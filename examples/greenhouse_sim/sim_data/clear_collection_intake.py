"""Inspect completed native batches while collection continues; never approve.

Creates new engineering pools, strict-screen drafts and review pages. It cannot
launch Isaac, write decisions, alter a split, finalize a release or start training.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from .dataset_review import read_json, write_json, require
from .depth_preview import sha256


def completed_json(path):
    # Outcome writers use exclusive creation. A polling reader can briefly see
    # a newly created, incomplete JSON file. Never consume or mark that as seen.
    try:return read_json(path)
    except (FileNotFoundError,json.JSONDecodeError):return None


def checked_audit(record):
    require(record.get('state')=='audited','Only independently audited batches')
    batch=Path(record['batch']).resolve()
    require(sha256(batch/'manifest.json')==record['manifest_sha256'],'Batch changed')
    manifest=read_json(batch/'manifest.json')
    require(manifest['state']=='complete_bounded_batch_pending_visual_review' and len(manifest['jobs'])==1,
            'Expected one completed native worker')
    job=manifest['jobs'][0]
    require(job['job_id']==record['job_id'] and job['plant_family']==record['family'] and
            job['state']=='audited_prototype_pending_visual_review' and job['returncode']==0 and
            not job['timed_out'],'Unsuccessful native worker')
    audit=Path(record['audit']).resolve()
    require(audit==batch/record['job_id']/'audit/audit.json' and
            audit==Path(job['audit_path']).resolve() and sha256(audit)==job['audit_sha256'],'Audit binding mismatch')
    return audit


def materialize(audits, output):
    from .training_export import build as pool
    from .clear_cutpoint_release import build as clear, scan
    from .clear_cutpoint_review import build as review
    output=Path(output).resolve();require(not output.exists(),'New intake output required')
    output.mkdir(parents=True)
    pool(audits,output/'source',allow_incomplete=True)
    rows,exclusions=scan(output/'source')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                source_audits={str(p):sha256(p) for p in audits},
                clear_candidates=len(rows),excluded=len(exclusions),training_approved=False,
                visual_review_performed=False)
    if rows:
        m=clear(output/'source',output/'draft',audited_source=True,
                review_policy='assistant_reviewed_experiment_v1')
        result.update(state='draft_pending_visual_review',acceptance=m['acceptance'],review=m['review'])
        result['review_page']=review(output/'draft',output/'review')['path']
    else:result['state']='no_strict_clear_candidates'
    write_json(output/'result.json',result)
    return result


def watch(campaign,output,*,timeout=36000,poll=15):
    campaign,output=Path(campaign).resolve(),Path(output).resolve()
    require((campaign/'request.json').is_file(),'Missing campaign request')
    require(not output.exists() and not output.is_relative_to(campaign) and not campaign.is_relative_to(output),
            'New disjoint intake directory required')
    require(1<=poll<=60 and 60<=timeout<=43200,'Invalid bounded watcher duration')
    output.mkdir(parents=True);request_hash=sha256(campaign/'request.json')
    write_json(output/'request.json',dict(campaign=str(campaign),campaign_request_sha256=request_hash,
        timeout_s=timeout,poll_s=poll,automatic_approvals=False,training_started=False))
    seen={};audits=[];started=time.monotonic()
    try:
        while time.monotonic()-started<timeout:
            require(sha256(campaign/'request.json')==request_hash,'Campaign request changed')
            for path in sorted(campaign.glob('job_*_outcome.json')):
                if path.name in seen:
                    require(sha256(path)==seen[path.name],'Completed outcome changed');continue
                record=completed_json(path)
                if record is None:continue
                seen[path.name]=sha256(path)
                if record['state']!='audited':continue
                audit=checked_audit(record);audits.append(audit)
                result=materialize([audit],output/record['job_id'])
                print('CLEAR_INTAKE',record['job_id'],result,flush=True)
            if (campaign/'result.json').is_file():
                result=completed_json(campaign/'result.json')
                if result is None:
                    time.sleep(poll);continue
                require(result['state']=='completed_collection_pending_review','Collection stopped; no aggregate release')
                require(audits,'No successful audited jobs')
                aggregate=materialize(audits,output/'aggregate')
                write_json(output/'result.json',dict(state='intake_complete_pending_review',
                    audited_jobs=len(audits),aggregate=aggregate,training_approved=False))
                return
            time.sleep(poll)
        raise TimeoutError('Intake timeout; collection is not stopped or modified')
    except BaseException as exc:
        write_json(output/'failure.json',dict(error=str(exc),audited_jobs=len(audits),training_approved=False))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=36000)
    a=p.parse_args(argv);watch(a.campaign,a.output,timeout=a.timeout)


if __name__=='__main__':main()
