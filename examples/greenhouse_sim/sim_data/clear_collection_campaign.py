"""Serial native clearer-image collection with durable per-family outcomes.

Only no-screened-viewpoint exits may be skipped. All other failures stop.
This does not approve, label, export or train on a single frame automatically.
"""
import argparse
from pathlib import Path
import shutil
from .collection_plan import load_plan
from .collection_run import run_jobs
from .dataset_review import read_json,write_json,require
from .depth_preview import sha256


def outcome(result,batch):
    if result['state']=='complete_bounded_batch_pending_visual_review':return 'audited'
    jobs=result.get('jobs',[])
    if len(jobs)==1:
        j=jobs[0]
        if (j.get('returncode')==3 and not j.get('timed_out') and j.get('sample_count')==0 and
                j.get('capture_state')=='blocked_no_screened_viewpoints'):
            capture=read_json(Path(batch)/j['job_id']/'capture/manifest.json')
            if capture['state']=='blocked_no_screened_viewpoints' and not capture['samples'] and capture.get('source_assets_unchanged'):
                return 'no_clear_screened_view'
    return 'failure'


def run(plan_path,output,*,timeout=3600):
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve();plan,_=load_plan(plan_path)
    require(plan['configuration'].get('clear_capture')=='robot_head_close_diffuse_v1','Only the explicit clear-capture plan')
    require(not output.exists() and not output.is_relative_to(Path(plan['package'])),'New campaign outside sources required')
    require(type(timeout) is int and 60<=timeout<=14400,'Invalid timeout')
    output.mkdir(parents=True);digest=sha256(plan_path)
    write_json(output/'request.json',dict(plan=str(plan_path),plan_sha256=digest,workers=1,timeout_s=timeout,
        native_depth_required=True,automatic_retries=False,source_geometry_changes=False,training_started=False))
    records=[];state='running'
    try:
        for job in sorted(plan['jobs'],key=lambda j:(j['split']!='train',j['job_id'])):
            require(sha256(plan_path)==digest,'Plan changed')
            require(shutil.disk_usage(output).free>=40*2**30,'Insufficient disk reserve; no deletion or launch')
            batch=output/job['job_id']
            result=run_jobs(plan_path,batch,max_jobs=1,timeout_s=timeout,job_ids=[job['job_id']],
                            instance_backend='fast',render_budget='warm56_then8')
            status=outcome(result,batch)
            record=dict(job_id=job['job_id'],family=job['plant_family'],split=job['split'],state=status,
                batch=str(batch),manifest_sha256=sha256(batch/'manifest.json'))
            if status=='audited':record['audit']=str(batch/job['job_id']/'audit/audit.json')
            records.append(record);write_json(output/(job['job_id']+'_outcome.json'),record)
            print('CLEAR_CAMPAIGN_OUTCOME',record,flush=True)
            require(status!='failure','Native capture failed; campaign stopped without automatic retry')
        state='completed_collection_pending_review'
    except BaseException as exc:
        state='stopped_collection'
        write_json(output/'failure.json',dict(error=str(exc)))
        raise
    finally:
        write_json(output/'result.json',dict(state=state,jobs=records,training_approved=False))
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args(argv);run(a.plan,a.output,timeout=a.timeout)


if __name__=='__main__':main()
