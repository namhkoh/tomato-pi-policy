"""Generate hash-bound review cards for selected saved grounding frames.

May inspect completed sample files in a still-running job. That is explicitly
not a completed batch audit, human confirmation, or training-release approval.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from .collection_plan import load_plan
from .dataset_review import check_sample, read_json, reference_robot, require, review_card, safe_file, write_json
from .depth_preview import sha256
from .training_contract import derive_label


def preview(plan_path,batch,job_id,samples,output):
    plan,reports=load_plan(plan_path)
    job=next((j for j in plan['jobs'] if j['job_id']==job_id),None)
    require(job is not None,'Unknown job')
    require(bool(samples) and len(samples)==len(set(samples)) and
            all(re.fullmatch(r'sample_[0-9]{4}',s) for s in samples),'Invalid sample selection')
    capture=Path(batch)/job_id/'capture'
    output=Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(capture.resolve())
            and not output.is_relative_to(Path(plan['package'])),'Choose a new review directory outside sources')
    report=next(r for r in reports if r['plant_id']==job['plant_family'])
    model,mount,cal,_=reference_robot()
    checked=[]
    output.mkdir(parents=True)
    for sid in samples:
        directory=capture/sid
        meta_path=directory/'sample.json'
        initial=sha256(meta_path)
        metadata=read_json(meta_path)
        for name,detail in metadata['files'].items():
            require(sha256(safe_file(directory,name))==detail['sha256'],'Changed saved sample')
        draft=next(r for r in job['targets'] if r['draft_id']==metadata['supervision']['review_id'])
        audit,buffers=check_sample(directory,metadata,draft,model,mount,cal)
        label=derive_label(directory,metadata,report)
        review_card(output/(sid+'.png'),metadata,buffers)
        require(sha256(meta_path)==initial,'Sample changed during preview')
        checked.append(dict(sample_id=sid,sample_sha256=initial,engineering_check=audit,label=label,
                            review_card_sha256=sha256(output/(sid+'.png'))))
    result=dict(schema_version='greenhouse.grounding_visual_preview.v1',
                state='ready_for_visual_inspection_not_approval',source_plan_sha256=sha256(plan_path),
                source_capture=str(capture.resolve()),human_review_performed=False,
                complete_job_audit_claimed=False,training_release_approved=False,samples=checked)
    write_json(output/'preview.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--sample',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    r=preview(a.plan,a.batch,a.job,a.sample,a.output)
    print('GROUNDING_VISUAL_PREVIEW',len(r['samples']),r['state'],flush=True)


if __name__=='__main__': main()
