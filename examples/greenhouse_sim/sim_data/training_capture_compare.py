"""Matched-pose task-level comparison of independently audited capture profiles.

RGB stochastic variation is reported, not mistaken for a moved anatomical label.
This report does not itself grant visual QA or approve a production profile.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import audit_manifest
from .dataset_review import read_json, require, verify_bindings, write_json
from .depth_preview import sha256
from .training_contract import derive_label
from .training_export import view_signature


def verified_run(root):
    root=Path(root).resolve()
    manifest=read_json(root/'manifest.json')
    result=read_json(root.parent/'result.json')
    audit_path=root.parent/'audit/audit.json'
    require(manifest['state']=='pilot_ready_for_review' and manifest['source_assets_unchanged'] is True,
            'Incomplete capture cannot qualify a profile')
    require(result['returncode']==0 and not result['timed_out'] and result['audit_sha256']==sha256(audit_path),
            'Capture/audit did not finish cleanly')
    audit=read_json(audit_path)
    verify_bindings(audit['bindings_sha256'])
    plan=read_json(manifest['source_collection_plan_path'])
    require(sha256(manifest['source_collection_plan_path'])==manifest['source_collection_plan_sha256'],'Changed plan')
    job=next(j for j in plan['jobs'] if j['job_id']==manifest['collection_job_id'])
    report=audit_manifest(job['source_manifest_path'])
    rows=[]
    for entry in manifest['samples']:
        directory=root/entry['sample_id']
        meta=read_json(directory/'sample.json')
        rows.append(dict(directory=directory,metadata=meta,
                         signature=view_signature(meta,job['plant_family'],manifest['lighting']),
                         label=derive_label(directory,meta,report)))
    return manifest,rows


def pixel_evidence(reference,candidate,point):
    """Compare saved native arrays and a local RGB region; no image modifications."""
    a=np.asarray(Image.open(reference/'inputs/rgb.png')).astype(float)
    b=np.asarray(Image.open(candidate/'inputs/rgb.png')).astype(float)
    ta=np.asarray(Image.open(reference/'supervision/target_visible.png'))==255
    tb=np.asarray(Image.open(candidate/'supervision/target_visible.png'))==255
    da=np.load(reference/'inputs/depth_m.npy',allow_pickle=False)
    db=np.load(candidate/'inputs/depth_m.npy',allow_pickle=False)
    va=np.asarray(Image.open(reference/'inputs/depth_valid.png'))==255
    vb=np.asarray(Image.open(candidate/'inputs/depth_valid.png'))==255
    overlap=va&vb
    delta=np.abs(a-b)
    x,y=np.floor(point).astype(int)
    roi=delta[max(0,y-32):min(408,y+33),max(0,x-32):min(848,x+33)]
    union=int((ta|tb).sum())
    return dict(target_visible_mask_equal=bool(np.array_equal(ta,tb)),
                target_mask_IoU=float((ta&tb).sum()/union) if union else None,
                depth_validity_equal=bool(np.array_equal(va,vb)),
                native_depth_equal_0_2mm=bool(np.array_equal(va,vb) and np.allclose(da[overlap],db[overlap],atol=.0002,rtol=0)),
                RGB_mean_absolute_difference_0_255=float(delta.mean()),
                target_ROI_RGB_mean_absolute_difference_0_255=float(roi.mean()))


def compare(reference,candidate,output):
    output=Path(output)
    require(not output.exists(),'Choose a new comparison report')
    rm,rr=verified_run(reference); cm,cr=verified_run(candidate)
    require(rm['source_usd_sha256']==cm['source_usd_sha256'] and rm['lighting']==cm['lighting'],
            'Comparison requires the same scene assets and lighting')
    lookup={r['signature']:r for r in rr}
    require(len(lookup)==len(rr),'Reference contains duplicate poses')
    pairs=[]; unmatched=[]
    for c in cr:
        r=lookup.get(c['signature'])
        if r is None:
            unmatched.append(c['directory'].name); continue
        rl,cl=r['label'],c['label']
        require(rl['target_id']==cl['target_id'],'Matched pose has a different target')
        both=rl['eligible'] and cl['eligible']
        pairs.append(dict(reference_sample=r['directory'].name,candidate_sample=c['directory'].name,
            target_id=rl['target_id'],camera_scene_signature=c['signature'],
            reference_label=rl,candidate_label=cl,
            accepted_answer_equal=bool(rl['answer']==cl['answer']) if both else None,
            pixels=pixel_evidence(r['directory'],c['directory'],r['metadata']['supervision']['nominal_projected']['pixel_xy']),
            reference_capture_seconds=sum(r['metadata']['performance_seconds'].values()),
            candidate_capture_seconds=sum(c['metadata']['performance_seconds'].values()),
            reference_sample_sha256=sha256(r['directory']/'sample.json'),
            candidate_sample_sha256=sha256(c['directory']/'sample.json')))
    result=dict(schema_version='greenhouse.matched_capture_task_comparison.v1',
        state='ready_for_task_level_visual_QA_not_profile_approval',reference_run=str(Path(reference).resolve()),
        candidate_run=str(Path(candidate).resolve()),profile_approved=False,
        matched=len(pairs),unmatched_candidate_samples=unmatched,
        accepted_both_by_difficulty=dict(Counter(p['reference_label']['difficulty'] for p in pairs
            if p['reference_label']['eligible'] and p['candidate_label']['eligible'])),
        accepted_answer_mismatches=sum(p['accepted_answer_equal'] is False for p in pairs),
        pairs=pairs)
    write_json(output,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    r=compare(a.reference,a.candidate,a.output)
    print({k:v for k,v in r.items() if k!='pairs'},flush=True)


if __name__=='__main__': main()
