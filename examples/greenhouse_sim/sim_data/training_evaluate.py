"""Offline cut-point/abstention metrics and image-free shortcut baselines.

These baselines diagnose position/query shortcuts, not trained VLM performance.
Predictions are keyed by portable release row ID. No simulator or API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_review import read_json,require,safe_file,write_json
from .training_contract import validate_answer
from .training_export import read_jsonl,validate


def segment_distance(point,polyline):
    p=np.asarray(point,float); line=np.asarray(polyline,float)
    require(p.shape==(2,) and line.ndim==2 and line.shape[1]==2 and len(line)>=2
            and np.isfinite(p).all() and np.isfinite(line).all(),'Invalid projected interval')
    a,b=line[:-1],line[1:]; delta=b-a
    denom=(delta*delta).sum(axis=1)
    t=np.divide(((p-a)*delta).sum(axis=1),denom,out=np.zeros(len(a)),where=denom>0)
    closest=a+np.clip(t,0,1)[:,None]*delta
    return float(np.linalg.norm(closest-p,axis=1).min())


def score(rows,predictions,root):
    root=Path(root)
    require(len({r['id'] for r in rows})==len(rows),'Duplicate evaluation rows')
    require(set(predictions)=={r['id'] for r in rows},'Missing or unexpected prediction IDs')
    errors=[]; correct_status=0; false_localizations=0; interval_hits=0; target_hits=0
    positive_count=sum(r['answer']['status']=='localized' for r in rows)
    negative_count=len(rows)-positive_count; answered_positive=0
    for r in rows:
        p=validate_answer(predictions[r['id']]); truth=r['answer']
        correct_status+=p['status']==truth['status']
        if truth['status']=='abstain':
            false_localizations+=p['status']=='localized'
        elif p['status']=='localized':
            answered_positive+=1
            error=float(np.linalg.norm(np.asarray(p['cut_point_uv'])-truth['cut_point_uv']))
            errors.append(error)
            label=read_json(safe_file(root,r['files']['label']))
            interval_hits+=segment_distance(p['cut_point_uv'],label['accepted_interval_uv'])<=2.
            x,y=np.floor(p['cut_point_uv']).astype(int)
            target_hits+=np.asarray(Image.open(safe_file(root,r['files']['target_mask'])))[y,x]==255
    return dict(samples=len(rows),localized_ground_truth=positive_count,occluded_ground_truth=negative_count,
        status_accuracy=correct_status/len(rows) if rows else None,
        false_localization_rate_on_occluded=false_localizations/negative_count if negative_count else None,
        localized_answer_coverage=answered_positive/positive_count if positive_count else None,
        success_within_5px_including_abstention_failures=sum(e<=5 for e in errors)/positive_count if positive_count else None,
        projected_interval_hit_rate_2px=interval_hits/positive_count if positive_count else None,
        visible_target_hit_rate=target_hits/positive_count if positive_count else None,
        median_error_px_conditional_on_answer=float(np.median(errors)) if errors else None,
        metric_scope='2D_grounding_not_metric_3D_or_physical_cut_success')


def baselines(root,*,allow_incomplete=False):
    validate(root,allow_incomplete=allow_incomplete)
    rows=list(read_jsonl(Path(root)/'index.jsonl'))
    train=[r for r in rows if r['split']=='train' and r['answer']['status']=='localized']
    if not train:
        return {'state':'insufficient_localized_training_examples_for_baselines'}
    points=np.asarray([r['answer']['cut_point_uv'] for r in train])
    queries=np.asarray([r['query_pixel_uv'] for r in train])
    constant=np.median(points,axis=0); offset=np.median(points-queries,axis=0)
    results={}
    for split in ('validation','test'):
        held=[r for r in rows if r['split']==split]
        if not held: continue
        results[split]={}
        for name in ('constant_pixel','copy_query','query_plus_train_median_offset','always_abstain'):
            predictions={}
            for r in held:
                if name=='always_abstain':
                    p=dict(status='abstain',cut_point_uv=None,visibility='occluded',next_action='change_viewpoint')
                else:
                    point=constant if name=='constant_pixel' else np.asarray(r['query_pixel_uv'])+(offset if name=='query_plus_train_median_offset' else 0)
                    point=np.clip(point,[0,0],[847.9,407.9])
                    p=dict(status='localized',cut_point_uv=point.tolist(),visibility='clear',next_action='inspect_cut_region')
                predictions[r['id']]=p
            results[split][name]=score(held,predictions,root)
    return dict(state='complete_image_free_baselines_not_VLM_evaluation',fitted_on='training_localized_rows_only',
                training_constant_pixel=constant.tolist(),training_query_offset=offset.tolist(),results=results)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args(argv)
    require(not a.output.exists(),'Choose a new baseline report')
    result=baselines(a.dataset,allow_incomplete=a.allow_incomplete)
    write_json(a.output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
