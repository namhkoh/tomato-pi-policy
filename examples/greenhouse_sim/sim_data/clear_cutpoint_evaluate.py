"""Visible-only point evaluation. No test access by default; no motion authority."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_review import read_json,write_json,safe_file,require
from .depth_preview import sha256
from .training_export import read_jsonl
from .training_contract import validate_answer
from .training_evaluate import segment_distance
from .clear_cutpoint_release import validate


def summarize(items):
    n=len(items);errors=[i['error_px'] for i in items if i['error_px'] is not None]
    def rate(key): return sum(bool(i[key]) for i in items)/n if n else None
    return dict(rows=n,valid_json_answer_rate=rate('valid'),localized_coverage=rate('localized'),
                interval_hit_2px=rate('interval_hit'),interval_and_target_hit=rate('joint_hit'),
                target_mask_hit=rate('target_hit'),success_5px=rate('within_5px'),
                median_error_answered_px=float(np.median(errors)) if errors else None,
                p90_error_answered_px=float(np.percentile(errors,90)) if errors else None,
                error_denominator='localized_answers_only; success rates include invalid/abstain/missing failures')


def metrics(rows, raw_predictions, root, coordinates='normalized_1000'):
    from .qwen_coordinates import answer_to_pixels
    require(coordinates in ('pixels','normalized_1000'),'Unknown coordinates')
    require(rows and len({r['id'] for r in rows})==len(rows),'Empty/duplicate rows')
    require(set(raw_predictions)<={r['id'] for r in rows},'Unexpected prediction IDs')
    items=[]
    for r in rows:
        item=dict(id=r['id'],target=r['target_id'],family=r['source_plant_family'],valid=False,
                  localized=False,interval_hit=False,target_hit=False,joint_hit=False,within_5px=False,error_px=None)
        try:
            answer=json.loads(raw_predictions[r['id']])  # No repair of malformed output.
            answer=answer_to_pixels(answer) if coordinates=='normalized_1000' else validate_answer(answer)
            item['valid']=True;item['localized']=answer['status']=='localized'
            if item['localized']:
                point=answer['cut_point_uv'];label=read_json(safe_file(root,r['files']['label']))
                item['error_px']=float(np.linalg.norm(np.asarray(point)-r['answer']['cut_point_uv']))
                item['within_5px']=item['error_px']<=5
                item['interval_hit']=segment_distance(point,label['accepted_interval_uv'])<=2
                x,y=np.floor(point).astype(int)
                with Image.open(safe_file(root,r['files']['target_mask'])) as mask:
                    item['target_hit']=bool(np.asarray(mask)[y,x]==255)
                item['joint_hit']=item['interval_hit'] and item['target_hit']
        except (KeyError,ValueError,TypeError) as e:
            # Missing/invalid predictions fail. Dataset errors must not be swallowed.
            if item['valid']: raise
            item['failure']=type(e).__name__
        items.append(item)
    result=dict(overall=summarize(items),per_example=items)
    for key in ('target','family'):
        groups=defaultdict(list)
        for i in items: groups[i[key]].append(i)
        result['by_'+key]={k:summarize(v) for k,v in sorted(groups.items())}
        result[key+'_macro_interval_hit_2px']=float(np.mean([summarize(v)['interval_hit_2px'] for v in groups.values()]))
        result[key+'_macro_joint_hit']=float(np.mean([summarize(v)['interval_and_target_hit'] for v in groups.values()]))
    result['scope']='Visible-only synthetic 2D localization, not abstention ability, XYZ accuracy or cut safety'
    result['wrong_organ_scope']='Off-target mask hits measured; leaf/fruit/main-stem attribution unavailable in portable sidecars'
    return result


def score_run(args):
    root=Path(args.dataset);validate(root);require(not args.output.exists(),'New report path required')
    headers=[];records={}
    for path in sorted(Path(args.run).glob('predictions-shard-*.jsonl')):
        data=list(read_jsonl(path));require(data and data[0]['kind']=='header','Missing shard header')
        headers.append(data[0]);require(len(data)-1==data[0]['rows'],'Incomplete shard output')
        for r in data[1:]:
            require(r['kind']=='prediction' and r['id'] not in records,'Duplicate/invalid prediction')
            records[r['id']]=r
    require(headers,'No predictions')
    keys=('split','coordinates','coordinate_decimals','query_crop','depth_input','model_path',
          'adapter','tuned_model','release_manifest_sha256','shard_count','max_new_tokens','decoding')
    require(all(all(h.get(k)==headers[0].get(k) for k in keys) for h in headers),'Mixed evaluation contracts')
    h=headers[0];require(len(headers)==h['shard_count'] and
        {x['shard_index'] for x in headers}==set(range(h['shard_count'])),'Incomplete/duplicate shards')
    require(h['release_manifest_sha256']==sha256(root/'manifest.json'),'Evaluation dataset changed')
    require(h['split']!='test' or all(x.get('test_access_authorized') for x in headers),'Test not authorized')
    rows=[r for r in read_jsonl(root/'index.jsonl') if r['split']==h['split']]
    report=metrics(rows,{k:('' if v.get('truncated') else v['raw']) for k,v in records.items()},root,h['coordinates'])
    complete=set(records)=={r['id'] for r in rows} and not any(x.get('limited_run') for x in headers)
    report.update(state='complete_clear_evaluation' if complete else 'partial_clear_evaluation_not_final',
                  split=h['split'],header=h,predictions=len(records),release_manifest_sha256=sha256(root/'manifest.json'))
    for key in ('latency_s','prompt_tokens','generated_tokens'):
        values=np.asarray([r[key] for r in records.values()],float)
        require(len(values)>0 and np.isfinite(values).all() and (values>=0).all(),'Invalid timing/token receipt')
        report[key]=dict(p50=float(np.median(values)),p95=float(np.percentile(values,95)),mean=float(np.mean(values)))
    report['truncated_predictions']=sum(bool(r.get('truncated')) for r in records.values())
    write_json(args.output,report);print(json.dumps(report['overall'],indent=2))


def baselines(root,split='validation',allow_test=False):
    require(split in ('validation','test') and (split!='test' or allow_test),'Test sealed')
    root=Path(root);validate(root);rows=list(read_jsonl(root/'index.jsonl'))
    train=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']==split]
    points=np.asarray([r['answer']['cut_point_uv'] for r in train]);queries=np.asarray([r['query_pixel_uv'] for r in train])
    constant=np.median(points,axis=0);offset=np.median(points-queries,axis=0);results={}
    for name in ('constant_pixel','copy_query','query_plus_train_median_offset'):
        predictions={}
        for r in held:
            p=constant if name=='constant_pixel' else np.asarray(r['query_pixel_uv'])+(offset if name.endswith('offset') else 0)
            answer=dict(status='localized',cut_point_uv=np.clip(p,[0,0],[847.9,407.9]).tolist(),visibility='clear',next_action='inspect_cut_region')
            predictions[r['id']]=json.dumps(answer)
        results[name]=metrics(held,predictions,root,'pixels')
    return dict(fitted_on='train_only',split=split,results=results)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--allow-test',action='store_true');a=p.parse_args(argv)
    require(not a.output.exists(),'New output required');write_json(a.output,baselines(a.dataset,a.split,a.allow_test))


if __name__=='__main__':main()
