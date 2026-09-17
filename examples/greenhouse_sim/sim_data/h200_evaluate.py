"""Greedy generation and canonical-pixel scoring for untuned/tuned Qwen3-VL-8B.

Server-only companion to h200_train.py. `generate` runs one local base snapshot,
optionally with a saved PEFT adapter, over one frozen split (validation by
default) using the SAME normalized prompt as training, and stores raw answers,
IDs, the coordinate contract and per-example latency. `score` decodes with
`answer_to_pixels` before `training_evaluate.score`, reports invalid JSON /
coordinates separately (never repaired), and breaks results down per status,
visibility and family. Nothing here trains, touches test labels unless the
split is explicitly requested, or authorizes a physical cut.
"""
import argparse
import importlib.metadata
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate')
    g.add_argument('--dataset',type=Path,required=True)
    g.add_argument('--model',type=Path,required=True,help='Already downloaded LOCAL base snapshot')
    g.add_argument('--adapter',type=Path,default=None,help='Saved PEFT adapter directory; omit for the untuned base')
    g.add_argument('--tuned-model',type=Path,default=None,help='Saved full fine-tuned model directory (from --method full)')
    g.add_argument('--output',type=Path,required=True,help='Run directory; shards are appended as new files')
    g.add_argument('--split',choices=('train','validation','test'),default='validation')
    g.add_argument('--ids-file',type=Path,default=None,help='JSON with train_ids (e.g. an overfit run_contract.json) to restrict rows')
    g.add_argument('--coordinates',choices=('normalized_1000','pixels'),default='normalized_1000')
    g.add_argument('--shard-index',type=int,default=0)
    g.add_argument('--shard-count',type=int,default=1)
    g.add_argument('--limit',type=int,default=None,help='Only the first N rows of this shard (smoke use)')
    g.add_argument('--max-new-tokens',type=int,default=256)
    s=sub.add_parser('score')
    s.add_argument('--dataset',type=Path,required=True)
    s.add_argument('--run',type=Path,required=True,help='Directory holding predictions-shard-*.jsonl')
    s.add_argument('--output',type=Path,required=True,help='New report JSON path')
    args=parser.parse_args(argv)
    if args.command=='generate' and not (args.shard_count>=1 and 0<=args.shard_index<args.shard_count):
        parser.error('Invalid shard')
    if args.command=='generate' and args.adapter and args.tuned_model: parser.error('Choose --adapter or --tuned-model')
    return args


def generate(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from .dataset_review import write_json
    from .training_export import read_jsonl
    from .qwen_adapter import model_messages
    from .qwen_coordinates import COORDINATE_ADAPTER
    root=args.dataset.resolve();model_path=args.model.resolve();output=args.output.resolve()
    if not (root/'manifest.json').is_file(): raise ValueError('Local release required')
    if output==root or output.is_relative_to(root) or output.is_relative_to(model_path):
        raise ValueError('Keep outputs separate from dataset and model')
    if args.split=='test': print('WARNING: generating on TEST; only do this after locking the configuration',flush=True)
    rows=list(read_jsonl(root/f'splits/{args.split}.jsonl'))
    if args.ids_file:
        keep=set(json.loads(args.ids_file.read_text())['train_ids'])
        rows=[r for r in rows if r['id'] in keep]
        if len(rows)!=len(keep): raise ValueError('ids-file rows missing from split')
    rows=rows[args.shard_index::args.shard_count]
    if args.limit is not None: rows=rows[:args.limit]
    if not rows: raise ValueError('Empty shard')
    output.mkdir(parents=True,exist_ok=True)
    target=output/f'predictions-shard-{args.shard_index}-of-{args.shard_count}.jsonl'
    if target.exists(): raise ValueError(f'{target} exists; choose a new run directory')
    tuned=args.tuned_model.resolve() if args.tuned_model else None
    decimals=None;depth_input=False;no_query=False
    if args.adapter or tuned:
        adapter=args.adapter.resolve() if args.adapter else None
        contract=json.loads(((adapter or tuned)/'grounding_adapter.json').read_text())
        if contract['coordinates']!=args.coordinates: raise ValueError('Adapter coordinate convention differs from request')
        decimals=contract.get('coordinate_decimals');depth_input=bool(contract.get('depth_input'));no_query=not contract.get('query_pixel_given',True)
        processor=AutoProcessor.from_pretrained(str(adapter or tuned),local_files_only=True,trust_remote_code=False)
    else:
        adapter=None
        processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(tuned or model_path),local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),local_files_only=True)
    model=model.to('cuda').eval()
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','accelerate')}
    from .qwen_coordinates import adapter_name
    header=dict(kind='header',split=args.split,model_path=str(model_path),adapter=str(adapter) if adapter else None,
        tuned_model=str(tuned) if tuned else None,coordinate_decimals=decimals,depth_input=depth_input,query_pixel_given=not no_query,
        coordinates=args.coordinates,coordinate_adapter=adapter_name(decimals) if args.coordinates=='normalized_1000' else 'canonical_task_v3_pixels',
        shard_index=args.shard_index,shard_count=args.shard_count,rows=len(rows),max_new_tokens=args.max_new_tokens,
        decoding='greedy',versions=versions)
    with open(target,'w') as handle:
        handle.write(json.dumps(header)+'\n')
        for n,row in enumerate(rows,1):
            messages=model_messages(row,root,coordinates=args.coordinates,decimals=decimals,depth_input=depth_input,no_query=no_query)
            assert [m['role'] for m in messages]==['system','user']
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt').to('cuda')
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            torch.cuda.synchronize();latency=time.perf_counter()-start
            new=out[0,inputs['input_ids'].shape[1]:]
            raw=processor.tokenizer.decode(new,skip_special_tokens=True)
            record=dict(kind='prediction',id=row['id'],raw=raw,latency_s=latency,prompt_tokens=int(inputs['input_ids'].shape[1]),
                generated_tokens=int(new.shape[0]),truncated=bool(new.shape[0]>=args.max_new_tokens))
            handle.write(json.dumps(record)+'\n');handle.flush()
            if n%25==0 or n==len(rows): print(f'shard {args.shard_index}: {n}/{len(rows)} last_latency={latency:.2f}s',flush=True)
    write_json(output/f'shard-{args.shard_index}-of-{args.shard_count}.done.json',dict(rows=len(rows),file=target.name))


def _decode(raw,coordinates):
    """Return (answer_in_canonical_pixels, failure_reason)."""
    from .qwen_coordinates import answer_to_pixels
    from .training_contract import validate_answer
    text=raw.strip()
    if text.startswith('```'):
        text=text.strip('`');text=text[4:] if text.startswith('json') else text;text=text.strip()
    try: parsed=json.loads(text)
    except Exception: return None,'invalid_json'
    try:
        answer=answer_to_pixels(parsed) if coordinates=='normalized_1000' else validate_answer(parsed)
    except Exception as error: return None,'invalid_answer:'+type(error).__name__
    return answer,None


def _metrics(rows,predictions,root):
    """Scorer on the valid subset plus exact counts penalizing invalid outputs."""
    from .training_evaluate import score
    valid_rows=[r for r in rows if r['id'] in predictions]
    positives=sum(r['answer']['status']=='localized' for r in rows);negatives=len(rows)-positives
    result=dict(samples=len(rows),valid_predictions=len(valid_rows),invalid_predictions=len(rows)-len(valid_rows),
        invalid_rate=(len(rows)-len(valid_rows))/len(rows) if rows else None,
        localized_ground_truth=positives,occluded_ground_truth=negatives)
    if valid_rows:
        subset=score(valid_rows,{r['id']:predictions[r['id']] for r in valid_rows},root)
        result['valid_subset']=subset
        sp=subset['localized_ground_truth'];sn=subset['occluded_ground_truth']
        def scale(value,numerator,denominator):
            return None if value is None or not denominator else value*numerator/denominator
        result['all_rows_invalid_counted_as_failure']=dict(
            status_accuracy=scale(subset['status_accuracy'],len(valid_rows),len(rows)),
            false_localization_rate_on_occluded=scale(subset['false_localization_rate_on_occluded'],sn,negatives),
            localized_answer_coverage=scale(subset['localized_answer_coverage'],sp,positives),
            success_within_5px=scale(subset['success_within_5px_including_abstention_failures'],sp,positives),
            projected_interval_hit_rate_2px=scale(subset['projected_interval_hit_rate_2px'],sp,positives),
            visible_target_hit_rate=scale(subset['visible_target_hit_rate'],sp,positives))
    return result


def score_run(args):
    from .dataset_review import write_json
    from .training_export import read_jsonl
    root=args.dataset.resolve();run=args.run.resolve()
    if args.output.exists(): raise ValueError('Choose a new report path')
    shards=sorted(run.glob('predictions-shard-*.jsonl'))
    if not shards: raise ValueError('No prediction shards')
    headers=[];records={}
    for shard in shards:
        for line in shard.read_text().splitlines():
            item=json.loads(line)
            if item['kind']=='header': headers.append(item)
            elif item['id'] in records: raise ValueError('Duplicate prediction id '+item['id'])
            else: records[item['id']]=item
    split={h['split'] for h in headers};coordinates={h['coordinates'] for h in headers};adapters={(h['adapter'],h.get('tuned_model')) for h in headers}
    if len(split)!=1 or len(coordinates)!=1 or len(adapters)!=1: raise ValueError('Mixed shards in one run')
    split=split.pop();coordinates=coordinates.pop()
    expected={h['shard_index'] for h in headers};counts={h['shard_count'] for h in headers}
    if len(counts)!=1 or expected!=set(range(counts.pop())): raise ValueError('Incomplete shard set')
    index={r['id']:r for r in read_jsonl(root/'index.jsonl') if r['split']==split}
    rows=[index[i] for i in records]
    if args.__dict__.get('ids_file'): pass
    if not records or set(records)-set(index): raise ValueError('Predictions outside the split')
    predictions={};failures=defaultdict(int);per_example=[]
    for row in rows:
        rec=records[row['id']];answer,failure=_decode(rec['raw'],coordinates)
        if failure: failures[failure]+=1
        else: predictions[row['id']]=answer
        per_example.append(dict(id=row['id'],raw=rec['raw'],decoded_pixels=answer,failure=failure,
            truth=row['answer'],latency_s=rec['latency_s'],truncated=rec.get('truncated')))
    latencies=sorted(r['latency_s'] for r in records.values())
    def pct(p): return latencies[min(len(latencies)-1,int(round(p*(len(latencies)-1))))]
    report=dict(state='validation_generation_evaluation_2D_only',split=split,coordinates=coordinates,
        adapter=headers[0]['adapter'],tuned_model=headers[0].get('tuned_model'),model_path=headers[0]['model_path'],decoding=headers[0]['decoding'],
        max_new_tokens=headers[0]['max_new_tokens'],versions=headers[0]['versions'],
        rows_in_split=len(index),rows_scored=len(rows),failures=dict(failures),
        latency_s=dict(p50=statistics.median(latencies),p95=pct(.95),mean=statistics.fmean(latencies),n=len(latencies)),
        overall=_metrics(rows,predictions,root),by_truth_status={},by_truth_visibility={},by_family={},by_difficulty={},
        scope='2D grounding proxy on synthetic data; not metric 3D accuracy, physical cut success or real-world safety')
    family_key=next((k for k in ('source_plant_family','family','target_family') if rows and k in rows[0]),None)
    difficulty_key=next((k for k in ('difficulty','difficulty_bin') if rows and k in rows[0]),None)
    groups={'by_truth_status':lambda r:r['answer']['status'],'by_truth_visibility':lambda r:r['answer']['visibility'],
        'by_family':(lambda r:str(r[family_key])) if family_key else None,
        'by_difficulty':(lambda r:str(r[difficulty_key])) if difficulty_key else None}
    for name,key in groups.items():
        if key is None: report[name]={'unavailable':'no such field in index.jsonl'};continue
        buckets=defaultdict(list)
        for r in rows: buckets[key(r)].append(r)
        report[name]={k:_metrics(v,predictions,root) for k,v in sorted(buckets.items())}
    write_json(args.output,report)
    write_json(args.output.with_name(args.output.stem+'.per_example.json'),per_example)
    summary={k:report['overall'].get(k) for k in ('samples','invalid_predictions','invalid_rate')}
    summary.update(report['overall'].get('all_rows_invalid_counted_as_failure',{}));summary['latency_p50_s']=report['latency_s']['p50'];summary['latency_p95_s']=report['latency_s']['p95']
    print(json.dumps(summary,indent=2),flush=True)


def main(argv=None):
    args=arguments(argv)
    generate(args) if args.command=='generate' else score_run(args)


if __name__=='__main__': main()
