"""Teacher-forced localize/abstain probe: does the model carry an occlusion signal?

For each row, scores log P(value tokens | prompt + '{"status":"') for value in
(abstain, localized) with the SAME normalized prompt as training. Reports AUC,
balanced accuracy at the greedy (zero-margin) threshold and at the best margin.
Diagnostic only: no training, no test split unless explicitly requested.
"""
import argparse, json, os, random
from pathlib import Path


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('probe'); g.add_argument('--dataset',type=Path,required=True); g.add_argument('--model',type=Path,required=True)
    g.add_argument('--adapter',type=Path,default=None); g.add_argument('--tuned-model',type=Path,default=None)
    g.add_argument('--split',choices=('train','validation','test'),default='validation'); g.add_argument('--per-class',type=int,default=None,help='Deterministic per-status subsample (seed 0)')
    g.add_argument('--coordinates',choices=('normalized_1000','pixels'),default='normalized_1000'); g.add_argument('--coordinate-decimals',type=int,default=None)
    g.add_argument('--output',type=Path,required=True); g.add_argument('--shard-index',type=int,default=0); g.add_argument('--shard-count',type=int,default=1)
    s=sub.add_parser('summarize'); s.add_argument('--run',type=Path,required=True); s.add_argument('--output',type=Path,required=True)
    return p.parse_args(argv)


def probe(a):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from .training_export import read_jsonl
    from .qwen_adapter import model_messages
    root=a.dataset.resolve(); rows=list(read_jsonl(root/f'splits/{a.split}.jsonl'))
    if a.per_class:
        rng=random.Random(0); groups={'localized':[],'abstain':[]}
        for r in rows: groups[json.loads(r['messages'][2]['content'])['status']].append(r)
        rows=[]; [rows.extend(rng.sample(g,min(a.per_class,len(g)))) for g in groups.values()]
    rows=rows[a.shard_index::a.shard_count]
    src=(a.tuned_model or a.adapter or a.model).resolve()
    processor=AutoProcessor.from_pretrained(str(src),local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(str((a.tuned_model or a.model).resolve()),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    if a.adapter:
        from peft import PeftModel; model=PeftModel.from_pretrained(model,str(a.adapter.resolve()),local_files_only=True)
    model=model.to('cuda').eval(); tok=processor.tokenizer
    kw={} if a.coordinate_decimals is None else dict(decimals=a.coordinate_decimals)
    if a.tuned_model or a.adapter:
        contract=json.loads((src/'grounding_adapter.json').read_text())
        if contract.get('depth_input'): kw['depth_input']=True
        if contract.get('coordinate_decimals') is not None: kw['decimals']=contract['coordinate_decimals']
    a.output.mkdir(parents=True,exist_ok=True); out=a.output/f'probe-shard-{a.shard_index}-of-{a.shard_count}.jsonl'
    with open(out,'w') as fh:
        for n,row in enumerate(rows,1):
            messages=model_messages(row,root,coordinates=a.coordinates,**kw)
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
            base=inputs['input_ids']; scores={}
            for value in ('abstain','localized'):
                cont=tok('{"status":"'+value,add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
                pre=tok('{"status":"',add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
                assert torch.equal(cont[:,:pre.shape[1]],pre),'status prefix tokenization not stable'
                ids=torch.cat([base,cont],dim=1); am=torch.ones_like(ids)
                with torch.inference_mode():
                    logits=model(input_ids=ids,attention_mask=am,pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw']).logits
                lp=torch.log_softmax(logits[0,:-1].float(),dim=-1); tgt=ids[0,1:]
                start=base.shape[1]+pre.shape[1]-1  # positions predicting the value tokens
                scores[value]=float(lp[start:,:].gather(1,tgt[start:,None]).sum())
            truth=json.loads(row['messages'][2]['content'])['status']
            fh.write(json.dumps(dict(id=row['id'],truth=truth,logp_abstain=scores['abstain'],logp_localized=scores['localized'],margin=scores['abstain']-scores['localized']))+'\n'); fh.flush()
            if n%50==0 or n==len(rows): print(f'shard {a.shard_index}: {n}/{len(rows)}',flush=True)


def summarize(a):
    import numpy as np
    recs=[json.loads(l) for f in sorted(a.run.glob('probe-shard-*.jsonl')) for l in f.read_text().splitlines()]
    y=np.array([r['truth']=='abstain' for r in recs]); m=np.array([r['margin'] for r in recs])
    pos=m[y]; neg=m[~y]
    auc=float((pos[:,None]>neg[None,:]).mean()+.5*(pos[:,None]==neg[None,:]).mean()) if len(pos) and len(neg) else None
    def ba(t): return float(((m[y]>t).mean()+(m[~y]<=t).mean())/2)
    grid=np.quantile(m,np.linspace(0,1,201)); best=max(grid,key=ba)
    rep=dict(n=len(recs),n_abstain=int(y.sum()),n_localized=int((~y).sum()),auc_abstain_vs_localized=auc,
        greedy_margin0=dict(balanced_accuracy=ba(0.),abstain_rate=float((m>0).mean()),tpr_abstain=float((m[y]>0).mean()),fpr_localize_on_occluded=float((m[y]<=0).mean())),
        best_threshold=dict(margin=float(best),balanced_accuracy=ba(float(best)),abstain_rate=float((m>best).mean())),
        margin_stats=dict(abstain_mean=float(pos.mean()) if len(pos) else None,localized_mean=float(neg.mean()) if len(neg) else None,std=float(m.std())))
    a.output.write_text(json.dumps(rep,indent=1)); print(json.dumps(rep,indent=1))


def main(argv=None):
    a=arguments(argv); probe(a) if a.command=='probe' else summarize(a)


if __name__=='__main__': main()
