"""Greedy validation generation, scoring in canonical pixels, depth metrics, and a teacher-forced status probe
for 3D HAMSTER cut-point models. Reuses the task-v3 scorer through sim_data.h200_evaluate._metrics.

  generate --dataset D --model M --output RUN [--split validation] [--shard-index i --shard-count n] [--limit N]
  score    --dataset D --run RUN --output report.json
  probe    --dataset D --model M --output RUN [--shard-index i --shard-count n]   (logP('[') - logP('null') after "point_3d": )
  summarize-probe --run RUN --output summary.json
"""
import argparse, json, os, statistics, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutpoint_data as cd
from sim_data.h200_evaluate import _metrics  # noqa: E402
from sim_data.dataset_review import write_json  # noqa: E402


def load(model_path):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from hamster3d.model import register_qwen3_vl_geometry
    register_qwen3_vl_geometry()
    proc = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='sdpa').to('cuda').eval()
    return proc, model


def inputs_for(proc, model, root, row, longest_edge, text_override=None, add_generation_prompt=True, query=True):
    import torch
    rgb, depth, valid = cd.load_frame(root, row); img, geo, dmap = cd.prepare(rgb, depth, longest_edge)
    msgs = cd.messages(row, query=query)
    text = text_override if text_override is not None else proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    mi = proc(text=[text], images=[img], return_tensors='pt').to('cuda'); dt = next(model.parameters()).dtype
    for k, v in list(mi.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v): mi[k] = v.to(dt)
    mi['geometry_encoder_inputs'] = [geo.to('cuda', dt)]; mi['depth_maps'] = [dmap.to('cuda', dt)]
    return mi, depth, valid


def rows_for(root, split, shard_index, shard_count, limit=None, ids_file=None):
    index = {r['id']: r for r in cd.read_jsonl(Path(root) / 'index.jsonl')}
    rows = [index[r['id']] for r in cd.read_jsonl(Path(root) / f'splits/{split}.jsonl')]
    if ids_file:
        keep = set(json.loads(Path(ids_file).read_text())['train_ids']); rows = [r for r in rows if r['id'] in keep]
    rows = rows[shard_index::shard_count]
    return rows[:limit] if limit else rows


def generate(a):
    import torch
    root = a.dataset.resolve(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((a.model / 'grounding_adapter.json').read_text()) if (a.model / 'grounding_adapter.json').exists() else {}
    le = contract.get('longest_edge', a.longest_edge); query = contract.get('query_pixel_given', True)
    proc, model = load(a.model); rows = rows_for(root, a.split, a.shard_index, a.shard_count, a.limit, a.ids_file)
    target = out / f'predictions-shard-{a.shard_index}-of-{a.shard_count}.jsonl'
    if target.exists(): raise ValueError(f'{target} exists')
    with open(target, 'w') as fh:
        fh.write(json.dumps(dict(kind='header', split=a.split, model=str(a.model), longest_edge=le, contract=contract.get('contract', cd.CONTRACT), query_pixel_given=query, shard_index=a.shard_index, shard_count=a.shard_count, rows=len(rows))) + '\n')
        for n, row in enumerate(rows, 1):
            mi, depth, valid = inputs_for(proc, model, root, row, le, query=query)
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.inference_mode(): o = model.generate(**mi, max_new_tokens=a.max_new_tokens, do_sample=False, temperature=None, top_p=None)
            torch.cuda.synchronize(); lat = time.perf_counter() - t
            raw = proc.batch_decode(o[:, mi['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(kind='prediction', id=row['id'], raw=raw, latency_s=lat)) + '\n'); fh.flush()
            if n % 25 == 0 or n == len(rows): print(f'shard {a.shard_index}: {n}/{len(rows)}', flush=True)


def score(a):
    root = a.dataset.resolve(); run = a.run.resolve()
    if a.output.exists(): raise ValueError('choose a new report path')
    recs = {}; headers = []
    for f in sorted(run.glob('predictions-shard-*.jsonl')):
        for line in f.read_text().splitlines():
            it = json.loads(line)
            if it['kind'] == 'header': headers.append(it)
            else: recs[it['id']] = it
    split = headers[0]['split']; index = {r['id']: r for r in cd.read_jsonl(root / 'index.jsonl') if r['split'] == split}
    rows = [index[i] for i in recs]; preds = {}; fails = defaultdict(int); per = []; depth_err = []; consist_pred = []; consist_query = []
    for r in rows:
        raw = recs[r['id']]['raw']; ans, d, fail = cd.parse_output(raw)
        rec = dict(id=r['id'], raw=raw, decoded=ans, pred_depth_m=d, failure=fail, truth=r['answer'], latency_s=recs[r['id']]['latency_s'])
        if fail: fails[fail] += 1
        else:
            preds[r['id']] = ans
            if ans['status'] == 'localized' and d is not None:
                _, depth, valid = cd.load_frame(root, r)
                zp = cd.depth_at(depth, valid, *ans['cut_point_uv']); zq = cd.depth_at(depth, valid, *r['query_pixel_uv'])
                rec.update(native_depth_at_pred_m=zp, native_depth_at_query_m=zq)
                if zp is not None: consist_pred.append((abs(d - zp), r['answer']['status']))
                if zq is not None: consist_query.append((abs(d - zq), r['answer']['status']))
                if r['answer']['status'] == 'localized':
                    zl = cd.depth_at(depth, valid, *r['answer']['cut_point_uv'])
                    if zl is not None: depth_err.append(abs(d - zl)); rec['depth_error_m'] = abs(d - zl)
        per.append(rec)
    lat = sorted(x['latency_s'] for x in recs.values())
    def auc(pairs):
        pos = np.array([v for v, s in pairs if s == 'abstain']); neg = np.array([v for v, s in pairs if s == 'localized'])
        return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else None
    rep = dict(state='validation_generation_evaluation_2D_plus_depth', split=split, model=headers[0]['model'], contract=cd.CONTRACT, longest_edge=headers[0]['longest_edge'],
        rows_scored=len(rows), failures=dict(fails), latency_s=dict(p50=statistics.median(lat), p95=lat[min(len(lat) - 1, int(round(.95 * (len(lat) - 1))))]),
        overall=_metrics(rows, preds, root), by_truth_status={}, by_family={},
        depth=dict(predicted_vs_native_at_label_median_m=float(np.median(depth_err)) if depth_err else None, n=len(depth_err),
                   within_2cm=float(np.mean(np.array(depth_err) <= .02)) if depth_err else None,
                   pred_depth_vs_native_at_pred_pixel_auc_occluded=auc(consist_pred), pred_depth_vs_native_at_query_auc_occluded=auc(consist_query)))
    for name, key in (('by_truth_status', lambda r: r['answer']['status']), ('by_family', lambda r: r['source_plant_family'])):
        b = defaultdict(list)
        for r in rows: b[key(r)].append(r)
        rep[name] = {k: _metrics(v, preds, root) for k, v in sorted(b.items())}
    write_json(a.output, rep); write_json(a.output.with_name(a.output.stem + '.per_example.json'), per)
    o = rep['overall']; s = {k: o.get(k) for k in ('samples', 'invalid_predictions')}; s.update(o.get('all_rows_invalid_counted_as_failure', {})); s['depth'] = rep['depth']; s['failures'] = rep['failures']
    print(json.dumps(s, indent=1))


def probe(a):
    import torch
    root = a.dataset.resolve(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((a.model / 'grounding_adapter.json').read_text()) if (a.model / 'grounding_adapter.json').exists() else {}
    le = contract.get('longest_edge', a.longest_edge); query = contract.get('query_pixel_given', True); proc, model = load(a.model); tok = proc.tokenizer
    rows = rows_for(root, a.split, a.shard_index, a.shard_count, a.limit, a.ids_file)
    prefix_text = '```json\n[{"point_3d":'
    with open(out / f'probe-shard-{a.shard_index}-of-{a.shard_count}.jsonl', 'w') as fh:
        for n, row in enumerate(rows, 1):
            base_text = proc.apply_chat_template(cd.messages(row, query=query), tokenize=False, add_generation_prompt=True)
            scores = {}
            for val in (' [', ' null'):
                mi, _, _ = inputs_for(proc, model, root, row, le, text_override=base_text + prefix_text + val)
                pre, _, _ = inputs_for(proc, model, root, row, le, text_override=base_text + prefix_text)
                ids = mi['input_ids']; k = pre['input_ids'].shape[1]
                if not torch.equal(ids[:, :k], pre['input_ids']): raise RuntimeError('prefix tokenization unstable')
                with torch.inference_mode(): lp = torch.log_softmax(model(**mi).logits[0, :-1].float(), dim=-1)
                tgt = ids[0, 1:]; scores[val] = float(lp[k - 1:, :].gather(1, tgt[k - 1:, None]).sum())
            fh.write(json.dumps(dict(id=row['id'], truth=row['answer']['status'], logp_null=scores[' null'], logp_point=scores[' ['], margin=scores[' null'] - scores[' ['])) + '\n'); fh.flush()
            if n % 50 == 0 or n == len(rows): print(f'probe shard {a.shard_index}: {n}/{len(rows)}', flush=True)


def summarize_probe(a):
    recs = [json.loads(l) for f in sorted(a.run.glob('probe-shard-*.jsonl')) for l in f.read_text().splitlines()]
    y = np.array([r['truth'] == 'abstain' for r in recs]); m = np.array([r['margin'] for r in recs]); pos, neg = m[y], m[~y]
    auc = float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())
    ba = lambda t: float(((m[y] > t).mean() + (m[~y] <= t).mean()) / 2)
    rep = dict(n=len(recs), auc_abstain_vs_localized=auc, greedy_margin0=dict(balanced_accuracy=ba(0.), abstain_rate=float((m > 0).mean())),
               best_threshold=dict(balanced_accuracy=max(ba(t) for t in np.quantile(m, np.linspace(0, 1, 201)))))
    a.output.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('generate', 'probe'):
        g = sub.add_parser(name); g.add_argument('--dataset', type=Path, required=True); g.add_argument('--model', type=Path, required=True); g.add_argument('--output', type=Path, required=True)
        g.add_argument('--split', default='validation'); g.add_argument('--shard-index', type=int, default=0); g.add_argument('--shard-count', type=int, default=1)
        g.add_argument('--limit', type=int, default=None); g.add_argument('--ids-file', type=Path, default=None); g.add_argument('--longest-edge', type=int, default=640); g.add_argument('--max-new-tokens', type=int, default=128)
    s = sub.add_parser('score'); s.add_argument('--dataset', type=Path, required=True); s.add_argument('--run', type=Path, required=True); s.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('summarize-probe'); q.add_argument('--run', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    {'generate': generate, 'score': score, 'probe': probe, 'summarize-probe': summarize_probe}[a.cmd](a)


if __name__ == '__main__': main()
