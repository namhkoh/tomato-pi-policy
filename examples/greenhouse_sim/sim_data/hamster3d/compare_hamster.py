"""Compare a hamster-* evaluation against full-02 on the validation split (same scorer, same rows)."""
import json, sys, numpy as np
from pathlib import Path
R=Path('/workspace/nhkoh/tomato-vlm/runs'); name=sys.argv[1] if len(sys.argv)>1 else 'hamster-01'
def rep(n): return json.loads((R/f'eval-{n}'/'report.json').read_text())
def errs(n):
    pe=json.loads((R/f'eval-{n}'/'report.per_example.json').read_text()); key='decoded' if 'decoded' in pe[0] else 'decoded_pixels'
    return np.array([float(np.hypot(x[key]['cut_point_uv'][0]-x['truth']['cut_point_uv'][0],x[key]['cut_point_uv'][1]-x['truth']['cut_point_uv'][1])) for x in pe if x.get(key) and x[key]['cut_point_uv'] and x['truth']['cut_point_uv']])
cols=[('Full v2 (RGB, Qwen3-VL-8B)','full-02-validation','probe-full-02-validation'),(f'{name} (3D HAMSTER, RGB-D)',f'{name}-validation',f'probe-{name}-validation')]
reps={c[1]:rep(c[1]) for c in cols}; E={c[1]:errs(c[1]) for c in cols}
def P(c):
    p=R/c[2]/'summary.json'; return json.loads(p.read_text())['auc_abstain_vs_localized'] if p.exists() else None
def A(n,k): return reps[n]['overall']['all_rows_invalid_counted_as_failure'].get(k)
def ab(n): return reps[n]['by_truth_status']['abstain']['all_rows_invalid_counted_as_failure']['status_accuracy']
rows=[('Valid answers',lambda c:f"{reps[c[1]]['rows_scored']-reps[c[1]]['overall']['invalid_predictions']}/{reps[c[1]]['rows_scored']}"),
('Abstains correctly on occluded (346)',lambda c:f'{ab(c[1]):.3f}'),('Fabricates a point on occluded',lambda c:f"{A(c[1],'false_localization_rate_on_occluded'):.3f}"),
('Answers on visible (1,012)',lambda c:f"{A(c[1],'localized_answer_coverage'):.3f}"),('Balanced accuracy of the decision',lambda c:f"{(A(c[1],'localized_answer_coverage')+ab(c[1]))/2:.3f}"),
('Status-score AUC (probe)',lambda c:'n/a' if P(c) is None else f'{P(c):.3f}'),('Median error of answers',lambda c:f'{np.median(E[c[1]]):.1f} px'),
('Answers within 10 / 20 px',lambda c:f"{(E[c[1]]<=10).mean():.2f} / {(E[c[1]]<=20).mean():.2f}"),('Inside the 10-20 mm interval (2 px)',lambda c:f"{A(c[1],'projected_interval_hit_rate_2px'):.3f}"),
('Within 5 px of label',lambda c:f"{A(c[1],'success_within_5px'):.3f}"),('Latency p50',lambda c:f"{reps[c[1]]['latency_s']['p50']:.2f} s"),
('Predicted depth vs native at label (median, within 2 cm)',lambda c:'n/a' if 'depth' not in reps[c[1]] else f"{reps[c[1]]['depth']['predicted_vs_native_at_label_median_m']*100:.1f} cm, {reps[c[1]]['depth']['within_2cm']:.2f}"),
('AUC of |pred depth - native at pred pixel| for occluded',lambda c:'n/a' if 'depth' not in reps[c[1]] else f"{reps[c[1]]['depth']['pred_depth_vs_native_at_pred_pixel_auc_occluded']:.3f}")]
print('| Validation metric | '+' | '.join(c[0] for c in cols)+' |'); print('|---|'+'---:|'*len(cols))
for n,f in rows: print(f'| {n} | '+' | '.join(f(c) for c in cols)+' |')
print('\nfailures:',{c[1]:reps[c[1]]['failures'] for c in cols})
print('by family status acc:',{c[1]:{k:round(v['all_rows_invalid_counted_as_failure']['status_accuracy'],3) for k,v in reps[c[1]]['by_family'].items()} for c in cols})
