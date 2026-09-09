"""Re-screen a previous visual bundle without migrating its approvals.

Revalidates immutable source audits, applies the current task contract, and
creates a fresh pending bundle plus a per-image comparison. Excluded sources
remain on disk. This is not training approval or physical-cut certification.
"""
from __future__ import annotations

import argparse
import html
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_review import read_json, require, safe_file, write_json
from .depth_preview import sha256
from .query_visibility import QueryVisibility
from .training_contract import CONTRACT, contract_hash
from .training_export import gather, check_release_rows
from .training_release_review import SCHEMA, POLICY, identity, task_review_canvas


def write_index(output, comparisons):
    """Lightweight local one-by-one browser; it never records approvals."""
    output=Path(output); parts=['<!doctype html><meta charset="utf-8"><title>Query re-screen v3</title>',
        '<style>body{background:#161a21;color:#eee;font:16px sans-serif;max-width:1200px;margin:24px auto}',
        'a{color:#75d9ff}details{border:1px solid #555;padding:12px;margin:10px 0}summary{cursor:pointer}',
        'img{max-width:100%;height:auto}button{margin:8px;padding:8px}</style>',
        f'<h1>One-by-one query re-screen</h1><p>{sum(r["current_state"]!="excluded" for r in comparisons)}/{len(comparisons)} eligible pending new review. ',
        'Eligibility is not visual approval. No old approvals are migrated. Original RGB/native Isaac depth are unchanged.</p>',
        '<button onclick="filterRows(false)">All rows</button><button onclick="filterRows(true)">Excluded only</button>',
        '<script>function filterRows(excluded){document.querySelectorAll("details").forEach(x=>x.hidden=excluded&&x.dataset.state!=="excluded")}</script>']
    for n,r in enumerate(comparisons,1):
        e=html.escape; rgb=(Path(r['source_sample'])/'inputs/rgb.png').as_uri()
        parts.extend([f'<details data-state="{e(r["current_state"])}"><summary>{n:02d}. {e(r["id"])} — {e(r["current_state"])}</summary>',
            f'<p>Previous query: {e(str(r["previous_query"]))}; current: {e(str(r["current_query"]))}. Reason: {e(str(r["reason"]))}</p>',
            f'<p>Previous query warnings: {e(str(r["previous_query_usability"]["reasons"]))}</p>',
            f'<p><a href="{e(rgb,quote=True)}">Unmodified model RGB</a> | <a href="{e(r["card"],quote=True)}">Full review card</a></p>',
            f'<img loading="lazy" src="{e(r["card"],quote=True)}" alt="{e(r["id"],quote=True)} review evidence"></details>'])
    with (output/'index.html').open('x',encoding='utf-8') as stream: stream.write('\n'.join(parts))


def rescreen(previous_bundle, output):
    previous_bundle, output=Path(previous_bundle).resolve(),Path(output).resolve()
    require(not output.exists(),'Choose a new rescreen directory')
    require(not output.is_relative_to(previous_bundle.parent),'Keep new review separate from previous evidence')
    old=read_json(previous_bundle)
    require(old.get('schema_version') in ('greenhouse.grounding_stratified_visual_QA.v1',SCHEMA),'Unknown previous bundle')
    audits=old['source_audits_sha256']
    for p,h in audits.items(): require(sha256(p)==h,'Changed original source audit')
    for e in old['entries']:
        require(sha256(safe_file(previous_bundle.parent,e['card']))==e['card_sha256'],'Changed original review card')
    require(len({e['id'] for e in old['entries']})==len(old['entries']),'Duplicate prior entry')
    gathered=gather(list(audits),output=output)
    candidates={r['id']:r for r in gathered['candidates']}
    sources={r['id']:(r['directory'],r['metadata'],None) for r in candidates.values()}
    for r in gathered['exclusions']:
        d=Path(r['source_sample']); m=read_json(d/'sample.json')
        sid=r['family']+'_'+m['files']['inputs/rgb.png']['sha256'][:20]
        sources.setdefault(sid,(d,m,r['reason']))
    output.mkdir(parents=True)
    comparisons=[]; entries=[]
    for e in old['entries']:
        require(e['id'] in sources,'Original sample missing from audited sources')
        directory,meta,reason=sources[e['id']]
        require(sha256(directory/'sample.json')==e['source_sample_sha256'] and
                meta['files']['inputs/rgb.png']['sha256']==e['rgb_sha256'],'Changed original sample identity')
        rgb=np.asarray(Image.open(directory/'inputs/rgb.png'))
        depth=np.load(directory/'inputs/depth_m.npy',allow_pickle=False)
        valid=np.asarray(Image.open(directory/'inputs/depth_valid.png'))==255
        target=np.asarray(Image.open(directory/'supervision/target_visible.png'))==255
        old_usability=QueryVisibility(rgb,target).inspect(e['query_pixel_uv'])
        current=candidates.get(e['id']); card=e['id']+'.png'
        display=current if current else dict(**identity(e),metadata=meta)
        if current is None:
            display['answer']=dict(status='excluded',reason=reason)
        task_review_canvas(display,(rgb,depth,valid,target)).save(output/card)
        if current:
            entries.append(dict(**identity(current),card=card,card_sha256=sha256(output/card)))
        comparisons.append(dict(id=e['id'],target_id=e['target_id'],source_sample=str(directory),
            previous_query=e['query_pixel_uv'],previous_answer=e['answer'],previous_query_usability=old_usability,
            current_state='eligible_pending_new_visual_review' if current else 'excluded',reason=reason,
            current_query=current['query_pixel_uv'] if current else None,
            current_answer=current['answer'] if current else None,
            current_query_usability=current['label']['query_usability'] if current else None,
            query_changed=bool(current and current['query_pixel_uv']!=e['query_pixel_uv']),
            card=card,card_sha256=sha256(output/card)))
    now=datetime.now(timezone.utc).isoformat()
    write_json(output/'bundle.json',dict(schema_version=SCHEMA,state='pending_actual_visual_inspection',policy=POLICY,
        created_utc=now,entries=entries,source_audits_sha256=audits,training_release_approved=False,human_review_performed=False))
    summary=dict(schema_version='greenhouse.grounding_rescreen.v1',created_utc=now,
        state='rescreened_not_visually_approved_not_a_release',previous_bundle=str(previous_bundle),
        previous_bundle_sha256=sha256(previous_bundle),contract=CONTRACT,contract_sha256=contract_hash(),
        inspected_prior_bundle_size=len(comparisons),surviving_prior_rows=len(entries),
        excluded_prior_rows=len(comparisons)-len(entries),changed_queries=sum(r['query_changed'] for r in comparisons),
        old_query_gate_failures=sum(not r['previous_query_usability']['passed'] for r in comparisons),
        source_audit_count=len(audits),all_source_candidates=len(candidates),all_source_exclusions=len(gathered['exclusions']),
        all_source_exclusion_counts=dict(Counter(r['reason'] for r in gathered['exclusions'])),
        all_source_coverage=check_release_rows(list(candidates.values())),
        old_approvals_migrated=False,original_images_depth_or_reviews_modified=False,
        implementation_sha256={p:sha256(Path(__file__).with_name(p)) for p in
            ('training_rescreen.py','training_contract.py','query_visibility.py','dataset_review.py','training_release_review.py')})
    write_json(output/'comparison.json',comparisons)
    write_json(output/'summary.json',summary)
    write_index(output,comparisons)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-bundle',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv); result=rescreen(a.previous_bundle,a.output)
    print('RESCREEN_COMPLETE_PENDING_VISUAL_REVIEW',result['surviving_prior_rows'],'eligible /',
          result['excluded_prior_rows'],'excluded',flush=True)


if __name__=='__main__': main()
