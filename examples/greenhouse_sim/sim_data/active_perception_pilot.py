"""Prepare a bounded task-v4 REVIEW pilot from existing native captures.

No rendering, image edits, depth synthesis, human decisions or training export.
Matched frozen views are contrasts, not executed reveal demonstrations.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .active_perception_contract import (TASK_ID, CONTRACT, SYSTEM_PROMPT, contract_hash,
    motion_capture_gaps, user_prompt, validate_evidence)
from .capture_contract import fingerprint
from .dataset_review import read_json, require, safe_file, write_json
from .depth_preview import sha256
from .query_visibility import QueryVisibility
from .training_review_gui import TrainingReviewApp, DEFAULT_BUNDLE, DEFAULT_SUGGESTIONS

SCHEMA = 'greenhouse.active_perception_review_pilot.v1'


def proposed_answer(kind, original=None):
    visible = kind == 'visible'
    negative = kind == 'invalid_candidate'
    return dict(target_state='ineligible' if negative else 'eligible' if visible else 'unknown',
        cut_visibility='not_applicable' if negative else original['visibility'] if visible else 'occluded',
        decision='reject' if negative else 'localize' if visible else 'inspect',
        cut_point_uv=original['cut_point_uv'] if visible else None,
        next_action='skip_candidate' if negative else 'inspect_cut_region' if visible else 'change_viewpoint',
        execution_feasibility='unverified',
        evidence_note=('The indicated object is a protected organ, not an eligible petiole.' if negative else
                       'The target petiole and proximal cut region are identifiable in this view.' if visible else
                       'The cut region is obscured; target eligibility and the cutting position need further inspection.'))


def read_source(app, sid):
    app._check_source(sid)
    source = app.sources[sid]
    metadata_path = next(p for p in source['files'] if p.name == 'sample.json')
    return metadata_path.parent, read_json(metadata_path)


def negative_query(directory, metadata, audit, organ):
    """Select an actual visible renderer-identity interior, never a random label.

    This remains a draft until someone judges RGB anatomy. Unmapped background
    is NEVER interpreted as no eligible target.
    """
    paths = {}
    for name in ('supervision/component_id.npy', 'supervision/identities.json'):
        path = safe_file(directory, name); expected = metadata['files'][name]['sha256']
        require(audit['bindings_sha256'].get(str(path)) == expected and sha256(path) == expected,
                'Negative query needs independently bound native identity')
        paths[str(path)] = expected
    ids = np.load(directory/'supervision/component_id.npy', allow_pickle=False)
    require(ids.shape == (408, 848) and np.issubdtype(ids.dtype, np.integer), 'Invalid native component buffer')
    catalogue = read_json(directory/'supervision/identities.json')['component_catalogue']
    rgb = np.asarray(Image.open(directory/'inputs/rgb.png').convert('RGB'))
    depth = np.load(directory/'inputs/depth_m.npy', allow_pickle=False)
    valid = np.asarray(Image.open(directory/'inputs/depth_valid.png')) == 255
    require(depth.shape == ids.shape and depth.dtype == np.float32, 'Expected saved native optical-Z')
    counts = Counter(ids.ravel().tolist())
    candidates = sorted((c for c in catalogue if c['organ_type'] == organ and counts[c['component_index']] >= 100),
                        key=lambda c: (-counts[c['component_index']], c['component_index']))
    for c in candidates[:6]:
        mask = ids == c['component_index']; screen = QueryVisibility(rgb, mask)
        interior = screen.interior.copy(); interior[:16] = 0; interior[-16:] = 0; interior[:, :16] = 0; interior[:, -16:] = 0
        interior[~valid | ~np.isfinite(depth) | (depth <= 0)] = 0
        for _ in range(8):
            y, x = np.unravel_index(np.argmax(interior), interior.shape)
            if interior[y, x] < 1.5: break
            query = [float(x)+.5, float(y)+.5]; checked = screen.inspect(query)
            if checked['passed']:
                return dict(query=query, component=c, usability=checked, bindings=paths)
            interior[max(0,y-8):y+9, max(0,x-8):x+9] = 0
    return None


def prepare(bundle, output, *, suggestions=None, max_pairs=8, max_negatives=8, max_unknown_regions=0):
    require(type(max_pairs) is int and 1 <= max_pairs <= 20 and type(max_negatives) is int and 0 <= max_negatives <= 40,
            'Pilot bounds: 1-20 pairs and 0-40 negatives')
    require(type(max_unknown_regions) is int and 0 <= max_unknown_regions <= 20, 'Region pilot bound: 0-20')
    output = Path(output).resolve(); require(not output.exists(), 'Choose a new pilot directory')
    app = TrainingReviewApp(bundle, suggestions)
    state = app.state(); state_hash = fingerprint(state)
    eligible, excluded = {}, []
    for s in state['samples']:
        reasons = []
        if s['source_blocks']: reasons.append('active_source_hold')
        if s['decision'] and s['decision']['decision'] != 'accept': reasons.append('human_task_hold')
        if s['previous_decision'] and s['previous_decision']['decision'] != 'accept': reasons.append('legacy_task_hold')
        if s['suggestion'] and s['suggestion']['suggested_decision'] == 'hold':
            final = s['decision'] or {}
            resolved = (final.get('decision') == 'accept'
                        and (final.get('suggestion_context') or {}).get('sha256') == s['suggestion']['sha256'])
            if not resolved: reasons.append('advisory_hold_conservatively_excluded')
        if reasons: excluded.append(dict(id=s['id'], reasons=reasons))
        else: eligible[s['id']] = s
    groups = defaultdict(list); sources = {}; family_splits = {}; image_splits = {}
    for sid in eligible:
        e = app.entries[sid]; family = e['source_plant_family']
        require(family_splits.setdefault(family, e['split']) == e['split'], 'Family split leakage')
        require(image_splits.setdefault(e['rgb_sha256'], e['split']) == e['split'], 'Image split leakage')
        directory, meta = read_source(app, sid); sources[sid] = directory, meta
        anatomy = fingerprint(dict(proposal=meta['supervision']['cut_region_proposal'],
                                  plant_to_world=meta['supervision']['plant_to_world_usd_row_vectors']))
        groups[family, e['target_id'], anatomy].append(sid)
    pairs, used_families = [], set()
    for (family, target, anatomy), ids in sorted(groups.items()):
        visible = sorted(s for s in ids if app.entries[s]['answer']['status'] == 'localized')
        hidden = sorted(s for s in ids if app.entries[s]['answer']['status'] == 'abstain')
        if visible and hidden and family not in used_families and len(pairs) < max_pairs:
            pairs.append(dict(id='contrast_'+str(len(pairs)+1).zfill(3), family=family, target=target,
                anatomy_sha256=anatomy, visible=visible[0], occluded=hidden[0],
                split=family_splits[family], temporal_episode=False, action_validated=False))
            used_families.add(family)
    require(pairs, 'No unheld same-anatomy visible/occluded contrasts; new captures required')
    examples, bindings = [], {str(app.bundle_path): app.bundle_hash}
    bindings.update({str(p): h for p, h in app.suggestions.bindings.items()})

    def add(sid, kind, *, negative=None, contrast=None, region=None):
        e = app.entries[sid]; directory, meta = sources[sid]
        query = negative['query'] if negative else e['query_pixel_uv']
        answer = proposed_answer(kind, e['answer'])
        if region:
            query = None
            answer.update(cut_visibility='unknown', evidence_note='Foliage obscures the specified region. I cannot establish whether an eligible cutting target is behind it; inspect another view.')
        evidence = dict(target_identifiable=kind == 'visible', cut_region_identifiable=kind == 'visible',
                        eligibility_observable=kind == 'visible')
        if negative: evidence['identified_ineligible_organ'] = negative['component']['organ_type']
        mode = 'scene_region' if region else 'candidate_query'
        if region: evidence['occluder_identifiable'] = True
        validate_evidence(answer, evidence, mode=mode)
        sid4 = sid+'__'+(negative['component']['organ_type'] if negative else kind)
        examples.append(dict(id=sid4, task_id=TASK_ID, task_contract_sha256=contract_hash(),
            kind=kind, source_id=sid, source_plant_family=e['source_plant_family'], split=e['split'], contrast_id=contrast,
            model_input=dict(rgb_path=str(directory/'inputs/rgb.png'), query_pixel_uv=query,
                             **({'region_xyxy':region['region_xyxy']} if region else {}),
                             instruction=user_prompt(region=region['region_xyxy']) if region else user_prompt(query=query), history=[]),
            proposed_answer=answer, proposed_observable_evidence=evidence,
            private_evaluator=dict(world_target_state='unknown' if region else 'ineligible' if negative else 'eligible_under_prototype_rule',
                original_target_id=e['target_id'], nominal_world_m=None if negative or region else meta['supervision']['nominal_world_m'],
                observed_organ=negative['component'] if negative else region['foreground_component'] if region else None),
            native_sidecars=dict(depth_path=str(directory/'inputs/depth_m.npy'), depth_valid_path=str(directory/'inputs/depth_valid.png'),
                calibration=meta['calibration'], robot_snapshot=meta['robot_snapshot'], synchronization=meta['synchronization']),
            query_usability=negative['usability'] if negative else None,
            motion_readiness=motion_capture_gaps(meta),
            review_status='pending_new_task_visual_review', human_confirmation=False,
            training_eligible=False, physical_execution_approved=False))
        bindings.update({str(p): h for p, h in app.sources[sid]['files'].items()})
        bindings[str(app.sources[sid]['audit_path'])] = e['source_audit_sha256']
        if negative: bindings.update(negative['bindings'])
        if region: bindings.update(region['bindings'])

    for pair in pairs:
        add(pair['visible'], 'visible', contrast=pair['id'])
        add(pair['occluded'], 'occluded', contrast=pair['id'])
    negative_count = 0
    for pair_index, pair in enumerate(pairs):
        if negative_count >= max_negatives: break
        sid = pair['visible']; directory, meta = sources[sid]
        organs = ('main_stem', 'fruit', 'peduncle')
        start = pair_index % len(organs)
        for organ in organs[start:]+organs[:start]:
            candidate = negative_query(directory, meta, app.sources[sid]['audit'], organ)
            if candidate:
                add(sid, 'invalid_candidate', negative=candidate); negative_count += 1
                break
    region_count = 0
    if max_unknown_regions:
        from .active_perception_regions import propose
        for pair in pairs:
            if region_count >= max_unknown_regions: break
            for sid in (pair['occluded'], pair['visible']):
                directory, meta = sources[sid]
                region = propose(directory, meta, app.sources[sid]['audit'])
                if region:
                    add(sid, 'uncertain_region', region=region)
                    region_count += 1
                    break
    requests = [dict(case=case, state='not_collected', training_eligible=False, required_evidence=proof)
        for case, proof in (
            ('no_eligible_target_in_region', 'Fully observed bounded region; reviewed absence, not unmapped background'),
            ('unidentifiable_dense_foliage', 'Unknown eligibility; null cut; RGB readability review'),
            ('successful_viewpoint_reveal', 'Executed collision-checked camera motion and synchronized before/after frames'),
            ('successful_left_arm_reveal', 'Validated foliage contact, native timed RGB-D, left-hand transition to target grasp'),
            ('failed_reveal_and_recovery', 'Executed attempt with no visibility gain; bounded recovery or stop'),
        ('visible_but_execution_blocked', 'Measured IK/collision constraint; do not label anatomically invalid'))]
    if region_count:
        requests[1].update(state='static_leaf_obscured_region_drafts_prepared', count=region_count)
    # Hash all prior decision records used to exclude sources. Do not copy their approvals to v4.
    for folder in (app.records, app.bundle_path.parent/'human_decisions'):
        for path in folder.glob('*.json'): bindings[str(path.resolve())] = sha256(path)
    for source in app.sources.values():
        for path in (source['audit_path'].parent/'records').glob('*.json'):
            bindings[str(path.resolve())] = sha256(path)
    require(fingerprint(app.state()) == state_hash, 'Review history changed during pilot preparation; rerun')
    for name, expected in bindings.items(): require(sha256(name) == expected, 'Pilot source changed')
    result = dict(schema=SCHEMA, task_id=TASK_ID, contract=CONTRACT, task_contract_sha256=contract_hash(),
        system_prompt=SYSTEM_PROMPT, created_utc=datetime.now(timezone.utc).isoformat(),
        state='review_only_static_contrasts_not_action_demonstrations', source_bundle_sha256=app.bundle_hash,
        examples=examples, contrasts=pairs, pending_capture_requests=requests, excluded_sources=excluded,
        counts=dict(Counter(e['kind'] for e in examples)), bindings_sha256=bindings,
        human_confirmation=False, training_eligible=False, physical_execution_approved=False,
        dynamic_episodes_collected=0)
    output.mkdir(parents=True)
    write_json(output/'pilot.json', result)
    write_json(output/'contract.json', dict(contract=CONTRACT, sha256=contract_hash(), system_prompt=SYSTEM_PROMPT))
    write_json(output/'capture_requests.json', requests)
    write_review(output, result)
    return result


def write_review(output, pilot):
    cards = []
    for e in pilot['examples']:
        rgb = Path(e['model_input']['rgb_path'])
        relative = Path(os.path.relpath(rgb, output)).as_posix()
        query = e['model_input']['query_pixel_uv']
        if query is not None:
            u,v = query
            marker = f'<circle cx="{u}" cy="{v}" r="5" fill="none" stroke="cyan" stroke-width="1.5"/>'
        else:
            x,y,r,b = e['model_input']['region_xyxy']
            marker = f'<rect x="{x}" y="{y}" width="{r-x}" height="{b-y}" fill="none" stroke="cyan" stroke-width="1.5"/>'
        cards.append(f'<section><h2>{escape(e["kind"])} | {escape(e["id"])}</h2>'
            f'<p>{escape(e["split"])} | {escape(e["model_input"]["instruction"])}</p>'
            f'<div class="frame"><img src="{escape(relative, quote=True)}" alt="Original robot-head RGB">'
            f'<svg viewBox="0 0 848 408">{marker}</svg></div>'
            f'<p><a href="{escape(relative, quote=True)}">Open untouched RGB</a> | Cyan is the review-only query or assessed region, NOT the cut.</p>'
            f'<pre>{escape(json.dumps(e["proposed_answer"], indent=2))}</pre>'
            '<p>Draft native-derived labels: visual review required. No copied human approval or execution claim.</p></section>')
    html = ('<!doctype html><meta charset="utf-8"><title>Active perception v4 pilot - REVIEW ONLY</title>'
        '<style>body{background:#f3f5f3;font:15px system-ui;max-width:1000px;margin:auto;padding:20px}'
        'section{background:white;border:1px solid #ccd;padding:18px;margin:20px 0}h2{font-size:17px;overflow-wrap:anywhere}'
        '.frame{position:relative}.frame img{width:100%;display:block}.frame svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}'
        'pre{white-space:pre-wrap}#markers:not(:checked)~main svg{display:none}</style>'
        '<h1>Task-v4 static perception pilot</h1><p>Review only. Different frozen views of matched anatomy are NOT a robot reveal episode. '
        'Invalid-object queries are not proof of an empty plant. Depth remains the original Isaac native array.</p>'
        '<input type="checkbox" id="markers"><label for="markers">Show candidate-query markers (original RGB first)</label>'
        '<main>'+''.join(cards)+'</main>')
    with (output/'review.html').open('x', encoding='utf-8') as stream: stream.write(html)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    p.add_argument('--suggestions', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-pairs', type=int, default=8)
    p.add_argument('--max-negatives', type=int, default=8)
    p.add_argument('--max-unknown-regions', type=int, default=8)
    a = p.parse_args(argv)
    advice = a.suggestions
    if advice is None and a.bundle.resolve() == DEFAULT_BUNDLE.resolve():
        require(DEFAULT_SUGGESTIONS.is_dir(), 'Default pilot requires the independent advisory assessment')
        advice = DEFAULT_SUGGESTIONS
    result = prepare(a.bundle, a.output, suggestions=advice, max_pairs=a.max_pairs, max_negatives=a.max_negatives, max_unknown_regions=a.max_unknown_regions)
    print(json.dumps(dict(output=str(a.output.resolve()), counts=result['counts'], dynamic_episodes_collected=0)), flush=True)


if __name__ == '__main__': main()
