"""Explicit reset8 original-camera contract; adaptation controls are never TRAIN."""
from pathlib import Path
import ast
import importlib.util
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from . import native848_original_direct_plan_v2 as original

SCHEMA = 'greenhouse.original848_short_camera_batch_plan.v1'
SAMPLE_SCHEMA = 'greenhouse.original848_short_camera_sample.v1'
RESULT_STATE = 'original848_short_batch_captured_pending_independent_audit'
PROFILE = 'original848_cached_reset8_after_product56_native_quality.v1'
BUDGET = 8
RESET_API = 'omni.usd.get_context().reset_renderer_accumulation'


def implementation_bindings():
    root = Path(__file__).resolve().parent
    roots = (root.parent, root.parents[1])
    pending = [root / ('native848_original_short_' + name + '_v1.py')
        for name in ('plan', 'worker', 'audit', 'admission')]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        found[str(path)] = sha256(path)
        base = next(p for p in roots if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.' * node.level + (node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name] + [name + '.' + a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in roots:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate / '__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def bound_json(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}
        and Path(binding['path']).is_absolute() and len(binding['sha256']) == 64
        and sha256(binding['path']) == binding['sha256'], 'Missing or changed sealed evidence')
    return read_json(binding['path'])


def qualification(plan):
    q = bound_json(plan['qualification_evidence'])
    require(q['schema'] == 'greenhouse.native848_short_profile_qualification.v1'
        and q['selected_budget_subframes'] == BUDGET
        and q['diagnostic_profile_qualified'] is True
        and q['qualified_for_original_adaptation'] is True
        and q['production_profile_qualified'] is False
        and q['training_approved'] is False and q['accepted_training_increment'] == 0,
        'Diagnostic qualification does not authorize original adaptation')
    verify_bindings(q['source_bindings'])
    from .native848_reset_probe_v2 import SETTINGS, checked_subframe_floor
    require(set(q['render_settings']) == set(SETTINGS)
        and q['render_settings']['/rtx/rendermode'] == 'RealTimePathTracing',
        'Exact diagnostic render settings required')
    checked_subframe_floor(q['render_settings'], BUDGET)
    require(plan['render_settings'] == q['render_settings'], 'Changed qualified renderer settings')
    return q


def adaptation_evidence(plan, *, full=False):
    if plan['mode'] == 'adaptation':
        require(plan['original_adaptation_evidence'] is None, 'Adaptation cannot claim its own prior approval')
        return None
    q = bound_json(plan['original_adaptation_evidence'])
    require(q['schema'] == 'greenhouse.original848_short_adaptation_qualification.v1'
        and q['profile'] == PROFILE and q['selected_budget_subframes'] == BUDGET
        and q['runtime_adaptation_passed'] is True and q['individual_visual_review_passed'] is True
        and q['qualified_for_original_production'] is True and q['training_approved'] is False
        and q['accepted_training_increment'] == 0
        and q['qualification_evidence'] == plan['qualification_evidence'], 'Original adaptation approval required')
    verify_bindings(q['source_bindings'])
    audit_path, reviews_path = Path(q['audit_path']).resolve(), Path(q['visual_reviews_path']).resolve()
    require(q['source_bindings'].get(str(audit_path)) == sha256(audit_path) == q['audit_sha256']
        and q['source_bindings'].get(str(reviews_path)) == sha256(reviews_path) == q['visual_reviews_sha256'],
        'Original adaptation raw audit/reviews must be bound')
    audit = read_json(audit_path)
    require(audit['schema'] == 'greenhouse.original848_short_buffer_geometry_audit.v1'
        and audit['mode'] == 'adaptation' and audit['records'] == []
        and audit['runtime_adaptation']['passed'] is True and len(audit['adaptation_records']) == 3,
        'Complete excluded A-B-A raw replay required')
    verify_bindings(audit['source_bindings'])
    reviews = read_json(reviews_path)
    for row in audit['adaptation_records']:
        require(row['capture_role'] == 'adaptation_control' and row['actual_automated_criteria_passed'] is True
            and any(v['sample_id'] == row['sample_id'] and v['decision'] == 'accept'
                and v['rgb_sha256'] == row['rgb_sha256'] and v['source_sample_sha256'] == row['sample_sha256']
                and v['full_native_image_inspected'] is True
                and v['unscaled_lossless_association_crop_inspected'] is True for v in reviews),
            'Every original adaptation image needs actual visual acceptance')
    if full:
        from .native848_original_short_audit_v1 import audit_capture
        trial = Path(q['adaptation_trial']).resolve()
        intent = read_json(trial / 'intent.json')
        require(audit_path == trial / 'audit.json' and not (trial / 'failure.json').exists(), 'Changed adaptation trial')
        require(audit_capture(trial / 'capture', plan_path=intent['plan_path'], plan_sha256=intent['plan_sha256'],
            result_sha256=sha256(trial / 'capture/result.json')) == audit, 'Original adaptation replay differs')
    return q


def schedule_check(plan, cache):
    rows = plan['schedule']
    require(isinstance(rows, list) and 1 <= len(rows) <= 30 and len(rows) == plan['max_frames']
        and all(set(r) == {'sample_id', 'source_pose_id', 'capture_role'} for r in rows), 'Invalid short capture schedule')
    names = [r['sample_id'] for r in rows]
    source = [r['source_pose_id'] for r in rows]
    require(len(set(names)) == len(names) and all(n and Path(n).name == n and n not in ('.', '..') for n in names),
        'Unique safe observation IDs required')
    by = {r['sample_id']: r for r in cache['records']}
    require(set(source) <= set(by), 'Unknown source pose')
    if plan['mode'] == 'adaptation':
        require(len(rows) == 3 and source[0] == source[2] != source[1]
            and all(r['capture_role'] == 'adaptation_control' for r in rows)
            and plan['selected_sample_ids'] == [], 'Adaptation is exactly three excluded A-B-A controls')
        require(not np.allclose(by[source[0]]['camera_to_world_usd_row_vectors'],
            by[source[1]]['camera_to_world_usd_row_vectors'], atol=1e-9, rtol=0), 'A and B cameras must differ')
    else:
        require(len(set(source)) == len(source) and all(r['capture_role'] == 'production' for r in rows)
            and source == plan['selected_sample_ids'], 'Production cannot duplicate source poses or contain controls')
    return rows


def prepare(cache_path, schedule, qualification_evidence, *, mode='adaptation', original_adaptation_evidence=None):
    path = Path(cache_path).resolve()
    q = bound_json(qualification_evidence)
    plan = dict(schema=SCHEMA, cache_path=str(path), cache_sha256=sha256(path), mode=mode,
        schedule=schedule, selected_sample_ids=[r['source_pose_id'] for r in schedule if r['capture_role'] == 'production'],
        resolution=[848, 408], profile=PROFILE, render_budget_subframes=BUDGET,
        qualification_evidence=qualification_evidence, original_adaptation_evidence=original_adaptation_evidence,
        render_settings=q['render_settings'], warmup_request_subframes=[8]*7, warmup_per_product=1,
        reset_api=RESET_API, max_frames=len(schedule), instance_backend='fast',
        full_greenhouse_stage_count=1, capture_render_product_count=1, broad_pose_search=False,
        simulated_motion=False, generated_geometry_used=False, source_cap_reset=False, training_approved=False,
        long_reference_probe_ids=[], implementation_bindings=implementation_bindings())
    check(plan)
    return plan


def check(plan, *, full=False):
    require(plan['schema'] == SCHEMA and plan['mode'] in ('adaptation', 'production')
        and plan['resolution'] == [848, 408] and plan['profile'] == PROFILE
        and type(plan['render_budget_subframes']) is int and plan['render_budget_subframes'] == BUDGET
        and plan['warmup_request_subframes'] == [8]*7 and plan['warmup_per_product'] == 1
        and plan['reset_api'] == RESET_API and plan['instance_backend'] == 'fast'
        and plan['full_greenhouse_stage_count'] == plan['capture_render_product_count'] == 1
        and plan['long_reference_probe_ids'] == []
        and all(plan[k] is False for k in ('broad_pose_search', 'simulated_motion', 'generated_geometry_used',
            'source_cap_reset', 'training_approved')), 'Changed original short scope')
    require(plan['implementation_bindings'] == implementation_bindings(), 'Changed original short implementation')
    verify_bindings(plan['implementation_bindings'])
    qualification(plan)
    require(sha256(plan['cache_path']) == plan['cache_sha256'], 'Changed original camera cache')
    cache = read_json(plan['cache_path'])
    schedule_check(plan, cache)
    # Invoke the existing immutable anatomy/FK/cache contract with unique source IDs.
    # This is CPU validation of the original cache, not fixed56 capture approval.
    unique_ids = list(dict.fromkeys(r['source_pose_id'] for r in plan['schedule']))
    legacy = original.prepare(plan['cache_path'], unique_ids)
    result = original.check(legacy, full=full)
    adaptation_evidence(plan, full=full)
    return result
