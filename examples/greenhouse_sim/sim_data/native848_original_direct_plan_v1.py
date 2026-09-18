"""Explicit original-only native848 cached-pose capture contract."""
from pathlib import Path
import ast
import importlib.util
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from . import native848_pair_plan_v2 as pair
from .native_view_pose import bounded_reference_root

SCHEMA = 'greenhouse.original848_direct_camera_batch_plan.v1'
SAMPLE_SCHEMA = 'greenhouse.original848_direct_camera_sample.v1'
RESULT_STATE = 'original848_direct_batch_captured_pending_independent_audit'
PROFILE = 'original848_direct_cached_poses_established56_every_frame.v1'


def implementation_bindings():
    root = Path(__file__).resolve().parent
    roots = (root.parent, root.parents[1])
    pending = [root/name for name in ('native848_original_direct_plan_v1.py',
        'native848_original_direct_worker_v1.py', 'native848_original_direct_audit_v1.py',
        'native848_original_direct_admission_v1.py')]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes()
        found[str(path)] = sha256(path)
        base = next(p for p in roots if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.'*node.level+(node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name]+[name+'.'+a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in roots:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def prepare(cache_path, selected_sample_ids):
    path = Path(cache_path).resolve()
    plan = dict(schema=SCHEMA, cache_path=str(path), cache_sha256=sha256(path),
        selected_sample_ids=list(selected_sample_ids), resolution=[848, 408], profile=PROFILE,
        max_frames=len(selected_sample_ids), instance_backend='fast', full_greenhouse_stage_count=1,
        capture_render_product_count=1, broad_pose_search=False, simulated_motion=False,
        generated_geometry_used=False, source_cap_reset=False, training_approved=False,
        long_reference_probe_ids=[], implementation_bindings=implementation_bindings())
    check(plan)
    return plan


def check(plan, *, full=False):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from .capture_contract import project
    from .native848_pair_audit_v2 import world_from_row
    from .collection_plan import load_plan
    require(plan['schema'] == SCHEMA and plan['resolution'] == [848, 408]
        and plan['profile'] == PROFILE and plan['instance_backend'] == 'fast'
        and type(plan['max_frames']) is int and 1 <= plan['max_frames'] <= 30
        and plan['full_greenhouse_stage_count'] == plan['capture_render_product_count'] == 1
        and plan['long_reference_probe_ids'] == []
        and all(plan[k] is False for k in ('broad_pose_search', 'simulated_motion',
            'generated_geometry_used', 'source_cap_reset', 'training_approved')), 'Changed original848 scope')
    require(plan['implementation_bindings'] == implementation_bindings(), 'Changed original848 implementation')
    verify_bindings(plan['implementation_bindings'])
    require(sha256(plan['cache_path']) == plan['cache_sha256'], 'Changed original camera cache')
    cache = read_json(plan['cache_path'])
    require(cache['schema'] == 'greenhouse.original848_cached_embodied_camera_poses.v1'
        and cache['resolution'] == [848, 408] and cache['generated_geometry_used'] is False
        and cache['frozen_original_assets'] is True and cache['training_approved'] is False,
        'Original-only cache required')
    verify_bindings(cache['source_bindings'])
    anchor_path = cache['anchor_pair_plan_path']
    require(sha256(anchor_path) == cache['anchor_pair_plan_sha256'], 'Changed reconstructed-scene anchor')
    anchor = read_json(anchor_path)
    pair.check(anchor, full=full)
    source_plan, reports = load_plan(anchor['source_collection_plan'])
    source_rows = {r['target_id']: r for job in source_plan['jobs'] for r in job['targets']
        if job['plant_family'] == anchor['source_family'] and job['split'] == 'train'}
    report = next(r for r in reports if r['plant_id'] == anchor['source_family'])
    by = {r['sample_id']: r for r in cache['records']}
    ids = plan['selected_sample_ids']
    require(len(by) == len(cache['records']) and len(ids) == len(set(ids)) == plan['max_frames']
        and set(ids) <= set(by), 'Invalid selected original poses')
    model = Rby1Kinematics()
    ref = anchor['expected_robot_snapshot']
    for name in ids:
        rec = by[name]
        target = rec['target_id']
        require(target in source_rows and rec['source_row'] == source_rows[target]
            and rec['source_target'] == target and rec['anchor_pair_plan_path'] == anchor_path
            and rec['anchor_pair_plan_sha256'] == cache['anchor_pair_plan_sha256'],
            'Original target or anchor differs')
        require(rec['source_collection_plan'] == anchor['source_collection_plan']
            and sha256(rec['source_collection_plan']) == rec['source_collection_plan_sha256'],
            'Original collection source changed')
        world = world_from_row(anchor, rec['source_row'])
        require(np.allclose(rec['target_world_m'], world['nominal_world_m'], atol=1e-9, rtol=0)
            and np.allclose(rec['interval_world_m'], world['interval_world_m'], atol=1e-9, rtol=0),
            'Original target geometry differs')
        root = bounded_reference_root(ref, rec['requested_spec'], rec['target_world_m'])
        require(np.allclose(root, np.asarray(rec['robot_root_to_world_usd_row_vectors']).T, atol=1e-9, rtol=0),
            'Cached original base differs')
        require(all(rec['joint_degrees'][k] == v for k, v in ref['joint_degrees'].items()
            if k not in ('head_0', 'head_1')), 'Original arm or torso changed')
        mount = np.asarray(rec['camera_to_head_column_vectors'])
        require(np.allclose(mount, ref['camera_to_head_column_vectors'], atol=1e-9, rtol=0), 'Camera mount differs')
        links = model.all_link_transforms(rec['joint_degrees'])
        camera = root @ links['link_head_2'] @ mount
        require(np.allclose(camera.T, rec['camera_to_world_usd_row_vectors'], atol=1e-9, rtol=0)
            and np.allclose(camera.T, rec['calibration']['camera_to_world_usd_row_vectors'], atol=1e-9, rtol=0),
            'Original cached camera FK differs')
        require(all(rec['calibration'][k] == anchor['expected_calibration'][k] for k in rec['calibration']
            if k != 'camera_to_world_usd_row_vectors'), 'Original optical calibration differs')
        observed = project([rec['target_world_m']], rec['calibration'])[0]
        require(observed['projection_status'] == 'in_frame' and np.linalg.norm(
            np.asarray(observed['pixel_xy'])-rec['requested_spec']['desired_pixel_xy']) < 2,
            'Original target framing differs')
    return cache, anchor, report
