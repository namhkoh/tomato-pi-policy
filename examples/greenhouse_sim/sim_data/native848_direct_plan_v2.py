"""Pinned, bounded direct camera batch at the user's native848 resolution."""
from pathlib import Path
import ast
import importlib.util
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from . import native848_pair_plan_v2 as pair
from .native_view_pose import bounded_reference_root

SCHEMA = 'greenhouse.native848_direct_camera_batch_plan.v2'
SAMPLE_SCHEMA = 'greenhouse.native848_direct_camera_sample.v2'
RESULT_STATE = 'native848_direct_batch_captured_pending_independent_audit'
PROFILE = 'native848_direct_cached_poses_established56_every_frame.v2'


def implementation_bindings():
    root = Path(__file__).resolve().parent
    roots = (root.parent, root.parents[1])
    pending = [root / name for name in (
        'native848_direct_plan_v2.py', 'native848_direct_worker_v2.py',
        'native848_direct_audit_v2.py', 'native848_direct_admission_v2.py')]
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
                name = '.' * node.level + (node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name] + [name + '.' + a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in roots:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate / '__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def prepare(cache_path, *, selected_sample_ids=None):
    path = Path(cache_path).resolve()
    cache = read_json(path)
    selected = [r['sample_id'] for r in cache['records'] if r['state'].startswith('cached_')]
    if selected_sample_ids is not None:
        selected = list(selected_sample_ids)
    plan = dict(schema=SCHEMA, cache_path=str(path), cache_sha256=sha256(path),
        selected_sample_ids=selected, resolution=[848, 408], profile=PROFILE,
        long_reference_probe_ids=[],
        instance_backend='fast', max_frames=len(selected), full_greenhouse_stage_count=1,
        capture_render_product_count=1, broad_pose_search=False, simulated_motion=False,
        source_cap_reset=False, training_approved=False,
        implementation_bindings=implementation_bindings())
    check(plan)
    return plan


def check(plan, *, full=False):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from .capture_contract import project
    require(plan['schema'] == SCHEMA and plan['resolution'] == [848, 408]
        and plan['profile'] == PROFILE and plan['instance_backend'] == 'fast'
        and type(plan['max_frames']) is int and 1 <= plan['max_frames'] <= 30 and plan['full_greenhouse_stage_count'] == 1
        and plan['capture_render_product_count'] == 1
        and all(plan[k] is False for k in ('broad_pose_search', 'simulated_motion', 'source_cap_reset', 'training_approved')),
        'Changed direct native848 scope')
    require(plan['implementation_bindings'] == implementation_bindings(), 'Changed direct implementation closure')
    verify_bindings(plan['implementation_bindings'])
    require(sha256(plan['cache_path']) == plan['cache_sha256'], 'Changed embodied pose cache')
    cache = read_json(plan['cache_path'])
    require(cache['schema'] == 'greenhouse.native848_cached_embodied_camera_poses.v1'
        and cache['resolution'] == [848, 408] and cache['training_approved'] is False,
        'Wrong pose cache')
    verify_bindings(cache['source_bindings'])
    records = {r['sample_id']: r for r in cache['records']}
    ids = plan['selected_sample_ids']
    require(len(ids) == len(set(ids)) == plan['max_frames'] and set(ids) <= set(records), 'Invalid bounded camera selection')
    require(plan['long_reference_probe_ids'] == [], 'Established56 profile has no short-render probes')
    plans = {}
    generated = None
    model = Rby1Kinematics()
    anchor = None
    for key in ids:
        rec = records[key]
        path = rec['pair_plan_path']
        require(rec['state'].startswith('cached_') and sha256(path) == rec['pair_plan_sha256'], 'Unqualified or changed cached pose')
        if path not in plans:
            p = read_json(path)
            g = pair.check(p, full=full)
            plans[path] = p
            if generated is None:
                generated = g
            if anchor is None:
                anchor = p
            require(all(p[k] == anchor[k] for k in ('variant_directory', 'source_collection_plan',
                'source_family', 'scene_variants', 'original_variant', 'expected_scene_counts')),
                'Direct batch mixes incompatible scenes')
            require(np.allclose(p['expected_robot_snapshot']['camera_to_head_column_vectors'],
                anchor['expected_robot_snapshot']['camera_to_head_column_vectors'], atol=1e-9, rtol=0), 'Different mounted cameras')
            require(all(p['expected_calibration'][k] == anchor['expected_calibration'][k]
                for k in p['expected_calibration'] if k != 'camera_to_world_usd_row_vectors'), 'Different camera optics')
        p = plans[path]
        ref = p['expected_robot_snapshot']
        require(rec['target_id'] == p['generated_row']['target_id']
            and rec['source_target'] == p['conservative_view_cap_group'], 'Target ancestry changed')
        root = bounded_reference_root(ref, rec['requested_spec'], rec['target_world_m'])
        require(np.allclose(root, np.asarray(rec['robot_root_to_world_usd_row_vectors']).T, atol=1e-9, rtol=0), 'Cached base differs')
        require(all(rec['joint_degrees'][k] == v for k, v in ref['joint_degrees'].items()
            if k not in ('head_0', 'head_1')), 'Cached arm or torso changed')
        mount = np.asarray(rec['camera_to_head_column_vectors'])
        require(np.allclose(mount, ref['camera_to_head_column_vectors'], atol=1e-9, rtol=0), 'Cached mount differs')
        links = model.all_link_transforms(rec['joint_degrees'])
        camera = root @ links['link_head_2'] @ mount
        require(np.allclose(camera.T, rec['camera_to_world_usd_row_vectors'], atol=1e-9, rtol=0)
            and np.allclose(camera.T, rec['calibration']['camera_to_world_usd_row_vectors'], atol=1e-9, rtol=0), 'Cached camera FK differs')
        require(all(rec['calibration'][k] == p['expected_calibration'][k] for k in rec['calibration']
            if k != 'camera_to_world_usd_row_vectors'), 'Cached optical calibration differs')
        observed = project([rec['target_world_m']], rec['calibration'])[0]
        require(observed['projection_status'] == 'in_frame' and np.linalg.norm(
            np.asarray(observed['pixel_xy']) - rec['requested_spec']['desired_pixel_xy']) < 2, 'Cached target framing differs')
    return cache, plans, generated
