"""Synchronous endpoint-only withdrawal check; never step, move or release.

Call after the final fetched physics sample, before stopping/resetting physics.
The caller owns cut/sequence, retention and all ongoing native contact guards.
Plant frames must be the complete current rig.body_paths-ordered sample.

Only the park REFERENCE uses FK. Every screened robot body, including fingers,
comes from robot_bodies' native path-labelled transforms. Cached shape geometry
is read-only; copied screens get fresh plant snapshots and a new static-query
epoch. This is conservative sampled endpoint evidence, NOT a path certificate,
complete native shape equivalence, verified tissue cutting or physical safety.
"""
from copy import copy, deepcopy
import json
import operator

import numpy as np

from .withdrawal_evidence import withdrawal_evidence, _pose, _sample_id


def _sample(step_id):
    if isinstance(step_id, (tuple, list)):
        return _sample_id(step_id), 'explicit_episode_and_step'
    if isinstance(step_id, (bool, np.bool_)):
        raise ValueError('Nonnegative physics step required')
    try:
        step = operator.index(step_id)
    except TypeError as exc:
        raise ValueError('Nonnegative physics step required') from exc
    if step < 0:
        raise ValueError('Nonnegative physics step required')
    # Stateless single-call namespace only; never a claimed persistent episode.
    return (0, int(step)), 'single_call_step_not_cross_episode_identity'


def _paths(values, label):
    if isinstance(values, (str, bytes)):
        raise ValueError('Ordered exact ' + label + ' paths required')
    paths = tuple(values)
    if (not paths or len(set(paths)) != len(paths)
            or any(not isinstance(p, str) or not p.startswith('/') or p.endswith('/') for p in paths)):
        raise ValueError('Unique absolute ' + label + ' paths required')
    return paths


def _wrist(fixture):
    from .runtime import pose_matrices
    expected = fixture.root + '/ee_right'
    paths = _paths(fixture.right_palm.prim_paths, 'right wrist')
    if paths != (expected,) or fixture.right_palm.count != 1:
        raise ValueError('Exact single native right-wrist view required')
    poses = pose_matrices(np.array(fixture.right_palm.get_transforms(), copy=True))
    if poses.shape != (1, 4, 4):
        raise ValueError('Single native right-wrist transform required')
    return _pose(poses[0])


def _robot_world(fixture):
    from .runtime import pose_matrices
    paths = _paths(fixture.robot_bodies.prim_paths, 'native robot body')
    expected = _paths(fixture.body_paths, 'authored robot body inventory')
    if set(paths) != set(expected):
        raise ValueError('Native robot body coverage differs from complete fixture inventory')
    poses = pose_matrices(np.array(fixture.robot_bodies.get_transforms(), copy=True))
    if poses.shape != (len(paths), 4, 4):
        raise ValueError('Native robot body path/transform count mismatch')
    prefix = fixture.root + '/'
    if any(not p.startswith(prefix) for p in paths):
        raise ValueError('Native robot body outside exact robot root')
    world = {p[len(prefix):]: _pose(m) for p, m in zip(paths, poses, strict=True)}
    # Screens use root-relative link names, not basename guesses or FK.
    for _, body, link, _, _ in fixture.self_screen.shapes:
        if body != prefix + link or body not in paths:
            raise ValueError('Screen shape missing its exact native rigid body')
    if 'ee_right' not in world:
        raise ValueError('Right wrist missing from complete native body inventory')
    return world


def _screens(fixture, frames):
    from .held_plant_screen import HeldPlantScreen
    original = fixture.held_plant_screen
    if (original.arm != 'right' or original.workspace is None
            or not original.local or not original.static):
        raise ValueError('Existing complete right-arm/local-static geometry cache required')
    paths = _paths(fixture.rig.body_paths, 'plant body')
    values = np.array(frames, dtype=float, copy=True)
    if values.shape != (len(paths), 4, 4):
        raise ValueError('Complete same-step native plant body frame sample required')
    values = np.array([_pose(m) for m in values])
    if any(not 0 <= i < len(paths) or not path.startswith(paths[i] + '/')
           for path, i, _, _ in original.local):
        raise ValueError('Cached plant shapes do not match exact body frame inventory')
    selected = fixture.grasp_path
    if selected not in paths or not fixture.rig.cut_index <= paths.index(selected):
        raise ValueError('Exact detachable left-grasp body required')
    right = copy(original)
    # Only immutable shape data/indices are shared. All fields written by
    # snapshot/check are replaced on these copies, never on the controller.
    right.static = list(original.static); right.local = list(original.local)
    right.static_indices = dict(getattr(original, 'static_indices', {}))
    right.workspace = tuple(np.array(v, copy=True) for v in original.workspace)
    right.shapes = list(original.shapes); right.native_static_query = None
    expected_right = [s for s in fixture.self_screen.shapes
                      if s[2].startswith(('link_right_arm_', 'ee_right', 'ee_finger_r'))]
    if not right.shapes or [s[0] for s in right.shapes] != [s[0] for s in expected_right]:
        raise ValueError('Incomplete cached right-arm shape coverage')
    right.snapshot(values)
    # Same HeldPlantScreen algorithm and exact existing grasp allowances;
    # reusing its cached local geometry avoids a Fabric-stale USD pose read.
    left = copy(right); left.arm = 'left'
    left.shapes = [s for s in fixture.self_screen.shapes
                   if s[2].startswith(('link_left_arm_', 'ee_left', 'ee_finger_l'))]
    if not left.shapes:
        raise ValueError('Missing complete left-arm shape coverage')
    index = paths.index(selected)
    left.grasp_collider = selected + '/StemCollider'
    left.grasp_colliders = {p + '/StemCollider'
                           for p in paths[max(fixture.rig.cut_index, index-1):index+2]}
    left.last_failure = None
    if not isinstance(right, HeldPlantScreen):
        raise ValueError('Expected existing HeldPlantScreen geometry')
    return right, left


def check(fixture, frames, step_id):
    """Return current endpoint/clearance diagnostics; errors never complete.

    An integer step_id is scoped to this synchronous call; alternatively pass
    (episode, physics_step). Wrong park pose performs NO clearance query.
    Query creation always uses the existing 8-second default (no budget edit).
    No plan, planning_slides, controller snapshot or existing query is changed.
    """
    sample, sample_basis = _sample(step_id)
    actual = _wrist(fixture)
    park = _pose(fixture.kin.forward('right', np.array(fixture.right, copy=True),
                                    np.array(fixture.base, copy=True)))
    result = withdrawal_evidence(actual, park, clearance_verified=False,
        measured_sample_id=sample, clearance_sample_id=sample)
    result.update(checker='native_current_endpoint_bounds_v1', sample_id_basis=sample_basis,
        measured_wrist_world=actual.tolist(), park_reference_world=park.tolist(),
        robot_pose_source='post_fetch_native_path_labelled_body_transforms',
        park_pose_source='FK_of_unchanged_park_reference_only',
        plant_pose_source='caller_supplied_complete_post_fetch_sample',
        clearance_attempted=False, self_screen=None, right_scene=None, left_scene=None,
        native_static=None, errors=[], whole_scene_native_collision_certified=False)
    if not result['endpoint_attained']:
        result['reason'] = 'park_endpoint_not_attained'
        return result
    native = None; geometry_clear = False
    result['clearance_attempted'] = True
    result['native_static'] = dict(initialization_status='not_started',
        query_count=0, query_count_known=True, final_validation_passed=False)
    try:
        from .native_static_clearance import current_scene_query
        right, left = _screens(fixture, frames)
        result['native_static'].update(initialization_status='in_progress',
                                       query_count=None, query_count_known=False)
        native = current_scene_query(fixture.stage, right.static)
        result['native_static']['initialization_status'] = 'ready'
        right.native_static_query = native
        left.native_static_query = native  # Existing left screen has no native refinement allowance.
        world = _robot_world(fixture)
        # Independent native views must describe the same wrist. Never screen
        # a body set different from the one that attained the endpoint.
        if not np.allclose(actual, world['ee_right'], atol=1e-7, rtol=0):
            raise RuntimeError('Native wrist/body views disagree within withdrawal sample')
        actual = world['ee_right']
        pose_check = withdrawal_evidence(actual, park, clearance_verified=False,
            measured_sample_id=sample, clearance_sample_id=sample)
        result.update(pose_check)
        result['measured_wrist_world'] = actual.tolist()
        if not pose_check['endpoint_attained']:
            result['reason'] = 'park_endpoint_not_attained_in_body_snapshot'
        else:
            self_result = copy(fixture.self_screen).check(world)  # unchanged default 3 mm
            result['self_screen'] = deepcopy(self_result)
            self_clear = (self_result.get('passed') is True
                          and self_result.get('all_shape_bounds_screened') is True
                          and not self_result.get('unsupported_shapes')
                          and self_result.get('checked_pairs', 0) > 0
                          and np.isfinite(self_result.get('minimum_clearance_m', np.nan)))
            right_clear = right.check(world, stroke=False)  # unchanged default 1 mm
            result['right_scene'] = dict(passed=right_clear, stroke_allowance=False,
                margin_m=.001, failure=deepcopy(right.last_failure))
            left_clear = left.check(world, grasp=True, stroke=False)
            result['left_scene'] = dict(passed=left_clear, margin_m=.001,
                expected_grasp_colliders=sorted(left.grasp_colliders),
                failure=deepcopy(left.last_failure))
            geometry_clear = bool(self_clear and right_clear is True and left_clear is True)
            result['reason'] = 'endpoint_bounds_clear' if geometry_clear else 'endpoint_bounds_rejected'
            if geometry_clear:
                native.validate()  # Same-epoch final positive coverage BEFORE success.
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
        result['reason'] = 'clearance_unavailable_not_collision_proof'
    finally:
        if native is not None:
            try:
                native.close()
            except Exception as exc:
                result['errors'].append('close: ' + type(exc).__name__ + ': ' + str(exc))
            try:
                result['native_static'] = deepcopy(native.report())
                result['native_static']['initialization_status'] = 'ready'
                result['native_static']['query_count_known'] = type(result['native_static'].get('query_count')) is int
            except Exception as exc:
                result['errors'].append('report: ' + type(exc).__name__ + ': ' + str(exc))
        elif result['native_static']['initialization_status'] == 'in_progress':
            result['native_static']['initialization_status'] = 'failed'
    report = result['native_static']
    epoch = report.get('epoch') or {}
    validated = (report.get('final_validation_passed') is True and report.get('closed') is True
                 and not report.get('errors') and epoch.get('revision') == 0
                 and epoch.get('subscriptions_closed') is True and not epoch.get('cleanup_errors')
                 and not epoch.get('invalidation_reasons')
                 and set(report.get('used_static_colliders', ())) <= set(report.get('final_coverage_checked', ())))
    completed = bool(geometry_clear and validated and not result['errors'])
    result.update(withdrawal_evidence(actual, park, clearance_verified=completed,
        measured_sample_id=sample, clearance_sample_id=sample))
    if geometry_clear and not completed:
        result['reason'] = 'native_epoch_or_cleanup_not_validated'
    # Return detached finite JSON data, never screen/query objects or Infinity.
    # Malformed diagnostics raise, still AFTER owned query cleanup.
    return json.loads(json.dumps(result, allow_nan=False))
