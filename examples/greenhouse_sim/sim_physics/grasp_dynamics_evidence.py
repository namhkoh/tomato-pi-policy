"""Pure post-fetch retention telemetry; no USD, controller or decision authority.

The caller must supply ALL chain-body frames and both finger-body frames from
the same post-fetch sample. One step_id labels that assertion; this helper
cannot authenticate freshness, native provenance or complete body coverage.

rows are ShaftGraspEvidence.rows: (finger_index, other_collider, world_point,
normal_on_finger, impulse_on_finger, separation). Those are already oriented,
normalized-normal, signed core rows, NOT original-order raw ContactData. Copy
them unchanged, including negative/tiny impulses and rejected/unknown contacts.
Never pass friction anchors. Local points refer to BODY frames, not collider
frames or material anchors tracked across time. No force closure is inferred.
"""
from functools import lru_cache
import operator

import numpy as np


MAX_CONTACT_ROWS = 256
PD_STIFFNESS_N_M = 200.
PD_DAMPING_N_S_M = 5.


def _index(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label + ' must be a nonnegative integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(label + ' must be a nonnegative integer') from exc
    if result < 0:
        raise ValueError(label + ' must be a nonnegative integer')
    return int(result)


def _path(value):
    if (not isinstance(value, str) or not value.startswith('/') or value == '/'
            or value.endswith('/') or '//' in value
            or any(p in ('.', '..') for p in value.split('/'))):
        raise ValueError('Exact absolute body/collider path required')
    return value


def _paths(values):
    if isinstance(values, (str, bytes)):
        raise ValueError('Ordered body path sequence required')
    paths = tuple(_path(p) for p in values)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Nonempty unique body paths required')
    return paths


@lru_cache(maxsize=16)
def _chain_lookup(body_paths):
    # Immutable input key; no per-row path traversal or prefix attribution.
    return {path + '/StemCollider': i for i, path in enumerate(body_paths)}


def _array(value, shape, label):
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Finite ' + label + ' array required') from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Finite ' + label + ' array with shape ' + str(shape) + ' required')
    return array


def _frames(value, count, label):
    frames = _array(value, (count, 4, 4), label)
    r = frames[:, :3, :3]
    # Match shaft_grasp's rigid-frame tolerance; do not silently repair poses.
    with np.errstate(over='ignore', invalid='ignore'):
        rigid = (np.allclose(frames[:, 3], [0., 0., 0., 1.], atol=1e-7, rtol=0)
                 and np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-5, rtol=0)
                 and np.allclose(np.linalg.det(r), 1., atol=1e-5, rtol=0))
    if not rigid:
        raise ValueError('Proper rigid unscaled ' + label + ' required')
    return frames


def _scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('Finite numeric ' + label + ' required')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Finite numeric ' + label + ' required') from exc
    if not np.isfinite(result):
        raise ValueError('Finite numeric ' + label + ' required')
    return result


def grasp_dynamics_evidence(body_frames, body_paths, finger_frames, finger_paths, rows,
                            *, qf, qdotf, targetf, gravityf, drivecapsf, dt, step_id):
    """Finite JSON copy only; no pass/fail, clipping, force sum or artifact write.

    body_paths is the caller's complete ordered chain inventory (15 in native49).
    qf/qdotf/targetf/gravityf/drivecapsf use the EXACT finger_paths order.
    PD = 200*(target-q)-5*qdot is an unclipped reconstruction, not native effort.
    Gravity and remaining drive caps are supplied readbacks/configuration, not
    inferred limits. Their magnitudes are never used as grasp authorization.
    """
    step_id = _index(step_id, 'step_id')
    dt = _scalar(dt, 'dt')
    if dt <= 0:
        raise ValueError('Positive dt required')
    bodies = _paths(body_paths); fingers = _paths(finger_paths)
    if len(fingers) != 2 or set(bodies) & set(fingers):
        raise ValueError('Exactly two distinct finger bodies outside the plant chain required')
    bf = _frames(body_frames, len(bodies), 'chain body frames')
    ff = _frames(finger_frames, 2, 'finger body frames')
    q = _array(qf, (2,), 'finger positions')
    qdot = _array(qdotf, (2,), 'finger velocities')
    target = _array(targetf, (2,), 'finger commanded targets')
    gravity = _array(gravityf, (2,), 'finger gravity efforts')
    caps = _array(drivecapsf, (2,), 'finger remaining drive caps')
    if np.any(caps < 0):
        raise ValueError('Nonnegative remaining drive caps required')
    with np.errstate(over='ignore', invalid='ignore'):
        pd = PD_STIFFNESS_N_M * (target - q) - PD_DAMPING_N_S_M * qdot
        total_budget = np.abs(gravity) + caps
    if not np.isfinite(pd).all() or not np.isfinite(total_budget).all():
        raise ValueError('Unrepresentable reconstructed PD or force budget')
    lookup = _chain_lookup(bodies)
    contacts = []
    for raw in rows:
        if len(contacts) >= MAX_CONTACT_ROWS:
            raise ValueError('Contact telemetry overflow; no silent truncation')
        if not isinstance(raw, (tuple, list)) or len(raw) != 6:
            raise ValueError('Six-field oriented core contact row required')
        i, other, point, normal, impulse, separation = raw
        i = _index(i, 'finger index')
        if i >= 2:
            raise ValueError('Finger index must address exact ordered pair')
        other = _path(other)
        point = _array(point, (3,), 'contact world point')
        normal = _array(normal, (3,), 'oriented contact world normal')
        impulse = _array(impulse, (3,), 'signed impulse on finger')
        separation = _scalar(separation, 'contact separation')
        chain_index = lookup.get(other)
        with np.errstate(over='ignore', invalid='ignore'):
            finger_local = ff[i, :3, :3].T @ (point - ff[i, :3, 3])
            body_local = (None if chain_index is None else
                          bf[chain_index, :3, :3].T @ (point - bf[chain_index, :3, 3]))
        if not np.isfinite(finger_local).all() or (body_local is not None and not np.isfinite(body_local).all()):
            raise ValueError('Unrepresentable contact local coordinate')
        contacts.append(dict(finger_index=i, finger_path=fingers[i], other_collider=other,
            point_world_m=point.tolist(), normal_on_finger_world=normal.tolist(),
            impulse_on_finger_world_ns=impulse.tolist(), separation_m=separation,
            point_in_finger_body_m=finger_local.tolist(), chain_body_index=chain_index,
            chain_body_path=None if chain_index is None else bodies[chain_index],
            point_in_chain_body_m=None if body_local is None else body_local.tolist()))
    return dict(model='grasp_dynamics_post_fetch_telemetry_v1', step_id=step_id, dt_s=dt,
        sample_basis='caller_asserted_same_step_post_fetch',
        native_provenance_verified_by_helper=False,
        body_paths=list(bodies), body_frames_world_m=bf.tolist(),
        finger_paths=list(fingers), finger_frames_world_m=ff.tolist(),
        body_count=len(bodies), finger_count=2,
        contacts=contacts, contact_row_count=len(contacts),
        contact_row_contract='shaft_grasp_core_oriented_signed_normal_rows_v1',
        original_raw_collider_order=False, friction_included=False,
        contact_attribution='exact_chain_body_StemCollider_path_only_not_grasp_eligibility',
        local_coordinate_frame='body_not_collider_or_tracked_material_anchor',
        finger_positions_m=q.tolist(), finger_velocities_m_s=qdot.tolist(),
        finger_commanded_targets_m=target.tolist(), finger_gravity_effort_n=gravity.tolist(),
        finger_remaining_drive_caps_n=caps.tolist(),
        configured_gravity_plus_drive_upper_n=total_budget.tolist(),
        pd_stiffness_n_m=PD_STIFFNESS_N_M, pd_damping_n_s_m=PD_DAMPING_N_S_M,
        reconstructed_unclipped_pd_force_n=pd.tolist(),
        effort_basis='fixed_PD_model_reconstruction_not_actual_native_effort',
        actual_drive_effort_n=None, force_closure_verified=False,
        changes_grasp_or_release_decision=False)
