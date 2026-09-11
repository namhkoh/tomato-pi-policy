"""Pure source-bound contact momentum accounting; no native runtime or writes.

step_ledger uses ONE fixed world-space reference point for BOTH endpoint
momenta. Signed original collider0 impulses from normal AND friction rows are
summed once. Contact impulses must belong to the current step, not the previous
prediction geometry step. Internal pair impulses cancel globally.

No dynamic pass/fail threshold is introduced. The separate static_net_wrench
comparison retains the existing coupon 0.1 mN / 1 uNm limits; it is NOT a
dynamic conservation test, nor proof of settled state or material-law parity.
Full-stream and contacts-only declarations are caller assertions, not sensor
authentication. Missing friction cannot be inferred from an empty row list:
explicit missing-stream evidence or expected row counts are required.

Mass properties are supplied explicitly. source_authored means a source-based
reconstruction, NOT native inertia/COM readback. native_readback still describes
caller provenance only. COM positions must be exactly zero, and the inertia
must be about COM in the body-prim frame (not unrotated principal moments).
"""
import hashlib
import json

import numpy as np

from .contact_spring_probe import Coupon, layout

SCHEMA = 'contact_momentum_ledger_v1'
STATIC_FORCE_LIMIT_N = 1e-4
STATIC_TORQUE_LIMIT_NM = 1e-6
MAX_CONTACT_ROWS = 256
# Existing coupon bind_rigid source-consistency bounds, NOT residual gates.
SOURCE_RTOL = 2e-6
MASS_SOURCE_ATOL_KG = 1e-12
INERTIA_SOURCE_ATOL_KG_M2 = 1e-14
FRAME_ATOL = 1e-6


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError('Real '+name+' required')
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite '+name+' with shape '+str(shape)+' required')
    return a


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError('Integer '+name+' required')
    return int(value)


def _dt(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value <= 0):
        raise ValueError('Positive finite current step dt required')
    return float(value)


def _frames(value):
    f = _array(value, (3, 4, 4), 'three native body frames')
    if not np.array_equal(f[:, 3], np.tile([0., 0., 0., 1.], (3, 1))):
        raise ValueError('Homogeneous rigid frame required')
    for R in f[:, :3, :3]:
        if (not np.allclose(R.T@R, np.eye(3), atol=FRAME_ATOL, rtol=0)
                or abs(np.linalg.det(R)-1) > FRAME_ATOL):
            raise ValueError('Proper unscaled frame required; no reflection or repair')
    return f


def _properties(coupon, masses, inertias_body, com_local, basis):
    if not isinstance(coupon, Coupon):
        raise ValueError('Explicit source-bound Coupon required')
    if basis not in ('source_authored', 'native_readback'):
        raise ValueError('Explicit mass_properties_basis required')
    m = _array(masses, (3,), 'unchanged body masses')
    I = _array(inertias_body, (3, 3, 3), 'COM inertias in body-prim frame')
    com = _array(com_local, (3, 3), 'local COM positions')
    if np.any(com != 0):
        raise ValueError('This ledger requires exactly zero local COM positions')
    if np.any(m <= 0):
        raise ValueError('Positive body masses required')
    for tensor in I:
        scale = max(float(np.max(abs(tensor))), np.finfo(float).tiny)
        if np.max(abs(tensor-tensor.T)) > 64*np.finfo(float).eps*scale:
            raise ValueError('Symmetric inertia required; not silently symmetrized')
        eig = np.linalg.eigvalsh(tensor)
        if eig[0] <= 0 or eig[-1] > eig[:-1].sum()+64*np.finfo(float).eps*scale:
            raise ValueError('Positive physical inertia required')
    expected_m = np.asarray(coupon.masses); expected_I = np.asarray(coupon.inertias)
    # Do not replace actual supplied properties with the nominal source values.
    if basis == 'source_authored':
        matches = np.array_equal(m, expected_m) and np.array_equal(I, expected_I)
    else:
        matches = (np.allclose(m, expected_m, rtol=SOURCE_RTOL, atol=MASS_SOURCE_ATOL_KG)
                   and np.allclose(I, expected_I, rtol=SOURCE_RTOL, atol=INERTIA_SOURCE_ATOL_KG_M2))
    if not matches:
        raise ValueError('Mass/inertia source mismatch')
    provenance = dict(basis=basis, masses_kg=m.tolist(), inertias_body_kg_m2=I.tolist(),
        com_local_m=com.tolist(), supplied_COM_positions_verified_zero=True,
        native_properties_authenticated=False,
        max_mass_source_difference_kg=float(np.max(abs(m-expected_m))),
        max_inertia_source_difference_kg_m2=float(np.max(abs(I-expected_I))),
        source_comparison=('exact' if basis == 'source_authored' else
            dict(rtol=SOURCE_RTOL, mass_atol_kg=MASS_SOURCE_ATOL_KG,
                 inertia_atol_kg_m2=INERTIA_SOURCE_ATOL_KG_M2)),
        comparison_tolerances_are_not_dynamic_residual_gates=True)
    return m, I, provenance


def _state(coupon, state):
    step = _integer(state['step_id'], 'snapshot step')
    if state.get('source_sha256') != coupon.source_sha256:
        raise ValueError('Snapshot source mismatch')
    if ('body_paths' in state and list(state['body_paths']) != layout(coupon)['body_paths']):
        raise ValueError('Snapshot body order differs from coupon')
    if step and _dt(state.get('dt_s')) != coupon.dt:
        raise ValueError('Snapshot dt differs from coupon')
    return step, _frames(state['frames_world_m']), _array(
        state['body_velocities_world'], (3, 6), 'world COM linear/angular velocities')


def step_ledger(coupon, previous, current, *, masses, inertias_body, com_local,
                mass_properties_basis, contact_evidence, origin_world_m):
    """Return finite metrics only; no force application or dynamic qualification.

previous/current carry source_sha256, step_id, frames_world_m and
body_velocities_world; ordinary snapshots also carry dt_s. current additionally
contains contact_rows with collider0/collider1/kind/point_world_m/impulse_on_0_ns.
Initial step0 must explicitly identify its post-bootstrap or pre-solve state.
All state/property arrays follow layout(coupon)['body_paths'] order. Optional
snapshot body_paths are checked, but missing path metadata is caller-asserted
ordering, not a claim that native object identities were authenticated.

contact_evidence requires matching source_sha256/step_id/dt_s plus TRUE
normal_rows_complete, friction_rows_complete, contacts_are_all_external_loads.
Optional expected_normal_rows/expected_friction_rows detect counted omissions;
without them a falsely asserted complete stream is not independently detected.
"""
    m, I, properties = _properties(coupon, masses, inertias_body, com_local, mass_properties_basis)
    step0, F0, V0 = _state(coupon, previous)
    step1, F1, V1 = _state(coupon, current)
    if step1 < 1 or step1 != step0+1:
        raise ValueError('Contiguous previous/current snapshots required')
    h = _dt(current['dt_s'])
    initial_basis = None
    if step0 == 0:
        initial_basis = previous.get('initial_state_basis')
        boot = previous.get('bootstrap_dt_s')
        if initial_basis == 'post_bootstrap_native_readback':
            _dt(boot)
        elif initial_basis == 'pre_solve_native_readback':
            if isinstance(boot, (bool, np.bool_)) or boot != 0:
                raise ValueError('Pre-solve readback must have zero bootstrap duration')
        else:
            raise ValueError('Explicit initial readback/bootstrap distinction required')
    if 'native_prediction_before_step' in current:
        expected = current['native_prediction_before_step'].get('frames_world_m')
        if expected is not None and not np.array_equal(_frames(expected), F0):
            raise ValueError('Previous frame differs from current recorded pre-step frame')
    e = contact_evidence
    if (not isinstance(e, dict) or e.get('source_sha256') != coupon.source_sha256
            or _integer(e.get('step_id'), 'contact step', 1) != step1
            or _dt(e.get('dt_s')) != h):
        raise ValueError('Contact source/step/dt mismatch')
    for key in ('normal_rows_complete', 'friction_rows_complete', 'contacts_are_all_external_loads'):
        if e.get(key) is not True:
            raise ValueError('Explicit complete contacts-only evidence required: '+key)
    if current.get('complete_stream_caller_asserted', True) is not True:
        raise ValueError('Recorded incomplete contact stream')
    origin = _array(origin_world_m, (3,), 'same fixed world origin')
    data = layout(coupon)
    bodies = set(data['collider_paths']); pads = set(data['pads'])
    linear_impulse = np.zeros(3); angular_impulse = np.zeros(3)
    counts = dict(normal=0, friction=0); internal = 0; normal_impulse = np.zeros(3); friction_impulse = np.zeros(3)
    for row in current['contact_rows']:
        if sum(counts.values()) >= MAX_CONTACT_ROWS:
            raise ValueError('Contact row bound exhausted')
        a, b, kind = row['collider0'], row['collider1'], row['kind']
        if (a == b or a not in bodies|pads or b not in bodies|pads
                or not ({a, b} & bodies) or kind not in counts):
            raise ValueError('Unknown/unattributed contact collider or kind')
        p = _array(row['point_world_m'], (3,), 'contact world point')
        j = _array(row['impulse_on_0_ns'], (3,), 'signed original-order impulse')
        if 'normal_on_0' in row:
            _array(row['normal_on_0'], (3,), 'original normal')
        if 'separation_m' in row:
            _array(row['separation_m'], (), 'native separation')
        counts[kind] += 1
        internal += int(a in bodies and b in bodies)
        for path, sign in ((a, 1), (b, -1)):
            if path in bodies:
                linear_impulse += sign*j
                angular_impulse += np.cross(p-origin, sign*j)
                if kind == 'normal': normal_impulse += sign*j
                else: friction_impulse += sign*j
    normalized_evidence = dict(e, step_id=step1, dt_s=h)
    for kind in counts:
        key = 'expected_'+kind+'_rows'
        if key in e and _integer(e[key], key) != counts[kind]:
            raise ValueError('Contact '+kind+' row count mismatch')
        if key in e:
            normalized_evidence[key] = int(e[key])
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        R0 = F0[:, :3, :3]; R1 = F1[:, :3, :3]
        IW0 = R0@I@R0.transpose(0, 2, 1); IW1 = R1@I@R1.transpose(0, 2, 1)
        r0 = F0[:, :3, 3]-origin; r1 = F1[:, :3, 3]-origin
        p0 = m[:, None]*V0[:, :3]; p1 = m[:, None]*V1[:, :3]
        P0 = p0.sum(0); P1 = p1.sum(0)
        S0 = np.einsum('bij,bj->bi', IW0, V0[:, 3:]).sum(0)
        S1 = np.einsum('bij,bj->bi', IW1, V1[:, 3:]).sum(0)
        L0 = np.cross(r0, p0).sum(0); L1 = np.cross(r1, p1).sum(0)
        H0 = S0+L0; H1 = S1+L1
        force = linear_impulse/h; torque = angular_impulse/h
        dP = (P1-P0)/h; dH = (H1-H0)/h
        force_error = dP-force; torque_error = dH-torque
        spin_acceleration = np.einsum('bij,bj->bi', .5*(IW0+IW1),
                                      (V1[:, 3:]-V0[:, 3:])/h).sum(0)
        orientation = np.einsum('bij,bj->bi', (IW1-IW0)/h,
                                .5*(V0[:, 3:]+V1[:, 3:])).sum(0)
        orbital_acceleration = np.cross(.5*(r0+r1), (p1-p0)/h).sum(0)
        lever_change = np.cross((r1-r0)/h, .5*(p0+p1)).sum(0)
        fnorm = float(np.linalg.norm(force)); tnorm = float(np.linalg.norm(torque))
    # Explicit finite copying: overflow never becomes a JSON NaN or an implicit pass.
    result = dict(schema=SCHEMA, source_sha256=coupon.source_sha256,
        coupon_sha256=_digest(coupon.report()), step_id=step1, previous_step_id=step0, dt_s=h,
        first_interval_initial_state_basis=initial_basis,
        initial_bootstrap_dt_s=previous.get('bootstrap_dt_s') if step0 == 0 else None,
        bootstrap_impulses_included=False, mass_properties=properties,
        body_paths=data['body_paths'], origin_world_m=origin.tolist(),
        body_order_authenticated=False,
        same_fixed_origin_both_endpoints=True,
        contact_evidence=dict(_copy(normalized_evidence), assertion_not_authentication=True,
            missing_friction_inferred=False, observed_normal_rows=counts['normal'],
            observed_friction_rows=counts['friction'], internal_contact_rows=internal),
        previous_linear_momentum_ns=P0.tolist(), current_linear_momentum_ns=P1.tolist(),
        previous_angular_momentum_nms=H0.tolist(), current_angular_momentum_nms=H1.tolist(),
        external_contact_impulse_ns=linear_impulse.tolist(),
        external_contact_angular_impulse_nms=angular_impulse.tolist(),
        normal_net_force_n=(normal_impulse/h).tolist(), friction_net_force_n=(friction_impulse/h).tolist(),
        contact_force_n=force.tolist(), contact_torque_nm=torque.tolist(),
        delta_p_dt_n=dP.tolist(), delta_h_dt_nm=dH.tolist(),
        force_residual_n=force_error.tolist(), torque_residual_nm=torque_error.tolist(),
        force_residual_norm_n=float(np.linalg.norm(force_error)),
        torque_residual_norm_nm=float(np.linalg.norm(torque_error)),
        body_linear_acceleration_m_s2=((V1[:, :3]-V0[:, :3])/h).tolist(),
        body_angular_acceleration_rad_s2=((V1[:, 3:]-V0[:, 3:])/h).tolist(),
        spin_momentum_rate_nm=((S1-S0)/h).tolist(), orbital_momentum_rate_nm=((L1-L0)/h).tolist(),
        spin_acceleration_term_nm=spin_acceleration.tolist(),
        orientation_change_term_nm=orientation.tolist(),
        orbital_acceleration_term_nm=orbital_acceleration.tolist(), lever_change_term_nm=lever_change.tolist(),
        momentum_decomposition_error_nm=(dH-spin_acceleration-orientation-orbital_acceleration-lever_change).tolist(),
        orientation_term_basis='exact finite R I R^T change; includes gyro contribution without assuming pose-rate/native-omega equality',
        static_net_wrench=dict(force_norm_n=fnorm, torque_norm_nm=tnorm,
            force_limit_n=STATIC_FORCE_LIMIT_N, torque_limit_nm=STATIC_TORQUE_LIMIT_NM,
            force_pass=fnorm < STATIC_FORCE_LIMIT_N, torque_pass=tnorm < STATIC_TORQUE_LIMIT_NM,
            passed=fnorm < STATIC_FORCE_LIMIT_N and tnorm < STATIC_TORQUE_LIMIT_NM,
            basis='existing coupon static limits only; not settled-state or dynamic qualification'),
        dynamic_pass=None, dynamic_residual_tolerance=None, native_qualified=False,
        material_law_qualified=False, force_applied=False)
    return _copy(result)


def summarize(ledgers):
    """Summary without relaxing or introducing any physical qualification gate."""
    rows = list(ledgers)
    if not rows: raise ValueError('At least one ledger required')
    first = rows[0]
    for i, row in enumerate(rows):
        if (row.get('schema') != SCHEMA or row['source_sha256'] != first['source_sha256']
                or row['coupon_sha256'] != first['coupon_sha256'] or row['dt_s'] != first['dt_s']
                or row['mass_properties'] != first['mass_properties']
                or (i and row['step_id'] != rows[i-1]['step_id']+1)):
            raise ValueError('Contiguous equally source-bound ledgers required')
    def vectors(key): return _array([row[key] for row in rows], (len(rows), 3), key)
    force = vectors('force_residual_n'); torque = vectors('torque_residual_nm')
    contact_torque = vectors('contact_torque_nm')
    return _copy(dict(schema=SCHEMA, first_step=first['step_id'], last_step=rows[-1]['step_id'],
        sample_count=len(rows), dt_s=first['dt_s'], source_sha256=first['source_sha256'],
        max_force_residual_n=float(np.max(np.linalg.norm(force, axis=1))),
        rms_force_residual_n=float(np.sqrt(np.mean(np.sum(force*force, axis=1)))),
        max_torque_residual_nm=float(np.max(np.linalg.norm(torque, axis=1))),
        rms_torque_residual_nm=float(np.sqrt(np.mean(np.sum(torque*torque, axis=1)))),
        max_contact_torque_nm=float(np.max(np.linalg.norm(contact_torque, axis=1))),
        static_wrench_failed_samples=sum(not row['static_net_wrench']['passed'] for row in rows),
        static_torque_failed_samples=sum(not row['static_net_wrench']['torque_pass'] for row in rows),
        dynamic_pass=None, native_qualified=False))


def trace_ledger(coupon, samples, initial_readback, stepping, *, masses, inertias_body,
                 com_local, mass_properties_basis, contact_assertions,
                 origin_world_m=(0., 0., 0.), use_current_middle_origin=False):
    """Adapt a COMPLETE trace beginning at step1, never treating bootstrap as dt.

initial_readback is the explicit report readback, potentially already moving
after bootstrap. Its source binding is caller asserted through coupon. The
bootstrap contact rows are NOT included or divided by the ordinary step dt.
use_current_middle_origin reproduces sample.net_contact_torque's reference:
each interval still uses that one current point for BOTH endpoint momenta.
Returned records may be sliced and passed to summarize for any contiguous tail.
"""
    if type(use_current_middle_origin) is not bool:
        raise ValueError('Explicit origin mode required')
    startup_steps = _integer(stepping['startup_physics_steps'], 'startup steps')
    boot = stepping['startup_physics_dt_s']
    if startup_steps not in (0, 1):
        raise ValueError('Explicit zero or one bootstrap supported')
    if startup_steps:
        _dt(boot); basis = 'post_bootstrap_native_readback'
    else:
        if isinstance(boot, (bool, np.bool_)) or boot != 0:
            raise ValueError('Zero startup steps requires zero bootstrap duration')
        basis = 'pre_solve_native_readback'
    if initial_readback.get('source_sha256', coupon.source_sha256) != coupon.source_sha256:
        raise ValueError('Initial readback source mismatch')
    previous = dict(step_id=0, source_sha256=coupon.source_sha256,
        frames_world_m=initial_readback['frames'],
        body_velocities_world=initial_readback['native_body_velocities'],
        initial_state_basis=basis, bootstrap_dt_s=boot)
    records = []
    for row in samples:
        evidence = dict(contact_assertions)
        # Caller supplies completeness, not separate potentially stale timing.
        if set(evidence) & {'source_sha256', 'step_id', 'dt_s'}:
            raise ValueError('Trace contact assertions must not override sample binding')
        evidence.update(source_sha256=row['source_sha256'], step_id=row['step_id'], dt_s=row['dt_s'])
        origin = _frames(row['frames_world_m'])[1, :3, 3] if use_current_middle_origin else origin_world_m
        records.append(step_ledger(coupon, previous, row, masses=masses, inertias_body=inertias_body,
            com_local=com_local, mass_properties_basis=mass_properties_basis,
            contact_evidence=evidence, origin_world_m=origin))
        previous = row
    return dict(records=records, summary=summarize(records),
        initial_readback_source_binding='caller asserted, not independently authenticated',
        bootstrap_physics_steps_excluded=startup_steps, bootstrap_dt_s_excluded=float(boot),
        native_qualified=False)
