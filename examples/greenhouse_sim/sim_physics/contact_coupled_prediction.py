"""Coupon-only frozen-contact prediction. No native imports, writes or actuators.

API: compile_geometry(coupon, frames, velocities, native_jacobian, ...)
returns (J, gap, surface_speed, report). solve(M, q, v, K, C, J, gap,
surface_speed, kc, dc, h, f, law=...) returns JSON-copyable predictions.
Only tau_spring is an effort proposal; predicted contact forces are NEVER to
be applied. The caller must zero/read back both native joint K/C, preserve
native contacts/friction, check effort caps and existing qualification gates.

Two explicit response contracts are provided, neither asserted to match PhysX:
unilateral_kv_v1: max(0, -kc*g_next-dc*relative_speed), only while g_next<=0;
signed_overlap_kv_v1: retain that signed response while g_next<=0. Negative
overlap response is damping, not evidence of compressive support. Both drop
the spring on separation; neither pins the root or adheres across a gap.

The finite-face approximation uses TWO source-seeded persistent segment
anchors per capsule/pad pair, not an infinite plane or a midpoint spring.
Native normal rows are matched at their contact-generation snapshot. Current
points are capsule_transform*segment_anchor-radius*pad_normal, not newly
clipped edges or rigidly transported capsule surface points. This follows
PhysX's PCM capsule contact construction (GuPersistentContactManifold.cpp,
addManifoldContactsToContactBuffer capsule overload). Native21 independently
agrees within 20 nm; neither that observation nor public source establishes
general native law equivalence. No force or stale separation is replayed.
The seed pad anchors must lie on the original finite face; current capsule
coverage is checked separately. A cached representative point may lie just
outside the finite edge while its seed pad anchor remains on the face.
Cap/other-face contacts and changed unsupported manifolds raise.
"""
from dataclasses import dataclass

import numpy as np

from .contact_spring_probe import (
    CAPSULE_LENGTH_M, PAD_SIZE_M, RADIUS_M, SPACING_M, layout,
)


MAX_ROWS = 256
MAX_ITERATIONS = 32
POINT_TOL_M = 2e-6
SEPARATION_TOL_M = 2e-6
NORMAL_TOL = 2e-4
CONTACT_OFFSET_SUM_M = .001  # Two unchanged .0005 shape contact offsets.
LAW_NAMES = ('unilateral_kv_v1', 'signed_overlap_kv_v1')


@dataclass(frozen=True)
class MaterialLaw:
    """Explicit mathematical choice, NEVER inferred from measured force signs."""
    response: str

    def __post_init__(self):
        if self.response not in LAW_NAMES:
            raise ValueError('Explicit supported material response required')


def _array(value, shape, name):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite '+name+' with shape '+str(shape)+' required')
    return a


def _integer(value, name, low=0, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError('Bounded integer '+name+' required')
    return int(value)


def _matrix(value, n, name, *, positive=False):
    a = np.array(value, dtype=float, copy=True)
    if a.shape == (n,):
        a = np.diag(a)
    a = _array(a, (n, n), name)
    scale = max(float(np.max(abs(a))), np.finfo(float).tiny)
    if np.max(abs(a-a.T)) > 2e-6*scale:
        raise ValueError(name+' must be symmetric; only float32 readback roundoff allowed')
    a = (a+a.T)*.5
    if positive:
        np.linalg.cholesky(a)  # No diagonal regularization or inertia inflation.
    elif np.linalg.eigvalsh(a)[0] < 0:
        raise ValueError(name+' must be positive semidefinite')
    return a


def _response(gap, speed, kc, dc, law):
    raw = -kc*gap-dc*speed
    active = gap <= 0
    if law.response == 'unilateral_kv_v1':
        active &= raw > 0
    return active, raw


def solve(M, q, v, K, C, J, g, s, kc, dc, h, f, *, law,
          root_dofs=6, max_iterations=16, feature_observed=None):
    """Bounded piecewise-linear solve; unresolved => tau_spring=None.

g>0 means separated; gdot=J*v-s. K/C may be diagonal vectors or PSD
matrices, with EXACT zero root rows/columns. Six floating root coordinates
participate in M; this routine never takes a constrained-root block.
Geometry/contact manifold and inertia are frozen for this one prediction.
Caller owns contact feature coverage and native friction, which is omitted
from prediction. No gain, force, sign or contact-count fitting is performed.
When using compile_geometry, pass report['feature_observed']; an unmatched
feature that would become active returns no effort. Omitting this mask is
permitted for pure algebra only, and reports feature coverage as unbound.
"""
    if not isinstance(law, MaterialLaw):
        raise ValueError('Explicit MaterialLaw required; no native parity default')
    q = np.array(q, dtype=float, copy=True)
    if q.ndim != 1 or not 1 <= q.size <= 64:
        raise ValueError('Bounded nonempty generalized coordinate vector required')
    n = len(q)
    q = _array(q, (n,), 'q'); v = _array(v, (n,), 'v'); f = _array(f, (n,), 'f')
    root_dofs = _integer(root_dofs, 'root_dofs', 0, n)
    max_iterations = _integer(max_iterations, 'max_iterations', 1, MAX_ITERATIONS)
    if isinstance(h, (bool, np.bool_)) or not np.isscalar(h) or not np.isfinite(h) or h <= 0:
        raise ValueError('Positive finite timestep required')
    M = _matrix(M, n, 'M', positive=True)
    K = _matrix(K, n, 'K'); C = _matrix(C, n, 'C')
    if any(np.any(a[:root_dofs]) or np.any(a[:, :root_dofs]) for a in (K, C)):
        raise ValueError('Root spring rows and columns must be exactly zero')
    g = np.array(g, dtype=float, copy=True)
    if g.ndim != 1 or len(g) > MAX_ROWS:
        raise ValueError('Bounded contact gap vector required')
    m = len(g)
    g = _array(g, (m,), 'g'); J = _array(J, (m, n), 'J')
    s = _array(s, (m,), 's'); kc = _array(kc, (m,), 'kc'); dc = _array(dc, (m,), 'dc')
    coverage_bound = feature_observed is not None
    observed = np.ones(m, dtype=bool)
    if coverage_bound:
        observed = np.asarray(feature_observed)
        if observed.shape != (m,) or (m and observed.dtype != np.bool_):
            raise ValueError('Explicit boolean feature-observed mask required')
        observed = observed.astype(bool, copy=True)
    if np.any(kc <= 0) or np.any(dc < 0):
        raise ValueError('Positive contact stiffness and nonnegative damping required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        initial_speed = J@v-s
        active, _ = _response(g, initial_speed, kc, dc, law)
        seen = set()
        for iteration in range(1, max_iterations+1):
            if np.any(active & ~observed):
                return _unresolved(law, iteration-1, 'unobserved_active_feature')
            key = tuple(active.tolist())
            if key in seen:
                return _unresolved(law, iteration-1, 'active_set_cycle')
            seen.add(key)
            ka = kc*active; da = dc*active
            A = M+h*C+h*h*K+(J.T*(h*da+h*h*ka))@J
            b = M@v+h*(f-K@q+J.T@(-ka*g+(da+h*ka)*s))
            if not np.isfinite(A).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite coupled system')
            np.linalg.cholesky(A)
            vp = np.linalg.solve(A, b)
            speed = J@vp-s
            gp = g+h*speed
            next_active, raw = _response(gp, speed, kc, dc, law)
            if not np.array_equal(next_active, active):
                active = next_active
                continue
            tau = -K@(q+h*vp)-C@vp
            contact = np.where(active, raw, 0.)
            residual = M@(vp-v)-h*(f+tau+J.T@contact)
            scale = max(np.linalg.norm(M@v), np.linalg.norm(M@vp),
                        h*np.linalg.norm(f+tau), h*np.linalg.norm(J.T@contact), 1e-30)
            relative_residual = float(np.linalg.norm(residual)/scale)
            if (not all(np.isfinite(x).all() for x in (vp, gp, raw, tau, contact))
                    or relative_residual > 1e-10 or np.any(tau[:root_dofs] != 0)):
                return _unresolved(law, iteration, 'nonfinite_or_equilibrium_residual')
            return dict(status='resolved', iterations=iteration,
                v_pred=vp.tolist(), tau_spring=tau.tolist(),
                tau_joint=tau[root_dofs:].tolist(), gap_initial_m=g.tolist(),
                gap_pred_m=gp.tolist(), relative_speed_initial_m_s=initial_speed.tolist(),
                relative_speed_pred_m_s=speed.tolist(),
                opening=(speed > 0).tolist(), separated_pred=(gp > 0).tolist(),
                active=active.tolist(), scalar_raw_n=raw.tolist(), scalar_response_n=contact.tolist(),
                negative_overlap_response=(contact < 0).tolist(),
                relative_equilibrium_residual=relative_residual,
                material_law=law.response, contact_force_applied=False, root_actuated=False,
                native_contact_law_parity=False, native_qualified=False,
                feature_coverage_bound=coverage_bound,
                friction_in_prediction=False, basis='frozen M/J/feature backward Euler; spring effort only')
    return _unresolved(law, max_iterations, 'iteration_limit')


def _unresolved(law, iterations, reason):
    return dict(status='unresolved', reason=reason, iterations=iterations,
        tau_spring=None, tau_joint=None, v_pred=None, material_law=law.response,
        native_contact_law_parity=False, native_qualified=False, contact_force_applied=False)


def _frames(value):
    frames = _array(value, (3, 4, 4), 'three native frames')
    for t in frames:
        if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                or not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(t[:3, :3])-1) > 1e-7):
            raise ValueError('Proper rigid frames required; no scale/reflection')
    for i in range(2):
        a = frames[i]; b = frames[i+1]
        if (np.linalg.norm(a[:3, 3]+a[:3, 2]*SPACING_M/2-b[:3, 3]+b[:3, 2]*SPACING_M/2) > 1e-5
                or np.linalg.norm(a[:3, 1]-b[:3, 1]) > 1e-4):
            raise ValueError('Coupon central anchors/hinge axes outside existing bounds')
    return frames


def _features(coupon, frames):
    """Source-seeded PCM segment points; no per-run force/position fitting.

Seed the two finite-face z edges using the ORIGINAL authored poses. PhysX
stores a capsule centerline point, not the projected cylindrical surface
point. Keep that local centerline anchor and subtract the world pad normal
times radius on every evaluation. Re-clipping each current surface at the
pad edge incorrectly moved native21's points by up to 6.347 micrometres.
An unsupported new/replaced native manifold fails the unchanged match gate.
"""
    data = layout(coupon)
    radius = float(np.float32(RADIUS_M))
    half_shaft = float(np.float32(CAPSULE_LENGTH_M-2*RADIUS_M))/2
    features = []
    for i, frame in enumerate(frames):
        for side, suffix in ((-1, '_minus'), (1, '_plus')) if coupon.held_contacts else ():
            pad_path = coupon.root+'/Pad_'+str(i)+suffix
            pad = data['pads'][pad_path]
            rp = pad[:3, :3]; center = rp.T@(frame[:3, 3]-pad[:3, 3])
            axis = rp.T@frame[:3, 2]
            normal = np.array([-side, 0., 0.])  # pad -> capsule
            # Project face normal onto the cylindrical cross-section.
            radial = normal-axis*np.dot(axis, normal)
            radial_norm = np.linalg.norm(radial)
            if abs(axis[2]) < .99 or radial_norm < .99:
                raise ValueError('Unsupported non-face capsule orientation')
            radial /= radial_norm
            face_x = -side*PAD_SIZE_M[0]/2
            seed_frame = data['frames'][i]
            seed_center = rp.T@(seed_frame[:3, 3]-pad[:3, 3])
            seed_axis = rp.T@seed_frame[:3, 2]
            for edge in (-1, 1):
                z = edge*PAD_SIZE_M[2]/2
                along = (z-center[2]+radius*radial[2])/axis[2]
                line = center+along*axis
                if abs(along) >= half_shaft-POINT_TOL_M:
                    raise ValueError('Capsule cap/finite-face clipping transition unsupported')
                if abs(line[1])+radius >= PAD_SIZE_M[1]/2-POINT_TOL_M:
                    raise ValueError('Pad y-edge/corner contact unsupported')
                seed_along = (z-seed_center[2])/seed_axis[2]
                if abs(seed_along) >= half_shaft-POINT_TOL_M:
                    raise ValueError('Source segment anchor outside cylindrical shaft')
                seed_line = seed_center+seed_along*seed_axis
                if abs(seed_line[1])+radius >= PAD_SIZE_M[1]/2-POINT_TOL_M:
                    raise ValueError('Source segment anchor outside finite pad face')
                seed_pad = np.array([face_x, seed_line[1], z])
                normal_world = rp@normal
                world = frame[:3, 3]+frame[:3, 2]*seed_along-radius*normal_world
                point = rp.T@(world-pad[:3, 3])
                gap = float(normal[0]*(point[0]-face_x))
                features.append(dict(id=pad_path+':z'+str(edge), body=i,
                    collider=data['collider_paths'][i], pad=pad_path,
                    point=world, normal=normal_world, gap=gap,
                    body_local=frame[:3, :3].T@(world-frame[:3, 3]),
                    cylinder_axis_coordinate_m=float(seed_along),
                    source_pad_anchor=pad[:3, 3]+rp@seed_pad,
                    representative_pad_edge_excess_m=max(0., abs(point[2])-PAD_SIZE_M[2]/2)))
    return features


def _kinematic_jacobian(frames):
    jac = np.zeros((3, 6, 8))
    root = frames[0, :3, 3]
    for i in range(3):
        jac[i, :3, :3] = np.eye(3); jac[i, 3:, 3:6] = np.eye(3)
        arm = frames[i, :3, 3]-root
        jac[i, :3, 3:6] = np.cross(np.eye(3), arm).T
        for j in range(i):
            axis = frames[j, :3, 1]
            anchor = frames[j, :3, 3]+frames[j, :3, 2]*SPACING_M/2
            jac[i, :3, 6+j] = np.cross(axis, frames[i, :3, 3]-anchor)
            jac[i, 3:, 6+j] = axis
    return jac


def compile_geometry(coupon, frames, velocities, nativeJac, *, native_rows,
                     generalized_velocity, step_id, native_geometry_step_id,
                     native_geometry_frames=None):
    """Return J,g,s,report, requiring a bounded original-order native comparison.

frames/velocities/Jac are the CURRENT pre-step native state, COM=body origin.
Native rows must describe contacts generated from native_geometry_frames
(defaults to current frames); their generation step must be current or one
earlier. The caller must retain that snapshot, not mislabel post-fetch poses.
Rows: collider0/1, kind, point_world_m, normal_on_0, separation_m,
impulse_on_0_ns. Friction rows are counted, not used in prediction.

Feature points use original finite-face seeded PCM segment anchors, carried
by the current capsule pose minus radius along the static world pad normal.
They are not freshly clipped to the current pad edge. Replacement manifolds
that no longer match these seeds fail closed; no adaptive position fitting.

Both candidate edge features are compared when present. Missing penetrating
features, unknown pairs, duplicate rows and geometry mismatches raise. Missing
separated features are permitted but reported; the current geometry still
provides those candidates for the bounded opening/closing solve. No impulses
are used in J/g or to choose/infer the response law.
"""
    if coupon.contact_model != 'compliant':
        raise ValueError('Unchanged compliant coupon material required')
    step_id = _integer(step_id, 'step_id')
    reference_step = _integer(native_geometry_step_id, 'native_geometry_step_id')
    if step_id-reference_step not in (0, 1):
        raise ValueError('Contact-generation reference must be current or one step old')
    if reference_step != step_id and native_geometry_frames is None:
        raise ValueError('Earlier contact generation requires its explicit frame snapshot')
    frames = _frames(frames)
    reference = frames if native_geometry_frames is None else _frames(native_geometry_frames)
    vel = _array(velocities, (3, 6), 'native world COM velocities')
    jac = _array(nativeJac, (3, 6, 8), 'native floating Jacobian')
    v = _array(generalized_velocity, (8,), 'native generalized velocity')
    analytical = _kinematic_jacobian(frames)
    jac_error = float(np.max(abs(jac-analytical)))
    if not np.allclose(jac, analytical, atol=2e-6, rtol=2e-5):
        raise ValueError('Native Jacobian does not match coupon world/COM/DOF convention')
    velocity_error = float(np.max(abs(np.einsum('bij,j->bi', jac, v)-vel)))
    if not np.allclose(np.einsum('bij,j->bi', jac, v), vel, atol=2e-5, rtol=2e-5):
        raise ValueError('Native Jacobian and same-step body velocities disagree')
    current = _features(coupon, frames); refs = _features(coupon, reference)
    rows = []
    for row in native_rows:
        if len(rows) >= MAX_ROWS:
            raise ValueError('Native contact row overflow')
        rows.append(row)
    matches = []; matched = set(); friction_count = 0
    known_pairs = {(x['collider'], x['pad']) for x in refs}
    for index, row in enumerate(rows):
        pair = (row['collider0'], row['collider1'])
        sign = 1 if pair in known_pairs else -1
        oriented_pair = pair if sign == 1 else pair[::-1]
        if oriented_pair not in known_pairs:
            raise ValueError('Unknown native collider pair; no unmodeled contact attribution')
        point = _array(row['point_world_m'], (3,), 'native contact point')
        impulse = _array(row['impulse_on_0_ns'], (3,), 'signed original impulse')
        if row['kind'] == 'friction':
            friction_count += 1
            continue
        if row['kind'] != 'normal':
            raise ValueError('Unknown contact row kind')
        normal = _array(row['normal_on_0'], (3,), 'native normal')*sign
        separation = _array(row['separation_m'], (), 'native separation').item()
        candidates = [(np.linalg.norm(point-x['point']), j, x) for j, x in enumerate(refs)
                      if (x['collider'], x['pad']) == oriented_pair]
        distance, j, feature = min(candidates, key=lambda item: (item[0], item[1]))
        normal_error = float(np.linalg.norm(normal-feature['normal']))
        separation_error = abs(separation-feature['gap'])
        if (j in matched or distance > POINT_TOL_M or normal_error > NORMAL_TOL
                or separation_error > SEPARATION_TOL_M
                or separation > CONTACT_OFFSET_SUM_M+SEPARATION_TOL_M):
            raise ValueError('Native finite-face feature mismatch/duplicate: row '+str(index))
        matched.add(j)
        matches.append(dict(row=index, feature=feature['id'],
            point_error_m=float(distance), normal_error=normal_error,
            separation_error_m=float(separation_error),
            native_separation_m=float(separation),
            signed_impulse_on_body_ns=(sign*impulse).tolist()))
    missing = [x for j, x in enumerate(refs) if j not in matched]
    if any(x['gap'] <= 0 for x in missing):
        raise ValueError('Missing penetrating native finite-face feature')
    # A previously absent feature may not silently become a loaded proxy.
    if any(current[j]['gap'] <= 0 for j in range(len(refs)) if j not in matched):
        raise ValueError('New penetrating feature lacks native manifold comparison')
    J = np.empty((len(current), 8)); g = np.empty(len(current))
    features_report = []
    for j, x in enumerate(current):
        i = x['body']; arm = x['point']-frames[i, :3, 3]
        point_jac = jac[i, :3]+np.cross(jac[i, 3:].T, arm).T
        J[j] = x['normal']@point_jac
        g[j] = x['gap']
        features_report.append(dict(id=x['id'], collider=x['collider'], pad=x['pad'],
            point_world_m=x['point'].tolist(), normal_on_body=x['normal'].tolist(),
            body_local_m=x['body_local'].tolist(), gap_m=x['gap'],
            cylinder_axis_coordinate_m=x['cylinder_axis_coordinate_m'],
            source_pad_anchor_world_m=x['source_pad_anchor'].tolist(),
            representative_pad_edge_excess_m=x['representative_pad_edge_excess_m'],
            native_reference_matched=j in matched))
    report = dict(model='coupled_contact_prediction', source_sha256=coupon.source_sha256,
        source_path=coupon.source_path, step_id=step_id, native_geometry_step_id=reference_step,
        normal_rows=len(matches), friction_rows=friction_count,
        candidate_features=len(current), matches=matches, features=features_report,
        feature_observed=[j in matched for j in range(len(current))],
        missing_separated_features=[x['id'] for x in missing],
        point_tolerance_m=POINT_TOL_M, separation_tolerance_m=SEPARATION_TOL_M,
        normal_tolerance=NORMAL_TOL, jacobian_max_error=jac_error,
        velocity_mapping_max_error=velocity_error, surface_velocity_basis='authored static pads',
        native_feature_comparison=True, native_contact_law_parity=False,
        native_qualified=False, friction_in_prediction=False, impulse_replayed=False,
        stiffness_per_normal_feature_n_m=coupon.contact_stiffness,
        damping_per_normal_feature_n_s_m=coupon.contact_damping,
        feature_model='source_seeded_persistent_capsule_segment_v1',
        gap_linearization='frozen capsule segment anchor; static pad normal; current pose')
    return J, g, np.zeros(len(current)), report
