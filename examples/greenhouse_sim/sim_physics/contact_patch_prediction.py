"""Coupon-only, frozen-geometry Coulomb patch approximation; no native imports.

compile_patches has compile_geometry's API plus anchor_binding=None. Return
J,g,s,report; retain report['anchor_binding'] unchanged for this manifold/epoch.
Pass report['patches'] and report['feature_observed'] to solve. Native rows
provide ONLY identities, points, normals and separations: impulses are not read.
Optional previous_predicted_seed centers one all-stick KKT candidate; every
original equation/gate is checked before acceptance, else the cold solver runs.
Only commit next_predicted_seed after successful bounded effort write. Reset it
with anchor_binding on a new epoch; neither checksum is physical certification.
The first binding requires both observed friction anchors to match the two
observed normal features. Thereafter their exact collider0-local coordinates
are retained; changed, missing or duplicate anchors fail closed.

Model: unilateral_kv_v1 normals and two circular Coulomb disks per patch.
Each disk radius is mu * SUM(patch normal forces) / 2, not mu times its nearest
normal row. At each anchor, zero relative tangent speed is stick; otherwise
force opposes slip on the disk boundary. All six free-root DOFs participate.
There is no persistent tangent spring, positional-friction bias, measured-force
warm start, guessed static anchor constraint, or root pin.

Public PhysX 5.9 source (commit 517a0073715120e114ee055b63b26c95e00d9039):
DyContactPrep.cpp halves mu for two anchors, emits two tangent rows per anchor,
and reports body0-local anchors transformed into world coordinates.
DySolverConstraints.cpp solveExtContact clamps each scalar tangent row using
the accumulated normal patch impulse and static/dynamic mu. Those row-wise
bounds and strong-friction position bias are NOT this circular-cone model.
Sources: https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/lowleveldynamics/src/DyContactPrep.cpp
https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/lowleveldynamics/src/DySolverConstraints.cpp

A bounded semismooth projection solve reports measured algebraic residuals,
not native parity. Only tau_spring is an actuation proposal. Contact forces
returned for diagnostics MUST NOT be applied; native contacts remain the sole
physical contact response. Caller must bind source M/K/C/mu, zero/read back
native joint K/C, preserve all native material/force/geometry gates and caps.
Prediction passivity is not a certificate of the split native execution.
"""
import hashlib
import json

import numpy as np

from .contact_coupled_prediction import (
    MaterialLaw, compile_geometry, solve as normal_solve, _array, _frames,
    _integer, _matrix, _features, MAX_ROWS, POINT_TOL_M,
)
from .contact_spring_probe import layout

MODEL = 'two_anchor_circular_coulomb_v1'
TOLERANCE = 1e-8
MAX_ITERATIONS = 64
MAX_BACKTRACKS = 12
# Numerical least-squares trust steps ONLY, never a term in M/K/C/contact law.
TRUST_DAMPING = (1e-8, 1e-6, 1e-4, 1e-2, 1., 100.)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def compile_patches(coupon, frames, velocities, nativeJac, *, native_rows,
                    generalized_velocity, step_id, native_geometry_step_id,
                    native_geometry_frames=None, anchor_binding=None):
    """Geometry-only binding; reject partial/changed manifolds, no impulse access.

The native reference frame and rows must be from the SAME contact-generation
snapshot, current or one ordinary step old. Initial binding is explicit;
subsequent calls MUST pass the returned binding (the pure function cannot
detect a caller discarding it). No rebinding during a running manifold/epoch.
Original collider0 owns each reported friction anchor, including reversed
headers. Pad frames are authored static; moving pads are unsupported.
"""
    rows = []
    for row in native_rows:
        if len(rows) >= MAX_ROWS:
            raise ValueError('Native contact row overflow')
        # Deliberately do not access an impulse or use its magnitude/direction.
        r = {k: row[k] for k in ('collider0', 'collider1', 'kind', 'point_world_m')}
        r['impulse_on_0_ns'] = [0., 0., 0.]  # existing geometry API requires this field
        if row['kind'] == 'normal':
            r.update(normal_on_0=row['normal_on_0'], separation_m=row['separation_m'])
        rows.append(r)
    J, g, s, report = compile_geometry(coupon, frames, velocities, nativeJac,
        native_rows=rows, generalized_velocity=generalized_velocity, step_id=step_id,
        native_geometry_step_id=native_geometry_step_id,
        native_geometry_frames=native_geometry_frames)
    # These zeros are interface placeholders, never reported as observations.
    for match in report['matches']:
        match.pop('signed_impulse_on_body_ns')
    if not all(report['feature_observed']):
        raise ValueError('Patch prediction requires every native normal feature observed')
    current = _frames(frames)
    reference = current if native_geometry_frames is None else _frames(native_geometry_frames)
    jac = _array(nativeJac, (3, 6, 8), 'native Jacobian')
    source = layout(coupon)
    refs = _features(coupon, reference)
    groups = [(i, i+1) for i in range(0, len(refs), 2)]
    binding = None
    if anchor_binding is not None:
        binding = json.loads(json.dumps(anchor_binding, allow_nan=False))
        digest = binding.pop('binding_sha256', None)
        if (digest != _digest(binding) or binding.get('model') != MODEL
                or binding.get('coupon_sha256') != _digest(coupon.report())
                or binding.get('source_sha256') != coupon.source_sha256
                or binding.get('root') != coupon.root
                or len(binding.get('patches', [])) != len(groups)):
            raise ValueError('Source/geometry anchor binding changed')
        seed_step = _integer(binding.get('seed_geometry_step_id'), 'seed geometry step')
        if seed_step > native_geometry_step_id:
            raise ValueError('Future anchor binding')
    else:
        binding = dict(model=MODEL, source_sha256=coupon.source_sha256,
            coupon_sha256=_digest(coupon.report()), root=coupon.root,
            seed_geometry_step_id=native_geometry_step_id, patches=[])

    def owner_frame(owner, fs):
        if owner in source['collider_paths']:
            return fs[source['collider_paths'].index(owner)]
        if owner in source['pads']:
            return source['pads'][owner]
        raise ValueError('Unknown friction anchor owner')

    tangent = np.zeros((len(groups), 2, 2, 8))
    patch_reports = []
    used_rows = set()
    for p, indices in enumerate(groups):
        feature = refs[indices[0]]
        pair = (feature['collider'], feature['pad'])
        relevant = [(ridx, r) for ridx, r in enumerate(rows)
                    if r['kind'] == 'friction'
                    and (r['collider0'], r['collider1']) in (pair, pair[::-1])]
        if len(relevant) != 2:
            raise ValueError('Exactly two native friction anchors per patch required')
        if len({(r['collider0'], r['collider1']) for _, r in relevant}) != 1:
            raise ValueError('Mixed original header orders inside one patch')
        anchors = []
        diagnostics = []
        existing = binding['patches'][p] if anchor_binding is not None else None
        if existing is not None and existing.get('normal_ids') != [refs[i]['id'] for i in indices]:
            raise ValueError('Patch feature identities changed')
        for a, i in enumerate(indices):
            if existing is None:
                expected = refs[i]['point']
            else:
                previous = existing['anchors'][a]
                if previous['owner'] not in pair:
                    raise ValueError('Anchor owner outside bound patch')
                local = _array(previous['local_point_m'], (3,), 'bound local anchor')
                pose = owner_frame(previous['owner'], reference)
                expected = pose[:3, 3]+pose[:3, :3]@local
            candidates = [(float(np.linalg.norm(_array(r['point_world_m'], (3,), 'anchor')-expected)),
                           ridx, r) for ridx, r in relevant if ridx not in used_rows]
            if not candidates:
                raise ValueError('Duplicate friction anchor')
            error, ridx, row = min(candidates, key=lambda item: (item[0], item[1]))
            if error > POINT_TOL_M:
                raise ValueError('Changed/unobserved friction anchor manifold')
            if existing is not None and row['collider0'] != previous['owner']:
                raise ValueError('Friction anchor original owner changed')
            used_rows.add(ridx)
            owner = row['collider0']
            pose = owner_frame(owner, reference)
            raw_point = _array(row['point_world_m'], (3,), 'anchor')
            if existing is None:
                local = pose[:3, :3].T@(raw_point-pose[:3, 3])
            anchor = dict(owner=owner, local_point_m=local.tolist())
            anchors.append(anchor)
            pose = owner_frame(owner, current)
            point = pose[:3, 3]+pose[:3, :3]@local
            b = feature['body']
            point_jac = jac[b, :3]+np.cross(jac[b, 3:].T, point-current[b, :3, 3]).T
            # Deterministic authored pad Y/Z, not directions inferred from impulses.
            basis = source['pads'][feature['pad']][:3, 1:3].T
            tangent[p, a] = basis@point_jac
            diagnostics.append(dict(native_row=ridx, reference_point_error_m=error,
                native_reference_point_world_m=raw_point.tolist(),
                current_bound_point_world_m=point.tolist(),
                tangent_basis_world=basis.tolist(), normal_feature=refs[i]['id']))
        if existing is None:
            binding['patches'].append(dict(normal_ids=[refs[i]['id'] for i in indices], anchors=anchors))
        patch_reports.append(dict(collider=pair[0], pad=pair[1], anchors=diagnostics))
    if len(used_rows) != report['friction_rows']:
        raise ValueError('Unattributed native friction anchor')
    binding['binding_sha256'] = _digest(binding)
    report.update(model=MODEL, anchor_binding=binding, patch_geometry=patch_reports,
        patches=dict(model=MODEL, tangent_jacobians=tangent.tolist(),
            surface_speeds_m_s=np.zeros((len(groups), 2, 2)).tolist(),
            normal_indices=[list(x) for x in groups], mu=coupon.friction,
            observed=[True]*len(groups),
            prediction_context=_prediction_context(coupon, binding, step_id)),
        native_impulses_read=False, impulse_replayed=False, friction_in_prediction=True,
        native_contact_law_parity=False, native_qualified=False,
        friction_law_basis='instantaneous circular Coulomb disks; no strong-friction positional bias',
        anchor_transport_basis='exact bound original collider0-local point; current rigid frame')
    return J, g, s, report


def _patch_arrays(patches, m):
    if not isinstance(patches, dict) or patches.get('model') != MODEL:
        raise ValueError('Explicit two-anchor Coulomb patch model required')
    indices = np.asarray(patches.get('normal_indices'))
    p = m//2
    if m > 12 or m % 2 or indices.shape != ((p, 2) if p else (0,)):
        raise ValueError('At most six patches, exactly two normal features each')
    if p and (indices.dtype.kind not in 'iu' or sorted(indices.ravel().tolist()) != list(range(m))):
        raise ValueError('Patch normal indices must partition every feature exactly once')
    indices = indices.astype(int).reshape(p, 2)
    T = np.asarray(patches.get('tangent_jacobians'), dtype=float)
    st = np.asarray(patches.get('surface_speeds_m_s'), dtype=float)
    if not p:
        if T.size or st.size:
            raise ValueError('Free prediction cannot contain tangent rows')
        T = np.zeros((0, 2, 2, 8)); st = np.zeros((0, 2, 2))
    T = _array(T, (p, 2, 2, 8), 'two tangent dimensions at two anchors')
    st = _array(st, (p, 2, 2), 'static surface speeds')
    if np.any(st != 0):
        raise ValueError('Coupon static pads only; no inferred surface motion')
    observed = np.asarray(patches.get('observed'))
    if observed.shape != (p,) or (p and observed.dtype != np.bool_):
        raise ValueError('Explicit patch observation mask required')
    mu = patches.get('mu')
    if isinstance(mu, (bool, np.bool_)) or not np.isscalar(mu) or not np.isfinite(mu) or mu < 0:
        raise ValueError('Finite nonnegative unchanged source mu required')
    return indices, T.reshape(4*p, 8), st.reshape(4*p), float(mu), observed.astype(bool)


def _failed(reason, iterations, **metrics):
    return dict(status='unresolved', reason=reason, iterations=iterations,
        tau_spring=None, tau_joint=None, v_pred=None, model=MODEL,
        native_qualified=False, native_contact_law_parity=False,
        contact_force_applied=False, root_actuated=False,
        measured_friction_used=False, **metrics)


SEED_SCHEMA = 'previous_predicted_patch_forces_v1'
CONTEXT_SCHEMA = 'coupon_patch_prediction_context_v1'


def _prediction_context(coupon, binding, step_id):
    return dict(schema=CONTEXT_SCHEMA, source_sha256=coupon.source_sha256,
        coupon_sha256=_digest(coupon.report()),
        anchor_binding_sha256=binding['binding_sha256'], root=coupon.root,
        dt_s=float(coupon.dt), step_id=_integer(step_id, 'prediction step'))


def _seed_inputs(patches, h, law, K, C, kc, dc, previous):
    """Checksums bind software provenance, NOT sensor authenticity or native parity."""
    context = patches.get('prediction_context') if isinstance(patches, dict) else None
    if context is None:
        if previous is not None:
            raise ValueError('Predicted seed requires compiled source/anchor/step context')
        return None, None, None
    context = json.loads(json.dumps(context, allow_nan=False))
    keys = {'schema', 'source_sha256', 'coupon_sha256', 'anchor_binding_sha256',
            'root', 'dt_s', 'step_id'}
    if not isinstance(context, dict) or set(context) != keys or context['schema'] != CONTEXT_SCHEMA:
        raise ValueError('Versioned prediction context required')
    for name in ('source_sha256', 'coupon_sha256', 'anchor_binding_sha256'):
        value = context[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Prediction context SHA256 required')
    _integer(context['step_id'], 'prediction step')
    if (not isinstance(context['root'], str) or not context['root'].startswith('/World/')
            or isinstance(context['dt_s'], bool) or not isinstance(context['dt_s'], (int, float))
            or not np.isfinite(context['dt_s']) or context['dt_s'] <= 0
            or context['dt_s'] != h):
        raise ValueError('Prediction source root/timestep mismatch')
    if not isinstance(law, MaterialLaw) or law.response != 'unilateral_kv_v1':
        raise ValueError('Explicit unilateral law required')
    constitutive = _digest(dict(law=law.response, model=MODEL,
        K=_matrix(K, 8, 'K').tolist(), C=_matrix(C, 8, 'C').tolist(),
        kc=np.asarray(kc, dtype=float).tolist(), dc=np.asarray(dc, dtype=float).tolist(),
        mu=float(patches['mu']), normal_indices=patches['normal_indices']))
    if previous is None:
        return context, constitutive, None
    seed = json.loads(json.dumps(previous, allow_nan=False))
    if not isinstance(seed, dict):
        raise ValueError('Versioned previous predicted seed required')
    seed_context = seed.get('context')
    if not isinstance(seed_context, dict):
        raise ValueError('Previous prediction context required')
    _integer(seed_context.get('step_id'), 'previous prediction step')
    digest = seed.pop('seed_sha256', None)
    expected_context = dict(context, step_id=context['step_id']-1)
    if (digest != _digest(seed) or seed.get('schema') != SEED_SCHEMA
            or seed.get('basis') != 'previous_resolved_prediction_only'
            or seed.get('context') != expected_context
            or seed.get('constitutive_sha256') != constitutive
            or set(seed) != {'schema', 'basis', 'context', 'constitutive_sha256',
                             'normal_forces_n', 'friction_forces_n'}):
        raise ValueError('Stale/changed/corrupt predicted seed provenance')
    p = len(patches['normal_indices'])
    normals = _array(seed['normal_forces_n'], (2*p,), 'previous predicted normals')
    if np.any(normals < 0):
        raise ValueError('Predicted normals must be nonnegative')
    force = np.asarray(seed['friction_forces_n'], dtype=float)
    if not p and force.size == 0:
        force = force.reshape(0, 2, 2)
    force = _array(force, (p, 2, 2), 'previous predicted tangent forces')
    return context, constitutive, force.ravel()


def _all_stick_candidate(M, q, v, K, C, J, g, s, kc, dc, h, f,
                         T, st, active, centre):
    """One mass-whitened KKT solution, nearest predicted multipliers or zero.

Rank deficiency is allowed: redundant anchor forces are not unique. The eight
velocity coordinates are NEVER reduced/pinned. This computes a proposal only;
rank truncation, normal-branch changes and cones still face original residuals.
"""
    A = M+h*C+h*h*K + J.T@((active*(h*dc+h*h*kc))[:, None]*J)
    b = M@v+h*(f-K@q+J.T@(active*(-kc*g+(dc+h*kc)*s)))
    vf = np.linalg.solve(A, b)
    D = h*np.linalg.solve(A, T.T)
    centre = np.zeros(len(T)) if centre is None else centre
    H = np.linalg.solve(np.linalg.cholesky(A), T.T).T
    U, singular, _ = np.linalg.svd(H, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0]*max(H.shape)*np.finfo(float).eps))
    rhs = st-T@(vf+D@centre)
    force = centre.copy()
    if rank:
        force += U[:, :rank]@((U[:, :rank].T@rhs)/(h*singular[:rank]**2))
    vp = vf+D@force
    speed = J@vp-s; gap = g+h*speed
    raw = -kc*gap-dc*speed
    actual_active = (gap <= 0) & (raw > 0)
    if not np.array_equal(actual_active, active):
        return None, dict(status='rejected', reason='normal_branch_changed', rank=rank)
    z = np.r_[np.where(actual_active, raw, 0.), force]
    if not np.isfinite(z).all():
        return None, dict(status='rejected', reason='nonfinite_candidate', rank=rank)
    return z, dict(status='candidate', rank=rank)


def solve(M, q, v, K, C, J, g, s, kc, dc, h, f, *, law, patches,
          feature_observed, max_iterations=64, previous_predicted_seed=None):
    """Optional seed is ONLY this solver's preceding resolved prediction.

Use compile_patches' current report['patches'] unchanged. Commit the returned
next_predicted_seed only AFTER the caller successfully writes bounded effort;
clear both seed and anchor binding at a new native epoch. No seed is emitted
on failure. Source/anchor/material/dt and contiguous steps are checked exactly.
Checksums cannot distinguish identical restarted epochs: caller owns lifecycle.
Unbound pure algebra calls remain supported cold, but cannot use/emit a seed.
A seed centers ONE all-stick KKT candidate; rejected candidates leave the
original cold semismooth initialization/fallback and every gate unchanged.
"""
    context, constitutive, centre = _seed_inputs(patches, h, law, K, C, kc, dc,
                                                previous_predicted_seed)
    result = _solve(M, q, v, K, C, J, g, s, kc, dc, h, f, law=law,
        patches=patches, feature_observed=feature_observed,
        max_iterations=max_iterations, _seed_center=centre)
    result['previous_predicted_seed_used'] = previous_predicted_seed is not None
    result['next_predicted_seed'] = None
    if context is not None and result['status'] == 'resolved':
        seed = dict(schema=SEED_SCHEMA, basis='previous_resolved_prediction_only',
            context=context, constitutive_sha256=constitutive,
            normal_forces_n=result.get('normal_forces_n', []),
            friction_forces_n=result['friction_forces_n'])
        seed['seed_sha256'] = _digest(seed)
        result['next_predicted_seed'] = json.loads(json.dumps(seed, allow_nan=False))
    return result


def _solve(M, q, v, K, C, J, g, s, kc, dc, h, f, *, law, patches,
           feature_observed, max_iterations=64, _seed_center=None, _try_all_stick=True):
    """Eight-coordinate coupon solve; no actual or lagged friction-force argument.

Returned normal/tangent forces are algebraic diagnostics ONLY, never additional
actuation. Emitted effort is -K(q+h*v_pred)-C*v_pred with EXACT zero root rows.
Unresolved/invalid/overflow gives no effort (invalid input may raise).
Normal complementarity is the explicit unilateral overlap KV approximation;
Coulomb complementarity is F=projection_disk(F-rho*relative_speed, mu*N/2).
rho is numerical inverse response, not a material gain. Semismooth Newton uses
minimum-norm linear solves for redundant sticking anchors, without modifying
M/K/C or adding physical compliance. Limits/backtracking are deterministic.
If the rank-deficient Newton direction fails, bounded damped least-squares
steps globalize this numerical root solve. Damping affects only its proposed
iteration increment; convergence is ALWAYS tested on the original undamped
physical equations. This is not added physical damping or mass regularization.
"""
    if not isinstance(law, MaterialLaw) or law.response != 'unilateral_kv_v1':
        raise ValueError('Patch model explicitly requires unilateral_kv_v1, not native sign inference')
    max_iterations = _integer(max_iterations, 'max_iterations', 1, MAX_ITERATIONS)
    q = _array(q, (8,), 'coupon q'); v = _array(v, (8,), 'coupon v')
    f = _array(f, (8,), 'known noncontact generalized force')
    M = _matrix(M, 8, 'M', positive=True)
    K = _matrix(K, 8, 'K'); C = _matrix(C, 8, 'C')
    g = np.asarray(g, dtype=float)
    if g.ndim != 1:
        raise ValueError('Normal gap vector required')
    m = len(g)
    indices, T, st, mu, observed_patches = _patch_arrays(patches, m)
    # Reuse all normal input/root validation; this also gives a force-free
    # initialization, never a native measured normal/friction warm start.
    initial = normal_solve(M, q, v, K, C, J, g, s, kc, dc, h, f,
        law=law, feature_observed=feature_observed)
    if feature_observed is None:
        raise ValueError('Explicit native normal feature observation mask required')
    if not m:
        initial.update(model=MODEL, friction_in_prediction=True, measured_friction_used=False,
            friction_force_applied=False, friction_forces_n=[], friction_modes=[],
            cone_max_violation_n=0., friction_fixed_point_residual_n=0.,
            normal_law_residual_n=0., anchor_relative_power_w=[])
        return initial
    J = _array(J, (m, 8), 'normal Jacobian')
    s = _array(s, (m,), 'normal surface speeds')
    kc = _array(kc, (m,), 'normal stiffness')
    dc = _array(dc, (m,), 'normal damping')
    observed_normals = np.asarray(feature_observed, dtype=bool)
    p = len(indices); nf = 4*p; total = m+nf
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        A = M+h*C+h*h*K
        vfree = np.linalg.solve(A, M@v+h*(f-K@q))
        B = np.vstack((J, T))
        D = h*np.linalg.solve(A, B.T)  # all six root columns retained
        JD = J@D; TD = T@D
        rho = np.zeros(2*p)
        for a in range(2*p):
            block = TD[2*a:2*a+2, m+2*a:m+2*a+2]
            eig = np.linalg.eigvalsh((block+block.T)*.5)
            if eig[0] <= 0:
                raise ValueError('Degenerate tangent anchor response')
            rho[a] = 1/eig[-1]
        identity = np.eye(total)
        z = np.zeros(total)
        if initial['status'] == 'resolved':
            z[:m] = initial['scalar_response_n']
        scale = max(float(np.max(abs(z))), float(np.max(abs(f))),
                    float(np.max(abs(K@q))), float(np.max(abs(M@v/h))), 1e-30)

        def evaluate(z, derivative):
            vp = vfree+D@z
            speed = J@vp-s; gp = g+h*speed
            raw = -kc*gp-dc*speed
            active = (gp <= 0) & (raw > 0)
            N = np.where(active, raw, 0.)
            residual = np.zeros(total)
            residual[:m] = z[:m]-N
            jacobian = identity.copy() if derivative else None
            if derivative:
                jacobian[:m] += ((h*kc+dc)*active)[:, None]*JD
            u = (T@vp-st).reshape(2*p, 2)
            cap = np.zeros(2*p)
            for a in range(2*p):
                cols = slice(m+2*a, m+2*a+2)
                ni = indices[a//2]
                normal_sum = float(np.sum(z[ni]))
                cap[a] = mu*.5*max(0., normal_sum)
                trial = z[cols]-rho[a]*u[a]
                length = float(np.linalg.norm(trial))
                if length <= cap[a]:
                    projection = trial
                    dp = np.eye(2); radius_gradient = np.zeros(2)
                elif length > 0:
                    direction = trial/length
                    projection = cap[a]*direction
                    dp = (cap[a]/length)*(np.eye(2)-np.outer(direction, direction))
                    radius_gradient = direction
                else:
                    projection = np.zeros(2)
                    dp = np.zeros((2, 2)); radius_gradient = np.zeros(2)
                residual[cols] = z[cols]-projection
                if derivative:
                    jacobian[cols] -= dp@(identity[cols]-rho[a]*TD[2*a:2*a+2])
                    if normal_sum > 0:
                        jacobian[cols, ni[0]] -= radius_gradient*mu*.5
                        jacobian[cols, ni[1]] -= radius_gradient*mu*.5
            if not all(np.isfinite(x).all() for x in (vp, residual, N, cap, u)):
                raise ValueError('Nonfinite patch prediction')
            return residual, jacobian, vp, N, gp, speed, active, cap, u

        def qualified(z, evaluated, iteration, backtracks, trust_steps):
            r, _, vp, N, gp, speed, active, cap, u = evaluated
            error = float(np.max(abs(r))/scale)
            if error > TOLERANCE:
                return None
            if np.any(active & ~observed_normals) or any(
                    np.any(active[ids]) and not observed_patches[k]
                    for k, ids in enumerate(indices)):
                return _failed('unobserved_active_feature', iteration)
            force = z[m:].reshape(2*p, 2)
            tau = -K@(q+h*vp)-C@vp
            residual = M@(vp-v)-h*(f+tau+J.T@z[:m]+T.T@z[m:])
            denom = max(float(np.linalg.norm(M@vp)), float(np.linalg.norm(M@v)),
                        h*float(np.linalg.norm(f+tau)),
                        h*float(np.linalg.norm(B.T@z)), 1e-30)
            equilibrium = float(np.linalg.norm(residual)/denom)
            violation = float(np.max(np.maximum(0., np.linalg.norm(force, axis=1)-cap)))
            power = np.einsum('ij,ij->i', force, u)
            # Verify algebraic momentum, cone feasibility and dissipativity;
            # never use a native measured contact to decide these gates.
            if (equilibrium > TOLERANCE or violation > TOLERANCE*scale
                    or np.any(tau[:6] != 0) or not np.isfinite(tau).all()):
                return _failed('equilibrium_or_cone_residual', iteration)
            # Projection variational inequality with e=F-proj(F-rho*u):
            # F.u <= |e| (|proj|/rho + |u|). Near exact stick u is
            # roundoff-small; a bound proportional only to |u| falsely
            # rejects signed-zero work. Account for the ACTUAL projection
            # residual plus floating arithmetic, not physical friction gain.
            force_norm = np.linalg.norm(force, axis=1)
            projection_error = np.linalg.norm(r[m:].reshape(2*p, 2), axis=1)
            projection_error += 32*np.finfo(float).eps*np.maximum(force_norm, scale)
            power_allowance = projection_error*((force_norm+projection_error)/rho
                                               +np.linalg.norm(u, axis=1))
            if np.any(power > power_allowance):
                return _failed('nondissipative_friction_residual', iteration)
            modes = []
            for a in range(2*p):
                if cap[a] == 0:
                    modes.append('unsupported')
                elif rho[a]*np.linalg.norm(u[a]) <= TOLERANCE*scale:
                    modes.append('stick')
                else:
                    modes.append('slip')
            return dict(status='resolved', model=MODEL, iterations=iteration,
                backtracks=backtracks, numerical_trust_steps=trust_steps, v_pred=vp.tolist(),
                tau_spring=tau.tolist(), tau_joint=tau[6:].tolist(),
                normal_forces_n=z[:m].tolist(), normal_response_n=N.tolist(),
                friction_forces_n=force.reshape(p, 2, 2).tolist(),
                anchor_limits_n=cap.reshape(p, 2).tolist(),
                friction_modes=np.asarray(modes).reshape(p, 2).tolist(),
                tangent_speeds_m_s=u.reshape(p, 2, 2).tolist(),
                anchor_relative_power_w=power.reshape(p, 2).tolist(),
                anchor_power_error_bound_w=power_allowance.reshape(p, 2).tolist(),
                normal_active=active.tolist(), gap_pred_m=gp.tolist(),
                normal_relative_speed_m_s=speed.tolist(),
                normal_law_residual_n=float(np.max(abs(r[:m]))),
                friction_fixed_point_residual_n=float(np.max(abs(r[m:]))),
                relative_fixed_point_residual=error,
                relative_equilibrium_residual=equilibrium,
                cone_max_violation_n=violation, material_law=law.response,
                mu=mu, normal_support_basis='sum predicted compressive patch normals / 2 per anchor',
                root_actuated=False, contact_force_applied=False, friction_force_applied=False,
                measured_friction_used=False, friction_in_prediction=True,
                native_contact_law_parity=False, native_qualified=False,
                feature_coverage_bound=True, friction_position_bias=False,
                basis='bounded instantaneous circular Coulomb/KV approximation; spring effort only')

        fast = dict(status='not_attempted', reason='normal_initialization_unresolved')
        if _try_all_stick and initial['status'] == 'resolved':
            try:
                candidate, fast = _all_stick_candidate(
                    M, q, v, K, C, J, g, s, kc, dc, h, f, T, st,
                    np.asarray(initial['active'], dtype=bool), _seed_center)
                if candidate is not None:
                    assessed = evaluate(candidate, False)
                    result = qualified(candidate, assessed, 0, 0, 0)
                    if result is not None and result['status'] == 'resolved':
                        result['all_stick_candidate'] = dict(fast, status='accepted')
                        return result
                    fast.update(status='rejected',
                        reason=result['reason'] if result else 'original_fixed_point_residual',
                        relative_fixed_point_residual=float(np.max(abs(assessed[0]))/scale))
            except (np.linalg.LinAlgError, FloatingPointError):
                fast = dict(status='rejected', reason='numerical_candidate_failure')
        if not _try_all_stick:
            fast = dict(status='not_attempted', reason='cold_reference')
        backtracks = 0
        trust_steps = 0
        for iteration in range(1, max_iterations+1):
            evaluated = evaluate(z, True)
            r, derivative = evaluated[:2]
            error = float(np.max(abs(r))/scale)
            result = qualified(z, evaluated, iteration, backtracks, trust_steps)
            if result is not None:
                result['all_stick_candidate'] = fast
                return result
            delta = np.linalg.lstsq(derivative, -r, rcond=None)[0]
            merit = float(r@r)
            accepted = False
            for bt in range(MAX_BACKTRACKS+1):
                alpha = 2.**(-bt)
                candidate = z+alpha*delta
                candidate[:m] = np.maximum(candidate[:m], 0.)
                trial = evaluate(candidate, False)[0]
                if float(trial@trial) < merit*(1-1e-4*alpha):
                    z = candidate
                    backtracks += bt
                    accepted = True
                    break
            if not accepted:
                for damping in TRUST_DAMPING:
                    delta = np.linalg.lstsq(
                        np.vstack((derivative, np.sqrt(damping)*identity)),
                        np.r_[-r, np.zeros(total)], rcond=None)[0]
                    for bt in range(MAX_BACKTRACKS+1):
                        alpha = 2.**(-bt)
                        candidate = z+alpha*delta
                        candidate[:m] = np.maximum(candidate[:m], 0.)
                        trial = evaluate(candidate, False)[0]
                        if float(trial@trial) < merit*(1-1e-4*alpha):
                            z = candidate
                            backtracks += bt
                            trust_steps += 1
                            accepted = True
                            break
                    if accepted:
                        break
            if not accepted:
                return _failed('semismooth_line_search', iteration, all_stick_candidate=fast,
                    relative_fixed_point_residual=error,
                    normal_law_residual_n=float(np.max(abs(r[:m]))),
                    friction_fixed_point_residual_n=float(np.max(abs(r[m:]))))
        return _failed('iteration_limit', max_iterations, all_stick_candidate=fast,
                       relative_fixed_point_residual=error,
                       normal_law_residual_n=float(np.max(abs(r[:m]))),
                       friction_fixed_point_residual_n=float(np.max(abs(r[m:]))))
