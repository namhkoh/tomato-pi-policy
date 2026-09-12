"""Bounded offline static contact-patch audit, never execution authorization.

A fixed set of supplied contact points, an inscribed eight-sided Coulomb cone,
and freely redistributed forces are approximations. Failure does not prove all
possible contacts infeasible; success proves neither a stable native grasp nor
dynamic retention. Do not use hidden geometry here as an execution target.
No state setters, controller commands, contact replay or material changes.
"""
import numpy as np
from scipy.optimize import linprog


def _array(value, shape, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite ' + label + ' with shape ' + str(shape) + ' required')
    return result


def gravity_wrench(masses, world_coms, origin, *, gravity=(0., 0., -9.81)):
    """Caller supplies only detached bodies, matching mass/COM order, in SI."""
    m = np.array(masses, dtype=float, copy=True)
    if m.ndim != 1 or not 1 <= len(m) <= 1024 or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError('Positive complete detached-body masses required')
    points = _array(world_coms, (len(m), 3), 'world COMs')
    centre = _array(origin, (3,), 'wrench origin')
    g = _array(gravity, (3,), 'gravity')
    forces = m[:, None] * g
    result = np.r_[forces.sum(0), np.cross(points-centre, forces).sum(0)]
    if not np.isfinite(result).all():
        raise ValueError('Unrepresentable gravity wrench')
    return result


def static_capacity(points, normals_on_object, finger_ids, applied_wrench, *, origin,
                    friction=.5, per_finger_budget=(.5, .5)):
    """Minimum worst-finger budget utilization for a static six-axis balance.

    normals point INTO the held object, not the fingers. Both fingers required.
    Budgets refer to sums of normal magnitudes PLUS friction upper bounds; they
    are NOT actuator forces. The result never increases or submits those limits.
    Inputs may be native samples or ideal proposals; provenance is caller-owned.
    Run outside the real-time control loop, not once per physics step.
    """
    p = np.array(points, dtype=float, copy=True)
    if p.ndim != 2 or p.shape[1:] != (3,) or not 2 <= len(p) <= 256 or not np.isfinite(p).all():
        raise ValueError('Two to 256 finite contact points required')
    n = _array(normals_on_object, p.shape, 'unit object normals')
    if not np.allclose(np.linalg.norm(n, axis=1), 1., rtol=0, atol=1e-6):
        raise ValueError('Unit object normals required; no silent orientation repair')
    ids = np.asarray(finger_ids)
    if ids.shape != (len(p),) or ids.dtype.kind not in 'iu' or set(ids.tolist()) != {0, 1}:
        raise ValueError('Exact integer identities for both fingers required')
    if (isinstance(friction, (bool, np.bool_)) or not np.isscalar(friction)
            or not np.isfinite(friction) or not 0 <= friction <= 1):
        raise ValueError('Finite friction coefficient in [0,1] required')
    caps = _array(per_finger_budget, (2,), 'per-finger contact budgets')
    if np.any(caps <= 0) or np.any(caps > .5):
        raise ValueError('Existing positive contact budgets at most 0.5 N required')
    centre = _array(origin, (3,), 'wrench origin')
    load = _array(applied_wrench, (6,), 'applied force/torque wrench')
    lever = p-centre
    # A metre-scale bound catches coordinate-frame/unit mistakes in this small
    # gripper audit. It is not a physical collision or reachability criterion.
    if np.any(np.linalg.norm(lever, axis=1) > 1):
        raise ValueError('Contact patch must be within one metre of wrench origin')
    tangent = np.cross(n, np.eye(3)[np.argmin(abs(n), axis=1)])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    bitangent = np.cross(n, tangent)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    rays = n[:, None, :] + friction*(np.cos(angles)[None, :, None]*tangent[:, None, :]
                                   + np.sin(angles)[None, :, None]*bitangent[:, None, :])
    columns = np.concatenate([rays, np.cross(lever[:, None, :], rays)], axis=2).reshape(-1, 6).T
    cone_ids = np.repeat(ids, 8)
    # Last variable is utilization, so infeasible-at-budget proposals remain
    # useful diagnostics WITHOUT relaxing any actual controller/contact limit.
    budgets = np.array([(1+friction)*(cone_ids == i) for i in (0, 1)], dtype=float)
    inequality = np.c_[budgets, -caps]
    equality = np.c_[columns, np.zeros(6)]
    objective = np.r_[np.zeros(columns.shape[1]), 1.]
    result = linprog(objective, A_eq=equality, b_eq=-load, A_ub=inequality, b_ub=np.zeros(2),
                     bounds=(0, None), method='highs', options={'time_limit': 2.,
                         'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    receipt = dict(model='fixed_patch_static_wrench_audit_v1', solver_status=int(result.status),
        solver_message=str(result.message), balance_found=False, within_contact_budgets=False,
        minimum_worst_finger_utilization=None, normal_load_per_finger_n=None,
        friction_coefficient=float(friction), contact_budgets_n=caps.tolist(), contact_points=len(p),
        applied_wrench_n_nm=load.tolist(), origin_world_m=centre.tolist(),
        assumptions=['fixed contact patch', 'inscribed 8-sided friction cones',
                     'force redistribution', 'static balance only', 'no actuator feasibility'],
        input_native_provenance_verified=False, retention_verified=False, execution_authorized=False)
    if result.status != 0 or result.x is None:
        return receipt
    weights = result.x[:-1]
    residual = columns@weights + load
    bound = budgets@weights
    verified = (np.isfinite(result.x).all() and np.all(weights >= -1e-10)
                and np.max(abs(residual[:3])) <= 1e-7 and np.max(abs(residual[3:])) <= 1e-9
                and np.all(bound <= caps*result.x[-1]+1e-8))
    receipt.update(balance_found=bool(verified), wrench_residual_n_nm=residual.tolist())
    if verified:
        receipt.update(minimum_worst_finger_utilization=float(result.x[-1]),
            normal_load_per_finger_n=[float(weights[cone_ids == i].sum()) for i in (0, 1)],
            contact_load_bounds_n=bound.tolist(), within_contact_budgets=bool(np.all(bound <= caps+1e-8)))
    return receipt
