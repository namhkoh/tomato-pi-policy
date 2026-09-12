"""One-step measured contact-load estimate for an isolated spring predictor.

Maps ORIGINAL signed normal/friction impulses through a caller-bound native
COM Jacobian. The result is an external-load ESTIMATE inside elastic_effort,
never an additional applied contact/root wrench. This explicit partition is
not a simultaneous spring/contact solve and needs native qualification.
"""
import numpy as np


def generalized_contact_load(rows, *, collider_bodies, body_com_world, com_jacobians, dt):
    com = np.asarray(body_com_world, dtype=float)
    jac = np.asarray(com_jacobians, dtype=float)
    if (com.ndim != 2 or com.shape[1] != 3 or not 1 <= len(com) <= 256
            or jac.ndim != 3 or jac.shape[:2] != (len(com), 6)
            or not 6 < jac.shape[2] <= 771
            or not np.isfinite(com).all() or not np.isfinite(jac).all()
            or type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0
            or not isinstance(rows, list) or len(rows) > 256
            or not isinstance(collider_bodies, dict) or not collider_bodies
            or any(not isinstance(p, str) or not p.startswith('/') or type(i) is not int
                   or not 0 <= i < len(com) for p, i in collider_bodies.items())):
        raise ValueError('Finite exact bounded contact/Jacobian binding required')
    wrench = np.zeros((len(com), 6))
    for row in rows:
        if row.get('kind') not in ('normal', 'friction'):
            raise ValueError('Explicit original normal/friction impulse kind required')
        a, b = row['collider0'], row['collider1']
        if a == b or not ({a, b} & collider_bodies.keys()):
            raise ValueError('Contact must involve an exact bound body collider')
        point = np.asarray(row['point_world_m'], float)
        impulse = np.asarray(row['impulse_on_0_ns'], float)
        if point.shape != (3,) or impulse.shape != (3,) or not np.isfinite([point, impulse]).all():
            raise ValueError('Finite original point and impulse required')
        for path, sign in ((a, 1.), (b, -1.)):
            if path not in collider_bodies: continue
            i = collider_bodies[path]; force = sign*impulse/dt
            wrench[i, :3] += force
            wrench[i, 3:] += np.cross(point-com[i], force)
    load = np.einsum('bij,bi->j', jac, wrench)
    if not np.isfinite(load).all(): raise ValueError('Unrepresentable contact-load estimate')
    return load


class LaggedContactLoad:
    def __init__(self, dofs):
        if type(dofs) is not int or not 6 < dofs <= 771:
            raise ValueError('Floating generalized coordinate count required')
        self.n = dofs; self.last_step = 0; self.last_command = 0
        self.load = np.zeros(dofs); self.dt = None

    def command(self, *, step, dt):
        if (type(step) is not int or step != self.last_step+1 or step != self.last_command+1
                or type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0
                or self.dt is not None and dt != self.dt):
            raise ValueError('Exactly next fixed-rate command after contact fetch required')
        self.last_command = step; self.dt = dt
        return self.load.copy(), dict(mode='lagged_native_contact_load_estimate_v1',
            command_step=step, observation_step=self.last_step if self.last_step else None,
            bootstrap_zero_load=self.last_step == 0, estimate_age_steps=1 if self.last_step else None,
            measured_impulses_used_in_predictor=True, contact_force_reapplied=False,
            root_actuation_applied=False, simultaneous_contact_solve=False,
            native_physics_qualified=False, training_eligible=False)

    def observe(self, rows, *, step, dt, guards_passed, full_normal_friction_stream, **binding):
        if (type(step) is not int or step != self.last_command or step != self.last_step+1
                or guards_passed is not True or full_normal_friction_stream is not True or dt != self.dt):
            raise ValueError('Fresh guard-accepted full native contact stream required')
        load = generalized_contact_load(rows, dt=dt, **binding)
        if load.shape != (self.n,): raise ValueError('Contact coordinate inventory changed')
        self.load = load; self.last_step = step
