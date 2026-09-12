"""Bundled Newton AVBD reference coupon, NOT a production backend replacement.

Same source SI mass/inertia/K/C and small synthetic geometry as the PhysX coupon.
Contact law differs: one-sided compression damping and regularized IPC friction,
not PhysX patch friction. No robot, asset import, grasp weld or force replay.
"""
import numpy as np

from .contact_spring_probe import (CAPSULE_LENGTH_M, RADIUS_M, SPACING_M,
    PAD_SIZE_M, INITIAL_JOINT_ANGLE_RAD, MAX_CONTACT_ROWS, layout)


def damping_ratio(stiffness, damping):
    """VBD represents physical damping C as Rayleigh coefficient C/K [seconds]."""
    k, c = np.asarray(stiffness, float), np.asarray(damping, float)
    if k.shape != c.shape or not np.isfinite(k).all() or not np.isfinite(c).all() or np.any(k <= 0) or np.any(c < 0):
        raise ValueError('Finite same-shaped positive stiffness/nonnegative damping required')
    return c/k


def build(coupon, *, device='cpu', iterations=32, friction_epsilon=.01):
    import newton
    import warp as wp
    from scipy.spatial.transform import Rotation
    if device != 'cpu' or type(iterations) is not int or iterations not in (32, 128):
        raise ValueError('Owned CPU reference with 32 or128 iterations required')
    if friction_epsilon not in (.01, .0001):
        raise ValueError('Explicit 10 mm/s or0.1 mm/s IPC friction regularization required')
    if coupon.contact_model != 'compliant' or coupon.dt != 1/240:
        raise ValueError('Original compliant240 Hz reference required')
    data = layout(coupon)
    def pose(frame):
        return wp.transform(frame[:3, 3], Rotation.from_matrix(frame[:3, :3]).as_quat())
    builder = newton.ModelBuilder(gravity=0.)
    cfg = newton.ModelBuilder.ShapeConfig(density=0., ke=coupon.contact_stiffness,
        kd=float(damping_ratio(coupon.contact_stiffness, coupon.contact_damping)),
        mu=coupon.friction, margin=0., gap=.001, restitution=0.)
    bodies=[]; shapes={}
    for i, frame in enumerate(data['frames']):
        body=builder.add_link(xform=pose(frame), mass=coupon.masses[i],
            inertia=np.asarray(coupon.inertias[i]), armature=0., lock_inertia=True,
            label=data['body_paths'][i])
        bodies.append(body)
        shape=builder.add_shape_capsule(body, radius=RADIUS_M,
            half_height=(CAPSULE_LENGTH_M-2*RADIUS_M)/2, cfg=cfg, label=data['collider_paths'][i])
        shapes[shape]=data['collider_paths'][i]
    joints=[builder.add_joint_free(bodies[0])]
    ratio=damping_ratio(coupon.stiffness, coupon.damping)
    for i in range(2):
        j=builder.add_joint_revolute(bodies[i], bodies[i+1], axis=(0.,1.,0.),
            parent_xform=wp.transform((0.,0.,SPACING_M/2), wp.quat_identity()),
            child_xform=wp.transform((0.,0.,-SPACING_M/2), wp.quat_identity()),
            target_pos=0., target_vel=0., target_ke=coupon.stiffness[i], target_kd=float(ratio[i]),
            limit_lower=-1e6, limit_upper=1e6, limit_ke=0., limit_kd=0.,
            armature=0., friction=0., label=data['joint_paths'][i])
        builder.joint_q[builder.joint_q_start[j]]=INITIAL_JOINT_ANGLE_RAD
        joints.append(j)
    builder.add_articulation(joints)
    for path, frame in data['pads'].items():
        shape=builder.add_shape_box(-1, xform=pose(frame), hx=PAD_SIZE_M[0]/2,
            hy=PAD_SIZE_M[1]/2, hz=PAD_SIZE_M[2]/2, cfg=cfg, label=path)
        shapes[shape]=path
    builder.color()
    model=builder.finalize(device=device)
    solver=newton.solvers.SolverVBD(model, iterations=iterations, friction_epsilon=friction_epsilon,
        rigid_contact_hard=False, rigid_contact_history=False,
        rigid_contact_k_start=coupon.contact_stiffness, rigid_avbd_beta=0.,
        rigid_contact_stick_motion_eps=0., rigid_contact_stick_freeze_translation_eps=0.,
        rigid_contact_stick_freeze_angular_eps=0.)
    receipt=check_model(model, coupon)
    receipt.update(engine=newton.__version__, warp=wp.__version__, device=device,
        iterations=iterations, joint_damping_convention='C=K*kd; kd=C/K seconds',
        contact_law='unilateral_penalty_with_compression_only_damping_and_regularized_IPC_friction',
        friction_epsilon_m_s=solver.friction_epsilon, physx_contact_law_parity=False,
        hard_contact=False, static_friction_patch_equivalence=False,
        hidden_state_replay=False, external_body_wrench=False, production_qualified=False)
    return model, solver, shapes, receipt


def check_model(model, coupon):
    """Read actual arrays; no trusting builder intent or implicit mass defaults."""
    observed={}
    expected=dict(body_mass=coupon.masses, body_inertia=coupon.inertias,
        body_com=np.zeros((3,3)), joint_target_ke=np.r_[np.zeros(6),coupon.stiffness],
        joint_target_kd=np.r_[np.zeros(6),damping_ratio(coupon.stiffness,coupon.damping)])
    for name, target in expected.items():
        actual=np.asarray(getattr(model,name).numpy(), float); target=np.asarray(target,float)
        if actual.shape != target.shape or not np.isfinite(actual).all() or not np.allclose(actual,target,rtol=2e-6,atol=1e-14):
            raise RuntimeError('Newton source parameter mismatch: '+name)
        observed[name]=actual.tolist()
    for name,target in (('shape_material_ke',coupon.contact_stiffness),
                        ('shape_material_kd',float(damping_ratio(coupon.contact_stiffness,coupon.contact_damping))),
                        ('shape_material_mu',coupon.friction)):
        actual=np.asarray(getattr(model,name).numpy(),float)
        if actual.shape != (3+(6 if coupon.held_contacts else 0),) or not np.allclose(actual,target,rtol=2e-6,atol=1e-10):
            raise RuntimeError('Newton source material mismatch: '+name)
        observed[name]=actual.tolist()
    return observed


def contact_rows(*, shape0, shape1, points0, points1, normals, forces1, shapes, dt):
    """Split the native total force once; retain BOTH physical application points.

    Newton normal and reported total force act on shape1. Our oracle rows act on
    shape0; reverse both signs, never absolute-value the applied force.
    """
    a=np.asarray(shape0); b=np.asarray(shape1); n=len(a)
    vectors=[np.asarray(x,float) for x in (points0,points1,normals,forces1)]
    if a.shape != (n,) or b.shape != (n,) or n*2 > MAX_CONTACT_ROWS or not np.isfinite(dt) or dt<=0:
        raise ValueError('Bounded contact count and positive timestep required')
    if any(x.shape!=(n,3) or not np.isfinite(x).all() for x in vectors):
        raise ValueError('Complete finite contact vectors required')
    p0,p1,norm,f1=vectors; rows=[]
    for i in range(n):
        if int(a[i]) not in shapes or int(b[i]) not in shapes or a[i]==b[i] or abs(np.linalg.norm(norm[i])-1)>1e-5:
            raise ValueError('Known distinct shape indices and unit native normal required')
        normal_force=float(f1[i]@norm[i])
        if normal_force < -1e-7:
            raise ValueError('Native force/normal sign contract violated')
        normal1=normal_force*norm[i]
        for kind,force in (('normal',normal1),('friction',f1[i]-normal1)):
            row=dict(collider0=shapes[int(a[i])],collider1=shapes[int(b[i])],kind=kind,
                point_world_m=p0[i].tolist(),point_on_1_world_m=p1[i].tolist(),
                impulse_on_0_ns=(-force*dt).tolist())
            if kind=='normal':
                row.update(normal_on_0=(-norm[i]).tolist(),separation_m=float((p1[i]-p0[i])@norm[i]))
            rows.append(row)
    return rows
