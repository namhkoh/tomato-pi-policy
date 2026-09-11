"""Small-revolute-coupon section springs; no runtime owner or qualification.

Two massless D6 constraints at joint +/-r X, each with ONLY a transZ force
drive in body0's joint frame. Their other five coordinates are unconstrained.
The existing central Y revolute supplies the centre constraint, not these
springs. Author only before parsing a NEW diagnostic stage; never a live rig.

The ideal planar law is E=K sin(theta)^2/2, tau=-K sin(theta) cos(theta)
and tau_d=-C cos(theta)^2 omega. It matches K/C near zero, NOT at large
angles. These energy/passivity statements are continuous mathematical oracles,
not a claim that the native discrete solver realizes them. No fracture model.
"""
import math

import numpy as np


MODEL = 'off_axis_transz_section_springs_v1'
MAX_FORCE = float(np.finfo(np.float32).max)


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError('Finite scalar ' + name + ' required')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('Finite positive ' + name + ' required')
    return value


def _frame(value):
    f = np.array(value, dtype=float, copy=True)
    if (f.shape != (4, 4) or not np.isfinite(f).all()
            or not np.allclose(f[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(f[:3, :3].T @ f[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(f[:3, :3])-1) > 1e-7):
        raise ValueError('Proper rigid frame required; no scale/reflection')
    return f


def coefficients(stiffness, damping, radius):
    """SI K [Nm/rad], C [Nm s/rad] -> per-connector k [N/m], c [N s/m]."""
    k = _scalar(stiffness, 'stiffness', positive=True)
    c = _scalar(damping, 'damping', positive=True)
    r = _scalar(radius, 'radius', positive=True)
    denominator = 2*r*r
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError('Unrepresentable radius squared')
    linear = [k/denominator, c/denominator]
    if any(not math.isfinite(x) or x > MAX_FORCE or x < np.finfo(np.float32).tiny for x in linear):
        raise ValueError('Unrepresentable native linear coefficients')
    return dict(model=MODEL, radius_m=r, angular_stiffness_nm_rad=k,
                angular_damping_nm_s_rad=c, linear_stiffness_n_m=linear[0],
                linear_damping_n_s_m=linear[1], connector_count=2,
                finite_angle_law='E=K*sin(theta)^2/2; damping torque=-C*cos(theta)^2*omega',
                native_qualified=False)


def local_anchors(parent_joint_local, child_joint_local, radius):
    """(minus/plus, body0/body1, 4,4) local frames, coincident at ZERO strain.

    Joint local frames encode the zero-strain material axes. Do not derive
    fresh coincident anchors from already strained current body poses.
    """
    r = _scalar(radius, 'radius', positive=True)
    centres = [_frame(parent_joint_local), _frame(child_joint_local)]
    result = np.empty((2, 2, 4, 4))
    for i, side in enumerate((-1, 1)):
        shift = np.eye(4); shift[0, 3] = side*r
        for j, centre in enumerate(centres):
            result[i, j] = centre @ shift
    if not np.isfinite(result).all():
        raise ValueError('Overflowing local anchors')
    return result


def displacements(parent_world, child_world, parent_joint_local, child_joint_local, radius):
    """Two signed Z gaps in body0 constraint frames; not Euclidean distances."""
    a, b = _frame(parent_world), _frame(child_world)
    anchors = local_anchors(parent_joint_local, child_joint_local, radius)
    result = []
    for first, second in anchors:
        x, y = a @ first, b @ second
        result.append(float(x[:3, 2] @ (y[:3, 3]-x[:3, 3])))
    if not np.isfinite(result).all():
        raise ValueError('Overflowing section displacement')
    return np.array(result)


def planar_response(theta, omega, stiffness, damping, radius):
    """Ideal continuous planar energy/torque oracle, not native force readback."""
    p = coefficients(stiffness, damping, radius)
    q, v = _scalar(theta, 'angle'), _scalar(omega, 'angular velocity')
    s, c = math.sin(q), math.cos(q)
    z = np.array([1., -1.])*p['radius_m']*s
    dz = np.array([1., -1.])*p['radius_m']*c
    speed = dz*v
    force = -p['linear_stiffness_n_m']*z-p['linear_damping_n_s_m']*speed
    elastic = -p['angular_stiffness_nm_rad']*s*c
    viscous = -p['angular_damping_nm_s_rad']*c*c*v
    result = dict(displacement_z_m=z.tolist(), velocity_z_m_s=speed.tolist(),
        force_on_child_z_n=force.tolist(), energy_j=.5*p['angular_stiffness_nm_rad']*s*s,
        elastic_torque_nm=elastic, damping_torque_nm=viscous,
        total_torque_nm=elastic+viscous, damping_power_w=viscous*v,
        energy_rate_w=-elastic*v, total_power_w=(elastic+viscous)*v)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Overflowing section response')
    return result


def displacement_rates(parent_world, child_world, parent_velocity, child_velocity,
                       parent_joint_local, child_joint_local, radius):
    """Signed gap rates including rotation of body0's Z measurement axis.

    Velocities are WORLD [linear at body origin, angular]. This coupon authors
    COM=origin. Other callers must convert COM velocity first; no guessing.
    A common rigid translation/rotation contributes exactly zero gap rate.
    """
    a, b = _frame(parent_world), _frame(child_world)
    velocities = np.array([parent_velocity, child_velocity], dtype=float)
    if velocities.shape != (2, 6) or not np.isfinite(velocities).all():
        raise ValueError('Two finite world body-origin velocities required')
    anchors = local_anchors(parent_joint_local, child_joint_local, radius)
    result = []
    for first, second in anchors:
        x, y = a @ first, b @ second
        va = velocities[0, :3]+np.cross(velocities[0, 3:], x[:3, 3]-a[:3, 3])
        vb = velocities[1, :3]+np.cross(velocities[1, 3:], y[:3, 3]-b[:3, 3])
        axis = x[:3, 2]
        rate = axis @ (vb-va)+np.cross(velocities[0, 3:], axis) @ (y[:3, 3]-x[:3, 3])
        result.append(float(rate))
    if not np.isfinite(result).all(): raise ValueError('Overflowing section gap rates')
    return np.array(result)


def _prepare(stage, central_path, stiffness, damping, radius, topology):
    from pxr import Gf, UsdPhysics
    if topology not in ('articulated', 'maximal'):
        raise ValueError('Explicit articulated/maximal central topology required')
    p = coefficients(stiffness, damping, radius)
    prim = stage.GetPrimAtPath(central_path)
    if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
        raise ValueError('Existing central revolute required')
    joint = UsdPhysics.RevoluteJoint(prim)
    if (joint.GetAxisAttr().Get() != 'Y' or not joint.GetJointEnabledAttr().Get()
            or joint.GetExcludeFromArticulationAttr().Get() != (topology == 'maximal')
            or not joint.GetCollisionEnabledAttr().Get()):
        raise ValueError('Enabled central Y revolute with matching explicit topology/collisions required')
    if topology == 'maximal' and any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()):
        raise ValueError('Maximal coupon must have no articulation root')
    drive = UsdPhysics.DriveAPI(prim, 'angular')
    if (not drive or drive.GetStiffnessAttr().Get() != 0
            or drive.GetDampingAttr().Get() != 0
            or drive.GetTargetPositionAttr().Get() != 0
            or drive.GetTargetVelocityAttr().Get() != 0):
        raise ValueError('Angular drive must be zero BEFORE section authoring/parsing')
    bodies = []; frames = []
    for side in (0, 1):
        paths = getattr(joint, 'GetBody'+str(side)+'Rel')().GetTargets()
        if len(paths) != 1 or not stage.GetPrimAtPath(paths[0]).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Two existing rigid bodies required; never world attachment')
        if UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(paths[0])).GetKinematicEnabledAttr().Get():
            raise ValueError('Dynamic section bodies required; no kinematic support')
        bodies.append(str(paths[0]))
        quat = getattr(joint, 'GetLocalRot'+str(side)+'Attr')().Get()
        frame = np.array(Gf.Matrix4d().SetRotate(Gf.Quatd(quat)), dtype=float).T
        frame[:3, 3] = getattr(joint, 'GetLocalPos'+str(side)+'Attr')().Get()
        frames.append(_frame(frame))
    if bodies[0] == bodies[1]: raise ValueError('Distinct central bodies required')
    anchors = local_anchors(*frames, radius)
    paths = [str(central_path)+'_Section_'+side for side in ('minus', 'plus')]
    return p, bodies, anchors, paths


def author_pair(stage, central_path, *, stiffness, damping, radius, topology='articulated'):
    """Add ONLY two D6 prims; caller owns NEW pre-parse stage. Non-atomic.

    USD generic Joint has all six coordinates free when LimitAPI is absent.
    CollisionEnabled=True preserves the central coupon's collision policy.
    No bodies, masses, materials, limits, angular drives or state overrides.
    """
    from pxr import Gf, UsdPhysics
    p, bodies, anchors, paths = _prepare(stage, central_path, stiffness, damping, radius, topology)
    if any(stage.GetPrimAtPath(path) for path in paths):
        raise ValueError('New section paths required; no overwrite')
    for path, pair in zip(paths, anchors):
        joint = UsdPhysics.Joint.Define(stage, path)
        joint.CreateJointEnabledAttr(True); joint.CreateCollisionEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        for side, frame in enumerate(pair):
            getattr(joint, 'CreateBody'+str(side)+'Rel')().SetTargets([bodies[side]])
            getattr(joint, 'CreateLocalPos'+str(side)+'Attr')(Gf.Vec3f(*frame[:3, 3]))
            rotation = Gf.Matrix4d(frame.T.tolist()).ExtractRotationQuat()
            getattr(joint, 'CreateLocalRot'+str(side)+'Attr')(Gf.Quatf(rotation))
        d = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'transZ')
        d.CreateTypeAttr('force'); d.CreateStiffnessAttr(p['linear_stiffness_n_m'])
        d.CreateDampingAttr(p['linear_damping_n_s_m'])
        d.CreateTargetPositionAttr(0.); d.CreateTargetVelocityAttr(0.); d.CreateMaxForceAttr(MAX_FORCE)
    return inspect_pair(stage, central_path, stiffness=stiffness, damping=damping, radius=radius, topology=topology)


def inspect_pair(stage, central_path, *, stiffness, damping, radius, topology='articulated'):
    """USD provenance only; external D6 native coefficients are NOT verified."""
    from pxr import Gf, UsdPhysics
    p, bodies, anchors, paths = _prepare(stage, central_path, stiffness, damping, radius, topology)
    for path, pair in zip(paths, anchors):
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetTypeName() != 'PhysicsJoint':
            raise ValueError('Missing generic section D6')
        schemas = list(prim.GetAppliedSchemas())
        if ([s for s in schemas if s.startswith('PhysicsDriveAPI:')] != ['PhysicsDriveAPI:transZ']
                or any(s.startswith('PhysicsLimitAPI:') for s in schemas)
                or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            raise ValueError('Only transZ spring allowed; no limits/other drives/bodies')
        joint = UsdPhysics.Joint(prim)
        if (not joint.GetJointEnabledAttr().Get() or not joint.GetCollisionEnabledAttr().Get()
                or not joint.GetExcludeFromArticulationAttr().Get()):
            raise ValueError('Enabled external section with collisions required')
        for side, frame in enumerate(pair):
            if list(map(str, getattr(joint, 'GetBody'+str(side)+'Rel')().GetTargets())) != [bodies[side]]:
                raise ValueError('Section body binding changed')
            pos = getattr(joint, 'GetLocalPos'+str(side)+'Attr')().Get()
            quat = getattr(joint, 'GetLocalRot'+str(side)+'Attr')().Get()
            rotation = np.array(Gf.Matrix4d().SetRotate(Gf.Quatd(quat)), dtype=float).T[:3, :3]
            if (not np.allclose(pos, frame[:3, 3], atol=1e-9, rtol=2e-7)
                    or not np.allclose(rotation, frame[:3, :3], atol=1e-6, rtol=0)):
                raise ValueError('Section local anchor binding changed')
        d = UsdPhysics.DriveAPI(prim, 'transZ')
        if (d.GetTypeAttr().Get() != 'force' or d.GetTargetPositionAttr().Get() != 0
                or d.GetTargetVelocityAttr().Get() != 0 or d.GetMaxForceAttr().Get() != MAX_FORCE
                or not np.allclose([d.GetStiffnessAttr().Get(), d.GetDampingAttr().Get()],
                    [p['linear_stiffness_n_m'], p['linear_damping_n_s_m']], rtol=2e-7, atol=0)):
            raise ValueError('Section drive coefficients/targets changed')
    return dict(**p, central_joint_path=str(central_path), body_paths=bodies, joint_paths=paths,
        central_topology=topology,
        local_anchor_frames=anchors.tolist(), all_axes_unlimited=True,
        excludes_articulation=True, collision_enabled=True, authored_usd_verified=True,
        native_external_drive_coefficients_verified=False)
