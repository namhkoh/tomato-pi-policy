"""Preparations for a SMALL contact-coupled spring experiment; no app/step owner.

Three free-root links, two Y revolute joints and six static pads.
contact_model='compliant' preserves source contact K/C. 'rigid_control' is an
explicit diagnostic override: retain those coefficients in reports, omit their
material authoring. Geometry, friction, joint K/C and all gates are unchanged.
Only contact loads the links: no world joint, root pin, injected body wrench,
gravity compensation, or contact-force replay. Outer pad pairs are tilted in
opposite directions. Actual net contact wrench must balance; symmetry is NOT
accepted as evidence of force neutrality. This deliberately elementary planar
coupon cannot qualify the original spherical-joint plant or tissue mechanics.

Alternative model='section_springs': author(stage,coupon,model=...) adds four
external transZ D6 springs and authors both central angular drives at zero.
bind(...,model=...,stage=stage) checks their USD bindings and exact native zero
angular k/c; no predictor. Native external D6 gains are NOT read back here.
Pass the same model to sample(...,model=...). assess_tail infers it from rows
and uses K*sin(theta)*cos(theta), with all existing numerical gates unchanged.

section_springs_maximal uses the SAME bodies/rest frames and section fibers,
but no articulation API: central revolutes are external. bind_rigid returns
(report, None), requiring native rows in the exact layout body_paths order.
rigid_coordinates(world_frames, world_COM_velocities) returns (q,qdot,axis_errors):
geometric angles, projected relative native angular velocities, and hinge-axis
misalignment. These are NOT articulation readbacks. Pass model to author,
settings_readback and sample. Free-mode assess_tail additionally requires
whole_run_samples from step 1; a settled tail cannot hide earlier energy growth.
angular_d6_maximal instead uses two generic D6s (only rotY free/driven), no
fibers. Pass its model explicitly to bind_rigid; K/C use the same pi/180 USD
conversion. Energy/K*q are small-angle references, not native drive readbacks.

Caller workflow (in a NEW, stopped diagnostic stage, never production):
    coupon = from_report(path_to_original_native52_report)
    layout = author(stage, coupon)                 # before physics parsing
    view = simulation_view.create_articulation_view(layout['body_paths'][0])
    binding, predictor = bind(view, coupon, model='native')
    # Caller owns full normal+friction subscription, stepping and recording.
    # predictor is None for native, or the EXISTING unknown-contact predictor.
    # If present: predictor.step(dt, root_constrained=False) before each step.
    # After fetch: sample(...), then assess_tail(last_half_second, coupon).

Use IDENTICAL coupon parameters for native and implicit_effort comparisons.
Only iterations (16/4, 32/0, 128/32), timestep and whole-coupon orientation may vary
as labelled numerical comparisons. No mass/gain/friction tuning is provided.
The probe does not launch Kit, create tensor views, subscribe, step, or write
artifacts. Native execution and same-step sensor provenance remain caller-owned.

Earlier joint_probe_20260910_12/13: TGS fixed-base angle agreement hid residual
velocity; PGS fixed-base passed, externally anchored rotated variants failed.
Those reports omit native iteration readback. _body also authored 16/4 on every
link, so root-only 32/0 was not evidence of a uniform effective 32/0. We author
the requested pair on every body AND the articulation and label USD readback
honestly: installed tensors api.py has no solver-iteration getter; the property
query articulation response does not expose iterations either.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SPACING_M = .03
CAPSULE_LENGTH_M = .025
RADIUS_M = .0028
PAD_COMPRESSION_M = .00005
PAD_TILT_RAD = .01
INITIAL_JOINT_ANGLE_RAD = .005
INITIAL_Q_TOLERANCE_RAD = 1e-6
PAD_SIZE_M = (.008, .016, .016)
MAX_CONTACT_ROWS = 256
MODELS = ('native', 'implicit_effort', 'section_springs', 'section_springs_maximal', 'angular_d6_maximal',
          'native_damping_explicit_stiffness', 'coupled_contact_prediction')
SECTION_MODELS = ('section_springs', 'section_springs_maximal')
MAXIMAL_MODELS = ('section_springs_maximal', 'angular_d6_maximal')
D6_LOCKED_AXES = ('transX', 'transY', 'transZ', 'rotX', 'rotZ')
CONTACT_MODELS = ('compliant', 'rigid_control')
INITIAL_FRAME_POSITION_TOLERANCE_M = 1e-6
INITIAL_FRAME_ANGLE_TOLERANCE_RAD = 1e-6
# Predeclared diagnostic numerical allowance, NOT an inferred native error bound.
# Referenced to initial energy, not a growing maximum; no accumulated allowance.
ENERGY_ROUNDOFF_ATOL_J = 1e-9
ENERGY_ROUNDOFF_RTOL = 1e-4


def _model(model):
    if model not in MODELS:
        raise ValueError('Explicit comparison model required')
    return model


def _array(value, shape, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite ' + label + ' with shape ' + str(shape) + ' required')
    return result


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError('Integer ' + label + ' required')
    return int(value)


def _rotation(value):
    r = _array(value, (3, 3), 'proper rotation')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(r)-1) > 1e-8:
        raise ValueError('Proper rotation required; no scaling/reflection')
    return r


def _poses(value):
    frames = _array(value, (3, 4, 4), 'three native body frames')
    for frame in frames:
        if not np.allclose(frame[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
            raise ValueError('Rigid native frame required')
        r = frame[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or abs(np.linalg.det(r)-1) > 1e-6:
            raise ValueError('Rigid native rotation required')
    return frames


def _ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass(frozen=True)
class Coupon:
    source_path: str
    source_sha256: str
    stiffness: tuple
    damping: tuple
    masses: tuple
    inertias: tuple
    contact_stiffness: float
    contact_damping: float
    friction: float
    root: str = '/World/ContactSpringProbe'
    rotation: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    iterations: tuple = (16, 4)
    dt: float = 1/240
    solver: str = 'PGS'
    held_contacts: bool = True
    contact_model: str = 'compliant'

    def __post_init__(self):
        if (not isinstance(self.source_path, str) or not self.source_path
                or len(self.source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.source_sha256)):
            raise ValueError('Source report path/SHA256 required')
        if (not isinstance(self.root, str) or not self.root.startswith('/World/')
                or any(not p.isidentifier() for p in self.root[1:].split('/'))):
            raise ValueError('New absolute USD root required')
        for name, count in (('stiffness', 2), ('damping', 2), ('masses', 3)):
            v = _array(getattr(self, name), (count,), name)
            if np.any(v <= 0): raise ValueError('Positive ' + name + ' required')
            object.__setattr__(self, name, tuple(v.tolist()))
        inertia = _array(self.inertias, (3, 3, 3), 'inertias')
        for m in inertia:
            if np.max(abs(m-m.T)) > 1e-5*np.max(abs(m)) + 1e-15:
                raise ValueError('Symmetric physical inertia required')
            eig = np.linalg.eigvalsh((m+m.T)/2)
            if min(eig) <= 0 or max(eig) > sum(eig)-max(eig)+1e-14:
                raise ValueError('Positive physical inertia required')
        object.__setattr__(self, 'inertias', tuple(tuple(tuple(row) for row in m) for m in inertia))
        object.__setattr__(self, 'rotation', tuple(tuple(row) for row in _rotation(self.rotation)))
        if len(self.iterations) != 2: raise ValueError('Position/velocity iteration pair required')
        iterations = tuple(_integer(v, 'iterations') for v in self.iterations)
        if iterations not in ((16, 4), (32, 0), (128, 32)):
            raise ValueError('Only explicit 16/4, 32/0 and 128/32 numerical comparisons supported')
        object.__setattr__(self, 'iterations', iterations)
        if self.solver not in ('PGS', 'TGS'): raise ValueError('Explicit native solver required')
        if type(self.held_contacts) is not bool: raise ValueError('Explicit held/free contact control required')
        if not isinstance(self.contact_model, str) or self.contact_model not in CONTACT_MODELS:
            raise ValueError('Explicit compliant or rigid_control contact model required')
        for v in (self.dt, self.contact_stiffness, self.contact_damping, self.friction):
            if isinstance(v, (bool, np.bool_)) or not np.isscalar(v) or not np.isfinite(v) or v <= 0:
                raise ValueError('Positive finite timestep/material values required')
        if not 1/1920 <= self.dt <= 1/60: raise ValueError('Bounded diagnostic timestep required')

    def report(self):
        result = asdict(self)
        result.update(model='three_link_native_contact_spring_coupon_v1',
            coefficient_source='native report Joint_002:1 and Joint_003:1, unchanged SI k/c',
            inertia_source='native report bodies Segment_001..003; no mass/inertia inflation',
            geometry='synthetic straight capsules/pads, NOT original plant geometry',
            gravity_m_s2=0., applied_body_wrenches='none', root_weld=False,
            expected_root=('floating; held only by native '+self.contact_model+' contact' if self.held_contacts
                           else 'floating; contact-free recovery control'),
            source_contact_model='compliant', contact_model_override=self.contact_model != 'compliant',
            source_compliant_contact_coefficients=dict(stiffness_n_m=self.contact_stiffness,
                damping_n_s_m=self.contact_damping, force_based=True),
            compliant_contact_coefficients_authored=self.contact_model == 'compliant',
            contact_model_scope='new diagnostic coupon only; not production contact qualification',
            nominal_pad_inner_gap_m=2*(RADIUS_M-PAD_COMPRESSION_M),
            nominal_shaft_diameter_m=2*RADIUS_M,
            nominal_rigid_pinch_interference_m=2*PAD_COMPRESSION_M,
            rigid_held_control_ineligible=self.held_contacts and self.contact_model == 'rigid_control',
            rigid_held_control_limitation='Centered rigid pinch cannot fit the authored gap; escape is not held qualification',
            initial_joint_angle_rad=INITIAL_JOINT_ANGLE_RAD,
            training_eligible=False, native_qualified=False)
        return json.loads(json.dumps(result, allow_nan=False))


def from_report(path, **numerical_options):
    """Copy coefficients, not fitted values; hash the exact bytes first."""
    if set(numerical_options) - {'root', 'rotation', 'iterations', 'dt', 'solver', 'held_contacts', 'contact_model'}:
        raise ValueError('Only explicitly labelled numerical/placement variants allowed')
    path = Path(path).resolve(); raw = path.read_bytes(); report = json.loads(raw)
    native = report['native_drive_parameters']; names = native['names']
    indices = [names.index(n) for n in ('Joint_002:1', 'Joint_003:1')]
    k = _array(native['stiffness'], (1, len(names)), 'native stiffness')[0, indices]
    c = _array(native['damping'], (1, len(names)), 'native damping')[0, indices]
    mass = np.asarray(native['native_masses'], float).reshape(-1)[1:4]
    inertia = np.asarray(native['native_inertias'], float).reshape(-1, 3, 3)[1:4]
    material = report['robot_probe']['finger_contact_compliance']
    if material.get('force_based') is not True: raise ValueError('Force-based contact material required')
    return Coupon(str(path), hashlib.sha256(raw).hexdigest(), tuple(k), tuple(c), tuple(mass),
        tuple(inertia), material['stiffness_n_m'], material['damping_n_s_m'],
        report['configuration']['finger_friction'], **numerical_options)


def layout(coupon):
    r = np.asarray(coupon.rotation); frames = np.tile(np.eye(4), (3, 1, 1))
    # Identical small initial strain in held and contact-free controls. This
    # is pre-start authoring, never a running state override.
    for i in range(3): frames[i, :3, :3] = r @ _ry((i-1)*INITIAL_JOINT_ANGLE_RAD)
    frames[0, :3, 3] = r @ [0, 0, -SPACING_M]
    for i in range(1, 3):
        frames[i, :3, 3] = (frames[i-1, :3, 3]
            + frames[i-1, :3, :3] @ [0, 0, SPACING_M/2]
            + frames[i, :3, :3] @ [0, 0, SPACING_M/2])
    bodies = [coupon.root + '/Link_' + str(i) for i in range(3)]
    pads = {}
    for i, angle in enumerate((-PAD_TILT_RAD, 0., PAD_TILT_RAD)) if coupon.held_contacts else ():
        pad_r = r @ _ry(angle)
        for side in (-1, 1):
            pose = np.eye(4); pose[:3, :3] = pad_r
            pose[:3, 3] = frames[i, :3, 3] + pad_r @ [side*(RADIUS_M-PAD_COMPRESSION_M+PAD_SIZE_M[0]/2), 0, 0]
            pads[coupon.root + '/Pad_' + str(i) + ('_minus' if side < 0 else '_plus')] = pose
    return dict(body_paths=bodies, collider_paths=[p+'/Collider' for p in bodies],
        joint_paths=[coupon.root+'/Joint_001', coupon.root+'/Joint_002'],
        frames=frames, pads=pads, scene_path=coupon.root+'/Physics')


def author(stage, coupon, *, model='native'):
    """Author only an empty diagnostic stage; never call on parsed/live actors.

    Pre-start ownership is a caller obligation. No timeline import or startup.
    Mutation is non-atomic; discard this NEW stage if authoring raises.
    """
    model = _model(model)
    maximal = model in MAXIMAL_MODELS
    angular_d6 = model == 'angular_d6_maximal'
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    from .plant import _body, _collision, matrix_attr, physics_schema
    if any(str(p.GetPath()) != '/World' for p in stage.Traverse()):
        raise ValueError('Empty standalone stage required; no existing actors/assets')
    data = layout(coupon)
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, coupon.root)
    scene = UsdPhysics.Scene.Define(stage, data['scene_path'])
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(0.)
    physics_schema(scene.GetPrim(), 'PhysxSceneAPI', [
        ('physxScene:solverType', Sdf.ValueTypeNames.Token, coupon.solver),
        ('physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool, False),
        ('physxScene:broadphaseType', Sdf.ValueTypeNames.Token, 'MBP'),
        ('physxScene:frictionType', Sdf.ValueTypeNames.Token, 'patch')])
    material = UsdShade.Material.Define(stage, coupon.root+'/ContactMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(coupon.friction); api.CreateDynamicFrictionAttr(coupon.friction)
    api.CreateRestitutionAttr(0.)
    material_attrs = [
        ('physxMaterial:compliantContactStiffness', Sdf.ValueTypeNames.Float, coupon.contact_stiffness),
        ('physxMaterial:compliantContactDamping', Sdf.ValueTypeNames.Float, coupon.contact_damping),
        ('physxMaterial:compliantContactAccelerationSpring', Sdf.ValueTypeNames.Bool, False)
    ] if coupon.contact_model == 'compliant' else []
    material_attrs.append(('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token, 'min'))
    physics_schema(material.GetPrim(), 'PhysxMaterialAPI', material_attrs)
    for i, path in enumerate(data['body_paths']):
        m = np.array(coupon.inertias[i]); eigen, axes = np.linalg.eigh((m+m.T)/2)
        if np.linalg.det(axes) < 0: axes[:, 0] *= -1
        body = _body(stage, path, data['frames'][i],
            dict(mass=coupon.masses[i], inertia=eigen, principal_axes=axes), kinematic=False)
        physics_schema(body, 'PhysxRigidBodyAPI', [
            ('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, coupon.iterations[0]),
            ('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, coupon.iterations[1]),
            ('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, 0.)])
        if i == 0 and not maximal:
            UsdPhysics.ArticulationRootAPI.Apply(body)
            physics_schema(body, 'PhysxArticulationAPI', [
                ('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool, True),
                ('physxArticulation:sleepThreshold', Sdf.ValueTypeNames.Float, 0.),
                ('physxArticulation:solverPositionIterationCount', Sdf.ValueTypeNames.Int, coupon.iterations[0]),
                ('physxArticulation:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, coupon.iterations[1])])
        shape = UsdGeom.Capsule.Define(stage, data['collider_paths'][i])
        shape.CreateAxisAttr('Z'); shape.CreateRadiusAttr(RADIUS_M)
        shape.CreateHeightAttr(CAPSULE_LENGTH_M-2*RADIUS_M); _collision(shape.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose='physics')
    for i, path in enumerate(data['joint_paths']):
        joint = (UsdPhysics.Joint if angular_d6 else UsdPhysics.RevoluteJoint).Define(stage, path)
        joint.CreateBody0Rel().SetTargets([data['body_paths'][i]])
        joint.CreateBody1Rel().SetTargets([data['body_paths'][i+1]])
        if not angular_d6: joint.CreateAxisAttr('Y')
        joint.CreateJointEnabledAttr(True)
        if angular_d6:
            for axis in D6_LOCKED_AXES:
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1.); limit.CreateHighAttr(-1.)
        joint.CreateExcludeFromArticulationAttr(maximal); joint.CreateCollisionEnabledAttr(True)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, SPACING_M/2))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, -SPACING_M/2))
        joint.CreateLocalRot0Attr(Gf.Quatf(1)); joint.CreateLocalRot1Attr(Gf.Quatf(1))
        if not maximal:
            physics_schema(joint.GetPrim(), 'PhysicsJointStateAPI:angular', [
                ('state:angular:physics:position', Sdf.ValueTypeNames.Float, math.degrees(INITIAL_JOINT_ANGLE_RAD)),
                ('state:angular:physics:velocity', Sdf.ValueTypeNames.Float, 0.)])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'rotY' if angular_d6 else 'angular')
        drive.CreateTypeAttr('force'); drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
        # Section mode has NO parallel angular spring, including bootstrap.
        drive.CreateStiffnessAttr(0. if model in SECTION_MODELS else coupon.stiffness[i]*math.pi/180)
        drive.CreateDampingAttr(0. if model in SECTION_MODELS else coupon.damping[i]*math.pi/180)
        drive.CreateMaxForceAttr(float(np.finfo(np.float32).max))
    if model in SECTION_MODELS:
        from .section_springs import author_pair
        data['section_springs'] = [author_pair(stage, path, stiffness=coupon.stiffness[i],
            damping=coupon.damping[i], radius=RADIUS_M, topology='maximal' if maximal else 'articulated')
            for i, path in enumerate(data['joint_paths'])]
    for path, pose in data['pads'].items():
        cube = UsdGeom.Cube.Define(stage, path); cube.CreateSizeAttr(1.)
        matrix_attr(cube.GetPrim(), pose @ np.diag([*PAD_SIZE_M, 1.]))
        _collision(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
    data['comparison_model'] = model
    data['contact_model'] = coupon.contact_model
    data['native_view_kind'] = 'rigid_body' if maximal else 'articulation'
    return data


def settings_readback(stage, coupon, *, actual_dt, model='native'):
    """USD after-reset readback, explicitly NOT effective native iterations."""
    model = _model(model)
    from pxr import UsdGeom, UsdPhysics
    data = layout(coupon); scene = stage.GetPrimAtPath(data['scene_path'])
    observed = dict(solver=scene.GetAttribute('physxScene:solverType').Get(),
        gpu=scene.GetAttribute('physxScene:enableGPUDynamics').Get(),
        gravity=scene.GetAttribute('physics:gravityMagnitude').Get(),
        friction=scene.GetAttribute('physxScene:frictionType').Get())
    if (not np.isfinite(actual_dt) or abs(actual_dt-coupon.dt) > 1e-12
            or observed != dict(solver=coupon.solver, gpu=False, gravity=0., friction='patch')
            or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
        raise ValueError('Context changed the requested diagnostic configuration')
    pairs = []
    checked = [(p, 'physxRigidBody') for p in data['body_paths']]
    if model in MAXIMAL_MODELS:
        if any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()):
            raise ValueError('Maximal coupon must have no articulation root')
    else:
        checked.append((data['body_paths'][0], 'physxArticulation'))
    for path, api in checked:
        prim = stage.GetPrimAtPath(path)
        pair = [prim.GetAttribute(api+':solver'+which+'IterationCount').Get() for which in ('Position', 'Velocity')]
        if tuple(pair) != coupon.iterations: raise ValueError('Authored iteration settings changed')
        pairs.append(dict(path=path, api=api, iterations=pair))
    return dict(scene_usd_after_reset=observed, actual_physics_dt_s=float(actual_dt),
        authored_iteration_readback=pairs, native_iteration_readback=None,
        effective_native_iterations_verified=False,
        limitation='Installed tensor/property-query APIs expose no exact iteration getter')


def bind(articulation, coupon, *, model='native', stage=None, contact_law=None, patch_friction=False):
    """Check unstepped native initial strain and coefficients before any writes.

    Call after parsing/reset, BEFORE the first physics step. Initial USD state
    attributes alone do not establish native excitation. Do not repair a bad
    native state here or silently accept an already-stepped, relaxed coupon.
    """
    model = _model(model)
    coupled = model == 'coupled_contact_prediction'
    if type(patch_friction) is not bool or (patch_friction and not coupled):
        raise ValueError('Boolean patch friction only valid for coupled diagnostic')
    if coupled:
        from .contact_coupled_prediction import MaterialLaw
        if not isinstance(contact_law, MaterialLaw):
            raise ValueError('Explicit contact material law required for coupled diagnostic')
        if patch_friction and contact_law.response != 'unilateral_kv_v1':
            raise ValueError('Patch friction requires unilateral_kv_v1')
    elif contact_law is not None:
        raise ValueError('Contact material prediction law only valid for coupled diagnostic')
    if model in MAXIMAL_MODELS:
        raise ValueError('Maximal model requires bind_rigid, not articulation readbacks')
    sections = None
    if model == 'section_springs':
        if stage is None: raise ValueError('Section binding requires its authored stage')
        from .section_springs import inspect_pair
        sections = [inspect_pair(stage, path, stiffness=coupon.stiffness[i],
            damping=coupon.damping[i], radius=RADIUS_M) for i, path in enumerate(layout(coupon)['joint_paths'])]
    data = layout(coupon); a = articulation
    if (a.count != 1 or bool(a.shared_metatype.fixed_base)
            or list(a.link_paths[0]) != data['body_paths']
            or list(a.shared_metatype.dof_names) != ['Joint_001', 'Joint_002']):
        raise ValueError('Exact floating three-link/two-revolute native articulation required')
    initial_q = _array(a.get_dof_positions(), (1, 2), 'initial native joint positions')[0]
    if np.any(np.abs(initial_q-INITIAL_JOINT_ANGLE_RAD) > INITIAL_Q_TOLERANCE_RAD):
        raise ValueError('Initial native joint strain differs from authored 0.005 rad; '
                         'actual_q_rad='+repr(initial_q.tolist())+'; missing excitation or already stepped')
    for value, expected, label in ((a.get_dof_stiffnesses(), coupon.stiffness, 'stiffness'),
            (a.get_dof_dampings(), coupon.damping, 'damping')):
        v = _array(value, (1, 2), 'native '+label)[0]
        if model == 'section_springs' and np.any(v != 0):
            raise ValueError('Section native angular '+label+' must already be exactly zero')
        if model != 'section_springs' and not np.allclose(v, expected, rtol=2e-6, atol=1e-10):
            raise ValueError('Native SI '+label+' differs; do not retune')
    masses = _array(a.get_masses(), (1, 3), 'native masses')[0]
    if not np.allclose(masses, coupon.masses, rtol=2e-6, atol=1e-12):
        raise ValueError('Native masses differ')
    prediction = None
    if model == 'implicit_effort':
        from .implicit_springs import ImplicitJointSprings
        prediction = ImplicitJointSprings(a)
    split = model == 'native_damping_explicit_stiffness'
    if split: prediction = NativeDampingExplicitStiffness(a, coupon)
    if coupled:
        from .contact_coupled_native import ContactCoupledCouponSpring
        prediction = ContactCoupledCouponSpring(a, coupon, law=contact_law, patch_friction=patch_friction)
    return dict(model=model, source_sha256=coupon.source_sha256, contact_model=coupon.contact_model,
        initial_native_q_rad=initial_q.tolist(), expected_initial_q_rad=[INITIAL_JOINT_ANGLE_RAD]*2,
        initial_native_q_verified=True, initial_q_tolerance_rad=INITIAL_Q_TOLERANCE_RAD,
        native_fixed_base=False, native_si_coefficients_verified=model != 'section_springs' and not split and not coupled,
        initial_native_si_coefficients_verified=model != 'section_springs',
        native_drives_disabled=model == 'implicit_effort' or coupled, root_constraint=False,
        native_angular_drives_disabled=model not in ('native', 'native_damping_explicit_stiffness'),
        external_section_drives_authored=model == 'section_springs',
        disabled_drive_scope='angular stiffness only; native damping retained' if split else 'central articulation angular drives only',
        native_angular_drives_zero_verified=model == 'section_springs',
        section_springs=sections, external_d6_native_coefficients_verified=False,
        contact_prediction=('finite normal and circular patch friction prediction; spring effort only' if patch_friction else
                            'finite normal feature implicit prediction; spring effort only' if coupled else
                            'none; explicit stiffness with native damping/contacts' if split else
                            'omitted_legacy_control' if model == 'implicit_effort' else 'native_solver_coupled'),
        split_scheme=prediction.report() if split else None,
        coupled_prediction=prediction.report() if coupled else None,
        native_iteration_readback=None, native_qualified=False), prediction


class NativeDampingExplicitStiffness:
    """Coupon-only split integrator: exact -K*q command, ONLY native C active.

    Construct through bind(), after original native K/C/initial-q verification.
    Original angular K remains during the runner's disclosed insertion bootstrap.
    No mass matrix, contact replay, root effort, state override or gain fitting.
    Constant-M theory assumes q_new=q+h*v_new; native realization is unqualified.
    """
    def __init__(self, articulation, coupon):
        self.articulation = articulation; self.dt = coupon.dt
        self.indices = np.array([0], dtype=np.uint32)
        self.k = _array(articulation.get_dof_stiffnesses(), (1,2), 'initial native K')[0]
        self.c = _array(articulation.get_dof_dampings(), (1,2), 'initial native C')[0]
        self.caps = _array(articulation.get_dof_max_forces(), (1,2), 'native drive caps')[0]
        # Installed tensors api.py get_drive_types: 0=None, 1=Force, 2=Acceleration.
        self.drive_types = _array(articulation.get_drive_types(), (1,2), 'native drive types')[0]
        if np.any(self.drive_types != 1): raise ValueError('Native Force drive type 1 required')
        self.paths = list(articulation.link_paths[0])
        self.names = list(articulation.shared_metatype.dof_names)
        if (np.any(self.k <= 0) or np.any(self.c <= 0) or np.any(self.caps <= 0)
                or np.any(self.c-self.dt*self.k/2 < 0)
                or np.any(_array(articulation.get_dof_velocity_targets(), (1,2), 'native velocity targets') != 0)):
            raise ValueError('Positive K/C/caps, zero velocity target and C >= h*K/2 required')
        articulation.set_dof_stiffnesses(np.zeros((1,2), dtype=np.float32), self.indices)
        self._check()  # A failed readback aborts this diagnostic, never runs it.

    def _check(self):
        a = self.articulation
        if (a.count != 1 or a.shared_metatype.fixed_base or list(a.link_paths[0]) != self.paths
                or list(a.shared_metatype.dof_names) != self.names
                or np.any(_array(a.get_dof_stiffnesses(), (1,2), 'native K') != 0)
                or not np.array_equal(_array(a.get_dof_dampings(), (1,2), 'native C')[0], self.c)
                or not np.array_equal(_array(a.get_dof_max_forces(), (1,2), 'native drive caps')[0], self.caps)
                or np.any(_array(a.get_drive_types(), (1,2), 'native drive types') != 1)
                or np.any(_array(a.get_dof_velocity_targets(), (1,2), 'native velocity targets') != 0)):
            raise ValueError('Split stiffness/damping native contract changed')

    def step(self, dt, *, root_constrained=False):
        if (isinstance(dt, (bool,np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt)
                or abs(dt-self.dt) > 1e-12 or root_constrained is not False):
            raise ValueError('Exact coupon timestep and unconstrained root required')
        self._check()
        q = _array(self.articulation.get_dof_positions(), (1,2), 'native joint position')[0]
        effort = -self.k*q
        if np.any(abs(q) >= .05) or not np.isfinite(effort).all() or np.any(abs(effort) > self.caps):
            raise ValueError('Split coupon angle/effort bound exceeded; no clipping')
        command = effort.astype(np.float32)
        self.articulation.set_dof_actuation_forces(command[None,:], self.indices)
        return command.astype(float)  # Exact values submitted, not measured force.

    def report(self):
        return dict(original_native_k=self.k.tolist(), retained_native_c=self.c.tolist(),
            retained_native_drive_caps=self.caps.tolist(), native_stiffness_zero_verified=True,
            native_drive_types=self.drive_types.astype(int).tolist(), native_force_drive_type=1,
            stiffness_zeroing_timing='after original native K/C and initial-state checks; runner bootstrap retains original native K',
            explicit_effort_caps_enforced_by_helper=True,
            native_drive_caps_assumed_to_limit_explicit_effort=False,
            native_damping_and_caps_unchanged=True, effort='-original_native_K * pre-step_native_q',
            root_actuated=False, contact_force_injected=False, native_coupling_verified=False,
            damping_minus_half_dt_stiffness=(self.c-self.dt*self.k/2).tolist(),
            passivity_basis='constant-M consistent implicit-C update only; not native qualification')


def _rigid_joint_state(frames, velocities):
    """Derived geometry/rate crosscheck, NEVER an articulation sensor value."""
    angles = []; rates = []; axis_errors = []
    for i in range(2):
        relative = frames[i, :3, :3].T@frames[i+1, :3, :3]
        angles.append(math.atan2(relative[0, 2]-relative[2, 0], relative[0, 0]+relative[2, 2]))
        a, b = frames[i, :3, 1], frames[i+1, :3, 1]
        rates.append(float(a@(velocities[i+1, 3:]-velocities[i, 3:])))
        axis_errors.append(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a@b)))
    return np.array(angles), np.array(rates), np.array(axis_errors)


def rigid_coordinates(frames, velocities):
    """Three pure derived arrays; no FD substitution or articulation readback.

    frames are canonical Link_0..2 body-prim world poses, velocities are native
    world [linear COM, angular]. Identity Y central rest frames are coupon-only.
    Hinge-axis errors remain visible; this projection cannot certify a joint.
    """
    return _rigid_joint_state(_poses(frames), _array(velocities, (3, 6), 'native body velocities'))


def modeled_energy(coupon, frames, velocities, *, model):
    """Modeled mechanical energy, not native drive work/energy readback.

    COM=body origin is an authored/binding requirement. Inertias are source
    body-frame tensors; R I R^T uses actual body orientation and native WORLD
    angular velocity, never qdot or pose finite differences. Section potential
    includes common-mode anchor displacement using float32-authored k/anchors.
    Contact potential, solver corrections and actuator work are not included;
    therefore free-mode energy qualification is valid only with zero contacts.
    """
    model = _model(model); frames = _poses(frames)
    velocity = _array(velocities, (3, 6), 'native body velocities')
    inertia = np.array(coupon.inertias); inertia = (inertia+inertia.transpose(0, 2, 1))/2
    rotation = frames[:, :3, :3]
    world_inertia = rotation@inertia@rotation.transpose(0, 2, 1)
    linear = .5*np.array(coupon.masses)*np.sum(velocity[:, :3]**2, axis=1)
    angular = .5*np.einsum('ni,nij,nj->n', velocity[:, 3:], world_inertia, velocity[:, 3:])
    if model in SECTION_MODELS:
        from .section_springs import coefficients, displacements
        local0 = np.eye(4); local0[2, 3] = float(np.float32(SPACING_M/2))
        local1 = np.eye(4); local1[2, 3] = -local0[2, 3]
        potential = 0.
        for i in range(2):
            k = float(np.float32(coefficients(coupon.stiffness[i], coupon.damping[i],
                RADIUS_M)['linear_stiffness_n_m']))
            z = displacements(frames[i], frames[i+1], local0, local1, float(np.float32(RADIUS_M)))
            potential += .5*k*(z@z)
        potential_basis = 'pose-derived section fibers, float32-authored anchors/gains'
    else:
        q = _rigid_joint_state(frames, velocity)[0]
        potential = .5*np.dot(np.array(coupon.stiffness)*q, q)
        potential_basis = 'pose-derived linear angular spring coordinate energy'
    result = dict(modeled_linear_kinetic_per_body_j=linear.tolist(),
        modeled_angular_kinetic_per_body_j=angular.tolist(),
        modeled_kinetic_energy_j=float(np.sum(linear+angular)),
        modeled_potential_energy_j=float(potential),
        modeled_total_mechanical_energy_j=float(np.sum(linear+angular)+potential),
        modeled_energy_basis='source masses/inertias; COM=origin; native world velocities; '+potential_basis,
        native_energy_readback=False, contact_potential_energy_included=False)
    return json.loads(json.dumps(result, allow_nan=False))


def _inspect_angular_d6(stage, path, bodies, k, c):
    """Exact coupon USD binding, NOT a native drive-coefficient getter."""
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path); joint = UsdPhysics.Joint(prim)
    expected = {'PhysicsDriveAPI:rotY'} | {'PhysicsLimitAPI:'+a for a in D6_LOCKED_AXES}
    if (not prim or prim.GetTypeName() != 'PhysicsJoint' or set(prim.GetAppliedSchemas()) != expected
            or not joint.GetJointEnabledAttr().Get() or not joint.GetExcludeFromArticulationAttr().Get()
            or not joint.GetCollisionEnabledAttr().Get()):
        raise ValueError('Exact enabled external angular D6 topology required')
    for axis in D6_LOCKED_AXES:
        limit = UsdPhysics.LimitAPI(prim, axis)
        if (limit.GetLowAttr().Get(), limit.GetHighAttr().Get()) != (1., -1.):
            raise ValueError('D6 coupon lock changed')
    for side in (0, 1):
        pos = getattr(joint, 'GetLocalPos'+str(side)+'Attr')().Get()
        quat = getattr(joint, 'GetLocalRot'+str(side)+'Attr')().Get()
        if (list(map(str, getattr(joint, 'GetBody'+str(side)+'Rel')().GetTargets())) != [bodies[side]]
                or not np.allclose(pos, [0, 0, (1-2*side)*float(np.float32(SPACING_M/2))], atol=1e-10, rtol=0)
                or not np.isfinite([quat.GetReal(), *quat.GetImaginary()]).all()
                or abs(abs(quat.GetReal())-1) > 1e-7 or np.linalg.norm(quat.GetImaginary()) > 1e-7):
            raise ValueError('D6 coupon rest frame/body binding changed')
    drive = UsdPhysics.DriveAPI(prim, 'rotY')
    if (drive.GetTypeAttr().Get() != 'force' or drive.GetTargetPositionAttr().Get() != 0
            or drive.GetTargetVelocityAttr().Get() != 0 or drive.GetMaxForceAttr().Get() != float(np.finfo(np.float32).max)
            or not np.allclose([drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get()],
                              np.array([k, c])*math.pi/180, rtol=2e-7, atol=0)):
        raise ValueError('D6 coupon angular drive changed; exact SI-to-USD conversion required')
    return dict(path=path, stiffness_usd=drive.GetStiffnessAttr().Get(), damping_usd=drive.GetDampingAttr().Get())


def bind_rigid(view, coupon, *, stage, model='section_springs_maximal'):
    """Bind an explicit maximal coupon to EXACT native RigidBodyView rows.

    Returns (report, None), with no setters, predictors or fake DOF readbacks.
    Caller uses world subspace '/', owns fresh-stage parsing/bootstrap and must
    invoke before recorded steps. Initial poses/strain must still match authoring.
    USD absence of ArticulationRootAPI is reported, not a native actor-type query.
    """
    from pxr import UsdPhysics
    from .runtime import pose_matrices
    from .section_springs import inspect_pair, local_anchors
    if model not in MAXIMAL_MODELS: raise ValueError('Explicit maximal binding model required')
    if stage is None: raise ValueError('Maximal binding requires authored stage')
    data = layout(coupon)
    if view.count != 3 or list(view.prim_paths) != data['body_paths']:
        raise ValueError('Exact ordered three native rigid body paths required')
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if set(bodies) != set(data['body_paths']):
        raise ValueError('Exact three authored dynamic bodies required')
    if (any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
            or any(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p)).GetKinematicEnabledAttr().Get() for p in bodies)):
        raise ValueError('Maximal coupon requires dynamic bodies and no articulation root')
    sections = []; angular_drives = []
    local0 = np.eye(4); local0[2, 3] = float(np.float32(SPACING_M/2))
    local1 = np.eye(4); local1[2, 3] = -local0[2, 3]
    expected_anchors = local_anchors(local0, local1, RADIUS_M)
    for i, path in enumerate(data['joint_paths']):
        if model == 'angular_d6_maximal':
            angular_drives.append(_inspect_angular_d6(stage, path, data['body_paths'][i:i+2],
                coupon.stiffness[i], coupon.damping[i]))
            continue
        section = inspect_pair(stage, path, stiffness=coupon.stiffness[i], damping=coupon.damping[i],
                               radius=RADIUS_M, topology='maximal')
        central = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
        if (section['body_paths'] != data['body_paths'][i:i+2]
                or not np.allclose(section['local_anchor_frames'], expected_anchors, atol=1e-10, rtol=0)
                or central.GetLowerLimitAttr().Get() != -float('inf')
                or central.GetUpperLimitAttr().Get() != float('inf')
                or any(s.startswith('PhysicsLimitAPI:') for s in central.GetPrim().GetAppliedSchemas())
                or [s for s in central.GetPrim().GetAppliedSchemas() if s.startswith('PhysicsDriveAPI:')]
                    != ['PhysicsDriveAPI:angular']):
            raise ValueError('Original maximal centre geometry/topology changed')
        sections.append(section)
    expected_joints = set(data['joint_paths']) | {p for s in sections for p in s['joint_paths']}
    if {str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)} != expected_joints:
        raise ValueError('Unexpected extra/missing maximal coupon joint')
    masses = _array(view.get_masses(), (3, 1), 'native rigid masses')[:, 0]
    if not np.allclose(masses, coupon.masses, rtol=2e-6, atol=1e-12):
        raise ValueError('Native rigid masses differ')
    poses = _array(view.get_transforms(), (3, 7), 'initial native rigid poses')
    com = _array(view.get_coms(), (3, 7), 'native local COM/principal axes')
    if (np.max(abs(np.linalg.norm(poses[:, 3:], axis=1)-1)) > 1e-5
            or np.max(abs(np.linalg.norm(com[:, 3:], axis=1)-1)) > 1e-5
            or np.max(abs(com[:, :3])) > 1e-9):
        raise ValueError('Unit native quaternions and original zero COM required')
    # API documents column-major 3x3 tensors at COM, in the body-prim frame.
    inertia = _array(view.get_inertias(), (3, 9), 'native rigid inertias').reshape(3, 3, 3).transpose(0, 2, 1)
    expected_inertia = np.array(coupon.inertias); expected_inertia = (expected_inertia+expected_inertia.transpose(0, 2, 1))/2
    if (not np.allclose(inertia, expected_inertia, rtol=2e-6, atol=1e-14)
            or np.any(np.linalg.eigvalsh((inertia+inertia.transpose(0, 2, 1))/2) <= 0)):
        raise ValueError('Native rigid inertias differ; no mass/inertia retuning')
    frames = _poses(pose_matrices(poses))
    velocity = _array(view.get_velocities(), (3, 6), 'initial native rigid velocities')
    translation_error = np.linalg.norm(frames[:, :3, 3]-data['frames'][:, :3, 3], axis=1)
    rotation_error = []
    for actual, expected in zip(frames, data['frames']):
        relative = expected[:3, :3].T@actual[:3, :3]
        sine = .5*np.linalg.norm([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]])
        rotation_error.append(math.atan2(float(sine), float((np.trace(relative)-1)/2)))
    q, qdot, axis_errors = _rigid_joint_state(frames, velocity)
    if (np.any(translation_error > INITIAL_FRAME_POSITION_TOLERANCE_M)
            or max(rotation_error) > INITIAL_FRAME_ANGLE_TOLERANCE_RAD
            or np.any(abs(q-INITIAL_JOINT_ANGLE_RAD) > INITIAL_Q_TOLERANCE_RAD)):
        raise ValueError('Initial native rigid frames/strain differ; actual_geometric_q_rad='+repr(q.tolist()))
    return dict(model=model, source_sha256=coupon.source_sha256, contact_model=coupon.contact_model,
        native_view_kind='rigid_body', native_body_paths=list(view.prim_paths),
        native_masses=masses.tolist(), native_inertias_body_frame=inertia.tolist(),
        native_local_com_poses=com.tolist(), initial_native_frames=frames.tolist(),
        initial_native_body_velocities=velocity.tolist(),
        initial_geometric_q_rad=q.tolist(), initial_projected_relative_qdot_rad_s=qdot.tolist(),
        initial_hinge_axis_error_rad=axis_errors.tolist(), initial_native_frames_verified=True,
        initial_position_error_m=translation_error.tolist(), initial_angle_error_rad=rotation_error,
        initial_position_tolerance_m=INITIAL_FRAME_POSITION_TOLERANCE_M,
        initial_angle_tolerance_rad=INITIAL_FRAME_ANGLE_TOLERANCE_RAD,
        initial_q_tolerance_rad=INITIAL_Q_TOLERANCE_RAD,
        native_masses_inertias_com_verified=True, native_articulation_readbacks_used=False,
        authored_articulation_roots_absent=True, native_actor_topology_independently_verified=False,
        central_angular_drives_authored_zero=model in SECTION_MODELS, native_angular_drives_zero_verified=False,
        angular_d6_usd_drives=angular_drives, angular_d6_authored_verified=model == 'angular_d6_maximal',
        native_drives_disabled=False, root_constraint=False, predictor=None,
        section_springs=sections, external_d6_native_coefficients_verified=False,
        native_qualified=False), None


def sample(coupon, *, step_id, frames, velocities, q, qdot, contact_rows,
           full_normal_friction_stream, model='native'):
    """Pure post-fetch telemetry/oracle. Native freshness is caller asserted.

    Rows: {collider0, collider1, point_world_m, impulse_on_0_ns, kind}.
    Normal rows additionally require normal_on_0, a unit world normal.
    kind is normal or friction. Preserve ORIGINAL order/sign for both kinds;
    copy friction anchors once, not once per normal point. No abs physical load.
    No synthetic force injection or native callbacks are performed here.
    """
    model = _model(model)
    step = _integer(step_id, 'positive physics step', 1)
    if full_normal_friction_stream is not True: raise ValueError('Complete same-step native contact stream required')
    frames = _poses(frames); velocity = _array(velocities, (3, 6), 'native body velocities')
    q = _array(q, (2,), 'native joint positions'); qdot = _array(qdot, (2,), 'native joint velocities')
    geometric_q, projected_qdot, axis_errors = _rigid_joint_state(frames, velocity)
    if model in MAXIMAL_MODELS:
        if (np.max(abs(q-geometric_q)) > 1e-6 or np.max(abs(qdot-projected_qdot)) > 1e-6):
            raise ValueError('Maximal q/qdot must match geometric angle/projected native angular velocity')
    rows = list(contact_rows)
    if len(rows) > MAX_CONTACT_ROWS: raise ValueError('Contact row bound exhausted')
    data = layout(coupon); colliders = {p: i for i, p in enumerate(data['collider_paths'])}
    known = set(colliders) | set(data['pads']); wrench = np.zeros((3, 6)); upper = np.zeros(3)
    pad_normals = dict.fromkeys(data['pads'], 0.); copied = []
    for row in rows:
        first, second = row['collider0'], row['collider1']
        if first == second or first not in known or second not in known or row['kind'] not in ('normal', 'friction'):
            raise ValueError('Unknown contact collider/kind; no incomplete attribution')
        p = _array(row['point_world_m'], (3,), 'contact point')
        impulse = _array(row['impulse_on_0_ns'], (3,), 'signed original-order contact impulse')
        force = impulse/coupon.dt
        for path, sign in ((first, 1), (second, -1)):
            if path in colliders:
                i = colliders[path]; f = sign*force
                wrench[i, :3] += f; wrench[i, 3:] += np.cross(p-frames[i, :3, 3], f)
                upper[i] += math.hypot(*force)
        item = dict(collider0=first, collider1=second, kind=row['kind'],
            point_world_m=p.tolist(), impulse_on_0_ns=impulse.tolist())
        if row['kind'] == 'normal':
            normal = _array(row['normal_on_0'], (3,), 'normal on collider0')
            if abs(math.hypot(*normal)-1) > 1e-5: raise ValueError('Unit contact normal required')
            scalar = float(impulse @ normal)
            if math.hypot(*(impulse-scalar*normal)) > 1e-12+1e-5*math.hypot(*impulse):
                raise ValueError('Normal impulse contains non-normal load')
            for pad in {first, second} & set(data['pads']): pad_normals[pad] += scalar
            item['normal_on_0'] = normal.tolist()
            if row.get('separation_m') is not None:
                item['separation_m'] = float(_array(row['separation_m'], (), 'native signed separation'))
        copied.append(item)
    moments = []; measured_q = []; anchor_error = []
    for i in range(2):
        parent, child = frames[i], frames[i+1]
        anchor = parent[:3, 3] + parent[:3, :3] @ [0, 0, SPACING_M/2]
        opposite = child[:3, 3] + child[:3, :3] @ [0, 0, -SPACING_M/2]
        anchor_error.append(float(np.linalg.norm(anchor-opposite)))
        torque = sum((wrench[j, 3:] + np.cross(frames[j, :3, 3]-anchor, wrench[j, :3])
                      for j in range(i+1, 3)), start=np.zeros(3))
        moments.append(float(parent[:3, 1] @ torque))
        relative = parent[:3, :3].T @ child[:3, :3]
        measured_q.append(math.atan2(relative[0, 2]-relative[2, 0], relative[0, 0]+relative[2, 2]))
    origin = frames[1, :3, 3]
    total_torque = sum((wrench[i, 3:] + np.cross(frames[i, :3, 3]-origin, wrench[i, :3])
                        for i in range(3)), start=np.zeros(3))
    result = dict(step_id=step, dt_s=coupon.dt, source_sha256=coupon.source_sha256,
        comparison_model=model, contact_model=coupon.contact_model,
        joint_position_basis=('derived_relative_native_body_pose' if model in MAXIMAL_MODELS
                              else 'native_articulation_dof_position'),
        joint_velocity_basis=('projected_relative_native_body_angular_velocity' if model in MAXIMAL_MODELS
                              else 'native_articulation_dof_velocity'),
        independent_articulation_readback=model not in MAXIMAL_MODELS,
        hinge_axis_error_rad=axis_errors.tolist(),
        frames_world_m=frames.tolist(), body_velocities_world=velocity.tolist(),
        q_rad=q.tolist(), qdot_rad_s=qdot.tolist(), measured_relative_angle_rad=measured_q,
        contact_rows=copied, active_normal_pads=sorted(p for p, j in pad_normals.items() if j > 0),
        per_body_contact_upper_bound_n=upper.tolist(), contact_wrenches_at_body_com=wrench.tolist(),
        contact_joint_moments_nm=moments, spring_kq_nm=(np.array(coupon.stiffness)*q).tolist(),
        damping_cqdot_nm=(np.array(coupon.damping)*qdot).tolist(),
        elastic_coordinate_energy_j=float(.5*np.dot(np.array(coupon.stiffness)*q, q)),
        net_contact_force_n=wrench[:, :3].sum(axis=0).tolist(), net_contact_torque_nm=total_torque.tolist(),
        anchor_error_m=anchor_error, complete_stream_caller_asserted=True,
        native_provenance_verified_by_helper=False, static_equilibrium_verified=False)
    result.update(modeled_energy(coupon, frames, velocity, model=model))
    result['modeled_initial_energy_reference_j'] = modeled_energy(coupon, data['frames'],
        np.zeros((3, 6)), model=model)['modeled_total_mechanical_energy_j']
    result['modeled_initial_energy_reference_basis'] = 'pre-start authored strain/rest anchors; zero initial native body velocity; not post-bootstrap reset'
    if model in SECTION_MODELS:
        from .section_springs import coefficients, displacements, displacement_rates, planar_response
        # Use the actual float32-authored coordinates/gains for geometry-derived
        # fiber energy. The ideal planar law below remains the original SI K/C.
        parent_local = np.eye(4); parent_local[2, 3] = float(np.float32(SPACING_M/2))
        child_local = np.eye(4); child_local[2, 3] = -parent_local[2, 3]
        authored_radius = float(np.float32(RADIUS_M))
        sections = []
        for i in range(2):
            p = coefficients(coupon.stiffness[i], coupon.damping[i], RADIUS_M)
            k_linear = float(np.float32(p['linear_stiffness_n_m']))
            c_linear = float(np.float32(p['linear_damping_n_s_m']))
            z = displacements(frames[i], frames[i+1], parent_local, child_local, authored_radius)
            zdot = displacement_rates(frames[i], frames[i+1], velocity[i], velocity[i+1],
                parent_local, child_local, authored_radius)
            centre0 = (frames[i]@parent_local)[:3, 3]
            centre1 = (frames[i+1]@child_local)[:3, 3]
            ideal = planar_response(q[i], qdot[i], coupon.stiffness[i], coupon.damping[i], RADIUS_M)
            sections.append(dict(joint_path=data['joint_paths'][i],
                actual_displacement_z_m=z.tolist(), actual_fiber_energy_j=float(.5*k_linear*(z@z)),
                native_velocity_derived_gap_rate_m_s=zdot.tolist(),
                ideal_dashpot_power_from_native_velocity_w=float(-c_linear*(zdot@zdot)),
                common_mode_z_m=float(np.mean(z)), axial_stiffness_n_m=2*k_linear,
                ideal_common_mode_axial_force_n=float(-2*k_linear*np.mean(z)),
                centre_anchor_error_m=float(np.linalg.norm(centre1-centre0)), ideal_planar=ideal,
                basis='native poses/velocities + float32-authored anchors/gains; NOT native drive force/energy readback'))
        result['section_springs'] = sections
        result['linear_reference_fields_only'] = ['spring_kq_nm', 'damping_cqdot_nm', 'elastic_coordinate_energy_j']
        result['section_actual_fiber_energy_j'] = sum(s['actual_fiber_energy_j'] for s in sections)
        result['section_ideal_elastic_resistance_nm'] = [s['ideal_planar']['elastic_torque_nm']*-1 for s in sections]
        result['section_ideal_damping_resistance_nm'] = [s['ideal_planar']['damping_torque_nm']*-1 for s in sections]
    return json.loads(json.dumps(result, allow_nan=False))


def assess_tail(samples, coupon, *, model=None, whole_run_samples=None):
    """Require a contiguous >=0.5 s tail, stiffness AND velocity/contact balance.

    Fixed diagnostic tolerances, not calibration and not native force readback:
    1 uNm +2% moment residual, 0.01 rad/s joint/body angular RMS,
    0.001 m/s body linear RMS, 0.1 mN /1 uNm global wrench balance.
    Moment equilibrium kq (section: K*sin(q)*cos(q)) is tested only in the
    small-angle (<0.05 rad) coupon. Neither law is native drive-force readback.
    """
    rows = list(samples)
    if len(rows) < 2: raise ValueError('At least two consecutive post-fetch samples required')
    model = _model(rows[0].get('comparison_model', 'native') if model is None else model)
    for i, row in enumerate(rows):
        _integer(row['step_id'], 'tail step', 1)
        if (row['source_sha256'] != coupon.source_sha256 or row['dt_s'] != coupon.dt
                or row.get('comparison_model', 'native') != model
                or row.get('contact_model', 'compliant') != coupon.contact_model
                or row.get('complete_stream_caller_asserted') is not True
                or (i and row['step_id'] != rows[i-1]['step_id']+1)):
            raise ValueError('Stale, incomplete or differently bound tail')
    def values(key, shape):
        return _array([r[key] for r in rows], (len(rows), *shape), key)
    q = values('q_rad', (2,)); v = values('qdot_rad_s', (2,))
    body_v = values('body_velocities_world', (3, 6))
    torque = values('contact_joint_moments_nm', (2,)); elastic = q*np.array(coupon.stiffness)
    if model in SECTION_MODELS:
        elastic = np.sin(q)*np.cos(q)*np.array(coupon.stiffness)
    residual = abs(torque-elastic); tolerance = 1e-6+.02*np.maximum(abs(torque), abs(elastic))
    geometry_q = values('measured_relative_angle_rad', (2,))
    force = values('net_contact_force_n', (3,)); moment = values('net_contact_torque_nm', (3,))
    gates = dict(duration=(len(rows)-1)*coupon.dt >= .5-1e-12,
        loaded_or_free_recovered=(bool(np.all(np.mean(abs(q), axis=0) > 1e-4)) if coupon.held_contacts
                                  else bool(np.max(abs(q)) < 1e-4)),
        small_angle=bool(np.max(abs(q)) < .05),
        native_pose_agrees=bool(np.max(abs(geometry_q-q)) < 1e-4),
        joint_anchors=bool(np.max(values('anchor_error_m', (2,))) < 1e-5),
        stiffness_equilibrium=bool(np.all(residual < tolerance)),
        joint_velocity=bool(np.sqrt(np.mean(v*v)) < .01),
        body_linear_velocity=bool(np.sqrt(np.mean(body_v[:, :, :3]**2)) < .001),
        body_angular_velocity=bool(np.sqrt(np.mean(body_v[:, :, 3:]**2)) < .01),
        global_force_balance=bool(np.max(np.linalg.norm(force, axis=1)) < 1e-4),
        global_torque_balance=bool(np.max(np.linalg.norm(moment, axis=1)) < 1e-6),
        all_pads_observed=all(set(r['active_normal_pads']) == set(layout(coupon)['pads']) for r in rows))
    if coupon.held_contacts and coupon.contact_model == 'rigid_control':
        # This unchanged fixture intentionally preloads compliant pads. A hard
        # 5.50 mm gap cannot admit its 5.60 mm centered shaft; never qualify it
        # through solver penetration or escape from the finite pads.
        gates['rigid_control_pinch_feasible'] = False
    if model in MAXIMAL_MODELS:
        # No independent DOF sensor exists in maximal coordinates. Do not turn
        # geometry compared with itself into an articulation agreement claim.
        gates['derived_state_consistency'] = gates.pop('native_pose_agrees')
        gates['hinge_axis_alignment'] = bool(np.max(values('hinge_axis_error_rad', (2,))) < 1e-4)
    energy = None
    if not coupon.held_contacts:
        if whole_run_samples is not None:
            whole_run = list(whole_run_samples)
            energy = assess_free_energy(whole_run, coupon, model=model)
            if (not whole_run or whole_run[-1]['step_id'] != rows[-1]['step_id']
                    or len(whole_run) < len(rows) or whole_run[-len(rows):] != rows):
                raise ValueError('Whole-run energy evidence must include this exact tail')
        gates['whole_run_free_energy'] = energy is not None and energy['passed']
    return dict(passed=all(gates.values()), gates=gates,
        whole_run_free_energy=energy,
        comparison_model=model, contact_model=coupon.contact_model,
        elastic_law='K*sin(theta)*cos(theta)' if model in SECTION_MODELS else 'K*theta',
        max_static_moment_residual_nm=float(np.max(residual)),
        joint_velocity_rms_rad_s=float(np.sqrt(np.mean(v*v))),
        source_sha256=coupon.source_sha256, native_qualified=False,
        effective_native_iterations_verified=False, whole_plant_qualified=False,
        basis='caller-supplied native samples; elementary planar equilibrium only')


def assess_free_energy(samples, coupon, *, model):
    """Entire free run, starting step 1; no tail-only or contact-loaded proof.

    Require total modeled energy <= its authored initial reference and no rise
    above the running minimum, allowing 1 nJ + 1e-4 * initial energy. This
    disclosed diagnostic allowance is not fitted to a run or claimed to be an
    exact bound on PhysX rounding. Running-min comparison prevents accumulation.
    Caller must separately expose any pre-step parsing/bootstrap activity.
    """
    model = _model(model); rows = list(samples)
    if coupon.held_contacts or not rows:
        raise ValueError('Nonempty entire contact-free run required')
    reference = modeled_energy(coupon, layout(coupon)['frames'], np.zeros((3, 6)),
        model=model)['modeled_total_mechanical_energy_j']
    energy = []; no_contacts = True
    for i, row in enumerate(rows):
        if (_integer(row['step_id'], 'energy step', 1) != i+1
                or row['source_sha256'] != coupon.source_sha256 or row['dt_s'] != coupon.dt
                or row.get('comparison_model', 'native') != model
                or row.get('contact_model', 'compliant') != coupon.contact_model
                or row.get('complete_stream_caller_asserted') is not True):
            raise ValueError('Whole free run must be source/model bound and contiguous from step 1')
        no_contacts = no_contacts and not row['contact_rows']
        # Recompute rather than trusting copied/overwritten energy fields.
        energy.append(modeled_energy(coupon, row['frames_world_m'], row['body_velocities_world'],
            model=model)['modeled_total_mechanical_energy_j'])
    energy = np.array(energy); tolerance = ENERGY_ROUNDOFF_ATOL_J+ENERGY_ROUNDOFF_RTOL*reference
    previous_min = np.minimum.accumulate(np.r_[reference, energy[:-1]])
    max_rise = float(np.max(energy-previous_min))
    gates = dict(no_contacts=bool(no_contacts), bounded_from_initial=bool(np.max(energy) <= reference+tolerance),
        nonincreasing_with_roundoff=bool(max_rise <= tolerance))
    return dict(passed=all(gates.values()), gates=gates, initial_reference_j=reference,
        max_total_modeled_energy_j=float(np.max(energy)), max_rise_above_prior_minimum_j=max_rise,
        roundoff_allowance_j=tolerance, roundoff_atol_j=ENERGY_ROUNDOFF_ATOL_J,
        roundoff_initial_energy_rtol=ENERGY_ROUNDOFF_RTOL, steps=len(rows),
        source_sha256=coupon.source_sha256, comparison_model=model, contact_model=coupon.contact_model,
        native_energy_readback=False, native_qualified=False)
