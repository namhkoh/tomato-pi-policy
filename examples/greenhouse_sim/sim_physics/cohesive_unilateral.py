"""USD-only unilateral secant adapter; uncalibrated, never a cut/controller.

107.3 primary semantics (D6 soft limits, force units, excluded articulations):
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html
Installed PhysxSchema generatedSchema.usda: PhysxLimitAPI stiffness/damping.
Joint X is the A->B material normal; J=(G0 L0)^-1 (G1 L1), so positive X
is opening. A soft upper limit at zero resists opening. The finite -2 mm lower
bound is inactive ONLY within the caller's norm(anchor jump) <= 1 mm envelope;
this is a bounded diagnostic approximation, not a globally unilateral spring.
PhysX 5.6.1 PxJointLinearLimitPair::isValid requires finite low/high/width.
At zero retention both normal limit APIs and their properties are REMOVED,
never made into a zero-stiffness hard stop or an infinite native limit pair.
Compression and
friction are separate native contact channels. Pure compression cannot damage
cohesive.py; shear under compression still can. Never add returned law forces
on top of these native implicit constraints.

Configure disabled material_band fracture D6s BEFORE starting native physics.
Commit accepted measured cohesive states after fetch; stiffness takes effect in
the NEXT solve (lagged secant, not a converged nonlinear cohesive solve). Caller
owns state measurement, acceptance, energy residuals, all guards and activation.
This helper never enables joints, steps physics, changes geometry/mass/contact,
or uses a command, time, contact switch or rounded damage to select a branch.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .cohesive import CohesiveState, ENGINEERING_STATUS


MAXIMUM_ANCHOR_JUMP_M = .001  # Caller must check actual measured vectors, including compression.
_NORMAL_LOW_M = -.002       # Diagnostic envelope, NOT a material parameter.
_NORMAL_LIMIT_SCHEMAS = {'PhysicsLimitAPI:transX', 'PhysxLimitAPI:transX'}
_NORMAL_LIMIT_PREFIXES = ('limit:transX:', 'physxLimit:transX:')


@dataclass(frozen=True)
class UnilateralSettings:
    normal_stiffness_n_m: float
    tangent_stiffness_n_m: float

    @property
    def normal_low_m(self):
        # Infinities describe a REMOVED limit; they are never authored to USD.
        return _NORMAL_LOW_M if self.normal_stiffness_n_m > 0 else -math.inf

    @property
    def normal_high_m(self):
        return 0. if self.normal_stiffness_n_m > 0 else math.inf


def unilateral_settings(state, area_m2):
    """Area-weighted retained stiffness; necessary state envelope check only.

    Effective separation omits compression and weights shear by sqrt(Kt/Kn).
    Its history must fit the measured-vector envelope, but passing this check
    cannot certify compression or the actual vector norm: caller checks both.
    """
    if (not isinstance(state, CohesiveState) or isinstance(area_m2, (bool, np.bool_))
            or not isinstance(area_m2, Real)):
        raise ValueError('CohesiveState and positive reference quadrature area required')
    try:
        area_m2 = float(area_m2)
    except OverflowError as exc:
        raise ValueError('Unrepresentable reference area') from exc
    if not math.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError('Finite positive reference area required')
    if (state.maximum_effective_separation_m / max(1., state.parameters.shear_weight)
            > MAXIMUM_ANCHOR_JUMP_M):
        raise ValueError('Cohesive history outside necessary 1 mm anchor-jump envelope')
    retained = state.retained_stiffness
    values = [float(area_m2) * retained * k for k in
              (state.parameters.normal_stiffness_pa_m, state.parameters.shear_stiffness_pa_m)]
    # A positive spring rounded/flushed to zero would become a HARD limit.
    f32 = np.finfo(np.float32)
    if any(not math.isfinite(k) or (retained > 0 and not float(f32.tiny) <= k <= float(f32.max)) for k in values):
        raise ValueError('Native float32 spring unrepresentable; never clamp or prematurely fail material')
    return UnilateralSettings(*values)


def _schema_names(prim):
    op = prim.GetMetadata('apiSchemas')
    return set(prim.GetAppliedSchemas()) | (set(op.GetAppliedItems()) if op else set())


def _signature(joint):
    """Immutable body identities and joint-local frames; no current-pose inference."""
    result = []
    for i in (0, 1):
        targets = getattr(joint, f'GetBody{i}Rel')().GetTargets()
        pos = getattr(joint, f'GetLocalPos{i}Attr')().Get()
        quat = getattr(joint, f'GetLocalRot{i}Attr')().Get()
        q = (quat.GetReal(), *quat.GetImaginary())
        if len(targets) != 1 or not np.isfinite([*pos, *q]).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-6, rtol=0):
            raise ValueError('Two finite rigid material-anchor frames required')
        result.append((str(targets[0]), tuple(pos), tuple(q)))
    if result[0][0] == result[1][0]:
        raise ValueError('Distinct material bodies required')
    return tuple(result)


def _common(joint):
    from pxr import UsdGeom, UsdPhysics
    prim = joint.GetPrim(); stage = prim.GetStage()
    if (prim.GetTypeName() != 'PhysicsJoint' or not joint.GetExcludeFromArticulationAttr().Get()
            or not joint.GetCollisionEnabledAttr().Get()
            or joint.GetBreakForceAttr().Get() != math.inf or joint.GetBreakTorqueAttr().Get() != math.inf
            or UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.):
        raise ValueError('Unbreakable excluded D6 with contact enabled in metre/kg units required')
    if any(a.GetNumTimeSamples() for a in prim.GetAttributes()):
        raise ValueError('No animated joint/drive/limit settings')
    signature = _signature(joint)
    for path, _, _ in signature:
        body = stage.GetPrimAtPath(path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Existing native rigid material bodies required')
        api = UsdPhysics.RigidBodyAPI(body)
        if not api.GetRigidBodyEnabledAttr().Get() or api.GetKinematicEnabledAttr().Get():
            raise ValueError('Both fracture material bodies must remain dynamic')
    return signature


def _drives(joint, axes, expected):
    from pxr import UsdPhysics
    prim = joint.GetPrim(); names = _schema_names(prim)
    if {s for s in names if s.startswith('PhysicsDriveAPI:')} != {'PhysicsDriveAPI:' + a for a in axes}:
        raise ValueError('Unexpected/missing drive; no parallel normal or angular spring')
    result = []
    for axis, stiffness in zip(axes, expected):
        d = UsdPhysics.DriveAPI(prim, axis)
        if (d.GetTypeAttr().Get() != 'force' or d.GetDampingAttr().Get() != 0.
                or d.GetTargetPositionAttr().Get() != 0. or d.GetTargetVelocityAttr().Get() != 0.
                or d.GetMaxForceAttr().Get() != math.inf
                or not np.isclose(d.GetStiffnessAttr().Get(), stiffness, atol=0., rtol=2e-6)):
            raise ValueError('Unexpected force drive settings or area-weighted stiffness')
        result.append(d)
    return result


def _prepare(joint, state, area_m2):
    from pxr import Gf, UsdGeom
    settings = unilateral_settings(state, area_m2)
    signature = _common(joint); prim = joint.GetPrim()
    if joint.GetJointEnabledAttr().Get() or any(s.startswith(('PhysicsLimitAPI:', 'PhysxLimitAPI:')) for s in _schema_names(prim)):
        raise ValueError('Configure only a disabled, unrestricted fracture D6 before native start')
    virgin = unilateral_settings(CohesiveState(state.parameters), area_m2)
    _drives(joint, ('transX', 'transY', 'transZ'),
            (virgin.normal_stiffness_n_m, virgin.tangent_stiffness_n_m, virgin.tangent_stiffness_n_m))
    cache = UsdGeom.XformCache(); frames = []; centres = []
    for path, pos, q in signature:
        body = np.asarray(cache.GetLocalToWorldTransform(prim.GetStage().GetPrimAtPath(path))).T
        if (not np.isfinite(body).all() or not np.allclose(body[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(body[:3, :3].T @ body[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(body[:3, :3]), 1., atol=1e-6, rtol=0)):
            raise ValueError('Rigid unscaled native body frames required')
        local = np.eye(4); local[:3, 3] = pos
        local[:3, :3] = np.asarray(Gf.Matrix3d(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))).T
        frames.append(body @ local); centres.append(body[:3, 3])
    if (not np.allclose(frames[0][:3, 3], frames[1][:3, 3], atol=2e-8, rtol=0)
            or not np.allclose(frames[0][:3, :3], frames[1][:3, :3], atol=2e-6, rtol=0)
            or np.dot(centres[1] - centres[0], frames[0][:3, 0]) <= 0):
        raise ValueError('Coincident stress-free anchors and +X normal from A toward B required')
    return joint, state, float(area_m2), signature, settings


class UnilateralLink:
    """Use configure_unilateral/configure_band_interfaces, not direct construction."""
    def __init__(self, prepared):
        from pxr import Sdf, UsdPhysics
        self.joint, self.state, self.area_m2, self.signature, settings = prepared
        prim = self.joint.GetPrim()
        if not prim.RemoveAPI(UsdPhysics.DriveAPI, 'transX'):
            raise ValueError('Could not remove normal DriveAPI')
        for name in UsdPhysics.DriveAPI.GetSchemaAttributeNames(False, 'transX'):
            prim.RemoveProperty(name)
        if any(n.startswith('drive:transX:') for n in prim.GetPropertyNames()):
            raise ValueError('Composed normal drive properties remain; isolate authoring layer')
        self.limit = UsdPhysics.LimitAPI.Apply(prim, 'transX')
        try:
            from pxr import PhysxSchema
        except ImportError:
            # Exact schema authoring for stock-USD tests; NOT native readiness.
            prim.AddAppliedSchema('PhysxLimitAPI:transX')
        else:
            PhysxSchema.PhysxLimitAPI.Apply(prim, 'transX')
        self.native_schema_registered = 'PhysxLimitAPI:transX' in prim.GetAppliedSchemas()
        self.soft = {name: prim.CreateAttribute('physxLimit:transX:' + name, Sdf.ValueTypeNames.Float, custom=False)
                     for name in ('stiffness', 'damping', 'restitution', 'bounceThreshold')}
        for name in ('damping', 'restitution', 'bounceThreshold'):
            self.soft[name].Set(0.)
        self.tangents = [UsdPhysics.DriveAPI(prim, a) for a in ('transY', 'transZ')]
        self._write(settings)

    def _write(self, settings):
        if settings.normal_stiffness_n_m == 0:
            if self.limit is not None:
                self._remove_normal_limit()
            return
        self.limit.CreateLowAttr(settings.normal_low_m)
        self.soft['stiffness'].Set(settings.normal_stiffness_n_m)
        self.limit.CreateHighAttr(0.)
        for drive in self.tangents:
            drive.GetStiffnessAttr().Set(settings.tangent_stiffness_n_m)

    def _remove_normal_limit(self):
        """Atomic Sdf-only removal; no transient hard limit or USD reads in block."""
        from pxr import Sdf
        prim = self.joint.GetPrim(); target = prim.GetStage().GetEditTarget()
        spec = target.GetLayer().GetPrimAtPath(target.MapToSpecPath(prim.GetPath()))
        properties = [p for p in prim.GetProperties()
                      if p.GetName().startswith(_NORMAL_LIMIT_PREFIXES) and p.GetPropertyStack()]
        if not spec or not spec.HasInfo('apiSchemas') or any(
                len(p.GetPropertyStack()) != 1 or p.GetPropertyStack()[0].owner != spec for p in properties):
            raise ValueError('Normal limit removal requires isolated authoring layer; no weaker properties')
        op = spec.GetInfo('apiSchemas')
        if op.isExplicit:
            op.explicitItems = [n for n in op.explicitItems if n not in _NORMAL_LIMIT_SCHEMAS]
        else:
            for field in ('prependedItems', 'appendedItems', 'addedItems', 'orderedItems'):
                setattr(op, field, [n for n in getattr(op, field) if n not in _NORMAL_LIMIT_SCHEMAS])
            op.deletedItems = sorted(set(op.deletedItems) | _NORMAL_LIMIT_SCHEMAS)
        removals = [spec.properties[p.GetName()] for p in properties]
        tangents = [spec.attributes.get(d.GetStiffnessAttr().GetName()) for d in self.tangents]
        if not all(tangents):
            raise ValueError('Tangential stiffness must be authored in the same isolated layer')
        # Sdf only: consumers get the final schema/property/stiffness change together.
        with Sdf.ChangeBlock():
            spec.SetInfo('apiSchemas', op)
            for prop in removals:
                spec.RemoveProperty(prop)
            for attr in tangents:
                attr.default = 0.
        if (_schema_names(prim) & _NORMAL_LIMIT_SCHEMAS
                or any(n.startswith(_NORMAL_LIMIT_PREFIXES) for n in prim.GetPropertyNames())):
            raise ValueError('Composed normal limit remains after removal; stop before next solve')
        self.limit = None; self.soft = {}

    def update(self, state):
        """Commit an ACCEPTED post-fetch measured law state for the next solve.

        No native force readback or solver convergence is claimed. Caller must
        enforce increment/force/penetration/energy guards before this operation,
        including norm(actual anchor jump) <= MAXIMUM_ANCHOR_JUMP_M each step.
        Law state alone cannot check compression; never infer it is in bounds.
        Compression does not unbind/rebind the limit; only zero retention does.
        """
        settings = unilateral_settings(state, self.area_m2)
        if (state.parameters != self.state.parameters or state.bonded != self.state.bonded
                or state.maximum_effective_separation_m < self.state.maximum_effective_separation_m):
            raise ValueError('Material/connectivity changes or healing are forbidden')
        if _common(self.joint) != self.signature:
            raise ValueError('Material joint identities/local frames changed')
        prim = self.joint.GetPrim(); names = _schema_names(prim)
        old = unilateral_settings(self.state, self.area_m2)
        expected = _NORMAL_LIMIT_SCHEMAS if old.normal_stiffness_n_m > 0 else set()
        if {s for s in names if s.startswith(('PhysicsLimitAPI:', 'PhysxLimitAPI:'))} != expected:
            raise ValueError('Unexpected or missing unilateral limits')
        _drives(self.joint, ('transY', 'transZ'), (old.tangent_stiffness_n_m,) * 2)
        if expected and (self.limit.GetLowAttr().Get() != float(np.float32(old.normal_low_m))
                or self.limit.GetHighAttr().Get() != 0.
                or any(self.soft[n].Get() != 0. for n in ('damping', 'restitution', 'bounceThreshold'))
                or not np.isclose(self.soft['stiffness'].Get(), old.normal_stiffness_n_m, atol=0., rtol=2e-6)):
            raise ValueError('Unilateral normal settings changed outside retained law')
        if not expected and any(n.startswith(_NORMAL_LIMIT_PREFIXES) for n in prim.GetPropertyNames()):
            raise ValueError('Removed normal limit properties reappeared')
        self._write(settings); self.state = state
        return self.report()

    def report(self):
        settings = unilateral_settings(self.state, self.area_m2)
        return dict(status=ENGINEERING_STATUS, joint=str(self.joint.GetPath()), area_m2=self.area_m2,
            normal_stiffness_n_m=settings.normal_stiffness_n_m, tangent_stiffness_n_m=settings.tangent_stiffness_n_m,
            normal_limit_unbounded=settings.normal_high_m == math.inf, normal_drive_removed=True,
            normal_limit_removed=self.limit is None,
            normal_low_m=None if self.limit is None else settings.normal_low_m,
            normal_high_m=None if self.limit is None else settings.normal_high_m,
            maximum_anchor_jump_m=MAXIMUM_ANCHOR_JUMP_M,
            actual_jump_envelope_checked=False, actual_jump_guard_owned_by_caller=True,
            bounded_unilateral_approximation=True,
            native_schema_registered=self.native_schema_registered,  # At configuration, even after removal.
            damage=self.state.damage, fully_separated=self.state.fully_separated,
            activation_owned_by_caller=True, native_response_validated=False,
            force_is_not_native_readback=True, lagged_secant=True, training_eligible=False)


def configure_unilateral(joint, *, state, area_m2):
    """Replace ONLY a disabled fracture D6's normal drive; keep it disabled."""
    return UnilateralLink(_prepare(joint, state, area_m2))


def configure_band_interfaces(authored):
    """Preflight ALL fracture anchors before modifying any; no bulk/source edits.

    Returns (face index, anchor index, UnilateralLink) tuples. Old transX drive
    handles in material_band records are now INVALID for use: update through
    these links only. Caller continues to own accepted states and activation.
    """
    from pxr import Gf
    from .material_band import validate
    validate(authored['band'])
    records = authored['interfaces']; faces = authored['band'].interfaces
    if len(records) != len(faces) or any(r['face'] is not f for r, f in zip(records, faces)):
        raise ValueError('Exact complete material_band interface inventory required')
    prepared = []
    stage = records[0]['anchors'][0]['joint'].GetPrim().GetStage()
    for record in records:
        face = record['face']
        if not face.fracture:
            continue
        if len(record['anchors']) != 3 or len(record['states']) != 3:
            raise ValueError('Three persistent quadrature anchors/states required')
        for i, (anchor, state, area) in enumerate(zip(record['anchors'], record['states'], face.weights_m2)):
            joint = anchor['joint']
            expected = (authored['paths'][face.a], authored['paths'][face.b])
            signature = _signature(joint)
            if (joint.GetPrim().GetStage() != stage or tuple(x[0] for x in signature) != expected
                    or state.parameters != authored['band'].config.cohesive_material):
                raise ValueError('Fracture body/material identity mismatch')
            for (_, position, q), cell in zip(signature, (face.a, face.b)):
                local = face.anchors[i] - authored['band'].cells[cell].centre
                rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))).T
                if (not np.allclose(position, local, atol=2e-9, rtol=0)
                        or not np.allclose(rotation, face.frame, atol=2e-6, rtol=0)):
                    raise ValueError('Joint no longer matches its exact material quadrature anchor/frame')
            prepared.append((face.index, i, _prepare(joint, state, area)))
    if len({str(p[2][0].GetPath()) for p in prepared}) != len(prepared):
        raise ValueError('Duplicate material anchor joint')
    return [(face, i, UnilateralLink(p)) for face, i, p in prepared]
