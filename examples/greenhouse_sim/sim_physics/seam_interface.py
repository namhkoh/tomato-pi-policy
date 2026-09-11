"""Author-only prescribed cohesive plane between TWO existing dynamic bodies.

Eight sectors of an inscribed 32-gon, three degree-two quadrature anchors each.
Reuse only material_band.face_anchors; no material cells, bulk springs or app.
The explicit plane has columns [normal A->B, tangent, bitangent, centre].
This is a reduced-order prescribed-interface model, not a discovered crack,
continuum deformation, sharp cutting, blade contact or grasp qualification.

All cohesive material inputs are REQUIRED and uncalibrated. In particular,
the old 10 kPa coupon strength is not a production prior: a full r=3 mm
section has only about 0.00021 N m nominal sigma*I/r bending capacity and
may fail under the existing branch/leaf gravity before a blade arrives.
Attached gravity and grasp must be qualified with the SAME material law;
never introduce knife-triggered weakening or tune strength to a force guard.

Pre-start authoring ONLY: creation notices can precede the disabled opinion;
this helper is not an atomic live-physics insertion. Creates only a NEW scope
and 24 DISABLED unilateral D6 joints. Caller owns
flat-ended contact geometry, mass/source provenance, removal of any old weld
AFTER separate review, load-path/contact audits, native activation and guards.
Never add cohesive_response forces in parallel with native drives. Persist
each returned link's accepted measured state; no timer/displacement command
may constitute fracture evidence. The helper's finite lower limit is valid
only under its caller-enforced <=1 mm measured anchor-jump norm envelope.
"""
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np

from .cohesive import CohesiveParameters, CohesiveState
from .cohesive_unilateral import configure_unilateral, unilateral_settings, MAXIMUM_ANCHOR_JUMP_M
from .material_band import face_anchors


STATUS = "uncalibrated_prescribed_seam_authoring_only"


@dataclass(frozen=True)
class SeamConfig:
    """Required engineering inputs; source identity is not tissue calibration."""
    source_manifest_path: str
    source_target: str
    radius_m: float
    material: CohesiveParameters

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.source_manifest_path, self.source_target)):
            raise ValueError("Explicit source identity required; authoring does not read/verify source assets")
        if (not isinstance(self.material, CohesiveParameters)
                or isinstance(self.radius_m, (bool, np.bool_)) or not isinstance(self.radius_m, Real)
                or not np.isfinite(self.radius_m) or self.radius_m <= 0):
            raise ValueError("Explicit cohesive material and finite positive radius required")
        object.__setattr__(self, "radius_m", float(self.radius_m))
        if not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("Radius must remain representable as a Python float")


def partition(config):
    """Local plane X=0, normal +X; no USD or body creation."""
    if not isinstance(config, SeamConfig):
        raise ValueError("SeamConfig required")
    theta = np.arange(32)*2*np.pi/32
    ring = np.column_stack((np.zeros(32), config.radius_m*np.cos(theta), config.radius_m*np.sin(theta)))
    facets = []
    for sector in range(8):
        vertices = np.vstack((np.zeros(3), ring[(4*sector+np.arange(5)) % 32]))
        anchors, weights = face_anchors(vertices)
        facets.append(dict(sector=sector, vertices_m=vertices, anchors_m=anchors, weights_m2=weights))
    anchors = np.vstack([f["anchors_m"] for f in facets])
    weights = np.concatenate([f["weights_m2"] for f in facets])
    area = 16*config.radius_m**2*np.sin(np.pi/16)
    second = area*config.radius_m**2*(2+np.cos(np.pi/16))/12
    if (not np.isfinite(np.r_[anchors.ravel(), weights, area, second]).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), area, rtol=1e-12, atol=0)
            or not np.allclose(weights@anchors, 0., rtol=0, atol=area*config.radius_m*1e-12)
            or not np.allclose(np.einsum("n,ni,nj->ij", weights, anchors, anchors),
                               np.diag([0., second, second]), rtol=1e-11, atol=second*1e-12)):
        raise ValueError("Unrepresentable/inconsistent area or second-moment quadrature")
    return dict(facets=facets, anchors_m=anchors, weights_m2=weights, area_m2=float(area),
                second_area_moment_m4=np.diag([0., second, second]),
                circular_area_fraction=float(np.sin(np.pi/16)/(np.pi/16)))


def _rigid(frame):
    f = np.asarray(frame)
    if f.shape != (4, 4) or f.dtype.kind not in "iuf":
        raise ValueError("Finite proper rigid 4x4 frame required")
    f = np.array(f, dtype=float, copy=True)
    if (not np.isfinite(f).all() or not np.allclose(f[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(f[:3, :3].T@f[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(f[:3, :3]), 1., atol=1e-8, rtol=0)):
        raise ValueError("Unscaled, unreflected, orthonormal material/plane frame required")
    return f


def author(stage, *, root, body_a, body_b, plane_world, config, states=None):
    """Explicit attachment only; returns 24 persistent helper links, all disabled.

    Body origins must bracket the plane along its +X normal. No material/body
    property or existing joint is modified. A partial authoring failure leaves
    ONLY new disabled joints and authoringComplete=false, never an active seam.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    geometry = partition(config)
    plane = _rigid(plane_world)
    paths = [Sdf.Path(p) for p in (body_a, body_b)]
    root_path = Sdf.Path(root)
    if (not root_path.IsAbsolutePath() or not root_path.IsPrimPath() or root_path == Sdf.Path.absoluteRootPath
            or stage.GetPrimAtPath(root_path) or not stage.GetPrimAtPath(root_path.GetParentPath())
            or any(not p.IsAbsolutePath() or not p.IsPrimPath() for p in paths)
            or paths[0] == paths[1] or any(root_path.HasPrefix(p) or p.HasPrefix(root_path) for p in paths)
            or UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.):
        raise ValueError("New separate scope, distinct existing body paths and metre/kg stage required")
    cache = UsdGeom.XformCache(); body_frames = []
    for path in paths:
        body = stage.GetPrimAtPath(path)
        if not body or not body.IsActive() or not body.IsLoaded() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Two existing native rigid material bodies required")
        api = UsdPhysics.RigidBodyAPI(body)
        if not api.GetRigidBodyEnabledAttr().Get() or api.GetKinematicEnabledAttr().Get():
            raise ValueError("Both material bodies must be enabled and dynamic")
        current = body
        while current and not current.IsPseudoRoot():
            if any(a.GetNumTimeSamples() for a in UsdGeom.Xformable(current).GetOrderedXformOps()):
                raise ValueError("No animated material frames")
            current = current.GetParent()
        body_frames.append(_rigid(np.asarray(cache.GetLocalToWorldTransform(body)).T))
    normal, centre = plane[:3, 0], plane[:3, 3]
    if not (normal@(body_frames[0][:3, 3]-centre) < 0 < normal@(body_frames[1][:3, 3]-centre)):
        raise ValueError("Body origins must bracket the prescribed plane; normal must point A to B")
    states = tuple(CohesiveState(config.material) for _ in range(24)) if states is None else tuple(states)
    if len(states) != 24 or any(not isinstance(s, CohesiveState) or s.parameters != config.material for s in states):
        raise ValueError("Exactly 24 persistent states with the explicit material required")
    if any(s.effective_separation_m != 0 for s in states):
        raise ValueError("Coincident authoring requires unloaded states; preserve damage history but reconcile initial energy")
    for state, area in zip(states, geometry["weights_m2"]):
        unilateral_settings(state, area)
        unilateral_settings(CohesiveState(config.material), area)

    def quat(rotation):
        m = np.eye(4); m[:3, :3] = rotation
        return Gf.Quatf(Gf.Matrix4d(m.T.tolist()).ExtractRotationQuat())

    # Transform offsets relative to body centres before adding world position:
    # avoid subtracting two large world anchor coordinates in local authoring.
    offsets = geometry["anchors_m"]@plane[:3, :3].T
    local = np.array([(offsets+(centre-b[:3, 3]))@b[:3, :3] for b in body_frames]).transpose(1, 0, 2)
    local = local.astype(np.float32).astype(float)
    rotations = [quat(b[:3, :3].T@plane[:3, :3]) for b in body_frames]
    reference_gap = ((body_frames[1][:3, 3]-body_frames[0][:3, 3])
                     +local[:, 1]@body_frames[1][:3, :3].T-local[:, 0]@body_frames[0][:3, :3].T)
    if not np.isfinite(local).all() or np.max(np.linalg.norm(reference_gap, axis=1)) > 2e-8:
        raise ValueError("USD float32 anchor representation exceeds helper coincidence tolerance")
    # Coincidence alone cannot detect every anchor collapsing onto one point.
    # Check the DISTRIBUTION after float32 authoring in each material frame.
    weights = geometry["weights_m2"]
    radius = config.radius_m
    maximum_anchor_error = 0.
    for side, body in enumerate(body_frames):
        recovered = (local[:, side]@body[:3, :3].T+(body[:3, 3]-centre))@plane[:3, :3]
        error = float(np.max(np.linalg.norm(recovered-geometry["anchors_m"], axis=1)))
        second = np.einsum("n,ni,nj->ij", weights, recovered, recovered)
        if (not np.isfinite(recovered).all() or error > radius*1e-5
                or np.linalg.norm(weights@recovered) > geometry["area_m2"]*radius*1e-5
                or not np.allclose(second, geometry["second_area_moment_m4"], rtol=1e-4,
                                   atol=geometry["area_m2"]*radius**2*1e-5)):
            raise ValueError("Float32 authoring does not preserve area-weighted seam anchor moments")
        maximum_anchor_error = max(maximum_anchor_error, error)
    existing = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            a, b = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if len(a) == len(b) == 1 and set(a+b) == set(paths):
                existing.append(dict(path=str(prim.GetPath()), type=prim.GetTypeName(), enabled=j.GetJointEnabledAttr().Get()))
    scope = UsdGeom.Scope.Define(stage, root_path).GetPrim()
    scope.CreateAttribute("seam:status", Sdf.ValueTypeNames.String).Set(STATUS)
    complete = scope.CreateAttribute("seam:authoringComplete", Sdf.ValueTypeNames.Bool); complete.Set(False)
    scope.CreateAttribute("seam:sourceManifest", Sdf.ValueTypeNames.String).Set(config.source_manifest_path)
    scope.CreateAttribute("seam:sourceTarget", Sdf.ValueTypeNames.String).Set(config.source_target)
    links = []
    for i, (position, state, area) in enumerate(zip(local, states, geometry["weights_m2"])):
        j = UsdPhysics.Joint.Define(stage, root_path.AppendChild(f"Sector_{i//3:02d}_Anchor_{i%3}"))
        j.CreateJointEnabledAttr(False); j.CreateCollisionEnabledAttr(True)
        j.CreateExcludeFromArticulationAttr(True)
        for side in (0, 1):
            getattr(j, f"CreateBody{side}Rel")().SetTargets([paths[side]])
            getattr(j, f"CreateLocalPos{side}Attr")(Gf.Vec3f(*position[side]))
            getattr(j, f"CreateLocalRot{side}Attr")(rotations[side])
        for axis, stiffness in zip(("transX", "transY", "transZ"),
                area*np.array([config.material.normal_stiffness_pa_m]+[config.material.shear_stiffness_pa_m]*2)):
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
            d.CreateTypeAttr("force"); d.CreateStiffnessAttr(float(stiffness)); d.CreateDampingAttr(0.)
            d.CreateTargetPositionAttr(0.); d.CreateTargetVelocityAttr(0.); d.CreateMaxForceAttr(float("inf"))
        links.append(configure_unilateral(j, state=state, area_m2=float(area)))
    complete.Set(True)
    return dict(config=config, geometry=geometry, links=links, local_anchors_m=local,
        plane_world=plane, body_frames=body_frames, body_paths=[str(p) for p in paths],
        reference_anchor_gap_m=reference_gap, summary=dict(status=STATUS, config=asdict(config),
            sectors=8, quadrature_anchors=24, outer_polygon_sides=32, reference_area_m2=geometry["area_m2"],
            circular_area_fraction=geometry["circular_area_fraction"],
            maximum_authored_anchor_gap_m=float(np.max(np.linalg.norm(reference_gap, axis=1))),
            maximum_rounded_anchor_error_m=maximum_anchor_error,
            rounded_anchor_moments_checked=True, requires_stopped_simulation=True,
            atomic_live_insertion_supported=False, initial_states_unloaded=True,
            existing_direct_joints=existing, existing_joints_removed=False, external_load_path_audit_required=True,
            flat_contact_geometry_authored=False, flat_contact_geometry_verified=False,
            body_geometry_mass_contacts_modified=False, source_assets_loaded_or_modified=False,
            source_identity_verified=False, prescribed_plane_not_discovered_crack=True,
            maximum_measured_anchor_jump_m=MAXIMUM_ANCHOR_JUMP_M, caller_must_enforce_jump_guard=True,
            all_new_joints_disabled=True, native_schema_registered=all(h.native_schema_registered for h in links),
            activation_owned_by_caller=True, native_response_validated=False, continuum_qualified=False,
            material_calibrated=False, attached_gravity_grasp_qualified=False,
            physical_cut_verified=False, training_eligible=False))
