"""Engineering-only 32-cell rest/axial-elastic/preseparated contact coupon.

No knife, plant wiring, source editing, inferred cut, or application on import.
Native entry is explicit and main-owned. E/rho/cohesive priors are not fitted.
The reference is an independently assembled small-displacement anchor network,
NOT continuum EA/L. An initially unbonded control is NOT simulated fracture.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from . import material_band as band_api
from .cohesive import CohesiveParameters, CohesiveState, cohesive_response
from .mechanics import Material
from .cohesive_unilateral import configure_band_interfaces
from .cohesive_closing_probe import coefficient_snapshot
from .runtime import pose_matrices


ROOT = "/World/MaterialBandCoupon"
FORCE_N, CARRIAGE_K, CARRIAGE_C, FORCE_CAP_N = .05, 300., 8., .08
MOMENTUM_N, CONTACT_N, ANCHOR_FORCE_N = .00051, .1, .1
REST_DRIFT_M, BULK_JUMP_M, JUMP_M = 1e-7, 1e-5, .001
PENETRATION_M, SPEED_M_S, ANGULAR_SPEED, ROTATION_RAD = 1e-5, .01, 1., .01
NATIVE_LINEAR_CAP_M_S, NATIVE_DEPENETRATION_CAP_M_S = 1., 1.
MAX_CONTACT_ROWS = 4096
REFERENCE_TOLERANCE = .05


@dataclass(frozen=True)
class Config:
    band: band_api.BandConfig
    case: str
    physics_hz: int = 960

    def __post_init__(self):
        b = self.band
        if not isinstance(b, band_api.BandConfig) or self.case not in ("rest", "elastic", "preseparated"):
            raise ValueError("Explicit BandConfig and bounded case required")
        if (type(self.physics_hz) is not int or self.physics_hz not in (960, 1920)
                or b.axial_layers != 4 or b.radius_m != .003 or b.length_m != .008
                or b.density_kg_m3 != 950. or b.bulk_material != Material()
                or b.cohesive_material != CohesiveParameters(1e8, 2e8, 1e4, 2.)):
            raise ValueError("First coupon fixes 32 cells, r3/L8 mm and explicit unchanged engineering priors")

    @property
    def duration(self):
        return {"rest": 1., "elastic": 9., "preseparated": 6.5}[self.case]

    @property
    def initial_gap(self):
        return .0001 if self.case == "preseparated" else 0.


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def network(config):
    """Pure 193-DOF geometry/reference description; one guided carriage DOF."""
    band = band_api.build(config.band)
    centres = np.vstack(([c.centre for c in band.cells], [0., 0., .005]))
    masses = np.array([c.mass_kg for c in band.cells]+[sum(c.mass_kg for c in band.cells[:8])])
    inertias = [c.inertia_kg_m2 for c in band.cells]
    cap_inertia = np.zeros((3, 3))
    for c in band.cells[:8]:
        offset = np.r_[c.centre[:2], 0.]
        cap_inertia += c.inertia_kg_m2+c.mass_kg*(np.dot(offset, offset)*np.eye(3)-np.outer(offset, offset))
    inertias = np.array(inertias+[cap_inertia])
    records = []

    def face(kind, a, b, frame, points, ks, face_index=None):
        for q, (p, k) in enumerate(zip(points, ks)):
            records.append(dict(kind=kind, a=a, b=b, p=np.array(p), frame=frame,
                                k=np.array(k), face_index=face_index, quadrature=q))

    for f in band.interfaces:
        face("cohesive" if f.fracture else "bulk", f.a, f.b, f.frame, f.anchors, f.stiffness_n_m, f.index)
    E = config.band.bulk_material.youngs_modulus_pa
    G = E/(2*(1+config.band.bulk_material.poisson_ratio))
    for cell, ids in band.boundary_faces:
        points = band.vertices[list(ids)]
        if np.allclose(points[:, 2], -.004, atol=1e-13, rtol=0):
            points = points[::-1]
            _, _, frame, _, _ = band_api.face_geometry(points)
            anchors, weights = band_api.face_anchors(points)
            face("support", -1, cell, frame, anchors, weights[:, None]*[E/.001, G/.001, G/.001])
        elif np.allclose(points[:, 2], .004, atol=1e-13, rtol=0):
            _, _, frame, _, _ = band_api.face_geometry(points)
            anchors, weights = band_api.face_anchors(points)
            face("platen", cell, 32, frame, anchors, weights[:, None]*[E/.002, G/.002, G/.002])
    if len(records) != 216:
        raise ValueError("Expected 144 bulk, 24 cohesive, 24 support, 24 platen anchors")
    jacobians = []
    for r in records:
        J = np.zeros((3, 193))
        for index, sign in ((r["a"], -1.), (r["b"], 1.)):
            if index < 0:
                continue
            if index == 32:
                J[:, -1] += sign*r["frame"].T@np.array([0., 0., 1.])
            else:
                # Rotation DOFs scaled to metres for conditioning; theta=qrot/r.
                J[:, 6*index:6*index+6] += sign*r["frame"].T@np.column_stack(
                    (np.eye(3), -skew(r["p"]-centres[index])/config.band.radius_m))
        jacobians.append(J)
    J = np.asarray(jacobians)
    k = np.array([r["k"] for r in records])
    K = np.einsum("nai,na,naj->ij", J, k, J)
    load = np.zeros(193); load[-1] = FORCE_N
    try:
        np.linalg.cholesky(K)
        q = np.linalg.solve(K, load)
    except np.linalg.LinAlgError as error:
        raise ValueError("Unsupported/singular independent anchor reference") from error
    if np.linalg.norm(K@q-load) > 1e-8 or q[-1] <= 0:
        raise ValueError("Independent reference equilibrium failed")
    predicted = np.einsum("nai,i->na", J, q)
    return dict(band=band, centres=centres, masses=masses, inertias=inertias,
                records=records, jacobians=J, stiffness=k, reference_jumps=predicted,
                reference_q=q, reference_energy=.5*float(q@load),
                command_open=q[-1]+FORCE_N/CARRIAGE_K)


def protocol(t, config, net):
    if isinstance(t, bool) or not np.isfinite(t) or not 0 <= t <= config.duration:
        raise ValueError("Finite time inside fixed protocol required")
    if config.case == "rest":
        return "rest", 0.
    if config.case == "preseparated":
        closed = -config.initial_gap-.025/CARRIAGE_K
        knots = [(0., 0.), (.5, 0.), (2.5, closed), (3.5, closed), (5.5, 0.), (6.5, 0.)]
        names = ["rest", "closing", "compression_hold", "reopening", "separated_hold"]
    else:
        opened = net["command_open"]
        knots = [(0., 0.), (.5, 0.), (2.5, opened), (3.5, opened),
                 (5., 0.), (6., -.025/CARRIAGE_K), (7., -.025/CARRIAGE_K),
                 (8., opened), (9., opened)]
        names = ["rest", "opening", "elastic_hold", "unloading", "closing",
                 "compression_hold", "reopening", "elastic_rehold"]
    i = min(np.searchsorted([p[0] for p in knots], t, side="right")-1, len(names)-1)
    lo, hi = knots[i:i+2]
    return names[i], float(lo[1]+(hi[1]-lo[1])*(t-lo[0])/(hi[0]-lo[0]))


def author(stage, config):
    """In-memory USD only. All 24 cohesive constraints remain disabled."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    net = network(config)
    fixture = band_api.author(stage, net["band"], root=ROOT, proximal_support=True)
    if config.case == "preseparated":
        for r in fixture["interfaces"]:
            if r["face"].fracture:
                r["states"] = tuple(CohesiveState(config.band.cohesive_material, bonded=False) for _ in range(3))
    handles = configure_band_interfaces(fixture)
    links = [h for _, _, h in handles]
    lookup = {(i, q): h for i, q, h in handles}
    paths = fixture["paths"]+[ROOT+"/LaboratoryCarriage"]
    cap = UsdGeom.Xform.Define(stage, paths[-1])
    cap.AddTranslateOp().Set(Gf.Vec3d(*net["centres"][-1]))
    UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim()).CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(cap.GetPrim())
    mass.CreateMassAttr(float(net["masses"][-1]))
    mass.CreateDensityAttr(950.)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*np.diag(net["inertias"][-1])))
    cap.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    for name, kind, value in (("sleepThreshold", Sdf.ValueTypeNames.Float, 0.),
            ("linearDamping", Sdf.ValueTypeNames.Float, 0.), ("angularDamping", Sdf.ValueTypeNames.Float, 0.),
            ("solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
            ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 8)):
        cap.GetPrim().CreateAttribute("physxRigidBody:"+name, kind, custom=False).Set(value)

    def quat(rotation):
        m = np.eye(4); m[:3, :3] = rotation
        return Gf.Quatf(Gf.Matrix4d(m.T.tolist()).ExtractRotationQuat())

    joints = []
    for r in net["records"]:
        if r["kind"] in ("bulk", "cohesive"):
            j = fixture["interfaces"][r["face_index"]]["anchors"][r["quadrature"]]["joint"]
        elif r["kind"] == "support":
            j = UsdPhysics.Joint.Get(stage, ROOT+f'/RemoteProximal/Cell_{r["b"]:03d}/Anchor_{r["quadrature"]}')
        else:
            j = UsdPhysics.Joint.Define(stage, ROOT+f'/RemoteDistal/Cell_{r["a"]:03d}/Anchor_{r["quadrature"]}')
            j.CreateJointEnabledAttr(True); j.CreateCollisionEnabledAttr(True)
            j.CreateExcludeFromArticulationAttr(True)
            for side, index in enumerate((r["a"], r["b"])):
                getattr(j, f"CreateBody{side}Rel")().SetTargets([paths[index]])
                getattr(j, f"CreateLocalPos{side}Attr")(Gf.Vec3f(*(r["p"]-net["centres"][index])))
                getattr(j, f"CreateLocalRot{side}Attr")(quat(r["frame"]))
            for axis, stiffness in zip(("transX", "transY", "transZ"), r["k"]):
                d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
                d.CreateTypeAttr("force"); d.CreateStiffnessAttr(float(stiffness))
                d.CreateDampingAttr(0.); d.CreateMaxForceAttr(float("inf"))
                d.CreateTargetPositionAttr(0.); d.CreateTargetVelocityAttr(0.)
        if not j:
            raise ValueError("Missing expected material/boundary anchor")
        joints.append(j)
    guide = UsdPhysics.PrismaticJoint.Define(stage, ROOT+"/CarriageGuide")
    guide.CreateBody1Rel().SetTargets([paths[-1]])
    guide.CreateAxisAttr("Z"); guide.CreateExcludeFromArticulationAttr(True)
    position = net["centres"][-1]+[0., 0., config.initial_gap]
    guide.CreateLocalPos0Attr(Gf.Vec3f(*position)); guide.CreateLocalPos1Attr(Gf.Vec3f(0.))
    loading = UsdPhysics.DriveAPI.Apply(guide.GetPrim(), "linear")
    loading.CreateTypeAttr("force"); loading.CreateStiffnessAttr(CARRIAGE_K)
    loading.CreateDampingAttr(CARRIAGE_C); loading.CreateMaxForceAttr(FORCE_CAP_N)
    loading.CreateTargetPositionAttr(0.); loading.CreateTargetVelocityAttr(0.)
    initial = net["centres"].copy()
    if config.initial_gap:
        initial[16:, 2] += config.initial_gap
        for index in range(16, 33):
            UsdGeom.Xformable(stage.GetPrimAtPath(paths[index])).GetOrderedXformOps()[0].Set(Gf.Vec3d(*initial[index]))
    local, rotations = [], []
    for j in joints:
        local.append([list(j.GetLocalPos0Attr().Get()), list(j.GetLocalPos1Attr().Get())])
        q = j.GetLocalRot0Attr().Get()
        rotations.append(np.asarray(Gf.Matrix3d(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))).T)
    applied = net["stiffness"].copy()
    for i, (r, j) in enumerate(zip(net["records"], joints)):
        if r["kind"] != "cohesive":
            applied[i] = [UsdPhysics.DriveAPI(j.GetPrim(), axis).GetStiffnessAttr().Get()
                          for axis in ("transX", "transY", "transZ")]
            if not np.allclose(applied[i], r["k"], atol=0., rtol=2e-6):
                raise ValueError("Authored native bulk/boundary coefficients differ from explicit law")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        for name, value in (("maxLinearVelocity", NATIVE_LINEAR_CAP_M_S),
                            ("maxDepenetrationVelocity", NATIVE_DEPENETRATION_CAP_M_S)):
            prim.CreateAttribute("physxRigidBody:"+name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    body_settings = {p: {name: stage.GetPrimAtPath(p).GetAttribute("physxRigidBody:"+name).Get()
        for name in ("maxLinearVelocity", "maxDepenetrationVelocity", "linearDamping", "angularDamping",
                     "sleepThreshold", "solverPositionIterationCount", "solverVelocityIterationCount")}
        for p in paths}
    return dict(config=config, net=net, authored=fixture, paths=paths, links=links,
                link_lookup=lookup, joints=joints, local=np.asarray(local),
                rotations=np.asarray(rotations), initial_centres=initial, loading=loading,
                guide=guide, applied_stiffness=applied, body_settings=body_settings)


def measured(fixture, frames):
    """Native material-anchor jumps; never command travel as strain evidence."""
    records, local = fixture["net"]["records"], fixture["local"]
    anchors = np.zeros((len(records), 2, 3))
    axes = np.empty((len(records), 3, 3))
    for i, r in enumerate(records):
        for side, body in enumerate((r["a"], r["b"])):
            anchors[i, side] = local[i, side] if body < 0 else frames[body, :3, :3]@local[i, side]+frames[body, :3, 3]
        axes[i] = fixture["rotations"][i] if r["a"] < 0 else frames[r["a"], :3, :3]@fixture["rotations"][i]
    jumps = np.einsum("nij,ni->nj", axes, anchors[:, 1]-anchors[:, 0])
    return jumps, anchors, axes


def source_receipt(config):
    from .cohesive_native_probe import _hash
    manifest = Path(config.band.source_manifest_path).resolve(strict=True)
    with manifest.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("units") != "meters":
        raise ValueError("Metre source manifest required")
    identity = config.band.source_target.split("/")
    if len(identity) != 2 or identity[0] != manifest.parent.name:
        raise ValueError("Exact source family/component identity required")
    components = [c for c in data["components"] if c["id"] == identity[1]]
    if len(components) != 1 or components[0]["type"] != "sub_stem":
        raise ValueError("Unique source sub-stem required")
    asset = (manifest.parent/components[0]["file"]).resolve(strict=True)
    if asset.parent != manifest.parent:
        raise ValueError("Source asset must remain inside its manifest directory")
    return {str(p): _hash(p) for p in (manifest, asset)}


def contact_wrenches(fixture, rows, friction, forces, frames, dt):
    """Fixed header order: +J/dt on collider0, -J/dt on collider1; never fit.

    NativeContacts preserves the full callback's [collider0, collider1].
    PhysX contact normals point shape1 -> shape0. The tensor net is NORMAL
    only: api.py get_contact_data normal*scalar reduction/get_friction_data,
    also NVIDIA 6.0.1 python_scripting/environment_setup.html contact example.
    Friction anchors contribute separately to total force, torque and work.
    """
    if len(rows)+len(friction) > MAX_CONTACT_ROWS:
        raise RuntimeError("Full-contact record budget exceeded; never truncate to pass")
    if not np.isfinite(dt) or dt <= 0 or np.shape(forces) != (32, 3) or not np.isfinite(forces).all():
        raise RuntimeError("Invalid normal tensor/time evidence")
    colliders = {p+"/Solid": i for i, p in enumerate(fixture["paths"][:32])}
    normal, tangent, torque = np.zeros((32, 3)), np.zeros((32, 3)), np.zeros((32, 3))
    total, minimum, nonface = 0., 0., 0
    face_pairs = {(r.a, r.b) for r in fixture["net"]["band"].interfaces}
    for record_index, record in enumerate(rows+friction):
        pair = record["colliders"]
        if len(pair) != 2 or pair[0] == pair[1] or any(p not in colliders for p in pair):
            raise RuntimeError("Unexpected external/native contact pair in isolated band")
        indices = [colliders[p] for p in pair]
        key = tuple(sorted(indices))  # Membership ONLY; never reorder impulse/header IDs.
        if key in fixture["net"]["band"].bulk_pairs:
            raise RuntimeError("Native contact on permanently filtered bulk pair")
        nonface += key not in face_pairs
        impulse = np.asarray(record["impulse"], float)/dt
        point = np.asarray(record["position"], float)
        if impulse.shape != (3,) or point.shape != (3,) or not np.isfinite(np.r_[impulse, point]).all():
            raise RuntimeError("Nonfinite full contact evidence")
        total += np.linalg.norm(impulse)
        accumulator = normal if record_index < len(rows) else tangent
        if record_index < len(rows):
            separation = float(record["separation_m"])
            if not np.isfinite(separation):
                raise RuntimeError("Nonfinite native separation")
            minimum = min(minimum, separation)
        for body, sign in zip(indices, (1., -1.)):
            accumulator[body] += sign*impulse
            torque[body] += np.cross(point-frames[body, :3, 3], sign*impulse)
    error = float(np.max(np.linalg.norm(normal-forces, axis=1)))
    if error > 1e-5:
        raise RuntimeError("Fixed collider0-positive NORMAL callback disagrees with native NORMAL tensor")
    return dict(torque=torque, normal=normal, friction=tangent, force=normal+tangent,
                total=float(total), minimum=float(minimum), nonface_records=int(nonface),
                callback_tensor_max_error_n=error)


def cap_status(fixture, rows, frames, velocities, dt):
    """Conservative endpoint cap-risk screen, NOT hidden solver clamp telemetry."""
    if not np.isfinite(velocities).all() or not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Nonfinite velocity/cap evidence")
    peak_speed = float(np.max(np.linalg.norm(velocities[:, :3], axis=1)))
    demand = 0.
    colliders = {p+"/Solid": i for i, p in enumerate(fixture["paths"][:32])}
    for row in rows:
        normal = np.asarray(row["normal"], float)
        point = np.asarray(row["position"], float)
        separation = float(row["separation_m"])
        if (normal.shape != (3,) or point.shape != (3,)
                or not np.isfinite(np.r_[normal, point, separation]).all()
                or abs(np.linalg.norm(normal)-1.) > 1e-4):
            raise RuntimeError("Invalid native normal/separation for cap-risk accounting")
        a, b = (colliders[p] for p in row["colliders"])
        point_v = [velocities[i, :3]+np.cross(velocities[i, 3:], point-frames[i, :3, 3]) for i in (a, b)]
        # Full penetration correction in one dt plus either-direction relative
        # normal motion: deliberately conservative; no ERP/sign fit.
        demand = max(demand, max(-separation, 0.)/dt+abs(float(normal@(point_v[0]-point_v[1]))))
    return dict(maximum_body_speed_m_s=peak_speed,
                native_linear_cap_may_bind=peak_speed >= .99*NATIVE_LINEAR_CAP_M_S,
                depenetration_demand_screen_m_s=demand,
                native_depenetration_cap_may_bind=demand >= .99*NATIVE_DEPENETRATION_CAP_M_S,
                native_cap_activation_observed=None)


def check_caps_and_speed(row):
    if row["native_linear_cap_may_bind"] or row["native_depenetration_cap_may_bind"]:
        raise RuntimeError("Native velocity/depenetration cap may bind; dynamics not accepted")
    if row["maximum_body_speed_m_s"] > SPEED_M_S:
        raise RuntimeError("Native speed watchdog exceeded below last-resort cap")


def anchor_response(fixture, frames, stiffness, states):
    net = fixture["net"]
    jumps, anchors, axes = measured(fixture, frames)
    kinds = np.array([r["kind"] for r in net["records"]])
    cohesive = np.flatnonzero(kinds == "cohesive")
    if (not np.isfinite(jumps).all() or np.max(np.linalg.norm(jumps, axis=1)) > JUMP_M
            or np.max(np.linalg.norm(jumps[kinds != "cohesive"], axis=1)) > BULK_JUMP_M):
        raise RuntimeError("Measured material/bulk anchor-jump guard")
    responses = [cohesive_response(state, jumps[i], [1., 0., 0.], area_m2=link.area_m2)
                 for state, i, link in zip(states, cohesive, fixture["links"])]
    if any(r.state.damage != s.damage or r.dissipated_energy_j != 0 for r, s in zip(responses, states)):
        raise RuntimeError("Fundamentals coupon must not produce damage/fracture work")
    elastic = jumps.copy()
    elastic[cohesive, 0] = np.maximum(elastic[cohesive, 0], 0.)
    local_force = stiffness*elastic
    world_force = np.einsum("nij,nj->ni", axes, local_force)
    if np.max(np.linalg.norm(world_force, axis=1)) > ANCHOR_FORCE_N:
        raise RuntimeError("Individual anchor force guard")
    forces, torque = np.zeros((33, 3)), np.zeros((33, 3))
    for i, r in enumerate(net["records"]):
        for side, body, sign in ((0, r["a"], 1.), (1, r["b"], -1.)):
            if body >= 0:
                forces[body] += sign*world_force[i]
                torque[body] += np.cross(anchors[i, side]-frames[body, :3, 3], sign*world_force[i])
    U = .5*np.sum(stiffness*elastic*elastic, axis=1)
    return dict(jumps=jumps, anchors=anchors, force=forces, torque=torque,
                energy=float(U.sum()), energy_by_kind={kind: float(U[kinds == kind].sum())
                    for kind in ("bulk", "cohesive", "support", "platen")},
                states=[r.state for r in responses])


def kinetic_and_momentum(frames, velocities, masses, inertias):
    I = frames[:, :3, :3]@inertias@frames[:, :3, :3].transpose(0, 2, 1)
    L = np.einsum("nij,nj->ni", I, velocities[:, 3:])
    kinetic = .5*np.sum(masses[:, None]*velocities[:, :3]**2)+.5*np.sum(velocities[:, 3:]*L)
    return float(kinetic), L


def check_sample(fixture, row, trial, initial_jumps):
    """Fixed engineering gates, including independent bulk strain reference."""
    config, net = fixture["config"], fixture["net"]
    values = [row[k] for k in ("t", "maximum_cell_momentum_residual_n",
        "global_axial_momentum_residual_n", "maximum_cell_angular_residual_nm",
        "cumulative_energy_residual_j", "maximum_cell_drift_m", "full_contact_force_n",
        "carriage_force_n", "distal_half_contact_axial_n", "mean_seam_gap_m")]
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite acceptance evidence")
    if (row["maximum_cell_momentum_residual_n"] > MOMENTUM_N
            or abs(row["global_axial_momentum_residual_n"]) > MOMENTUM_N
            or row["maximum_cell_angular_residual_nm"] > MOMENTUM_N*config.band.radius_m):
        raise RuntimeError("Predeclared cell/global force or angular momentum residual")
    energy_limit = 1e-9 if config.case == "rest" else .05*net["reference_energy"]
    if abs(row["cumulative_energy_residual_j"]) > energy_limit:
        raise RuntimeError("Predeclared cumulative energy residual")
    if row["phase"] == "rest":
        if row["maximum_cell_drift_m"] > REST_DRIFT_M or row["full_contact_force_n"] > MOMENTUM_N:
            raise RuntimeError("Rest drift/preload, including nonface contacts")
    ends = ({"rest": 1.} if config.case == "rest" else
            {"compression_hold": 3.5, "separated_hold": 6.5} if config.case == "preseparated" else
            {"elastic_hold": 3.5, "compression_hold": 7., "elastic_rehold": 9.})
    tail = row["phase"] in ends and row["t"] > ends[row["phase"]]-.2+1e-10
    if not tail:
        return None
    if row["phase"].startswith("elastic"):
        F = row["carriage_force_n"]
        if abs(F-FORCE_N) > REFERENCE_TOLERANCE*FORCE_N:
            raise RuntimeError("Settled axial load outside 0.05 N +/-5%")
        expected = net["reference_jumps"]*(F/FORCE_N)
        bulk = np.array([r["kind"] == "bulk" for r in net["records"]])
        weighted_error = np.sqrt(net["stiffness"][bulk])*(trial["jumps"][bulk]-initial_jumps[bulk]-expected[bulk])
        weighted_reference = np.sqrt(net["stiffness"][bulk])*expected[bulk]
        row["bulk_reference_relative_error"] = float(np.linalg.norm(weighted_error)/np.linalg.norm(weighted_reference))
        if row["bulk_reference_relative_error"] > REFERENCE_TOLERANCE:
            raise RuntimeError("Measured bulk anchor strain differs from assembled-network reference by >5%")
        cohesive = np.array([r["kind"] == "cohesive" for r in net["records"]])
        if np.max(abs(trial["jumps"][cohesive, 0]-expected[cohesive, 0])) > REFERENCE_TOLERANCE*np.max(expected[cohesive, 0]):
            raise RuntimeError("Settled cohesive opening differs from elastic network reference")
    elif row["phase"] == "compression_hold":
        if row["distal_half_contact_axial_n"] <= 1e-6 or row["mean_seam_gap_m"] > 1e-7:
            raise RuntimeError("Compression hold lacks independent native contact reaction")
    elif row["phase"] == "separated_hold":
        if (abs(row["mean_seam_gap_m"]-config.initial_gap) > REFERENCE_TOLERANCE*config.initial_gap
                or row["full_contact_force_n"] > 1e-5):
            raise RuntimeError("Initially unbonded halves did not reopen without cohesive restraint")
    return row["phase"]


def run(sim, fixture, *, native_errors):
    """Caller owns native context; 32 material bodies plus one fixture carriage."""
    from .blade_loading_probe import NativeContacts
    from greenhouse_sim.physics_clock import PhysicsClock
    native_errors.check("band_run_entry")
    config, net = fixture["config"], fixture["net"]
    dt = 1/config.physics_hz
    if not np.isclose(sim.get_physics_dt(), dt, rtol=1e-8, atol=1e-12):
        raise RuntimeError("Native timestep mismatch")
    view = sim.physics_sim_view; view.set_subspace_roots("/")
    cells = view.create_rigid_body_view(ROOT+"/Cells/*")
    carriage = view.create_rigid_body_view(fixture["paths"][-1])
    if set(cells.prim_paths) != set(fixture["paths"][:32]) or list(carriage.prim_paths) != fixture["paths"][-1:]:
        raise RuntimeError("Expected exactly 32 material bodies and one laboratory carriage")
    order = [list(cells.prim_paths).index(p) for p in fixture["paths"][:32]]
    sensor = view.create_rigid_contact_view(ROOT+"/Cells/*")
    if set(sensor.sensor_paths) != set(fixture["paths"][:32]):
        raise RuntimeError("Native sensor-path inventory mismatch")
    sensor_order = [list(sensor.sensor_paths).index(p) for p in fixture["paths"][:32]]

    def read(method):
        return np.concatenate((np.asarray(getattr(cells, method)())[order],
                               np.asarray(getattr(carriage, method)())), axis=0).copy()

    masses = read("get_masses").reshape(33).astype(float)
    inertias = read("get_inertias").reshape(33, 3, 3).transpose(0, 2, 1).astype(float)
    coms = read("get_coms")
    if (not np.isfinite(np.r_[masses, inertias.ravel(), coms.ravel()]).all()
            or not np.allclose(masses, net["masses"], rtol=.005, atol=0)
            or np.max(abs(coms[:, :3])) > 1e-9
            or np.max(np.linalg.norm(inertias-net["inertias"], axis=(1, 2))/np.linalg.norm(net["inertias"], axis=(1, 2))) > .005):
        raise RuntimeError("Native source-volume mass/COM/inertia validation failed")
    poses, velocities = read("get_transforms"), read("get_velocities").astype(float)
    frames = pose_matrices(poses)
    initial_frames = frames.copy()
    states = [link.state for link in fixture["links"]]
    stiffness = fixture["applied_stiffness"].copy()
    cohesive = np.array([r["kind"] == "cohesive" for r in net["records"]])
    settings, stiffness[cohesive] = coefficient_snapshot(fixture)
    previous = anchor_response(fixture, frames, stiffness, states)
    states = previous["states"]
    initial_jumps = previous["jumps"].copy()
    initial_U = previous["energy"]
    initial_ke, previous_L = kinetic_and_momentum(frames, velocities, masses, inertias)
    native_errors.check("initial_band_state_readback")
    clock = PhysicsClock(sim, physics_hz=config.physics_hz, render_hz=0)
    steps = round(config.duration*config.physics_hz)
    arrays = {name: np.full(shape, np.nan, dtype) for name, shape, dtype in (
        ("poses", (steps, 33, 7), np.float32), ("velocities", (steps, 33, 6), np.float32),
        ("anchor_jumps_m", (steps, 216, 3), np.float64), ("contact_forces_n", (steps, 32, 3), np.float32),
        ("normal_contact_forces_n", (steps, 32, 3), np.float32),
        ("friction_contact_forces_n", (steps, 32, 3), np.float32),
        ("joint_forces_n", (steps, 33, 3), np.float64), ("joint_torques_nm", (steps, 33, 3), np.float64))}
    contacts = NativeContacts(); contacts.subscribe()
    rows, tails, error = [], {}, None
    command, phase, native_time, act_work, contact_work = 0., "rest", float(sim.current_time), 0., 0.

    def before(stamp, step_dt):
        nonlocal command, phase, settings
        native_errors.check("before_band_coefficient_writes")
        contacts.rows, contacts.friction = [], []
        for link, state in zip(fixture["links"], states):
            link.update(state)
        settings, stiffness[cohesive] = coefficient_snapshot(fixture)
        if (not fixture["guide"].GetJointEnabledAttr().Get()
                or not all(j.GetJointEnabledAttr().Get() for j in fixture["joints"])):
            raise RuntimeError("Unexpected disabled bulk/boundary/cohesive joint")
        phase, command = protocol(stamp.simulation_time_s, config, net)
        fixture["loading"].GetTargetPositionAttr().Set(command)
        native_errors.check("after_band_writes_before_solve")

    def after(stamp, step_dt):
        nonlocal frames, velocities, states, previous, previous_L, native_time, act_work, contact_work
        native_errors.check("after_band_solve")
        p, v = read("get_transforms"), read("get_velocities").astype(float)
        f = pose_matrices(p)
        cf = np.asarray(sensor.get_net_contact_forces(step_dt), float)[sensor_order].copy()
        i = len(rows)
        row = dict(t=stamp.simulation_time_s, dt=step_dt, phase=phase, command_m=command,
                   accepted=False, contacts=list(contacts.rows), friction=list(contacts.friction),
                   array_sample=i)
        rows.append(row); arrays["poses"][i] = p; arrays["velocities"][i] = v
        arrays["normal_contact_forces_n"][i] = cf
        # Stop even for empty error strings. Never ignore unfiltered central-axis contacts.
        if contacts.error is not None:
            raise RuntimeError(contacts.error)
        now = float(sim.current_time)
        if not np.isclose(now-native_time, step_dt, rtol=1e-6, atol=1e-9):
            raise RuntimeError("Native clock did not advance exactly one step")
        if (not np.isfinite(np.r_[v.ravel(), cf.ravel()]).all()
                or np.max(np.linalg.norm(v[:, 3:], axis=1)) > ANGULAR_SPEED
                or np.max(np.linalg.norm(f[:, :3, :3]-np.eye(3), axis=(1, 2))) > ROTATION_RAD
                or abs(f[-1, 2, 3]-fixture["initial_centres"][-1, 2]) > JUMP_M):
            raise RuntimeError("Native finite pose/velocity/carriage travel guard")
        cw = contact_wrenches(fixture, contacts.rows, contacts.friction, cf, f, step_dt)
        row.update(cap_status(fixture, contacts.rows, f, v, step_dt))
        arrays["friction_contact_forces_n"][i] = cw["friction"]
        arrays["contact_forces_n"][i] = cw["force"]
        check_caps_and_speed(row)
        if cw["total"] > CONTACT_N or cw["minimum"] < -PENETRATION_M:
            raise RuntimeError("Full native contact/penetration guard")
        trial = anchor_response(fixture, f, stiffness, states)
        arrays["anchor_jumps_m"][i] = trial["jumps"]
        arrays["joint_forces_n"][i] = trial["force"]; arrays["joint_torques_nm"][i] = trial["torque"]
        F = float(np.clip(CARRIAGE_K*(command-(f[-1, 2, 3]-fixture["initial_centres"][-1, 2]))
                          -CARRIAGE_C*v[-1, 2], -FORCE_CAP_N, FORCE_CAP_N))
        inertial = masses[:, None]*(v[:, :3]-velocities[:, :3])/step_dt
        residual = inertial-trial["force"]
        residual[:32] -= cw["force"]; residual[-1, 2] -= F
        ke, L = kinetic_and_momentum(f, v, masses, inertias)
        angular = (L-previous_L)/step_dt-trial["torque"]
        angular[:32] -= cw["torque"]
        dx = f[:, :3, 3]-frames[:, :3, 3]
        dR = f[:, :3, :3]@frames[:, :3, :3].transpose(0, 2, 1)
        # Tiny guarded rotation: sine-rotation endpoint work; approximation is explicit.
        dtheta = .5*np.stack((dR[:, 2, 1]-dR[:, 1, 2], dR[:, 0, 2]-dR[:, 2, 0], dR[:, 1, 0]-dR[:, 0, 1]), axis=1)
        act_work += F*dx[-1, 2]
        contact_work += float(np.sum(cw["force"]*dx[:32])+np.sum(cw["torque"]*dtheta[:32]))
        balance = ke-initial_ke+trial["energy"]-initial_U-act_work-contact_work
        row.update(native_context_time_s=now, carriage_force_n=F,
            maximum_cell_momentum_residual_n=float(np.max(np.linalg.norm(residual[:32], axis=1))),
            global_axial_momentum_residual_n=float(residual[:, 2].sum()),
            maximum_cell_angular_residual_nm=float(np.max(np.linalg.norm(angular[:32], axis=1))),
            full_contact_force_n=cw["total"], minimum_contact_separation_m=cw["minimum"],
            nonface_contact_records=cw["nonface_records"],
            callback_tensor_max_error_n=cw["callback_tensor_max_error_n"],
            maximum_cell_drift_m=float(np.max(np.linalg.norm(f[:32, :3, 3]-initial_frames[:32, :3, 3], axis=1))),
            distal_half_contact_axial_n=float(cw["force"][16:, 2].sum()),
            mean_seam_gap_m=float(trial["jumps"][cohesive, 0].mean()),
            stored_energy_j=trial["energy"], energy_by_kind_j=trial["energy_by_kind"],
            kinetic_energy_j=ke, cumulative_carriage_work_j=act_work,
            cumulative_contact_work_j=contact_work, cumulative_energy_residual_j=balance,
            maximum_damage=max(s.damage for s in trial["states"]),
            initially_unbonded=config.case == "preseparated", new_fracture_work_j=0.)
        native_errors.check("band_before_state_commit")
        tail = check_sample(fixture, row, trial, initial_jumps)
        if tail is not None:
            tails[tail] = tails.get(tail, 0)+1
        row["accepted"] = True
        frames, velocities, previous, previous_L, native_time = f, v, trial, L, now
        states = trial["states"]

    try:
        for _ in range(steps):
            clock.tick(before=before, after=after)
        required = 1 if config.case == "rest" else 2 if config.case == "preseparated" else 3
        if len(tails) != required or any(n != round(.2*config.physics_hz) for n in tails.values()):
            raise RuntimeError("Incomplete bounded hold-tail evidence")
        native_errors.check("band_protocol_finished")
    except Exception as fault:
        error = str(fault)
        if rows:
            rows[-1]["stop_reason"] = error
    finally:
        fixture["loading"].GetMaxForceAttr().Set(0.)
        contacts.subscription = None
    return dict(error=error, bounded_protocol_accepted=error is None and len(rows) == steps
                and not native_errors.report()["faulted"] and not native_errors.report()["logging_interface_injected"],
        records=rows, arrays={k: v[:len(rows)] for k, v in arrays.items()}, timing=clock.report(),
        config=asdict(config), native_masses_kg=masses.tolist(), native_inertias_kg_m2=inertias.tolist(),
        native_com_principal_frames=coms.tolist(), initial_constitutive_energy_j=initial_U,
        density_kg_m3=config.band.density_kg_m3,
        density_evidence="explicit rho*exact_polygon_volume authored mass; native mass/inertia readback; no native density accessor",
        hold_tail_samples=tails, native_error_observer=native_errors.report(),
        reference=dict(force_n=FORCE_N, carriage_extension_m=net["reference_q"][-1],
            stored_energy_j=net["reference_energy"], anchor_jumps_m=net["reference_jumps"].tolist()),
        guards=dict(momentum_n=MOMENTUM_N, angular_momentum_nm=MOMENTUM_N*.003,
            energy_j=1e-9 if config.case == "rest" else .05*net["reference_energy"],
            contact_n=CONTACT_N, penetration_m=PENETRATION_M, rest_drift_m=REST_DRIFT_M,
            bulk_jump_m=BULK_JUMP_M, all_anchor_jump_m=JUMP_M, reference_relative=REFERENCE_TOLERANCE),
        band=net["band"].summary(), body_paths=fixture["paths"],
        fixture_carriage="noncolliding_laboratory_boundary_2mm_32gon_mass_NOT_tissue_or_knife_proxy",
        physics_boundaries=dict(body_usd_settings=fixture["body_settings"],
            native_linear_cap_m_s=NATIVE_LINEAR_CAP_M_S, speed_watchdog_m_s=SPEED_M_S,
            native_depenetration_cap_m_s=NATIVE_DEPENETRATION_CAP_M_S,
            cap_activation_telemetry_available=False, unclamped_solver_dynamics_verified=False,
            cap_screen="stop at >=99% endpoint speed or penetration/dt + abs(relative point normal speed); not internal solver telemetry",
            support="24 EA/h,GA/h remote-proximal world anchors; no seam bypass",
            distal="24 EA/h,GA/h anchors spanning all 8 end cells; noncolliding guided carriage",
            carriage=dict(stiffness_n_m=CARRIAGE_K, damping_ns_m=CARRIAGE_C, force_cap_n=FORCE_CAP_N),
            contact_material=dict(friction=config.band.friction, restitution=0., compliant_contact=False,
                                  contact_offset_m=config.band.contact_offset_m, rest_offset_m=0.),
            scene=dict(solver="PGS", gpu_dynamics=False, broadphase="MBP", friction="patch", gravity_m_s2=0.)),
        contact_force_contract="original header IDs; +J/dt collider0, -J/dt collider1; compare normal callback ONLY to normal tensor; add separate friction for momentum/work; no sign fit",
        applied_stiffness_n_m=stiffness.tolist(), force_work_provenance="implicit_native_drives; reconstructed_joint_wrenches; native_contact_forces; endpoint_work",
        contact_angular_work="sine_rotation_endpoint_approximation_under_small_rotation_guard",
        material_calibrated=False, continuum_qualified=False, native_response_validated=False,
        fracture_produced=False, physical_cut_verified=False, training_eligible=False)


def _native_session(config, output, observer):
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics
    from isaacsim.core.api import SimulationContext
    observer.check("before_band_authoring")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    scene_path = "/World/BandPhysics"
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityMagnitudeAttr(0.)
    api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    api.CreateSolverTypeAttr("PGS"); api.CreateEnableGPUDynamicsAttr(False)
    api.CreateBroadphaseTypeAttr("MBP"); api.CreateFrictionTypeAttr("patch")
    fixture = author(stage, config)
    observer.check("after_band_authoring_before_activation")
    if not all(link.report()["native_schema_registered"] for link in fixture["links"]):
        raise RuntimeError("Registered native limit schema required; stock USD is not a native test")
    coefficient_snapshot(fixture)
    for j in fixture["joints"]:
        j.GetJointEnabledAttr().Set(True)
    observer.check("after_band_activation_before_reset")
    stage.GetRootLayer().Export(str(output/"coupon_authored.usda"))
    sim = SimulationContext(physics_dt=1/config.physics_hz, rendering_dt=1/60,
        physics_prim_path=scene_path, stage_units_in_meters=1., backend="numpy", set_defaults=False)
    result = None
    try:
        sim.get_physics_context().enable_fabric(True)
        sim.reset()
        observer.check("after_band_reset")
        if api.GetSolverTypeAttr().Get() != "PGS" or api.GetEnableGPUDynamicsAttr().Get() or scene.GetGravityMagnitudeAttr().Get() != 0:
            raise RuntimeError("Native scene settings changed")
        result = run(sim, fixture, native_errors=observer)
    finally:
        try:
            sim.stop()
            observer.check("after_band_stop")
        except Exception as error:
            if result is None:
                raise
            result["error"] = str(error); result["bounded_protocol_accepted"] = False
    return result


def _write_evidence(output, result):
    from .cohesive_native_probe import _hash
    with (output/"native_samples.npz").open("xb") as stream:
        np.savez_compressed(stream, **result.pop("arrays"))
    with (output/"trace.jsonl").open("x", encoding="utf-8") as stream:
        for row in result.pop("records"):
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    result["evidence_sha256"] = {name: _hash(output/name) for name in ("native_samples.npz", "trace.jsonl")}
    result["array_contract"] = "row.array_sample indexes NPZ; bodies 0..31 material, 32 fixture; unavailable rejected-sample arrays are NaN"


def main(argv=None):
    import traceback
    from .cohesive_native_probe import _hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("rest", "elastic", "preseparated"), required=True)
    parser.add_argument("--physics-hz", type=int, choices=(960, 1920), default=960)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-target", required=True)
    for name in ("radius-m", "length-m", "density-kg-m3", "youngs-pa", "poisson",
                 "kn-pa-m", "kt-pa-m", "strength-pa", "gc-j-m2"):
        parser.add_argument("--"+name, type=float, required=True)
    args = parser.parse_args(argv)
    if not args.native_run:
        parser.error("Main's explicit reviewed --native-run required")
    material = Material(youngs_modulus_pa=args.youngs_pa, density_kg_m3=args.density_kg_m3, poisson_ratio=args.poisson)
    config = Config(band_api.BandConfig(source_manifest_path=args.source_manifest, source_target=args.source_target,
        radius_m=args.radius_m, length_m=args.length_m, density_kg_m3=args.density_kg_m3,
        bulk_material=material, cohesive_material=CohesiveParameters(
            args.kn_pa_m, args.kt_pa_m, args.strength_pa, args.gc_j_m2)), args.case, args.physics_hz)
    sources = source_receipt(config)
    output = args.output.resolve()
    allowed = Path(__file__).resolve().parents[3]/"data/sim_physics"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("New child of repository data/sim_physics required")
    output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False, "sync_loads": False})
    observer, result, hashes, exit_code = None, None, {}, 2
    try:
        from .native_errors import NativeErrors
        from greenhouse_sim import physics_clock
        dependencies = [Path(__file__), Path(physics_clock.__file__)]
        dependencies += [Path(__file__).with_name(name+".py") for name in (
            "material_band", "cohesive", "cohesive_unilateral", "cohesive_closing_probe",
            "cohesive_native_probe", "mechanics", "runtime", "blade_loading_probe", "native_errors")]
        hashes = {str(p): _hash(p) for p in dependencies}
        observer = NativeErrors()
        with observer:
            result = _native_session(config, output, observer)
            observer.check("before_band_observer_removal")
        result["native_error_observer"] = observer.report()
        unchanged = all(_hash(p) == h for p, h in {**sources, **hashes}.items())
        result.update(source_sha256=sources, implementation_sha256=hashes,
            source_and_implementation_unchanged=unchanged,
            native_error_scope="coupon_authoring_through_stop_and_logger_removal_not_entire_app_or_flushed_async")
        if not unchanged:
            result["error"] = "Source or implementation changed during coupon"
        result["bounded_protocol_accepted"] = bool(result["bounded_protocol_accepted"]
            and result["error"] is None and not observer.report()["faulted"] and observer.report()["closed"])
        result["state"] = "band_completed_unqualified" if result["error"] is None else "band_stopped_unqualified"
        _write_evidence(output, result)
        with (output/"report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        exit_code = 0 if result["bounded_protocol_accepted"] else 2
        print(json.dumps(dict(state=result["state"], error=result["error"], output=str(output))), flush=True)
    except Exception as error:
        if result is not None:
            result["bounded_protocol_accepted"] = False
            result["error"] = str(error)
            result["native_error_observer"] = observer.report() if observer else None
            if "records" in result:
                _write_evidence(output, result)
        with (output/"failure.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(error=str(error), traceback=traceback.format_exc(),
                source_sha256=sources, implementation_sha256=hashes, protocol_result=result,
                native_error_observer=observer.report() if observer else None,
                bounded_protocol_accepted=False, physical_cut_verified=False, training_eligible=False), stream, indent=2)
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
