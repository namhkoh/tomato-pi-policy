"""Standalone two-body, four-facet native cohesive COUPON; never a plant cut.

Integration with James's cohesive.py: B-minus-A stress-free anchor jump in the
supported A frame, normal +X, reference area in m^2, traction stiffness in Pa/m.
Each translational D6 stiffness is area * retained_stiffness * material stiffness
[N/m]. Rotations and translations across the interface have NO rigid limits,
FixedJoint, residual damper, articulation, or parallel weld. B is a density-
authored box on a force-limited laboratory prismatic carriage, NOT a tool/grasp.

Native drives implicitly solve the last accepted positive secant stiffness;
damage is committed from measured POST-step anchors, then used next step. This
is a partitioned/lagged constitutive update, NOT a converged implicit nonlinear
cohesive solve. Stepwise damage/jump guards and explicit lag/work residual logs
prevent silently presenting that approximation as native constitutive truth.

D6 drives are bilateral. This deliberately restricted tensile/mixed-mode
coupon aborts on compression beyond floating-point tolerance; it does NOT
claim unilateral cohesive/contact coupling through closing/reopening cycles.
Frictionless box contact is separately enabled and never drives damage. The
native carriage holds transverse translation/orientation relative to the lab;
no bending, large rotation, compression, or continuum convergence is qualified.

Installed primary evidence (Isaac 6.0.1 / physics 110.1.13):
  omni.physx.demos/.../RigidBodyRopeDemo.py:_createJoint (D6/DriveAPI)
  omni.physx.tests/.../PhysicsJoint.py:_run_physics_joint_prismatic_limits_test
  omni.physics.tensors/omni/physics/tensors/api.py:RigidBodyView, RigidContactView
  omni.physx/omni/physx/bindings/_physx.pyi (no standalone drive-force readback)
  omni.usd.schema.physx/.../schema.usda (density/contact/solver semantics)
Native spring convention:
https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/_api_build/structPxD6Drive.html
https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/Joints.html

Drive force/work below are ENDPOINT RECONSTRUCTIONS, not sensor readbacks.
Only poses, velocities, masses/inertias, contacts and simulation time are native
measurements. No routine in this module is run automatically on import.
Main owns the single SimulationApp and must review code before --native-run.
Use a NEW output directory for every case/rate; never overwrite prior evidence.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from .cohesive import CohesiveParameters, CohesiveState, cohesive_response
from .runtime import pose_matrices


ENGINEERING = "uncalibrated_engineering_coupon_only"
ROOT = "/World/CohesiveCoupon"
AXES = ("transX", "transY", "transZ")


@dataclass(frozen=True)
class CouponConfig:
    material: CohesiveParameters  # Deliberately no fitted/default material.
    case: str = "softening"
    physics_hz: int = 960
    shear_ratio: float = 0.0  # B translation Y / X; tensile normal always > 0.
    density_kg_m3: float = 1000.0  # Toy solid coupon, NOT measured plant/tool.
    area_m2: float = 4e-6
    carriage_stiffness_n_m: float = 300.0
    carriage_damping_ns_m: float = 8.0
    carriage_max_force_n: float = 0.08
    maximum_speed_m_s: float = 0.01
    maximum_contact_force_n: float = 0.1
    maximum_interface_force_n: float = 0.4
    maximum_damage_increment: float = 0.05
    maximum_jump_increment_r0: float = 0.05
    compression_tolerance_m: float = 1e-7

    def __post_init__(self):
        if not isinstance(self.material, CohesiveParameters):
            raise ValueError("Explicit cohesive material parameters required")
        if self.case not in ("softening", "damage_disabled", "subcritical"):
            raise ValueError("Unknown coupon case")
        if type(self.physics_hz) is not int or self.physics_hz not in (480, 960, 1920):
            raise ValueError("Bounded 480/960/1920 Hz only; compare rates separately")
        numbers = {k: v for k, v in vars(self).items()
                   if k not in ("material", "case", "physics_hz")}
        if any(isinstance(v, (bool, str)) or not np.isfinite(v) for v in numbers.values()):
            raise ValueError("Finite numeric coupon configuration required")
        if not -1 <= self.shear_ratio <= 1:
            raise ValueError("Only bounded tensile/mixed-mode carriage directions")
        if any(v <= 0 for k, v in numbers.items() if k != "shear_ratio"):
            raise ValueError("Positive coupon scales required")
        if not (100 <= self.density_kg_m3 <= 5000 and 1e-7 <= self.area_m2 <= 1e-5):
            raise ValueError("Density/reference area outside isolated coupon bounds")
        if not (10 <= self.carriage_stiffness_n_m <= 500
                and 1 <= self.carriage_damping_ns_m <= 20
                and self.carriage_max_force_n <= 0.1
                and self.maximum_interface_force_n <= 0.4
                and self.maximum_contact_force_n <= 0.1
                and self.maximum_speed_m_s <= 0.01
                and self.maximum_damage_increment <= 0.05
                and self.maximum_jump_increment_r0 <= 0.05
                and self.compression_tolerance_m <= 1e-7):
            raise ValueError("Coupon limits may not relax the engineering guards")
        p = self.material
        if not 1e-5 <= p.initiation_separation_m < p.final_separation_m <= 0.0005:
            raise ValueError("Uncalibrated material separation scales outside coupon envelope")
        if self.amplitude_m / 4 > 0.001:
            raise ValueError("Commanded carriage speed exceeds 1 mm/s")
        # Avoid a known quasi-static snap-back from a softer lab spring than
        # the negative mixed-mode envelope slope; NOT a convergence proof.
        direction_scale2 = self.direction[0]**2 + (p.shear_weight*self.direction[1])**2
        softening_slope = self.area_m2 * p.normal_strength_pa * direction_scale2 / (p.final_separation_m-p.initiation_separation_m)
        if self.carriage_stiffness_n_m <= 1.1 * softening_slope:
            raise ValueError("Carriage spring too soft for bounded envelope; choose explicit material/config")

    @property
    def direction(self):
        v = np.array([1.0, self.shear_ratio, 0.0])
        return v / np.linalg.norm(v)

    @property
    def amplitude_m(self):
        p = self.material
        effective_per_travel = np.hypot(self.direction[0], p.shear_weight * self.direction[1])
        r = 0.25 * p.initiation_separation_m if self.case == "subcritical" else 1.4 * p.final_separation_m
        return r / effective_per_travel


def command_travel(t, config):
    """Lab actuator INPUT only. Never supplied to cohesive_response as evidence.

    0.5 s settling, 4 s pull, 1 s hold, 4 s return to 20% travel, 0.5 s hold.
    Return stays tensile; compression/contact testing requires another bridge.
    """
    if not np.isfinite(t) or t < 0:
        raise ValueError("Invalid command time")
    if t < 0.5:
        fraction = 0.0
    elif t < 4.5:
        fraction = (t - 0.5) / 4
    elif t < 5.5:
        fraction = 1.0
    else:
        fraction = max(0.2, 1 - 0.8 * (t - 5.5) / 4)
    return float(config.amplitude_m * fraction)


def reference_geometry(area_m2):
    """Four equal-area Gauss points on a square patch, inside larger box faces."""
    half_length = 0.005
    transverse = np.sqrt(area_m2) / (2 * np.sqrt(3))
    world = np.array([[0., y, z] for y in (-transverse, transverse)
                      for z in (-transverse, transverse)])
    frames = np.tile(np.eye(4), (2, 1, 1))
    frames[:, 0, 3] = [-half_length, half_length]
    local = world[None, :, :] - frames[:, None, :3, 3]
    return frames, local


def measured_jumps(frames, local):
    """Actual B anchor minus A anchor, transported to supported A's frame."""
    frames, local = np.asarray(frames, float), np.asarray(local, float)
    if (frames.shape != (2, 4, 4) or local.shape != (2, 4, 3)
            or not np.isfinite(frames).all() or not np.isfinite(local).all()):
        raise ValueError("Finite two-body transforms and four persistent anchors required")
    for f in frames:
        if not np.allclose(f[:3, :3].T @ f[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError("Rigid material frame required")
    anchors = np.einsum("bij,bfj->bfi", frames[:, :3, :3], local) + frames[:, None, :3, 3]
    return (anchors[1] - anchors[0]) @ frames[0, :3, :3], anchors


class FacetBridge:
    """Pure state adapter; inputs are measured anchors, never a command/time."""

    def __init__(self, config):
        self.config = config
        self.states = [CohesiveState(config.material) for _ in range(4)]
        self.last_jumps = np.zeros((4, 3))
        self.stored_energy_j = 0.0

    def trial(self, jumps):
        jumps = np.asarray(jumps, float)
        if jumps.shape != (4, 3) or not np.isfinite(jumps).all():
            raise ValueError("Four finite native anchor jumps required")
        c, p = self.config, self.config.material
        if np.max(np.linalg.norm(jumps, axis=1)) > 0.001:
            raise RuntimeError("1 mm measured material-anchor separation safety stop")
        if np.min(jumps[:, 0]) < -c.compression_tolerance_m:
            raise RuntimeError("Bilateral D6 bridge cannot qualify compression; stop, do not fake contact law")
        if np.max(np.linalg.norm((jumps - self.last_jumps) * [1, p.shear_weight, p.shear_weight], axis=1)) > c.maximum_jump_increment_r0 * p.initiation_separation_m:
            raise RuntimeError("Measured jump increment too large: reduce timestep; state not committed")
        responses = [cohesive_response(s, j, (1., 0., 0.), area_m2=c.area_m2 / 4)
                     for s, j in zip(self.states, jumps, strict=True)]
        shadow = np.array([r.state.damage for r in responses])
        if c.case != "damage_disabled" and max(r.state.damage - s.damage for r, s in zip(responses, self.states, strict=True)) > c.maximum_damage_increment:
            raise RuntimeError("Constitutive damage increment too large: reduce timestep; state not committed")
        retained = (np.ones(4) if c.case == "damage_disabled"
                    else np.array([r.state.retained_stiffness for r in responses]))
        stiffness = retained[:, None] * (c.area_m2 / 4) * np.array(
            [p.normal_stiffness_pa_m, p.shear_stiffness_pa_m, p.shear_stiffness_pa_m])
        tensile = jumps.copy()
        tensile[:, 0] = np.maximum(tensile[:, 0], 0.)
        forces = stiffness * tensile
        stored = float(0.5 * np.sum(forces * tensile))
        dissipated = 0. if c.case == "damage_disabled" else sum(r.dissipated_energy_j for r in responses)
        increment = 0. if c.case == "damage_disabled" else sum(r.dissipation_increment_j for r in responses)
        if c.case == "subcritical" and np.max(shadow) > 0:
            raise RuntimeError("Subcritical control damaged under measured load; no negative-control success")
        if np.sum(np.linalg.norm(forces, axis=1)) > c.maximum_interface_force_n:
            raise RuntimeError("Constitutive interface force guard")
        return dict(responses=responses, jumps=jumps.copy(), stiffness=stiffness,
                    force_a=forces, force_b=-forces, applied_damage=1-retained,
                    shadow_damage=shadow, stored=stored, dissipated=dissipated,
                    dissipation_increment=increment,
                    constitutive_work=stored-self.stored_energy_j+increment)

    def commit(self, trial):
        self.states = [r.state for r in trial["responses"]]
        self.last_jumps = trial["jumps"].copy()
        self.stored_energy_j = trial["stored"]


def _drive(prim, axis, stiffness, damping, limit):
    from pxr import UsdPhysics
    drive = UsdPhysics.DriveAPI.Apply(prim, axis)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(float(limit))
    drive.CreateTargetPositionAttr(0.)
    drive.CreateTargetVelocityAttr(0.)
    return drive


def author(stage, config, *, root=ROOT):
    """Pure USD authoring on an otherwise empty metre stage, before native reset."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    if stage.GetPrimAtPath(root):
        raise ValueError("New isolated coupon root required")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError("Explicit metre stage required")
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
           for p in stage.Traverse()):
        raise ValueError("Isolated stage only; do not inject coupon into a plant/robot scene")
    frames, local = reference_geometry(config.area_m2)
    UsdGeom.Xform.Define(stage, root)
    material = UsdShade.Material.Define(stage, root + "/CompressionContact")
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(0.)
    contact.CreateDynamicFrictionAttr(0.)
    contact.CreateRestitutionAttr(0.)
    paths = [root + "/Bodies/SupportedA", root + "/Bodies/CarriageB"]
    dimensions = np.array([0.01, 0.02, 0.02])
    # Author density; let native geometry supply mass/inertia, then read them back.
    for i, path in enumerate(paths):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*frames[i, :3, 3]))
        api = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        api.CreateKinematicEnabledAttr(i == 0)
        api.CreateVelocityAttr(Gf.Vec3f(0))
        api.CreateAngularVelocityAttr(Gf.Vec3f(0))
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateDensityAttr(config.density_kg_m3)
        for schema in ("PhysxRigidBodyAPI", "PhysxContactReportAPI"):
            body.GetPrim().AddAppliedSchema(schema)
        for name, kind, value in (
            ("sleepThreshold", Sdf.ValueTypeNames.Float, 0.),
            ("linearDamping", Sdf.ValueTypeNames.Float, 0.),
            ("angularDamping", Sdf.ValueTypeNames.Float, 0.),
            ("solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
            ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 8),
            ("maxLinearVelocity", Sdf.ValueTypeNames.Float, 0.02),
            ("maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, 0.01)):
            body.GetPrim().CreateAttribute("physxRigidBody:" + name, kind, custom=False).Set(value)
        body.GetPrim().CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float, custom=False).Set(0.)
        shape = UsdGeom.Cube.Define(stage, path + "/CompressionCollider")
        shape.CreateSizeAttr(1.)
        shape.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        shape.CreateDisplayColorAttr([Gf.Vec3f(.2, .5, .8) if i else Gf.Vec3f(.6, .6, .6)])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        shape.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        shape.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, custom=False).Set(1e-5)
        shape.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, custom=False).Set(0.)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose="physics")
    bridge = FacetBridge(config)
    virgin = bridge.trial(np.zeros((4, 3)))
    facet_drives, joints = [], []
    for facet in range(4):
        joint = UsdPhysics.Joint.Define(stage, root + f"/Facets/Facet_{facet}")
        joints.append(joint)
        joint.CreateBody0Rel().SetTargets([paths[0]])
        joint.CreateBody1Rel().SetTargets([paths[1]])
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateJointEnabledAttr(True)
        joint.CreateCollisionEnabledAttr(True)  # Independent compression contact survives damage.
        for side in (0, 1):
            getattr(joint, f"CreateLocalPos{side}Attr")(Gf.Vec3f(*local[side, facet]))
            getattr(joint, f"CreateLocalRot{side}Attr")(Gf.Quatf(1.))
        facet_drives.append([_drive(joint.GetPrim(), a, k, 0., 0.15)
                             for a, k in zip(AXES, virgin["stiffness"][facet], strict=True)])
    # World-to-B prismatic laboratory guide, not an A-to-B connection.
    guide = UsdPhysics.PrismaticJoint.Define(stage, root + "/LaboratoryCarriage")
    guide.CreateBody1Rel().SetTargets([paths[1]])
    guide.CreateExcludeFromArticulationAttr(True)
    guide.CreateAxisAttr("X")
    guide.CreateLocalPos0Attr(Gf.Vec3f(*frames[1, :3, 3]))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0))
    rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*config.direction)).GetQuat()
    guide.CreateLocalRot0Attr(Gf.Quatf(rotation))
    guide.CreateLocalRot1Attr(Gf.Quatf(rotation))
    # No axial hard stop within or beyond the test path: limits are omitted/free.
    loading = _drive(guide.GetPrim(), "linear", config.carriage_stiffness_n_m,
                     config.carriage_damping_ns_m, config.carriage_max_force_n)
    if any(p.IsA(UsdPhysics.FixedJoint) or p.HasAPI(UsdPhysics.ArticulationRootAPI)
           for p in stage.Traverse()):
        raise RuntimeError("Unapproved rigid connection/articulation in coupon")
    return dict(config=config, bridge=bridge, frames=frames, local=local, paths=paths,
                facet_drives=facet_drives, joints=joints, loading=loading,
                dimensions=dimensions, root=root)


def set_coefficients(fixture, stiffness):
    """USD change listener updates native implicit drives; zero after failure."""
    for row, values in zip(fixture["facet_drives"], stiffness, strict=True):
        for drive, k in zip(row, values, strict=True):
            drive.GetStiffnessAttr().Set(float(k))
            drive.GetDampingAttr().Set(0.)  # Never leave a post-failure dashpot.
            drive.GetMaxForceAttr().Set(0.15 if k > 0 else 0.)


def run(sim, fixture):
    """Caller supplies initialized single native app/context; no launch here."""
    from .blade_loading_probe import NativeContacts
    from greenhouse_sim.physics_clock import PhysicsClock
    c, bridge = fixture["config"], fixture["bridge"]
    if not np.isclose(sim.get_physics_dt(), 1 / c.physics_hz, rtol=1e-8):
        raise RuntimeError("Native dt differs from bounded configuration")
    view = sim.physics_sim_view
    view.set_subspace_roots("/")
    bodies = view.create_rigid_body_view(fixture["root"] + "/Bodies/*")
    if set(bodies.prim_paths) != set(fixture["paths"]):
        raise RuntimeError("Unexpected native coupon bodies")
    order = [list(bodies.prim_paths).index(p) for p in fixture["paths"]]
    masses = np.asarray(bodies.get_masses(), float)[order].reshape(2)
    inertias = np.asarray(bodies.get_inertias(), float)[order].reshape(2, 3, 3)
    expected_mass = float(c.density_kg_m3 * np.prod(fixture["dimensions"]))
    if (not np.isfinite(masses).all() or np.any(masses <= 0)
            or not np.allclose(masses, expected_mass, rtol=0.005)
            or not np.isfinite(inertias).all() or np.any(np.linalg.eigvalsh(inertias) <= 0)):
        raise RuntimeError("Native density-derived mass/inertia verification failed")
    contacts = NativeContacts()
    contacts.subscribe()
    clock = PhysicsClock(sim, physics_hz=c.physics_hz, render_hz=0)
    records, fault = [], None
    previous_frames = pose_matrices(bodies.get_transforms()[order])
    previous_velocity = np.asarray(bodies.get_velocities(), float)[order].copy()
    previous_native_time = float(sim.current_time)
    initial_jumps, _ = measured_jumps(previous_frames, fixture["local"])
    current = bridge.trial(initial_jumps)
    bridge.commit(current)
    set_coefficients(fixture, current["stiffness"])
    command = 0.
    cumulative = dict(constitutive_work_j=0., reconstructed_drive_absorbed_work_j=0.,
                      reconstructed_carriage_work_j=0.)

    def before(stamp, dt):
        nonlocal command
        contacts.rows = []
        contacts.friction = []
        command = command_travel(stamp.simulation_time_s, c)
        fixture["loading"].GetTargetPositionAttr().Set(command)

    def after(stamp, dt):
        nonlocal current, previous_frames, previous_velocity, previous_native_time
        frames = pose_matrices(bodies.get_transforms()[order])
        velocity = np.asarray(bodies.get_velocities(), float)[order].copy()
        jumps, anchors = measured_jumps(frames, fixture["local"])
        travel = float((frames[1, :3, 3] - fixture["frames"][1, :3, 3]) @ c.direction)
        speed = float(velocity[1, :3] @ c.direction)
        contact_magnitude = sum(np.linalg.norm(r["impulse"]) for r in contacts.rows + contacts.friction) / dt
        row = dict(t=stamp.simulation_time_s, dt=dt,
                   native_context_time_s=float(sim.current_time),
                   command_travel_m=command, command_is_not_separation_evidence=True,
                   native_frames=frames.tolist(), native_velocities=velocity.tolist(),
                   native_anchors_world_m=anchors.tolist(), measured_jumps_a_m=jumps.tolist(),
                   measured_carriage_travel_m=travel, measured_carriage_speed_m_s=speed,
                   native_contacts=list(contacts.rows), native_friction_anchors=list(contacts.friction),
                   native_contact_force_magnitude_n=float(contact_magnitude),
                   actual_joint_drive_forces_measured=False, accepted_step=False)
        records.append(row)  # Preserve offending observations before a guard raises.
        if not np.isclose(row["native_context_time_s"]-previous_native_time, dt, rtol=1e-6, atol=1e-9):
            raise RuntimeError("Native context did not advance exactly one configured timestep")
        if contacts.error:
            raise RuntimeError(contacts.error)
        if (not np.isfinite(velocity).all() or np.max(np.linalg.norm(velocity[:, :3], axis=1)) > c.maximum_speed_m_s
                or np.max(np.linalg.norm(velocity[:, 3:], axis=1)) > 0.01
                or not np.allclose(frames[0], fixture["frames"][0], atol=1e-6)
                or not np.allclose(frames[1, :3, :3], np.eye(3), atol=1e-4)
                or abs(travel) > min(0.001, max(1.2 * c.amplitude_m, 5e-5))
                or contact_magnitude > c.maximum_contact_force_n):
            raise RuntimeError("Native speed/pose/travel/contact safety guard")
        if not all(bool(j.GetJointEnabledAttr().Get()) for j in fixture["joints"]):
            raise RuntimeError("Unexpected native interface topology change")
        trial = bridge.trial(jumps)
        # Bilateral endpoint reconstruction of the coefficients used THIS solve.
        # Do not substitute the newly softened constitutive force into this log.
        lag_force = current["stiffness"] * jumps
        if np.max(np.abs(lag_force)) >= 0.15 or np.sum(np.linalg.norm(lag_force, axis=1)) > c.maximum_interface_force_n:
            raise RuntimeError("Possible cohesive drive saturation/force guard; coupling not qualified")
        raw_actuator = c.carriage_stiffness_n_m * (command - travel) - c.carriage_damping_ns_m * speed
        actuator = float(np.clip(raw_actuator, -c.carriage_max_force_n, c.carriage_max_force_n))
        old_travel = float((previous_frames[1, :3, 3] - fixture["frames"][1, :3, 3]) @ c.direction)
        absorbed = float(np.sum(lag_force * (jumps - bridge.last_jumps)))
        actuator_work = actuator * (travel - old_travel)
        world_force = lag_force @ frames[0, :3, :3].T
        constitutive_world = trial["force_a"] @ frames[0, :3, :3].T
        inertial = masses[1] * float((velocity[1, :3] - previous_velocity[1, :3]) @ c.direction) / dt
        residual = float(inertial - actuator + np.sum(world_force @ c.direction))
        def kinetic(f, v):
            omega = f[1, :3, :3].T @ v[1, 3:]
            return float(.5 * masses[1] * (v[1, :3] @ v[1, :3]) + .5 * omega @ inertias[1] @ omega)
        delta_ke = kinetic(frames, velocity) - kinetic(previous_frames, previous_velocity)
        cumulative["constitutive_work_j"] += trial["constitutive_work"]
        cumulative["reconstructed_drive_absorbed_work_j"] += absorbed
        cumulative["reconstructed_carriage_work_j"] += actuator_work
        row.update(accepted_step=True, applied_damage=trial["applied_damage"].tolist(),
                   shadow_law_damage=trial["shadow_damage"].tolist(),
                   used_drive_stiffness_n_m=current["stiffness"].tolist(),
                   next_drive_stiffness_n_m=trial["stiffness"].tolist(),
                   constitutive_force_a_world_n=constitutive_world.tolist(),
                   constitutive_force_b_world_n=(-constitutive_world).tolist(),
                   reconstructed_drive_force_a_world_n=world_force.tolist(),
                   reconstructed_drive_force_b_world_n=(-world_force).tolist(),
                   drive_minus_constitutive_force_a_n=(world_force-constitutive_world).tolist(),
                   reconstructed_carriage_force_n=actuator, carriage_force_limit_n=c.carriage_max_force_n,
                   native_inertial_force_along_carriage_n=float(inertial),
                   momentum_residual_n_if_no_contact=residual if contact_magnitude < 1e-8 else None,
                   actual_action_reaction_independently_measured=False,
                   stored_energy_j=trial["stored"], dissipated_energy_j=trial["dissipated"],
                   constitutive_work_increment_j=trial["constitutive_work"],
                   reconstructed_drive_absorbed_work_increment_j=absorbed,
                   drive_minus_constitutive_work_increment_j=absorbed-trial["constitutive_work"],
                   reconstructed_carriage_work_increment_j=actuator_work,
                   native_state_kinetic_energy_j=kinetic(frames, velocity),
                   reconstructed_energy_residual_j_if_no_contact=(delta_ke + trial["constitutive_work"] - actuator_work) if contact_magnitude < 1e-8 else None,
                   cumulative=dict(cumulative))
        bridge.commit(trial)
        current = trial
        set_coefficients(fixture, trial["stiffness"])
        previous_frames, previous_velocity = frames, velocity
        previous_native_time = row["native_context_time_s"]

    try:
        for _ in range(10 * c.physics_hz):
            clock.tick(before=before, after=after)
    except Exception as error:
        fault = str(error)
        if records:
            records[-1]["stop_reason"] = fault
    finally:
        fixture["loading"].GetMaxForceAttr().Set(0.)
        contacts.subscription = None
    return dict(state="bounded_coupon_completed_unqualified" if fault is None else "coupon_stopped_unqualified",
                error=fault, records=records, config=asdict(c), timing=clock.report(),
                material_status=ENGINEERING, native_masses_kg=masses.tolist(),
                native_inertias_kg_m2=inertias.tolist(), density_authored_kg_m3=c.density_kg_m3,
                native_effective_density_kg_m3=(masses/np.prod(fixture["dimensions"])).tolist(),
                native_dt_s=float(sim.get_physics_dt()),
                dt_sqrt_initial_k_over_m=float((1/c.physics_hz)*np.sqrt(c.area_m2*max(c.material.normal_stiffness_pa_m,c.material.shear_stiffness_pa_m)/masses[1])),
                negatives="subcritical and damage_disabled require separate new runs; not inferred here",
                drive_force_readback="unavailable_for_standalone_D6_in_installed_tensor_API",
                fully_implicit_constitutive_solve=False, constitutive_update_lag_steps=1,
                timestep_convergence_verified=False, native_drive_coupling_verified=False,
                material_calibrated=False, physical_cut_verified=False, robot_grasp_verified=False,
                seam_release_authorized=False, training_eligible=False)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", action="store_true", help="Main-agent opt-in AFTER code review")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("softening", "damage_disabled", "subcritical"), default="softening")
    parser.add_argument("--physics-hz", type=int, choices=(480, 960, 1920), default=960)
    parser.add_argument("--shear-ratio", type=float, default=0.)
    for name in ("kn-pa-m", "kt-pa-m", "strength-pa", "gc-j-m2"):
        parser.add_argument("--" + name, type=float, required=True)
    args = parser.parse_args(argv)
    config = CouponConfig(CohesiveParameters(args.kn_pa_m, args.kt_pa_m, args.strength_pa, args.gc_j_m2),
                          case=args.case, physics_hz=args.physics_hz, shear_ratio=args.shear_ratio)
    if not args.native_run:
        parser.error("No native launch without main-agent --native-run after code review")
    output = args.output.resolve()
    allowed = Path(__file__).resolve().parents[3] / "data" / "sim_physics"
    if not output.is_relative_to(allowed.resolve()) or output == allowed.resolve():
        raise ValueError("New child directory under repository data/sim_physics required")
    output.mkdir(parents=True, exist_ok=False)
    # ONLY main's explicit invocation reaches SimulationApp. No source assets used.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False, "sync_loads": False})
    sim = None
    exit_code = 2
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from isaacsim.core.api import SimulationContext
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, "Z")
        scene_path = "/World/CouponPhysics"
        physics = UsdPhysics.Scene.Define(stage, scene_path)
        physics.CreateGravityMagnitudeAttr(0.)  # Isolated zero-gravity mechanical coupon.
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
        api.CreateSolverTypeAttr("PGS")
        api.CreateEnableGPUDynamicsAttr(False)
        api.CreateBroadphaseTypeAttr("MBP")
        api.CreateFrictionTypeAttr("patch")
        fixture = author(stage, config)
        # Snapshot authoring, not a source/release asset; exclusive output directory.
        stage.GetRootLayer().Export(str(output / "coupon_authored.usda"))
        sim = SimulationContext(physics_dt=1/config.physics_hz, rendering_dt=1/60,
                                physics_prim_path=scene_path, stage_units_in_meters=1,
                                backend="numpy", set_defaults=False)
        sim.get_physics_context().enable_fabric(True)
        sim.reset()
        if (api.GetSolverTypeAttr().Get() != "PGS" or api.GetEnableGPUDynamicsAttr().Get()
                or physics.GetGravityMagnitudeAttr().Get() != 0.):
            raise RuntimeError("Native coupon scene configuration changed")
        result = run(sim, fixture)
        result["scene"] = dict(solver="PGS", gpu_dynamics=False, gravity_m_s2=0.,
                               compression_contact="separate_frictionless_box_contacts")
        result["implementation_sha256"] = {str(p): _hash(p) for p in
            (Path(__file__), Path(__file__).with_name("cohesive.py"))}
        with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:
            for record in result.pop("records"):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        result["trace_sha256"] = _hash(output / "trace.jsonl")
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({"state": result["state"], "error": result["error"], "output": str(output)}), flush=True)
        exit_code = 0 if result["error"] is None else 2
    except Exception as error:
        with (output / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(state="coupon_failed_unqualified", error=str(error), traceback=traceback.format_exc(),
                           physical_cut_verified=False, training_eligible=False), stream, indent=2)
    finally:
        if sim is not None:
            sim.stop()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
