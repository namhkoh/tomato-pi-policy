"""Uncalibrated two-body closing/reopening COUPON, never blade/grasp evidence.

Reuses the unchanged hashed tensile coupon geometry. The native unilateral
helper owns the soft upper normal limit and freeing that axis on full failure.
Constitutive damage is committed from accepted native material-anchor jumps,
then applied before the next solve. Native implicit unilateral elasticity does
not make this a fully implicit nonlinear damage solve.

No source assets, plant/material-band integration, or runtime launch on import.
Main must review this module and the helper before any explicit native run.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import traceback

import numpy as np

from .cohesive import CohesiveParameters, CohesiveState, cohesive_response
from . import cohesive_native_probe as tensile
from .runtime import pose_matrices


ROOT = "/World/CohesiveClosingCoupon"
ENGINEERING = "uncalibrated_native_unilateral_closing_coupon"
DURATION_S = 11
MOMENTUM_LIMIT_N = .00041  # 1e-5 N + 1% of fixed .04 N reference.
ENERGY_LIMIT_J = .4e-6  # 5% of fixed 8 microjoule fracture-work reference.
INTACT_ENERGY_LIMIT_J = .025e-6  # 5% of intended .5 microjoule elastic peak.
HOLD_TAIL_S = .2
HOLD_END_S = {"open_hold": 4., "compression_hold": 8., "reopen_hold": 11.}


class ProtocolAcceptance:
    """Predeclared coupon checks; no native launch, force application or repair.

    Called on each computed candidate row BEFORE history commit. A first
    violation latches failure, including excursions that could later cancel.
    Native error observation is separately mandatory in run().
    """

    def __init__(self, config):
        self.config = config
        self.count = 0
        self.tails = dict.fromkeys(HOLD_END_S, 0)
        self.previous_damage = np.zeros(4)
        self.previous_failed = np.zeros(4, dtype=bool)
        self.partial_damage = None
        self.partial_dissipation = None
        self.momentum_squares = self.maximum_momentum = self.maximum_energy = 0.
        self.failure = None
        self.complete = False

    def observe(self, row):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        try:
            self._observe(row)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            self.failure = str(error)
            raise RuntimeError(self.failure) from error

    def _observe(self, row):
        dt = 1/self.config.coupon.physics_hz
        if (self.complete or not np.isclose(row["t"], (self.count+1)*dt, atol=1e-9, rtol=0)
                or not np.isclose(row["dt"], dt, atol=1e-12, rtol=0)
                or row["phase"] != protocol(self.count*dt, self.config)[0]):
            raise RuntimeError("Acceptance requires every consecutive protocol sample")
        residual = float(row["momentum_residual_with_native_contact_n"])
        energy = float(row["cumulative"]["energy_balance_residual_j"])
        damage = np.asarray(row["damage"], float)
        failed = np.asarray(row["fully_separated"])
        jumps = np.asarray(row["measured_jumps_a_m"], float)
        stiffness = np.asarray(row["used_stiffness_n_m"], float)
        normal_force = np.asarray(row["reconstructed_normal_force_a_n"], float)
        contact = np.asarray(row["native_contact_force_b_world_n"], float)
        dissipated = float(row["dissipated_energy_j"])
        if (damage.shape != (4,) or failed.shape != (4,) or failed.dtype != bool
                or jumps.shape != (4, 3) or stiffness.shape != (4, 3)
                or normal_force.shape != (4,) or contact.shape != (3,)
                or not np.isfinite(np.r_[residual, energy, dissipated, damage, jumps.ravel(),
                                          stiffness.ravel(), normal_force, contact]).all()
                or np.any((damage < 0) | (damage > 1)) or np.any(stiffness < 0)
                or dissipated < 0):
            raise RuntimeError("Nonfinite/invalid acceptance evidence")
        if abs(residual) > MOMENTUM_LIMIT_N:
            raise RuntimeError("Per-step momentum residual exceeds 0.00041 N")
        limit = INTACT_ENERGY_LIMIT_J if self.config.case == "intact" else ENERGY_LIMIT_J
        if abs(energy) > limit:
            raise RuntimeError(f"Cumulative energy residual exceeds {limit:g} J")
        if np.any(damage < self.previous_damage) or np.any(self.previous_failed & ~failed):
            raise RuntimeError("Irreversible facet state healed")
        if np.any(normal_force[jumps[:, 0] <= 0] != 0):
            raise RuntimeError("Normal cohesive force present in compression")
        # The first newly failed sample used the previous retained stiffness.
        # On the NEXT solve, every previously failed facet must be fully free.
        settings = row["used_unilateral_settings"]
        if len(settings) != 4:
            raise RuntimeError("Missing four-facet normal settings")
        def free(index):
            return (np.all(stiffness[index] == 0)
                    and settings[index]["normal_limit_present"] is False
                    and settings[index]["normal_axis_free"] is True
                    and settings[index]["normal_low"] is None
                    and settings[index]["normal_high"] is None)
        if any(not free(i) for i in np.flatnonzero(self.previous_failed)):
            raise RuntimeError("Failed facet retained stiffness or a bounded normal axis")
        if self.config.case == "intact" and (np.any(damage != 0) or dissipated != 0):
            raise RuntimeError("Intact acceptance requires zero damage and dissipation")
        if self.config.case == "damaged" and row["phase"] in (
                "closing", "compression_hold", "reopening", "reopen_hold"):
            if (self.partial_damage is None or not np.array_equal(damage, self.partial_damage)
                    or dissipated != self.partial_dissipation):
                raise RuntimeError("Partial damage/dissipation not preserved through closing/reopening")
        phase = row["phase"]
        in_tail = phase in HOLD_END_S and row["t"] > HOLD_END_S[phase]-HOLD_TAIL_S+1e-10
        if in_tail:
            if phase == "compression_hold":
                if (contact@self.config.coupon.direction <= 0
                        or np.any(jumps[:, 0] > self.config.coupon.compression_tolerance_m)):
                    raise RuntimeError("Compression hold lacks positive native compression contact")
            elif phase == "reopen_hold" and np.any(jumps[:, 0] <= 0):
                raise RuntimeError("Reopen hold lacks measured positive facet opening")
            if self.config.case == "intact" and phase in ("open_hold", "reopen_hold"):
                expected = 25e-6 if phase == "open_hold" else 50e-6
                if np.any((jumps[:, 0] < .95*expected) | (jumps[:, 0] > 1.05*expected)):
                    raise RuntimeError("Intact settled opening outside fixed 25/50 micrometre +/-5% band")
            if self.config.case == "damaged" and phase == "open_hold":
                if not np.all((damage > 0) & (damage < 1)) or np.any(failed):
                    raise RuntimeError("Open hold requires strictly partial damage on every facet")
                self.partial_damage = damage.copy()
                self.partial_dissipation = dissipated
            if self.config.case == "failed" and (not np.all(failed) or not all(free(i) for i in range(4))):
                raise RuntimeError("Failed-case hold requires all facets failed, zero stiffness and free normal axes")
            self.tails[phase] += 1
        self.previous_damage, self.previous_failed = damage.copy(), failed.copy()
        self.count += 1
        self.momentum_squares += residual*residual
        self.maximum_momentum = max(self.maximum_momentum, abs(residual))
        self.maximum_energy = max(self.maximum_energy, abs(energy))

    def finish(self):
        if (self.failure is not None or self.count != DURATION_S*self.config.coupon.physics_hz
                or any(n != round(HOLD_TAIL_S*self.config.coupon.physics_hz) for n in self.tails.values())):
            self.failure = self.failure if self.failure is not None else "Incomplete acceptance protocol/hold tails"
            raise RuntimeError(self.failure)
        self.complete = True

    def report(self):
        return dict(complete=self.complete and self.failure is None, failure=self.failure,
            accepted_samples=self.count, hold_tail_samples=dict(self.tails),
            momentum_limit_n=MOMENTUM_LIMIT_N, energy_limit_j=ENERGY_LIMIT_J,
            intact_additional_energy_limit_j=INTACT_ENERGY_LIMIT_J,
            hold_tail_s=HOLD_TAIL_S, intact_opening_targets_m=[25e-6, 50e-6],
            intact_opening_relative_tolerance=.05,
            maximum_abs_accepted_momentum_residual_n=self.maximum_momentum,
            rms_accepted_momentum_residual_n=(self.momentum_squares/self.count)**.5 if self.count else None,
            maximum_abs_accepted_cumulative_energy_residual_j=self.maximum_energy,
            scope="single_uncalibrated_coupon_protocol_NOT_native_material_or_cut_validation")


@dataclass(frozen=True)
class ClosingConfig:
    coupon: tensile.CouponConfig
    case: str

    def __post_init__(self):
        if not isinstance(self.coupon, tensile.CouponConfig):
            raise ValueError("Explicit existing CouponConfig/material required")
        c = self.coupon
        if self.case not in ("intact", "damaged", "failed"):
            raise ValueError("Closing case must be intact, damaged or failed")
        if c.physics_hz not in (960, 1920) or c.shear_ratio != 0 or c.case != "softening":
            raise ValueError("Closing probe is pure normal, active damage, 960/1920 Hz only")
        if max(self.open_command_m, self.reopen_command_m, self.compression_command_m) > .001:
            raise ValueError("Closing protocol command outside 1 mm bound")

    @property
    def open_command_m(self):
        c, p = self.coupon, self.coupon.material
        if self.case == "failed":
            return 1.4*p.final_separation_m
        if self.case == "intact":
            opening = .25*p.initiation_separation_m
            force = c.area_m2*p.normal_stiffness_pa_m*opening
        else:
            opening = (p.initiation_separation_m+p.final_separation_m)/2
            force = c.area_m2*p.normal_strength_pa*(p.final_separation_m-opening)/(
                p.final_separation_m-p.initiation_separation_m)
        # Quasistatic INPUT estimate only. Actual opening/damage must be measured.
        return opening+force/c.carriage_stiffness_n_m

    @property
    def compression_command_m(self):
        return .25*self.coupon.material.initiation_separation_m

    @property
    def reopen_command_m(self):
        c = self.coupon
        return .5*c.material.initiation_separation_m*(
            1+c.area_m2*c.material.normal_stiffness_pa_m/c.carriage_stiffness_n_m)


def protocol(t, config):
    """11 s: settle .5, open 3, hold .5, close 3, hold 1, reopen 2, hold 1."""
    if isinstance(t, bool) or not np.isfinite(t) or not 0 <= t <= DURATION_S:
        raise ValueError("Finite protocol time in [0, 11] required")
    opened, closed, reopened = config.open_command_m, -config.compression_command_m, config.reopen_command_m
    if t < .5:
        return "settle", 0.
    if t < 3.5:
        return "opening", float(opened*(t-.5)/3)
    if t < 4.:
        return "open_hold", float(opened)
    if t < 7.:
        return "closing", float(opened+(closed-opened)*(t-4)/3)
    if t < 8.:
        return "compression_hold", float(closed)
    if t < 10.:
        return "reopening", float(closed+(reopened-closed)*(t-8)/2)
    return "reopen_hold", float(reopened)


class ClosingHistory:
    """Pure accepted-state adapter. No commanded travel argument or USD writes."""

    def __init__(self, config):
        self.config = config
        self.states = tuple(CohesiveState(config.coupon.material) for _ in range(4))
        self.last_jumps = np.zeros((4, 3))
        self.stored_energy_j = 0.

    def trial(self, jumps):
        c, p = self.config.coupon, self.config.coupon.material
        jumps = np.asarray(jumps, float)
        if jumps.shape != (4, 3) or not np.isfinite(jumps).all():
            raise ValueError("Four finite native facet jumps required")
        if np.max(np.linalg.norm(jumps, axis=1)) > .001:
            raise RuntimeError("1 mm material-anchor separation guard")
        increments = (jumps-self.last_jumps)*[1., p.shear_weight, p.shear_weight]
        if np.max(np.linalg.norm(increments, axis=1)) > c.maximum_jump_increment_r0*p.initiation_separation_m:
            raise RuntimeError("Material-jump increment guard")
        responses = tuple(cohesive_response(s, j, (1., 0., 0.), area_m2=c.area_m2/4)
                          for s, j in zip(self.states, jumps))
        if max(r.state.damage-s.damage for r, s in zip(responses, self.states)) > c.maximum_damage_increment:
            raise RuntimeError("Damage increment guard")
        if self.config.case == "intact" and any(r.state.damage > 0 for r in responses):
            raise RuntimeError("Intact control developed damage")
        forces = np.array([r.force_a_n for r in responses])
        if np.sum(np.linalg.norm(forces, axis=1)) > c.maximum_interface_force_n:
            raise RuntimeError("Constitutive interface force guard")
        retained = np.array([r.state.retained_stiffness for r in responses])
        stiffness = retained[:, None]*(c.area_m2/4)*[p.normal_stiffness_pa_m,
                                                  p.shear_stiffness_pa_m, p.shear_stiffness_pa_m]
        return dict(responses=responses, jumps=jumps.copy(), stiffness=stiffness, force_a=forces,
            stored=sum(r.stored_energy_j for r in responses),
            dissipated=sum(r.dissipated_energy_j for r in responses),
            constitutive_work=sum(r.constitutive_work_increment_j for r in responses),
            damage=np.array([r.state.damage for r in responses]),
            fully_separated=[r.state.fully_separated for r in responses])

    def commit(self, trial):
        self.states = tuple(r.state for r in trial["responses"])
        self.last_jumps = trial["jumps"].copy()
        self.stored_energy_j = trial["stored"]


def reconstructed_interface_force(stiffness, jumps):
    """Endpoint reconstruction, NOT native joint-force readback.

    Soft-limit activation at a within-step zero crossing still needs native
    validation. Keeping this separate exposes that discrepancy in momentum.
    """
    stiffness, jumps = np.asarray(stiffness, float), np.asarray(jumps, float)
    if (stiffness.shape != (4, 3) or jumps.shape != (4, 3)
            or not np.isfinite(stiffness).all() or not np.isfinite(jumps).all()
            or np.min(stiffness) < 0):
        raise ValueError("Finite nonnegative four-facet coefficients required")
    unilateral = jumps.copy()
    unilateral[:, 0] = np.maximum(unilateral[:, 0], 0.)
    return stiffness*unilateral


def author(stage, config):
    """Pure USD setup; helper conversion complete, fracture joints DISABLED."""
    from pxr import Gf, UsdPhysics
    from .cohesive_unilateral import configure_unilateral
    if not isinstance(config, ClosingConfig):
        raise ValueError("ClosingConfig required")
    fixture = tensile.author(stage, config.coupon, root=ROOT)
    # Helper's contract requires dynamic material on BOTH sides. The remote
    # laboratory support replaces kinematic A without changing geometry/mass.
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(fixture["paths"][0])).GetKinematicEnabledAttr().Set(False)
    support = UsdPhysics.FixedJoint.Define(stage, ROOT+"/RemoteWorldSupport")
    support.CreateBody1Rel().SetTargets([fixture["paths"][0]])
    support.CreateLocalPos0Attr(Gf.Vec3f(*fixture["frames"][0, :3, 3]))
    support.CreateLocalPos1Attr(Gf.Vec3f(0))
    support.CreateExcludeFromArticulationAttr(True)
    support.CreateCollisionEnabledAttr(True)
    support.CreateJointEnabledAttr(True)
    history = ClosingHistory(config)
    for joint, drives in zip(fixture["joints"], fixture["facet_drives"]):
        joint.GetJointEnabledAttr().Set(False)
        # Soft-limit API has no per-force cap. Use uncapped elastic constraints
        # and explicit measured/reconstructed guards, not silent force clipping.
        for drive in drives:
            drive.GetMaxForceAttr().Set(float("inf"))
    links = [configure_unilateral(joint, state=state, area_m2=config.coupon.area_m2/4)
             for joint, state in zip(fixture["joints"], history.states)]
    # The old fixture's transX Drive handles have been removed by the helper.
    fixture.pop("facet_drives")
    fixture.pop("bridge")
    fixture.update(closing_config=config, history=history, links=links, support=support)
    return fixture


def coefficient_snapshot(fixture):
    """Read actual USD coefficients applied to native constraints; JSON-safe."""
    from pxr import UsdPhysics
    from .cohesive_unilateral import unilateral_settings
    rows, stiffness = [], []
    for link in fixture["links"]:
        expected = unilateral_settings(link.state, link.area_m2)
        prim = link.joint.GetPrim()
        op = prim.GetMetadata("apiSchemas")
        names = set(prim.GetAppliedSchemas()) | (set(op.GetAppliedItems()) if op else set())
        limits = {n for n in names if n.startswith(("PhysicsLimitAPI:", "PhysxLimitAPI:"))}
        active = expected.normal_stiffness_n_m > 0
        if active:
            if link.limit is None or limits != {"PhysicsLimitAPI:transX", "PhysxLimitAPI:transX"}:
                raise RuntimeError("Missing active normal soft limit")
            low, high = link.limit.GetLowAttr().Get(), link.limit.GetHighAttr().Get()
            kn = float(link.soft["stiffness"].Get())
            if (low != float(np.float32(expected.normal_low_m)) or high != expected.normal_high_m
                    or not np.isfinite([low, high]).all() or low >= -.001 or high != 0. or kn <= 0
                    or any(link.soft[n].Get() != 0 for n in ("damping", "restitution", "bounceThreshold"))):
                raise RuntimeError("Unexpected/hard/unsafe normal limit configuration")
        else:
            if (link.limit is not None or link.soft or limits or any(n.startswith(
                    ("limit:transX:", "physxLimit:transX:")) for n in prim.GetPropertyNames())):
                raise RuntimeError("Failed normal limit schema/properties not removed")
            low, high, kn = None, None, 0.
        kt = [float(d.GetStiffnessAttr().Get()) for d in link.tangents]
        if (not np.isfinite([kn, *kt]).all() or min(kn, *kt) < 0
                or not np.allclose([kn, *kt], [expected.normal_stiffness_n_m,
                    expected.tangent_stiffness_n_m, expected.tangent_stiffness_n_m], rtol=2e-6, atol=0)
                or link.joint.GetPrim().HasAPI(UsdPhysics.DriveAPI, "transX")):
            raise RuntimeError("Unexpected/hard/parallel normal limit configuration")
        rows.append(dict(normal_low=low, normal_high=high,
                         normal_limit_present=active, normal_axis_free=not active,
                         normal_stiffness_n_m=kn, tangent_stiffness_n_m=kt,
                         normal_damping_ns_m=0., normal_restitution=0.))
        stiffness.append([kn, *kt])
    return rows, np.asarray(stiffness)


def activate_for_native(fixture):
    """Main's post-review setup step only; does not launch or step anything."""
    if not all(link.report()["native_schema_registered"] for link in fixture["links"]):
        raise RuntimeError("Native PhysxLimitAPI schema not registered; USD fallback is not native readiness")
    coefficient_snapshot(fixture)
    for joint in fixture["joints"]:
        joint.GetJointEnabledAttr().Set(True)


def _evidence(records, config):
    accepted = [r for r in records if r.get("accepted_step")]
    opened = [r for r in accepted if r["phase"] == "open_hold"]
    closed = [r for r in accepted if r["phase"] == "compression_hold"]
    reopened = [r for r in accepted if r["phase"] == "reopen_hold"]
    return dict(
        open_hold_observed=bool(opened),
        compression_contact_observed=any(r["native_contact_force_magnitude_n"] > 1e-6 for r in closed),
        reopened_positive_gap_observed=any(np.mean(np.asarray(r["measured_jumps_a_m"])[:, 0]) >
            .1*config.coupon.material.initiation_separation_m for r in reopened),
        maximum_damage=max((max(r["damage"]) for r in accepted), default=0.),
        all_facets_separated_observed=any(all(r["fully_separated"]) for r in accepted),
        native_response_qualified=False,
        note="Observed phase flags are not convergence, work, force or cutting qualification.")


def run(sim, fixture, *, native_errors=None):
    """Caller owns the initialized single native context and all launch authority."""
    from .blade_loading_probe import NativeContacts
    from .cohesive_unilateral import unilateral_settings
    from greenhouse_sim.physics_clock import PhysicsClock
    # Optional only for stub tests; absence can NEVER yield bounded acceptance.
    def check_errors(phase):
        if native_errors is not None:
            native_errors.check(phase)
    check_errors("run_entry")
    config, history = fixture["closing_config"], fixture["history"]
    c = config.coupon
    if not np.isclose(sim.get_physics_dt(), 1/c.physics_hz, rtol=1e-8, atol=1e-12):
        raise RuntimeError("Native timestep mismatch")
    if not all(j.GetJointEnabledAttr().Get() for j in fixture["joints"]):
        raise RuntimeError("Explicit reviewed activation is required")
    view = sim.physics_sim_view
    view.set_subspace_roots("/")
    bodies = view.create_rigid_body_view(ROOT+"/Bodies/*")
    if set(bodies.prim_paths) != set(fixture["paths"]):
        raise RuntimeError("Unexpected native body inventory")
    order = [list(bodies.prim_paths).index(p) for p in fixture["paths"]]
    masses = np.asarray(bodies.get_masses(), float)[order].reshape(2)
    inertias = np.asarray(bodies.get_inertias(), float)[order].reshape(2, 3, 3)
    expected_mass = c.density_kg_m3*np.prod(fixture["dimensions"])
    if (not np.isfinite(masses).all() or not np.allclose(masses, expected_mass, rtol=.005, atol=0)
            or not np.isfinite(inertias).all() or np.any(np.linalg.eigvalsh(inertias) <= 0)):
        raise RuntimeError("Native density/mass/inertia verification failed")
    # Unlike raw callback collider-order signs, this API documents force ON B.
    sensor = view.create_rigid_contact_view(fixture["paths"][1], filter_patterns=[fixture["paths"][0]])
    if sensor.sensor_count != 1:
        raise RuntimeError("Expected one signed native contact sensor on B")
    contacts = NativeContacts()
    contacts.subscribe()
    clock = PhysicsClock(sim, physics_hz=c.physics_hz, render_hz=0)
    records, fault = [], None
    acceptance = ProtocolAcceptance(config)
    previous_frames = pose_matrices(bodies.get_transforms()[order])
    previous_velocity = np.asarray(bodies.get_velocities(), float)[order].copy()
    previous_native_time = float(sim.current_time)
    initial_jumps, _ = tensile.measured_jumps(previous_frames, fixture["local"])
    current = history.trial(initial_jumps)
    check_errors("initial_readback_before_state_commit")
    history.commit(current)
    initial_stored = current["stored"]
    command, phase, used_rows, used_stiffness = 0., "settle", [], None
    cumulative = dict(constitutive_work_j=0., reconstructed_interface_absorbed_work_j=0.,
        reconstructed_carriage_work_j=0., native_contact_endpoint_work_j=0.,
        energy_balance_residual_j=0.)

    def before(stamp, dt):
        nonlocal command, phase, used_rows, used_stiffness
        check_errors("before_coefficient_writes")
        contacts.rows, contacts.friction = [], []
        # Preflight all scalar settings. No step occurs between these writes.
        # A's helper removes the normal schemas/properties on complete failure.
        for response in current["responses"]:
            unilateral_settings(response.state, c.area_m2/4)
        for link, response in zip(fixture["links"], current["responses"]):
            link.update(response.state)
        used_rows, used_stiffness = coefficient_snapshot(fixture)
        phase, command = protocol(stamp.simulation_time_s, config)
        fixture["loading"].GetTargetPositionAttr().Set(command)
        check_errors("after_coefficient_writes_before_solve")

    def after(stamp, dt):
        nonlocal current, previous_frames, previous_velocity, previous_native_time
        check_errors("after_step_before_readback")
        frames = pose_matrices(bodies.get_transforms()[order])
        velocity = np.asarray(bodies.get_velocities(), float)[order].copy()
        jumps, anchors = tensile.measured_jumps(frames, fixture["local"])
        native_time = float(sim.current_time)
        travel = float((frames[1, :3, 3]-fixture["frames"][1, :3, 3])@c.direction)
        speed = float(velocity[1, :3]@c.direction)
        contact_b = np.asarray(sensor.get_net_contact_forces(dt), float).reshape(1, 3)[0].copy()
        normal_magnitude = sum(np.linalg.norm(r["impulse"]) for r in contacts.rows)/dt
        friction_magnitude = sum(np.linalg.norm(r["impulse"]) for r in contacts.friction)/dt
        contact_magnitude = float(normal_magnitude+friction_magnitude)
        min_separation = min((r["separation_m"] for r in contacts.rows), default=0.)
        row = dict(t=stamp.simulation_time_s, dt=dt, native_context_time_s=native_time,
            phase=phase, command_travel_m=command, command_is_not_separation_evidence=True,
            native_frames=frames.tolist(), native_velocities=velocity.tolist(),
            native_anchors_world_m=anchors.tolist(), measured_jumps_a_m=jumps.tolist(),
            measured_carriage_travel_m=travel, measured_carriage_speed_m_s=speed,
            native_contacts=list(contacts.rows), native_friction_anchors=list(contacts.friction),
            native_contact_force_b_world_n=contact_b.tolist(),
            native_contact_force_magnitude_n=contact_magnitude,
            native_friction_force_magnitude_n=float(friction_magnitude),
            minimum_contact_separation_m=float(min_separation), used_unilateral_settings=used_rows,
            used_stiffness_n_m=used_stiffness.tolist(), actual_joint_forces_measured=False,
            accepted_step=False)
        records.append(row)
        if contacts.error is not None:
            raise RuntimeError(contacts.error)
        check_errors("after_readback_before_state_commit")
        if not np.isclose(native_time-previous_native_time, dt, rtol=1e-6, atol=1e-9):
            raise RuntimeError("Native context did not advance one timestep")
        if (not np.isfinite(np.r_[velocity.ravel(), contact_b, contact_magnitude, min_separation]).all()
                or np.max(np.linalg.norm(velocity[:, :3], axis=1)) > c.maximum_speed_m_s
                or np.max(np.linalg.norm(velocity[:, 3:], axis=1)) > .01
                or not np.allclose(frames[0], fixture["frames"][0], atol=1e-6, rtol=0)
                or not np.allclose(frames[1, :3, :3], np.eye(3), atol=1e-4, rtol=0)
                or abs(travel) > .001 or min_separation < -.001
                or contact_magnitude > c.maximum_contact_force_n
                or friction_magnitude > 1e-7
                or np.linalg.norm(contact_b) > contact_magnitude+1e-6):
            raise RuntimeError("Native pose/velocity/travel/contact/friction/readback guard")
        if not fixture["support"].GetJointEnabledAttr().Get() or not all(j.GetJointEnabledAttr().Get() for j in fixture["joints"]):
            raise RuntimeError("Unexpected support/interface joint disable")
        trial = history.trial(jumps)
        force_a = reconstructed_interface_force(used_stiffness, jumps)
        if np.max(np.abs(force_a)) >= .15 or np.sum(np.linalg.norm(force_a, axis=1)) > c.maximum_interface_force_n:
            raise RuntimeError("Reconstructed interface force guard")
        world_force_a = force_a@frames[0, :3, :3].T
        constitutive_world = trial["force_a"]@frames[0, :3, :3].T
        raw_actuator = c.carriage_stiffness_n_m*(command-travel)-c.carriage_damping_ns_m*speed
        actuator = float(np.clip(raw_actuator, -c.carriage_max_force_n, c.carriage_max_force_n))
        displacement_b = frames[1, :3, 3]-previous_frames[1, :3, 3]
        interface_work = float(np.sum(force_a*(jumps-history.last_jumps)))
        actuator_work = float(actuator*(displacement_b@c.direction))
        contact_work = float(contact_b@displacement_b)
        inertial = float(masses[1]*((velocity[1, :3]-previous_velocity[1, :3])@c.direction)/dt)
        residual = float(inertial-actuator+np.sum(world_force_a@c.direction)-contact_b@c.direction)

        def kinetic(f, v):
            omega = f[1, :3, :3].T@v[1, 3:]
            return float(.5*masses[1]*(v[1, :3]@v[1, :3])+.5*omega@inertias[1]@omega)

        ke = kinetic(frames, velocity)
        delta_ke = ke-kinetic(previous_frames, previous_velocity)
        balance = delta_ke+trial["constitutive_work"]-actuator_work-contact_work
        for key, value in (("constitutive_work_j", trial["constitutive_work"]),
                ("reconstructed_interface_absorbed_work_j", interface_work),
                ("reconstructed_carriage_work_j", actuator_work),
                ("native_contact_endpoint_work_j", contact_work), ("energy_balance_residual_j", balance)):
            cumulative[key] += value
        row.update(damage=trial["damage"].tolist(),
            fully_separated=trial["fully_separated"], next_constitutive_stiffness_n_m=trial["stiffness"].tolist(),
            reconstructed_interface_force_a_world_n=world_force_a.tolist(),
            reconstructed_interface_force_b_world_n=(-world_force_a).tolist(),
            reconstructed_normal_force_a_n=force_a[:, 0].tolist(),
            constitutive_force_a_world_n=constitutive_world.tolist(),
            drive_minus_constitutive_force_a_world_n=(world_force_a-constitutive_world).tolist(),
            normal_sign_crossing=((jumps[:, 0]>0) != (history.last_jumps[:, 0]>0)).tolist(),
            reconstructed_carriage_force_n=actuator, native_inertial_force_along_carriage_n=inertial,
            momentum_residual_with_native_contact_n=residual,
            stored_energy_j=trial["stored"], dissipated_energy_j=trial["dissipated"],
            native_state_kinetic_energy_j=ke, constitutive_work_increment_j=trial["constitutive_work"],
            reconstructed_interface_absorbed_work_increment_j=interface_work,
            reconstructed_carriage_work_increment_j=actuator_work,
            native_contact_endpoint_work_increment_j=contact_work,
            energy_balance_residual_increment_j=balance,
            interface_work_minus_constitutive_work_increment_j=interface_work-trial["constitutive_work"],
            cumulative=dict(cumulative))
        check_errors("computed_response_before_state_commit")
        acceptance.observe(row)
        row["accepted_step"] = True
        history.commit(trial)
        current = trial
        previous_frames, previous_velocity, previous_native_time = frames, velocity, native_time

    try:
        for _ in range(DURATION_S*c.physics_hz):
            clock.tick(before=before, after=after)
        check_errors("protocol_finished")
        acceptance.finish()
    except Exception as error:
        fault = str(error)
        if records:
            records[-1]["stop_reason"] = fault
    finally:
        fixture["loading"].GetMaxForceAttr().Set(0.)
        contacts.subscription = None
    return dict(state="closing_coupon_completed_unqualified" if fault is None else "closing_coupon_stopped_unqualified",
        error=fault, records=records, config=asdict(config), timing=clock.report(),
        bounded_protocol_accepted=bool(fault is None and acceptance.complete and native_errors is not None
            and not native_errors.report()["faulted"]
            and not native_errors.report().get("logging_interface_injected", True)),
        acceptance_contract=acceptance.report(),
        native_error_observer=native_errors.report() if native_errors is not None else None,
        native_error_gate="mandatory_caller_observer_checked_before_after_each_solve",
        material_status=ENGINEERING, protocol_duration_s=DURATION_S,
        native_masses_kg=masses.tolist(), native_inertias_kg_m2=inertias.tolist(),
        density_authored_kg_m3=c.density_kg_m3, native_dt_s=float(sim.get_physics_dt()),
        initial_constitutive_stored_energy_j=initial_stored, protocol_evidence=_evidence(records, config),
        remote_support="world_to_dynamic_A_fixed_joint_NOT_across_interface",
        normal_model="bounded_native_soft_limit_minus_2mm_to_zero_removed_on_failure",
        coefficient_update="accepted_post_fetch_states_applied_before_next_native_solve",
        force_readback="contact_on_B_native_tensor; interface_and_carriage_endpoint_reconstructions",
        contact_work="native_contact_force_times_measured_B_displacement_endpoint_quadrature",
        joint_contact_distance_setting="not_exposed_in_installed_PhysxLimitAPI",
        fully_implicit_constitutive_solve=False, constitutive_update_lag_steps=1,
        native_response_validated=False, timestep_convergence_verified=False,
        material_calibrated=False, physical_cut_verified=False, robot_grasp_verified=False,
        training_eligible=False)


def _native_session(config, output, native_errors):
    """Called ONLY inside the CLI-owned observer context; never on import."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from isaacsim.core.api import SimulationContext
    stage = omni.usd.get_context().get_stage()
    native_errors.check("before_authoring")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    scene_path = "/World/ClosingCouponPhysics"
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityMagnitudeAttr(0.)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    api.CreateSolverTypeAttr("PGS")
    api.CreateEnableGPUDynamicsAttr(False)
    api.CreateBroadphaseTypeAttr("MBP")
    api.CreateFrictionTypeAttr("patch")
    fixture = author(stage, config)
    native_errors.check("after_authoring_before_activation")
    activate_for_native(fixture)
    native_errors.check("after_activation_before_reset")
    stage.GetRootLayer().Export(str(output/"coupon_authored.usda"))
    sim = SimulationContext(physics_dt=1/config.coupon.physics_hz, rendering_dt=1/60,
        physics_prim_path=scene_path, stage_units_in_meters=1, backend="numpy", set_defaults=False)
    result = None
    try:
        sim.get_physics_context().enable_fabric(True)
        sim.reset()
        native_errors.check("after_reset")
        if (api.GetSolverTypeAttr().Get() != "PGS" or api.GetEnableGPUDynamicsAttr().Get()
                or scene.GetGravityMagnitudeAttr().Get() != 0.):
            raise RuntimeError("Closing coupon scene settings changed")
        result = run(sim, fixture, native_errors=native_errors)
        result["scene"] = dict(solver="PGS", gpu_dynamics=False, gravity_m_s2=0.,
            compression_contact="separate_frictionless_box_contacts", articulation=False)
    finally:
        try:
            sim.stop()
            native_errors.check("after_sim_stop")
        except Exception as error:
            if result is None:
                raise
            result["error"] = str(error)
            result["state"] = "closing_coupon_stopped_unqualified"
            result["bounded_protocol_accepted"] = False
    return result


def _write_trace(output, result):
    """Exclusive new output only; also preserve trace if observer exit fails."""
    with (output/"trace.jsonl").open("x", encoding="utf-8") as stream:
        for row in result.pop("records"):
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    result["trace_sha256"] = tensile._hash(output/"trace.jsonl")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", action="store_true", help="Main's explicit opt-in AFTER code/helper review")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("intact", "damaged", "failed"), required=True)
    parser.add_argument("--physics-hz", type=int, choices=(960, 1920), default=960)
    for name in ("kn-pa-m", "kt-pa-m", "strength-pa", "gc-j-m2"):
        parser.add_argument("--"+name, type=float, required=True)
    args = parser.parse_args(argv)
    if not args.native_run:
        parser.error("No native launch without main's --native-run after review")
    config = ClosingConfig(tensile.CouponConfig(
        CohesiveParameters(args.kn_pa_m, args.kt_pa_m, args.strength_pa, args.gc_j_m2),
        physics_hz=args.physics_hz), args.case)
    output = args.output.resolve()
    allowed = Path(__file__).resolve().parents[3]/"data"/"sim_physics"
    if output == allowed.resolve() or not output.is_relative_to(allowed.resolve()):
        raise ValueError("New child directory under repository data/sim_physics required")
    output.mkdir(parents=True, exist_ok=False)
    # Only main's explicit invocation reaches SimulationApp. Never called by tests.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False, "sync_loads": False})
    exit_code, observer, result, hashes = 2, None, None, {}
    try:
        from . import cohesive_unilateral, native_errors
        from greenhouse_sim import physics_clock
        dependencies = [Path(__file__), Path(tensile.__file__), Path(cohesive_unilateral.__file__),
            Path(__file__).with_name("cohesive.py"), Path(__file__).with_name("runtime.py"),
            Path(__file__).with_name("blade_loading_probe.py"), Path(physics_clock.__file__),
            Path(native_errors.__file__)]
        hashes = {str(p): tensile._hash(p) for p in dependencies}
        observer = native_errors.NativeErrors()  # Immediate registration, before any authoring.
        with observer:
            result = _native_session(config, output, observer)
            observer.check("before_observer_context_exit")
        # No success publication until context exit/removal checks have finished.
        result["native_error_observer"] = observer.report()
        result["native_error_observation_scope"] = (
            "registration_before_coupon_authoring_through_sim_stop_and_owned_logger_removal;"
            "not_app_startup_shutdown_or_flushed_async_coverage")
        result["bounded_protocol_accepted"] = bool(result["bounded_protocol_accepted"]
            and result["error"] is None and not observer.report()["faulted"]
            and observer.report()["closed"] and not observer.report()["logging_interface_injected"])
        result["implementation_sha256"] = hashes
        result["implementation_unchanged_during_run"] = all(tensile._hash(p) == h for p, h in hashes.items())
        if not result["implementation_unchanged_during_run"]:
            result["error"] = "Implementation changed during run; evidence unqualified"
            result["state"] = "closing_coupon_stopped_unqualified"
            result["bounded_protocol_accepted"] = False
        _write_trace(output, result)
        with (output/"report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(state=result["state"], error=result["error"], output=str(output))), flush=True)
        exit_code = 0 if result["error"] is None and result["bounded_protocol_accepted"] else 2
    except Exception as error:
        if result is not None:
            result["bounded_protocol_accepted"] = False
            result["state"] = "closing_coupon_stopped_unqualified"
            result["error"] = str(error)
            if "records" in result:
                _write_trace(output, result)
        with (output/"failure.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(state="closing_coupon_failed_unqualified", error=str(error),
                bounded_protocol_accepted=False,
                native_error_observer=observer.report() if observer is not None else None,
                implementation_sha256=hashes, protocol_result=result,
                traceback=traceback.format_exc(), physical_cut_verified=False,
                training_eligible=False), stream, indent=2)
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
