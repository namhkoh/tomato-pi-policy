"""Sever vine organs, and measure how well the cut was placed.

Cutting is implemented by disabling a joint, not by editing geometry. Isaac Sim
cannot change mesh topology at runtime, and deleting prims mid-simulation
desynchronises USD from PhysX and invalidates tensor views -- so the organ stays
in the scene and simply stops being held on.

Two severance modes are distinguished because they are agronomically different
outcomes, not two spellings of the same one:

* a **cut** is commanded, when a blade shears the petiole; the joint is released
  wherever the blade crossed.
* a **tear** happens on its own, when a pull exceeds what the junction can hold.

Tearing is detected by watching joint force, not by `physics:breakForce`. The
plant is an articulation -- it has to be, or its stiff light chain cannot be
integrated -- and breakForce is silently ignored inside one. Monitoring is
better anyway: breakForce fires on solver transients during settling and tore
every petiole off in the first frames, whereas a threshold applied to a measured
force can require the load to persist before it counts.

The blade plane is measured continuously along the centreline. PhysX cannot
detach an internal link from a live reduced-coordinate articulation, however,
so the runtime physical release uses the organ's pre-authored maximal-coordinate
base joint. The nearest geometric link joint and its quantisation are persisted
alongside that physical release rather than hidden; a future deformable/topology
backend can replace this explicit approximation without changing benchmark
cut evidence.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from greenhouse_sim import skeleton as skeleton_module
from greenhouse_sim import usd_env
from greenhouse_sim import vine_physics

usd_env.ensure_pxr()

from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

# Agronomic bins for residual petiole stub length. Pruning flush to the stem
# leaves a wound that resists Botrytis cinerea almost absolutely, while a stub
# is highly susceptible, with lesions advancing 3-5 mm per day.
FLUSH_STUB_M = 0.005
MARGINAL_STUB_M = 0.020


@dataclasses.dataclass(frozen=True)
class CutGateParameters:
    """Numerically stable conditions for a blade-mediated petiole cut."""

    minimum_forward_speed_m_s: float = 0.01
    maximum_axis_dot: float = math.sin(math.radians(35.0))
    cut_zone_before_m: float = 0.002
    cut_zone_after_m: float = 0.025
    radial_tolerance_m: float = 0.004
    minimum_cut_travel_m: float = 0.003
    force_cap_multiple: float = 3.0
    contact_memory_steps: int = 4


@dataclasses.dataclass(frozen=True)
class CutTarget:
    """A cylindrical petiole cut zone in world coordinates."""

    key: str
    organ_label: str
    centre_m: np.ndarray
    axis: np.ndarray
    radius_m: float
    cut_force_n: float

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre_m, dtype=np.float64)
        axis = np.asarray(self.axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if centre.shape != (3,) or axis.shape != (3,) or norm <= 0.0:
            raise ValueError("cut target centre and non-zero axis must be three-vectors")
        if self.radius_m <= 0.0 or self.cut_force_n <= 0.0:
            raise ValueError("cut target radius and force must be positive")
        object.__setattr__(self, "centre_m", centre)
        object.__setattr__(self, "axis", axis / norm)

    def required_work_j(self, parameters: CutGateParameters) -> float:
        """Work needed to carry a loaded edge through the petiole diameter."""
        travel = max(2.0 * self.radius_m, parameters.minimum_cut_travel_m)
        return float(self.cut_force_n * travel)


@dataclasses.dataclass(frozen=True)
class BladeContactSample:
    """One physics step of aggregated cutting-edge contact."""

    point_m: np.ndarray
    impulse_ns: np.ndarray
    edge_centre_m: np.ndarray
    edge_axis: np.ndarray
    cutting_direction: np.ndarray
    edge_velocity_m_s: np.ndarray
    dt_s: float
    counterhold_active: bool = False
    commanded_edge_velocity_m_s: np.ndarray | None = None


@dataclasses.dataclass
class CutProgress:
    """State accumulated only while contact satisfies the cut geometry."""

    work_j: float = 0.0
    peak_force_n: float = 0.0
    peak_speed_m_s: float = 0.0
    forward_travel_m: float = 0.0
    contact_steps: int = 0
    valid_contact_steps: int = 0
    gap_steps: int = 0
    minimum_signed_side_m: float = float("inf")
    maximum_signed_side_m: float = float("-inf")
    last_relative_edge_m: np.ndarray | None = None
    last_edge_centre_m: np.ndarray | None = None
    last_stub_m: float = 0.0
    edge_axis_alignment: float = 0.0
    motion_transverse_alignment: float = 0.0
    last_effective_force_n: float = 0.0
    last_forward_speed_m_s: float = 0.0
    last_contact_valid: bool = False
    virtual_penetration_m: float = 0.0
    counterheld_contact_steps: int = 0
    counterhold_start_side_m: float | None = None
    rejections: dict[str, int] = dataclasses.field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def reset_contact_window(self) -> None:
        """Discard disconnected bumps; physical cutting must be sustained."""
        self.work_j = 0.0
        self.forward_travel_m = 0.0
        self.valid_contact_steps = 0
        self.minimum_signed_side_m = float("inf")
        self.maximum_signed_side_m = float("-inf")
        self.last_relative_edge_m = None
        self.last_edge_centre_m = None
        self.virtual_penetration_m = 0.0
        self.counterheld_contact_steps = 0
        self.counterhold_start_side_m = None


@dataclasses.dataclass(frozen=True)
class PhysicalCutDecision:
    """Evidence that is persisted with a physically triggered severance."""

    target_key: str
    organ_label: str
    requested_stub_m: float
    cut_work_j: float
    required_work_j: float
    peak_force_n: float
    peak_speed_m_s: float
    forward_travel_m: float
    contact_steps: int
    valid_contact_steps: int
    edge_axis_alignment: float
    motion_transverse_alignment: float
    virtual_penetration_m: float
    counterheld_contact_steps: int


class DirectionalCutGate:
    """Accept only sustained, forceful leading-edge sweeps through a petiole."""

    def __init__(self, parameters: CutGateParameters | None = None) -> None:
        self.parameters = parameters or CutGateParameters()
        self._progress: dict[str, CutProgress] = {}
        self._completed: set[str] = set()

    def progress_for(self, target_key: str) -> CutProgress:
        return self._progress.setdefault(target_key, CutProgress())

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray | None:
        values = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        return values / norm if values.shape == (3,) and norm > 1e-12 else None

    def observe(
        self,
        target: CutTarget,
        sample: BladeContactSample,
    ) -> PhysicalCutDecision | None:
        """Consume one aggregated contact step and return a cut once qualified."""
        if target.key in self._completed:
            return None
        progress = self.progress_for(target.key)
        progress.contact_steps += 1
        progress.gap_steps = 0

        point = np.asarray(sample.point_m, dtype=np.float64)
        impulse = np.asarray(sample.impulse_ns, dtype=np.float64)
        edge_centre = np.asarray(sample.edge_centre_m, dtype=np.float64)
        velocity = np.asarray(sample.edge_velocity_m_s, dtype=np.float64)
        edge_axis = self._unit(sample.edge_axis)
        cut_direction = self._unit(sample.cutting_direction)
        commanded_velocity = (
            None
            if sample.commanded_edge_velocity_m_s is None
            else np.asarray(
                sample.commanded_edge_velocity_m_s,
                dtype=np.float64,
            )
        )
        motion_velocity = (
            commanded_velocity
            if sample.counterhold_active
            and commanded_velocity is not None
            and commanded_velocity.shape == (3,)
            else velocity
        )
        velocity_direction = self._unit(motion_velocity)
        if (
            point.shape != (3,)
            or impulse.shape != (3,)
            or edge_centre.shape != (3,)
            or velocity.shape != (3,)
            or edge_axis is None
            or cut_direction is None
            or velocity_direction is None
            or sample.dt_s <= 0.0
        ):
            progress.reject("invalid_sample")
            return None

        offset = point - target.centre_m
        stub_m = float(np.dot(offset, target.axis))
        radial = offset - stub_m * target.axis
        radial_distance = float(np.linalg.norm(radial))
        edge_axis_dot = abs(float(np.dot(edge_axis, target.axis)))
        motion_axis_dot = abs(float(np.dot(velocity_direction, target.axis)))
        forward_speed = float(np.dot(motion_velocity, cut_direction))
        effective_force = abs(float(np.dot(impulse, cut_direction))) / sample.dt_s
        relative_edge = edge_centre - target.centre_m

        progress.last_effective_force_n = effective_force
        progress.last_forward_speed_m_s = forward_speed

        progress.peak_force_n = max(progress.peak_force_n, effective_force)
        progress.peak_speed_m_s = max(progress.peak_speed_m_s, max(forward_speed, 0.0))
        progress.edge_axis_alignment = 1.0 - edge_axis_dot
        progress.motion_transverse_alignment = 1.0 - motion_axis_dot
        progress.last_stub_m = stub_m

        checks = (
            (
                -self.parameters.cut_zone_before_m
                <= stub_m
                <= self.parameters.cut_zone_after_m,
                "outside_axial_cut_zone",
            ),
            (
                radial_distance <= target.radius_m + self.parameters.radial_tolerance_m,
                "outside_radial_cut_zone",
            ),
            (edge_axis_dot <= self.parameters.maximum_axis_dot, "edge_not_transverse"),
            (motion_axis_dot <= self.parameters.maximum_axis_dot, "motion_not_transverse"),
            (
                forward_speed >= self.parameters.minimum_forward_speed_m_s,
                "wrong_or_slow_direction",
            ),
            (effective_force >= target.cut_force_n, "insufficient_force"),
        )
        failed = [reason for accepted, reason in checks if not accepted]
        progress.last_contact_valid = not failed
        if failed:
            for reason in failed:
                progress.reject(reason)
            progress.last_relative_edge_m = relative_edge.copy()
            progress.last_edge_centre_m = edge_centre.copy()
            return None

        relative_forward_step = 0.0
        if progress.last_relative_edge_m is not None:
            relative_forward_step = max(
                float(
                    np.dot(
                        relative_edge - progress.last_relative_edge_m,
                        cut_direction,
                    )
                ),
                0.0,
            )
        world_forward_step = 0.0
        if progress.last_edge_centre_m is not None:
            world_forward_step = max(
                float(
                    np.dot(
                        edge_centre - progress.last_edge_centre_m,
                        cut_direction,
                    )
                ),
                0.0,
            )
        commanded_forward_step = (
            max(forward_speed * sample.dt_s, 0.0)
            if sample.counterhold_active and commanded_velocity is not None
            else 0.0
        )
        progress.last_relative_edge_m = relative_edge.copy()
        progress.last_edge_centre_m = edge_centre.copy()
        progress.valid_contact_steps += 1
        forward_step = relative_forward_step
        if sample.counterhold_active:
            # A rigid capsule cannot yield under a sharp edge, so its contact
            # frame deflects with the knife even though real tissue would
            # fracture. Once an opposed physical grasp holds the same organ,
            # count forward world-edge travel as virtual penetration through
            # that rigid contact surface. Unheld pushing still uses only true
            # blade/target relative motion and remains non-qualifying.
            progress.counterheld_contact_steps += 1
            forward_step = max(
                forward_step,
                world_forward_step,
                commanded_forward_step,
            )
            progress.virtual_penetration_m += max(
                max(world_forward_step, commanded_forward_step)
                - relative_forward_step,
                0.0,
            )
        progress.forward_travel_m += forward_step
        capped_force = min(
            effective_force,
            target.cut_force_n * self.parameters.force_cap_multiple,
        )
        progress.work_j += capped_force * forward_step
        side = float(np.dot(relative_edge, cut_direction))
        progress.minimum_signed_side_m = min(progress.minimum_signed_side_m, side)
        progress.maximum_signed_side_m = max(progress.maximum_signed_side_m, side)
        if sample.counterhold_active and progress.counterhold_start_side_m is None:
            progress.counterhold_start_side_m = side

        physically_crossed_centre = (
            progress.minimum_signed_side_m <= -0.25 * target.radius_m
            and progress.maximum_signed_side_m >= 0.0
        )
        virtually_crossed_centre = bool(
            progress.counterhold_start_side_m is not None
            and progress.counterhold_start_side_m <= -0.25 * target.radius_m
            and progress.counterhold_start_side_m + progress.forward_travel_m >= 0.0
        )
        crossed_centre = physically_crossed_centre or virtually_crossed_centre
        required_work = target.required_work_j(self.parameters)
        if progress.work_j < required_work or not crossed_centre:
            return None

        self._completed.add(target.key)
        return PhysicalCutDecision(
            target_key=target.key,
            organ_label=target.organ_label,
            requested_stub_m=float(
                np.clip(stub_m, 0.0, self.parameters.cut_zone_after_m)
            ),
            cut_work_j=progress.work_j,
            required_work_j=required_work,
            peak_force_n=progress.peak_force_n,
            peak_speed_m_s=progress.peak_speed_m_s,
            forward_travel_m=progress.forward_travel_m,
            contact_steps=progress.contact_steps,
            valid_contact_steps=progress.valid_contact_steps,
            edge_axis_alignment=progress.edge_axis_alignment,
            motion_transverse_alignment=progress.motion_transverse_alignment,
            virtual_penetration_m=progress.virtual_penetration_m,
            counterheld_contact_steps=progress.counterheld_contact_steps,
        )

    def finish_step(self, contacted_target_keys: set[str]) -> None:
        """Advance contact gaps and reject damage assembled from separate taps."""
        for key, progress in self._progress.items():
            if key in contacted_target_keys or key in self._completed:
                continue
            progress.last_effective_force_n = 0.0
            progress.last_forward_speed_m_s = 0.0
            progress.last_contact_valid = False
            progress.gap_steps += 1
            if progress.gap_steps > self.parameters.contact_memory_steps:
                progress.reset_contact_window()

    def summary(self) -> dict[str, dict]:
        return {
            key: {
                "work_j": progress.work_j,
                "peak_force_n": progress.peak_force_n,
                "peak_speed_m_s": progress.peak_speed_m_s,
                "forward_travel_mm": progress.forward_travel_m * 1000.0,
                "contact_steps": progress.contact_steps,
                "valid_contact_steps": progress.valid_contact_steps,
                "last_effective_force_n": progress.last_effective_force_n,
                "last_forward_speed_m_s": progress.last_forward_speed_m_s,
                "last_contact_valid": progress.last_contact_valid,
                "last_stub_mm": progress.last_stub_m * 1000.0,
                "minimum_signed_side_mm": progress.minimum_signed_side_m * 1000.0,
                "maximum_signed_side_mm": progress.maximum_signed_side_m * 1000.0,
                "virtual_penetration_mm": progress.virtual_penetration_m * 1000.0,
                "counterheld_contact_steps": progress.counterheld_contact_steps,
                "rejections": dict(progress.rejections),
                "completed": key in self._completed,
            }
            for key, progress in self._progress.items()
        }


class CutError(Exception):
    """Raised when a severance cannot be applied."""


@dataclasses.dataclass(frozen=True)
class Cut:
    """The outcome of severing one organ."""

    organ_label: str
    joint_path: str
    torn: bool
    # Where the blade crossed, along the organ's centreline from its base.
    requested_stub_m: float
    # Where the joint that actually released sits, along the same centreline.
    realised_stub_m: float
    # Load at the junction when it let go; only meaningful for a tear.
    load_n: float = 0.0
    trigger: str = "debug_forced"
    cut_work_j: float = 0.0
    required_work_j: float = 0.0
    peak_force_n: float = 0.0
    peak_speed_m_s: float = 0.0
    forward_travel_m: float = 0.0
    contact_steps: int = 0
    valid_contact_steps: int = 0
    edge_axis_alignment: float = 0.0
    motion_transverse_alignment: float = 0.0
    virtual_penetration_m: float = 0.0
    counterheld_contact_steps: int = 0
    geometric_joint_path: str = ""
    geometric_stub_m: float = 0.0
    release_mode: str = "joint"

    @property
    def quantisation_error_m(self) -> float:
        """Gap between the commanded cut and the joint that could realise it."""
        return abs(self.realised_stub_m - self.requested_stub_m)

    @property
    def grade(self) -> str:
        """Agronomic quality of the resulting wound."""
        if self.requested_stub_m <= FLUSH_STUB_M:
            return "flush"
        if self.requested_stub_m <= MARGINAL_STUB_M:
            return "marginal"
        return "stub"


class Severer:
    """Applies and records severances on a rigged plant."""

    def __init__(
        self,
        stage: Usd.Stage,
        rig: vine_physics.PlantRig,
        skeletons: dict[int, skeleton_module.Skeleton],
        organ_indices: dict[str, int],
    ) -> None:
        self._stage = stage
        self._rig = rig
        self._skeletons = skeletons
        self._organ_indices = organ_indices
        self._cuts: list[Cut] = []
        self._sustained: dict[str, int] = {}
        # Captured before the simulation runs, while the plant is in its
        # authored rest pose, so joint deflection is measured from zero.
        self._rest_rotations: dict[str, np.ndarray] = {}
        for label, junction in rig.junctions.items():
            rotation = self._relative_rotation(junction)
            if rotation is not None:
                self._rest_rotations[label] = rotation

    @property
    def cuts(self) -> list[Cut]:
        return list(self._cuts)

    def severable(self) -> list[str]:
        """Organ labels that can still be cut."""
        return [label for label in self._rig.cut_joints if not self._is_severed(label)]

    def cut(
        self,
        organ_label: str,
        stub_length_m: float = 0.0,
        *,
        trigger: str = "debug_forced",
        decision: PhysicalCutDecision | None = None,
    ) -> Cut:
        """Sever `organ_label`, leaving `stub_length_m` attached to the parent.

        A stub of zero is the agronomically correct flush cut at the junction.
        Benchmark cuts pass a :class:`PhysicalCutDecision`; calls without one
        are retained only as an explicit debug/acceptance shortcut.
        """
        if organ_label not in self._rig.cut_joints:
            raise CutError(f"{organ_label} has no severable joint")
        if self._is_severed(organ_label):
            raise CutError(f"{organ_label} is already severed")
        if decision is not None:
            if decision.organ_label != organ_label:
                raise CutError(
                    f"physical cut decision is for {decision.organ_label}, not {organ_label}"
                )
            stub_length_m = decision.requested_stub_m
            trigger = "physical_blade"

        geometric_joint_path, geometric_stub_m = self._joint_for_stub(
            organ_label,
            stub_length_m,
        )
        base_joint = self._rig.cut_joints[organ_label]
        # Internal joints belong to one reduced-coordinate articulation.
        # Changing jointEnabled updates USD but does not split the live PhysX
        # articulation, so a held downstream link remains tethered. The base
        # joint was deliberately authored outside the articulation and is the
        # runtime-detachable physical boundary.
        internal_articulation_release = geometric_joint_path != base_joint
        joint_path = base_joint if internal_articulation_release else geometric_joint_path
        realised = 0.0 if internal_articulation_release else geometric_stub_m
        self._set_joint_enabled(joint_path, enabled=False)

        record = Cut(
            organ_label=organ_label,
            joint_path=joint_path,
            torn=False,
            requested_stub_m=float(max(stub_length_m, 0.0)),
            realised_stub_m=realised,
            trigger=trigger,
            cut_work_j=decision.cut_work_j if decision is not None else 0.0,
            required_work_j=decision.required_work_j if decision is not None else 0.0,
            peak_force_n=decision.peak_force_n if decision is not None else 0.0,
            peak_speed_m_s=decision.peak_speed_m_s if decision is not None else 0.0,
            forward_travel_m=decision.forward_travel_m if decision is not None else 0.0,
            contact_steps=decision.contact_steps if decision is not None else 0,
            valid_contact_steps=decision.valid_contact_steps if decision is not None else 0,
            edge_axis_alignment=(
                decision.edge_axis_alignment if decision is not None else 0.0
            ),
            motion_transverse_alignment=(
                decision.motion_transverse_alignment if decision is not None else 0.0
            ),
            virtual_penetration_m=(
                decision.virtual_penetration_m if decision is not None else 0.0
            ),
            counterheld_contact_steps=(
                decision.counterheld_contact_steps if decision is not None else 0
            ),
            geometric_joint_path=geometric_joint_path,
            geometric_stub_m=geometric_stub_m,
            release_mode=(
                "maximal_coordinate_organ_base"
                if internal_articulation_release
                else "selected_joint"
            ),
        )
        self._cuts.append(record)
        return record

    def poll_tears(self, *, sustain_steps: int = 3) -> list[Cut]:
        """Sever organs whose junction is being pulled past what it can hold.

        Call once per simulation step. A junction must exceed its threshold for
        `sustain_steps` consecutive steps before it lets go, which stops a
        single solver spike from stripping the plant -- the exact failure mode
        that made `physics:breakForce` unusable here.
        """
        torn = []
        for label, junction in self._rig.junctions.items():
            if junction.tear_force_n <= 0.0 or self._is_severed(label):
                self._sustained.pop(label, None)
                continue

            force = self._junction_load(label, junction)
            if force is None or force < junction.tear_force_n:
                self._sustained[label] = 0
                continue

            self._sustained[label] = self._sustained.get(label, 0) + 1
            if self._sustained[label] < sustain_steps:
                continue

            self._set_joint_enabled(junction.joint_path, enabled=False)
            record = Cut(
                organ_label=label,
                joint_path=junction.joint_path,
                torn=True,
                requested_stub_m=0.0,
                realised_stub_m=0.0,
                load_n=force,
                trigger="tear",
            )
            self._cuts.append(record)
            torn.append(record)
        return torn

    def _junction_load(self, label: str, junction) -> float | None:
        """Force at a junction, from how far its joint is bent.

        The joint's angular drive is authored with a known stiffness, so its
        restoring torque is stiffness times deflection, and the equivalent force
        is that torque over the organ's lever arm. Reading it from transforms
        keeps this independent of physics-tensor internals, which are invalidated
        by the very scene edits severing performs.
        """
        rest = self._rest_rotations.get(label)
        if rest is None:
            return None
        current = self._relative_rotation(junction)
        if current is None:
            return None

        # Angle between the rest and current relative orientations.
        delta = current @ rest.T
        cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        deflection = float(np.arccos(cosine))
        torque = junction.stiffness_nm_per_rad * deflection
        return torque / max(junction.lever_m, 1e-4)

    def _relative_rotation(self, junction) -> np.ndarray | None:
        """Child orientation expressed in the parent's frame."""
        parent = self._rotation_of(junction.parent_path) if junction.parent_path else np.eye(3)
        child = self._rotation_of(junction.child_path)
        if parent is None or child is None:
            return None
        return parent.T @ child

    def _rotation_of(self, path: str) -> np.ndarray | None:
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows = np.array([[matrix[r][c] for c in range(3)] for r in range(3)])
        # Strip any scale so the result is a pure rotation.
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(norms > 0, norms, 1.0)

    def _is_severed(self, organ_label: str) -> bool:
        return any(cut.organ_label == organ_label for cut in self._cuts)

    def _joint_for_stub(self, organ_label: str, stub_length_m: float) -> tuple[str, float]:
        """Joint closest to the requested cut position, and where it truly sits."""
        base_joint = self._rig.cut_joints[organ_label]
        if stub_length_m <= 0.0:
            return base_joint, 0.0

        organ_index = self._organ_indices.get(organ_label)
        centreline = self._skeletons.get(organ_index) if organ_index is not None else None
        if centreline is None:
            return base_joint, 0.0

        # Internal joints sit at centreline nodes 1..n, so node k is k segments
        # of arc length from the organ's base.
        arcs = centreline.arc_lengths()
        prefix = f"{base_joint.rsplit('/', 1)[0]}/Joint_"
        candidates: list[tuple[float, str, float]] = [(arcs[0], base_joint, 0.0)]
        for node in range(1, len(arcs)):
            path = f"{prefix}{node:03d}"
            if self._stage.GetPrimAtPath(path).IsValid():
                candidates.append((abs(arcs[node] - stub_length_m), path, float(arcs[node])))
        candidates[0] = (abs(arcs[0] - stub_length_m), base_joint, 0.0)

        _, path, realised = min(candidates, key=lambda item: item[0])
        return path, realised

    def _joint_prim(self, joint_path: str) -> Usd.Prim:
        prim = self._stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            raise CutError(f"joint {joint_path} is not on the stage")
        return prim

    def _set_joint_enabled(self, joint_path: str, *, enabled: bool) -> None:
        # The attribute is authored when the rig is built, so this only changes
        # a value; creating it here would force a full PhysX resync instead.
        attribute = UsdPhysics.Joint(self._joint_prim(joint_path)).GetJointEnabledAttr()
        if not attribute or not attribute.IsValid():
            raise CutError(f"joint {joint_path} has no jointEnabled attribute; was the rig authored by vine_physics?")
        attribute.Set(enabled)

    def _joint_enabled(self, joint_path: str) -> bool:
        attribute = UsdPhysics.Joint(self._joint_prim(joint_path)).GetJointEnabledAttr()
        value = attribute.Get() if attribute and attribute.IsValid() else None
        return True if value is None else bool(value)


def summarise(cuts: list[Cut]) -> dict[str, float]:
    """Aggregate cut quality the way the benchmark reports it."""
    if not cuts:
        return {"count": 0}
    stubs = np.array([cut.requested_stub_m for cut in cuts])
    return {
        "count": float(len(cuts)),
        "torn": float(sum(1 for cut in cuts if cut.torn)),
        "median_stub_mm": float(np.median(stubs) * 1000.0),
        "p90_stub_mm": float(np.percentile(stubs, 90) * 1000.0),
        "flush_fraction": float(np.mean(stubs <= FLUSH_STUB_M)),
        "marginal_fraction": float(np.mean((stubs > FLUSH_STUB_M) & (stubs <= MARGINAL_STUB_M))),
        "stub_fraction": float(np.mean(stubs > MARGINAL_STUB_M)),
        "max_quantisation_mm": float(max(cut.quantisation_error_m for cut in cuts) * 1000.0),
    }
