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

Physics releases at the nearest joint, but the score is measured against the
blade plane's true position along the centreline. Those differ by up to half a
segment, and that residual is reported per cut rather than hidden, because
residual stub length is the benchmark's headline metric and an error term it
cannot see would quietly flatter every policy.
"""

from __future__ import annotations

import dataclasses

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

    def cut(self, organ_label: str, stub_length_m: float = 0.0) -> Cut:
        """Sever `organ_label`, leaving `stub_length_m` attached to the parent.

        A stub of zero is the agronomically correct flush cut at the junction.
        """
        if organ_label not in self._rig.cut_joints:
            raise CutError(f"{organ_label} has no severable joint")
        if self._is_severed(organ_label):
            raise CutError(f"{organ_label} is already severed")

        joint_path, realised = self._joint_for_stub(organ_label, stub_length_m)
        self._set_joint_enabled(joint_path, enabled=False)

        record = Cut(
            organ_label=organ_label,
            joint_path=joint_path,
            torn=False,
            requested_stub_m=float(max(stub_length_m, 0.0)),
            realised_stub_m=realised,
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
