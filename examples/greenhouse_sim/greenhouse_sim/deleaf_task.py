"""Benchmark state for the required bi-manual deleafing sequence."""

from __future__ import annotations

import dataclasses
import enum

import numpy as np


class Phase(enum.Enum):
    SEEK_GRASP = "seek_grasp"
    GRASPED = "grasped"
    ORPHAN_RETAINED = "orphan_retained"
    TRANSPORTED = "transported"
    RELEASED = "released"
    DEPOSITED = "deposited"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class TaskParameters:
    minimum_grasp_force_n: float = 1.0
    required_grasp_steps: int = 3
    minimum_transport_clearance_m: float = 0.15
    maximum_drop_speed_m_s: float = 0.15
    floor_tolerance_m: float = 0.05
    drop_zone_min_m: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    drop_zone_max_m: tuple[float, float, float] = (1.0, 1.0, 1.0)


class BimanualDeleafTask:
    """Require left grasp -> right physical cut -> retain -> floor deposit."""

    def __init__(
        self,
        target_vine: str,
        target_organ: str,
        parameters: TaskParameters | None = None,
    ) -> None:
        self.target_vine = target_vine
        self.target_organ = target_organ
        self.parameters = parameters or TaskParameters()
        self.phase = Phase.SEEK_GRASP
        self.step_index = 0
        self.grasp_body: str | None = None
        self.grasp_active = False
        self._grasp_steps = 0
        self.maximum_grasp_force_n = 0.0
        self.maximum_transport_clearance_m = 0.0
        self.failures: list[dict] = []
        self.events: list[dict] = []

    @property
    def target_key(self) -> str:
        return f"{self.target_vine}/{self.target_organ}"

    @property
    def succeeded(self) -> bool:
        return self.phase is Phase.DEPOSITED

    def _event(self, name: str, **values) -> None:
        self.events.append({"step": self.step_index, "event": name, **values})

    def fail(self, reason: str, **values) -> None:
        self.failures.append({"step": self.step_index, "reason": reason, **values})
        self.phase = Phase.FAILED

    def advance(self) -> None:
        self.step_index += 1

    def observe_grasp(
        self,
        *,
        vine: str,
        organ: str,
        body_path: str,
        finger_contacts: set[str],
        force_n: float,
    ) -> bool:
        """Accept a grasp only after both left fingers load the target branch."""
        if self.phase is Phase.FAILED:
            return False
        if vine != self.target_vine or organ != self.target_organ:
            self._grasp_steps = 0
            return False
        opposing = {"left_finger_1", "left_finger_2"}.issubset(finger_contacts)
        loaded = force_n >= self.parameters.minimum_grasp_force_n
        if not opposing or not loaded:
            self._grasp_steps = 0
            return False

        self._grasp_steps += 1
        self.maximum_grasp_force_n = max(self.maximum_grasp_force_n, force_n)
        if self._grasp_steps < self.parameters.required_grasp_steps:
            return False
        if self.phase is Phase.SEEK_GRASP:
            self.phase = Phase.GRASPED
            self.grasp_body = body_path
            self.grasp_active = True
            self._event(
                "left_grasp_established",
                body_path=body_path,
                force_n=force_n,
            )
        return self.phase is Phase.GRASPED

    def observe_cut(
        self,
        *,
        vine: str,
        organ: str,
        physical_blade: bool,
        intended_target: bool,
        safe_path: bool = True,
    ) -> bool:
        """Record the real severance while enforcing grasp-before-cut order."""
        if self.phase is Phase.FAILED:
            return False
        if not physical_blade:
            self.fail("debug_or_nonphysical_cut")
            return False
        if vine != self.target_vine or organ != self.target_organ or not intended_target:
            self.fail("unintended_organ_cut", vine=vine, organ=organ)
            return False
        if not safe_path:
            self.fail("protected_contact_before_cut")
            return False
        if self.phase is not Phase.GRASPED or not self.grasp_active:
            self.fail("cut_before_left_grasp")
            return False
        self.phase = Phase.ORPHAN_RETAINED
        self._event("right_blade_severed_target")
        return True

    def observe_hold(self, *, grasp_active: bool) -> None:
        if self.phase in {Phase.FAILED, Phase.RELEASED, Phase.DEPOSITED}:
            return
        self.grasp_active = bool(grasp_active)
        if self.phase in {Phase.ORPHAN_RETAINED, Phase.TRANSPORTED} and not grasp_active:
            self.fail("orphan_dropped_before_commanded_release")

    def observe_transport(self, clearance_m: float) -> bool:
        if self.phase is Phase.FAILED:
            return False
        self.maximum_transport_clearance_m = max(
            self.maximum_transport_clearance_m,
            float(clearance_m),
        )
        if (
            self.phase is Phase.ORPHAN_RETAINED
            and self.grasp_active
            and clearance_m >= self.parameters.minimum_transport_clearance_m
        ):
            self.phase = Phase.TRANSPORTED
            self._event("orphan_cleared_vine_row", clearance_m=float(clearance_m))
        return self.phase is Phase.TRANSPORTED

    def observe_release(self) -> bool:
        if self.phase is Phase.FAILED:
            return False
        if self.phase is not Phase.TRANSPORTED or not self.grasp_active:
            self.fail("release_before_safe_transport")
            return False
        self.grasp_active = False
        self.phase = Phase.RELEASED
        self._event("left_gripper_released_orphan")
        return True

    def observe_deposit(
        self,
        *,
        centroid_m: np.ndarray,
        speed_m_s: float,
        floor_contact: bool,
        lowest_height_m: float | None = None,
    ) -> bool:
        if self.phase is not Phase.RELEASED:
            return False
        centroid = np.asarray(centroid_m, dtype=np.float64)
        lower = np.asarray(self.parameters.drop_zone_min_m, dtype=np.float64)
        upper = np.asarray(self.parameters.drop_zone_max_m, dtype=np.float64)
        inside_xy = bool(np.all(centroid[:2] >= lower[:2]) and np.all(centroid[:2] <= upper[:2]))
        lowest = centroid[2] if lowest_height_m is None else float(lowest_height_m)
        near_floor = bool(lowest <= lower[2] + self.parameters.floor_tolerance_m)
        settled = speed_m_s <= self.parameters.maximum_drop_speed_m_s
        if inside_xy and near_floor and settled and floor_contact:
            self.phase = Phase.DEPOSITED
            self._event(
                "orphan_deposited_on_floor",
                centroid_m=centroid.tolist(),
                lowest_height_m=lowest,
                speed_m_s=float(speed_m_s),
            )
            return True
        return False

    @property
    def summary(self) -> dict:
        return {
            "target": self.target_key,
            "phase": self.phase.value,
            "succeeded": self.succeeded,
            "grasp_active": self.grasp_active,
            "grasp_body": self.grasp_body,
            "maximum_grasp_force_n": self.maximum_grasp_force_n,
            "maximum_transport_clearance_m": self.maximum_transport_clearance_m,
            "parameters": dataclasses.asdict(self.parameters),
            "events": list(self.events),
            "failures": list(self.failures),
        }
