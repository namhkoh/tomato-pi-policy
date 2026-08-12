"""Framework-neutral online RL contract for the deleafing benchmark.

The Isaac process owns physics and implements :class:`DeleafRuntime`.  This
module deliberately depends only on NumPy so it can be tested without Kit and
used from Isaac Sim's bundled Python.  A trainer may wrap ``OnlineDeleafEnv``
with Gymnasium, but Gymnasium is not a simulator dependency.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

import numpy as np


ACTION_SIZE = 15
LEFT_ARM_ACTION_SLICE = slice(0, 7)
RIGHT_ARM_ACTION_SLICE = slice(7, 14)
GRIPPER_ACTION_INDEX = 14
LEFT_GRASP_DELTA_SLICE = slice(29, 32)
PHASE_OBSERVATION_SLICE = slice(44, 51)
DEFAULT_GRIPPER_ACTIVATION_DISTANCE_M = 0.05
PHASES = (
    "seek_grasp",
    "grasped",
    "orphan_retained",
    "transported",
    "released",
    "deposited",
    "failed",
)
OBSERVATION_SIZE = 56


@dataclasses.dataclass(frozen=True)
class ActionParameters:
    """Convert normalized policy actions to bounded simulator commands."""

    maximum_arm_speed_degrees_s: float = 35.0
    maximum_arm_acceleration_degrees_s2: float = 120.0
    maximum_gripper_speed_per_s: float = 2.0
    maximum_gripper_acceleration_per_s2: float = 8.0
    control_hz: float = 20.0
    physics_hz: float = 240.0

    def __post_init__(self) -> None:
        values = (
            self.maximum_arm_speed_degrees_s,
            self.maximum_arm_acceleration_degrees_s2,
            self.maximum_gripper_speed_per_s,
            self.maximum_gripper_acceleration_per_s2,
            self.control_hz,
            self.physics_hz,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("RL action rates must be finite and positive")
        ratio = self.physics_hz / self.control_hz
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("physics_hz must be an integer multiple of control_hz")

    @property
    def physics_steps_per_action(self) -> int:
        return int(round(self.physics_hz / self.control_hz))


@dataclasses.dataclass(frozen=True)
class RewardParameters:
    """Dense shaping with event bonuses from the strict benchmark state."""

    time_penalty: float = 0.01
    action_penalty: float = 0.002
    action_delta_penalty: float = 0.01
    reach_scale_m: float = 0.12
    grasp_close_scale: float = 0.5
    grasp_close_reach_scale_m: float = 0.04
    grasp_bonus: float = 10.0
    cut_bonus: float = 30.0
    transported_bonus: float = 15.0
    released_bonus: float = 5.0
    deposited_bonus: float = 50.0
    failure_penalty: float = 30.0
    unsafe_contact_penalty: float = 50.0
    cut_progress_scale: float = 8.0


@dataclasses.dataclass(frozen=True)
class DeleafState:
    """One policy observation derived from live benchmark physics."""

    left_joint_position: np.ndarray
    right_joint_position: np.ndarray
    left_joint_velocity: np.ndarray
    right_joint_velocity: np.ndarray
    gripper_openness: float
    left_grasp_delta_m: np.ndarray
    blade_cut_delta_m: np.ndarray
    target_axis: np.ndarray
    blade_edge_axis: np.ndarray
    blade_cut_direction: np.ndarray
    phase: str
    grasp_force_fraction: float
    cut_force_fraction: float
    cut_work_fraction: float
    transport_fraction: float
    unsafe_contact_count: int = 0
    safety_clear: bool = True
    target_key: str = ""

    def __post_init__(self) -> None:
        vector_fields = {
            "left_joint_position": (self.left_joint_position, 7),
            "right_joint_position": (self.right_joint_position, 7),
            "left_joint_velocity": (self.left_joint_velocity, 7),
            "right_joint_velocity": (self.right_joint_velocity, 7),
            "left_grasp_delta_m": (self.left_grasp_delta_m, 3),
            "blade_cut_delta_m": (self.blade_cut_delta_m, 3),
            "target_axis": (self.target_axis, 3),
            "blade_edge_axis": (self.blade_edge_axis, 3),
            "blade_cut_direction": (self.blade_cut_direction, 3),
        }
        for name, (value, size) in vector_fields.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite {size}-vector")
            object.__setattr__(self, name, array)
        scalars = (
            self.gripper_openness,
            self.grasp_force_fraction,
            self.cut_force_fraction,
            self.cut_work_fraction,
            self.transport_fraction,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("RL state scalars must be finite")
        if self.phase not in PHASES:
            raise ValueError(f"unknown deleafing phase: {self.phase!r}")
        if self.unsafe_contact_count < 0:
            raise ValueError("unsafe_contact_count cannot be negative")

    def vector(self) -> np.ndarray:
        """Return the stable 56-value low-dimensional observation."""

        phase = np.zeros(len(PHASES), dtype=np.float32)
        phase[PHASES.index(self.phase)] = 1.0
        vector = np.concatenate(
            (
                self.left_joint_position,
                self.right_joint_position,
                self.left_joint_velocity,
                self.right_joint_velocity,
                np.asarray([self.gripper_openness], dtype=np.float32),
                self.left_grasp_delta_m,
                self.blade_cut_delta_m,
                self.target_axis,
                self.blade_edge_axis,
                self.blade_cut_direction,
                phase,
                np.asarray(
                    [
                        np.clip(self.grasp_force_fraction, 0.0, 3.0),
                        np.clip(self.cut_force_fraction, 0.0, 3.0),
                        np.clip(self.cut_work_fraction, 0.0, 2.0),
                        np.clip(self.transport_fraction, 0.0, 2.0),
                        min(float(self.unsafe_contact_count), 10.0) / 10.0,
                    ],
                    dtype=np.float32,
                ),
            )
        )
        if vector.shape != (OBSERVATION_SIZE,):
            raise AssertionError(f"unexpected RL observation shape: {vector.shape}")
        return vector

    def info(self) -> dict:
        return {
            "target": self.target_key,
            "phase": self.phase,
            "safety_clear": bool(self.safety_clear),
            "unsafe_contact_count": int(self.unsafe_contact_count),
            "grasp_force_fraction": float(self.grasp_force_fraction),
            "cut_force_fraction": float(self.cut_force_fraction),
            "cut_work_fraction": float(self.cut_work_fraction),
            "transport_fraction": float(self.transport_fraction),
            "left_grasp_distance_m": float(np.linalg.norm(self.left_grasp_delta_m)),
            "blade_cut_distance_m": float(np.linalg.norm(self.blade_cut_delta_m)),
        }


def phase_action_mask(
    observation,
    *,
    gripper_activation_distance_m: float = DEFAULT_GRIPPER_ACTIVATION_DISTANCE_M,
) -> np.ndarray:
    """Return deterministic task-order masks for one or batched observations.

    The cutter arm cannot move before a strict grasp exists. The gripper also
    stays open while its jaw centre is outside the local grasp neighbourhood;
    this prevents the policy from collecting closure shaping at a distance.
    """

    values = np.asarray(observation, dtype=np.float32)
    if values.shape[-1:] != (OBSERVATION_SIZE,) or not np.isfinite(values).all():
        raise ValueError(
            f"observation must end in a finite {OBSERVATION_SIZE}-vector"
        )
    if (
        not math.isfinite(float(gripper_activation_distance_m))
        or gripper_activation_distance_m <= 0.0
    ):
        raise ValueError("gripper activation distance must be finite and positive")
    mask = np.ones(values.shape[:-1] + (ACTION_SIZE,), dtype=np.float32)
    phases = values[..., PHASE_OBSERVATION_SLICE]
    seek_grasp = phases[..., PHASES.index("seek_grasp")] > 0.5
    mask[..., RIGHT_ARM_ACTION_SLICE] = np.where(
        np.expand_dims(seek_grasp, axis=-1), 0.0, 1.0
    )
    grasp_distance = np.linalg.norm(values[..., LEFT_GRASP_DELTA_SLICE], axis=-1)
    enable_gripper = np.logical_or(
        ~seek_grasp, grasp_distance <= gripper_activation_distance_m
    )
    mask[..., GRIPPER_ACTION_INDEX] = enable_gripper.astype(np.float32)
    return mask


class DeleafRuntime(Protocol):
    """Physics adapter implemented by the running Isaac greenhouse."""

    def reset(self, *, seed: int) -> DeleafState:
        """Restore one episode and return its first measured state."""

    def apply_action(
        self,
        action: np.ndarray,
        parameters: ActionParameters,
    ) -> DeleafState:
        """Apply one normalized action and advance the configured substeps."""


class OnlineDeleafEnv:
    """Gym-style single-environment API around strict physical task events."""

    metadata = {"render_modes": ("human", "rgb_array"), "render_fps": 20}

    def __init__(
        self,
        runtime: DeleafRuntime,
        *,
        action_parameters: ActionParameters | None = None,
        reward_parameters: RewardParameters | None = None,
        maximum_episode_steps: int = 1200,
        terminal_phase: str | None = None,
    ) -> None:
        if maximum_episode_steps < 1:
            raise ValueError("maximum_episode_steps must be positive")
        if terminal_phase is not None and terminal_phase not in PHASES[1:-1]:
            raise ValueError("terminal_phase must be a reachable non-failure task phase")
        self.runtime = runtime
        self.action_parameters = action_parameters or ActionParameters()
        self.reward_parameters = reward_parameters or RewardParameters()
        self.maximum_episode_steps = int(maximum_episode_steps)
        self.terminal_phase = terminal_phase
        self._steps = 0
        self._state: DeleafState | None = None
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (ACTION_SIZE,)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (OBSERVATION_SIZE,)

    def reset(self, *, seed: int = 0) -> tuple[np.ndarray, dict]:
        self._steps = 0
        self._previous_action.fill(0.0)
        self._state = self.runtime.reset(seed=int(seed))
        info = self._state.info()
        info.update(episode_seed=int(seed), reset=True)
        return self._state.vector(), info

    def _potential(self, state: DeleafState) -> float:
        params = self.reward_parameters
        left_distance = float(np.linalg.norm(state.left_grasp_delta_m))
        blade_distance = float(np.linalg.norm(state.blade_cut_delta_m))
        left_reach = math.exp(-left_distance / params.reach_scale_m)
        blade_reach = math.exp(-blade_distance / params.reach_scale_m)
        if state.phase == "seek_grasp":
            near_grasp = math.exp(
                -left_distance / params.grasp_close_reach_scale_m
            )
            closing = 1.0 - float(np.clip(state.gripper_openness, 0.0, 1.0))
            return (
                left_reach + params.grasp_close_scale * near_grasp * closing
            )
        if state.phase == "grasped":
            return 1.0 + blade_reach + params.cut_progress_scale * min(
                state.cut_work_fraction, 1.0
            )
        phase_potential = {
            "orphan_retained": 3.0 + min(state.transport_fraction, 1.0),
            "transported": 5.0,
            "released": 6.0,
            "deposited": 8.0,
            "failed": 0.0,
        }
        return phase_potential[state.phase]

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._state is None:
            raise RuntimeError("reset must be called before step")
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (ACTION_SIZE,) or not np.isfinite(values).all():
            raise ValueError(f"action must be a finite {ACTION_SIZE}-vector")
        requested = values.copy()
        values = np.clip(requested, -1.0, 1.0)
        mask = phase_action_mask(self._state.vector())
        clipped_values = values.copy()
        values *= mask
        action_delta = values - self._previous_action
        previous = self._state
        current = self.runtime.apply_action(values, self.action_parameters)
        self._state = current
        self._steps += 1

        params = self.reward_parameters
        reward = self._potential(current) - self._potential(previous)
        reward -= params.time_penalty
        reward -= params.action_penalty * float(np.mean(np.square(values)))
        reward -= params.action_delta_penalty * float(
            np.mean(np.square(action_delta))
        )
        self._previous_action = values.copy()
        reward += params.cut_progress_scale * max(
            current.cut_work_fraction - previous.cut_work_fraction,
            0.0,
        )

        transitions = {
            "grasped": params.grasp_bonus,
            "orphan_retained": params.cut_bonus,
            "transported": params.transported_bonus,
            "released": params.released_bonus,
            "deposited": params.deposited_bonus,
        }
        if current.phase != previous.phase:
            reward += transitions.get(current.phase, 0.0)

        unsafe = not current.safety_clear or current.unsafe_contact_count > 0
        objective_reached = (
            self.terminal_phase is not None and current.phase == self.terminal_phase
        )
        terminated = current.phase in {"deposited", "failed"} or unsafe or objective_reached
        truncated = self._steps >= self.maximum_episode_steps and not terminated
        termination_reason = None
        if unsafe:
            reward -= params.unsafe_contact_penalty
            termination_reason = "unsafe_contact"
        elif current.phase == "failed":
            reward -= params.failure_penalty
            termination_reason = "task_failed"
        elif current.phase == "deposited":
            termination_reason = "success"
        elif objective_reached:
            termination_reason = f"curriculum_{self.terminal_phase}"
        elif truncated:
            termination_reason = "time_limit"

        info = current.info()
        info.update(
            episode_step=self._steps,
            success=current.phase == "deposited" and not unsafe,
            objective_reached=bool(
                (objective_reached or current.phase == "deposited") and not unsafe
            ),
            terminal_phase=self.terminal_phase,
            termination_reason=termination_reason,
            action_clipped=not np.array_equal(clipped_values, requested),
            action_phase_masked=not np.array_equal(values, clipped_values),
            action_mask=mask.tolist(),
            action_delta_rms=float(np.sqrt(np.mean(np.square(action_delta)))),
        )
        return current.vector(), float(reward), terminated, truncated, info
