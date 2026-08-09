"""Safe file-mailbox contract for leader-arm control of the simulator only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
import pathlib
import tempfile

SCHEMA = "greenhouse.teleop.v1"


class TeleopCommandError(ValueError):
    """A command cannot safely be applied to the simulator."""


@dataclasses.dataclass(frozen=True)
class ArmCommand:
    joint_degrees: tuple[float, ...]
    enabled: bool


@dataclasses.dataclass(frozen=True)
class TeleopCommand:
    sequence: int
    monotonic_time_s: float
    left: ArmCommand
    right: ArmCommand
    left_gripper_closed: bool
    recording: bool


@dataclasses.dataclass(frozen=True)
class GatedCommand:
    sequence: int
    left_target_degrees: tuple[float, ...]
    right_target_degrees: tuple[float, ...]
    apply_left: bool
    apply_right: bool
    left_gripper_closed: bool
    recording: bool
    age_s: float
    rate_limited: bool


def _finite_vector(values, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TeleopCommandError(f"{name} must contain exactly seven joint angles")
    try:
        if len(values) != 7:
            raise TeleopCommandError(f"{name} must contain exactly seven joint angles")
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TeleopCommandError(f"{name} must contain exactly seven joint angles") from exc
    if not all(math.isfinite(value) for value in result):
        raise TeleopCommandError(f"{name} must contain only finite joint angles")
    return result


def parse_command(payload: Mapping) -> TeleopCommand:
    if payload.get("schema") != SCHEMA:
        raise TeleopCommandError(f"expected schema {SCHEMA!r}")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise TeleopCommandError("sequence must be a non-negative integer")
    timestamp = float(payload.get("monotonic_time_s", float("nan")))
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise TeleopCommandError("monotonic_time_s must be finite and non-negative")

    def arm(side: str) -> ArmCommand:
        value = payload.get(side)
        if not isinstance(value, Mapping):
            raise TeleopCommandError(f"{side} must be an object")
        enabled = value.get("enabled")
        if not isinstance(enabled, bool):
            raise TeleopCommandError(f"{side}.enabled must be boolean")
        return ArmCommand(
            joint_degrees=_finite_vector(value.get("joint_degrees"), f"{side}.joint_degrees"),
            enabled=enabled,
        )

    gripper = payload.get("left_gripper_closed")
    recording = payload.get("recording", False)
    if not isinstance(gripper, bool):
        raise TeleopCommandError("left_gripper_closed must be boolean")
    if not isinstance(recording, bool):
        raise TeleopCommandError("recording must be boolean")
    return TeleopCommand(
        sequence=sequence,
        monotonic_time_s=timestamp,
        left=arm("left"),
        right=arm("right"),
        left_gripper_closed=gripper,
        recording=recording,
    )


def command_payload(command: TeleopCommand) -> dict:
    return {
        "schema": SCHEMA,
        "sequence": command.sequence,
        "monotonic_time_s": command.monotonic_time_s,
        "left": dataclasses.asdict(command.left),
        "right": dataclasses.asdict(command.right),
        "left_gripper_closed": command.left_gripper_closed,
        "recording": command.recording,
    }


def atomic_write_command(path: pathlib.Path, command: TeleopCommand) -> None:
    """Publish one complete command; readers never observe a partial JSON file."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            json.dump(command_payload(command), handle, separators=(",", ":"))
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class CommandMailbox:
    """Read strictly increasing, atomically published teleop commands."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.last_sequence = -1

    def poll(self) -> TeleopCommand | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise TeleopCommandError(f"cannot read command mailbox: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise TeleopCommandError("command mailbox root must be an object")
        command = parse_command(payload)
        if command.sequence <= self.last_sequence:
            return None
        self.last_sequence = command.sequence
        return command


class TeleopSafetyGate:
    """Apply watchdog, deadman, URDF-limit, and slew-rate constraints."""

    def __init__(
        self,
        arm_limits_degrees: Mapping[str, tuple[Sequence[float], Sequence[float]]],
        *,
        watchdog_s: float = 0.25,
        maximum_joint_speed_deg_s: float = 45.0,
    ) -> None:
        if watchdog_s <= 0.0 or maximum_joint_speed_deg_s <= 0.0:
            raise ValueError("watchdog and maximum joint speed must be positive")
        self.watchdog_s = float(watchdog_s)
        self.maximum_joint_speed_deg_s = float(maximum_joint_speed_deg_s)
        self._limits = {}
        for side in ("left", "right"):
            lower, upper = arm_limits_degrees[side]
            lower_values = _finite_vector(lower, f"{side} lower limits")
            upper_values = _finite_vector(upper, f"{side} upper limits")
            if any(lo >= hi for lo, hi in zip(lower_values, upper_values, strict=True)):
                raise ValueError(f"invalid {side} arm limits")
            self._limits[side] = (lower_values, upper_values)
        self.last_sequence = -1

    def _target(self, side: str, requested, current, dt_s: float) -> tuple[tuple[float, ...], bool]:
        requested_values = _finite_vector(requested, f"{side} requested joints")
        current_values = _finite_vector(current, f"{side} current joints")
        lower, upper = self._limits[side]
        if any(
            value < lo or value > hi
            for value, lo, hi in zip(requested_values, lower, upper, strict=True)
        ):
            raise TeleopCommandError(f"{side} command exceeds the RB-Y1 URDF joint limits")
        maximum_delta = self.maximum_joint_speed_deg_s * max(float(dt_s), 0.0)
        limited = []
        was_limited = False
        for value, actual in zip(requested_values, current_values, strict=True):
            delta = value - actual
            bounded = max(-maximum_delta, min(maximum_delta, delta))
            limited.append(actual + bounded)
            was_limited = was_limited or not math.isclose(delta, bounded, abs_tol=1e-12)
        return tuple(limited), was_limited

    def accept(
        self,
        command: TeleopCommand,
        *,
        now_s: float,
        dt_s: float,
        current_left_degrees,
        current_right_degrees,
    ) -> GatedCommand:
        if command.sequence <= self.last_sequence:
            raise TeleopCommandError("command sequence did not increase")
        age = float(now_s) - command.monotonic_time_s
        if age < -0.050:
            raise TeleopCommandError("command timestamp is in the future")
        if age > self.watchdog_s:
            raise TeleopCommandError(
                f"command is stale ({age * 1000.0:.1f} ms > {self.watchdog_s * 1000.0:.1f} ms)"
            )
        left_current = _finite_vector(current_left_degrees, "left current joints")
        right_current = _finite_vector(current_right_degrees, "right current joints")
        left_target, left_limited = (
            self._target("left", command.left.joint_degrees, left_current, dt_s)
            if command.left.enabled
            else (left_current, False)
        )
        right_target, right_limited = (
            self._target("right", command.right.joint_degrees, right_current, dt_s)
            if command.right.enabled
            else (right_current, False)
        )
        self.last_sequence = command.sequence
        return GatedCommand(
            sequence=command.sequence,
            left_target_degrees=left_target,
            right_target_degrees=right_target,
            apply_left=command.left.enabled,
            apply_right=command.right.enabled,
            left_gripper_closed=command.left_gripper_closed,
            recording=command.recording,
            age_s=max(age, 0.0),
            rate_limited=left_limited or right_limited,
        )


class DemonstrationRecorder:
    """Append synchronized simulator observation/action records as JSON Lines."""

    def __init__(self, directory: pathlib.Path, metadata: Mapping) -> None:
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_directory = self.directory / "frames"
        self.frames_directory.mkdir(exist_ok=True)
        self.metadata_path = self.directory / "metadata.json"
        self.steps_path = self.directory / "steps.jsonl"
        self.metadata_path.write_text(json.dumps(dict(metadata), indent=2), encoding="utf-8")
        self._stream = self.steps_path.open("a", encoding="utf-8")
        self.samples = 0

    def append(self, sample: Mapping) -> None:
        self._stream.write(json.dumps(dict(sample), separators=(",", ":"), default=str) + "\n")
        self._stream.flush()
        self.samples += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
