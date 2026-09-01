"""Mirror measured physical RB-Y1 upper-body state into the greenhouse simulator.

This process is strictly read-only with respect to the physical robot. It
connects to the RB-Y1 SDK state stream and the teleop PC's HTTP gripper status,
then publishes measured torso, head, arm, and left-gripper state to the
simulator's atomic mailbox. It never powers, servos, enables, or commands
RB-Y1 or either physical gripper.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import contextlib
import json
import pathlib
import signal
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import teleop


_LEFT_READY_DEGREES = np.asarray((0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0))
_RIGHT_READY_DEGREES = np.asarray((0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0))
_TORSO_READY_DEGREES = np.asarray((0.0, 45.0, -90.0, 45.0, 0.0, 0.0))
_HEAD_READY_DEGREES = np.zeros(2, dtype=np.float64)
_LEFT_GRIPPER_MOTOR_ID = 1
_DEFAULT_GRIPPER_STATUS_URL = "http://192.168.50.243:8765/status"
_HEAD_LIMITS_DEGREES = (
    np.asarray((-29.965988, -20.053523), dtype=np.float64),
    np.asarray((29.965988, 89.954374), dtype=np.float64),
)
_MEASURED_LIMIT_CLAMP_TOLERANCE_DEGREES = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="RB-Y1 SDK host:port")
    parser.add_argument("--model", default="a")
    parser.add_argument(
        "--command-file",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/teleop_command.json"),
    )
    parser.add_argument("--rate", type=float, default=50.0, help="RB-Y1 state stream Hz")
    parser.add_argument(
        "--mirror",
        choices=("left", "right", "both"),
        default="both",
        help="physical arms explicitly enabled for simulator mirroring",
    )
    parser.add_argument("--no-torso", action="store_true")
    parser.add_argument("--no-head", action="store_true")
    parser.add_argument("--no-left-gripper", action="store_true")
    parser.add_argument(
        "--gripper-status-url",
        default=_DEFAULT_GRIPPER_STATUS_URL,
        help="read-only teleop-PC HTTP status endpoint",
    )
    parser.add_argument(
        "--gripper-rate",
        type=float,
        default=10.0,
        help="maximum read-only gripper status polling rate in Hz",
    )
    parser.add_argument("--gripper-timeout", type=float, default=0.20)
    parser.add_argument(
        "--gripper-stale",
        type=float,
        default=0.35,
        help="disable gripper updates when the last HTTP sample is older than this",
    )
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


def _next_sequence(path: pathlib.Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(int(payload.get("sequence", -1)) + 1, 0)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def _extract_degrees(
    state, model
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(state.position, dtype=np.float64)
    index_groups = (
        model.left_arm_idx,
        model.right_arm_idx,
        model.torso_idx,
        model.head_idx,
    )
    required = max(index for indices in index_groups for index in indices) + 1
    if position.ndim != 1 or position.size < required or not np.isfinite(position).all():
        raise RuntimeError("RB-Y1 returned a malformed or non-finite joint state")
    return tuple(
        np.degrees(position[np.asarray(indices, dtype=np.int64)])
        for indices in index_groups
    )




def _clamp_measured_head_degrees(
    head_degrees,
    *,
    tolerance_degrees: float = _MEASURED_LIMIT_CLAMP_TOLERANCE_DEGREES,
) -> tuple[np.ndarray, bool]:
    """Reconcile sub-degree encoder/model boundary drift before publication."""
    values = np.asarray(head_degrees, dtype=np.float64)
    tolerance = float(tolerance_degrees)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise RuntimeError("measured RB-Y1 head state is malformed or non-finite")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("measured head clamp tolerance must be finite and non-negative")
    lower, upper = _HEAD_LIMITS_DEGREES
    bounded = np.clip(values, lower, upper)
    excess = float(np.max(np.abs(values - bounded)))
    if excess > tolerance + 1e-12:
        raise RuntimeError(
            "measured RB-Y1 head state exceeds the Model A limits by "
            f"{excess:.6f} degrees"
        )
    return bounded, bool(excess > 0.0)
def _normalize_left_gripper_openness(payload: Mapping) -> float:
    """Return semantic openness from this session's numeric encoder stops.

    The lab's left motor (id 1) reaches its *open* mechanical stop at the
    numeric minimum and its *closed* stop at the numeric maximum.  The HTTP
    fields are ordered numeric extrema, not `closed/open` semantic extrema.
    """

    if not isinstance(payload, Mapping):
        raise RuntimeError("gripper status root is not an object")
    motors = payload.get("motor_states")
    if not isinstance(motors, list):
        raise RuntimeError("gripper status has no motor_states list")
    try:
        motor = next(
            item
            for item in motors
            if isinstance(item, Mapping)
            and int(item.get("id", -1)) == _LEFT_GRIPPER_MOTOR_ID
        )
        position = float(motor["position"])
        minimum = float(payload["gripper_min_q"][_LEFT_GRIPPER_MOTOR_ID])
        maximum = float(payload["gripper_max_q"][_LEFT_GRIPPER_MOTOR_ID])
    except (KeyError, IndexError, TypeError, ValueError, StopIteration) as exc:
        raise RuntimeError("left gripper status/calibration is incomplete") from exc
    values = np.asarray((position, minimum, maximum), dtype=np.float64)
    if not np.isfinite(values).all() or abs(maximum - minimum) <= 1e-9:
        raise RuntimeError("left gripper status/calibration is non-finite or degenerate")
    return float(np.clip((maximum - position) / (maximum - minimum), 0.0, 1.0))


class GripperStatusPoller:
    """Poll the read-only gripper cache without blocking the 50 Hz robot stream."""

    def __init__(self, url: str, *, rate_hz: float, timeout_s: float) -> None:
        self.url = str(url)
        self.period_s = 1.0 / float(rate_hz)
        self.timeout_s = float(timeout_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="rby1-read-only-gripper-status",
            daemon=True,
        )
        self._latest_openness: float | None = None
        self._latest_time_s: float | None = None
        self._latest_error: str | None = "waiting for first gripper status sample"

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout_s) as response:
                    payload = json.load(response)
                openness = _normalize_left_gripper_openness(payload)
                with self._lock:
                    self._latest_openness = openness
                    self._latest_time_s = time.monotonic()
                    self._latest_error = None
            except Exception as exc:
                with self._lock:
                    self._latest_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            self._stop.wait(max(self.period_s - elapsed, 0.0))

    def snapshot(self, *, now_s: float, stale_s: float) -> tuple[float | None, bool, str | None]:
        with self._lock:
            openness = self._latest_openness
            sample_time = self._latest_time_s
            error = self._latest_error
        fresh = bool(
            openness is not None
            and sample_time is not None
            and now_s - sample_time <= stale_s
        )
        if not fresh and error is None:
            error = "last gripper status sample is stale"
        return openness, fresh, error

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(self.timeout_s + self.period_s, 0.25))


def _publish(
    path: pathlib.Path,
    sequence: int,
    *,
    left_degrees,
    right_degrees,
    torso_degrees,
    head_degrees,
    left_enabled: bool,
    right_enabled: bool,
    torso_enabled: bool,
    head_enabled: bool,
    left_gripper_openness: float,
    left_gripper_enabled: bool,
    recording: bool,
) -> None:
    teleop.atomic_write_command(
        path,
        teleop.TeleopCommand(
            sequence=sequence,
            monotonic_time_s=time.monotonic(),
            left=teleop.ArmCommand(tuple(map(float, left_degrees)), left_enabled),
            right=teleop.ArmCommand(tuple(map(float, right_degrees)), right_enabled),
            left_gripper_closed=bool(left_gripper_openness <= 0.10),
            recording=recording,
            torso=teleop.JointGroupCommand(
                tuple(map(float, torso_degrees)), torso_enabled
            ),
            head=teleop.JointGroupCommand(
                tuple(map(float, head_degrees)), head_enabled
            ),
            left_gripper=teleop.GripperCommand(
                openness=float(left_gripper_openness),
                enabled=left_gripper_enabled,
            ),
        ),
    )


def main() -> int:
    args = parse_args()
    numeric = (
        args.rate,
        args.gripper_rate,
        args.gripper_timeout,
        args.gripper_stale,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in numeric):
        raise ValueError("state/gripper rates, timeout, and stale interval must be positive")

    import rby1_sdk as rby

    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"could not connect read-only RB-Y1 state source at {args.address}")
    model = robot.model()
    sequence = _next_sequence(args.command_file)
    stop = threading.Event()
    lock = threading.Lock()
    latest = {
        "left": _LEFT_READY_DEGREES.copy(),
        "right": _RIGHT_READY_DEGREES.copy(),
        "torso": _TORSO_READY_DEGREES.copy(),
        "head": _HEAD_READY_DEGREES.copy(),
        "left_gripper_openness": 1.0,
    }
    poller = None
    if not args.no_left_gripper:
        poller = GripperStatusPoller(
            args.gripper_status_url,
            rate_hz=min(args.gripper_rate, 10.0),
            timeout_s=args.gripper_timeout,
        )
        poller.start()
    last_gripper_warning = None
    head_limit_clamped = False

    def callback(state, *_unused) -> None:
        nonlocal sequence, last_gripper_warning, head_limit_clamped
        try:
            left, right, torso, head = _extract_degrees(state, model)
            head, head_was_clamped = _clamp_measured_head_degrees(head)
            if head_was_clamped and not head_limit_clamped:
                print(
                    "measured head state clamped to the exact Model A limit",
                    file=sys.stderr,
                )
            head_limit_clamped = head_was_clamped
            now = time.monotonic()
            openness = None
            gripper_fresh = False
            gripper_error = None
            if poller is not None:
                openness, gripper_fresh, gripper_error = poller.snapshot(
                    now_s=now, stale_s=args.gripper_stale
                )
            with lock:
                latest["left"] = left.copy()
                latest["right"] = right.copy()
                latest["torso"] = torso.copy()
                latest["head"] = head.copy()
                if openness is not None:
                    latest["left_gripper_openness"] = float(openness)
                current_openness = float(latest["left_gripper_openness"])
                current_sequence = sequence
                sequence += 1
            _publish(
                args.command_file,
                current_sequence,
                left_degrees=left,
                right_degrees=right,
                torso_degrees=torso,
                head_degrees=head,
                left_enabled=args.mirror in {"left", "both"},
                right_enabled=args.mirror in {"right", "both"},
                torso_enabled=not args.no_torso,
                head_enabled=not args.no_head,
                left_gripper_openness=current_openness,
                left_gripper_enabled=gripper_fresh and not args.no_left_gripper,
                recording=args.record,
            )
            warning = None if gripper_fresh or args.no_left_gripper else gripper_error
            if warning != last_gripper_warning:
                if warning is not None:
                    print(
                        f"left gripper mirroring held: {warning}",
                        file=sys.stderr,
                    )
                elif last_gripper_warning is not None:
                    print("left gripper status recovered", file=sys.stderr)
                last_gripper_warning = warning
        except Exception as exc:
            print(f"RB-Y1 state publishing stopped: {exc}", file=sys.stderr)
            stop.set()

    def handle_stop(_signum, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    # The RB-Y1 Python binding returns None here; stream failures are reported
    # by the binding or callback, not as a Boolean result.
    robot.start_state_update(callback, rate=args.rate)
    print(
        f"read-only RB-Y1 state mirror active at {args.address}; "
        f"arms={args.mirror}, torso={not args.no_torso}, head={not args.no_head}, "
        f"left_gripper={not args.no_left_gripper}; no physical command API is used"
    )
    try:
        while not stop.wait(0.25):
            pass
    finally:
        with contextlib.suppress(Exception):
            robot.stop_state_update()
        if poller is not None:
            poller.close()
        with lock:
            final_sequence = sequence
            final = {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in latest.items()
            }
        _publish(
            args.command_file,
            final_sequence,
            left_degrees=final["left"],
            right_degrees=final["right"],
            torso_degrees=final["torso"],
            head_degrees=final["head"],
            left_enabled=False,
            right_enabled=False,
            torso_enabled=False,
            head_enabled=False,
            left_gripper_openness=final["left_gripper_openness"],
            left_gripper_enabled=False,
            recording=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())