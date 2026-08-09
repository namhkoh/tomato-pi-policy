"""Publish RB-Y1 leader-arm state to the greenhouse simulator only.

This process never connects to, powers, or commands the physical RB-Y1. It
opens only the vendor leader-arm/trigger device and publishes an atomic JSON
mailbox consumed by ``interactive_greenhouse.py``.
"""

# The local package import intentionally follows the script-directory insertion.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import pathlib
import signal
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import teleop


_VENDOR_SAMPLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "third_party"
    / "rby1-sdk"
    / "examples"
    / "python"
    / "35_leader_arm_teleop_with_monitor.py"
)
_LEFT_READY_DEGREES = (0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0)
_RIGHT_READY_DEGREES = (0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command-file",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/teleop_command.json"),
    )
    parser.add_argument("--device", default=None, help="leader-arm serial device; auto-detect by default")
    parser.add_argument("--period", type=float, default=0.01, help="leader sampling period in seconds")
    parser.add_argument(
        "--gripper-threshold",
        type=int,
        default=500,
        help="left trigger value at or above which the simulated gripper closes",
    )
    parser.add_argument("--record", action="store_true", help="request synchronized simulator recording")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="publish one disabled ready-pose command without opening leader hardware",
    )
    return parser.parse_args()


def _next_sequence(path: pathlib.Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(int(payload.get("sequence", -1)) + 1, 0)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def _load_vendor_sample():
    if not _VENDOR_SAMPLE.exists():
        raise FileNotFoundError(f"vendored leader-arm sample is missing: {_VENDOR_SAMPLE}")
    spec = importlib.util.spec_from_file_location("rby1_leader_arm_vendor", _VENDOR_SAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored leader-arm sample: {_VENDOR_SAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _publish(
    path: pathlib.Path,
    sequence: int,
    *,
    left_degrees,
    right_degrees,
    left_enabled: bool,
    right_enabled: bool,
    gripper_closed: bool,
    recording: bool,
) -> None:
    teleop.atomic_write_command(
        path,
        teleop.TeleopCommand(
            sequence=sequence,
            monotonic_time_s=time.monotonic(),
            left=teleop.ArmCommand(tuple(float(value) for value in left_degrees), left_enabled),
            right=teleop.ArmCommand(tuple(float(value) for value in right_degrees), right_enabled),
            left_gripper_closed=bool(gripper_closed),
            recording=bool(recording),
        ),
    )


def main() -> int:
    args = parse_args()
    if args.period <= 0.0:
        raise ValueError("--period must be positive")
    if not 0 <= args.gripper_threshold <= 1000:
        raise ValueError("--gripper-threshold must be between 0 and 1000")
    sequence = _next_sequence(args.command_file)
    if args.dry_run:
        _publish(
            args.command_file,
            sequence,
            left_degrees=_LEFT_READY_DEGREES,
            right_degrees=_RIGHT_READY_DEGREES,
            left_enabled=False,
            right_enabled=False,
            gripper_closed=False,
            recording=args.record,
        )
        print(f"published disabled simulator-only command {sequence} to {args.command_file}")
        return 0

    vendor = _load_vendor_sample()
    leader = vendor.LeaderArm(
        **({"dev_name": args.device} if args.device is not None else {}),
        control_period=args.period,
        check_goal_position=False,
    )
    active_ids = leader.initialize(verbose=True)
    if len(active_ids) != leader.DEVICE_COUNT:
        with contextlib.suppress(Exception):
            leader.DisableTorque()
        raise RuntimeError(
            f"leader-arm discovery found {len(active_ids)} of {leader.DEVICE_COUNT} devices; refusing teleop"
        )

    stop = threading.Event()
    sequence_lock = threading.Lock()

    def safety_function(state) -> None:
        stop.set()
        with contextlib.suppress(Exception):
            leader.DisableTorque()
        print(
            "leader-arm communication fault; torque disabled and simulator publishing stopped: "
            f"joints={state.joint_fault_ids}, tools={state.tool_fault_ids}",
            file=sys.stderr,
        )

    def callback(state):
        nonlocal sequence
        with sequence_lock:
            current_sequence = sequence
            sequence += 1
        _publish(
            args.command_file,
            current_sequence,
            right_degrees=np.degrees(state.q_joint[0:7]),
            left_degrees=np.degrees(state.q_joint[7:14]),
            right_enabled=state.button_right.button == 1,
            left_enabled=state.button_left.button == 1,
            gripper_closed=state.button_left.trigger >= args.gripper_threshold,
            recording=args.record,
        )

        # Gravity compensation and joint-limit resistance act only on the
        # hand-guided leader device. No physical RB-Y1 command stream exists.
        control = vendor.LeaderArm.ControlInput()
        control.target_operating_mode.fill(vendor.rby.DynamixelBus.CurrentControlMode)
        lower = np.deg2rad(
            [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
        )
        upper = np.deg2rad(
            [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
        )
        barrier = 0.5 * (
            np.maximum(lower - state.q_joint, 0.0)
            + np.minimum(upper - state.q_joint, 0.0)
        )
        viscous = np.asarray(
            [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2
        ) * state.qvel_joint
        limits = np.asarray([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
        control.target_torque[:] = np.clip(
            state.gravity_term + barrier + viscous,
            -limits,
            limits,
        )
        return control

    def handle_stop(signum, frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    if not leader.start_control(callback, safety_function=safety_function):
        leader.DisableTorque()
        raise RuntimeError("leader-arm control loop did not start")
    print(
        "publishing leader arms to the simulator only; hold each tool button to enable that arm; "
        "Ctrl+C stops and disables leader torque"
    )
    try:
        while not stop.wait(0.25):
            pass
    finally:
        leader.close()
        _publish(
            args.command_file,
            sequence,
            left_degrees=_LEFT_READY_DEGREES,
            right_degrees=_RIGHT_READY_DEGREES,
            left_enabled=False,
            right_enabled=False,
            gripper_closed=False,
            recording=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
