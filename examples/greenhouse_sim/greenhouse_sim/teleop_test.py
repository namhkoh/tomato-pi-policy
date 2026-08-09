import json

import numpy as np
import pytest

from greenhouse_sim import teleop


def _command(sequence=1, timestamp=10.0, *, left=True, right=True, value=5.0):
    arm = teleop.ArmCommand((value,) * 7, enabled=True)
    return teleop.TeleopCommand(
        sequence=sequence,
        monotonic_time_s=timestamp,
        left=teleop.ArmCommand(arm.joint_degrees, left),
        right=teleop.ArmCommand(arm.joint_degrees, right),
        left_gripper_closed=True,
        recording=True,
    )


def _gate(speed=45.0):
    limits = dict.fromkeys(("left", "right"), ((-180.0,) * 7, (180.0,) * 7))
    return teleop.TeleopSafetyGate(limits, maximum_joint_speed_deg_s=speed)


def test_atomic_mailbox_only_returns_new_commands(tmp_path):
    path = tmp_path / "command.json"
    command = _command()
    teleop.atomic_write_command(path, command)
    mailbox = teleop.CommandMailbox(path)
    assert mailbox.poll() == command
    assert mailbox.poll() is None
    assert json.loads(path.read_text())["schema"] == teleop.SCHEMA


def test_gate_requires_fresh_strictly_increasing_commands():
    gate = _gate()
    zeros = (0.0,) * 7
    accepted = gate.accept(
        _command(), now_s=10.1, dt_s=0.1, current_left_degrees=zeros, current_right_degrees=zeros
    )
    assert accepted.sequence == 1
    with pytest.raises(teleop.TeleopCommandError, match="increase"):
        gate.accept(
            _command(), now_s=10.1, dt_s=0.1, current_left_degrees=zeros, current_right_degrees=zeros
        )
    with pytest.raises(teleop.TeleopCommandError, match="stale"):
        _gate().accept(
            _command(), now_s=11.0, dt_s=0.1, current_left_degrees=zeros, current_right_degrees=zeros
        )


def test_gate_holds_disabled_arm_and_rate_limits_enabled_arm():
    zeros = (0.0,) * 7
    accepted = _gate(speed=10.0).accept(
        _command(left=False, value=5.0),
        now_s=10.01,
        dt_s=0.1,
        current_left_degrees=zeros,
        current_right_degrees=zeros,
    )
    assert not accepted.apply_left
    assert accepted.left_target_degrees == zeros
    assert accepted.right_target_degrees == (1.0,) * 7
    assert accepted.rate_limited


def test_gate_rejects_out_of_limit_angles():
    zeros = (0.0,) * 7
    with pytest.raises(teleop.TeleopCommandError, match="URDF"):
        _gate().accept(
            _command(value=181.0),
            now_s=10.01,
            dt_s=0.1,
            current_left_degrees=zeros,
            current_right_degrees=zeros,
        )


def test_gate_accepts_numpy_urdf_limit_arrays():
    limits = {
        side: (np.full(7, -180.0), np.full(7, 180.0))
        for side in ("left", "right")
    }
    gate = teleop.TeleopSafetyGate(limits)
    assert gate.maximum_joint_speed_deg_s == 45.0


def test_recorder_writes_metadata_and_jsonl(tmp_path):
    with teleop.DemonstrationRecorder(tmp_path / "episode", {"target": "vine/petiole"}) as recorder:
        recorder.append({"step": 1, "action": {"sequence": 2}})
    assert json.loads((tmp_path / "episode" / "metadata.json").read_text())["target"] == "vine/petiole"
    assert json.loads((tmp_path / "episode" / "steps.jsonl").read_text())["step"] == 1
