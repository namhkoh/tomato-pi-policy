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


def test_atomic_mailbox_retries_transient_windows_replace_contention(tmp_path, monkeypatch):
    path = tmp_path / "command.json"
    original_replace = teleop.os.replace
    attempts = []
    delays = []

    def flaky_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError("destination is briefly open")
        return original_replace(source, destination)

    monkeypatch.setattr(teleop.os, "replace", flaky_replace)
    monkeypatch.setattr(teleop.time, "sleep", delays.append)

    command = _command()
    teleop.atomic_write_command(path, command)

    assert teleop.CommandMailbox(path).poll() == command
    assert len(attempts) == 3
    assert delays == [0.001, 0.002]


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


def test_extended_command_round_trips_full_upper_body_and_gripper():
    base = _command()
    command = teleop.TeleopCommand(
        sequence=base.sequence,
        monotonic_time_s=base.monotonic_time_s,
        left=base.left,
        right=base.right,
        left_gripper_closed=False,
        recording=base.recording,
        torso=teleop.JointGroupCommand((1.0,) * 6, enabled=True),
        head=teleop.JointGroupCommand((2.0, 3.0), enabled=True),
        left_gripper=teleop.GripperCommand(openness=0.35, enabled=True),
    )

    assert teleop.parse_command(teleop.command_payload(command)) == command


def test_gate_limits_and_rate_limits_torso_and_head():
    base = _command()
    command = teleop.TeleopCommand(
        sequence=base.sequence,
        monotonic_time_s=base.monotonic_time_s,
        left=base.left,
        right=base.right,
        left_gripper_closed=False,
        recording=False,
        torso=teleop.JointGroupCommand((5.0,) * 6, enabled=True),
        head=teleop.JointGroupCommand((5.0,) * 2, enabled=True),
        left_gripper=teleop.GripperCommand(openness=0.25, enabled=True),
    )
    gate = teleop.TeleopSafetyGate(
        dict.fromkeys(("left", "right"), ((-180.0,) * 7, (180.0,) * 7)),
        maximum_joint_speed_deg_s=100.0,
        joint_group_limits_degrees={
            "torso": ((-10.0,) * 6, (10.0,) * 6),
            "head": ((-10.0,) * 2, (10.0,) * 2),
        },
        joint_group_maximum_speeds_deg_s={"torso": 20.0, "head": 30.0},
    )

    accepted = gate.accept(
        command,
        now_s=10.01,
        dt_s=0.1,
        current_left_degrees=(0.0,) * 7,
        current_right_degrees=(0.0,) * 7,
        current_torso_degrees=(0.0,) * 6,
        current_head_degrees=(0.0,) * 2,
    )

    assert accepted.torso_target_degrees == (2.0,) * 6
    assert accepted.head_target_degrees == (3.0,) * 2
    assert accepted.apply_torso and accepted.apply_head
    assert accepted.apply_left_gripper
    assert accepted.left_gripper_openness == pytest.approx(0.25)
    assert accepted.rate_limited


@pytest.mark.parametrize("openness", (-0.01, 1.01, float("nan")))
def test_parse_rejects_invalid_continuous_gripper_openness(openness):
    payload = teleop.command_payload(_command())
    payload["left_gripper"] = {"openness": openness, "enabled": True}
    with pytest.raises(teleop.TeleopCommandError, match="between 0 and 1"):
        teleop.parse_command(payload)