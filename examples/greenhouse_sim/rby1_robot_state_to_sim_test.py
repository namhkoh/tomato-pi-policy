from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import rby1_robot_state_to_sim as bridge


_MODEL = SimpleNamespace(
    torso_idx=(2, 3, 4, 5, 6, 7),
    right_arm_idx=(8, 9, 10, 11, 12, 13, 14),
    left_arm_idx=(15, 16, 17, 18, 19, 20, 21),
    head_idx=(22, 23),
)


def test_extract_degrees_uses_rby1_model_joint_indices() -> None:
    degrees = np.arange(24, dtype=np.float64) - 12.0
    state = SimpleNamespace(position=np.radians(degrees))

    left, right, torso, head = bridge._extract_degrees(state, _MODEL)

    np.testing.assert_allclose(left, degrees[list(_MODEL.left_arm_idx)])
    np.testing.assert_allclose(right, degrees[list(_MODEL.right_arm_idx)])
    np.testing.assert_allclose(torso, degrees[list(_MODEL.torso_idx)])
    np.testing.assert_allclose(head, degrees[list(_MODEL.head_idx)])


@pytest.mark.parametrize(
    "position",
    (
        np.zeros(21, dtype=np.float64),
        np.concatenate((np.zeros(23, dtype=np.float64), [np.nan])),
    ),
)
def test_extract_degrees_rejects_malformed_state(position: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="malformed or non-finite"):
        bridge._extract_degrees(SimpleNamespace(position=position), _MODEL)




def test_measured_head_boundary_drift_is_clamped_to_model_limit() -> None:
    bounded, clamped = bridge._clamp_measured_head_degrees(
        (-29.970696889772263, 0.17578121342974934)
    )

    assert clamped
    np.testing.assert_allclose(bounded, (-29.965988, 0.17578121342974934))


def test_live_measured_head_boundary_drift_is_clamped_to_model_limit() -> None:
    bounded, clamped = bridge._clamp_measured_head_degrees(
        (-30.146478, 0.17578121342974934)
    )

    assert clamped
    np.testing.assert_allclose(bounded, (-29.965988, 0.17578121342974934))


def test_measured_head_state_far_outside_model_limit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="exceeds the Model A limits"):
        bridge._clamp_measured_head_degrees((-30.5, 0.0))
def test_normalize_left_gripper_uses_session_homing_stops() -> None:
    payload = {
        "motor_states": [
            {"id": 0, "position": 5.0},
            {"id": 1, "position": 0.0},
        ],
        "gripper_min_q": [-4.0, -4.5],
        "gripper_max_q": [5.0, 4.5],
    }

    assert bridge._normalize_left_gripper_openness(payload) == pytest.approx(0.5)


def test_normalize_left_gripper_maps_numeric_minimum_to_open() -> None:
    payload = {
        "motor_states": [{"id": 1, "position": -4.50223267578125}],
        "gripper_min_q": [0.0, -4.50223267578125],
        "gripper_max_q": [1.0, 4.570970527343751],
    }

    assert bridge._normalize_left_gripper_openness(payload) == 1.0


def test_normalize_left_gripper_maps_numeric_maximum_to_closed() -> None:
    payload = {
        "motor_states": [{"id": 1, "position": 4.570970527343751}],
        "gripper_min_q": [0.0, -4.50223267578125],
        "gripper_max_q": [1.0, 4.570970527343751],
    }

    assert bridge._normalize_left_gripper_openness(payload) == 0.0

def test_normalize_left_gripper_clamps_encoder_noise() -> None:
    payload = {
        "motor_states": [{"id": 1, "position": 4.6}],
        "gripper_min_q": [0.0, -4.5],
        "gripper_max_q": [1.0, 4.5],
    }

    assert bridge._normalize_left_gripper_openness(payload) == 0.0


def test_normalize_left_gripper_rejects_missing_calibration() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        bridge._normalize_left_gripper_openness({"motor_states": []})