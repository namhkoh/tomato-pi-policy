from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from collect_online_rl_grasp_demo import (  # noqa: E402
    decode_left_arm_degrees,
    expert_action,
)
from greenhouse_sim import rl_env  # noqa: E402


LOWER = np.asarray([-180.0, -1.0, -180.0, -150.0, -180.0, -90.0, -155.0])
UPPER = np.asarray([180.0, 180.0, 180.0, 1.0, 180.0, 110.0, 155.0])


def _observation(distance_m=0.1):
    observation = np.zeros(rl_env.OBSERVATION_SIZE, dtype=np.float32)
    observation[rl_env.LEFT_GRASP_DELTA_SLICE] = (distance_m, 0.0, 0.0)
    observation[
        rl_env.PHASE_OBSERVATION_SLICE.start
        + rl_env.PHASES.index("seek_grasp")
    ] = 1.0
    return observation


def test_decode_left_arm_position_inverts_runtime_normalization():
    observation = _observation()
    observation[:7] = np.asarray([-1.0, 0.0, 1.0, 0.5, -0.5, 0.25, -0.25])
    decoded = decode_left_arm_degrees(observation, LOWER, UPPER)
    expected = 0.5 * (LOWER + UPPER) + observation[:7] * 0.5 * (UPPER - LOWER)
    np.testing.assert_allclose(decoded, expected)


def test_expert_action_moves_only_left_arm_and_respects_phase_mask():
    observation = _observation(distance_m=0.1)
    measured = decode_left_arm_degrees(observation, LOWER, UPPER)
    target = measured + 10.0
    action = expert_action(
        observation,
        target,
        LOWER,
        UPPER,
        maximum_arm_speed_degrees_s=20.0,
        position_gain=1.0,
        close_gripper=True,
    )
    np.testing.assert_allclose(action[:7], 0.5)
    np.testing.assert_array_equal(action[7:], np.zeros(8))


def test_expert_action_closes_only_in_grasp_neighbourhood():
    observation = _observation(distance_m=0.02)
    measured = decode_left_arm_degrees(observation, LOWER, UPPER)
    action = expert_action(
        observation,
        measured,
        LOWER,
        UPPER,
        maximum_arm_speed_degrees_s=35.0,
        position_gain=4.0,
        close_gripper=True,
    )
    assert action[14] == -1.0
    with pytest.raises(ValueError):
        expert_action(
            observation,
            measured,
            LOWER,
            UPPER,
            maximum_arm_speed_degrees_s=0.0,
            position_gain=4.0,
        )
