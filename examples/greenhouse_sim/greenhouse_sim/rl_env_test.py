from __future__ import annotations

import numpy as np
import pytest

from greenhouse_sim import rl_env


def _state(phase="seek_grasp", **changes):
    values = {
        "left_joint_position": np.zeros(7),
        "right_joint_position": np.zeros(7),
        "left_joint_velocity": np.zeros(7),
        "right_joint_velocity": np.zeros(7),
        "gripper_openness": 1.0,
        "left_grasp_delta_m": np.asarray([0.1, 0.0, 0.0]),
        "blade_cut_delta_m": np.asarray([0.2, 0.0, 0.0]),
        "target_axis": np.asarray([0.0, 0.0, 1.0]),
        "blade_edge_axis": np.asarray([0.0, 1.0, 0.0]),
        "blade_cut_direction": np.asarray([1.0, 0.0, 0.0]),
        "phase": phase,
        "grasp_force_fraction": 0.0,
        "cut_force_fraction": 0.0,
        "cut_work_fraction": 0.0,
        "transport_fraction": 0.0,
        "target_key": "Vine_0002/SubStem_02",
    }
    values.update(changes)
    return rl_env.DeleafState(**values)


class _Runtime:
    def __init__(self, states):
        self.states = list(states)
        self.actions = []
        self.seeds = []

    def reset(self, *, seed):
        self.seeds.append(seed)
        return self.states.pop(0)

    def apply_action(self, action, parameters):
        self.actions.append((action.copy(), parameters.physics_steps_per_action))
        return self.states.pop(0)


def test_observation_contract_is_stable_and_finite():
    observation = _state().vector()
    assert observation.shape == (rl_env.OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert observation[44] == 1.0  # seek-grasp phase one-hot


def test_grasp_cut_transport_deposit_receive_strict_event_bonuses():
    runtime = _Runtime(
        [
            _state(),
            _state("grasped", grasp_force_fraction=1.2),
            _state("orphan_retained", cut_work_fraction=1.0),
            _state("transported", transport_fraction=1.0),
            _state("released", transport_fraction=1.0),
            _state("deposited", transport_fraction=1.0),
        ]
    )
    env = rl_env.OnlineDeleafEnv(runtime)
    env.reset(seed=7)
    rewards = []
    for _ in range(5):
        _, reward, terminated, truncated, info = env.step(np.zeros(15))
        rewards.append(reward)
    assert runtime.seeds == [7]
    assert all(reward > 0.0 for reward in rewards)
    assert terminated and not truncated
    assert info["success"]
    assert info["termination_reason"] == "success"


def test_unsafe_contact_terminates_even_if_task_phase_is_not_failed():
    runtime = _Runtime([_state(), _state(unsafe_contact_count=1, safety_clear=False)])
    env = rl_env.OnlineDeleafEnv(runtime)
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.zeros(15))
    assert terminated and not truncated
    assert reward < -40.0
    assert info["termination_reason"] == "unsafe_contact"


def test_time_limit_truncates_without_claiming_success():
    runtime = _Runtime([_state(), _state()])
    env = rl_env.OnlineDeleafEnv(runtime, maximum_episode_steps=1)
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros(15))
    assert not terminated and truncated
    assert not info["success"]


def test_action_validation_and_clipping():
    runtime = _Runtime([_state(), _state()])
    env = rl_env.OnlineDeleafEnv(runtime)
    env.reset()
    _, _, _, _, info = env.step(np.full(15, 2.0))
    assert info["action_clipped"]
    np.testing.assert_array_equal(runtime.actions[0][0][:7], np.ones(7))
    np.testing.assert_array_equal(runtime.actions[0][0][7:], np.zeros(8))
    assert runtime.actions[0][1] == 12
    with pytest.raises(RuntimeError):
        rl_env.OnlineDeleafEnv(_Runtime([])).step(np.zeros(15))


def test_seek_grasp_masks_right_arm_and_far_gripper():
    observation = _state().vector()
    mask = rl_env.phase_action_mask(observation)
    np.testing.assert_array_equal(mask[:7], np.ones(7))
    np.testing.assert_array_equal(mask[7:14], np.zeros(7))
    assert mask[14] == 0.0

    runtime = _Runtime([_state(), _state()])
    env = rl_env.OnlineDeleafEnv(runtime)
    env.reset()
    _, _, _, _, info = env.step(np.ones(15))
    np.testing.assert_array_equal(runtime.actions[0][0][:7], np.ones(7))
    np.testing.assert_array_equal(runtime.actions[0][0][7:], np.zeros(8))
    assert info["action_phase_masked"]


def test_seek_grasp_enables_gripper_near_target_and_batch_masking():
    far = _state().vector()
    near = _state(left_grasp_delta_m=np.asarray([0.02, 0.0, 0.0])).vector()
    grasped = _state("grasped").vector()
    mask = rl_env.phase_action_mask(np.stack((far, near, grasped)))
    assert mask.shape == (3, rl_env.ACTION_SIZE)
    assert mask[0, 14] == 0.0
    assert mask[1, 14] == 1.0
    np.testing.assert_array_equal(mask[2], np.ones(rl_env.ACTION_SIZE))


def test_grasp_curriculum_terminates_without_claiming_full_task_success():
    runtime = _Runtime([_state(), _state("grasped", grasp_force_fraction=1.1)])
    env = rl_env.OnlineDeleafEnv(runtime, terminal_phase="grasped")
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.zeros(15))
    assert reward > 9.0
    assert terminated and not truncated
    assert info["objective_reached"]
    assert not info["success"]
    assert info["termination_reason"] == "curriculum_grasped"


def test_unsafe_grasp_does_not_count_as_curriculum_objective():
    runtime = _Runtime(
        [
            _state(),
            _state(
                "grasped", grasp_force_fraction=1.1,
                unsafe_contact_count=1, safety_clear=False,
            ),
        ]
    )
    env = rl_env.OnlineDeleafEnv(runtime, terminal_phase="grasped")
    env.reset()
    _, _, terminated, _, info = env.step(np.zeros(15))
    assert terminated
    assert not info["objective_reached"]
    assert info["termination_reason"] == "unsafe_contact"


def test_curriculum_terminal_validation():
    with pytest.raises(ValueError):
        rl_env.OnlineDeleafEnv(_Runtime([]), terminal_phase="seek_grasp")
    with pytest.raises(ValueError):
        rl_env.OnlineDeleafEnv(_Runtime([]), terminal_phase="failed")



def test_action_change_penalty_discourages_command_reversals():
    steady_runtime = _Runtime([_state(), _state(), _state()])
    reversing_runtime = _Runtime([_state(), _state(), _state()])
    steady = rl_env.OnlineDeleafEnv(steady_runtime)
    reversing = rl_env.OnlineDeleafEnv(reversing_runtime)
    steady.reset()
    reversing.reset()
    action = np.full(15, 0.5)
    steady.step(action)
    reversing.step(action)
    _, steady_reward, _, _, steady_info = steady.step(action)
    _, reversing_reward, _, _, reversing_info = reversing.step(-action)
    assert steady_info["action_delta_rms"] == pytest.approx(0.0)
    assert reversing_info["action_delta_rms"] == pytest.approx(np.sqrt(7.0 / 15.0))
    assert reversing_reward < steady_reward


def test_action_acceleration_parameters_must_be_positive():
    with pytest.raises(ValueError):
        rl_env.ActionParameters(maximum_arm_acceleration_degrees_s2=0.0)
    with pytest.raises(ValueError):
        rl_env.ActionParameters(maximum_gripper_acceleration_per_s2=0.0)


def test_near_target_gripper_closing_has_dense_grasp_shaping():
    runtime = _Runtime(
        [
            _state(left_grasp_delta_m=np.asarray([0.02, 0.0, 0.0])),
            _state(
                left_grasp_delta_m=np.asarray([0.02, 0.0, 0.0]),
                gripper_openness=0.8,
            ),
        ]
    )
    env = rl_env.OnlineDeleafEnv(runtime)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(15))
    assert reward > 0.04
