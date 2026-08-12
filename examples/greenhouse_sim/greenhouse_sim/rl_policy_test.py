from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from greenhouse_sim import rl_env, rl_policy  # noqa: E402
from train_online_rl import ActorCritic  # noqa: E402


def _observations(count=8):
    values = np.zeros((count, rl_env.OBSERVATION_SIZE), dtype=np.float32)
    values[:, rl_env.LEFT_GRASP_DELTA_SLICE] = (0.02, 0.0, 0.0)
    values[
        :, rl_env.PHASE_OBSERVATION_SLICE.start
        + rl_env.PHASES.index("seek_grasp")
    ] = 1.0
    return values


def test_torch_phase_mask_matches_environment_numpy_mask():
    observations = _observations(3)
    observations[0, rl_env.LEFT_GRASP_DELTA_SLICE] = (0.1, 0.0, 0.0)
    observations[2, rl_env.PHASE_OBSERVATION_SLICE] = 0.0
    observations[
        2,
        rl_env.PHASE_OBSERVATION_SLICE.start
        + rl_env.PHASES.index("grasped"),
    ] = 1.0
    expected = rl_env.phase_action_mask(observations)
    actual = rl_policy.phase_action_mask_tensor(
        torch.as_tensor(observations)
    ).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_demonstration_loader_validates_and_concatenates(tmp_path):
    observations = _observations(2)
    actions = np.zeros((2, rl_env.ACTION_SIZE), dtype=np.float32)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez_compressed(first, observations=observations, actions=actions)
    np.savez_compressed(second, observations=observations, actions=actions)
    loaded_observations, loaded_actions, sources = rl_policy.load_demonstrations(
        (first, second),
        observation_size=rl_env.OBSERVATION_SIZE,
        action_size=rl_env.ACTION_SIZE,
    )
    assert loaded_observations.shape == (4, rl_env.OBSERVATION_SIZE)
    assert loaded_actions.shape == (4, rl_env.ACTION_SIZE)
    assert [source["transitions"] for source in sources] == [2, 2]

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, observations=observations, actions=np.ones((2, 3)))
    with pytest.raises(ValueError):
        rl_policy.load_demonstrations(
            (invalid,),
            observation_size=rl_env.OBSERVATION_SIZE,
            action_size=rl_env.ACTION_SIZE,
        )


def test_behavior_cloning_reduces_active_action_error():
    torch.manual_seed(3)
    observations = _observations(32)
    actions = np.zeros((32, rl_env.ACTION_SIZE), dtype=np.float32)
    actions[:, :7] = 0.35
    actions[:, 14] = -0.75
    model = ActorCritic(rl_env.OBSERVATION_SIZE, rl_env.ACTION_SIZE)
    report = rl_policy.behavior_clone(
        model,
        observations,
        actions,
        epochs=30,
        minibatch_size=16,
        learning_rate=1e-3,
        maximum_gradient_norm=1.0,
        device=torch.device("cpu"),
        seed=3,
    )
    assert report["final_loss"] < report["initial_loss"] * 0.1
    assert report["final_mae"] < report["initial_mae"]
