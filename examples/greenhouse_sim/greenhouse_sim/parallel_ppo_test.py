from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from train_online_rl_parallel import _vector_gae  # noqa: E402


def test_vector_gae_keeps_worker_trajectories_independent():
    rewards = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    values = np.zeros_like(rewards)
    dones = np.asarray([[False, True], [False, False]])
    bootstrap = np.asarray([3.0, 30.0], dtype=np.float32)
    advantages, returns = _vector_gae(
        rewards, values, dones, bootstrap, gamma=1.0, lam=1.0
    )
    np.testing.assert_allclose(advantages[:, 0], [6.0, 5.0])
    np.testing.assert_allclose(advantages[:, 1], [10.0, 50.0])
    np.testing.assert_allclose(returns, advantages)


def test_vector_gae_rejects_mismatched_worker_shapes():
    with pytest.raises(ValueError):
        _vector_gae(
            np.zeros((2, 2)),
            np.zeros((2, 1)),
            np.zeros((2, 2)),
            np.zeros(2),
            gamma=0.99,
            lam=0.95,
        )
