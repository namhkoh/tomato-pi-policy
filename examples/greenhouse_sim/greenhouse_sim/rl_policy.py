"""Policy-side curriculum helpers shared by PPO training and evaluation."""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from . import rl_env


def phase_action_mask_tensor(observation: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`rl_env.phase_action_mask`."""

    if observation.shape[-1:] != (rl_env.OBSERVATION_SIZE,):
        raise ValueError("observation tensor has the wrong final dimension")
    mask = torch.ones(
        observation.shape[:-1] + (rl_env.ACTION_SIZE,),
        dtype=observation.dtype,
        device=observation.device,
    )
    seek_grasp = (
        observation[
            ..., rl_env.PHASE_OBSERVATION_SLICE.start
            + rl_env.PHASES.index("seek_grasp")
        ]
        > 0.5
    )
    mask[..., rl_env.RIGHT_ARM_ACTION_SLICE] = (~seek_grasp).to(
        observation.dtype
    ).unsqueeze(-1)
    grasp_distance = torch.linalg.vector_norm(
        observation[..., rl_env.LEFT_GRASP_DELTA_SLICE], dim=-1
    )
    enable_gripper = torch.logical_or(
        ~seek_grasp,
        grasp_distance <= rl_env.DEFAULT_GRIPPER_ACTIVATION_DISTANCE_M,
    )
    mask[..., rl_env.GRIPPER_ACTION_INDEX] = enable_gripper.to(observation.dtype)
    return mask


def load_demonstrations(
    paths,
    *,
    observation_size: int,
    action_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load and validate one or more accepted expert trajectory datasets."""

    observations = []
    actions = []
    sources = []
    for source in paths:
        path = pathlib.Path(source)
        with np.load(path, allow_pickle=False) as archive:
            source_observations = np.asarray(archive["observations"], dtype=np.float32)
            source_actions = np.asarray(archive["actions"], dtype=np.float32)
        if (
            source_observations.ndim != 2
            or source_observations.shape[1] != observation_size
            or source_actions.shape != (source_observations.shape[0], action_size)
        ):
            raise ValueError(f"demonstration has incompatible shapes: {path}")
        if source_observations.shape[0] < 1:
            raise ValueError(f"demonstration contains no transitions: {path}")
        if (
            not np.isfinite(source_observations).all()
            or not np.isfinite(source_actions).all()
            or np.max(np.abs(source_actions)) > 1.000001
        ):
            raise ValueError(f"demonstration contains invalid values: {path}")
        observations.append(source_observations)
        actions.append(source_actions)
        sources.append(
            {
                "path": str(path),
                "transitions": int(source_observations.shape[0]),
            }
        )
    if not observations:
        raise ValueError("at least one demonstration path is required")
    return np.concatenate(observations), np.concatenate(actions), sources


def behavior_clone(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    minibatch_size: int,
    learning_rate: float,
    maximum_gradient_norm: float,
    device: torch.device,
    seed: int,
) -> dict:
    """Warm-start active actor dimensions from accepted physical trajectories."""

    if epochs < 1 or minibatch_size < 1 or learning_rate <= 0.0:
        raise ValueError("behavior-cloning counts and learning rate must be positive")
    observation_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=device
    )
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    if observation_tensor.ndim != 2 or action_tensor.shape != (
        observation_tensor.shape[0], model.actor.out_features
    ):
        raise ValueError("behavior-cloning arrays have incompatible shapes")
    mask = phase_action_mask_tensor(observation_tensor)
    active_count = torch.clamp(mask.sum(), min=1.0)

    def metrics() -> tuple[float, float]:
        with torch.no_grad():
            distribution, _ = model.distribution_and_value(observation_tensor)
            predicted = torch.tanh(distribution.mean) * mask
            difference = (predicted - action_tensor) * mask
            loss = torch.square(difference).sum() / active_count
            mae = torch.abs(difference).sum() / active_count
            return float(loss.cpu()), float(mae.cpu())

    initial_loss, initial_mae = metrics()
    parameters = list(model.body.parameters()) + list(model.actor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    random = np.random.default_rng(seed)
    indices = np.arange(observation_tensor.shape[0])
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch = indices[start : start + minibatch_size]
            distribution, _ = model.distribution_and_value(observation_tensor[batch])
            batch_mask = mask[batch]
            predicted = torch.tanh(distribution.mean) * batch_mask
            difference = (predicted - action_tensor[batch]) * batch_mask
            loss = torch.square(difference).sum() / torch.clamp(
                batch_mask.sum(), min=1.0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm)
            optimizer.step()
    final_loss, final_mae = metrics()
    return {
        "epochs": epochs,
        "transitions": int(observation_tensor.shape[0]),
        "active_values": int(mask.sum().item()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_mae": initial_mae,
        "final_mae": final_mae,
    }
