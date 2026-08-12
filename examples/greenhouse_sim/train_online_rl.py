"""Reference single-environment PPO trainer for the greenhouse RL server.

Start ``interactive_greenhouse.py --headless --rl-server`` first.  The trainer
runs outside Isaac Sim so policy dependencies and GPU memory are isolated from
the physics process.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import rl_policy  # noqa: E402
from greenhouse_sim.rl_client import DeleafClient  # noqa: E402


class ActorCritic(torch.nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(observation_size, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
        )
        self.actor = torch.nn.Linear(256, action_size)
        self.critic = torch.nn.Linear(256, 1)
        self.log_std = torch.nn.Parameter(torch.full((action_size,), -1.5))
        self._initialize()

    def distribution_and_value(self, observation):
        features = self.body(observation)
        mean = self.actor(features)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), self.critic(features).squeeze(-1)

    def _initialize(self) -> None:
        for layer in self.body:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=2.0 ** 0.5)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        torch.nn.init.zeros_(self.actor.bias)
        torch.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        torch.nn.init.zeros_(self.critic.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/rl/ppo_deleaf.pt"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _gae(rewards, values, dones, bootstrap, gamma: float, lam: float):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = 0.0
    next_value = float(bootstrap)
    for index in reversed(range(len(rewards))):
        continuing = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * continuing - values[index]
        carry = delta + gamma * lam * continuing * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, advantages + values


def train(args: argparse.Namespace) -> None:
    if args.total_steps < 1 or args.rollout_steps < 1 or args.minibatch_size < 1:
        raise ValueError("step and minibatch counts must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with DeleafClient(args.host, args.port) as client:
        model = ActorCritic(client.observation_size, client.action_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        observation, reset_info = client.reset(seed=args.seed)
        print(f"connected: target={reset_info['target']} device={device}")
        total_steps = 0
        episode_index = 0
        episode_return = 0.0

        while total_steps < args.total_steps:
            observations = []
            latent_actions = []
            log_probabilities = []
            rewards = []
            values = []
            dones = []

            count = min(args.rollout_steps, args.total_steps - total_steps)
            for _ in range(count):
                tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                with torch.no_grad():
                    distribution, value = model.distribution_and_value(tensor)
                    latent = distribution.sample()
                    action_mask = rl_policy.phase_action_mask_tensor(tensor)
                    action = torch.tanh(latent) * action_mask
                    log_probability = (
                        distribution.log_prob(latent) * action_mask
                    ).sum()
                next_observation, reward, terminated, truncated, info = client.step(
                    action.cpu().numpy()
                )
                done = terminated or truncated
                observations.append(observation)
                latent_actions.append(latent.cpu().numpy())
                log_probabilities.append(float(log_probability.cpu()))
                rewards.append(reward)
                values.append(float(value.cpu()))
                dones.append(done)
                episode_return += reward
                total_steps += 1
                observation = next_observation
                if done:
                    print(
                        f"episode={episode_index} steps={total_steps} "
                        f"return={episode_return:.3f} phase={info['phase']} "
                        f"reason={info['termination_reason']}"
                    )
                    episode_index += 1
                    episode_return = 0.0
                    observation, _ = client.reset(seed=args.seed + episode_index)

            with torch.no_grad():
                _, bootstrap_value = model.distribution_and_value(
                    torch.as_tensor(observation, dtype=torch.float32, device=device)
                )
            advantages, returns = _gae(
                np.asarray(rewards, dtype=np.float32),
                np.asarray(values, dtype=np.float32),
                np.asarray(dones, dtype=np.float32),
                float(bootstrap_value.cpu()),
                args.gamma,
                args.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            observation_tensor = torch.as_tensor(np.asarray(observations), device=device)
            latent_tensor = torch.as_tensor(np.asarray(latent_actions), device=device)
            old_log_probability = torch.as_tensor(log_probabilities, device=device)
            advantage_tensor = torch.as_tensor(advantages, device=device)
            return_tensor = torch.as_tensor(returns, device=device)
            indices = np.arange(count)

            for _ in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, count, args.minibatch_size):
                    batch = indices[start : start + args.minibatch_size]
                    distribution, value = model.distribution_and_value(
                        observation_tensor[batch]
                    )
                    action_mask = rl_policy.phase_action_mask_tensor(
                        observation_tensor[batch]
                    )
                    log_probability = (
                        distribution.log_prob(latent_tensor[batch]) * action_mask
                    ).sum(-1)
                    ratio = torch.exp(log_probability - old_log_probability[batch])
                    unclipped = ratio * advantage_tensor[batch]
                    clipped = torch.clamp(
                        ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                    ) * advantage_tensor[batch]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(value, return_tensor[batch])
                    entropy = (
                        distribution.entropy() * action_mask
                    ).sum(-1).mean()
                    loss = (
                        policy_loss
                        + args.value_coefficient * value_loss
                        - args.entropy_coefficient * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.maximum_gradient_norm
                    )
                    optimizer.step()

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_steps": total_steps,
                    "episode_index": episode_index,
                    "observation_size": client.observation_size,
                    "action_size": client.action_size,
                    "seed": args.seed,
                },
                args.checkpoint,
            )
            print(f"updated PPO at step={total_steps}; checkpoint={args.checkpoint}")


if __name__ == "__main__":
    train(parse_args())
