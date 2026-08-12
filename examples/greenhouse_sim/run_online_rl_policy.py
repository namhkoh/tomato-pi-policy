"""Run a saved PPO checkpoint against the synchronous greenhouse RL server."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim.rl_client import DeleafClient  # noqa: E402
from train_online_rl import ActorCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--maximum-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/rl/policy_trial.json"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.episodes < 1 or args.maximum_steps < 1:
        raise ValueError("episodes and maximum steps must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    trials = []
    with DeleafClient(args.host, args.port) as client:
        model = ActorCritic(client.observation_size, client.action_size).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        remaining = int(args.maximum_steps)
        for episode in range(args.episodes):
            if remaining <= 0:
                break
            observation, reset_info = client.reset(seed=args.seed + episode)
            episode_return = 0.0
            records = []
            terminated = False
            truncated = False
            while not (terminated or truncated) and remaining > 0:
                with torch.no_grad():
                    distribution, _ = model.distribution_and_value(
                        torch.as_tensor(observation, dtype=torch.float32, device=device)
                    )
                    latent = distribution.sample() if args.stochastic else distribution.mean
                    action = torch.tanh(latent).cpu().numpy()
                observation, reward, terminated, truncated, info = client.step(action)
                records.append(
                    {
                        "step": len(records),
                        "reward": reward,
                        "phase": info["phase"],
                        "unsafe_contact_count": info["unsafe_contact_count"],
                        "left_grasp_distance_m": info["left_grasp_distance_m"],
                        "blade_cut_distance_m": info["blade_cut_distance_m"],
                    }
                )
                episode_return += reward
                remaining -= 1
            trials.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "target": reset_info["target"],
                    "steps": len(records),
                    "return": episode_return,
                    "terminated": terminated,
                    "truncated": truncated,
                    "final": info if records else reset_info,
                    "records": records,
                }
            )
    result = {
        "schema": "greenhouse.online_rl.policy_trial.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_training_steps": checkpoint.get("total_steps"),
        "stochastic": bool(args.stochastic),
        "requested_episodes": args.episodes,
        "requested_maximum_steps": args.maximum_steps,
        "trials": trials,
        "successful_trials": sum(
            bool(trial["final"].get("success")) for trial in trials
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
