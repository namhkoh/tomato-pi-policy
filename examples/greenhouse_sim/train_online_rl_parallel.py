"""Parallel PPO trainer for independent greenhouse Isaac Sim workers.

Each port must serve one ``interactive_greenhouse.py --headless --rl-server``
process.  Socket calls are issued concurrently, while policy inference and PPO
updates remain batched in one process so every worker improves one checkpoint.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import rl_policy  # noqa: E402
from greenhouse_sim.rl_client import DeleafClient  # noqa: E402
from train_online_rl import ActorCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", type=int, nargs="+", default=(8766, 8767, 8768, 8769))
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=256,
        help="policy steps collected per worker before each PPO update",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--entropy-coefficient", type=float, default=0.003)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--demonstrations", type=pathlib.Path, nargs="+", default=())
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--bc-minibatch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--bc-log-std",
        type=float,
        default=-2.3,
        help="initial PPO log standard deviation after behavior cloning",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/rl/ppo_deleaf_parallel.pt"),
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/rl/ppo_deleaf_parallel.json"),
    )
    return parser.parse_args()


def _vector_gae(rewards, values, dones, bootstrap, gamma: float, lam: float):
    """Compute GAE independently along every worker trajectory."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    bootstrap = np.asarray(bootstrap, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("reward, value, and done arrays must share [time, worker] shape")
    if rewards.ndim != 2 or bootstrap.shape != (rewards.shape[1],):
        raise ValueError("parallel GAE expects [time, worker] and [worker] arrays")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = bootstrap.copy()
    for index in reversed(range(rewards.shape[0])):
        continuing = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * continuing - values[index]
        carry = delta + gamma * lam * continuing * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, advantages + values


def _parallel(executor: ThreadPoolExecutor, calls):
    futures = [executor.submit(function, *arguments) for function, arguments in calls]
    return [future.result() for future in futures]


def _write_report(path: pathlib.Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict:
    worker_count = len(args.ports)
    if worker_count < 1:
        raise ValueError("at least one worker port is required")
    if len(set(args.ports)) != worker_count:
        raise ValueError("worker ports must be unique")
    if any(not 1 <= port <= 65535 for port in args.ports):
        raise ValueError("worker ports must be between 1 and 65535")
    positive = (
        args.total_steps,
        args.rollout_steps,
        args.epochs,
        args.minibatch_size,
        args.learning_rate,
        args.target_kl,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training counts and learning rate must be positive")
    if args.total_steps % worker_count:
        raise ValueError("total steps must be divisible by the worker count")
    if args.demonstrations and (
        args.bc_epochs < 1
        or args.bc_minibatch_size < 1
        or args.bc_learning_rate <= 0.0
        or not np.isfinite(args.bc_log_std)
    ):
        raise ValueError("behavior-cloning settings must be finite and positive")
    if args.bc_only and not args.demonstrations:
        raise ValueError("--bc-only requires --demonstrations")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    clients: list[DeleafClient] = []
    start_time = time.time()
    report = {
        "schema": "greenhouse.online_rl.parallel_ppo.v1",
        "host": args.host,
        "ports": list(args.ports),
        "workers": worker_count,
        "requested_total_steps": args.total_steps,
        "rollout_steps_per_worker": args.rollout_steps,
        "seed": args.seed,
        "device": str(device),
        "episodes": [],
        "updates": [],
        "demonstrations": [str(path) for path in args.demonstrations],
        "status": "connecting",
    }
    _write_report(args.report, report)

    try:
        clients = [DeleafClient(args.host, port, timeout_s=300.0) for port in args.ports]
        observation_size = clients[0].observation_size
        action_size = clients[0].action_size
        if any(
            client.observation_size != observation_size
            or client.action_size != action_size
            for client in clients
        ):
            raise RuntimeError("parallel workers expose different RL contracts")

        model = ActorCritic(observation_size, action_size).to(device)
        behavior_cloning = None
        demonstration_sources = []
        if args.demonstrations:
            demo_observations, demo_actions, demonstration_sources = (
                rl_policy.load_demonstrations(
                    args.demonstrations,
                    observation_size=observation_size,
                    action_size=action_size,
                )
            )
            behavior_cloning = rl_policy.behavior_clone(
                model,
                demo_observations,
                demo_actions,
                epochs=args.bc_epochs,
                minibatch_size=args.bc_minibatch_size,
                learning_rate=args.bc_learning_rate,
                maximum_gradient_norm=args.maximum_gradient_norm,
                device=device,
                seed=args.seed,
            )
            with torch.no_grad():
                model.log_std.fill_(args.bc_log_std)
            report["behavior_cloning"] = behavior_cloning
            report["demonstration_sources"] = demonstration_sources
            _write_report(args.report, report)
            print(f"behavior cloning: {behavior_cloning}", flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.bc_only:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_steps": 0,
                    "update_index": 0,
                    "episode_counts": [0] * worker_count,
                    "observation_size": observation_size,
                    "action_size": action_size,
                    "seed": args.seed,
                    "workers": worker_count,
                    "ports": list(args.ports),
                    "behavior_cloning": behavior_cloning,
                    "demonstration_sources": demonstration_sources,
                    "phase_action_mask": True,
                },
                args.checkpoint,
            )
            report.update(
                status="complete", total_steps=0, successful_episodes=0,
                objective_episodes=0, elapsed_s=time.time() - start_time,
            )
            _write_report(args.report, report)
            return report
        executor = ThreadPoolExecutor(max_workers=worker_count)
        futures = [
            executor.submit(client.reset, seed=args.seed + worker * 1_000_000)
            for worker, client in enumerate(clients)
        ]
        initial = [future.result() for future in futures]


        observations = np.stack([item[0] for item in initial])
        targets = [item[1]["target"] for item in initial]
        episode_counts = np.zeros(worker_count, dtype=np.int64)
        episode_returns = np.zeros(worker_count, dtype=np.float64)
        minimum_grasp_distances = np.full(worker_count, np.inf, dtype=np.float64)
        maximum_action_delta_rms = np.zeros(worker_count, dtype=np.float64)
        total_steps = 0
        update_index = 0
        report.update(
            status="training",
            observation_size=observation_size,
            action_size=action_size,
            targets=targets,
        )
        _write_report(args.report, report)
        print(
            f"connected workers={worker_count} ports={list(args.ports)} "
            f"targets={targets} device={device}",
            flush=True,
        )

        while total_steps < args.total_steps:
            remaining_per_worker = (args.total_steps - total_steps) // worker_count
            time_steps = min(args.rollout_steps, remaining_per_worker)
            rollout_observations = []
            rollout_latents = []
            rollout_log_probabilities = []
            rollout_rewards = []
            rollout_values = []
            rollout_dones = []

            for _ in range(time_steps):
                tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                with torch.no_grad():
                    distribution, value = model.distribution_and_value(tensor)
                    latent = distribution.sample()
                    action_mask = rl_policy.phase_action_mask_tensor(tensor)
                    actions = torch.tanh(latent) * action_mask
                    log_probability = (
                        distribution.log_prob(latent) * action_mask
                    ).sum(-1)
                action_array = actions.cpu().numpy()
                results = _parallel(
                    executor,
                    [
                        (client.step, (action_array[worker],))
                        for worker, client in enumerate(clients)
                    ],
                )

                next_observations = np.stack([result[0] for result in results])
                rewards = np.asarray([result[1] for result in results], dtype=np.float32)
                dones = np.asarray(
                    [result[2] or result[3] for result in results], dtype=np.float32
                )
                rollout_observations.append(observations.copy())
                rollout_latents.append(latent.cpu().numpy())
                rollout_log_probabilities.append(log_probability.cpu().numpy())
                rollout_rewards.append(rewards)
                rollout_values.append(value.cpu().numpy())
                rollout_dones.append(dones)
                episode_returns += rewards
                total_steps += worker_count

                reset_futures = {}
                for worker, result in enumerate(results):
                    info = result[4]
                    minimum_grasp_distances[worker] = min(
                        minimum_grasp_distances[worker],
                        float(info["left_grasp_distance_m"]),
                    )
                    maximum_action_delta_rms[worker] = max(
                        maximum_action_delta_rms[worker],
                        float(info.get("action_delta_rms", 0.0)),
                    )
                    if not dones[worker]:
                        continue
                    episode_record = {
                        "worker": worker,
                        "port": args.ports[worker],
                        "episode": int(episode_counts[worker]),
                        "global_step": total_steps,
                        "return": float(episode_returns[worker]),
                        "phase": info["phase"],
                        "reason": info["termination_reason"],
                        "success": bool(info.get("success")),
                        "objective_reached": bool(info.get("objective_reached")),
                        "minimum_grasp_distance_m": float(
                            minimum_grasp_distances[worker]
                        ),
                        "maximum_action_delta_rms": float(
                            maximum_action_delta_rms[worker]
                        ),
                    }
                    report["episodes"].append(episode_record)
                    print(
                        f"worker={worker} episode={episode_counts[worker]} "
                        f"steps={total_steps} return={episode_returns[worker]:.3f} "
                        f"phase={info['phase']} reason={info['termination_reason']} "
                        f"min_grasp={minimum_grasp_distances[worker]:.4f}",
                        flush=True,
                    )
                    episode_counts[worker] += 1
                    episode_returns[worker] = 0.0
                    minimum_grasp_distances[worker] = np.inf
                    maximum_action_delta_rms[worker] = 0.0
                    reset_seed = (
                        args.seed
                        + worker * 1_000_000
                        + int(episode_counts[worker])
                    )
                    reset_futures[worker] = executor.submit(
                        clients[worker].reset, seed=reset_seed
                    )
                for worker, future in reset_futures.items():
                    next_observations[worker] = future.result()[0]
                observations = next_observations

            with torch.no_grad():
                _, bootstrap = model.distribution_and_value(
                    torch.as_tensor(observations, dtype=torch.float32, device=device)
                )
            advantages, returns = _vector_gae(
                np.asarray(rollout_rewards),
                np.asarray(rollout_values),
                np.asarray(rollout_dones),
                bootstrap.cpu().numpy(),
                args.gamma,
                args.gae_lambda,
            )
            flat_advantages = advantages.reshape(-1)
            flat_advantages = (
                flat_advantages - flat_advantages.mean()
            ) / (flat_advantages.std() + 1e-8)
            observation_tensor = torch.as_tensor(
                np.asarray(rollout_observations).reshape(-1, observation_size),
                dtype=torch.float32,
                device=device,
            )
            latent_tensor = torch.as_tensor(
                np.asarray(rollout_latents).reshape(-1, action_size),
                dtype=torch.float32,
                device=device,
            )
            old_log_probability = torch.as_tensor(
                np.asarray(rollout_log_probabilities).reshape(-1),
                dtype=torch.float32,
                device=device,
            )
            advantage_tensor = torch.as_tensor(
                flat_advantages, dtype=torch.float32, device=device
            )
            return_tensor = torch.as_tensor(
                returns.reshape(-1), dtype=torch.float32, device=device
            )
            sample_count = observation_tensor.shape[0]
            indices = np.arange(sample_count)
            losses = []
            entropies = []
            approximate_kls = []
            clip_fractions = []
            early_stopped = False

            for _ in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, sample_count, args.minibatch_size):
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
                    log_ratio = log_probability - old_log_probability[batch]
                    ratio = torch.exp(log_ratio)
                    approximate_kl = float(
                        ((ratio - 1.0) - log_ratio).mean().detach().cpu()
                    )
                    if approximate_kl > args.target_kl:
                        approximate_kls.append(approximate_kl)
                        early_stopped = True
                        break
                    unclipped = ratio * advantage_tensor[batch]
                    clipped = torch.clamp(
                        ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                    ) * advantage_tensor[batch]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(
                        value, return_tensor[batch]
                    )
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
                    with torch.no_grad():
                        losses.append(float(loss.cpu()))
                        entropies.append(float(entropy.cpu()))
                        approximate_kls.append(approximate_kl)
                        clip_fractions.append(
                            float((torch.abs(ratio - 1.0) > args.clip_ratio).float().mean().cpu())
                        )
                if early_stopped:
                    break

            update_index += 1
            update_record = {
                "update": update_index,
                "total_steps": total_steps,
                "samples": sample_count,
                "mean_loss": float(np.mean(losses)),
                "mean_entropy": float(np.mean(entropies)),
                "mean_approximate_kl": float(np.mean(approximate_kls)),
                "mean_clip_fraction": float(np.mean(clip_fractions)),
                "target_kl": args.target_kl,
                "kl_early_stopped": early_stopped,
                "elapsed_s": time.time() - start_time,
            }
            report["updates"].append(update_record)
            report["total_steps"] = total_steps
            report["successful_episodes"] = sum(
                bool(episode["success"]) for episode in report["episodes"]
            )
            report["objective_episodes"] = sum(
                bool(episode["objective_reached"]) for episode in report["episodes"]
            )
            report["status"] = (
                "complete" if total_steps >= args.total_steps else "training"
            )
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_steps": total_steps,
                    "update_index": update_index,
                    "episode_counts": episode_counts.tolist(),
                    "observation_size": observation_size,
                    "action_size": action_size,
                    "seed": args.seed,
                    "workers": worker_count,
                    "ports": list(args.ports),
                    "behavior_cloning": behavior_cloning,
                    "demonstration_sources": demonstration_sources,
                    "phase_action_mask": True,
                },
                args.checkpoint,
            )
            _write_report(args.report, report)
            print(
                f"update={update_index} steps={total_steps} samples={sample_count} "
                f"loss={update_record['mean_loss']:.4f} "
                f"kl={update_record['mean_approximate_kl']:.6f} "
                f"checkpoint={args.checkpoint}",
                flush=True,
            )
        report["elapsed_s"] = time.time() - start_time
        report["status"] = "complete"
        _write_report(args.report, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_s"] = time.time() - start_time
        _write_report(args.report, report)
        raise
    finally:
        if "executor" in locals():
            close_futures = [executor.submit(client.close) for client in clients]
            for future in close_futures:
                try:
                    future.result()
                except Exception:
                    pass
            executor.shutdown(wait=True)
        else:
            for client in clients:
                try:
                    client.close()
                except Exception:
                    pass


if __name__ == "__main__":
    result = train(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "total_steps": result.get("total_steps"),
                "episodes": len(result["episodes"]),
                "successful_episodes": result.get("successful_episodes", 0),
                "elapsed_s": result.get("elapsed_s"),
            },
            indent=2,
        )
    )
