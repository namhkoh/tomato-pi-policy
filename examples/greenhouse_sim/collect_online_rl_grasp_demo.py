"""Collect strict physical grasp demonstrations through the online-RL API.

Start ``interactive_greenhouse.py --rl-server`` at the documented 100 mm
SubStem_02 curriculum pose.  This controller does not teleport the robot or
bypass contact: it velocity-controls the validated collision-clear IK route,
then asks the physical gripper to close.  Only trajectories that reach the
strict ``grasped`` task phase are written to the behavior-cloning dataset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import rl_env  # noqa: E402
from greenhouse_sim.rl_client import DeleafClient  # noqa: E402
from greenhouse_sim.robot_kinematics import Rby1Kinematics  # noqa: E402


GRASP_WAYPOINTS_DEGREES = np.asarray(
    [
        [-36.4987161317, -0.9994270697, -34.8408616563, -83.0688385330,
         127.4820280373, -53.0529909849, 51.0633864438],
        [-41.3099029640, 0.3320312852, -34.7162528006, -77.2307820511,
         129.1943859815, -51.9644672180, 46.4778694203],
        [-45.1398727634, -0.1763214636, -33.1438201854, -70.4048332680,
         128.7344251107, -49.1852283916, 49.8046347412],
        [-47.1604784437, -0.7117635794, -32.1769727761, -66.4670708032,
         127.9870718298, -47.5549475324, 52.8771945702],
        [-49.3611373175, -0.9994270697, -31.8221285130, -62.3834985867,
         127.3018258586, -46.2359693493, 61.8268245219],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-arm-speed", type=float, default=35.0)
    parser.add_argument("--position-gain", type=float, default=4.0)
    parser.add_argument("--waypoint-tolerance-degrees", type=float, default=1.25)
    parser.add_argument("--dwell-steps", type=int, default=3)
    parser.add_argument("--maximum-waypoint-steps", type=int, default=80)
    parser.add_argument("--maximum-close-steps", type=int, default=40)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/rl/grasp_expert.npz"),
    )
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser.parse_args()


def decode_left_arm_degrees(
    observation,
    lower_degrees,
    upper_degrees,
) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    lower = np.asarray(lower_degrees, dtype=np.float32)
    upper = np.asarray(upper_degrees, dtype=np.float32)
    if observation.shape != (rl_env.OBSERVATION_SIZE,):
        raise ValueError("observation has the wrong shape")
    if lower.shape != (7,) or upper.shape != (7,) or np.any(upper <= lower):
        raise ValueError("arm limits must be ordered seven-vectors")
    centre = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    return centre + observation[:7] * half_range


def expert_action(
    observation,
    target_degrees,
    lower_degrees,
    upper_degrees,
    *,
    maximum_arm_speed_degrees_s: float,
    position_gain: float,
    close_gripper: bool = False,
) -> np.ndarray:
    """Compute one normalized velocity action from measured joint feedback."""

    if maximum_arm_speed_degrees_s <= 0.0 or position_gain <= 0.0:
        raise ValueError("expert controller rates must be positive")
    target = np.asarray(target_degrees, dtype=np.float32)
    if target.shape != (7,):
        raise ValueError("target_degrees must contain seven values")
    measured = decode_left_arm_degrees(
        observation, lower_degrees, upper_degrees
    )
    action = np.zeros(rl_env.ACTION_SIZE, dtype=np.float32)
    action[:7] = np.clip(
        position_gain * (target - measured) / maximum_arm_speed_degrees_s,
        -1.0,
        1.0,
    )
    if close_gripper:
        action[rl_env.GRIPPER_ACTION_INDEX] = -1.0
    action *= rl_env.phase_action_mask(observation)
    return action


def _phase(observation) -> str:
    phases = np.asarray(observation)[rl_env.PHASE_OBSERVATION_SLICE]
    return rl_env.PHASES[int(np.argmax(phases))]


def _collect_episode(
    client: DeleafClient,
    *,
    seed: int,
    lower_degrees: np.ndarray,
    upper_degrees: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    observation, reset_info = client.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    minimum_distance = float(reset_info["left_grasp_distance_m"])
    final_info = reset_info
    failure = None

    def take_step(action: np.ndarray):
        nonlocal observation, minimum_distance, final_info
        observations.append(observation.copy())
        actions.append(action.copy())
        observation, reward, terminated, truncated, final_info = client.step(action)
        minimum_distance = min(
            minimum_distance, float(final_info["left_grasp_distance_m"])
        )
        return reward, terminated, truncated

    for waypoint_index, target in enumerate(GRASP_WAYPOINTS_DEGREES):
        stable_steps = 0
        for _ in range(args.maximum_waypoint_steps):
            action = expert_action(
                observation,
                target,
                lower_degrees,
                upper_degrees,
                maximum_arm_speed_degrees_s=args.maximum_arm_speed,
                position_gain=args.position_gain,
            )
            _, terminated, truncated = take_step(action)
            if _phase(observation) == "grasped":
                break
            if terminated or truncated:
                failure = final_info.get("termination_reason") or "episode_ended"
                break
            measured = decode_left_arm_degrees(
                observation, lower_degrees, upper_degrees
            )
            position_error = float(np.max(np.abs(target - measured)))
            stable_steps = (
                stable_steps + 1
                if position_error <= args.waypoint_tolerance_degrees
                else 0
            )
            if stable_steps >= args.dwell_steps:
                break
        else:
            failure = f"waypoint_{waypoint_index}_timeout"
        if failure is not None or _phase(observation) == "grasped":
            break

    if failure is None and _phase(observation) != "grasped":
        target = GRASP_WAYPOINTS_DEGREES[-1]
        for _ in range(args.maximum_close_steps):
            action = expert_action(
                observation,
                target,
                lower_degrees,
                upper_degrees,
                maximum_arm_speed_degrees_s=args.maximum_arm_speed,
                position_gain=args.position_gain,
                close_gripper=True,
            )
            _, terminated, truncated = take_step(action)
            if _phase(observation) == "grasped":
                break
            if terminated or truncated:
                failure = final_info.get("termination_reason") or "episode_ended"
                break
        else:
            failure = "grasp_close_timeout"

    accepted = _phase(observation) == "grasped" and failure is None
    if not accepted and failure is None:
        failure = "strict_grasp_not_reached"
    final_measured = decode_left_arm_degrees(
        observation, lower_degrees, upper_degrees
    )
    record = {
        "seed": seed,
        "target": reset_info.get("target"),
        "accepted": accepted,
        "steps": len(actions),
        "minimum_grasp_distance_m": minimum_distance,
        "final_phase": _phase(observation),
        "termination_reason": final_info.get("termination_reason"),
        "failure": failure,
        "final_grasp_force_fraction": float(
            final_info.get("grasp_force_fraction", 0.0)
        ),
        "final_joint_error_degrees": (
            GRASP_WAYPOINTS_DEGREES[-1] - final_measured
        ).tolist(),
    }
    return record, observations, actions


def collect(args: argparse.Namespace) -> dict:
    positive = (
        args.episodes,
        args.maximum_arm_speed,
        args.position_gain,
        args.waypoint_tolerance_degrees,
        args.dwell_steps,
        args.maximum_waypoint_steps,
        args.maximum_close_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("collector counts, rates, and tolerances must be positive")
    lower, upper = Rby1Kinematics().arm_limits_degrees("left")
    report_path = args.report or args.output.with_suffix(".json")
    episode_reports = []
    accepted_observations = []
    accepted_actions = []
    episode_ids = []
    step_ids = []
    with DeleafClient(args.host, args.port, timeout_s=300.0) as client:
        if (
            client.observation_size != rl_env.OBSERVATION_SIZE
            or client.action_size != rl_env.ACTION_SIZE
        ):
            raise RuntimeError("server exposes an incompatible RL contract")
        for episode in range(args.episodes):
            record, observations, actions = _collect_episode(
                client,
                seed=args.seed + episode,
                lower_degrees=lower,
                upper_degrees=upper,
                args=args,
            )
            episode_reports.append(record)
            print(
                f"episode={episode} seed={record['seed']} accepted={record['accepted']} "
                f"steps={record['steps']} phase={record['final_phase']} "
                f"min_grasp={record['minimum_grasp_distance_m']:.4f} "
                f"failure={record['failure']}",
                flush=True,
            )
            if not record["accepted"]:
                continue
            accepted_observations.extend(observations)
            accepted_actions.extend(actions)
            episode_ids.extend([episode] * len(actions))
            step_ids.extend(range(len(actions)))

    report = {
        "schema": "greenhouse.online_rl.grasp_expert.v1",
        "host": args.host,
        "port": args.port,
        "requested_episodes": args.episodes,
        "accepted_episodes": sum(item["accepted"] for item in episode_reports),
        "accepted_transitions": len(accepted_actions),
        "waypoints_degrees": GRASP_WAYPOINTS_DEGREES.tolist(),
        "episodes": episode_reports,
        "dataset": str(args.output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not accepted_actions:
        raise RuntimeError(
            f"no strict grasp demonstrations accepted; see {report_path}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.asarray(accepted_observations, dtype=np.float32),
        actions=np.asarray(accepted_actions, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        step_ids=np.asarray(step_ids, dtype=np.int32),
        arm_lower_degrees=np.asarray(lower, dtype=np.float32),
        arm_upper_degrees=np.asarray(upper, dtype=np.float32),
        waypoints_degrees=GRASP_WAYPOINTS_DEGREES,
    )
    return report


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), indent=2))
