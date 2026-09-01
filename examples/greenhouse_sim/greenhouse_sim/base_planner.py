"""Deterministic target-conditioned RB-Y1 base pre-positioning.

The greenhouse benchmark does not move the mobile base during an episode.  It
does, however, need a target-conditioned starting pose: a pose accepted for one
petiole can force the left wrist to use a proximal, foliage-intersecting grasp
on another.  This module searches a short, explicit approach line before the
robot is authored and admits a pose only when exact arm IK reaches a distal
physical segment with clearance for the wrist D405 body.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np

from greenhouse_sim import robot_kinematics


@dataclasses.dataclass(frozen=True)
class GraspCandidate:
    collider: str
    body: str
    segment: int
    role: str
    centre_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    axial_offset_m: float = 0.0
    excluded_finger_colliders: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class BaseAttempt:
    advance_m: float
    lateral_m: float
    position_m: tuple[float, float, float]
    collider: str
    segment: int
    grasp_axial_offset_m: float
    grasp_approach_yaw_offset_degrees: float
    grasp_transverse_sign: float
    solution: robot_kinematics.IKResult
    joint_limit_margin_degrees: float
    camera_clearance_m: float
    nearest_obstacle: str | None
    payload_clearance_m: float
    nearest_payload_obstacle: str | None
    inter_arm_clearance_m: float | None
    nearest_inter_arm_pair: str | None
    body_clearance_m: float
    nearest_body_obstacle: str | None
    body_foliage_clearance_m: float
    nearest_body_foliage_obstacle: str | None
    grasp_arm_clearance_m: float
    nearest_grasp_arm_obstacle: str | None
    grasp_arm_foliage_clearance_m: float
    nearest_grasp_arm_foliage_obstacle: str | None
    camera_foliage_clearance_m: float
    nearest_camera_foliage_obstacle: str | None
    payload_foliage_clearance_m: float
    nearest_payload_foliage_obstacle: str | None
    trajectory_arm_clearance_m: float
    nearest_trajectory_arm_obstacle: str | None
    trajectory_camera_clearance_m: float
    nearest_trajectory_camera_obstacle: str | None
    trajectory_payload_clearance_m: float
    nearest_trajectory_payload_obstacle: str | None
    trajectory_arm_foliage_clearance_m: float
    nearest_trajectory_arm_foliage_obstacle: str | None
    trajectory_camera_foliage_clearance_m: float
    nearest_trajectory_camera_foliage_obstacle: str | None
    trajectory_payload_foliage_clearance_m: float
    nearest_trajectory_payload_foliage_obstacle: str | None
    trajectory_inter_arm_clearance_m: float | None
    nearest_trajectory_inter_arm_pair: str | None
    approach_route_index: int
    approach_waypoints_degrees: tuple[tuple[float, ...], ...]
    additional_feasibility: dict | None = None
    approach_route_screens: tuple[dict, ...] = ()


@dataclasses.dataclass(frozen=True)
class BasePlan:
    nominal_position_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    offset_m: tuple[float, float, float]
    advance_m: float
    lateral_m: float
    reach_reserve_m: float
    minimum_segment: int
    selected_grasp_collider: str
    selected_grasp_body: str
    selected_grasp_segment: int
    selected_grasp_axial_offset_m: float
    selected_grasp_approach_yaw_offset_degrees: float
    selected_grasp_transverse_sign: float
    camera_clearance_m: float
    nearest_obstacle: str | None
    payload_clearance_m: float
    nearest_payload_obstacle: str | None
    minimum_inter_arm_clearance_m: float | None
    nearest_inter_arm_pair: str | None
    body_clearance_m: float
    nearest_body_obstacle: str | None
    body_foliage_clearance_m: float
    nearest_body_foliage_obstacle: str | None
    grasp_arm_foliage_clearance_m: float
    nearest_grasp_arm_foliage_obstacle: str | None
    camera_foliage_clearance_m: float
    nearest_camera_foliage_obstacle: str | None
    payload_foliage_clearance_m: float
    nearest_payload_foliage_obstacle: str | None
    trajectory_arm_clearance_m: float
    nearest_trajectory_arm_obstacle: str | None
    trajectory_camera_clearance_m: float
    nearest_trajectory_camera_obstacle: str | None
    trajectory_payload_clearance_m: float
    nearest_trajectory_payload_obstacle: str | None
    trajectory_arm_foliage_clearance_m: float
    nearest_trajectory_arm_foliage_obstacle: str | None
    trajectory_camera_foliage_clearance_m: float
    nearest_trajectory_camera_foliage_obstacle: str | None
    trajectory_payload_foliage_clearance_m: float
    nearest_trajectory_payload_foliage_obstacle: str | None
    trajectory_inter_arm_clearance_m: float | None
    nearest_trajectory_inter_arm_pair: str | None
    approach_route_index: int
    approach_waypoints_degrees: tuple[tuple[float, ...], ...]
    approach_start_degrees: tuple[float, ...] | None
    solution: robot_kinematics.IKResult
    attempts: tuple[BaseAttempt, ...]


def _tool_payload_clearance(
    model,
    joint_degrees,
    base_matrix,
    obstacles,
    payload_boxes,
    excluded_finger_colliders=(),
    *,
    ee_matrix=None,
):
    """Return the minimum vine clearance of every wrist payload box."""
    if not payload_boxes:
        return robot_kinematics.ClearanceResult(float("inf"), None)
    excluded = set(excluded_finger_colliders)
    if ee_matrix is None:
        ee_matrix = model.forward("left", joint_degrees, base_matrix)
    best = robot_kinematics.ClearanceResult(float("inf"), None)
    for component, local_centre, local_rotation, half_extents in payload_boxes:
        component_obstacles = (
            tuple(
                obstacle
                for obstacle in obstacles
                if obstacle.path not in excluded
            )
            if str(component).startswith("ee_finger_")
            else obstacles
        )
        world_centre = (
            ee_matrix
            @ np.append(np.asarray(local_centre, dtype=np.float64), 1.0)
        )[:3]
        world_rotation = ee_matrix[:3, :3] @ np.asarray(
            local_rotation, dtype=np.float64
        )
        clearance = robot_kinematics.oriented_box_capsule_clearance(
            world_centre,
            world_rotation,
            half_extents,
            component_obstacles,
        )
        if clearance.clearance_m < best.clearance_m:
            best = robot_kinematics.ClearanceResult(
                clearance.clearance_m,
                f"{component} <-> {clearance.nearest_obstacle}",
            )
    return best


def _arm_foliage_clearance(
    model,
    side: str,
    joint_degrees,
    base_matrix,
    foliage_obstacles,
):
    if not foliage_obstacles:
        return robot_kinematics.ClearanceResult(float("inf"), None)
    return model.arm_oriented_box_clearance(
        side,
        joint_degrees,
        base_matrix,
        foliage_obstacles,
    )


def _fixed_body_foliage_clearance(model, base_matrix, foliage_obstacles):
    if not foliage_obstacles:
        return robot_kinematics.ClearanceResult(float("inf"), None)
    return model.fixed_body_oriented_box_clearance(
        base_matrix,
        foliage_obstacles,
    )


def _tool_camera_foliage_clearance(
    model,
    joint_degrees,
    base_matrix,
    camera_local_centre_m,
    camera_radius_m: float,
    foliage_obstacles,
):
    if not foliage_obstacles:
        return robot_kinematics.ClearanceResult(float("inf"), None)
    local = np.append(np.asarray(camera_local_centre_m, dtype=np.float64), 1.0)
    centre = (model.forward("left", joint_degrees, base_matrix) @ local)[:3]
    return robot_kinematics.sphere_oriented_box_clearance(
        centre,
        camera_radius_m,
        foliage_obstacles,
    )


def _tool_payload_foliage_clearance(
    model,
    joint_degrees,
    base_matrix,
    foliage_obstacles,
    payload_boxes,
    excluded_finger_colliders=(),
    *,
    ee_matrix=None,
):
    """Require every open-gripper payload box to clear all live foliage.

    ``excluded_finger_colliders`` remains in the call contract because the
    rigid petiole capsule is intentionally allowed between open jaws. Foliage
    is different: exempting target-branch leaf proxies let an approaching
    finger sweep the leaf and displace the petiole before closure.
    """
    if not foliage_obstacles or not payload_boxes:
        return robot_kinematics.ClearanceResult(float("inf"), None)
    if ee_matrix is None:
        ee_matrix = model.forward("left", joint_degrees, base_matrix)
    best = robot_kinematics.ClearanceResult(float("inf"), None)
    for component, local_centre, local_rotation, half_extents in payload_boxes:
        world_centre = (
            ee_matrix
            @ np.append(np.asarray(local_centre, dtype=np.float64), 1.0)
        )[:3]
        world_rotation = ee_matrix[:3, :3] @ np.asarray(
            local_rotation,
            dtype=np.float64,
        )
        clearance = robot_kinematics.oriented_box_oriented_box_clearance(
            world_centre,
            world_rotation,
            half_extents,
            foliage_obstacles,
        )
        if clearance.clearance_m < best.clearance_m:
            best = robot_kinematics.ClearanceResult(
                clearance.clearance_m,
                f"{component} <-> {clearance.nearest_obstacle}",
            )
    return best


def _tool_payload_clears_foliage(
    model,
    joint_degrees,
    base_matrix,
    foliage_obstacles,
    payload_boxes,
    minimum_clearance_m: float,
    *,
    ee_matrix=None,
) -> bool:
    """Threshold-only counterpart to payload foliage diagnostics."""
    if not foliage_obstacles or not payload_boxes:
        return True
    if ee_matrix is None:
        ee_matrix = model.forward("left", joint_degrees, base_matrix)
    for _component, local_centre, local_rotation, half_extents in payload_boxes:
        world_centre = (
            ee_matrix
            @ np.append(np.asarray(local_centre, dtype=np.float64), 1.0)
        )[:3]
        world_rotation = ee_matrix[:3, :3] @ np.asarray(
            local_rotation,
            dtype=np.float64,
        )
        if not robot_kinematics.oriented_box_clears_oriented_boxes(
            world_centre,
            world_rotation,
            half_extents,
            foliage_obstacles,
            minimum_clearance_m,
        ):
            return False
    return True


def _arm_joint_limit_margin_degrees(model, side: str, joint_degrees) -> float:
    """Return authored joint reserve, or a neutral value for test adapters."""
    evaluator = getattr(model, "arm_joint_limit_margin_degrees", None)
    if evaluator is None:
        return float("inf")
    return float(evaluator(side, joint_degrees))


def _sample_grasp_trajectory(
    model,
    *,
    start_degrees,
    waypoint_degrees,
    target_degrees,
    base_matrix,
    obstacles,
    foliage_obstacles,
    camera_local_centre_m,
    camera_radius_m: float,
    payload_boxes,
    excluded_finger_colliders,
    right_waiting_degrees,
    samples: int,
    clearance_thresholds=None,
):
    """Return minimum clearance along every chord of a waypoint route."""
    target = np.asarray(target_degrees, dtype=np.float64)
    start = target if start_degrees is None else np.asarray(start_degrees, dtype=np.float64)
    waypoints = tuple(np.asarray(values, dtype=np.float64) for values in waypoint_degrees)
    if start.shape != (7,) or target.shape != (7,) or any(
        waypoint.shape != (7,) for waypoint in waypoints
    ):
        raise ValueError("approach start, waypoints, and target must contain seven joints")
    arm_best = robot_kinematics.ClearanceResult(float("inf"), None)
    thresholds = (
        None
        if clearance_thresholds is None
        else np.asarray(clearance_thresholds, dtype=np.float64)
    )
    if thresholds is not None and thresholds.shape != (5,):
        raise ValueError("clearance_thresholds must contain five values")
    camera_best = robot_kinematics.ClearanceResult(float("inf"), None)
    payload_best = robot_kinematics.ClearanceResult(float("inf"), None)
    foliage_arm_best = robot_kinematics.ClearanceResult(float("inf"), None)
    foliage_camera_best = robot_kinematics.ClearanceResult(float("inf"), None)
    foliage_payload_best = robot_kinematics.ClearanceResult(float("inf"), None)
    inter_arm_best = (
        None
        if right_waiting_degrees is None
        else robot_kinematics.ClearanceResult(float("inf"), None)
    )
    def route_result():
        return (
            arm_best,
            camera_best,
            payload_best,
            inter_arm_best,
            foliage_arm_best,
            foliage_camera_best,
            foliage_payload_best,
        )

    chord_start = start
    for chord_target in (*waypoints, target):
        for fraction in np.linspace(0.0, 1.0, samples):
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            joints = chord_start + smooth * (chord_target - chord_start)
            arm = model.arm_obstacle_clearance("left", joints, base_matrix, obstacles)
            if arm.clearance_m < arm_best.clearance_m:
                arm_best = arm
            if thresholds is not None and arm_best.clearance_m < thresholds[0]:
                return route_result()
            camera = robot_kinematics.tool_sphere_clearance(
                model,
                "left",
                joints,
                base_matrix,
                camera_local_centre_m,
                camera_radius_m,
                obstacles,
            )
            if camera.clearance_m < camera_best.clearance_m:
                camera_best = camera
            if thresholds is not None and camera_best.clearance_m < thresholds[1]:
                return route_result()
            payload = _tool_payload_clearance(
                model,
                joints,
                base_matrix,
                obstacles,
                payload_boxes,
                excluded_finger_colliders,
            )
            if payload.clearance_m < payload_best.clearance_m:
                payload_best = payload
            if thresholds is not None and payload_best.clearance_m < thresholds[2]:
                return route_result()
            if (
                thresholds is not None
                and callable(getattr(model, "arm_capsules", None))
                and callable(getattr(model, "forward", None))
            ):
                left_capsules = model.arm_capsules("left", joints, base_matrix)
                if not robot_kinematics.capsules_clear_oriented_boxes(
                    left_capsules,
                    foliage_obstacles,
                    thresholds[3],
                ):
                    foliage_arm_best = _arm_foliage_clearance(
                        model,
                        "left",
                        joints,
                        base_matrix,
                        foliage_obstacles,
                    )
                    return route_result()
                ee_matrix = model.forward("left", joints, base_matrix)
                camera_centre = (
                    ee_matrix
                    @ np.append(
                        np.asarray(camera_local_centre_m, dtype=np.float64),
                        1.0,
                    )
                )[:3]
                if not robot_kinematics.sphere_clears_oriented_boxes(
                    camera_centre,
                    camera_radius_m,
                    foliage_obstacles,
                    thresholds[3],
                ):
                    foliage_camera_best = (
                        robot_kinematics.sphere_oriented_box_clearance(
                            camera_centre,
                            camera_radius_m,
                            foliage_obstacles,
                        )
                    )
                    return route_result()
                if not _tool_payload_clears_foliage(
                    model,
                    joints,
                    base_matrix,
                    foliage_obstacles,
                    payload_boxes,
                    thresholds[3],
                    ee_matrix=ee_matrix,
                ):
                    foliage_payload_best = _tool_payload_foliage_clearance(
                        model,
                        joints,
                        base_matrix,
                        foliage_obstacles,
                        payload_boxes,
                        excluded_finger_colliders,
                        ee_matrix=ee_matrix,
                    )
                    return route_result()
                if inter_arm_best is not None:
                    inter_arm = model.inter_arm_clearance(
                        joints,
                        right_waiting_degrees,
                        base_matrix,
                    )
                    if inter_arm.clearance_m < inter_arm_best.clearance_m:
                        inter_arm_best = inter_arm
                    if inter_arm_best.clearance_m < thresholds[4]:
                        return route_result()
                continue
            foliage_arm = _arm_foliage_clearance(
                model,
                "left",
                joints,
                base_matrix,
                foliage_obstacles,
            )
            if foliage_arm.clearance_m < foliage_arm_best.clearance_m:
                foliage_arm_best = foliage_arm
            if thresholds is not None and foliage_arm_best.clearance_m < thresholds[3]:
                return route_result()
            foliage_camera = _tool_camera_foliage_clearance(
                model,
                joints,
                base_matrix,
                camera_local_centre_m,
                camera_radius_m,
                foliage_obstacles,
            )
            if foliage_camera.clearance_m < foliage_camera_best.clearance_m:
                foliage_camera_best = foliage_camera
            if thresholds is not None and foliage_camera_best.clearance_m < thresholds[3]:
                return route_result()
            foliage_payload = _tool_payload_foliage_clearance(
                model,
                joints,
                base_matrix,
                foliage_obstacles,
                payload_boxes,
                excluded_finger_colliders,
            )
            if foliage_payload.clearance_m < foliage_payload_best.clearance_m:
                foliage_payload_best = foliage_payload
            if thresholds is not None and foliage_payload_best.clearance_m < thresholds[3]:
                return route_result()
            if inter_arm_best is not None:
                inter_arm = model.inter_arm_clearance(
                    joints,
                    right_waiting_degrees,
                    base_matrix,
                )
                if inter_arm.clearance_m < inter_arm_best.clearance_m:
                    inter_arm_best = inter_arm
                if (
                    thresholds is not None
                    and inter_arm_best.clearance_m < thresholds[4]
                ):
                    return route_result()
            if thresholds is not None and (
                arm_best.clearance_m < thresholds[0]
                or camera_best.clearance_m < thresholds[1]
                or payload_best.clearance_m < thresholds[2]
                or foliage_arm_best.clearance_m < thresholds[3]
                or foliage_camera_best.clearance_m < thresholds[3]
                or foliage_payload_best.clearance_m < thresholds[3]
                or (
                    inter_arm_best is not None
                    and inter_arm_best.clearance_m < thresholds[4]
                )
            ):
                return (
                    arm_best,
                    camera_best,
                    payload_best,
                    inter_arm_best,
                    foliage_arm_best,
                    foliage_camera_best,
                    foliage_payload_best,
                )
        chord_start = chord_target
    return (
        arm_best,
        camera_best,
        payload_best,
        inter_arm_best,
        foliage_arm_best,
        foliage_camera_best,
        foliage_payload_best,
    )

def _plan_joint_space_route(
    start_degrees,
    goal_degrees,
    lower_degrees,
    upper_degrees,
    is_valid,
    *,
    seed: int,
    max_iterations: int,
    step_fraction: float = 0.10,
    edge_resolution: float = 0.025,
    validation_edge_resolution: float | None = None,
):
    """Deterministic bidirectional RRT-Connect with fine route validation."""
    start = np.asarray(start_degrees, dtype=np.float64)
    goal = np.asarray(goal_degrees, dtype=np.float64)
    lower = np.asarray(lower_degrees, dtype=np.float64)
    upper = np.asarray(upper_degrees, dtype=np.float64)
    span = upper - lower
    if (
        start.shape != goal.shape
        or start.shape != lower.shape
        or lower.shape != upper.shape
        or np.any(span <= 0.0)
    ):
        raise ValueError("joint route bounds and endpoints must have matching positive spans")
    if max_iterations < 1 or step_fraction <= 0.0 or edge_resolution <= 0.0:
        raise ValueError("joint route search parameters must be positive")
    if (
        validation_edge_resolution is not None
        and validation_edge_resolution <= 0.0
    ):
        raise ValueError("route validation resolution must be positive")


    def normalize(values):
        return (np.asarray(values, dtype=np.float64) - lower) / span

    def denormalize(values):
        return lower + np.asarray(values, dtype=np.float64) * span

    def edge_valid(first_normalized, second_normalized, resolution=None):
        delta = np.asarray(second_normalized) - np.asarray(first_normalized)
        spacing = edge_resolution if resolution is None else float(resolution)
        samples = max(2, int(np.ceil(np.linalg.norm(delta) / spacing)) + 1)
        for fraction in np.linspace(0.0, 1.0, samples):
            if not is_valid(denormalize(first_normalized + fraction * delta)):
                return False
        return True

    def nearest_index(tree, sample):
        distances = np.asarray(
            [np.linalg.norm(node - sample) for node in tree["nodes"]],
            dtype=np.float64,
        )
        return int(np.argmin(distances))

    def steer(first, second):
        direction = second - first
        distance = float(np.linalg.norm(direction))
        if distance <= step_fraction:
            return second.copy()
        return first + direction * (step_fraction / distance)

    def append_toward(tree, target, *, connect):
        parent = nearest_index(tree, target)
        current = tree["nodes"][parent]
        while True:
            candidate = steer(current, target)
            if np.allclose(candidate, current) or not edge_valid(current, candidate):
                return None, False
            tree["nodes"].append(candidate)
            tree["parents"].append(parent)
            parent = len(tree["nodes"]) - 1
            reached = bool(np.allclose(candidate, target, atol=1e-12))
            if reached or not connect:
                return parent, reached
            current = candidate

    def root_path(tree, index):
        result = []
        while index >= 0:
            result.append(tree["nodes"][index])
            index = tree["parents"][index]
        result.reverse()
        return result

    start_normalized = normalize(start)
    goal_normalized = normalize(goal)
    if not is_valid(start) or not is_valid(goal):
        return None
    if edge_valid(start_normalized, goal_normalized) and (
        validation_edge_resolution is None
        or edge_valid(
            start_normalized,
            goal_normalized,
            validation_edge_resolution,
        )
    ):
        return ()

    rng = np.random.default_rng(seed)
    trees = [
        {"root": "start", "nodes": [start_normalized], "parents": [-1]},
        {"root": "goal", "nodes": [goal_normalized], "parents": [-1]},
    ]
    connected = None
    for iteration in range(max_iterations):
        active = trees[iteration % 2]
        opposite = trees[(iteration + 1) % 2]
        draw = rng.random()
        if draw < 0.25:
            sample = opposite["nodes"][rng.integers(len(opposite["nodes"]))]
        elif draw < 0.75:
            fraction = rng.random()
            sample = (
                start_normalized
                + fraction * (goal_normalized - start_normalized)
                + rng.normal(0.0, 0.20, size=start.shape)
            )
            sample = np.clip(sample, 0.0, 1.0)
        else:
            sample = rng.random(start.shape)
        active_index, _ = append_toward(active, sample, connect=False)
        if active_index is None:
            continue
        meeting = active["nodes"][active_index]
        opposite_index, reached = append_toward(opposite, meeting, connect=True)
        if reached:
            connected = (active, active_index, opposite, opposite_index)
            break
    if connected is None:
        return None

    active, active_index, opposite, opposite_index = connected
    paths = {
        active["root"]: root_path(active, active_index),
        opposite["root"]: root_path(opposite, opposite_index),
    }
    path = paths["start"] + list(reversed(paths["goal"]))[1:]

    shortened = [path[0]]
    cursor = 0
    while cursor < len(path) - 1:
        next_index = len(path) - 1
        while next_index > cursor + 1 and not edge_valid(path[cursor], path[next_index]):
            next_index -= 1
        shortened.append(path[next_index])
        cursor = next_index
    if validation_edge_resolution is not None and any(
        not edge_valid(
            first,
            second,
            validation_edge_resolution,
        )
        for first, second in zip(shortened, shortened[1:])
    ):
        return None

    return tuple(
        tuple(float(value) for value in denormalize(node))
        for node in shortened[1:-1]
    )

def plan_target_conditioned_base(
    model,
    *,
    nominal_position_m,
    yaw_degrees: float,
    candidates: tuple[GraspCandidate, ...],
    obstacles: tuple[robot_kinematics.CapsuleObstacle, ...],
    foliage_obstacles: tuple[robot_kinematics.OrientedBoxObstacle, ...] = (),
    jaw_local_point_m,
    camera_local_centre_m,
    camera_radius_m: float,
    seeds,
    advances_m=(0.0, 0.03, 0.06, 0.09),
    lateral_offsets_m=(0.0,),
    advance_direction_xy=None,
    left_waiting_degrees=None,
    left_approach_start_degrees=None,
    left_approach_waypoint_routes=((),),
    target_relative_ingress_offsets_m=(),
    target_relative_ingress_gateways_m=(),
    right_waiting_degrees=None,
    reach_reserve_m: float = 0.02,
    minimum_camera_clearance_m: float = 0.005,
    left_payload_boxes=(),
    minimum_payload_clearance_m: float | None = None,
    minimum_inter_arm_clearance_m: float = 0.005,
    minimum_arm_obstacle_clearance_m: float = 0.005,
    minimum_foliage_clearance_m: float = 0.0005,
    minimum_trajectory_clearance_m: float | None = None,
    minimum_body_clearance_m: float = 0.01,
    trajectory_samples: int = 31,
    joint_space_search_iterations: int = 0,
    maximum_joint_space_route_searches: int | None = None,
    joint_space_search_seed: int = 0,
    maximum_ik_evaluations: int = 5000,
    grasp_approach_yaw_offsets_degrees=(0.0,),
    grasp_transverse_signs=(1.0,),
    candidates_are_preferred: bool = False,
    stop_on_first_feasible: bool = False,
    position_feasibility_check=None,
    endpoint_feasibility_check=None,
    additional_feasibility_check=None,
    diagnostics: dict | None = None,
) -> BasePlan | None:
    """Return the safest target-facing base pose with a clear grasp chord."""
    if not candidates:
        return None
    nominal = np.asarray(nominal_position_m, dtype=np.float64)
    if nominal.shape != (3,):
        raise ValueError("nominal_position_m must contain three values")
    trajectory_threshold = (
        minimum_arm_obstacle_clearance_m
        if minimum_trajectory_clearance_m is None
        else float(minimum_trajectory_clearance_m)
    )
    payload_threshold = (
        minimum_camera_clearance_m
        if minimum_payload_clearance_m is None
        else float(minimum_payload_clearance_m)
    )
    if (
        camera_radius_m < 0.0
        or minimum_camera_clearance_m < 0.0
        or payload_threshold < 0.0
        or minimum_inter_arm_clearance_m < 0.0
        or minimum_arm_obstacle_clearance_m < 0.0
        or minimum_foliage_clearance_m < 0.0
        or trajectory_threshold < 0.0
        or minimum_body_clearance_m < 0.0
        or reach_reserve_m < 0.0
    ):
        raise ValueError("camera radius, clearance, and reach reserve must be non-negative")
    if trajectory_samples < 2:
        raise ValueError("trajectory_samples must be at least two")
    if joint_space_search_iterations < 0:
        raise ValueError("joint_space_search_iterations cannot be negative")
    if maximum_ik_evaluations < 1:
        raise ValueError("maximum_ik_evaluations must be positive")
    target_relative_ingress_offsets = tuple(
        float(value) for value in target_relative_ingress_offsets_m
    )
    target_relative_ingress_gateways = tuple(
        tuple(float(component) for component in gateway)
        for gateway in target_relative_ingress_gateways_m
    )
    if (
        any(
            not math.isfinite(value) or value <= 0.0
            for value in target_relative_ingress_offsets
        )
        or any(
            current <= previous
            for previous, current in zip(
                target_relative_ingress_offsets,
                target_relative_ingress_offsets[1:],
            )
        )
    ):
        raise ValueError(
            "target_relative_ingress_offsets_m must be strictly increasing "
            "positive finite distances"
        )
    if any(
        len(gateway) not in (3, 4)
        or not np.isfinite(gateway).all()
        or gateway[0] <= 0.0
        or (len(gateway) == 4 and gateway[3] <= 0.0)
        for gateway in target_relative_ingress_gateways
    ):
        raise ValueError(
            "target_relative_ingress_gateways_m must contain finite "
            "(standoff, lateral, vertical[, taper_power]) tuples with "
            "positive standoff and taper power"
        )
    if target_relative_ingress_gateways and not target_relative_ingress_offsets:
        raise ValueError(
            "target-relative ingress gateways require ingress offsets"
        )
    transverse_signs = tuple(float(value) for value in grasp_transverse_signs)
    if not transverse_signs or any(value not in (-1.0, 1.0) for value in transverse_signs):
        raise ValueError("grasp_transverse_signs must contain only -1 or 1")
    approach_yaw_offsets = tuple(
        float(value) for value in grasp_approach_yaw_offsets_degrees
    )
    if (
        not approach_yaw_offsets
        or not np.isfinite(approach_yaw_offsets).all()
        or any(abs(value) > 60.0 for value in approach_yaw_offsets)
    ):
        raise ValueError(
            "grasp_approach_yaw_offsets_degrees must be finite and within +/-60"
        )
    if (
        maximum_joint_space_route_searches is not None
        and maximum_joint_space_route_searches < 0
    ):
        raise ValueError("maximum_joint_space_route_searches cannot be negative")
    advances = tuple(float(value) for value in advances_m)
    lateral_offsets = tuple(float(value) for value in lateral_offsets_m)
    if not advances or any(value < 0.0 for value in advances):
        raise ValueError("advances_m must contain non-negative distances")
    if not lateral_offsets:
        raise ValueError("lateral_offsets_m cannot be empty")

    # For one organ, retain only its distal two physical links. Automatic
    # target selection supplies one already-preferred distal link per organ;
    # comparing those unrelated local segment indices globally would silently
    # discard valid short petioles whenever another organ has more links.
    if candidates_are_preferred:
        ordered = tuple(candidates)
        distal = ordered
        minimum_segment = min(item.segment for item in ordered)
    else:
        ordered = tuple(
            sorted(candidates, key=lambda item: item.segment, reverse=True)
        )
        minimum_segment = max(0, max(item.segment for item in ordered) - 1)
        distal = tuple(
            item for item in ordered if item.segment >= minimum_segment
        )
    if advance_direction_xy is None:
        target_xy = np.mean(
            np.asarray([item.centre_m[:2] for item in distal]), axis=0
        )
        direction_xy = target_xy - nominal[:2]
    else:
        direction_xy = np.asarray(advance_direction_xy, dtype=np.float64)
        if direction_xy.shape != (2,) or not np.all(np.isfinite(direction_xy)):
            raise ValueError("advance_direction_xy must contain two finite values")
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9:
        raise ValueError("base advance direction must be non-zero")
    direction_xy /= norm
    yaw_rad = np.radians(float(yaw_degrees))
    lateral_xy = np.asarray([-np.sin(yaw_rad), np.cos(yaw_rad)])
    approach_start = (
        left_waiting_degrees
        if left_approach_start_degrees is None
        else left_approach_start_degrees
    )
    approach_routes = tuple(
        tuple(tuple(float(value) for value in waypoint) for waypoint in route)
        for route in left_approach_waypoint_routes
    )
    if not approach_routes:
        raise ValueError("left_approach_waypoint_routes cannot be empty")
    if any(len(waypoint) != 7 for route in approach_routes for waypoint in route):
        raise ValueError("every approach waypoint must contain seven joints")

    attempts: list[BaseAttempt] = []
    feasible: list[dict] = []
    position_rejections: list[dict] = []
    joint_space_route_searches = 0
    stop_search = False
    position_attempts = tuple(
        (advance_m, lateral_m)
        for lateral_m in lateral_offsets
        for advance_m in advances
    )
    for advance_m, lateral_m in position_attempts:
        position = nominal.copy()
        position[:2] += direction_xy * advance_m + lateral_xy * lateral_m
        base_matrix = robot_kinematics.base_transform(position, yaw_degrees)
        body_clearance = model.fixed_body_clearance(base_matrix, obstacles)
        if body_clearance.clearance_m < minimum_body_clearance_m:
            position_rejections.append(
                {
                    "reason": "fixed_body_clearance",
                    "position_m": position.tolist(),
                    "clearance_m": body_clearance.clearance_m,
                    "nearest_obstacle": body_clearance.nearest_obstacle,
                }
            )
            continue

        body_foliage_clearance = _fixed_body_foliage_clearance(
            model,
            base_matrix,
            foliage_obstacles,
        )
        if body_foliage_clearance.clearance_m < minimum_foliage_clearance_m:
            position_rejections.append(
                {
                    "reason": "fixed_body_foliage_clearance",
                    "position_m": position.tolist(),
                    "clearance_m": body_foliage_clearance.clearance_m,
                    "nearest_obstacle": (
                        body_foliage_clearance.nearest_obstacle
                    ),
                }
            )
            continue

        waiting_arm_clearances = tuple(
            model.arm_obstacle_clearance(side, degrees, base_matrix, obstacles)
            for side, degrees in (
                ("left", left_waiting_degrees),
                ("right", right_waiting_degrees),
            )
            if degrees is not None
        )
        unsafe_waiting_arms = tuple(
            clearance
            for clearance in waiting_arm_clearances
            if clearance.clearance_m < minimum_arm_obstacle_clearance_m
        )
        if unsafe_waiting_arms:
            worst = min(unsafe_waiting_arms, key=lambda item: item.clearance_m)
            position_rejections.append(
                {
                    "reason": "waiting_arm_clearance",
                    "position_m": position.tolist(),
                    "clearance_m": worst.clearance_m,
                    "nearest_obstacle": worst.nearest_obstacle,
                }
            )
            continue


        waiting_arm_foliage_clearances = tuple(
            _arm_foliage_clearance(
                model,
                side,
                degrees,
                base_matrix,
                foliage_obstacles,
            )
            for side, degrees in (
                ("left", left_waiting_degrees),
                ("right", right_waiting_degrees),
            )
            if degrees is not None
        )
        unsafe_waiting_foliage = tuple(
            clearance
            for clearance in waiting_arm_foliage_clearances
            if clearance.clearance_m < minimum_foliage_clearance_m
        )
        if unsafe_waiting_foliage:
            worst = min(
                unsafe_waiting_foliage,
                key=lambda item: item.clearance_m,
            )
            position_rejections.append(
                {
                    "reason": "waiting_arm_foliage_clearance",
                    "position_m": position.tolist(),
                    "clearance_m": worst.clearance_m,
                    "nearest_obstacle": worst.nearest_obstacle,
                }
            )
            continue

        position_candidates = distal
        if position_feasibility_check is not None:
            position_additional = dict(
                position_feasibility_check(
                    base_matrix=base_matrix.copy(),
                    position_m=position.copy(),
                    candidates=distal,
                )
            )
            if 'feasible' not in position_additional:
                raise ValueError(
                    'position_feasibility_check must return '
                    'a mapping containing feasible'
                )
            if not bool(position_additional['feasible']):
                position_rejections.append(
                    {
                        'reason': 'additional_position_feasibility',
                        'position_m': position.tolist(),
                        'additional_feasibility': position_additional,
                    }
                )
                continue

            eligible_colliders = position_additional.get(
                'eligible_candidate_colliders'
            )
            if eligible_colliders is not None:
                eligible_colliders = frozenset(
                    str(collider) for collider in eligible_colliders
                )
                position_candidates = tuple(
                    candidate
                    for candidate in distal
                    if candidate.collider in eligible_colliders
                )
                if not position_candidates:
                    position_rejections.append(
                        {
                            'reason': 'additional_position_feasibility',
                            'position_m': position.tolist(),
                            'additional_feasibility': {
                                **position_additional,
                                'feasible': False,
                                'reason': (
                                    'position feasibility admitted no supplied '
                                    'grasp candidate'
                                ),
                            },
                        }
                    )
                    continue

        for candidate in position_candidates:
            candidate_bearing = np.asarray(candidate.centre_m, dtype=np.float64) - position
            candidate_bearing[2] = 0.0
            candidate_bearing_norm = float(np.linalg.norm(candidate_bearing))
            if candidate_bearing_norm <= 1e-9:
                continue
            candidate_bearing /= candidate_bearing_norm
            candidate_reach_reserve = candidate_bearing * reach_reserve_m
            orientation_specs = []
            for approach_yaw_offset in approach_yaw_offsets:
                pointing_direction = robot_kinematics.rotate_horizontal_direction(
                    -candidate_bearing,
                    approach_yaw_offset,
                )
                for transverse_sign in transverse_signs:
                    try:
                        transverse_direction = (
                            robot_kinematics.signed_transverse_direction(
                                pointing_direction,
                                candidate.axis,
                                transverse_sign,
                            )
                        )
                    except ValueError:
                        continue
                    orientation_specs.extend(
                        (
                            approach_yaw_offset,
                            transverse_sign,
                            seed_index,
                            seed_degrees,
                            pointing_direction,
                            transverse_direction,
                        )
                        for seed_index, seed_degrees in enumerate(seeds)
                    )
            for (
                approach_yaw_offset,
                transverse_sign,
                seed_index,
                seed_degrees,
                pointing_direction,
                transverse_direction,
            ) in orientation_specs:
                solution = model.solve_position_axes(
                    "left",
                    local_point_m=jaw_local_point_m,
                    target_point_m=(
                        np.asarray(candidate.centre_m) + candidate_reach_reserve
                    ),
                    seed_degrees=seed_degrees,
                    base_matrix=base_matrix,
                    pointing_axis=2,
                    # The jaw centre is on local -Z. Point +Z back toward the
                    # selected base so -Z enters the crop along the actual
                    # target bearing, including lateral base placements.
                    pointing_direction=pointing_direction,
                    transverse_axis=0,
                    transverse_to=candidate.axis,
                    transverse_direction=transverse_direction,
                    position_scale_m=0.002,
                    maximum_evaluations=maximum_ik_evaluations,
                )
                camera = robot_kinematics.tool_sphere_clearance(
                    model,
                    "left",
                    solution.joint_degrees,
                    base_matrix,
                    camera_local_centre_m,
                    camera_radius_m,
                    obstacles,
                )
                grasp_arm = model.arm_obstacle_clearance(
                    "left", solution.joint_degrees, base_matrix, obstacles
                )
                payload = _tool_payload_clearance(
                    model,
                    solution.joint_degrees,
                    base_matrix,
                    obstacles,
                    left_payload_boxes,
                    candidate.excluded_finger_colliders,
                )
                grasp_arm_foliage = _arm_foliage_clearance(
                    model,
                    "left",
                    solution.joint_degrees,
                    base_matrix,
                    foliage_obstacles,
                )
                camera_foliage = _tool_camera_foliage_clearance(
                    model,
                    solution.joint_degrees,
                    base_matrix,
                    camera_local_centre_m,
                    camera_radius_m,
                    foliage_obstacles,
                )
                payload_foliage = _tool_payload_foliage_clearance(
                    model,
                    solution.joint_degrees,
                    base_matrix,
                    foliage_obstacles,
                    left_payload_boxes,
                    candidate.excluded_finger_colliders,
                )
                inter_arm = (
                    None
                    if right_waiting_degrees is None
                    else model.inter_arm_clearance(
                        solution.joint_degrees,
                        right_waiting_degrees,
                        base_matrix,
                    )
                )
                endpoint_clear = bool(
                    solution.succeeded
                    and camera.clearance_m >= minimum_camera_clearance_m
                    and payload.clearance_m >= payload_threshold
                    and grasp_arm.clearance_m >= minimum_arm_obstacle_clearance_m
                    and grasp_arm_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and camera_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and payload_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and (
                        inter_arm is None
                        or inter_arm.clearance_m >= minimum_inter_arm_clearance_m
                    )
                )
                endpoint_additional = None
                if endpoint_clear and endpoint_feasibility_check is not None:
                    endpoint_additional = dict(
                        endpoint_feasibility_check(
                            base_matrix=base_matrix.copy(),
                            position_m=position.copy(),
                            candidate=candidate,
                            left_solution=solution,
                        )
                    )
                    if 'feasible' not in endpoint_additional:
                        raise ValueError(
                            'endpoint_feasibility_check must return '
                            'a mapping containing feasible'
                        )
                    endpoint_clear = bool(
                        endpoint_additional['feasible']
                    )
                route_evaluations = []
                candidate_approach_routes = approach_routes
                if endpoint_clear:
                    # Joint interpolation directly into a safe grasp endpoint
                    # can bow the distal links into a neighbouring vine. Build
                    # a target-relative Cartesian pre-grasp ladder so the last
                    # 60 mm follows the audited jaw approach instead.
                    target_relative_solutions = []
                    target_relative_seed = solution.joint_degrees
                    target_point = (
                        np.asarray(candidate.centre_m, dtype=np.float64)
                        + candidate_reach_reserve
                    )
                    for offset_m in target_relative_ingress_offsets:
                        waypoint_solution = model.solve_position_axes(
                            "left",
                            local_point_m=jaw_local_point_m,
                            target_point_m=(
                                target_point - offset_m * candidate_bearing
                            ),
                            seed_degrees=target_relative_seed,
                            base_matrix=base_matrix,
                            pointing_axis=2,
                            pointing_direction=pointing_direction,
                            transverse_axis=0,
                            transverse_to=candidate.axis,
                            transverse_direction=transverse_direction,
                            position_scale_m=0.002,
                            maximum_evaluations=maximum_ik_evaluations,
                        )
                        if not waypoint_solution.succeeded:
                            target_relative_solutions = []
                            break
                        target_relative_solutions.append(
                            waypoint_solution.joint_degrees
                        )
                        target_relative_seed = waypoint_solution.joint_degrees
                    if (
                        target_relative_ingress_offsets
                        and len(target_relative_solutions)
                        == len(target_relative_ingress_offsets)
                    ):
                        target_relative_ladder = tuple(
                            reversed(target_relative_solutions)
                        )
                        gateway_routes = []
                        lateral_direction = np.asarray(
                            (-candidate_bearing[1], candidate_bearing[0], 0.0),
                            dtype=np.float64,
                        )
                        vertical_direction = np.asarray(
                            (0.0, 0.0, 1.0),
                            dtype=np.float64,
                        )
                        for gateway in target_relative_ingress_gateways:
                            standoff_m, lateral_m, vertical_m = gateway[:3]
                            taper_power = (
                                1.0 if len(gateway) == 3 else gateway[3]
                            )
                            lane_solutions = []
                            lane_seed = solution.joint_degrees
                            far_offset_m = target_relative_ingress_offsets[-1]
                            for offset_m in target_relative_ingress_offsets:
                                lane_solution = model.solve_position_axes(
                                    "left",
                                    local_point_m=jaw_local_point_m,
                                    target_point_m=(
                                        target_point
                                        - offset_m * candidate_bearing
                                        + lateral_m
                                        * (offset_m / far_offset_m) ** taper_power
                                        * lateral_direction
                                        + vertical_m
                                        * (offset_m / far_offset_m) ** taper_power
                                        * vertical_direction
                                    ),
                                    seed_degrees=lane_seed,
                                    base_matrix=base_matrix,
                                    pointing_axis=2,
                                    pointing_direction=pointing_direction,
                                    transverse_axis=0,
                                    transverse_to=candidate.axis,
                                    transverse_direction=transverse_direction,
                                    position_scale_m=0.002,
                                    maximum_evaluations=maximum_ik_evaluations,
                                )
                                if not lane_solution.succeeded:
                                    lane_solutions = []
                                    break
                                lane_solutions.append(
                                    lane_solution.joint_degrees
                                )
                                lane_seed = lane_solution.joint_degrees
                            if len(lane_solutions) != len(
                                target_relative_ingress_offsets
                            ):
                                continue
                            gateway_solution = model.solve_position_axes(
                                "left",
                                local_point_m=jaw_local_point_m,
                                target_point_m=(
                                    target_point
                                    - standoff_m * candidate_bearing
                                    + lateral_m * lateral_direction
                                    + vertical_m * vertical_direction
                                ),
                                seed_degrees=lane_solutions[-1],
                                base_matrix=base_matrix,
                                pointing_axis=2,
                                pointing_direction=pointing_direction,
                                transverse_axis=0,
                                transverse_to=candidate.axis,
                                transverse_direction=transverse_direction,
                                position_scale_m=0.002,
                                maximum_evaluations=maximum_ik_evaluations,
                            )
                            if gateway_solution.succeeded:
                                gateway_routes.append(
                                    (
                                        gateway_solution.joint_degrees,
                                        *reversed(lane_solutions),
                                    )
                                )
                        candidate_approach_routes = (
                            *approach_routes,
                            target_relative_ladder,
                            *gateway_routes,
                        )

                    def sample_route(route_waypoints, sample_count):
                        return _sample_grasp_trajectory(
                            model,
                            start_degrees=approach_start,
                            waypoint_degrees=route_waypoints,
                            target_degrees=solution.joint_degrees,
                            base_matrix=base_matrix,
                            obstacles=obstacles,
                            foliage_obstacles=foliage_obstacles,
                            camera_local_centre_m=camera_local_centre_m,
                            camera_radius_m=camera_radius_m,
                            payload_boxes=left_payload_boxes,
                            excluded_finger_colliders=(
                                candidate.excluded_finger_colliders
                            ),
                            right_waiting_degrees=right_waiting_degrees,
                            samples=sample_count,
                            clearance_thresholds=(
                                trajectory_threshold,
                                minimum_camera_clearance_m,
                                payload_threshold,
                                minimum_foliage_clearance_m,
                                minimum_inter_arm_clearance_m,
                            ),
                        )

                    for route_index, route_waypoints in enumerate(
                        candidate_approach_routes
                    ):
                        print(
                            f"[base-planner] screening preset ingress "
                            f"{route_index + 1}/{len(candidate_approach_routes)}",
                            flush=True,
                        )
                        coarse_samples = min(7, trajectory_samples)
                        route_metrics = sample_route(
                            route_waypoints,
                            coarse_samples,
                        )
                        coarse_route_clear = bool(
                            route_metrics[0].clearance_m >= trajectory_threshold
                            and route_metrics[1].clearance_m
                            >= minimum_camera_clearance_m
                            and route_metrics[2].clearance_m >= payload_threshold
                            and route_metrics[4].clearance_m
                            >= minimum_foliage_clearance_m
                            and route_metrics[5].clearance_m
                            >= minimum_foliage_clearance_m
                            and route_metrics[6].clearance_m
                            >= minimum_foliage_clearance_m
                            and (
                                route_metrics[3] is None
                                or route_metrics[3].clearance_m
                                >= minimum_inter_arm_clearance_m
                            )
                        )
                        if coarse_route_clear and coarse_samples < trajectory_samples:
                            route_metrics = sample_route(
                                route_waypoints,
                                trajectory_samples,
                            )
                        (
                            route_arm,
                            route_camera,
                            route_payload,
                            route_inter_arm,
                            route_arm_foliage,
                            route_camera_foliage,
                            route_payload_foliage,
                        ) = route_metrics
                        route_is_clear = bool(
                            route_arm.clearance_m >= trajectory_threshold
                            and route_camera.clearance_m
                            >= minimum_camera_clearance_m
                            and route_payload.clearance_m >= payload_threshold
                            and route_arm_foliage.clearance_m
                            >= minimum_foliage_clearance_m
                            and route_camera_foliage.clearance_m
                            >= minimum_foliage_clearance_m
                            and route_payload_foliage.clearance_m
                            >= minimum_foliage_clearance_m
                            and (
                                route_inter_arm is None
                                or route_inter_arm.clearance_m
                                >= minimum_inter_arm_clearance_m
                            )
                        )
                        route_bottlenecks = [
                            ("arm", route_arm),
                            ("camera", route_camera),
                            ("payload", route_payload),
                            ("arm_foliage", route_arm_foliage),
                            ("camera_foliage", route_camera_foliage),
                            ("payload_foliage", route_payload_foliage),
                        ]
                        if route_inter_arm is not None:
                            route_bottlenecks.append(
                                ("inter_arm", route_inter_arm)
                            )
                        route_bottleneck_name, route_bottleneck = min(
                            route_bottlenecks,
                            key=lambda item: item[1].clearance_m,
                        )

                        route_evaluations.append(
                            {
                                "index": route_index,
                                "waypoints": route_waypoints,
                                "arm": route_arm,
                                "camera": route_camera,
                                "payload": route_payload,
                                "inter_arm": route_inter_arm,
                                "arm_foliage": route_arm_foliage,
                                "camera_foliage": route_camera_foliage,
                                "payload_foliage": route_payload_foliage,
                                "bottleneck": route_bottleneck_name,
                                "bottleneck_nearest": (
                                    route_bottleneck.nearest_obstacle
                                ),
                                "clearance_m": min(
                                    route_arm.clearance_m,
                                    route_camera.clearance_m,
                                    route_payload.clearance_m,
                                    float("inf")
                                    if route_inter_arm is None
                                    else route_inter_arm.clearance_m,
                                    route_arm_foliage.clearance_m,
                                    route_camera_foliage.clearance_m,
                                    route_payload_foliage.clearance_m,
                                ),
                            }
                        )
                        print(
                            f"[base-planner] preset ingress {route_index + 1} "
                            f"{'accepted' if route_is_clear else 'rejected'} "
                            f"bottleneck={route_bottleneck_name} "
                            f"clearance={route_bottleneck.clearance_m * 1000.0:.3f}mm "
                            f"nearest={route_bottleneck.nearest_obstacle}",
                            flush=True,
                        )
                        if (
                            route_is_clear
                            and (
                                stop_on_first_feasible
                                or maximum_joint_space_route_searches == 0
                            )
                        ):
                            break
                else:
                    skipped = robot_kinematics.ClearanceResult(
                        float("-inf"),
                        "endpoint_not_clear",
                    )
                    route_evaluations.append(
                        {
                            "index": 0,
                            "waypoints": (),
                            "arm": skipped,
                            "camera": skipped,
                            "payload": skipped,
                            "inter_arm": None,
                            "arm_foliage": skipped,
                            "camera_foliage": skipped,
                            "payload_foliage": skipped,
                            "clearance_m": float("-inf"),
                        }
                    )
                preset_route = max(
                    route_evaluations,
                    key=lambda record: record["clearance_m"],
                )
                preset_route_clear = bool(
                    preset_route["arm"].clearance_m >= trajectory_threshold
                    and preset_route["camera"].clearance_m
                    >= minimum_camera_clearance_m
                    and preset_route["payload"].clearance_m
                    >= payload_threshold
                    and preset_route["arm_foliage"].clearance_m
                    >= minimum_foliage_clearance_m
                    and preset_route["camera_foliage"].clearance_m
                    >= minimum_foliage_clearance_m
                    and preset_route["payload_foliage"].clearance_m
                    >= minimum_foliage_clearance_m
                    and (
                        preset_route["inter_arm"] is None
                        or preset_route["inter_arm"].clearance_m
                        >= minimum_inter_arm_clearance_m
                    )
                )
                if (
                    endpoint_clear
                    and not preset_route_clear
                    and approach_start is not None
                    and joint_space_search_iterations > 0
                    and (
                        maximum_joint_space_route_searches is None
                        or joint_space_route_searches
                        < maximum_joint_space_route_searches
                    )
                ):
                    joint_space_route_searches += 1
                    lower_degrees, upper_degrees = model.arm_limits_degrees("left")
                    route_search_foliage_threshold = (
                        minimum_foliage_clearance_m + 0.001
                    )
                    reuse_transforms = all(
                        callable(getattr(model, method, None))
                        for method in ("arm_capsules", "forward")
                    )
                    right_waiting_capsules = (
                        model.arm_capsules(
                            "right",
                            right_waiting_degrees,
                            base_matrix,
                        )
                        if reuse_transforms and right_waiting_degrees is not None
                        else None
                    )

                    @functools.lru_cache(maxsize=None)
                    def joint_configuration_clear_cached(values):
                        left_capsules = None
                        ee_matrix = None
                        if reuse_transforms:
                            left_capsules = model.arm_capsules(
                                "left",
                                values,
                                base_matrix,
                            )
                            arm_result = robot_kinematics.capsule_sets_clearance(
                                left_capsules,
                                obstacles,
                            )
                        else:
                            arm_result = model.arm_obstacle_clearance(
                                "left", values, base_matrix, obstacles
                            )
                        if arm_result.clearance_m < trajectory_threshold:
                            return False
                        if reuse_transforms:
                            ee_matrix = model.forward("left", values, base_matrix)
                            camera_centre = (
                                ee_matrix
                                @ np.append(
                                    np.asarray(
                                        camera_local_centre_m,
                                        dtype=np.float64,
                                    ),
                                    1.0,
                                )
                            )[:3]
                            camera_result = robot_kinematics.sphere_capsule_clearance(
                                camera_centre,
                                camera_radius_m,
                                obstacles,
                            )
                        else:
                            camera_result = robot_kinematics.tool_sphere_clearance(
                                model,
                                "left",
                                values,
                                base_matrix,
                                camera_local_centre_m,
                                camera_radius_m,
                                obstacles,
                            )
                        if camera_result.clearance_m < minimum_camera_clearance_m:
                            return False
                        payload_result = _tool_payload_clearance(
                            model,
                            values,
                            base_matrix,
                            obstacles,
                            left_payload_boxes,
                            candidate.excluded_finger_colliders,
                            ee_matrix=ee_matrix,
                        )
                        if payload_result.clearance_m < payload_threshold:
                            return False
                        if reuse_transforms:
                            if not robot_kinematics.capsules_clear_oriented_boxes(
                                left_capsules,
                                foliage_obstacles,
                                route_search_foliage_threshold,
                            ):
                                return False
                            if not robot_kinematics.sphere_clears_oriented_boxes(
                                camera_centre,
                                camera_radius_m,
                                foliage_obstacles,
                                route_search_foliage_threshold,
                            ):
                                return False
                            if not _tool_payload_clears_foliage(
                                model,
                                values,
                                base_matrix,
                                foliage_obstacles,
                                left_payload_boxes,
                                route_search_foliage_threshold,
                                ee_matrix=ee_matrix,
                            ):
                                return False
                        if (
                            not reuse_transforms
                            and
                            _arm_foliage_clearance(
                                model,
                                "left",
                                values,
                                base_matrix,
                                foliage_obstacles,
                            ).clearance_m
                            < route_search_foliage_threshold
                        ):
                            return False
                        if (
                            not reuse_transforms
                            and
                            _tool_camera_foliage_clearance(
                                model,
                                values,
                                base_matrix,
                                camera_local_centre_m,
                                camera_radius_m,
                                foliage_obstacles,
                            ).clearance_m
                            < route_search_foliage_threshold
                        ):
                            return False
                        if (
                            not reuse_transforms
                            and
                            _tool_payload_foliage_clearance(
                                model,
                                values,
                                base_matrix,
                                foliage_obstacles,
                                left_payload_boxes,
                                candidate.excluded_finger_colliders,
                            ).clearance_m
                            < route_search_foliage_threshold
                        ):
                            return False
                        if right_waiting_degrees is None:
                            return True
                        if reuse_transforms:
                            return (
                                robot_kinematics.capsule_sets_clearance(
                                    left_capsules,
                                    right_waiting_capsules,
                                ).clearance_m
                                >= minimum_inter_arm_clearance_m
                            )
                        return (
                            model.inter_arm_clearance(
                                values,
                                right_waiting_degrees,
                                base_matrix,
                            ).clearance_m
                            >= minimum_inter_arm_clearance_m
                        )

                    def joint_configuration_clear(values):
                        # RRT-Connect revisits tree nodes while connecting and
                        # shortening.  The settled scene is immutable here, so
                        # exact joint-state clearance results are reusable.
                        return joint_configuration_clear_cached(
                            tuple(float(value) for value in values)
                        )

                    generated_waypoints = _plan_joint_space_route(
                        approach_start,
                        solution.joint_degrees,
                        lower_degrees,
                        upper_degrees,
                        joint_configuration_clear,
                        seed=(
                            int(joint_space_search_seed)
                            + 7919 * len(attempts)
                            + 101 * seed_index
                            + candidate.segment
                        ),
                        max_iterations=joint_space_search_iterations,
                        edge_resolution=0.075,
                        validation_edge_resolution=0.025,
                    )
                    if generated_waypoints is not None:
                        (
                            route_arm,
                            route_camera,
                            route_payload,
                            route_inter_arm,
                            route_arm_foliage,
                            route_camera_foliage,
                            route_payload_foliage,
                        ) = _sample_grasp_trajectory(
                                model,
                                start_degrees=approach_start,
                                waypoint_degrees=generated_waypoints,
                                target_degrees=solution.joint_degrees,
                                base_matrix=base_matrix,
                                obstacles=obstacles,
                                foliage_obstacles=foliage_obstacles,
                                camera_local_centre_m=camera_local_centre_m,
                                camera_radius_m=camera_radius_m,
                                payload_boxes=left_payload_boxes,
                                excluded_finger_colliders=(
                                    candidate.excluded_finger_colliders
                                ),
                                right_waiting_degrees=right_waiting_degrees,
                                samples=trajectory_samples,
                        )
                        route_evaluations.append(
                            {
                                "index": len(candidate_approach_routes),
                                "waypoints": generated_waypoints,
                                "arm": route_arm,
                                "camera": route_camera,
                                "payload": route_payload,
                                "inter_arm": route_inter_arm,
                                "arm_foliage": route_arm_foliage,
                                "camera_foliage": route_camera_foliage,
                                "payload_foliage": route_payload_foliage,
                                "clearance_m": min(
                                    route_arm.clearance_m,
                                    route_camera.clearance_m,
                                    route_payload.clearance_m,
                                    float("inf")
                                    if route_inter_arm is None
                                    else route_inter_arm.clearance_m,
                                    route_arm_foliage.clearance_m,
                                    route_camera_foliage.clearance_m,
                                    route_payload_foliage.clearance_m,
                                ),
                            }
                        )
                selected_route = max(
                    route_evaluations,
                    key=lambda record: record["clearance_m"],
                )
                route_arm = selected_route["arm"]
                route_camera = selected_route["camera"]
                route_payload = selected_route["payload"]
                route_inter_arm = selected_route["inter_arm"]
                route_arm_foliage = selected_route["arm_foliage"]
                route_camera_foliage = selected_route["camera_foliage"]
                route_payload_foliage = selected_route["payload_foliage"]
                attempt = BaseAttempt(
                    advance_m=advance_m,
                    lateral_m=lateral_m,
                    position_m=tuple(float(value) for value in position),
                    collider=candidate.collider,
                    segment=candidate.segment,
                    grasp_axial_offset_m=candidate.axial_offset_m,
                    grasp_approach_yaw_offset_degrees=approach_yaw_offset,
                    grasp_transverse_sign=transverse_sign,
                    solution=solution,
                    joint_limit_margin_degrees=(
                        _arm_joint_limit_margin_degrees(
                            model,
                            "left", solution.joint_degrees
                        )
                    ),
                    camera_clearance_m=camera.clearance_m,
                    nearest_obstacle=camera.nearest_obstacle,
                    payload_clearance_m=payload.clearance_m,
                    nearest_payload_obstacle=payload.nearest_obstacle,
                    body_clearance_m=body_clearance.clearance_m,
                    nearest_body_obstacle=body_clearance.nearest_obstacle,
                    body_foliage_clearance_m=(
                        body_foliage_clearance.clearance_m
                    ),
                    nearest_body_foliage_obstacle=(
                        body_foliage_clearance.nearest_obstacle
                    ),
                    grasp_arm_clearance_m=grasp_arm.clearance_m,
                    nearest_grasp_arm_obstacle=grasp_arm.nearest_obstacle,
                    grasp_arm_foliage_clearance_m=(
                        grasp_arm_foliage.clearance_m
                    ),
                    nearest_grasp_arm_foliage_obstacle=(
                        grasp_arm_foliage.nearest_obstacle
                    ),
                    camera_foliage_clearance_m=camera_foliage.clearance_m,
                    nearest_camera_foliage_obstacle=(
                        camera_foliage.nearest_obstacle
                    ),
                    payload_foliage_clearance_m=payload_foliage.clearance_m,
                    nearest_payload_foliage_obstacle=(
                        payload_foliage.nearest_obstacle
                    ),
                    inter_arm_clearance_m=(
                        None if inter_arm is None else inter_arm.clearance_m
                    ),
                    nearest_inter_arm_pair=(
                        None if inter_arm is None else inter_arm.nearest_obstacle
                    ),
                    trajectory_arm_clearance_m=route_arm.clearance_m,
                    nearest_trajectory_arm_obstacle=route_arm.nearest_obstacle,
                    trajectory_camera_clearance_m=route_camera.clearance_m,
                    nearest_trajectory_camera_obstacle=route_camera.nearest_obstacle,
                    trajectory_payload_clearance_m=route_payload.clearance_m,
                    nearest_trajectory_payload_obstacle=(
                        route_payload.nearest_obstacle
                    ),
                    trajectory_arm_foliage_clearance_m=(
                        route_arm_foliage.clearance_m
                    ),
                    nearest_trajectory_arm_foliage_obstacle=(
                        route_arm_foliage.nearest_obstacle
                    ),
                    trajectory_camera_foliage_clearance_m=(
                        route_camera_foliage.clearance_m
                    ),
                    nearest_trajectory_camera_foliage_obstacle=(
                        route_camera_foliage.nearest_obstacle
                    ),
                    trajectory_payload_foliage_clearance_m=(
                        route_payload_foliage.clearance_m
                    ),
                    nearest_trajectory_payload_foliage_obstacle=(
                        route_payload_foliage.nearest_obstacle
                    ),
                    trajectory_inter_arm_clearance_m=(
                        None
                        if route_inter_arm is None
                        else route_inter_arm.clearance_m
                    ),
                    nearest_trajectory_inter_arm_pair=(
                        None
                        if route_inter_arm is None
                        else route_inter_arm.nearest_obstacle
                    ),
                    approach_route_index=selected_route["index"],
                    approach_waypoints_degrees=selected_route["waypoints"],
                    additional_feasibility=endpoint_additional,
                    approach_route_screens=tuple(
                        {
                            "index": route["index"],
                            "bottleneck": route.get("bottleneck"),
                            "bottleneck_nearest": route.get(
                                "bottleneck_nearest"
                            ),
                            "clearance_m": route["clearance_m"],
                            "arm_clearance_m": route["arm"].clearance_m,
                            "camera_clearance_m": route[
                                "camera"
                            ].clearance_m,
                            "payload_clearance_m": route[
                                "payload"
                            ].clearance_m,
                            "inter_arm_clearance_m": (
                                None
                                if route["inter_arm"] is None
                                else route["inter_arm"].clearance_m
                            ),
                            "arm_foliage_clearance_m": route[
                                "arm_foliage"
                            ].clearance_m,
                            "camera_foliage_clearance_m": route[
                                "camera_foliage"
                            ].clearance_m,
                            "payload_foliage_clearance_m": route[
                                "payload_foliage"
                            ].clearance_m,
                        }
                        for route in route_evaluations
                    ),
                )
                attempts.append(attempt)
                if (
                    endpoint_additional is not None
                    and not endpoint_clear
                    and bool(endpoint_additional.get("prune_candidate"))
                ):
                    break

                if (
                    solution.succeeded
                    and camera.clearance_m >= minimum_camera_clearance_m
                    and payload.clearance_m >= payload_threshold
                    and grasp_arm.clearance_m >= minimum_arm_obstacle_clearance_m
                    and grasp_arm_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and camera_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and payload_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and route_arm.clearance_m >= trajectory_threshold
                    and route_camera.clearance_m >= minimum_camera_clearance_m
                    and route_payload.clearance_m >= payload_threshold
                    and route_arm_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and route_camera_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and route_payload_foliage.clearance_m
                    >= minimum_foliage_clearance_m
                    and (
                        inter_arm is None
                        or inter_arm.clearance_m >= minimum_inter_arm_clearance_m
                    )
                    and (
                        route_inter_arm is None
                        or route_inter_arm.clearance_m
                        >= minimum_inter_arm_clearance_m
                    )
                ):
                    if additional_feasibility_check is not None:
                        additional = dict(
                            additional_feasibility_check(
                                base_matrix=base_matrix.copy(),
                                position_m=position.copy(),
                                candidate=candidate,
                                left_solution=solution,
                                left_route_waypoints=selected_route[
                                    "waypoints"
                                ],
                            )
                        )
                        if "feasible" not in additional:
                            raise ValueError(
                                "additional_feasibility_check must return "
                                "a mapping containing 'feasible'"
                            )
                        attempt = dataclasses.replace(
                            attempt,
                            additional_feasibility=additional,
                        )
                        attempts[-1] = attempt
                        if not bool(additional["feasible"]):
                            continue
                    clearances = [
                        body_clearance.clearance_m,
                        camera.clearance_m,
                        payload.clearance_m,
                        grasp_arm.clearance_m,
                        route_arm.clearance_m,
                        route_camera.clearance_m,
                        route_payload.clearance_m,
                        body_foliage_clearance.clearance_m,
                        grasp_arm_foliage.clearance_m,
                        camera_foliage.clearance_m,
                        payload_foliage.clearance_m,
                        route_arm_foliage.clearance_m,
                        route_camera_foliage.clearance_m,
                        route_payload_foliage.clearance_m,
                    ]
                    if inter_arm is not None:
                        clearances.append(inter_arm.clearance_m)
                    if route_inter_arm is not None:
                        clearances.append(route_inter_arm.clearance_m)
                    feasible.append(
                        {
                            "attempt": attempt,
                            "candidate": candidate,
                            "position": position.copy(),
                            "safety_clearance_m": min(clearances),
                            "joint_limit_margin_degrees": (
                                attempt.joint_limit_margin_degrees
                            ),
                        }
                    )
                    if stop_on_first_feasible:
                        stop_search = True
                        break
            if stop_search:
                break
        if stop_search:
            break

    selected = None
    if feasible:
        selected = max(
            feasible,
            key=lambda record: (
                record["joint_limit_margin_degrees"] >= 5.0,
                min(record["joint_limit_margin_degrees"], 15.0),
                record["safety_clearance_m"],
                record["candidate"].segment,
                -abs(record["attempt"].lateral_m),
                -record["attempt"].advance_m,
                -record["attempt"].solution.cost,
            ),
        )
    if diagnostics is not None:
        diagnostics.update(
            minimum_segment=minimum_segment,
            trajectory_samples=trajectory_samples,
            minimum_trajectory_clearance_m=trajectory_threshold,
            minimum_payload_clearance_m=payload_threshold,
            minimum_foliage_clearance_m=minimum_foliage_clearance_m,
            foliage_obstacles=len(foliage_obstacles),
            joint_space_route_searches=joint_space_route_searches,
            maximum_joint_space_route_searches=maximum_joint_space_route_searches,
            maximum_ik_evaluations=maximum_ik_evaluations,
            grasp_approach_yaw_offsets_degrees=approach_yaw_offsets,
            grasp_transverse_signs=transverse_signs,
            candidates_are_preferred=bool(candidates_are_preferred),
            stop_on_first_feasible=bool(stop_on_first_feasible),
            position_rejections=position_rejections,
            attempts=[dataclasses.asdict(attempt) for attempt in attempts],
            feasible_attempts=len(feasible),
        )
    if selected is None:
        return None

    attempt = selected["attempt"]
    candidate = selected["candidate"]
    if diagnostics is not None:
        diagnostics["selected_attempt"] = dataclasses.asdict(attempt)
    return BasePlan(
        nominal_position_m=tuple(float(value) for value in nominal),
        position_m=tuple(float(value) for value in selected["position"]),
        offset_m=tuple(float(value) for value in selected["position"] - nominal),
        advance_m=attempt.advance_m,
        lateral_m=attempt.lateral_m,
        reach_reserve_m=reach_reserve_m,
        minimum_segment=minimum_segment,
        selected_grasp_collider=candidate.collider,
        selected_grasp_body=candidate.body,
        selected_grasp_segment=candidate.segment,
        selected_grasp_axial_offset_m=candidate.axial_offset_m,
        selected_grasp_approach_yaw_offset_degrees=(
            attempt.grasp_approach_yaw_offset_degrees
        ),
        selected_grasp_transverse_sign=attempt.grasp_transverse_sign,
        camera_clearance_m=attempt.camera_clearance_m,
        nearest_obstacle=attempt.nearest_obstacle,
        payload_clearance_m=attempt.payload_clearance_m,
        nearest_payload_obstacle=attempt.nearest_payload_obstacle,
        body_clearance_m=attempt.body_clearance_m,
        nearest_body_obstacle=attempt.nearest_body_obstacle,
        body_foliage_clearance_m=attempt.body_foliage_clearance_m,
        nearest_body_foliage_obstacle=(
            attempt.nearest_body_foliage_obstacle
        ),
        grasp_arm_foliage_clearance_m=(
            attempt.grasp_arm_foliage_clearance_m
        ),
        nearest_grasp_arm_foliage_obstacle=(
            attempt.nearest_grasp_arm_foliage_obstacle
        ),
        camera_foliage_clearance_m=attempt.camera_foliage_clearance_m,
        nearest_camera_foliage_obstacle=(
            attempt.nearest_camera_foliage_obstacle
        ),
        payload_foliage_clearance_m=attempt.payload_foliage_clearance_m,
        nearest_payload_foliage_obstacle=(
            attempt.nearest_payload_foliage_obstacle
        ),
        minimum_inter_arm_clearance_m=attempt.inter_arm_clearance_m,
        nearest_inter_arm_pair=attempt.nearest_inter_arm_pair,
        trajectory_arm_clearance_m=attempt.trajectory_arm_clearance_m,
        nearest_trajectory_arm_obstacle=attempt.nearest_trajectory_arm_obstacle,
        trajectory_camera_clearance_m=attempt.trajectory_camera_clearance_m,
        nearest_trajectory_camera_obstacle=attempt.nearest_trajectory_camera_obstacle,
        trajectory_payload_clearance_m=attempt.trajectory_payload_clearance_m,
        nearest_trajectory_payload_obstacle=(
            attempt.nearest_trajectory_payload_obstacle
        ),
        trajectory_arm_foliage_clearance_m=(
            attempt.trajectory_arm_foliage_clearance_m
        ),
        nearest_trajectory_arm_foliage_obstacle=(
            attempt.nearest_trajectory_arm_foliage_obstacle
        ),
        trajectory_camera_foliage_clearance_m=(
            attempt.trajectory_camera_foliage_clearance_m
        ),
        nearest_trajectory_camera_foliage_obstacle=(
            attempt.nearest_trajectory_camera_foliage_obstacle
        ),
        trajectory_payload_foliage_clearance_m=(
            attempt.trajectory_payload_foliage_clearance_m
        ),
        nearest_trajectory_payload_foliage_obstacle=(
            attempt.nearest_trajectory_payload_foliage_obstacle
        ),
        trajectory_inter_arm_clearance_m=attempt.trajectory_inter_arm_clearance_m,
        nearest_trajectory_inter_arm_pair=attempt.nearest_trajectory_inter_arm_pair,
        approach_route_index=attempt.approach_route_index,
        approach_waypoints_degrees=attempt.approach_waypoints_degrees,
        approach_start_degrees=(
            None
            if approach_start is None
            else tuple(float(value) for value in approach_start)
        ),
        solution=attempt.solution,
        attempts=tuple(attempts),
    )
