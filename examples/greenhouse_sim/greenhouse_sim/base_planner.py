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
    excluded_finger_colliders: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class BaseAttempt:
    advance_m: float
    lateral_m: float
    position_m: tuple[float, float, float]
    collider: str
    segment: int
    solution: robot_kinematics.IKResult
    camera_clearance_m: float
    nearest_obstacle: str | None
    payload_clearance_m: float
    nearest_payload_obstacle: str | None
    inter_arm_clearance_m: float | None
    nearest_inter_arm_pair: str | None
    body_clearance_m: float
    nearest_body_obstacle: str | None
    grasp_arm_clearance_m: float
    nearest_grasp_arm_obstacle: str | None
    trajectory_arm_clearance_m: float
    nearest_trajectory_arm_obstacle: str | None
    trajectory_camera_clearance_m: float
    nearest_trajectory_camera_obstacle: str | None
    trajectory_payload_clearance_m: float
    nearest_trajectory_payload_obstacle: str | None
    trajectory_inter_arm_clearance_m: float | None
    nearest_trajectory_inter_arm_pair: str | None
    approach_route_index: int
    approach_waypoints_degrees: tuple[tuple[float, ...], ...]


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
    camera_clearance_m: float
    nearest_obstacle: str | None
    payload_clearance_m: float
    nearest_payload_obstacle: str | None
    minimum_inter_arm_clearance_m: float | None
    nearest_inter_arm_pair: str | None
    body_clearance_m: float
    nearest_body_obstacle: str | None
    trajectory_arm_clearance_m: float
    nearest_trajectory_arm_obstacle: str | None
    trajectory_camera_clearance_m: float
    nearest_trajectory_camera_obstacle: str | None
    trajectory_payload_clearance_m: float
    nearest_trajectory_payload_obstacle: str | None
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
):
    """Return the minimum vine clearance of every wrist payload box."""
    if not payload_boxes:
        return robot_kinematics.ClearanceResult(float("inf"), None)
    excluded = set(excluded_finger_colliders)
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


def _sample_grasp_trajectory(
    model,
    *,
    start_degrees,
    waypoint_degrees,
    target_degrees,
    base_matrix,
    obstacles,
    camera_local_centre_m,
    camera_radius_m: float,
    payload_boxes,
    excluded_finger_colliders,
    right_waiting_degrees,
    samples: int,
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
    camera_best = robot_kinematics.ClearanceResult(float("inf"), None)
    payload_best = robot_kinematics.ClearanceResult(float("inf"), None)
    inter_arm_best = (
        None
        if right_waiting_degrees is None
        else robot_kinematics.ClearanceResult(float("inf"), None)
    )
    chord_start = start
    for chord_target in (*waypoints, target):
        for fraction in np.linspace(0.0, 1.0, samples):
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            joints = chord_start + smooth * (chord_target - chord_start)
            arm = model.arm_obstacle_clearance("left", joints, base_matrix, obstacles)
            if arm.clearance_m < arm_best.clearance_m:
                arm_best = arm
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
            if inter_arm_best is not None:
                inter_arm = model.inter_arm_clearance(
                    joints,
                    right_waiting_degrees,
                    base_matrix,
                )
                if inter_arm.clearance_m < inter_arm_best.clearance_m:
                    inter_arm_best = inter_arm
        chord_start = chord_target
    return arm_best, camera_best, payload_best, inter_arm_best

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
):
    """Deterministic bidirectional RRT-Connect with exact edge validation."""
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

    def normalize(values):
        return (np.asarray(values, dtype=np.float64) - lower) / span

    def denormalize(values):
        return lower + np.asarray(values, dtype=np.float64) * span

    def edge_valid(first_normalized, second_normalized):
        delta = np.asarray(second_normalized) - np.asarray(first_normalized)
        samples = max(2, int(np.ceil(np.linalg.norm(delta) / edge_resolution)) + 1)
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
    if edge_valid(start_normalized, goal_normalized):
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
    jaw_local_point_m,
    camera_local_centre_m,
    camera_radius_m: float,
    seeds,
    advances_m=(0.0, 0.03, 0.06, 0.09),
    lateral_offsets_m=(0.0,),
    left_waiting_degrees=None,
    left_approach_start_degrees=None,
    left_approach_waypoint_routes=((),),
    right_waiting_degrees=None,
    reach_reserve_m: float = 0.02,
    minimum_camera_clearance_m: float = 0.005,
    left_payload_boxes=(),
    minimum_payload_clearance_m: float | None = None,
    minimum_inter_arm_clearance_m: float = 0.005,
    minimum_arm_obstacle_clearance_m: float = 0.005,
    minimum_trajectory_clearance_m: float | None = None,
    minimum_body_clearance_m: float = 0.01,
    trajectory_samples: int = 31,
    joint_space_search_iterations: int = 0,
    joint_space_search_seed: int = 0,
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
        or trajectory_threshold < 0.0
        or minimum_body_clearance_m < 0.0
        or reach_reserve_m < 0.0
    ):
        raise ValueError("camera radius, clearance, and reach reserve must be non-negative")
    if trajectory_samples < 2:
        raise ValueError("trajectory_samples must be at least two")
    if joint_space_search_iterations < 0:
        raise ValueError("joint_space_search_iterations cannot be negative")
    advances = tuple(float(value) for value in advances_m)
    lateral_offsets = tuple(float(value) for value in lateral_offsets_m)
    if not advances or any(value < 0.0 for value in advances):
        raise ValueError("advances_m must contain non-negative distances")
    if not lateral_offsets:
        raise ValueError("lateral_offsets_m cannot be empty")

    ordered = tuple(sorted(candidates, key=lambda item: item.segment, reverse=True))
    minimum_segment = max(0, max(item.segment for item in ordered) - 1)
    distal = tuple(item for item in ordered if item.segment >= minimum_segment)
    target_xy = np.mean(np.asarray([item.centre_m[:2] for item in distal]), axis=0)
    direction_xy = target_xy - nominal[:2]
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9:
        raise ValueError("target and nominal base must have distinct plan positions")
    direction_xy /= norm
    yaw_rad = np.radians(float(yaw_degrees))
    lateral_xy = np.asarray([-np.sin(yaw_rad), np.cos(yaw_rad)])
    reach_reserve = np.asarray(
        [direction_xy[0] * reach_reserve_m, direction_xy[1] * reach_reserve_m, 0.0],
        dtype=np.float64,
    )
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

        for candidate in distal:
            for seed_index, seed_degrees in enumerate(seeds):
                solution = model.solve_position_axes(
                    "left",
                    local_point_m=jaw_local_point_m,
                    target_point_m=np.asarray(candidate.centre_m) + reach_reserve,
                    seed_degrees=seed_degrees,
                    base_matrix=base_matrix,
                    pointing_axis=2,
                    pointing_direction=(0.0, 1.0, 0.0),
                    transverse_axis=0,
                    transverse_to=candidate.axis,
                    position_scale_m=0.002,
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
                inter_arm = (
                    None
                    if right_waiting_degrees is None
                    else model.inter_arm_clearance(
                        solution.joint_degrees,
                        right_waiting_degrees,
                        base_matrix,
                    )
                )
                route_evaluations = []
                candidate_routes = (
                    approach_routes if solution.succeeded else approach_routes[:1]
                )
                for route_index, route_waypoints in enumerate(candidate_routes):
                    route_arm, route_camera, route_payload, route_inter_arm = (
                        _sample_grasp_trajectory(
                            model,
                            start_degrees=approach_start,
                            waypoint_degrees=route_waypoints,
                            target_degrees=solution.joint_degrees,
                            base_matrix=base_matrix,
                            obstacles=obstacles,
                            camera_local_centre_m=camera_local_centre_m,
                            camera_radius_m=camera_radius_m,
                            payload_boxes=left_payload_boxes,
                            excluded_finger_colliders=(
                                candidate.excluded_finger_colliders
                            ),
                            right_waiting_degrees=right_waiting_degrees,
                            samples=trajectory_samples,
                        )
                    )
                    route_evaluations.append(
                        {
                            "index": route_index,
                            "waypoints": route_waypoints,
                            "arm": route_arm,
                            "camera": route_camera,
                            "payload": route_payload,
                            "inter_arm": route_inter_arm,
                            "clearance_m": min(
                                route_arm.clearance_m,
                                route_camera.clearance_m,
                                route_payload.clearance_m,
                                float("inf")
                                if route_inter_arm is None
                                else route_inter_arm.clearance_m,
                            ),
                        }
                    )
                preset_route = max(
                    route_evaluations,
                    key=lambda record: record["clearance_m"],
                )
                endpoint_clear = bool(
                    solution.succeeded
                    and camera.clearance_m >= minimum_camera_clearance_m
                    and payload.clearance_m >= payload_threshold
                    and grasp_arm.clearance_m >= minimum_arm_obstacle_clearance_m
                    and (
                        inter_arm is None
                        or inter_arm.clearance_m >= minimum_inter_arm_clearance_m
                    )
                )
                preset_route_clear = bool(
                    preset_route["arm"].clearance_m >= trajectory_threshold
                    and preset_route["camera"].clearance_m
                    >= minimum_camera_clearance_m
                    and preset_route["payload"].clearance_m
                    >= payload_threshold
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
                ):
                    lower_degrees, upper_degrees = model.arm_limits_degrees("left")

                    def joint_configuration_clear(values):
                        arm_result = model.arm_obstacle_clearance(
                            "left", values, base_matrix, obstacles
                        )
                        if arm_result.clearance_m < trajectory_threshold:
                            return False
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
                        )
                        if payload_result.clearance_m < payload_threshold:
                            return False
                        if right_waiting_degrees is None:
                            return True
                        return (
                            model.inter_arm_clearance(
                                values,
                                right_waiting_degrees,
                                base_matrix,
                            ).clearance_m
                            >= minimum_inter_arm_clearance_m
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
                    )
                    if generated_waypoints is not None:
                        route_arm, route_camera, route_payload, route_inter_arm = (
                            _sample_grasp_trajectory(
                                model,
                                start_degrees=approach_start,
                                waypoint_degrees=generated_waypoints,
                                target_degrees=solution.joint_degrees,
                                base_matrix=base_matrix,
                                obstacles=obstacles,
                                camera_local_centre_m=camera_local_centre_m,
                                camera_radius_m=camera_radius_m,
                                payload_boxes=left_payload_boxes,
                                excluded_finger_colliders=(
                                    candidate.excluded_finger_colliders
                                ),
                                right_waiting_degrees=right_waiting_degrees,
                                samples=trajectory_samples,
                            )
                        )
                        route_evaluations.append(
                            {
                                "index": len(approach_routes),
                                "waypoints": generated_waypoints,
                                "arm": route_arm,
                                "camera": route_camera,
                                "payload": route_payload,
                                "inter_arm": route_inter_arm,
                                "clearance_m": min(
                                    route_arm.clearance_m,
                                    route_camera.clearance_m,
                                    route_payload.clearance_m,
                                    float("inf")
                                    if route_inter_arm is None
                                    else route_inter_arm.clearance_m,
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
                attempt = BaseAttempt(
                    advance_m=advance_m,
                    lateral_m=lateral_m,
                    position_m=tuple(float(value) for value in position),
                    collider=candidate.collider,
                    segment=candidate.segment,
                    solution=solution,
                    camera_clearance_m=camera.clearance_m,
                    nearest_obstacle=camera.nearest_obstacle,
                    payload_clearance_m=payload.clearance_m,
                    nearest_payload_obstacle=payload.nearest_obstacle,
                    body_clearance_m=body_clearance.clearance_m,
                    nearest_body_obstacle=body_clearance.nearest_obstacle,
                    grasp_arm_clearance_m=grasp_arm.clearance_m,
                    nearest_grasp_arm_obstacle=grasp_arm.nearest_obstacle,
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
                )
                attempts.append(attempt)
                if (
                    solution.succeeded
                    and camera.clearance_m >= minimum_camera_clearance_m
                    and payload.clearance_m >= payload_threshold
                    and grasp_arm.clearance_m >= minimum_arm_obstacle_clearance_m
                    and route_arm.clearance_m >= trajectory_threshold
                    and route_camera.clearance_m >= minimum_camera_clearance_m
                    and route_payload.clearance_m >= payload_threshold
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
                    clearances = [
                        body_clearance.clearance_m,
                        camera.clearance_m,
                        payload.clearance_m,
                        grasp_arm.clearance_m,
                        route_arm.clearance_m,
                        route_camera.clearance_m,
                        route_payload.clearance_m,
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
                        }
                    )

    selected = None
    if feasible:
        selected = max(
            feasible,
            key=lambda record: (
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
        camera_clearance_m=attempt.camera_clearance_m,
        nearest_obstacle=attempt.nearest_obstacle,
        payload_clearance_m=attempt.payload_clearance_m,
        nearest_payload_obstacle=attempt.nearest_payload_obstacle,
        body_clearance_m=attempt.body_clearance_m,
        nearest_body_obstacle=attempt.nearest_body_obstacle,
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