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


@dataclasses.dataclass(frozen=True)
class BaseAttempt:
    advance_m: float
    position_m: tuple[float, float, float]
    collider: str
    segment: int
    solution: robot_kinematics.IKResult
    camera_clearance_m: float
    nearest_obstacle: str | None


@dataclasses.dataclass(frozen=True)
class BasePlan:
    nominal_position_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    offset_m: tuple[float, float, float]
    advance_m: float
    reach_reserve_m: float
    minimum_segment: int
    selected_grasp_collider: str
    selected_grasp_body: str
    selected_grasp_segment: int
    camera_clearance_m: float
    nearest_obstacle: str | None
    solution: robot_kinematics.IKResult
    attempts: tuple[BaseAttempt, ...]


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
    reach_reserve_m: float = 0.02,
    minimum_camera_clearance_m: float = 0.005,
) -> BasePlan | None:
    """Return the nearest target-facing base pose with a safe distal grasp."""
    if not candidates:
        return None
    nominal = np.asarray(nominal_position_m, dtype=np.float64)
    if nominal.shape != (3,):
        raise ValueError("nominal_position_m must contain three values")
    if (
        camera_radius_m < 0.0
        or minimum_camera_clearance_m < 0.0
        or reach_reserve_m < 0.0
    ):
        raise ValueError("camera radius, clearance, and reach reserve must be non-negative")
    advances = tuple(float(value) for value in advances_m)
    if not advances or any(value < 0.0 for value in advances):
        raise ValueError("advances_m must contain non-negative distances")

    ordered = tuple(sorted(candidates, key=lambda item: item.segment, reverse=True))
    minimum_segment = max(0, max(item.segment for item in ordered) - 1)
    distal = tuple(item for item in ordered if item.segment >= minimum_segment)
    target_xy = np.mean(np.asarray([item.centre_m[:2] for item in distal]), axis=0)
    direction_xy = target_xy - nominal[:2]
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9:
        raise ValueError("target and nominal base must have distinct plan positions")
    direction_xy /= norm
    reach_reserve = np.asarray(
        [direction_xy[0] * reach_reserve_m, direction_xy[1] * reach_reserve_m, 0.0],
        dtype=np.float64,
    )

    attempts: list[BaseAttempt] = []
    for advance_m in advances:
        position = nominal.copy()
        position[:2] += direction_xy * advance_m
        base_matrix = robot_kinematics.base_transform(position, yaw_degrees)
        for candidate in distal:
            valid: list[tuple[robot_kinematics.IKResult, robot_kinematics.ClearanceResult]] = []
            for seed in seeds:
                solution = model.solve_position_axes(
                    "left",
                    local_point_m=jaw_local_point_m,
                    target_point_m=np.asarray(candidate.centre_m) + reach_reserve,
                    seed_degrees=seed,
                    base_matrix=base_matrix,
                    pointing_axis=2,
                    pointing_direction=(0.0, 1.0, 0.0),
                    transverse_axis=0,
                    transverse_to=candidate.axis,
                    position_scale_m=0.002,
                )
                clearance = robot_kinematics.tool_sphere_clearance(
                    model,
                    "left",
                    solution.joint_degrees,
                    base_matrix,
                    camera_local_centre_m,
                    camera_radius_m,
                    obstacles,
                )
                attempts.append(
                    BaseAttempt(
                        advance_m=advance_m,
                        position_m=tuple(float(value) for value in position),
                        collider=candidate.collider,
                        segment=candidate.segment,
                        solution=solution,
                        camera_clearance_m=clearance.clearance_m,
                        nearest_obstacle=clearance.nearest_obstacle,
                    )
                )
                if solution.succeeded and clearance.clearance_m >= minimum_camera_clearance_m:
                    valid.append((solution, clearance))
            if valid:
                solution, clearance = max(valid, key=lambda item: item[1].clearance_m)
                return BasePlan(
                    nominal_position_m=tuple(float(value) for value in nominal),
                    position_m=tuple(float(value) for value in position),
                    offset_m=tuple(float(value) for value in position - nominal),
                    advance_m=advance_m,
                    reach_reserve_m=reach_reserve_m,
                    minimum_segment=minimum_segment,
                    selected_grasp_collider=candidate.collider,
                    selected_grasp_body=candidate.body,
                    selected_grasp_segment=candidate.segment,
                    camera_clearance_m=clearance.clearance_m,
                    nearest_obstacle=clearance.nearest_obstacle,
                    solution=solution,
                    attempts=tuple(attempts),
                )
    return None
