"""Exact, lightweight RB-Y1 v1.0 kinematics for benchmark probes.

The simulator asset is generated from the same URDF, so this module provides a
single source of truth for collision-test waypoints without depending on Lula
robot-description files or a second robot model. Angles exposed to the rest of
the greenhouse example are degrees, matching USD angular-drive targets.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_URDF = REPOSITORY_ROOT / "third_party" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model_v1.0.urdf"
DEFAULT_TORSO_DEGREES = (0.0, 45.0, -90.0, 45.0, 0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class IKResult:
    joint_degrees: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float
    cost: float
    succeeded: bool
    evaluations: int | None = None


@dataclasses.dataclass(frozen=True)
class ForceCapacity:
    """Quasi-static point-force demand against authored joint limits."""

    joint_torques_nm: tuple[float, ...]
    joint_utilization: tuple[float, ...]
    maximum_utilization: float
    force_capacity_n: float


@dataclasses.dataclass(frozen=True)
class CapsuleObstacle:
    """World-space capsule centreline used for conservative tool clearance."""

    path: str
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    radius_m: float


@dataclasses.dataclass(frozen=True)
class BoxObstacle:
    """World-space axis-aligned box used for rigid greenhouse clearance."""

    path: str
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class OrientedBoxObstacle:
    """World-space oriented box used for live foliage clearance."""

    path: str
    centre_m: tuple[float, float, float]
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    half_extents_m: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class ClearanceResult:
    clearance_m: float
    nearest_obstacle: str | None


def select_tool_clearance_route(candidates: list[dict], tolerance_m: float = 1e-6) -> dict:
    """Select a route direction first, then maximize separation within it."""
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m cannot be negative")
    feasible = [
        candidate
        for candidate in candidates
        if candidate["feasible"]
        and len(candidate["solutions"]) == len(candidate["offsets"])
    ]
    if not feasible:
        raise ValueError("no route candidate has complete IK")
    best_minimum = max(
        candidate["minimum_clearance"]["clearance_m"]
        for candidate in feasible
    )
    clearance_tied = [
        candidate
        for candidate in feasible
        if candidate["minimum_clearance"]["clearance_m"]
        >= best_minimum - tolerance_m
    ]
    direction_signs = sorted(
        {candidate["x_sign"] for candidate in clearance_tied}
    )
    selected_sign = max(
        direction_signs,
        key=lambda sign: max(
            candidate["mean_clearance_m"]
            for candidate in clearance_tied
            if candidate["x_sign"] == sign
        ),
    )
    selected_direction = [
        candidate
        for candidate in clearance_tied
        if candidate["x_sign"] == selected_sign
    ]
    return max(
        selected_direction,
        key=lambda candidate: (
            candidate["lateral_distance_m"],
            candidate["lift_distance_m"],
            candidate["mean_clearance_m"],
        ),
    )


@dataclasses.dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower_rad: float
    upper_rad: float


@dataclasses.dataclass(frozen=True)
class _LinkCapsule:
    link: str
    index: int
    origin: np.ndarray
    radius_m: float
    cylinder_length_m: float


# Match build_robot.py: the URDF arm_5 cylinder is longer than the endpoint
# comment in the source model and extends through the attached wrist tooling.
_TASK_CONTACT_CAPSULE_OVERRIDES = {
    "link_right_arm_5": ((0.0, 0.0, -0.024), 0.052),
    "link_left_arm_5": ((0.0, 0.0, -0.024), 0.052),
}


def _numbers(value: str | None) -> np.ndarray:
    return np.asarray([float(item) for item in (value or "0 0 0").split()], dtype=np.float64)


def _rotation_xyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )


def base_transform(position_m, yaw_degrees: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _axis_rotation(np.asarray([0.0, 0.0, 1.0]), math.radians(yaw_degrees))
    matrix[:3, 3] = np.asarray(position_m, dtype=np.float64)
    return matrix


def signed_transverse_direction(
    pointing_direction,
    transverse_to,
    sign: float = 1.0,
) -> np.ndarray:
    """Return one signed axis perpendicular to the pointing and target axes.

    For the RB-Y1 gripper, the returned vector is the desired local-X jaw
    closing direction. The two signs represent the physically distinct wrist
    rolls that place the same petiole between the open fingers.
    """
    if float(sign) not in (-1.0, 1.0):
        raise ValueError("sign must be -1 or 1")
    pointing = np.asarray(pointing_direction, dtype=np.float64)
    transverse = np.asarray(transverse_to, dtype=np.float64)
    if pointing.shape != (3,) or transverse.shape != (3,):
        raise ValueError("pointing_direction and transverse_to must contain three values")
    pointing_norm = float(np.linalg.norm(pointing))
    transverse_norm = float(np.linalg.norm(transverse))
    if pointing_norm <= 1e-12 or transverse_norm <= 1e-12:
        raise ValueError("pointing_direction and transverse_to must be non-zero")
    pointing /= pointing_norm
    transverse /= transverse_norm
    direction = np.cross(transverse, pointing)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        raise ValueError("pointing and target axes are parallel")
    return float(sign) * direction / direction_norm


def rotate_horizontal_direction(direction, yaw_degrees: float) -> np.ndarray:
    """Rotate a non-zero direction around world Z and preserve unit length."""
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("direction must contain three values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(float(yaw_degrees)):
        raise ValueError("direction must be non-zero and yaw_degrees finite")
    return _axis_rotation(
        np.asarray([0.0, 0.0, 1.0]), math.radians(float(yaw_degrees))
    ) @ (vector / norm)


def sphere_capsule_clearance(
    centre_m,
    radius_m: float,
    obstacles: tuple[CapsuleObstacle, ...],
) -> ClearanceResult:
    """Return signed separation between a sphere and the nearest capsule."""
    centre = np.asarray(centre_m, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError("centre_m must contain three values")
    if radius_m < 0.0:
        raise ValueError("radius_m cannot be negative")
    best = float("inf")
    nearest = None
    for obstacle in obstacles:
        start = np.asarray(obstacle.start_m, dtype=np.float64)
        end = np.asarray(obstacle.end_m, dtype=np.float64)
        span = end - start
        denominator = float(np.dot(span, span))
        fraction = 0.0 if denominator <= 1e-18 else float(
            np.clip(np.dot(centre - start, span) / denominator, 0.0, 1.0)
        )
        axis_point = start + fraction * span
        clearance = float(
            np.linalg.norm(centre - axis_point) - radius_m - obstacle.radius_m
        )
        if clearance < best:
            best = clearance
            nearest = obstacle.path
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)



def _segment_segment_distance(first_start, first_end, second_start, second_end) -> float:
    """Return the exact Euclidean distance between two finite 3-D segments."""
    first_start = np.asarray(first_start, dtype=np.float64)
    first_end = np.asarray(first_end, dtype=np.float64)
    second_start = np.asarray(second_start, dtype=np.float64)
    second_end = np.asarray(second_end, dtype=np.float64)
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_length_squared = float(np.dot(first_direction, first_direction))
    second_length_squared = float(np.dot(second_direction, second_direction))
    direction_dot = float(np.dot(first_direction, second_direction))
    first_offset_dot = float(np.dot(first_direction, offset))
    second_offset_dot = float(np.dot(second_direction, offset))
    epsilon = 1e-15

    if first_length_squared <= epsilon and second_length_squared <= epsilon:
        return float(np.linalg.norm(offset))
    if first_length_squared <= epsilon:
        first_fraction = 0.0
        second_fraction = float(
            np.clip(second_offset_dot / second_length_squared, 0.0, 1.0)
        )
    elif second_length_squared <= epsilon:
        second_fraction = 0.0
        first_fraction = float(
            np.clip(-first_offset_dot / first_length_squared, 0.0, 1.0)
        )
    else:
        denominator = (
            first_length_squared * second_length_squared - direction_dot * direction_dot
        )
        if denominator > epsilon:
            first_fraction = float(
                np.clip(
                    (direction_dot * second_offset_dot - first_offset_dot * second_length_squared)
                    / denominator,
                    0.0,
                    1.0,
                )
            )
        else:
            first_fraction = 0.0
        second_fraction = (
            direction_dot * first_fraction + second_offset_dot
        ) / second_length_squared
        if second_fraction < 0.0:
            second_fraction = 0.0
            first_fraction = float(
                np.clip(-first_offset_dot / first_length_squared, 0.0, 1.0)
            )
        elif second_fraction > 1.0:
            second_fraction = 1.0
            first_fraction = float(
                np.clip(
                    (direction_dot - first_offset_dot) / first_length_squared,
                    0.0,
                    1.0,
                )
            )
    delta = offset + first_fraction * first_direction - second_fraction * second_direction
    return float(np.linalg.norm(delta))


def capsule_capsule_clearance(first: CapsuleObstacle, second: CapsuleObstacle) -> float:
    """Return signed separation between two world-space capsules."""
    return _segment_segment_distance(
        first.start_m, first.end_m, second.start_m, second.end_m
    ) - first.radius_m - second.radius_m


def _segment_to_segments_distances(
    first_start,
    first_end,
    second_starts: np.ndarray,
    second_ends: np.ndarray,
) -> np.ndarray:
    """Vectorized exact distance from one segment to many finite segments."""
    first_start = np.asarray(first_start, dtype=np.float64)
    first_end = np.asarray(first_end, dtype=np.float64)
    second_starts = np.asarray(second_starts, dtype=np.float64)
    second_ends = np.asarray(second_ends, dtype=np.float64)
    first_direction = first_end - first_start
    second_directions = second_ends - second_starts
    offsets = first_start - second_starts
    first_length_squared = float(np.dot(first_direction, first_direction))
    second_length_squared = np.einsum("ij,ij->i", second_directions, second_directions)
    direction_dots = second_directions @ first_direction
    first_offset_dots = offsets @ first_direction
    second_offset_dots = np.einsum("ij,ij->i", second_directions, offsets)
    epsilon = 1e-15
    first_fractions = np.zeros(len(second_starts), dtype=np.float64)
    second_fractions = np.zeros(len(second_starts), dtype=np.float64)
    second_nondegenerate = second_length_squared > epsilon

    if first_length_squared <= epsilon:
        second_fractions[second_nondegenerate] = np.clip(
            second_offset_dots[second_nondegenerate]
            / second_length_squared[second_nondegenerate],
            0.0,
            1.0,
        )
    else:
        denominators = (
            first_length_squared * second_length_squared
            - direction_dots * direction_dots
        )
        nonparallel = second_nondegenerate & (denominators > epsilon)
        first_fractions[nonparallel] = np.clip(
            (
                direction_dots[nonparallel] * second_offset_dots[nonparallel]
                - first_offset_dots[nonparallel] * second_length_squared[nonparallel]
            )
            / denominators[nonparallel],
            0.0,
            1.0,
        )
        second_fractions[second_nondegenerate] = (
            direction_dots[second_nondegenerate]
            * first_fractions[second_nondegenerate]
            + second_offset_dots[second_nondegenerate]
        ) / second_length_squared[second_nondegenerate]
        below = second_nondegenerate & (second_fractions < 0.0)
        second_fractions[below] = 0.0
        first_fractions[below] = np.clip(
            -first_offset_dots[below] / first_length_squared,
            0.0,
            1.0,
        )
        above = second_nondegenerate & (second_fractions > 1.0)
        second_fractions[above] = 1.0
        first_fractions[above] = np.clip(
            (direction_dots[above] - first_offset_dots[above])
            / first_length_squared,
            0.0,
            1.0,
        )
        degenerate = ~second_nondegenerate
        first_fractions[degenerate] = np.clip(
            -first_offset_dots[degenerate] / first_length_squared,
            0.0,
            1.0,
        )

    deltas = (
        offsets
        + first_fractions[:, None] * first_direction
        - second_fractions[:, None] * second_directions
    )
    return np.linalg.norm(deltas, axis=1)


def _capsule_sets_clearance(
    first: tuple[CapsuleObstacle, ...],
    second: tuple[CapsuleObstacle, ...],
) -> ClearanceResult:
    """Return exact minimum clearance using a vectorized obstacle batch."""
    if not first or not second:
        return ClearanceResult(clearance_m=float("inf"), nearest_obstacle=None)
    starts = np.asarray([item.start_m for item in second], dtype=np.float64)
    ends = np.asarray([item.end_m for item in second], dtype=np.float64)
    radii = np.asarray([item.radius_m for item in second], dtype=np.float64)
    best = float("inf")
    nearest = None
    for capsule in first:
        clearances = (
            _segment_to_segments_distances(
                capsule.start_m,
                capsule.end_m,
                starts,
                ends,
            )
            - capsule.radius_m
            - radii
        )
        index = int(np.argmin(clearances))
        clearance = float(clearances[index])
        if clearance < best:
            best = clearance
            nearest = f"{capsule.path} <-> {second[index].path}"
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)

def oriented_box_capsule_clearance(
    centre_m,
    rotation: np.ndarray,
    half_extents_m,
    obstacles: tuple[CapsuleObstacle, ...],
) -> ClearanceResult:
    """Return exact signed separation from an oriented box to capsules."""
    centre = np.asarray(centre_m, dtype=np.float64)
    basis = np.asarray(rotation, dtype=np.float64)
    half_extents = np.asarray(half_extents_m, dtype=np.float64)
    if centre.shape != (3,) or basis.shape != (3, 3) or half_extents.shape != (3,):
        raise ValueError("box centre, rotation, and half extents have invalid shapes")
    if np.any(half_extents < 0.0):
        raise ValueError("box half extents cannot be negative")
    best = float("inf")
    nearest = None
    # An enclosing-sphere separation is a conservative lower bound on the
    # exact OBB/capsule separation. Evaluate likely-nearest capsules first and
    # skip the expensive segment/AABB solve once that bound cannot beat the
    # exact minimum already found.
    box_radius = float(np.linalg.norm(half_extents))
    bounded_obstacles = []
    for obstacle in obstacles:
        start = np.asarray(obstacle.start_m, dtype=np.float64)
        end = np.asarray(obstacle.end_m, dtype=np.float64)
        obstacle_centre = 0.5 * (start + end)
        obstacle_radius = (
            0.5 * float(np.linalg.norm(end - start))
            + float(obstacle.radius_m)
        )
        lower_bound = (
            float(np.linalg.norm(obstacle_centre - centre))
            - box_radius
            - obstacle_radius
        )
        bounded_obstacles.append((lower_bound, obstacle, start, end))
    bounded_obstacles.sort(key=lambda item: item[0])
    for lower_bound, obstacle, start, end in bounded_obstacles:
        if lower_bound >= best:
            break
        local_start = basis.T @ (start - centre)
        local_end = basis.T @ (end - centre)
        clearance = (
            _segment_aabb_distance(local_start, local_end, half_extents)
            - obstacle.radius_m
        )
        if clearance < best:
            best = clearance
            nearest = obstacle.path
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)


def _box_centre_half_extents(obstacle: BoxObstacle) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(obstacle.minimum_m, dtype=np.float64)
    maximum = np.asarray(obstacle.maximum_m, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum < minimum):
        raise ValueError(f"invalid box obstacle bounds: {obstacle.path}")
    return 0.5 * (minimum + maximum), 0.5 * (maximum - minimum)


def _oriented_box_aabb_separation(
    centre_m,
    rotation: np.ndarray,
    half_extents_m,
    obstacle: BoxObstacle,
) -> float:
    """Return conservative signed SAT separation from an OBB to an AABB."""
    obstacle_centre, obstacle_half_extents = _box_centre_half_extents(obstacle)
    return _oriented_box_obb_separation(
        centre_m,
        rotation,
        half_extents_m,
        obstacle_centre,
        np.eye(3, dtype=np.float64),
        obstacle_half_extents,
    )


def _oriented_box_obb_separation(
    centre_m,
    rotation: np.ndarray,
    half_extents_m,
    obstacle_centre_m,
    obstacle_rotation: np.ndarray,
    obstacle_half_extents_m,
) -> float:
    """Return conservative signed SAT separation between two oriented boxes."""
    centre = np.asarray(centre_m, dtype=np.float64)
    basis = np.asarray(rotation, dtype=np.float64)
    half_extents = np.asarray(half_extents_m, dtype=np.float64)
    obstacle_centre = np.asarray(obstacle_centre_m, dtype=np.float64)
    obstacle_basis = np.asarray(obstacle_rotation, dtype=np.float64)
    obstacle_half_extents = np.asarray(
        obstacle_half_extents_m,
        dtype=np.float64,
    )
    if (
        centre.shape != (3,)
        or basis.shape != (3, 3)
        or half_extents.shape != (3,)
        or obstacle_centre.shape != (3,)
        or obstacle_basis.shape != (3, 3)
        or obstacle_half_extents.shape != (3,)
    ):
        raise ValueError("box centre, rotation, and half extents have invalid shapes")
    if np.any(half_extents < 0.0) or np.any(obstacle_half_extents < 0.0):
        raise ValueError("box half extents cannot be negative")
    relative_rotation = basis.T @ obstacle_basis
    absolute_rotation = np.abs(relative_rotation) + 1e-12
    translation = basis.T @ (obstacle_centre - centre)
    gaps: list[float] = []

    for axis in range(3):
        gaps.append(
            abs(float(translation[axis]))
            - float(
                half_extents[axis]
                + np.dot(absolute_rotation[axis, :], obstacle_half_extents)
            )
        )
    for axis in range(3):
        gaps.append(
            abs(float(np.dot(translation, relative_rotation[:, axis])))
            - float(
                obstacle_half_extents[axis]
                + np.dot(half_extents, absolute_rotation[:, axis])
            )
        )
    for first_axis in range(3):
        first_next = (first_axis + 1) % 3
        first_last = (first_axis + 2) % 3
        for second_axis in range(3):
            second_next = (second_axis + 1) % 3
            second_last = (second_axis + 2) % 3
            projected_distance = abs(
                float(
                    translation[first_last]
                    * relative_rotation[first_next, second_axis]
                    - translation[first_next]
                    * relative_rotation[first_last, second_axis]
                )
            )
            first_radius = float(
                half_extents[first_next]
                * absolute_rotation[first_last, second_axis]
                + half_extents[first_last]
                * absolute_rotation[first_next, second_axis]
            )
            second_radius = float(
                obstacle_half_extents[second_next]
                * absolute_rotation[first_axis, second_last]
                + obstacle_half_extents[second_last]
                * absolute_rotation[first_axis, second_next]
            )
            gaps.append(projected_distance - first_radius - second_radius)
    return max(gaps)


def _oriented_obstacle_arrays(
    obstacle: OrientedBoxObstacle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = np.asarray(obstacle.centre_m, dtype=np.float64)
    rotation = np.asarray(obstacle.rotation, dtype=np.float64)
    half_extents = np.asarray(obstacle.half_extents_m, dtype=np.float64)
    if centre.shape != (3,) or rotation.shape != (3, 3) or half_extents.shape != (3,):
        raise ValueError(f"invalid oriented box obstacle: {obstacle.path}")
    if np.any(half_extents < 0.0):
        raise ValueError(f"invalid oriented box half extents: {obstacle.path}")
    return centre, rotation, half_extents


def oriented_box_box_clearance(
    centre_m,
    rotation: np.ndarray,
    half_extents_m,
    obstacles: tuple[BoxObstacle, ...],
) -> ClearanceResult:
    """Return signed separation from an oriented box to rigid structure boxes."""
    centre = np.asarray(centre_m, dtype=np.float64)
    half_extents = np.asarray(half_extents_m, dtype=np.float64)
    if centre.shape != (3,) or half_extents.shape != (3,):
        raise ValueError("box centre and half extents have invalid shapes")
    box_radius = float(np.linalg.norm(half_extents))
    bounded_obstacles = []
    for obstacle in obstacles:
        obstacle_centre, obstacle_half_extents = _box_centre_half_extents(obstacle)
        lower_bound = (
            float(np.linalg.norm(obstacle_centre - centre))
            - box_radius
            - float(np.linalg.norm(obstacle_half_extents))
        )
        bounded_obstacles.append((lower_bound, obstacle))
    bounded_obstacles.sort(key=lambda item: item[0])

    best = float("inf")
    nearest = None
    for lower_bound, obstacle in bounded_obstacles:
        if lower_bound >= best:
            break
        clearance = _oriented_box_aabb_separation(
            centre,
            rotation,
            half_extents,
            obstacle,
        )
        if clearance < best:
            best = clearance
            nearest = obstacle.path
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)


def oriented_box_oriented_box_clearance(
    centre_m,
    rotation: np.ndarray,
    half_extents_m,
    obstacles: tuple[OrientedBoxObstacle, ...],
) -> ClearanceResult:
    """Return signed separation from a tool OBB to live foliage OBBs."""
    centre = np.asarray(centre_m, dtype=np.float64)
    half_extents = np.asarray(half_extents_m, dtype=np.float64)
    if centre.shape != (3,) or half_extents.shape != (3,):
        raise ValueError("box centre and half extents have invalid shapes")
    box_radius = float(np.linalg.norm(half_extents))
    bounded_obstacles = []
    for obstacle in obstacles:
        obstacle_centre, obstacle_rotation, obstacle_half_extents = (
            _oriented_obstacle_arrays(obstacle)
        )
        lower_bound = (
            float(np.linalg.norm(obstacle_centre - centre))
            - box_radius
            - float(np.linalg.norm(obstacle_half_extents))
        )
        bounded_obstacles.append(
            (
                lower_bound,
                obstacle,
                obstacle_centre,
                obstacle_rotation,
                obstacle_half_extents,
            )
        )
    bounded_obstacles.sort(key=lambda item: item[0])

    best = float("inf")
    nearest = None
    for (
        lower_bound,
        obstacle,
        obstacle_centre,
        obstacle_rotation,
        obstacle_half_extents,
    ) in bounded_obstacles:
        if lower_bound >= best:
            break
        clearance = _oriented_box_obb_separation(
            centre,
            rotation,
            half_extents,
            obstacle_centre,
            obstacle_rotation,
            obstacle_half_extents,
        )
        if clearance < best:
            best = clearance
            nearest = obstacle.path
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)


def sphere_oriented_box_clearance(
    centre_m,
    radius_m: float,
    obstacles: tuple[OrientedBoxObstacle, ...],
) -> ClearanceResult:
    """Return exact sphere separation from live foliage OBBs."""
    centre = np.asarray(centre_m, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError("centre_m must contain three values")
    if radius_m < 0.0:
        raise ValueError("radius_m cannot be negative")
    best = float("inf")
    nearest = None
    for obstacle in obstacles:
        obstacle_centre, obstacle_rotation, half_extents = (
            _oriented_obstacle_arrays(obstacle)
        )
        local = obstacle_rotation.T @ (centre - obstacle_centre)
        closest = np.clip(local, -half_extents, half_extents)
        clearance = float(np.linalg.norm(local - closest) - radius_m)
        if clearance < best:
            best = clearance
            nearest = obstacle.path
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)


def capsule_oriented_box_clearance(
    capsule: CapsuleObstacle,
    obstacle: OrientedBoxObstacle,
) -> float:
    """Return exact capsule-centreline separation from a foliage OBB."""
    obstacle_centre, obstacle_rotation, half_extents = _oriented_obstacle_arrays(
        obstacle
    )
    start = obstacle_rotation.T @ (
        np.asarray(capsule.start_m, dtype=np.float64) - obstacle_centre
    )
    end = obstacle_rotation.T @ (
        np.asarray(capsule.end_m, dtype=np.float64) - obstacle_centre
    )
    return _segment_aabb_distance(start, end, half_extents) - float(
        capsule.radius_m
    )


def capsules_oriented_box_clearance(
    capsules: tuple[CapsuleObstacle, ...],
    obstacles: tuple[OrientedBoxObstacle, ...],
) -> ClearanceResult:
    """Return exact minimum capsule-set separation from foliage OBBs."""
    best = float("inf")
    nearest = None
    for capsule in capsules:
        start = np.asarray(capsule.start_m, dtype=np.float64)
        end = np.asarray(capsule.end_m, dtype=np.float64)
        centre = 0.5 * (start + end)
        bounding_radius = (
            0.5 * float(np.linalg.norm(end - start))
            + float(capsule.radius_m)
        )
        for obstacle in obstacles:
            obstacle_centre, _, half_extents = _oriented_obstacle_arrays(
                obstacle
            )
            lower_bound = (
                float(np.linalg.norm(obstacle_centre - centre))
                - bounding_radius
                - float(np.linalg.norm(half_extents))
            )
            if lower_bound >= best:
                continue
            clearance = capsule_oriented_box_clearance(capsule, obstacle)
            if clearance < best:
                best = clearance
                nearest = f"{capsule.path} <-> {obstacle.path}"
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)


def sphere_box_clearance(
    centre_m,
    radius_m: float,
    obstacles: tuple[BoxObstacle, ...],
) -> ClearanceResult:
    """Return signed separation from a sphere to rigid structure boxes."""
    centre = np.asarray(centre_m, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError("centre_m must contain three values")
    if radius_m < 0.0:
        raise ValueError("radius_m cannot be negative")
    best = float("inf")
    nearest = None
    for obstacle in obstacles:
        obstacle_centre, half_extents = _box_centre_half_extents(obstacle)
        excess = np.maximum(np.abs(centre - obstacle_centre) - half_extents, 0.0)
        clearance = float(np.linalg.norm(excess) - radius_m)
        if clearance < best:
            best = clearance
            nearest = obstacle.path
    return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)


def capsule_box_clearance(capsule: CapsuleObstacle, obstacle: BoxObstacle) -> float:
    """Return signed separation from a capsule to an axis-aligned box."""
    centre, half_extents = _box_centre_half_extents(obstacle)
    return (
        _segment_aabb_distance(
            np.asarray(capsule.start_m, dtype=np.float64) - centre,
            np.asarray(capsule.end_m, dtype=np.float64) - centre,
            half_extents,
        )
        - capsule.radius_m
    )


def _segment_aabb_distance(start, end, half_extents) -> float:
    """Exact distance between a segment and an origin-centred axis-aligned box."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    half_extents = np.asarray(half_extents, dtype=np.float64)
    delta = end - start
    breakpoints = [0.0, 1.0]
    for axis in range(3):
        if abs(float(delta[axis])) <= 1e-15:
            continue
        for boundary in (-half_extents[axis], half_extents[axis]):
            fraction = float((boundary - start[axis]) / delta[axis])
            if 0.0 < fraction < 1.0:
                breakpoints.append(fraction)
    breakpoints = sorted(set(breakpoints))

    def squared_distance(fraction: float) -> float:
        point = start + fraction * delta
        excess = np.maximum(np.abs(point) - half_extents, 0.0)
        return float(np.dot(excess, excess))

    best_squared = float("inf")
    for lower, upper in itertools.pairwise(breakpoints):
        midpoint = 0.5 * (lower + upper)
        point = start + midpoint * delta
        offsets = np.zeros(3, dtype=np.float64)
        active_delta = np.zeros(3, dtype=np.float64)
        below = point < -half_extents
        above = point > half_extents
        offsets[below] = start[below] + half_extents[below]
        offsets[above] = start[above] - half_extents[above]
        active_delta[below | above] = delta[below | above]
        denominator = float(np.dot(active_delta, active_delta))
        candidates = [lower, upper]
        if denominator > 1e-18:
            stationary = -float(np.dot(offsets, active_delta)) / denominator
            candidates.append(float(np.clip(stationary, lower, upper)))
        best_squared = min(
            best_squared,
            *(squared_distance(fraction) for fraction in candidates),
        )
    return float(np.sqrt(max(best_squared, 0.0)))


def tool_sphere_clearance(
    model,
    side: str,
    arm_degrees,
    base_matrix: np.ndarray,
    local_centre_m,
    radius_m: float,
    obstacles: tuple[CapsuleObstacle, ...],
) -> ClearanceResult:
    """Transform a tool-attached sphere and measure its capsule clearance."""
    local = np.append(np.asarray(local_centre_m, dtype=np.float64), 1.0)
    centre = (model.forward(side, arm_degrees, base_matrix) @ local)[:3]
    return sphere_capsule_clearance(centre, radius_m, obstacles)


def tool_box_clearance(
    model,
    side: str,
    arm_degrees,
    base_matrix: np.ndarray,
    local_centre_m,
    local_rotation: np.ndarray,
    half_extents_m,
    obstacles: tuple[CapsuleObstacle, ...],
) -> ClearanceResult:
    """Transform a tool-attached box and measure its capsule clearance."""
    tool = model.forward(side, arm_degrees, base_matrix)
    local = np.append(np.asarray(local_centre_m, dtype=np.float64), 1.0)
    centre = (tool @ local)[:3]
    rotation = tool[:3, :3] @ np.asarray(local_rotation, dtype=np.float64)
    return oriented_box_capsule_clearance(
        centre,
        rotation,
        half_extents_m,
        obstacles,
    )


class Rby1Kinematics:
    """Forward and bounded inverse kinematics for either seven-axis arm."""

    def __init__(self, urdf_path: pathlib.Path = DEFAULT_URDF) -> None:
        root = ET.parse(pathlib.Path(urdf_path)).getroot()
        self._by_child: dict[str, _Joint] = {}
        self._by_name: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            origin_element = element.find("origin")
            xyz = _numbers(None if origin_element is None else origin_element.get("xyz"))
            rpy = _numbers(None if origin_element is None else origin_element.get("rpy"))
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rotation_xyz(rpy)
            origin[:3, 3] = xyz
            axis_element = element.find("axis")
            axis = _numbers(None if axis_element is None else axis_element.get("xyz"))
            limit = element.find("limit")
            lower = -math.inf if limit is None or limit.get("lower") is None else float(limit.get("lower"))
            upper = math.inf if limit is None or limit.get("upper") is None else float(limit.get("upper"))
            joint = _Joint(
                name=element.attrib["name"],
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                kind=element.attrib["type"],
                origin=origin,
                axis=axis,
                lower_rad=lower,
                upper_rad=upper,
            )
            self._by_child[joint.child] = joint
            self._by_name[joint.name] = joint
        self._link_capsules: dict[str, tuple[_LinkCapsule, ...]] = {}
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            capsules: list[_LinkCapsule] = []
            for index, collision in enumerate(link.findall("collision")):
                capsule = collision.find("geometry/capsule")
                if capsule is None:
                    continue
                origin_element = collision.find("origin")
                xyz = _numbers(
                    None if origin_element is None else origin_element.get("xyz")
                )
                rpy = _numbers(
                    None if origin_element is None else origin_element.get("rpy")
                )
                length = float(capsule.attrib["length"])
                override = _TASK_CONTACT_CAPSULE_OVERRIDES.get(link_name)
                if override is not None:
                    xyz = np.asarray(override[0], dtype=np.float64)
                    length = float(override[1])
                origin = np.eye(4, dtype=np.float64)
                origin[:3, :3] = _rotation_xyz(rpy)
                origin[:3, 3] = xyz
                capsules.append(
                    _LinkCapsule(
                        link=link_name,
                        index=index,
                        origin=origin,
                        radius_m=float(capsule.attrib["radius"]),
                        cylinder_length_m=length,
                    )
                )
            if capsules:
                self._link_capsules[link_name] = tuple(capsules)



    def _chain(self, child: str) -> tuple[_Joint, ...]:
        chain: list[_Joint] = []
        while child != "base":
            joint = self._by_child[child]
            chain.append(joint)
            child = joint.parent
        return tuple(reversed(chain))

    def _link_transform(
        self,
        link: str,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> np.ndarray:
        arm_radians = np.radians(np.asarray(arm_degrees, dtype=np.float64))
        torso_radians = np.radians(np.asarray(torso_degrees, dtype=np.float64))
        values = {
            **{f"torso_{index}": value for index, value in enumerate(torso_radians)},
            **{f"{side}_arm_{index}": value for index, value in enumerate(arm_radians)},
        }
        matrix = np.asarray(base_matrix, dtype=np.float64).copy()
        for joint in self._chain(link):
            matrix = matrix @ joint.origin
            if joint.kind == "revolute":
                rotation = np.eye(4, dtype=np.float64)
                rotation[:3, :3] = _axis_rotation(
                    joint.axis, values.get(joint.name, 0.0)
                )
                matrix = matrix @ rotation
        return matrix


    def arm_limits_degrees(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        joints = [self._by_name[f"{side}_arm_{index}"] for index in range(7)]
        return (
            np.degrees([joint.lower_rad for joint in joints]),
            np.degrees([joint.upper_rad for joint in joints]),
        )

    def arm_joint_limit_margin_degrees(self, side: str, arm_degrees) -> float:
        """Return the smallest distance from an arm pose to an authored limit."""
        values = np.asarray(arm_degrees, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError("arm_degrees must contain seven joint values")
        lower, upper = self.arm_limits_degrees(side)
        return float(np.min(np.minimum(values - lower, upper - values)))

    def forward(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> np.ndarray:
        return self._link_transform(
            f"ee_{side}", side, arm_degrees, base_matrix, torso_degrees

        )

    def arm_capsules(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> tuple[CapsuleObstacle, ...]:
        """Transform one arm's authored URDF contact capsules to world space."""
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported arm side {side!r}")
        result: list[CapsuleObstacle] = []
        prefix = f"link_{side}_arm_"
        for link, capsules in self._link_capsules.items():
            if not link.startswith(prefix):
                continue
            link_matrix = self._link_transform(
                link, side, arm_degrees, base_matrix, torso_degrees
            )
            for capsule in capsules:
                collision_matrix = link_matrix @ capsule.origin
                half_length = 0.5 * capsule.cylinder_length_m
                start = (collision_matrix @ np.asarray([0.0, 0.0, -half_length, 1.0]))[:3]
                end = (collision_matrix @ np.asarray([0.0, 0.0, half_length, 1.0]))[:3]
                result.append(
                    CapsuleObstacle(
                        path=f"{link}/capsule_{capsule.index:02d}",
                        start_m=tuple(float(value) for value in start),
                        end_m=tuple(float(value) for value in end),
                        radius_m=capsule.radius_m,
                    )
                )
        return tuple(result)

    def fixed_body_capsules(
        self,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> tuple[CapsuleObstacle, ...]:
        """Transform torso and head URDF contact capsules to world space."""
        result: list[CapsuleObstacle] = []
        for link, capsules in self._link_capsules.items():
            if not link.startswith(("link_torso_", "link_head_")):
                continue
            link_matrix = self._link_transform(
                link,
                "left",
                (0.0,) * 7,
                base_matrix,
                torso_degrees,
            )
            for capsule in capsules:
                collision_matrix = link_matrix @ capsule.origin
                half_length = 0.5 * capsule.cylinder_length_m
                start = (
                    collision_matrix
                    @ np.asarray([0.0, 0.0, -half_length, 1.0])
                )[:3]
                end = (
                    collision_matrix
                    @ np.asarray([0.0, 0.0, half_length, 1.0])
                )[:3]
                result.append(
                    CapsuleObstacle(
                        path=f"{link}/capsule_{capsule.index:02d}",
                        start_m=tuple(float(value) for value in start),
                        end_m=tuple(float(value) for value in end),
                        radius_m=capsule.radius_m,
                    )
                )
        return tuple(result)

    def fixed_body_clearance(
        self,
        base_matrix: np.ndarray,
        obstacles: tuple[CapsuleObstacle, ...],
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ClearanceResult:
        """Return minimum torso/head separation from physical vine capsules."""
        return _capsule_sets_clearance(
            self.fixed_body_capsules(base_matrix, torso_degrees),
            obstacles,
        )

    def fixed_body_oriented_box_clearance(
        self,
        base_matrix: np.ndarray,
        obstacles: tuple[OrientedBoxObstacle, ...],
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ClearanceResult:
        """Return exact torso/head separation from live foliage OBBs."""
        return capsules_oriented_box_clearance(
            self.fixed_body_capsules(base_matrix, torso_degrees),
            obstacles,
        )

    def arm_obstacle_clearance(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        obstacles: tuple[CapsuleObstacle, ...],
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ClearanceResult:
        """Return minimum signed separation between one arm and vine capsules."""
        return _capsule_sets_clearance(
            self.arm_capsules(side, arm_degrees, base_matrix, torso_degrees),
            obstacles,
        )

    def inter_arm_clearance(
        self,
        left_degrees,
        right_degrees,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ClearanceResult:
        """Return the minimum signed separation between left/right arm capsules."""
        left_capsules = self.arm_capsules(
            "left", left_degrees, base_matrix, torso_degrees
        )
        right_capsules = self.arm_capsules(
            "right", right_degrees, base_matrix, torso_degrees
        )
        return _capsule_sets_clearance(left_capsules, right_capsules)

    def arm_structure_clearance(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        obstacles: tuple[BoxObstacle, ...],
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ClearanceResult:
        """Return minimum arm-capsule separation from rigid greenhouse boxes."""
        capsules = self.arm_capsules(
            side,
            arm_degrees,
            base_matrix,
            torso_degrees,
        )
        best = float("inf")
        nearest = None
        for capsule in capsules:
            for obstacle in obstacles:
                clearance = capsule_box_clearance(capsule, obstacle)
                if clearance < best:
                    best = clearance
                    nearest = f"{capsule.path} <-> {obstacle.path}"
        return ClearanceResult(clearance_m=best, nearest_obstacle=nearest)

    def arm_oriented_box_clearance(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        obstacles: tuple[OrientedBoxObstacle, ...],
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ClearanceResult:
        """Return minimum arm-capsule separation from live foliage OBBs."""
        return capsules_oriented_box_clearance(
            self.arm_capsules(
                side,
                arm_degrees,
                base_matrix,
                torso_degrees,
            ),
            obstacles,
        )

    def point_jacobian(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        local_point_m,
        torso_degrees=DEFAULT_TORSO_DEGREES,
        *,
        delta_rad: float = 1e-5,
    ) -> np.ndarray:
        """Return the 3x7 world translational Jacobian at a tool point."""
        if delta_rad <= 0.0:
            raise ValueError("delta_rad must be positive")
        joints = np.asarray(arm_degrees, dtype=np.float64)
        local = np.append(np.asarray(local_point_m, dtype=np.float64), 1.0)
        if joints.shape != (7,) or local.shape != (4,):
            raise ValueError("expected seven arm angles and a three-vector point")
        perturbation_degrees = math.degrees(delta_rad)
        columns = []
        for index in range(7):
            delta = np.zeros(7, dtype=np.float64)
            delta[index] = perturbation_degrees
            forward_point = (
                self.forward(
                    side,
                    joints + delta,
                    base_matrix,
                    torso_degrees,
                )
                @ local
            )[:3]
            backward_point = (
                self.forward(
                    side,
                    joints - delta,
                    base_matrix,
                    torso_degrees,
                )
                @ local
            )[:3]
            columns.append((forward_point - backward_point) / (2.0 * delta_rad))
        return np.column_stack(columns)

    def point_force_capacity(
        self,
        side: str,
        arm_degrees,
        base_matrix: np.ndarray,
        local_point_m,
        force_direction,
        force_n: float,
        effort_limits_nm,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> ForceCapacity:
        """Estimate tool-force capacity without exceeding any joint effort."""
        direction = np.asarray(force_direction, dtype=np.float64)
        limits = np.asarray(effort_limits_nm, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or norm <= 1e-12:
            raise ValueError("force_direction must be a non-zero three-vector")
        if limits.shape != (7,) or np.any(limits <= 0.0):
            raise ValueError("effort_limits_nm must contain seven positive values")
        if force_n < 0.0:
            raise ValueError("force_n cannot be negative")
        direction /= norm
        jacobian = self.point_jacobian(
            side,
            arm_degrees,
            base_matrix,
            local_point_m,
            torso_degrees,
        )
        torque_per_newton = jacobian.T @ direction
        torques = torque_per_newton * float(force_n)
        utilization = np.abs(torques) / limits
        loaded = np.abs(torque_per_newton) > 1e-12
        capacity = float(
            np.min(limits[loaded] / np.abs(torque_per_newton[loaded]))
        ) if np.any(loaded) else float("inf")
        return ForceCapacity(
            joint_torques_nm=tuple(float(value) for value in torques),
            joint_utilization=tuple(float(value) for value in utilization),
            maximum_utilization=float(np.max(utilization)),
            force_capacity_n=capacity,
        )

    def solve_pose(
        self,
        side: str,
        desired: np.ndarray,
        seed_degrees,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> IKResult:
        """Solve a full end-effector pose while staying on the seed branch."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        desired = np.asarray(desired, dtype=np.float64)
        seed = np.radians(np.asarray(seed_degrees, dtype=np.float64))
        lower, upper = self.arm_limits_degrees(side)
        lower = np.radians(lower) + 1e-5
        upper = np.radians(upper) - 1e-5

        def residual(radians: np.ndarray) -> np.ndarray:
            actual = self.forward(side, np.degrees(radians), base_matrix, torso_degrees)
            rotation = Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.concatenate(
                (
                    (actual[:3, 3] - desired[:3, 3]) / 0.01,
                    rotation / 0.15,
                    (radians - seed) * 0.01,
                )
            )

        result = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=4000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        actual = self.forward(side, np.degrees(result.x), base_matrix, torso_degrees)
        position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        orientation_error = float(
            np.linalg.norm(
                Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec()
            )
        )
        return IKResult(
            joint_degrees=tuple(float(value) for value in np.degrees(result.x)),
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            cost=float(result.cost),
            succeeded=bool(result.success and position_error < 5e-4 and orientation_error < 5e-3),
        )

    def solve_position(
        self,
        side: str,
        *,
        local_point_m,
        target_point_m,
        seed_degrees,
        base_matrix: np.ndarray,
        torso_degrees=DEFAULT_TORSO_DEGREES,
    ) -> IKResult:
        """Place one tool point while retaining the seed's redundant branch."""
        from scipy.optimize import least_squares

        local_point = np.append(np.asarray(local_point_m, dtype=np.float64), 1.0)
        target = np.asarray(target_point_m, dtype=np.float64)
        seed = np.radians(np.asarray(seed_degrees, dtype=np.float64))
        lower, upper = self.arm_limits_degrees(side)
        lower = np.radians(lower) + 1e-5
        upper = np.radians(upper) - 1e-5

        def residual(radians: np.ndarray) -> np.ndarray:
            actual = self.forward(
                side,
                np.degrees(radians),
                base_matrix,
                torso_degrees,
            )
            point = (actual @ local_point)[:3]
            return np.concatenate(
                (
                    (point - target) / 0.005,
                    (radians - seed) * 0.003,
                )
            )

        result = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=5000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        actual = self.forward(
            side,
            np.degrees(result.x),
            base_matrix,
            torso_degrees,
        )
        position_error = float(
            np.linalg.norm((actual @ local_point)[:3] - target)
        )
        return IKResult(
            joint_degrees=tuple(float(value) for value in np.degrees(result.x)),
            position_error_m=position_error,
            orientation_error_rad=0.0,
            cost=float(result.cost),
            succeeded=bool(result.success and position_error < 1e-3),
        )

    def solve_position_axes(
        self,
        side: str,
        *,
        local_point_m,
        target_point_m,
        seed_degrees,
        base_matrix: np.ndarray,
        pointing_axis: int,
        pointing_direction,
        transverse_axis: int,
        transverse_to,
        transverse_direction=None,
        torso_degrees=DEFAULT_TORSO_DEGREES,
        position_scale_m: float = 0.005,
        maximum_evaluations: int = 5000,
    ) -> IKResult:
        """Solve point placement with a pointing axis and transverse closing axis."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        local_point = np.append(np.asarray(local_point_m, dtype=np.float64), 1.0)
        target = np.asarray(target_point_m, dtype=np.float64)
        pointing = np.asarray(pointing_direction, dtype=np.float64)
        pointing /= max(float(np.linalg.norm(pointing)), 1e-12)
        transverse = np.asarray(transverse_to, dtype=np.float64)
        transverse /= max(float(np.linalg.norm(transverse)), 1e-12)
        desired_transverse = (
            None
            if transverse_direction is None
            else np.asarray(transverse_direction, dtype=np.float64)
        )
        if desired_transverse is not None:
            if desired_transverse.shape != (3,):
                raise ValueError("transverse_direction must contain three values")
            desired_transverse_norm = float(np.linalg.norm(desired_transverse))
            if desired_transverse_norm <= 1e-12:
                raise ValueError("transverse_direction must be non-zero")
            desired_transverse /= desired_transverse_norm
        if position_scale_m <= 0.0:
            raise ValueError("position_scale_m must be positive")
        if maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive")
        seed = np.radians(np.asarray(seed_degrees, dtype=np.float64))
        lower, upper = self.arm_limits_degrees(side)
        lower = np.radians(lower) + 1e-5
        upper = np.radians(upper) - 1e-5

        def residual(radians: np.ndarray) -> np.ndarray:
            actual = self.forward(side, np.degrees(radians), base_matrix, torso_degrees)
            point = (actual @ local_point)[:3]
            return np.concatenate(
                (
                    (point - target) / position_scale_m,
                    (actual[:3, pointing_axis] - pointing) / 0.4,
                    (
                        np.asarray(
                            [np.dot(actual[:3, transverse_axis], transverse) / 0.3]
                        )
                        if desired_transverse is None
                        else (
                            actual[:3, transverse_axis] - desired_transverse
                        )
                        / 0.4
                    ),
                    (radians - seed) * 0.003,
                )
            )

        result = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=int(maximum_evaluations),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        actual = self.forward(side, np.degrees(result.x), base_matrix, torso_degrees)
        point = (actual @ local_point)[:3]
        position_error = float(np.linalg.norm(point - target))
        axis_error = float(
            math.acos(np.clip(np.dot(actual[:3, pointing_axis], pointing), -1.0, 1.0))
        )
        transverse_error = (
            0.0
            if desired_transverse is None
            else float(
                math.acos(
                    np.clip(
                        np.dot(
                            actual[:3, transverse_axis], desired_transverse
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        return IKResult(
            joint_degrees=tuple(float(value) for value in np.degrees(result.x)),
            position_error_m=position_error,
            orientation_error_rad=max(axis_error, transverse_error),
            cost=float(result.cost),
            succeeded=bool(
                result.success
                and position_error < 1e-3
                and axis_error < math.radians(50.0)
                and (
                    abs(float(np.dot(actual[:3, transverse_axis], transverse))) < 0.1
                    if desired_transverse is None
                    else transverse_error < math.radians(50.0)
                )
            ),
            evaluations=int(result.nfev),
        )
