"""Exact, lightweight RB-Y1 v1.0 kinematics for benchmark probes.

The simulator asset is generated from the same URDF, so this module provides a
single source of truth for collision-test waypoints without depending on Lula
robot-description files or a second robot model. Angles exposed to the rest of
the greenhouse example are degrees, matching USD angular-drive targets.
"""

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class ForceCapacity:
    """Quasi-static point-force demand against authored joint limits."""

    joint_torques_nm: tuple[float, ...]
    joint_utilization: tuple[float, ...]
    maximum_utilization: float
    force_capacity_n: float


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

    def _chain(self, child: str) -> tuple[_Joint, ...]:
        chain: list[_Joint] = []
        while child != "base":
            joint = self._by_child[child]
            chain.append(joint)
            child = joint.parent
        return tuple(reversed(chain))

    def arm_limits_degrees(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        joints = [self._by_name[f"{side}_arm_{index}"] for index in range(7)]
        return (
            np.degrees([joint.lower_rad for joint in joints]),
            np.degrees([joint.upper_rad for joint in joints]),
        )

    def forward(
        self,
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
        for joint in self._chain(f"ee_{side}"):
            matrix = matrix @ joint.origin
            if joint.kind == "revolute":
                rotation = np.eye(4, dtype=np.float64)
                rotation[:3, :3] = _axis_rotation(joint.axis, values.get(joint.name, 0.0))
                matrix = matrix @ rotation
        return matrix

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
        torso_degrees=DEFAULT_TORSO_DEGREES,
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
        seed = np.radians(np.asarray(seed_degrees, dtype=np.float64))
        lower, upper = self.arm_limits_degrees(side)
        lower = np.radians(lower) + 1e-5
        upper = np.radians(upper) - 1e-5

        def residual(radians: np.ndarray) -> np.ndarray:
            actual = self.forward(side, np.degrees(radians), base_matrix, torso_degrees)
            point = (actual @ local_point)[:3]
            return np.concatenate(
                (
                    (point - target) / 0.005,
                    (actual[:3, pointing_axis] - pointing) / 0.4,
                    np.asarray([np.dot(actual[:3, transverse_axis], transverse) / 0.3]),
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
        actual = self.forward(side, np.degrees(result.x), base_matrix, torso_degrees)
        point = (actual @ local_point)[:3]
        position_error = float(np.linalg.norm(point - target))
        axis_error = float(
            math.acos(np.clip(np.dot(actual[:3, pointing_axis], pointing), -1.0, 1.0))
        )
        return IKResult(
            joint_degrees=tuple(float(value) for value in np.degrees(result.x)),
            position_error_m=position_error,
            orientation_error_rad=axis_error,
            cost=float(result.cost),
            succeeded=bool(
                result.success
                and position_error < 1e-3
                and axis_error < math.radians(50.0)
                and abs(float(np.dot(actual[:3, transverse_axis], transverse))) < 0.1
            ),
        )
