"""Isaac Sim adapter and synchronous transport for online deleafing RL."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import socket
from collections.abc import Callable

import numpy as np

from greenhouse_sim import rl_env


_LEFT_JAW_CENTRE_M = np.asarray([0.0, 0.0, -0.1025], dtype=np.float64)

def _advance_one_physics_step(context, *, render: bool) -> None:
    """Advance one physics sample, then optionally refresh rendered UI.

    ``SimulationContext.step(render=True)`` advances one rendering interval,
    which is four physics samples at this scene's 60/240 Hz timing.
    """

    context.step(render=False)
    if render:
        context.render()



@dataclasses.dataclass
class _ArticulationSnapshot:
    view: object
    positions: np.ndarray
    orientations: np.ndarray
    joint_positions: np.ndarray


class IsaacDeleafRuntime:
    """Drive live RB-Y1 physics and restore deterministic episode snapshots."""

    def __init__(
        self,
        *,
        stage,
        context,
        runtimes,
        selected_target,
        blade_monitor,
        grasp_manager,
        contact_diagnostics,
        report: dict,
        apply_cut_decisions: Callable,
        unsafe_contacts: Callable,
        airflow,
        frame_recorder=None,
        render: bool = False,
        reset_settle_steps: int = 24,
        reset_joint_noise_degrees: float = 1.0,
    ) -> None:
        from isaacsim.core.prims import Articulation
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics

        self._stage = stage
        self._context = context
        self._runtimes = tuple(runtimes)
        self._target = selected_target
        self._blade = blade_monitor
        self._grasp = grasp_manager
        self._contacts = contact_diagnostics
        self._report = report
        self._apply_cut_decisions = apply_cut_decisions
        self._unsafe_contacts = unsafe_contacts
        self._airflow = airflow
        self._frame_recorder = frame_recorder
        self._recorded_frames = 0
        self._render = bool(render)
        self._reset_settle_steps = int(reset_settle_steps)
        self._reset_joint_noise_degrees = float(reset_joint_noise_degrees)
        self._Gf = Gf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        if self._reset_settle_steps < 0:
            raise ValueError("reset_settle_steps cannot be negative")
        if self._reset_joint_noise_degrees < 0.0:
            raise ValueError("reset_joint_noise_degrees cannot be negative")

        self._robot = Articulation(
            prim_paths_expr="/World/RBY1",
            name="greenhouse_online_rl_rby1",
            reset_xform_properties=False,
        )
        self._robot.initialize()
        self._dof_indices = {
            side: [
                self._robot.get_dof_index(f"{side}_arm_{index}")
                for index in range(7)
            ]
            for side in ("left", "right")
        }
        if any(index < 0 for values in self._dof_indices.values() for index in values):
            raise RuntimeError("RB-Y1 arm DOFs are incomplete")
        self._arm_limits = {
            side: self._read_arm_limits(side) for side in ("left", "right")
        }
        self._robot_snapshot = self._snapshot_articulation(self._robot)

        self._vine_snapshots: list[_ArticulationSnapshot] = []
        articulation_index = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/InteractiveVines/"):
                continue
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", path).strip("_")
            view = Articulation(
                prim_paths_expr=path,
                name=f"greenhouse_online_rl_vine_{articulation_index}_{safe_name}",
                reset_xform_properties=False,
            )
            view.initialize()
            self._vine_snapshots.append(self._snapshot_articulation(view))
            articulation_index += 1
        if not self._vine_snapshots:
            raise RuntimeError("no physics-vine articulations found for RL reset")

        self._initial_drive_degrees = {
            side: self._joint_state(side)[0].copy()
            for side in ("left", "right")
        }
        self._arm_target_degrees = {
            side: values.copy()
            for side, values in self._initial_drive_degrees.items()
        }
        self._commanded_arm_velocity_degrees_s = {
            side: np.zeros(7, dtype=np.float64) for side in ("left", "right")
        }
        self._commanded_gripper_velocity_per_s = 0.0
        self._gripper_openness = 1.0
        self._episode_seed = 0
        self._episode_index = -1

    def _read_arm_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for index in range(7):
            joint = self._stage.GetPrimAtPath(
                f"/World/RBY1/joints/{side}_arm_{index}"
            )
            revolute = self._UsdPhysics.RevoluteJoint(joint)
            lower.append(float(revolute.GetLowerLimitAttr().Get()))
            upper.append(float(revolute.GetUpperLimitAttr().Get()))
        return np.asarray(lower), np.asarray(upper)

    @staticmethod
    def _snapshot_articulation(view) -> _ArticulationSnapshot:
        positions, orientations = view.get_world_poses()
        joints = view.get_joint_positions()
        return _ArticulationSnapshot(
            view=view,
            positions=np.asarray(positions, dtype=np.float64).copy(),
            orientations=np.asarray(orientations, dtype=np.float64).copy(),
            joint_positions=np.asarray(joints, dtype=np.float64).copy(),
        )

    @staticmethod
    def _restore_articulation(snapshot: _ArticulationSnapshot) -> None:
        view = snapshot.view
        view.set_world_poses(
            positions=snapshot.positions.copy(),
            orientations=snapshot.orientations.copy(),
        )
        zero_root = np.zeros((snapshot.positions.shape[0], 3), dtype=np.float64)
        view.set_linear_velocities(zero_root)
        view.set_angular_velocities(zero_root)
        if snapshot.joint_positions.size:
            view.set_joint_positions(snapshot.joint_positions.copy())
            view.set_joint_velocities(np.zeros_like(snapshot.joint_positions))

    def _joint_state(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        indices = self._dof_indices[side]
        positions = np.degrees(
            np.asarray(
                self._robot.get_joint_positions(joint_indices=indices),
                dtype=np.float64,
            ).reshape(-1)
        )
        velocities = np.degrees(
            np.asarray(
                self._robot.get_joint_velocities(joint_indices=indices),
                dtype=np.float64,
            ).reshape(-1)
        )
        if positions.shape != (7,) or velocities.shape != (7,):
            raise RuntimeError(f"unexpected live {side} arm state")
        return positions, velocities

    def _set_arm_targets(self, side: str, degrees: np.ndarray) -> None:
        for index, value in enumerate(degrees):
            joint = self._stage.GetPrimAtPath(
                f"/World/RBY1/joints/{side}_arm_{index}"
            )
            drive = self._UsdPhysics.DriveAPI.Get(joint, "angular")
            if not drive:
                raise RuntimeError(f"missing {side} arm drive {index}")
            drive.CreateTargetPositionAttr(float(value))
            drive.CreateTargetVelocityAttr(0.0)

    def _world_point(self, path: str, local_m: np.ndarray) -> np.ndarray:
        matrix = self._UsdGeom.Xformable(
            self._stage.GetPrimAtPath(path)
        ).ComputeLocalToWorldTransform(self._Usd.TimeCode.Default())
        return np.asarray(
            matrix.Transform(self._Gf.Vec3d(*local_m.tolist())),
            dtype=np.float64,
        )

    def _physics_tick(self, *, render: bool = False) -> None:
        previous_edge = self._blade.edge_centre_m.copy()
        self._airflow.step()
        _advance_one_physics_step(self._context, render=render)
        current_edge = self._blade.edge_centre_m.copy()
        self._blade.set_commanded_edge_velocity(
            (current_edge - previous_edge) * 240.0
        )
        self._grasp.process()
        self._apply_cut_decisions(
            self._context,
            self._blade,
            self._report,
            grasp_manager=self._grasp,
        )

    def _clear_report_episode_state(self) -> None:
        for key in (
            "physical_blade_cuts",
            "blade_traversal_cuts",
            "blade_cut_errors",
            "benchmark_failures",
        ):
            self._report.pop(key, None)

    def reset(self, *, seed: int) -> rl_env.DeleafState:
        self._episode_seed = int(seed)
        self._episode_index += 1
        random = np.random.default_rng(self._episode_seed)
        self._airflow.reset(phase_s=float(random.uniform(0.0, 10.0)))
        self._context.pause()
        try:
            self._grasp.reset_episode(
                self._target.vine_name,
                self._target.organ_label,
            )
            self._blade.reset_episode(
                self._target.vine_name,
                self._target.organ_label,
            )
            self._contacts.reset_episode()
            for runtime in self._runtimes:
                runtime.severer.reset()
            self._clear_report_episode_state()
            for snapshot in self._vine_snapshots:
                self._restore_articulation(snapshot)
            self._restore_articulation(self._robot_snapshot)
            for side, target in self._initial_drive_degrees.items():
                lower, upper = self._arm_limits[side]
                randomized = np.clip(
                    target
                    + random.uniform(
                        -self._reset_joint_noise_degrees,
                        self._reset_joint_noise_degrees,
                        size=7,
                    ),
                    lower,
                    upper,
                )
                self._set_arm_targets(side, randomized)
                self._arm_target_degrees[side] = randomized.copy()
                self._commanded_arm_velocity_degrees_s[side].fill(0.0)
            self._commanded_gripper_velocity_per_s = 0.0
            self._gripper_openness = 1.0
            self._grasp.request_open()
        finally:
            self._context.play()

        for _ in range(self._reset_settle_steps):
            self._physics_tick(render=False)
        # Reset settling is outside the policy episode and may contain stale
        # contact-lost callbacks from the previous trajectory.
        self._blade.reset_episode(
            self._target.vine_name,
            self._target.organ_label,
        )
        self._grasp.reset_episode(
            self._target.vine_name,
            self._target.organ_label,
        )
        self._contacts.reset_episode()
        self._report["online_rl_episode"] = {
            "episode_index": self._episode_index,
            "seed": self._episode_seed,
            "target": self._target.key,
            "joint_noise_degrees": self._reset_joint_noise_degrees,
            "airflow": self._airflow.summary,
        }
        return self._state()

    def apply_action(
        self,
        action: np.ndarray,
        parameters: rl_env.ActionParameters,
    ) -> rl_env.DeleafState:
        values = np.asarray(action, dtype=np.float64)
        targets = {}
        for side, action_slice in (("left", values[:7]), ("right", values[7:14])):
            desired_velocity = (
                action_slice * parameters.maximum_arm_speed_degrees_s
            )
            maximum_velocity_change = (
                parameters.maximum_arm_acceleration_degrees_s2
                / parameters.control_hz
            )
            previous_velocity = self._commanded_arm_velocity_degrees_s[side]
            commanded_velocity = np.clip(
                desired_velocity,
                previous_velocity - maximum_velocity_change,
                previous_velocity + maximum_velocity_change,
            )
            self._commanded_arm_velocity_degrees_s[side] = commanded_velocity
            delta = commanded_velocity / parameters.control_hz
            lower, upper = self._arm_limits[side]
            target = np.clip(
                self._arm_target_degrees[side] + delta, lower, upper
            )
            self._arm_target_degrees[side] = target.copy()
            self._set_arm_targets(side, target)
            targets[side] = target.tolist()

        desired_gripper_velocity = values[14] * parameters.maximum_gripper_speed_per_s
        maximum_gripper_velocity_change = (
            parameters.maximum_gripper_acceleration_per_s2 / parameters.control_hz
        )
        self._commanded_gripper_velocity_per_s = float(np.clip(
            desired_gripper_velocity,
            self._commanded_gripper_velocity_per_s - maximum_gripper_velocity_change,
            self._commanded_gripper_velocity_per_s + maximum_gripper_velocity_change,
        ))
        self._gripper_openness = float(
            np.clip(
                self._gripper_openness
                + self._commanded_gripper_velocity_per_s / parameters.control_hz,
                0.0,
                1.0,
            )
        )
        self._grasp.request_openness(self._gripper_openness)
        for substep in range(parameters.physics_steps_per_action):
            self._physics_tick(
                render=self._render
                and substep == parameters.physics_steps_per_action - 1
            )
        if self._frame_recorder is not None:
            images, errors = self._frame_recorder.capture(self._recorded_frames)
            self._report["online_rl_episode"]["latest_frames"] = images
            if errors:
                self._report["online_rl_episode"]["frame_errors"] = errors
            self._recorded_frames += 1
            self._report["online_rl_episode"]["recorded_frame_sets"] = (
                self._recorded_frames
            )
        self._report["online_rl_episode"].update(
            arm_target_degrees=targets,
            gripper_openness=self._gripper_openness,
            commanded_arm_velocity_degrees_s={
                side: values.tolist()
                for side, values in self._commanded_arm_velocity_degrees_s.items()
            },
            commanded_gripper_velocity_per_s=self._commanded_gripper_velocity_per_s,
        )
        return self._state(parameters)

    def _state(
        self,
        parameters: rl_env.ActionParameters | None = None,
    ) -> rl_env.DeleafState:
        parameters = parameters or rl_env.ActionParameters()
        position_parts = {}
        velocity_parts = {}
        for side in ("left", "right"):
            positions, velocities = self._joint_state(side)
            lower, upper = self._arm_limits[side]
            centre = 0.5 * (lower + upper)
            half_range = np.maximum(0.5 * (upper - lower), 1e-6)
            position_parts[side] = np.clip((positions - centre) / half_range, -1.0, 1.0)
            velocity_parts[side] = np.clip(
                velocities / parameters.maximum_arm_speed_degrees_s,
                -3.0,
                3.0,
            )

        grasp_geometry = self._grasp.target_geometry
        blade_geometry = self._blade.target_geometry
        if grasp_geometry is None or blade_geometry is None:
            raise RuntimeError("RL target geometry is unavailable")
        left_jaw = self._world_point("/World/RBY1/ee_left", _LEFT_JAW_CENTRE_M)
        blade_tool = self._blade.tool_point_geometry(np.zeros(3))

        grasp_summary = self._grasp.summary
        task = grasp_summary.get("task") or {}
        task_parameters = task.get("parameters") or {}
        minimum_grasp = max(float(task_parameters.get("minimum_grasp_force_n", 1.0)), 1e-6)
        minimum_transport = max(
            float(task_parameters.get("minimum_transport_clearance_m", 0.15)),
            1e-6,
        )
        feedback = self._blade.active_cut_feedback or {}
        target_summary = self._blade.summary.get("active_target_geometry") or {}
        required_force = max(float(target_summary.get("cut_force_n", 66.3)), 1e-6)
        radius = max(float(target_summary.get("radius_m", 0.003)), 1e-6)
        minimum_travel = float(
            self._blade.summary.get("parameters", {}).get(
                "minimum_cut_travel_m", 0.003
            )
        )
        required_work = required_force * max(2.0 * radius, minimum_travel)
        unsafe = self._unsafe_contacts(
            self._contacts.active_summary,
            blade_geometry,
            grasp_geometry,
        )
        safety_clear = self._blade.safety_clear and not unsafe

        return rl_env.DeleafState(
            left_joint_position=position_parts["left"],
            right_joint_position=position_parts["right"],
            left_joint_velocity=velocity_parts["left"],
            right_joint_velocity=velocity_parts["right"],
            gripper_openness=self._gripper_openness,
            left_grasp_delta_m=np.asarray(grasp_geometry["centre_m"]) - left_jaw,
            blade_cut_delta_m=np.asarray(blade_geometry["centre_m"])
            - np.asarray(blade_tool["point_m"]),
            target_axis=np.asarray(blade_geometry["axis"]),
            blade_edge_axis=np.asarray(blade_tool["edge_axis"]),
            blade_cut_direction=np.asarray(blade_tool["cut_direction"]),
            phase=str(task.get("phase", "seek_grasp")),
            grasp_force_fraction=float(task.get("maximum_grasp_force_n", 0.0))
            / minimum_grasp,
            cut_force_fraction=float(feedback.get("effective_force_n", 0.0))
            / required_force,
            cut_work_fraction=float(feedback.get("work_j", 0.0)) / required_work,
            transport_fraction=float(
                task.get("maximum_transport_clearance_m", 0.0)
            )
            / minimum_transport,
            unsafe_contact_count=len(unsafe),
            safety_clear=safety_clear,
            target_key=self._target.key,
        )


def _send_json(stream, payload: dict) -> None:
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
    stream.flush()


def serve(
    env: rl_env.OnlineDeleafEnv,
    *,
    host: str,
    port: int,
    app,
    report: dict,
    report_path: pathlib.Path,
    emit: Callable[[dict, pathlib.Path], None],
) -> dict:
    """Serve one synchronous JSON-lines RL client until it closes."""

    summary = {
        "schema": "greenhouse.online_rl.v1",
        "host": host,
        "port": int(port),
        "action_size": rl_env.ACTION_SIZE,
        "observation_size": rl_env.OBSERVATION_SIZE,
        "connections": 0,
        "resets": 0,
        "steps": 0,
        "episodes_completed": 0,
        "last_error": None,
    }
    report["online_rl"] = summary
    emit(report, report_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, int(port)))
        server.listen(1)
        server.settimeout(0.5)
        while app.is_running():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            summary["connections"] += 1
            with connection, connection.makefile("rwb") as stream:
                _send_json(
                    stream,
                    {
                        "ok": True,
                        "schema": summary["schema"],
                        "action_size": rl_env.ACTION_SIZE,
                        "observation_size": rl_env.OBSERVATION_SIZE,
                    },
                )
                for raw in stream:
                    try:
                        request = json.loads(raw)
                        command = request.get("command")
                        if command == "spec":
                            response = {
                                "ok": True,
                                "action_size": rl_env.ACTION_SIZE,
                                "observation_size": rl_env.OBSERVATION_SIZE,
                                "action_low": -1.0,
                                "action_high": 1.0,
                                "physics_steps_per_action": (
                                    env.action_parameters.physics_steps_per_action
                                ),
                                "maximum_episode_steps": env.maximum_episode_steps,
                            }
                        elif command == "reset":
                            observation, info = env.reset(seed=int(request.get("seed", 0)))
                            summary["resets"] += 1
                            response = {
                                "ok": True,
                                "observation": observation.tolist(),
                                "info": info,
                            }
                        elif command == "step":
                            observation, reward, terminated, truncated, info = env.step(
                                request.get("action")
                            )
                            summary["steps"] += 1
                            summary["episodes_completed"] += int(terminated or truncated)
                            response = {
                                "ok": True,
                                "observation": observation.tolist(),
                                "reward": reward,
                                "terminated": terminated,
                                "truncated": truncated,
                                "info": info,
                            }
                        elif command == "close":
                            _send_json(stream, {"ok": True, "closed": True})
                            report["online_rl"] = summary
                            emit(report, report_path)
                            return summary
                        else:
                            raise ValueError(f"unknown RL command: {command!r}")
                    except Exception as exc:
                        summary["last_error"] = str(exc)
                        response = {"ok": False, "error": str(exc)}
                    _send_json(stream, response)
                    report["online_rl"] = summary
                    emit(report, report_path)
    return summary
