"""Run the fitted RB-Y1 and physics-enabled tomato vines in the greenhouse.

The generated greenhouse scene stays immutable. Selected static vine references
are hidden in the USD session layer and replaced at the same bed transforms by
the verified articulated GLB rigs.

Controls while the simulation is running:
- Shift + left-drag a stem or petiole contact zone to pull with force.
- [ and ] select the previous or next deleafing petiole.
- V selects the next physics-enabled vine.
- C force-cuts the selected petiole for debugging only.
Benchmark cuts are triggered only by the right knife's physical leading edge.
The Vine Interaction window exposes the same selection and cut controls.
"""

# SimulationApp must exist before any omni or pxr import.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_DEFAULT_SCENE = pathlib.Path("data/greenhouse_sim/scenes/deleafing_bench.usd")
_DEFAULT_VINE_DIR = pathlib.Path("greenhouse/tomato_glb_20")
_DEFAULT_ROBOT = pathlib.Path("data/greenhouse_sim/robots/rby1a_v1.0.usd")
_DEFAULT_REPORT = pathlib.Path("data/greenhouse_sim/interactive_greenhouse.json")


_RBY1_CHASSIS_MIN_LOCAL_M = (-0.325, -0.250)
_RBY1_CHASSIS_MAX_LOCAL_M = (0.295, 0.250)
_MINIMUM_BASE_STRUCTURE_CLEARANCE_M = 0.03
_MINIMUM_GREENHOUSE_CLEARANCE_M = 0.01
_WARM_START_COMFORTABLE_CLEARANCE_M = 0.012


@dataclasses.dataclass
class VineRuntime:
    """One static greenhouse placement upgraded to a dynamic vine."""

    name: str
    source: pathlib.Path
    root_path: str
    plant: object
    skeletons: dict
    rig: object
    clips: list
    severer: object
    organ_indices: dict[str, int]
    cut_sites: dict[str, tuple[float, float, float]]
    rest_positions: object | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=pathlib.Path, default=_DEFAULT_SCENE)
    parser.add_argument("--vine-dir", type=pathlib.Path, default=_DEFAULT_VINE_DIR)
    parser.add_argument("--robot", type=pathlib.Path, default=_DEFAULT_ROBOT)
    parser.add_argument("--no-robot", action="store_true", help="run the accepted vine-only environment")
    parser.add_argument(
        "--robot-position",
        type=float,
        nargs=3,
        default=None,
        help="base XYZ; defaults to a target-gutter-relative pose on the local cultivation floor",
    )
    parser.add_argument(
        "--robot-position-mode",
        choices=("target-conditioned", "fixed"),
        default="target-conditioned",
        help="pre-position for a distal collision-clear target grasp, or preserve --robot-position exactly",
    )
    parser.add_argument("--robot-yaw", type=float, default=90.0)
    parser.add_argument("--physics-vines", type=int, default=1)
    parser.add_argument(
        "--target-vine",
        default="Vine_0002",
        help="physics-enabled vine name, or 'auto' for seeded selection",
    )
    parser.add_argument(
        "--target-organ",
        default="SubStem_00",
        help="physical petiole label, or 'auto' for seeded selection",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=0,
        help="deterministic target-selection seed when either target selector is auto",
    )
    parser.add_argument(
        "--neighbor-safety-vines",
        type=int,
        default=2,
        help="nearest static vines given sparse physical collision proxies",
    )
    parser.add_argument("--segment", type=float, default=0.02)
    parser.add_argument("--clip-spacing", type=float, default=0.30)
    parser.add_argument(
        "--collision-mode", choices=("interaction", "none", "all"), default="interaction"
    )
    parser.add_argument("--show-colliders", action="store_true")
    parser.add_argument("--cut-force", type=float, default=66.3)
    parser.add_argument("--cut-min-speed", type=float, default=0.01)
    parser.add_argument("--cut-max-axis-angle", type=float, default=35.0)
    parser.add_argument("--tear-force", type=float, default=0.0)
    parser.add_argument("--drag-stiffness", type=float, default=10.0)
    parser.add_argument("--drag-damping", type=float, default=0.02)
    parser.add_argument("--drag-max-force", type=float, default=1.0)
    parser.add_argument("--airflow-speed", type=float, default=1.0)
    parser.add_argument("--airflow-frequency", type=float, default=0.18)
    parser.add_argument("--airflow-direction", type=float, default=20.0)
    parser.add_argument("--airflow-probe-steps", type=int, default=0)
    parser.add_argument("--visual-pull-probe", action="store_true")
    parser.add_argument(
        "--bimanual-probe",
        choices=("left_approach", "right_approach", "full"),
        default=None,
        help="run a staged robot approach or full deleafing acceptance",
    )
    parser.add_argument(
        "--motion-steps",
        type=int,
        default=180,
        help="240 Hz steps used for each smooth arm waypoint",
    )
    parser.add_argument(
        "--drop-steps",
        type=int,
        default=1200,
        help="steps allowed for the released orphan to settle on the aisle floor",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screenshot", type=pathlib.Path, default=None)
    parser.add_argument(
        "--capture-camera",
        choices=("inspection", "head", "left_wrist", "right_wrist"),
        default="inspection",
        help="camera used by --screenshot",
    )
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--pull-probe", type=str, default=None, metavar="SUBSTEM")
    parser.add_argument("--pull-accel", type=float, default=500.0)
    parser.add_argument("--pull-steps", type=int, default=60)
    parser.add_argument("--recover-steps", type=int, default=240)
    parser.add_argument("--cut", type=str, default=None, metavar="SUBSTEM")
    parser.add_argument("--post-cut-steps", type=int, default=240)
    parser.add_argument(
        "--contact-diagnostics",
        action="store_true",
        help="record robot contact pairs and impulses during settling",
    )
    parser.add_argument(
        "--teleop-command-file",
        type=pathlib.Path,
        default=None,
        help="atomic greenhouse.teleop.v1 mailbox written by a simulator input device",
    )
    parser.add_argument(
        "--teleop-record-dir",
        type=pathlib.Path,
        default=None,
        help="parent directory for synchronized JSONL and D405 demonstration episodes",
    )
    parser.add_argument("--teleop-watchdog-ms", type=float, default=250.0)
    parser.add_argument("--teleop-max-joint-speed", type=float, default=240.0, metavar="DEG_S")
    parser.add_argument(
        "--teleop-contact-policy",
        choices=("monitor", "rollback"),
        default="monitor",
        help=(
            "log physical contacts without pausing, or roll back while an "
            "unsafe contact remains active"
        ),
    )
    parser.add_argument("--teleop-record-hz", type=float, default=10.0)
    parser.add_argument(
        "--teleop-cameras",
        nargs="+",
        choices=("head", "left_wrist", "right_wrist"),
        default=("head", "left_wrist", "right_wrist"),
    )
    parser.add_argument("--teleop-width", type=int, default=640)
    parser.add_argument("--teleop-height", type=int, default=360)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera-target", type=float, nargs=3, default=None)
    parser.add_argument("--report", type=pathlib.Path, default=_DEFAULT_REPORT)
    return parser.parse_args()


def _emit(report: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    scene_path = args.scene.resolve()
    vine_dir = args.vine_dir.resolve()
    robot_path = args.robot.resolve()
    if not scene_path.exists():
        print(f"scene not found: {scene_path}; run build_scene.py first")
        return 1
    vine_sources = sorted(vine_dir.glob("tomato_*.glb"))
    if not vine_sources:
        print(f"no tomato GLBs found under {vine_dir}")
        return 1
    if args.physics_vines < 1:
        print("--physics-vines must be at least 1")
        return 1
    if args.neighbor_safety_vines < 0:
        print("--neighbor-safety-vines cannot be negative")
        return 1
    if args.teleop_command_file is not None and args.no_robot:
        print("--teleop-command-file requires the fitted robot")
        return 1
    if args.teleop_record_dir is not None and args.teleop_command_file is None:
        print("--teleop-record-dir requires --teleop-command-file")
        return 1
    if args.teleop_watchdog_ms <= 0.0 or args.teleop_max_joint_speed <= 0.0:
        print("teleop watchdog and maximum joint speed must be positive")
        return 1
    if args.teleop_record_hz <= 0.0 or args.teleop_width < 1 or args.teleop_height < 1:
        print("teleop recording rate and dimensions must be positive")
        return 1
    if not args.no_robot and not robot_path.exists():
        print(f"fitted robot not found: {robot_path}; run build_robot.py first or pass --no-robot")
        return 1

    report: dict = {
        "stage": "starting",
        "scene": str(scene_path),
        "physics_vines_requested": args.physics_vines,
        "collision_mode": args.collision_mode,
        "robot_requested": not args.no_robot,
    }
    _emit(report, args.report)

    from isaacsim import SimulationApp

    headless = args.headless or args.screenshot is not None
    app = SimulationApp({"headless": headless})

    from greenhouse_sim import base_planner
    from greenhouse_sim import cutting
    from greenhouse_sim import deleaf_task
    from greenhouse_sim import episode
    from greenhouse_sim import glb
    from greenhouse_sim import organs
    from greenhouse_sim import robot_hardware
    from greenhouse_sim import robot_kinematics
    from greenhouse_sim import robot_scene
    from greenhouse_sim import skeleton as skeleton_module
    from greenhouse_sim import teleop
    from greenhouse_sim import vine_interaction
    from greenhouse_sim import vine_physics
    from greenhouse_sim import vine_usd
    from greenhouse_sim import vine_visuals
    import numpy as np
    import omni.usd
    import time
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    omni.usd.get_context().open_stage(str(scene_path))
    for _ in range(10):
        app.update()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())

    robot_placement = None
    teleop_startup_gripper_openness = None

    static_scope = stage.GetPrimAtPath("/World/Vines")
    static_vines = list(static_scope.GetChildren()) if static_scope and static_scope.IsValid() else []
    try:
        selected = _select_physics_vines(
            static_vines,
            args.target_vine,
            args.physics_vines,
        )
    except ValueError as exc:
        report.update(stage="failed", error=str(exc))
        _emit(report, args.report)
        app.close()
        return 1
    if not selected:
        report.update(stage="failed", error="no /World/Vines placements in scene")
        _emit(report, args.report)
        app.close()
        return 1

    vine_physics.apply_scene_physics(stage)
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/InteractiveVines"))

    runtimes: list[VineRuntime] = []
    safety_colliders = []
    placement_centres = []
    for static_prim in selected:
        placement = UsdGeom.Xformable(static_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        placement_centre = np.array(placement.ExtractTranslation(), dtype=np.float64)
        placement_centres.append(placement_centre)
        source = _source_for_placement(static_prim.GetName(), vine_sources)
        static_prim.SetActive(False)

        plant = organs.load_plant(source)
        asset = glb.read_glb(source)
        skeletons = skeleton_module.skeletonise_plant(plant, segment_length=args.segment)
        points_to_stage, directions_to_stage = _placement_transforms(placement, vine_usd, Gf, np)

        root_path = f"/World/InteractiveVines/{static_prim.GetName()}"
        UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
        rig = vine_physics.author_plant_physics(
            stage,
            plant,
            root_path,
            skeletons,
            points_to_stage,
            properties=vine_physics.TissueProperties(
                cut_force_n=args.cut_force,
                tear_force_n=args.tear_force,
            ),
            visible_colliders=args.show_colliders,
            collision_mode=args.collision_mode,
        )
        vine_visuals.attach_organ_visuals(
            stage,
            rig,
            plant,
            asset,
            points_to_stage,
            to_stage_directions=directions_to_stage,
        )
        if args.collision_mode == "interaction":
            foliage_colliders = vine_physics.author_foliage_contact_proxies(
                stage,
                rig,
                plant,
                points_to_stage,
                visible=args.show_colliders,
            )
            rig = dataclasses.replace(
                rig,
                collider_paths=rig.collider_paths
                + tuple(info.path for info in foliage_colliders),
                colliders=rig.colliders + foliage_colliders,
            )
        clips = vine_physics.add_trellis_clips(stage, rig, plant.root, spacing=args.clip_spacing)
        # The supplied greenhouse already owns collision ground at the measured
        # cultivation-floor height. A hidden tray at gutter height made an
        # orphan appear deposited while it was still 1.19 m above the aisle.

        organ_indices = {organ.label: organ.index for organ in plant.organs}
        cut_sites = {
            organ.label: tuple(float(value) for value in points_to_stage([organ.attachment])[0])
            for organ in plant.organs
            if organ.label.startswith("SubStem_") and organ.attachment is not None
        }
        runtimes.append(
            VineRuntime(
                name=static_prim.GetName(),
                source=source,
                root_path=root_path,
                plant=plant,
                skeletons=skeletons,
                rig=rig,
                clips=clips,
                severer=cutting.Severer(stage, rig, skeletons, organ_indices),
                organ_indices=organ_indices,
                cut_sites=cut_sites,
            )
        )

    UsdGeom.Scope.Define(stage, Sdf.Path("/World/NeighbourSafety"))
    selected_paths = {str(prim.GetPath()) for prim in selected}
    safety_candidates = []
    for static_prim in static_vines:
        if str(static_prim.GetPath()) in selected_paths:
            continue
        placement = UsdGeom.Xformable(static_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        centre = np.asarray(placement.ExtractTranslation(), dtype=np.float64)
        distance = float(np.linalg.norm(centre[:2] - placement_centres[0][:2]))
        safety_candidates.append((distance, static_prim, placement))
    safety_candidates.sort(key=lambda item: item[0])
    for _, static_prim, placement in safety_candidates[: args.neighbor_safety_vines]:
        source = _source_for_placement(static_prim.GetName(), vine_sources)
        plant = organs.load_plant(source)
        skeletons = skeleton_module.skeletonise_plant(
            plant,
            segment_length=args.segment,
        )
        points_to_stage, _ = _placement_transforms(placement, vine_usd, Gf, np)
        safety_colliders.extend(
            vine_physics.author_safety_proxies(
                stage,
                plant,
                f"/World/NeighbourSafety/{static_prim.GetName()}",
                skeletons,
                points_to_stage,
                vine_name=static_prim.GetName(),
                visible=args.show_colliders,
            )
        )

    camera_path = _author_camera(stage, placement_centres[0], args, Gf, Sdf, UsdGeom)
    try:
        selected_target = episode.resolve_target(
            runtimes,
            vine_name=args.target_vine,
            organ_label=args.target_organ,
            seed=args.episode_seed,
        )
    except ValueError as exc:
        report.update(stage="failed", error=str(exc))
        _emit(report, args.report)
        app.close()
        return 1
    selected_runtime = next(
        runtime for runtime in runtimes if runtime.name == selected_target.vine_name
    )

    def target_conditioned_base_plan(
        nominal_position_m,
        lateral_offsets_m,
        diagnostics,
        gripper_geometry=None,
    ):
        camera_centre_m, camera_radius_m = robot_hardware.wrist_d405_body_sphere(side="left")
        payload_boxes = [
            ("wrist_d405", *robot_hardware.wrist_d405_body_box(side="left")),
            (
                "wrist_camera_bracket",
                *robot_hardware.wrist_camera_bracket_box(side="left"),
            ),
        ]
        for name, geometry in (
            (gripper_geometry or {}).get("fingers") or {}
        ).items():
            minimum = np.asarray(
                geometry["collider_min_ee_m"], dtype=np.float64
            )
            maximum = np.asarray(
                geometry["collider_max_ee_m"], dtype=np.float64
            )
            payload_boxes.append(
                (
                    name,
                    0.5 * (minimum + maximum),
                    np.eye(3, dtype=np.float64),
                    0.5 * (maximum - minimum),
                )
            )
        return base_planner.plan_target_conditioned_base(
            robot_kinematics.Rby1Kinematics(),
            nominal_position_m=nominal_position_m,
            yaw_degrees=args.robot_yaw,
            candidates=_target_grasp_candidates(
                stage,
                selected_runtime,
                selected_target.organ_label,
                base_planner,
            ),
            obstacles=_vine_capsule_obstacles(stage, robot_kinematics),
            jaw_local_point_m=_LEFT_JAW_CENTRE_M,
            camera_local_centre_m=camera_centre_m,
            camera_radius_m=camera_radius_m,
            left_payload_boxes=tuple(payload_boxes),
            minimum_payload_clearance_m=0.005,
            seeds=(
                _LEFT_AISLE_CLEARANCE_WAYPOINTS_DEGREES[-1],
                _LEFT_APPROACH_SEEDS_DEGREES[0.0],
                *_LEFT_MULTISTART_SEEDS_DEGREES,
            ),
            advances_m=(0.0,),
            lateral_offsets_m=lateral_offsets_m,
            left_waiting_degrees=_LEFT_READY_DEGREES,
            left_approach_start_degrees=_LEFT_READY_DEGREES,
            left_approach_waypoint_routes=(
                (),
                (_LEFT_APPROACH_SEEDS_DEGREES[0.10],),
                (_LEFT_APPROACH_SEEDS_DEGREES[0.06],),
                (_LEFT_APPROACH_SEEDS_DEGREES[0.04],),
                (_LEFT_APPROACH_SEEDS_DEGREES[0.02],),
                (_LEFT_MULTISTART_SEEDS_DEGREES[0],),
                tuple(
                    _LEFT_APPROACH_SEEDS_DEGREES[offset]
                    for offset in (0.10, 0.06, 0.04, 0.02)
                ),
            ),
            right_waiting_degrees=_RIGHT_SAFE_DEGREES,
            reach_reserve_m=0.0,
            minimum_trajectory_clearance_m=0.005,
            trajectory_samples=31,
            joint_space_search_iterations=2500,
            joint_space_search_seed=args.episode_seed,
            minimum_inter_arm_clearance_m=0.05,
            diagnostics=diagnostics,
        )
    greenhouse_structure_obstacles = ()
    greenhouse_wall_colliders = ()
    if not args.no_robot:
        target_reference = np.asarray(
            selected_runtime.cut_sites[selected_target.organ_label],
            dtype=np.float64,
        )
        aisle_bounds = _approach_aisle_base_bounds(
            stage,
            target_reference,
            args.robot_yaw,
            Usd,
            UsdGeom,
            np,
        )
        requested_robot_position = (
            None
            if args.robot_position is None
            else tuple(float(value) for value in args.robot_position)
        )
        if args.robot_position is None:
            args.robot_position = _default_robot_position(
                stage,
                target_reference,
                args.robot_yaw,
                Usd,
                UsdGeom,
                np,
            )
        elif args.robot_position_mode == "target-conditioned":
            args.robot_position = (
                float(args.robot_position[0]),
                aisle_bounds["selected_base_y_m"],
                float(args.robot_position[2]),
            )
        elif not (
            aisle_bounds["minimum_base_y_m"]
            <= float(args.robot_position[1])
            <= aisle_bounds["maximum_base_y_m"]
        ):
            report.update(
                stage="failed",
                error="fixed robot position intersects the gutter or greenhouse wall",
                opposite_aisle=aisle_bounds,
            )
            _emit(report, args.report)
            app.close()
            return 1
        nominal_robot_position = tuple(float(value) for value in args.robot_position)
        if args.robot_position_mode == "target-conditioned":
            base_diagnostics = {}
            base_plan = target_conditioned_base_plan(
                nominal_robot_position,
                (
                    0.0,
                    -0.15,
                    0.15,
                    -0.30,
                    0.30,
                    -0.45,
                    0.45,
                ),
                base_diagnostics,
            )
            if base_plan is None:
                report.update(
                    stage="failed",
                    error=(
                        "no target-conditioned base pose reaches a distal petiole "
                        "segment with wrist-camera clearance"
                    ),
                    base_planning=base_diagnostics,
                )
                _emit(report, args.report)
                app.close()
                return 1
            args.robot_position = base_plan.position_m
            report["robot_preposition"] = {
                "mode": args.robot_position_mode,
                **dataclasses.asdict(base_plan),
                "requested_position_m": requested_robot_position,
                "opposite_aisle": aisle_bounds,
            }
        else:
            report["robot_preposition"] = {
                "mode": args.robot_position_mode,
                "nominal_position_m": nominal_robot_position,
                "position_m": nominal_robot_position,
                "offset_m": (0.0, 0.0, 0.0),
                "requested_position_m": requested_robot_position,
                "opposite_aisle": aisle_bounds,
            }
        yaw_radians = np.radians(float(args.robot_yaw))
        robot_forward_xy = np.asarray(
            [np.cos(yaw_radians), np.sin(yaw_radians)],
            dtype=np.float64,
        )
        target_delta_xy = target_reference[:2] - np.asarray(
            args.robot_position[:2], dtype=np.float64
        )
        target_forward_distance_m = float(
            np.dot(target_delta_xy, robot_forward_xy)
        )
        report["robot_preposition"]["target_forward_distance_m"] = (
            target_forward_distance_m
        )
        if target_forward_distance_m <= 0.15:
            report.update(
                stage="failed",
                error="selected tomato petiole is not in front of the robot",
            )
            _emit(report, args.report)
            app.close()
            return 1
        robot_placement = robot_scene.add_fitted_robot(
            stage,
            robot_path,
            position_m=args.robot_position,
            yaw_degrees=args.robot_yaw,
        )
        if args.teleop_command_file is not None:
            startup_targets = {
                "left": tuple(
                    robot_scene.SDK_READY_POSE_DEGREES[f"left_arm_{index}"]
                    for index in range(7)
                ),
                "right": tuple(
                    robot_scene.SDK_READY_POSE_DEGREES[f"right_arm_{index}"]
                    for index in range(7)
                ),
                "torso": tuple(
                    robot_scene.SDK_READY_POSE_DEGREES[f"torso_{index}"]
                    for index in range(6)
                ),
                "head": tuple(
                    robot_scene.SDK_READY_POSE_DEGREES[f"head_{index}"]
                    for index in range(2)
                ),
            }
            startup_source = "sdk_ready_fallback"
            startup_error = None
            startup_age_ms = None
            try:
                startup_payload = json.loads(
                    pathlib.Path(args.teleop_command_file).read_text(
                        encoding="utf-8"
                    )
                )
                startup_command = teleop.parse_command(startup_payload)
                startup_age_s = time.monotonic() - startup_command.monotonic_time_s
                if startup_age_s < -0.050:
                    raise teleop.TeleopCommandError(
                        "startup command timestamp is in the future"
                    )
                if startup_age_s > args.teleop_watchdog_ms / 1000.0:
                    raise teleop.TeleopCommandError(
                        f"startup command is stale ({startup_age_s * 1000.0:.1f} ms)"
                    )
                if startup_command.left.enabled:
                    startup_targets["left"] = startup_command.left.joint_degrees
                if startup_command.right.enabled:
                    startup_targets["right"] = startup_command.right.joint_degrees
                if startup_command.torso is not None and startup_command.torso.enabled:
                    startup_targets["torso"] = startup_command.torso.joint_degrees
                if startup_command.head is not None and startup_command.head.enabled:
                    startup_targets["head"] = startup_command.head.joint_degrees
                if (
                    startup_command.left_gripper is not None
                    and startup_command.left_gripper.enabled
                ):
                    teleop_startup_gripper_openness = (
                        startup_command.left_gripper.openness
                    )
                startup_source = "fresh_measured_robot_mailbox"
                startup_age_ms = max(startup_age_s, 0.0) * 1000.0
            except (
                FileNotFoundError,
                OSError,
                json.JSONDecodeError,
                teleop.TeleopCommandError,
            ) as exc:
                startup_error = str(exc)
            for group, target_degrees in startup_targets.items():
                _set_joint_group_initial_state(stage, group, target_degrees)
            report["teleop_startup_pose"] = {
                "source": startup_source,
                "command_age_ms": startup_age_ms,
                "fallback_reason": startup_error,
                "joint_degrees": {
                    group: list(values)
                    for group, values in startup_targets.items()
                },
                "left_gripper_openness": teleop_startup_gripper_openness,
            }
        greenhouse_structure_obstacles = _greenhouse_structure_obstacles(
            stage,
            robot_kinematics,
            args.robot_position,
        )
        greenhouse_wall_colliders = _author_greenhouse_wall_colliders(
            stage,
            greenhouse_structure_obstacles,
        )
        report["robot"] = dataclasses.asdict(robot_placement)
        report["greenhouse_safety"] = {
            "planning_obstacle_count": len(greenhouse_structure_obstacles),
            "planning_obstacles": [
                dataclasses.asdict(obstacle)
                for obstacle in greenhouse_structure_obstacles
            ],
            "physical_wall_colliders": list(greenhouse_wall_colliders),
        }
    report.update(
        stage="rigged",
        static_vines=len(static_vines),
        physics_vines=len(runtimes),
        links=sum(len(runtime.rig.links) for runtime in runtimes),
        clips=sum(len(runtime.clips) for runtime in runtimes),
        severable=sum(len(runtime.rig.cut_joints) for runtime in runtimes),
        contact_colliders=sum(len(runtime.rig.collider_paths) for runtime in runtimes),
        neighbour_safety_vines=len(
            {collider.vine_name for collider in safety_colliders}
        ),
        neighbour_safety_colliders=len(safety_colliders),
        sources=[str(runtime.source) for runtime in runtimes],
        episode_target={
            **dataclasses.asdict(selected_target),
            "key": selected_target.key,
            "selection_mode": (
                "seeded_auto"
                if "auto" in (args.target_vine, args.target_organ)
                else "exact"
            ),
        },
    )
    _emit(report, args.report)

    for _ in range(10):
        app.update()

    from isaacsim.core.api import SimulationContext

    contact_diagnostics = (
        RobotContactDiagnostics(stage)
        if (
            args.contact_diagnostics
            or args.bimanual_probe is not None
            or args.teleop_command_file is not None
            or (not headless and robot_placement is not None)
        )
        else None
    )
    blade_cutting = None
    grasp_manager = None
    if robot_placement is not None:
        blade_cutting = BladeContactMonitor(
            stage,
            runtimes,
            robot_placement,
            cutting.CutGateParameters(
                minimum_forward_speed_m_s=args.cut_min_speed,
                maximum_axis_dot=float(
                    np.sin(np.radians(args.cut_max_axis_angle))
                ),
            ),
        )
        blade_cutting.add_safety_colliders(safety_colliders)
        blade_cutting.set_active_target(
            selected_target.vine_name,
            selected_target.organ_label,
        )
        report["blade_cutting"] = blade_cutting.summary
        floor_z = float(args.robot_position[2])
        grasp_manager = LeftGraspManager(
            stage,
            runtimes,
            robot_placement,
            deleaf_task.TaskParameters(
                drop_zone_min_m=(
                    float(args.robot_position[0]) - 1.20,
                    float(args.robot_position[1]) - 1.00,
                    floor_z,
                ),
                drop_zone_max_m=(
                    float(args.robot_position[0]) - 0.30,
                    float(args.robot_position[1]) + 0.70,
                    floor_z + 0.60,
                ),
            ),
        )
        grasp_manager.set_active_target(
            selected_target.vine_name,
            selected_target.organ_label,
        )
        if teleop_startup_gripper_openness is not None:
            grasp_manager.request_openness(teleop_startup_gripper_openness)
        blade_cutting.set_counterhold_provider(grasp_manager.holds_target)
        report["bimanual_task"] = grasp_manager.summary
    context = SimulationContext(
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
    )
    context.initialize_physics()
    if contact_diagnostics is not None:
        contact_diagnostics.subscribe()
    if blade_cutting is not None:
        blade_cutting.subscribe()
    if grasp_manager is not None:
        grasp_manager.subscribe()
    context.get_physics_context().set_gravity(-9.81)
    for runtime in runtimes:
        runtime.rest_positions = _positions(stage, _base_link_paths(runtime.rig))

    if not headless:
        report["native_transform_selector_disabled"] = _disable_native_mouse_interaction(app)
        _focus_viewport(camera_path)

    context.play()
    for _ in range(args.settle_steps):
        context.step(render=False)
        if grasp_manager is not None:
            grasp_manager.process()
        if blade_cutting is not None:
            _apply_blade_cut_decisions(
                context,
                blade_cutting,
                report,
                grasp_manager=grasp_manager,
            )

    if (
        robot_placement is not None
        and args.robot_position_mode == "target-conditioned"
    ):
        settled_base_diagnostics = {}
        settled_base_plan = target_conditioned_base_plan(
            tuple(float(value) for value in args.robot_position),
            (0.0,),
            settled_base_diagnostics,
            grasp_manager.summary.get("gripper_geometry"),
        )
        report["settled_base_planning"] = settled_base_diagnostics
        if settled_base_plan is None:
            report.update(
                stage="failed",
                succeeded=False,
                error=(
                    "no collision-clear left grasp route exists at the parked "
                    "robot base after vine settling"
                ),
            )
            _emit(report, args.report)
            context.stop()
            app.close()
            return 1
        previous_preposition = report["robot_preposition"]
        report["robot_preposition"] = {
            "mode": args.robot_position_mode,
            **dataclasses.asdict(settled_base_plan),
            "requested_position_m": previous_preposition.get(
                "requested_position_m"
            ),
            "opposite_aisle": previous_preposition.get("opposite_aisle"),
            "target_forward_distance_m": previous_preposition.get(
                "target_forward_distance_m"
            ),
            "geometry_source": "post_settle_live_vines",
        }
        _emit(report, args.report)
    if headless:
        success = _run_headless_checks(
            stage,
            context,
            runtimes,
            selected_target,
            args,
            report,
        )
        if blade_cutting is not None:
            report["initial_tool_safety_clear"] = blade_cutting.safety_clear
            success = success and blade_cutting.safety_clear

        if args.bimanual_probe is not None:
            probe = _bimanual_probe(
                stage,
                context,
                selected_runtime,
                args,
                report,
                blade_cutting,
                grasp_manager,
                contact_diagnostics,
                robot_hardware,
                robot_kinematics,
                greenhouse_structure_obstacles,
            )
            report["bimanual_probe"] = probe
            success = success and bool(probe["succeeded"])
        if contact_diagnostics is not None:
            report["robot_contacts"] = contact_diagnostics.summary
            contact_diagnostics.close()
        if args.screenshot is not None:
            capture_camera = camera_path
            if args.capture_camera != "inspection":
                if robot_placement is None:
                    raise ValueError("a fitted robot is required to capture a D405 camera")
                camera_indices = {"head": 0, "left_wrist": 1, "right_wrist": 2}
                capture_camera = robot_placement.cameras[camera_indices[args.capture_camera]]
            _capture(capture_camera, args, app)
            report["screenshot"] = str(args.screenshot)
            report["capture_camera"] = capture_camera
        report.update(stage="done", succeeded=bool(success))
        if blade_cutting is not None:
            report["blade_cutting"] = blade_cutting.summary
            blade_cutting.close()
        if grasp_manager is not None:
            report["bimanual_task"] = grasp_manager.summary
            grasp_manager.close()
        _emit(report, args.report)
        context.stop()
        app.close()
        return 0 if success else 1

    if contact_diagnostics is not None:
        report["robot_contacts"] = contact_diagnostics.summary

    # Render the selected inspection camera before the overlay reads its
    # projection matrices. All settling frames above are deliberately headless.
    for _ in range(4):
        context.step(render=True)
    camera_views = {"inspection": camera_path}
    if robot_placement is not None:
        camera_views.update(
            head=robot_placement.cameras[0],
            left_wrist=robot_placement.cameras[1],
            right_wrist=robot_placement.cameras[2],
        )

    def run_full_ik_from_ui() -> dict:
        if (
            robot_placement is None
            or blade_cutting is None
            or grasp_manager is None
            or contact_diagnostics is None
        ):
            return {
                "mode": "full",
                "succeeded": False,
                "error": "fitted robot and safety monitors are required",
            }
        original_mode = args.bimanual_probe
        args.bimanual_probe = "full"
        try:
            baseline_clear = _run_headless_checks(
                stage,
                context,
                runtimes,
                selected_target,
                args,
                report,
            )
            probe = _bimanual_probe(
                stage,
                context,
                selected_runtime,
                args,
                report,
                blade_cutting,
                grasp_manager,
                contact_diagnostics,
                robot_hardware,
                robot_kinematics,
                greenhouse_structure_obstacles,
            )
        finally:
            args.bimanual_probe = original_mode
        report["bimanual_probe"] = probe
        report["robot_contacts"] = contact_diagnostics.summary
        report["blade_cutting"] = blade_cutting.summary
        report["bimanual_task"] = grasp_manager.summary
        report.update(
            stage="probe_complete",
            succeeded=bool(baseline_clear and probe["succeeded"]),
        )
        _emit(report, args.report)
        return probe

    controller = InteractionController(
        stage,
        runtimes,
        report,
        args.report,
        args,
        vine_interaction,
        camera_views,
        blade_cutting,
        grasp_manager,
        selected_target,
        run_full_ik=run_full_ik_from_ui,
    )
    teleop_controller = None
    if args.teleop_command_file is not None:
        # Replicator render products need explicit warmup, but that must not
        # advance unobserved physics or accumulate cut/grasp contacts.
        context.pause()
        try:
            teleop_controller = SimulatorTeleop(
                stage,
                args,
                camera_views,
                selected_target,
                report,
                args.report,
                teleop,
                robot_hardware,
                robot_kinematics,
                blade_cutting,
                grasp_manager,
                contact_diagnostics,
            )
        finally:
            context.play()
    report.update(stage="running", controls=controller.controls)
    _emit(report, args.report)
    if args.bimanual_probe is not None:
        success = _run_headless_checks(
            stage,
            context,
            runtimes,
            selected_target,
            args,
            report,
        )
        probe = _bimanual_probe(
            stage,
            context,
            selected_runtime,
            args,
            report,
            blade_cutting,
            grasp_manager,
            contact_diagnostics,
            robot_hardware,
            robot_kinematics,
            greenhouse_structure_obstacles,
        )
        report["bimanual_probe"] = probe
        success = success and bool(probe["succeeded"])
        if contact_diagnostics is not None:
            report["robot_contacts"] = contact_diagnostics.summary
        if blade_cutting is not None:
            report["blade_cutting"] = blade_cutting.summary
        if grasp_manager is not None:
            report["bimanual_task"] = grasp_manager.summary
        report.update(stage="probe_complete", succeeded=bool(success))
        _emit(report, args.report)
        print(
            "Visible bimanual probe complete; close the window when inspection "
            "is finished."
        )
    if args.visual_pull_probe:
        visual_probe = controller.run_visual_pull_probe(context, stage, runtimes[0])
        report["visual_pull_probe"] = visual_probe
        report.update(stage="done", succeeded=bool(visual_probe["succeeded"]))
        _emit(report, args.report)
        controller.close()
        context.stop()
        app.close()
        return 0 if visual_probe["succeeded"] else 1
    print(
        "Shift-drag pulls; [ / ] selects; G closes left grasp; O releases; "
        "C DEBUG-force-cuts; V changes vine; 1-4 switches cameras"
    )
    try:
        simulation_step = 0
        while app.is_running():
            if teleop_controller is not None:
                teleop_controller.process(simulation_step)
            context.step(render=True)
            simulation_step += 1
            controller.process(context)
            if args.tear_force > 0.0:
                controller.poll_tears()
    finally:
        if teleop_controller is not None:
            teleop_controller.close()
        controller.close()
        if blade_cutting is not None:
            report["blade_cutting"] = blade_cutting.summary
            blade_cutting.close()
        if grasp_manager is not None:
            report["bimanual_task"] = grasp_manager.summary
            grasp_manager.close()
        if contact_diagnostics is not None:
            report["robot_contacts"] = contact_diagnostics.summary
            contact_diagnostics.close()
        report["stage"] = "closed"
        _emit(report, args.report)
        context.stop()
        app.close()
    return 0


def _source_for_placement(name: str, sources: list[pathlib.Path]) -> pathlib.Path:
    match = re.search(r"(\d+)$", name)
    index = int(match.group(1)) if match else 0
    return sources[index % len(sources)]


def _select_physics_vines(static_vines, target_vine: str, count: int):
    """Select the requested target first, then fill the remaining budget."""
    budget = min(int(count), len(static_vines))
    if target_vine == "auto":
        return list(static_vines[:budget])
    target = next(
        (prim for prim in static_vines if prim.GetName() == target_vine),
        None,
    )
    if target is None:
        available = ", ".join(prim.GetName() for prim in static_vines)
        raise ValueError(
            f"target vine {target_vine!r} is not present; available: {available}"
        )
    return [target, *[prim for prim in static_vines if prim != target]][:budget]


def _opposite_aisle_base_bounds(
    stage,
    placement,
    yaw_degrees: float,
    Usd,
    UsdGeom,
    np,
) -> dict:
    """Return a chassis-clear base interval between the gutter and outer wall."""
    target = np.asarray(placement, dtype=np.float64)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bed_candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.endswith("/Bed/Bed_Bed/mesh"):
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower = np.asarray(extent.GetMin(), dtype=np.float64)
        upper = np.asarray(extent.GetMax(), dtype=np.float64)
        if lower[0] - 1e-6 <= target[0] <= upper[0] + 1e-6:
            bed_candidates.append(
                (
                    abs(float(0.5 * (lower[1] + upper[1]) - target[1])),
                    path,
                    lower,
                    upper,
                )
            )
    if not bed_candidates:
        raise RuntimeError("could not resolve the selected gutter bed bounds")
    _, bed_path, _, bed_upper = min(bed_candidates, key=lambda item: item[0])

    wall_candidates = []
    environment = stage.GetPrimAtPath("/World/Main_Cultivation_Zone/Env")
    for prim in environment.GetChildren() if environment.IsValid() else ():
        if not prim.GetName().startswith("Wall_"):
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower = np.asarray(extent.GetMin(), dtype=np.float64)
        upper = np.asarray(extent.GetMax(), dtype=np.float64)
        if lower[1] <= target[1] or upper[2] - lower[2] < 2.0:
            continue
        wall_candidates.append((float(lower[1]), str(prim.GetPath()), lower, upper))
    if not wall_candidates:
        raise RuntimeError("could not resolve the positive-Y greenhouse wall")
    wall_inner_y, wall_path, _, _ = min(wall_candidates, key=lambda item: item[0])

    yaw = np.radians(float(yaw_degrees))
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    world_y_offsets = [
        sine * local_x + cosine * local_y
        for local_x in (_RBY1_CHASSIS_MIN_LOCAL_M[0], _RBY1_CHASSIS_MAX_LOCAL_M[0])
        for local_y in (_RBY1_CHASSIS_MIN_LOCAL_M[1], _RBY1_CHASSIS_MAX_LOCAL_M[1])
    ]
    minimum_y = (
        float(bed_upper[1])
        + _MINIMUM_BASE_STRUCTURE_CLEARANCE_M
        - min(world_y_offsets)
    )
    maximum_y = (
        wall_inner_y
        - _MINIMUM_BASE_STRUCTURE_CLEARANCE_M
        - max(world_y_offsets)
    )
    if minimum_y > maximum_y:
        raise RuntimeError(
            f"RBY1 chassis does not fit opposite aisle: {minimum_y:.3f}>{maximum_y:.3f}"
        )
    return {
        "minimum_base_y_m": minimum_y,
        "maximum_base_y_m": maximum_y,
        "selected_base_y_m": 0.5 * (minimum_y + maximum_y),
        "gutter_outer_y_m": float(bed_upper[1]),
        "wall_inner_y_m": wall_inner_y,
        "bed_path": bed_path,
        "wall_path": wall_path,
    }


def _approach_aisle_base_bounds(
    stage,
    placement,
    yaw_degrees: float,
    Usd,
    UsdGeom,
    np,
) -> dict:
    """Return the chassis-clear aisle on the robot-facing side of a gutter."""
    target = np.asarray(placement, dtype=np.float64)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    beds = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.endswith("/Bed/Bed_Bed/mesh"):
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower = np.asarray(extent.GetMin(), dtype=np.float64)
        upper = np.asarray(extent.GetMax(), dtype=np.float64)
        if lower[0] - 1e-6 <= target[0] <= upper[0] + 1e-6:
            beds.append(
                (
                    float(0.5 * (lower[1] + upper[1])),
                    path,
                    lower,
                    upper,
                )
            )
    if not beds:
        raise RuntimeError("could not resolve the selected gutter bed bounds")

    target_bed = min(beds, key=lambda item: abs(item[0] - target[1]))
    target_centre_y, bed_path, bed_lower, bed_upper = target_bed
    yaw = np.radians(float(yaw_degrees))
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    if abs(sine) < 0.5:
        raise RuntimeError("greenhouse approach requires the robot to face +/-Y")

    if sine > 0.0:
        neighbours = [
            bed for bed in beds if bed[0] < target_centre_y - 1e-3
        ]
        if not neighbours:
            raise RuntimeError("could not resolve the neighbouring negative-Y gutter")
        neighbour = max(neighbours, key=lambda item: item[0])
        lower_boundary_y = float(neighbour[3][1])
        upper_boundary_y = float(bed_lower[1])
        lower_boundary_path = neighbour[1]
        upper_boundary_path = bed_path
        aisle_side = "inter_gutter_negative_y"
    else:
        walls = []
        environment = stage.GetPrimAtPath("/World/Main_Cultivation_Zone/Env")
        for prim in environment.GetChildren() if environment.IsValid() else ():
            if not prim.GetName().startswith("Wall_"):
                continue
            extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = np.asarray(extent.GetMin(), dtype=np.float64)
            upper = np.asarray(extent.GetMax(), dtype=np.float64)
            if lower[1] <= target[1] or upper[2] - lower[2] < 2.0:
                continue
            walls.append((float(lower[1]), str(prim.GetPath())))
        if not walls:
            raise RuntimeError("could not resolve the positive-Y greenhouse wall")
        wall_inner_y, wall_path = min(walls, key=lambda item: item[0])
        lower_boundary_y = float(bed_upper[1])
        upper_boundary_y = wall_inner_y
        lower_boundary_path = bed_path
        upper_boundary_path = wall_path
        aisle_side = "outer_wall_positive_y"

    world_y_offsets = [
        sine * local_x + cosine * local_y
        for local_x in (
            _RBY1_CHASSIS_MIN_LOCAL_M[0],
            _RBY1_CHASSIS_MAX_LOCAL_M[0],
        )
        for local_y in (
            _RBY1_CHASSIS_MIN_LOCAL_M[1],
            _RBY1_CHASSIS_MAX_LOCAL_M[1],
        )
    ]
    minimum_y = (
        lower_boundary_y
        + _MINIMUM_BASE_STRUCTURE_CLEARANCE_M
        - min(world_y_offsets)
    )
    maximum_y = (
        upper_boundary_y
        - _MINIMUM_BASE_STRUCTURE_CLEARANCE_M
        - max(world_y_offsets)
    )
    if minimum_y > maximum_y:
        raise RuntimeError(
            f"RBY1 chassis does not fit {aisle_side}: "
            f"{minimum_y:.3f}>{maximum_y:.3f}"
        )
    target_edge_margin_m = min(0.085, 0.25 * (maximum_y - minimum_y))
    selected_y = (
        maximum_y - target_edge_margin_m
        if sine > 0.0
        else minimum_y + target_edge_margin_m
    )
    return {
        "aisle_side": aisle_side,
        "minimum_base_y_m": minimum_y,
        "maximum_base_y_m": maximum_y,
        "selected_base_y_m": selected_y,
        "lower_boundary_y_m": lower_boundary_y,
        "upper_boundary_y_m": upper_boundary_y,
        "lower_boundary_path": lower_boundary_path,
        "upper_boundary_path": upper_boundary_path,
        "bed_path": bed_path,
        "target_forward_distance_y_m": sine * (target[1] - selected_y),
    }


def _default_robot_position(
    stage, placement, yaw_degrees, Usd, UsdGeom, np
) -> tuple[float, float, float]:
    """Place the base beside the selected gutter on its local cultivation floor."""
    target = np.asarray(placement, dtype=np.float64)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    floor_candidates: list[float] = []
    for prim in stage.Traverse():
        if prim.GetName() != "GroundPlane":
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if extent.IsEmpty():
            continue
        lower = np.asarray(extent.GetMin(), dtype=np.float64)
        upper = np.asarray(extent.GetMax(), dtype=np.float64)
        contains_target = bool(
            np.all(target[:2] >= lower[:2] - 1e-6)
            and np.all(target[:2] <= upper[:2] + 1e-6)
        )
        if contains_target and upper[2] <= target[2] + 1e-6:
            floor_candidates.append(float(upper[2]))
    if not floor_candidates:
        raise RuntimeError(
            f"no ground plane below selected vine placement {tuple(float(v) for v in target)}"
        )
    aisle = _approach_aisle_base_bounds(
        stage, placement, yaw_degrees, Usd, UsdGeom, np
    )
    return (
        float(target[0] + 0.12),
        aisle["selected_base_y_m"],
        max(floor_candidates),
    )


def _placement_transforms(matrix, vine_usd, Gf, np):
    """Position and direction transforms from GLB coordinates to greenhouse world."""

    def points(values):
        converted = vine_usd.gltf_to_usd(values)
        return np.asarray([matrix.Transform(Gf.Vec3d(*row)) for row in converted], dtype=np.float64)

    def directions(values):
        converted = vine_usd.gltf_to_usd(values)
        return np.asarray(
            [matrix.TransformDir(Gf.Vec3d(*row)) for row in converted], dtype=np.float64
        )

    return points, directions


def _author_camera(stage, placement, args, Gf, Sdf, UsdGeom) -> str:
    target = (
        Gf.Vec3d(*args.camera_target)
        if args.camera_target is not None
        else Gf.Vec3d(float(placement[0]), float(placement[1]), float(placement[2]) + 0.8)
    )
    eye = Gf.Vec3d(*args.eye) if args.eye is not None else target + Gf.Vec3d(-1.6, 2.4, 0.8)
    path = Sdf.Path("/World/InteractiveInspectionCamera")
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    xform.AddTransformOp().Set(view.GetInverse())
    return str(path)


def _disable_native_mouse_interaction(app) -> bool:
    """Give the benchmark's visible-mesh pull exclusive ownership of selection."""
    import carb.settings
    import omni.physx.bindings._physx as physx_bindings

    settings = carb.settings.get_settings()
    settings.set_bool(physx_bindings.SETTING_MOUSE_GRAB, False)
    settings.set_bool(physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED, False)

    # Isaac Sim 5.1's mixed USD/Fabric transform selector can forward a handled
    # selection as None to the next manipulator.  prim.core then indexes that
    # None as a path type and raises KeyError(NoneType).  This environment uses
    # its own renderer raycast and never needs the native transform gizmo, so
    # clear stale selection and unsubscribe only that selector's stage listener.
    selector_disabled = False
    try:
        from omni.kit.manipulator.selector import get_manipulator_selector
        import omni.usd

        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
        selector = get_manipulator_selector("")
        if selector is not None:
            selector.destroy()
            selector_disabled = True
    except Exception as exc:
        print(f"could not disable native transform selector: {exc}")
    for _ in range(2):
        app.update()
    return selector_disabled


def _focus_viewport(camera_path: str) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(camera_path)
    except Exception as exc:
        print(f"could not focus inspection camera: {exc}")


class RobotContactDiagnostics:
    """Opt-in contact trace for locating unsafe task-pose intersections."""

    def __init__(self, stage):
        from pxr import PhysxSchema
        from pxr import UsdPhysics

        self._pairs: dict[tuple[str, str], dict] = {}
        self._active_pairs: set[tuple[str, str]] = set()
        self._subscription = None
        self._reported_bodies = 0
        self._phase = "settling"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/World/RBY1/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
                self._reported_bodies += 1

    def subscribe(self) -> None:
        from omni.physx import get_physx_simulation_interface

        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contacts
        )

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)

    @staticmethod
    def _vector(value) -> list[float]:
        return [float(value.x), float(value.y), float(value.z)]

    def _on_contacts(self, headers, data) -> None:
        import numpy as np
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        for header in headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            pair_key = tuple(sorted((collider0, collider1)))
            pair = self._pairs.setdefault(
                pair_key,
                {
                    "collider0": pair_key[0],
                    "collider1": pair_key[1],
                    "events": 0,
                    "contacts": 0,
                    "minimum_separation_mm": float("inf"),
                    "maximum_impulse_ns": 0.0,
                    "maximum_impulse_vector_ns": [0.0, 0.0, 0.0],
                    "maximum_impulse_position_m": None,
                    "phases": [],
                    "active": False,
                },
            )
            pair["events"] += 1
            if header.type == ContactEventType.CONTACT_LOST:
                self._active_pairs.discard(pair_key)
                pair["active"] = False
                continue

            self._active_pairs.add(pair_key)
            pair["active"] = True
            if self._phase not in pair["phases"]:
                pair["phases"].append(self._phase)
            start = header.contact_data_offset
            stop = start + header.num_contact_data
            pair["contacts"] += header.num_contact_data
            for contact in data[start:stop]:
                impulse = np.asarray(self._vector(contact.impulse), dtype=np.float64)
                magnitude = float(np.linalg.norm(impulse))
                pair["minimum_separation_mm"] = min(
                    pair["minimum_separation_mm"],
                    float(contact.separation) * 1000.0,
                )
                if magnitude > pair["maximum_impulse_ns"]:
                    pair["maximum_impulse_ns"] = magnitude
                    pair["maximum_impulse_vector_ns"] = impulse.tolist()
                    pair["maximum_impulse_position_m"] = self._vector(
                        contact.position
                    )
    @property
    def summary(self) -> dict:
        pairs = sorted(
            self._pairs.values(),
            key=lambda item: item["maximum_impulse_ns"],
            reverse=True,
        )
        return {"reported_bodies": self._reported_bodies, "pairs": pairs}

    @property
    def active_summary(self) -> dict:
        """Return only contacts that have not emitted CONTACT_LOST."""
        pairs = sorted(
            (
                self._pairs[key]
                for key in self._active_pairs
                if key in self._pairs
            ),
            key=lambda item: item["maximum_impulse_ns"],
            reverse=True,
        )
        return {"reported_bodies": self._reported_bodies, "pairs": pairs}
    def close(self) -> None:
        self._subscription = None


class BladeContactMonitor:
    """Translate real knife contacts into cut work and safety evidence."""

    _BLOCKING_SAFETY_CATEGORIES = frozenset(
        {
            "neighbouring_vine_contact",
            "main_stem_contact",
            "non_target_organ_contact",
            "protected_structure_contact",
            "robot_self_contact",
        }
    )

    def __init__(self, stage, runtimes, robot_placement, parameters) -> None:
        from greenhouse_sim import cutting as cutting_module
        from greenhouse_sim import robot_hardware
        import numpy as np
        from pxr import Gf
        from pxr import PhysxSchema
        from pxr import Usd
        from pxr import UsdGeom

        self._cutting = cutting_module
        self._np = np
        self._cut_direction_local = robot_hardware.KNIFE_CUT_DIRECTION_LOCAL.copy()
        self._edge_axis_local = robot_hardware.KNIFE_EDGE_AXIS_LOCAL.copy()
        self._stage = stage
        self._runtimes = {runtime.name: runtime for runtime in runtimes}
        self._gate = cutting_module.DirectionalCutGate(parameters)
        self._edge_path = robot_placement.knife_cutting_edge
        knife_root = self._edge_path.rsplit("/", 1)[0]
        self._blade_collider = f"{knife_root}/BladeCollision"
        self._arc_collider = f"{knife_root}/ArcCollision"
        self._targets = {}
        self._target_frames = {}
        self._target_colliders = {}
        self._collider_target_frames = {}
        self._collider_info = {}
        self._pending: list[dict] = []
        self._subscription = None
        self._previous_edge_centre = None
        self._commanded_edge_velocity = np.zeros(3, dtype=np.float64)
        self._active_target: str | None = None
        self._counterhold_provider = lambda _key: False
        self._violations: dict[tuple[str, str, str], dict] = {}
        self._physical_cuts: list[dict] = []

        for runtime in runtimes:
            for info in runtime.rig.colliders:
                self._collider_info[info.path] = {
                    "vine": runtime.name,
                    "label": info.organ_label,
                    "role": info.role,
                }
                if "petiole_cut_zone" not in info.role:
                    continue
                junction = runtime.rig.junctions.get(info.organ_label)
                if junction is None:
                    continue
                key = f"{runtime.name}/{info.organ_label}"
                self._target_colliders[info.path] = key
                if key not in self._targets:
                    self._targets[key] = (
                        runtime,
                        cutting_module.CutTarget(
                            key=key,
                            organ_label=info.organ_label,
                            centre_m=junction.cut_position_m,
                            axis=junction.cut_axis,
                            radius_m=junction.cut_radius_m,
                            cut_force_n=junction.cut_force_n,
                        ),
                    )
                    body_matrix = UsdGeom.Xformable(
                        stage.GetPrimAtPath(junction.child_path)
                    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    self._target_frames[key] = {
                        "body": junction.child_path,
                        "centre_local": body_matrix.GetInverse().Transform(
                            Gf.Vec3d(*junction.cut_position_m.tolist())
                        ),
                        "axis_local": body_matrix.GetInverse().TransformDir(
                            Gf.Vec3d(*junction.cut_axis.tolist())
                        ),
                    }

                # Planning uses the live proximal junction frame above.  Cut
                # scoring follows the exact articulated capsule that produced
                # contact, with projection still expressed as cumulative stub
                # distance from the junction.  One shared frame for all links
                # cannot represent a bent petiole; overwriting it per link is
                # equally wrong because the final collider wins globally.
                virtual_centre, segment_axis, arc_start_m = (
                    runtime.rig.cut_segment_frame(info)
                )
                body_matrix = UsdGeom.Xformable(
                    stage.GetPrimAtPath(info.body_path)
                ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                self._collider_target_frames[info.path] = {
                    "body": info.body_path,
                    "centre_local": body_matrix.GetInverse().Transform(
                        Gf.Vec3d(*virtual_centre.tolist())
                    ),
                    "axis_local": body_matrix.GetInverse().TransformDir(
                        Gf.Vec3d(*segment_axis.tolist())
                    ),
                    "segment": info.segment,
                    "arc_start_m": arc_start_m,
                }

        ee_right = stage.GetPrimAtPath(f"{robot_placement.root_path}/ee_right")
        if not ee_right.IsValid():
            raise ValueError("right end-effector body is missing for blade contact reporting")
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(ee_right)
        report_api.CreateThresholdAttr().Set(0.0)

    def add_safety_colliders(self, entries) -> None:
        for entry in entries:
            self._collider_info[entry.path] = {
                "vine": entry.vine_name,
                "label": entry.organ_label,
                "role": entry.role,
            }

    def set_active_target(self, vine_name: str, organ_label: str) -> None:
        self._active_target = f"{vine_name}/{organ_label}"

    def set_counterhold_provider(self, provider) -> None:
        """Supply verified physical-grasp state for rigid-tissue fracture."""
        self._counterhold_provider = provider

    def set_commanded_edge_velocity(self, velocity_m_s) -> None:
        """Record policy-commanded edge motion for traction separation."""
        velocity = self._np.asarray(velocity_m_s, dtype=self._np.float64)
        if velocity.shape != (3,) or not self._np.isfinite(velocity).all():
            raise ValueError("commanded edge velocity must be a finite three-vector")
        self._commanded_edge_velocity = velocity.copy()

    @property
    def edge_centre_m(self):
        """Return the live cutting-edge centre, including torso motion."""
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom

        matrix = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(self._edge_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return self._np.asarray(
            matrix.Transform(Gf.Vec3d(0.0)), dtype=self._np.float64
        )

    def _current_target(self, key: str, collider_path: str | None = None):
        from pxr import Usd
        from pxr import UsdGeom

        runtime, authored = self._targets[key]
        frame = (
            self._collider_target_frames[collider_path]
            if collider_path is not None
            else self._target_frames[key]
        )
        matrix = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(frame["body"])
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centre = self._np.asarray(
            matrix.Transform(frame["centre_local"]), dtype=self._np.float64
        )
        axis = self._np.asarray(
            matrix.TransformDir(frame["axis_local"]).GetNormalized(),
            dtype=self._np.float64,
        )
        return runtime, self._cutting.CutTarget(
            key=authored.key,
            organ_label=authored.organ_label,
            centre_m=centre,
            axis=axis,
            radius_m=authored.radius_m,
            cut_force_n=authored.cut_force_n,
        )

    @property
    def target_geometry(self) -> dict | None:
        key = self._active_target or ""
        if key not in self._targets:
            return None
        _, target = self._current_target(key)
        colliders = [
            path
            for path, key in self._target_colliders.items()
            if key == self._active_target
        ]
        return {
            "key": target.key,
            "centre_m": target.centre_m.copy(),
            "axis": target.axis.copy(),
            "radius_m": target.radius_m,
            "colliders": tuple(colliders),
        }

    @property
    def active_cut_feedback(self) -> dict | None:
        """Return the latest physical contact load for closed-loop motion."""
        key = self._active_target or ""
        if key not in self._targets:
            return None
        _, target = self._current_target(key)
        progress = self._gate.progress_for(key)
        return {
            "target": key,
            "required_force_n": target.cut_force_n,
            "effective_force_n": progress.last_effective_force_n,
            "forward_speed_m_s": progress.last_forward_speed_m_s,
            "contact_valid": progress.last_contact_valid,
            "work_j": progress.work_j,
            "forward_travel_m": progress.forward_travel_m,
            "gap_steps": progress.gap_steps,
        }

    def tool_point_geometry(self, local_point_m) -> dict:
        """Return a knife-local point and tool axes in the live world frame."""
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom

        knife_root = self._edge_path.rsplit("/", 1)[0]
        matrix = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(knife_root)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        point = self._np.asarray(
            matrix.Transform(Gf.Vec3d(*local_point_m)), dtype=self._np.float64
        )
        edge_axis = self._np.asarray(
            matrix.TransformDir(
                Gf.Vec3d(*self._edge_axis_local.tolist())
            ).GetNormalized(),
            dtype=self._np.float64,
        )
        cut_direction = self._np.asarray(
            matrix.TransformDir(
                Gf.Vec3d(*self._cut_direction_local.tolist())
            ).GetNormalized(),
            dtype=self._np.float64,
        )
        return {
            "point_m": point,
            "edge_axis": edge_axis,
            "cut_direction": cut_direction,
        }

    def target_path_geometry(self, stub_m: float) -> dict | None:
        """Return the live articulated centreline point/tangent at a stub."""
        key = self._active_target or ""
        if key not in self._targets:
            return None
        candidates = [
            (path, frame)
            for path, frame in self._collider_target_frames.items()
            if self._target_colliders.get(path) == key
        ]
        if not candidates:
            return None
        before = [
            item
            for item in candidates
            if float(item[1]["arc_start_m"]) <= float(stub_m) + 1e-9
        ]
        collider_path, frame = max(
            before or candidates,
            key=lambda item: float(item[1]["arc_start_m"]),
        )
        _, target = self._current_target(key, collider_path)
        point = target.centre_m + target.axis * float(stub_m)
        return {
            "key": key,
            "point_m": point,
            "axis": target.axis.copy(),
            "radius_m": target.radius_m,
            "collider": collider_path,
            "segment": int(frame["segment"]),
            "arc_start_m": float(frame["arc_start_m"]),
            "virtual_junction_m": target.centre_m.copy(),
        }

    def subscribe(self) -> None:
        from omni.physx import get_physx_simulation_interface

        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contacts
        )

    def _vector(self, value):
        return self._np.asarray(
            [float(value.x), float(value.y), float(value.z)],
            dtype=self._np.float64,
        )

    def _inside_cutting_edge(self, point_m) -> bool:
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom

        prim = self._stage.GetPrimAtPath(self._edge_path)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local = self._np.asarray(
            matrix.GetInverse().Transform(Gf.Vec3d(*point_m.tolist())),
            dtype=self._np.float64,
        )
        return bool(self._np.all(self._np.abs(local) <= 0.60))

    def _on_contacts(self, headers, data) -> None:
        from pxr import PhysicsSchemaTools

        for header in headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if self._blade_collider in (collider0, collider1):
                tool = "blade"
                other = collider1 if collider0 == self._blade_collider else collider0
            elif self._arc_collider in (collider0, collider1):
                tool = "arc"
                other = collider1 if collider0 == self._arc_collider else collider0
            else:
                continue

            start = header.contact_data_offset
            stop = start + header.num_contact_data
            for contact in data[start:stop]:
                point = self._vector(contact.position)
                impulse = self._vector(contact.impulse)
                # PhysX also emits contact-offset proximity points with
                # positive separation and exactly zero impulse. They remain in
                # RobotContactDiagnostics, but are not physical tool contact.
                if float(self._np.linalg.norm(impulse)) <= 1e-12:
                    continue
                self._pending.append(
                    {
                        "tool": tool,
                        "other": other,
                        "point": point,
                        "impulse": impulse,
                        "inside_edge": (
                            tool == "blade" and self._inside_cutting_edge(point)
                        ),
                    }
                )

    def _edge_kinematics(self, dt_s: float) -> tuple[np.ndarray, ...]:
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom

        matrix = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(self._edge_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centre = self._np.asarray(
            matrix.Transform(Gf.Vec3d(0.0)), dtype=self._np.float64
        )
        edge_axis = self._np.asarray(
            matrix.TransformDir(
                Gf.Vec3d(*self._edge_axis_local.tolist())
            ).GetNormalized(),
            dtype=self._np.float64,
        )
        cut_direction = self._np.asarray(
            matrix.TransformDir(
                Gf.Vec3d(*self._cut_direction_local.tolist())
            ).GetNormalized(),
            dtype=self._np.float64,
        )
        velocity = (
            self._np.zeros(3, dtype=self._np.float64)
            if self._previous_edge_centre is None
            else (centre - self._previous_edge_centre) / dt_s
        )
        self._previous_edge_centre = centre.copy()
        return centre, edge_axis, cut_direction, velocity

    def _record_violation(self, event: dict, dt_s: float) -> None:
        info = self._collider_info.get(event["other"])
        target_key = self._target_colliders.get(event["other"])
        if info is None:
            if event["other"].startswith("/World/RBY1/"):
                category = "robot_self_contact"
            else:
                category = "protected_structure_contact"
            vine = ""
            label = event["other"]
        else:
            vine = info["vine"]
            label = info["label"]
            if vine != (self._active_target or "").partition("/")[0]:
                category = "neighbouring_vine_contact"
            elif info["role"] == "protected_main_stem":
                category = "main_stem_contact"
            elif target_key != self._active_target:
                category = "non_target_organ_contact"
            elif event["tool"] == "arc":
                category = "support_arc_target_contact"
            else:
                category = "blade_face_target_contact"

        impulse = float(self._np.linalg.norm(event["impulse"]))
        force = impulse / max(dt_s, 1e-12)
        key = (category, event["tool"], event["other"])
        record = self._violations.setdefault(
            key,
            {
                "category": category,
                "tool": event["tool"],
                "collider": event["other"],
                "vine": vine,
                "organ": label,
                "events": 0,
                "maximum_impulse_ns": 0.0,
                "maximum_force_n": 0.0,
            },
        )
        record["events"] += 1
        record["maximum_impulse_ns"] = max(record["maximum_impulse_ns"], impulse)
        record["maximum_force_n"] = max(record["maximum_force_n"], force)

    def process(self, dt_s: float = 1.0 / 240.0) -> list[dict]:
        centre, edge_axis, cut_direction, velocity = self._edge_kinematics(dt_s)
        aggregates: dict[str, dict] = {}
        for event in self._pending:
            target_key = self._target_colliders.get(event["other"])
            if event["tool"] == "blade" and event["inside_edge"] and target_key is not None:
                aggregate = aggregates.setdefault(
                    target_key,
                    {
                        "point_sum": self._np.zeros(3),
                        "weight": 0.0,
                        "impulse": self._np.zeros(3),
                        "collider_weights": {},
                    },
                )
                weight = max(float(self._np.linalg.norm(event["impulse"])), 1e-12)
                aggregate["point_sum"] += weight * event["point"]
                aggregate["weight"] += weight
                aggregate["impulse"] += event["impulse"]
                aggregate["collider_weights"][event["other"]] = (
                    aggregate["collider_weights"].get(event["other"], 0.0)
                    + weight
                )
            else:
                self._record_violation(event, dt_s)
        self._pending.clear()

        decisions = []
        contacted = set(aggregates)
        for key, aggregate in aggregates.items():
            dominant_collider = max(
                aggregate["collider_weights"],
                key=aggregate["collider_weights"].get,
            )
            runtime, target = self._current_target(key, dominant_collider)
            sample = self._cutting.BladeContactSample(
                point_m=aggregate["point_sum"] / aggregate["weight"],
                impulse_ns=aggregate["impulse"],
                edge_centre_m=centre,
                edge_axis=edge_axis,
                cutting_direction=cut_direction,
                edge_velocity_m_s=velocity,
                dt_s=dt_s,
                counterhold_active=bool(self._counterhold_provider(key)),
                commanded_edge_velocity_m_s=(
                    self._commanded_edge_velocity.copy()
                ),
            )
            decision = self._gate.observe(target, sample)
            if decision is not None:
                decisions.append(
                    {
                        "runtime": runtime,
                        "decision": decision,
                        "intended_target": key == self._active_target,
                    }
                )
        self._gate.finish_step(contacted)
        return decisions

    def record_cut(self, event: dict) -> None:
        self._physical_cuts.append(event)

    @property
    def blocking_safety_violations(self) -> list[dict]:
        return [
            dict(record)
            for record in self._violations.values()
            if record["category"] in self._BLOCKING_SAFETY_CATEGORIES
        ]

    @property
    def safety_clear(self) -> bool:
        return not self.blocking_safety_violations

    @property
    def summary(self) -> dict:
        active_target = (
            self._current_target(self._active_target)[1]
            if self._active_target in self._targets
            else None
        )
        return {
            "model": (
                "directional leading-edge force/work gate with "
                "physical-counterhold rigid-tissue fracture"
            ),
            "cutting_edge": self._edge_path,
            "physical_blade_collider": self._blade_collider,
            "active_target": self._active_target,
            "active_target_geometry": (
                {
                    "centre_m": active_target.centre_m.tolist(),
                    "axis": active_target.axis.tolist(),
                    "radius_m": active_target.radius_m,
                    "cut_force_n": active_target.cut_force_n,
                }
                if active_target is not None
                else None
            ),
            "targets": len(self._targets),
            "parameters": dataclasses.asdict(self._gate.parameters),
            "progress": self._gate.summary(),
            "physical_cuts": list(self._physical_cuts),
            "safety_clear": self.safety_clear,
            "blocking_safety_violations": sorted(
                self.blocking_safety_violations,
                key=lambda item: item["maximum_impulse_ns"],
                reverse=True,
            ),
            "safety_violations": sorted(
                self._violations.values(),
                key=lambda item: item["maximum_impulse_ns"],
                reverse=True,
            ),
        }

    def close(self) -> None:
        self._subscription = None


class LeftGraspManager:
    """Drive the retained left tongs and hold an orphan branch after cutting."""

    def __init__(
        self,
        stage,
        runtimes,
        robot_placement,
        task_parameters,
        *,
        open_width_m: float = 0.025,
        close_overtravel_m: float = 0.006,
    ) -> None:
        from greenhouse_sim import deleaf_task as task_module
        import numpy as np
        from pxr import Gf
        from pxr import PhysxSchema
        from pxr import Sdf
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics

        self._task_module = task_module
        self._np = np
        self._Gf = Gf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._stage = stage
        self._runtimes = {runtime.name: runtime for runtime in runtimes}
        self._root_path = robot_placement.root_path
        self._task_parameters = task_parameters
        self._open_width_m = float(open_width_m)
        self._close_overtravel_m = float(close_overtravel_m)
        self._close_requested = False
        self._requested_openness = 1.0
        self._close_request_threshold = 0.95
        self._release_threshold = 0.75
        self._pending: list[dict] = []
        self._subscription = None
        self._active_target: str | None = None
        self._task = None
        self._grasp_colliders: dict[str, dict] = {}
        self._grasp_colliders_for_key: dict[str, list[str]] = {}
        self._grasp_collider_for_key: dict[str, str] = {}
        self._grasp_bodies: dict[str, str] = {}
        self._joint_paths: dict[str, str] = {}
        self._orphan_paths: dict[str, list[str]] = {}
        self._active_joint_key: str | None = None
        self._active_grasp_body: str | None = None
        self._active_grasp_collider: str | None = None
        self._active_grasp_point_local = None
        self._cut_grasp_position = None
        self._previous_orphan_centroid = None
        self._latest_orphan_state = None
        self._latest_finger_contact: dict | None = None
        self._finger_contact_serial = 0

        self._finger_colliders = {
            f"{self._root_path}/ee_finger_l1/restored_collisions/contact_proxy":
                "left_finger_1",
            f"{self._root_path}/ee_finger_l2/restored_collisions/contact_proxy":
                "left_finger_2",
        }
        for link in ("ee_finger_l1", "ee_finger_l2"):
            prim = stage.GetPrimAtPath(f"{self._root_path}/{link}")
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)

        grasp_scope = Sdf.Path(
            f"{self._root_path}/ee_left/BenchmarkGrasps"
        )
        UsdGeom.Scope.Define(stage, grasp_scope)
        for runtime in runtimes:
            for info in runtime.rig.colliders:
                key = f"{runtime.name}/{info.organ_label}"
                if (
                    "grasp" not in info.role
                    and "petiole_cut_zone" not in info.role
                ):
                    continue
                self._grasp_colliders[info.path] = {
                    "key": key,
                    "vine": runtime.name,
                    "organ": info.organ_label,
                    "body": info.body_path,
                    "collider": info.path,
                    "segment": int(info.segment),
                    "role": info.role,
                }
                self._grasp_colliders_for_key.setdefault(key, []).append(
                    info.path
                )
                if (
                    key not in self._grasp_collider_for_key
                    or "petiole_grasp" in info.role
                ):
                    self._grasp_collider_for_key[key] = info.path
                    self._grasp_bodies[key] = info.body_path
            for label in runtime.rig.junctions:
                key = f"{runtime.name}/{label}"
                body = self._grasp_bodies.get(key)
                if body is None:
                    continue
                safe_name = f"{runtime.name}_{label}".replace("-", "_")
                joint_path = str(grasp_scope.AppendChild(safe_name))
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                body0_path = f"{self._root_path}/ee_left"
                joint.CreateBody0Rel().SetTargets([body0_path])
                joint.CreateBody1Rel().SetTargets([body])
                body0_matrix = UsdGeom.Xformable(
                    stage.GetPrimAtPath(body0_path)
                ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                body1_matrix = UsdGeom.Xformable(
                    stage.GetPrimAtPath(body)
                ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                anchor = body1_matrix.ExtractTranslation()
                joint.CreateLocalPos0Attr().Set(
                    Gf.Vec3f(body0_matrix.GetInverse().Transform(anchor))
                )
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
                relative = (
                    body0_matrix.ExtractRotationQuat().GetInverse()
                    * body1_matrix.ExtractRotationQuat()
                )
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(relative))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
                joint.CreateExcludeFromArticulationAttr(defaultValue=True)
                joint.CreateJointEnabledAttr(defaultValue=False)
                self._joint_paths[key] = joint_path

                organ_index = runtime.organ_indices[label]
                descendants = set(runtime.plant.descendants_of(organ_index))
                self._orphan_paths[key] = [
                    link.path
                    for link in runtime.rig.links
                    if link.organ in descendants
                ]

        self._finger_drives = (
            (
                UsdPhysics.DriveAPI.Get(
                    stage.GetPrimAtPath(
                        f"{self._root_path}/joints/gripper_finger_l1"
                    ),
                    "linear",
                ),
                -self._open_width_m,
            ),
            (
                UsdPhysics.DriveAPI.Get(
                    stage.GetPrimAtPath(
                        f"{self._root_path}/joints/gripper_finger_l2"
                    ),
                    "linear",
                ),
                self._open_width_m,
            ),
        )
        if not all(drive for drive, _ in self._finger_drives):
            raise ValueError("left gripper linear drives are missing")
        self._finger_drive_configuration = {
            "type": "force",
            "stiffness_n_m": 800.0,
            "damping_n_s_m": 10.0,
            "maximum_force_n": 40.0,
        }
        for drive, _ in self._finger_drives:
            drive.CreateTypeAttr(self._finger_drive_configuration["type"])
            drive.CreateStiffnessAttr(
                self._finger_drive_configuration["stiffness_n_m"]
            )
            drive.CreateDampingAttr(
                self._finger_drive_configuration["damping_n_s_m"]
            )
            drive.CreateMaxForceAttr(
                self._finger_drive_configuration["maximum_force_n"]
            )
        self._set_finger_targets(opened=True)

    def _set_finger_openness(self, openness: float) -> None:
        for drive, open_target in self._finger_drives:
            closed_target = (
                -self._np.sign(open_target) * self._close_overtravel_m
            )
            target = closed_target + openness * (open_target - closed_target)
            drive.CreateTargetPositionAttr(float(target))
            drive.CreateTargetVelocityAttr(0.0)

    def _set_finger_targets(self, *, opened: bool) -> None:
        self._set_finger_openness(1.0 if opened else 0.0)

    def set_active_target(self, vine_name: str, organ_label: str) -> None:
        key = f"{vine_name}/{organ_label}"
        if key not in self._joint_paths:
            return
        if self._active_target == key:
            return
        if self._task is not None and self._task.phase is not self._task_module.Phase.SEEK_GRASP:
            return
        self._active_target = key
        self._task = self._task_module.BimanualDeleafTask(
            vine_name,
            organ_label,
            self._task_parameters,
        )

    @property
    def target_candidates(self) -> list[dict]:
        """Return live physical colliders eligible for the selected grasp."""
        from pxr import Gf

        key = self._active_target or ""
        candidates = []
        for path in self._grasp_colliders_for_key.get(key, ()):
            info = self._grasp_colliders[path]
            # Foliage proxies are valid physical pinch contacts, but automated
            # IK remains benchmarked against the repeatable petiole corridor.
            if info["role"] == "foliage_grasp":
                continue
            body_matrix = self._body_matrix(info["body"])
            collider_matrix = self._body_matrix(path)
            candidates.append(
                {
                    **info,
                    "centre_m": self._np.asarray(
                        collider_matrix.ExtractTranslation(), dtype=self._np.float64
                    ),
                    "axis": self._np.asarray(
                        body_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized(),
                        dtype=self._np.float64,
                    ),
                    "preferred": path == self._grasp_collider_for_key.get(key),
                }
            )
        return candidates

    def set_planned_grasp_collider(self, collider_path: str) -> None:
        """Plan against one eligible collider; opposed contact chooses the final body."""
        info = self._grasp_colliders.get(collider_path)
        if info is None or info["key"] != self._active_target:
            raise ValueError(f"collider is not eligible for active target: {collider_path}")
        key = info["key"]
        self._grasp_collider_for_key[key] = collider_path
        self._grasp_bodies[key] = info["body"]

    @property
    def target_geometry(self) -> dict | None:
        from pxr import Gf

        key = self._active_target or ""
        path = (
            self._active_grasp_collider
            or self._grasp_collider_for_key.get(key)
        )
        if path is None:
            return None
        body = self._active_grasp_body or self._grasp_bodies[key]
        matrix = self._body_matrix(body)
        centre = (
            self._np.asarray(
                matrix.Transform(self._active_grasp_point_local),
                dtype=self._np.float64,
            )
            if self._active_grasp_point_local is not None
            else self._np.asarray(
                self._body_matrix(path).ExtractTranslation(),
                dtype=self._np.float64,
            )
        )
        axis = self._np.asarray(
            matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized(),
            dtype=self._np.float64,
        )
        tool_local_point = None
        if self._active_grasp_point_local is not None:
            ee_inverse = self._body_matrix(
                f"{self._root_path}/ee_left"
            ).GetInverse()
            tool_local_point = self._np.asarray(
                ee_inverse.Transform(Gf.Vec3d(*centre.tolist())),
                dtype=self._np.float64,
            )
        physics_organ = path.partition("/Link_")[0]
        organ_prim = self._stage.GetPrimAtPath(physics_organ)
        organ_colliders = tuple(
            str(prim.GetPath())
            for prim in self._Usd.PrimRange(organ_prim)
            if prim.GetName() == "Collider"
        )
        orphan_colliders = tuple(
            str(prim.GetPath())
            for link_path in self._orphan_paths.get(key, ())
            for prim in self._Usd.PrimRange(
                self._stage.GetPrimAtPath(link_path)
            )
            if prim.GetName() == "Collider"
        )
        branch_contact_colliders = tuple(
            self._grasp_colliders_for_key.get(key, ())
        )
        return {
            "key": key,
            "collider": path,
            "colliders": tuple(
                organ_colliders or (path,)
            ),
            # The selected petiole and all of its descendants form the branch
            # that the left fingers are expected to handle and remove. Keep
            # this separate from ``colliders``: the wrist camera and bracket
            # must still clear the branch, while finger contact with it is an
            # intended task interaction rather than a neighbouring-organ hit.
            "orphan_colliders": tuple(
                dict.fromkeys(
                    orphan_colliders
                    + organ_colliders
                    + branch_contact_colliders
                    + (path,)
                )
            ),
            "body": body,
            "centre_m": centre,
            "axis": axis,
            "tool_local_point_m": tool_local_point,
        }

    @property
    def task_succeeded(self) -> bool:
        return bool(self._task is not None and self._task.succeeded)

    @property
    def task_phase(self) -> str | None:
        return self._task.phase.value if self._task is not None else None

    def holds_target(self, key: str) -> bool:
        """Report a live, opposed-finger physical hold on exactly ``key``."""
        return bool(
            key == self._active_target
            and self._active_joint_key == key
            and self._task is not None
            and self._task.grasp_active
        )

    @property
    def latest_finger_contact(self) -> dict | None:
        """Return the latest graspable branch touched by either left finger."""
        return (
            None
            if self._latest_finger_contact is None
            else dict(self._latest_finger_contact)
        )

    @property
    def close_requested(self) -> bool:
        """Return whether the measured physical aperture requests a pinch."""
        return self._close_requested

    def subscribe(self) -> None:
        from omni.physx import get_physx_simulation_interface

        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contacts
        )

    def _vector(self, value):
        return self._np.asarray(
            [float(value.x), float(value.y), float(value.z)],
            dtype=self._np.float64,
        )

    def _on_contacts(self, headers, data) -> None:
        from pxr import PhysicsSchemaTools

        for header in headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if collider0 in self._finger_colliders:
                finger = self._finger_colliders[collider0]
                other = collider1
            elif collider1 in self._finger_colliders:
                finger = self._finger_colliders[collider1]
                other = collider0
            else:
                continue
            target = self._grasp_colliders.get(other)
            if target is None:
                continue
            start = header.contact_data_offset
            stop = start + header.num_contact_data
            for contact in data[start:stop]:
                impulse = self._vector(contact.impulse)
                # CONTACT_LOST/contact-offset records can carry the same
                # collider pair with exactly zero load. They must not replace
                # the latest physical pinch or count toward grasp persistence.
                if float(self._np.linalg.norm(impulse)) <= 1e-12:
                    continue
                self._pending.append(
                    {
                        **target,
                        "finger": finger,
                        "point": self._vector(contact.position),
                        "impulse": impulse,
                    }
                )

    def _release_active_grasp(self) -> None:
        if self._active_joint_key is None or self._task is None:
            return
        self._task.observe_release()
        self._previous_orphan_centroid = None
        self._latest_orphan_state = None
        self._set_joint_enabled(self._active_joint_key, False)
        self._active_joint_key = None
        self._active_grasp_body = None
        self._active_grasp_collider = None
        self._active_grasp_point_local = None

    def request_openness(self, openness: float) -> None:
        """Track the measured physical aperture and preserve contact semantics."""
        openness = float(openness)
        if not self._np.isfinite(openness) or not 0.0 <= openness <= 1.0:
            raise ValueError("left gripper openness must be between 0 and 1")
        if (
            self._active_joint_key is not None
            and openness >= self._release_threshold
        ):
            self._release_active_grasp()
        self._requested_openness = openness
        self._close_requested = openness <= self._close_request_threshold
        self._set_finger_openness(openness)

    def request_close(self) -> None:
        self.request_openness(0.0)

    def request_open(self) -> None:
        self.request_openness(1.0)
    def _set_joint_enabled(self, key: str, enabled: bool) -> None:
        joint = self._UsdPhysics.Joint(
            self._stage.GetPrimAtPath(self._joint_paths[key])
        )
        joint.GetJointEnabledAttr().Set(bool(enabled))

    def _body_matrix(self, path: str):
        from pxr import Usd

        return self._UsdGeom.Xformable(
            self._stage.GetPrimAtPath(path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def _body_position(self, path: str):
        return self._np.asarray(
            self._body_matrix(path).ExtractTranslation(),
            dtype=self._np.float64,
        )

    def _gripper_geometry(self, target_position) -> dict:
        """Report live finger contact volumes in the left-EE frame."""
        ee = self._body_matrix(f"{self._root_path}/ee_left")
        ee_inverse = ee.GetInverse()
        target_local = (
            self._np.asarray(
                ee_inverse.Transform(self._Gf.Vec3d(*target_position)),
                dtype=self._np.float64,
            ).tolist()
            if target_position is not None
            else None
        )
        fingers = {}
        for name in ("ee_finger_l1", "ee_finger_l2"):
            body_path = f"{self._root_path}/{name}"
            collider_path = f"{body_path}/restored_collisions/contact_proxy"
            body_world = self._body_matrix(body_path)
            collider_world = self._body_matrix(collider_path)
            corners = []
            for x in (-0.5, 0.5):
                for y in (-0.5, 0.5):
                    for z in (-0.5, 0.5):
                        world = collider_world.Transform(self._Gf.Vec3d(x, y, z))
                        corners.append(
                            self._np.asarray(
                                ee_inverse.Transform(world), dtype=self._np.float64
                            )
                        )
            corners = self._np.asarray(corners)
            fingers[name] = {
                "body_origin_ee_m": self._np.asarray(
                    ee_inverse.Transform(body_world.ExtractTranslation()),
                    dtype=self._np.float64,
                ).tolist(),
                "collider_min_ee_m": self._np.min(corners, axis=0).tolist(),
                "collider_max_ee_m": self._np.max(corners, axis=0).tolist(),
            }
        return {"target_position_ee_m": target_local, "fingers": fingers}

    def _activate_joint(
        self,
        key: str,
        point_m,
        body1_path: str,
        collider_path: str,
    ) -> None:
        joint = self._UsdPhysics.FixedJoint.Get(
            self._stage,
            self._joint_paths[key],
        )
        body0_path = f"{self._root_path}/ee_left"
        joint.GetBody1Rel().SetTargets([body1_path])
        body0 = self._body_matrix(body0_path)
        body1 = self._body_matrix(body1_path)
        point = self._Gf.Vec3d(*point_m.tolist())
        joint.GetLocalPos0Attr().Set(
            self._Gf.Vec3f(body0.GetInverse().Transform(point))
        )
        local_point = body1.GetInverse().Transform(point)
        joint.GetLocalPos1Attr().Set(self._Gf.Vec3f(local_point))
        child_rotation = body1.ExtractRotationQuat()
        relative = body0.ExtractRotationQuat().GetInverse() * child_rotation
        joint.GetLocalRot0Attr().Set(self._Gf.Quatf(relative))
        joint.GetLocalRot1Attr().Set(self._Gf.Quatf(1.0))
        joint.GetJointEnabledAttr().Set(True)
        self._active_joint_key = key
        self._active_grasp_body = body1_path
        self._active_grasp_collider = collider_path
        self._active_grasp_point_local = self._Gf.Vec3d(local_point)

    def notify_cut(self, event: dict) -> bool:
        if self._task is None:
            return False
        accepted = self._task.observe_cut(
            vine=event["vine"],
            organ=event["organ_label"],
            physical_blade=event.get("trigger") == "physical_blade",
            intended_target=bool(event.get("intended_target")),
            safe_path=bool(event.get("safety_clear", False)),
        )
        if accepted and self._active_joint_key is not None:
            self._cut_grasp_position = self._body_position(
                self._active_grasp_body
                or self._grasp_bodies[self._active_joint_key]
            )
        return accepted

    def process(self, dt_s: float = 1.0 / 240.0) -> None:
        if self._task is None:
            self._pending.clear()
            return
        self._task.advance()
        grouped: dict[tuple[str, str], dict] = {}
        for event in self._pending:
            aggregate = grouped.setdefault(
                (event["key"], event["body"]),
                {
                    "key": event["key"],
                    "vine": event["vine"],
                    "organ": event["organ"],
                    "body": event["body"],
                    "collider": event["collider"],
                    "fingers": set(),
                    "impulse": 0.0,
                    "point_sum": self._np.zeros(3),
                    "weight": 0.0,
                },
            )
            magnitude = float(self._np.linalg.norm(event["impulse"]))
            aggregate["fingers"].add(event["finger"])
            aggregate["impulse"] += magnitude
            aggregate["point_sum"] += max(magnitude, 1e-12) * event["point"]
            aggregate["weight"] += max(magnitude, 1e-12)
        self._pending.clear()

        if grouped:
            contacted = max(
                grouped.values(),
                key=lambda aggregate: (
                    {"left_finger_1", "left_finger_2"}.issubset(
                        aggregate["fingers"]
                    ),
                    aggregate["impulse"],
                ),
            )
            self._finger_contact_serial += 1
            self._latest_finger_contact = {
                "serial": self._finger_contact_serial,
                "key": contacted["key"],
                "vine": contacted["vine"],
                "organ": contacted["organ"],
                "body": contacted["body"],
                "collider": contacted["collider"],
                "fingers": sorted(contacted["fingers"]),
                "force_n": contacted["impulse"] / max(dt_s, 1e-12),
            }

        active_candidates = [
            aggregate
            for (key, _), aggregate in grouped.items()
            if key == (self._active_target or "")
        ]
        active = (
            max(
                active_candidates,
                key=lambda aggregate: (
                    {"left_finger_1", "left_finger_2"}.issubset(
                        aggregate["fingers"]
                    ),
                    aggregate["impulse"],
                ),
            )
            if active_candidates
            else None
        )
        if (
            self._close_requested
            and self._active_joint_key is None
            and active is not None
        ):
            established = self._task.observe_grasp(
                vine=active["vine"],
                organ=active["organ"],
                body_path=active["body"],
                finger_contacts=active["fingers"],
                force_n=active["impulse"] / max(dt_s, 1e-12),
            )
            if established and self._active_joint_key is None:
                point = active["point_sum"] / active["weight"]
                self._activate_joint(
                    self._active_target,
                    point,
                    active["body"],
                    active["collider"],
                )
        elif self._close_requested and self._active_joint_key is None:
            # Required grasp steps are consecutive physical-contact steps;
            # proximity reports or a contact gap must restart the sequence.
            planned = self._grasp_colliders.get(
                self._grasp_collider_for_key.get(self._active_target or "", "")
            )
            if planned is not None:
                self._task.observe_grasp(
                    vine=planned["vine"],
                    organ=planned["organ"],
                    body_path=planned["body"],
                    finger_contacts=set(),
                    force_n=0.0,
                )

        if self._active_joint_key is not None:
            self._task.observe_hold(grasp_active=True)
            if (
                self._cut_grasp_position is not None
                and self._task.phase
                in {
                    self._task_module.Phase.ORPHAN_RETAINED,
                    self._task_module.Phase.TRANSPORTED,
                }
            ):
                current = self._body_position(
                    self._active_grasp_body
                    or self._grasp_bodies[self._active_joint_key]
                )
                self._task.observe_transport(
                    float(self._np.linalg.norm(current - self._cut_grasp_position))
                )
        elif self._task.phase in {
            self._task_module.Phase.ORPHAN_RETAINED,
            self._task_module.Phase.TRANSPORTED,
        }:
            self._task.observe_hold(grasp_active=False)

        if self._task.phase is self._task_module.Phase.RELEASED:
            paths = self._orphan_paths.get(self._active_target or "", [])
            if paths:
                positions = _positions(self._stage, paths)
                centroid = self._np.mean(positions, axis=0)
                speed = 0.0
                if self._previous_orphan_centroid is not None:
                    speed = float(
                        self._np.linalg.norm(
                            centroid - self._previous_orphan_centroid
                        )
                        / dt_s
                    )
                self._previous_orphan_centroid = centroid.copy()
                floor = self._task.parameters.drop_zone_min_m[2]
                lowest = float(self._np.min(positions[:, 2]))
                floor_contact = bool(
                    lowest
                    <= floor + self._task.parameters.floor_tolerance_m
                )
                self._latest_orphan_state = {
                    "centroid_m": centroid.tolist(),
                    "speed_m_s": speed,
                    "lowest_height_m": lowest,
                    "floor_contact": floor_contact,
                    "body_count": len(paths),
                }
                self._task.observe_deposit(
                    centroid_m=centroid,
                    lowest_height_m=lowest,
                    speed_m_s=speed,
                    floor_contact=floor_contact,
                )

    @property
    def summary(self) -> dict:
        target_body = self._active_grasp_body
        planned_body = self._grasp_bodies.get(self._active_target or "")
        target_position = (
            self._body_position(target_body).tolist()
            if target_body is not None
            else None
        )
        left_ee_position = self._body_position(
            f"{self._root_path}/ee_left"
        ).tolist()
        target_distance_mm = (
            float(
                self._np.linalg.norm(
                    self._np.asarray(left_ee_position) - target_position
                )
                * 1000.0
            )
            if target_position is not None
            else None
        )
        return {
            "model": "opposed-left-finger fixed-joint grasp",
            "active_target": self._active_target,
            "planned_grasp_body": planned_body,
            "active_grasp_body": target_body,
            "active_grasp_position_m": target_position,
            "left_ee_position_m": left_ee_position,
            "left_ee_distance_to_grasp_mm": target_distance_mm,
            "gripper_geometry": self._gripper_geometry(target_position),
            "close_requested": self._close_requested,
            "requested_openness": self._requested_openness,
            "latest_finger_contact": self.latest_finger_contact,
            "active_joint": (
                self._joint_paths.get(self._active_joint_key)
                if self._active_joint_key is not None
                else None
            ),
            "graspable_targets": len(self._joint_paths),
            "target_candidates": [
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key not in {"centre_m", "axis"}
                    },
                    "centre_m": candidate["centre_m"].tolist(),
                    "axis": candidate["axis"].tolist(),
                }
                for candidate in self.target_candidates
            ],
            "finger_drive_configuration": dict(self._finger_drive_configuration),
            "finger_close_overtravel_m": self._close_overtravel_m,
            "orphan_state": self._latest_orphan_state,
            "task": self._task.summary if self._task is not None else None,
        }

    def close(self) -> None:
        self._subscription = None


def _apply_blade_cut_decisions(
    context,
    monitor: BladeContactMonitor,
    report: dict,
    report_path: pathlib.Path | None = None,
    grasp_manager: LeftGraspManager | None = None,
) -> list[dict]:
    applied = []
    for event in monitor.process():
        runtime = event["runtime"]
        decision = event["decision"]
        context.pause()
        try:
            record = runtime.severer.cut(
                decision.organ_label,
                decision=decision,
            )
        except Exception as exc:
            report.setdefault("blade_cut_errors", []).append(
                {
                    "target": decision.target_key,
                    "error": str(exc),
                }
            )
        else:
            persisted = {
                **dataclasses.asdict(record),
                "vine": runtime.name,
                "intended_target": bool(event["intended_target"]),
                "safety_clear": monitor.safety_clear,
                "blocking_safety_violations": monitor.blocking_safety_violations,
            }
            bimanual_valid = (
                grasp_manager.notify_cut(persisted)
                if grasp_manager is not None
                else False
            )
            persisted["bimanual_sequence_valid"] = bimanual_valid
            persisted["benchmark_valid"] = bool(
                event["intended_target"]
                and persisted["safety_clear"]
                and bimanual_valid
            )
            report.setdefault("physical_blade_cuts", []).append(persisted)
            if not event["intended_target"]:
                report.setdefault("benchmark_failures", []).append(
                    {
                        "reason": "unintended_organ_cut",
                        "vine": runtime.name,
                        "organ": decision.organ_label,
                    }
                )
            if not persisted["safety_clear"]:
                report.setdefault("benchmark_failures", []).append(
                    {
                        "reason": "protected_contact_before_cut",
                        "vine": runtime.name,
                        "organ": decision.organ_label,
                        "violations": persisted[
                            "blocking_safety_violations"
                        ],
                    }
                )
            monitor.record_cut(persisted)
            applied.append(persisted)
        finally:
            context.play()
    if applied or report.get("blade_cut_errors"):
        report["blade_cutting"] = monitor.summary
        if grasp_manager is not None:
            report["bimanual_task"] = grasp_manager.summary
        if report_path is not None:
            _emit(report, report_path)
    return applied


def _opposed_finger_contact(contact: dict | None) -> bool:
    """Return whether a nonzero contact loads both left fingers."""
    if contact is None or float(contact.get("force_n", 0.0)) <= 0.0:
        return False
    return {"left_finger_1", "left_finger_2"}.issubset(
        set(contact.get("fingers", ()))
    )


class InteractionController:
    """Queue UI/input actions and apply them between simulation steps."""

    controls = {
        "mouse": "Shift + left-drag visible stem, petiole, or leaf geometry",
        "previous_target": "[",
        "next_target": "]",
        "next_vine": "V",
        "debug_force_cut": "C (not benchmark-valid)",
        "physical_cut": "right leading-edge contact + direction + force + work",
        "left_grasp_close": "G",
        "left_grasp_open_release": "O",
        "target_pinched_leaf": "T (explicit manual target change)",
        "run_full_ik": "Run Full IK Sequence button",
        "camera_views": "1 inspection, 2 head, 3 left wrist, 4 right wrist",
    }

    def __init__(
        self,
        stage,
        runtimes: list[VineRuntime],
        report: dict,
        report_path: pathlib.Path,
        args,
        vine_interaction,
        camera_views: dict[str, str],
        blade_cutting: BladeContactMonitor | None,
        grasp_manager: LeftGraspManager | None,
        selected_target,
        run_full_ik=None,
    ):
        import carb
        import omni.appwindow
        import omni.ui as ui

        self._carb = carb
        self._runtimes = runtimes
        self._report = report
        self._report_path = report_path
        self._pending: list[str] = []
        self._camera_views = camera_views
        self._blade_cutting = blade_cutting
        self._grasp_manager = grasp_manager
        self._active_camera = "inspection"
        self._targets = [self._target_labels(runtime) for runtime in runtimes]
        self._vine = next(
            index
            for index, runtime in enumerate(runtimes)
            if runtime.name == selected_target.vine_name
        )
        self._target = self._targets[self._vine].index(selected_target.organ_label)
        self._run_full_ik = run_full_ik
        self._ik_running = False
        self._reported_contact_serial = None

        self._window = ui.Window("Vine Interaction", width=420, height=495)
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Pull: Shift + left-drag visible vine geometry", height=24)
                self._target_label = ui.Label("", height=24)
                self._status_label = ui.Label("Ready; gentle airflow enabled", height=48, word_wrap=True)
                with ui.HStack(spacing=6, height=34):
                    ui.Button("Previous [", clicked_fn=lambda: self._queue("previous"))
                    ui.Button("Next ]", clicked_fn=lambda: self._queue("next"))
                with ui.HStack(spacing=6, height=34):
                    ui.Button("Next vine V", clicked_fn=lambda: self._queue("vine"))
                    ui.Button("DEBUG FORCE CUT C", clicked_fn=lambda: self._queue("cut"))
                with ui.HStack(spacing=6, height=34):
                    ui.Button("Close left grasp G", clicked_fn=lambda: self._queue("grasp"))
                    ui.Button("Open / release O", clicked_fn=lambda: self._queue("release"))
                ui.Button(
                    "Target pinched leaf T",
                    clicked_fn=lambda: self._queue("target_contact"),
                    height=34,
                )
                self._run_ik_button = ui.Button(
                    "Run Full IK Sequence",
                    clicked_fn=lambda: self._queue("run_full_ik"),
                    height=38,
                )
                ui.Label("Viewport camera / video observation", height=22)
                with ui.HStack(spacing=6, height=34):
                    ui.Button("Inspection 1", clicked_fn=lambda: self._queue("camera:inspection"))
                    ui.Button("Head D405 2", clicked_fn=lambda: self._queue("camera:head"))
                with ui.HStack(spacing=6, height=34):
                    ui.Button("Left D405 3", clicked_fn=lambda: self._queue("camera:left_wrist"))
                    ui.Button("Right D405 4", clicked_fn=lambda: self._queue("camera:right_wrist"))

        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self._airflow = vine_interaction.Airflow(
            stage,
            runtimes,
            speed_m_s=args.airflow_speed,
            frequency_hz=args.airflow_frequency,
            direction_deg=args.airflow_direction,
        )
        report["airflow"] = self._airflow.summary
        body_paths = [link.path for runtime in runtimes for link in runtime.rig.links]
        self._visual_pull = vine_interaction.VisualPull(
            stage,
            body_paths,
            report,
            persist=lambda: _emit(self._report, self._report_path),
            status=lambda message: setattr(self._status_label, "text", message),
            stiffness_n_m=args.drag_stiffness,
            damping_n_s_m=args.drag_damping,
            max_force_n=args.drag_max_force,
        )
        self._refresh()

    def _target_labels(self, runtime: VineRuntime) -> list[str]:
        labelled = [
            organ
            for organ in runtime.plant.organs
            if organ.label.startswith("SubStem_") and organ.label in runtime.rig.cut_joints
        ]
        labelled.sort(
            key=lambda organ: float(organ.attachment[1])
            if organ.attachment is not None
            else float("inf")
        )
        return [organ.label for organ in labelled]

    def _queue(self, action: str) -> None:
        self._pending.append(action)

    def _on_key(self, event, *args, **kwargs) -> bool:
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True
        action = {
            "LEFT_BRACKET": "previous",
            "RIGHT_BRACKET": "next",
            "V": "vine",
            "C": "cut",
            "G": "grasp",
            "O": "release",
            "T": "target_contact",
            "KEY_1": "camera:inspection",
            "KEY_2": "camera:head",
            "KEY_3": "camera:left_wrist",
            "KEY_4": "camera:right_wrist",
        }.get(event.input.name)
        if action is not None:
            self._queue(action)
        return True


    def process(self, context) -> None:
        if self._grasp_manager is not None:
            self._grasp_manager.process()
            contact = self._grasp_manager.latest_finger_contact
            if (
                contact is not None
                and contact["serial"] != self._reported_contact_serial
            ):
                self._reported_contact_serial = contact["serial"]
                current = (
                    f"{self._runtimes[self._vine].name}/"
                    f"{self._targets[self._vine][self._target]}"
                )
                if contact["key"] != current:
                    if (
                        self._grasp_manager.close_requested
                        and _opposed_finger_contact(contact)
                    ):
                        self._pending.insert(0, "target_contact:auto")
                        self._status_label.text = (
                            f"Opposed physical pinch detected on {contact['key']}; "
                            "selecting that branch"
                        )
                    else:
                        self._status_label.text = (
                            f"Finger contact is {contact['key']}, not selected {current}. "
                            "Press T or click Target pinched leaf."
                        )
        if self._blade_cutting is not None:
            applied = _apply_blade_cut_decisions(
                context,
                self._blade_cutting,
                self._report,
                self._report_path,
                self._grasp_manager,
            )
            if applied:
                latest = applied[-1]
                qualifier = "intended" if latest["intended_target"] else "UNINTENDED"
                self._status_label.text = (
                    f"Physical blade cut {latest['vine']}/{latest['organ_label']} "
                    f"({qualifier}, {latest['peak_force_n']:.1f} N)"
                )
        self._airflow.step()
        self._visual_pull.step()
        while self._pending:
            action = self._pending.pop(0)
            if action.startswith("camera:"):
                self._set_camera(action.partition(":")[2])
                continue
            targets = self._targets[self._vine]
            if not targets:
                self._status_label.text = "No deleafing targets on this vine"
                continue
            if action == "previous":
                self._target = (self._target - 1) % len(targets)
            elif action == "next":
                self._target = (self._target + 1) % len(targets)
            elif action == "vine":
                self._vine = (self._vine + 1) % len(self._runtimes)
                self._target = 0
            elif action == "grasp" and self._grasp_manager is not None:
                self._grasp_manager.request_close()
                self._status_label.text = (
                    "Left gripper closing; two-finger target contact is required"
                )
                self._report["bimanual_task"] = self._grasp_manager.summary
                _emit(self._report, self._report_path)
            elif action == "release" and self._grasp_manager is not None:
                self._grasp_manager.request_open()
                self._status_label.text = (
                    "Left gripper opened; release is valid only after safe transport"
                )
                self._report["bimanual_task"] = self._grasp_manager.summary
                _emit(self._report, self._report_path)
            elif action.startswith("target_contact") and self._grasp_manager is not None:
                automatic = action.endswith(":auto")
                contact = self._grasp_manager.latest_finger_contact
                if contact is None:
                    self._status_label.text = (
                        "No left-finger leaf contact has been detected yet"
                    )
                else:
                    vine_index = next(
                        (
                            index
                            for index, runtime in enumerate(self._runtimes)
                            if runtime.name == contact["vine"]
                        ),
                        None,
                    )
                    target_index = (
                        self._targets[vine_index].index(contact["organ"])
                        if vine_index is not None
                        and contact["organ"] in self._targets[vine_index]
                        else None
                    )
                    if vine_index is None or target_index is None:
                        self._status_label.text = (
                            f"Contacted {contact['key']} is not a severable target"
                        )
                    else:
                        previous = (
                            f"{self._runtimes[self._vine].name}/"
                            f"{self._targets[self._vine][self._target]}"
                        )
                        self._vine = vine_index
                        self._target = target_index
                        change = {
                            "from": previous,
                            "to": contact["key"],
                            "trigger": (
                                "opposed_finger_physical_pinch"
                                if automatic
                                else "explicit_pinched_leaf_selection"
                            ),
                        }
                        self._report.setdefault(
                            "contact_target_changes", []
                        ).append(change)
                        if not automatic:
                            self._report.setdefault(
                                "manual_target_changes", []
                            ).append(change)
                        qualifier = "Automatically" if automatic else "Manually"
                        self._status_label.text = (
                            f"{qualifier} selected {contact['key']} from physical contact; "
                            "maintain the opposed pinch"
                        )
            elif action == "run_full_ik":
                if self._run_full_ik is None:
                    self._status_label.text = "Full IK sequence is unavailable"
                elif self._ik_running:
                    self._status_label.text = "Full IK sequence is already running"
                else:
                    self._ik_running = True
                    self._run_ik_button.enabled = False
                    self._status_label.text = (
                        "Running grasp, cut, retract, and deposit sequence..."
                    )
                    try:
                        result = self._run_full_ik()
                    except Exception as exc:
                        self._status_label.text = (
                            f"Full IK sequence failed: {exc}"
                        )
                        self._report["ui_full_ik_error"] = str(exc)
                    else:
                        self._status_label.text = (
                            "Full IK sequence passed"
                            if result["succeeded"]
                            else "Full IK sequence failed: "
                            + result.get("error", "acceptance gate")
                        )
                    finally:
                        self._ik_running = False
                        _emit(self._report, self._report_path)
            elif action == "cut":
                self._cut(context)
            self._refresh()

    def _set_camera(self, name: str) -> None:
        camera_path = self._camera_views.get(name)
        if camera_path is None:
            self._status_label.text = f"Camera view is unavailable: {name}"
            return
        self._visual_pull.end()
        _focus_viewport(camera_path)
        self._active_camera = name
        self._report["active_camera_view"] = name
        self._report["active_camera_path"] = camera_path
        self._status_label.text = f"Viewport: {name} ({camera_path})"
        _emit(self._report, self._report_path)

    def run_visual_pull_probe(self, context, stage, runtime: VineRuntime) -> dict:
        import numpy as np
        from pxr import Usd
        from pxr import UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        candidates = []
        projected_samples = []
        visual_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(runtime.root_path)):
            if not prim.IsA(UsdGeom.Mesh) or not prim.GetName().startswith("Visual_"):
                continue
            visual_count += 1
            centre = cache.ComputeWorldBound(prim).ComputeCentroid()
            ndc = self._visual_pull.project_world(centre)
            if len(projected_samples) < 8:
                projected_samples.append(
                    {"visual": str(prim.GetPath()), "world": list(centre), "ndc": list(ndc)}
                )
            if np.isfinite(ndc).all() and abs(ndc[0]) < 0.95 and abs(ndc[1]) < 0.95:
                candidates.append((ndc[0] ** 2 + ndc[1] ** 2, ndc, str(prim.GetPath())))
        candidates.sort(key=lambda row: row[0])

        selected = None
        for _, ndc, visual in candidates[:24]:
            self._visual_pull.begin(ndc[:2])
            for _ in range(16):
                context.step(render=True)
                self._airflow.step()
                self._visual_pull.step()
            if self._visual_pull.active_body is not None:
                selected = (ndc, visual, self._visual_pull.active_body)
                break
            self._visual_pull.end()
        if selected is None:
            return {
                "succeeded": False,
                "error": "renderer raycast did not resolve dynamic visible geometry",
                "visual_meshes": visual_count,
                "on_screen_candidates": len(candidates),
                "projected_samples": projected_samples,
            }

        ndc, visual, body = selected
        start = _positions(stage, [body])[0]
        peak = 0.0
        self._visual_pull.update((min(ndc[0] + 0.08, 0.95), ndc[1]))
        for _ in range(60):
            context.step(render=True)
            self._airflow.step()
            self._visual_pull.step()
            peak = max(peak, float(np.linalg.norm(_positions(stage, [body])[0] - start)))
        self._visual_pull.end()
        for _ in range(240):
            context.step(render=True)
            self._airflow.step()
        final = _positions(stage, [body])[0]
        residual = float(np.linalg.norm(final - start))
        finite = bool(np.isfinite(final).all())
        return {
            "visual_candidate": visual,
            "body": body,
            "peak_displacement_mm": peak * 1000.0,
            "residual_displacement_mm": residual * 1000.0,
            "finite": finite,
            "succeeded": bool(finite and 0.001 < peak < 0.5 and residual < 0.1),
        }

    def _cut(self, context) -> None:
        runtime = self._runtimes[self._vine]
        label = self._targets[self._vine][self._target]
        context.pause()
        try:
            record = runtime.severer.cut(label, trigger="debug_forced")
        except Exception as exc:
            self._status_label.text = str(exc)
        else:
            forced = {
                **dataclasses.asdict(record),
                "vine": runtime.name,
                "benchmark_valid": False,
            }
            self._report.setdefault("debug_forced_cuts", []).append(forced)
            if self._grasp_manager is not None:
                self._grasp_manager.notify_cut(forced)
                self._report["bimanual_task"] = self._grasp_manager.summary
            self._status_label.text = (
                f"DEBUG forced cut {runtime.name}/{label}: {record.grade}; "
                "not benchmark-valid"
            )
            _emit(self._report, self._report_path)
        finally:
            context.play()

    def poll_tears(self) -> None:
        for runtime in self._runtimes:
            records = runtime.severer.poll_tears()
            if records:
                self._report.setdefault("tears", []).extend(
                    dataclasses.asdict(record) for record in records
                )
                self._status_label.text = f"Tore {runtime.name}/{records[-1].organ_label}"
                _emit(self._report, self._report_path)

    def _refresh(self) -> None:
        targets = self._targets[self._vine]
        label = targets[self._target] if targets else "none"
        self._target_label.text = (
            f"Vine {self._vine + 1}/{len(self._runtimes)}: "
            f"{self._runtimes[self._vine].name} | target {label}"
        )
        if self._blade_cutting is not None and targets:
            self._blade_cutting.set_active_target(
                self._runtimes[self._vine].name,
                label,
            )
        if self._grasp_manager is not None and targets:
            self._grasp_manager.set_active_target(
                self._runtimes[self._vine].name,
                label,
            )
        self._report["interaction_target"] = {
            "vine": self._runtimes[self._vine].name,
            "organ": label,
        }

    def close(self) -> None:
        self._visual_pull.close()
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        self._window.visible = False


class _TeleopCameraRecorder:
    """Read the latest rendered D405 RGB observations into an episode folder."""

    def __init__(self, camera_views, camera_names, resolution, frames_directory) -> None:
        import numpy as np
        import omni.replicator.core as rep
        from PIL import Image

        self._np = np
        self._rep = rep
        self._Image = Image
        self._frames_directory = pathlib.Path(frames_directory)
        self._streams = {}
        for name in camera_names:
            camera_path = camera_views.get(name)
            if camera_path is None:
                raise ValueError(f"teleop camera is unavailable: {name}")
            product = rep.create.render_product(camera_path, resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            self._streams[name] = (camera_path, product, annotator)
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=4)

    def capture(self, sample_index: int) -> tuple[dict[str, str], dict[str, str]]:
        images = {}
        errors = {}
        for name, (_, _, annotator) in self._streams.items():
            try:
                rgba = self._np.asarray(annotator.get_data())
                if rgba.ndim != 3 or rgba.shape[2] < 3 or rgba.size == 0:
                    raise ValueError(f"unexpected RGB shape {rgba.shape}")
                path = self._frames_directory / f"{sample_index:06d}_{name}.png"
                self._Image.fromarray(rgba[:, :, :3].astype(self._np.uint8)).save(path)
                images[name] = str(path.relative_to(self._frames_directory.parent))
            except Exception as exc:
                errors[name] = str(exc)
        return images, errors

    def close(self) -> None:
        for _, product, annotator in self._streams.values():
            with contextlib.suppress(Exception):
                annotator.detach([product])
            with contextlib.suppress(Exception):
                product.destroy()
        self._streams.clear()


class SimulatorTeleop:
    """Rate-limited, watchdog-protected leader-arm input for the simulator."""

    def __init__(
        self,
        stage,
        args,
        camera_views,
        selected_target,
        report,
        report_path,
        teleop_module,
        robot_hardware,
        robot_kinematics,
        blade_monitor,
        grasp_manager,
        contact_diagnostics,
    ) -> None:
        import os
        import time

        import numpy as np
        from isaacsim.core.prims import Articulation
        from pxr import Usd
        from pxr import UsdGeom

        self._time = time
        self._np = np
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._stage = stage
        self._args = args
        self._teleop = teleop_module
        self._report = report
        self._report_path = report_path
        self._blade_monitor = blade_monitor
        self._grasp_manager = grasp_manager
        self._contact_diagnostics = contact_diagnostics
        self._camera_views = dict(camera_views)
        self._mailbox = teleop_module.CommandMailbox(args.teleop_command_file)
        self._model = robot_kinematics.Rby1Kinematics()
        limits = {
            side: self._model.arm_limits_degrees(side) for side in ("left", "right")
        }
        self._gate = teleop_module.TeleopSafetyGate(
            limits,
            watchdog_s=args.teleop_watchdog_ms / 1000.0,
            maximum_joint_speed_deg_s=args.teleop_max_joint_speed,
            joint_group_limits_degrees={
                "torso": _RBY1_TORSO_LIMITS_DEGREES,
                "head": _RBY1_HEAD_LIMITS_DEGREES,
            },
            joint_group_maximum_speeds_deg_s={
                "torso": 120.0,
                "head": 175.0,
            },
        )
        self._articulation = Articulation(
            prim_paths_expr="/World/RBY1",
            name="greenhouse_teleop_rby1",
            reset_xform_properties=False,
        )
        self._articulation.initialize()
        self._joint_group_dof_names = {
            "left": [f"left_arm_{index}" for index in range(7)],
            "right": [f"right_arm_{index}" for index in range(7)],
            "torso": [f"torso_{index}" for index in range(6)],
            "head": [f"head_{index}" for index in range(2)],
        }
        missing_dofs = sorted(
            name
            for names in self._joint_group_dof_names.values()
            for name in names
            if name not in self._articulation.dof_names
        )
        if missing_dofs:
            raise RuntimeError(f"RB-Y1 articulation is missing DOFs: {missing_dofs}")
        self._joint_group_dof_indices = {
            group: [self._articulation.get_dof_index(name) for name in names]
            for group, names in self._joint_group_dof_names.items()
        }
        self._previous_blade_point = None
        self._last_process_time = time.monotonic()
        self._last_command_time = None
        self._last_record_time = -float("inf")
        self._last_publish_time = -float("inf")
        self._last_gripper_closed = None
        self._last_gripper_openness = None
        self._commanded_targets = None
        self._last_safe_targets = None
        self._hold_targets = None
        self._hold_captures = 0
        self._contact_recoveries = 0
        self._latest_error = None
        self._unsafe_contacts = []
        self._unsafe_latched = False
        self._accepted_commands = 0
        self._rejected_commands = 0
        self._rate_limited_commands = 0
        self._watchdog_holds = 0
        self._last_sequence = None
        self._latest_age_ms = None
        self._recording = False
        self._closed = False
        tracked_groups = ("left", "right", "torso", "head")
        self._latest_simulated_joint_state = {
            group: None for group in tracked_groups
        }
        self._latest_source_joint_degrees = {
            group: None for group in tracked_groups
        }
        self._latest_source_joint_degrees["left_gripper_openness"] = None
        self._latest_applied_target_degrees = {
            group: None for group in tracked_groups
        }

        self._drive_configuration = {
            side: _configure_arm_drives(stage, side) for side in ("left", "right")
        }
        self._drive_configuration["torso"] = _configure_joint_group_drives(
            stage,
            "torso",
            _RBY1_TORSO_EFFORT_LIMITS_NM,
            _RBY1_TORSO_STIFFNESS_NM_RAD,
            _RBY1_TORSO_DAMPING_NM_S_RAD,
        )
        self._drive_configuration["head"] = _configure_joint_group_drives(
            stage,
            "head",
            _RBY1_HEAD_EFFORT_LIMITS_NM,
            _RBY1_HEAD_STIFFNESS_NM_RAD,
            _RBY1_HEAD_DAMPING_NM_S_RAD,
        )
        self._recorder = None
        self._camera_recorder = None
        self._episode_directory = None
        if args.teleop_record_dir is not None:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            self._episode_directory = pathlib.Path(args.teleop_record_dir) / (
                f"episode_{stamp}_{os.getpid()}"
            )
            self._recorder = teleop_module.DemonstrationRecorder(
                self._episode_directory,
                {
                    "schema": "greenhouse.demonstration.v1",
                    "target": selected_target.key,
                    "episode_seed": selected_target.seed,
                    "command_schema": teleop_module.SCHEMA,
                    "command_file": str(pathlib.Path(args.teleop_command_file).resolve()),
                    "simulator_only": True,
                    "physical_robot_commanded": False,
                    "physics_hz": 240.0,
                    "record_hz": args.teleop_record_hz,
                    "camera_resolution": [args.teleop_width, args.teleop_height],
                    "cameras": list(args.teleop_cameras),
                    "watchdog_ms": args.teleop_watchdog_ms,
                    "maximum_joint_speed_deg_s": args.teleop_max_joint_speed,
                },
            )

        self._publish(force=True)

    def _joint_group_state(self, group: str) -> dict:
        indices = self._joint_group_dof_indices[group]
        expected = len(indices)
        positions = self._np.asarray(
            self._articulation.get_joint_positions(joint_indices=indices),
            dtype=self._np.float64,
        ).reshape(-1)
        velocities = self._np.asarray(
            self._articulation.get_joint_velocities(joint_indices=indices),
            dtype=self._np.float64,
        ).reshape(-1)
        if positions.size != expected or velocities.size != expected:
            raise RuntimeError(
                f"live PhysX state for {group} has unexpected shape "
                f"({positions.size}, {velocities.size})"
            )
        if not self._np.isfinite(positions).all() or not self._np.isfinite(velocities).all():
            raise RuntimeError(f"live PhysX state for {group} is non-finite")
        return {
            "position_degrees": self._np.degrees(positions).tolist(),
            "velocity_degrees_s": self._np.degrees(velocities).tolist(),
        }
    def _ee_matrix(self, side: str) -> list[list[float]]:
        matrix = self._UsdGeom.Xformable(
            self._stage.GetPrimAtPath(f"/World/RBY1/ee_{side}")
        ).ComputeLocalToWorldTransform(self._Usd.TimeCode.Default())
        return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]

    def _current_unsafe_contacts(self) -> list[dict]:
        if self._contact_diagnostics is None:
            return []
        blade_geometry = self._blade_monitor.target_geometry
        grasp_geometry = self._grasp_manager.target_geometry
        if blade_geometry is None or grasp_geometry is None:
            return []
        return _probe_unsafe_contacts(
            self._contact_diagnostics.active_summary,
            blade_geometry,
            grasp_geometry,
        )

    def _set_blade_velocity(self, *, moving: bool, dt_s: float) -> None:
        point = self._blade_monitor.edge_centre_m
        velocity = self._np.zeros(3, dtype=self._np.float64)
        if moving and self._previous_blade_point is not None and dt_s > 0.0:
            velocity = (point - self._previous_blade_point) / dt_s
        self._previous_blade_point = point.copy()
        self._blade_monitor.set_commanded_edge_velocity(velocity)
    def _record(self, simulation_step: int, now_s: float, command, states) -> None:
        if self._recorder is None or not command.recording:
            return
        if now_s - self._last_record_time < 1.0 / self._args.teleop_record_hz:
            return
        if self._camera_recorder is None:
            self._camera_recorder = _TeleopCameraRecorder(
                self._camera_views,
                self._args.teleop_cameras,
                (self._args.teleop_width, self._args.teleop_height),
                self._recorder.frames_directory,
            )
        sample_index = self._recorder.samples
        images, camera_errors = self._camera_recorder.capture(sample_index)
        grasp_summary = self._grasp_manager.summary
        task = grasp_summary.get("task") or {}
        blade_summary = self._blade_monitor.summary
        self._recorder.append(
            {
                "schema": "greenhouse.demonstration.step.v1",
                "sample_index": sample_index,
                "simulation_step": simulation_step,
                "simulation_time_s": simulation_step / 240.0,
                "host_monotonic_time_s": now_s,
                "target": blade_summary.get("active_target"),
                "observation": {
                    "left_arm": states["left"],
                    "right_arm": states["right"],
                    "torso": states["torso"],
                    "head": states["head"],
                    "left_gripper_openness": command.left_gripper_openness,
                    "left_ee_world_matrix": self._ee_matrix("left"),
                    "right_ee_world_matrix": self._ee_matrix("right"),
                    "cameras": images,
                    "camera_errors": camera_errors,
                },
                "action": dataclasses.asdict(command),
                "task": {
                    "phase": task.get("phase"),
                    "grasp_active": task.get("grasp_active"),
                    "task_succeeded": self._grasp_manager.task_succeeded,
                    "active_grasp_body": grasp_summary.get("active_grasp_body"),
                    "physical_cuts": len(blade_summary.get("physical_cuts", ())),
                    "cut_feedback": self._blade_monitor.active_cut_feedback,
                },
                "safety": {
                    "blade_safety_clear": self._blade_monitor.safety_clear,
                    "unsafe_contact_count": len(self._unsafe_contacts),
                    "unsafe_latched": self._unsafe_latched,
                },
            }
        )
        self._last_record_time = now_s

    @property
    def summary(self) -> dict:
        now = self._time.monotonic()
        watchdog_age_ms = (
            None
            if self._last_command_time is None
            else (now - self._last_command_time) * 1000.0
        )
        return {
            "schema": self._teleop.SCHEMA,
            "simulator_only": True,
            "physical_robot_commanded": False,
            "command_file": str(pathlib.Path(self._args.teleop_command_file).resolve()),
            "watchdog_ms": self._args.teleop_watchdog_ms,
            "maximum_joint_speed_deg_s": self._args.teleop_max_joint_speed,
            "accepted_commands": self._accepted_commands,
            "rejected_commands": self._rejected_commands,
            "rate_limited_commands": self._rate_limited_commands,
            "watchdog_holds": self._watchdog_holds,
            "last_sequence": self._last_sequence,
            "latest_command_age_ms": self._latest_age_ms,
            "watchdog_age_ms": watchdog_age_ms,
            "watchdog_fresh": (
                watchdog_age_ms is not None
                and watchdog_age_ms <= self._args.teleop_watchdog_ms
            ),
            "unsafe_latched": self._unsafe_latched,
            "unsafe_contacts": self._unsafe_contacts,
            "contact_policy": self._args.teleop_contact_policy,
            "contact_recoveries": self._contact_recoveries,
            "recording": self._recording,
            "recorded_samples": self._recorder.samples if self._recorder is not None else 0,
            "episode_directory": (
                str(self._episode_directory) if self._episode_directory is not None else None
            ),
            "latest_error": self._latest_error,
            "hold_active": self._hold_targets is not None,
            "hold_captures": self._hold_captures,
            "hold_target_degrees": self._hold_targets,
            "simulated_joint_state": self._latest_simulated_joint_state,
            "source_joint_degrees": self._latest_source_joint_degrees,
            "applied_target_degrees": self._latest_applied_target_degrees,
            "drive_configuration": self._drive_configuration,
        }

    def _publish(self, *, force: bool = False) -> None:
        now = self._time.monotonic()
        if not force and now - self._last_publish_time < 1.0:
            return
        self._report["teleoperation"] = self.summary
        _emit(self._report, self._report_path)
        self._last_publish_time = now

    def _ensure_commanded_targets(self, states) -> None:
        if self._commanded_targets is None:
            self._commanded_targets = {
                group: list(state["position_degrees"])
                for group, state in states.items()
            }

    def _hold_joint_groups(self, states) -> None:
        """Hold one captured pose; never chase a gravity-driven falling state."""
        self._ensure_commanded_targets(states)
        if self._hold_targets is None:
            self._hold_targets = {
                group: list(state["position_degrees"])
                for group, state in states.items()
            }
            self._hold_captures += 1
        for group, target in self._hold_targets.items():
            _set_joint_group_drive_targets(self._stage, group, target)
        self._latest_applied_target_degrees = {
            group: list(target)
            for group, target in self._hold_targets.items()
        }

    def _rollback_to_last_safe(self, states) -> None:
        """Drive back one contact-free sample instead of freezing in contact."""
        self._ensure_commanded_targets(states)
        if self._hold_targets is None:
            source = self._last_safe_targets or {
                group: list(state["position_degrees"])
                for group, state in states.items()
            }
            self._hold_targets = {
                group: list(target) for group, target in source.items()
            }
            self._hold_captures += 1
        for group, target in self._hold_targets.items():
            _set_joint_group_drive_targets(self._stage, group, target)
        self._latest_applied_target_degrees = {
            group: list(target)
            for group, target in self._hold_targets.items()
        }
    def process(self, simulation_step: int) -> None:
        now = self._time.monotonic()
        dt_s = max(now - self._last_process_time, 1.0 / 240.0)
        self._last_process_time = now
        states = {
            group: self._joint_group_state(group)
            for group in ("left", "right", "torso", "head")
        }
        self._latest_simulated_joint_state = states
        self._ensure_commanded_targets(states)
        was_recovering = self._unsafe_latched
        self._unsafe_contacts = self._current_unsafe_contacts()
        if (
            self._unsafe_contacts
            and self._args.teleop_contact_policy == "rollback"
        ):
            if not was_recovering:
                self._contact_recoveries += 1
            self._unsafe_latched = True
            self._rollback_to_last_safe(states)
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            self._latest_error = (
                "unsafe robot contact active; rolling back to the last "
                "contact-free pose"
            )
            self._publish()
            return

        if was_recovering:
            self._unsafe_latched = False
            self._hold_targets = None
            self._commanded_targets = {
                group: list(state["position_degrees"])
                for group, state in states.items()
            }
            self._latest_error = None
        if not self._unsafe_contacts:
            self._last_safe_targets = {
                group: list(state["position_degrees"])
                for group, state in states.items()
            }
        try:
            raw = self._mailbox.poll()
        except self._teleop.TeleopCommandError as exc:
            self._rejected_commands += 1
            self._latest_error = str(exc)
            self._hold_joint_groups(states)
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            self._publish()
            return
        if raw is None:
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            if (
                self._last_command_time is not None
                and now - self._last_command_time > self._args.teleop_watchdog_ms / 1000.0
            ):
                self._hold_joint_groups(states)
                self._recording = False
                self._watchdog_holds += 1
                self._latest_error = "teleop watchdog expired; measured pose is held"
            self._publish()
            return

        self._latest_source_joint_degrees = {
            "left": list(raw.left.joint_degrees),
            "right": list(raw.right.joint_degrees),
            "torso": (
                list(raw.torso.joint_degrees) if raw.torso is not None else None
            ),
            "head": (
                list(raw.head.joint_degrees) if raw.head is not None else None
            ),
            "left_gripper_openness": (
                raw.left_gripper.openness
                if raw.left_gripper is not None
                else None
            ),
        }
        try:
            command = self._gate.accept(
                raw,
                now_s=now,
                dt_s=dt_s,
                current_left_degrees=states["left"]["position_degrees"],
                current_right_degrees=states["right"]["position_degrees"],
                current_torso_degrees=states["torso"]["position_degrees"],
                current_head_degrees=states["head"]["position_degrees"],
            )
        except self._teleop.TeleopCommandError as exc:
            self._rejected_commands += 1
            self._latest_error = str(exc)
            self._hold_joint_groups(states)
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            self._publish()
            return

        targets = {
            "left": command.left_target_degrees,
            "right": command.right_target_degrees,
            "torso": command.torso_target_degrees,
            "head": command.head_target_degrees,
        }
        apply = {
            "left": command.apply_left,
            "right": command.apply_right,
            "torso": command.apply_torso,
            "head": command.apply_head,
        }
        self._hold_targets = None
        for group in ("left", "right", "torso", "head"):
            target = targets[group]
            if apply[group] and target is not None:
                self._commanded_targets[group] = list(target)
            _set_joint_group_drive_targets(
                self._stage, group, self._commanded_targets[group]
            )
        self._latest_applied_target_degrees = {
            group: list(target)
            for group, target in self._commanded_targets.items()
        }

        if command.apply_left_gripper:
            if command.left_gripper_openness is not None:
                openness = float(command.left_gripper_openness)
                if (
                    self._last_gripper_openness is None
                    or abs(openness - self._last_gripper_openness) >= 1e-4
                ):
                    self._grasp_manager.request_openness(openness)
                    self._last_gripper_openness = openness
                self._last_gripper_closed = openness <= 0.10
            elif command.left_gripper_closed != self._last_gripper_closed:
                if command.left_gripper_closed:
                    self._grasp_manager.request_close()
                else:
                    self._grasp_manager.request_open()
                self._last_gripper_closed = command.left_gripper_closed
                self._last_gripper_openness = (
                    0.0 if command.left_gripper_closed else 1.0
                )

        self._set_blade_velocity(
            moving=command.apply_right or command.apply_torso,
            dt_s=dt_s,
        )
        self._accepted_commands += 1
        self._rate_limited_commands += int(command.rate_limited)
        self._last_sequence = command.sequence
        self._last_command_time = now
        self._latest_age_ms = command.age_s * 1000.0
        self._latest_error = None
        self._recording = bool(command.recording and self._recorder is not None)
        self._record(simulation_step, now, command, states)
        self._publish()
    def close(self) -> None:
        if self._closed:
            return
        self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
        if self._camera_recorder is not None:
            self._camera_recorder.close()
        if self._recorder is not None:
            self._recorder.close()
        self._recording = False
        self._closed = True
        self._publish(force=True)


def _run_headless_checks(stage, context, runtimes, selected_target, args, report: dict) -> bool:
    import numpy as np

    success = True
    stability = []
    for runtime in runtimes:
        paths = _base_link_paths(runtime.rig)
        now = _positions(stage, paths)
        moved = np.linalg.norm(now - runtime.rest_positions, axis=1)
        finite = bool(np.isfinite(now).all())
        runaways = int(np.sum(moved > 0.1))
        largest = np.argsort(moved)[::-1][:5]
        stability.append(
            {
                "vine": runtime.name,
                "finite": finite,
                "max_displacement_mm": float(np.nanmax(moved) * 1000.0),
                "runaway_organs": runaways,
                "organs_tracked": len(paths),
                "largest_displacements": [
                    {
                        "path": paths[int(index)],
                        "displacement_mm": float(moved[int(index)] * 1000.0),
                    }
                    for index in largest
                ],
            }
        )
        success = success and finite and runaways == 0
    report["stability"] = stability

    if report.get("robot_requested"):
        robot_stability = _robot_stability(stage, args)
        report["robot_stability"] = robot_stability
        success = success and bool(robot_stability["succeeded"])
        selected_runtime = next(
            runtime for runtime in runtimes if runtime.name == selected_target.vine_name
        )
        robot_precontact = _robot_precontact(
            stage,
            selected_runtime,
            selected_target.organ_label,
            args.robot_yaw,
        )
        report["robot_precontact"] = robot_precontact
        success = success and bool(robot_precontact["succeeded"])

    if args.airflow_probe_steps > 0:
        from greenhouse_sim import vine_interaction

        airflow = vine_interaction.Airflow(
            stage,
            runtimes,
            speed_m_s=args.airflow_speed,
            frequency_hz=args.airflow_frequency,
            direction_deg=args.airflow_direction,
        )
        report["airflow"] = airflow.summary
        airflow_probe = _airflow_probe(stage, context, airflow, args.airflow_probe_steps)
        report["airflow_probe"] = airflow_probe
        success = success and bool(airflow_probe["succeeded"])

    if args.pull_probe:
        pull = _pull_probe(stage, context, runtimes[0], args.pull_probe, args)
        report["pull_probe"] = pull
        success = success and bool(pull["succeeded"])
    if args.cut:
        cut = _cut_probe(stage, context, runtimes[0], args.cut, args.post_cut_steps)
        report["cut_probe"] = cut
        success = success and bool(cut["succeeded"])
    return success


def _robot_stability(stage, args) -> dict:
    import numpy as np
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root_path = "/World/RBY1"
    rigid_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{root_path}/") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    positions = _positions(stage, rigid_paths)
    base = stage.GetPrimAtPath(f"{root_path}/base")
    if not rigid_paths or not base.IsValid():
        return {"succeeded": False, "error": "fitted robot rigid bodies are missing"}

    base_matrix = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    base_position = np.asarray(base_matrix.ExtractTranslation(), dtype=np.float64)
    expected = np.asarray(args.robot_position, dtype=np.float64)
    displacement = base_position - expected
    up = np.asarray(base_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)), dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    tilt_degrees = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    finite = bool(np.isfinite(positions).all() and np.isfinite(up).all())
    succeeded = bool(
        finite
        and len(rigid_paths) == 34
        and np.linalg.norm(displacement[:2]) < 0.05
        and abs(displacement[2]) < 0.10
        and tilt_degrees < 10.0
    )
    return {
        "rigid_bodies": len(rigid_paths),
        "finite": finite,
        "base_position_m": base_position.tolist(),
        "base_displacement_mm": (displacement * 1000.0).tolist(),
        "base_tilt_degrees": tilt_degrees,
        "succeeded": succeeded,
    }


def _robot_precontact(
    stage,
    runtime: VineRuntime,
    target_label: str,
    robot_yaw_degrees: float,
) -> dict:
    """Measure the settled knife pose against the selected physical petiole."""
    from greenhouse_sim import robot_hardware
    import numpy as np
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom

    target_position = runtime.cut_sites.get(target_label)
    if target_position is None:
        return {"succeeded": False, "error": f"missing cut site {target_label}"}

    knife_root_path = "/World/RBY1/ee_right/attachments/DeleafKnife"
    blade_path = f"{knife_root_path}/Blade"
    arc_path = f"{knife_root_path}/Arc"
    knife_root = stage.GetPrimAtPath(knife_root_path)
    blade = stage.GetPrimAtPath(blade_path)
    arc = stage.GetPrimAtPath(arc_path)
    if not knife_root.IsValid() or not blade.IsValid() or not arc.IsValid():
        return {"succeeded": False, "error": "fitted knife geometry is missing"}

    target = np.asarray(target_position, dtype=np.float64)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )

    def bounds_and_distance(prim):
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = np.asarray(extent.GetMin(), dtype=np.float64)
        maximum = np.asarray(extent.GetMax(), dtype=np.float64)
        outside = np.maximum(np.maximum(minimum - target, target - maximum), 0.0)
        return minimum, maximum, float(np.linalg.norm(outside))

    blade_min, blade_max, blade_distance = bounds_and_distance(blade)
    arc_min, arc_max, arc_distance = bounds_and_distance(arc)
    knife_matrix = UsdGeom.Xformable(knife_root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    blade_matrix = UsdGeom.Xformable(blade).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    blade_extension = np.asarray(
        blade_matrix.TransformDir(
            Gf.Vec3d(*robot_hardware.KNIFE_CUT_DIRECTION_LOCAL.tolist())
        ).GetNormalized(),
        dtype=np.float64,
    )
    arc_facing = np.asarray(
        blade_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized(),
        dtype=np.float64,
    )
    finite = bool(
        np.isfinite(target).all()
        and np.isfinite(blade_min).all()
        and np.isfinite(blade_max).all()
        and np.isfinite(arc_min).all()
        and np.isfinite(arc_max).all()
        and np.isfinite(blade_extension).all()
        and np.isfinite(arc_facing).all()
    )
    # Both components must remain within the arm's conservative one-metre
    # approach envelope and outside a 5 mm no-spawn-contact margin. The wider
    # upper bound accommodates the lower-gutter-safe raised/rearward stow; the
    # full probe separately proves bounded IK, inter-arm clearance, contact
    # safety, and the physical cut. The blade points into the row and the U faces up.
    yaw_radians = np.radians(float(robot_yaw_degrees))
    robot_forward_xy = np.asarray(
        [np.cos(yaw_radians), np.sin(yaw_radians)], dtype=np.float64
    )
    blade_forward_alignment = float(np.dot(blade_extension[:2], robot_forward_xy))
    succeeded = bool(
        finite
        and 0.005 < blade_distance < 1.000
        and 0.005 < arc_distance < 1.000
        and blade_forward_alignment > 0.70
        and arc_facing[2] > 0.70
    )
    return {
        "vine": runtime.name,
        "target": target_label,
        "target_position_m": target.tolist(),
        "knife_root_position_m": list(knife_matrix.ExtractTranslation()),
        "blade_extension": blade_extension.tolist(),
        "blade_forward_alignment": blade_forward_alignment,
        "arc_facing": arc_facing.tolist(),
        "blade_distance_to_target_mm": blade_distance * 1000.0,
        "arc_distance_to_target_mm": arc_distance * 1000.0,
        "blade_bounds_m": {"min": blade_min.tolist(), "max": blade_max.tolist()},
        "arc_bounds_m": {"min": arc_min.tolist(), "max": arc_max.tolist()},
        "finite": finite,
        "succeeded": succeeded,
    }


_LEFT_AISLE_CLEARANCE_WAYPOINTS_DEGREES = (
    (13.953, 28.379, 26.822, -149.999, 77.017, 106.965, 0.0),
    (-3.647, 19.746, 40.289, -149.999, 115.225, 89.831, 0.0),
    (-39.855, 4.358, 18.724, -137.646, 99.514, 70.217, 0.0),
    (-61.416, -0.999, 9.339, -119.583, 87.697, 58.299, 0.0),
    (-69.063, -0.999, 7.054, -106.906, 80.808, 52.222, 0.0),
)
_LEFT_READY_DEGREES = (0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0)
_RIGHT_SAFE_DEGREES = (
    -146.58120585198483,
    -83.52711902452849,
    44.100760676248115,
    -118.2726158812136,
    -84.20799011233542,
    109.99942703110308,
    -67.89880648450034,
)
_LEFT_JAW_CENTRE_M = (0.0, 0.0, -0.1025)
_RIGHT_MULTISTART_SEEDS_DEGREES = (
    (-74.089, -77.473, 61.273, -109.016, 26.816, 40.399, -61.070),
    (-141.544, -59.888, 102.853, -128.673, -43.169, 109.999, -118.738),
)

# Track the outer flat-blade wing against the live articulated petiole rather
# than extending Link 0 as if the bent chain were straight. The yawed support
# provides parent-stem clearance.
_RIGHT_EDGE_WING_M = (-0.014, -0.07047998, 0.0)
# The physical plate is 13 mm thick along the petiole tangent. A 12 mm target
# lets its proximal face overlap the main-stem envelope, while a 20 mm target
# lets the loaded compliant contact migrate beyond the 25 mm admissible zone.
# The controlled load moved the 18 mm plate contact just beyond the admissible
# zone; 16 mm keeps the full physical plate inside it while force feedback stops
# the late high-load main-stem excursion seen in the earlier uncontrolled run.
_RIGHT_CUT_STUB_M = 0.016
_RIGHT_COMMITTED_CUT_STUB_CANDIDATES_M = (
    0.013,
    0.014,
    0.015,
    0.016,
    0.017,
    0.018,
    0.019,
)
_RIGHT_KNIFE_YAW_DEGREES = -25.0
_RIGHT_KNIFE_ROLL_DEGREES = 0.0
_RIGHT_SERVO_MAX_ATTEMPTS = 7
_RIGHT_COMFORTABLE_PAYLOAD_CLEARANCE_M = 0.012
_LEFT_APPROACH_SEEDS_DEGREES = {
    0.10: (-103.781, -0.999, -25.819, -115.255, -18.501, 109.999, -23.873),
    0.06: (-107.042, -0.999, -22.579, -104.214, -15.552, 109.999, -19.398),
    0.04: (-109.778, -0.999, -20.964, -96.981, -13.813, 109.999, -16.244),
    0.02: (-113.954, -0.999, -19.378, -87.399, -11.571, 109.999, -11.855),
    0.01: (-117.218, -0.999, -18.627, -80.651, -9.925, 109.999, -8.657),
    0.00: (-120.779, 0.740, -20.440, -74.103, -5.287, 109.523, -3.747),
}
# Ordered fallback reached the second supplied petiole with 0.167 mm point
# error while retaining the same aisle-side shoulder branch. It is considered
# only when both the previous live solution and the accepted SubStem_00 seed
# fail the unchanged 1 mm / 50 degree point-axis criteria.
_LEFT_MULTISTART_SEEDS_DEGREES = (
    (-139.249, -0.999, -89.957, -20.278, 60.752, 49.790, -48.653),
)
_LEFT_TRANSPORT_SEED_DEGREES = (
    -108.062,
    -0.999,
    -26.702,
    -124.530,
    -20.523,
    109.999,
    -29.550,
)
_RBY1_ARM_EFFORT_LIMITS_NM = (70.0, 70.0, 70.0, 40.0, 10.0, 10.0, 8.0)
_RBY1_ARM_STIFFNESS_NM_RAD = (350.0, 350.0, 300.0, 250.0, 80.0, 80.0, 50.0)
_RBY1_ARM_DAMPING_NM_S_RAD = (40.0, 40.0, 36.0, 30.0, 8.0, 8.0, 6.0)
# Exact Model A v1.0 URDF ranges, expressed in the degree units used by
# Isaac's angular drive targets. Torso speed is capped at its slowest 120 deg/s
# hardware limit; both head joints are capped below their 180 deg/s limit.
_RBY1_TORSO_LIMITS_DEGREES = (
    (-15.0, -30.0, -150.0, -45.0, -30.0, -135.0),
    (15.0, 90.0, 90.0, 90.0, 30.0, 135.0),
)
_RBY1_HEAD_LIMITS_DEGREES = (
    (-29.965988, -20.053523),
    (29.965988, 89.954374),
)
_RBY1_TORSO_EFFORT_LIMITS_NM = (270.0, 270.0, 270.0, 120.0, 120.0, 120.0)
_RBY1_TORSO_STIFFNESS_NM_RAD = (1200.0, 1200.0, 1200.0, 800.0, 600.0, 500.0)
_RBY1_TORSO_DAMPING_NM_S_RAD = (120.0, 120.0, 120.0, 80.0, 60.0, 50.0)
_RBY1_HEAD_EFFORT_LIMITS_NM = (20.0, 20.0)
_RBY1_HEAD_STIFFNESS_NM_RAD = (120.0, 120.0)
_RBY1_HEAD_DAMPING_NM_S_RAD = (16.0, 16.0)
_LEFT_COUNTERHOLD_CUT_FORCE_SHARE = 0.5
_LEFT_PRETENSION_PULL_M = 0.015
_LEFT_PRETENSION_MAX_ORIENTATION_ERROR_DEGREES = 3.0
_MINIMUM_WRIST_CAMERA_CLEARANCE_M = 0.005
_RIGHT_PLANNING_PAYLOAD_CLEARANCE_M = 0.008
# The committed blade stroke has an exact live path and is checked every 240 Hz
# sample by the unchanged 5 mm payload stop plus the zero-tolerance protected
# PhysX contact ledger. Live compliant motion reduced the fixed-roll centreline
# prediction to 5.184 mm while the preceding physical sample still held 7.334
# mm. Retain a 0.1 mm planning reserve so that route remains executable; the
# per-sample 5 mm hard stop is unchanged.
_RIGHT_COMMITTED_CUT_PLANNING_CLEARANCE_M = 0.0051
# The conservative knife-support boxes may enter the selected orphan branch's
# envelope while the blade advances.  Preserve a non-contact geometric reserve
# for that intended target only; protected neighbours and the parent plant keep
# their normal planning/runtime margins, and actual PhysX contact still fails
# the probe through the contact ledger.
_RIGHT_TARGET_ORPHAN_SUPPORT_PLANNING_CLEARANCE_M = 0.0005
_MINIMUM_INTER_ARM_CLEARANCE_M = 0.005


def _target_grasp_candidates(stage, runtime, organ_label: str, base_planner):
    """Read startup grasp geometry from the same authored physical colliders."""
    import numpy as np
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom

    organ_index = runtime.organ_indices[organ_label]
    descendants = set(runtime.plant.descendants_of(organ_index))
    orphan_bodies = {
        link.path for link in runtime.rig.links if link.organ in descendants
    }
    excluded_finger_colliders = tuple(
        info.path
        for info in runtime.rig.colliders
        if info.body_path in orphan_bodies
    )
    candidates = []
    for info in runtime.rig.colliders:
        if info.organ_label != organ_label or (
            "grasp" not in info.role and "petiole_cut_zone" not in info.role
        ):
            continue
        collider_matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath(info.path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath(info.body_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        axis = np.asarray(
            body_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized(),
            dtype=np.float64,
        )
        candidates.append(
            base_planner.GraspCandidate(
                collider=info.path,
                body=info.body_path,
                segment=int(info.segment),
                role=info.role,
                centre_m=tuple(
                    float(value) for value in collider_matrix.ExtractTranslation()
                ),
                axis=tuple(float(value) for value in axis),
                excluded_finger_colliders=excluded_finger_colliders,
            )
        )
    return tuple(candidates)


def _vine_capsule_obstacles(stage, robot_kinematics):
    """Return current physical vine/safety capsules as world-space obstacles."""
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom

    axis_vectors = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    obstacles = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(("/World/InteractiveVines/", "/World/NeighbourSafety/")):
            continue
        if not prim.IsA(UsdGeom.Capsule):
            continue
        capsule = UsdGeom.Capsule(prim)
        radius = float(capsule.GetRadiusAttr().Get() or 0.0)
        height = float(capsule.GetHeightAttr().Get() or 0.0)
        if radius <= 0.0:
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        centre = matrix.ExtractTranslation()
        axis_token = str(capsule.GetAxisAttr().Get() or "Z").upper()
        direction = matrix.TransformDir(axis_vectors[axis_token]).GetNormalized()
        half = direction * (0.5 * height)
        obstacles.append(
            robot_kinematics.CapsuleObstacle(
                path=path,
                start_m=tuple(float(value) for value in centre - half),
                end_m=tuple(float(value) for value in centre + half),
                radius_m=radius,
            )
        )
    return tuple(obstacles)


class _VineCapsuleObstacleCache:
    """Cache vine prim discovery and one transform snapshot per physics step."""

    def __init__(self, stage, robot_kinematics):
        from pxr import UsdGeom

        self._stage = stage
        self._robot_kinematics = robot_kinematics
        self._entries = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(
                ("/World/InteractiveVines/", "/World/NeighbourSafety/")
            ):
                continue
            if not prim.IsA(UsdGeom.Capsule):
                continue
            capsule = UsdGeom.Capsule(prim)
            radius = float(capsule.GetRadiusAttr().Get() or 0.0)
            if radius <= 0.0:
                continue
            self._entries.append(
                (
                    prim,
                    path,
                    radius,
                    float(capsule.GetHeightAttr().Get() or 0.0),
                    str(capsule.GetAxisAttr().Get() or "Z").upper(),
                )
            )
        self._snapshot = None
        self.snapshot_builds = 0
        self.cache_hits = 0

    def invalidate(self) -> None:
        self._snapshot = None

    def snapshot(self):
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom

        if self._snapshot is not None:
            self.cache_hits += 1
            return self._snapshot
        axis_vectors = {
            "X": Gf.Vec3d(1.0, 0.0, 0.0),
            "Y": Gf.Vec3d(0.0, 1.0, 0.0),
            "Z": Gf.Vec3d(0.0, 0.0, 1.0),
        }
        obstacles = []
        for prim, path, radius, height, axis_token in self._entries:
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            centre = matrix.ExtractTranslation()
            direction = matrix.TransformDir(
                axis_vectors[axis_token]
            ).GetNormalized()
            half = direction * (0.5 * height)
            obstacles.append(
                self._robot_kinematics.CapsuleObstacle(
                    path=path,
                    start_m=tuple(float(value) for value in centre - half),
                    end_m=tuple(float(value) for value in centre + half),
                    radius_m=radius,
                )
            )
        self._snapshot = tuple(obstacles)
        self.snapshot_builds += 1
        return self._snapshot

    @property
    def summary(self) -> dict:
        return {
            "capsule_prims": len(self._entries),
            "snapshot_builds": self.snapshot_builds,
            "cache_hits": self.cache_hits,
        }


def _greenhouse_structure_obstacles(
    stage,
    robot_kinematics,
    reference_position_m,
    horizontal_radius_m: float = 2.75,
):
    """Build compact AABB proxies for nearby rigid greenhouse structure."""
    import numpy as np
    from pxr import Usd
    from pxr import UsdGeom

    reference = np.asarray(reference_position_m, dtype=np.float64)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    candidates = []
    environment = stage.GetPrimAtPath("/World/Main_Cultivation_Zone/Env")
    for prim in environment.GetChildren() if environment.IsValid() else ():
        if prim.GetName() in {"Wall_01", "Wall_02", "Wall_03", "Wall_04"}:
            candidates.append(prim)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Main_Cultivation_Zone/Beds/"):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not (
            path.endswith("/Bed/Bed_Bed/mesh")
            or "/Pipe_" in path
        ):
            continue
        candidates.append(prim)

    obstacles = []
    for prim in candidates:
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if extent.IsEmpty():
            continue
        minimum = np.asarray(extent.GetMin(), dtype=np.float64)
        maximum = np.asarray(extent.GetMax(), dtype=np.float64)
        horizontal_excess = np.maximum(
            np.maximum(minimum[:2] - reference[:2], reference[:2] - maximum[:2]),
            0.0,
        )
        if float(np.linalg.norm(horizontal_excess)) > horizontal_radius_m:
            continue
        if np.any(maximum - minimum <= 1e-6):
            continue
        obstacles.append(
            robot_kinematics.BoxObstacle(
                path=str(prim.GetPath()),
                minimum_m=tuple(float(value) for value in minimum),
                maximum_m=tuple(float(value) for value in maximum),
            )
        )
    return tuple(obstacles)


def _author_greenhouse_wall_colliders(stage, obstacles) -> tuple[str, ...]:
    """Author invisible static wall proxies so PhysX agrees with the planner."""
    import numpy as np

    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root = "/World/BenchmarkSafety/GreenhouseColliders"
    UsdGeom.Scope.Define(stage, Sdf.Path(root))
    paths = []
    wall_obstacles = [
        obstacle for obstacle in obstacles if "/Env/Wall_" in obstacle.path
    ]
    for index, obstacle in enumerate(wall_obstacles):
        minimum = np.asarray(obstacle.minimum_m, dtype=np.float64)
        maximum = np.asarray(obstacle.maximum_m, dtype=np.float64)
        path = f"{root}/Wall_{index:02d}"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (minimum + maximum))))
        xform.AddScaleOp().Set(Gf.Vec3f(*(maximum - minimum)))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().CreateAttribute(
            "tomato:sourcePrim",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(obstacle.path)
        paths.append(path)
    return tuple(paths)


def _configure_joint_group_drives(
    stage,
    group: str,
    maximum_forces,
    stiffness_values,
    damping_values,
) -> dict:
    """Apply effort-bounded position gains to a torso or head joint group."""
    from pxr import UsdPhysics

    joints = []
    for index, (maximum_force, stiffness, damping) in enumerate(
        zip(maximum_forces, stiffness_values, damping_values, strict=True)
    ):
        joint = stage.GetPrimAtPath(f"/World/RBY1/joints/{group}_{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"missing {group} drive {index}")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
        drive.CreateMaxForceAttr(float(maximum_force))
        joints.append(
            {
                "joint": str(joint.GetPath()),
                "maximum_force_nm": float(maximum_force),
                "stiffness_nm_per_rad": float(stiffness),
                "damping_nm_s_per_rad": float(damping),
            }
        )
    return {
        "group": group,
        "type": "force",
        "control": "fixed-target hardware-effort-bounded position tracking",
        "joints": joints,
    }

def _configure_arm_drives(stage, side: str) -> dict:
    """Apply stable hardware-bounded position gains to one RB-Y1 arm."""
    from pxr import UsdPhysics

    joints = []
    gains = zip(
        _RBY1_ARM_EFFORT_LIMITS_NM,
        _RBY1_ARM_STIFFNESS_NM_RAD,
        _RBY1_ARM_DAMPING_NM_S_RAD,
        strict=True,
    )
    for index, (maximum_force, stiffness, damping) in enumerate(gains):
        joint = stage.GetPrimAtPath(f"/World/RBY1/joints/{side}_arm_{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"missing {side} arm drive {index}")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(maximum_force)
        joints.append(
            {
                "joint": str(joint.GetPath()),
                "maximum_force_nm": maximum_force,
                "stiffness_nm_per_rad": stiffness,
                "damping_nm_s_per_rad": damping,
            }
        )
    return {
        "side": side,
        "type": "force",
        "control": "critically-damped hardware-bounded position tracking",
        "joints": joints,
    }


def _set_joint_group_initial_state(stage, group: str, degrees) -> None:
    """Seed drive and PhysX state before initialization, avoiding startup sweeps."""
    from pxr import PhysxSchema
    from pxr import UsdPhysics

    prefix = f"{group}_arm" if group in {"left", "right"} else group
    for index, value in enumerate(degrees):
        joint = stage.GetPrimAtPath(f"/World/RBY1/joints/{prefix}_{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"missing {group} drive {index}")
        drive.CreateTargetPositionAttr(float(value))
        drive.CreateTargetVelocityAttr(0.0)
        state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.CreatePositionAttr().Set(float(value))
        state.CreateVelocityAttr().Set(0.0)

def _set_joint_group_drive_targets(stage, group: str, degrees) -> None:
    from pxr import UsdPhysics

    prefix = f"{group}_arm" if group in {"left", "right"} else group
    for index, value in enumerate(degrees):
        joint = stage.GetPrimAtPath(f"/World/RBY1/joints/{prefix}_{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"missing {group} drive {index}")
        drive.CreateTargetPositionAttr(float(value))
        drive.CreateTargetVelocityAttr(0.0)


def _set_arm_drive_targets(stage, side: str, degrees) -> None:
    _set_joint_group_drive_targets(stage, side, degrees)


def _required_probe_payload_clearance(
    side: str,
    payload_minimum: dict,
    grasp_geometry: dict,
    protected_clearance_m: float,
) -> float:
    """Return the target-semantic payload clearance for one measured pair."""
    nearest_obstacle = str(payload_minimum.get("nearest_obstacle", ""))
    if "/Main_Cultivation_Zone/" in nearest_obstacle:
        return max(
            float(protected_clearance_m),
            _MINIMUM_GREENHOUSE_CLEARANCE_M,
        )
    orphan_colliders = set(
        grasp_geometry.get(
            "orphan_colliders",
            grasp_geometry.get("colliders", (grasp_geometry.get("collider"),)),
        )
    )
    if (
        side == "right"
        and str(payload_minimum.get("component", "")).startswith("knife_support_")
        and payload_minimum.get("nearest_obstacle") in orphan_colliders
    ):
        return min(
            float(protected_clearance_m),
            _RIGHT_TARGET_ORPHAN_SUPPORT_PLANNING_CLEARANCE_M,
        )
    return float(protected_clearance_m)


def _probe_unsafe_contacts(contact_summary: dict, blade_geometry: dict, grasp_geometry: dict) -> list[dict]:
    """Reject robot contacts except floor support and intended tool contacts."""
    cut_colliders = set(blade_geometry["colliders"])
    grasp_colliders = set(
        grasp_geometry.get(
            "orphan_colliders",
            grasp_geometry.get("colliders", (grasp_geometry["collider"],)),
        )
    )
    unsafe = []
    for pair in contact_summary["pairs"]:
        if float(pair.get("maximum_impulse_ns", 0.0)) <= 1e-12:
            continue
        paths = (pair["collider0"], pair["collider1"])
        robot = next((path for path in paths if path.startswith("/World/RBY1/")), None)
        if robot is None:
            continue
        other = paths[1] if paths[0] == robot else paths[0]
        if "GroundPlane/CollisionPlane" in other:
            continue
        if other in grasp_colliders and "/ee_finger_l" in robot:
            continue
        if other in cut_colliders and robot.endswith(("/BladeCollision", "/ArcCollision")):
            continue
        if other.startswith(("/World/InteractiveVines/", "/World/NeighbourSafety/")):
            unsafe.append(dict(pair))
            continue
        if other.startswith(
            (
                "/World/Main_Cultivation_Zone",
                "/World/Main_Cultivation_Zone_01",
                "/World/BenchmarkSafety/GreenhouseColliders",
            )
        ):
            unsafe.append(dict(pair))
    return unsafe


def _bimanual_probe(
    stage,
    context,
    runtime: VineRuntime,
    args,
    report: dict,
    blade_monitor: BladeContactMonitor,
    grasp_manager: LeftGraspManager,
    contact_diagnostics: RobotContactDiagnostics,
    robot_hardware,
    robot_kinematics,
    greenhouse_structure_obstacles=(),
) -> dict:
    """Execute staged, force-limited dual-arm deleafing acceptance motion."""
    import numpy as np
    from pxr import Usd
    from pxr import UsdGeom

    if blade_monitor is None or grasp_manager is None or contact_diagnostics is None:
        return {"succeeded": False, "error": "robot cut/grasp/contact monitors are required"}
    if args.motion_steps < 30 or args.drop_steps < 60:
        return {"succeeded": False, "error": "probe step counts are too small for bounded motion"}

    blade_geometry = blade_monitor.target_geometry
    grasp_geometry = grasp_manager.target_geometry
    if blade_geometry is None or grasp_geometry is None:
        return {"succeeded": False, "error": "active cut or grasp target geometry is missing"}

    report["bimanual_probe_progress"] = {
        "stage": "probe_initializing",
        "arm": "both",
        "steps_completed": 0,
    }
    _emit(report, args.report)
    model = robot_kinematics.Rby1Kinematics()
    base_gf = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/RBY1/base")
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Gf uses row-vector matrices; the pure kinematics module uses columns.
    base_matrix = np.asarray(base_gf, dtype=np.float64).T
    robot_forward = base_matrix[:3, 0].copy()
    robot_forward /= np.linalg.norm(robot_forward)
    robot_lateral = base_matrix[:3, 1].copy()
    robot_lateral /= np.linalg.norm(robot_lateral)
    aisle_direction = -robot_forward
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    def task_offset(lateral_m: float, aisle_m: float, vertical_m: float):
        return (
            lateral_m * robot_lateral
            + aisle_m * aisle_direction
            + vertical_m * world_up
        )

    vine_obstacle_cache = _VineCapsuleObstacleCache(stage, robot_kinematics)
    greenhouse_structure_obstacles = tuple(greenhouse_structure_obstacles)

    def vine_obstacles():
        return vine_obstacle_cache.snapshot()

    left_camera_local_centre_m, left_camera_radius_m = (
        robot_hardware.wrist_d405_body_sphere(side="left")
    )
    left_camera_box_centre, left_camera_box_rotation, left_camera_box_half_extents = (
        robot_hardware.wrist_d405_body_box(side="left")
    )
    left_tool_boxes = [
        (
            "wrist_d405",
            left_camera_box_centre,
            left_camera_box_rotation,
            left_camera_box_half_extents,
        ),
        (
            "wrist_camera_bracket",
            *robot_hardware.wrist_camera_bracket_box(side="left"),
        ),
    ]
    gripper_geometry = grasp_manager.summary.get("gripper_geometry") or {}
    for name, geometry in (gripper_geometry.get("fingers") or {}).items():
        minimum = np.asarray(geometry["collider_min_ee_m"], dtype=np.float64)
        maximum = np.asarray(geometry["collider_max_ee_m"], dtype=np.float64)
        left_tool_boxes.append(
            (
                name,
                0.5 * (minimum + maximum),
                np.eye(3, dtype=np.float64),
                0.5 * (maximum - minimum),
            )
        )

    right_camera_box_centre, right_camera_box_rotation, right_camera_box_half_extents = (
        robot_hardware.wrist_d405_body_box(side="right")
    )
    bracket_box_centre, bracket_box_rotation, bracket_box_half_extents = (
        robot_hardware.wrist_camera_bracket_box(side="right")
    )
    knife_blade_centre, knife_blade_half_extents = robot_hardware.knife_blade_box()
    right_tool_boxes = [
        (
            "wrist_d405",
            right_camera_box_centre,
            right_camera_box_rotation,
            right_camera_box_half_extents,
        ),
        (
            "wrist_camera_bracket",
            bracket_box_centre,
            bracket_box_rotation,
            bracket_box_half_extents,
        ),
        (
            "knife_blade",
            robot_hardware.KNIFE_TRANSLATION_M
            + robot_hardware.KNIFE_ROTATION @ knife_blade_centre,
            robot_hardware.KNIFE_ROTATION,
            knife_blade_half_extents,
        ),
    ]
    right_tool_boxes.extend(
        (
            f"knife_support_{index:02d}",
            robot_hardware.KNIFE_TRANSLATION_M
            + robot_hardware.KNIFE_ROTATION @ support_centre,
            robot_hardware.KNIFE_ROTATION,
            support_half_extents,
        )
        for index, (support_centre, support_half_extents) in enumerate(
            robot_hardware.knife_support_boxes()
        )
    )

    def left_camera_clearance(solution):
        ee_matrix = model.forward(
            "left",
            solution.joint_degrees,
            base_matrix,
        )
        centre = (
            ee_matrix
            @ np.append(
                np.asarray(left_camera_local_centre_m, dtype=np.float64),
                1.0,
            )
        )[:3]
        clearances = (
            robot_kinematics.sphere_capsule_clearance(
                centre,
                left_camera_radius_m,
                vine_obstacles(),
            ),
            robot_kinematics.sphere_box_clearance(
                centre,
                left_camera_radius_m,
                greenhouse_structure_obstacles,
            ),
        )
        return min(clearances, key=lambda result: result.clearance_m)

    def left_payload_clearance(solution, excluded_colliders=()):
        excluded = set(excluded_colliders)
        obstacles = vine_obstacles()
        records = []
        joint_degrees = getattr(solution, "joint_degrees", solution)
        ee_matrix = model.forward(
            "left",
            joint_degrees,
            base_matrix,
        )
        for component, local_centre, local_rotation, half_extents in left_tool_boxes:
            component_obstacles = (
                tuple(
                    obstacle
                    for obstacle in obstacles
                    if obstacle.path not in excluded
                )
                if component.startswith("ee_finger_")
                else obstacles
            )
            world_centre = (
                ee_matrix @ np.append(np.asarray(local_centre, dtype=np.float64), 1.0)
            )[:3]
            world_rotation = ee_matrix[:3, :3] @ np.asarray(
                local_rotation,
                dtype=np.float64,
            )
            clearances = (
                robot_kinematics.oriented_box_capsule_clearance(
                    world_centre,
                    world_rotation,
                    half_extents,
                    component_obstacles,
                ),
                robot_kinematics.oriented_box_box_clearance(
                    world_centre,
                    world_rotation,
                    half_extents,
                    greenhouse_structure_obstacles,
                ),
            )
            clearance = min(clearances, key=lambda result: result.clearance_m)
            records.append(
                {
                    "component": component,
                    "clearance_m": clearance.clearance_m,
                    "nearest_obstacle": clearance.nearest_obstacle,
                }
            )
        arm_vine_clearance = model.arm_obstacle_clearance(
            "left",
            joint_degrees,
            base_matrix,
            obstacles,
        )
        records.append(
            {
                "component": "left_arm_capsules",
                "clearance_m": arm_vine_clearance.clearance_m,
                "nearest_obstacle": arm_vine_clearance.nearest_obstacle,
            }
        )
        arm_structure_clearance = model.arm_structure_clearance(
            "left",
            joint_degrees,
            base_matrix,
            greenhouse_structure_obstacles,
        )
        records.append(
            {
                "component": "left_arm_structure",
                "clearance_m": arm_structure_clearance.clearance_m,
                "nearest_obstacle": arm_structure_clearance.nearest_obstacle,
            }
        )
        return min(records, key=lambda item: item["clearance_m"]), records

    def left_payload_trajectory_clearance(
        target_degrees, excluded_colliders=(), samples: int = 31
    ):
        """Sample full wrist/finger clearance along a joint-space chord."""
        start = current["left"]
        target = np.asarray(target_degrees, dtype=np.float64)
        best = {
            "clearance_m": float("inf"),
            "sample": 0,
            "component": None,
            "nearest_obstacle": None,
        }
        for index, fraction in enumerate(np.linspace(0.0, 1.0, samples), start=0):
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            commanded = start + smooth * (target - start)
            minimum, _ = left_payload_clearance(
                commanded, excluded_colliders=excluded_colliders
            )
            if minimum["clearance_m"] < best["clearance_m"]:
                best = {
                    **minimum,
                    "sample": index,
                    "fraction": float(fraction),
                }
        return best


    def right_payload_clearance(solution, excluded_colliders=()):
        excluded = set(excluded_colliders)
        obstacles = tuple(obstacle for obstacle in vine_obstacles() if obstacle.path not in excluded)
        records = []
        joint_degrees = getattr(solution, "joint_degrees", solution)
        ee_matrix = model.forward("right", joint_degrees, base_matrix)
        for component, local_centre, local_rotation, half_extents in right_tool_boxes:
            world_centre = (
                ee_matrix @ np.append(np.asarray(local_centre, dtype=np.float64), 1.0)
            )[:3]
            world_rotation = ee_matrix[:3, :3] @ np.asarray(local_rotation, dtype=np.float64)
            clearances = (
                robot_kinematics.oriented_box_capsule_clearance(
                    world_centre,
                    world_rotation,
                    half_extents,
                    obstacles,
                ),
                robot_kinematics.oriented_box_box_clearance(
                    world_centre,
                    world_rotation,
                    half_extents,
                    greenhouse_structure_obstacles,
                ),
            )
            clearance = min(clearances, key=lambda result: result.clearance_m)
            records.append(
                {
                    "component": component,
                    "clearance_m": clearance.clearance_m,
                    "nearest_obstacle": clearance.nearest_obstacle,
                }
            )
        arm_vine_clearance = model.arm_obstacle_clearance(
            "right",
            joint_degrees,
            base_matrix,
            tuple(vine_obstacles()),
        )
        records.append(
            {
                "component": "right_arm_capsules",
                "clearance_m": arm_vine_clearance.clearance_m,
                "nearest_obstacle": arm_vine_clearance.nearest_obstacle,
            }
        )
        arm_structure_clearance = model.arm_structure_clearance(
            "right",
            joint_degrees,
            base_matrix,
            greenhouse_structure_obstacles,
        )
        records.append(
            {
                "component": "right_arm_structure",
                "clearance_m": arm_structure_clearance.clearance_m,
                "nearest_obstacle": arm_structure_clearance.nearest_obstacle,
            }
        )
        return min(records, key=lambda item: item["clearance_m"]), records

    def right_payload_trajectory_clearance(
        target_degrees, excluded_colliders=(), samples: int = 31
    ):
        """Sample the complete right payload along a joint-space chord."""
        start = current["right"]
        target = np.asarray(target_degrees, dtype=np.float64)
        best = {
            "clearance_m": float("inf"),
            "sample": 0,
            "component": None,
            "nearest_obstacle": None,
        }
        for index, fraction in enumerate(np.linspace(0.0, 1.0, samples), start=0):
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            commanded = start + smooth * (target - start)
            minimum, _ = right_payload_clearance(
                commanded, excluded_colliders=excluded_colliders
            )
            if minimum["clearance_m"] < best["clearance_m"]:
                best = {**minimum, "sample": index, "fraction": float(fraction)}
        return best


    def inter_arm_trajectory_clearance(side, target_degrees, samples: int = 31):
        start = current[side]
        target = np.asarray(target_degrees, dtype=np.float64)
        other_side = "right" if side == "left" else "left"
        best = {
            "clearance_m": float("inf"),
            "nearest_obstacle": None,
            "sample": 0,
            "fraction": 0.0,
        }
        for index, fraction in enumerate(np.linspace(0.0, 1.0, samples), start=0):
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            commanded = start + smooth * (target - start)
            pair = {
                side: commanded,
                other_side: current[other_side],
            }
            clearance = model.inter_arm_clearance(
                pair["left"], pair["right"], base_matrix
            )
            if clearance.clearance_m < best["clearance_m"]:
                best = {
                    "clearance_m": clearance.clearance_m,
                    "nearest_obstacle": clearance.nearest_obstacle,
                    "sample": index,
                    "fraction": float(fraction),
                }
        return best


    stages = []
    preferred_cut_direction = (
        np.cos(np.radians(15.0)) * robot_forward
        + np.sin(np.radians(15.0)) * world_up
    )
    initial_cut_path = blade_monitor.target_path_geometry(_RIGHT_CUT_STUB_M)
    if initial_cut_path is None:
        return {"succeeded": False, "error": "initial physical cut path is missing"}
    initial_knife_rotation = robot_hardware.cut_aligned_knife_rotation(
        initial_cut_path["axis"],
        preferred_cut_direction,
    )
    blade_local_centre_m, blade_half_extents_m = robot_hardware.knife_blade_box()
    target_cut_colliders = set(blade_geometry["colliders"])
    blade_obstacles = tuple(
        obstacle
        for obstacle in vine_obstacles()
        if obstacle.path not in target_cut_colliders
    )
    wing_candidates = (
        np.asarray(_RIGHT_EDGE_WING_M, dtype=np.float64),
        np.asarray(
            [-_RIGHT_EDGE_WING_M[0], _RIGHT_EDGE_WING_M[1], _RIGHT_EDGE_WING_M[2]],
            dtype=np.float64,
        ),
    )
    wing_records = []
    for wing in wing_candidates:
        root = initial_cut_path["point_m"] - initial_knife_rotation @ wing
        blade_centre = root + initial_knife_rotation @ blade_local_centre_m
        clearances = (
            robot_kinematics.oriented_box_capsule_clearance(
                blade_centre,
                initial_knife_rotation,
                blade_half_extents_m,
                blade_obstacles,
            ),
            robot_kinematics.oriented_box_box_clearance(
                blade_centre,
                initial_knife_rotation,
                blade_half_extents_m,
                greenhouse_structure_obstacles,
            ),
        )
        clearance = min(clearances, key=lambda result: result.clearance_m)
        wing_records.append(
            {
                "edge_wing_local_m": wing.tolist(),
                "blade_centre_m": blade_centre.tolist(),
                "clearance_m": clearance.clearance_m,
                "nearest_obstacle": clearance.nearest_obstacle,
            }
        )
    selected_wing_index = max(
        range(len(wing_records)),
        key=lambda index: wing_records[index]["clearance_m"],
    )
    wing_selection_mode = "maximum_initial_blade_clearance"
    if args.bimanual_probe == "full":
        selected_wing_index = max(
            range(len(wing_candidates)),
            key=lambda index: wing_candidates[index][0],
        )
        wing_selection_mode = "bimanual_support_away_from_neighbour"
    right_edge_wing_m = wing_candidates[selected_wing_index]
    stages.append(
        {
            "stage": "right_blade_wing_selection",
            "selected_edge_wing_local_m": right_edge_wing_m.tolist(),
            "candidates": wing_records,
            "selection_mode": wing_selection_mode,
            "excluded_target_colliders": sorted(target_cut_colliders),
        }
    )
    commanded_blade_local = np.append(
        robot_hardware.KNIFE_ROTATION
        @ right_edge_wing_m,
        1.0,
    )
    previous_commanded_blade_point = None
    applied_cuts = []
    left_counterpull_start = None
    left_counterpull_target = None
    left_hold_capacity_n = None
    left_hold_local_point = None
    minimum_left_hold_capacity_n = None
    current = {
        "left": np.asarray(_LEFT_READY_DEGREES, dtype=np.float64),
        "right": np.asarray(_RIGHT_SAFE_DEGREES, dtype=np.float64),
    }
    minimum_inter_arm_clearance = {
        "clearance_m": float("inf"),
        "nearest_pair": None,
        "phase": "initial",
    }

    def require_inter_arm_clearance(left_degrees, right_degrees, phase: str):
        clearance = model.inter_arm_clearance(
            left_degrees,
            right_degrees,
            base_matrix,
        )
        if clearance.clearance_m < minimum_inter_arm_clearance["clearance_m"]:
            minimum_inter_arm_clearance.update(
                clearance_m=clearance.clearance_m,
                nearest_pair=clearance.nearest_obstacle,
                phase=phase,
            )
        report["minimum_inter_arm_clearance"] = dict(minimum_inter_arm_clearance)
        if clearance.clearance_m < _MINIMUM_INTER_ARM_CLEARANCE_M:
            raise RuntimeError(
                f"inter-arm clearance {clearance.clearance_m * 1000.0:.1f} mm "
                f"below {_MINIMUM_INTER_ARM_CLEARANCE_M * 1000.0:.1f} mm in {phase}: "
                f"{clearance.nearest_obstacle}"
            )
        return clearance

    initial_inter_arm_clearance = require_inter_arm_clearance(
        current["left"], current["right"], "initial"
    )
    stages.append(
        {
            "stage": "initial_inter_arm_clearance",
            "minimum_required_m": _MINIMUM_INTER_ARM_CLEARANCE_M,
            "clearance_m": initial_inter_arm_clearance.clearance_m,
            "nearest_pair": initial_inter_arm_clearance.nearest_obstacle,
        }
    )

    stages.append(
        {
            "stage": "right_arm_drive_configuration",
            "configuration": _configure_arm_drives(stage, "right"),
            "settled_base_matrix": base_matrix.tolist(),
        }
    )
    stages.append(
        {
            "stage": "left_arm_drive_configuration",
            "configuration": _configure_arm_drives(stage, "left"),
        }
    )
    if args.bimanual_probe in {"left_approach", "full"}:
        selection_diagnostics = []
        selected_collider = None
        minimum_grasp_segment = int(
            report.get("robot_preposition", {}).get("minimum_segment", 0)
        )
        preplanned_collider = report.get("robot_preposition", {}).get(
            "selected_grasp_collider"
        )
        target_conditioned = (
            report.get("robot_preposition", {}).get("mode")
            == "target-conditioned"
        )
        preplanned_candidate = next(
            (
                candidate
                for candidate in grasp_manager.target_candidates
                if candidate["collider"] == preplanned_collider
                and int(candidate["segment"]) >= minimum_grasp_segment
            ),
            None,
        )
        selection_source = "live_multistart"
        if target_conditioned and preplanned_collider:
            # Target-conditioned placement already solved this exact physical
            # collider with the distal-segment floor, reach reserve, and D405
            # clearance before robot authoring. Repeating every multistart IK
            # here blocked the visible app for several minutes before its first
            # waypoint. Live IK is still solved at every approach waypoint.
            selected_collider = preplanned_collider
            selection_source = "target_conditioned_base_plan"
            selection_diagnostics.append({
                "collider": preplanned_collider,
                "body": (
                    None
                    if preplanned_candidate is None
                    else preplanned_candidate["body"]
                ),
                "segment": (
                    None
                    if preplanned_candidate is None
                    else preplanned_candidate["segment"]
                ),
                "role": (
                    None
                    if preplanned_candidate is None
                    else preplanned_candidate["role"]
                ),
                "centre_m": (
                    None
                    if preplanned_candidate is None
                    else preplanned_candidate["centre_m"].tolist()
                ),
                "axis": (
                    None
                    if preplanned_candidate is None
                    else preplanned_candidate["axis"].tolist()
                ),
                "preferred": (
                    None
                    if preplanned_candidate is None
                    else preplanned_candidate["preferred"]
                ),
                "solutions": [],
                "source": selection_source,
            })
        selection_seeds = (
            _LEFT_AISLE_CLEARANCE_WAYPOINTS_DEGREES[-1],
            _LEFT_APPROACH_SEEDS_DEGREES[0.0],
            *_LEFT_MULTISTART_SEEDS_DEGREES,
        )
        # Fixed-position regressions do not carry a target-conditioned plan,
        # so preserve the full live multistart search for that explicit mode.
        candidate_sequence = (
            reversed(grasp_manager.target_candidates)
            if selected_collider is None
            else ()
        )
        for candidate in candidate_sequence:
            if int(candidate["segment"]) < minimum_grasp_segment:
                continue
            solutions = []
            solution_records = []
            for seed in selection_seeds:
                solution = model.solve_position_axes(
                    "left",
                    local_point_m=_LEFT_JAW_CENTRE_M,
                    target_point_m=candidate["centre_m"],
                    seed_degrees=seed,
                    base_matrix=base_matrix,
                    pointing_axis=2,
                    pointing_direction=(0.0, 1.0, 0.0),
                    transverse_axis=0,
                    transverse_to=candidate["axis"],
                    position_scale_m=0.002,
                )
                clearance = left_camera_clearance(solution)
                solutions.append(solution)
                solution_records.append(
                    {
                        "seed_degrees": [float(value) for value in seed],
                        "solution": dataclasses.asdict(solution),
                        "camera_clearance_m": clearance.clearance_m,
                        "nearest_obstacle": clearance.nearest_obstacle,
                    }
                )
            selection_diagnostics.append(
                {
                    "collider": candidate["collider"],
                    "body": candidate["body"],
                    "segment": candidate["segment"],
                    "role": candidate["role"],
                    "centre_m": candidate["centre_m"].tolist(),
                    "axis": candidate["axis"].tolist(),
                    "preferred": candidate["preferred"],
                    "solutions": solution_records,
                }
            )
            if any(
                solution.succeeded
                and record["camera_clearance_m"]
                >= _MINIMUM_WRIST_CAMERA_CLEARANCE_M
                for solution, record in zip(solutions, solution_records, strict=True)
            ):
                selected_collider = candidate["collider"]
                break
        stages.append(
            {
                "stage": "left_grasp_candidate_selection",
                "selected_collider": selected_collider,
                "minimum_segment": minimum_grasp_segment,
                "selection_source": selection_source,
                "candidates": selection_diagnostics,
            }
        )
        report["bimanual_probe_progress"] = {
            "stage": "left_grasp_candidate_selection",
            "arm": "left",
            "steps_completed": 0,
            "selection_source": selection_source,
            "selected_collider": selected_collider,
        }
        _emit(report, args.report)
        if selected_collider is None:
            return {
                "mode": args.bimanual_probe,
                "stages": stages,
                "physical_cuts": applied_cuts,
                "unsafe_contacts": [],
                "blade_safety_clear": blade_monitor.safety_clear,
                "task": grasp_manager.summary,
                "succeeded": False,
                "error": (
                    "no selected-petiole grasp collider is reachable with "
                    "wrist-camera clearance"
                ),
            }
        grasp_manager.set_planned_grasp_collider(selected_collider)
        grasp_geometry = grasp_manager.target_geometry

    render_probe = not (args.headless or args.screenshot is not None)
    physics_dt_s = float(context.get_physics_dt())
    rendering_dt_s = float(context.get_rendering_dt())
    physics_steps_per_render = max(
        int(round(rendering_dt_s / physics_dt_s)),
        1,
    )
    probe_physics_step = 0
    stages.append(
        {
            "stage": "probe_step_timing",
            "physics_dt_s": physics_dt_s,
            "rendering_dt_s": rendering_dt_s,
            "physics_steps_per_render": physics_steps_per_render,
            "render_requested": render_probe,
        }
    )

    def tick() -> None:
        nonlocal probe_physics_step
        # SimulationContext.step(render=True) advances one *rendering* step.
        # At 60 Hz rendering and 240 Hz physics that silently performs four
        # physics substeps before the contact/grasp monitors run, inflating
        # measured force by roughly 4x and changing the compliant trajectory.
        # Advance exactly one physics step per control sample, then refresh the
        # UI without simulation at the requested render cadence.
        context.step(render=False)
        vine_obstacle_cache.invalidate()
        grasp_manager.process()
        applied_cuts.extend(
            _apply_blade_cut_decisions(
                context,
                blade_monitor,
                report,
                grasp_manager=grasp_manager,
            )
        )
        probe_physics_step += 1
        if render_probe and probe_physics_step % physics_steps_per_render == 0:
            context.render()

    def record_commanded_blade_motion(right_degrees=None) -> None:
        """Map right joint targets to the commanded knife-wing velocity."""
        nonlocal previous_commanded_blade_point
        if right_degrees is None:
            blade_monitor.set_commanded_edge_velocity(np.zeros(3))
            previous_commanded_blade_point = None
            return
        point = (
            model.forward("right", right_degrees, base_matrix)
            @ commanded_blade_local
        )[:3]
        velocity = (
            np.zeros(3, dtype=np.float64)
            if previous_commanded_blade_point is None
            else (point - previous_commanded_blade_point) * 240.0
        )
        previous_commanded_blade_point = point.copy()
        blade_monitor.set_commanded_edge_velocity(velocity)

    def pose_sample(side: str) -> list[float]:
        matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"/World/RBY1/ee_{side}")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return [float(value) for value in matrix.ExtractTranslation()]

    def physical_ee_pose(side: str) -> np.ndarray:
        matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"/World/RBY1/ee_{side}")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.asarray(matrix, dtype=np.float64).T

    def synchronize_measured_arm_state(side: str, commanded) -> dict:
        """Reconstruct the force-driven arm state from its physical EE pose."""
        physical_gf = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"/World/RBY1/ee_{side}")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        physical_pose = np.asarray(physical_gf, dtype=np.float64).T
        commanded_values = np.asarray(commanded, dtype=np.float64)
        measured = model.solve_pose(
            side,
            physical_pose,
            commanded_values,
            base_matrix,
        )
        record = dataclasses.asdict(measured)
        if not measured.succeeded:
            raise RuntimeError(
                f"could not reconstruct measured {side} arm state: "
                f"{measured.position_error_m * 1000.0:.2f} mm / "
                f"{np.degrees(measured.orientation_error_rad):.2f} deg"
            )
        measured_values = np.asarray(measured.joint_degrees, dtype=np.float64)
        joint_lag = commanded_values - measured_values
        record.update(
            commanded_degrees=commanded_values.tolist(),
            measured_degrees=measured_values.tolist(),
            joint_lag_degrees=joint_lag.tolist(),
            maximum_joint_lag_degrees=float(np.max(np.abs(joint_lag))),
        )
        current[side] = measured_values
        return record

    def move(
        side: str,
        target,
        name: str,
        *,
        steps: int | None = None,
        hold_steps: int = 0,
        settle_to_pose: bool = False,
    ) -> None:
        start = current[side].copy()
        target_values = np.asarray(target, dtype=np.float64)
        count = int(steps or args.motion_steps)
        contact_diagnostics.set_phase(name)
        samples = []
        movement_clearance_m = float("inf")
        movement_nearest_pair = None
        movement_payload_clearance_m = float("inf")
        movement_payload_required_m = _MINIMUM_WRIST_CAMERA_CLEARANCE_M
        movement_payload_nearest = None
        for index in range(1, count + 1):
            fraction = index / count
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            commanded = start + smooth * (target_values - start)
            paired = {
                side: commanded,
                "right" if side == "left" else "left": current[
                    "right" if side == "left" else "left"
                ],
            }
            clearance = require_inter_arm_clearance(
                paired["left"], paired["right"], f"{name}:{index}/{count}"
            )
            if clearance.clearance_m < movement_clearance_m:
                movement_clearance_m = clearance.clearance_m
                movement_nearest_pair = clearance.nearest_obstacle
            live_grasp = grasp_manager.target_geometry or grasp_geometry
            if side == "left":
                payload_clearance = left_payload_clearance
                excluded_payload_colliders = live_grasp.get(
                    "orphan_colliders", live_grasp.get("colliders", ())
                )
            else:
                live_cut = blade_monitor.target_geometry or blade_geometry
                payload_clearance = right_payload_clearance
                excluded_payload_colliders = live_cut.get("colliders", ())
            payload_minimum, _ = payload_clearance(
                commanded,
                excluded_colliders=excluded_payload_colliders,
            )
            payload_required_m = _required_probe_payload_clearance(
                side,
                payload_minimum,
                live_grasp,
                _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
            )
            if payload_minimum["clearance_m"] < movement_payload_clearance_m:
                movement_payload_clearance_m = payload_minimum["clearance_m"]
                movement_payload_required_m = payload_required_m
                movement_payload_nearest = {
                    "component": payload_minimum["component"],
                    "obstacle": payload_minimum["nearest_obstacle"],
                }
            if payload_minimum["clearance_m"] < payload_required_m:
                raise RuntimeError(
                    f"{side} payload clearance {payload_minimum['clearance_m'] * 1000.0:.1f} mm "
                    f"below {payload_required_m * 1000.0:.1f} mm in "
                    f"{name}:{index}/{count}: {payload_minimum['component']} vs "
                    f"{payload_minimum['nearest_obstacle']}"
                )

            _set_arm_drive_targets(stage, side, commanded)
            record_commanded_blade_motion(
                commanded if side == "right" else None
            )
            tick()
            if index in {1, count // 2, count}:
                samples.append({"step": index, "ee_position_m": pose_sample(side)})
        for _ in range(int(hold_steps)):
            _set_arm_drive_targets(stage, side, target_values)
            record_commanded_blade_motion(
                target_values if side == "right" else None
            )
            tick()
        convergence = None
        if settle_to_pose:
            expected_pose = model.forward(side, target_values, base_matrix)
            consecutive = 0
            settle_limit = max(4 * args.motion_steps, 240)
            settle_samples = []
            for settle_index in range(1, settle_limit + 1):
                _set_arm_drive_targets(stage, side, target_values)
                record_commanded_blade_motion(
                    target_values if side == "right" else None
                )
                tick()
                physical_pose = physical_ee_pose(side)
                position_error_m = float(
                    np.linalg.norm(
                        physical_pose[:3, 3] - expected_pose[:3, 3]
                    )
                )
                relative_rotation = (
                    expected_pose[:3, :3] @ physical_pose[:3, :3].T
                )
                orientation_error_rad = float(
                    np.arccos(
                        np.clip(
                            0.5 * (np.trace(relative_rotation) - 1.0),
                            -1.0,
                            1.0,
                        )
                    )
                )
                converged = bool(
                    position_error_m <= 0.002
                    and orientation_error_rad <= np.radians(2.0)
                )
                consecutive = consecutive + 1 if converged else 0
                if settle_index in {1, settle_limit} or consecutive == 1:
                    settle_samples.append(
                        {
                            "step": settle_index,
                            "position_error_m": position_error_m,
                            "orientation_error_rad": orientation_error_rad,
                            "consecutive_converged": consecutive,
                        }
                    )
                if consecutive >= 4:
                    break
            convergence = {
                "succeeded": consecutive >= 4,
                "steps": settle_index,
                "maximum_steps": settle_limit,
                "required_position_error_m": 0.002,
                "required_orientation_error_rad": float(np.radians(2.0)),
                "final_position_error_m": position_error_m,
                "final_orientation_error_rad": orientation_error_rad,
                "samples": settle_samples,
            }
            if not convergence["succeeded"]:
                raise RuntimeError(
                    f"{side} arm failed to physically converge in {name}: "
                    f"{position_error_m * 1000.0:.1f} mm / "
                    f"{np.degrees(orientation_error_rad):.1f} degrees"
                )
        if hold_steps:
            samples.append(
                {
                    "step": count + int(hold_steps),
                    "ee_position_m": pose_sample(side),
                    "endpoint_hold": True,
                }
            )
        measured_state = synchronize_measured_arm_state(side, target_values)
        if settle_to_pose:
            current[side] = target_values.copy()
            measured_state["planning_state"] = "converged_commanded_target"
        stages.append(
            {
                "stage": name,
                "arm": side,
                "steps": count,
                "hold_steps": int(hold_steps),
                "target_degrees": target_values.tolist(),
                "samples": samples,
                "blade_safety_clear": blade_monitor.safety_clear,
                "minimum_inter_arm_clearance_m": movement_clearance_m,
                "minimum_required_inter_arm_clearance_m": _MINIMUM_INTER_ARM_CLEARANCE_M,
                "nearest_inter_arm_pair": movement_nearest_pair,
                "payload_side": side,
                "minimum_payload_clearance_m": movement_payload_clearance_m,
                "minimum_required_payload_clearance_m": movement_payload_required_m,
                "protected_payload_clearance_m": _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
                "nearest_payload_pair": movement_payload_nearest,
                "measured_state": measured_state,
                "physical_convergence": convergence,
            }
        )
        report["bimanual_probe_progress"] = {
            "stage": name,
            "arm": side,
            "steps_completed": count + int(hold_steps),
        }
        _emit(report, args.report)

    def move_bimanual(
        right_target,
        left_target,
        name: str,
        *,
        steps: int,
        force_control: dict | None = None,
    ) -> None:
        starts = {side: current[side].copy() for side in ("right", "left")}
        targets = {
            "right": np.asarray(right_target, dtype=np.float64),
            "left": np.asarray(left_target, dtype=np.float64),
        }
        count = int(steps)
        contact_diagnostics.set_phase(name)
        samples = []
        command_fraction = 0.0
        maximum_observed_force_n = 0.0
        loaded_advance_steps = 0
        unload_steps = 0
        executed_steps = 0
        movement_clearance_m = float("inf")
        movement_nearest_pair = None
        movement_payload_clearance_m = {
            "left": float("inf"),
            "right": float("inf"),
        }
        movement_payload_required_m = {
            "left": _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
            "right": _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
        }
        movement_payload_nearest = {
            "left": None,
            "right": None,
        }
        for index in range(1, count + 1):
            executed_steps = index
            control_mode = "nominal"
            if force_control is None:
                fraction = index / count
            else:
                feedback = blade_monitor.active_cut_feedback or {}
                force_n = float(feedback.get("effective_force_n", 0.0))
                maximum_observed_force_n = max(
                    maximum_observed_force_n,
                    force_n,
                )
                increment = 1.0 / count
                if force_n > float(force_control["maximum_force_n"]):
                    command_fraction = max(
                        -1.0,
                        command_fraction
                        - float(force_control["unload_fraction"]) * increment,
                    )
                    unload_steps += 1
                    control_mode = "unload"
                elif force_n >= float(force_control["target_force_n"]):
                    command_fraction = min(
                        1.0,
                        command_fraction
                        + float(force_control["loaded_advance_fraction"])
                        * increment,
                    )
                    loaded_advance_steps += 1
                    control_mode = "loaded_advance"
                else:
                    command_fraction = min(1.0, command_fraction + increment)
                fraction = command_fraction
            smooth = (
                fraction
                if force_control is not None
                else fraction * fraction * (3.0 - 2.0 * fraction)
            )
            commanded_pair = {
                side: starts[side] + smooth * (targets[side] - starts[side])
                for side in ("right", "left")
            }
            clearance = require_inter_arm_clearance(
                commanded_pair["left"],
                commanded_pair["right"],
                f"{name}:{index}/{count}",
            )
            if clearance.clearance_m < movement_clearance_m:
                movement_clearance_m = clearance.clearance_m
                movement_nearest_pair = clearance.nearest_obstacle
            live_grasp = grasp_manager.target_geometry or grasp_geometry
            live_cut = blade_monitor.target_geometry or blade_geometry
            payload_specs = (
                (
                    "left",
                    left_payload_clearance,
                    live_grasp.get(
                        "orphan_colliders", live_grasp.get("colliders", ())
                    ),
                ),
                (
                    "right",
                    right_payload_clearance,
                    live_cut.get("colliders", ()),
                ),
            )
            for payload_side, payload_clearance, excluded_colliders in payload_specs:
                payload_minimum, _ = payload_clearance(
                    commanded_pair[payload_side],
                    excluded_colliders=excluded_colliders,
                )
                payload_required_m = _required_probe_payload_clearance(
                    payload_side,
                    payload_minimum,
                    live_grasp,
                    _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
                )
                if (
                    payload_minimum["clearance_m"]
                    < movement_payload_clearance_m[payload_side]
                ):
                    movement_payload_clearance_m[payload_side] = payload_minimum[
                        "clearance_m"
                    ]
                    movement_payload_required_m[payload_side] = payload_required_m
                    movement_payload_nearest[payload_side] = {
                        "component": payload_minimum["component"],
                        "obstacle": payload_minimum["nearest_obstacle"],
                    }
                if payload_minimum["clearance_m"] < payload_required_m:
                    raise RuntimeError(
                        f"{payload_side} payload clearance "
                        f"{payload_minimum['clearance_m'] * 1000.0:.1f} mm below "
                        f"{payload_required_m * 1000.0:.1f} mm in "
                        f"{name}:{index}/{count}: {payload_minimum['component']} vs "
                        f"{payload_minimum['nearest_obstacle']}"
                    )
            for side in ("right", "left"):
                _set_arm_drive_targets(stage, side, commanded_pair[side])
                if side == "right":
                    record_commanded_blade_motion(commanded_pair[side])
            tick()
            if index in {1, count // 2, count} or (
                force_control is not None and control_mode == "unload" and len(samples) < 8
            ):
                feedback = blade_monitor.active_cut_feedback or {}
                samples.append(
                    {
                        "step": index,
                        "command_fraction": fraction,
                        "force_control_mode": control_mode,
                        "blade_force_n": float(
                            feedback.get("effective_force_n", 0.0)
                        ),
                        "right_ee_position_m": pose_sample("right"),
                        "left_ee_position_m": pose_sample("left"),
                    }
                )
            if force_control is not None and applied_cuts:
                break
        if force_control is None:
            commanded_targets = targets
            command_fraction = 1.0
        else:
            smooth = command_fraction
            commanded_targets = {
                side: starts[side] + smooth * (targets[side] - starts[side])
                for side in ("right", "left")
            }
        measured_states = {
            side: synchronize_measured_arm_state(side, commanded_targets[side])
            for side in ("right", "left")
        }
        stages.append(
            {
                "stage": name,
                "arm": "both",
                "steps": executed_steps,
                "hold_steps": 0,
                "target_degrees": {
                    side: target.tolist()
                    for side, target in commanded_targets.items()
                },
                "requested_target_degrees": {
                    side: target.tolist() for side, target in targets.items()
                },
                "command_fraction": command_fraction,
                "force_control": (
                    {
                        **force_control,
                        "maximum_observed_force_n": maximum_observed_force_n,
                        "loaded_advance_steps": loaded_advance_steps,
                        "unload_steps": unload_steps,
                    }
                    if force_control is not None
                    else None
                ),
                "samples": samples,
                "blade_safety_clear": blade_monitor.safety_clear,
                "minimum_inter_arm_clearance_m": movement_clearance_m,
                "minimum_required_inter_arm_clearance_m": _MINIMUM_INTER_ARM_CLEARANCE_M,
                "nearest_inter_arm_pair": movement_nearest_pair,
                "minimum_payload_clearance_m": dict(
                    movement_payload_clearance_m
                ),
                "minimum_required_payload_clearance_m": dict(
                    movement_payload_required_m
                ),
                "protected_payload_clearance_m": _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
                "nearest_payload_pair": dict(movement_payload_nearest),
                "measured_state": measured_states,
            }
        )
        report["bimanual_probe_progress"] = {
            "stage": name,
            "arm": "both",
            "steps_completed": executed_steps,
        }
        _emit(report, args.report)

    def solve_left(point, seed, alternate_seed=None, diagnostics=None) -> object:
        primary = model.solve_position_axes(
            "left",
            local_point_m=(
                _LEFT_JAW_CENTRE_M
                if grasp_geometry.get("tool_local_point_m") is None
                else grasp_geometry["tool_local_point_m"]
            ),
            target_point_m=point,
            seed_degrees=seed,
            base_matrix=base_matrix,
            pointing_axis=2,
            pointing_direction=(0.0, 1.0, 0.0),
            transverse_axis=0,
            transverse_to=grasp_geometry["axis"],
            position_scale_m=0.005,
        )
        primary_camera = left_camera_clearance(primary)
        primary_payload_minimum, primary_payload_components = (
            left_payload_clearance(
                primary,
                excluded_colliders=grasp_geometry.get(
                    "orphan_colliders",
                    grasp_geometry["colliders"],
                ),
            )
        )
        primary_trajectory = left_payload_trajectory_clearance(
            primary.joint_degrees,
            excluded_colliders=grasp_geometry.get(
                "orphan_colliders",
                grasp_geometry["colliders"],
            ),
        )
        primary_inter_arm = inter_arm_trajectory_clearance(
            "left",
            primary.joint_degrees,
        )
        primary_margin_m = min(
            primary_camera.clearance_m,
            primary_payload_minimum["clearance_m"],
            primary_trajectory["clearance_m"],
            primary_inter_arm["clearance_m"],
        )
        if diagnostics is not None:
            diagnostics.append(
                {
                    "seed_degrees": [float(value) for value in seed],
                    "position_scale_m": 0.005,
                    "solution": dataclasses.asdict(primary),
                    "camera_clearance_m": primary_camera.clearance_m,
                    "camera_nearest_obstacle": primary_camera.nearest_obstacle,
                    "payload_minimum_clearance": primary_payload_minimum,
                    "payload_components": primary_payload_components,
                    "trajectory_minimum_clearance": primary_trajectory,
                    "inter_arm_trajectory_clearance": primary_inter_arm,
                    "warm_start": True,
                }
            )
        if (
            primary.succeeded
            and primary_margin_m >= _WARM_START_COMFORTABLE_CLEARANCE_M
        ):
            return primary

        seeds = [seed]
        if alternate_seed is not None:
            seeds.append(alternate_seed)
        seeds.extend(_LEFT_MULTISTART_SEEDS_DEGREES)
        candidate_specs = [
            (candidate_seed, position_scale_m)
            for position_scale_m in (0.005, 0.002)
            for candidate_seed in seeds
        ]
        candidates = [
            model.solve_position_axes(
                "left",
                local_point_m=(
                    _LEFT_JAW_CENTRE_M
                    if grasp_geometry.get("tool_local_point_m") is None
                    else grasp_geometry["tool_local_point_m"]
                ),
                target_point_m=point,
                seed_degrees=candidate_seed,
                base_matrix=base_matrix,
                pointing_axis=2,
                pointing_direction=(0.0, 1.0, 0.0),
                transverse_axis=0,
                transverse_to=grasp_geometry["axis"],
                position_scale_m=position_scale_m,
            )
            for candidate_seed, position_scale_m in candidate_specs
        ]
        camera_clearances = [left_camera_clearance(result) for result in candidates]
        payload_clearances = [
            left_payload_clearance(
                result,
                excluded_colliders=grasp_geometry.get(
                    "orphan_colliders", grasp_geometry["colliders"]
                ),
            )
            for result in candidates
        ]
        trajectory_clearances = [
            left_payload_trajectory_clearance(
                result.joint_degrees,
                excluded_colliders=grasp_geometry.get(
                    "orphan_colliders", grasp_geometry["colliders"]
                ),
            )
            for result in candidates
        ]
        inter_arm_clearances = [
            inter_arm_trajectory_clearance("left", result.joint_degrees)
            for result in candidates
        ]
        if diagnostics is not None:
            diagnostics.extend(
                {
                    "seed_degrees": [float(value) for value in candidate_seed],
                    "position_scale_m": position_scale_m,
                    "solution": dataclasses.asdict(result),
                    "camera_clearance_m": camera.clearance_m,
                    "camera_nearest_obstacle": camera.nearest_obstacle,
                    "payload_minimum_clearance": payload_minimum,
                    "payload_components": payload_components,
                    "trajectory_minimum_clearance": trajectory,
                    "inter_arm_trajectory_clearance": inter_arm,
                }
                for (candidate_seed, position_scale_m), result, camera, (
                    payload_minimum, payload_components
                ), trajectory, inter_arm in zip(
                    candidate_specs,
                    candidates,
                    camera_clearances,
                    payload_clearances,
                    trajectory_clearances,
                    inter_arm_clearances,
                    strict=True,
                )
            )
        clear = [
            (result, camera, payload_minimum, trajectory, inter_arm)
            for result, camera, (payload_minimum, _), trajectory, inter_arm in zip(
                candidates,
                camera_clearances,
                payload_clearances,
                trajectory_clearances,
                inter_arm_clearances,
                strict=True,
            )
            if result.succeeded
            and camera.clearance_m >= _MINIMUM_WRIST_CAMERA_CLEARANCE_M
            and payload_minimum["clearance_m"] >= _MINIMUM_WRIST_CAMERA_CLEARANCE_M
            and trajectory["clearance_m"] >= _MINIMUM_WRIST_CAMERA_CLEARANCE_M
            and inter_arm["clearance_m"] >= _MINIMUM_INTER_ARM_CLEARANCE_M
        ]
        if clear:
            return max(
                clear,
                key=lambda item: min(
                    item[1].clearance_m,
                    item[2]["clearance_m"],
                    item[3]["clearance_m"],
                    item[4]["clearance_m"],
                ),
            )[0]
        successful = [
            (result, camera, payload_minimum, trajectory, inter_arm)
            for result, camera, (payload_minimum, _), trajectory, inter_arm in zip(
                candidates,
                camera_clearances,
                payload_clearances,
                trajectory_clearances,
                inter_arm_clearances,
                strict=True,
            )
            if result.succeeded
        ]
        if successful:
            best, _, _, _, _ = max(
                successful,
                key=lambda item: min(
                    item[1].clearance_m,
                    item[2]["clearance_m"],
                    item[3]["clearance_m"],
                    item[4]["clearance_m"],
                ),
            )
            return dataclasses.replace(best, succeeded=False)
        return min(
            candidates,
            key=lambda result: (
                result.position_error_m,
                result.orientation_error_rad,
            ),
        )

    right_goal = {}
    right_cut_roll_degrees = None
    right_direction_candidates = []

    def knife_rotation_for_roll(axis, roll_degrees):
        base_rotation = robot_hardware.cut_aligned_knife_rotation(
            axis, preferred_cut_direction
        )
        support = base_rotation[:, 2]
        cut_direction = (
            base_rotation @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
        )
        angle = np.radians(roll_degrees)
        cut_direction = (
            cut_direction * np.cos(angle)
            + np.cross(support, cut_direction) * np.sin(angle)
        )
        knife_x = -cut_direction
        edge_axis = np.cross(support, knife_x)
        edge_axis /= np.linalg.norm(edge_axis)
        return np.column_stack((knife_x, edge_axis, support))
    def solve_right(
        side_m: float,
        seed,
        translation_correction=None,
        cut_geometry_override=None,
        planning_payload_clearance_m=_RIGHT_PLANNING_PAYLOAD_CLEARANCE_M,
        roll_candidates_override=None,
        wing_candidates_override=None,
        prefer_joint_continuity=False,
    ) -> object:
        nonlocal right_cut_roll_degrees, right_direction_candidates
        nonlocal left_hold_capacity_n, right_edge_wing_m
        nonlocal commanded_blade_local, previous_commanded_blade_point
        cut_geometry = (
            blade_monitor.target_path_geometry(_RIGHT_CUT_STUB_M)
            if cut_geometry_override is None
            else cut_geometry_override
        )
        if cut_geometry is None:
            raise RuntimeError("live articulated cut path disappeared")
        seed_degrees = np.asarray(seed, dtype=np.float64)

        def joint_continuity_key(solution):
            delta = (
                np.asarray(solution.joint_degrees, dtype=np.float64)
                - seed_degrees
            )
            return float(np.max(np.abs(delta))), float(np.linalg.norm(delta))

        def required_payload_clearance(payload_clearance):
            return _required_probe_payload_clearance(
                "right",
                payload_clearance,
                grasp_geometry,
                planning_payload_clearance_m,
            )

        # Stay inside the benchmark's 25 mm admissible physical stub zone and
        # follow the live bent centreline rather than extending Link-0 as if
        # the whole articulated petiole were straight.
        cut_point = cut_geometry["point_m"]
        correction = (
            np.zeros(3, dtype=np.float64)
            if translation_correction is None
            else np.asarray(translation_correction, dtype=np.float64)
        )
        selecting_direction = True
        base_roll_candidates = (
            0.0,
            5.0,
            -5.0,
            10.0,
            -10.0,
            15.0,
            -15.0,
            30.0,
            -30.0,
            45.0,
            -45.0,
            60.0,
            -60.0,
            75.0,
            -75.0,
            90.0,
            -90.0,
            105.0,
            -105.0,
            120.0,
            -120.0,
        )
        counterhold_required = bool(
            minimum_left_hold_capacity_n is not None
            and side_m >= -0.035
        )
        if roll_candidates_override is not None:
            roll_candidates = tuple(roll_candidates_override)
        elif right_cut_roll_degrees is None or counterhold_required:
            roll_candidates = base_roll_candidates
        else:
            roll_candidates = tuple(
                dict.fromkeys(
                    (
                        *base_roll_candidates,
                        *(
                            right_cut_roll_degrees + delta
                            for delta in (
                                0.0,
                                5.0,
                                -5.0,
                                10.0,
                                -10.0,
                                15.0,
                                -15.0,
                                30.0,
                                -30.0,
                                45.0,
                                -45.0,
                                60.0,
                                -60.0,
                            )
                            if -120.0
                            <= right_cut_roll_degrees + delta
                            <= 120.0
                        ),
                    )
                )
            )
        records = []
        waypoint_wing_candidates = (
            tuple(wing_candidates_override)
            if wing_candidates_override is not None
            else (
                right_edge_wing_m,
                *(
                    wing
                    for wing in wing_candidates
                    if not np.allclose(wing, right_edge_wing_m)
                ),
            )
        )
        roll_wing_candidates = (
            (roll_degrees, edge_wing_m)
            for roll_degrees in roll_candidates
            for edge_wing_m in waypoint_wing_candidates
        )
        for roll_degrees, edge_wing_m in roll_wing_candidates:
            knife_rotation = knife_rotation_for_roll(
                cut_geometry["axis"], roll_degrees
            )
            cut_direction = (
                knife_rotation @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
            )
            if float(np.dot(cut_direction, robot_forward)) < 0.20:
                continue
            edge = cut_point + cut_direction * side_m
            root = edge - knife_rotation @ edge_wing_m + correction
            desired = np.eye(4, dtype=np.float64)
            desired[:3, :3] = knife_rotation @ robot_hardware.KNIFE_ROTATION.T
            desired[:3, 3] = root
            def assess_solution(candidate):
                candidate_payload = {
                    "clearance_m": float("-inf"),
                    "component": None,
                    "nearest_obstacle": None,
                }
                candidate_arm_clearance_m = float("-inf")
                candidate_arm_nearest = None
                if candidate.succeeded:
                    candidate_payload = right_payload_trajectory_clearance(
                        candidate.joint_degrees,
                        excluded_colliders=target_cut_colliders,
                    )
                    candidate_arm_clearance_m = float("inf")
                    start = current["right"]
                    target = np.asarray(candidate.joint_degrees, dtype=np.float64)
                    for fraction in np.linspace(0.0, 1.0, 31):
                        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                        commanded = start + smooth * (target - start)
                        clearance = model.inter_arm_clearance(
                            current["left"], commanded, base_matrix
                        )
                        if clearance.clearance_m < candidate_arm_clearance_m:
                            candidate_arm_clearance_m = clearance.clearance_m
                            candidate_arm_nearest = clearance.nearest_obstacle
                return {
                    "solution": candidate,
                    "payload_clearance": candidate_payload,
                    "required_payload_clearance_m": (
                        required_payload_clearance(candidate_payload)
                    ),
                    "inter_arm_clearance_m": candidate_arm_clearance_m,
                    "inter_arm_nearest": candidate_arm_nearest,
                }

            primary_solution = model.solve_pose(
                "right",
                desired,
                seed,
                base_matrix,
            )
            primary_assessment = assess_solution(primary_solution)
            primary_comfortable = bool(
                primary_solution.succeeded
                and primary_assessment["payload_clearance"]["clearance_m"]
                >= max(
                    primary_assessment["required_payload_clearance_m"],
                    _WARM_START_COMFORTABLE_CLEARANCE_M,
                )
                and primary_assessment["inter_arm_clearance_m"]
                >= _WARM_START_COMFORTABLE_CLEARANCE_M
                and model.arm_joint_limit_margin_degrees(
                    "right",
                    primary_solution.joint_degrees,
                )
                >= 5.0
            )
            solution_assessments = [primary_assessment]
            if not primary_comfortable:
                solution_assessments.extend(
                    assess_solution(
                        model.solve_pose(
                            "right",
                            desired,
                            candidate_seed,
                            base_matrix,
                        )
                    )
                    for candidate_seed in _RIGHT_MULTISTART_SEEDS_DEGREES
                )
            geometrically_clear = [
                assessment
                for assessment in solution_assessments
                if assessment["solution"].succeeded
                and assessment["payload_clearance"]["clearance_m"]
                >= assessment["required_payload_clearance_m"]
                and assessment["inter_arm_clearance_m"]
                >= _MINIMUM_INTER_ARM_CLEARANCE_M
            ]
            if geometrically_clear:
                comfortable_assessments = [
                    assessment
                    for assessment in geometrically_clear
                    if min(
                        assessment["payload_clearance"]["clearance_m"],
                        assessment["inter_arm_clearance_m"],
                    )
                    >= _RIGHT_COMFORTABLE_PAYLOAD_CLEARANCE_M
                ]
                assessment_pool = comfortable_assessments or geometrically_clear
                if prefer_joint_continuity:
                    selected_assessment = min(
                        assessment_pool,
                        key=lambda assessment: joint_continuity_key(
                            assessment["solution"]
                        ),
                    )
                elif counterhold_required:
                    selected_assessment = max(
                        assessment_pool,
                        key=lambda assessment: (
                            model.arm_joint_limit_margin_degrees(
                                "right", assessment["solution"].joint_degrees
                            ),
                            min(
                                assessment["payload_clearance"]["clearance_m"],
                                assessment["inter_arm_clearance_m"],
                            ),
                        ),
                    )
                else:
                    selected_assessment = max(
                        assessment_pool,
                        key=lambda assessment: min(
                            assessment["payload_clearance"]["clearance_m"],
                            assessment["inter_arm_clearance_m"],
                        ),
                    )
            else:
                selected_assessment = min(
                    solution_assessments,
                    key=lambda assessment: (
                        not assessment["solution"].succeeded,
                        assessment["solution"].position_error_m,
                        assessment["solution"].orientation_error_rad,
                    ),
                )
            result = selected_assessment["solution"]
            payload_clearance = selected_assessment["payload_clearance"]
            arm_clearance_m = selected_assessment["inter_arm_clearance_m"]
            arm_nearest = selected_assessment["inter_arm_nearest"]
            hold_capacity = None
            if (
                result.succeeded
                and left_hold_local_point is not None
                and minimum_left_hold_capacity_n is not None
            ):
                hold_capacity = model.point_force_capacity(
                    "left",
                    current["left"],
                    base_matrix,
                    left_hold_local_point,
                    cut_direction,
                    required_cut_force_n,
                    _RBY1_ARM_EFFORT_LIMITS_NM,
                )

            eligible = bool(
                result.succeeded
                and payload_clearance["clearance_m"]
                >= selected_assessment["required_payload_clearance_m"]
                and arm_clearance_m >= _MINIMUM_INTER_ARM_CLEARANCE_M
                and (
                    not counterhold_required
                    or (
                        hold_capacity is not None
                        and hold_capacity.force_capacity_n
                        >= minimum_left_hold_capacity_n
                    )
                )
            )
            records.append(
                {
                    "roll_degrees": float(roll_degrees),
                    "edge_wing_m": edge_wing_m.copy(),
                    "knife_rotation": knife_rotation,
                    "cut_direction": cut_direction,
                    "edge": edge,
                    "desired": desired,
                    "solution": result,
                    "payload_clearance": payload_clearance,
                    "required_payload_clearance_m": selected_assessment[
                        "required_payload_clearance_m"
                    ],
                    "inter_arm_clearance_m": arm_clearance_m,
                    "inter_arm_nearest": arm_nearest,
                    "left_hold_capacity": (
                        None if hold_capacity is None else dataclasses.asdict(hold_capacity)
                    ),
                    "minimum_left_hold_capacity_n": minimum_left_hold_capacity_n,
                    "counterhold_clear": bool(hold_capacity is None or hold_capacity.force_capacity_n >= minimum_left_hold_capacity_n),
                    "counterhold_required": counterhold_required,
                    "eligible": eligible,
                }
            )
        if not records:
            raise RuntimeError("no knife direction faces into the crop row")

        eligible_records = [record for record in records if record["eligible"]]
        if eligible_records:
            if prefer_joint_continuity:
                selected = min(
                    eligible_records,
                    key=lambda record: joint_continuity_key(record["solution"]),
                )
            elif (
                minimum_left_hold_capacity_n is not None
                and not counterhold_required
            ):
                force_ready_records = [
                    record
                    for record in eligible_records
                    if (
                        record["left_hold_capacity"] is not None
                        and record["left_hold_capacity"]["force_capacity_n"]
                        >= minimum_left_hold_capacity_n
                    )
                ]
                selection_pool = force_ready_records or eligible_records
                # A capacity above the required physical counterhold is enough;
                # surplus force must not outrank collision margin. The old
                # lexicographic force-first rule selected an 8 mm blade-wing
                # corridor instead of a 13 mm force-capable corridor at the
                # -50 mm waypoint. Millimetric vine motion could then leave no
                # continuation at -42 mm even though the safer wing remained
                # reachable.
                selected = max(
                    selection_pool,
                    key=lambda record: (
                        min(
                            record["payload_clearance"]["clearance_m"],
                            record["inter_arm_clearance_m"],
                        ),
                        (
                            float("-inf")
                            if record["left_hold_capacity"] is None
                            else record["left_hold_capacity"]["force_capacity_n"]
                        ),
                    ),
                )
            else:
                comfortable_records = [
                    record
                    for record in eligible_records
                    if min(
                        record["payload_clearance"]["clearance_m"],
                        record["inter_arm_clearance_m"],
                    )
                    >= _RIGHT_COMFORTABLE_PAYLOAD_CLEARANCE_M
                ]
                selection_pool = comfortable_records or eligible_records
                if counterhold_required and comfortable_records:
                    # Loaded cutting motion needs kinematic reserve as well as
                    # collision clearance. At the final -15 mm waypoint the
                    # former clearance-only ranking changed from the 75-degree
                    # roll (29.2 degrees of joint reserve) to 60 degrees (20.9
                    # degrees). Joint 1 then saturated at +1 degree after two
                    # 5 mm sweep segments. Keep a >=12 mm collision corridor,
                    # then maximize distance from every authored joint limit.
                    selected = max(
                        selection_pool,
                        key=lambda record: (
                            model.arm_joint_limit_margin_degrees(
                                "right", record["solution"].joint_degrees
                            ),
                            min(
                                record["payload_clearance"]["clearance_m"],
                                record["inter_arm_clearance_m"],
                            ),
                        ),
                    )
                else:
                    selected = max(
                        selection_pool,
                        key=lambda record: min(
                            record["payload_clearance"]["clearance_m"],
                            record["inter_arm_clearance_m"],
                        ),
                    )
        else:
            selected = min(
                records,
                key=lambda record: (
                    not record["solution"].succeeded,
                    record["solution"].position_error_m,
                    record["solution"].orientation_error_rad,
                ),
            )
        if selecting_direction and selected["eligible"]:
            right_cut_roll_degrees = selected["roll_degrees"]
            selected_wing = selected["edge_wing_m"]
            if not np.allclose(selected_wing, right_edge_wing_m):
                right_edge_wing_m = selected_wing.copy()
                commanded_blade_local = np.append(
                    robot_hardware.KNIFE_ROTATION @ right_edge_wing_m,
                    1.0,
                )
                previous_commanded_blade_point = None
        selected_hold_capacity = selected["left_hold_capacity"]
        left_hold_capacity_n = (
            None
            if selected_hold_capacity is None
            else selected_hold_capacity["force_capacity_n"]
        )
        if records:
            right_direction_candidates = [
                {
                    "roll_degrees": record["roll_degrees"],
                    "edge_wing_local_m": record["edge_wing_m"].tolist(),
                    "cut_direction": record["cut_direction"].tolist(),
                    "solution": dataclasses.asdict(record["solution"]),
                    "payload_trajectory_clearance": (
                        record["payload_clearance"]
                        if record["solution"].succeeded
                        else None
                    ),
                    "required_payload_clearance_m": record[
                        "required_payload_clearance_m"
                    ],
                    "minimum_inter_arm_clearance_m": (
                        record["inter_arm_clearance_m"]
                        if record["solution"].succeeded
                        else None
                    ),
                    "nearest_inter_arm_pair": record["inter_arm_nearest"],
                    "left_hold_capacity": record["left_hold_capacity"],
                    "minimum_left_hold_capacity_n": record["minimum_left_hold_capacity_n"],
                    "counterhold_clear": record["counterhold_clear"],
                    "eligible": record["eligible"],
                }
                for record in records
            ]

        result = selected["solution"]
        right_goal.clear()
        right_goal.update(
            desired_pose=selected["desired"].tolist(),
            cut_point_m=cut_point.tolist(),
            cut_axis=cut_geometry["axis"].tolist(),
            cut_direction=selected["cut_direction"].tolist(),
            edge_point_m=selected["edge"].tolist(),
            selected_edge_wing_local_m=selected["edge_wing_m"].tolist(),
            selected_roll_degrees=selected["roll_degrees"],
            selected_left_hold_capacity=selected["left_hold_capacity"],
            minimum_left_hold_capacity_n=minimum_left_hold_capacity_n,
            planning_payload_clearance_m=planning_payload_clearance_m,
            counterhold_clear=selected["counterhold_clear"],
            direction_candidates=right_direction_candidates,
        )
        if not selected["eligible"] and result.succeeded:
            return dataclasses.replace(result, succeeded=False)
        return result

    try:
        if args.bimanual_probe in {"left_approach", "full"}:
            preposition = report.get("robot_preposition", {})
            preplanned_solution = preposition.get("solution") or {}
            preplanned_goal_degrees = preplanned_solution.get("joint_degrees")
            preplanned_start_degrees = preposition.get("approach_start_degrees")
            preplanned_waypoints = tuple(
                preposition.get("approach_waypoints_degrees") or ()
            )
            use_preplanned_route = bool(
                target_conditioned
                and preplanned_goal_degrees is not None
                and preplanned_start_degrees is not None
            )
            if use_preplanned_route:
                planned_start = np.asarray(
                    preplanned_start_degrees, dtype=np.float64
                )
                if planned_start.shape != (7,) or any(
                    np.asarray(waypoint).shape != (7,)
                    for waypoint in preplanned_waypoints
                ):
                    raise RuntimeError(
                        "target-conditioned left approach route is malformed"
                    )
                start_error_degrees = float(
                    np.max(np.abs(current["left"] - planned_start))
                )
                if start_error_degrees > 1e-6:
                    raise RuntimeError(
                        "left arm is not at the planned approach start: "
                        f"{start_error_degrees:.3f} degree maximum error"
                    )
                stages.append(
                    {
                        "stage": "left_clearance_route",
                        "mode": "target_conditioned_joint_space_route",
                        "approach_route_index": preposition.get(
                            "approach_route_index"
                        ),
                        "waypoint_count": len(preplanned_waypoints),
                        "planned_start_degrees": planned_start.tolist(),
                        "maximum_start_error_degrees": start_error_degrees,
                        "planned_static_clearance_m": {
                            "arm_vine": preposition.get(
                                "trajectory_arm_clearance_m"
                            ),
                            "d405_vine": preposition.get(
                                "trajectory_camera_clearance_m"
                            ),
                            "full_payload_vine": preposition.get(
                                "trajectory_payload_clearance_m"
                            ),
                            "inter_arm": preposition.get(
                                "trajectory_inter_arm_clearance_m"
                            ),
                        },
                    }
                )
                for index, waypoint in enumerate(preplanned_waypoints):
                    move(
                        "left",
                        waypoint,
                        f"left_planned_route_{index}",
                        steps=max(args.motion_steps, 180),
                        settle_to_pose=True,
                    )
            else:
                lateral_base_offset_m = float(preposition.get("lateral_m", 0.0))
                if abs(lateral_base_offset_m) < 0.05:
                    for index, waypoint in enumerate(
                        _LEFT_AISLE_CLEARANCE_WAYPOINTS_DEGREES
                    ):
                        move("left", waypoint, f"left_clearance_{index}")
                else:
                    stages.append(
                        {
                            "stage": "left_clearance_route",
                            "mode": "lateral_base_direct_aisle_retreat",
                            "lateral_base_offset_m": lateral_base_offset_m,
                        }
                    )

                # Preserve the high-clearance wrist orientation while moving away
                # from the crop. Reorienting during the descent otherwise swings a
                # finger through upper foliage even though both endpoints are clear.
                grasp_geometry = grasp_manager.target_geometry
                if grasp_geometry is None:
                    raise RuntimeError(
                        "live grasp geometry disappeared before aisle retreat"
                    )
                retreat_target = model.forward(
                    "left", current["left"], base_matrix
                ).copy()
                retreat_target[:3, 3] += task_offset(0.0, 0.05, 0.0)
                retreat_solution = model.solve_pose(
                    "left", retreat_target, current["left"], base_matrix
                )
                retreat_clearance = left_payload_trajectory_clearance(
                    retreat_solution.joint_degrees,
                    excluded_colliders=grasp_geometry.get(
                        "orphan_colliders", grasp_geometry["colliders"]
                    ),
                )
                stages.append(
                    {
                        "stage": "left_aisle_retreat_ik",
                        "target_pose": retreat_target.tolist(),
                        "solution": dataclasses.asdict(retreat_solution),
                        "trajectory_minimum_clearance": retreat_clearance,
                    }
                )
                retreat_admissible = bool(
                    retreat_solution.succeeded
                    and retreat_clearance["clearance_m"]
                    >= _MINIMUM_WRIST_CAMERA_CLEARANCE_M
                )
                if retreat_admissible:
                    move(
                        "left",
                        retreat_solution.joint_degrees,
                        "left_aisle_retreat",
                    )
                elif abs(lateral_base_offset_m) >= 0.05:
                    stages[-1]["skipped"] = (
                        "unreachable_or_obstructed_outer_retreat"
                    )
                elif not retreat_solution.succeeded:
                    raise RuntimeError("left aisle-retreat IK failed")
                else:
                    raise RuntimeError(
                        "left aisle-retreat payload trajectory is obstructed"
                    )
            left_solutions = []
            reached_offsets = []
            # Enter along the petiole axis in short Cartesian increments. A
            # direct chord from the aisle-clearance posture to 100 mm is
            # endpoint-clear but sweeps the wrist payload through upper foliage
            # in the lower-gutter greenhouse.
            offsets = (
                (0.0,)
                if use_preplanned_route
                else (0.55, 0.40, 0.30, 0.20, 0.10, 0.06, 0.04, 0.02)
            )
            if args.bimanual_probe == "full" and not use_preplanned_route:
                offsets += (0.01, 0.0)
            seed = current["left"]
            for offset in offsets:
                # The compliant vine continues to sway during the multi-second
                # approach. Reacquire the live grasp body before every solve so
                # the final fingers close around its current pose instead of
                # pulling it back to a stale pre-motion target.
                grasp_geometry = grasp_manager.target_geometry
                if grasp_geometry is None:
                    raise RuntimeError("live grasp geometry disappeared during approach")
                goal = grasp_geometry["centre_m"] + task_offset(0.0, offset, 0.0)
                candidate_diagnostics = []
                solution = solve_left(
                    goal,
                    seed,
                    (
                        preplanned_goal_degrees
                        if use_preplanned_route
                        else _LEFT_APPROACH_SEEDS_DEGREES.get(offset)
                    ),
                    candidate_diagnostics,
                )
                left_solutions.append(dataclasses.asdict(solution))
                stages.append(
                    {
                        "stage": f"left_ik_{offset:.3f}",
                        "target_point_m": goal.tolist(),
                        "target_axis": grasp_geometry["axis"].tolist(),
                        "candidate_solutions": candidate_diagnostics,
                        "solution": dataclasses.asdict(solution),
                    }
                )
                if not solution.succeeded:
                    stages[-1]["skipped"] = "unreachable_or_obstructed_outer_waypoint"
                    continue
                move(
                    "left",
                    solution.joint_degrees,
                    f"left_approach_{offset:.3f}",
                    hold_steps=args.motion_steps,
                    settle_to_pose=use_preplanned_route,
                )
                seed = current["left"].copy()
                reached_offsets.append(offset)
            if not reached_offsets or not np.isclose(reached_offsets[-1], offsets[-1]):
                raise RuntimeError(
                    "left IK failed to reach required final aisle offset "
                    f"{offsets[-1]:.3f} m"
                )
            stages.append({"stage": "left_ik", "solutions": left_solutions})

            if args.bimanual_probe == "left_approach":
                unsafe = _probe_unsafe_contacts(
                    contact_diagnostics.summary, blade_geometry, grasp_geometry
                )
                return {
                    "mode": args.bimanual_probe,
                    "stages": stages,
                    "unsafe_contacts": unsafe,
                    "minimum_inter_arm_clearance": dict(minimum_inter_arm_clearance),
                    "succeeded": bool(not unsafe and blade_monitor.safety_clear),
                }

            contact_diagnostics.set_phase("left_grasp_close")
            grasp_manager.request_close()
            # The imported left-finger drives settle asymmetrically under the
            # greenhouse pose. Give both physical fingers a full second at the
            # default 240 Hz; the grasp gate still requires opposed contact and
            # measured force, so elapsed time alone cannot establish a grasp.
            grasp_close_steps = 2 * args.motion_steps
            for _ in range(grasp_close_steps):
                tick()
            stages.append(
                {
                    "stage": "left_grasp_close",
                    "steps": grasp_close_steps,
                    "task_phase": grasp_manager.task_phase,
                    "summary": grasp_manager.summary,
                }
            )
            if grasp_manager.task_phase != "grasped":
                raise RuntimeError("left two-finger grasp was not established")

            # Pull the grasped petiole into the aisle before the knife enters
            # the loaded cut zone. The old implementation assigned identical
            # counterpull start/target joints, so no tension was ever applied.
            grasp_geometry = grasp_manager.target_geometry
            if grasp_geometry is None:
                raise RuntimeError("live grasp geometry disappeared before counterhold")
            pull_pose = model.forward("left", current["left"], base_matrix).copy()
            pull_direction = np.asarray(
                grasp_geometry["axis"], dtype=np.float64
            )
            pull_direction /= np.linalg.norm(pull_direction)
            if float(np.dot(pull_direction, aisle_direction)) < 0.0:
                pull_direction *= -1.0
            pull_pose[:3, 3] += _LEFT_PRETENSION_PULL_M * pull_direction
            pull_candidates = [
                model.solve_pose("left", pull_pose, candidate_seed, base_matrix)
                for candidate_seed in (
                    current["left"],
                    _LEFT_APPROACH_SEEDS_DEGREES[0.0],
                    *_LEFT_MULTISTART_SEEDS_DEGREES,
                )
            ]
            pull_assessments = []
            for candidate in pull_candidates:
                payload_clearance = left_payload_trajectory_clearance(
                    candidate.joint_degrees,
                    excluded_colliders=grasp_geometry.get(
                        "orphan_colliders", grasp_geometry["colliders"]
                    ),
                )
                arm_clearance = inter_arm_trajectory_clearance(
                    "left", candidate.joint_degrees
                )
                pull_assessments.append(
                    {
                        "solution": candidate,
                        "payload_clearance": payload_clearance,
                        "inter_arm_clearance": arm_clearance,
                    }
                )
            clear_pull_assessments = [
                assessment
                for assessment in pull_assessments
                if (
                    assessment["solution"].succeeded
                    or (
                        assessment["solution"].position_error_m <= 0.001
                        and assessment["solution"].orientation_error_rad
                        <= np.radians(
                            _LEFT_PRETENSION_MAX_ORIENTATION_ERROR_DEGREES
                        )
                    )
                )
                and assessment["payload_clearance"]["clearance_m"]
                >= _MINIMUM_WRIST_CAMERA_CLEARANCE_M
                and assessment["inter_arm_clearance"]["clearance_m"]
                >= _MINIMUM_INTER_ARM_CLEARANCE_M
            ]
            stages.append(
                {
                    "stage": "left_pretension_ik",
                    "pull_distance_m": _LEFT_PRETENSION_PULL_M,
                    "pull_direction": pull_direction.tolist(),
                    "maximum_position_error_m": 0.001,
                    "maximum_orientation_error_degrees": (
                        _LEFT_PRETENSION_MAX_ORIENTATION_ERROR_DEGREES
                    ),
                    "target_pose": pull_pose.tolist(),
                    "candidates": [
                        {
                            "solution": dataclasses.asdict(assessment["solution"]),
                            "payload_clearance": assessment["payload_clearance"],
                            "inter_arm_clearance": assessment["inter_arm_clearance"],
                        }
                        for assessment in pull_assessments
                    ],
                }
            )
            if not clear_pull_assessments:
                raise RuntimeError("left pre-tension pull IK is obstructed")
            selected_pull = max(
                clear_pull_assessments,
                key=lambda assessment: min(
                    assessment["payload_clearance"]["clearance_m"],
                    assessment["inter_arm_clearance"]["clearance_m"],
                ),
            )
            move(
                "left",
                selected_pull["solution"].joint_degrees,
                "left_pretension_pull",
            )
            if grasp_manager.task_phase != "grasped":
                raise RuntimeError("left grasp was lost during pre-tension pull")
            grasp_geometry = grasp_manager.target_geometry
            if grasp_geometry is None:
                raise RuntimeError("live grasp geometry disappeared after pre-tension")
            hold_cut_geometry = blade_monitor.target_path_geometry(
                _RIGHT_CUT_STUB_M
            )
            if hold_cut_geometry is None:
                raise RuntimeError("live cut geometry disappeared before counterhold")
            preferred_cut_direction = (
                np.cos(np.radians(15.0)) * robot_forward
                + np.sin(np.radians(15.0)) * world_up
            )
            hold_knife_rotation = robot_hardware.cut_aligned_knife_rotation(
                hold_cut_geometry["axis"],
                preferred_cut_direction,
            )
            hold_cut_direction = (
                hold_knife_rotation @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
            )
            feedback = blade_monitor.active_cut_feedback
            required_cut_force_n = float(
                args.cut_force
                if feedback is None
                else feedback["required_force_n"]
            )
            left_hold_local_point = (
                _LEFT_JAW_CENTRE_M
                if grasp_geometry.get("tool_local_point_m") is None
                else grasp_geometry["tool_local_point_m"]
            )
            hold_capacity = model.point_force_capacity(
                "left",
                current["left"],
                base_matrix,
                left_hold_local_point,
                hold_cut_direction,
                required_cut_force_n,
                _RBY1_ARM_EFFORT_LIMITS_NM,
            )
            baseline_hold_capacity_n = hold_capacity.force_capacity_n
            minimum_left_hold_capacity_n = (
                1.10
                * _LEFT_COUNTERHOLD_CUT_FORCE_SHARE
                * required_cut_force_n
            )
            stages.append(
                {
                    "stage": "left_static_counterhold",
                    "joint_degrees": current["left"].tolist(),
                    "grasp_body": grasp_geometry["body"],
                    "grasp_collider": grasp_geometry["collider"],
                    "grasp_anchor_m": grasp_geometry["centre_m"].tolist(),
                    "tool_local_anchor_m": np.asarray(left_hold_local_point).tolist(),
                    "opposed_cut_direction": hold_cut_direction.tolist(),
                    "required_cut_force_n": required_cut_force_n,
                    "counterhold_cut_force_share": (
                        _LEFT_COUNTERHOLD_CUT_FORCE_SHARE
                    ),
                    "minimum_capacity_n": minimum_left_hold_capacity_n,
                    "capacity": dataclasses.asdict(hold_capacity),
                    "direction_selection_deferred": bool(
                        baseline_hold_capacity_n < minimum_left_hold_capacity_n
                    ),
                }
            )
            left_counterpull_start = current["left"].copy()
            left_counterpull_target = current["left"].copy()

        if args.bimanual_probe in {"right_approach", "full"}:
            # Seed the right approach from the live cut line. The yawed support
            # path is contact-clear, so every waypoint can safely reacquire the
            # compliant target instead of following a stale frame while the
            # left arm tensions the petiole.
            blade_geometry = blade_monitor.target_geometry
            if blade_geometry is None:
                raise RuntimeError("live cut geometry disappeared before right approach")
            # The opposite aisle is bounded by the crop gutter and greenhouse
            # wall. Rank several same-orientation lift/lateral routes instead
            # of blindly moving +250 mm through the wall. A successful warm
            # branch is accepted only after the complete payload and arm chord
            # clears vines, gutter, pipes, wall, and the left arm.
            retreat_origin = model.forward(
                "right", current["right"], base_matrix
            ).copy()
            retreat_offsets = (
                (0.00, 0.25, 0.00),
                (0.00, 0.10, 0.12),
                (0.20, 0.05, 0.10),
                (-0.20, 0.05, 0.10),
                (0.25, 0.00, 0.12),
                (-0.25, 0.00, 0.12),
                (0.00, 0.00, 0.15),
            )
            retreat_records = []
            for offset_m in retreat_offsets:
                retreat_target = retreat_origin.copy()
                retreat_target[:3, 3] += task_offset(*offset_m)
                seeds = (current["right"], *_RIGHT_MULTISTART_SEEDS_DEGREES)
                for seed_index, retreat_seed in enumerate(seeds):
                    solution = model.solve_pose(
                        "right",
                        retreat_target,
                        retreat_seed,
                        base_matrix,
                    )
                    clearance = {
                        "clearance_m": float("-inf"),
                        "component": None,
                        "nearest_obstacle": None,
                    }
                    required_clearance_m = _MINIMUM_WRIST_CAMERA_CLEARANCE_M
                    if solution.succeeded:
                        clearance = right_payload_trajectory_clearance(
                            solution.joint_degrees,
                            excluded_colliders=blade_geometry["colliders"],
                        )
                        required_clearance_m = _required_probe_payload_clearance(
                            "right",
                            clearance,
                            grasp_geometry,
                            _MINIMUM_WRIST_CAMERA_CLEARANCE_M,
                        )
                    admissible = bool(
                        solution.succeeded
                        and clearance["clearance_m"] >= required_clearance_m
                    )
                    retreat_records.append(
                        {
                            "offset_m": offset_m,
                            "seed_index": seed_index,
                            "solution": solution,
                            "trajectory_minimum_clearance": clearance,
                            "required_clearance_m": required_clearance_m,
                            "admissible": admissible,
                        }
                    )
                    if admissible:
                        break
            admissible_retreats = [
                record for record in retreat_records if record["admissible"]
            ]
            if not admissible_retreats:
                raise RuntimeError(
                    "no greenhouse-clear right aisle-retreat route is reachable"
                )
            selected_retreat = max(
                admissible_retreats,
                key=lambda record: (
                    record["trajectory_minimum_clearance"]["clearance_m"],
                    -float(
                        np.linalg.norm(
                            np.asarray(
                                record["solution"].joint_degrees,
                                dtype=np.float64,
                            )
                            - current["right"]
                        )
                    ),
                ),
            )
            retreat_solution = selected_retreat["solution"]
            retreat_clearance = selected_retreat[
                "trajectory_minimum_clearance"
            ]
            stages.append(
                {
                    "stage": "right_aisle_retreat_ik",
                    "selected_offset_m": selected_retreat["offset_m"],
                    "solution": dataclasses.asdict(retreat_solution),
                    "trajectory_minimum_clearance": retreat_clearance,
                    "required_clearance_m": selected_retreat[
                        "required_clearance_m"
                    ],
                    "candidates": [
                        {
                            **{
                                key: value
                                for key, value in record.items()
                                if key != "solution"
                            },
                            "solution": dataclasses.asdict(record["solution"]),
                        }
                        for record in retreat_records
                    ],
                }
            )
            move("right", retreat_solution.joint_degrees, "right_aisle_retreat")

            right_solutions = []
            # Establish the rolled tool frame well outside the canopy, then
            # enter along the intended Cartesian cut line.
            # Keep the final approach increments at 5 mm.  A 10 mm joint-space
            # chord between otherwise clear endpoint poses can swing the knife
            # support into a distal target-branch link mid-trajectory even
            # though the intended Cartesian cut line remains unobstructed.
            right_offsets = (
                -0.100,
                -0.060,
                -0.050,
                -0.042,
                -0.035,
                -0.030,
                -0.025,
                -0.020,
                -0.015,
            )
            seed = current["right"]
            right_waypoints = {}
            translation_correction = np.zeros(3, dtype=np.float64)
            for side_m in right_offsets:
                live_cut_geometry = blade_monitor.target_geometry
                if live_cut_geometry is None:
                    raise RuntimeError(
                        f"live cut geometry disappeared before side {side_m:.3f} m"
                    )
                blade_geometry = live_cut_geometry
                tracking_error = float("inf")
                convergence_m = 0.005 if side_m <= -0.035 else 0.002
                hard_limit_m = 0.010 if side_m <= -0.035 else 0.005
                for servo_attempt in range(_RIGHT_SERVO_MAX_ATTEMPTS):
                    live_cut_geometry = blade_monitor.target_geometry
                    if live_cut_geometry is None:
                        raise RuntimeError(
                            f"live cut geometry disappeared during side {side_m:.3f} m"
                        )
                    blade_geometry = live_cut_geometry
                    proven_orientation = right_cut_roll_degrees is not None
                    preferred_roll_degrees = (
                        right_cut_roll_degrees
                        if proven_orientation
                        else _RIGHT_KNIFE_ROLL_DEGREES
                    )
                    optional_hard_bound_correction = bool(
                        servo_attempt > 0 and tracking_error <= hard_limit_m
                    )
                    solution = solve_right(
                        side_m,
                        seed,
                        translation_correction,
                        roll_candidates_override=(preferred_roll_degrees,),
                        wing_candidates_override=(right_edge_wing_m.copy(),),
                        prefer_joint_continuity=proven_orientation,
                    )
                    orientation_search = (
                        (
                            "proven_optional_hard_bound_correction"
                            if optional_hard_bound_correction
                            else "proven_orientation"
                        )
                        if proven_orientation
                        else "default_orientation"
                    )
                    if (
                        not solution.succeeded
                        and proven_orientation
                        and not optional_hard_bound_correction
                    ):
                        # Near the crop, millimetric support clearances can
                        # change materially between coarse roll samples. Rank
                        # the complete local neighborhood together by safety
                        # clearance; accepting the first eligible roll can
                        # latch a marginal orientation that has no continuation
                        # at the next waypoint.
                        nearby_rolls = tuple(
                            preferred_roll_degrees + roll_delta
                            for magnitude in range(1, 11)
                            for roll_delta in (float(magnitude), -float(magnitude))
                            if -120.0
                            <= preferred_roll_degrees + roll_delta
                            <= 120.0
                        )
                        nearby_solution = solve_right(
                            side_m,
                            seed,
                            translation_correction,
                            roll_candidates_override=nearby_rolls,
                            wing_candidates_override=(right_edge_wing_m.copy(),),
                        )
                        if nearby_solution.succeeded:
                            solution = nearby_solution
                            orientation_search = "proven_nearby_ranked"
                    if not solution.succeeded and not optional_hard_bound_correction:
                        solution = solve_right(
                            side_m,
                            seed,
                            translation_correction,
                        )
                        orientation_search = (
                            "proven_then_full_fallback"
                            if proven_orientation
                            else "default_then_full_fallback"
                        )
                    solution_record = dataclasses.asdict(solution)
                    solution_record.update(
                        {
                            "side_m": side_m,
                            "servo_attempt": servo_attempt,
                            "orientation_search": orientation_search,
                            "translation_correction_m": translation_correction.tolist(),
                        }
                    )
                    right_solutions.append(solution_record)
                    stages.append(
                        {
                            "stage": f"right_ik_{side_m:.3f}_{servo_attempt}",
                            "goal": dict(right_goal),
                            "solution": solution_record,
                        }
                    )
                    if not solution.succeeded:
                        if servo_attempt > 0 and tracking_error <= hard_limit_m:
                            # The previously executed physical pose already
                            # satisfies the hard Cartesian bound.  A tighter
                            # optional correction may have no collision-free
                            # continuation; retain the measured settled pose
                            # instead of invalidating it.
                            stages[-1]["accepted_previous_settled_pose"] = {
                                "tracking_error_m": tracking_error,
                                "hard_limit_m": hard_limit_m,
                            }
                            right_solutions[-1][
                                "accepted_previous_settled_pose"
                            ] = True
                            break
                        raise RuntimeError(
                            f"right IK failed at side {side_m:.3f} m, "
                            f"servo attempt {servo_attempt}"
                        )
                    right_waypoints[side_m] = solution.joint_degrees
                    phase = (
                        f"right_sweep_{side_m:.3f}"
                        if servo_attempt == 0
                        else f"right_servo_{side_m:.3f}_{servo_attempt}"
                    )
                    move(
                        "right",
                        solution.joint_degrees,
                        phase,
                        hold_steps=args.motion_steps,
                    )
                    seed = current["right"].copy()
                    live_target = blade_monitor.target_geometry
                    live_path = blade_monitor.target_path_geometry(_RIGHT_CUT_STUB_M)
                    if live_target is None or live_path is None:
                        raise RuntimeError(
                            f"live cut geometry disappeared after side {side_m:.3f} m"
                        )
                    tool = blade_monitor.tool_point_geometry(right_edge_wing_m)
                    planned_point = (
                        live_path["point_m"]
                        + tool["cut_direction"] * side_m
                    )
                    correction_error = planned_point - tool["point_m"]
                    tracking_error = float(np.linalg.norm(correction_error))
                    delta = tool["point_m"] - live_path["virtual_junction_m"]
                    axial = float(np.dot(delta, live_path["axis"]))
                    radial = delta - axial * live_path["axis"]
                    stages[-1]["cut_alignment"] = {
                        "live_target_centre_m": live_path["point_m"].tolist(),
                        "live_target_axis": live_path["axis"].tolist(),
                        "live_target_segment": live_path["segment"],
                        "live_target_collider": live_path["collider"],
                        "blade_wing_point_m": tool["point_m"].tolist(),
                        "planned_point_error_mm": tracking_error * 1000.0,
                        "live_goal_error_mm": tracking_error * 1000.0,
                        "live_axial_stub_mm": axial * 1000.0,
                        "live_radial_distance_mm": float(
                            np.linalg.norm(radial) * 1000.0
                        ),
                        "live_signed_side_mm": float(
                            np.dot(
                                tool["point_m"] - live_path["point_m"],
                                tool["cut_direction"],
                            )
                            * 1000.0
                        ),
                    }
                    if tracking_error <= convergence_m or applied_cuts:
                        break
                    correction_step = correction_error
                    if tracking_error > 0.010:
                        correction_step = correction_error * (0.010 / tracking_error)
                    translation_correction += correction_step
                if tracking_error > hard_limit_m and not applied_cuts:
                    raise RuntimeError(
                        f"right Cartesian servo remained {tracking_error * 1000.0:.1f} mm "
                        f"from sweep side {side_m:.3f} m"
                    )
            stages.append(
                {
                    "stage": "right_ik",
                    "cut_stub_m": _RIGHT_CUT_STUB_M,
                    "edge_wing_local_m": right_edge_wing_m.tolist(),
                    "knife_yaw_degrees": _RIGHT_KNIFE_YAW_DEGREES,
                    "knife_roll_degrees": _RIGHT_KNIFE_ROLL_DEGREES,
                    "orientation_mode": "live_segment_transverse_support_up",
                    "servo_max_attempts": _RIGHT_SERVO_MAX_ATTEMPTS,
                    "sweep_offsets_m": list(right_offsets),
                    "solutions": right_solutions,
                }
            )

            if args.bimanual_probe == "right_approach":
                unsafe = _probe_unsafe_contacts(
                    contact_diagnostics.summary, blade_geometry, grasp_geometry
                )
                return {
                    "mode": args.bimanual_probe,
                    "stages": stages,
                    "unsafe_contacts": unsafe,
                    "minimum_inter_arm_clearance": dict(minimum_inter_arm_clearance),
                    "succeeded": bool(not unsafe and blade_monitor.safety_clear),
                }

            # Track the compliant petiole online during the actual cut. A
            # single joint-space jump between accurate endpoints lets the
            # tensioned branch move between them without ever touching the
            # blade. These short consecutive segments preserve forward motion
            # while refreshing the physical target frame before every solve.
            sweep_solutions = []
            sweep_start_m = -0.015
            sweep_stop_m = 0.035
            # Use at least 48 integration/contact samples per 5 mm IK segment.
            # At the default this is a 2.0 s, 0.025 m/s nominal sweep: still
            # above the 0.01 m/s cut gate, but long enough for the physical
            # joint drives to reach the centreline before planning beyond it.
            # Measured edge force can slow or unload the command further before
            # the left arm's hardware-bounded counterhold is overpowered.
            sweep_segments = 10
            segment_steps = max(48, args.motion_steps // 4)
            # Reacquire the compliant target through pre-contact, then commit
            # to this physical cut plane. Continuing to chase the target while
            # loaded makes the blade push the petiole sideways; a latched plane
            # creates relative shear travel while the left arm holds tension.
            committed_cut_candidates = []
            pre_cut_roll_degrees = right_cut_roll_degrees
            committed_roll_candidates = tuple(
                dict.fromkeys(
                    pre_cut_roll_degrees + delta
                    for delta in (0.0, 15.0, -15.0, 30.0, -30.0)
                    if -120.0 <= pre_cut_roll_degrees + delta <= 120.0
                )
            )
            committed_wing_candidates = (right_edge_wing_m.copy(),)
            for committed_stub_m in _RIGHT_COMMITTED_CUT_STUB_CANDIDATES_M:
                candidate_geometry = blade_monitor.target_path_geometry(
                    committed_stub_m
                )
                if candidate_geometry is None:
                    continue
                candidate_solution = solve_right(
                    0.0,
                    current["right"],
                    translation_correction,
                    candidate_geometry,
                    _RIGHT_COMMITTED_CUT_PLANNING_CLEARANCE_M,
                    committed_roll_candidates,
                    committed_wing_candidates,
                )
                candidate_goal = dict(right_goal)
                selected_direction = next(
                    (
                        candidate
                        for candidate in candidate_goal["direction_candidates"]
                        if candidate["eligible"]
                        and candidate["roll_degrees"]
                        == candidate_goal["selected_roll_degrees"]
                        and np.allclose(
                            candidate["edge_wing_local_m"],
                            candidate_goal["selected_edge_wing_local_m"],
                        )
                    ),
                    None,
                )
                committed_cut_candidates.append(
                    {
                        "stub_m": committed_stub_m,
                        "geometry": candidate_geometry,
                        "solution": candidate_solution,
                        "goal": candidate_goal,
                        "payload_clearance_m": (
                            float("-inf")
                            if selected_direction is None
                            else selected_direction[
                                "payload_trajectory_clearance"
                            ]["clearance_m"]
                        ),
                    }
                )
            feasible_cut_candidates = [
                candidate
                for candidate in committed_cut_candidates
                if candidate["solution"].succeeded
            ]
            cut_plane_stage = {
                "stage": "committed_cut_plane_selection",
                "selected_stub_m": None,
                "selected_payload_clearance_m": None,
                "roll_candidates_degrees": list(committed_roll_candidates),
                "candidates": [
                    {
                        "stub_m": candidate["stub_m"],
                        "solution": dataclasses.asdict(candidate["solution"]),
                        "goal": candidate["goal"],
                        "payload_clearance_m": candidate["payload_clearance_m"],
                    }
                    for candidate in committed_cut_candidates
                ],
            }
            stages.append(cut_plane_stage)
            if not feasible_cut_candidates:
                raise RuntimeError("no neighbour-clear committed cut plane is reachable")
            selected_cut_candidate = max(
                feasible_cut_candidates,
                key=lambda candidate: candidate["payload_clearance_m"],
            )
            cut_plane_stage["selected_stub_m"] = selected_cut_candidate["stub_m"]
            cut_plane_stage["selected_payload_clearance_m"] = selected_cut_candidate[
                "payload_clearance_m"
            ]
            latched_cut_geometry = selected_cut_candidate["geometry"]
            right_goal.clear()
            right_goal.update(selected_cut_candidate["goal"])
            right_cut_roll_degrees = right_goal["selected_roll_degrees"]
            right_edge_wing_m = np.asarray(
                right_goal["selected_edge_wing_local_m"], dtype=np.float64
            )
            commanded_blade_local = np.append(
                robot_hardware.KNIFE_ROTATION @ right_edge_wing_m,
                1.0,
            )
            previous_commanded_blade_point = None
            latched_cut_roll_degrees = right_cut_roll_degrees
            latched_cut_wing_m = right_edge_wing_m.copy()
            selected_hold_capacity = right_goal["selected_left_hold_capacity"]
            left_hold_capacity_n = selected_hold_capacity["force_capacity_n"]
            cut_feedback = blade_monitor.active_cut_feedback
            required_cut_force_n = float(
                args.cut_force
                if cut_feedback is None
                else cut_feedback["required_force_n"]
            )
            maximum_control_force_n = min(
                1.04 * required_cut_force_n,
                0.93
                * float(
                    left_hold_capacity_n
                    if left_hold_capacity_n is not None
                    else 1.25 * required_cut_force_n
                )
                / _LEFT_COUNTERHOLD_CUT_FORCE_SHARE,
            )
            force_control = {
                "target_force_n": 1.005 * required_cut_force_n,
                "maximum_force_n": maximum_control_force_n,
                "loaded_advance_fraction": 0.25,
                "unload_fraction": 1.00,
            }
            if force_control["target_force_n"] >= force_control["maximum_force_n"]:
                raise RuntimeError("left counterhold has no usable force-control band")
            if not applied_cuts:
                for index in range(1, sweep_segments + 1):
                    side_m = sweep_start_m + (
                        (sweep_stop_m - sweep_start_m) * index / sweep_segments
                    )
                    live_cut_geometry = blade_monitor.target_geometry
                    if live_cut_geometry is None:
                        raise RuntimeError(
                            f"live cut geometry disappeared during sweep segment {index}"
                        )
                    blade_geometry = live_cut_geometry
                    solution = solve_right(
                        side_m,
                        current["right"],
                        translation_correction,
                        latched_cut_geometry,
                        _RIGHT_COMMITTED_CUT_PLANNING_CLEARANCE_M,
                        (latched_cut_roll_degrees,),
                        (latched_cut_wing_m,),
                        True,
                    )
                    record = dataclasses.asdict(solution)
                    record.update(
                        {
                            "segment": index,
                            "side_m": side_m,
                            "translation_correction_m": (
                                translation_correction.tolist()
                            ),
                        }
                    )
                    sweep_solutions.append(record)
                    stages.append(
                        {
                            "stage": f"right_cut_ik_{index:02d}",
                            "goal": dict(right_goal),
                            "solution": record,
                        }
                    )
                    if not solution.succeeded:
                        raise RuntimeError(
                            f"right live sweep IK failed at segment {index}"
                        )
                    phase = f"right_cut_sweep_{index:02d}"
                    if (
                        left_counterpull_start is not None
                        and left_counterpull_target is not None
                    ):
                        left_fraction = index / sweep_segments
                        left_target = left_counterpull_start + left_fraction * (
                            left_counterpull_target - left_counterpull_start
                        )
                        move_bimanual(
                            solution.joint_degrees,
                            left_target,
                            phase,
                            steps=segment_steps,
                            force_control=force_control,
                        )
                    else:
                        move(
                            "right",
                            solution.joint_degrees,
                            phase,
                            steps=segment_steps,
                        )
                    live_target = blade_monitor.target_geometry
                    live_path = blade_monitor.target_path_geometry(_RIGHT_CUT_STUB_M)
                    if live_target is None or live_path is None:
                        raise RuntimeError(
                            f"live cut geometry disappeared after sweep segment {index}"
                        )
                    tool = blade_monitor.tool_point_geometry(right_edge_wing_m)
                    live_goal = (
                        live_path["point_m"]
                        + tool["cut_direction"] * side_m
                    )
                    correction_error = live_goal - tool["point_m"]
                    tracking_error = float(np.linalg.norm(correction_error))
                    delta = tool["point_m"] - live_path["virtual_junction_m"]
                    axial = float(np.dot(delta, live_path["axis"]))
                    radial = delta - axial * live_path["axis"]
                    stages[-1]["cut_alignment"] = {
                        "live_target_centre_m": live_path["point_m"].tolist(),
                        "live_target_axis": live_path["axis"].tolist(),
                        "live_target_segment": live_path["segment"],
                        "live_target_collider": live_path["collider"],
                        "blade_wing_point_m": tool["point_m"].tolist(),
                        "live_goal_error_mm": tracking_error * 1000.0,
                        "live_axial_stub_mm": axial * 1000.0,
                        "live_radial_distance_mm": float(
                            np.linalg.norm(radial) * 1000.0
                        ),
                        "live_signed_side_mm": float(
                            np.dot(
                                tool["point_m"] - live_path["point_m"],
                                tool["cut_direction"],
                            )
                            * 1000.0
                        ),
                    }
                    if not blade_monitor.safety_clear:
                        raise RuntimeError(
                            f"protected contact during live sweep segment {index}"
                        )
                    if applied_cuts:
                        break
                    # Keep the committed stroke registered on radial/axial
                    # target motion, but never chase displacement along the
                    # cutting direction. The latter must remain real relative
                    # blade travel through tissue, not controller compensation.
                    transverse_error = correction_error - (
                        float(np.dot(correction_error, tool["cut_direction"]))
                        * tool["cut_direction"]
                    )
                    transverse_error_norm = float(
                        np.linalg.norm(transverse_error)
                    )
                    stages[-1]["cut_alignment"][
                        "transverse_correction_error_mm"
                    ] = transverse_error_norm * 1000.0
                    if transverse_error_norm > 0.002:
                        correction_step = transverse_error
                        if transverse_error_norm > 0.005:
                            correction_step = transverse_error * (
                                0.005 / transverse_error_norm
                            )
                        translation_correction += correction_step
            # Rigid collision geometry cannot be progressively split, so once
            # the edge is loaded at the safe physical plane, repeat a bounded
            # 5 mm slicing target instead of driving farther into the row.
            # Only forward, direction-valid, force-qualified travel contributes
            # fracture work; reverse unload strokes contribute nothing.
            fracture_cycle_solutions = []
            maximum_fracture_cycles = 20
            fracture_side_m = 0.015
            if not applied_cuts:
                for cycle in range(1, maximum_fracture_cycles + 1):
                    solution = solve_right(
                        fracture_side_m,
                        current["right"],
                        translation_correction,
                        latched_cut_geometry,
                        _RIGHT_COMMITTED_CUT_PLANNING_CLEARANCE_M,
                        (latched_cut_roll_degrees,),
                        (latched_cut_wing_m,),
                        True,
                    )
                    record = dataclasses.asdict(solution)
                    record.update(
                        {
                            "cycle": cycle,
                            "side_m": fracture_side_m,
                            "translation_correction_m": translation_correction.tolist(),
                        }
                    )
                    fracture_cycle_solutions.append(record)
                    stages.append(
                        {
                            "stage": f"right_fracture_ik_{cycle:02d}",
                            "goal": dict(right_goal),
                            "solution": record,
                        }
                    )
                    if not solution.succeeded:
                        raise RuntimeError(
                            f"right fracture-cycle IK failed at cycle {cycle}"
                        )
                    move_bimanual(
                        solution.joint_degrees,
                        current["left"],
                        f"right_cut_fracture_cycle_{cycle:02d}",
                        steps=segment_steps,
                        force_control=force_control,
                    )
                    if not blade_monitor.safety_clear:
                        raise RuntimeError(
                            f"protected contact during fracture cycle {cycle}"
                        )
                    if applied_cuts:
                        break
            stages.append(
                {
                    "stage": "right_live_cut_sweep",
                    "control_mode": (
                        "live_precontact_latched_cut_axis_transverse_tracking"
                    ),
                    "latched_target_point_m": latched_cut_geometry[
                        "point_m"
                    ].tolist(),
                    "latched_target_axis": latched_cut_geometry["axis"].tolist(),
                    "latched_target_segment": latched_cut_geometry["segment"],
                    "latched_target_collider": latched_cut_geometry["collider"],
                    "start_side_m": sweep_start_m,
                    "stop_side_m": sweep_stop_m,
                    "segments_requested": sweep_segments,
                    "segments_executed": len(sweep_solutions),
                    "steps_per_segment": segment_steps,
                    "force_control": force_control,
                    "solutions": sweep_solutions,
                    "maximum_fracture_cycles": maximum_fracture_cycles,
                    "fracture_cycles_executed": len(fracture_cycle_solutions),
                    "fracture_cycle_side_m": fracture_side_m,
                    "fracture_cycle_solutions": fracture_cycle_solutions,
                }
            )

            if not applied_cuts:
                raise RuntimeError("right leading-edge sweep did not physically sever the petiole")
            if grasp_manager.task_phase != "orphan_retained":
                raise RuntimeError(
                    f"physical cut did not enter orphan_retained: {grasp_manager.task_phase}"
                )
            # Do not reverse the broad knife plate through the severed plane.
            # The free side of the target differs between petioles, so plan
            # both lateral directions (and both lift orders) against the live
            # non-target capsules. This prevents a route that clears
            # SubStem_01 from becoming a hard-coded collision on SubStem_00.
            retract_origin = model.forward(
                "right",
                current["right"],
                base_matrix,
            )
            # Ten-millimetre Cartesian targets bound the chord error of the
            # intervening joint interpolation. A single 60 mm endpoint kept
            # the correct final pose but bowed the broad plate downward on the
            # way there and touched the lower neighbouring petiole.
            retract_obstacles = tuple(
                obstacle
                for obstacle in vine_obstacles()
                if obstacle.path not in target_cut_colliders
            )
            blade_centre_ee_m = (
                robot_hardware.KNIFE_TRANSLATION_M
                + robot_hardware.KNIFE_ROTATION @ blade_local_centre_m
            )
            knife_boxes = [
                ("blade", blade_centre_ee_m, blade_half_extents_m),
            ]
            knife_boxes.extend(
                (
                    f"support_arc_{index:02d}",
                    robot_hardware.KNIFE_TRANSLATION_M
                    + robot_hardware.KNIFE_ROTATION @ support_centre,
                    support_half_extents,
                )
                for index, (support_centre, support_half_extents) in enumerate(
                    robot_hardware.knife_support_boxes()
                )
            )
            route_specs = (
                ("negative_x_then_lift", -1.0, 0.040, 0.040, False),
                ("positive_x_then_lift", 1.0, 0.040, 0.040, False),
                ("lift_then_negative_x", -1.0, 0.040, 0.040, True),
                ("lift_then_positive_x", 1.0, 0.040, 0.040, True),
                ("positive_x_wide_then_lift", 1.0, 0.080, 0.040, False),
                ("positive_x_then_high_lift", 1.0, 0.040, 0.080, False),
                ("positive_x_wide_high", 1.0, 0.080, 0.080, False),
                ("high_lift_then_positive_x_wide", 1.0, 0.080, 0.080, True),
                ("positive_x_extra_wide_high", 1.0, 0.120, 0.080, False),
                ("positive_x_wide_extra_high", 1.0, 0.080, 0.120, False),
                ("extra_high_no_lateral", 0.0, 0.0, 0.120, True),
            )
            route_candidates = []
            for (
                route_name,
                x_sign,
                lateral_distance,
                lift_distance,
                lift_first,
            ) in route_specs:
                lateral_steps = round(lateral_distance / 0.010)
                lift_steps = round(lift_distance / 0.010)
                if lift_first:
                    lateral_lift_offsets = (
                        tuple(
                            np.asarray([0.0, 0.0, 0.010 * index])
                            for index in range(1, lift_steps + 1)
                        )
                        + tuple(
                            np.asarray(
                                [
                                    x_sign * 0.010 * index,
                                    0.0,
                                    lift_distance,
                                ]
                            )
                            for index in range(1, lateral_steps + 1)
                        )
                    )
                else:
                    lateral_lift_offsets = (
                        tuple(
                            np.asarray([x_sign * 0.010 * index, 0.0, 0.0])
                            for index in range(1, lateral_steps + 1)
                        )
                        + tuple(
                            np.asarray(
                                [
                                    x_sign * lateral_distance,
                                    0.0,
                                    0.010 * index,
                                ]
                            )
                            for index in range(1, lift_steps + 1)
                        )
                    )
                retract_offsets = lateral_lift_offsets + tuple(
                    np.asarray(
                        [
                            x_sign * lateral_distance,
                            0.020 * index,
                            lift_distance,
                        ]
                    )
                    for index in range(1, 14)
                )
                retract_offsets = tuple(
                    task_offset(*offset) for offset in retract_offsets
                )
                seed = current["right"].copy()
                solutions = []
                clearance_samples = []
                feasible = True
                failure_waypoint = None
                for waypoint_index, offset in enumerate(retract_offsets, start=1):
                    desired = retract_origin.copy()
                    desired[:3, 3] += offset
                    solution = model.solve_pose(
                        "right",
                        desired,
                        seed,
                        base_matrix,
                    )
                    solutions.append(solution)
                    if not solution.succeeded:
                        feasible = False
                        failure_waypoint = waypoint_index
                        break
                    target_degrees = np.asarray(
                        solution.joint_degrees,
                        dtype=np.float64,
                    )
                    for sample_index in range(1, 9):
                        fraction = sample_index / 8.0
                        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                        sampled_degrees = seed + smooth * (target_degrees - seed)
                        component_clearances = []
                        for component, centre_m, half_extents_m in knife_boxes:
                            clearance = robot_kinematics.tool_box_clearance(
                                model,
                                "right",
                                sampled_degrees,
                                base_matrix,
                                centre_m,
                                robot_hardware.KNIFE_ROTATION,
                                half_extents_m,
                                retract_obstacles,
                            )
                            component_clearances.append(
                                (component, clearance)
                            )
                        component, clearance = min(
                            component_clearances,
                            key=lambda item: item[1].clearance_m,
                        )
                        clearance_samples.append(
                            {
                                "component": component,
                                "clearance_m": clearance.clearance_m,
                                "nearest_obstacle": clearance.nearest_obstacle,
                                "waypoint": waypoint_index,
                                "sample": sample_index,
                            }
                        )
                    seed = target_degrees
                minimum_clearance = (
                    min(
                        clearance_samples,
                        key=lambda sample: sample["clearance_m"],
                    )
                    if clearance_samples
                    else {
                        "clearance_m": float("-inf"),
                        "nearest_obstacle": None,
                        "component": None,
                        "waypoint": None,
                        "sample": None,
                    }
                )
                route_candidates.append(
                    {
                        "name": route_name,
                        "x_sign": x_sign,
                        "lateral_distance_m": lateral_distance,
                        "lift_distance_m": lift_distance,
                        "offsets": retract_offsets,
                        "solutions": solutions,
                        "feasible": feasible,
                        "failure_waypoint": failure_waypoint,
                        "minimum_clearance": minimum_clearance,
                        "mean_clearance_m": (
                            float(
                                np.mean(
                                    [
                                        sample["clearance_m"]
                                        for sample in clearance_samples
                                    ]
                                )
                            )
                            if clearance_samples
                            else float("-inf")
                        ),
                    }
                )
            try:
                selected_retract_route = (
                    robot_kinematics.select_tool_clearance_route(
                        route_candidates
                    )
                )
            except ValueError as exc:
                raise RuntimeError(
                    "no post-cut right retract route has complete IK"
                ) from exc
            stages.append(
                {
                    "stage": "right_retract_route_selection",
                    "selected_route": selected_retract_route["name"],
                    "candidates": [
                        {
                            "name": candidate["name"],
                            "x_sign": candidate["x_sign"],
                            "lateral_distance_m": candidate[
                                "lateral_distance_m"
                            ],
                            "lift_distance_m": candidate["lift_distance_m"],
                            "offsets_m": [
                                offset.tolist() for offset in candidate["offsets"]
                            ],
                            "feasible": candidate["feasible"],
                            "failure_waypoint": candidate["failure_waypoint"],
                            "minimum_clearance": candidate["minimum_clearance"],
                            "mean_clearance_m": candidate["mean_clearance_m"],
                            "solutions": [
                                dataclasses.asdict(solution)
                                for solution in candidate["solutions"]
                            ],
                        }
                        for candidate in route_candidates
                    ],
                    "excluded_target_colliders": sorted(target_cut_colliders),
                }
            )
            retract_offsets = selected_retract_route["offsets"]
            retract_solutions = selected_retract_route["solutions"]
            for index, (offset, solution) in enumerate(
                zip(retract_offsets, retract_solutions, strict=True),
                start=1,
            ):
                desired = retract_origin.copy()
                desired[:3, 3] += offset
                stages.append(
                    {
                        "stage": f"right_retract_ik_{index:02d}",
                        "offset_m": offset.tolist(),
                        "solution": dataclasses.asdict(solution),
                    }
                )
                if not solution.succeeded:
                    raise RuntimeError(
                        f"right lifted retract IK failed at waypoint {index}"
                    )
                move(
                    "right",
                    solution.joint_degrees,
                    f"right_lifted_retract_{index:02d}",
                    steps=max(args.motion_steps // 4, 30),
                    hold_steps=(30 if index == len(retract_offsets) else 0),
                )
                unsafe = _probe_unsafe_contacts(
                    contact_diagnostics.summary,
                    blade_geometry,
                    grasp_geometry,
                )
                if unsafe:
                    raise RuntimeError(
                        f"unsafe robot contact during lifted retract waypoint {index}"
                    )
            stages.append(
                {
                    "stage": "right_lifted_retract",
                    "control_mode": (
                        "collision_ranked_pose_preserving_extended_aisle_stow"
                    ),
                    "selected_route": selected_retract_route["name"],
                    "origin_m": retract_origin[:3, 3].tolist(),
                    "offsets_m": [offset.tolist() for offset in retract_offsets],
                    "solutions": [
                        dataclasses.asdict(solution)
                        for solution in retract_solutions
                    ],
                }
            )

        grasp_geometry = grasp_manager.target_geometry
        if grasp_geometry is None:
            raise RuntimeError("live grasp geometry disappeared before transport")
        transport_origin = grasp_geometry["centre_m"].copy()
        transport_ee_origin = model.forward(
            "left",
            current["left"],
            base_matrix,
        )
        # Select a payload-clear route before moving. The D405 and both finger
        # envelopes are scored against every live non-target capsule; short
        # negative-X waypoints let the second petiole leave the row without
        # crossing SubStem_02, while the original route remains available for
        # targets whose local canopy is clear.
        transport_route_candidates = {
            "row_normal": (
                np.asarray([0.0, 0.04, 0.04]),
                np.asarray([0.0, 0.10, 0.06]),
                np.asarray([0.0, 0.18, 0.08]),
                np.asarray([-0.06, 0.20, 0.07]),
                np.asarray([-0.12, 0.22, 0.04]),
            ),
            "low_aisle_clearance": (
                np.asarray([0.0, 0.04, 0.04]),
                np.asarray([0.0, 0.10, 0.02]),
                np.asarray([0.0, 0.18, 0.02]),
                np.asarray([-0.06, 0.20, 0.03]),
                np.asarray([-0.12, 0.22, 0.04]),
            ),
            "negative_x_clearance": (
                np.asarray([0.0, 0.04, 0.04]),
                np.asarray([-0.04, 0.07, 0.05]),
                np.asarray([-0.08, 0.10, 0.06]),
                np.asarray([-0.12, 0.14, 0.08]),
                np.asarray([-0.12, 0.18, 0.07]),
                np.asarray([-0.12, 0.22, 0.04]),
            ),
            "positive_x_clearance": (
                np.asarray([0.0, 0.04, 0.04]),
                np.asarray([0.04, 0.08, 0.05]),
                np.asarray([0.04, 0.13, 0.07]),
                np.asarray([0.01, 0.18, 0.08]),
                np.asarray([-0.06, 0.21, 0.06]),
                np.asarray([-0.12, 0.22, 0.04]),
            ),
        }
        transport_route_candidates = {
            name: tuple(task_offset(*offset) for offset in offsets)
            for name, offsets in transport_route_candidates.items()
        }
        excluded_grasp_colliders = tuple(
            grasp_geometry.get(
                "orphan_colliders", grasp_geometry.get("colliders", ())
            )
        )
        # The radial-support box/capsule proxy over-bounds the proven-clear
        # first corridor and its adjacent leaf proxies by up to 10.3 mm. Admit
        # a 12 mm measured model bias for route
        # ranking only; PhysX contact impulses remain the zero-tolerance gate.
        minimum_payload_clearance_m = -0.012
        route_records = []
        feasible_routes = []
        for route_name, offsets in transport_route_candidates.items():
            route_seed = current["left"]
            route_plan = []
            minimum_clearance_m = float("inf")
            route_feasible = True
            for index, offset in enumerate(offsets, start=1):
                desired = transport_ee_origin.copy()
                desired[:3, 3] += offset
                pose_transport = model.solve_pose(
                    "left",
                    desired,
                    route_seed,
                    base_matrix,
                )
                pose_clearance, pose_components = left_payload_clearance(
                    pose_transport,
                    excluded_grasp_colliders,
                )
                candidates = [
                    (
                        "pose_preserving",
                        pose_transport,
                        pose_clearance,
                        pose_components,
                    )
                ]
                if (
                    not pose_transport.succeeded
                    or pose_clearance["clearance_m"] < minimum_payload_clearance_m
                ):
                    axes_transport = solve_left(
                        transport_origin + offset,
                        route_seed,
                        _LEFT_TRANSPORT_SEED_DEGREES,
                    )
                    axes_clearance, axes_components = left_payload_clearance(
                        axes_transport,
                        excluded_grasp_colliders,
                    )
                    candidates.append(
                        (
                            "point_axes",
                            axes_transport,
                            axes_clearance,
                            axes_components,
                        )
                    )
                valid = [
                    candidate
                    for candidate in candidates
                    if candidate[1].succeeded
                    and candidate[2]["clearance_m"] >= minimum_payload_clearance_m
                ]
                selected = (
                    max(valid, key=lambda item: item[2]["clearance_m"])
                    if valid
                    else max(candidates, key=lambda item: item[2]["clearance_m"])
                )
                solver, transport, clearance, component_clearances = selected
                segment_clearances = []
                route_start = np.asarray(route_seed, dtype=np.float64)
                route_target = np.asarray(transport.joint_degrees, dtype=np.float64)
                for sample_index in range(1, 9):
                    fraction = sample_index / 8.0
                    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                    sampled_joints = route_start + smooth * (
                        route_target - route_start
                    )
                    sampled_solution = dataclasses.replace(
                        transport,
                        joint_degrees=tuple(float(value) for value in sampled_joints),
                    )
                    sampled_clearance, sampled_components = left_payload_clearance(
                        sampled_solution,
                        excluded_grasp_colliders,
                    )
                    segment_clearances.append(
                        {
                            "fraction": fraction,
                            "clearance": sampled_clearance,
                            "component_clearances": sampled_components,
                        }
                    )
                swept_clearance = min(
                    (sample["clearance"] for sample in segment_clearances),
                    key=lambda item: item["clearance_m"],
                )
                if swept_clearance["clearance_m"] < clearance["clearance_m"]:
                    clearance = swept_clearance
                route_plan.append(
                    {
                        "index": index,
                        "offset_m": offset,
                        "solver": solver,
                        "solution": transport,
                        "clearance": clearance,
                        "component_clearances": component_clearances,
                        "segment_clearances": segment_clearances,
                        "candidate_solutions": {
                            name: {
                                "solution": dataclasses.asdict(solution),
                                "clearance": candidate_clearance,
                                "component_clearances": candidate_components,
                            }
                            for name, solution, candidate_clearance, candidate_components in candidates
                        },
                    }
                )
                minimum_clearance_m = min(
                    minimum_clearance_m,
                    clearance["clearance_m"],
                )
                if not valid:
                    route_feasible = False
                    break
                if clearance["clearance_m"] < minimum_payload_clearance_m:
                    route_feasible = False
                    break
                route_seed = np.asarray(transport.joint_degrees, dtype=np.float64)
            route_record = {
                "name": route_name,
                "feasible": route_feasible,
                "minimum_payload_clearance_m": minimum_clearance_m,
                "waypoints": [
                    {
                        **{key: value for key, value in waypoint.items() if key != "solution"},
                        "offset_m": waypoint["offset_m"].tolist(),
                        "solution": dataclasses.asdict(waypoint["solution"]),
                    }
                    for waypoint in route_plan
                ],
            }
            route_records.append(route_record)
            if route_feasible:
                feasible_routes.append((minimum_clearance_m, route_name, route_plan))
        if not feasible_routes:
            stages.append(
                {
                    "stage": "left_transport_route_selection",
                    "selected_route": None,
                    "minimum_payload_clearance_m": minimum_payload_clearance_m,
                    "routes": route_records,
                }
            )
            raise RuntimeError("no collision-clear left payload transport route")
        _, selected_route_name, selected_route_plan = max(
            feasible_routes,
            key=lambda item: item[0],
        )
        transport_offsets = tuple(
            waypoint["offset_m"] for waypoint in selected_route_plan
        )
        stages.append(
            {
                "stage": "left_transport_route_selection",
                "selected_route": selected_route_name,
                "minimum_payload_clearance_m": minimum_payload_clearance_m,
                "routes": route_records,
            }
        )
        transport_solutions = []
        for waypoint in selected_route_plan:
            index = waypoint["index"]
            offset = waypoint["offset_m"]
            transport = waypoint["solution"]
            transport_solutions.append(dataclasses.asdict(transport))
            stages.append(
                {
                    "stage": f"left_transport_ik_{index:02d}",
                    "target_point_m": (transport_origin + offset).tolist(),
                    "offset_m": offset.tolist(),
                    "solver": waypoint["solver"],
                    "payload_clearance": waypoint["clearance"],
                    "component_clearances": waypoint["component_clearances"],
                    "segment_clearances": waypoint["segment_clearances"],
                    "candidate_solutions": waypoint["candidate_solutions"],
                    "solution": dataclasses.asdict(transport),
                }
            )
            if not transport.succeeded:
                raise RuntimeError(f"left transport IK failed at waypoint {index}")
            move(
                "left",
                transport.joint_degrees,
                f"left_transport_{index:02d}",
                steps=args.motion_steps,
                hold_steps=(
                    max(args.motion_steps // 2, 30)
                    if index == len(transport_offsets)
                    else 0
                ),
            )
            grasp_geometry = grasp_manager.target_geometry
            if grasp_geometry is None:
                raise RuntimeError(
                    f"live grasp geometry disappeared at transport waypoint {index}"
                )
            unsafe = _probe_unsafe_contacts(
                contact_diagnostics.summary,
                blade_geometry,
                grasp_geometry,
            )
            if unsafe:
                raise RuntimeError(
                    f"unsafe robot contact during transport waypoint {index}"
                )
        stages.append(
            {
                "stage": "left_transport_ik",
                "origin_m": transport_origin.tolist(),
                "offsets_m": [offset.tolist() for offset in transport_offsets],
                "solutions": transport_solutions,
            }
        )
        if grasp_manager.task_phase != "transported":
            raise RuntimeError(f"orphan did not reach transport clearance: {grasp_manager.task_phase}")

        contact_diagnostics.set_phase("orphan_release")
        grasp_manager.request_open()
        for _ in range(args.drop_steps):
            tick()
        stages.append(
            {
                "stage": "orphan_release_and_drop",
                "steps": args.drop_steps,
                "task_phase": grasp_manager.task_phase,
                "summary": grasp_manager.summary,
            }
        )
    except Exception as exc:
        unsafe = _probe_unsafe_contacts(
            contact_diagnostics.summary, blade_geometry, grasp_geometry
        )
        return {
            "mode": args.bimanual_probe,
            "stages": stages,
            "physical_cuts": applied_cuts,
            "unsafe_contacts": unsafe,
            "blade_safety_clear": blade_monitor.safety_clear,
            "minimum_inter_arm_clearance": dict(minimum_inter_arm_clearance),
            "task": grasp_manager.summary,
            "succeeded": False,
            "error": str(exc),
        }

    unsafe = _probe_unsafe_contacts(contact_diagnostics.summary, blade_geometry, grasp_geometry)
    benchmark_cuts = [cut for cut in applied_cuts if cut.get("benchmark_valid")]
    return {
        "mode": args.bimanual_probe,
        "stages": stages,
        "physical_cuts": applied_cuts,
        "unsafe_contacts": unsafe,
        "blade_safety_clear": blade_monitor.safety_clear,
        "minimum_inter_arm_clearance": dict(minimum_inter_arm_clearance),
        "task": grasp_manager.summary,
        "succeeded": bool(
            grasp_manager.task_succeeded
            and len(benchmark_cuts) == 1
            and not unsafe
            and blade_monitor.safety_clear
        ),
    }


def _airflow_probe(stage, context, airflow, steps: int) -> dict:
    import numpy as np

    paths = airflow.body_paths
    if not paths:
        return {"steps": steps, "targets": 0, "succeeded": False, "error": "no foliage targets"}
    start = _positions(stage, paths)
    minimum = start.copy()
    maximum = start.copy()
    max_from_start = 0.0
    finite = True
    for step in range(steps):
        airflow.step()
        context.step(render=False)
        if step % 4 != 0 and step + 1 != steps:
            continue
        current = _positions(stage, paths)
        finite = finite and bool(np.isfinite(current).all())
        minimum = np.minimum(minimum, current)
        maximum = np.maximum(maximum, current)
        max_from_start = max(
            max_from_start,
            float(np.nanmax(np.linalg.norm(current - start, axis=1))),
        )
    peak_to_peak = float(np.nanmax(maximum - minimum))
    return {
        "steps": steps,
        "duration_s": steps / 240.0,
        "targets": len(paths),
        "peak_displacement_mm": max_from_start * 1000.0,
        "peak_to_peak_axis_motion_mm": peak_to_peak * 1000.0,
        "finite": finite,
        "succeeded": bool(finite and 0.00001 < peak_to_peak < 0.1),
    }

def _contact_body(runtime: VineRuntime, label: str) -> str:
    from pxr import Sdf

    organ_index = runtime.organ_indices.get(label)
    if organ_index is None:
        raise KeyError(f"unknown organ {label}")
    contact_bodies = {str(Sdf.Path(path).GetParentPath()) for path in runtime.rig.collider_paths}
    candidates = [
        link.path
        for link in runtime.rig.links
        if link.organ == organ_index and link.path in contact_bodies
    ]
    if not candidates:
        raise RuntimeError(f"{label} has no interaction collider")
    return candidates[0]


def _pull_probe(stage, context, runtime: VineRuntime, label: str, args) -> dict:
    import carb
    import numpy as np
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    from pxr import Sdf
    from pxr import UsdPhysics
    from pxr import UsdUtils

    body = _contact_body(runtime, label)
    prim = stage.GetPrimAtPath(body)
    mass_value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
    if mass_value is None or float(mass_value) <= 0.0:
        return {"target": label, "body": body, "succeeded": False, "error": "body has no positive mass"}
    mass_kg = float(mass_value)
    force_n = mass_kg * float(args.pull_accel)
    body_path = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(body))
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation = get_physx_simulation_interface()
    start = _positions(stage, [body])[0]
    peak = 0.0
    for _ in range(args.pull_steps):
        current = _positions(stage, [body])[0]
        simulation.apply_force_at_pos(
            stage_id,
            body_path,
            carb.Float3(force_n, 0.0, 0.0),
            carb.Float3(*current),
            "Force",
        )
        context.step(render=False)
        peak = max(peak, float(np.linalg.norm(_positions(stage, [body])[0] - start)))
    for _ in range(args.recover_steps):
        context.step(render=False)
    final = _positions(stage, [body])[0]
    recovery = float(np.linalg.norm(final - start))
    finite = bool(np.isfinite(final).all())
    succeeded = finite and 0.001 < peak < 0.5 and recovery < 0.5
    return {
        "target": label,
        "body": body,
        "mass_kg": mass_kg,
        "acceleration_m_s2": args.pull_accel,
        "force_n": force_n,
        "peak_displacement_mm": peak * 1000.0,
        "recovery_displacement_mm": recovery * 1000.0,
        "finite": finite,
        "succeeded": succeeded,
    }


def _cut_probe(stage, context, runtime: VineRuntime, label: str, steps: int) -> dict:
    import numpy as np
    from pxr import UsdPhysics

    organ_index = runtime.organ_indices.get(label)
    if organ_index is None:
        return {"target": label, "succeeded": False, "error": "unknown organ"}
    joint_path = runtime.rig.cut_joints.get(label)
    if joint_path is None:
        return {"target": label, "succeeded": False, "error": "organ has no cut joint"}
    joint_attr = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path)).GetJointEnabledAttr()
    enabled_before = bool(joint_attr.Get())
    tracked = runtime.rig.link_paths_for(organ_index)
    before = _positions(stage, tracked)
    context.pause()
    try:
        record = runtime.severer.cut(label)
    except Exception as exc:
        context.play()
        return {"target": label, "succeeded": False, "error": str(exc)}
    enabled_after = bool(
        UsdPhysics.Joint(stage.GetPrimAtPath(record.joint_path)).GetJointEnabledAttr().Get()
    )
    context.play()
    for _ in range(steps):
        context.step(render=False)
    after = _positions(stage, tracked)
    travel = np.linalg.norm(after - before, axis=1)
    drop = float(np.nanmean(before[:, 2]) - np.nanmean(after[:, 2]))
    max_travel = float(np.nanmax(travel))
    finite = bool(np.isfinite(after).all())
    return {
        "target": label,
        "joint": record.joint_path,
        "joint_enabled_before": enabled_before,
        "joint_enabled_after": enabled_after,
        "grade": record.grade,
        "drop_mm": drop * 1000.0,
        "max_travel_mm": max_travel * 1000.0,
        "finite": finite,
        "succeeded": bool(finite and enabled_before and not enabled_after and max_travel > 0.005),
    }


def _base_link_paths(rig) -> list[str]:
    bases = {}
    for link in rig.links:
        current = bases.get(link.organ)
        if current is None or link.index < current.index:
            bases[link.organ] = link
    return [bases[index].path for index in sorted(bases)]


def _positions(stage, paths: list[str]):
    import numpy as np
    from pxr import Usd
    from pxr import UsdGeom

    rows = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            rows.append([np.nan, np.nan, np.nan])
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows.append(list(matrix.ExtractTranslation()))
    return np.asarray(rows, dtype=np.float64)


def _capture(camera_path: str, args, app) -> None:
    import numpy as np
    import omni.replicator.core as rep
    from PIL import Image

    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    product = rep.create.render_product(camera_path, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    for _ in range(args.warmup):
        rep.orchestrator.step(rt_subframes=4)
    rgba = np.asarray(annotator.get_data())
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(args.screenshot)
    app.update()


if __name__ == "__main__":
    raise SystemExit(main())
