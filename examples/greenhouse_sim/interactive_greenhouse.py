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
    parser.add_argument("--robot-position", type=float, nargs=3, default=(6.99114, 3.93, -0.3050817))
    parser.add_argument("--robot-yaw", type=float, default=-90.0)
    parser.add_argument("--physics-vines", type=int, default=1)
    parser.add_argument(
        "--target-vine",
        default="Vine_0000",
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
        help="run a staged headless robot approach or full deleafing acceptance",
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
    parser.add_argument("--teleop-max-joint-speed", type=float, default=45.0, metavar="DEG_S")
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

    headless = args.headless or args.screenshot is not None or args.bimanual_probe is not None
    app = SimulationApp({"headless": headless})

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
    if not args.no_robot:
        robot_placement = robot_scene.add_fitted_robot(
            stage,
            robot_path,
            position_m=args.robot_position,
            yaw_degrees=args.robot_yaw,
        )
        report["robot"] = dataclasses.asdict(robot_placement)

    static_scope = stage.GetPrimAtPath("/World/Vines")
    static_vines = list(static_scope.GetChildren()) if static_scope and static_scope.IsValid() else []
    selected = static_vines[: min(args.physics_vines, len(static_vines))]
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

    if headless:
        success = _run_headless_checks(
            stage,
            context,
            runtimes,
            selected_target,
            args,
            report,
        )
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
            context.step(render=True)
            simulation_step += 1
            controller.process(context)
            if teleop_controller is not None:
                teleop_controller.process(simulation_step)
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
        import omni.usd
        from omni.kit.manipulator.selector import get_manipulator_selector

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
                },
            )
            if self._phase not in pair["phases"]:
                pair["phases"].append(self._phase)
            pair["events"] += 1
            start = header.contact_data_offset
            stop = start + header.num_contact_data
            pair["contacts"] += header.num_contact_data
            for contact in data[start:stop]:
                impulse = np.asarray(self._vector(contact.impulse), dtype=np.float64)
                magnitude = float(np.linalg.norm(impulse))
                pair["minimum_separation_mm"] = min(
                    pair["minimum_separation_mm"], float(contact.separation) * 1000.0
                )
                if magnitude > pair["maximum_impulse_ns"]:
                    pair["maximum_impulse_ns"] = magnitude
                    pair["maximum_impulse_vector_ns"] = impulse.tolist()
                    pair["maximum_impulse_position_m"] = self._vector(contact.position)

    @property
    def summary(self) -> dict:
        pairs = sorted(
            self._pairs.values(), key=lambda item: item["maximum_impulse_ns"], reverse=True
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
        import numpy as np
        from pxr import Gf
        from pxr import PhysxSchema
        from pxr import Usd
        from pxr import UsdGeom

        self._cutting = cutting_module
        self._np = np
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
            matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized(),
            dtype=self._np.float64,
        )
        cut_direction = self._np.asarray(
            matrix.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)).GetNormalized(),
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
            matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized(),
            dtype=self._np.float64,
        )
        cut_direction = self._np.asarray(
            matrix.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)).GetNormalized(),
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
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._stage = stage
        self._runtimes = {runtime.name: runtime for runtime in runtimes}
        self._root_path = robot_placement.root_path
        self._task_parameters = task_parameters
        self._open_width_m = float(open_width_m)
        self._close_overtravel_m = float(close_overtravel_m)
        self._close_requested = False
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
                if (
                    "grasp" not in info.role
                    and "petiole_cut_zone" not in info.role
                ):
                    continue
                key = f"{runtime.name}/{info.organ_label}"
                self._grasp_colliders[info.path] = {
                    "key": key,
                    "vine": runtime.name,
                    "organ": info.organ_label,
                    "body": info.body_path,
                    "collider": info.path,
                    "role": info.role,
                }
                self._grasp_colliders_for_key.setdefault(key, []).append(
                    info.path
                )
                if "grasp" in info.role or key not in self._grasp_collider_for_key:
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

    def _set_finger_targets(self, *, opened: bool) -> None:
        for drive, open_target in self._finger_drives:
            closed_target = (
                -self._np.sign(open_target) * self._close_overtravel_m
            )
            drive.CreateTargetPositionAttr(
                open_target if opened else float(closed_target)
            )
            drive.CreateTargetVelocityAttr(0.0)

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
        return {
            "key": key,
            "collider": path,
            "colliders": tuple(self._grasp_colliders_for_key.get(key, (path,))),
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
                self._pending.append(
                    {
                        **target,
                        "finger": finger,
                        "point": self._vector(contact.position),
                        "impulse": self._vector(contact.impulse),
                    }
                )

    def request_close(self) -> None:
        self._close_requested = True
        self._set_finger_targets(opened=False)

    def request_open(self) -> None:
        self._close_requested = False
        if self._active_joint_key is not None and self._task is not None:
            self._task.observe_release()
            self._previous_orphan_centroid = None
            self._latest_orphan_state = None
            self._set_joint_enabled(self._active_joint_key, False)
            self._active_joint_key = None
            self._active_grasp_body = None
            self._active_grasp_collider = None
            self._active_grasp_point_local = None
        self._set_finger_targets(opened=True)

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

        self._window = ui.Window("Vine Interaction", width=420, height=405)
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
        self._mailbox = teleop_module.CommandMailbox(args.teleop_command_file)
        self._model = robot_kinematics.Rby1Kinematics()
        limits = {
            side: self._model.arm_limits_degrees(side) for side in ("left", "right")
        }
        self._gate = teleop_module.TeleopSafetyGate(
            limits,
            watchdog_s=args.teleop_watchdog_ms / 1000.0,
            maximum_joint_speed_deg_s=args.teleop_max_joint_speed,
        )
        base_gf = UsdGeom.Xformable(
            stage.GetPrimAtPath("/World/RBY1/base")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self._base_matrix = np.asarray(base_gf, dtype=np.float64).T
        self._blade_local = np.append(
            robot_hardware.KNIFE_ROTATION
            @ np.asarray(_RIGHT_EDGE_WING_M, dtype=np.float64),
            1.0,
        )
        self._previous_blade_point = None
        self._last_process_time = time.monotonic()
        self._last_command_time = None
        self._last_record_time = -float("inf")
        self._last_publish_time = -float("inf")
        self._last_gripper_closed = None
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

        self._drive_configuration = {
            side: _configure_arm_drives(stage, side) for side in ("left", "right")
        }
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
            self._camera_recorder = _TeleopCameraRecorder(
                camera_views,
                args.teleop_cameras,
                (args.teleop_width, args.teleop_height),
                self._recorder.frames_directory,
            )
        self._publish(force=True)

    def _arm_state(self, side: str) -> dict:
        from pxr import PhysxSchema
        from pxr import UsdPhysics

        positions = []
        velocities = []
        for index in range(7):
            joint = self._stage.GetPrimAtPath(f"/World/RBY1/joints/{side}_arm_{index}")
            state = PhysxSchema.JointStateAPI.Get(joint, "angular")
            position = state.GetPositionAttr().Get() if state else None
            velocity = state.GetVelocityAttr().Get() if state else None
            if position is None:
                drive = UsdPhysics.DriveAPI.Get(joint, "angular")
                position = drive.GetTargetPositionAttr().Get() if drive else 0.0
            positions.append(float(position or 0.0))
            velocities.append(float(velocity or 0.0))
        return {"position_degrees": positions, "velocity_degrees_s": velocities}

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
            self._contact_diagnostics.summary,
            blade_geometry,
            grasp_geometry,
        )

    def _set_blade_velocity(self, right_target, *, apply_right: bool, dt_s: float) -> None:
        point = (self._model.forward("right", right_target, self._base_matrix) @ self._blade_local)[:3]
        velocity = self._np.zeros(3, dtype=self._np.float64)
        if apply_right and self._previous_blade_point is not None and dt_s > 0.0:
            velocity = (point - self._previous_blade_point) / dt_s
        self._previous_blade_point = point.copy()
        self._blade_monitor.set_commanded_edge_velocity(velocity)

    def _record(self, simulation_step: int, now_s: float, command, left_state, right_state) -> None:
        if self._recorder is None or not command.recording:
            return
        if now_s - self._last_record_time < 1.0 / self._args.teleop_record_hz:
            return
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
                    "left_arm": left_state,
                    "right_arm": right_state,
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
            "recording": self._recording,
            "recorded_samples": self._recorder.samples if self._recorder is not None else 0,
            "episode_directory": (
                str(self._episode_directory) if self._episode_directory is not None else None
            ),
            "latest_error": self._latest_error,
            "drive_configuration": self._drive_configuration,
        }

    def _publish(self, *, force: bool = False) -> None:
        now = self._time.monotonic()
        if not force and now - self._last_publish_time < 1.0:
            return
        self._report["teleoperation"] = self.summary
        _emit(self._report, self._report_path)
        self._last_publish_time = now

    def process(self, simulation_step: int) -> None:
        now = self._time.monotonic()
        dt_s = max(now - self._last_process_time, 1.0 / 240.0)
        self._last_process_time = now
        left_state = self._arm_state("left")
        right_state = self._arm_state("right")
        self._unsafe_contacts = self._current_unsafe_contacts()
        if self._unsafe_contacts:
            self._unsafe_latched = True
        if self._unsafe_latched:
            _set_arm_drive_targets(self._stage, "left", left_state["position_degrees"])
            _set_arm_drive_targets(self._stage, "right", right_state["position_degrees"])
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            self._latest_error = "unsafe robot contact latched; teleop motion is held"
            self._publish()
            return

        try:
            raw = self._mailbox.poll()
        except self._teleop.TeleopCommandError as exc:
            self._rejected_commands += 1
            self._latest_error = str(exc)
            _set_arm_drive_targets(self._stage, "left", left_state["position_degrees"])
            _set_arm_drive_targets(self._stage, "right", right_state["position_degrees"])
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            self._publish()
            return
        if raw is None:
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            if (
                self._last_command_time is not None
                and now - self._last_command_time > self._args.teleop_watchdog_ms / 1000.0
            ):
                _set_arm_drive_targets(self._stage, "left", left_state["position_degrees"])
                _set_arm_drive_targets(self._stage, "right", right_state["position_degrees"])
                self._recording = False
                self._watchdog_holds += 1
                self._latest_error = "teleop watchdog expired; measured pose is held"
            self._publish()
            return
        try:
            command = self._gate.accept(
                raw,
                now_s=now,
                dt_s=dt_s,
                current_left_degrees=left_state["position_degrees"],
                current_right_degrees=right_state["position_degrees"],
            )
        except self._teleop.TeleopCommandError as exc:
            self._rejected_commands += 1
            self._latest_error = str(exc)
            _set_arm_drive_targets(self._stage, "left", left_state["position_degrees"])
            _set_arm_drive_targets(self._stage, "right", right_state["position_degrees"])
            self._blade_monitor.set_commanded_edge_velocity(self._np.zeros(3))
            self._publish()
            return

        if command.apply_left:
            _set_arm_drive_targets(self._stage, "left", command.left_target_degrees)
            if command.left_gripper_closed != self._last_gripper_closed:
                if command.left_gripper_closed:
                    self._grasp_manager.request_close()
                else:
                    self._grasp_manager.request_open()
                self._last_gripper_closed = command.left_gripper_closed
        else:
            _set_arm_drive_targets(self._stage, "left", left_state["position_degrees"])
        if command.apply_right:
            _set_arm_drive_targets(self._stage, "right", command.right_target_degrees)
        else:
            _set_arm_drive_targets(self._stage, "right", right_state["position_degrees"])
        self._set_blade_velocity(
            command.right_target_degrees,
            apply_right=command.apply_right,
            dt_s=dt_s,
        )
        self._accepted_commands += 1
        self._rate_limited_commands += int(command.rate_limited)
        self._last_sequence = command.sequence
        self._last_command_time = now
        self._latest_age_ms = command.age_s * 1000.0
        self._latest_error = None
        self._recording = bool(command.recording and self._recorder is not None)
        self._record(simulation_step, now, command, left_state, right_state)
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


def _robot_precontact(stage, runtime: VineRuntime, target_label: str) -> dict:
    """Measure the settled knife pose against the selected physical petiole."""
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
        blade_matrix.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)).GetNormalized(),
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
    # Both components must be within the staged arm's verified reachable
    # approach envelope but outside a 5 mm no-spawn-contact margin. The full
    # probe separately proves bounded IK, contact safety, and the physical cut.
    # The blade points into the row and the U faces up.
    succeeded = bool(
        finite
        and 0.005 < blade_distance < 0.300
        and 0.005 < arc_distance < 0.300
        and blade_extension[1] < -0.70
        and arc_facing[2] > 0.70
    )
    return {
        "vine": runtime.name,
        "target": target_label,
        "target_position_m": target.tolist(),
        "knife_root_position_m": list(knife_matrix.ExtractTranslation()),
        "blade_extension": blade_extension.tolist(),
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
_RIGHT_SAFE_DEGREES = (-101.724, -83.623, 34.196, -135.683, -57.431, 94.832, -74.920)
_LEFT_JAW_CENTRE_M = (0.0, 0.0, -0.1025)
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
_RIGHT_KNIFE_YAW_DEGREES = -25.0
_RIGHT_KNIFE_ROLL_DEGREES = 0.0
_RIGHT_SERVO_MAX_ATTEMPTS = 7
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


def _configure_arm_drives(stage, side: str) -> dict:
    """Apply hardware-bounded position gains to one RB-Y1 arm."""
    from pxr import UsdPhysics

    joints = []
    for index, maximum_force in enumerate(_RBY1_ARM_EFFORT_LIMITS_NM):
        joint = stage.GetPrimAtPath(f"/World/RBY1/joints/{side}_arm_{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"missing {side} arm drive {index}")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(20.0)
        drive.CreateDampingAttr(2.0)
        drive.CreateMaxForceAttr(maximum_force)
        joints.append(
            {
                "joint": str(joint.GetPath()),
                "maximum_force_nm": maximum_force,
            }
        )
    return {
        "side": side,
        "type": "force",
        "stiffness_nm_per_degree": 20.0,
        "damping_nm_s_per_degree": 2.0,
        "joints": joints,
    }


def _set_arm_drive_targets(stage, side: str, degrees) -> None:
    from pxr import UsdPhysics

    for index, value in enumerate(degrees):
        joint = stage.GetPrimAtPath(f"/World/RBY1/joints/{side}_arm_{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"missing {side} arm drive {index}")
        drive.CreateTargetPositionAttr(float(value))
        drive.CreateTargetVelocityAttr(0.0)


def _probe_unsafe_contacts(contact_summary: dict, blade_geometry: dict, grasp_geometry: dict) -> list[dict]:
    """Reject robot contacts except floor support and intended tool contacts."""
    cut_colliders = set(blade_geometry["colliders"])
    grasp_colliders = set(
        grasp_geometry.get("colliders", (grasp_geometry["collider"],))
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
        if other.startswith(("/World/Main_Cultivation_Zone", "/World/Main_Cultivation_Zone_01")):
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

    model = robot_kinematics.Rby1Kinematics()
    base_gf = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/RBY1/base")
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Gf uses row-vector matrices; the pure kinematics module uses columns.
    base_matrix = np.asarray(base_gf, dtype=np.float64).T
    commanded_blade_local = np.append(
        robot_hardware.KNIFE_ROTATION
        @ np.asarray(_RIGHT_EDGE_WING_M, dtype=np.float64),
        1.0,
    )
    previous_commanded_blade_point = None
    stages = []
    applied_cuts = []
    left_counterpull_start = None
    left_counterpull_target = None
    left_hold_capacity_n = None
    current = {
        "left": np.asarray(_LEFT_READY_DEGREES, dtype=np.float64),
        "right": np.asarray(_RIGHT_SAFE_DEGREES, dtype=np.float64),
    }
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
        selection_seeds = (
            _LEFT_AISLE_CLEARANCE_WAYPOINTS_DEGREES[-1],
            _LEFT_APPROACH_SEEDS_DEGREES[0.0],
            *_LEFT_MULTISTART_SEEDS_DEGREES,
        )
        # Prefer the authored distal grasp segment, but fall back proximally
        # when exact RB-Y1 kinematics prove it unreachable. All candidates are
        # physical colliders on the same selected petiole; opposed contact
        # still determines the body that the fixed grasp joint actually binds.
        for candidate in reversed(grasp_manager.target_candidates):
            solutions = [
                model.solve_position_axes(
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
                for seed in selection_seeds
            ]
            selection_diagnostics.append(
                {
                    "collider": candidate["collider"],
                    "body": candidate["body"],
                    "role": candidate["role"],
                    "centre_m": candidate["centre_m"].tolist(),
                    "axis": candidate["axis"].tolist(),
                    "preferred": candidate["preferred"],
                    "solutions": [dataclasses.asdict(solution) for solution in solutions],
                }
            )
            if any(solution.succeeded for solution in solutions):
                selected_collider = candidate["collider"]
                break
        stages.append(
            {
                "stage": "left_grasp_candidate_selection",
                "selected_collider": selected_collider,
                "candidates": selection_diagnostics,
            }
        )
        if selected_collider is None:
            return {
                "mode": args.bimanual_probe,
                "stages": stages,
                "physical_cuts": applied_cuts,
                "unsafe_contacts": [],
                "blade_safety_clear": blade_monitor.safety_clear,
                "task": grasp_manager.summary,
                "succeeded": False,
                "error": "no selected-petiole grasp collider is reachable",
            }
        grasp_manager.set_planned_grasp_collider(selected_collider)
        grasp_geometry = grasp_manager.target_geometry

    def tick() -> None:
        context.step(render=False)
        grasp_manager.process()
        applied_cuts.extend(
            _apply_blade_cut_decisions(
                context,
                blade_monitor,
                report,
                grasp_manager=grasp_manager,
            )
        )

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

    def move(
        side: str,
        target,
        name: str,
        *,
        steps: int | None = None,
        hold_steps: int = 0,
    ) -> None:
        start = current[side].copy()
        target_values = np.asarray(target, dtype=np.float64)
        count = int(steps or args.motion_steps)
        contact_diagnostics.set_phase(name)
        samples = []
        for index in range(1, count + 1):
            fraction = index / count
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            commanded = start + smooth * (target_values - start)
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
        if hold_steps:
            samples.append(
                {
                    "step": count + int(hold_steps),
                    "ee_position_m": pose_sample(side),
                    "endpoint_hold": True,
                }
            )
        current[side] = target_values
        stages.append(
            {
                "stage": name,
                "arm": side,
                "steps": count,
                "hold_steps": int(hold_steps),
                "target_degrees": target_values.tolist(),
                "samples": samples,
                "blade_safety_clear": blade_monitor.safety_clear,
            }
        )

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
            for side in ("right", "left"):
                commanded = starts[side] + smooth * (
                    targets[side] - starts[side]
                )
                _set_arm_drive_targets(stage, side, commanded)
                if side == "right":
                    record_commanded_blade_motion(commanded)
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
        current.update(commanded_targets)
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
            }
        )

    def solve_left(point, seed, alternate_seed=None, diagnostics=None) -> object:
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
        if diagnostics is not None:
            diagnostics.extend(
                {
                    "seed_degrees": [float(value) for value in candidate_seed],
                    "position_scale_m": position_scale_m,
                    "solution": dataclasses.asdict(result),
                }
                for (candidate_seed, position_scale_m), result in zip(
                    candidate_specs, candidates, strict=True
                )
            )
        # Preserve the live trajectory branch whenever it passes. Additional
        # deterministic branches are fallbacks, not a global minimum that may
        # introduce a large joint-space jump into an already valid path.
        return next(
            (result for result in candidates if result.succeeded),
            min(
                candidates,
                key=lambda result: (
                    result.position_error_m,
                    result.orientation_error_rad,
                ),
            ),
        )

    def solve_right(
        side_m: float,
        seed,
        translation_correction=None,
        cut_geometry_override=None,
    ) -> object:
        preferred_cut_direction = np.asarray(
            [0.0, -np.cos(np.radians(15.0)), np.sin(np.radians(15.0))],
            dtype=np.float64,
        )
        cut_geometry = (
            blade_monitor.target_path_geometry(_RIGHT_CUT_STUB_M)
            if cut_geometry_override is None
            else cut_geometry_override
        )
        if cut_geometry is None:
            raise RuntimeError("live articulated cut path disappeared")
        knife_rotation = robot_hardware.cut_aligned_knife_rotation(
            cut_geometry["axis"],
            preferred_cut_direction,
        )
        cut_direction = knife_rotation @ np.asarray([0.0, -1.0, 0.0])
        # Stay inside the benchmark's 25 mm admissible physical stub zone and
        # follow the live bent centreline rather than extending Link-0 as if
        # the whole articulated petiole were straight.
        cut_point = cut_geometry["point_m"]
        edge = cut_point + cut_direction * side_m
        correction = (
            np.zeros(3, dtype=np.float64)
            if translation_correction is None
            else np.asarray(translation_correction, dtype=np.float64)
        )
        root = edge - knife_rotation @ np.asarray(_RIGHT_EDGE_WING_M) + correction
        desired = np.eye(4, dtype=np.float64)
        desired[:3, :3] = knife_rotation @ robot_hardware.KNIFE_ROTATION.T
        desired[:3, 3] = root
        return model.solve_pose("right", desired, seed, base_matrix)

    try:
        if args.bimanual_probe in {"left_approach", "full"}:
            for index, waypoint in enumerate(_LEFT_AISLE_CLEARANCE_WAYPOINTS_DEGREES):
                move("left", waypoint, f"left_clearance_{index}")

            left_solutions = []
            offsets = (0.10, 0.06, 0.04, 0.02)
            if args.bimanual_probe == "full":
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
                goal = grasp_geometry["centre_m"] + np.asarray([0.0, offset, 0.0])
                candidate_diagnostics = []
                solution = solve_left(
                    goal,
                    seed,
                    _LEFT_APPROACH_SEEDS_DEGREES[offset],
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
                    raise RuntimeError(f"left IK failed at aisle offset {offset:.3f} m")
                move("left", solution.joint_degrees, f"left_approach_{offset:.3f}")
                seed = solution.joint_degrees
            stages.append({"stage": "left_ik", "solutions": left_solutions})

            if args.bimanual_probe == "left_approach":
                unsafe = _probe_unsafe_contacts(
                    contact_diagnostics.summary, blade_geometry, grasp_geometry
                )
                return {
                    "mode": args.bimanual_probe,
                    "stages": stages,
                    "unsafe_contacts": unsafe,
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

            # Hold the physical grasp at its established pose during cutting.
            # An earlier 50 mm pre-pull rotated wrist joint 4 into a posture
            # with only ~71 N point-force capacity, barely above the 66.3 N
            # cut threshold, and swept finger 1 through another organ. The
            # unpulled grasp posture has measurable margin and supplies the
            # required bimanual counterforce without preloading a neighbour.
            grasp_geometry = grasp_manager.target_geometry
            if grasp_geometry is None:
                raise RuntimeError("live grasp geometry disappeared before counterhold")
            hold_cut_geometry = blade_monitor.target_path_geometry(
                _RIGHT_CUT_STUB_M
            )
            if hold_cut_geometry is None:
                raise RuntimeError("live cut geometry disappeared before counterhold")
            preferred_cut_direction = np.asarray(
                [0.0, -np.cos(np.radians(15.0)), np.sin(np.radians(15.0))],
                dtype=np.float64,
            )
            hold_knife_rotation = robot_hardware.cut_aligned_knife_rotation(
                hold_cut_geometry["axis"],
                preferred_cut_direction,
            )
            hold_cut_direction = hold_knife_rotation @ np.asarray(
                [0.0, -1.0, 0.0], dtype=np.float64
            )
            feedback = blade_monitor.active_cut_feedback
            required_cut_force_n = float(
                args.cut_force
                if feedback is None
                else feedback["required_force_n"]
            )
            hold_local_point = (
                _LEFT_JAW_CENTRE_M
                if grasp_geometry.get("tool_local_point_m") is None
                else grasp_geometry["tool_local_point_m"]
            )
            hold_capacity = model.point_force_capacity(
                "left",
                current["left"],
                base_matrix,
                hold_local_point,
                hold_cut_direction,
                required_cut_force_n,
                _RBY1_ARM_EFFORT_LIMITS_NM,
            )
            left_hold_capacity_n = hold_capacity.force_capacity_n
            minimum_hold_capacity_n = 1.10 * required_cut_force_n
            stages.append(
                {
                    "stage": "left_static_counterhold",
                    "joint_degrees": current["left"].tolist(),
                    "grasp_body": grasp_geometry["body"],
                    "grasp_collider": grasp_geometry["collider"],
                    "grasp_anchor_m": grasp_geometry["centre_m"].tolist(),
                    "tool_local_anchor_m": np.asarray(hold_local_point).tolist(),
                    "opposed_cut_direction": hold_cut_direction.tolist(),
                    "required_cut_force_n": required_cut_force_n,
                    "minimum_capacity_n": minimum_hold_capacity_n,
                    "capacity": dataclasses.asdict(hold_capacity),
                }
            )
            if left_hold_capacity_n < minimum_hold_capacity_n:
                raise RuntimeError(
                    "left counterhold posture lacks hardware effort margin: "
                    f"{left_hold_capacity_n:.1f} N capacity for "
                    f"{required_cut_force_n:.1f} N cut"
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
            right_solutions = []
            # Establish the rolled tool frame well outside the canopy, then
            # enter along the intended Cartesian cut line.
            right_offsets = (-0.100, -0.060, -0.035, -0.015)
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
                    solution = solve_right(
                        side_m,
                        seed,
                        translation_correction,
                    )
                    solution_record = dataclasses.asdict(solution)
                    solution_record.update(
                        {
                            "side_m": side_m,
                            "servo_attempt": servo_attempt,
                            "translation_correction_m": translation_correction.tolist(),
                        }
                    )
                    right_solutions.append(solution_record)
                    stages.append(
                        {
                            "stage": f"right_ik_{side_m:.3f}_{servo_attempt}",
                            "solution": solution_record,
                        }
                    )
                    if not solution.succeeded:
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
                    seed = solution.joint_degrees
                    live_target = blade_monitor.target_geometry
                    live_path = blade_monitor.target_path_geometry(_RIGHT_CUT_STUB_M)
                    if live_target is None or live_path is None:
                        raise RuntimeError(
                            f"live cut geometry disappeared after side {side_m:.3f} m"
                        )
                    tool = blade_monitor.tool_point_geometry(_RIGHT_EDGE_WING_M)
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
                    "edge_wing_local_m": list(_RIGHT_EDGE_WING_M),
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
            # Use at least 24 integration/contact samples per 5 mm IK segment.
            # At the default this is a 1.0 s, 0.05 m/s nominal sweep; measured
            # edge force can slow or unload it further before the left arm's
            # hardware-bounded counterhold is overpowered.
            sweep_segments = 10
            segment_steps = max(24, args.motion_steps // 8)
            # Reacquire the compliant target through pre-contact, then commit
            # to this physical cut plane. Continuing to chase the target while
            # loaded makes the blade push the petiole sideways; a latched plane
            # creates relative shear travel while the left arm holds tension.
            latched_cut_geometry = blade_monitor.target_path_geometry(
                _RIGHT_CUT_STUB_M
            )
            if latched_cut_geometry is None:
                raise RuntimeError("live cut geometry disappeared before committed stroke")
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
                ),
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
                    tool = blade_monitor.tool_point_geometry(_RIGHT_EDGE_WING_M)
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
            for side_m in (-0.035, -0.060, -0.100):
                move(
                    "right",
                    right_waypoints[side_m],
                    f"right_retract_{side_m:.3f}",
                    hold_steps=args.motion_steps,
                )

        grasp_geometry = grasp_manager.target_geometry
        if grasp_geometry is None:
            raise RuntimeError("live grasp geometry disappeared before transport")
        transport_origin = grasp_geometry["centre_m"].copy()
        # Keep the grasp anchor on a short Cartesian aisle-clearance route.
        # One large joint-space interpolation arced finger 1 through Organ_0009
        # and never produced the requested anchor displacement.
        transport_offsets = (
            np.asarray([0.0, 0.04, 0.04]),
            np.asarray([0.0, 0.10, 0.06]),
            np.asarray([0.0, 0.18, 0.08]),
            # Only translate sideways after the orphan is clear of the row.
            # The earlier rearward route placed the falling branch over the
            # 620 x 500 mm chassis footprint; this aisle-side route keeps the
            # release point at least 100 mm outside that footprint.
            np.asarray([-0.22, 0.22, 0.04]),
        )
        transport_ee_origin = model.forward(
            "left",
            current["left"],
            base_matrix,
        )
        transport_solutions = []
        for index, offset in enumerate(transport_offsets, start=1):
            grasp_geometry = grasp_manager.target_geometry
            if grasp_geometry is None:
                raise RuntimeError(
                    f"live grasp geometry disappeared at transport waypoint {index}"
                )
            if index <= 3:
                desired = transport_ee_origin.copy()
                desired[:3, 3] += offset
                pose_transport = model.solve_pose(
                    "left",
                    desired,
                    current["left"],
                    base_matrix,
                )
                axes_transport = solve_left(
                    transport_origin + offset,
                    current["left"],
                )
                candidates = (
                    ("pose_preserving", pose_transport),
                    ("point_axes", axes_transport),
                )
                if pose_transport.succeeded:
                    solver, transport = candidates[0]
                else:
                    solver, transport = candidates[1]
                candidate_solutions = {
                    name: dataclasses.asdict(solution)
                    for name, solution in candidates
                }
            else:
                transport = solve_left(
                    transport_origin + offset,
                    current["left"],
                    _LEFT_TRANSPORT_SEED_DEGREES,
                )
                solver = "point_axes"
                candidate_solutions = {
                    solver: dataclasses.asdict(transport),
                }
            transport_solutions.append(dataclasses.asdict(transport))
            stages.append(
                {
                    "stage": f"left_transport_ik_{index:02d}",
                    "target_point_m": (transport_origin + offset).tolist(),
                    "offset_m": offset.tolist(),
                    "solver": solver,
                    "candidate_solutions": candidate_solutions,
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
