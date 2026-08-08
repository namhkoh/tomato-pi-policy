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
    parser.add_argument("--robot-position", type=float, nargs=3, default=(6.99114, 3.78, -0.3050817))
    parser.add_argument("--robot-yaw", type=float, default=-90.0)
    parser.add_argument("--physics-vines", type=int, default=1)
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

    from greenhouse_sim import cutting
    from greenhouse_sim import deleaf_task
    from greenhouse_sim import glb
    from greenhouse_sim import organs
    from greenhouse_sim import robot_scene
    from greenhouse_sim import skeleton as skeleton_module
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
        floor = min(float(link.start[2]) for link in rig.links)
        vine_physics.add_ground_plane(
            stage,
            path=f"{root_path}/CatchPlane",
            height=floor - 0.005,
            size=1.2,
            centre_xy=(float(placement_centre[0]), float(placement_centre[1])),
            filtered_paths=(f"{robot_placement.root_path}/base",)
            if robot_placement is not None
            else (),
        )

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
    )
    _emit(report, args.report)

    for _ in range(10):
        app.update()

    from isaacsim.core.api import SimulationContext

    contact_diagnostics = RobotContactDiagnostics(stage) if args.contact_diagnostics else None
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
        if runtimes and "SubStem_00" in runtimes[0].rig.junctions:
            blade_cutting.set_active_target(runtimes[0].name, "SubStem_00")
        report["blade_cutting"] = blade_cutting.summary
        floor_z = float(args.robot_position[2])
        grasp_manager = LeftGraspManager(
            stage,
            runtimes,
            robot_placement,
            deleaf_task.TaskParameters(
                drop_zone_min_m=(
                    float(args.robot_position[0]) - 0.45,
                    float(args.robot_position[1]) - 0.35,
                    floor_z,
                ),
                drop_zone_max_m=(
                    float(args.robot_position[0]) + 0.45,
                    float(args.robot_position[1]) + 0.35,
                    floor_z + 0.60,
                ),
            ),
        )
        if runtimes and "SubStem_00" in runtimes[0].rig.junctions:
            grasp_manager.set_active_target(runtimes[0].name, "SubStem_00")
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

    if contact_diagnostics is not None:
        report["robot_contacts"] = contact_diagnostics.summary
        contact_diagnostics.close()

    if headless:
        success = _run_headless_checks(stage, context, runtimes, args, report)
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
    )
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
        while app.is_running():
            context.step(render=True)
            controller.process(context)
            if args.tear_force > 0.0:
                controller.poll_tears()
    finally:
        controller.close()
        if blade_cutting is not None:
            report["blade_cutting"] = blade_cutting.summary
            blade_cutting.close()
        if grasp_manager is not None:
            report["bimanual_task"] = grasp_manager.summary
            grasp_manager.close()
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
                },
            )
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
        from pxr import PhysxSchema

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
        self._target_colliders = {}
        self._collider_info = {}
        self._pending: list[dict] = []
        self._subscription = None
        self._previous_edge_centre = None
        self._active_target: str | None = None
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
                self._target_colliders[info.path] = key

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
                    },
                )
                weight = max(float(self._np.linalg.norm(event["impulse"])), 1e-12)
                aggregate["point_sum"] += weight * event["point"]
                aggregate["weight"] += weight
                aggregate["impulse"] += event["impulse"]
            else:
                self._record_violation(event, dt_s)
        self._pending.clear()

        decisions = []
        contacted = set(aggregates)
        for key, aggregate in aggregates.items():
            runtime, target = self._targets[key]
            sample = self._cutting.BladeContactSample(
                point_m=aggregate["point_sum"] / aggregate["weight"],
                impulse_ns=aggregate["impulse"],
                edge_centre_m=centre,
                edge_axis=edge_axis,
                cutting_direction=cut_direction,
                edge_velocity_m_s=velocity,
                dt_s=dt_s,
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
        active = self._targets.get(self._active_target or "")
        active_target = active[1] if active is not None else None
        return {
            "model": "directional leading-edge force/work gate",
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
        self._close_requested = False
        self._pending: list[dict] = []
        self._subscription = None
        self._active_target: str | None = None
        self._task = None
        self._grasp_colliders: dict[str, dict] = {}
        self._grasp_bodies: dict[str, str] = {}
        self._joint_paths: dict[str, str] = {}
        self._orphan_paths: dict[str, list[str]] = {}
        self._active_joint_key: str | None = None
        self._cut_grasp_position = None
        self._previous_orphan_centroid = None

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
                if "grasp" not in info.role:
                    continue
                key = f"{runtime.name}/{info.organ_label}"
                self._grasp_colliders[info.path] = {
                    "key": key,
                    "vine": runtime.name,
                    "organ": info.organ_label,
                    "body": info.body_path,
                }
                self._grasp_bodies.setdefault(key, info.body_path)
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
        self._set_finger_targets(opened=True)

    def _set_finger_targets(self, *, opened: bool) -> None:
        for drive, open_target in self._finger_drives:
            drive.CreateTargetPositionAttr(open_target if opened else 0.0)
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
            self._set_joint_enabled(self._active_joint_key, False)
            self._active_joint_key = None
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

    def _activate_joint(self, key: str, point_m) -> None:
        joint = self._UsdPhysics.FixedJoint.Get(
            self._stage,
            self._joint_paths[key],
        )
        body0_path = f"{self._root_path}/ee_left"
        body1_path = self._grasp_bodies[key]
        body0 = self._body_matrix(body0_path)
        body1 = self._body_matrix(body1_path)
        point = self._Gf.Vec3d(*point_m.tolist())
        joint.GetLocalPos0Attr().Set(
            self._Gf.Vec3f(body0.GetInverse().Transform(point))
        )
        joint.GetLocalPos1Attr().Set(
            self._Gf.Vec3f(body1.GetInverse().Transform(point))
        )
        child_rotation = body1.ExtractRotationQuat()
        relative = body0.ExtractRotationQuat().GetInverse() * child_rotation
        joint.GetLocalRot0Attr().Set(self._Gf.Quatf(relative))
        joint.GetLocalRot1Attr().Set(self._Gf.Quatf(1.0))
        joint.GetJointEnabledAttr().Set(True)
        self._active_joint_key = key

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
                self._grasp_bodies[self._active_joint_key]
            )
        return accepted

    def process(self, dt_s: float = 1.0 / 240.0) -> None:
        if self._task is None:
            self._pending.clear()
            return
        self._task.advance()
        grouped: dict[str, dict] = {}
        for event in self._pending:
            aggregate = grouped.setdefault(
                event["key"],
                {
                    "vine": event["vine"],
                    "organ": event["organ"],
                    "body": event["body"],
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

        active = grouped.get(self._active_target or "")
        if self._close_requested and active is not None:
            established = self._task.observe_grasp(
                vine=active["vine"],
                organ=active["organ"],
                body_path=active["body"],
                finger_contacts=active["fingers"],
                force_n=active["impulse"] / max(dt_s, 1e-12),
            )
            if established and self._active_joint_key is None:
                point = active["point_sum"] / active["weight"]
                self._activate_joint(self._active_target, point)

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
                    self._grasp_bodies[self._active_joint_key]
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
                self._task.observe_deposit(
                    centroid_m=centroid,
                    lowest_height_m=lowest,
                    speed_m_s=speed,
                    floor_contact=(
                        lowest
                        <= floor + self._task.parameters.floor_tolerance_m
                    ),
                )

    @property
    def summary(self) -> dict:
        target_body = self._grasp_bodies.get(self._active_target or "")
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
            "active_grasp_body": target_body,
            "active_grasp_position_m": target_position,
            "left_ee_position_m": left_ee_position,
            "left_ee_distance_to_grasp_mm": target_distance_mm,
            "close_requested": self._close_requested,
            "active_joint": (
                self._joint_paths.get(self._active_joint_key)
                if self._active_joint_key is not None
                else None
            ),
            "graspable_targets": len(self._joint_paths),
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
    ):
        import carb
        import omni.appwindow
        import omni.ui as ui

        self._carb = carb
        self._runtimes = runtimes
        self._report = report
        self._report_path = report_path
        self._vine = 0
        self._target = 0
        self._pending: list[str] = []
        self._camera_views = camera_views
        self._blade_cutting = blade_cutting
        self._grasp_manager = grasp_manager
        self._active_camera = "inspection"
        self._targets = [self._target_labels(runtime) for runtime in runtimes]

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


def _run_headless_checks(stage, context, runtimes, args, report: dict) -> bool:
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
        robot_precontact = _robot_precontact(stage, runtimes[0])
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


def _robot_precontact(stage, runtime: VineRuntime) -> dict:
    """Measure the settled knife pose against the first real lower petiole."""
    import numpy as np
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom

    target_label = "SubStem_00"
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
    # Both components must be within a short final approach but outside a 5 mm
    # no-spawn-contact margin.  The blade points into the row and the U faces up.
    succeeded = bool(
        finite
        and 0.005 < blade_distance < 0.200
        and 0.005 < arc_distance < 0.200
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
