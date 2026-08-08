"""Run the fitted RB-Y1 and physics-enabled tomato vines in the greenhouse.

The generated greenhouse scene stays immutable. Selected static vine references
are hidden in the USD session layer and replaced at the same bed transforms by
the verified articulated GLB rigs.

Controls while the simulation is running:
- Shift + left-drag a stem or petiole contact zone to pull with force.
- [ and ] select the previous or next deleafing petiole.
- V selects the next physics-enabled vine.
- C cuts the selected petiole.
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
    parser.add_argument("--segment", type=float, default=0.02)
    parser.add_argument("--clip-spacing", type=float, default=0.30)
    parser.add_argument(
        "--collision-mode", choices=("interaction", "none", "all"), default="interaction"
    )
    parser.add_argument("--show-colliders", action="store_true")
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
            properties=vine_physics.TissueProperties(tear_force_n=args.tear_force),
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

    camera_path = _author_camera(stage, placement_centres[0], args, Gf, Sdf, UsdGeom)
    report.update(
        stage="rigged",
        static_vines=len(static_vines),
        physics_vines=len(runtimes),
        links=sum(len(runtime.rig.links) for runtime in runtimes),
        clips=sum(len(runtime.clips) for runtime in runtimes),
        severable=sum(len(runtime.rig.cut_joints) for runtime in runtimes),
        contact_colliders=sum(len(runtime.rig.collider_paths) for runtime in runtimes),
        sources=[str(runtime.source) for runtime in runtimes],
    )
    _emit(report, args.report)

    for _ in range(10):
        app.update()

    from isaacsim.core.api import SimulationContext

    contact_diagnostics = RobotContactDiagnostics(stage) if args.contact_diagnostics else None
    context = SimulationContext(
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
    )
    context.initialize_physics()
    if contact_diagnostics is not None:
        contact_diagnostics.subscribe()
    context.get_physics_context().set_gravity(-9.81)
    for runtime in runtimes:
        runtime.rest_positions = _positions(stage, _base_link_paths(runtime.rig))

    if not headless:
        report["native_transform_selector_disabled"] = _disable_native_mouse_interaction(app)
        _focus_viewport(camera_path)

    context.play()
    for _ in range(args.settle_steps):
        context.step(render=False)

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
        stage, runtimes, report, args.report, args, vine_interaction, camera_views
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
    print("Shift-drag pulls; [ / ] selects; C cuts; V changes vine; 1-4 switches cameras")
    try:
        while app.is_running():
            context.step(render=True)
            controller.process(context)
            if args.tear_force > 0.0:
                controller.poll_tears()
    finally:
        controller.close()
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


class InteractionController:
    """Queue UI/input actions and apply them between simulation steps."""

    controls = {
        "mouse": "Shift + left-drag visible stem, petiole, or leaf geometry",
        "previous_target": "[",
        "next_target": "]",
        "next_vine": "V",
        "cut": "C",
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
        self._active_camera = "inspection"
        self._targets = [self._target_labels(runtime) for runtime in runtimes]

        self._window = ui.Window("Vine Interaction", width=420, height=350)
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
                    ui.Button("CUT target C", clicked_fn=lambda: self._queue("cut"))
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
            "KEY_1": "camera:inspection",
            "KEY_2": "camera:head",
            "KEY_3": "camera:left_wrist",
            "KEY_4": "camera:right_wrist",
        }.get(event.input.name)
        if action is not None:
            self._queue(action)
        return True


    def process(self, context) -> None:
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
            record = runtime.severer.cut(label)
        except Exception as exc:
            self._status_label.text = str(exc)
        else:
            self._report.setdefault("cuts", []).append(dataclasses.asdict(record))
            self._status_label.text = f"Cut {runtime.name}/{label}: {record.grade}"
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
        stability.append(
            {
                "vine": runtime.name,
                "finite": finite,
                "max_displacement_mm": float(np.nanmax(moved) * 1000.0),
                "runaway_organs": runaways,
                "organs_tracked": len(paths),
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
