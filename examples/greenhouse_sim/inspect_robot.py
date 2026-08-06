"""Render an unobstructed fit check of RBY1-A, its D405s, and knife.

This complements the greenhouse physics acceptance image: greenhouse trough
cladding can occlude the externally mounted hardware from aisle viewpoints, so
this inspector places the exact same generated robot asset on a neutral floor.

    D:\\isaac-sim\\python.bat examples\\greenhouse_sim\\inspect_robot.py
"""

# SimulationApp must exist before omni/pxr imports.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot", type=pathlib.Path, default=pathlib.Path("data/greenhouse_sim/robots/rby1a_v1.0.usd")
    )
    parser.add_argument("--screenshot", type=pathlib.Path, default=pathlib.Path("data/greenhouse_sim/robot_fit_isolated.png"))
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("data/greenhouse_sim/robot_fit_isolated.json"))
    parser.add_argument("--view", choices=("overall", "right_tool", "left_wrist", "head_camera"), default="overall")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot_path = args.robot.resolve()
    if not robot_path.exists():
        print(f"fitted robot not found: {robot_path}; run build_robot.py first")
        return 1

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux

    from greenhouse_sim import robot_scene
    from greenhouse_sim import vine_physics

    omni.usd.get_context().new_stage()
    for _ in range(4):
        app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    placement = robot_scene.add_fitted_robot(stage, robot_path, position_m=(0.0, 0.0, 0.0), yaw_degrees=0.0)

    vine_physics.apply_scene_physics(stage)
    vine_physics.add_ground_plane(stage, path="/World/GroundCollision", height=0.0, size=5.0)
    floor = UsdGeom.Cube.Define(stage, "/World/FloorVisual")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.16, 0.17, 0.18)])
    floor_xform = UsdGeom.Xformable(floor.GetPrim())
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.026))
    floor_xform.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.05))

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1000.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(3000.0)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -20.0))

    camera_path = "/World/FitCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(30.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    eye = Gf.Vec3d(2.7, -2.8, 1.65)
    target = Gf.Vec3d(0.0, 0.0, 0.70)
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(view.GetInverse())

    from isaacsim.core.api import SimulationContext

    context = SimulationContext(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    context.initialize_physics()
    context.get_physics_context().set_gravity(-9.81)
    context.play()
    for _ in range(20):
        context.step(render=False)

    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = bounds_cache.ComputeWorldBound(stage.GetPrimAtPath(placement.root_path)).ComputeAlignedRange()
    attachment_views = {
        "right_tool": f"{placement.root_path}/ee_right/attachments",
        "left_wrist": f"{placement.root_path}/ee_left/attachments",
        "head_camera": f"{placement.root_path}/link_head_2/attachments",
    }
    attachment_bounds = {}
    for name, prim_path in attachment_views.items():
        extent = bounds_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        minimum = np.asarray(extent.GetMin(), dtype=np.float64)
        maximum = np.asarray(extent.GetMax(), dtype=np.float64)
        attachment_bounds[name] = {"min": minimum.tolist(), "max": maximum.tolist()}

    active_camera = camera_path
    if args.view != "overall":
        selected = attachment_bounds[args.view]
        minimum = np.asarray(selected["min"], dtype=np.float64)
        maximum = np.asarray(selected["max"], dtype=np.float64)
        centre = 0.5 * (minimum + maximum)
        radius = float(np.linalg.norm(maximum - minimum))
        direction = np.array([1.0, -1.0 if args.view != "left_wrist" else 1.0, 0.55], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        distance = max(0.20, 3.0 * radius)
        active_camera = "/World/FitCameraCloseup"
        close_camera = UsdGeom.Camera.Define(stage, active_camera)
        close_camera.CreateFocalLengthAttr(45.0)
        close_camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
        close_eye = Gf.Vec3d(*(centre + distance * direction))
        close_target = Gf.Vec3d(*centre)
        close_view = Gf.Matrix4d().SetLookAt(close_eye, close_target, Gf.Vec3d(0.0, 0.0, 1.0))
        UsdGeom.Xformable(close_camera.GetPrim()).AddTransformOp().Set(close_view.GetInverse())

    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    product = rep.create.render_product(active_camera, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    for _ in range(35):
        rep.orchestrator.step(rt_subframes=4)
    rgba = np.asarray(annotator.get_data())
    image_finite = bool(rgba.size and np.isfinite(rgba).all())
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(args.screenshot)

    camera_poses = {}
    for sensor_path in placement.cameras:
        sensor_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(sensor_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        sensor_forward = sensor_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
        sensor_up = sensor_matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized()
        camera_poses[sensor_path] = {
            "position_m": list(sensor_matrix.ExtractTranslation()),
            "forward": list(sensor_forward),
            "up": list(sensor_up),
        }

    blade_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(placement.knife_blade)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    knife_axes = {
        "blade_extension": list(blade_matrix.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)).GetNormalized()),
        "arc_facing": list(blade_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized()),
    }

    report = {
        "succeeded": image_finite,
        "robot": str(robot_path),
        "view": args.view,
        "screenshot": str(args.screenshot),
        "ready_joints": len(placement.initialized_joints),
        "cameras": list(placement.cameras),
        "camera_poses": camera_poses,
        "knife_axes": knife_axes,
        "knife_blade": placement.knife_blade,
        "knife_support": placement.knife_support,
        "world_bounds_m": {
            "min": list(bounds.GetMin()),
            "max": list(bounds.GetMax()),
        },
        "attachment_bounds_m": attachment_bounds,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    context.stop()
    app.close()
    return 0 if report["succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
