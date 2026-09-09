"""Render Model A v1.2 mount close-ups and all three real camera prims.

No teleop or physical-robot API is imported. Writes a bounded, isolated smoke
report and real Isaac RGB images, not fabricated example camera observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sim_data/robot_v12_fit"))
    parser.add_argument("--right-tool", choices=("knife_only", "gripper"), default="knife_only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 848, "height": 408,
                         "renderer": "RaytracedLighting", "multi_gpu": False})
    exit_code = 1
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdLux
        from greenhouse_sim import robot_model
        from sim_data.robot_preview import add_robot_preview

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        robot = add_robot_preview(stage, gutter_x=-1.0, right_tool=args.right_tool)
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(1200)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2200)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-30, 20, -30))
        floor = UsdGeom.Cube.Define(stage, "/World/Floor")
        floor.CreateSizeAttr(1)
        floor.CreateDisplayColorAttr([Gf.Vec3f(.2, .22, .24)])
        UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3d(5, 5, .04))
        UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, 0, -.51))
        # A real, labelled test target in front of the robot makes orientation
        # visible in sensor frames without pretending it is greenhouse data.
        for i, color in enumerate(((.8,.15,.1),(.1,.7,.15),(.15,.2,.8))):
            cube = UsdGeom.Cube.Define(stage, f"/World/TestTarget{i}")
            cube.CreateSizeAttr(.08)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(-.65, (i-1)*.15, 1.25))
        bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        views = {}
        for name, path, direction in (
            ("overall", "/World/RBY1", (-1, -1, .6)),
            ("head_mount", "/World/RBY1/link_head_2/attachments", (-1, -1, .6)),
            ("left_mount", "/World/RBY1/ee_left/attachments", (-1, -1, .65)),
            ("right_mount", "/World/RBY1/ee_right/attachments", (-1, 1, .65)),
        ):
            extent = bounds_cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            center = np.asarray(extent.GetMidpoint())
            radius = np.linalg.norm(np.asarray(extent.GetSize())) / 2
            direction = np.asarray(direction) / np.linalg.norm(direction)
            eye = center + direction * max(.3, radius * 5.5)
            camera = UsdGeom.Camera.Define(stage, f"/World/Fit_{name}")
            camera.CreateFocalLengthAttr(35)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 50))
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0,0,1)).GetInverse())
            views[name] = str(camera.GetPath())
        # Include both sibling finger links, not only the wrist attachments.
        for side, direction in (("left", (-1, -1, .6)), ("right", (-1, 1, .6))):
            extent = Gf.Range3d()
            for link in (f"ee_{side}", f"ee_finger_{side[0]}1", f"ee_finger_{side[0]}2"):
                extent.UnionWith(bounds_cache.ComputeWorldBound(stage.GetPrimAtPath(f"/World/RBY1/{link}")).ComputeAlignedRange())
            center = np.asarray(extent.GetMidpoint())
            radius = np.linalg.norm(np.asarray(extent.GetSize())) / 2
            direction = np.asarray(direction) / np.linalg.norm(direction)
            eye = center + direction * max(.4, radius * 7)
            camera = UsdGeom.Camera.Define(stage, f"/World/Fit_{side}_hand")
            camera.CreateFocalLengthAttr(24)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 50))
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0,0,1)).GetInverse())
            views[f"{side}_hand"] = str(camera.GetPath())
        views.update({name.split()[0].lower() + "_rgb": path for name, path in robot["camera_paths"].items()})
        report = {"robot": robot, "physics_tested": False, "views": {}, "passed": False}
        for name, camera_path in views.items():
            product = rep.create.render_product(camera_path, robot_model.D405_RESOLUTION)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            try:
                for _ in range(12):
                    rep.orchestrator.step(rt_subframes=2, pause_timeline=True)
                pixels = np.asarray(annotator.get_data())
                if pixels.shape[:2] != (408, 848) or not np.isfinite(pixels).all():
                    raise AssertionError(f"Invalid sensor render: {name}, {pixels.shape}")
                output = args.output / f"{name}.png"
                Image.fromarray(pixels[:, :, :3].astype(np.uint8)).save(output)
                camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
                world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                report["views"][name] = {"camera": camera_path, "image": str(output),
                                         "resolution": [848,408], "pixel_std": float(pixels[:,:,:3].std()),
                                         "world_from_usd_camera_row_matrix": [list(row) for row in world]}
                print("CAMERA_CAPTURED", name, str(output), flush=True)
            finally:
                annotator.detach([product])
                product.destroy()
        report["passed"] = True
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("ROBOT_V12_FIT_SMOKE_PASSED", flush=True)
        exit_code = 0
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
