"""Session-only Model A v1.2 geometry and physically mounted RGB cameras.

This package's Phase 1 preview intentionally has no robot or plant dynamics.
The existing interactive greenhouse remains the separate physics workflow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from greenhouse_sim import robot_hardware, robot_kinematics, robot_model, robot_scene
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def add_robot_preview(stage, *, gutter_x, asset=robot_model.DEFAULT_ASSET):
    asset = Path(asset).resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"Build the v1.2 robot first: {asset}")
    path = "/World/RBY1"
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Robot preview already exists at {path}")
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().GetReferences().AddReference(asset.as_posix())
    if root.GetPrim().GetAttribute("tomato:robotModel").Get() != robot_model.ROBOT_NAME:
        raise ValueError("Robot asset is stale or not Model A v1.2; rebuild it")
    robot_hardware._set_transform(root.GetPrim(), robot_hardware.rotation_z(180),
                                 np.array([gutter_x + 1.0, 0.0, 0.0]))
    pose = dict(robot_scene.SDK_READY_POSE_DEGREES)
    pose["head_1"] = -15.0
    links = robot_kinematics.Rby1Kinematics().all_link_transforms(pose)
    for link, matrix in links.items():
        prim = stage.GetPrimAtPath(f"{path}/{link}")
        if not prim:
            raise ValueError(f"Missing v1.2 robot link {link}")
        robot_hardware._set_transform(prim, matrix[:3, :3], matrix[:3, 3])
    for prim in Usd.PrimRange(root.GetPrim()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    root.GetPrim().CreateAttribute("tomato:previewOnly", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    cameras = {
        "Robot head D405": f"{path}/link_head_2/attachments/HeadCamera/D405/DepthCamera",
        "Left wrist D405": f"{path}/ee_left/attachments/LeftWristCamera/D405/DepthCamera",
        "Right wrist D405": f"{path}/ee_right/attachments/RightWristCamera/D405/DepthCamera",
    }
    for camera in cameras.values():
        if not stage.GetPrimAtPath(camera).IsA(UsdGeom.Camera):
            raise ValueError(f"Missing fitted camera {camera}")
    return {"asset": str(asset), "root": path, "model": robot_model.ROBOT_NAME,
            "camera_paths": cameras, "resolution": list(robot_model.D405_RESOLUTION),
            "pose_degrees": pose, "robot_dynamics_enabled": False,
            "wrist_adapter_physical_fit_validated": False}


def select_camera(viewport, camera_path):
    sensor = "/D405/DepthCamera" in camera_path
    viewport.set_texture_resolution(robot_model.D405_RESOLUTION if sensor else (1280, 720))
    viewport.set_active_camera(camera_path)
