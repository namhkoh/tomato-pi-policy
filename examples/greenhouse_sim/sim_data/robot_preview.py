"""Session-only Model A v1.2 geometry and physically mounted RGB cameras.

This package's Phase 1 preview intentionally has no robot or plant dynamics.
The existing interactive greenhouse remains the separate physics workflow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from greenhouse_sim import robot_hardware, robot_kinematics, robot_model, robot_scene
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def configure_right_tool(stage, robot_path, configuration):
    """Restore stock gripper visuals for annotation, leaving source USD intact."""
    if configuration not in ("gripper", "knife_only"):
        raise ValueError(f"Unknown right tool: {configuration}")
    right = stage.GetPrimAtPath(f"{robot_path}/ee_right")
    knife = stage.GetPrimAtPath(f"{robot_path}/ee_right/attachments/DeleafKnife")
    if not right or not knife:
        raise ValueError("Expected fitted right tool asset is missing")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        knife.SetActive(configuration == "knife_only")
        for name in robot_hardware.RIGHT_GRIPPER_LINKS:
            link = stage.GetPrimAtPath(f"{robot_path}/{name}")
            visual = stage.GetPrimAtPath(f"{robot_path}/{name}/visuals")
            if not link or not visual:
                raise ValueError(f"Original right gripper geometry is missing: {name}")
            visual.SetActive(configuration == "gripper")
            link.CreateAttribute("tomato:originalGripperRemoved", Sdf.ValueTypeNames.Bool, custom=True).Set(configuration == "knife_only")
        right.CreateAttribute("tomato:toolConfiguration", Sdf.ValueTypeNames.String, custom=True).Set(configuration)
    stage.Load(robot_path)


def add_robot_preview(stage, *, gutter_x, asset=robot_model.DEFAULT_ASSET, floor_path=None,
                      right_tool="knife_only"):
    asset = Path(asset).resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"Build the v1.2 robot first: {asset}")
    path = "/World/RBY1"
    existing = stage.GetPrimAtPath(path)
    # Kit can save camera API overrides below an otherwise undefined robot
    # root. Those overs are not a loaded robot; keep them when adding the asset.
    if existing and (existing.IsDefined() or existing.HasAuthoredReferences() or existing.HasAuthoredPayloads()):
        raise ValueError(f"Robot preview already exists at {path}")
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().GetReferences().AddReference(asset.as_posix())
    # Also load visual payloads when the host stage was opened with LoadNone.
    stage.Load(path)
    if root.GetPrim().GetAttribute("tomato:robotModel").Get() != robot_model.ROBOT_NAME:
        raise ValueError("Robot asset is stale or not Model A v1.2; rebuild it")
    configure_right_tool(stage, path, right_tool)
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
    alignment = {"method": "not_requested", "physics_validated": False}
    if floor_path is not None:
        from .floor_alignment import align_robot_to_floor
        alignment = align_robot_to_floor(stage, path, floor_path)
    return {"asset": str(asset), "root": path, "model": robot_model.ROBOT_NAME,
            "right_tool_configuration": right_tool,
            "floor_alignment": alignment,
            "camera_paths": cameras, "resolution": list(robot_model.D405_RESOLUTION),
            "pose_degrees": pose, "robot_dynamics_enabled": False,
            "wrist_adapter_physical_fit_validated": False}


def select_camera(viewport, camera_path):
    sensor = "/D405/DepthCamera" in camera_path
    viewport.set_texture_resolution(robot_model.D405_RESOLUTION if sensor else (1280, 720))
    viewport.set_active_camera(camera_path)
