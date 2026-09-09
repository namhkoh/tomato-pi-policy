from copy import deepcopy
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from greenhouse_sim.robot_kinematics import Rby1Kinematics
from greenhouse_sim.robot_model import DEFAULT_ASSET
from sim_data.capture_contract import project
from sim_data.capture_scene import calibration, capture_root, freeze_rigid_bodies, mounted_camera_to_head, plan_head_pose, scene_guard
from sim_data.review_camera import HEAD_CAMERA
from sim_data.robot_preview import add_robot_preview


def test_scene_guard_detects_overlay_pose_visibility_and_binding_changes():
    stage=Usd.Stage.CreateInMemory()
    prim=UsdGeom.Cube.Define(stage,"/World/Cube")
    original=scene_guard(stage)
    prim.AddTranslateOp().Set(Gf.Vec3d(1,0,0))
    moved=scene_guard(stage)
    assert original!=moved
    prim.CreateVisibilityAttr("invisible")
    assert scene_guard(stage)!=moved
    UsdGeom.Xform.Define(stage,"/World/DraftLabels")
    with pytest.raises(ValueError,match="Diagnostic"): scene_guard(stage)


@pytest.mark.skipif(not DEFAULT_ASSET.is_file(),reason="Built v1.2 robot required")
def test_head_pose_uses_actual_mount_and_model_limits_without_camera_teleport():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,"Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    robot=add_robot_preview(stage,gutter_x=-.2,right_tool="gripper")
    original=stage.GetSessionLayer().ExportToString()
    cal=calibration(stage)
    mount=mounted_camera_to_head(stage)
    root=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot["root"]))).T
    model=Rby1Kinematics()
    target=np.array([.03,.4,1.3])
    desired=[310,190]
    pose,error=plan_head_pose(model,robot["pose_degrees"],root,mount,target,cal["intrinsics"],desired)
    assert error<.1
    assert all(pose[k]==v for k,v in robot["pose_degrees"].items() if not k.startswith("head_"))
    camera=root @ model.all_link_transforms(pose)["link_head_2"] @ mount
    actual=project([target],{**cal,"camera_to_world_usd_row_vectors":camera.T.tolist()})[0]
    assert actual["projection_status"]=="in_frame"
    assert np.allclose(actual["pixel_xy"],desired,atol=.01)
    assert stage.GetSessionLayer().ExportToString()==original


def test_calibration_requires_correct_scene_and_camera():
    stage=Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError): calibration(stage)
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,"Z")
    camera=UsdGeom.Camera.Define(stage,HEAD_CAMERA)
    camera.CreateProjectionAttr("orthographic")
    with pytest.raises(ValueError): calibration(stage)


@pytest.mark.parametrize("kind",["animated","rigid"])
def test_static_guard_rejects_dynamic_scenes(kind):
    stage=Usd.Stage.CreateInMemory()
    cube=UsdGeom.Cube.Define(stage,"/World/Cube")
    if kind=="animated":
        cube.AddTranslateOp().Set(Gf.Vec3d(1,0,0),Usd.TimeCode(1))
    else:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateRigidBodyEnabledAttr(True)
    with pytest.raises(ValueError): scene_guard(stage)


def test_freeze_physics_is_session_only_and_preserves_geometry():
    stage=Usd.Stage.CreateInMemory()
    cube=UsdGeom.Cube.Define(stage,"/World/Cube")
    body=UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    original=stage.GetRootLayer().ExportToString()
    assert freeze_rigid_bodies(stage)==["/World/Cube"]
    assert not body.GetRigidBodyEnabledAttr().Get()
    assert stage.GetRootLayer().ExportToString()==original
    assert scene_guard(stage)


def test_capture_wrapper_preserves_source_and_relative_asset_resolution(tmp_path):
    child=Usd.Stage.CreateNew(str(tmp_path/"child.usda"))
    cube=UsdGeom.Cube.Define(child,"/Mesh")
    cube.CreateSizeAttr(.25)
    child.SetDefaultPrim(cube.GetPrim())
    child.GetRootLayer().Save()
    source=Usd.Stage.CreateNew(str(tmp_path/"source.usda"))
    source.DefinePrim("/World/Referenced").GetReferences().AddReference("child.usda")
    source.GetRootLayer().Save()
    original=source.GetRootLayer().ExportToString()
    root=capture_root(tmp_path/"source.usda")
    composed=Usd.Stage.Open(root)
    assert UsdGeom.Cube(composed.GetPrimAtPath("/World/Referenced")).GetSizeAttr().Get()==.25
    composed.SetEndTimeCode(123)
    assert root.anonymous and root.dirty
    assert not source.GetRootLayer().dirty
    assert source.GetRootLayer().ExportToString()==original
