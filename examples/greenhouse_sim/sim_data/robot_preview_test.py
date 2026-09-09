"""Preview-only right tool selection preserves the benchmark's source asset."""
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from greenhouse_sim.robot_model import DEFAULT_ASSET
from sim_data.robot_preview import add_robot_preview, configure_right_tool


@pytest.mark.parametrize("tool", ["gripper", "knife_only"])
def test_right_tool_visibility_cameras_and_static_state(tool):
    if not DEFAULT_ASSET.is_file():
        pytest.skip("Generated Model A v1.2 asset required")
    before = DEFAULT_ASSET.read_bytes()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    result = add_robot_preview(stage, gutter_x=-.2, right_tool=tool)
    root = result["root"]
    assert result["right_tool_configuration"] == tool
    assert stage.GetPrimAtPath(root + "/ee_right/attachments/DeleafKnife").IsActive() == (tool == "knife_only")
    for name in ("ee_right", "ee_finger_r1", "ee_finger_r2"):
        prim = stage.GetPrimAtPath(f"{root}/{name}/visuals")
        assert prim.IsActive() == (tool == "gripper")
        if tool == "gripper":
            meshes = [p for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
            assert meshes
            assert all(UsdGeom.Imageable(p).ComputeVisibility() != "invisible" for p in meshes)
    for name in ("ee_left", "ee_finger_l1", "ee_finger_l2"):
        assert stage.GetPrimAtPath(f"{root}/{name}/visuals").IsActive()
    assert len(result["camera_paths"]) == 3
    assert all(stage.GetPrimAtPath(p).IsA(UsdGeom.Camera) for p in result["camera_paths"].values())
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            assert not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if prim.IsA(UsdPhysics.Joint):
            assert not UsdPhysics.Joint(prim).GetJointEnabledAttr().Get()
    assert DEFAULT_ASSET.read_bytes() == before
    # Switching visual configuration also leaves the source asset unchanged.
    configure_right_tool(stage, root, "gripper" if tool == "knife_only" else "knife_only")
    assert DEFAULT_ASSET.read_bytes() == before


def test_unknown_right_tool_rejected_without_any_stage_change():
    stage = Usd.Stage.CreateInMemory()
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError, match="Unknown right tool"):
        configure_right_tool(stage, "/World/RBY1", "invalid")
    assert stage.GetSessionLayer().ExportToString() == before


def test_camera_only_overs_are_preserved_but_defined_robot_is_not_duplicated():
    if not DEFAULT_ASSET.is_file():
        pytest.skip("Generated Model A v1.2 asset required")
    stage = Usd.Stage.CreateInMemory()
    camera_path = "/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera"
    stage.OverridePrim(camera_path).SetCustomDataByKey("saved_exposure_note", "keep")
    before = stage.GetRootLayer().ExportToString()
    stage.SetEditTarget(stage.GetSessionLayer())
    add_robot_preview(stage, gutter_x=-.2, right_tool="gripper")
    assert stage.GetPrimAtPath(camera_path).GetCustomDataByKey("saved_exposure_note") == "keep"
    assert stage.GetRootLayer().ExportToString() == before
    with pytest.raises(ValueError, match="already exists"):
        add_robot_preview(stage, gutter_x=-.2)
