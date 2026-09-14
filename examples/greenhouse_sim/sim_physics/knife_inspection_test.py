from types import SimpleNamespace
import numpy as np
import pytest
from pxr import Gf,Usd,UsdGeom
from .knife_inspection import viewpoints,refresh


def fixture():
    stage=Usd.Stage.CreateInMemory()
    for path in viewpoints(np.eye(4)):
        camera=UsdGeom.Camera.Define(stage,path)
        camera.AddTransformOp().Set(Gf.Matrix4d(1))
    robot=UsdGeom.Camera.Define(stage,'/World/Robot/HeadD405')
    robot.AddTransformOp().Set(Gf.Matrix4d(1))
    body=SimpleNamespace(prim_paths=['/World/Robot/ee_right'],count=1,
        get_transforms=lambda:np.array([[.03,.56,1.41,0,0,0,1]]))
    return SimpleNamespace(stage=stage,root='/World/Robot',right_palm=body,
        knife=SimpleNamespace(frame=lambda value:value.copy()))


def test_current_native_edge_centred_and_robot_camera_unchanged():
    f=fixture();robot=f.stage.GetPrimAtPath('/World/Robot/HeadD405')
    before=UsdGeom.Xformable(robot).GetLocalTransformation()
    result=refresh(f)
    for path in result['camera_paths']:
        world=np.asarray(UsdGeom.Xformable(f.stage.GetPrimAtPath(path)).GetLocalTransformation()).T
        ray=np.array([.03,.56,1.41])-world[:3,3];ray/=np.linalg.norm(ray)
        assert np.allclose(-world[:3,2],ray)
    assert UsdGeom.Xformable(robot).GetLocalTransformation()==before
    assert not result['robot_cameras_changed']


def test_invalid_native_identity_never_writes_stage():
    f=fixture();f.right_palm.prim_paths=['/World/Robot/ee_left']
    before=f.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):refresh(f)
    assert f.stage.GetRootLayer().ExportToString()==before


def test_validate_all_camera_transforms_before_any_write():
    f=fixture();paths=list(viewpoints(np.eye(4)))
    UsdGeom.Xformable(f.stage.GetPrimAtPath(paths[-1])).AddTranslateOp()
    before=f.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):refresh(f)
    assert f.stage.GetRootLayer().ExportToString()==before
