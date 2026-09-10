import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from sim_physics.self_screen import SelfCapsuleScreen


def test_native_adjacent_and_explicit_filters_without_adding_any():
    stage=Usd.Stage.CreateInMemory()
    for name in ('torso','arm0','arm1'):
        body=UsdGeom.Xform.Define(stage,'/R/'+name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        cap=UsdGeom.Capsule.Define(stage,'/R/'+name+'/capsule')
        cap.CreateRadiusAttr(.1);cap.CreateHeightAttr(.2)
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    joint=UsdPhysics.RevoluteJoint.Define(stage,'/R/joint')
    joint.CreateBody0Rel().SetTargets(['/R/torso']);joint.CreateBody1Rel().SetTargets(['/R/arm0'])
    joint.CreateCollisionEnabledAttr(False)
    before=stage.GetRootLayer().ExportToString()
    check=SelfCapsuleScreen(stage,'/R')
    frames={name:np.eye(4) for name in ('torso','arm0','arm1')}
    result=check.check(frames)
    assert not result['passed'] and result['checked_pairs']==2
    assert '/R/arm1/capsule' in result['nearest_pair']
    frames['arm1'][0,3]=1
    assert check.check(frames)['passed']
    assert stage.GetRootLayer().ExportToString()==before
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/R/torso')).CreateFilteredPairsRel().SetTargets(['/R/arm1'])
    assert SelfCapsuleScreen(stage,'/R').check(frames)['checked_pairs']==1


def test_tool_box_screen_detects_camera_finger_collision_missed_by_arm_capsules():
    from pxr import Gf
    stage=Usd.Stage.CreateInMemory()
    for name,x in (('left',0.),('right',2.)):
        body=UsdGeom.Xform.Define(stage,'/R/'+name).GetPrim();UsdPhysics.RigidBodyAPI.Apply(body)
        cap=UsdGeom.Capsule.Define(stage,'/R/'+name+'/arm')
        cap.CreateRadiusAttr(.05);cap.CreateHeightAttr(.1)
        UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(x,0,0));UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    for path,z in (('/R/left/finger',.2),('/R/right/camera',.22)):
        box=UsdGeom.Cube.Define(stage,path);box.CreateSizeAttr(.05)
        UsdGeom.Xformable(box).AddTranslateOp().Set(Gf.Vec3d(0,0,z));UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    before=stage.GetRootLayer().ExportToString();frames={n:np.eye(4) for n in ('left','right')}
    assert SelfCapsuleScreen(stage,'/R').check(frames)['passed']
    screen=SelfCapsuleScreen(stage,'/R',include_tool_boxes=True);result=screen.check(frames)
    assert not result['passed'] and set(result['nearest_pair'])=={'/R/left/finger','/R/right/camera'}
    assert result['all_shape_bounds_screened'] and not result['whole_robot_self_collision_certified']
    frames['right'][0,3]=.2
    assert screen.check(frames)['passed']
    assert stage.GetRootLayer().ExportToString()==before
