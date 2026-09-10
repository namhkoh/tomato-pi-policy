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
