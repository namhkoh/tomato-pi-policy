import numpy as np
import pytest
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from sim_physics.gutter_instances import batch


def make_stage():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/Source')
    UsdGeom.Cube.Define(stage,'/Source/Visual').CreateSizeAttr(.3)
    for i in range(3):
        path=f'/World/Gutters/Gutter_{i}/Gutter_Module_00'
        mod=UsdGeom.Xform.Define(stage,path)
        mod.GetPrim().GetReferences().AddInternalReference('/Source')
        mod.AddTranslateOp().Set((i*2.,3.,.8));mod.AddRotateZOp().Set(180.)
        mod.GetPrim().SetInstanceable(True)
        collider=UsdGeom.Cube.Define(stage,f'/World/Gutters/Gutter_{i}/Collision')
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    return stage


def test_exact_visual_transforms_colliders_and_session_only():
    stage=make_stage();before=stage.GetRootLayer().ExportToString()
    modules=[p for g in stage.GetPrimAtPath('/World/Gutters').GetChildren() for p in g.GetChildren()
             if p.GetName().startswith('Gutter_Module_')]
    transforms=[UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) for p in modules]
    report=batch(stage)
    inst=UsdGeom.PointInstancer(stage.GetPrimAtPath('/World/GutterVisualInstances'))
    actual=inst.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(),Usd.TimeCode.Default())
    np.testing.assert_allclose(actual,transforms,rtol=0,atol=1e-6)
    assert stage.GetPrimAtPath('/World/GutterVisualInstances/Prototype/Visual').IsActive()
    assert report['collision_proxies_retained']==3 and report['module_count']==3
    assert stage.GetRootLayer().ExportToString()==before
    with pytest.raises(ValueError): batch(stage)


def test_physical_or_animated_modules_are_rejected_without_edits():
    stage=make_stage()
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath('/Source/Visual'))
    before=stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError,match='containing physics'): batch(stage)
    assert stage.GetSessionLayer().ExportToString()==before
    stage=make_stage()
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Gutters/Gutter_0/Gutter_Module_00')).GetOrderedXformOps()[0].Set((0,0,0),1.)
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Gutters/Gutter_0/Gutter_Module_00')).GetOrderedXformOps()[0].Set((1,0,0),2.)
    with pytest.raises(ValueError,match='animated'): batch(stage)
