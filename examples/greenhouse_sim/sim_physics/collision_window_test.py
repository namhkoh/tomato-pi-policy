import numpy as np
import pytest
from sim_physics.collision_window import intersects_xy,inside_window


def test_bounds_not_origins_decide_collision_retention():
    assert intersects_xy([-4,0,10],[4,1,20],[0,0],2)
    assert intersects_xy([2,-1,0],[3,1,2],[0,0],2)
    assert not intersects_xy([2.01,-1,0],[3,1,2],[0,0],2)
    with pytest.raises(ValueError): intersects_xy([float('nan'),0,0],[3,1,2],[0,0],2)


def test_body_radius_and_predictive_margin_are_guarded():
    assert inside_window([[1,0,30]],[.5],[0,0],2)
    assert not inside_window([[1.4,0,1]],[.5],[0,0],2)
    assert not inside_window([[0,-1.9,1]],[0],[0,0],2)
    with pytest.raises(ValueError): inside_window([[0,0,0]],[-1],[0,0],2)


def test_hidden_collider_radius_is_not_zero():
    from pxr import Usd,UsdGeom
    from sim_physics.collision_window import body_radii
    s=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s,'/Body')
    box=UsdGeom.Cube.Define(s,'/Body/Shape');box.CreateSizeAttr(1.)
    box.CreateVisibilityAttr('invisible')
    box.CreatePurposeAttr('guide')
    np.testing.assert_allclose(body_radii(s,['/Body'],['/Body/Shape']),[np.sqrt(3)/2])
    np.testing.assert_allclose(body_radii(s,['/Body/Shape'],['/Body/Shape']),[np.sqrt(3)/2])


def test_tighter_window_checks_full_initial_bounds_before_any_scene_edit():
    from types import SimpleNamespace
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from sim_physics.collision_window import configure
    s=Usd.Stage.CreateInMemory()
    def shape(path,x):
        cube=UsdGeom.Cube.Define(s,path);cube.CreateSizeAttr(.2)
        cube.AddTranslateOp().Set(Gf.Vec3d(x,0,0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube
    UsdGeom.Xform.Define(s,'/World/GutterWires')
    wire=shape('/World/GutterWires/Far',3);wire.CreatePurposeAttr('guide')
    visual=UsdGeom.Cube.Define(s,'/World/GutterWires/VisibleWire')
    UsdGeom.Xform.Define(s,'/Robot')
    shape('/Robot/Collision',1.4)
    UsdGeom.Xform.Define(s,'/Plant')
    shape('/Plant/Collision',0)
    r=SimpleNamespace(base=np.eye(4),body_paths=['/Robot'],collider_paths=['/Robot/Collision'],
        rig=SimpleNamespace(root='/Plant',body_paths=['/Plant']))
    before=s.GetRootLayer().ExportToString();session=s.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError,match='Initial complete'):
        configure(s,r,half_extent=1.)
    assert s.GetRootLayer().ExportToString()==before
    assert s.GetSessionLayer().ExportToString()==session
    assert wire.GetPrim().IsActive()
    report=configure(s,r,half_extent=2.)
    assert report['initial_full_collision_bounds_checked_before_culling']
    assert visual.GetPrim().IsActive() and not wire.GetPrim().IsActive()
    assert s.GetRootLayer().ExportToString()==before
