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
