import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim_physics.tool_bounds import arc_box


def test_sloped_arc_enclosure_removes_empty_box_wedge_not_material():
    points=np.array(list(itertools.product((-.003,.003),(-.004,.004),(-.02,.02))))
    r=Rotation.from_euler('x',30,degrees=True).as_matrix();points=points@r.T
    frame=np.eye(4);frame[:3,:3]=Rotation.from_euler('xyz',[7,21,43],degrees=True).as_matrix()
    frame[:3,3]=[.1,-.2,.3]
    centre,axes,half=arc_box(points,frame)
    world=points@frame[:3,:3].T+frame[:3,3]
    assert np.all(abs((world-centre)@axes)<=half+1e-12)
    assert 8*np.prod(half)<np.prod(np.ptp(points,axis=0))*.5
    np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=1e-12)
    assert np.linalg.det(axes)==pytest.approx(1.)


def test_every_original_arc_partition_is_enclosed_without_stage_or_native_edits():
    from pxr import Usd,UsdGeom
    from greenhouse_sim.robot_model import DEFAULT_ASSET,ROBOT_ROOT
    from sim_physics.arc_contacts import refine_arc_contacts
    from sim_physics.self_screen import SelfCapsuleScreen
    stage=Usd.Stage.Open(str(DEFAULT_ASSET));stage.SetEditTarget(stage.GetSessionLayer())
    result=refine_arc_contacts(stage,ROBOT_ROOT)
    root_before=stage.GetRootLayer().ExportToString();session_before=stage.GetSessionLayer().ExportToString()
    screen=SelfCapsuleScreen(stage,ROBOT_ROOT,include_tool_boxes=True,fit_plate=True)
    by_path={s[0]:s for s in screen.shapes};cache=UsdGeom.XformCache()
    for path in result['collider_paths']:
        prim=stage.GetPrimAtPath(path);s=by_path[path]
        local=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(s[1]))).T)@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
        original=points.copy();centre,axes,half=s[4]
        body_points=points@local[:3,:3].T+local[:3,3]
        assert np.all(abs((body_points-centre)@axes)<=half+1e-10)
        assert np.all(half>0)
        np.testing.assert_array_equal(points,original)
    assert stage.GetRootLayer().ExportToString()==root_before
    assert stage.GetSessionLayer().ExportToString()==session_before


@pytest.mark.parametrize('fault',['nan','flat','scale','reflect','bottom'])
def test_invalid_or_nonrigid_bounds_are_not_silently_repaired(fault):
    p=np.array(list(itertools.product((-.003,.003),(-.01,.01),(-.02,.02))))
    frame=np.eye(4)
    if fault=='nan':p[0,0]=np.nan
    if fault=='flat':p[:,0]=0
    if fault=='scale':frame[0,0]=2
    if fault=='reflect':frame[0,0]=-1
    if fault=='bottom':frame[3,0]=1
    with pytest.raises(ValueError):arc_box(p,frame)
