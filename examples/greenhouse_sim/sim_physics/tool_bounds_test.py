import itertools
import numpy as np
import pytest
from sim_physics.tool_bounds import plate_box
from sim_physics.downward_cut import downward_angles,downward_direction


def test_slanted_plate_box_encloses_all_source_points_and_removes_empty_wedge():
    points=np.array(list(itertools.product((-.014,.014),(-.03,.03),(-.003,.003))))
    angle=np.radians(2);r=np.array([[np.cos(angle),-np.sin(angle),0],
        [np.sin(angle),np.cos(angle),0],[0,0,1.]])
    points=points@r.T
    frame=np.eye(4);frame[:3,3]=[.4,.5,.6]
    centre,axes,half=plate_box(points,frame)
    assert np.all(abs((points+frame[:3,3]-centre)@axes)<=half)
    assert np.prod(2*half)<np.prod(np.ptp(points,axis=0))
    np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=1e-12)
    assert np.linalg.det(axes)==pytest.approx(1)


def test_source_blade_bound_encloses_original_convex_mesh_without_editing():
    from pxr import UsdGeom
    # Reuse the source asset directly: no need to solve an unrelated fixture IK.
    from greenhouse_sim.robot_model import DEFAULT_ASSET,ROBOT_ROOT
    from pxr import Usd
    from sim_physics.blade_contacts import refine_blade_contacts
    from sim_physics.knife import mount_forward
    from sim_physics.self_screen import SelfCapsuleScreen
    s=Usd.Stage.Open(str(DEFAULT_ASSET));s.SetEditTarget(s.GetSessionLayer())
    mount_forward(s,ROBOT_ROOT,alignment='camera');refine_blade_contacts(s,ROBOT_ROOT)
    before=s.GetSessionLayer().ExportToString()
    screen=SelfCapsuleScreen(s,ROBOT_ROOT,include_tool_boxes=True,fit_plate=True)
    path=ROBOT_ROOT+'/ee_right/attachments/DeleafKnife/BladePlateContact'
    value=next(v for v in screen.shapes if v[0]==path)
    c=UsdGeom.XformCache();local=np.linalg.inv(np.asarray(c.GetLocalToWorldTransform(s.GetPrimAtPath(value[1]))).T)@np.asarray(c.GetLocalToWorldTransform(s.GetPrimAtPath(path))).T
    points=np.asarray(UsdGeom.Mesh.Get(s,path).GetPointsAttr().Get())@local[:3,:3].T+local[:3,3]
    centre,axes,half=value[4]
    assert np.all(abs((points-centre)@axes)<=half)
    assert s.GetSessionLayer().ExportToString()==before


@pytest.mark.parametrize('axis',[[1,0,0],[.6638,.6961,.2735],[.9,0,.435]])
def test_downward_fan_never_leaves_downward_cone(axis):
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis);d=downward_direction(axis)
    angles=downward_angles(axis)
    assert angles[0]==0 and len(angles)>1
    for a in angles:
        candidate=d*np.cos(np.radians(a))+np.cross(axis,d)*np.sin(np.radians(a))
        assert candidate[2]<=-np.cos(np.pi/6)+1e-12
        assert abs(candidate@axis)<1e-12


def test_bad_source_or_frame_cannot_produce_a_clearance_bound():
    with pytest.raises(ValueError):plate_box(np.zeros((4,3)),np.diag([2,1,1,1]))
