import numpy as np
import pytest
from sim_physics.downward_cut import downward_direction,arm_extension


def test_downward_is_transverse_and_never_upward():
    axis=np.array([.6638,.6961,.2735]);axis/=np.linalg.norm(axis)
    d=downward_direction(axis)
    assert np.linalg.norm(d)==pytest.approx(1)
    assert abs(d@axis)<1e-12 and d[2]<-.96
    np.testing.assert_allclose(d,downward_direction(-axis),atol=1e-12)


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('tilt',[-15,-10,0,10,15])
def test_vertical_proposal_uses_real_stem_for_unchanged_angular_limits(sign,tilt):
    from sim_physics.downward_cut import vertical_cut_frame
    axis=np.array([.6638,.6961,.2735]);axis/=np.linalg.norm(axis)
    original=axis.copy()
    d,n=vertical_cut_frame(axis,sign,tilt)
    np.testing.assert_array_equal(axis,original)
    np.testing.assert_array_equal(d,[0,0,-1])
    frame=np.column_stack([-d,np.cross(n,-d),n])
    np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-12)
    assert np.linalg.det(frame)==pytest.approx(1)
    assert abs(d@axis)<.3 and abs(frame[:,1]@axis)<.3
    assert abs(d@axis)>.27  # No falsely transverse, substituted anatomical axis.


@pytest.mark.parametrize('z',[.3,-.3,.435,-.435,1.,-1.])
def test_vertical_proposal_rejects_existing_gate_boundary_or_steeper(z):
    from sim_physics.downward_cut import vertical_cut_frame
    assert vertical_cut_frame([np.sqrt(1-z*z),0,z]) is None


@pytest.mark.parametrize('axis',[[0,0,0],[1,0,1],[float('nan'),0,0],[1,0]])
def test_vertical_proposal_rejects_invalid_anatomy(axis):
    from sim_physics.downward_cut import vertical_cut_frame
    with pytest.raises(ValueError):vertical_cut_frame(axis)


@pytest.mark.parametrize('axis',[[0,0,1],[0,0,-1],[1,0,1],[0,0,0],[float('nan'),0,0]])
def test_unsuitable_or_invalid_stem_does_not_fall_back_to_side_cut(axis):
    with pytest.raises(ValueError):downward_direction(axis)


def test_extension_uses_geometry_not_a_joint_angle_assumption():
    frames={'link_right_arm_'+str(i):np.eye(4) for i in (2,3,4)}
    frames['link_right_arm_3'][:3,3]=[.3,0,0]
    frames['link_right_arm_4'][:3,3]=[.6,0,0]
    assert arm_extension(frames)==pytest.approx(1)
    frames['link_right_arm_4'][:3,3]=[.3,.3,0]
    assert arm_extension(frames)==pytest.approx(1/np.sqrt(2))


def test_camera_aligned_mount_keeps_flange_camera_and_source_unchanged():
    from pxr import Usd,UsdGeom
    from greenhouse_sim.robot_model import DEFAULT_ASSET,ROBOT_ROOT
    from sim_physics.knife import mount_forward
    s=Usd.Stage.Open(str(DEFAULT_ASSET));s.SetEditTarget(s.GetSessionLayer())
    source=s.GetRootLayer().ExportToString();c=UsdGeom.XformCache()
    wrist=s.GetPrimAtPath(ROBOT_ROOT+'/ee_right')
    inverse=np.linalg.inv(np.asarray(c.GetLocalToWorldTransform(wrist)).T)
    camera=s.GetPrimAtPath(str(wrist.GetPath())+'/attachments/RightWristCamera')
    before=np.asarray(c.GetLocalToWorldTransform(camera)).copy()
    root=s.GetPrimAtPath(str(wrist.GetPath())+'/attachments/DeleafKnife')
    flange=(inverse@np.asarray(c.GetLocalToWorldTransform(root)).T)[:3,3]
    result=mount_forward(s,ROBOT_ROOT,alignment='camera');c.Clear()
    rotation=(inverse@np.asarray(c.GetLocalToWorldTransform(root)).T)[:3,:3]
    # Source +Z is the arc's radial side. Source +Y points back to the wrist.
    np.testing.assert_allclose(rotation[:,2],[0,-1,0],atol=1e-10)
    np.testing.assert_allclose(rotation[:,1],[0,0,1],atol=1e-10)
    np.testing.assert_allclose((inverse@np.asarray(c.GetLocalToWorldTransform(root)).T)[:3,3],flange)
    np.testing.assert_array_equal(np.asarray(c.GetLocalToWorldTransform(camera)),before)
    assert result['additional_roll_from_legacy_degrees']==pytest.approx(-90)
    assert not mount_forward(s,ROBOT_ROOT,alignment='camera')['changed']
    assert mount_forward(s,ROBOT_ROOT,alignment='legacy')['changed']
    assert mount_forward(s,ROBOT_ROOT,alignment='camera')['changed']
    assert s.GetRootLayer().ExportToString()==source


def test_new_style_rejects_legacy_proposals_before_launch(tmp_path):
    from sim_physics.benchmark import main
    for extra in (['--cut-proposal-json','unused.json'],['--right-ik-fixed-joint','1','-.75']):
        with pytest.raises(ValueError,match='cannot reuse'):
            main(['--output',str(tmp_path/'none'),'--bimanual-cut','--cut-style','downward',*extra])
    assert not (tmp_path/'none').exists()


def test_straight_transit_screens_every_sample_and_has_no_detour_fallback():
    from types import SimpleNamespace as S
    from sim_physics.downward_cut import cartesian_transit
    def fk(side,q,base):
        frame=np.eye(4);frame[0,3]=q[0];return frame
    calls=[]
    def solve(desired,seed):
        q=np.array(seed,copy=True);q[0]=desired[0,3]
        return S(succeeded=True,joint_degrees=q)
    robot=S(right=np.zeros(7),base=np.eye(4),solve_right_pose=solve,
        kin=S(forward=fk,inter_arm_clearance=lambda *a:S(clearance_m=.1)),
        check_self=lambda *a:dict(passed=True),
        check_held_plant=lambda left,q:(calls.append(q.copy()) or True))
    goal=np.array([.01,0,0,0,0,0,0])
    path,clearance,evidence=cartesian_transit(robot,np.zeros(7),goal)
    np.testing.assert_allclose(path[:,0],np.linspace(0,.01,6))
    assert len(calls)==5 and clearance==.1
    assert evidence['method']=='straight_cartesian_no_detour'
    robot.check_held_plant=lambda left,q:q[0]<.005
    assert cartesian_transit(robot,np.zeros(7),goal) is None


def test_exact_material_grasp_tracks_rotation_not_segment_centre():
    from sim_physics.full_robot import FullRobotGripper
    r=FullRobotGripper.__new__(FullRobotGripper);r.body_index=1;r.grasp_offset_m=-.011
    frames=np.repeat(np.eye(4)[None],3,axis=0)
    frames[1,:3,3]=[1,2,3]
    np.testing.assert_allclose(r.grasp_point(frames),[1,2,2.989])
    frames[1,:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    np.testing.assert_allclose(r.grasp_point(frames),[.989,2,3])
    r.grasp_offset_m=0
    np.testing.assert_array_equal(r.grasp_point(frames),frames[1,:3,3])


def test_exact_grasp_restricted_before_kit_start(tmp_path):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError,match='Exact grasp arc requires'):
        main(['--output',str(tmp_path/'none'),'--exact-grasp-arc'])
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('style', ['legacy','downward'])
def test_downward_endpoint_and_all_fallback_seeds_keep_joint_reserve(style):
    from types import SimpleNamespace as S
    from sim_physics.bimanual import BimanualRobot
    calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs)
        return S(succeeded=False)
    robot=BimanualRobot.__new__(BimanualRobot)
    robot.base=np.eye(4);robot.cut_style=style;robot.right_ik_fixed_joint=None
    robot.kin=S(solve_pose=solve)
    robot.solve_right_pose(np.eye(4),np.zeros(7))
    assert len(calls)==(4 if style=='downward' else 1)
    assert all(c.get('joint_limit_margin_degrees',0.)==(3. if style=='downward' else 0.) for c in calls)
