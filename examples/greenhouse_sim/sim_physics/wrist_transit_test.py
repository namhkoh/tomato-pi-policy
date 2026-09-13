from types import SimpleNamespace as S
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim_physics.wrist_transit import frames,screen_modes,MODES
from sim_physics.downward_cut import cartesian_transit


def endpoints():
    a=np.eye(4);b=np.eye(4);b[:3,3]=[.03,-.04,.012]
    b[:3,:3]=Rotation.from_euler('z',70,degrees=True).as_matrix()
    return a,b


@pytest.mark.parametrize('mode',MODES)
def test_exact_endpoints_bounded_translation_rotation_and_no_mutation(mode):
    a,b=endpoints();original=np.array([a,b]);path=frames(a,b,mode=mode)
    np.testing.assert_array_equal(path[0],a);np.testing.assert_array_equal(path[-1],b)
    np.testing.assert_array_equal([a,b],original)
    assert np.max(np.linalg.norm(np.diff(path[:,:3,3],axis=0),axis=1))<=.002+1e-12
    turns=Rotation.from_matrix(path[1:,:3,:3]@path[:-1,:3,:3].transpose(0,2,1)).magnitude()
    assert np.max(turns)<=np.radians(1)+1e-12
    if mode!='simultaneous':
        moving=np.linalg.norm(path[:,:3,3]-a[:3,3],axis=1)>1e-12
        np.testing.assert_allclose(path[moving,:3,:3],np.broadcast_to(b[:3,:3],path[moving,:3,:3].shape),atol=1e-12)


@pytest.mark.parametrize('zero',['rotation','translation','both'])
def test_degenerate_phases_do_not_divide_by_zero(zero):
    a,b=endpoints()
    if zero in ('rotation','both'):b[:3,:3]=a[:3,:3]
    if zero in ('translation','both'):b[:3,3]=a[:3,3]
    for mode in MODES:
        path=frames(a,b,mode=mode)
        assert len(path)>=2 and np.isfinite(path).all()
        np.testing.assert_array_equal(path[-1],b)


@pytest.mark.parametrize('fault',['mode','nan','scale','reflection','bottom','unbounded'])
def test_unknown_or_nonrigid_proposal_refused(fault):
    a,b=endpoints();mode='simultaneous'
    if fault=='mode':mode='teleport'
    if fault=='nan':b[0,3]=np.nan
    if fault=='scale':a[0,0]=2
    if fault=='reflection':a[0,0]=-1
    if fault=='bottom':a[3,1]=1
    if fault=='unbounded':b[0,3]=3
    with pytest.raises(ValueError):frames(a,b,mode=mode)


def test_rigid_preview_can_select_orientation_first_without_allowing_seam_contact():
    a,b=endpoints();seen=[]
    def check(path,*,stroke):
        assert stroke is False;seen.append(path)
        # Synthetic obstacle intersects simultaneous motion only. This is not
        # evidence about a real plant: the native screen supplies actual checks.
        translated=np.linalg.norm(path[:,:3,3]-a[:3,3],axis=1)>1e-12
        aligned=np.max(abs(path[:,:3,:3]-b[:3,:3]),axis=(1,2))<1e-10
        return dict(passed=not np.any(translated&~aligned),motion_authorized=False)
    modes,evidence=screen_modes(S(check=check),a,b)
    assert modes==MODES[1:] and len(seen)==len(MODES)
    assert not evidence['simultaneous']['passed']
    assert not evidence['orient_then_translate']['motion_authorized']


def test_query_error_cannot_become_a_clear_fallback():
    def fail(*a,**kw):raise RuntimeError('stale native query')
    with pytest.raises(RuntimeError,match='stale'):screen_modes(S(check=fail),*endpoints())


def robot_fixture():
    def fk(side,q,base):
        f=np.eye(4);f[0,3]=q[0];f[:3,:3]=Rotation.from_euler('z',q[1],degrees=True).as_matrix()
        return f
    def ik(desired,seed):
        q=seed.copy();q[0]=desired[0,3]
        q[1]=Rotation.from_matrix(desired[:3,:3]).as_euler('xyz',degrees=True)[2]
        return S(succeeded=True,joint_degrees=q)
    calls=[]
    robot=S(right=np.zeros(7),base=np.eye(4),solve_right_pose=ik,
        kin=S(forward=fk,inter_arm_clearance=lambda *a:S(clearance_m=.1)),
        check_self=lambda *a:dict(passed=True),
        check_held_plant=lambda left,q:calls.append(q.copy()) or True)
    return robot,calls


def test_actual_staged_ik_path_checks_orientation_phase_and_translation():
    robot,calls=robot_fixture();goal=np.array([.01,30,0,0,0,0,0])
    path,clearance,evidence=cartesian_transit(robot,np.zeros(7),goal,
        replan_stroke_from_endpoint=True,mode='orient_then_translate')
    np.testing.assert_allclose(path[-1],goal,atol=1e-12)
    assert any(q[0]==0 and q[1]>0 for q in calls)
    assert all(abs(q[1]-30)<1e-10 for q in calls if q[0]>1e-12)
    assert evidence['method']=='orient_then_straight_cartesian_no_detour'
    assert not evidence['whole_scene_certified']
    np.testing.assert_array_equal(robot.right,np.zeros(7))


def test_collision_during_rotation_refuses_entire_actual_path():
    robot,calls=robot_fixture();robot.check_held_plant=lambda left,q:q[1]<10
    detail={}
    assert cartesian_transit(robot,np.zeros(7),np.array([.01,30,0,0,0,0,0]),
        mode='orient_then_translate',diagnostics=detail) is None
    assert detail['reason']=='plant_or_scene_clearance' and not detail['motion_authorized']


def test_staging_cannot_be_enabled_on_unqualified_cli_fixture(tmp_path):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError,match='complete isolated downward'):
        main(['--output',str(tmp_path/'none'),'--staged-downward-transit'])
    assert not (tmp_path/'none').exists()
