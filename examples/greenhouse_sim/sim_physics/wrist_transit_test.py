from types import SimpleNamespace as S
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim_physics.wrist_transit import frames,screen_modes,try_modes,MODES,LIFT_OFFSETS,RETREAT_OFFSETS,GOAL_SIDE_OFFSETS
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
    if mode!='simultaneous' and mode not in LIFT_OFFSETS and mode not in RETREAT_OFFSETS and mode not in GOAL_SIDE_OFFSETS:
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
    assert modes==tuple(m for m in MODES if m!='simultaneous' and m not in LIFT_OFFSETS and m not in RETREAT_OFFSETS and m not in GOAL_SIDE_OFFSETS)
    assert len(seen)==len(MODES)
    assert not evidence['simultaneous']['passed']
    assert not evidence['orient_then_translate']['motion_authorized']


@pytest.mark.parametrize('mode',LIFT_OFFSETS)
def test_lift_precedes_rotation_then_translation_and_final_descent(mode):
    a,b=endpoints();path=frames(a,b,mode=mode)
    height=max(a[2,3],b[2,3])+LIFT_OFFSETS[mode]
    changed=np.max(abs(path[:,:3,:3]-a[:3,:3]),axis=(1,2))>1e-9
    first=int(np.flatnonzero(changed)[0])
    np.testing.assert_allclose(path[:first,:3,:3],np.broadcast_to(a[:3,:3],(first,3,3)),atol=1e-12)
    assert path[first,2,3]==pytest.approx(height)
    np.testing.assert_allclose(path[:first,:2,3],np.broadcast_to(a[:2,3],(first,2)),atol=1e-12)
    descending=np.flatnonzero(np.diff(path[:,2,3])<-1e-12)
    assert len(descending)
    np.testing.assert_allclose(path[descending,:2,3],np.broadcast_to(b[:2,3],(len(descending),2)),atol=1e-12)
    np.testing.assert_allclose(path[descending,:3,:3],np.broadcast_to(b[:3,:3],(len(descending),3,3)),atol=1e-12)


def test_lift_collision_is_rejected_without_a_stroke_allowance():
    a,b=endpoints()
    def check(path,*,stroke):
        assert stroke is False
        return dict(passed=bool(np.max(path[:,2,3])<.05),motion_authorized=False)
    modes,evidence=screen_modes(S(check=check),a,b)
    for mode in LIFT_OFFSETS:
        assert mode not in modes and not evidence[mode]['passed']


@pytest.mark.parametrize('mode',GOAL_SIDE_OFFSETS)
def test_final_side_entry_is_horizontal_and_aligned(mode):
    a,b=endpoints();path=frames(a,b,mode=mode)
    count=int(np.ceil(abs(GOAL_SIDE_OFFSETS[mode])/.002))+1
    np.testing.assert_allclose(path[-count:,2,3],b[2,3],atol=1e-12)
    np.testing.assert_allclose(path[-count:,:3,:3],np.broadcast_to(b[:3,:3],(count,3,3)),atol=1e-12)
    side=b[:3,0].copy();side[2]=0;side/=np.linalg.norm(side)
    np.testing.assert_allclose(path[-count,:3,3],b[:3,3]+GOAL_SIDE_OFFSETS[mode]*side,atol=1e-12)


@pytest.mark.parametrize('mode',RETREAT_OFFSETS)
def test_retreat_moves_away_unrotated_before_any_lift(mode):
    a,b=endpoints();path=frames(a,b,mode=mode);distance,height=RETREAT_OFFSETS[mode]
    rising=np.flatnonzero(path[:,2,3]>a[2,3]+1e-12)[0]
    away=a[:2,3]-b[:2,3];away/=np.linalg.norm(away)
    np.testing.assert_allclose(path[rising-1,:2,3],a[:2,3]+distance*away,atol=1e-12)
    np.testing.assert_allclose(path[:rising,:3,:3],np.broadcast_to(a[:3,:3],(rising,3,3)),atol=1e-12)
    assert max(path[:,2,3])==pytest.approx(max(a[2,3],b[2,3])+height)


def test_lazy_preview_retains_full_arm_fallbacks_and_stops_only_on_full_success():
    a,b=endpoints();tested=[];accepted=[];evidence={}
    def check(path,*,stroke):
        assert not stroke;tested.append(path)
        return dict(passed=True)
    def accept(mode):
        accepted.append(mode)
        return len(accepted)==3  # First two fail full-arm validation.
    assert try_modes(S(check=check),a,b,accept,evidence)
    assert accepted==list(MODES[:3]) and list(evidence)==list(MODES[:3])
    assert len(tested)==3


def test_lazy_rejected_preview_never_reaches_ik_and_errors_propagate():
    def accept(mode):raise AssertionError('Rejected preview may not authorize IK/path')
    assert not try_modes(S(check=lambda *a,**k:dict(passed=False)),*endpoints(),accept,{})
    def fail(*a,**kw):raise RuntimeError('native epoch changed')
    with pytest.raises(RuntimeError):try_modes(S(check=fail),*endpoints(),accept,{})


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
