from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
import pytest
from .neutral_ready import initialize,ready_arms,time_path,interpolate


def test_prephysics_initialization_preserves_torso_base_and_goal_without_native_reset():
    from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES as ready
    pose=dict(ready);pose.update({f'torso_{i}':0. for i in range(6)})
    original=dict(pose);base=np.eye(4)
    f=S(initial_q=np.arange(7),right=-np.arange(7),pose=pose,base=base.copy(),restore_authored_state=Mock())
    r=initialize(f)
    f.restore_authored_state.assert_called_once_with()
    assert f.neutral_goal_pose==original and f.pose==original
    np.testing.assert_array_equal(f.base,base)
    np.testing.assert_array_equal(r['goal_arm_degrees'],np.r_[np.arange(7),-np.arange(7)])
    assert not r['execution_passed'] and not r['cut_authorized'] and not r['torso_moved']
    with pytest.raises(RuntimeError):initialize(f)
    with pytest.raises(RuntimeError):initialize(S(robot=Mock()))


def test_time_parameterization_stays_on_path_and_below_fifteen_degrees_per_second():
    path=np.array([ready_arms(),ready_arms()+1,ready_arms()+np.arange(14)/10])
    knots=time_path(path)
    np.testing.assert_array_equal(interpolate(path,knots,0),path[0])
    np.testing.assert_array_equal(interpolate(path,knots,100),path[-1])
    ts=np.linspace(0,knots[-1],10001)
    qs=np.array([interpolate(path,knots,t) for t in ts])
    assert np.max(abs(np.diff(qs,axis=0)/(ts[1]-ts[0])))<=15.+1e-5
    for t,q in zip(ts[::100],qs[::100]):
        i=min(np.searchsorted(knots,t,side='right')-1,len(path)-2)
        assert np.all(q>=np.minimum(path[i],path[i+1])-1e-9)
        assert np.all(q<=np.maximum(path[i],path[i+1])+1e-9)


@pytest.mark.parametrize('speed',[0,16,True,float('nan')])
def test_bad_speeds_fail_closed(speed):
    with pytest.raises(ValueError):time_path(np.zeros((2,14)),speed=speed)


def test_public_option_forwards_but_raw_incomplete_profile_cannot_create_output(tmp_path,monkeypatch):
    from . import benchmark,ground_truth_trial
    with pytest.raises(ValueError,match='Neutral ready'):
        benchmark.main(['--output',str(tmp_path/'rejected'),'--neutral-ready-start'])
    assert not (tmp_path/'rejected').exists()
    calls=[];monkeypatch.setattr(benchmark,'main',lambda a:calls.append(benchmark.parser().parse_args(a)))
    ground_truth_trial.main(['--output',str(tmp_path/'unused'),'--mode','bimanual','--milestone','cut_action',
        '--process-zone-trial','--neutral-ready-start'])
    assert calls[0].neutral_ready_start


def test_failed_prelude_never_enters_grasp_or_cut_controller(tmp_path,monkeypatch):
    from . import neutral_ready,bimanual_probe
    monkeypatch.setattr(neutral_ready,'run',lambda *a:dict(execution_passed=False,error='blocked path'))
    f=Mock()
    result=bimanual_probe.run(None,None,None,None,None,f,S(neutral_ready_start=True),tmp_path)
    assert result['state']=='failed_neutral_ready_approach' and not result['cut_action']['passed']
    f.bind.assert_not_called()


def test_neutral_open_hand_uses_original_pd_ceiling_not_total_motor_budget():
    from .neutral_ready import apply_open_finger_effort
    f=S(force_limits=np.array([[.8,.2,10.]]),finger_indices=[0,1],
        force_closer=S(drive_limit_n=.3),robot=Mock(),index=[0],finger_effort=Mock())
    apply_open_finger_effort(f,step=0,dt=1/480)
    np.testing.assert_array_equal(f.force_limits,[[.3,.2,10.]])
    f.finger_effort.apply.assert_called_once_with(f,step=0,dt=1/480)


def test_contact_failure_receipt_preserves_paths_and_normal_plus_friction():
    from .neutral_ready import contact_diagnostic
    pair=('/World/Robot/Wrist','/World/Target/Leaf')
    m=S(pairs={pair:.006/480},normal_pairs={pair:.004/480},friction_pairs={pair:.002/480})
    rows=contact_diagnostic(m,1/480)
    assert rows==[dict(collider0=pair[0],collider1=pair[1],
        normal_upper_bound_n=.004,friction_upper_bound_n=.002,total_upper_bound_n=.006)]
    assert m.pairs[pair]==.006/480
    with pytest.raises(ValueError):contact_diagnostic(m,0)


def test_invalid_native_contact_load_cannot_become_valid_diagnostic():
    from .neutral_ready import contact_diagnostic
    pair=('/World/R/Wrist','/World/P/Leaf')
    m=S(pairs={pair:float('nan')},normal_pairs={pair:0.},friction_pairs={pair:0.})
    with pytest.raises(ValueError):contact_diagnostic(m,1/480)


def run_mock_transit(tmp_path,monkeypatch,*,contact_fault=False):
    """Controller ordering only: not a native motion/physics qualification."""
    from . import neutral_ready as n,runtime as rt
    frames=np.repeat(np.eye(4)[None],2,axis=0);actual=np.radians(ready_arms())[None].copy()
    target=actual.copy();sim=S(current_time=0.,physics_sim_view=object(),is_playing=lambda:True)
    f=S(neutral_ready={},bind=Mock(),release_grasp_observer=Mock(),cut_authorized=False,
        left_indices=list(range(7)),right_indices=list(range(7,14)),targets=target,
        expected_right=np.eye(4),base=np.eye(4),stop_requested=False,
        explicit_finger_effort=False,prepare_step=Mock(),on_sample=Mock(),
        cut_event=None,neutral_goal_pose={'marker':'goal'},pose={'marker':'neutral'},
        restore_authored_state=Mock(),
        check_self=lambda *a:dict(passed=True),
        kin=S(forward=lambda *a:np.eye(4),inter_arm_clearance=lambda *a:S(clearance_m=.02)))
    f.robot=S(get_dof_positions=lambda:actual.copy(),get_dof_velocities=lambda:np.zeros_like(actual))
    f.palm=f.right_palm=S(get_transforms=lambda:np.eye(4)[None])
    f.event_monitor=S(begin_step=Mock(),pairs={},normal_pairs={},friction_pairs={})
    f._command_left_drives=lambda q,p:target.__setitem__((0,f.left_indices),np.radians(q))
    def step(**kw):
        actual[:]=target;sim.current_time+=1/480
    sim.step=step
    def metrics(*a):
        load=.006 if contact_fault and sim.current_time>1.02 else 0.
        pair=('/World/Robot/Wrist','/World/Target/Leaf')
        f.event_monitor.pairs={pair:load/480}
        f.event_monitor.normal_pairs={pair:load/960}
        f.event_monitor.friction_pairs={pair:load/960}
        return dict(per_finger_contact_upper_bound_n={'left1':0.,'left2':0.},
            allowed_tool_contact_n=0.,unwanted_contact_n=load)
    f.check=metrics;times=[]
    def planner(fixture,seen,start):
        times.append(sim.current_time)
        np.testing.assert_array_equal(seen,frames)
        return np.array([start,ready_arms()+.02]),dict(passed=True)
    monkeypatch.setattr(n,'plan',planner);monkeypatch.setattr(rt,'pose_matrices',lambda x:x)
    runtime=S(frames=frames,sample=lambda:(frames,np.zeros((2,6))),sync_visuals=Mock())
    result=n.run(S(is_running=lambda:True),sim,S(cut=False,cut_index=1,rest_frames=frames),
        runtime,S(step=Mock()),f,S(physics_hz=480,render_hz=0,coupled_fingers_trial=False),tmp_path)
    f.restore_authored_state.assert_not_called()
    return result,times,f


def test_neutral_path_is_planned_after_native_hold_then_endpoint_rechecked(tmp_path,monkeypatch):
    r,times,f=run_mock_transit(tmp_path,monkeypatch)
    assert r['execution_passed'] and r['reobserved_after_initial_hold']
    assert r['motion_planning']['passed'] and r['endpoint_screen']['passed']
    assert len(times)==2 and times[0]>=1.-1e-10 and times[1]>times[0]
    assert r['last_sample']['native_guards_passed'] and not r['native_state_reset']
    assert f.pose==f.neutral_goal_pose and not f.cut_authorized


def test_rejected_neutral_contact_sample_is_saved_without_cut_continuation(tmp_path,monkeypatch):
    r,times,f=run_mock_transit(tmp_path,monkeypatch,contact_fault=True)
    assert not r['execution_passed'] and 'Unexpected plant/tool contact' in r['error']
    assert len(times)==1 and not r['last_sample']['native_guards_passed']
    assert r['last_sample']['robot']['unwanted_contact_n']==.006
    assert r['last_sample']['contact_pairs'][0]['total_upper_bound_n']==.006
    assert f.pose=={'marker':'neutral'} and not f.cut_authorized
