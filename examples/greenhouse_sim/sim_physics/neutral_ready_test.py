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
