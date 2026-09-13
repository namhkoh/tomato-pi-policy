from types import SimpleNamespace as S
import numpy as np
import pytest
from .left_ready_park import initialize,stationary_path
from .startup_station_search_test import Robot


def robot():
    r=Robot();r.cut_strategy='right_only';r.park_left_ready=True
    def forbidden(*a,**k):raise AssertionError('No left grasp IK may run in task-independent park')
    r.kin.solve_pose=forbidden
    initialize(r)
    return r


def test_no_target_or_ik_required_to_initialize_and_keep_left_stationary():
    r=robot();original=r.initial_q.copy()
    assert r.pregrasp_ik_attempts==0
    from .full_robot import FullRobotGripper
    FullRobotGripper.plan_approach(r)
    np.testing.assert_array_equal(r.path_q,[original,original])
    np.testing.assert_array_equal(r.start,r.goal)
    assert r.minimum_interarm==.02


def test_park_is_not_available_for_bimanual_grasp():
    r=Robot();r.cut_strategy='bimanual'
    with pytest.raises(ValueError,match='right-only'):initialize(r)
    with pytest.raises(ValueError,match='bimanual'):stationary_path(r)


def test_park_does_not_disable_original_interarm_or_tracking_guards():
    r=robot();r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.009)
    with pytest.raises(RuntimeError,match='inter-arm'):stationary_path(r)
    r=robot();r.goal[0,3]+=.001
    with pytest.raises(ValueError,match='pose changed'):stationary_path(r)


def test_exact_source_limit_reserve_remains_required():
    r=Robot();r.cut_strategy='right_only'
    r.kin.arm_limits_degrees=lambda *a:(np.full(7,-120.),np.full(7,180.))
    with pytest.raises(ValueError,match='source joint limits'):initialize(r)


def test_cut_orbit_uses_same_park_joint_configuration_at_new_station():
    from .cut_station_orbit_test import fixture
    from .cut_station_orbit import search
    r,poses=fixture();r.cut_strategy='right_only';r.park_left_ready=True
    initialize(r);calls=[]
    def native(world,shapes):
        calls.append(world);np.testing.assert_array_equal(world['left'],r.initial_q)
        assert shapes==r.self_screen.shapes
        return dict(passed=True)
    out=search(r,S(check=native),lambda:None)
    assert len(calls)==2 and out['proposed_station']
    assert not any(arm=='left' for arm,pose in poses)
    assert out['cut_entry_checked_with_left']=='SDK_ready_park'
    assert not out['motion_authorized']


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_public_option_validation_before_launch(tmp_path,monkeypatch,mode):
    from pathlib import Path
    from .ground_truth_trial import main
    class ReachedLaunch(Exception):pass
    def stop(*a,**k):raise ReachedLaunch()
    monkeypatch.setattr(Path,'mkdir',stop)
    expected=ReachedLaunch if mode=='right_only' else ValueError
    with pytest.raises(expected):
        main(['--output',str(tmp_path/'unused'),'--mode',mode,'--milestone','cut_action',
              '--process-zone-trial','--park-left-ready'])
