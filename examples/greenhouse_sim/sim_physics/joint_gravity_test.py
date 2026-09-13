from types import SimpleNamespace as S
import numpy as np
import pytest
from .joint_gravity import allocate


def test_source_budget_excludes_wheels_and_fingers_not_missing_arm_limits():
    from .joint_gravity import source_limits
    names=['left_wheel','right_wheel','torso_0','left_arm_0','right_arm_0',
           'gripper_finger_l1','gripper_finger_l2','head_0']
    effort=dict(torso_0=100,left_arm_0=80,right_arm_0=80,head_0=5)
    indices,limits=source_limits(names,effort)
    assert indices==[2,3,4,7]
    np.testing.assert_array_equal(limits,[100,80,80,5])
    with pytest.raises(KeyError):source_limits(names,{})


def test_reserves_source_effort_without_legacy_gravity_bias():
    ff,pd=allocate([60.,-40.,0.],[100.,100.,100.],[70.,70.,70.])
    np.testing.assert_array_equal(ff,[60.,-40.,0.])
    np.testing.assert_array_equal(pd,[40.,60.,70.])
    assert ff.dtype==pd.dtype==np.float32


def test_float32_rounding_never_enlarges_either_budget():
    rng=np.random.default_rng(71)
    total=rng.uniform(.01,1000,100000)
    gravity=rng.uniform(-.9,.9,100000)*total
    ceiling=.7*total
    before=[a.copy() for a in (gravity,total,ceiling)]
    ff,pd=allocate(gravity,total,ceiling)
    assert np.all(np.abs(ff.astype(float))+pd.astype(float)<=total)
    assert np.all(pd.astype(float)<=ceiling)
    assert np.all(np.abs(ff.astype(float))<=np.abs(gravity))
    np.testing.assert_allclose(ff,gravity,rtol=1.2e-7,atol=0)
    for a,b in zip((gravity,total,ceiling),before):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('gravity,total,ceiling',[
    ([91],[100],[70]),([-91],[100],[70]),([float('nan')],[100],[70]),
    ([1],[float('inf')],[70]),([1],[0],[0]),([1],[100],[9]),
    ([1],[100],[101]),([1],[100],[float('nan')]),([],[],[]),
    ([[1]],[[100]],[[70]]),([1,2],[100],[70]),([1],[1e50],[7e49])])
def test_bad_or_overloaded_state_fails_closed(gravity,total,ceiling):
    with pytest.raises(ValueError):allocate(gravity,total,ceiling)


@pytest.mark.parametrize('enabled',[False,True])
def test_native_command_allocates_angular_budget_but_preserves_fingers(enabled):
    from .full_robot import FullRobotGripper
    calls=[]
    gravity=np.array([[60.,-.314,.314,0]],dtype=np.float32)
    api=S(set_dof_position_targets=lambda *a:calls.append(('targets',a[0].copy())),
          get_gravity_compensation_forces=lambda:gravity.copy(),
          set_dof_max_forces=lambda *a:calls.append(('limits',a[0].copy())),
          set_dof_actuation_forces=lambda *a:calls.append(('force',a[0].copy())))
    r=S(targets=np.zeros((1,4),np.float32),left_indices=[0],index=np.array([0]),
        robot=api,feedforward_limit=np.array([[30,0,0,30]],np.float32),
        force_limits=np.array([[70,.5,.5,70]],np.float32),
        budgeted_joint_gravity=enabled,angular_indices=[0,3],
        angular_total_effort=np.array([100.,100.]),angular_pd_ceiling=np.array([70.,70.]),
        names=['left_arm_0','gripper_finger_l1','gripper_finger_l2','right_arm_0'],
        finger_gravity=True,finger_indices=[1,2],finger_actuator_limit_n=.8)
    FullRobotGripper._command_left_drives(r,[12],np.eye(4))
    assert [c[0] for c in calls]==['targets','limits','force']
    np.testing.assert_array_equal(calls[-1][1][0,1:3],gravity[0,1:3])
    np.testing.assert_allclose(calls[1][1][0,1:3],.8-np.abs(gravity[0,1:3]),atol=1e-7)
    assert calls[-1][1][0,0]==(60 if enabled else 30)
    assert calls[1][1][0,0]==(40 if enabled else 70)
    if enabled:assert r.joint_gravity_sample['legacy_clipping_residual_nm']==[30.,0.]


def test_option_is_explicit_and_requires_native_robot(tmp_path,monkeypatch):
    from . import benchmark,ground_truth_trial
    assert not benchmark.parser().parse_args(['--output','unused']).budgeted_joint_gravity
    with pytest.raises(ValueError,match='full native robot'):
        benchmark.main(['--output',str(tmp_path/'unused'),'--budgeted-joint-gravity'])
    assert not (tmp_path/'unused').exists()
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda argv:captured.append(benchmark.parser().parse_args(argv)))
    ground_truth_trial.main(['--output',str(tmp_path/'unused'),'--mode','right_only',
                             '--budgeted-joint-gravity'])
    assert captured[0].budgeted_joint_gravity
