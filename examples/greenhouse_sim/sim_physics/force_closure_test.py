import numpy as np
import pytest

from sim_physics.force_closure import ForceClosure


def sample(step, support=(0., 0.)):
    return dict(step_id=step, adapter_valid=True, normal_only=True,
        stem_only=True, compressive_support_n=list(support))


def test_near_contact_speed_and_compression_bounded_without_inventing_grasp():
    c=ForceClosure(.003,.0005)
    for step in range(2400):
        before=c.gaps.copy()
        gaps=c.command(min(step/240,1.),step=step,dt=1/240)
        if max(before)<=c.slow_gap+1e-12:
            assert np.max(abs(gaps-before))<=.0005/240+1e-12
        assert np.all(gaps>=.0025) and np.all(gaps<=.025)
        assert c.receipt['grasp_verified'] is False
        c.observe(sample(step+1),[0.,0.],step=step+1,guards_passed=True)
    assert np.allclose(c.gaps,.0025)


def test_independent_force_feedback_and_all_contact_backoff():
    c=ForceClosure(.003,.0005)
    c.command(1.,step=0,dt=1/240)
    c.observe(sample(1,(.12,.12)),[.12,.31],step=1,guards_passed=True)
    before=c.gaps.copy()
    after=c.command(1.,step=1,dt=1/240)
    assert after[0]==before[0]
    assert after[1]>before[1]


def test_no_stale_missing_or_repeated_sample():
    c=ForceClosure(.003,.0005)
    c.command(0.,step=0,dt=1/240)
    with pytest.raises(RuntimeError):c.command(0.,step=1,dt=1/240)
    c.observe(sample(1),[0.,0.],step=1,guards_passed=True)
    with pytest.raises(RuntimeError):c.observe(sample(1),[0.,0.],step=1,guards_passed=True)
    with pytest.raises(RuntimeError):c.command(0.,step=0,dt=1/240)


@pytest.mark.parametrize('change',[
    {'adapter_valid':False},{'normal_only':False},{'stem_only':None},
    {'step_id':2},{'compressive_support_n':[float('nan'),0.]},
    {'compressive_support_n':[0.]}])
def test_invalid_observation_does_not_advance(change):
    c=ForceClosure(.003,.0005);c.command(0.,step=0,dt=1/240)
    contact=sample(1);contact.update(change)
    with pytest.raises(RuntimeError):c.observe(contact,[0.,0.],step=1,guards_passed=True)
    assert c.observed_step==0


@pytest.mark.parametrize('loads,passed',[([.5,0.],True),([-1.,0.],True),([0.,0.],False)])
def test_guard_failure_never_reenters_controller(loads,passed):
    c=ForceClosure(.003,.0005);c.command(0.,step=0,dt=1/240)
    with pytest.raises(RuntimeError):c.observe(sample(1),loads,step=1,guards_passed=passed)


def test_feedback_wait_is_explicit_without_changing_legacy_timing():
    from sim_physics.bimanual_probe import sequence_times
    legacy=sequence_times(0.);feedback=sequence_times(0.,True)
    for key in ('plan','approach','stroke','end'):
        assert feedback[key]==legacy[key]+5
    assert sequence_times(.008,True)['plan']==feedback['plan']+1.5


def test_native_command_caps_do_not_exceed_the_remaining_gravity_budget():
    from sim_physics.bimanual import BimanualRobot
    from types import SimpleNamespace
    from unittest.mock import Mock
    fixture=SimpleNamespace(force_closure_enabled=True,force_closer=ForceClosure(.003,.0005),
        force_limits=np.array([[.10,.30]]),finger_indices=[0,1],
        targets=np.zeros((1,2)),index=np.array([0],dtype=np.uint32),robot=Mock())
    BimanualRobot.close(fixture,1.,step=0,dt=1/240)
    np.testing.assert_allclose(fixture.force_limits,[[.10,.15]])
    fixture.robot.set_dof_max_forces.assert_called_once()
    fixture.robot.set_dof_position_targets.assert_called_once()
    assert fixture.targets[0,0]<0<fixture.targets[0,1]


def test_unqualified_geometry_cannot_squeeze_and_never_counts_as_grasp():
    c=ForceClosure(.003,.0005);c.command(1.,step=0,dt=1/240)
    contact=sample(1,(.2,.2));contact['stem_only']=False
    c.observe(contact,[.25,0.],step=1,guards_passed=True)
    before=c.gaps.copy();after=c.command(1.,step=1,dt=1/240)
    assert after[0]>before[0] and after[1]==before[1]
    assert not c.receipt['geometry_valid'] and not c.receipt['grasp_verified']
    np.testing.assert_array_equal(c.support,[0,0])
