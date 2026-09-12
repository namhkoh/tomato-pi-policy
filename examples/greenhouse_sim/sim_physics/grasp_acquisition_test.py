import pytest
from sim_physics.bimanual_probe import grasp_acquisition_state,schedule_after_grasp,sequence_times


def test_feedback_waits_for_original_native_dwell_not_nominal_deadline():
    assert grasp_acquisition_state(3.,240,240,True)=='waiting'
    assert grasp_acquisition_state(3.5,24,240,True)=='verified'
    assert grasp_acquisition_state(8.5,0,240,True)=='waiting'
    assert grasp_acquisition_state(11.,23,240,True)=='waiting'
    assert grasp_acquisition_state(11.,24,240,True)=='verified'
    assert grasp_acquisition_state(13.5,0,240,True)=='timeout'
    assert grasp_acquisition_state(13.5,24,240,True)=='verified'
    assert grasp_acquisition_state(13.6,240,240,True)=='timeout'


def test_geometric_control_keeps_original_deadline_and_contact_requirement():
    assert grasp_acquisition_state(3.5,23,240,False)=='timeout'
    assert grasp_acquisition_state(3.5,24,240,False)=='verified'
    assert grasp_acquisition_state(3.4,24,240,False)=='waiting'
    assert schedule_after_grasp(3.5,0)==sequence_times(0)


@pytest.mark.parametrize('reposition',[0.,.008])
def test_every_phase_shifts_from_verified_time_with_unchanged_phase_durations(reposition):
    original=sequence_times(reposition)
    actual=schedule_after_grasp(11.25,reposition)
    for key in original:assert actual[key]==pytest.approx(original[key]+7.75)
    assert actual['plan']>=11.25
    assert actual['approach']-actual['plan']==.5
    assert actual['stroke']-actual['approach']==4
    assert actual['end']-actual['stroke']==6


@pytest.mark.parametrize('args',[(float('nan'),1,240,True),(-1,1,240,True),
    (4,-1,240,True),(4,True,240,True),(4,1,0,True),(4,1,240,1)])
def test_invalid_acquisition_cannot_verify(args):
    with pytest.raises(ValueError):grasp_acquisition_state(*args)


@pytest.mark.parametrize('t',[float('nan'),0.,3.49,14.])
def test_invalid_verification_time_cannot_schedule_motion(t):
    with pytest.raises(ValueError):schedule_after_grasp(t,0)
