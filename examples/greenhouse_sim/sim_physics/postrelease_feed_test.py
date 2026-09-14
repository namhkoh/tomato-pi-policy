import pytest
from .postrelease_feed import loaded_speed,validate,reverse_maximum
from .cut_retraction_test import fixture,observe,command
from .cut_section_test import contact


@pytest.mark.parametrize('bad',[True,False,None,float('nan'),float('inf'),.00029,.00201,-1.])
def test_bad_caps_fail_before_execution(bad):
    with pytest.raises(ValueError):validate(bad)


@pytest.mark.parametrize('load',[0.,.01,.1,.15,.2,.4,.5])
def test_default_is_exactly_the_old_rate(load):
    assert loaded_speed(load,.0003)==.0003


def test_speed_decreases_with_load_without_raising_force_limits():
    values=[loaded_speed(i/100,.001) for i in range(51)]
    assert values[0]==.001 and values[20]==.0003
    assert all(a>=b for a,b in zip(values,values[1:]))


@pytest.mark.parametrize('upper,state',[(.41,'hold_contact_or_support'),(.5,'hold_contact_or_support')])
def test_faster_reverse_still_stops_at_original_guard(upper,state):
    f=fixture();f.maximum_feed=.001
    observe(f,contact(z=-.0031,upper=upper));command(f)
    assert f.receipt['command_state']==state and f.receipt['command_speed_m_s']==0


def test_faster_command_never_means_observed_completion():
    f=fixture();f.maximum_feed=.001
    observe(f,contact(z=-.0031,upper=.08));command(f)
    assert f.receipt['command_speed_m_s']==.001
    assert not f.complete and not f.receipt['commanded_motion_used_as_completion']


@pytest.mark.parametrize('maximum',[.0003,.001,.002])
def test_support_aware_insertion_keeps_original_reverse_rate(maximum):
    assert reverse_maximum(maximum,support_aware_insertion=True)==.0003
    assert reverse_maximum(maximum,support_aware_insertion=False)==maximum


@pytest.mark.parametrize('flag',[None,0,1,'true'])
def test_reverse_profile_cannot_be_implicitly_enabled(flag):
    with pytest.raises(ValueError):reverse_maximum(.001,support_aware_insertion=flag)


def test_bad_maximum_is_not_hidden_by_slow_reverse_selection():
    with pytest.raises(ValueError):reverse_maximum(float('nan'),support_aware_insertion=True)
