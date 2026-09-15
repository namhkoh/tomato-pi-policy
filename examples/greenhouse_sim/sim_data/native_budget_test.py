import pytest
from sim_data.native_budget import requested_subframes,evidence,REFERENCE,TRIAL

@pytest.mark.parametrize('captured',[0,1,7,1000])
def test_reference_never_shortens(captured):
    assert requested_subframes(REFERENCE,captured)==56

def test_trial_retains_initial_warmup():
    assert requested_subframes(TRIAL,0)==56
    assert requested_subframes(TRIAL,1)==8
    for n in (0,1):
        e=evidence(TRIAL,n)
        assert e['native_step_calls']*e['subframes_per_step']==e['requested_subframes']
        assert e['experimental_short_profile'] and not e['high_resolution_short_profile_qualified']
        assert not e['training_release_approved'] and not e['photometric_equivalence_claimed']

@pytest.mark.parametrize('profile,count',[('unknown',0),(TRIAL,-1),(TRIAL,True),(TRIAL,2.0)])
def test_invalid_or_ambiguous_budget_fails(profile,count):
    with pytest.raises(ValueError):requested_subframes(profile,count)

