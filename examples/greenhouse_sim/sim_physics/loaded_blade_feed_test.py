"""Synthetic feedback checks, not evidence of native cutting success."""
import numpy as np
import pytest
from .blade_feed import BladeFeed
from .knife import DOWNWARD_CUT_MODEL,KNIFE_IMPULSE_CONTRACT


def fixture(**changes):
    options=dict(radius=.003,dwell_feedback=True,compliant_rate=True,
        friction_budget=True,physics_hz=480,loaded_advance=True)
    options.update(changes)
    return BladeFeed([-.008,-.004,0,.005],**options)


def step(feed,normal=.25,upper=.3,geometry=True,**changes):
    k=dict(edge_signed_resistance_n=normal,tool_contact_upper_bound_n=upper,
        force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows_complete=True,
        cut_model=DOWNWARD_CUT_MODEL,loading_geometry_verified=geometry)
    k.update(changes)
    feed.observe(k,step=feed.observed_step+1,guards_passed=True,released=False)
    return feed.command(step=feed.observed_step,dt=1/480)


def test_aligned_loaded_contact_advances_without_authorizing_release():
    f=fixture();start=f.offset
    for _ in range(480):step(f)
    assert f.offset-start==pytest.approx(.00015)
    assert f.receipt['state']=='loaded_downward_advance'
    assert f.receipt['current_loading_geometry_verified'] is True
    assert f.receipt['hard_full_contact_guard_n']==.5
    assert f.receipt['cut_authorized'] is False
    assert f.released is False  # Even commanded 0.15 mm is NOT cut evidence.


@pytest.mark.parametrize('normal,upper',[(.25,.3),(.1,.15),(-.01,.02)])
def test_bad_geometry_or_lost_support_never_increases_contact_load(normal,upper):
    f=fixture();start=f.offset
    step(f,normal,upper,False)
    assert f.offset==start and f.receipt['state']=='hold_unqualified_contact'


def test_rejected_alignment_revokes_previous_loading_on_the_next_command():
    f=fixture();step(f);start=f.offset
    step(f,geometry=False)
    assert f.offset==start
    assert f.receipt['current_loading_geometry_verified'] is False


@pytest.mark.parametrize('normal,upper',[(.29,.31),(.25,.401)])
@pytest.mark.parametrize('geometry',[True,False])
def test_existing_load_backoff_has_priority_over_motion_and_alignment(normal,upper,geometry):
    f=fixture();f.offset=-.004;start=f.offset;step(f,normal,upper,geometry)
    assert f.offset==pytest.approx(start-.0005/480)
    assert f.receipt['state']=='backoff_load'


def test_low_force_can_be_acquired_only_with_current_valid_geometry():
    f=fixture();start=f.offset;step(f,.1,.15,True)
    assert 0<f.offset-start<=.0003/480
    assert f.receipt['state']=='load_feedback'


def test_free_approach_does_not_require_a_contact_before_contact_exists():
    f=fixture();start=f.offset;step(f,0.,0.,False)
    assert f.offset>start and f.receipt['state']=='free_space'


@pytest.mark.parametrize('changes',[dict(cut_model='legacy'),dict(geometry=None),
    dict(geometry=1),dict(geometry=np.bool_(True))])
def test_missing_or_stale_contract_does_not_advance_observation(changes):
    f=fixture()
    with pytest.raises(RuntimeError):step(f,**changes)
    assert f.observed_step==0


@pytest.mark.parametrize('changes',[dict(loaded_advance=1),dict(physics_hz=240),
    dict(dwell_feedback=False),dict(friction_budget=False),dict(compliant_rate=False)])
def test_incomplete_profile_rejected(changes):
    with pytest.raises(ValueError):fixture(**changes)
