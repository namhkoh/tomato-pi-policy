from pathlib import Path
import pytest
from .blade_feed import BladeFeed
from .knife import DOWNWARD_CUT_MODEL,KNIFE_IMPULSE_CONTRACT


def feed(faster=True):
    return BladeFeed([-.015,-.005,0.,.0045],radius=.003,dwell_feedback=True,
        compliant_rate=True,friction_budget=True,physics_hz=480,loaded_advance=True,faster_cut=faster)


def sample(f,normal=0.,upper=0.,geometry=True,released=False):
    f.observe(dict(edge_signed_resistance_n=normal,tool_contact_upper_bound_n=upper,
        force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows_complete=True,
        cut_model=DOWNWARD_CUT_MODEL,loading_geometry_verified=geometry),
        step=f.observed_step+1,guards_passed=True,released=released)


def advance(f):return f.command(step=f.observed_step,dt=1/480)


def test_explicit_profile_preserves_old_default_and_contact_boundaries():
    fast,old=feed(),feed(False)
    assert (old.free_speed,old.near_speed,old.loaded_speed)==(.002,.0003,.00015)
    assert (fast.free_speed,fast.near_speed,fast.loaded_speed)==(.010,.001,.00075)
    assert fast.backoff_load==old.backoff_load==.40
    assert fast.acquisition_target==old.acquisition_target==.25
    assert fast.near_offset==old.near_offset
    fast.offset=fast.near_offset-1e-6;sample(fast);advance(fast)
    assert fast.offset==pytest.approx(fast.near_offset)


@pytest.mark.parametrize('normal,upper,geometry,state,speed',[
    (.25,.30,True,'loaded_downward_advance',.00075),
    (.10,.15,True,'load_feedback',.00075),
    (.25,.41,True,'backoff_load',-.0005),
    (.29,.32,True,'backoff_load',-.0005),
    (.25,.30,False,'hold_unqualified_contact',0.),
    (0.,.11,True,'hold_nonqualifying_load',0.)])
def test_faster_commands_keep_same_force_geometry_backoff(normal,upper,geometry,state,speed):
    f=feed();f.offset=-.004
    sample(f,normal,upper,geometry);advance(f)
    assert f.receipt['state']==state
    assert f.offset==pytest.approx(-.004+speed/480)
    assert f.receipt['hard_full_contact_guard_n']==.5
    assert not f.receipt['cut_authorized'] and not f.receipt['force_or_geometry_limits_changed']


def test_same_hard_limit_freshness_and_release_latch():
    f=feed()
    with pytest.raises(RuntimeError):sample(f,.3,.501)
    assert f.observed_step==0
    sample(f);advance(f)
    with pytest.raises(RuntimeError):advance(f)
    sample(f,released=True)
    with pytest.raises(RuntimeError):advance(f)


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_only_explicit_boolean_profile(value):
    with pytest.raises(ValueError):feed(value)


def test_faster_profile_requires_native_loaded_advance():
    with pytest.raises(ValueError,match='Faster cut'):
        BladeFeed([-.015,0,.005],radius=.003,faster_cut=True)


def test_cli_rejects_partial_profile_before_any_output(tmp_path):
    from .benchmark import main
    p=tmp_path/'new'
    with pytest.raises(ValueError,match='Faster cut'):main(['--output',str(p),'--faster-cut-trial'])
    assert not p.exists()


def test_wrapper_complete_profile_and_two_mm_feed_required(monkeypatch,tmp_path):
    from .ground_truth_trial import main
    class BeforeWrite(Exception):pass
    def stop(*args,**kwargs):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'new'),'--mode','right_only','--milestone','cut_action',
        '--process-zone-trial','--through-stroke-trial','--material-clearance-trial',
        '--postcut-egress-trial','--faster-cut-trial','--neutral-ready-start',
        '--support-aware-feed-trial','--park-left-ready','--torso-degrees','0','0','0','0','0','0']
    with pytest.raises(ValueError,match='Support-aware|Faster cut'):main(args)
    with pytest.raises(BeforeWrite):main(args+['--postrelease-feed-m-s','.002'])
