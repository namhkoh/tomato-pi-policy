import inspect
import numpy as np
import pytest
from sim_physics.blade_feed import BladeFeed
from sim_physics.knife import KNIFE_IMPULSE_CONTRACT


def fixture():return BladeFeed([-.008,-.004,0,.005],radius=.003)


def observed(feed,normal=0.,upper=0.,**changes):
    k=dict(edge_signed_resistance_n=normal,tool_contact_upper_bound_n=upper,
        force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows_complete=True)
    k.update(changes)
    feed.observe(k,step=feed.observed_step+1,guards_passed=True,released=False)


def advance(feed):return feed.command(step=feed.observed_step,dt=1/240)


def test_no_fabricated_first_observation():
    f=fixture()
    with pytest.raises(RuntimeError):f.command(step=0,dt=1/240)


def test_fresh_clock_and_no_duplicate_commands():
    f=fixture();observed(f);advance(f)
    with pytest.raises(RuntimeError):advance(f)
    with pytest.raises(RuntimeError):f.command(step=2,dt=1/240)
    observed(f);advance(f)


def test_free_approach_stops_at_slow_transition_and_stays_bounded():
    f=fixture();offsets=[f.offset]
    for _ in range(365):observed(f);advance(f);offsets.append(f.offset)
    assert -.005<f.offset<-.00499
    assert max(np.diff(offsets))<=.002/240+1e-12
    assert any(abs(x+.005)<1e-12 for x in offsets)
    assert np.diff(offsets)[-1]==pytest.approx(.0001/240)


@pytest.mark.parametrize('normal,upper,state,speed',[
    (.1,.15,'load_feedback',.000032),(.24,.30,'hold_valid_load',0),
    (.29,.31,'backoff_load',-.0005),(.2,.34,'backoff_load',-.0005),
    (0,.11,'hold_nonqualifying_load',0),(-.02,.05,'load_feedback',.00005)])
def test_signed_feedback_and_conservative_total_load(normal,upper,state,speed):
    f=fixture();f.offset=-.004
    observed(f,normal,upper);advance(f)
    assert f.offset==pytest.approx(-.004+speed/240)
    assert f.receipt['state']==state
    assert not f.receipt['cut_authorized'] and not f.receipt['force_limit_guaranteed']


@pytest.mark.parametrize('normal,upper',[(float('nan'),.2),(.2,float('inf')),(.3,.2),(0,-.01),(0,.501),(True,.2),(0,False),(None,.2)])
def test_invalid_load_does_not_advance_observation(normal,upper):
    f=fixture()
    with pytest.raises(RuntimeError):observed(f,normal,upper)
    assert f.observed_step==0


@pytest.mark.parametrize('changes',[dict(force_contract='unsigned'),dict(raw_normal_rows_complete=False)])
def test_incomplete_or_unsigned_reports_rejected(changes):
    f=fixture()
    with pytest.raises(RuntimeError):observed(f,**changes)


@pytest.mark.parametrize('step,passed,released',[(2,True,False),(1,False,False),(True,True,False),(1,True,0)])
def test_failed_guards_or_stale_epoch_cannot_authorize_feed(step,passed,released):
    f=fixture()
    with pytest.raises(RuntimeError):f.observe({},step=step,guards_passed=passed,released=released)


def test_no_timed_or_commanded_release_and_no_motion_after_actual_release():
    f=fixture();observed(f,.25,.3);advance(f)
    k=dict(edge_signed_resistance_n=.25,tool_contact_upper_bound_n=.3,
        force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows_complete=True)
    f.observe(k,step=2,guards_passed=True,released=True)
    with pytest.raises(RuntimeError):advance(f)
    assert not f.receipt['cut_authorized']
    with pytest.raises(RuntimeError):f.observe(k,step=3,guards_passed=True,released=False)


def test_hold_is_bounded_by_timeout_not_release():
    f=fixture();f.offset=-.004
    for _ in range(7200):observed(f,.25,.3);advance(f)
    observed(f,.25,.3)
    with pytest.raises(RuntimeError,match='timed out'):advance(f)
    assert f.offset==-.004


@pytest.mark.parametrize('dt',[1/120,0,float('nan'),True])
def test_only_qualified_dt(dt):
    f=fixture();observed(f)
    with pytest.raises(ValueError):f.command(step=1,dt=dt)


def test_exhausted_path_never_becomes_success():
    f=fixture();f.offset=f.end-1e-8;observed(f)
    with pytest.raises(RuntimeError,match='exhausted'):advance(f)


@pytest.mark.parametrize('offsets,radius',[([-.008,-.009,.005],.003),([-.008,float('nan'),.005],.003),
    ([-.03,.005],.003),([-.008,.005],0),([-.008,.005],True)])
def test_invalid_path(offsets,radius):
    with pytest.raises(ValueError):BladeFeed(offsets,radius=radius)


def test_opt_in_fails_before_stage_creation_without_complete_fixture(tmp_path):
    from sim_physics.benchmark import parser,main
    assert not parser().parse_args(['--output','unused']).blade_force_feed
    with pytest.raises(ValueError,match='isolated downward'):
        main(['--output',str(tmp_path/'unused'),'--blade-force-feed'])
    assert not (tmp_path/'unused').exists()


def test_runtime_guard_order_and_release_latch_remain_intact():
    from sim_physics.bimanual_probe import run
    src=inspect.getsource(run)
    assert src.index('fixture.check(dt,palm)')<src.index('fixture.inspect_cut(')<src.index('blade_feed.observe(')
    assert 'cut_fraction=released_stroke_fraction(last_right_command)' in src
    assert 't>=stroke_end and blade_feed is None' in src
    assert "if cut_time is None and blade_feed is not None:" in src
