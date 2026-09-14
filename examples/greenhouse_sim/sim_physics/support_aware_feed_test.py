import copy
import pytest
from .support_aware_feed import SupportAwareFeed


def record(step=1,load=.25,slip=.0005,cut_only=False):
    r=dict(t=step/480,native_guards_passed=True,cut=True,
        contact=dict(adapter_valid=True,step_id=step,bilateral=not cut_only),
        robot=dict(per_finger_contact_upper_bound_n={'/Robot/ee_finger_l1':load,'/Robot/ee_finger_l2':load}),
        knife={'tool_contact_upper_bound_n':.05},slip_m=None if cut_only else slip)
    if cut_only:r['cut_only_park']=dict(model='native_unheld_left_park_v1',step_id=step)
    return r


def advance(f,n=500,load=.25):
    values=[]
    for i in range(1,n+1):
        r=record(i,load=load);original=copy.deepcopy(r)
        f.observe(r,step=i);values.append(f.command(.001,step=i,dt=1/480))
        assert r==original
    return values


def test_acceleration_requires_stable_measured_support_and_is_slew_bounded():
    f=SupportAwareFeed(.001);speeds=advance(f)
    assert speeds[:143]==[.0003]*143 and speeds[-1]>.0003
    assert all(b-a<=.001/480+1e-12 for a,b in zip(speeds,speeds[1:]))
    assert max(speeds)<=.001 and not f.receipt['cut_authorized']
    assert not f.receipt['force_limits_changed']


def test_high_finger_load_stops_acceleration_even_when_blade_load_is_low():
    f=SupportAwareFeed(.001);advance(f,200)
    f.observe(record(201,load=.41),step=201)
    assert f.command(.001,step=201,dt=1/480)==0


def test_rapid_slip_prevents_speedup_and_large_slip_stops_before_original_limit():
    f=SupportAwareFeed(.001)
    for i in range(1,100):
        f.observe(record(i,slip=.0005+(i%2)*.0004),step=i)
        assert f.command(.001,step=i,dt=1/480)==.0003
    f.observe(record(100,slip=.0021),step=100)
    assert f.command(.001,step=100,dt=1/480)==0


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(native_guards_passed=False),
    lambda r:r['contact'].update(adapter_valid=False),
    lambda r:r['contact'].update(bilateral=False),
    lambda r:r.update(slip_m=None),
    lambda r:r['robot']['per_finger_contact_upper_bound_n'].update({'/Robot/ee_finger_l1':.5})])
def test_bad_or_missing_support_cannot_authorize_acceleration(mutate):
    r=record();mutate(r)
    with pytest.raises(RuntimeError):SupportAwareFeed(.001).observe(r,step=1)


def test_no_fake_grasp_needed_for_verified_cut_only_park():
    f=SupportAwareFeed(.001)
    for i in range(1,500):
        f.observe(record(i,load=0.,cut_only=True),step=i)
        f.command(.001,step=i,dt=1/480)
    assert f.speed>.0003 and f.receipt['cut_only']
    assert f.receipt['slip_m'] is None


def test_stop_backoff_and_freshness_remain_immediate():
    f=SupportAwareFeed(.001);f.observe(record(),step=1)
    assert f.command(-.0005,step=1,dt=1/480)==-.0005
    with pytest.raises(RuntimeError):f.command(.001,step=1,dt=1/480)
    with pytest.raises(RuntimeError):f.observe(record(3),step=3)


def test_explicit_profile_required_before_output(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Support-aware acceleration'):
        main(['--output',str(tmp_path/'no'),'--support-aware-feed-trial'])
    assert not (tmp_path/'no').exists()
