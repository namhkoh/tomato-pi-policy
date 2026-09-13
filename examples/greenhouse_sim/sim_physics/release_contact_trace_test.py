from types import SimpleNamespace as S
import pytest
from sim_physics.contact_events import ContactEvents
from sim_physics.release_contact_trace import ReleaseContactTrace


def setup():
    monitor=ContactEvents(robot_root='/Robot', target_root='/Plant', fingers=[], floor_root=None)
    monitor.native_full_contact_reporting=True
    trace=ReleaseContactTrace(monitor,['/Plant/Stem'])
    return monitor,trace


def test_records_plant_support_contact_without_reclassifying_robot_loads():
    m,t=setup()
    m.consume('/Plant/Stem','/Support/Stem',[[0,0,.001],[0,0,-.001]],
              [[0,0,0]]*2,[[0,0,1]]*2,[-.0001,.0002],
              friction_impulses=[[.0002,0,0]],friction_points=[[0,0,0]])
    r=t.measured_snapshot(step=1,dt=1/240,cut=True)
    assert m.pairs=={} and m.unwanted_impulse==0
    assert r['row_count']==3 and r['rows'][1]['impulse_on_0_ns'][2]==-.001
    p=r['pair_summary'][0]
    assert p['observed_force_upper_n']==pytest.approx(.528)
    assert p['minimum_separation_m']==-.0001
    assert p['normal_rows']==2 and p['friction_rows']==1
    assert not r['native_completeness_verified'] and not r['actuation_authorized']
    r['rows'].clear()
    assert len(t.rows)==3
    m.begin_step()
    assert t.measured_snapshot(step=2,dt=1/240,cut=True)['rows']==[]


def test_never_replaces_observer_or_claims_missing_native_reporting():
    for m in (S(full_contact_observer=object(),native_full_contact_reporting=True),
              S(full_contact_observer=None,native_full_contact_reporting=False)):
        old=m.full_contact_observer
        with pytest.raises(ValueError):ReleaseContactTrace(m,['/Plant/Stem'])
        assert m.full_contact_observer is old


@pytest.mark.parametrize('step,dt,cut',[(True,1/240,False),(0,1/240,False),
    (1,0,False),(1,float('nan'),False),(1,True,False),(1,1/240,1)])
def test_invalid_binding_rejected(step,dt,cut):
    _,t=setup()
    with pytest.raises(ValueError):t.measured_snapshot(step=step,dt=dt,cut=cut)


def test_replay_gap_and_callback_fault_rejected():
    m,t=setup();t.measured_snapshot(step=5,dt=1/240,cut=False)
    for step in (5,7):
        with pytest.raises(ValueError):t.measured_snapshot(step=step,dt=1/240,cut=True)
    t.invalidate(RuntimeError('partial native buffer'))
    with pytest.raises(RuntimeError):t.measured_snapshot(step=6,dt=1/240,cut=True)
