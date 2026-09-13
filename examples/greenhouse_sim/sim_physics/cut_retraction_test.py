import pytest
from .cut_retraction import CutRetraction
from .cut_section_test import contact,KNIFE,FACES


def fixture():
    return CutRetraction([-.008,0,.005],fraction=.85,endpoint=[0,0,.008],
        direction=[0,0,-1],step=20,section=dict(knife=KNIFE,faces=FACES))


def observe(f,r=None,**changes):
    opts=dict(step=f.step+1,support_ready=True);opts.update(changes)
    return f.observe(contact(z=-.0031) if r is None else r,**opts)


def command(f):return f.command(step=f.step,dt=1/480)


@pytest.mark.parametrize('upper,speed,state',[(.12,.0003,'loaded_reverse'),(0.,.002,'unloaded_reverse'),(.41,0.,'hold_contact_or_support')])
def test_retraction_uses_existing_contact_speed_without_force_cap_increase(upper,speed,state):
    f=fixture();observe(f,contact(z=-.0031,upper=upper));before=f.offset;command(f)
    assert f.receipt['command_speed_m_s']==speed
    assert f.offset==pytest.approx(before-speed/480)
    assert f.receipt['command_state']==state and not f.receipt['cut_authorized']


def test_lost_support_or_wrong_pair_cannot_drag_target():
    for support in (False,True):
        f=fixture();r=contact(z=-.0031)
        if support:r['native_contact_pairs_n']=[[KNIFE,'/Neighbor/StemCollider',.12]]
        observe(f,r,support_ready=support);before=f.offset;command(f)
        assert f.offset==before


def test_observed_unloaded_endpoint_dwell_not_command_fraction_or_time():
    f=fixture();observe(f);f.offset=f.start;command(f)
    for i in range(12):
        r=observe(f,contact(z=.008,upper=0.));assert r['complete'] is (i==11)
        if i<11:command(f)
    assert r['net_measured_reverse_m']==pytest.approx(.0111)
    assert r['unloaded_endpoint_dwell_s']==.025


def test_loaded_park_or_stationary_blade_never_completes():
    f=fixture();observe(f);f.offset=f.start
    for _ in range(20):command(f);assert not observe(f)['complete']
    f=fixture();observe(f);f.offset=f.start
    for _ in range(20):command(f);assert not observe(f,contact(z=.008,upper=.12))['complete']
    with pytest.raises(RuntimeError):observe(fixture(),contact(z=.008,upper=0.))


def test_stale_or_unsafe_data_refused():
    f=fixture();observe(f)
    with pytest.raises(RuntimeError):observe(f)
    command(f)
    with pytest.raises(RuntimeError):command(f)
    with pytest.raises(RuntimeError):observe(fixture(),contact(upper=.501))
    r=contact();r['cut']=False
    with pytest.raises(RuntimeError):observe(fixture(),r)
