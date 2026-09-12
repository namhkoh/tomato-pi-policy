import pytest
from .blade_feed import BladeFeed
from .blade_feed_test import observed, advance


def feed():
    f=BladeFeed([-.008,-.004,0,.005],radius=.003,dwell_feedback=True)
    f.offset=-.004
    return f


def test_low_recent_load_still_regulates_when_latest_peak_is_in_band():
    f=feed();observed(f,.18,.25);advance(f)
    observed(f,.24,.30);old=f.offset;advance(f)
    assert f.offset>old
    assert f.receipt['control_load_n']==.18
    assert not f.receipt['release_evidence_filtered'] and not f.receipt['cut_authorized']


def test_raw_peak_and_total_load_backoff_not_filtered():
    for normal,upper in ((.29,.31),(.2,.34)):
        f=feed();observed(f,.10,.15);advance(f)
        observed(f,normal,upper);old=f.offset;advance(f)
        assert f.offset==pytest.approx(old-.0005/240)
        assert f.receipt['state']=='backoff_load'


def test_window_is_bounded_and_cleared_on_contact_loss():
    f=feed();observed(f,.18,.25);advance(f)
    for _ in range(7): observed(f,.24,.3);advance(f)
    assert len(f.loads)==7 and f.receipt['state']=='hold_valid_load'
    observed(f,0,0);advance(f)
    assert not f.loads
    observed(f,.23,.28);old=f.offset;advance(f)
    assert f.offset==old and f.receipt['control_load_sample_count']==1


def test_nonqualifying_contact_stays_held_and_caps_unchanged():
    f=feed();observed(f,0,.2);old=f.offset;advance(f)
    assert f.offset==old and f.receipt['state']=='hold_nonqualifying_load'
    with pytest.raises(RuntimeError):observed(f,.2,.501)


@pytest.mark.parametrize('bad',[0,1,None,'yes'])
def test_explicit_profile_only(bad):
    with pytest.raises(ValueError):BladeFeed([-.008,-.004,0,.005],radius=.003,dwell_feedback=bad)


def test_bounded_timeout_is_still_not_release():
    f=feed()
    for _ in range(7200):observed(f,.24,.3);advance(f)
    observed(f,.24,.3)
    with pytest.raises(RuntimeError,match='timed out'):advance(f)


def test_cli_incomplete_protocol_rejected_before_launch(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='complete isolated'):
        main(['--output',str(tmp_path/'unused'),'--blade-dwell-feedback'])
    assert not (tmp_path/'unused').exists()
