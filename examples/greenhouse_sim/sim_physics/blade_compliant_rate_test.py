import pytest
from .blade_feed import BladeFeed
from .blade_feed_test import observed, advance


def test_opt_in_loading_increment_is_bounded_and_no_release_permission():
    f=BladeFeed([-.008,-.004,0,.005],radius=.003,compliant_rate=True)
    f.offset=-.004
    observed(f,.01,.03);old=f.offset;advance(f)
    assert f.offset-old == pytest.approx(.0003/240)
    assert f.receipt['maximum_loading_increment_m']==pytest.approx(1.25e-6)
    assert not f.receipt['cut_authorized'] and not f.receipt['force_limit_guaranteed']
    observed(f,.2,.26);old=f.offset;advance(f)
    assert f.offset-old == pytest.approx((.26-.2)*.00125/240)


@pytest.mark.parametrize('normal,upper,delta',[(.24,.3,0),(.29,.31,-.0005/240),(.1,.34,-.0005/240),(0,.11,0)])
def test_original_hold_backoff_and_nonqualifying_contact_guards(normal,upper,delta):
    f=BladeFeed([-.008,-.004,0,.005],radius=.003,compliant_rate=True)
    f.offset=-.004;observed(f,normal,upper);old=f.offset;advance(f)
    assert f.offset-old==pytest.approx(delta)


def test_near_transition_still_has_no_jump_and_no_contact_free_cut():
    f=BladeFeed([-.008,-.004,0,.005],radius=.003,compliant_rate=True)
    f.offset=f.near_offset-1e-7;observed(f);advance(f)
    assert f.offset==f.near_offset
    observed(f);old=f.offset;advance(f)
    assert f.offset-old==pytest.approx(.0003/240)
    with pytest.raises(RuntimeError):observed(f,.2,.501)


def test_configuration_requires_exact_compliant_fixture(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='isolated compliant'):
        main(['--output',str(tmp_path/'unused'),'--compliant-blade-rate'])
    assert not (tmp_path/'unused').exists()
    with pytest.raises(ValueError):BladeFeed([-.008,.005],radius=.003,compliant_rate=1)
