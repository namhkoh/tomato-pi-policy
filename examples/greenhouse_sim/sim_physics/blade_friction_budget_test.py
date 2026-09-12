import pytest
from .blade_feed import BladeFeed
from .blade_feed_test import observed, advance


def feed(**kw):
    return BladeFeed([-.008,-.004,0,.005],radius=.003,**kw)


def test_full_contact_control_budget_includes_friction_not_cut_authority():
    old=feed(compliant_rate=True,dwell_feedback=True)
    new=feed(compliant_rate=True,dwell_feedback=True,friction_budget=True)
    for f in (old,new):
        f.offset=-.004;observed(f,.215,.35);advance(f)
    assert old.receipt['state']=='backoff_load'
    assert new.receipt['state']=='load_feedback'
    assert 0<new.receipt['delta_m']<=1.25e-6
    assert new.receipt['backoff_load_n']==.4
    assert new.receipt['hard_full_contact_guard_n']==.5
    assert not new.receipt['cut_authorized'] and not new.receipt['release_evidence_filtered']
    assert not new.receipt['force_limit_guaranteed']


@pytest.mark.parametrize('normal,upper,state',[(.215,.401,'backoff_load'),(.281,.39,'backoff_load'),
    (.24,.36,'hold_valid_load'),(0,.11,'hold_nonqualifying_load')])
def test_existing_normal_and_nonqualifying_limits_remain(normal,upper,state):
    f=feed(compliant_rate=True,dwell_feedback=True,friction_budget=True)
    observed(f,normal,upper);advance(f)
    assert f.receipt['state']==state


def test_full_force_safety_cap_not_changed():
    f=feed(compliant_rate=True,dwell_feedback=True,friction_budget=True)
    with pytest.raises(RuntimeError):observed(f,.25,.500001)


@pytest.mark.parametrize('kw',[dict(friction_budget=True),dict(friction_budget=1),
    dict(compliant_rate=True,friction_budget=True),dict(dwell_feedback=True,friction_budget=True)])
def test_requires_explicit_matched_controller(kw):
    with pytest.raises(ValueError):feed(**kw)


def test_unsupported_launcher_combination_rejected_before_output(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='friction budget'):
        main(['--output',str(tmp_path/'unused'),'--blade-friction-budget'])
    assert not (tmp_path/'unused').exists()
