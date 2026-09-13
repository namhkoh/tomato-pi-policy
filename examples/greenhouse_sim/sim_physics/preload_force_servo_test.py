"""Synthetic controller checks, not native contact or cutting qualification."""
import numpy as np
import pytest
from .force_closure import ForceClosure


def controller(enabled=True):
    return ForceClosure(.003,.001,retention_preload=True,symmetric=True,
        effort_bounded_target=True,pregrasp_half_aperture=.008,preload_force_servo=enabled)


def contact(c, support, loads=None, valid=True):
    c.command(1.,step=0,dt=1/240)
    c.observe(dict(step_id=1,adapter_valid=True,normal_only=True,stem_only=valid,
        compressive_support_n=list(support)),list(support) if loads is None else loads,
        step=1,guards_passed=True)


def test_control_moves_off_readiness_boundary_without_raising_rate_or_effort():
    old=controller(False);new=controller()
    for c in (old,new):contact(c,[.215,.215])
    old_before=old.gaps.copy();before=new.gaps.copy()
    np.testing.assert_array_equal(old.command(1.,step=1,dt=1/240),old_before)
    gap=new.command(1.,step=1,dt=1/240)
    assert np.all(before-gap>0) and np.max(before-gap)<=.0005/240+1e-12
    for key in ('desired_support_n','drive_limit_n','backoff_contact_n','minimum','opening'):
        assert getattr(old,key)==getattr(new,key)
    assert new.receipt['native_hard_contact_limit_n']==.5
    assert new.receipt['actual_native_penetration_guard_m']==.001
    assert not new.receipt['grasp_verified']


@pytest.mark.parametrize('support,direction',[(.18,-1),(.215,-1),(.235,0),(.24,0),(.245,0),(.265,1),(.35,1)])
def test_inner_deadband_and_bounded_symmetric_negative_feedback(support,direction):
    c=controller();contact(c,[support]*2);before=c.gaps.copy()
    after=c.command(1.,step=1,dt=1/240)
    assert after[0]==after[1]
    assert np.sign(after[0]-before[0])==direction
    assert np.max(abs(after-before))<=.0005/240+1e-12


@pytest.mark.parametrize('valid,loads',[(False,[.2,.2]),(True,[.41,.2])])
def test_geometry_and_all_contact_backoff_override_preload_servo(valid,loads):
    c=controller();contact(c,[.1,.1],loads,valid);before=c.gaps.copy()
    after=c.command(1.,step=1,dt=1/240)
    np.testing.assert_allclose(after-before,.005/240)


def test_profile_is_explicit_and_cli_fails_before_output(tmp_path):
    from .benchmark import parser,main
    assert not parser().parse_args(['--output','unused']).preload_force_servo
    with pytest.raises(ValueError,match='Preload force servo'):
        main(['--output',str(tmp_path/'unused'),'--preload-force-servo'])
    assert not (tmp_path/'unused').exists()
    for kw in ({},{'retention_preload':True,'symmetric':True}):
        with pytest.raises(ValueError,match='Preload force servo'):
            ForceClosure(.003,.001,preload_force_servo=True,**kw)
    for v in (1,'yes',None):
        with pytest.raises(ValueError):ForceClosure(.003,.001,preload_force_servo=v)


def test_same_step_and_guard_failures_remain_fatal():
    c=controller();c.command(1.,step=0,dt=1/240)
    with pytest.raises(RuntimeError):c.command(1.,step=1,dt=1/240)
    with pytest.raises(RuntimeError):
        c.observe(dict(step_id=1,adapter_valid=True,normal_only=True,stem_only=True,
            compressive_support_n=[.24,.24]),[.5,.24],step=1,guards_passed=True)
    assert c.observed_step==0
