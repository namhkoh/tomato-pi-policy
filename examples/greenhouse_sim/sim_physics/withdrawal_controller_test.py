"""Controller orchestration tests; fake adapter receipts are NOT native proof."""
from copy import deepcopy
from types import SimpleNamespace as S

import numpy as np
import pytest

from .withdrawal_controller import (WithdrawalController,reference_rates,RIGHT,
    native_command_packet,measured_completion)


def plan():
    a=np.array([[-2.]*7,[0.]*7]);s=np.array([[0.]*7,[1.]*7,[2.]*7])
    a[:,1]=s[:,1]=0.
    return dict(approach=a,stroke=s,direction=[1.,0.,0.])


def test_rates_use_original_profiles_and_native_caps_without_tuning():
    rate=reference_rates(plan(),.5,np.radians([.5,1,1,1,1,1,1]))
    np.testing.assert_allclose(rate,[.5,.75,.75,.75,.75,.75,.75])


@pytest.mark.parametrize('fraction',[True,-.1,1.01,np.nan])
def test_bad_rate_source_fails(fraction):
    with pytest.raises(ValueError):reference_rates(plan(),fraction,np.ones(7))


@pytest.mark.parametrize('limits',[np.zeros(7),[1.]*6,[np.inf]*7])
def test_bad_native_rates_fail(limits):
    with pytest.raises(ValueError):reference_rates(plan(),.5,limits)


def setup(monkeypatch):
    from . import withdrawal_controller as module
    calls=[];seen=[];indices=np.arange(3,10);targets=np.zeros((1,12),dtype=np.float32)
    targets[0,:3]=[.001,.002,.003]
    names=['left_a','left_b','left_c',*RIGHT,'finger1','finger2']
    stamp=S(episode=2,step=10);clock=S(stamp=stamp,dt=1/240)
    f=S(names=names,right_indices=indices,targets=targets,expected_palm=np.eye(4),
        finger_indices=np.array([10,11]),plan=plan(),index=np.array([0],np.uint32),
        kin=S(arm_limits_degrees=lambda _: (np.full(7,-10.),np.full(7,10.))),
        knife=S(frame=lambda m:m),cut_authorized=True)
    native=S(targets=targets.copy())
    def send(t,i):calls.append(t.copy());native.targets=t.copy()
    f.robot=S(set_dof_position_targets=send,get_dof_position_targets=lambda:native.targets.copy(),
        get_dof_max_velocities=lambda:np.ones((1,12)))
    class Adapter:
        def __init__(self,*a):self.last_receipt=None
        def snapshot(self,sample,**kw):
            seen.append((sample,deepcopy(kw)))
            q=float(np.degrees(float(np.float32(np.radians(.2)))))
            return dict(sample_id=list(sample),right_q_deg=[q]*7)
        def wrist_fk(self,q):return np.eye(4)
        def validate(self,request):raise AssertionError('Fake helper must not claim native query')
        def _fresh(self):pass
    class Helper:
        def __init__(self,*paths,**kwargs):self.kwargs=kwargs;self.m=kwargs['measurement']
        def initial_command(self):return dict(status='ready',sample_id=self.m['sample_id'],
            target_right_q_deg=self.m['right_q_deg'],cut_authorized=False)
        def advance(self,m):return dict(status='complete',sample_id=m['sample_id'],
            target_right_q_deg=None,cut_authorized=False)
    monkeypatch.setattr(module,'NativeWithdrawalAdapter',Adapter)
    monkeypatch.setattr(module,'MeasuredWithdrawal',Helper)
    ctl=WithdrawalController(f,S(bodies='fake'),clock,cut_fraction=.5)
    record=dict(native_guards_passed=True,cut=True,robot={},contact={'step_id':10})
    return ctl,f,stamp,record,calls,seen


def test_only_right_drive_targets_sent_from_measured_start_once(monkeypatch):
    c,f,s,r,calls,seen=setup(monkeypatch);before=f.targets.copy()
    c.observe(s,r)
    assert seen[0][0]==(2,10) and seen[0][1]['guard_evidence']['passed'] is True
    np.testing.assert_allclose(c.controller.kwargs['measurement']['right_q_deg'],[.2]*7)
    c.command(s)
    assert len(calls)==1 and f.cut_authorized is False
    np.testing.assert_array_equal(f.targets[0,[0,1,2,10,11]],before[0,[0,1,2,10,11]])
    np.testing.assert_allclose(f.targets[0,f.right_indices],np.radians(.2),rtol=1e-7)
    with pytest.raises(RuntimeError,match='Missing'):c.command(s)
    assert len(calls)==1


@pytest.mark.parametrize('fault',['guards','cut','robot'])
def test_no_snapshot_or_command_before_original_native_guards(monkeypatch,fault):
    c,f,s,r,calls,seen=setup(monkeypatch)
    r[{'guards':'native_guards_passed','cut':'cut','robot':'robot'}[fault]]=None
    with pytest.raises(RuntimeError):c.observe(s,r)
    assert not seen and not calls


def test_stale_decision_has_no_command(monkeypatch):
    c,f,s,r,calls,seen=setup(monkeypatch);c.observe(s,r)
    s.step+=1
    with pytest.raises(RuntimeError,match='stale'):c.command(s)
    assert not calls


@pytest.mark.parametrize('bad',[np.nan,10.,-10.])
def test_bad_target_never_clipped_or_sent(monkeypatch,bad):
    c,f,s,r,calls,seen=setup(monkeypatch);c.observe(s,r)
    c.pending['target_right_q_deg'][0]=bad
    with pytest.raises(RuntimeError,match='target'):c.command(s)
    assert not calls


def test_completed_path_holds_existing_target_without_repeating_motion(monkeypatch):
    c,f,s,r,calls,seen=setup(monkeypatch);c.observe(s,r);c.command(s)
    s.step+=1;c.observe(s,r);c.command(s)
    assert len(calls)==1 and r['measured_withdrawal']['status']=='complete'


def test_freshness_failure_stops_before_native_write(monkeypatch):
    c,f,s,r,calls,seen=setup(monkeypatch);c.observe(s,r)
    c.adapter._fresh=lambda:(_ for _ in ()).throw(RuntimeError('changed native geometry'))
    with pytest.raises(RuntimeError,match='geometry'):c.command(s)
    assert not calls


def test_readback_mismatch_latches_blocked_without_retry(monkeypatch):
    c,f,s,r,calls,seen=setup(monkeypatch);c.observe(s,r)
    f.robot.get_dof_position_targets=lambda:np.ones((1,12))
    with pytest.raises(RuntimeError,match='readback'):c.command(s)
    assert len(calls)==1
    with pytest.raises(RuntimeError,match='readback'):c.command(s)
    assert len(calls)==1


def test_native_packet_is_representable_bounded_and_preserves_idle_state():
    rng=np.random.default_rng(42)
    for _ in range(200):
        q=np.degrees(rng.uniform(-2.5,2.5,7).astype(np.float32).astype(float))
        desired=q+rng.uniform(-.2,.2,7)
        packet=native_command_packet(q,desired)
        np.testing.assert_array_equal(np.degrees(np.radians(packet).astype(np.float32).astype(float)),packet)
        assert np.all((packet-q)*(desired-q)>=0)
        assert np.all(abs(packet-q)<=abs(desired-q))
        np.testing.assert_array_equal(native_command_packet(q,q),q)


def test_sub_ulp_motion_is_not_rounded_past_request():
    q=np.degrees(np.full(7,np.float32(1.2345)).astype(float))
    for delta in (-1e-10,1e-10):
        np.testing.assert_array_equal(native_command_packet(q,q+delta),q)


def test_packet_rejects_non_native_measurement_provenance():
    with pytest.raises(ValueError,match='provenance'):native_command_packet(np.full(7,.2),np.full(7,.3))


@pytest.mark.parametrize('fault',['status','completed','sample','receipt','receipt_sample'])
def test_completion_requires_same_step_helper_and_native_receipt(fault):
    w=dict(status='complete',right_withdrawal_completed=True,sample_id=[2,3],
        receipt=dict(passed=True,sample_id=[2,3]))
    record=dict(measured_withdrawal=w)
    assert measured_completion(record,(2,3))
    if fault=='status':w['status']='moving'
    elif fault=='completed':w['right_withdrawal_completed']=1
    elif fault=='sample':w['sample_id']=[2,2]
    elif fault=='receipt':w['receipt']['passed']=False
    else:w['receipt']['sample_id']=[2,2]
    assert not measured_completion(record,(2,3))
