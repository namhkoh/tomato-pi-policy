import numpy as np
import pytest
from sim_physics.force_closure import ForceClosure
from sim_physics.finger_effort import command,FingerEffort
from sim_physics.finger_target_antiwindup import project


def test_profile_changes_controller_not_hard_contact_or_compression_limit():
    old=ForceClosure(.003,.001);new=ForceClosure(.003,.001,retention_preload=True)
    assert old.desired_support_n==.12 and old.drive_limit_n==.15 and old.backoff_contact_n==.3
    assert new.desired_support_n==.24 and new.drive_limit_n==.3 and new.backoff_contact_n==.4
    assert old.minimum==new.minimum==.002
    new.command(0,step=0,dt=1/240)
    assert new.receipt['native_hard_contact_limit_n']==.5
    assert not new.receipt['grasp_verified']
    assert .24*1.5<new.backoff_contact_n<.5


@pytest.mark.parametrize('preload',[False,True])
def test_force_guard_is_not_relaxed_for_either_profile(preload):
    c=ForceClosure(.003,.001,retention_preload=preload);c.command(0,step=0,dt=1/240)
    evidence=dict(step_id=1,adapter_valid=True,normal_only=True,compressive_support_n=[.1,.1],stem_only=True)
    with pytest.raises(RuntimeError,match='unsafe'):
        c.observe(evidence,[.5,.1],step=1,guards_passed=True)


def test_higher_preload_backs_off_before_unchanged_hard_limit():
    c=ForceClosure(.003,.001,retention_preload=True)
    c.gaps[:]=.003;c.loads[:]=.41;c.support[:]=.24
    before=c.gaps.copy();after=c.command(1,step=0,dt=1/240)
    np.testing.assert_allclose(after,before+.005/240)
    assert c.receipt['maximum_non_gravity_drive_effort_n']==.3


def test_explicit_preload_changes_only_pd_budget_and_keeps_total_bounded():
    args=([-.004,.004],[0,0],[-.002,.002],[-.158,-.158],[.3,.3])
    with pytest.raises(ValueError):command(*args,dt=1/240)
    total,r=command(*args,dt=1/240,retention_preload=True)
    np.testing.assert_allclose(r['submitted_pd_n'],[.3,-.3])
    np.testing.assert_allclose(total,[.142,-.458])
    assert max(abs(total))<.8 and not r['actual_drive_effort_measured']
    with pytest.raises(ValueError):command(*args[:-1],[.301,.3],dt=1/240,retention_preload=True)


def test_preload_projection_tracks_higher_bounded_pd_without_changing_physical_state():
    args=([.005,.005],[-.003,.003],[0,0],[.3,.3])
    with pytest.raises(ValueError):project(*args,minimum=.002)
    gaps,r=project(*args,minimum=.002,retention_preload=True)
    np.testing.assert_allclose(gaps,[.0045,.0045])
    assert max(abs(np.array(r['resulting_unclipped_pd_n'])))<=.3+1e-12
    assert not r['changes_physical_state']


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_profile_flags_require_booleans(value):
    with pytest.raises(ValueError):ForceClosure(.003,.001,retention_preload=value)
    with pytest.raises(ValueError):command([0,0],[0,0],[0,0],[0,0],[.15,.15],dt=1/240,retention_preload=value)
    with pytest.raises(ValueError):project([.003,.003],[-.003,.003],[0,0],[.15,.15],minimum=.002,retention_preload=value)


def test_bound_native_profile_cannot_change_mid_trial():
    from sim_physics.finger_effort_test import fixture
    f=fixture();f.retention_preload=True;f.force_limits[0,1:]=.3
    controller=FingerEffort(f);controller.apply(f,step=0,dt=1/240)
    assert controller.receipt['maximum_non_gravity_pd_n']==.3
    f.retention_preload=False
    with pytest.raises(RuntimeError,match='profile changed'):controller.apply(f,step=1,dt=1/240)


def test_native_float32_cap_roundoff_is_not_an_extra_force_budget():
    args=([-.004,.004],[0,0],[-.002,.002],[0,0])
    total,_=command(*args,np.full(2,.3,dtype=np.float32),dt=1/240,retention_preload=True)
    assert max(abs(total))==float(np.float32(.3))
    with pytest.raises(ValueError):command(*args,[.3+2e-8,.3],dt=1/240,retention_preload=True)


def test_cli_requires_complete_isolated_protocol_before_kit(tmp_path):
    from sim_physics.benchmark import main,parser
    assert not parser().parse_args(['--output','unused']).retention_preload
    with pytest.raises(ValueError,match='Retention preload'):
        main(['--output',str(tmp_path/'unused'),'--retention-preload'])
    assert not (tmp_path/'unused').exists()
