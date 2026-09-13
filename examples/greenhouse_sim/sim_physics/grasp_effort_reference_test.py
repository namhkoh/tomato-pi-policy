"""Command-reference regression; synthetic observations are not grasp proof."""
import numpy as np
import pytest
from .force_closure import ForceClosure
from .finger_target_antiwindup import project
from .finger_effort import command


def controller(**kw):
    return ForceClosure(.003,.001,retention_preload=True,symmetric=True,
        pregrasp_half_aperture=.008,**kw)


def test_reference_bias_can_request_original_preload_without_raising_effort_caps():
    old=controller();new=controller(effort_bounded_target=True)
    assert old.minimum==pytest.approx(.002) and new.minimum==pytest.approx(.0015)
    assert 200*(.003-old.minimum)==pytest.approx(.2)
    assert 200*(.003-new.minimum)==pytest.approx(.30)
    for key in ('desired_support_n','drive_limit_n','backoff_contact_n','slow_gap','opening'):
        assert getattr(old,key)==getattr(new,key)
    target=.003-new.desired_support_n/200
    gap,receipt=project([target]*2,[-.003,.003],[0,0],[.3,.3],minimum=new.minimum,
        maximum=new.opening,retention_preload=True,symmetric=True)
    effort,applied=command([-.003,.003],[0,0],[-gap[0],gap[1]],[-.139,-.139],[.3,.3],
        dt=1/240,retention_preload=True)
    np.testing.assert_allclose(applied['submitted_pd_n'],[.24,-.24])
    assert max(abs(effort))<.8 and not receipt['changes_physical_state']


def test_slow_contact_approach_and_reference_floor_remain_bounded():
    c=controller(effort_bounded_target=True)
    for step in range(2600):
        before=c.gaps.copy();after=c.command(1.,step=step,dt=1/240)
        if max(before)<=c.slow_gap+1e-12:assert max(abs(after-before))<=.0005/240+1e-12
        assert min(after)>=c.minimum and max(after)<=c.opening
        c.observe(dict(step_id=step+1,adapter_valid=True,normal_only=True,stem_only=True,
            compressive_support_n=[0.,0.]),[0.,0.],step=step+1,guards_passed=True)
    assert c.receipt['actual_native_penetration_guard_m']==.001
    assert c.receipt['native_hard_contact_limit_n']==.5
    assert not c.receipt['grasp_verified'] and c.receipt['maximum_non_gravity_drive_effort_n']==.30


def test_measured_penetration_guard_remains_independent_of_reference_bias():
    from .shaft_grasp_test import setup,contact,result
    e,frames,links=setup()
    contact(e,0,separation=-.001001);contact(e,1)
    r=result(e,frames,links)
    assert not r['bilateral']
    assert r['rejected'][0]['reason']=='separation_outside_native_contact_guards'


@pytest.mark.parametrize('kw',[{'retention_preload':False,'symmetric':True},
    {'retention_preload':True,'symmetric':False}])
def test_no_unqualified_controller_profile(kw):
    with pytest.raises(ValueError):ForceClosure(.003,.001,effort_bounded_target=True,**kw)


def test_thin_shaft_cannot_request_negative_aperture():
    with pytest.raises(ValueError,match='zero-aperture'):
        ForceClosure(.0014,.001,retention_preload=True,symmetric=True,effort_bounded_target=True)


def test_cli_requires_the_complete_native_effort_retention_trial(tmp_path):
    from .benchmark import parser,main
    assert not parser().parse_args(['--output','unused']).effort_bounded_grasp_target
    with pytest.raises(ValueError,match='Effort-bounded target'):
        main(['--output',str(tmp_path/'unused'),'--effort-bounded-grasp-target'])
    assert not (tmp_path/'unused').exists()


def test_robot_report_separates_actual_reference_floor_from_legacy_bias(monkeypatch):
    from .bimanual import BimanualRobot
    from .full_robot import FullRobotGripper
    monkeypatch.setattr(FullRobotGripper,'report',lambda self:{})
    r=object.__new__(BimanualRobot)
    for name,value in dict(radius=.003,grasp_compression=.001,force_closure_enabled=True,
            force_closer=controller(effort_bounded_target=True),effort_bounded_grasp_target=True,
            physical_grasp_span=True,minimum_grasp_self_clearance=.01,knife_mount={},
            blade_contacts={},arc_contacts={},plan_diagnostics={},plan=None).items():setattr(r,name,value)
    report=r.report()['grasp_closure']
    assert report['minimum_commanded_half_aperture_m']==pytest.approx(.0015)
    assert report['maximum_nominal_target_bias_m']==pytest.approx(.0015)
    assert report['actual_native_penetration_guard_m']==.001
