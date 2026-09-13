from pathlib import Path
import numpy as np
import pytest
from .diagnostic_rate import frequency,matches
from .force_closure import ForceClosure
from .force_closure_test import sample
from .finger_effort import command
from .blade_feed import BladeFeed
from .retention_settle import PreloadSettle
from .retention_settle_test import record
from .retention_preflight import assess
from .retention_preflight_test import case
from .native_spring_handoff_test import source
from .native_spring_observer import restore_after_release


@pytest.mark.parametrize('bad',[120,960,1920,0,True,480.,'480',None])
def test_only_explicit_bounded_rate(bad):
    with pytest.raises(ValueError):frequency(bad)


@pytest.mark.parametrize('hz',[240,480])
def test_force_units_limits_and_controller_speed_preserved(hz):
    assert matches(1/hz,hz) and not matches(1/60,hz)
    c=ForceClosure(.003,.0005,physics_hz=hz)
    for step in range(hz):
        before=c.gaps.copy();c.command(1.,step=step,dt=1/hz)
        if np.max(before)<=c.slow_gap+1e-12:
            assert np.max(abs(c.gaps-before))<=.0005/hz+1e-12
        c.observe(sample(step+1),[0.,0.],step=step+1,guards_passed=True)
    total,receipt=command([-.003,.003],[0.,0.],[-.0015,.0015],[.3,.3],[.3,.3],
        dt=1/hz,retention_preload=True,physics_hz=hz)
    np.testing.assert_allclose(total,[.6,0.],atol=1e-15)
    assert receipt['maximum_non_gravity_pd_n']==.3
    with pytest.raises(ValueError):command([-.003,.003],[0.,0.],[-.002,.002],[0.,0.],[.3,.3],
        dt=1/60,retention_preload=True,physics_hz=hz)
    feed=BladeFeed([-.008,.008],radius=.003,physics_hz=hz)
    assert (feed.loads.maxlen-1)/hz==.025


def test_480_preload_requires_full_200ms_not_48_ticks():
    s=PreloadSettle(physics_hz=480)
    for i in range(1,97):
        step=1920+i;r=record(step);r['t']=step/480;r['grasp_dynamics']['dt_s']=1/480
        result=s.observe(r,step=step,time_s=step/480,start_time_s=4.)
        assert result['state']==('ready' if i==96 else 'waiting')
    assert result['required_steps']==96 and result['required_dwell_s']==.2
    s=PreloadSettle(physics_hz=480);r=record(1921)
    with pytest.raises(ValueError):s.observe(r,step=1921,time_s=r['t'],start_time_s=4.)


def test_static_preflight_rate_binding_keeps_gravity_and_force_budgets():
    r,kw=case();original=assess(r,**kw)
    r['grasp_dynamics'].update(step_id=960,dt_s=1/480);kw.update(step=960,physics_hz=480)
    high=assess(r,**kw)
    assert high['gravity_wrench_n_nm']==original['gravity_wrench_n_nm']
    assert high['capacity']==original['capacity']
    kw.pop('physics_hz')
    with pytest.raises(ValueError):assess(r,**kw)


def test_native_handoff_keeps_original_k_c_and_refuses_wrong_clock():
    s,a,w=source();native=restore_after_release(s,physics_hz=480)
    assert native.step(1/480,root_constrained=False) is None
    with pytest.raises(ValueError):native.step(1/240,root_constrained=False)
    np.testing.assert_array_equal(a.values['stiffness'][0],s.k)
    assert w==['efforts','stiffness','damping']


@pytest.mark.parametrize('hz',[240,480])
def test_contact_impulse_conversion_and_fresh_frame_binding(hz):
    from .shaft_grasp_native_test import fixture,bind,add,tensor
    f=fixture();a=bind(f,physics_hz=hz);a.begin_step()
    try:
        a.capture_contact_frames(f.frames,f.fingerframes,step_id=0)
        for i in range(2):
            # Synthetic helper authors J=0.01*force, so give it the exact
            # impulse corresponding to the same30 mN load at either rate.
            add(a,f,i,force=.03/(hz*.01));tensor(f,i,force=.03)
        r=a.evaluate(1/hz,f.frames,f.fingerframes,frames_step_id=1,require_pre_step_frames=True)
        assert r['bilateral'] and r['adapter_valid']
        np.testing.assert_allclose(r['forces'],[[.03,0,0],[-.03,0,0]])
        assert not r['native_grasp_certified']
    finally:a.close()


def cli(output):
    return ['--output',str(output),'--scene','package','--isolate-station','--full-robot-probe',
        '--bimanual-cut','--explicit-finger-effort','--isolated-cut-contact-trial','--force-closure',
        '--stem-contact-model','flat_cylinders_v1','--grasp-contact-frames','pre_solve_pgs_v1',
        '--spring-mode','implicit_effort','--uniform-solver-iterations','128','0',
        '--sparse-contacts','--finger-gravity','--compliant-fingers','--seconds','50',
        '--force-newton','0','--solver','PGS','--physics-hz','480','--cut-convergence-trial',
        '--fixed-root-cut-trial','--constraint-mode','fixed_articulation','--diagnostic-grasp-dynamics',
        '--native-drives-after-cut','--require-retention-screen','--physical-grasp-span',
        '--settle-retention-preload','--effort-bounded-grasp-target','--preload-force-servo',
        '--native-startup-clearance','--native-static-clearance','--retention-preload',
        '--symmetric-finger-closure','--finger-target-antiwindup','--blade-force-feed',
        '--seam-contact-compliance','--cut-style','downward','--cut-model','signed_edge_load_brittle_seam_v1']


def test_only_complete_headless_profile_reaches_precreation_boundary(tmp_path,monkeypatch):
    from .benchmark import main,parser
    assert not parser().parse_args(['--output','unused']).cut_convergence_trial
    out=tmp_path/'unused';args=cli(out)
    class Validated(Exception):pass
    def stop(self,*a,**kw):
        assert self==out
        raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(args)
    for missing in ('--cut-convergence-trial','--require-retention-screen','--physical-grasp-span','--native-drives-after-cut'):
        with pytest.raises(ValueError):main([a for a in args if a!=missing])
    for flag in ('--gui','--measured-withdrawal','--robot-interactive'):
        with pytest.raises(ValueError):main(args+[flag])
    assert not out.exists()
