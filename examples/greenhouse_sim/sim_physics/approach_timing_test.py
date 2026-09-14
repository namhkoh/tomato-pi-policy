from types import SimpleNamespace as S
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from .approach_timing import ApproachTiming


def robot(path):
    def forward(side,q,base):
        frame=np.eye(4);frame[0,3]=q[0]/20
        frame[:3,:3]=Rotation.from_euler('z',q[1],degrees=True).as_matrix()
        return frame
    return S(plan={'approach':path},kin=S(forward=forward),base=np.eye(4))


def test_nonuniform_knots_keep_exact_path_and_bound_interpolated_speed():
    q=np.zeros((5,7));q[:,0]=[0,1,1.001,5,10];q[:,1]=[0,20,21,30,40]
    saved=q.copy();r=robot(q);clock=ApproachTiming(r)
    t=np.linspace(0,clock.duration,3001)
    f=np.array([clock.fraction(v) for v in t]);where=f*(len(q)-1)
    index=np.minimum(where.astype(int),len(q)-2);a=where-index
    actual=(1-a[:,None])*q[index]+a[:,None]*q[index+1]
    speed=np.abs(np.diff(actual,axis=0))/np.diff(t)[:,None]
    assert np.max(speed)<=30.+1e-8 and np.max(speed[:,0])/20<=.1+1e-8
    assert np.max(speed[:,1])<=30.+1e-8
    assert f[0]==0 and f[-1]==1 and np.all(np.diff(f)>=0)
    np.testing.assert_array_equal(q,saved)
    for i,knot in enumerate(clock.knots):
        # Every original knot is on the SAME interpolated path, not a spline.
        u=knot/clock.knots[-1]
        from scipy.optimize import brentq
        phase=0. if u==0 else 1. if u==1 else brentq(lambda z:z*z*(3-2*z)-u,0,1)
        assert clock.fraction(phase*clock.duration)==pytest.approx(i/(len(q)-1),abs=1e-10)
    assert clock.duration>=4 and not clock.report()['path_knots_changed']
    assert not clock.report()['motion_authorized']
    assert not clock.report()['cut_feed_or_postrelease_deadlines_changed']


def test_noop_and_duplicate_knots_have_finite_bounded_time():
    clock=ApproachTiming(robot(np.zeros((8,7))))
    assert clock.duration==4. and clock.fraction(10)==1.
    assert clock.fraction(0)==0 and np.isfinite(clock.fraction(2))


@pytest.mark.parametrize('path',[np.zeros((1,7)),np.zeros((5,6)),np.full((3,7),np.nan),np.zeros((10001,7))])
def test_invalid_path_is_never_timed(path):
    with pytest.raises(ValueError):ApproachTiming(robot(path))


def test_unreasonably_long_approach_rejected_not_silently_accelerated():
    q=np.zeros((2,7));q[-1,0]=180.
    with pytest.raises(RuntimeError,match='45s'):ApproachTiming(robot(q))


@pytest.mark.parametrize('value',[True,None,float('nan'),-1.])
def test_invalid_progress_time_fails(value):
    clock=ApproachTiming(robot(np.zeros((2,7))))
    with pytest.raises((ValueError,TypeError)):clock.fraction(value)


def test_launcher_requires_complete_neutral_cut_profile_before_native_start(monkeypatch,tmp_path):
    from . import ground_truth_trial
    class BeforeWrite(Exception):pass
    def stop(*a,**k):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'run'),'--mode','right_only','--milestone','cut_action',
        '--process-zone-trial','--through-stroke-trial','--material-clearance-trial',
        '--postcut-egress-trial','--rate-limited-approach-trial','--park-left-ready']
    with pytest.raises(ValueError,match='Rate-limited approach'):ground_truth_trial.main(args)
    with pytest.raises(BeforeWrite):ground_truth_trial.main(args+['--neutral-ready-start','--torso-degrees','0','0','0','0','0','0'])


def test_extended_horizon_still_checks_final_withdrawal_and_preserves_deadlines():
    import inspect
    from . import bimanual_probe
    source=inspect.getsource(bimanual_probe.run)
    assert 'while clock.stamp.step<total_steps' in source
    assert 'if rig.cut and stamp.step==total_steps' in source
    assert 'completed=len(records)==total_steps' in source
    assert 'postrelease_deadlines_changed=False' in source


def test_velocity_solver_comparison_is_explicit_guarded_and_keeps_default(monkeypatch,tmp_path):
    from . import ground_truth_trial,benchmark
    received=[]
    monkeypatch.setattr(benchmark,'main',lambda argv:received.append(argv))
    base=['--output',str(tmp_path/'run'),'--mode','right_only','--milestone','cut_action']
    with pytest.raises(SystemExit):ground_truth_trial.main(base+['--velocity-solver-trial'])
    complete=base+['--process-zone-trial','--through-stroke-trial','--material-clearance-trial','--postcut-egress-trial']
    ground_truth_trial.main(complete)
    args=received[-1];i=args.index('--uniform-solver-iterations')
    assert args[i+1:i+3]==['128','0']
    ground_truth_trial.main(complete+['--velocity-solver-trial'])
    args=received[-1];i=len(args)-1-args[::-1].index('--uniform-solver-iterations')
    assert args[i+1:i+3]==['128','8']
    with pytest.raises(SystemExit):ground_truth_trial.main(complete+['--velocity-solver-trial','--solver-convergence-trial','96'])
