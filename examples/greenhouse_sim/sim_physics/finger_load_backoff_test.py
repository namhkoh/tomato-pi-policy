"""Controller-reference checks; measured native holding still needs a trial."""
from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
import pytest
from .finger_target_antiwindup import project
from .finger_effort import command


def recorded():
    # native418, last pre-step before the post-release 0.567N stop.
    return ([.0018830413464456797]*2,
        [-.0031724313739687204,.003174042096361518],
        [-.007152721285820007,.0073226215317845345],[.30,.30])


def test_measured_opening_jaws_cannot_outrun_force_backoff_reference():
    args=recorded();settings=dict(minimum=.0015,retention_preload=True,symmetric=True)
    old,_=project(*args,**settings)
    gaps,r=project(*args,**settings,closing_effort_cap_n=.12)
    assert np.all(gaps>old) and gaps[0]==gaps[1]
    pd=np.array(r['resulting_unclipped_pd_n'])
    assert np.max(np.array([1.,-1.])*pd)<=.12+1e-14
    assert np.max(abs(pd))<=.30 and not r['changes_physical_state']
    total,e=command(args[1],args[2],np.array([-1.,1.])*gaps,[-.24846673]*2,args[3],
        dt=1/480,retention_preload=True,physics_hz=480)
    np.testing.assert_allclose(e['submitted_pd_n'],pd,atol=1e-14)
    assert max(abs(total))<.37 and not e['measured_contact_used_as_effort']


@pytest.mark.parametrize('cap',[False,True,float('nan'),float('inf'),-.01,.31,[.12,.12]])
def test_invalid_closing_cap_cannot_change_targets(cap):
    with pytest.raises(ValueError):project(*recorded(),minimum=.0015,
        retention_preload=True,symmetric=True,closing_effort_cap_n=cap)


@pytest.mark.parametrize('options',[{},dict(symmetric=True),dict(retention_preload=True)])
def test_backoff_requires_symmetric_retention_profile(options):
    with pytest.raises(ValueError):project(*recorded(),minimum=.0015,closing_effort_cap_n=.12,**options)


def test_no_intersection_refuses_instead_of_relaxing_effort_or_geometry():
    with pytest.raises(RuntimeError,match='No symmetric'):
        project([.025]*2,[-.049,.049],[0,0],[.3,.3],minimum=.0015,
            retention_preload=True,symmetric=True,closing_effort_cap_n=.12)


@pytest.mark.parametrize('loads,servo,expected',[
    ([.35,.39],True,None),([.4,.3],True,None),([.41,.39],False,None),
    ([.41,.39],True,.12),([.1,.49],True,.12)])
@pytest.mark.parametrize('experiment',[False,True])
def test_native_close_uses_fresh_all_contact_load_only_in_explicit_experiment(loads,servo,expected,experiment):
    from .bimanual import BimanualRobot
    from .force_closure import ForceClosure
    c=ForceClosure(.003,.0005,retention_preload=True,symmetric=True,
        effort_bounded_target=True,preload_force_servo=servo,physics_hz=480)
    c.gaps[:]=.0019;c.support[:]=.24;c.loads[:]=loads
    a=Mock();a.get_dof_positions.return_value=np.array([recorded()[1]])
    a.get_dof_velocities.return_value=np.array([recorded()[2]])
    r=S(force_closure_enabled=True,finger_target_antiwindup=True,force_closer=c,
        measured_jaw_backoff_trial=experiment,
        force_limits=np.array([[.3,.3]]),finger_indices=[0,1],targets=np.zeros((1,2)),index=[0],robot=a)
    BimanualRobot.close(r,1.,step=0,dt=1/480)
    receipt=c.receipt['antiwindup']
    if not experiment:expected=None
    assert receipt['closing_effort_backoff_cap_n']==expected
    assert receipt['closing_effort_backoff_uses_measured_state']==(expected is not None)
    np.testing.assert_array_equal(r.force_limits,[[.3,.3]])
    a.set_dof_positions.assert_not_called();a.set_dof_velocities.assert_not_called()
    # The existing contiguous observation contract remains mandatory.
    with pytest.raises(RuntimeError,match='Fresh preceding'):
        BimanualRobot.close(r,1.,step=1,dt=1/480)


def test_backoff_experiment_not_in_default_or_watched_profile(tmp_path,monkeypatch):
    from .benchmark import parser,main
    from .ground_truth_trial import main as public,arguments
    assert not parser().parse_args(arguments('unused','bimanual','cut_action',watch=True)).measured_jaw_backoff_trial
    out=tmp_path/'unused'
    with pytest.raises(ValueError,match='Measured jaw backoff'):
        main(['--output',str(out),'--measured-jaw-backoff-trial'])
    with pytest.raises(SystemExit):public(['--output',str(out),'--mode','bimanual','--measured-jaw-backoff-trial'])
    assert not out.exists()
    import sim_physics.benchmark as benchmark
    calls=[];monkeypatch.setattr(benchmark,'main',lambda argv:calls.append(parser().parse_args(argv)))
    public(['--output',str(out),'--mode','bimanual','--milestone','cut_action',
        '--process-zone-trial','--through-stroke-trial','--native-retention-trial','--measured-jaw-backoff-trial'])
    assert calls[0].measured_jaw_backoff_trial and calls[0].native_retention_trial
