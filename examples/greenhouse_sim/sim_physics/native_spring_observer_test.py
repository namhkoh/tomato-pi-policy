from types import SimpleNamespace
import numpy as np
import pytest
from sim_physics.native_spring_observer import NativeSpringObserver
from sim_physics.benchmark import main


class View:
    count=1
    shared_metatype=SimpleNamespace(dof_names=['a','b'])
    def __init__(self):
        self.values={'get_dof_stiffnesses':[[.4,.5]],'get_dof_dampings':[[.1,.2]],
            'get_dof_position_targets':[[0,0]],'get_dof_velocity_targets':[[0,0]],
            'get_dof_actuation_forces':[[0,0]]}
    def __getattr__(self,key):
        if key in self.values:return lambda:np.array(self.values[key])
        raise AssertionError('Observer must not call native setters: '+key)


def test_native_observer_never_submits_or_invents_drive_effort():
    a=View();o=NativeSpringObserver(a)
    assert o.step(1/240,root_constrained=True) is None
    np.testing.assert_array_equal(o.k,[.4,.5])


def test_explicit_release_observer_preserves_all_drive_checks():
    a=View();o=NativeSpringObserver(a,allow_release=True)
    assert o.step(1/240,root_constrained=False) is None
    np.testing.assert_array_equal(o.k,[.4,.5])
    a.values['get_dof_actuation_forces'][0][0]=.01
    with pytest.raises(RuntimeError):o.step(1/240,root_constrained=False)


@pytest.mark.parametrize('flag',[1,None,'true'])
def test_no_implicit_release_flag(flag):
    with pytest.raises(ValueError):NativeSpringObserver(View(),allow_release=flag)


@pytest.mark.parametrize('extra',[[],['--isolated-cut-contact-trial','--spring-mode','implicit_effort'],
    ['--spring-mode','native']])
def test_native_cut_flag_cannot_skip_complete_trial_protocol(tmp_path,extra):
    output=tmp_path/'unused'
    with pytest.raises(ValueError):
        main(['--output',str(output),'--native-spring-cut-trial',*extra])
    assert not output.exists()


@pytest.mark.parametrize('key',list(View().values))
def test_mutated_drive_contract_stops_before_next_native_step(key):
    a=View();o=NativeSpringObserver(a);a.values[key][0][0]+=.01
    with pytest.raises(RuntimeError):o.step(1/240,root_constrained=True)


@pytest.mark.parametrize('dt,root',[(1/120,True),(float('nan'),True),(True,True),(1/240,False)])
def test_unsupported_timestep_or_detached_state_cannot_continue(dt,root):
    o=NativeSpringObserver(View())
    with pytest.raises(ValueError):o.step(dt,root_constrained=root)


@pytest.mark.parametrize('extra',[[],['--isolate-station'],['--bimanual-hold-control']])
def test_native_mode_does_not_silently_replace_full_robot_default(tmp_path,extra):
    output=tmp_path/'unused'
    with pytest.raises(ValueError):
        main(['--output',str(output),'--scene','package','--full-robot-probe','--bimanual-cut',
            '--sparse-contacts','--finger-gravity','--solver','PGS','--seconds','20',*extra])
    assert not output.exists()


def test_fixed_root_comparison_cannot_enable_cut_or_release(tmp_path):
    output=tmp_path/'unused'
    with pytest.raises(ValueError):
        main(['--output',str(output),'--scene','package','--isolate-station','--full-robot-probe',
            '--bimanual-cut','--constraint-mode','fixed_articulation','--attached-only',
            '--sparse-contacts','--finger-gravity','--solver','PGS','--seconds','20'])
    assert not output.exists()
