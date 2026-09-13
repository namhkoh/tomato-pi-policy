import numpy as np
import pytest
from .root_transition_test import View
from .implicit_springs import ImplicitJointSprings
from .native_spring_observer import restore_after_release
from .benchmark import main


def source():
    a=View(False);writes=[]
    for name in ('stiffness','damping','targets','velocity_targets'):
        a.values[name].fill(0)
    a.values['q'].fill(.01);a.values['v'].fill(0)
    a.get_drive_types=lambda:np.ones((1,3))
    def setter(name):
        def apply(value,index):
            writes.append(name);a.values[name]=value.copy()
        return apply
    a.set_dof_stiffnesses=setter('stiffness')
    a.set_dof_dampings=setter('damping')
    a.set_dof_actuation_forces=setter('efforts')
    s=object.__new__(ImplicitJointSprings);s.articulation=a;s.fixed_base=False
    s.k=np.array([.2,.1,.12],np.float32).astype(float)
    s.c=np.array([.01,.02,.03],np.float32).astype(float)
    return s,a,writes


def test_original_native_coefficients_no_double_effort_and_no_state_write():
    s,a,w=source();poses=a.values['poses'].copy();q=a.values['q'].copy()
    native=restore_after_release(s)
    assert w==['efforts','stiffness','damping']
    np.testing.assert_array_equal(a.values['stiffness'][0],s.k)
    np.testing.assert_array_equal(a.values['damping'][0],s.c)
    np.testing.assert_array_equal(a.values['q'],q)
    np.testing.assert_array_equal(a.values['poses'],poses)
    assert native.step(1/240,root_constrained=False) is None
    assert not native.handoff_receipt['physical_model_qualified']
    assert native.handoff_receipt['numerical_integrator_changed']


@pytest.mark.parametrize('bad',['fixed','stiffness','damping','targets','velocity_targets','type','nan','wrong_count'])
def test_invalid_start_never_writes(bad):
    s,a,w=source()
    if bad=='fixed':s.fixed_base=True
    elif bad=='type':a.get_drive_types=lambda:np.zeros((1,3))
    elif bad=='nan':s.k[0]=np.nan
    elif bad=='wrong_count':s.c=np.ones(4)
    else:a.values[bad].fill(1)
    with pytest.raises(ValueError):restore_after_release(s)
    assert not w


def test_native_setter_state_corruption_is_not_accepted_or_repaired():
    s,a,w=source();original=a.set_dof_dampings
    def corrupt(value,index):
        original(value,index);a.values['q'][0,0]+=.1
    a.set_dof_dampings=corrupt
    with pytest.raises(RuntimeError,match='changed state'):restore_after_release(s)
    assert len(w)==3


def test_cli_cannot_enable_outside_checked_fixed_root_trial(tmp_path):
    out=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='Post-cut native drives'):
        main(['--output',str(out),'--native-drives-after-cut'])
    assert not out.exists()
