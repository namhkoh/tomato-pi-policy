from types import SimpleNamespace as S
import numpy as np
import pytest
from .spring_work import account,capture,finish


def work(q0,q1,k,tau,**kw):
    return account(q0,q1,k,tau,before_step=0,after_step=1,dt=.01,**kw)


def test_midpoint_elastic_force_is_conservative():
    r=work([.1,-.2],[.12,-.18],[.3,.1],[-.033,.019])
    assert abs(r['apparent_constitutive_energy_j'])<1e-18
    assert r['whole_system_energy_balance'] is False
    assert r['effort_basis']=='caller_supplied_constant_generalized_effort_not_authenticated'


def test_backward_euler_spring_damping_is_dissipative():
    q0=np.array([.1,-.2]);q1=np.array([.12,-.18]);k=np.array([.3,.1]);c=.02
    r=work(q0,q1,k,-k*q1-c*(q1-q0)/.01)
    expected=-.5*np.dot(k*(q1-q0),q1-q0)-c*np.dot(q1-q0,q1-q0)/.01
    assert r['apparent_constitutive_energy_j']==pytest.approx(expected)


def test_wrong_predictor_cannot_hide_positive_energy_source():
    r=work([.1],[.2],[1.],[-.01])
    assert r['effort_work_j']==pytest.approx(-.001)
    assert r['quadratic_potential_change_j']==pytest.approx(.015)
    assert r['apparent_constitutive_energy_j']==pytest.approx(.014)


@pytest.mark.parametrize('args',[([0],[1],[1,2],[0]),([0],[1],[-1],[0]),
    ([True],[1],[1],[0]),([0],[float('nan')],[1],[0]),([],[],[],[])])
def test_invalid_vectors(args):
    with pytest.raises(ValueError):work(*args)


@pytest.mark.parametrize('a,b,dt',[(0,0,.01),(0,2,.01),(-1,0,.01),(True,2,.01),(0,1,0),(0,1,True)])
def test_invalid_binding(a,b,dt):
    with pytest.raises(ValueError):account([0],[0],[1],[0],before_step=a,after_step=b,dt=dt)


def fixture():
    q=np.array([[.1,.2]],dtype=np.float32);tau=np.array([[.01,-.02]],dtype=np.float32)
    a=S(shared_metatype=S(dof_names=['x','y']),get_dof_positions=lambda:q,
        get_dof_actuation_forces=lambda:tau)
    return a,q,tau


def test_native_copy_exact_command_and_adjacent_inventory():
    a,q,tau=fixture();s=capture(a,tau[0],step=5);q[:]=.25
    r=finish(s,a,[.1,.2],step=6,dt=.01)
    assert r['q_before_rad']!=r['q_after_rad']
    assert r['submitted_effort_nm']==tau[0].astype(float).tolist()
    assert r['effort_basis']=='constant_actuation_command_readback_not_measured_joint_torque'
    a.shared_metatype.dof_names.reverse()
    with pytest.raises(ValueError,match='inventory'):finish(s,a,[.1,.2],step=6,dt=.01)


def test_native_mismatch_is_not_measured_force_evidence():
    a,q,tau=fixture()
    with pytest.raises(ValueError,match='readback'):capture(a,[0,0],step=0)
