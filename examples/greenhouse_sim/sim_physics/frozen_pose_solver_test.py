from types import SimpleNamespace as S
import numpy as np
import pytest
from greenhouse_sim.robot_kinematics import Rby1Kinematics
from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES
from .frozen_pose_solver import FrozenRightPoseSolver


def case():
    kin=Rby1Kinematics();kin.set_default_torso_degrees(np.zeros(6))
    seed=np.array([SDK_READY_POSE_DEGREES[f'right_arm_{i}'] for i in range(7)])
    base=np.eye(4);pose=kin.forward('right',seed,base)
    return kin,pose,seed,base


def test_exact_stock_repeat_reuses_only_immutable_ik_not_native_checks():
    kin,p,q,b=case();guards=[];solver=FrozenRightPoseSolver(kin,lambda:guards.append(1),enabled=True)
    expected=kin.solve_pose('right',p,q,b,maximum_evaluations=200,joint_limit_margin_degrees=3.)
    a=solver.solve(p,q,b);repeat=solver.solve(p.copy(),q.copy(),b.copy())
    assert a==repeat==expected and a.succeeded
    assert solver.receipt()['actual_solves']==1 and solver.receipt()['cache_hits']==1
    assert len(guards)==4
    assert not solver.receipt()['native_clearance_cached']
    assert not solver.receipt()['motion_authorized']


def test_default_has_no_cache_and_unknown_solver_is_not_reused():
    kin,p,q,b=case();solver=FrozenRightPoseSolver(kin,lambda:None)
    solver.solve(p,q,b);solver.solve(p,q,b)
    assert solver.solves==2 and not solver.entries
    calls=[]
    fake=S(solve_pose=lambda *a,**k:(calls.append(1) or S(succeeded=False)))
    other=FrozenRightPoseSolver(fake,lambda:None,enabled=True)
    other.solve(p,q,b);other.solve(p,q,b)
    assert len(calls)==2 and not other.receipt()['enabled']


@pytest.mark.parametrize('kind',['pose','seed','base'])
def test_exact_input_change_requires_another_solve(kind):
    kin,p,q,b=case();solver=FrozenRightPoseSolver(kin,lambda:None,enabled=True)
    solver.solve(p,q,b)
    if kind=='pose':p[0,3]+=1e-6
    elif kind=='seed':q[6]+=1e-6
    else:b[0,3]+=1e-6
    solver.solve(p,q,b)
    assert solver.solves==2 and solver.hits==0


@pytest.mark.parametrize('kind',['torso','origin','axis','limit','implementation','ancestry'])
def test_mutated_model_is_revoked_even_for_identical_query(kind):
    kin,p,q,b=case();solver=FrozenRightPoseSolver(kin,lambda:None,enabled=True)
    solver.solve(p,q,b)
    if kind=='torso':kin.set_default_torso_degrees([0,1,0,0,0,0])
    elif kind=='origin':kin._by_name['right_arm_0'].origin[0,3]+=.001
    elif kind=='axis':kin._by_name['right_arm_0'].axis[0]+=.001
    elif kind=='implementation':kin.forward=lambda *a:np.eye(4)
    elif kind=='limit':
        from dataclasses import replace
        j=kin._by_name['right_arm_0'];kin._by_name[j.name]=replace(j,lower_rad=j.lower_rad+.01)
    else:kin._by_child.pop('ee_right')
    with pytest.raises(RuntimeError,match='[Ff]rozen'):solver.solve(p,q,b)
    assert solver.hits==0 and solver.solves==1


def test_epoch_failure_on_hit_is_not_hidden():
    kin,p,q,b=case();stale=False
    def guard():
        if stale:raise RuntimeError('changed native epoch')
    solver=FrozenRightPoseSolver(kin,guard,enabled=True);solver.solve(p,q,b);stale=True
    with pytest.raises(RuntimeError,match='epoch'):solver.solve(p,q,b)
    assert solver.hits==0


def test_model_change_during_solve_does_not_publish_or_cache_result():
    kin,p,q,b=case();calls=[]
    def guard():
        calls.append(1)
        if len(calls)==2:kin.set_default_torso_degrees([0,1,0,0,0,0])
    solver=FrozenRightPoseSolver(kin,guard,enabled=True)
    with pytest.raises(RuntimeError,match='torso changed'):solver.solve(p,q,b)
    assert not solver.entries


def test_cache_saturation_keeps_computing_without_losing_checks():
    kin,p,q,b=case();solver=FrozenRightPoseSolver(kin,lambda:None,enabled=True)
    result=solver.solve(p,q,b);solver.entries={i:result for i in range(128)}
    solver.solve(p,q,b);solver.solve(p,q,b)
    assert solver.solves==3 and solver.hits==0 and len(solver.entries)==128


@pytest.mark.parametrize('kind',['pose','seed','base'])
def test_nonfinite_inputs_never_cached(kind):
    kin,p,q,b=case();solver=FrozenRightPoseSolver(kin,lambda:None,enabled=True)
    {'pose':p,'seed':q,'base':b}[kind].flat[0]=np.nan
    with pytest.raises(ValueError,match='Finite'):solver.solve(p,q,b)
    assert not solver.entries and solver.solves==0
