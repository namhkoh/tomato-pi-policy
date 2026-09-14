"""Dispatch/failure contracts with synthetic checks, not native evidence."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from .bimanual import BimanualRobot
from .knife import DOWNWARD_CUT_MODEL


def setup(monkeypatch,*,endpoint_clear=True,full_result=True):
    import sim_physics.rigid_tool_screen as rigid
    import sim_physics.redundant_ik as ik
    r=BimanualRobot.__new__(BimanualRobot)
    r.staged_downward_transit=True;r.joint_transit_fallback=True;r.cut_style='downward';r.cut_model=DOWNWARD_CUT_MODEL
    r.stage=None;r.root='/World/R';r.rig=S(root='/World/P');r.radius=.003
    r.goal=np.eye(4);r.right=np.zeros(7);r.base=np.eye(4);r.stroke_offsets=np.linspace(-.008,.005,27)
    r.held_plant_screen=S(workspace=[np.zeros(3),np.ones(3)],static=[],snapshot=lambda f:None)
    axis=np.array([.66,.70,.27]);axis/=np.linalg.norm(axis)
    r.seam=lambda f:(np.zeros(3),axis)
    r.knife=S(size=np.array([.001,.028,.001]),wrist_for_edge=lambda *a:np.eye(4))
    r.kin=S(forward=lambda *a:np.eye(4),inter_arm_clearance=lambda *a:S(clearance_m=.02))
    r.solve_right_pose=lambda *a:S(succeeded=True,joint_degrees=np.ones(7),evaluations=1)
    r.check_self=lambda *a:dict(passed=endpoint_clear)
    r.check_held_plant=lambda *a:True
    attempts=[]
    def candidate(*a):
        assert r._joint_transit_proposal
        attempts.append(a)
        if isinstance(full_result,Exception):raise full_result
        if full_result:r.plan={'complete_synthetic_path':True}
        return full_result
    r._try_cut_candidate=candidate
    class Rigid:
        def __init__(self,*a):pass
        def check(self,path,*,stroke=True):return dict(passed=stroke)
    monkeypatch.setattr(rigid,'RigidToolScreen',Rigid)
    monkeypatch.setattr(ik,'pose_family',lambda *a,**kw:())
    return r,attempts


def test_failed_cartesian_templates_can_request_full_joint_path(monkeypatch):
    r,calls=setup(monkeypatch)
    r._plan_cut([],np.zeros(7))
    assert len(calls)==1 and not r._joint_transit_proposal
    assert r.plan['approach_joint_fallback'] and r.plan['stroke_basis']=='world_vertical'


@pytest.mark.parametrize('blocked_endpoint',[True,False])
def test_endpoint_or_full_path_failure_cannot_publish_a_plan(monkeypatch,blocked_endpoint):
    r,calls=setup(monkeypatch,endpoint_clear=not blocked_endpoint,full_result=False)
    with pytest.raises(RuntimeError,match='No bimanual'):r._plan_cut([],np.zeros(7))
    assert r.plan is None and not getattr(r,'_joint_transit_proposal',False)
    assert bool(calls) is (not blocked_endpoint)


def test_native_query_exception_restores_dispatch_and_propagates(monkeypatch):
    r,calls=setup(monkeypatch,full_result=RuntimeError('native epoch invalidated'))
    with pytest.raises(RuntimeError,match='epoch'):r._plan_cut([],np.zeros(7))
    assert len(calls)==1 and not r._joint_transit_proposal and r.plan is None


@pytest.mark.parametrize('fallback',[False,True])
def test_known_endpoint_ik_failure_does_not_repeat_useless_rigid_transits(monkeypatch,fallback):
    import sim_physics.rigid_tool_screen as rigid
    r,calls=setup(monkeypatch);r.joint_transit_fallback=fallback;solves=[];transits=[]
    def solve(*a):
        solves.append(1)
        return S(succeeded=False,joint_degrees=np.zeros(7),evaluations=1,
            position_error_m=.1,orientation_error_rad=.2)
    r.solve_right_pose=solve
    class Rigid:
        def __init__(self,*a):pass
        def check(self,path,*,stroke=True):
            if not stroke:transits.append(1)
            return dict(passed=True)
    monkeypatch.setattr(rigid,'RigidToolScreen',Rigid)
    with pytest.raises(RuntimeError,match='No bimanual'):r._plan_cut([],np.zeros(7))
    attempts=r.plan_diagnostics['endpoint_attempts']
    assert len(solves)==len(attempts)==50
    assert len(transits)==(0 if fallback else 50)
    assert all(a['rejection']=='endpoint_IK' for a in attempts)
    assert not calls and r.plan is None
