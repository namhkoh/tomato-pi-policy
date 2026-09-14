"""Synthetic path/dispatch contracts; not native grasp evidence."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from .grasp_path_replan import replan


def fixture(monkeypatch,*,alternatives=True,blocked_midpoint=False):
    import sim_physics.redundant_ik as module
    r=S(start=np.eye(4),goal=np.eye(4),initial_q=np.zeros(7),right=np.zeros(7),base=np.eye(4))
    r.goal[0,3]=.02
    def fk(side,q,base):
        pose=np.eye(4);pose[0,3]=q[0]/1000;return pose
    def solve(side,target,seed,base,**kw):
        q=seed.copy();q[0]=target[0,3]*1000
        return S(succeeded=True,joint_degrees=q)
    r.kin=S(arm_limits_degrees=lambda side:(np.full(7,-180.),np.full(7,180.)),
        forward=fk,solve_pose=solve,inter_arm_clearance=lambda *a:S(clearance_m=.02))
    def check(q,right):
        bad=q[6]<np.clip((q[0]-9.5)*.8,0.,.4)-1e-10
        if blocked_midpoint and 0<q[6]<.5:bad=True
        return dict(passed=not bad)
    r.check_self=check
    def family(kin,side,target,seed,base,**kw):
        if alternatives:
            q=seed.copy();q[6]+=.5;yield S(joint_degrees=q,succeeded=True)
    monkeypatch.setattr(module,'pose_family',family)
    return r


def test_same_wrist_path_can_use_nearby_elbow_without_mutating_robot(monkeypatch):
    r=fixture(monkeypatch);original=r.initial_q.copy()
    fractions,path,receipt=replan(r)
    assert receipt['alternative_poses']==1 and receipt['current_plant_screen_required']
    assert not receipt['motion_authorized'] and not receipt['desired_wrist_path_changed']
    np.testing.assert_array_equal(path[0],original);np.testing.assert_array_equal(r.initial_q,original)
    assert np.all(np.diff(fractions)>0) and fractions[-1]==pytest.approx(1.15)
    assert np.max(abs(np.diff(path,axis=0)))<=.5+1e-12
    np.testing.assert_allclose(path[:,0]/1000,fractions*.02,atol=1e-14)
    assert path[-1,6]==.5 and not hasattr(r,'path_q')


@pytest.mark.parametrize('kind',['no_alternative','midpoint','start','IK','exception'])
def test_failed_prerequisite_or_interpolation_never_returns_a_path(monkeypatch,kind):
    r=fixture(monkeypatch,alternatives=kind!='no_alternative',blocked_midpoint=kind=='midpoint')
    if kind=='start':r.initial_q[0]=2.
    if kind=='IK':r.kin.solve_pose=lambda *a,**kw:S(succeeded=False)
    if kind=='exception':
        def error(*a):raise RuntimeError('Invalid model snapshot')
        r.check_self=error
    with pytest.raises(RuntimeError):replan(r)
    assert not hasattr(r,'path_q')


def test_expired_wall_budget_cannot_publish_partial_path(monkeypatch):
    import sim_physics.grasp_path_replan as module
    r=fixture(monkeypatch);times=iter([0.,0.,9.])
    monkeypatch.setattr(module.time,'perf_counter',lambda:next(times))
    with pytest.raises(RuntimeError,match='budget'):replan(r)


def test_bimanual_fallback_only_handles_a_real_self_screen_rejection(monkeypatch):
    from .bimanual import BimanualRobot
    from .full_robot import FullRobotGripper
    import sim_physics.grasp_path_replan as module
    monkeypatch.setattr(FullRobotGripper,'plan_approach',lambda self:None)
    r=BimanualRobot.__new__(BimanualRobot);r.self_screen=True
    def fail():raise RuntimeError('Unknown native query failure')
    r.check_grasp_path=fail
    monkeypatch.setattr(module,'replan',lambda *a:pytest.fail('Invariant failure was masked'))
    with pytest.raises(RuntimeError,match='Unknown'):r.plan_approach()


def test_bimanual_installs_complete_replan_then_rechecks_it(monkeypatch):
    from .bimanual import BimanualRobot
    from .full_robot import FullRobotGripper
    import sim_physics.grasp_path_replan as module
    monkeypatch.setattr(FullRobotGripper,'plan_approach',lambda self:None)
    r=BimanualRobot.__new__(BimanualRobot);r.self_screen=True;checks=[]
    def check():
        checks.append(1)
        if len(checks)==1:raise module.GraspPathSelfCollision('nominal path rejected',index=1)
    r.check_grasp_path=check
    ff=np.array([0.,1.15]);q=np.zeros((2,7));receipt={'minimum_interarm_m':.02}
    monkeypatch.setattr(module,'replan',lambda *a:(ff,q,receipt))
    r.plan_approach()
    assert len(checks)==2 and r.fractions is ff and r.path_q is q
    assert r.grasp_elbow_replan is receipt


@pytest.mark.parametrize('budget',[True,False,float('nan'),0.,9.])
def test_invalid_budget_rejected(monkeypatch,budget):
    with pytest.raises(ValueError):replan(fixture(monkeypatch),wall_limit_s=budget)
