from types import SimpleNamespace as S
import numpy as np
import pytest
from .bimanual import BimanualRobot


@pytest.mark.parametrize('primary_success,has_hint',[(True,True),(False,True),(False,False)])
def test_entry_hint_only_after_failed_endpoint_never_changes_waiting(primary_success,has_hint):
    robot=object.__new__(BimanualRobot);robot.right=np.zeros(7)
    hint=np.arange(7,dtype=float);robot.right_entry_seed_degrees=hint.copy() if has_hint else None
    calls=[]
    def solve(desired,seed):
        calls.append(np.array(seed));return S(succeeded=primary_success or len(calls)==2)
    robot.solve_right_pose=solve
    result=robot.solve_right_entry_pose(np.eye(4),robot.right)
    fallback=not primary_success and has_hint
    assert len(calls)==(2 if fallback else 1) and robot.entry_seed_fallback_used is fallback
    assert result.succeeded is (primary_success or fallback)
    np.testing.assert_array_equal(robot.right,np.zeros(7))
    if has_hint:np.testing.assert_array_equal(robot.right_entry_seed_degrees,hint)


def test_failed_hint_stays_failed():
    robot=object.__new__(BimanualRobot);robot.right_entry_seed_degrees=np.ones(7)
    robot.solve_right_pose=lambda *a:S(succeeded=False)
    assert not robot.solve_right_entry_pose(np.eye(4),np.zeros(7)).succeeded


def test_entry_hint_does_not_change_stroke_solver_seed_or_failure():
    robot=object.__new__(BimanualRobot);robot.right_entry_seed_degrees=np.ones(7)*33.
    robot.cut_style='downward';robot.base=np.eye(4);calls=[]
    def solve(side,desired,seed,base,**kw):
        calls.append(np.array(seed));assert kw['joint_limit_margin_degrees']==3.
        return S(succeeded=False)
    robot.kin=S(solve_pose=solve)
    assert not robot.solve_right_pose(np.eye(4),np.zeros(7)).succeeded
    assert len(calls)==4 and all(np.all(q[:6]==0.) for q in calls)


def test_launcher_forwards_hint_separately_from_waiting_pose(monkeypatch,tmp_path):
    from . import benchmark,ground_truth_trial
    got=[];monkeypatch.setattr(benchmark,'main',lambda args:got.append(benchmark.parser().parse_args(args)))
    ground_truth_trial.main(['--output',str(tmp_path/'none'),'--mode','bimanual',
        '--milestone','cut_action','--process-zone-trial','--right-entry-seed-degrees',*['1']*7])
    assert got[0].right_entry_seed_degrees==[1.]*7
    assert got[0].right_ready_degrees!=[1.]*7
