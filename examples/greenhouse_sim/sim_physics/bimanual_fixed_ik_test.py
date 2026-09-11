"""IK options are proposals, not robot state changes or clearance permission."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.bimanual import BimanualRobot


def test_default_right_solver_contract_unchanged():
    robot=object.__new__(BimanualRobot);calls=[];result=object()
    robot.base=np.eye(4)
    robot.kin=S(solve_pose=lambda *a,**k:calls.append((a,k)) or result)
    desired=np.eye(4);seed=np.zeros(7)
    assert robot.solve_right_pose(desired,seed) is result
    assert calls[0][0][0]=='right' and calls[0][0][1] is desired
    assert calls[0][1]==dict(maximum_evaluations=250)


def test_fixed_joint_solver_applies_to_each_requested_stroke_pose(monkeypatch):
    import sim_physics.redundant_ik as module
    calls=[];robot=object.__new__(BimanualRobot)
    robot.kin=object();robot.base=np.eye(4);robot.right_ik_fixed_joint=(0,-35.)
    def solve(*a,**k):
        calls.append((a,k));return S(succeeded=True,joint_degrees=np.array([-35.,0,0,0,0,0,0]))
    monkeypatch.setattr(module,'solve_fixed_joint',solve)
    seed=np.zeros(7)
    for distance in [0.,.0005,.001]:
        desired=np.eye(4);desired[0,3]=distance
        result=robot.solve_right_pose(desired,seed);seed=result.joint_degrees
    assert len(calls)==3
    for args,kw in calls:
        assert args[0] is robot.kin and args[1]=='right'
        assert kw==dict(joint_index=0,joint_degrees=-35.,maximum_evaluations=250)


@pytest.mark.parametrize('fixed',[(7,0),(-1,0),(.5,0),(0,float('nan')),('0','3'),(0,),True])
def test_invalid_fixed_joint_rejected_before_robot_authoring(fixed):
    with pytest.raises(ValueError,match='fixed-joint'):
        BimanualRobot(None,None,right_ik_fixed_joint=fixed)
