"""Planner must reject a full tool-corridor conflict before running arm IK."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.bimanual import BimanualRobot


def test_complete_rigid_corridor_precedes_endpoint_ik(monkeypatch):
    import sim_physics.rigid_tool_screen as module
    robot=BimanualRobot.__new__(BimanualRobot)
    robot.stage=None;robot.root='/World/R';robot.rig=S(root='/World/P')
    robot.held_plant_screen=S(workspace=[np.zeros(3),np.ones(3)],static=[],snapshot=lambda frames:None)
    robot.goal=np.eye(4);robot.goal[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    robot.radius=.003;robot.right=np.zeros(7);robot.base=np.eye(4)
    robot.stroke_offsets=np.linspace(-.025,.005,61)
    robot.seam=lambda frames:(np.zeros(3),np.array([0.,0.,1.]))
    def wrist(point,*unused):
        result=np.eye(4);result[:3,3]=point;return result
    robot.knife=S(size=np.array([.002,.05,.006]),wrist_for_edge=wrist)
    def forbidden_ik(*args,**kwargs):raise AssertionError('Blocked tool corridor reached arm IK')
    robot.kin=S(solve_pose=forbidden_ik);checked=[]
    class Screen:
        def __init__(self,*args):pass
        def check(self,frames):
            assert frames.shape==(61,4,4)
            assert np.linalg.norm(frames[-1,:3,3]-frames[0,:3,3])==pytest.approx(.03)
            checked.append(frames)
            return dict(passed=False,reason='end_of_stroke_tool_collision',sample=60,native_validated=False)
    monkeypatch.setattr(module,'RigidToolScreen',Screen)
    with pytest.raises(RuntimeError,match='IK_attempted=0'):
        robot._plan_cut([],np.zeros(7))
    assert len(checked)==756
    attempts=robot.plan_diagnostics['endpoint_attempts']
    assert all(not a['ik_attempted'] and a['rejection']=='rigid_tool_corridor' for a in attempts)
    assert robot.plan is None and not robot.plan_diagnostics['whole_scene_path_certified']
