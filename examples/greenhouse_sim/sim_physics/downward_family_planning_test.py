"""Synthetic planner integration; not native clearance or cutting evidence."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.bimanual import BimanualRobot


@pytest.mark.parametrize('family_clear',[True,False])
def test_colliding_converged_endpoint_does_not_end_the_posture_search(monkeypatch,family_clear):
    import sim_physics.redundant_ik as ik
    import sim_physics.rigid_tool_screen as screen
    import sim_physics.downward_cut as downward
    r=BimanualRobot.__new__(BimanualRobot)
    r.stage=None;r.root='/World/Robot';r.rig=S(root='/World/Plant')
    r.cut_style='downward';r.cut_proposal=None;r.native_static_clearance=False
    r.right=np.zeros(7);r.base=np.eye(4);r.goal=np.eye(4);r.radius=.003
    r.stroke_offsets=np.array([-.008,.003]);r.seam=lambda f:(np.array([0,0,1.]),np.array([1.,0,0]))
    r.held_plant_screen=S(workspace=[np.full(3,-10.),np.full(3,10.)],static=[],
        snapshot=lambda f:None,last_failure={'held_leaf':True})
    r.knife=S(size=np.array([.002,.05,.006]),wrist_for_edge=lambda *args:np.eye(4))
    r.kin=S(inter_arm_clearance=lambda *args:S(clearance_m=.1))
    solution=lambda q:S(joint_degrees=np.full(7,q),position_error_m=0.,orientation_error_rad=0.,
        evaluations=1,succeeded=True)
    r.solve_right_pose=lambda *args:solution(0)
    checks=[];paths=[]
    r.body_world=lambda *args:{}
    r.check_self=lambda left,q:checks.append(('self',q[0])) or {'passed':True}
    r.check_held_plant=lambda left,q:checks.append(('leaf',q[0])) or (q[0]==1 and family_clear)
    def accept(left,centre,axis,candidate,tilt,failures):
        paths.append(candidate[3].copy());r.plan={'mock_dense_path_passed':True};return True
    r._try_cut_candidate=accept
    def family(kin,side,desired,seed,base,**options):
        assert options==dict(steps_per_direction=32,joint_limit_margin_degrees=3.)
        yield solution(1)
    monkeypatch.setattr(ik,'pose_family',family)
    monkeypatch.setattr(downward,'arm_extension',lambda f:.9)
    monkeypatch.setattr(screen,'RigidToolScreen',lambda *args:S(check=lambda f:{'passed':True}))
    if family_clear:
        r._plan_cut(np.empty((0,4,4)),np.zeros(7))
        assert checks[:4]==[('self',0),('leaf',0),('self',1),('leaf',1)]
        assert len(paths)==1 and paths[0][0]==1
        assert r.plan['stroke_basis']=='world_vertical'
        attempts=r.plan_diagnostics['endpoint_attempts']
        assert attempts[0]['rejection']=='endpoint_held_plant'
        assert attempts[1]['endpoint_family_index']==1 and 'rejection' not in attempts[1]
    else:
        with pytest.raises(RuntimeError,match='No bimanual arm-clearance path'):
            r._plan_cut(np.empty((0,4,4)),np.zeros(7))
        assert not paths and r.plan is None
