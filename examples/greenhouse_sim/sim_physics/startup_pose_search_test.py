from types import SimpleNamespace as S
import numpy as np
import pytest


def test_proposals_do_not_mutate_robot_or_authorize_motion(monkeypatch):
    from .startup_pose_search import search
    from . import redundant_ik
    q=np.arange(7,dtype=float);calls=[]
    monkeypatch.setattr(redundant_ik,'pose_family',lambda *a,**k:iter([S(joint_degrees=q+1),S(joint_degrees=q+2)]))
    r=S(right=q.copy(),base=np.eye(4),initial_q=q.copy(),path_q=[q,q],
        kin=S(forward=lambda *a:np.eye(4),inter_arm_clearance=lambda *a:S(clearance_m=.02)),
        check_self=lambda *a:dict(passed=True),body_world=lambda left,right:dict(q=right.copy()),
        self_screen=S(shapes=['all_shapes']))
    def check(world,shapes):
        assert shapes==['all_shapes'];calls.append(world['q']);return dict(passed=len(calls)==2)
    out=search(r,S(check=check),lambda:None)
    np.testing.assert_array_equal(r.right,q)
    assert out['proposed_right_ready_degrees']==list(q+2)
    assert out['physics_steps']==0 and not out['motion_authorized']
    assert out['relaunch_required'] and not out['whole_path_certified']


def test_search_success_cannot_clear_the_original_bad_native_spawn(monkeypatch):
    from .native_startup_screen_test import fixture
    from .native_startup_screen import _screen
    from . import startup_pose_search
    args,state=fixture();state.hit=True;args[1].startup_right_pose_search=True
    monkeypatch.setattr(startup_pose_search,'search',lambda *a:dict(proposed_right_ready_degrees=[0.]*7))
    out=_screen(*args)
    assert out['passed'] is False and out['physics_steps']==0
    assert out['right_pose_search']['proposed_right_ready_degrees']==[0.]*7
    assert out['native']['final_validation_passed']
    assert out['right_pose_search']['final_native_controls_passed']


def test_final_native_control_failure_revokes_a_provisional_proposal(monkeypatch):
    from .native_startup_screen_test import fixture
    from .native_startup_screen import _screen
    from . import startup_pose_search
    args,state=fixture();args[1].startup_right_pose_search=True
    def propose(*unused):
        state.missing='/World/Plant/collider'
        return dict(proposed_right_ready_degrees=[0.]*7)
    monkeypatch.setattr(startup_pose_search,'search',propose)
    out=_screen(*args)
    assert out['passed'] is False
    assert out['right_pose_search']['proposed_right_ready_degrees'] is None
    assert out['right_pose_search']['final_native_controls_passed'] is False


def test_disconnected_elbow_branch_is_screened_after_local_family_ends(monkeypatch):
    from .startup_pose_search import search
    from . import redundant_ik
    monkeypatch.setattr(redundant_ik,'pose_family',lambda *a,**k:iter(()))
    seeds=[];q=np.arange(7,dtype=float)
    def solve(*args,**kwargs):
        seeds.append(args[2].copy());assert kwargs['joint_limit_margin_degrees']==3.
        return S(succeeded=True,joint_degrees=q)
    robot=S(right=q.copy(),base=np.eye(4),initial_q=q.copy(),path_q=[q],
        kin=S(forward=lambda *a:np.eye(4),inter_arm_clearance=lambda *a:S(clearance_m=.02),
            arm_limits_degrees=lambda *a:(np.full(7,-180.),np.full(7,180.)),solve_pose=solve),
        check_self=lambda *a:dict(passed=True),body_world=lambda *a:{},self_screen=S(shapes=['all']))
    out=search(robot,S(check=lambda world,shapes:dict(passed=True)),lambda:None)
    assert out['ik_attempts']['global_ik_attempts']==1 and len(seeds)==1
    assert out['candidates'][0]['ik_origin']=='global_multistart'
    assert not out['motion_authorized'] and out['relaunch_required']


def test_cli_requires_complete_native_diagnostic_before_output_creation(tmp_path):
    from .benchmark import main
    out=tmp_path/'none'
    with pytest.raises(ValueError,match='Startup pose search'):
        main(['--output',str(out),'--native-startup-pose-search'])
    assert not out.exists()


def test_approach_proposals_change_only_desired_initial_translation():
    from .startup_pose_search import search
    q=np.arange(7,dtype=float);requested=[];checks=[];guard_calls=[]
    def solve(arm,pose,seed,base,**kwargs):
        assert kwargs['joint_limit_margin_degrees']==3.
        np.testing.assert_array_equal(pose[:3,:3],np.eye(3))
        requested.append(pose.copy());return S(succeeded=True,joint_degrees=q+len(requested))
    robot=S(startup_approach_search=True,right=q.copy(),base=np.eye(4),initial_q=q.copy(),path_q=[q,q],
        kin=S(forward=lambda *a:np.eye(4),solve_pose=solve,inter_arm_clearance=lambda *a:S(clearance_m=.02)),
        check_self=lambda *a:dict(passed=True),body_world=lambda *a:{},self_screen=S(shapes=['all']))
    def check(world,shapes):
        assert shapes==['all'];checks.append(world);return dict(passed=len(checks)==2)
    result=search(robot,S(check=check),lambda:guard_calls.append(1))
    assert result['proposed_wrist_offset_world_m']==[0.,0.,.02]
    np.testing.assert_array_equal(robot.right,q)
    assert not result['motion_authorized'] and not result['whole_path_certified']
    assert result['physics_steps']==0 and result['relaunch_required']
    assert len(guard_calls)>=6


def test_approach_search_never_accepts_self_collision():
    from .startup_pose_search import approach_start_search
    q=np.zeros(7)
    robot=S(right=q,base=np.eye(4),initial_q=q,path_q=[q],
        kin=S(forward=lambda *a:np.eye(4),solve_pose=lambda *a,**k:S(succeeded=True,joint_degrees=q),
            inter_arm_clearance=lambda *a:S(clearance_m=.005)),
        check_self=lambda *a:dict(passed=True))
    out=approach_start_search(robot,S(),lambda:None)
    assert out['proposed_right_ready_degrees'] is None and len(out['candidates'])==24
    assert out['maximum_translation_m']==pytest.approx(.2)


def test_farther_waiting_proposal_still_gets_every_native_shape_check():
    from .startup_pose_search import approach_start_search
    q=np.zeros(7);poses=[];checks=[]
    def solve(arm,pose,seed,base,**kwargs):
        poses.append(pose.copy());return S(succeeded=True,joint_degrees=q+len(poses))
    robot=S(right=q.copy(),base=np.eye(4),initial_q=q,path_q=[q],
        kin=S(forward=lambda *a:np.eye(4),solve_pose=solve,
            inter_arm_clearance=lambda *a:S(clearance_m=.02)),
        check_self=lambda *a:dict(passed=True),body_world=lambda *a:{},self_screen=S(shapes=['all']))
    def check(world,shapes):
        assert shapes==['all'];checks.append(1);return dict(passed=len(checks)==17)
    out=approach_start_search(robot,S(check=check),lambda:None)
    assert out['proposed_wrist_offset_world_m']==[0.,0.,.08]
    assert len(checks)==17 and out['physics_steps']==0
    assert not out['motion_authorized'] and out['original_spawn_unchanged']
    np.testing.assert_array_equal(robot.right,q)


def test_heading_search_rotates_whole_proposed_tool_never_mount_or_live_pose():
    from .startup_pose_search import search
    from .blade_contacts import CROSSBAR_EDGE
    q=np.zeros(7);poses=[];shapes=[];local=np.eye(4)
    def solve(arm,pose,seed,base,**kwargs):
        poses.append(pose.copy());return S(succeeded=True,joint_degrees=q+1)
    r=S(startup_heading_search=True,right=q.copy(),base=np.eye(4),initial_q=q,path_q=[q],
        knife=S(edge_mode=CROSSBAR_EDGE,local=local,frame=lambda f:f.copy(),arc_up=lambda f:np.array([0,0,1.])),
        kin=S(forward=lambda *a:np.eye(4),solve_pose=solve,
            arm_limits_degrees=lambda *a:(np.full(7,-180.),np.full(7,180.)),
            inter_arm_clearance=lambda *a:S(clearance_m=.02)),
        check_self=lambda *a:dict(passed=True),body_world=lambda *a:{},self_screen=S(shapes=['all']))
    def check(world,ss):shapes.append(ss);return dict(passed=True)
    result=search(r,S(check=check),lambda:None)
    assert shapes==[['all']]
    np.testing.assert_allclose(poses[0][:3,:3],np.diag([-1,-1,1]),atol=1e-15)
    np.testing.assert_allclose(poses[0][:3,3],[0,0,.03])
    np.testing.assert_array_equal(r.right,q);np.testing.assert_array_equal(r.knife.local,local)
    assert result['proposed_right_ready_degrees']==[1.]*7
    assert not result['motion_authorized'] and not result['mounting_changed']
    assert result['relaunch_required'] and result['physics_steps']==0


def test_heading_profile_is_explicit_and_mutually_exclusive(tmp_path,monkeypatch):
    from pathlib import Path
    from .ground_truth_trial import main
    class Validated(Exception):pass
    def stop(*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'new'),'--mode','bimanual','--milestone','cut_action',
        '--process-zone-trial','--greenhouse-trial','--screen-tool-heading']
    with pytest.raises(Validated):main(args)
    with pytest.raises(SystemExit):main(args+['--screen-ready-pose'])
