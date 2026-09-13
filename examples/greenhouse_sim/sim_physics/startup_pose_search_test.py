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
