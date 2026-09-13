from types import SimpleNamespace as S
import numpy as np
import pytest
from .startup_station_search import search


class Robot:
    def __init__(self):
        self.base=np.eye(4);self.base[0,3]=.5
        self.initial_q=np.zeros(7);self.right=np.ones(7);self.path_q=[np.zeros(7),np.ones(7)]
        self.self_screen=S(shapes=['all_original_shapes'])
        self.rig=S(chain_world=np.array([[0.,0.,0.]]))
        self.kin=S(forward=lambda *a:np.eye(4),solve_pose=self.solve,
                   arm_limits_degrees=lambda *a:(np.full(7,-180.),np.full(7,180.)),
                   inter_arm_clearance=lambda *a:S(clearance_m=.02))
    def solve(self,arm,pose,seed,base,**kwargs):
        assert kwargs['joint_limit_margin_degrees']==3.
        assert kwargs['maximum_evaluations']==200
        return S(succeeded=True,joint_degrees=np.asarray(seed)+.1)
    def body_world(self,left,right):return dict(base=self.base.copy(),left=left.copy(),right=right.copy())
    def check_self(self,*a):return dict(passed=True)


def test_two_arm_station_proposal_keeps_original_poses_and_full_native_inventory():
    r=Robot();original=r.base.copy();calls=[]
    def check(world,shapes):
        assert shapes==['all_original_shapes'];calls.append(world)
        return dict(passed=len(calls)==2)
    out=search(r,S(check=check),lambda:None)
    np.testing.assert_array_equal(r.base,original)
    np.testing.assert_array_equal(r.initial_q,np.zeros(7))
    np.testing.assert_array_equal(r.right,np.ones(7))
    assert out['proposed_station']['station_pose']==pytest.approx([.45,0.,0.])
    assert out['proposed_station']['right_ready_degrees']==[0.,-5.,0.,-120.,0.,70.,0.]
    assert len(out['proposed_station']['left_path_degrees'])==2
    assert not out['motion_authorized'] and not out['base_motion_certified']
    assert not out['whole_path_certified'] and out['physics_steps']==0
    assert out['relaunch_required'] and out['proposed_floor_height_requires_fresh_launch']


def test_never_accept_unreachable_or_colliding_station():
    r=Robot();r.kin.solve_pose=lambda *a,**k:S(succeeded=False)
    out=search(r,S(),lambda:None)
    assert out['proposed_station'] is None and len(out['candidates'])==31
    r=Robot()
    out=search(r,S(check=lambda *a:dict(passed=False)),lambda:None)
    assert out['proposed_station'] is None


def test_prior_family_proposes_only_fresh_current_anatomy_poses_and_native_checks():
    r=Robot();r.cut_priority=dict(tilt=15.,normal_sign=-1,wing_m=0.)
    r.blade_axial_aim_offset_m=.0015;r.stroke_offsets=np.array([-.008,.005])
    r.rig.rest_frames=np.eye(4)[None]
    r.seam=lambda frames:(np.array([.1,.2,.3]),np.array([1.,0.,0.]))
    requested=[]
    def wrist(point,*args):
        requested.append(point.copy());pose=np.eye(4);pose[:3,3]=point
        return pose
    r.knife=S(wrist_for_edge=wrist);calls=[]
    def check(world,shapes):
        assert shapes==['all_original_shapes'];calls.append(world)
        return dict(passed=True)
    out=search(r,S(check=check),lambda:None)
    assert out['cut_frame_priority_used'] and len(calls)==1
    assert out['candidates'][0]['right_start_mode']=='cut_frame_withdrawn_20mm'
    np.testing.assert_allclose(requested[0],[.1015,.2,.308])
    assert out['proposed_station'] is not None and not out['motion_authorized']
    np.testing.assert_array_equal(r.right,np.ones(7))


def test_bad_left_path_revokes_native_clear_start():
    r=Robot();r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.005)
    out=search(r,S(check=lambda *a:dict(passed=True)),lambda:None)
    assert out['proposed_station'] is None
    assert all(row['rejection']=='left_path_self_or_interarm' for row in out['candidates'])


def test_final_control_failure_revokes_base_and_both_arms(monkeypatch):
    from .native_startup_screen_test import fixture
    from .native_startup_screen import _screen
    from . import startup_pose_search
    args,state=fixture();args[1].startup_right_pose_search=True
    def propose(*unused):
        state.missing='/World/Plant/collider'
        return dict(proposed_station={'station_pose':[.5,0,0],'left_ik_seed_degrees':[0]*7},
                    proposed_right_ready_degrees=[0]*7)
    monkeypatch.setattr(startup_pose_search,'search',propose)
    out=_screen(*args)
    assert out['right_pose_search']['proposed_station'] is None
    assert out['right_pose_search']['proposed_right_ready_degrees'] is None
    assert not out['right_pose_search']['final_native_controls_passed']


def test_station_search_cli_is_explicit_and_exclusive(monkeypatch,tmp_path):
    from pathlib import Path
    from .ground_truth_trial import main
    class BeforeWrite(Exception):pass
    def stop(*a,**k):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'new'),'--mode','bimanual','--milestone','cut_action',
          '--process-zone-trial','--greenhouse-trial','--screen-station']
    with pytest.raises(BeforeWrite):main(args)
    with pytest.raises(SystemExit):main(args+['--screen-tool-heading'])
    with pytest.raises(SystemExit):main(args+['--watch'])
