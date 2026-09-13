from types import SimpleNamespace as S
import numpy as np
import pytest
from .cut_station_orbit import stations,search
from .startup_station_search_test import Robot


def fixture():
    r=Robot();r.cut_priority=dict(normal_sign=-1,tilt=15.,wing_m=0.)
    r.blade_axial_aim_offset_m=.0015;r.stroke_offsets=np.array([-.008,.005])
    r.rig.rest_frames=np.eye(4)[None]
    r.seam=lambda frames:(np.zeros(3),np.array([1.,0.,0.]))
    poses=[]
    def wrist(point,*args):
        pose=np.eye(4);pose[:3,3]=point;return pose
    r.knife=S(wrist_for_edge=wrist)
    body_world=r.body_world
    def complete_world(*args):
        world=body_world(*args)
        for i,p in ((2,[0.,0.,0.]),(3,[.45,np.sqrt(.5**2-.45**2),0.]),(4,[.9,0.,0.])):
            frame=np.eye(4);frame[:3,3]=p;world['link_right_arm_'+str(i)]=frame
        return world
    r.body_world=complete_world
    solve=r.kin.solve_pose
    def record(arm,pose,*args,**kwargs):
        poses.append((arm,pose.copy()));return solve(arm,pose,*args,**kwargs)
    r.kin.solve_pose=record
    return r,poses


def test_orbits_are_finite_bounded_rigid_face_target_and_do_not_mutate():
    base=np.eye(4);base[:3,3]=[.5,0,.123];original=base.copy()
    values=list(stations(base,np.zeros(3)))
    assert len(values)==294
    for m,meta in values:
        assert m[2,3]==.123 and .299999999999<=np.linalg.norm(m[:2,3])<=.750000000001
        np.testing.assert_allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-12)
        assert np.dot(m[:2,0],-m[:2,3])>0
    np.testing.assert_array_equal(base,original)


def test_both_native_endpoints_required_and_original_spawn_untouched():
    r,poses=fixture();base=r.base.copy();calls=[]
    def native(world,shapes):
        assert shapes==['all_original_shapes'];calls.append(world)
        return dict(passed=True)
    out=search(r,S(check=native),lambda:None)
    assert len(calls)==2 and out['proposed_station']
    assert out['proposed_station']['cut_frame_family']==dict(normal_sign=-1,tilt=15.,wing_m=0.)
    right=[pose for arm,pose in poses if arm=='right']
    np.testing.assert_allclose(right[0][:3,3]-right[1][:3,3],[0,0,.03])
    np.testing.assert_array_equal(r.base,base)
    np.testing.assert_array_equal(r.right,np.ones(7))
    assert out['proposed_floor_height_requires_fresh_launch']
    for key in ('motion_authorized','whole_path_certified','base_motion_certified','infeasibility_proof'):
        assert out[key] is False


def test_clear_waiting_pose_never_promotes_blocked_cut_entry():
    r,_=fixture();calls=[]
    def native(*args):
        calls.append(1);return dict(passed=len(calls)%2==1)
    out=search(r,S(check=native),lambda:None)
    assert out['proposed_station'] is None
    assert all(row['native_startup_clear'] and not row['native_cut_entry_clear'] for row in out['candidates'])


def test_endpoint_clear_does_not_bypass_left_interarm():
    r,_=fixture();r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.005)
    out=search(r,S(check=lambda *a:dict(passed=True)),lambda:None)
    assert out['proposed_station'] is None
    assert all(not row['left_self_path_clear'] for row in out['candidates'])


def test_guard_failure_propagates_to_owner_for_final_revocation():
    r,_=fixture()
    def fail():raise RuntimeError('stale epoch')
    with pytest.raises(RuntimeError,match='stale epoch'):search(r,S(),fail)


def test_folded_cut_entry_cannot_be_proposed_as_a_straight_arm_cut():
    r,_=fixture();world=r.body_world
    def folded(*args):
        w=world(*args);w['link_right_arm_4'][:3,3]=[0.,0.,0.];return w
    r.body_world=folded;calls=[]
    out=search(r,S(check=lambda *a:(calls.append(1) or dict(passed=True))),lambda:None)
    assert out['proposed_station'] is None
    assert all(a['rejection']=='folded_or_fully_extended_cut_entry'
               for row in out['candidates'] for a in row['right_attempts'])


def test_identical_right_seeds_are_not_solved_twice():
    from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES
    r,_=fixture();r.right=np.array([SDK_READY_POSE_DEGREES[f'right_arm_{i}'] for i in range(7)])
    out=search(r,S(check=lambda *a:dict(passed=False)),lambda:None)
    assert all(len(row['right_attempts'])==2 for row in out['candidates'])


def test_original_between_grid_station_is_checked_before_coarse_orbits():
    r,_=fixture();r.base[0,3]=.5123;calls=[]
    def native(world,shapes):
        calls.append(world['base'].copy());return dict(passed=True)
    out=search(r,S(check=native),lambda:None)
    assert out['candidates'][0]['original_station']
    assert out['proposed_station']['station_pose']==pytest.approx([.5123,0,0])
    assert len(calls)==2 and all(m[0,3]==.5123 for m in calls)
    assert not out['motion_authorized'] and out['maximum_candidates']==295


def test_no_prior_report_uses_fresh_anatomy_not_a_prior_path():
    r,_=fixture();r.cut_priority=None
    out=search(r,S(check=lambda *a:dict(passed=True)),lambda:None)
    assert out['proposed_station'] and not out['cut_frame_priority_used']
    assert not out['motion_authorized'] and not out['whole_path_certified']


def test_public_flag_requires_zero_motion_search(tmp_path):
    from .ground_truth_trial import main
    with pytest.raises(ValueError,match='zero-motion station'):
        main(['--output',str(tmp_path/'unused'),'--mode','right_only','--milestone','cut_action',
              '--process-zone-trial','--cut-station-orbit'])
    assert not (tmp_path/'unused').exists()


def test_waiting_offsets_keep_orientation_entry_and_default_unchanged():
    from .cut_station_orbit import waiting_poses
    entry=np.eye(4);entry[:3,3]=[1,2,3];original=entry.copy()
    d=np.array([0.,0.,-1.]);n=np.array([1.,0.,0.])
    default=list(waiting_poses(entry,d,n));expanded=list(waiting_poses(entry,d,n,True))
    assert len(default)==1 and len(expanded)==12
    np.testing.assert_array_equal(default[0][0],expanded[0][0])
    np.testing.assert_array_equal(entry,original)
    for pose,offset in expanded:
        np.testing.assert_array_equal(pose[:3,:3],entry[:3,:3])
        np.testing.assert_allclose(pose[:3,3]-entry[:3,3],offset)
        assert .01<=offset[2]<=.12 and np.linalg.norm(offset[:2])<=.08
    with pytest.raises(ValueError):list(waiting_poses(entry,d,n,1))


def test_alternate_waiting_pose_needs_its_own_native_entry_and_left_path():
    r,poses=fixture();r.station_waiting_search=True;calls=[]
    def solve(side,pose,seed,base,**kwargs):
        q=np.asarray(seed).copy()
        if side=='right':q[:3]=pose[:3,3]
        return S(succeeded=True,joint_degrees=q)
    r.kin.solve_pose=solve
    def native(world,shapes):
        assert shapes==['all_original_shapes'];z=float(world['right'][2]);calls.append(z)
        return dict(passed=z<.02)  # Only lower waiting and the entry are clear.
    out=search(r,S(check=native),lambda:None)
    assert out['expanded_waiting_pose_search'] and out['waiting_pose_candidates_per_frame']==12
    assert out['proposed_station']['waiting_offset_world_m']==[0.,0.,.01]
    assert out['candidates'][0]['native_cut_entry_clear'] and out['candidates'][0]['left_self_path_clear']
    assert calls==pytest.approx([.038,.038,.018,.008])
    assert not out['motion_authorized'] and out['physics_steps']==0
    np.testing.assert_array_equal(r.right,np.ones(7))


def test_expanded_waits_cannot_promote_a_blocked_cut_entry():
    r,_=fixture();r.station_waiting_search=True
    def solve(side,pose,seed,base,**kwargs):
        q=np.asarray(seed).copy()
        if side=='right':q[:3]=pose[:3,3]
        return S(succeeded=True,joint_degrees=q)
    r.kin.solve_pose=solve
    out=search(r,S(check=lambda world,shapes:dict(passed=world['right'][2]>.009)),lambda:None)
    assert out['proposed_station'] is None
    assert all(not row['native_cut_entry_clear'] for row in out['candidates'])


def test_expanded_public_option_cannot_launch_motion(tmp_path):
    from .ground_truth_trial import main
    with pytest.raises(ValueError,match='zero-motion'):
        main(['--output',str(tmp_path/'unused'),'--mode','bimanual','--milestone','cut_action',
              '--process-zone-trial','--station-waiting-search'])
    assert not (tmp_path/'unused').exists()


def test_query_budget_stop_does_not_consume_final_native_control_reserve():
    r,_=fixture();r.station_waiting_search=True
    def unexpected(*args):pytest.fail('No native query after soft budget stop')
    out=search(r,S(check=unexpected,can_check_with_final_controls=lambda *a:False),lambda:None)
    assert out['proposed_station'] is None and out['budget_exhausted']
    assert out['query_budget_reserved_for_final_controls']
    assert out['candidates'][0]['right_attempts'][0]['rejection']=='reserved_final_native_controls'
    assert not out['motion_authorized'] and not out['infeasibility_proof']
