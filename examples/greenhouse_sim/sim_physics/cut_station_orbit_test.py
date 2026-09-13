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


def test_missing_priority_is_not_an_implicit_pose_choice():
    r,_=fixture();r.cut_priority=None
    with pytest.raises(ValueError,match='prior frame'):search(r,S(),lambda:None)


def test_public_flag_requires_zero_motion_search(tmp_path):
    from .ground_truth_trial import main
    with pytest.raises(ValueError,match='zero-motion station'):
        main(['--output',str(tmp_path/'unused'),'--mode','right_only','--milestone','cut_action',
              '--process-zone-trial','--cut-station-orbit'])
    assert not (tmp_path/'unused').exists()
