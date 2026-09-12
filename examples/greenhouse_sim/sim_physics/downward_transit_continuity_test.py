"""Synthetic redundant-arm path contracts; not native cutting evidence."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.downward_cut import cartesian_transit
from sim_physics.bimanual import BimanualRobot


def redundant_robot():
    r=BimanualRobot.__new__(BimanualRobot)
    r.right=np.zeros(7);r.base=np.eye(4);r.cut_style='downward';r.plan=None
    r.stroke_offsets=np.array([0.,-.001,-.002])
    def fk(side,q,base):
        frame=np.eye(4);frame[0,3]=q[0];return frame
    def wrist(point,*args):
        frame=np.eye(4);frame[:3,3]=point;return frame
    r.kin=S(forward=fk,inter_arm_clearance=lambda *args:S(clearance_m=.1))
    r.check_self=lambda *a:{'passed':True}
    r.check_held_plant=lambda *a,**kw:True
    r.body_world=lambda *a:{}
    r.held_plant_screen=S(last_failure={'blocked':True})
    r.knife=S(wrist_for_edge=wrist)
    def solve(desired,seed):
        q=np.array(seed,copy=True);q[0]=desired[0,3]
        return S(succeeded=True,joint_degrees=q)
    r.solve_right_pose=solve
    return r


def test_redundant_pose_only_terminal_requires_explicit_stroke_rebuild():
    r=redundant_robot();q=np.array([.01,0,0,0,0,0,30.])
    assert cartesian_transit(r,np.zeros(7),q) is None
    path,_,evidence=cartesian_transit(r,np.zeros(7),q,replan_stroke_from_endpoint=True)
    assert path[-1,6]==0 and q[6]==30
    assert evidence['requires_stroke_replanning'] is True
    assert evidence['proposed_terminal_joint_difference_degrees']==30
    assert evidence['terminal_position_error_m']==0


@pytest.mark.parametrize('bad',[1,None,'yes'])
def test_replan_flag_is_not_truthy_coercion(bad):
    with pytest.raises(ValueError):cartesian_transit(redundant_robot(),np.zeros(7),np.zeros(7),replan_stroke_from_endpoint=bad)


def test_downward_stroke_starts_from_exact_transit_terminal(monkeypatch):
    import sim_physics.downward_cut as module
    monkeypatch.setattr(module,'arm_extension',lambda *a:.9)
    r=redundant_robot();q=np.array([.01,0,0,0,0,0,30.]);failures=[]
    candidate=(0.,0.,np.array([1.,0,0]),q,1,0.,np.array([0.,0,1]))
    assert r._try_cut_candidate(np.zeros(7),np.array([.01,0,0]),np.array([0.,0,1]),candidate,0.,failures)
    np.testing.assert_array_equal(r.plan['approach'][-1],r.plan['stroke'][0])
    assert np.all(r.plan['stroke'][:,6]==0)
    assert failures==[] and r.plan['stroke_samples']==3


@pytest.mark.parametrize('failure',['transit','stroke','extension','terminal'])
def test_new_branch_still_requires_all_path_guards(monkeypatch,failure):
    import sim_physics.downward_cut as module
    monkeypatch.setattr(module,'arm_extension',lambda *a:.7 if failure=='extension' else .9)
    r=redundant_robot();q=np.array([.01,0,0,0,0,0,30.]);failures=[]
    candidate=(0.,0.,np.array([1.,0,0]),q,1,0.,np.array([0.,0,1]))
    if failure=='transit':r.check_held_plant=lambda *a,**kw:False
    if failure=='stroke':r.check_held_plant=lambda *a,**kw:not kw.get('stroke',False)
    if failure=='terminal':r.right_transit=lambda *a:(np.zeros((2,7)),.1,{'synthetic_wrong_terminal':True})
    assert not r._try_cut_candidate(np.zeros(7),np.array([.01,0,0]),np.array([0.,0,1]),candidate,0.,failures)
    assert r.plan is None and len(failures)==1


@pytest.mark.parametrize('failure',['IK','interarm','self','plant','branch'])
def test_rejection_diagnostics_distinguish_failures_without_accepting_them(failure):
    import json
    r=redundant_robot();q=np.array([.01,0,0,0,0,0,30.]);details={'stale':True}
    expected=dict(IK='cartesian_IK',interarm='interarm_clearance',self='robot_self_clearance',
        plant='plant_or_scene_clearance',branch='terminal_joint_branch')
    if failure=='IK':r.solve_right_pose=lambda *a:S(succeeded=False)
    if failure=='interarm':r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.009)
    if failure=='self':r.check_self=lambda *a:{'passed':False,'nearest_pair':['a','b']}
    if failure=='plant':r.check_held_plant=lambda *a,**kw:False
    assert cartesian_transit(r,np.zeros(7),q,diagnostics=details) is None
    assert details['reason']==expected[failure] and not details['motion_authorized'] and 'stale' not in details
    json.dumps(details,allow_nan=False)


def test_successful_transit_clears_old_failure_details():
    r=redundant_robot();details={'reason':'old'}
    assert cartesian_transit(r,np.zeros(7),np.array([.01,0,0,0,0,0,0]),diagnostics=details) is not None
    assert details=={}


def test_diagnostics_type_is_not_silently_coerced():
    with pytest.raises(ValueError,match='dictionary'):
        cartesian_transit(redundant_robot(),np.zeros(7),np.zeros(7),diagnostics=[])
