import json
from types import SimpleNamespace as S
import numpy as np
import pytest
from .neutral_station_search import read_candidates,search


def document():
    return dict(schema='neutral_station_candidates_v1',plant='p',target='s',
        candidates=[dict(station_pose=[.5,0.,0.])])


@pytest.mark.parametrize('fault',['schema','plant','target','empty','many','nan','bool','yaw','extra','seed'])
def test_candidate_file_rejects_bad_or_mismatched_proposals(tmp_path,fault):
    d=document()
    if fault in ('schema','plant','target'):d[fault]='wrong'
    elif fault=='empty':d['candidates']=[]
    elif fault=='many':d['candidates']*=65
    elif fault in ('nan','bool','yaw'):
        d['candidates'][0]['station_pose'][2]={'nan':float('nan'),'bool':True,'yaw':181}[fault]
    elif fault=='extra':d['candidates'][0]['motion_authorized']=True
    elif fault=='seed':d['candidates'][0]['right_ik_seed_degrees']=[0]*6
    path=tmp_path/'candidates.json';path.write_text(json.dumps(d))
    with pytest.raises(ValueError):read_candidates(path,plant='p',target='s')


def test_candidate_file_has_content_receipt(tmp_path):
    path=tmp_path/'candidates.json';path.write_text(json.dumps(document()))
    rows,receipt=read_candidates(path,plant='p',target='s')
    assert rows==document()['candidates'] and len(receipt['sha256'])==64


class Robot:
    def __init__(self):
        self.base=np.eye(4);self.base[0,3]=.5
        self.initial_q=np.zeros(7);self.right=np.ones(7);self.path_q=[np.zeros(7)]
        self.goal=np.eye(4);self.cut_priority=dict(normal_sign=-1,tilt=15.,wing_m=0.)
        self.rig=S(rest_frames=np.eye(4)[None],chain_world=np.array([[0.,0.,0.]]))
        self.self_screen=S(shapes=['complete_inventory'])
        self.blade_axial_aim_offset_m=.0015;self.stroke_offsets=np.array([-.008,.005])
        self.knife=S(wrist_for_edge=lambda *a:np.eye(4))
        self.calls=[]
        self.kin=S(forward=lambda *a:np.eye(4),solve_pose=self.solve,
            arm_limits_degrees=lambda *a:(np.full(7,-180.),np.full(7,180.)),
            inter_arm_clearance=lambda *a:S(clearance_m=.02))
        self.neutral_station_candidates=document()['candidates']
        self.neutral_station_candidate_source=dict(sha256='test')
    def seam(self,*a):return np.zeros(3),np.array([1.,0.,0.])
    def solve(self,side,goal,seed,base,**kw):
        assert kw==dict(maximum_evaluations=250,joint_limit_margin_degrees=3.)
        self.calls.append(side);return S(succeeded=True,joint_degrees=np.asarray(seed)+.1)
    def body_world(self,left,right):return dict(base=self.base.copy(),left=np.array(left),right=np.array(right))
    def check_self(self,*a):return dict(passed=True)


def setup(monkeypatch,failed_native=None,reserve=True):
    from . import downward_cut
    monkeypatch.setattr(downward_cut,'arm_extension',lambda world:.9)
    calls=[]
    def check(world,shapes,margin):
        assert shapes==['complete_inventory'];calls.append((world,margin))
        return dict(passed=len(calls)!=failed_native)
    return Robot(),S(check=check,can_check_with_final_controls=lambda *a,**k:reserve),calls


def test_neutral_then_pregrasp_then_entry_no_original_state_writes(monkeypatch):
    r,b,calls=setup(monkeypatch);base=r.base.copy();out=search(r,b,lambda:None)
    assert [c[1] for c in calls]==[.005,.005,.001]
    assert r.calls==['left','left','right']
    np.testing.assert_array_equal(r.base,base)
    np.testing.assert_array_equal(r.initial_q,np.zeros(7))
    assert out['proposed_station']['right_ready_degrees']==[0.,-5.,0.,-120.,0.,70.,0.]
    assert out['physics_steps']==0 and not out['motion_authorized']
    assert not out['whole_path_certified'] and not out['grasp_or_cut_verified']
    assert not out['settled_scene_verified'] and out['relaunch_required']


@pytest.mark.parametrize('failed,expected,solves',[(1,'neutral_scene',0),(2,'pregrasp_scene',1),(3,'entry_scene',3)])
def test_native_rejection_never_returns_a_proposal(monkeypatch,failed,expected,solves):
    r,b,calls=setup(monkeypatch,failed);out=search(r,b,lambda:None)
    assert out['proposed_station'] is None and out['candidates'][0]['rejection']==expected
    assert len(r.calls)==solves


def test_final_control_query_reserve_stops_search(monkeypatch):
    r,b,calls=setup(monkeypatch,reserve=False);out=search(r,b,lambda:None)
    assert out['budget_limited'] and out['proposed_station'] is None
    assert not calls and not r.calls


def test_epoch_failure_propagates_to_native_owner(monkeypatch):
    r,b,calls=setup(monkeypatch)
    def invalid():raise RuntimeError('epoch changed')
    with pytest.raises(RuntimeError,match='epoch changed'):search(r,b,invalid)
    assert not calls


@pytest.mark.parametrize('failure',['ik','self','interarm','extension','seed'])
def test_unreachable_or_unsafe_endpoints_are_not_proposals(monkeypatch,failure):
    from . import downward_cut
    r,b,calls=setup(monkeypatch)
    if failure=='ik':r.kin.solve_pose=lambda *a,**k:S(succeeded=False)
    elif failure=='self':r.check_self=lambda *a:dict(passed=False)
    elif failure=='interarm':r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.001)
    elif failure=='extension':monkeypatch.setattr(downward_cut,'arm_extension',lambda *a:1.)
    else:r.neutral_station_candidates[0]['left_ik_seed_degrees']=[190.]*7
    assert search(r,b,lambda:None)['proposed_station'] is None


def test_batch_launcher_is_zero_motion_only(monkeypatch,tmp_path):
    from pathlib import Path
    from . import benchmark,ground_truth_trial,cut_priority
    path=tmp_path/'candidates.json';d=document();d.update(plant='seed101_full',target='SubStem_41')
    path.write_text(json.dumps(d))
    monkeypatch.setattr(cut_priority,'from_arguments',lambda args:dict(tilt=15.,normal_sign=-1,wing_m=0.))
    class BeforeWrite(Exception):pass
    def stop(*a,**k):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'none'),'--mode','bimanual','--milestone','cut_action',
        '--process-zone-trial','--screen-station','--neutral-station-candidates',str(path),
        '--torso-degrees','0','0','0','0','0','0']
    with pytest.raises(BeforeWrite):ground_truth_trial.main(args)
    with pytest.raises(ValueError,match='Neutral station batch'):ground_truth_trial.main(args+['--neutral-ready-start'])
    with pytest.raises(ValueError,match='Neutral station batch'):ground_truth_trial.main(args+['--cut-station-orbit'])
    without_search=[a for a in args if a!='--screen-station']
    with pytest.raises(ValueError,match='Neutral station batch'):ground_truth_trial.main(without_search)


@pytest.mark.parametrize('value',[True,-1,17,1.5,None])
def test_pose_family_budget_must_be_explicit_bounded_integer(tmp_path,value):
    d=document();d['candidates'][0]['right_pose_family_steps']=value
    path=tmp_path/'candidate.json';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='pose-family'):read_candidates(path,plant='p',target='s')


def test_clear_redundant_entry_uses_fresh_native_check_without_moving_original(monkeypatch):
    from . import redundant_ik
    r,b,calls=setup(monkeypatch,failed_native=3)
    r.neutral_station_candidates[0]['right_pose_family_steps']=8
    observed=[]
    def family(kin,side,entry,seed,base,**kwargs):
        observed.append(kwargs)
        yield S(joint_degrees=np.asarray(seed)+1.)
    monkeypatch.setattr(redundant_ik,'pose_family',family)
    before=r.base.copy();out=search(r,b,lambda:None)
    row=out['candidates'][0]
    assert [c[1] for c in calls]==[.005,.005,.001,.001]
    assert len(row['right_entry_family'])==2 and 'rejection' not in row
    assert observed==[dict(steps_per_direction=8,joint_limit_margin_degrees=3.)]
    assert out['proposed_station']['right_entry_seed_degrees']==row['right_entry_family'][1]['joint_degrees']
    assert np.array_equal(before,r.base) and not out['motion_authorized']


def test_family_cannot_spend_reserved_final_native_controls(monkeypatch):
    from . import redundant_ik
    r,b,calls=setup(monkeypatch,failed_native=3)
    r.neutral_station_candidates[0]['right_pose_family_steps']=8
    monkeypatch.setattr(redundant_ik,'pose_family',lambda *a,**k:iter([S(joint_degrees=np.ones(7))]))
    b.can_check_with_final_controls=lambda *a,**k:len(calls)<3
    out=search(r,b,lambda:None)
    assert out['budget_limited'] and out['proposed_station'] is None and len(calls)==3


@pytest.mark.parametrize('family',[None,{},dict(tilt=90,normal_sign=1,wing_m=0),
    dict(tilt=15,normal_sign=True,wing_m=0),dict(tilt=15,normal_sign=1,wing_m=float('nan')),
    dict(tilt=15,normal_sign=1,wing_m=.021),dict(tilt=15,normal_sign=1,wing_m=0,path='replay')])
def test_cut_frame_candidates_cannot_expand_original_orientation_limits(tmp_path,family):
    d=document();d['candidates'][0]['cut_frame_family']=family
    path=tmp_path/'candidate.json';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='family'):read_candidates(path,plant='p',target='s')


def test_candidate_frame_changes_tool_proposal_not_original_priority(monkeypatch):
    r,b,calls=setup(monkeypatch)
    family=dict(tilt=-10.,normal_sign=1,wing_m=.01)
    r.neutral_station_candidates[0]['cut_frame_family']=family
    observed=[]
    def wrist(point,direction,normal,wing):
        observed.append((direction.copy(),normal.copy(),wing));return np.eye(4)
    r.knife.wrist_for_edge=wrist
    from .downward_cut import vertical_cut_frame
    expected=vertical_cut_frame(r.seam()[1],1,-10.)
    out=search(r,b,lambda:None)
    np.testing.assert_allclose(observed[0][0],[0.,0.,-1.])
    np.testing.assert_allclose(observed[0][1],expected[1])
    assert observed[0][2]==.01 and out['proposed_station']['cut_frame_family']==family
    assert r.cut_priority==dict(normal_sign=-1,tilt=15.,wing_m=0.)
    assert len(calls)==3 and not out['motion_authorized']


def test_anatomically_invalid_vertical_frame_is_rejected_without_queries(monkeypatch):
    r,b,calls=setup(monkeypatch)
    r.seam=lambda *a:(np.zeros(3),np.array([0.,0.,1.]))
    out=search(r,b,lambda:None)
    assert out['proposed_station'] is None and out['candidates'][0]['rejection']=='downward_frame'
    assert not calls and not r.calls


def test_direct_cut_station_keeps_left_ready_relative_to_new_base_not_old_world_goal(monkeypatch):
    from .neutral_ready import ready_arms
    r,b,calls=setup(monkeypatch);r.park_left_ready=True;r.cut_strategy='right_only'
    r.neutral_station_candidates[0]['station_pose']=[.6,0.,15.]
    out=search(r,b,lambda:None)
    assert r.calls==['right'] and len(calls)==3
    for world,_ in calls:np.testing.assert_array_equal(world['left'],ready_arms()[:7])
    assert out['candidates'][0]['left_grasp_ik'] is None
    assert out['candidates'][0]['left_strategy']=='sdk_ready_park'
    assert out['proposed_station']['left_ik_seed_degrees']==ready_arms()[:7].tolist()
    assert not out['grasp_or_cut_verified'] and not out['motion_authorized']


def test_park_cannot_replace_required_bimanual_grasp(monkeypatch):
    r,b,calls=setup(monkeypatch);r.park_left_ready=True;r.cut_strategy='bimanual'
    with pytest.raises(ValueError,match='right-only'):search(r,b,lambda:None)
    assert not calls and not r.calls


def test_direct_neutral_station_launcher_requires_park_and_remains_zero_motion(monkeypatch,tmp_path):
    from pathlib import Path
    from . import ground_truth_trial,cut_priority
    d=document();d.update(plant='seed101_full',target='SubStem_41')
    path=tmp_path/'candidates.json';path.write_text(json.dumps(d))
    monkeypatch.setattr(cut_priority,'from_arguments',lambda args:dict(tilt=15.,normal_sign=-1,wing_m=0.))
    class BeforeWrite(Exception):pass
    def stop(*a,**k):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'none'),'--mode','right_only','--milestone','cut_action',
        '--process-zone-trial','--screen-station','--neutral-station-candidates',str(path),
        '--torso-degrees','0','0','0','0','0','0']
    with pytest.raises(ValueError,match='Neutral station batch'):ground_truth_trial.main(args)
    with pytest.raises(BeforeWrite):ground_truth_trial.main(args+['--park-left-ready'])
