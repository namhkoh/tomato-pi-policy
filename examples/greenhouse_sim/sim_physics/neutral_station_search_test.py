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
