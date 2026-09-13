import json
from collections import Counter
from types import SimpleNamespace as S
import pytest
from .cut_priority import load,tilt_order,proposal_order


def test_order_changes_but_no_candidate_added_removed_or_rewritten():
    tilts=(0.,-10.,10.,-15.,15.)
    p=dict(tilt=15.,normal_sign=-1,wing_m=.004)
    original=[(None,s,w) for w in (0.,-.004,.004,-.008,.008) for s in (1,-1)]
    new=proposal_order(original,15.,p)
    assert new[0]==(None,-1,.004) and Counter(new)==Counter(original)
    assert tilt_order(tilts,p)[0]==15. and Counter(tilt_order(tilts,p))==Counter(tilts)
    assert proposal_order(original,0.,p) is original
    assert proposal_order(original,15.,dict(p,wing_m=.007)) is original
    assert tilt_order(tilts,None) is tilts


def data():
    config=dict(plant='seed101_full',target='SubStem_41',cut_arc_m=.02,
                knife_edge_mode='source_crossbar_edge_v1',knife_alignment='camera')
    report=dict(configuration=config,gates={k:True for k in
        ('blade_contact_release','full_forward_cut_stroke_verified','right_withdrawal_completed')},
        robot=dict(cut_plan=dict(plane_tilt_degrees=15.,normal_sign=-1,wing_m=.004,
                                transit_q=['never_replay'])))
    return S(**config),report


def test_loader_uses_no_prior_metric_pose_or_joint_trajectory(tmp_path):
    args,r=data();path=tmp_path/'cut.json';path.write_text(json.dumps(r))
    p=load(path,args)
    assert p['tilt']==15. and p['normal_sign']==-1 and p['wing_m']==.004
    assert not p['prior_pose_or_path_replayed'] and not p['motion_authorized']
    assert 'transit_q' not in p and len(p['source_sha256'])==64


@pytest.mark.parametrize('bad',['target','partial','nan','unknown_tilt','bool_sign'])
def test_mismatched_or_invalid_priority_refused(tmp_path,bad):
    args,r=data();p=r['robot']['cut_plan']
    if bad=='target':r['configuration']['target']='different'
    elif bad=='partial':r['gates']['right_withdrawal_completed']=False
    elif bad=='nan':p['wing_m']=float('nan')
    elif bad=='unknown_tilt':p['plane_tilt_degrees']=30.
    else:p['normal_sign']=True
    path=tmp_path/'bad.json';path.write_text(json.dumps(r))
    with pytest.raises(ValueError):load(path,args)
