from copy import deepcopy
from types import SimpleNamespace as S
import pytest
from pxr import Usd,UsdPhysics
from .cut_action import assess
from .released_debris import ReleasedDebris
from .contact_events import ContactEvents
from .signed_blade_test import seam_fixture,evidence
from .knife import BRITTLE_CUT_MODEL


def fixture():
    rig,joint=seam_fixture()
    f=S(root='/World/R',rig=rig,cut_strategy='bimanual',cut_event=None,
        body_paths=['/World/R/link_torso_4','/World/R/base','/World/R/ee_left','/World/R/ee_right'],
        held_plant_screen=S(local=[('/World/A/StemCollider',0,'capsule',None),
                                  ('/World/B/StemCollider',1,'capsule',None),
                                  ('/World/B/Leaf/Collision',1,'hull',None)]))
    m=ContactEvents(robot_root=f.root,target_root='/World',fingers=[],floor_root=None)
    return f,joint,m


def release(f):
    f.cut_event=f.rig.release_from_blade(evidence(BRITTLE_CUT_MODEL))
    return ReleasedDebris(f)


def test_torso_debris_is_separate_and_full_impulses_are_preserved():
    f,j,m=fixture();m.released_debris_policy=release(f)
    m.consume('/World/B/Leaf/Collision','/World/R/link_torso_4/collider',[[.008,0,0]],
              friction_impulses=[[0,.002,0]])
    r=m.measurements(.01)
    assert r['released_debris_contact_n']==pytest.approx(1.)
    assert r['released_debris_friction_n']==pytest.approx(.2)
    assert r['unwanted_contact_n']==0 and r['total_contact_upper_bound_n']==pytest.approx(1.)
    assert sum(m.pairs.values())==pytest.approx(.01)
    assert not m.released_debris_policy.report()['native_collision_filters_changed']


@pytest.mark.parametrize('plant,robot',[
    ('/World/A/StemCollider','/World/R/link_torso_4/collider'),
    ('/World/B/Leaf/Collision_other','/World/R/link_torso_4/collider'),
    ('/World/Neighbor/Leaf/Collision','/World/R/link_torso_4/collider'),
    ('/World/B/Leaf/Collision','/World/R/ee_right/attachments/knife'),
    ('/World/B/Leaf/Collision','/World/R/ee_left/collider'),
    ('/World/B/Leaf/Collision','/World/R/link_torso_40/collider')])
def test_exact_released_target_and_passive_bodies_only(plant,robot):
    f,j,m=fixture();m.released_debris_policy=release(f)
    m.consume(plant,robot,[[.01,0,0]])
    r=m.measurements(.01)
    assert r['released_debris_contact_n']==0 and r['unwanted_contact_n']==1.


def test_attached_or_fake_release_cannot_enable_debris():
    f,j,m=fixture()
    with pytest.raises(ValueError):ReleasedDebris(f)
    release(f)
    with Usd.EditContext(f.rig.stage,f.rig.stage.GetSessionLayer()):
        j.GetJointEnabledAttr().Set(True)
    with pytest.raises(ValueError,match='disabled'):ReleasedDebris(f)


def test_release_identity_change_reset_and_close_fail_safe():
    f,j,m=fixture();p=release(f);m.released_debris_policy=p
    f.cut_event=deepcopy(f.cut_event)
    with pytest.raises(RuntimeError):p.matches('/World/R/base/collider','/World/B/StemCollider')
    f.rig.cut=False
    assert not p.matches('/World/R/base/collider','/World/B/StemCollider')
    m.close();assert m.released_debris_policy is None


def action_input():
    return dict(cut_only=False,grasp_verified=True,planned=True,cut_time=1.,fault=None,
        cut_event=dict(event='blade_contact_joint_release',evidence=dict(stable_left_grasp=True)),
        records=[dict(t=3.,cut=True,native_guards_passed=True)])


def test_action_success_is_not_withdrawal_retention_or_deposit_claim():
    r=assess(**action_input())
    assert r['passed']
    assert not any(r[k] for k in ['verified_withdrawal_required','retention_required',
                                 'deposit_required','landing_safety_verified','training_eligible'])


@pytest.mark.parametrize('failure',['fault','no_plan','no_cut','no_grasp','no_guard','no_after'])
def test_cut_action_keeps_execution_and_pre_cut_support_gates(failure):
    a=action_input()
    if failure=='fault':a['fault']='active knife hit neighbor'
    elif failure=='no_plan':a['planned']=False
    elif failure=='no_cut':a['cut_event']=None
    elif failure=='no_grasp':a['grasp_verified']=False
    elif failure=='no_guard':a['records'][0]['native_guards_passed']=False
    else:a['records'][0]['t']=2.99
    assert not assess(**a)['passed']


def test_cut_only_does_not_claim_left_grasp():
    a=action_input();a.update(cut_only=True,grasp_verified=False)
    a['cut_event']['evidence']=dict(cut_strategy='right_only',stable_left_grasp=False,
                                  grasp_slip_m=None,cut_only_ready=True)
    assert assess(**a)['passed']
    a['grasp_verified']=True
    assert not assess(**a)['passed']
