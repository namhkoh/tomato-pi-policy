"""Analytic shape reduction and actual Model A FK; no native motion proof."""
from dataclasses import replace
from types import SimpleNamespace as S
import numpy as np
import pytest
from greenhouse_sim.robot_kinematics import Rby1Kinematics
from .wrist_invariant import capsule_in_wrist
from .rigid_tool_screen_test import fixture


def capsule():
    return ('/R/link_right_arm_5/capsule_00','/R/link_right_arm_5',
        'link_right_arm_5','capsule',(np.array([0.,0.,-.05]),np.array([0.,0.,.002]),.075))


def test_actual_model_capsule_is_invariant_for_arbitrary_arm_branches():
    k=Rby1Kinematics();source=capsule();mapped=capsule_in_wrist(source,k)
    assert mapped[:2]==source[:2] and mapped[2:4]==('ee_right','capsule')
    rng=np.random.default_rng(41);low,high=k.arm_limits_degrees('right')
    for q in rng.uniform(low+3,high-3,(80,7)):
        frames=k.all_link_transforms({f'right_arm_{i}':v for i,v in enumerate(q)})
        a,b=frames[source[2]],frames['ee_right']
        actual=np.array(source[4][:2])@a[:3,:3].T+a[:3,3]
        transformed=np.array(mapped[4][:2])@b[:3,:3].T+b[:3,3]
        np.testing.assert_allclose(transformed,actual,rtol=0,atol=1e-14)
    assert 0<source[4][2]-mapped[4][2]<1e-9


@pytest.mark.parametrize('change',['kin','box','link','axis','fixed','parent','origin','offaxis'])
def test_unproven_geometry_is_not_added_to_rigid_subset(change):
    k=Rby1Kinematics();s=capsule()
    if change=='kin':k=None
    elif change=='box':s=(*s[:3],'box',s[4])
    elif change=='link':s=(*s[:2],'link_right_arm_4',*s[3:])
    elif change=='axis':k._by_child['link_right_arm_6']=replace(k._by_child['link_right_arm_6'],axis=np.array([1.,0,0]))
    elif change=='fixed':k._by_child['ee_right']=replace(k._by_child['ee_right'],kind='revolute')
    elif change=='parent':k._by_child['link_right_arm_6']=replace(k._by_child['link_right_arm_6'],parent='other')
    elif change=='origin':
        m=np.eye(4);m[0,0]=2
        k._by_child['ee_right']=replace(k._by_child['ee_right'],origin=m)
    else:s[4][0][0]=.0001
    assert capsule_in_wrist(s,k) is None


def test_roundoff_correction_is_contained_not_inflated():
    k=Rby1Kinematics();s=capsule();s[4][0][0]=3e-9
    mapped=capsule_in_wrist(s,k)
    assert mapped[4][2]<s[4][2]-3e-9
    assert mapped[4][0][0]==0


def test_forearm_camera_conflict_rejected_before_ik_without_source_changes():
    _,r,_=fixture();r.kin=Rby1Kinematics()
    source=capsule();r.self_screen.shapes[2]=source
    r.held_plant_screen.shapes[1]=source
    before=r.self_screen.shapes.copy()
    from .rigid_tool_screen import RigidToolScreen
    screen=RigidToolScreen(r,np.zeros(7))
    frame=np.eye(4);frame[0,3]=.06;frame[2,3]=-.10
    # Tool box is clear of the left box; actual coaxial forearm is not.
    result=screen.check(frame[None])
    assert not result['passed'] and result['reason']=='rigid_left_right_tool_interference'
    assert source[0] in result['detail']['nearest_pair']
    assert result['wrist_invariant_capsules']==[source[0]]
    assert r.self_screen.shapes==before
    assert len(r.self_screen.pairs)==3 and len(screen.self_screen.pairs)==2
    assert screen.plant_screen.shapes[-1][2]=='ee_right'
    assert r.held_plant_screen.shapes[-1][2]=='link_right_arm_5'
    assert not result['whole_arm_path_certified'] and not result['native_validated']
