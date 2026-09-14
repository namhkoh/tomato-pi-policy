import numpy as np
import pytest
from sim_physics.held_plant_screen_test import fixture
from sim_physics.held_plant_screen import HeldPlantScreen
from sim_physics.whole_robot_target import WholeRobotTargetScreen


def test_larger_requested_transit_margin_reaches_complete_target_screen(monkeypatch):
    screen,rig,poses=fixture()
    whole=WholeRobotTargetScreen(screen,screen.shapes)
    calls=[]
    monkeypatch.setattr(whole.screen,'check',lambda world,**kw:calls.append(kw) or True)
    assert whole.check(poses,margin=.005)['margin_m']==.005
    assert calls==[dict(grasp=False,margin=.005)]
    whole.check(poses)
    assert calls[-1]==dict(grasp=False,margin=.001)


def shape(link, centre):
    return ('/World/R/'+link+'/Shape','/World/R/'+link,link,'box',
            (np.array(centre,float),np.eye(3),np.full(3,.005)))


@pytest.mark.parametrize('link',['base','link_torso_5','link_head_2','ee_right','ee_left'])
def test_bodies_omitted_by_arm_only_preview_still_reject_leaf_contact(link):
    original,rig,_=fixture()
    safe=shape('link_left_arm_5',[.5,0,0]);other=shape(link,[.1,0,0])
    screen=HeldPlantScreen(rig,[safe,other],'unused',arm='left',grasp_path=rig.body_paths[1])
    screen.snapshot(rig.rest_frames)
    whole=WholeRobotTargetScreen(screen,[safe,other])
    poses={s[2]:np.eye(4) for s in (safe,other)}
    result=whole.check(poses,grasp=True)
    assert not result['passed']
    assert result['failure']['robot_collider']==other[0]
    assert result['failure']['plant_collider'].endswith('/Leaf')
    assert not result['motion_authorized'] and not result['whole_scene_certified']


def test_target_only_subset_does_not_weaken_original_workspace_or_context():
    screen,rig,poses=fixture()
    screen.workspace=(np.full(3,-.01),np.full(3,.01))
    far=rig.rest_frames.copy();far[1,0,3]=.5;screen.snapshot(far)
    whole=WholeRobotTargetScreen(screen,screen.shapes)
    assert whole.check(poses)['passed']
    assert not screen.check(poses)  # Original cache bound still enforced.
    assert screen.workspace is not None
    assert whole.screen.obstacles is not screen.obstacles
    assert not whole.check(poses)['context_scene_checked']
    # A new snapshot must be explicitly compiled; no silent stale-frame reuse.
    screen.snapshot(rig.rest_frames)
    assert not WholeRobotTargetScreen(screen,screen.shapes).check(poses)['passed']


@pytest.mark.parametrize('link,expected',[('ee_finger_l1',True),('ee_finger_l2',True),
                                        ('ee_left',False),('ee_right',False),('link_torso_5',False)])
def test_intended_grasp_allowance_stays_limited_to_left_fingers(link,expected):
    _,rig,_=fixture();s=shape(link,[0,0,0])
    screen=HeldPlantScreen(rig,[s,shape('link_left_arm_5',[.5,0,0])],'unused',arm='left',grasp_path=rig.body_paths[1])
    screen.snapshot(rig.rest_frames)
    assert WholeRobotTargetScreen(screen,[s]).check({link:np.eye(4)},grasp=True)['passed']==expected
    poses={link:np.eye(4)};poses[link][0,3]=.1
    assert not WholeRobotTargetScreen(screen,[s]).check(poses,grasp=True)['passed']


@pytest.mark.parametrize('fault',['empty','duplicate_robot','missing_target','duplicate_target'])
def test_incomplete_inventory_cannot_return_pass(fault):
    screen,rig,_=fixture();shapes=list(screen.shapes)
    if fault=='empty':shapes=[]
    if fault=='duplicate_robot':shapes*=2
    if fault=='missing_target':screen.obstacles.pop()
    if fault=='duplicate_target':screen.obstacles.append(screen.obstacles[0])
    with pytest.raises(ValueError):WholeRobotTargetScreen(screen,shapes)
