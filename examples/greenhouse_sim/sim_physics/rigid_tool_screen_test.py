from types import SimpleNamespace
import numpy as np
import pytest
from sim_physics.self_screen import SelfCapsuleScreen
from sim_physics.rigid_tool_screen import RigidToolScreen


class PlantStub:
    def __init__(self,shapes):
        self.shapes=shapes;self.obstacles=[];self.last_failure=None
    def check(self,world,*,stroke=False):
        self.last_failure={'plant_collider':'blocked'}
        return world['ee_right'][1,3]<1.


def fixture():
    box=(np.zeros(3),np.eye(3),np.full(3,.02))
    shapes=[('/R/'+link+'/shape','/R/'+link,link,'box',box)
        for link in ('ee_left','ee_right','link_right_arm_5')]
    original=object.__new__(SelfCapsuleScreen)
    original.shapes=shapes;original.pairs=[(0,1),(0,2),(1,2)]
    original.box_paths=[s[0] for s in shapes];original.unsupported=[]
    world={s[2]:np.eye(4) for s in shapes};world['link_right_arm_5'][0,3]=1.
    plant=PlantStub(shapes[1:])
    robot=SimpleNamespace(self_screen=original,held_plant_screen=plant,right=np.zeros(7),
        body_world=lambda *args:world)
    return RigidToolScreen(robot,np.zeros(7)),robot,world


def test_required_tool_subset_rejects_interference_without_changing_robot():
    screen,robot,world=fixture();before=world['ee_right'].copy()
    frames=np.repeat(np.eye(4)[None],2,axis=0);frames[0,0,3]=1.
    result=screen.check(frames)
    assert not result['passed'] and result['sample']==1
    assert result['reason']=='rigid_left_right_tool_interference'
    np.testing.assert_array_equal(world['ee_right'],before)
    assert len(robot.self_screen.shapes)==3 and len(robot.self_screen.pairs)==3
    # Clear wrist subset may overlap an untested right-arm shape: no full-path claim.
    result=screen.check(frames[:1])
    assert result['passed'] and not result['whole_arm_path_certified'] and not result['native_validated']


def test_scene_rejection_and_invalid_frames_are_not_hidden():
    screen,robot,_=fixture();frames=np.eye(4)[None];frames[0,:2,3]=[1.,2.]
    result=screen.check(frames)
    assert not result['passed'] and result['reason']=='rigid_tool_scene_interference'
    assert robot.held_plant_screen.last_failure is None
    for bad in (np.empty((0,4,4)),np.full((1,4,4),np.nan),np.eye(3)[None]):
        with pytest.raises(ValueError): screen.check(bad)
