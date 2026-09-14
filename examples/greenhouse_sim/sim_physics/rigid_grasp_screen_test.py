"""Pure geometry/provenance contracts; not physical grasp evidence."""
from types import SimpleNamespace as S
from copy import deepcopy
import numpy as np
import pytest
from greenhouse_sim.robot_kinematics import Rby1Kinematics
from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES
from .rigid_grasp_screen import RigidGraspScreen


def fixture(monkeypatch):
    from . import held_plant_screen
    shapes=[('/R/'+s+'/shape','/R/'+s,s,'box',(np.zeros(3),np.eye(3),np.ones(3)*.01))
        for s in ('ee_left','ee_finger_l1','ee_finger_l2','link_left_arm_5')]
    class Screen:
        def __init__(self,*a,**kw):self.shapes=list(shapes);self.last_failure=None;self.calls=[]
        def snapshot(self,frames):self.frames=frames.copy()
        def set_physical_grasp_span(self,*args):self.span=args;return dict(allowed_indices=[1])
        def check(self,world,*,grasp):
            assert grasp
            self.calls.append({k:v.copy() for k,v in world.items()});return True
    monkeypatch.setattr(held_plant_screen,'HeldPlantScreen',Screen)
    r=S(kin=Rby1Kinematics(),pose=dict(SDK_READY_POSE_DEGREES),slides={'gripper_finger_l1':-.008,'gripper_finger_l2':.008},
        rig=S(stem_contact_model='flat_cylinders_v1',body_paths=['a','b']),self_screen=S(shapes=shapes),
        knife=S(collider='/R/knife'),grasp_path='b',body_index=1,grasp_point=lambda frames:frames[1,:3,3],
        pregrasp_half_aperture=.008,effort_bounded_grasp_target=True,force_closer=S(minimum=.002),
        radius=.003,grasp_compression=.0005)
    frames=np.repeat(np.eye(4)[None],2,axis=0)
    return r,RigidGraspScreen(r,frames)


def test_actual_model_finger_fk_and_all_hand_shapes_are_preserved(monkeypatch):
    from scipy.spatial.transform import Rotation
    r,screen=fixture(monkeypatch);pose=deepcopy(r.pose);slides=deepcopy(r.slides)
    rng=np.random.default_rng(67)
    low,high=r.kin.arm_limits_degrees('left')
    for _ in range(20):
        candidate=dict(r.pose);candidate.update({f'left_arm_{i}':v for i,v in enumerate(rng.uniform(low+3,high-3))})
        base=np.eye(4);base[:3,:3]=Rotation.random(random_state=rng).as_matrix();base[:3,3]=rng.uniform(-1,1,3)
        for gap in screen.gaps:
            fk=r.kin.all_link_transforms(candidate,prismatic_m={'gripper_finger_l1':-gap,'gripper_finger_l2':gap})
            world=screen.world(base@fk['ee_left'],gap)
            for link in world:np.testing.assert_allclose(world[link],base@fk[link],atol=1e-12)
    assert {s[2] for s in screen.screen.shapes}=={'ee_left','ee_finger_l1','ee_finger_l2'}
    assert len(r.self_screen.shapes)==4 and r.pose==pose and r.slides==slides


def test_complete_sampled_approach_and_closure_never_authorize_motion(monkeypatch):
    r,screen=fixture(monkeypatch);start=np.eye(4);goal=start.copy();goal[0,3]=.02
    result=screen.check(start,goal)
    assert result['passed'] and result['checks']==21+len(screen.gaps)
    assert not any(result[k] for k in ('motion_authorized','static_context_checked',
        'whole_arm_path_certified','native_contact_verified','continuous_path_certified','collision_filters_changed'))
    assert result['physical_grasp_span']['allowed_indices']==[1]
    assert screen.screen.span[2]==1


@pytest.mark.parametrize('phase',['approach','closure'])
def test_leaf_obstruction_cannot_become_grasp_permission(monkeypatch,phase):
    r,screen=fixture(monkeypatch);start=np.eye(4);goal=start.copy();goal[0,3]=.02
    def check(world,*,grasp):
        palm=world['ee_left'];gap=world['ee_finger_l1'][0,3]-palm[0,3]-.003
        blocked=.009<palm[0,3]<.011 if phase=='approach' else gap<.005
        screen.screen.last_failure=dict(plant_collider='/Target/Leaf_000')
        return not blocked
    screen.screen.check=check
    result=screen.check(start,goal)
    assert not result['passed'] and result['failure']['plant_collider']=='/Target/Leaf_000'
    assert result['phase'].startswith(phase) and not result['motion_authorized']


@pytest.mark.parametrize('bad',['nonrigid','long','nan'])
def test_invalid_or_unbounded_wrist_path_refuses(monkeypatch,bad):
    _,screen=fixture(monkeypatch);start=np.eye(4);goal=start.copy()
    if bad=='nonrigid':goal[0,0]=2
    if bad=='long':goal[0,3]=.5
    if bad=='nan':goal[0,3]=np.nan
    with pytest.raises(ValueError):screen.check(start,goal)


def test_unknown_finger_chain_is_not_assumed_rigid(monkeypatch):
    r,_=fixture(monkeypatch);r.kin._by_child=dict(r.kin._by_child);r.kin._by_child.pop('ee_finger_l1')
    with pytest.raises(ValueError,match='prismatic'):RigidGraspScreen(r,np.repeat(np.eye(4)[None],2,axis=0))


def test_incomplete_frame_inventory_refuses(monkeypatch):
    r,_=fixture(monkeypatch)
    with pytest.raises(ValueError,match='inventory'):RigidGraspScreen(r,np.eye(4)[None])
