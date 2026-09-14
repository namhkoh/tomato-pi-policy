"""Reject blocked hand approaches before arm IK, using original target shapes.

This is a necessary target-only subset: no whole-robot, surrounding-scene,
continuous-path, native contact or retention certificate. No pose/drive setters.
"""
import numpy as np
from .grasp_frame import approach_rotation
from .pregrasp_aperture import closure_samples
from .shaft_grasp import _poses


class RigidGraspScreen:
    def __init__(self,robot,frames):
        from .held_plant_screen import HeldPlantScreen
        self.frames=_poses(frames).copy()
        if len(self.frames)!=len(robot.rig.body_paths) or not len(self.frames):
            raise ValueError('Complete ordered target frame inventory required')
        self.robot=robot
        self.screen=HeldPlantScreen(robot.rig,robot.self_screen.shapes,robot.knife.collider,
            arm='left',grasp_path=robot.grasp_path)
        self.screen.shapes=[s for s in self.screen.shapes
            if s[2] in ('ee_left','ee_finger_l1','ee_finger_l2')]
        if {s[2] for s in self.screen.shapes}!={'ee_left','ee_finger_l1','ee_finger_l2'}:
            raise ValueError('Complete original left palm and both finger geometry required')
        # Prove the movable fingers are direct, named prismatic children of
        # the wrist. Unknown mechanisms cannot inherit a rigid hand proposal.
        for link,joint in (('ee_finger_l1','gripper_finger_l1'),('ee_finger_l2','gripper_finger_l2')):
            parsed=robot.kin._by_child.get(link)
            if (parsed is None or parsed.parent!='ee_left' or parsed.child!=link
                    or parsed.kind!='prismatic' or parsed.name!=joint):
                raise ValueError('Original wrist-attached prismatic fingers required')
        self.screen.snapshot(self.frames)
        self.opening=robot.pregrasp_half_aperture
        self.minimum=(robot.force_closer.minimum if robot.effort_bounded_grasp_target
            else max(0.,robot.radius-robot.grasp_compression))
        self.gaps=closure_samples(self.opening,self.minimum)
        self.relative={}
        for gap in self.gaps:
            slides=dict(robot.slides)
            slides.update(gripper_finger_l1=-float(gap),gripper_finger_l2=float(gap))
            world=robot.kin.all_link_transforms(dict(robot.pose),prismatic_m=slides)
            inverse=np.linalg.inv(world['ee_left'])
            relative={link:inverse@world[link] for link in ('ee_left','ee_finger_l1','ee_finger_l2')}
            _poses(np.array(list(relative.values())))
            self.relative[float(gap)]=relative

    def world(self,wrist,gap):
        return {link:wrist@relative for link,relative in self.relative[float(gap)].items()}

    def check(self,start,goal):
        from scipy.spatial.transform import Rotation
        start,goal=_poses(np.array([start,goal]))
        span=None
        if getattr(self.robot.rig,'stem_contact_model',None) in ('continuous_internal_capsules_v1','flat_cylinders_v1'):
            span=self.screen.set_physical_grasp_span(self.frames,self.world(goal,self.minimum),
                self.robot.body_index,self.robot.grasp_point(self.frames))
        result=dict(model='rigid_hand_current_target_subset_v1',passed=False,checks=0,
            physical_grasp_span=span,whole_arm_path_certified=False,static_context_checked=False,
            continuous_path_certified=False,native_contact_verified=False,
            motion_authorized=False,collision_filters_changed=False)
        length=np.linalg.norm(goal[:3,3]-start[:3,3])
        angle=Rotation.from_matrix(goal[:3,:3]@start[:3,:3].T).magnitude()
        count=max(3,int(np.ceil(length/.001))+1,int(np.ceil(angle/np.radians(1)))+1)
        if count>256:raise ValueError('Bounded local hand approach required')
        def clear(wrist,gap,phase,coordinate):
            result['checks']+=1
            if self.screen.check(self.world(wrist,gap),grasp=True):return True
            result.update(failure=dict(self.screen.last_failure),phase=phase,coordinate=float(coordinate))
            return False
        for fraction in np.linspace(0.,1.,count):
            wrist=start.copy();wrist[:3,3]+=fraction*(goal[:3,3]-start[:3,3])
            wrist[:3,:3]=approach_rotation(start[:3,:3],goal[:3,:3],float(fraction))
            if not clear(wrist,self.opening,'approach_fraction',fraction):return result
        for gap in self.gaps:
            if not clear(goal,gap,'closure_half_aperture_m',gap):return result
        result['passed']=True
        return result
