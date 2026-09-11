"""Reject impossible wrist-tool corridors before spending arm IK evaluations.

This subset screen cannot certify an arm path, grasp, collision-free execution
or cutting. It only checks fixed left geometry against the requested right
wrist tool, plus that tool against the supplied current scene snapshot.
"""
from copy import copy
import numpy as np


class RigidToolScreen:
    def __init__(self,robot,left):
        source=robot.self_screen
        selected=[i for i,s in enumerate(source.shapes)
            if s[2]=='ee_right' or s[2].startswith(('link_left_arm_','ee_left','ee_finger_l'))]
        remap={old:new for new,old in enumerate(selected)}
        self.self_screen=copy(source)
        self.self_screen.shapes=[source.shapes[i] for i in selected]
        self.self_screen.pairs=[(remap[i],remap[j]) for i,j in source.pairs
            if i in remap and j in remap and (source.shapes[i][2]=='ee_right')!=(source.shapes[j][2]=='ee_right')]
        self.self_screen.box_paths=[s[0] for s in self.self_screen.shapes if s[3]=='box']
        if not self.self_screen.pairs: raise ValueError('Missing left-versus-right-tool pairs')
        self.plant_screen=copy(robot.held_plant_screen)
        self.plant_screen.shapes=[s for s in self.plant_screen.shapes if s[2]=='ee_right']
        if not self.plant_screen.shapes or not hasattr(self.plant_screen,'obstacles'):
            raise ValueError('Current plant snapshot and fitted right tool required')
        self.world=robot.body_world(left,robot.right)

    def check(self,wrist_frames):
        frames=np.asarray(wrist_frames,float)
        if frames.ndim!=3 or frames.shape[1:]!=(4,4) or not len(frames) or not np.isfinite(frames).all():
            raise ValueError('Nonempty finite wrist-frame sequence required')
        result=dict(passed=False,checked_frames=0,whole_arm_path_certified=False,
            native_validated=False,training_eligible=False)
        world=dict(self.world)
        for i,frame in enumerate(frames):
            world['ee_right']=frame
            pair=self.self_screen.check(world)
            result['checked_frames']+=1
            if not pair['passed']:
                result.update(reason='rigid_left_right_tool_interference',sample=i,detail=pair)
                return result
            # First pose must be free of seam contact too; later contact with
            # the intended seam is only a planning allowance, not a cut.
            if not self.plant_screen.check(world,stroke=i>0):
                result.update(reason='rigid_tool_scene_interference',sample=i,detail=self.plant_screen.last_failure)
                return result
        result.update(passed=True,reason='required_subset_clear_not_full_path')
        return result
