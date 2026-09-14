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
        from .wrist_invariant import capsule_in_wrist
        invariant={i:converted for i,s in enumerate(source.shapes)
            if (converted:=capsule_in_wrist(s,getattr(robot,'kin',None))) is not None}
        right={i for i,s in enumerate(source.shapes) if s[2]=='ee_right'}|set(invariant)
        selected=[i for i,s in enumerate(source.shapes)
            if i in right or s[2].startswith(('link_left_arm_','ee_left','ee_finger_l'))]
        remap={old:new for new,old in enumerate(selected)}
        self.self_screen=copy(source)
        self.self_screen.shapes=[invariant.get(i,source.shapes[i]) for i in selected]
        self.self_screen.pairs=[(remap[i],remap[j]) for i,j in source.pairs
            if i in remap and j in remap and (i in right)!=(j in right)]
        self.self_screen.box_paths=[s[0] for s in self.self_screen.shapes if s[3]=='box']
        if not self.self_screen.pairs: raise ValueError('Missing left-versus-right-tool pairs')
        self.plant_screen=copy(robot.held_plant_screen)
        converted={s[0]:s for s in invariant.values()}
        self.plant_screen.shapes=[converted.get(s[0],s) for s in self.plant_screen.shapes
            if s[2]=='ee_right' or s[0] in converted]
        self.invariant_paths=list(converted)
        if not self.plant_screen.shapes or not hasattr(self.plant_screen,'obstacles'):
            raise ValueError('Current plant snapshot and fitted right tool required')
        self.world=robot.body_world(left,robot.right)

    def check(self,wrist_frames,*,stroke=True):
        if type(stroke) is not bool:raise ValueError('Explicit stroke allowance required')
        frames=np.asarray(wrist_frames,float)
        if frames.ndim!=3 or frames.shape[1:]!=(4,4) or not len(frames) or not np.isfinite(frames).all():
            raise ValueError('Nonempty finite wrist-frame sequence required')
        result=dict(passed=False,checked_frames=0,whole_arm_path_certified=False,
            native_validated=False,training_eligible=False,
            wrist_invariant_capsules=self.invariant_paths)
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
            if not self.plant_screen.check(world,stroke=stroke and i>0):
                result.update(reason='rigid_tool_scene_interference',sample=i,detail=self.plant_screen.last_failure)
                return result
        result.update(passed=True,reason='required_subset_clear_not_full_path')
        return result
