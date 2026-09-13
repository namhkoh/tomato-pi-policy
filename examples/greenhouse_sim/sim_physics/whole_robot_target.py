"""Complete robot versus the current target snapshot, not a scene certificate.

The arm workspace cache cannot cover the base/torso. Check these bodies against
ALL target geometry separately, without claiming coverage of omitted context.
No native query, pose, force, collision filter or target geometry is changed.
"""
from copy import copy
import numpy as np


class WholeRobotTargetScreen:
    def __init__(self, held_screen, robot_shapes):
        shapes=list(robot_shapes)
        if not shapes or len({s[0] for s in shapes})!=len(shapes):
            raise ValueError('Complete unique robot collision shapes required')
        paths={row[0] for row in held_screen.local}
        if not paths or not hasattr(held_screen,'obstacles'):
            raise ValueError('Current complete target snapshot required')
        target=[row for row in held_screen.obstacles if row[0] in paths]
        if len(target)!=len(paths) or {row[0] for row in target}!=paths:
            raise ValueError('Missing or duplicate current target geometry')
        self.screen=copy(held_screen)
        self.screen.shapes=shapes
        self.screen.obstacles=target
        self.screen.lower=np.array([row[3] for row in target])
        self.screen.upper=np.array([row[4] for row in target])
        self.screen.static=[]
        self.screen.static_indices={}
        self.screen.workspace=None  # Target-only finite inventory; NOT scene coverage.
        self.screen.native_static_query=None
        self.screen.triangle_indices={p:v for p,v in held_screen.triangle_indices.items() if p in paths}
        self.checks=0

    def check(self, body_world, *, grasp=False):
        self.checks+=1
        passed=self.screen.check(body_world,grasp=grasp)
        return dict(model='complete_robot_current_target_subset_v1',passed=bool(passed),
            failure=self.screen.last_failure,robot_colliders=len(self.screen.shapes),
            target_colliders=len(self.screen.obstacles),checks=self.checks,
            context_scene_checked=False,whole_scene_certified=False,
            native_contact_verified=False,motion_authorized=False)
