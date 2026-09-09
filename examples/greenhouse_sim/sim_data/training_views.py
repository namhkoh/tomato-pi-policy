"""Real RB-Y1 torso-height diversity for static grounding snapshots.

Never move the camera relative to its bracket or raise the robot root off the
floor. All poses still go through exact URDF FK, joint limits and scene screens.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .dataset_review import require

PROTOCOL='real_base_head_and_torso_snapshots.v1'


def with_postures(specs,target_id):
    seed=int.from_bytes(hashlib.sha256(('torso-grounding-v1:'+target_id).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    return [{**s,'torso_bend_degrees':float(rng.uniform(0,45))} for s in specs]


def robot_for_spec(robot,spec):
    if 'torso_bend_degrees' not in spec: return robot
    bend=spec['torso_bend_degrees']
    require(type(bend) in (int,float) and np.isfinite(bend) and 0<=bend<=45,'Invalid static torso pose')
    pose={**robot['pose_degrees'],'torso_1':float(bend),'torso_2':float(-2*bend),'torso_3':float(bend)}
    return {**robot,'pose_degrees':pose}
