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
    if 'torso_bend_degrees' not in spec:
        require('torso_lean_degrees' not in spec,'Lean requires a complete torso pose')
        return robot
    bend=spec['torso_bend_degrees']
    require(type(bend) in (int,float) and np.isfinite(bend) and 0<=bend<=45,'Invalid static torso pose')
    lean=spec.get('torso_lean_degrees',0)
    require(type(lean) in (int,float) and np.isfinite(lean) and 0<=lean<=30,'Invalid static torso lean')
    pose={**robot['pose_degrees'],'torso_1':float(bend),'torso_2':float(-2*bend),'torso_3':float(bend+lean)}
    return {**robot,'pose_degrees':pose}


def with_lean(specs,target_id):
    """Opt-in static forward lean; exact FK/floor/scene checks remain mandatory.

    No motion, balance or self-collision certification is implied.
    """
    seed=int.from_bytes(hashlib.sha256(('lean-clear-v1:'+target_id).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    return [dict(s,candidate_id='lean_'+s['candidate_id'],torso_lean_degrees=float(rng.uniform(5,30))) for s in specs]
