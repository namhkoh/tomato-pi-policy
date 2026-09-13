"""Task-independent left park for an explicit unheld right-only diagnostic.

This changes a proposed INITIAL configuration, never measured poses. Every
source collider and runtime open/unloaded/tracking guard remains required.
"""
import numpy as np


def initialize(robot):
    from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES
    if getattr(robot,'cut_strategy',None)!='right_only':
        raise ValueError('Independent left park is only for explicit right-only cutting')
    q=np.array([SDK_READY_POSE_DEGREES[f'left_arm_{i}'] for i in range(7)],float)
    low,high=robot.kin.arm_limits_degrees('left')
    if not np.isfinite(q).all() or np.any(q<=np.asarray(low)+3) or np.any(q>=np.asarray(high)-3):
        raise ValueError('SDK left park must retain exact source joint limits and 3 degree reserve')
    robot.initial_q=q
    robot.start=robot.kin.forward('left',q,robot.base)
    robot.goal=robot.start.copy()
    robot.pregrasp_ik_attempts=0


def stationary_path(robot):
    if getattr(robot,'cut_strategy',None)!='right_only':
        raise ValueError('Stationary left path cannot replace a bimanual approach')
    pose=robot.kin.forward('left',robot.initial_q,robot.base)
    if (not np.isfinite(pose).all() or not np.allclose(robot.start,pose,atol=1e-10,rtol=0)
            or not np.allclose(robot.goal,pose,atol=1e-10,rtol=0)):
        raise ValueError('Task-independent left park pose changed')
    clearance=robot.kin.inter_arm_clearance(robot.initial_q,robot.right,robot.base).clearance_m
    if not np.isfinite(clearance) or clearance<.01:
        raise RuntimeError('Independent left park fails inter-arm clearance')
    robot.fractions=np.array([0.,1.15])
    robot.path_q=np.array([robot.initial_q.copy(),robot.initial_q.copy()])
    robot.minimum_interarm=float(clearance)
    robot.expected_palm=pose.copy()
