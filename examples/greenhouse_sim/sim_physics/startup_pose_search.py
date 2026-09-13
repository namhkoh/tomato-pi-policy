"""Bounded same-wrist elbow proposals in a frozen native scene, never motion.

All source/robot native controls remain owned by native_startup_screen. A
candidate does not repair the currently authored colliding spawn: save it and
relaunch, then repeat startup, whole approach, grasp and cut qualification.
"""
import time
import numpy as np


def search(robot,backend,guard):
    from .redundant_ik import pose_family
    seed=np.array(robot.right,float,copy=True)
    desired=robot.kin.forward('right',seed,robot.base)
    begin=time.monotonic();rows=[];candidate=None
    for solved in pose_family(robot.kin,'right',desired,seed,robot.base,
            steps_per_direction=12,joint_limit_margin_degrees=3.):
        guard()
        if time.monotonic()-begin>=30:break  # Leave time for final native controls.
        q=np.asarray(solved.joint_degrees,float)
        row=dict(right_ready_degrees=q.tolist(),native_startup_clear=False,
            left_self_path_clear=False,motion_authorized=False)
        rows.append(row)
        # Every existing dense left-arm path knot; no new grasp or path accepted.
        for left in robot.path_q:
            guard()
            arm=robot.kin.inter_arm_clearance(left,q,robot.base).clearance_m
            check=robot.check_self(left,q)
            if arm<.01 or not check['passed']:
                row.update(rejection='left_path_self_or_interarm',interarm_m=float(arm),self_screen=check)
                break
        else:
            row['left_self_path_clear']=True
            geometry=backend.check(robot.body_world(robot.initial_q,q),robot.self_screen.shapes)
            row['native_startup_geometry']=geometry
            row['native_startup_clear']=geometry['passed'] is True
            if geometry['passed']:
                candidate=q.tolist();break
    guard()
    return dict(model='frozen_native_same_wrist_elbow_search_v1',candidates=rows,
        proposed_right_ready_degrees=candidate,maximum_candidates=24,
        original_spawn_unchanged=True,relaunch_required=True,physics_steps=0,
        motion_authorized=False,whole_path_certified=False,grasp_or_cut_verified=False)
