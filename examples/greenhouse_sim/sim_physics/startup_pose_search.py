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
    attempts={'local_solutions':0,'global_ik_attempts':0}
    def proposals():
        for solved in pose_family(robot.kin,'right',desired,seed,robot.base,
                steps_per_direction=12,joint_limit_margin_degrees=3.):
            guard()
            if time.monotonic()-begin>=30:return
            attempts['local_solutions']+=1
            yield 'local_self_motion',solved
        # A connected self-motion family can end at a wrist limit while an
        # entirely different elbow branch reaches the SAME wrist pose. These
        # deterministic multistarts are proposals, never motion permissions.
        low,high=robot.kin.arm_limits_degrees('right')
        rng=np.random.default_rng(267)
        for _ in range(16):
            guard()
            if time.monotonic()-begin>=30:return
            proposed_seed=rng.uniform(np.asarray(low)+10,np.asarray(high)-10)
            attempts['global_ik_attempts']+=1
            solved=robot.kin.solve_pose('right',desired,proposed_seed,robot.base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            if solved.succeeded:yield 'global_multistart',solved
    for origin,solved in proposals():
        guard()
        if time.monotonic()-begin>=30 or len(rows)>=24:break  # Leave final-control reserve.
        q=np.asarray(solved.joint_degrees,float)
        row=dict(right_ready_degrees=q.tolist(),native_startup_clear=False,
            left_self_path_clear=False,motion_authorized=False,ik_origin=origin)
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
        ik_attempts=attempts,global_multistart_seed=267,
        proposed_right_ready_degrees=candidate,maximum_candidates=24,
        original_spawn_unchanged=True,relaunch_required=True,physics_steps=0,
        motion_authorized=False,whole_path_certified=False,grasp_or_cut_verified=False)
