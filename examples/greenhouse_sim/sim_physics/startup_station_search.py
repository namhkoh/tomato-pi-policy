"""Frozen-scene base/two-arm proposals. Never move the current robot or plant.

The existing grasp world path, actual tool mounting and every collider remain
unchanged. A proposal is a new initial configuration, NOT a base-motion path.
Native startup success does not certify the subsequent manipulation trajectory.
"""
from copy import copy
import time
import numpy as np


def search(robot,backend,guard):
    from scipy.spatial.transform import Rotation
    from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES
    ready=np.array([SDK_READY_POSE_DEGREES[f'right_arm_{i}'] for i in range(7)])
    low,high=robot.kin.arm_limits_degrees('right')
    if np.any(ready<=np.asarray(low)+3) or np.any(ready>=np.asarray(high)-3):
        raise ValueError('SDK ready proposal outside current exact arm limits/margin')
    start=time.monotonic();rows=[];proposal=None
    original_base=np.array(robot.base,copy=True)
    poses=[robot.kin.forward('left',q,original_base) for q in robot.path_q]
    right_pose=robot.kin.forward('right',robot.right,original_base).copy()
    right_pose[2,3]+=.03
    # Back away from the canopy, then change which side each elbow occupies.
    # Same world-space hand goals. No plants removed or contacts disabled.
    choices=[(back,side,yaw) for back in (.05,.10,.15) for side,yaw in
             ((0.,0.),(.10,0.),(-.10,0.),(0.,15.),(0.,-15.))]
    # A new initial arm pose need not already be at the cut. The source SDK
    # ready arm provides a visibly withdrawn alternative to the near-limit
    # precontact wrist. The later complete approach must still be solved.
    choices=[(0.,0.,0.,'sdk_ready')]+[(*v,'sdk_ready') for v in choices]+[(*v,'fixed_world_tool') for v in choices]
    desired_tools={'fixed_world_tool':right_pose}
    priority=getattr(robot,'cut_priority',None)
    if priority is not None:
        from .downward_cut import vertical_cut_frame
        from .blade_aim import edge_centre
        centre,axis=robot.seam(robot.rig.rest_frames)
        frame=vertical_cut_frame(axis,priority['normal_sign'],priority['tilt'])
        if frame is None:raise ValueError('Prior cut-frame family violates current anatomical angles')
        d,normal=frame;aim=edge_centre(centre,axis,robot.blade_axial_aim_offset_m)
        desired=robot.knife.wrist_for_edge(aim+robot.stroke_offsets[0]*d,d,normal,priority['wing_m'])
        withdrawn=desired.copy();withdrawn[:3,3]+=.02*desired[:3,2]
        desired_tools.update(cut_frame=desired,cut_frame_withdrawn_20mm=withdrawn)
        # A valid cut frame can place the camera/forearm differently from the
        # old near-limit staging pose. These are fresh initial-pose proposals,
        # not imported joint paths. All native shapes and left-path checks run.
        stations=[(0.,0.,0.)]+[(back,side,yaw) for back in (.05,.10,.15) for side,yaw in
            ((0.,0.),(.10,0.),(-.10,0.),(0.,15.),(0.,-15.))]
        choices=[(*v,mode) for v in stations for mode in ('cut_frame_withdrawn_20mm','cut_frame')]+choices
    for back,side,yaw,mode in choices:
        guard()
        if time.monotonic()-start>=30:break
        candidate=copy(robot);candidate.base=original_base.copy()
        candidate.base[:3,3]+=-back*original_base[:3,0]+side*original_base[:3,1]
        candidate.base[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@original_base[:3,:3]
        row=dict(base_back_m=back,base_side_m=side,base_yaw_delta_degrees=yaw,
                 right_start_mode=mode,
                 native_startup_clear=False,left_self_path_clear=False,motion_authorized=False)
        rows.append(row)
        distance=np.linalg.norm(candidate.base[:2,3]-robot.rig.chain_world[0,:2])
        if not .25<=distance<=1.:
            row['rejection']='station_distance_outside_original_bounds';continue
        left=robot.kin.solve_pose('left',poses[0],robot.initial_q,candidate.base,
            maximum_evaluations=200,joint_limit_margin_degrees=3.)
        right=None if mode=='sdk_ready' else robot.kin.solve_pose('right',desired_tools[mode],robot.right,candidate.base,
            maximum_evaluations=200,joint_limit_margin_degrees=3.)
        guard()
        if not left.succeeded or right is not None and not right.succeeded:
            row.update(rejection='initial_ik',left_ik=bool(left.succeeded),right_ik=right is None or bool(right.succeeded));continue
        lq=np.array(left.joint_degrees);rq=ready.copy() if right is None else np.array(right.joint_degrees)
        if not candidate.check_self(lq,rq)['passed']:
            row['rejection']='initial_self_collision';continue
        native=backend.check(candidate.body_world(lq,rq),robot.self_screen.shapes)
        row['native_startup_geometry']=native;row['native_startup_clear']=native['passed'] is True
        if not row['native_startup_clear']:continue
        initial=lq.copy();path=[]
        for pose in poses:
            guard()
            if time.monotonic()-start>=30:
                row['rejection']='bounded_path_solve_timeout';break
            solved=robot.kin.solve_pose('left',pose,lq,candidate.base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            if not solved.succeeded:
                row['rejection']='left_path_ik';break
            lq=np.array(solved.joint_degrees)
            if (robot.kin.inter_arm_clearance(lq,rq,candidate.base).clearance_m<.01
                    or not candidate.check_self(lq,rq)['passed']):
                row['rejection']='left_path_self_or_interarm';break
            path.append(lq.tolist())
        else:
            row['left_self_path_clear']=True
            station=[*candidate.base[:2,3],float(np.degrees(np.arctan2(candidate.base[1,0],candidate.base[0,0])))]
            proposal=dict(station_pose=list(map(float,station)),left_ik_seed_degrees=initial.tolist(),
                right_ready_degrees=rq.tolist(),left_path_degrees=path)
            break
    guard()
    return dict(model='frozen_native_two_arm_station_search_v1',candidates=rows,
        proposed_station=proposal,maximum_candidates=len(choices),
        cut_frame_priority_used=priority is not None,
        original_spawn_unchanged=True,plant_or_mounting_changed=False,
        physics_steps=0,motion_authorized=False,relaunch_required=True,
        whole_path_certified=False,base_motion_certified=False,
        proposed_floor_height_requires_fresh_launch=True)
