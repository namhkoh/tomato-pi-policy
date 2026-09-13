"""Frozen-scene cut-oriented station proposals, never robot motion.

Keep both hand goals, anatomy, camera mounting and every collider unchanged.
Unlike a ready-pose search, require native clearance at BOTH a raised waiting
pose and cut-entry pose. A fresh launch still checks the complete moving path,
floor height, bilateral grasp and changed plant; no endpoint implies a path.
"""
from copy import copy
import time
import numpy as np


def waiting_poses(entry,direction,normal,expanded=False):
    """Bounded INITIAL wrist proposals, not collision-free approach paths."""
    if type(expanded) is not bool:raise ValueError('Explicit waiting-pose search option required')
    offsets=[np.array([0.,0.,.03])]
    if expanded:
        edge=np.cross(normal,direction)
        offsets += [np.array([0.,0.,z]) for z in (.01,.06,.12)]
        offsets += [np.array([0.,0.,.03])+sign*distance*axis
                    for distance in (.04,.08) for axis in (normal,edge) for sign in (-1.,1.)]
    for offset in offsets:
        pose=entry.copy();pose[:3,3]+=offset
        yield pose,offset


def stations(original, centre):
    from scipy.spatial.transform import Rotation
    original=np.asarray(original,float);centre=np.asarray(centre,float)
    if (original.shape!=(4,4) or centre.shape!=(3,)
            or not np.isfinite(original).all() or not np.isfinite(centre).all()):
        raise ValueError('Finite station matrix and target centre required')
    angle=np.arctan2(*(original[:2,3]-centre[:2])[::-1])
    # Alternate sides and radii early; do not spend the budget only at one side.
    for delta in (0.,15.,-15.,30.,-30.,60.,-60.,90.,-90.,120.,-120.,150.,-150.,180.):
        for radius in (.35,.4,.3,.45,.55,.65,.75):
            for yaw_bias in (0.,-15.,15.):
                theta=angle+np.radians(delta)
                base=original.copy()
                base[:2,3]=centre[:2]+radius*np.array([np.cos(theta),np.sin(theta)])
                base[:3,:3]=Rotation.from_euler('z',theta+np.pi+np.radians(yaw_bias)).as_matrix()
                yield base,dict(orbit_degrees=delta,radius_m=radius,yaw_bias_degrees=yaw_bias)


def search(robot,backend,guard):
    from .downward_cut import vertical_cut_frame
    from .blade_aim import edge_centre
    from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES
    priority=getattr(robot,'cut_priority',None)
    priority_supplied=priority is not None
    if priority is None:priority=dict(normal_sign=-1,tilt=0.,wing_m=0.)
    centre,axis=robot.seam(robot.rig.rest_frames)
    frame=vertical_cut_frame(axis,priority['normal_sign'],priority['tilt'])
    if frame is None:raise ValueError('Current anatomy does not permit the requested downward cut')
    d,normal=frame
    aim=edge_centre(centre,axis,robot.blade_axial_aim_offset_m)
    expanded_waiting=getattr(robot,'station_waiting_search',False)
    entry_frames={}
    for sign in (priority['normal_sign'],-priority['normal_sign']):
        candidate_frame=vertical_cut_frame(axis,sign,priority['tilt'])
        if candidate_frame is None:continue
        direction,plane_normal=candidate_frame
        entry=robot.knife.wrist_for_edge(aim+robot.stroke_offsets[0]*direction,
            direction,plane_normal,priority['wing_m'])
        entry_frames[sign]=(entry,list(waiting_poses(entry,direction,plane_normal,expanded_waiting)))
    poses=[robot.kin.forward('left',q,robot.base) for q in robot.path_q]
    ready=np.array([SDK_READY_POSE_DEGREES[f'right_arm_{i}'] for i in range(7)])
    right_seeds=[robot.right]
    if not np.array_equal(robot.right,ready):right_seeds.append(ready)
    rows=[];proposal=None;began=time.monotonic();expired=False;query_limited=False
    def native_check(candidate,left,right):
        nonlocal expired,query_limited
        world=candidate.body_world(left,right)
        reserve=getattr(backend,'can_check_with_final_controls',None)
        if reserve is not None and not reserve(world,robot.self_screen.shapes):
            expired=True;query_limited=True;return None
        return backend.check(world,robot.self_screen.shapes)
    def check_time():
        guard()
        return time.monotonic()-began<45.
    choices=stations(robot.base,centre)
    reference=getattr(robot,'station_reference_search',False)
    if reference:
        from itertools import chain
        from .station_reference import local_stations
        choices=chain(local_stations(robot.base),choices)
    else:
        from itertools import chain
        # A new explicit/IK-derived initial station may lie between coarse
        # orbit radii. Do not discard it before checking its two endpoints.
        choices=chain(((robot.base.copy(),dict(original_station=True)),),choices)
    for base,meta in choices:
        if not check_time():expired=True;break
        candidate=copy(robot);candidate.base=base
        row=dict(**meta,native_startup_clear=False,native_cut_entry_clear=False,
                 left_self_path_clear=False,motion_authorized=False)
        rows.append(row)
        if not .25<=np.linalg.norm(base[:2,3]-robot.rig.chain_world[0,:2])<=1.:
            row['rejection']='station_distance_outside_original_bounds';continue
        parked=getattr(robot,'park_left_ready',False)
        if parked:
            if getattr(robot,'cut_strategy',None)!='right_only':
                raise ValueError('Independent left park requires right-only strategy')
            lq=np.asarray(robot.initial_q).copy()
        else:
            left=robot.kin.solve_pose('left',poses[0],robot.initial_q,base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            if not left.succeeded:row['rejection']='left_initial_ik';continue
            lq=np.asarray(left.joint_degrees)
        row['right_attempts']=[]
        # Either physical side of the shaft may offer access. Both still keep
        # arc-up and world-downward motion, and both receive full native checks.
        attempts=((sign,entry,waiting,offset,seed)
            for sign,(entry,poses_for_entry) in entry_frames.items()
            for waiting,offset in poses_for_entry for seed in right_seeds)
        for sign,entry,waiting,offset,seed in attempts:
            if not check_time():expired=True;break
            attempt=dict(cut_plane_normal_sign=sign,waiting_offset_world_m=offset.tolist())
            row['right_attempts'].append(attempt)
            solved=robot.kin.solve_pose('right',waiting,seed,base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            if not solved.succeeded:attempt['rejection']='raised_waiting_ik';continue
            rq=np.asarray(solved.joint_degrees)
            if not candidate.check_self(lq,rq)['passed']:
                attempt['rejection']='waiting_self_collision';continue
            native=native_check(candidate,lq,rq)
            if native is None:attempt['rejection']='reserved_final_native_controls';break
            attempt['native_waiting']=native
            if native['passed'] is not True:continue
            row['native_startup_clear']=True
            endpoint=robot.kin.solve_pose('right',entry,rq,base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            if not endpoint.succeeded:attempt['rejection']='entry_ik';continue
            eq=np.asarray(endpoint.joint_degrees)
            from .downward_cut import arm_extension
            extension=arm_extension(candidate.body_world(lq,eq))
            attempt['entry_right_arm_extension']=extension
            if not .8<=extension<=.98:
                attempt['rejection']='folded_or_fully_extended_cut_entry';continue
            if not candidate.check_self(lq,eq)['passed']:
                attempt['rejection']='entry_self_collision';continue
            native=native_check(candidate,lq,eq)
            if native is None:attempt['rejection']='reserved_final_native_controls';break
            attempt['native_cut_entry']=native
            if native['passed'] is not True:continue
            row['native_cut_entry_clear']=True;path=[];pq=lq.copy()
            for pose in poses:
                if not check_time():expired=True;break
                if not parked:
                    solved=robot.kin.solve_pose('left',pose,pq,base,
                        maximum_evaluations=200,joint_limit_margin_degrees=3.)
                    if not solved.succeeded:attempt['rejection']='left_path_ik';break
                    pq=np.asarray(solved.joint_degrees)
                if (robot.kin.inter_arm_clearance(pq,rq,base).clearance_m<.01
                        or not candidate.check_self(pq,rq)['passed']):
                    attempt['rejection']='left_path_self_or_interarm';break
                path.append(pq.tolist())
            else:
                row['left_self_path_clear']=True
                proposal=dict(station_pose=[float(base[0,3]),float(base[1,3]),
                    float(np.degrees(np.arctan2(base[1,0],base[0,0])))],
                    left_ik_seed_degrees=lq.tolist(),right_ready_degrees=rq.tolist(),
                    left_path_degrees=path,
                    waiting_offset_world_m=offset.tolist(),
                    cut_frame_family=dict(normal_sign=int(sign),tilt=float(priority['tilt']),
                                          wing_m=float(priority['wing_m'])))
                break
        if proposal is not None or expired:break
    guard()
    return dict(model='frozen_native_two_arm_station_search_v1',search_strategy='cut_frame_orbit_v1',
        candidates=rows,proposed_station=proposal,maximum_candidates=315 if reference else 295,
        reference_local_seed_search=reference,
        expanded_waiting_pose_search=expanded_waiting,
        waiting_pose_candidates_per_frame=12 if expanded_waiting else 1,
        cut_frame_priority_used=priority_supplied,initial_vertical_standoff_m=.03,
        cut_plane_normal_signs=list(entry_frames),
        original_spawn_unchanged=True,plant_or_mounting_changed=False,
        physics_steps=0,motion_authorized=False,relaunch_required=True,
        whole_path_certified=False,base_motion_certified=False,
        cut_entry_checked_with_left='SDK_ready_park' if getattr(robot,'park_left_ready',False) else 'unloaded_pregrasp_not_held_branch',
        proposed_floor_height_requires_fresh_launch=True,budget_exhausted=expired,
        query_budget_reserved_for_final_controls=query_limited,
        wall_seconds=time.monotonic()-began,infeasibility_proof=False)
