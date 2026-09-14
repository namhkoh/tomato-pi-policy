"""Bounded frozen-scene neutral/pregrasp/cut-entry proposals, never motion.

Unlike single-arm station IK, test SDK-ready whole-body clearance FIRST.
Keep all source colliders, actor controls, joint limits and the frozen epoch.
The output is not a path, settled-scene, physical grasp or cut certificate.
"""
from copy import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read_candidates(path, *, plant, target):
    raw=Path(path).read_bytes()
    if len(raw)>100_000:raise ValueError('Bounded neutral station candidate file required')
    data=json.loads(raw)
    if (not isinstance(data,dict) or set(data)!={'schema','plant','target','candidates'}
            or data['schema']!='neutral_station_candidates_v1'
            or data['plant']!=plant or data['target']!=target
            or not isinstance(data['candidates'],list) or not 1<=len(data['candidates'])<=64):
        raise ValueError('Same-target bounded neutral station candidates required')
    for row in data['candidates']:
        if not isinstance(row,dict) or set(row)-{'station_pose','left_ik_seed_degrees','right_ik_seed_degrees'} or 'station_pose' not in row:
            raise ValueError('Only station and optional IK seeds are proposals')
        for key,values in row.items():
            n=3 if key=='station_pose' else 7
            if (not isinstance(values,list) or len(values)!=n
                    or any(type(v) not in (int,float) for v in values) or not np.isfinite(values).all()):
                raise ValueError('Finite station/IK proposal required')
        if abs(row['station_pose'][2])>180:raise ValueError('Station yaw must be within180 degrees')
    return data['candidates'],dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest())


def search(robot,backend,guard):
    from scipy.spatial.transform import Rotation
    from .neutral_ready import ready_arms,TRANSIT_MARGIN_M
    from .downward_cut import vertical_cut_frame,arm_extension
    from .blade_aim import edge_centre
    ready=ready_arms();priority=robot.cut_priority
    if priority is None:raise ValueError('Explicit downward cut-frame family required')
    centre,axis=robot.seam(robot.rig.rest_frames)
    frame=vertical_cut_frame(axis,priority['normal_sign'],priority['tilt'])
    if frame is None:raise ValueError('Current shaft does not permit proposed downward frame')
    d,normal=frame;aim=edge_centre(centre,axis,robot.blade_axial_aim_offset_m)
    entry=robot.knife.wrist_for_edge(aim+robot.stroke_offsets[0]*d,d,normal,priority['wing_m'])
    pregrasp=robot.kin.forward('left',robot.path_q[0],robot.base)
    grasp=robot.goal.copy()
    rows=[];proposal=None;began=time.monotonic();limited=False
    def native(candidate,left,right,margin):
        nonlocal limited
        guard();world=candidate.body_world(left,right)
        if not backend.can_check_with_final_controls(world,robot.self_screen.shapes,margin=margin):
            limited=True;return None
        result=backend.check(world,robot.self_screen.shapes,margin=margin);guard()
        return result
    def solve(candidate,side,goal,seed):
        lo,hi=map(np.asarray,robot.kin.arm_limits_degrees(side));seed=np.asarray(seed,float)
        if np.any(seed<lo) or np.any(seed>hi):return None
        result=robot.kin.solve_pose(side,goal,seed,candidate.base,
            maximum_evaluations=250,joint_limit_margin_degrees=3.)
        guard()
        return np.asarray(result.joint_degrees) if result.succeeded else None
    for spec in robot.neutral_station_candidates:
        guard()
        if time.monotonic()-began>=40.:limited=True;break
        row=dict(station_pose=list(spec['station_pose']),motion_authorized=False);rows.append(row)
        candidate=copy(robot);candidate.base=robot.base.copy()
        candidate.base[:2,3]=spec['station_pose'][:2]
        candidate.base[:3,:3]=Rotation.from_euler('z',spec['station_pose'][2],degrees=True).as_matrix()
        if not .25<=np.linalg.norm(candidate.base[:2,3]-robot.rig.chain_world[0,:2])<=1.:
            row['rejection']='station_distance';continue
        if not candidate.check_self(ready[:7],ready[7:])['passed']:
            row['rejection']='neutral_self';continue
        checked=native(candidate,ready[:7],ready[7:],TRANSIT_MARGIN_M)
        row['neutral_native']=checked
        if limited:break
        if checked['passed'] is not True:row['rejection']='neutral_scene';continue
        lq=solve(candidate,'left',pregrasp,spec.get('left_ik_seed_degrees',robot.initial_q))
        if lq is None:row['rejection']='pregrasp_ik';continue
        if not candidate.check_self(lq,ready[7:])['passed']:
            row['rejection']='pregrasp_self';continue
        checked=native(candidate,lq,ready[7:],TRANSIT_MARGIN_M);row['pregrasp_native']=checked
        if limited:break
        if checked['passed'] is not True:row['rejection']='pregrasp_scene';continue
        lg=solve(candidate,'left',grasp,lq)
        rq=solve(candidate,'right',entry,spec.get('right_ik_seed_degrees',ready[7:]))
        row.update(left_grasp_ik=lg is not None,right_entry_ik=rq is not None)
        if lg is None or rq is None:row['rejection']='grasp_or_entry_ik';continue
        clearance=robot.kin.inter_arm_clearance(lg,rq,candidate.base).clearance_m
        extension=arm_extension(candidate.body_world(lg,rq))
        row.update(interarm_clearance_m=float(clearance),right_extension=float(extension))
        if clearance<.01 or not .8<=extension<=.98 or not candidate.check_self(lg,rq)['passed']:
            row['rejection']='grasp_entry_self_or_extension';continue
        # No grasp-contact exception in the frozen native query: leave left
        # at its clear pregrasp while checking the complete right entry tool.
        # Grasped geometry, full stroke and moving transit remain unqualified.
        checked=native(candidate,lq,rq,.001);row['entry_native_left_pregrasp']=checked
        if limited:break
        if checked['passed'] is not True:row['rejection']='entry_scene';continue
        proposal=dict(station_pose=list(spec['station_pose']),left_ik_seed_degrees=lq.tolist(),
            right_ready_degrees=ready[7:].tolist(),right_entry_seed_degrees=rq.tolist(),
            cut_frame_family=dict(tilt=priority['tilt'],normal_sign=priority['normal_sign'],wing_m=priority['wing_m']))
        break
    guard()
    return dict(model='frozen_native_neutral_station_search_v1',candidates=rows,
        candidate_source=robot.neutral_station_candidate_source,proposed_station=proposal,
        maximum_candidates=len(robot.neutral_station_candidates),budget_limited=limited,
        original_spawn_unchanged=True,physics_steps=0,motion_authorized=False,relaunch_required=True,
        whole_path_certified=False,grasp_or_cut_verified=False,settled_scene_verified=False,
        source_geometry_or_contacts_changed=False)
