"""Frozen native grasp-orientation proposals; never grasp/cut authorization.

Keep the exact anatomical material point, base, right arm and original assets.
Only propose a different initial left wrist orientation and its approach. The
caller owns scene/actor controls and revokes proposals if those controls fail.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp_frame import pitch_for_pad_span


def orientations(robot):
    if (getattr(robot,'park_left_ready',False) or robot.grasp_skew!=0
            or robot.approach_tilt!=0 or robot.grasp_roll not in (0,180)):
        raise ValueError('Grasp search requires unparked zero-skew/tilt original parallel jaws')
    # Undo the constructor's pitch/roll, then turn around the same shaft axis.
    original=pitch_for_pad_span(robot.goal[:3,:3],-robot.grasp_pitch)
    if robot.grasp_roll==180:original[:,:2]*=-1
    axis=original[:,1];point=np.array(robot.grasp_point(robot.rig.rest_frames),copy=True)
    if point.shape!=(3,) or not np.isfinite(point).all():
        raise ValueError('Finite original material grasp point required')
    for phi in (0.,-30.,30.,-60.,60.,-90.,90.,180.):
        vector=Rotation.from_rotvec(axis*np.radians(phi)).apply(original[:,2])
        for pitch in (60.,-60.,40.,-40.,0.,20.,-20.):
            for roll in (robot.grasp_roll,180-robot.grasp_roll):
                r=np.column_stack([np.cross(axis,vector),axis,vector])
                if roll==180:r[:,:2]*=-1
                goal=np.eye(4);goal[:3,:3]=pitch_for_pad_span(r,pitch)
                goal[:3,3]=point+robot.grasp_depth*goal[:3,2]
                if np.allclose(goal,robot.goal,rtol=0,atol=1e-12):continue
                start=goal.copy();start[:3,3]+=robot.approach_distance*goal[:3,2]
                yield start,goal,dict(approach_vector=vector.tolist(),
                    grasp_pitch=pitch,grasp_roll=roll,shaft_rotation_degrees=phi)


def search(robot,backend,guard):
    from .grasp_target import finger_seam_clearance
    from .cut_station_orbit import left_seed_proposals
    began=time.monotonic();rows=[];proposals=[];expired=False;reserved=False
    centre,axis=robot.seam(robot.rig.rest_frames)
    seeds=left_seed_proposals(robot.initial_q,True)
    def time_ok():
        guard()
        return time.monotonic()-began<45.
    for start,goal,meta in orientations(robot):
        if not time_ok():expired=True;break
        row=dict(**meta,native_startup_clear=False,left_self_path_clear=False,
                 motion_authorized=False);rows.append(row)
        try:
            row['finger_cut_clearance']=finger_seam_clearance(robot.stage,robot.root,goal,centre,axis)
        except ValueError as exc:
            row.update(rejection='finger_cut_clearance',detail=str(exc));continue
        for seed_index,seed in enumerate(seeds):
            if not time_ok():expired=True;break
            solved=robot.kin.solve_pose('left',start,seed,robot.base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            if not solved.succeeded:row['rejection']='left_initial_ik';continue
            initial=np.asarray(solved.joint_degrees)
            if (robot.kin.inter_arm_clearance(initial,robot.right,robot.base).clearance_m<.01
                    or not robot.check_self(initial,robot.right)['passed']):
                row['rejection']='initial_self_or_interarm';continue
            world=robot.body_world(initial,robot.right)
            reserve=getattr(backend,'can_check_with_final_controls',None)
            if reserve is not None and not reserve(world,robot.self_screen.shapes):
                expired=True;reserved=True;row['rejection']='reserved_final_native_controls';break
            native=backend.check(world,robot.self_screen.shapes);guard()
            row['native_startup_geometry']=native
            if native['passed'] is not True:row['rejection']='native_startup_obstruction';continue
            row['native_startup_clear']=True;pq=initial.copy();path=[]
            for fraction in np.linspace(0.,1.15,47):
                if not time_ok():expired=True;break
                pose=start.copy();pose[:3,3]+=fraction*(goal[:3,3]-start[:3,3])
                solved=robot.kin.solve_pose('left',pose,pq,robot.base,
                    maximum_evaluations=200,joint_limit_margin_degrees=3.)
                if not solved.succeeded:row['rejection']='left_path_ik';break
                pq=np.asarray(solved.joint_degrees)
                if (robot.kin.inter_arm_clearance(pq,robot.right,robot.base).clearance_m<.01
                        or not robot.check_self(pq,robot.right)['passed']):
                    row['rejection']='left_path_self_or_interarm';break
                path.append(pq.tolist())
            else:
                row['left_self_path_clear']=True;row.pop('rejection',None)
                proposal=dict(**meta,left_seed_index=seed_index,
                    left_ik_seed_degrees=initial.tolist(),right_ready_degrees=robot.right.tolist(),
                    left_path_degrees=path,grasp_arc_m=float(robot.arc),
                    station_pose=[float(robot.base[0,3]),float(robot.base[1,3]),
                        float(np.degrees(np.arctan2(robot.base[1,0],robot.base[0,0])))])
                proposals.append(proposal)
                break
            if expired:break
        if len(proposals)>=8 or expired:break
    guard()
    return dict(model='frozen_native_grasp_orientation_search_v1',candidates=rows,
        proposed_grasp=proposals[0] if proposals else None,proposed_grasps=proposals,
        maximum_proposals=8,maximum_orientations=112,maximum_seeds_per_orientation=6,
        budget_exhausted=expired,query_budget_reserved_for_final_controls=reserved,
        wall_seconds=time.monotonic()-began,original_spawn_unchanged=True,
        anatomical_grasp_point_unchanged=True,physics_steps=0,motion_authorized=False,
        whole_path_certified=False,cut_entry_checked_with_left=False,
        grasp_or_cut_verified=False,retention_verified=False,relaunch_required=True)
