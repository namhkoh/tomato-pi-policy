"""Bounded same-wrist elbow proposals in a frozen native scene, never motion.

All source/robot native controls remain owned by native_startup_screen. A
candidate does not repair the currently authored colliding spawn: save it and
relaunch, then repeat startup, whole approach, grasp and cut qualification.
"""
import time
import numpy as np


def search(robot,backend,guard):
    if getattr(robot,'startup_grasp_search',False):
        from .grasp_orientation_search import search as grasp_search
        return grasp_search(robot,backend,guard)
    if getattr(robot,'startup_station_search',False):
        from .startup_station_search import search as station_search
        return station_search(robot,backend,guard)
    if getattr(robot,'startup_heading_search',False):return heading_start_search(robot,backend,guard)
    if getattr(robot,'startup_approach_search',False):return approach_start_search(robot,backend,guard)
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


def waiting_target_screen(robot):
    """Conservative rest-target clearance, NOT a settled-scene certificate.

    Keep the whole right arm/tool away from the intended compliant target
    while the left arm approaches. No contact exceptions apply while waiting.
    A 5 mm proposal margin does not bound deformation; live reobservation and
    all native contact guards must still pass on a fresh physical launch.
    """
    from .held_plant_screen import HeldPlantScreen
    screen=HeldPlantScreen(robot.rig,robot.self_screen.shapes,robot.knife.collider)
    screen.snapshot(np.array(robot.rig.rest_frames,float,copy=True))
    return screen


def approach_start_search(robot,backend,guard):
    """Bounded higher/lateral waiting poses; no teleport or path authorization.

Keep the corrected blade orientation. These are proposed new INITIAL poses,
not commands on the current running robot. Every original dense left self-path
and complete native robot/scene startup screen must pass. A subsequent launch
must still plan and verify the entire right approach and cutting sequence.
"""
    initial=np.array(robot.right,float,copy=True)
    base_pose=robot.kin.forward('right',initial,robot.base)
    offsets=[(0.,0.,z) for z in (.01,.02,.03,.05)]
    offsets += [(x,y,z) for z in (.01,.03) for x,y in
                ((.03,0.),(-.03,0.),(0.,.03),(0.,-.03),(.05,0.),(0.,-.05))]
    # A 10--50 mm perturbation can leave the entire forearm in the SAME
    # neighboring leaf. Include visibly withdrawn waiting poses; these still
    # must clear every original shape and are NEVER commands on a live robot.
    offsets += [(0.,0.,z) for z in (.08,.12,.16,.20)]
    offsets += [(x,y,.06) for x,y in ((.10,0.),(-.10,0.),(0.,.10),(0.,-.10))]
    # Refine around native-clear coarse stations if fewer than eight choices
    # survive. Coarse IK can miss narrow safe waiting regions between leaves.
    # Candidates remain initial poses, never a live retreat or a sag prediction.
    coarse_count=len(offsets);refined=False
    started=time.monotonic();rows=[];proposals=[]
    guard();target_screen=waiting_target_screen(robot);guard()
    index=0
    while True:
        if index==coarse_count and not refined:
            refined=True;seen={tuple(np.round(o,9)) for o in offsets}
            for proposal in list(proposals):
                centre=np.array(proposal['wrist_offset_world_m'])
                for distance in (.01,.02):
                    for delta in np.r_[np.eye(3),-np.eye(3)]*distance:
                        value=tuple(np.round(centre+delta,9))
                        if value not in seen:offsets.append(value);seen.add(value)
        if index>=len(offsets):break
        offset=offsets[index];index+=1
        guard()
        if time.monotonic()-started>=30:break
        desired=base_pose.copy();desired[:3,3]+=offset
        solved=robot.kin.solve_pose('right',desired,initial,robot.base,
            maximum_evaluations=200,joint_limit_margin_degrees=3.)
        row=dict(wrist_offset_world_m=list(offset),ik_succeeded=bool(solved.succeeded),
            native_startup_clear=False,left_self_path_clear=False,motion_authorized=False)
        rows.append(row)
        if not solved.succeeded:continue
        q=np.asarray(solved.joint_degrees,float);row['right_ready_degrees']=q.tolist()
        for left in robot.path_q:
            guard()
            clearance=robot.kin.inter_arm_clearance(left,q,robot.base).clearance_m
            check=robot.check_self(left,q)
            if clearance<.01 or not check['passed']:
                row.update(rejection='left_self_path',interarm_m=float(clearance),self_screen=check);break
        else:
            row['left_self_path_clear']=True
            world=robot.body_world(robot.initial_q,q)
            target_clear=target_screen.check(world,margin=.005)
            row['rest_target_clearance']=dict(passed=bool(target_clear),margin_m=.005,
                failure=target_screen.last_failure,native_contact_verified=False,
                settled_scene_checked=False)
            if not target_clear:row['rejection']='right_waiting_rest_target_margin'
            else:
                guard()
                geometry=backend.check(world,robot.self_screen.shapes)
                row['native_startup_geometry']=geometry;row['native_startup_clear']=geometry['passed'] is True
                if geometry['passed']:
                    proposals.append(dict(right_ready_degrees=q.tolist(),wrist_offset_world_m=list(offset)))
                    if len(proposals)>=8:break
    guard()
    first=proposals[0] if proposals else {}
    return dict(model='frozen_native_approach_start_search_v2',candidates=rows,
        proposed_right_ready_degrees=first.get('right_ready_degrees'),
        proposed_wrist_offset_world_m=first.get('wrist_offset_world_m'),
        proposed_waiting_poses=proposals,maximum_proposals=8,
        rest_target_margin_m=.005,settled_scene_checked=False,
        maximum_candidates=len(offsets),maximum_translation_m=max(float(np.linalg.norm(o)) for o in offsets),
        maximum_candidate_bound=120,coarse_candidates=coarse_count,local_refinement_added=refined,
        original_spawn_unchanged=True,orientation_changed=False,relaunch_required=True,
        physics_steps=0,motion_authorized=False,whole_path_certified=False,grasp_or_cut_verified=False)


def heading_start_search(robot,backend,guard):
    """Different camera-aligned knife headings at a fixed edge station.

The mounted assembly is NEVER edited. Rotate the entire proposed wrist/tool
around world up, preserving arc-up; the later approach must independently
align with the stem. This can place the forearm on another side of a leaf
where same-wrist elbow/translation searches cannot. No live pose writes.
"""
    from scipy.spatial.transform import Rotation
    from .blade_contacts import CROSSBAR_EDGE
    if robot.knife.edge_mode!=CROSSBAR_EDGE:raise ValueError('Actual source crossbar required')
    initial=np.array(robot.right,float,copy=True)
    wrist=robot.kin.forward('right',initial,robot.base);edge=robot.knife.frame(wrist)
    if robot.knife.arc_up(wrist)[2]<np.cos(np.radians(5)):
        raise ValueError('Initial source arc must already be up; no mounting repair')
    low,high=robot.kin.arm_limits_degrees('right');rng=np.random.default_rng(288)
    started=time.monotonic();rows=[];candidate=None;selected=None
    for height in (.03,.08):
        for yaw in (180.,90.,-90.,45.,-45.):
            target=edge.copy();target[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@edge[:3,:3]
            target[2,3]+=height;desired=target@np.linalg.inv(robot.knife.local)
            for seed in (initial,rng.uniform(np.asarray(low)+10,np.asarray(high)-10)):
                guard()
                if time.monotonic()-started>=30:break
                solved=robot.kin.solve_pose('right',desired,seed,robot.base,
                    maximum_evaluations=200,joint_limit_margin_degrees=3.)
                row=dict(tool_heading_change_degrees=yaw,edge_height_offset_m=height,
                    ik_succeeded=bool(solved.succeeded),native_startup_clear=False,motion_authorized=False)
                rows.append(row)
                if not solved.succeeded:continue
                q=np.asarray(solved.joint_degrees,float);row['right_ready_degrees']=q.tolist()
                for left in robot.path_q:
                    guard();arm=robot.kin.inter_arm_clearance(left,q,robot.base).clearance_m
                    check=robot.check_self(left,q)
                    if arm<.01 or not check['passed']:
                        row.update(rejection='left_self_path',interarm_m=float(arm),self_screen=check);break
                else:
                    geometry=backend.check(robot.body_world(robot.initial_q,q),robot.self_screen.shapes)
                    row['native_startup_geometry']=geometry;row['native_startup_clear']=geometry['passed'] is True
                    if geometry['passed']:
                        candidate=q.tolist();selected=dict(yaw_degrees=yaw,edge_height_offset_m=height);break
            if candidate is not None or time.monotonic()-started>=30:break
        if candidate is not None or time.monotonic()-started>=30:break
    guard()
    return dict(model='frozen_native_tool_heading_search_v1',candidates=rows,
        proposed_right_ready_degrees=candidate,proposed_heading=selected,maximum_candidates=20,
        original_spawn_unchanged=True,mounting_changed=False,arc_up_preserved=True,
        relaunch_required=True,physics_steps=0,motion_authorized=False,whole_path_certified=False)
