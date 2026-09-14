"""Metric proposals from a frozen post-fetch plant snapshot; no commands.

The source's rest shape is not its gravity-settled shape. Check BOTH for
proposed initial right-arm poses, then require a new native launch. Static
geometry is conservative; no native clearance or motion authority is inferred.
"""
import time
import numpy as np


def offsets():
    values=[np.zeros(3)]
    for distance in (.02,.04,.06):
        values.extend(np.r_[np.eye(3),-np.eye(3)]*distance)
    values.extend(np.array([x,y,z]) for z in (-.04,.04) for x in (-.04,0.,.04)
                  for y in (-.04,0.,.04) if x or y)
    return values


def swept_target_screen(robot,history):
    """Enclose EVERY sampled target pose, not only the two end frames.

    Leaf hulls enclose all sampled native convex vertices; shaft boxes enclose
    their complete sampled capsules. No physical collider is changed. This
    conservative rejection filter is not continuous-time collision proof.
    """
    from .startup_pose_search import waiting_target_screen
    from .held_plant_screen import TriangleIndex
    from scipy.spatial import ConvexHull
    screen=waiting_target_screen(robot)
    poses=np.concatenate((np.asarray(robot.rig.rest_frames)[None],history),axis=0)
    rows=[]
    for path,i,kind,data in screen.local:
        rotations=poses[:,i,:3,:3];translations=poses[:,i,:3,3]
        if kind=='capsule':
            a,b,radius=data
            points=np.einsum('tij,vj->tvi',rotations,np.array([a,b]))+translations[:,None,:]
            low=points.min(axis=(0,1))-radius;high=points.max(axis=(0,1))+radius
            value=((low+high)/2,np.eye(3),(high-low)/2);kind='box'
        else:
            vertices=data[0]
            points=(np.einsum('tij,vj->tvi',rotations,vertices)+translations[:,None,:]).reshape(-1,3)
            hull=ConvexHull(points)
            low=points.min(0);high=points.max(0)
            value=(points[hull.simplices],hull.equations);kind='hull'
        rows.append((path,kind,value,low,high))
    screen.obstacles=rows;screen.lower=np.array([r[3] for r in rows]);screen.upper=np.array([r[4] for r in rows])
    screen.triangle_indices={p:TriangleIndex(data[0]) for p,kind,data,_,_ in rows if kind=='hull'}
    return screen


def search(robot,frames,record,*,step,time_s,physics_hz,guard,history):
    from .shaft_grasp import _poses
    from .held_plant_screen import HeldPlantScreen
    from .startup_pose_search import waiting_target_screen
    from scipy.spatial.transform import Rotation
    sample=record.get('grasp_dynamics',{})
    current=_poses(frames)
    if (type(step) is not int or step<1 or type(physics_hz) is not int or physics_hz<=0
            or not np.isfinite(time_s) or abs(time_s-step/physics_hz)>1e-10
            or record.get('t')!=time_s or record.get('native_guards_passed') is not True
            or record.get('cut') is not False or sample.get('step_id')!=step
            or sample.get('dt_s')!=1/physics_hz
            or tuple(sample.get('body_paths',()))!=tuple(robot.rig.body_paths)
            or sample.get('model')!='grasp_dynamics_post_fetch_telemetry_v1'
            or not np.array_equal(current,np.asarray(sample.get('body_frames_world_m'),float))):
        raise ValueError('Fresh attached guard-accepted native snapshot required')
    cached=robot.held_plant_screen
    if cached.workspace is None:raise ValueError('Original bounded static scene cache required')
    if not isinstance(history,list) or len(history)!=step or step>1024:
        raise ValueError('Complete bounded per-step settling history required')
    if [r[0] for r in history]!=list(range(1,step+1)):
        raise ValueError('No skipped/repeated settling samples allowed')
    observed=np.array([r[1] for r in history],float)
    if observed.shape!=(step,*current.shape) or not np.array_equal(observed[-1],current):
        raise ValueError('Settling history must end at the current exact native frames')
    _poses(observed.reshape(-1,4,4))
    guard();start=time.monotonic()
    rest=waiting_target_screen(robot)
    swept=swept_target_screen(robot,observed);guard()
    settled=HeldPlantScreen(robot.rig,robot.self_screen.shapes,robot.knife.collider)
    settled.static=list(cached.static);settled.static_indices=dict(cached.static_indices)
    settled.workspace=tuple(np.array(v,copy=True) for v in cached.workspace)
    settled.snapshot(current)
    q0=np.array(robot.right,float,copy=True);wrist=robot.kin.forward('right',q0,robot.base)
    low,high=map(np.asarray,robot.kin.arm_limits_degrees('right'))
    rows=[];proposals=[];grid=offsets()
    for offset in grid:
        guard()
        if time.monotonic()-start>=30:break
        goal=wrist.copy();goal[:3,3]+=offset
        solved=robot.kin.solve_pose('right',goal,q0,robot.base,
            maximum_evaluations=200,joint_limit_margin_degrees=3.)
        row=dict(wrist_offset_world_m=offset.tolist(),ik_succeeded=bool(solved.succeeded))
        rows.append(row)
        if not solved.succeeded:continue
        q=np.asarray(solved.joint_degrees,float)
        if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<low+3) or np.any(q>high-3):
            row['rejection']='independent_joint_limits';continue
        actual=robot.kin.forward('right',q,robot.base)
        if (np.linalg.norm(actual[:3,3]-goal[:3,3])>=.0005
                or Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude()>=.005):
            row['rejection']='independent_fk';continue
        for left in robot.path_q:
            guard()
            if (robot.kin.inter_arm_clearance(left,q,robot.base).clearance_m<.01
                    or not robot.check_self(left,q)['passed']):
                row['rejection']='left_self_or_interarm_path';break
        else:
            world=robot.body_world(robot.initial_q,q)
            for label,screen in (('rest_target',rest),('settled_target_and_static',settled),
                                 ('observed_settling_envelope',swept)):
                guard()
                if not screen.check(world,margin=.005):
                    row.update(rejection=label,failure=screen.last_failure);break
            else:
                proposal=dict(right_ready_degrees=q.tolist(),wrist_offset_world_m=offset.tolist())
                row['geometric_checks_passed']=True;proposals.append(proposal)
                if len(proposals)>=8:break
    guard()  # A stale final sample throws; no proposals leave this function.
    return dict(model='settled_waiting_geometric_restart_search_v1',step_id=step,time_s=time_s,
        source_target=robot.rig.source_target,physics_steps_during_search=0,
        candidates=rows,proposed_waiting_poses=proposals,maximum_proposals=8,
        maximum_candidates=len(grid),maximum_wall_s=30,wall_s=time.monotonic()-start,
        rest_and_current_target_margin_m=.005,original_spawn_unchanged=True,
        rest_and_current_target_checked=True,settled_equilibrium_verified=False,
        settling_history_samples=len(history),sampled_settling_envelope_checked=True,
        continuous_time_sweep_certified=False,
        full_left_scene_path_checked=False,native_startup_certified=False,
        motion_authorized=False,grasp_or_cut_verified=False,training_eligible=False,
        relaunch_and_all_native_checks_required=True)
