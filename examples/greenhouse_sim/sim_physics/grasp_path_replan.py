"""Bounded collision-aware elbow proposals for the unchanged left wrist path.

No motion, contact permission or source edits. The caller must still check the
ENTIRE returned path against the current plant/native scene before commanding.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp_frame import approach_rotation


class GraspPathSelfCollision(RuntimeError):
    """A valid self-screen rejected a nominal path, not a query/model error."""
    def __init__(self,message,*,index):
        super().__init__(message)
        self.path_index=index


def replan(robot,*,wall_limit_s=8.):
    from .redundant_ik import pose_family
    if isinstance(wall_limit_s,(bool,np.bool_)) or not np.isfinite(wall_limit_s) or not 0<wall_limit_s<=8:
        raise ValueError('Bounded grasp replanning budget required')
    start=np.array(robot.start,float,copy=True);goal=np.array(robot.goal,float,copy=True)
    initial=np.array(robot.initial_q,float,copy=True)
    began=time.perf_counter();checks=0;alternatives=0
    lower,upper=robot.kin.arm_limits_degrees('left')
    def deadline():
        if time.perf_counter()-began>wall_limit_s:raise RuntimeError('Grasp elbow replanning wall budget exhausted')
    def desired(fraction):
        frame=start.copy();frame[:3,3]+=fraction*(goal[:3,3]-start[:3,3])
        frame[:3,:3]=approach_rotation(start[:3,:3],goal[:3,:3],float(fraction))
        return frame
    def valid(q,fraction):
        nonlocal checks
        deadline();checks+=1
        if (q.shape!=(7,) or not np.isfinite(q).all()
                or np.any(q<=lower) or np.any(q>=upper)):return False
        target=desired(fraction);actual=robot.kin.forward('left',q,robot.base)
        if (np.linalg.norm(actual[:3,3]-target[:3,3])>.0005
                or np.linalg.norm(Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).as_rotvec())>.005):return False
        return (robot.kin.inter_arm_clearance(q,robot.right,robot.base).clearance_m>=.01
                and robot.check_self(q,robot.right)['passed'] is True)
    if not valid(initial,0.):raise RuntimeError('Current initial left configuration cannot seed a clear wrist path')
    fractions=[0.];path=[initial]
    minimum=robot.kin.inter_arm_clearance(initial,robot.right,robot.base).clearance_m
    for fraction in np.linspace(0,1.15,47)[1:]:
        deadline();target=desired(fraction);previous=path[-1]
        solution=robot.kin.solve_pose('left',target,previous,robot.base,maximum_evaluations=250,
            joint_limit_margin_degrees=3.)
        if not solution.succeeded:raise RuntimeError('Reobserved grasp wrist path IK failed')
        seed=np.asarray(solution.joint_degrees,float)
        def proposals():
            yield seed
            # Nearby self-motion only, not an unchecked jump to another branch.
            for member in pose_family(robot.kin,'left',target,seed,robot.base,
                    steps_per_direction=8,step_degrees=.5,joint_limit_margin_degrees=3.):
                yield np.asarray(member.joint_degrees,float)
        accepted=False
        for index,q in enumerate(proposals()):
            if index:alternatives+=1
            deadline()
            if not valid(q,fraction):continue
            count=max(3,int(np.ceil(np.max(abs(q-previous))/.5))+1)
            if count>65:continue  # Do not invent a distant arm-branch transition.
            rows=np.linspace(previous,q,count)[1:]
            ff=np.linspace(fractions[-1],fraction,count)[1:]
            if not all(valid(row,f) for row,f in zip(rows,ff)):continue
            for row in rows:
                minimum=min(minimum,robot.kin.inter_arm_clearance(row,robot.right,robot.base).clearance_m)
            path.extend(rows);fractions.extend(ff);accepted=True;break
        if not accepted:raise RuntimeError('No locally continuous collision-clear left elbow path at '+str(fraction))
    return np.asarray(fractions),np.asarray(path),dict(
        model='bounded_same_wrist_grasp_elbow_replan_v1',self_and_interarm_checked=True,
        maximum_joint_sample_step_degrees=.5,maximum_wrist_position_error_m=.0005,
        maximum_wrist_orientation_error_rad=.005,checks=checks,alternative_poses=alternatives,
        path_samples=len(path),minimum_interarm_m=minimum,wall_seconds=time.perf_counter()-began,
        start_joint_vector_preserved=True,desired_wrist_path_changed=False,
        current_plant_screen_required=True,whole_scene_path_certified=False,
        native_contact_verified=False,motion_authorized=False)
