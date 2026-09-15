"""Wider outward static views, still actual mounted-head robot snapshots.

The previous4cm policy remains unchanged. This opt-in policy permits at most
24cm outward and8cm lateral displacement, never inward/across the target aisle,
and keeps all non-head joints/body heading fixed. Callers MUST run the same
whole-scene native geometry screen. Navigation/dynamics are not certified.
"""
import math
import numpy as np
from ..dataset_review import require

POLICY='outward_24cm_lateral_8cm_static_mounted_head.v1'


def bounded_reference_root(reference,spec,target_world):
    row=np.asarray(reference['robot_root_to_world_usd_row_vectors'],float)
    target=np.asarray(target_world,float)
    require(row.shape==(4,4) and target.shape==(3,) and np.isfinite(row).all()
        and np.isfinite(target).all(),'Finite reference geometry required')
    require(np.allclose(row[:3,:3]@row[:3,:3].T,np.eye(3),atol=1e-6,rtol=0)
        and abs(np.linalg.det(row[:3,:3])-1)<1e-6
        and np.allclose(row[:3,3],0,atol=1e-9,rtol=0) and row[3,3]==1
        and np.allclose(row[2,:3],[0,0,1],atol=1e-9,rtol=0),'Proper rigid Z-up reference required')
    require(reference.get('visual_bound_screen',{}).get('passed') is True,'Screened source reference required')
    require(abs(row[3,0]-target[0])>.02,'Ambiguous aisle side')
    side=1 if row[3,0]>target[0] else -1
    x=float(spec['root_x_m']);y=float(target[1]+spec['y_offset_m'])
    outward=(x-row[3,0])*side;lateral=y-row[3,1]
    require(spec.get('view_policy')==POLICY,'Explicit wider-view policy required')
    require(np.isfinite([x,y,outward,lateral,spec['root_yaw_degrees']]).all()
        and -1e-9<=outward<=.24+1e-9 and abs(lateral)<=.08+1e-9,'Wider static offset outside bounds')
    heading=math.degrees(math.atan2(row[0,1],row[0,0]))
    require(abs((float(spec['root_yaw_degrees'])-heading+180)%360-180)<1e-8
        and spec.get('base_heading_preserved') is True,'Body heading must stay unchanged')
    require(np.isclose(spec['base_displacement_from_reference_m'],math.hypot(outward,lateral),atol=1e-9,rtol=0),
        'Declared displacement does not match actual base')
    result=row.T.copy();result[0,3]=x;result[1,3]=y
    return result


def set_reference_snapshot(stage,robot,reference,spec,target_world):
    from pxr import Usd,UsdGeom
    from greenhouse_sim import robot_hardware,robot_kinematics
    from ..floor_alignment import align_robot_to_floor,PACKAGE_FLOOR
    from ..capture_scene import calibration,mounted_camera_to_head,plan_head_pose
    matrix=bounded_reference_root(reference,spec,target_world)
    mount=mounted_camera_to_head(stage)
    require(np.allclose(mount,reference['camera_to_head_column_vectors'],atol=1e-9,rtol=0),'Camera mount changed')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot['root']),matrix[:3,:3],matrix[:3,3])
    floor=align_robot_to_floor(stage,robot['root'],PACKAGE_FLOOR)
    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot['root']))).T
    model=robot_kinematics.Rby1Kinematics()
    pose,error=plan_head_pose(model,reference['joint_degrees'],matrix,mount,target_world,
        calibration(stage)['intrinsics'],spec['desired_pixel_xy'])
    require(all(pose[k]==v for k,v in reference['joint_degrees'].items() if k not in ('head_0','head_1')),
        'Arm or torso posture changed')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for link,transform in model.all_link_transforms(pose).items():
            prim=stage.GetPrimAtPath(robot['root']+'/'+link);require(bool(prim),'Missing robot link')
            robot_hardware._set_transform(prim,transform[:3,:3],transform[:3,3])
    require(np.allclose(mounted_camera_to_head(stage),mount,atol=1e-9,rtol=0),'Camera mount changed while aiming')
    return dict(joint_degrees=pose,robot_root_to_world_usd_row_vectors=matrix.T.tolist(),
        camera_to_head_column_vectors=mount.tolist(),framing_error_degrees=error,
        desired_cut_pixel_xy=spec['desired_pixel_xy'],floor_alignment=floor,pose_sampling=POLICY,
        source_body_orientation_preserved=True,arm_and_torso_joints_preserved=True,
        joint_limits_checked=True,whole_robot_collision_checked=False,
        motion_between_snapshots_validated=False,arm_reachability='not_tested')
