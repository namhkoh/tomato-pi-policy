"""Reference-preserving static robot snapshots, not navigation or robot commands.

Uses the proven reference body orientation and only <=4 cm outward/lateral base
offsets. Real head joints aim the unchanged mount; floor/joint/native geometry
checks remain mandatory. This does not apply arbitrary target-facing base yaw.
"""
import math
import numpy as np
from .dataset_review import require


def bounded_reference_root(reference, spec, target_world):
    row=np.asarray(reference['robot_root_to_world_usd_row_vectors'],float)
    target=np.asarray(target_world,float)
    require(row.shape==(4,4) and target.shape==(3,) and np.isfinite(row).all()
            and np.isfinite(target).all(),'Finite reference pose and target required')
    require(np.allclose(row[:3,:3]@row[:3,:3].T,np.eye(3),atol=1e-6)
            and np.allclose(row[:3,3],0,atol=1e-9) and row[3,3]==1
            and np.allclose(row[2,:3],[0,0,1],atol=1e-9),'Rigid Z-up source base required')
    require(reference.get('visual_bound_screen',{}).get('passed') is True,
            'Previously screened reference robot pose required')
    require(abs(row[3,0]-target[0])>.02,'Ambiguous reference aisle side')
    side=1 if row[3,0]>target[0] else -1
    x,y=float(spec['root_x_m']),float(target[1]+spec['y_offset_m'])
    outward=(x-row[3,0])*side;lateral=y-row[3,1]
    require(np.isfinite([x,y,outward,lateral,spec['root_yaw_degrees']]).all()
            and -1e-9<=outward<=.04+1e-9 and abs(lateral)<=.04+1e-9,'Snapshot exceeds bounded reference offsets')
    heading=math.degrees(math.atan2(row[0,1],row[0,0]))
    require(abs((spec['root_yaw_degrees']-heading+180)%360-180)<1e-8
            and spec['base_heading_preserved'] is True,'Source body orientation must be preserved')
    result=row.T.copy();result[0,3]=x;result[1,3]=y
    return result


def set_reference_snapshot(stage,robot,reference,spec,target_world):
    from pxr import Usd,UsdGeom
    from greenhouse_sim import robot_hardware,robot_kinematics
    from .floor_alignment import align_robot_to_floor,PACKAGE_FLOOR
    from .capture_scene import calibration,mounted_camera_to_head,plan_head_pose
    matrix=bounded_reference_root(reference,spec,target_world)
    mount=mounted_camera_to_head(stage)
    require(np.allclose(mount,reference['camera_to_head_column_vectors'],atol=1e-9,rtol=0),
            'Source camera mount changed')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot['root']),matrix[:3,:3],matrix[:3,3])
    floor=align_robot_to_floor(stage,robot['root'],PACKAGE_FLOOR)
    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot['root']))).T
    model=robot_kinematics.Rby1Kinematics()
    pose,error=plan_head_pose(model,reference['joint_degrees'],matrix,mount,target_world,
                             calibration(stage)['intrinsics'],spec['desired_pixel_xy'])
    require(all(pose[k]==v for k,v in reference['joint_degrees'].items() if k not in ('head_0','head_1')),
            'Arm or torso joint changed')
    links=model.all_link_transforms(pose)
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for link,transform in links.items():
            prim=stage.GetPrimAtPath(robot['root']+'/'+link)
            require(bool(prim),'Missing robot link')
            robot_hardware._set_transform(prim,transform[:3,:3],transform[:3,3])
    require(np.allclose(mounted_camera_to_head(stage),mount,atol=1e-9,rtol=0),'Camera mount changed while aiming')
    return dict(joint_degrees=pose,robot_root_to_world_usd_row_vectors=matrix.T.tolist(),
        camera_to_head_column_vectors=mount.tolist(),framing_error_degrees=error,
        desired_cut_pixel_xy=spec['desired_pixel_xy'],floor_alignment=floor,
        pose_sampling='bounded_reference_offsets_and_real_head_joints.v1',
        source_body_orientation_preserved=True,arm_and_torso_joints_preserved=True,
        joint_limits_checked=True,whole_robot_collision_checked=False,
        motion_between_snapshots_validated=False,arm_reachability='not_tested')
