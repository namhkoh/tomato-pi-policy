"""Retarget only external diagnostic cameras to the fetched native knife.

The original setup cameras used the initial/neutral wrist, so later close-ups
could show the base instead of the cut. No D405 camera, calibration, robot or
plant transform is changed here. These paused views are not training sensors.
"""
import numpy as np


def viewpoints(edge):
    from .withdrawal_evidence import _pose
    e=_pose(edge);centre=e[:3,3]
    return {
        '/World/RightKnifeMountInspection':(centre+.23*e[:3,2]+.18*e[:3,1]-.12*e[:3,0],centre.copy()),
        '/World/BladePlaneInspection_front':(centre+.22*e[:3,2]+.045*e[:3,1]+.05*e[:3,0],centre.copy()),
        '/World/BladePlaneInspection_back':(centre-.22*e[:3,2]+.045*e[:3,1]+.05*e[:3,0],centre.copy())}


def refresh(fixture):
    from pxr import Gf,UsdGeom
    from .runtime import pose_matrices
    view=fixture.right_palm
    if tuple(view.prim_paths)!=(fixture.root+'/ee_right',) or view.count!=1:
        raise ValueError('Exact single native right-wrist view required for diagnostic cameras')
    frames=pose_matrices(np.array(view.get_transforms(),copy=True))
    if frames.shape!=(1,4,4):raise ValueError('One fetched native right-wrist transform required')
    cameras=viewpoints(fixture.knife.frame(frames[0]))
    # Validate the entire diagnostic-only write set before modifying a camera.
    for path in cameras:
        prim=fixture.stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise ValueError('Existing dedicated knife-inspection cameras required')
    for path,(eye,centre) in cameras.items():
        xf=UsdGeom.Xformable(fixture.stage.GetPrimAtPath(path))
        operations=xf.GetOrderedXformOps()
        if len(operations)!=1 or operations[0].GetOpType()!=UsdGeom.XformOp.TypeTransform:
            raise ValueError('Single existing diagnostic-camera transform required')
    for path,(eye,centre) in cameras.items():
        op=UsdGeom.Xformable(fixture.stage.GetPrimAtPath(path)).GetOrderedXformOps()[0]
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*centre),Gf.Vec3d(0,0,1)).GetInverse())
    return dict(source='fetched_native_right_wrist_not_initial_FK',
        camera_paths=list(cameras),robot_cameras_changed=False,training_eligible=False)
