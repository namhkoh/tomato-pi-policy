"""Cheap pose preselection, never native depth/visibility or label approval."""
from copy import deepcopy
import numpy as np
from ..capture_contract import project, fingerprint
from ..capture_sensor import HIRES_RESOLUTION
from ..native_clear_contract import POLICY as CLEAR_POLICY
from ..dataset_review import require

POLICY = dict(version='greenhouse.native_projected_sampling.v1',
    resolution=list(HIRES_RESOLUTION),minimum_interval_px=CLEAR_POLICY['minimum_interval_px'],
    minimum_estimated_diameter_px=CLEAR_POLICY['minimum_width_proxy_px'],
    scope='static_pose_selection_only_not_native_visibility_or_annotation')

def projection_screen(calibration, interval_world_m, radius_m):
    """Check the complete proposed interval, preserving bends and clipping.

    Radius is the existing native collector's NOMINAL petiole radius. Projected
    diameter is only its pinhole legibility estimate, not the actual mask width.
    A passed screen means 'worth rendering'; no depth values, visibility labels,
    model answers or physical actions are generated.
    """
    cal=deepcopy(calibration)
    require(cal.get('resolution')==list(HIRES_RESOLUTION) and cal.get('crop_resize') is None
            and cal.get('depth_convention')=='optical_axis_z_metres_not_ray_range',
            'Exact native optical-Z camera calibration required')
    points=np.asarray(interval_world_m,float)
    require(points.ndim==2 and points.shape[1]==3 and 2<=len(points)<=4096
            and np.isfinite(points).all(), 'Finite complete proposed cut interval required')
    require(type(radius_m) in (int,float) and np.isfinite(radius_m) and radius_m>0,
            'Positive finite nominal petiole radius required')
    matrix=np.asarray(cal['camera_to_world_usd_row_vectors'],float)
    require(matrix.shape==(4,4) and np.isfinite(matrix).all()
            and np.allclose(matrix[:,3],[0,0,0,1],atol=1e-9,rtol=0)
            and np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6,rtol=0)
            and np.isclose(np.linalg.det(matrix[:3,:3]),1,atol=1e-6,rtol=0),
            'Rigid native camera transform required')
    k=np.asarray(cal['intrinsics'],float)
    require(k.shape==(3,3) and np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0
            and np.allclose(k[2],[0,0,1],atol=1e-9,rtol=0), 'Finite pinhole intrinsics required')
    limits=np.asarray(cal['clipping_range_m'],float)
    require(limits.shape==(2,) and np.isfinite(limits).all() and 0<limits[0]<limits[1],
            'Valid native clipping range required')
    projected=project(points,cal)
    in_frame=all(p['projection_status']=='in_frame' for p in projected)
    uv=[p['pixel_xy'] for p in projected]
    length=float(np.linalg.norm(np.diff(uv,axis=0),axis=1).sum()) if all(p is not None for p in uv) else None
    nominal_z=projected[0]['camera_optical_xyz_m'][2]
    diameter=float(2*radius_m*min(k[0,0],k[1,1])/nominal_z) if nominal_z>0 else None
    reasons=[]
    if not in_frame:reasons.append('interval_outside_native_frame_or_clipping')
    if length is None or length<POLICY['minimum_interval_px']:reasons.append('projected_interval_below_legibility_threshold')
    if diameter is None or diameter<POLICY['minimum_estimated_diameter_px']:reasons.append('estimated_diameter_below_legibility_threshold')
    return dict(policy=deepcopy(POLICY),policy_sha256=fingerprint(POLICY),
        camera_sha256=fingerprint(cal),geometry_sha256=fingerprint(dict(interval=points.tolist(),radius_m=radius_m)),
        worth_rendering=not reasons,reasons=reasons,projected_interval=projected,
        projected_interval_length_px=length,estimated_nominal_diameter_px=diameter,
        all_interval_samples_in_frame=in_frame,estimated_diameter_is_not_native_mask_width=True,
        native_depth_computed=False,native_visibility_verified=False,training_approved=False,
        physical_execution_approved=False)
