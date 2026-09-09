"""Static full-greenhouse pilot poses and camera-state checks; no robot commands."""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from greenhouse_sim import robot_hardware, robot_kinematics
from .capture_contract import RESOLUTION, fingerprint, transform_points
from .review_camera import HEAD_CAMERA


def capture_root(scene):
    """Put Kit's root-layer bookkeeping above, never in, the supplied source USD."""
    from pathlib import Path
    source = Path(scene).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Sdf.Layer.CreateAnonymous("greenhouse_capture.usda")
    root.subLayerPaths = [source.as_posix()]
    return root


def calibration(stage):
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError("Capture requires metre-scale Z-up scene")
    camera = UsdGeom.Camera(stage.GetPrimAtPath(HEAD_CAMERA))
    if not camera or camera.GetProjectionAttr().Get() != "perspective":
        raise ValueError("Mounted perspective head camera required")
    focal = float(camera.GetFocalLengthAttr().Get())
    apertures = [float(camera.GetHorizontalApertureAttr().Get()), float(camera.GetVerticalApertureAttr().Get())]
    offsets = [float(camera.GetHorizontalApertureOffsetAttr().Get()), float(camera.GetVerticalApertureOffsetAttr().Get())]
    if not np.isfinite([focal, *apertures, *offsets]).all() or min(focal, *apertures) <= 0:
        raise ValueError("Invalid authored camera parameters")
    width, height = RESOLUTION
    k = [[width*focal/apertures[0], 0, width*(.5-offsets[0]/apertures[0])],
         [0, height*focal/apertures[1], height*(.5+offsets[1]/apertures[1])], [0, 0, 1]]
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
    if not np.allclose(np.asarray(matrix)[:3, :3] @ np.asarray(matrix)[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError("Camera must have a rigid world transform")
    return {"schema_version": "greenhouse.pinhole_calibration.v1", "camera_path": HEAD_CAMERA,
            "resolution": list(RESOLUTION), "intrinsics": k,
            "camera_to_world_usd_row_vectors": [list(row) for row in matrix],
            "optical_frame": "+X right, +Y down, +Z forward",
            "usd_camera_frame": "+X right, +Y up, -Z forward",
            "pixel_convention": "top-left image edge origin; pixel centres at (i+0.5,j+0.5)",
            "clipping_range_m": list(camera.GetClippingRangeAttr().Get()),
            "focal_length_mm": focal, "apertures_mm": apertures, "aperture_offsets_mm": offsets,
            "depth_convention": "optical_axis_z_metres_not_ray_range",
            "sensor_model": "ideal_simulator_pinhole_not_calibrated_real_D405_noise",
            "crop_resize": None}


def mounted_camera_to_head(stage):
    cache = UsdGeom.XformCache()
    camera_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HEAD_CAMERA))
    head_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/RBY1/link_head_2"))
    return np.asarray(camera_world * head_world.GetInverse()).T


def plan_head_pose(model, initial_pose, root_world_column, camera_to_head_column, target_world,
                   intrinsics, desired_pixel):
    """Solve only the two real head joints; do not teleport or re-aim the camera mount."""
    from scipy.optimize import least_squares
    target = np.asarray(target_world, float)
    k = np.asarray(intrinsics)
    pixel = np.asarray(desired_pixel, float)
    if (target.shape != (3,) or pixel.shape != (2,) or not np.isfinite([*target, *pixel]).all()
            or not (0 < pixel[0] < RESOLUTION[0] and 0 < pixel[1] < RESOLUTION[1])):
        raise ValueError("Invalid target/framing pixel")
    optical = np.linalg.solve(k, [*pixel, 1])
    desired_ray = optical * [1, -1, -1]
    desired_ray /= np.linalg.norm(desired_ray)
    names = ("head_0", "head_1")
    limits = [(math.degrees(model._by_name[n].lower_rad), math.degrees(model._by_name[n].upper_rad)) for n in names]

    def residual(angles):
        pose = {**initial_pose, **dict(zip(names, angles))}
        head = model.all_link_transforms(pose)["link_head_2"]
        camera = root_world_column @ head @ camera_to_head_column
        direction = target - camera[:3, 3]
        direction /= max(np.linalg.norm(direction), 1e-9)
        return camera[:3, :3] @ desired_ray - direction

    solved = least_squares(residual, [initial_pose.get(n, 0) for n in names],
                           bounds=([p[0]+.01 for p in limits], [p[1]-.01 for p in limits]), max_nfev=100,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    error = math.degrees(2 * math.asin(min(1, np.linalg.norm(residual(solved.x)) / 2)))
    if not solved.success or error > .1:
        raise ValueError(f"Target framing cannot be reached with the head joints: {error:.3f} deg")
    pose = {**initial_pose, **dict(zip(names, map(float, solved.x)))}
    model.all_link_transforms(pose)  # All arm/torso/head URDF limits, not only head.
    return pose, error


def set_snapshot_pose(stage, robot, target_world, y_offset, desired_pixel, *, root_x_m=None, root_yaw_degrees=None):
    from .floor_alignment import align_robot_to_floor, PACKAGE_FLOOR
    root = stage.GetPrimAtPath(robot["root"])
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root)).T.copy()
    if root_yaw_degrees is not None:
        if not np.isfinite(root_yaw_degrees) or not 150 <= root_yaw_degrees <= 210:
            raise ValueError("Focused base yaw must remain within 30 degrees of the original aisle facing")
        matrix[:3, :3] = robot_hardware.rotation_z(root_yaw_degrees)
    matrix[1, 3] = float(target_world[1] + y_offset)
    if root_x_m is not None:
        if not np.isfinite(root_x_m) or root_x_m <= target_world[0]:
            raise ValueError("Viewpoint must remain on the original +X robot aisle side")
        matrix[0, 3] = float(root_x_m)
    # Default pilot keeps its distance; bounded search may request a nearer snapshot.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_hardware._set_transform(root, matrix[:3, :3], matrix[:3, 3])
    floor = align_robot_to_floor(stage, robot["root"], PACKAGE_FLOOR)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root)).T
    mount = mounted_camera_to_head(stage)
    model = robot_kinematics.Rby1Kinematics()
    pose, error = plan_head_pose(model, robot["pose_degrees"], matrix, mount, target_world,
                                calibration(stage)["intrinsics"], desired_pixel)
    links = model.all_link_transforms(pose)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for link, transform in links.items():
            robot_hardware._set_transform(stage.GetPrimAtPath(robot["root"] + "/" + link),
                                         transform[:3, :3], transform[:3, 3])
    if not np.allclose(mounted_camera_to_head(stage), mount, atol=1e-9):
        raise ValueError("Camera mounting transform changed during pose sampling")
    return {"joint_degrees": pose, "robot_root_to_world_usd_row_vectors": matrix.T.tolist(),
            "camera_to_head_column_vectors": mount.tolist(), "framing_error_degrees": error,
            "desired_cut_pixel_xy": list(desired_pixel), "floor_alignment": floor,
            "pose_sampling": "geometry_guided_static_pilot_not_autonomous_view_selection",
            "joint_limits_checked": True, "whole_robot_collision_checked": False,
            "motion_between_snapshots_validated": False, "arm_reachability": "not_tested"}


def freeze_rigid_bodies(stage):
    """Pause physics in this disposable capture stage, without changing geometry/assets."""
    frozen = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(prim)
                if api.GetRigidBodyEnabledAttr().Get():
                    api.CreateRigidBodyEnabledAttr(False)
                    frozen.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    return frozen


def scene_guard(stage):
    """Fingerprint visible scene transforms/material bindings; excludes Replicator bookkeeping."""
    states = []
    cache = UsdGeom.XformCache()
    forbidden = ("/World/DraftLabels", "/World/AnatomyReview", "/World/Review", "/World/Reachability")
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
            raise ValueError(f"Static pilot refuses enabled rigid bodies: {path}")
        if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
            raise ValueError(f"Static pilot refuses animated attributes: {path}")
        if any(path.startswith(prefix) for prefix in forbidden):
            raise ValueError(f"Diagnostic overlays present: {path}")
        if not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        states.append((path, imageable.ComputeVisibility(),
                       [list(r) for r in cache.GetLocalToWorldTransform(prim)],
                       [str(p) for p in prim.GetRelationship("material:binding").GetTargets()]))
    return fingerprint(states)


def target_world_geometry(stage, variant, row):
    root = stage.GetPrimAtPath(variant["plant_root"])
    matrix = [list(r) for r in UsdGeom.XformCache().GetLocalToWorldTransform(root)]
    proposal = row["cut_region_proposal"]
    nominal = transform_points([proposal["nominal"]["point_plant_m"]], matrix)[0]
    # Sample the complete 10-20 mm polyline at 1 mm spacing for review projection.
    samples = proposal["accepted_centerline_interval"]["samples"]
    points = []
    for a, b in zip(samples, samples[1:]):
        steps = max(1, int(np.ceil((b["arc_distance_m"] - a["arc_distance_m"]) / .001)))
        for t in np.linspace(0, 1, steps, endpoint=False):
            points.append(np.asarray(a["point_plant_m"]) * (1-t) + np.asarray(b["point_plant_m"]) * t)
    points.append(samples[-1]["point_plant_m"])
    return {"plant_to_world_usd_row_vectors": matrix, "nominal_world_m": nominal.tolist(),
            "interval_world_m": transform_points(points, matrix).tolist()}
