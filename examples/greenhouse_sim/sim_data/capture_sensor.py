"""Explicit native sensor dimensions; no image resizing or depth reconstruction.

A higher-resolution render of the existing ideal pinhole is not a claim that
the physical D405 supports that operating mode. Legacy capture remains 848x408
until a caller explicitly selects and validates the separate native mode.
"""
from copy import deepcopy
import numpy as np

LEGACY_RESOLUTION = (848, 408)
HIRES_RESOLUTION = (1696, 816)
SUPPORTED_RESOLUTIONS = (LEGACY_RESOLUTION, HIRES_RESOLUTION)


def checked_resolution(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) is not int for v in value)
            or tuple(value) not in SUPPORTED_RESOLUTIONS):
        raise ValueError("Explicit supported native resolution required")
    return tuple(value)


def sensor_profile(resolution):
    width, height = checked_resolution(resolution)
    return dict(schema="greenhouse.native_sensor_dimensions.v1",
                mode=f"mounted_head_native_{width}x{height}_v1",
                resolution=[width, height], rgb_resized=False,
                depth_source="native_distance_to_image_plane",
                depth_units="metres", camera_mount_changed=False,
                optics_changed=False,
                hardware_resolution_validated=False,
                scope="ideal_simulator_pinhole_not_real_D405_mode_validation")


def _authored_intrinsics(calibration, resolution):
    width, height = checked_resolution(resolution)
    focal = calibration["focal_length_mm"]
    apertures = np.asarray(calibration["apertures_mm"], float)
    offsets = np.asarray(calibration["aperture_offsets_mm"], float)
    if (apertures.shape != (2,) or offsets.shape != (2,)
            or type(focal) not in (float, int) or not np.isfinite(focal)
            or focal <= 0 or not np.isfinite(apertures).all()
            or not np.isfinite(offsets).all() or (apertures <= 0).any()):
        raise ValueError("Invalid authored pinhole parameters")
    return np.asarray([[width*focal/apertures[0], 0, width*(.5-offsets[0]/apertures[0])],
                       [0, height*focal/apertures[1], height*(.5+offsets[1]/apertures[1])],
                       [0, 0, 1]], float)


def calibration_for_native_resolution(calibration, resolution):
    """Recompute pixel intrinsics for a NEW native product, not an existing image.

    Camera transform, apertures, focal length, clipping and metric depth
    convention are unchanged. A native payload must independently match these
    intrinsics before any capture is accepted.
    """
    resolution = checked_resolution(resolution)
    original = checked_resolution(calibration["resolution"])
    if calibration.get("crop_resize") is not None:
        raise ValueError("Uncropped native calibration required")
    if calibration.get("depth_convention") != "optical_axis_z_metres_not_ray_range":
        raise ValueError("Native metric optical-Z convention required")
    k = np.asarray(calibration["intrinsics"], float)
    if k.shape != (3, 3) or not np.allclose(k, _authored_intrinsics(calibration, original),
                                          atol=1e-7, rtol=0):
        raise ValueError("Source intrinsics do not match authored optics")
    matrix = np.asarray(calibration["camera_to_world_usd_row_vectors"], float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError("Rigid right-handed camera transform required")
    result = deepcopy(calibration)
    result["resolution"] = list(resolution)
    result["intrinsics"] = _authored_intrinsics(calibration, resolution).tolist()
    return result


def normalized_coordinates(points, resolution, *, inverse=False):
    """Convert coordinates only; never resample RGB, masks or native metric Z."""
    dimensions = np.asarray(checked_resolution(resolution), float)
    if type(inverse) is not bool:
        raise ValueError("Explicit coordinate direction required")
    points = np.asarray(points, float)
    bound = np.asarray([1000., 1000.]) if inverse else dimensions
    if (points.ndim not in (1, 2) or points.shape[-1] != 2
            or not points.size or not np.isfinite(points).all()
            or (points < 0).any() or (points >= bound).any()):
        raise ValueError("Coordinates outside the declared original frame")
    return points*dimensions/1000. if inverse else points*1000./dimensions
