"""Full-scene pilot capture validation, projection and review-only exports.

No simulator imports here: malformed/stale sensor payloads can be unit tested.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

RESOLUTION = (848, 408)
PILOT_TARGETS = ("B03", "B05", "B06")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def transform_points(points, matrix_row):
    points, matrix = np.asarray(points, float), np.asarray(matrix_row, float)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Invalid point/transform shape")
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError("Non-finite geometry")
    if not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("Expected an affine USD row-vector transform")
    return (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]


def project(points_world, calibration):
    """USD camera +X right,+Y up,-Z forward -> optical +X right,+Y down,+Z forward."""
    camera = transform_points(points_world, np.linalg.inv(calibration["camera_to_world_usd_row_vectors"]))
    optical = camera * [1, -1, -1]
    k = np.asarray(calibration["intrinsics"], float)
    width, height = calibration["resolution"]
    near, far = calibration["clipping_range_m"]
    result = []
    for p in optical:
        uv = (k @ p)[:2] / p[2] if p[2] > 0 else None
        status = ("behind_camera" if p[2] <= 0 else "outside_clipping_range" if not near <= p[2] <= far
                  else "in_frame" if 0 <= uv[0] < width and 0 <= uv[1] < height else "out_of_frame")
        result.append({"camera_optical_xyz_m": p.tolist(), "pixel_xy": None if uv is None else uv.tolist(),
                       "projection_status": status})
    return result


def validate_payload(payload, calibration, previous_reference=None):
    """One Replicator writer payload, one camera, no reused frame fallback."""
    required = {"rgb", "distance_to_image_plane", "camera_params", "ReferenceTime", "reference_time"}
    if not required <= payload.keys():
        raise ValueError(f"Incomplete synchronized payload: {sorted(required - payload.keys())}")
    reference = payload["reference_time"]
    if (len(reference) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in reference)
            or reference[0] < 0 or reference[1] <= 0):
        raise ValueError("Invalid render reference time")
    reference = [int(x) for x in reference]
    rt = payload["ReferenceTime"]
    if Fraction(*reference) != Fraction(int(rt["referenceTimeNumerator"]), int(rt["referenceTimeDenominator"])):
        raise ValueError("Mismatched annotator reference time")
    if previous_reference is not None and Fraction(*reference) <= Fraction(*previous_reference):
        raise ValueError(f"Stale or repeated render payload: {reference} <= {previous_reference}")
    products = [v for k, v in payload.items() if k.startswith("rp_")]
    if (len(products) != 1 or products[0]["camera"] != calibration["camera_path"]
            or list(products[0]["resolution"]) != list(RESOLUTION)):
        raise ValueError("Expected one mounted head-camera render product")
    rgb, depth = np.asarray(payload["rgb"]), np.asarray(payload["distance_to_image_plane"])
    if rgb.dtype != np.uint8 or rgb.shape != (408, 848, 4):
        raise ValueError(f"Invalid RGB shape/dtype: {rgb.shape}/{rgb.dtype}")
    if depth.shape != (408, 848) or depth.dtype != np.float32:
        raise ValueError(f"Invalid depth shape/dtype: {depth.shape}/{depth.dtype}")
    if float(rgb[:, :, :3].std()) < 1:
        raise ValueError("Blank RGB payload")
    cp = payload["camera_params"]
    if (list(cp["renderProductResolution"]) != list(RESOLUTION)
            or not np.isclose(cp["metersPerSceneUnit"], 1)):
        raise ValueError("Render calibration units/resolution mismatch")
    view = np.asarray(cp["cameraViewTransform"], float).reshape(4, 4)
    expected_view = np.linalg.inv(calibration["camera_to_world_usd_row_vectors"])
    if not np.allclose(view, expected_view, rtol=0, atol=5e-5):
        raise ValueError("Rendered camera does not match the captured USD pose")
    # Compare rendered projection to authored pinhole intrinsics, including offsets.
    projection = np.asarray(cp["cameraProjection"], float).reshape(4, 4)
    probes = np.asarray([[0, 0, -1], [.1, .15, -1.2], [-.2, -.1, -2]])
    clip = np.column_stack((probes, np.ones(3))) @ projection
    if not np.isfinite(clip).all() or np.any(np.abs(clip[:, 3]) < 1e-8):
        raise ValueError("Invalid rendered projection")
    ndc = clip[:, :2] / clip[:, 3, None]
    pixels = (ndc * [1, -1] + 1) * (np.asarray(RESOLUTION) / 2)
    world = transform_points(probes, calibration["camera_to_world_usd_row_vectors"])
    expected_pixels = np.asarray([p["pixel_xy"] for p in project(world, calibration)])
    if not np.allclose(pixels, expected_pixels, rtol=0, atol=.01):
        raise ValueError("Rendered projection does not match pixel intrinsics")
    near, far = calibration["clipping_range_m"]
    valid = np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)
    if valid.mean() < .05:
        raise ValueError("Insufficient valid scene depth")
    # Preserve raw depth exactly; validity is a separate array, never infer a surface at infinity.
    return rgb[:, :, :3].copy(), depth.copy(), valid, reference


def validate_static_capture(payload, calibration, guard_before, guard_after, callback_sequence, previous=None):
    """Static-state synchronization when native frame/time IDs are unavailable.

    Not a dynamic recorder. The caller must freeze all geometry/materials and
    reject time-sampled geometry or enabled rigid bodies before using this path.
    Never invent an engine frame ID from our application sequence.
    """
    if (not isinstance(guard_before, str) or len(guard_before) != 64 or guard_before != guard_after):
        raise ValueError("Static scene guard changed or unavailable")
    if not isinstance(callback_sequence, int) or callback_sequence <= 0:
        raise ValueError("Fresh writer callback required")
    rgb, depth, valid, reference = validate_payload(payload, calibration)
    token = {"callback_sequence":callback_sequence,"camera_sha256":fingerprint(calibration),
             "rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),
             "depth_sha256":hashlib.sha256(depth.tobytes()).hexdigest()}
    if previous is not None:
        if callback_sequence <= previous["callback_sequence"]:
            raise ValueError("Stale writer callback")
        if token["camera_sha256"] == previous["camera_sha256"]:
            raise ValueError("Pilot requires a distinct camera pose, not repeated observations")
        if token["rgb_sha256"] == previous["rgb_sha256"] or token["depth_sha256"] == previous["depth_sha256"]:
            raise ValueError("Unchanged RGB or depth after camera movement; cannot establish freshness")
    return rgb, depth, valid, reference, token


def depth_evidence(projected, depth, valid, radius_m):
    """Conservative one-pixel occlusion evidence, NOT an organ mask or visibility fraction."""
    result = {"method": "nominal_centerline_projection_depth_test.v1", "visibility_fraction": None,
              "occluder_identity": None, "point_is_centerline_not_surface": True}
    if projected["projection_status"] != "in_frame":
        return {**result, "status": projected["projection_status"]}
    u, v = projected["pixel_xy"]
    x, y = int(np.floor(u)), int(np.floor(v))
    if not valid[y, x]:
        return {**result, "status": "unknown_invalid_depth", "sampled_pixel_xy": [x, y]}
    z, surface = projected["camera_optical_xyz_m"][2], float(depth[y, x])
    # A centreline lies below its own surface. Also allow 3 mm for raster/geometry approximation.
    tolerance = float(radius_m) + .003
    gap = float(z - surface)
    status = ("foreground_occlusion_evidence" if gap > tolerance else
              "depth_consistent_not_visibility_verified" if abs(gap) <= tolerance else
              "unknown_no_target_surface_at_pixel")
    return {**result, "status": status, "sampled_pixel_xy": [x, y], "scene_depth_m": surface,
            "target_axis_depth_m": z, "foreground_gap_m": gap, "tolerance_m": tolerance}


def write_sample(directory, rgb, depth, valid, metadata):
    """Clean observations and review overlays have disjoint paths. Publish JSON last."""
    from PIL import Image, ImageDraw
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    inputs, review = directory / "inputs", directory / "review"
    inputs.mkdir()
    review.mkdir()
    Image.fromarray(rgb).save(inputs / "rgb.png")
    np.save(inputs / "depth_m.npy", depth, allow_pickle=False)
    Image.fromarray(valid.astype(np.uint8) * 255).save(inputs / "depth_valid.png")
    overlay = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(overlay)
    projected = metadata["supervision"]["projected_interval"]
    points = [p["pixel_xy"] for p in projected if p["projection_status"] == "in_frame"]
    if len(points) == len(projected) and len(points) >= 2:
        draw.line([tuple(p) for p in points], fill=(255, 0, 190), width=3)
    nominal = metadata["supervision"]["nominal_projected"]
    if nominal["projection_status"] == "in_frame":
        x, y = nominal["pixel_xy"]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="white", width=2)
        draw.line((x-9,y,x+9,y), fill="white", width=1)
        draw.line((x,y-9,x,y+9), fill="white", width=1)
    draw.rectangle((0, 0, 848, 33), fill="black")
    draw.text((8, 3), "REVIEW ONLY: projected proposal, NOT observed surface / execution approval", fill="white")
    draw.text((8, 17), metadata["supervision"]["depth_evidence"]["status"], fill="yellow")
    overlay.save(review / "overlay.png")
    metadata["files"] = {str(p.relative_to(directory)).replace("\\", "/"): {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "role": "review_only" if p.parent == review else "observation",
    } for p in [inputs / "rgb.png", inputs / "depth_m.npy", inputs / "depth_valid.png", review / "overlay.png"]}
    (directory / "sample.json").write_text(json.dumps(jsonable(metadata), indent=2, allow_nan=False), encoding="utf-8")
    return metadata
