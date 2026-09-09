"""Actual mounted-camera review context; never re-aim or teleport a camera."""
import numpy as np
from pxr import Gf, Usd, UsdGeom

HEAD_CAMERA = "/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera"


def has_head_camera(stage):
    prim = stage.GetPrimAtPath(HEAD_CAMERA)
    return bool(prim and prim.IsActive() and prim.IsA(UsdGeom.Camera))


def attachment_world(stage, paths, report, target):
    component = report["components"][target["component_id"]]
    local = np.asarray(target["attachment_plant_m"], dtype=float) - component["translation_plant_m"]
    prim = stage.GetPrimAtPath(paths[target["component_id"]])
    return UsdGeom.XformCache().GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(*local))


def select_review_view(stage, viewport, scene, target, mode):
    """Switch viewpoint only. Robot, camera extrinsics, and plant stay unchanged."""
    from .robot_preview import select_camera
    point = attachment_world(stage, scene.paths, scene.report, target)
    if mode == "robot_head":
        if not has_head_camera(stage):
            raise ValueError("Mounted head camera unavailable; select diagnostic close-up explicitly")
        scene.restore_visibility()
        select_camera(viewport, HEAD_CAMERA)
    elif mode == "diagnostic_closeup":
        scene.focus(viewport, point)
        viewport.set_texture_resolution((1280, 720))
    else:
        raise ValueError(f"Unknown review view: {mode}")
    return point


def view_context(stage, viewport, scene, target):
    """Projection/frustum test only: an in-frame point may still be occluded."""
    camera = UsdGeom.Camera(stage.GetPrimAtPath(viewport.camera_path))
    if not camera or camera.GetProjectionAttr().Get() != "perspective":
        raise ValueError("A perspective camera is required for review context")
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(camera.GetPrim())
    point = attachment_world(stage, scene.paths, scene.report, target)
    local = matrix.GetInverse().Transform(point)
    width, height = viewport.get_texture_resolution()
    focal = camera.GetFocalLengthAttr().Get()
    horizontal, vertical = camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()
    if min(width, height, focal, horizontal, vertical) <= 0:
        raise ValueError("Invalid camera resolution or aperture")
    fx, fy = width * focal / horizontal, height * focal / vertical
    cx = width * (.5 - camera.GetHorizontalApertureOffsetAttr().Get() / horizontal)
    cy = height * (.5 + camera.GetVerticalApertureOffsetAttr().Get() / vertical)
    depth = -float(local[2])
    near, far = camera.GetClippingRangeAttr().Get()
    pixel = None
    if depth <= 0:
        status = "behind_camera"
    else:
        pixel = [fx * float(local[0]) / depth + cx, cy - fy * float(local[1]) / depth]
        status = ("outside_clipping_range" if not near <= depth <= far else
                  "inside_image_frustum" if 0 <= pixel[0] < width and 0 <= pixel[1] < height else "outside_image")
    robot = stage.GetPrimAtPath("/World/RBY1/base")
    return {"schema_version": "greenhouse.review_view.v1", "camera_path": str(camera.GetPath()),
            "view_kind": "robot_head" if str(camera.GetPath()) == HEAD_CAMERA else "diagnostic_or_other_camera",
            "resolution": [width, height], "camera_to_world": [list(r) for r in matrix],
            "matrix_convention": "USD row vectors", "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "target_world_m": list(point), "attachment_pixel_xy": pixel, "projection_status": status,
            "projection_is_visibility_measurement": False, "occlusion": "not_measured",
            "robot_base_to_world": [list(r) for r in cache.GetLocalToWorldTransform(robot)] if robot else None,
            "plant_root": scene.plant_root,
            "plant_to_world": [list(r) for r in cache.GetLocalToWorldTransform(stage.GetPrimAtPath(scene.plant_root))],
            "isolated": bool(scene.visibility_paths), "overlays_present": True, "training_input_allowed": False}
