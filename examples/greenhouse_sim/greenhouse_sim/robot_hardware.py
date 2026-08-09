"""Author the supplied deleafing knife and D405 hardware on RBY1-A v1.0.

All mount transforms in this module are explicit and testable.  CAD is kept in
millimetres in ``greenhouse/robot_assets`` and converted to metres only while
authoring USD.  The flat knife plate is the sole cutting surface; the curved
piece is support geometry and can never trigger a cut.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import struct

import numpy as np

from greenhouse_sim import usd_env

usd_env.ensure_pxr()

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSET_DIR = REPOSITORY_ROOT / "greenhouse" / "robot_assets"
DERIVED_DIR = ASSET_DIR / "derived"
MANIFEST_PATH = DERIVED_DIR / "hardware.json"

ROBOT_ROOT = "/RBY1_A_v1_0"
END_EFFECTOR_LINKS = {"left": "ee_left", "right": "ee_right"}
HEAD_LINK = "link_head_2"
RIGHT_GRIPPER_LINKS = ("ee_right", "ee_finger_r1", "ee_finger_r2")

# Local mounts in the corresponding RBY1 link frame.  RBY1 tools extend along
# -Z.  The wrist-camera transform is measured from the supplied
# RBY1_Example_setup.FCStd reference, not fitted by eye.  In that assembly the
# bracket's two M3 axes are at x=+/-9, y=-39, z=38.5 mm in either local gripper
# frame.  The extracted STL was re-origined at the source bracket bounds, so its
# same bolt axes are x=+/-9, y=56.919538, z=30.119184 mm.  This translation
# aligns those axes exactly and seats the bracket against the wrist screw face.
WRIST_REFERENCE_BOLT_CENTRES_M = np.array(
    [[-0.009, -0.039, 0.0385], [0.009, -0.039, 0.0385]], dtype=np.float64
)
WRIST_BRACKET_BOLT_CENTRES_M = np.array(
    [
        [-0.009, 0.0569195383671445, 0.0301191835115545],
        [0.009, 0.0569195383671445, 0.0301191835115545],
    ],
    dtype=np.float64,
)
WRIST_CAMERA_TRANSLATION_M = np.array(
    [0.0, -0.0959195383671445, 0.0083808164884455], dtype=np.float64
)
LEFT_CAMERA_TRANSLATION_M = WRIST_CAMERA_TRANSLATION_M.copy()
RIGHT_CAMERA_TRANSLATION_M = WRIST_CAMERA_TRANSLATION_M.copy()
# The right gripper body is removed for the deleafing configuration.  The
# knife's CAD origin is its mounting face, so it mounts directly at the
# retained ee_right kinematic frame rather than at the old jaw tip.
KNIFE_TRANSLATION_M = np.zeros(3, dtype=np.float64)
CUTTING_EDGE_DEPTH_M = 0.002
HEAD_BRACKET_TRANSLATION_M = np.array([0.022, 0.0, 0.040], dtype=np.float64)


def wrist_d405_body_sphere(
    manifest: dict | None = None,
) -> tuple[np.ndarray, float]:
    """Return the conservative D405 body sphere in either wrist EE frame."""
    centre, _, half_extents = wrist_d405_body_box(manifest)
    return centre, float(np.linalg.norm(half_extents))


def wrist_d405_body_box(
    manifest: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return D405 body box centre, rotation, and half extents in wrist EE."""
    manifest = load_manifest() if manifest is None else manifest
    body_minimum, body_maximum = _bounds_m(_part(manifest, "d405_body"))
    body_centre = 0.5 * (body_minimum + body_maximum)
    half_extents = 0.5 * (body_maximum - body_minimum)
    mount = manifest["mounts"]["camera_bracket_to_d405"]
    mount_rotation = np.asarray(mount["rotation_matrix"], dtype=np.float64)
    mount_translation = np.asarray(mount["translation_mm"], dtype=np.float64) * 0.001
    centre = WRIST_CAMERA_TRANSLATION_M + mount_translation + mount_rotation @ body_centre
    return centre, mount_rotation, half_extents


def knife_blade_box(
    manifest: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return blade-plate centre and half extents in the knife CAD frame."""
    manifest = load_manifest() if manifest is None else manifest
    knife = _part(manifest, "deleaf_knife")
    blade = next(
        component
        for component in knife["components"]
        if component["name"] == "deleaf_knife_blade"
    )
    minimum = np.asarray(blade["min_mm"], dtype=np.float64) * 0.001
    maximum = np.asarray(blade["max_mm"], dtype=np.float64) * 0.001
    return 0.5 * (minimum + maximum), 0.5 * (maximum - minimum)


def knife_support_box(
    manifest: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative U-support bounds in the knife CAD frame."""
    manifest = load_manifest() if manifest is None else manifest
    knife = _part(manifest, "deleaf_knife")
    support = next(
        component
        for component in knife["components"]
        if component["name"] == "deleaf_knife_arc"
    )
    minimum = np.asarray(support["min_mm"], dtype=np.float64) * 0.001
    maximum = np.asarray(support["max_mm"], dtype=np.float64) * 0.001
    return 0.5 * (minimum + maximum), 0.5 * (maximum - minimum)


@dataclasses.dataclass(frozen=True)
class TriangleMesh:
    """A compact triangle mesh in metres."""

    points: np.ndarray
    triangles: np.ndarray

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)


@dataclasses.dataclass(frozen=True)
class HardwareReport:
    """Stable paths produced by :func:`attach_robot_hardware`."""

    cameras: tuple[str, ...]
    cutting_surfaces: tuple[str, ...]
    non_cutting_supports: tuple[str, ...]
    attachments: tuple[str, ...]
    removed_right_gripper_prims: tuple[str, ...]


def rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# Both reference camera assemblies have the same transform relative to their
# mirrored local gripper frames.  Mirroring is supplied by the robot kinematic
# chain; adding another right-side 180-degree rotation misaligns the M3 pair.
LEFT_CAMERA_ROTATION = np.eye(3, dtype=np.float64)
RIGHT_CAMERA_ROTATION = np.eye(3, dtype=np.float64)
# Preserve CAD -Y along tool -Z, but roll the plate around that blade axis so
# the U-shaped support on CAD +Z faces tool +X. In the greenhouse ready pose,
# tool +X is the most upward transverse direction.
KNIFE_ROTATION = rotation_z(90.0) @ rotation_x(90.0)
HEAD_BRACKET_ROTATION = rotation_z(90.0)
HEAD_CAMERA_TRANSLATION_M = np.array([0.0, 0.00123, 0.025], dtype=np.float64)
HEAD_CAMERA_ROTATION = rotation_z(180.0)


def compose_rotation_xyz(rpy_radians: tuple[float, float, float] | list[float]) -> np.ndarray:
    """URDF roll-pitch-yaw rotation, ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = rpy_radians
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def transform_direction(rotation: np.ndarray, direction: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Apply a conventional column-vector rotation to a direction."""
    return np.asarray(rotation, dtype=np.float64) @ np.asarray(direction, dtype=np.float64)


def cut_aligned_knife_rotation(
    target_axis: tuple[float, float, float] | np.ndarray,
    preferred_cut_direction: tuple[float, float, float] | np.ndarray,
    upward: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Orient the CAD knife for a transverse cut with its support facing up.

    The flat plate's local X axis is its edge, local -Y is cutting travel, and
    local +Z points toward the U-shaped support.  A valid guillotine-like cut
    therefore places both X and -Y transverse to the contacted petiole tangent.
    The support normal is selected from the two tangent directions so it has a
    positive world-up component, then the preferred aisle motion is projected
    into the transverse plane.
    """
    tangent = np.asarray(target_axis, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    preferred = np.asarray(preferred_cut_direction, dtype=np.float64)
    preferred_norm = float(np.linalg.norm(preferred))
    up = np.asarray(upward, dtype=np.float64)
    if tangent.shape != (3,) or tangent_norm <= 1e-12:
        raise ValueError("target axis must be a non-zero three-vector")
    if preferred.shape != (3,) or preferred_norm <= 1e-12:
        raise ValueError("preferred cut direction must be a non-zero three-vector")
    if up.shape != (3,) or float(np.linalg.norm(up)) <= 1e-12:
        raise ValueError("upward direction must be a non-zero three-vector")
    support = tangent / tangent_norm
    if float(np.dot(support, up)) < 0.0:
        support = -support
    cut_direction = preferred - float(np.dot(preferred, support)) * support
    cut_norm = float(np.linalg.norm(cut_direction))
    if cut_norm <= 1e-12:
        raise ValueError("preferred cut direction is parallel to the target axis")
    cut_direction /= cut_norm
    knife_y = -cut_direction
    edge_axis = np.cross(knife_y, support)
    edge_axis /= float(np.linalg.norm(edge_axis))
    return np.column_stack((edge_axis, knife_y, support))


def read_binary_stl(path: pathlib.Path, *, scale: float = 0.001) -> TriangleMesh:
    """Read a binary STL without adding a runtime CAD/trimesh dependency."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangle_count
    if len(data) != expected:
        raise ValueError(f"expected a binary STL of {expected} bytes, found {len(data)}: {path}")

    record = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    facets = np.frombuffer(data, dtype=record, count=triangle_count, offset=84)
    raw_points = np.asarray(facets["vertices"], dtype=np.float64).reshape(-1, 3)
    points, inverse = np.unique(raw_points, axis=0, return_inverse=True)
    return TriangleMesh(points=points * scale, triangles=inverse.reshape(-1, 3).astype(np.int64))


def _gf_matrix(rotation: np.ndarray, translation: np.ndarray) -> Gf.Matrix4d:
    values = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix = Gf.Matrix4d(1.0)
    # NumPy/CAD rotations here use column vectors. Gf transforms row vectors,
    # so the numeric matrix must be transposed at this boundary.
    matrix.SetRotate(Gf.Matrix3d(*values.T.reshape(-1).tolist()))
    matrix.SetTranslateOnly(Gf.Vec3d(*np.asarray(translation, dtype=np.float64).tolist()))
    return matrix


def _set_transform(prim: Usd.Prim, rotation: np.ndarray, translation: np.ndarray) -> None:
    transformable = UsdGeom.Xformable(prim)
    transformable.ClearXformOpOrder()
    transformable.AddTransformOp().Set(_gf_matrix(rotation, translation))


def _hardware_attr(prim: Usd.Prim, name: str, value) -> None:
    if isinstance(value, bool):
        value_type = Sdf.ValueTypeNames.Bool
    elif isinstance(value, float):
        value_type = Sdf.ValueTypeNames.Float
    else:
        value_type = Sdf.ValueTypeNames.String
    prim.CreateAttribute(f"tomato:{name}", value_type, custom=True).Set(value)


def _author_mesh(
    stage: Usd.Stage,
    path: str,
    source: pathlib.Path,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    data = read_binary_stl(source)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in data.points])
    mesh.CreateFaceVertexCountsAttr([3] * len(data.triangles))
    mesh.CreateFaceVertexIndicesAttr(data.triangles.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def _author_box_collider(
    stage: Usd.Stage,
    path: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
    *,
    collidable: bool = True,
) -> str:
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (minimum + maximum))))
    xform.AddScaleOp().Set(Gf.Vec3f(*(maximum - minimum)))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    if collidable:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return path


def _author_decomposed_collider(stage: Usd.Stage, path: str, source: pathlib.Path) -> str:
    """Author a dynamic concave part as multiple convex hulls.

    A single convex hull around the supplied U support fills its opening and
    turns the guide into a solid plate in PhysX. Convex decomposition keeps the
    collider dynamic while preserving the passage through which a petiole must
    reach the flat blade.
    """
    mesh = _author_mesh(stage, path, source, (0.2, 0.2, 0.2))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(
        mesh.GetPrim()
    ).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
    return path


def _part(manifest: dict, name: str) -> dict:
    return next(part for part in manifest["parts"] if part["part"] == name)


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(f"unsupported robot hardware manifest: {path}")
    return manifest


def _bounds_m(part: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(part["min_mm"], dtype=np.float64) * 0.001, np.asarray(part["max_mm"], dtype=np.float64) * 0.001


def _author_d405(
    stage: Usd.Stage,
    root_path: str,
    manifest: dict,
    rotation: np.ndarray,
    translation: np.ndarray,
    role: str,
    sensor_roll_degrees: float = 0.0,
) -> str:
    root = UsdGeom.Xform.Define(stage, root_path)
    _set_transform(root.GetPrim(), rotation, translation)
    _hardware_attr(root.GetPrim(), "hardwareRole", f"d405_{role}")

    body_part = _part(manifest, "d405_body")
    _author_mesh(stage, f"{root_path}/BodyVisual", DERIVED_DIR / "d405_body.stl", (0.08, 0.08, 0.09))
    minimum, maximum = _bounds_m(body_part)
    _author_box_collider(stage, f"{root_path}/BodyCollision", minimum, maximum)

    optical_mm = np.asarray(body_part["optical_origin_mm"], dtype=np.float64)
    camera_path = f"{root_path}/DepthCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    # USD cameras look down local -Z with +Y up.  Rx(+90) maps those axes to
    # D405 +Y (forward) and +Z (up), respectively.
    # The right camera body is mirrored across the robot. Keep the physical
    # body and bracket mirrored, but roll its optical frame so policy images
    # have the same upright convention as the head and left-wrist cameras.
    sensor_rotation = rotation_x(90.0) @ rotation_z(sensor_roll_degrees)
    _set_transform(camera.GetPrim(), sensor_rotation, optical_mm * 0.001)
    horizontal, vertical = body_part["depth_fov_degrees"]
    focal_length_mm = 10.0
    camera.CreateFocalLengthAttr(focal_length_mm)
    camera.CreateHorizontalApertureAttr(2.0 * focal_length_mm * math.tan(math.radians(horizontal / 2.0)))
    camera.CreateVerticalApertureAttr(2.0 * focal_length_mm * math.tan(math.radians(vertical / 2.0)))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.04, 10.0))
    _hardware_attr(camera.GetPrim(), "cameraModel", "Intel RealSense D405")
    _hardware_attr(camera.GetPrim(), "cameraRole", role)
    _hardware_attr(camera.GetPrim(), "horizontalFovDegrees", float(horizontal))
    _hardware_attr(camera.GetPrim(), "verticalFovDegrees", float(vertical))
    _hardware_attr(camera.GetPrim(), "sensorRollDegrees", float(sensor_roll_degrees))
    return camera_path


def _author_wrist_camera(
    stage: Usd.Stage,
    link_path: str,
    side: str,
    manifest: dict,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[str, str]:
    assembly_path = f"{link_path}/attachments/{side.title()}WristCamera"
    assembly = UsdGeom.Xform.Define(stage, assembly_path)
    _set_transform(assembly.GetPrim(), rotation, translation)
    _hardware_attr(assembly.GetPrim(), "hardwareRole", "wrist_camera_assembly")
    _hardware_attr(assembly.GetPrim(), "mountInterface", "rby1_wrist_m3_pair")
    _hardware_attr(assembly.GetPrim(), "mountBoltSpacingMillimeters", 18.0)

    bracket_part = _part(manifest, "camera_bracket_d405")
    _author_mesh(stage, f"{assembly_path}/BracketVisual", DERIVED_DIR / "camera_bracket_d405.stl", (0.12, 0.12, 0.14))
    minimum, maximum = _bounds_m(bracket_part)
    _author_box_collider(stage, f"{assembly_path}/BracketCollision", minimum, maximum)

    mount = manifest["mounts"]["camera_bracket_to_d405"]
    camera_path = _author_d405(
        stage,
        f"{assembly_path}/D405",
        manifest,
        np.asarray(mount["rotation_matrix"], dtype=np.float64),
        np.asarray(mount["translation_mm"], dtype=np.float64) * 0.001,
        f"{side}_wrist",
        180.0 if side == "right" else 0.0,
    )
    return assembly_path, camera_path


def _author_head_camera(stage: Usd.Stage, link_path: str, manifest: dict) -> tuple[str, str]:
    assembly_path = f"{link_path}/attachments/HeadCamera"
    assembly = UsdGeom.Xform.Define(stage, assembly_path)
    _set_transform(assembly.GetPrim(), HEAD_BRACKET_ROTATION, HEAD_BRACKET_TRANSLATION_M)
    _hardware_attr(assembly.GetPrim(), "hardwareRole", "head_camera_assembly")

    part = _part(manifest, "head_camera_bracket_d405")
    _author_mesh(stage, f"{assembly_path}/BracketVisual", ASSET_DIR / "HeadCam_Bracket_D405-Body.stl", (0.12, 0.12, 0.14))
    minimum, maximum = _bounds_m(part)
    _author_box_collider(stage, f"{assembly_path}/BracketCollision", minimum, maximum)
    camera_path = _author_d405(
        stage,
        f"{assembly_path}/D405",
        manifest,
        HEAD_CAMERA_ROTATION,
        HEAD_CAMERA_TRANSLATION_M,
        "head",
    )
    return assembly_path, camera_path


def _author_knife(
    stage: Usd.Stage,
    link_path: str,
    manifest: dict,
) -> tuple[str, str, str, str]:
    root_path = f"{link_path}/attachments/DeleafKnife"
    root = UsdGeom.Xform.Define(stage, root_path)
    _set_transform(root.GetPrim(), KNIFE_ROTATION, KNIFE_TRANSLATION_M)
    _hardware_attr(root.GetPrim(), "hardwareRole", "deleafing_knife")

    knife = _part(manifest, "deleaf_knife")
    components = {component["name"]: component for component in knife["components"]}
    blade = components["deleaf_knife_blade"]

    blade_path = f"{root_path}/Blade"
    blade_mesh = _author_mesh(stage, blade_path, DERIVED_DIR / "deleaf_knife_blade.stl", (0.58, 0.61, 0.64))
    _hardware_attr(blade_mesh.GetPrim(), "hardwareRole", "blade_plate")
    _hardware_attr(blade_mesh.GetPrim(), "cuttingSurface", False)
    blade_min = np.asarray(blade["min_mm"], dtype=np.float64) * 0.001
    blade_max = np.asarray(blade["max_mm"], dtype=np.float64) * 0.001
    blade_collision = _author_box_collider(stage, f"{root_path}/BladeCollision", blade_min, blade_max)
    blade_collision_prim = stage.GetPrimAtPath(blade_collision)
    _hardware_attr(blade_collision_prim, "hardwareRole", "blade_plate")
    _hardware_attr(blade_collision_prim, "cuttingSurface", False)

    # Only the distal two millimetres of the flat plate are the leading edge.
    # This non-colliding semantic volume is evaluated against contact points
    # from the full physical plate collider, avoiding duplicate/overlapping
    # PhysX shapes while ensuring a face or arc scrape cannot sever a petiole.
    edge_min = blade_min.copy()
    edge_max = blade_max.copy()
    edge_max[1] = min(edge_min[1] + CUTTING_EDGE_DEPTH_M, blade_max[1])
    edge_path = _author_box_collider(
        stage,
        f"{root_path}/CuttingEdge",
        edge_min,
        edge_max,
        collidable=False,
    )
    edge_prim = stage.GetPrimAtPath(edge_path)
    _hardware_attr(edge_prim, "hardwareRole", "cutting_edge")
    _hardware_attr(edge_prim, "cuttingSurface", True)
    _hardware_attr(edge_prim, "edgeDepthMillimeters", CUTTING_EDGE_DEPTH_M * 1000.0)
    edge_prim.CreateAttribute(
        "tomato:cuttingDirection", Sdf.ValueTypeNames.Float3, custom=True
    ).Set(Gf.Vec3f(0.0, -1.0, 0.0))
    edge_prim.CreateAttribute(
        "tomato:edgeAxis", Sdf.ValueTypeNames.Float3, custom=True
    ).Set(Gf.Vec3f(1.0, 0.0, 0.0))

    arc_path = f"{root_path}/Arc"
    arc_mesh = _author_mesh(stage, arc_path, DERIVED_DIR / "deleaf_knife_arc.stl", (0.16, 0.18, 0.20))
    _hardware_attr(arc_mesh.GetPrim(), "hardwareRole", "knife_support")
    _hardware_attr(arc_mesh.GetPrim(), "cuttingSurface", False)
    arc_collision = _author_decomposed_collider(
        stage,
        f"{root_path}/ArcCollision",
        DERIVED_DIR / "deleaf_knife_arc.stl",
    )
    arc_collision_prim = stage.GetPrimAtPath(arc_collision)
    _hardware_attr(arc_collision_prim, "hardwareRole", "knife_support")
    _hardware_attr(arc_collision_prim, "cuttingSurface", False)
    return root_path, blade_path, edge_path, arc_path


def _remove_original_right_gripper(stage: Usd.Stage, robot_root: str) -> tuple[str, ...]:
    """Remove rendered/contact geometry while retaining inert articulation frames."""
    removed: list[str] = []
    for link_name in RIGHT_GRIPPER_LINKS:
        link = stage.GetPrimAtPath(f"{robot_root}/{link_name}")
        if not link.IsValid():
            raise ValueError(f"RBY1-A v1.0 right gripper link is missing: {link_name}")
        link.CreateAttribute("tomato:originalGripperRemoved", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        for child_name in ("visuals", "collisions", "restored_collisions"):
            child = stage.GetPrimAtPath(f"{link.GetPath()}/{child_name}")
            if child.IsValid() and child.IsActive():
                child.SetActive(False)
                removed.append(str(child.GetPath()))
    stage.GetPrimAtPath(f"{robot_root}/ee_right").CreateAttribute(
        "tomato:toolConfiguration", Sdf.ValueTypeNames.String, custom=True
    ).Set("knife_only")
    return tuple(removed)


def attach_robot_hardware(stage: Usd.Stage, robot_root: str = ROBOT_ROOT) -> HardwareReport:
    """Attach all requested hardware to an imported RBY1-A v1.0 stage."""
    required = [END_EFFECTOR_LINKS["left"], END_EFFECTOR_LINKS["right"], HEAD_LINK]
    missing = [name for name in required if not stage.GetPrimAtPath(f"{robot_root}/{name}").IsValid()]
    if missing:
        raise ValueError(f"RBY1-A v1.0 attachment links are missing: {', '.join(missing)}")

    removed_right_gripper = _remove_original_right_gripper(stage, robot_root)
    manifest = load_manifest()
    attachments: list[str] = []
    cameras: list[str] = []
    for side, rotation, translation in (
        ("left", LEFT_CAMERA_ROTATION, LEFT_CAMERA_TRANSLATION_M),
        ("right", RIGHT_CAMERA_ROTATION, RIGHT_CAMERA_TRANSLATION_M),
    ):
        link_path = f"{robot_root}/{END_EFFECTOR_LINKS[side]}"
        attachment, camera = _author_wrist_camera(stage, link_path, side, manifest, rotation, translation)
        attachments.append(attachment)
        cameras.append(camera)

    head_attachment, head_camera = _author_head_camera(stage, f"{robot_root}/{HEAD_LINK}", manifest)
    attachments.append(head_attachment)
    cameras.append(head_camera)

    knife_root, blade, edge, arc = _author_knife(
        stage,
        f"{robot_root}/{END_EFFECTOR_LINKS['right']}",
        manifest,
    )
    attachments.append(knife_root)
    return HardwareReport(
        cameras=tuple(cameras),
        cutting_surfaces=(edge,),
        non_cutting_supports=(
            blade,
            f"{knife_root}/BladeCollision",
            arc,
            f"{knife_root}/ArcCollision",
        ),
        attachments=tuple(attachments),
        removed_right_gripper_prims=removed_right_gripper,
    )
