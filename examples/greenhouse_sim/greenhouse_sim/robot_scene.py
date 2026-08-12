"""Place the fitted RB-Y1 asset in the greenhouse at a safe ready pose."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from greenhouse_sim import usd_env

usd_env.ensure_pxr()

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_ASSET = REPOSITORY_ROOT / "data" / "greenhouse_sim" / "robots" / "rby1a_v1.0.usd"
DEFAULT_ROBOT_PATH = "/World/RBY1"
# Stand on the opposite (+Y) side of Vine_0000, facing back toward the row. The
# 150 mm rearward offset keeps a released branch clear of the wheels. Its plan
# footprint extends beneath the supplied neighbouring elevated gutter, while
# the validated 3-D scene has no robot/gutter collision.
# Main_Cultivation_Zone's authored collision floor is z=-0.3050817 m.
DEFAULT_POSITION_M = np.array([6.99114, 3.9300, -0.3050817], dtype=np.float64)
DEFAULT_YAW_DEGREES = -90.0
DEFAULT_POSE_NAME = "opposite_aisle_lower_gutter_safe_stow"

# The official Model A ready pose used by the SDK's multi-control and leader-arm
# examples.  Angular drive and PhysX joint-state attributes are in degrees.
SDK_READY_POSE_DEGREES = {
    **{name: value for name, value in zip(
        (f"torso_{index}" for index in range(6)),
        (0.0, 45.0, -90.0, 45.0, 0.0, 0.0),
        strict=True,
    )},
    **{name: value for name, value in zip(
        (f"right_arm_{index}" for index in range(7)),
        (0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0),
        strict=True,
    )},
    **{name: value for name, value in zip(
        (f"left_arm_{index}" for index in range(7)),
        (0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0),
        strict=True,
    )},
    "head_0": 0.0,
    "head_1": 0.0,
}

# Historical deterministic-route seed for the old -90 degree station. At the
# current +90 degree station this pose folds the knife arm backward, so it must
# not be used as the generic startup state.
# This numerical IK from the exact RB-Y1 Model A v1.0 URDF preserves the
# knife's flat cutting orientation while moving its root 150 mm rearward and
# 100 mm upward from the old high-gutter precontact pose. The old pose overlaps
# upper foliage when used with gh_tomato_test's lower gutter. The torso, left
# arm, and head retain the official SDK ready vector.
GREENHOUSE_PRECONTACT_RIGHT_ARM_DEGREES = (
    -146.58120585198483,
    -83.52711902452849,
    44.100760676248115,
    -118.2726158812136,
    -84.20799011233542,
    109.99942703110308,
    -67.89880648450034,
)
# Start interactive and online-RL runs from the symmetric Model A SDK pose.
# Scripted planners compute task approaches from this measured state.
READY_POSE_DEGREES = dict(SDK_READY_POSE_DEGREES)



@dataclasses.dataclass(frozen=True)
class RobotPlacement:
    root_path: str
    asset: pathlib.Path
    position_m: tuple[float, float, float]
    yaw_degrees: float
    pose_name: str
    initialized_joints: tuple[str, ...]
    cameras: tuple[str, ...]
    knife_blade: str
    knife_cutting_edge: str
    knife_support: str
    right_gripper_removed: bool


def _set_root_pose(prim: Usd.Prim, position_m: np.ndarray, yaw_degrees: float) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position_m.tolist()))
    xform.AddRotateZOp().Set(float(yaw_degrees))


def _initialize_ready_pose(stage: Usd.Stage, root_path: str) -> tuple[str, ...]:
    # PhysxSchema is extension-provided and becomes importable after
    # SimulationApp starts; keeping this import local leaves the pure frame
    # helpers testable with the standalone USD libraries.
    from pxr import PhysxSchema  # noqa: PLC0415

    initialized: list[str] = []
    for name, target_degrees in READY_POSE_DEGREES.items():
        joint_path = f"{root_path}/joints/{name}"
        joint = stage.GetPrimAtPath(joint_path)
        if not joint.IsValid():
            raise ValueError(f"fitted robot is missing ready-pose joint {joint_path}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise ValueError(f"fitted robot joint has no angular drive: {joint_path}")
        drive.CreateTargetPositionAttr(float(target_degrees))
        drive.CreateTargetVelocityAttr(0.0)
        state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.CreatePositionAttr().Set(float(target_degrees))
        state.CreateVelocityAttr().Set(0.0)
        initialized.append(joint_path)
    return tuple(initialized)


def _anchor_stationary_base(stage: Usd.Stage, root_path: str) -> str:
    """Fix the parked mobile base to its authored benchmark transform."""
    base_path = f"{root_path}/base"
    base = stage.GetPrimAtPath(base_path)
    if not base.IsValid():
        raise ValueError(f"fitted robot is missing base rigid body {base_path}")
    world = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    joint_path = f"{root_path}/joints/benchmark_world_fixed"
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody1Rel().SetTargets([base_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(world.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    return joint_path


def add_fitted_robot(
    stage: Usd.Stage,
    asset: pathlib.Path = DEFAULT_ROBOT_ASSET,
    *,
    root_path: str = DEFAULT_ROBOT_PATH,
    position_m: tuple[float, float, float] | np.ndarray = DEFAULT_POSITION_M,
    yaw_degrees: float = DEFAULT_YAW_DEGREES,
) -> RobotPlacement:
    """Reference the fitted robot and initialize it without a startup sweep."""
    asset = pathlib.Path(asset).resolve()
    if not asset.exists():
        raise FileNotFoundError(f"fitted robot asset not found: {asset}; run build_robot.py")
    if stage.GetPrimAtPath(root_path).IsValid():
        raise ValueError(f"robot placement already exists: {root_path}")

    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    if not root.GetPrim().GetReferences().AddReference(str(asset)):
        raise RuntimeError(f"could not reference fitted robot asset: {asset}")
    position = np.asarray(position_m, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("robot position must contain exactly three values")
    _set_root_pose(root.GetPrim(), position, yaw_degrees)

    _anchor_stationary_base(stage, root_path)
    initialized = _initialize_ready_pose(stage, root_path)
    cameras = (
        f"{root_path}/link_head_2/attachments/HeadCamera/D405/DepthCamera",
        f"{root_path}/ee_left/attachments/LeftWristCamera/D405/DepthCamera",
        f"{root_path}/ee_right/attachments/RightWristCamera/D405/DepthCamera",
    )
    for camera in cameras:
        if not stage.GetPrimAtPath(camera).IsA(UsdGeom.Camera):
            raise ValueError(f"fitted D405 camera is missing: {camera}")

    blade = f"{root_path}/ee_right/attachments/DeleafKnife/Blade"
    cutting_edge = f"{root_path}/ee_right/attachments/DeleafKnife/CuttingEdge"
    support = f"{root_path}/ee_right/attachments/DeleafKnife/Arc"
    if stage.GetPrimAtPath(blade).GetAttribute("tomato:cuttingSurface").Get() is not False:
        raise ValueError("the knife plate body must not be treated as its leading edge")
    if stage.GetPrimAtPath(cutting_edge).GetAttribute("tomato:cuttingSurface").Get() is not True:
        raise ValueError("the flat knife leading edge is not marked as the cutting surface")
    if stage.GetPrimAtPath(support).GetAttribute("tomato:cuttingSurface").Get() is not False:
        raise ValueError("the curved knife support must be explicitly non-cutting")
    right_gripper_visuals_removed = all(
        stage.GetPrimAtPath(f"{root_path}/{link}/visuals").IsValid()
        and not stage.GetPrimAtPath(f"{root_path}/{link}/visuals").IsActive()
        for link in ("ee_right", "ee_finger_r1", "ee_finger_r2")
    )
    right_collision_scopes = tuple(
        stage.GetPrimAtPath(f"{root_path}/{link}/{scope}")
        for link in ("ee_right", "ee_finger_r1", "ee_finger_r2")
        for scope in ("collisions", "restored_collisions")
        if stage.GetPrimAtPath(f"{root_path}/{link}/{scope}").IsValid()
    )
    right_gripper_removed = right_gripper_visuals_removed and all(
        not prim.IsActive() for prim in right_collision_scopes
    )
    right_tool_configuration = stage.GetPrimAtPath(f"{root_path}/ee_right").GetAttribute(
        "tomato:toolConfiguration"
    ).Get()
    if not right_gripper_removed or right_tool_configuration != "knife_only":
        raise ValueError("the original right gripper geometry/contact is still active")

    return RobotPlacement(
        root_path=root_path,
        asset=asset,
        position_m=tuple(float(value) for value in position),
        yaw_degrees=float(yaw_degrees),
        pose_name=DEFAULT_POSE_NAME,
        initialized_joints=initialized,
        cameras=cameras,
        knife_blade=blade,
        knife_cutting_edge=cutting_edge,
        knife_support=support,
        right_gripper_removed=right_gripper_removed,
    )
