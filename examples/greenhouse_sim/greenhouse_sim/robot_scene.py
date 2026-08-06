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
# Main_Cultivation_Zone's authored collision floor is z=-0.3050817 m.
DEFAULT_POSITION_M = np.array([6.8691, 2.0, -0.3050817], dtype=np.float64)
DEFAULT_YAW_DEGREES = 90.0

# The official Model A ready pose used by the SDK's multi-control and leader-arm
# examples.  Angular drive and PhysX joint-state attributes are in degrees.
READY_POSE_DEGREES = {
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


@dataclasses.dataclass(frozen=True)
class RobotPlacement:
    root_path: str
    asset: pathlib.Path
    position_m: tuple[float, float, float]
    yaw_degrees: float
    initialized_joints: tuple[str, ...]
    cameras: tuple[str, ...]
    knife_blade: str
    knife_support: str


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
    support = f"{root_path}/ee_right/attachments/DeleafKnife/Arc"
    if stage.GetPrimAtPath(blade).GetAttribute("tomato:cuttingSurface").Get() is not True:
        raise ValueError("the flat knife blade is not marked as a cutting surface")
    if stage.GetPrimAtPath(support).GetAttribute("tomato:cuttingSurface").Get() is not False:
        raise ValueError("the curved knife support must be explicitly non-cutting")

    return RobotPlacement(
        root_path=root_path,
        asset=asset,
        position_m=tuple(float(value) for value in position),
        yaw_degrees=float(yaw_degrees),
        initialized_joints=initialized,
        cameras=cameras,
        knife_blade=blade,
        knife_support=support,
    )
