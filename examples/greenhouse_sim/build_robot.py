"""Build the exact RB-Y1 Model A v1.0 USD with deleafing hardware.

The Isaac URDF importer does not understand RB-Y1's non-standard ``capsule``
elements.  This builder imports the licensed v1.0 model, restores all 17 active
declared link capsules, adds conservative mobile-base contacts, changes the two
wheel joints to velocity drives, and attaches the supplied knife/brackets/D405s.

Run from the repository root:

    D:\\isaac-sim\\python.bat examples\\greenhouse_sim\\build_robot.py
"""

# SimulationApp must exist before importing omni or pxr.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_DEFAULT_URDF = pathlib.Path("third_party/rby1-sdk/models/rby1a/urdf/model_v1.0.urdf")
_DEFAULT_OUTPUT = pathlib.Path("data/greenhouse_sim/robots/rby1a_v1.0.usd")
_DEFAULT_REPORT = pathlib.Path("data/greenhouse_sim/robots/rby1a_v1.0.json")
_EXPECTED_ROOT = "/RBY1_A_v1_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=pathlib.Path, default=_DEFAULT_URDF)
    parser.add_argument("--output", type=pathlib.Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--report", type=pathlib.Path, default=_DEFAULT_REPORT)
    return parser.parse_args()


def _numbers(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    parsed = tuple(float(item) for item in value.split())
    if len(parsed) != 3:
        raise ValueError(f"expected three values, found {value!r}")
    return parsed


def _restore_urdf_capsules(stage, urdf_path: pathlib.Path, robot_hardware, Gf, Sdf, UsdGeom, UsdPhysics) -> list[dict]:
    root = ET.parse(urdf_path).getroot()
    restored: list[dict] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        link_path = f"{_EXPECTED_ROOT}/{link_name}"
        if not stage.GetPrimAtPath(link_path).IsValid():
            raise RuntimeError(f"imported robot is missing link {link_path}")
        for index, collision in enumerate(link.findall("collision")):
            capsule_element = collision.find("geometry/capsule")
            if capsule_element is None:
                continue
            radius = float(capsule_element.attrib["radius"])
            length = float(capsule_element.attrib["length"])
            origin = collision.find("origin")
            xyz = _numbers(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
            rpy = _numbers(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))

            # The importer's empty collision scope is an instanceable
            # reference, so children there would be instance proxies. A
            # sibling scope belongs to the same rigid-body link and remains
            # authorable.
            path = f"{link_path}/restored_collisions/capsule_{index:02d}"
            capsule = UsdGeom.Capsule.Define(stage, Sdf.Path(path))
            capsule.CreateAxisAttr(UsdGeom.Tokens.z)
            capsule.CreateRadiusAttr(radius)
            capsule.CreateHeightAttr(length)
            capsule.CreatePurposeAttr(UsdGeom.Tokens.guide)
            robot_hardware._set_transform(  # kept in one matrix to preserve URDF RPY order
                capsule.GetPrim(),
                robot_hardware.compose_rotation_xyz(rpy),
                np.asarray(xyz, dtype=np.float64),
            )
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            capsule.GetPrim().CreateAttribute("tomato:restoredFromUrdf", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            restored.append(
                {
                    "path": path,
                    "link": link_name,
                    "radius_m": radius,
                    "cylinder_length_m": length,
                    "xyz_m": xyz,
                    "rpy_rad": rpy,
                }
            )
    return restored


def _add_mobile_base_colliders(stage, Gf, Sdf, UsdGeom, UsdPhysics) -> list[str]:
    """Add contacts absent from the source URDF for the chassis and wheels."""
    paths: list[str] = []

    # Keep the chassis proxy 20 mm above the wheel tangent so the differential
    # wheels, rather than a large friction box, carry the robot on flat ground.
    base_path = f"{_EXPECTED_ROOT}/base/restored_collisions/chassis_proxy"
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(base_path))
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    minimum = np.array([-0.325, -0.250, 0.020])
    maximum = np.array([0.295, 0.250, 0.345])
    xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (minimum + maximum))))
    xform.AddScaleOp().Set(Gf.Vec3f(*(maximum - minimum)))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().CreateAttribute("tomato:collisionRole", Sdf.ValueTypeNames.String, custom=True).Set("chassis")
    paths.append(base_path)

    for side in ("r", "l"):
        wheel_path = f"{_EXPECTED_ROOT}/wheel_{side}/restored_collisions/wheel_proxy"
        wheel = UsdGeom.Cylinder.Define(stage, Sdf.Path(wheel_path))
        wheel.CreateAxisAttr(UsdGeom.Tokens.y)
        wheel.CreateRadiusAttr(0.100)
        wheel.CreateHeightAttr(0.050)
        wheel.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(wheel.GetPrim())
        wheel.GetPrim().CreateAttribute("tomato:collisionRole", Sdf.ValueTypeNames.String, custom=True).Set("wheel")
        paths.append(wheel_path)
    return paths


def _add_gripper_colliders(stage, Gf, Sdf, UsdGeom, UsdPhysics) -> list[str]:
    """Restore contact on the retained left gripper only."""
    body_min = np.array([-0.0630, -0.0325, -0.0730])
    body_max = np.array([0.0630, 0.0325, 0.0])
    finger_min = np.array([-0.0030, -0.0160, -0.0605])
    finger_max = np.array([0.0130, 0.0160, 0.0015])
    links = {
        "ee_left": (body_min, body_max, "gripper_body"),
        "ee_finger_l1": (finger_min, finger_max, "finger"),
        "ee_finger_l2": (finger_min, finger_max, "finger"),
    }
    paths: list[str] = []
    for link, (minimum, maximum, role) in links.items():
        path = f"{_EXPECTED_ROOT}/{link}/restored_collisions/contact_proxy"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (minimum + maximum))))
        xform.AddScaleOp().Set(Gf.Vec3f(*(maximum - minimum)))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().CreateAttribute("tomato:collisionRole", Sdf.ValueTypeNames.String, custom=True).Set(role)
        paths.append(path)
    return paths


def _configure_wheel_drives(stage, UsdPhysics) -> list[dict]:
    drives: list[dict] = []
    for name in ("left_wheel", "right_wheel"):
        joint = stage.GetPrimAtPath(f"{_EXPECTED_ROOT}/joints/{name}")
        if not joint.IsValid():
            raise RuntimeError(f"imported robot is missing wheel joint {name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(40.0)
        drive.CreateMaxForceAttr(120.0)
        drive.CreateTargetVelocityAttr(0.0)
        drives.append(
            {
                "joint": str(joint.GetPath()),
                "mode": "velocity",
                "stiffness": 0.0,
                "damping": 40.0,
                "max_force_nm": 120.0,
            }
        )
    return drives


def main() -> int:
    args = parse_args()
    urdf_path = args.urdf.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if not urdf_path.exists():
        print(f"URDF not found: {urdf_path}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import omni.kit.commands
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    from greenhouse_sim import robot_hardware

    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        app.close()
        raise RuntimeError("Isaac Sim could not create a URDF import configuration")
    config.merge_fixed_joints = False
    config.convex_decomp = False
    config.import_inertia_tensor = True
    config.fix_base = False
    config.distance_scale = 1.0
    config.self_collision = False
    config.create_physics_scene = False
    config.make_default_prim = True

    status, imported_root = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=config,
        dest_path=str(output_path),
        get_articulation_root=False,
    )
    for _ in range(8):
        app.update()
    if not status:
        app.close()
        raise RuntimeError(f"Isaac Sim failed to import {urdf_path}")

    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        app.close()
        raise RuntimeError(f"Isaac Sim reported success but did not write {output_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid() or str(default_prim.GetPath()) != _EXPECTED_ROOT:
        app.close()
        raise RuntimeError(f"unexpected imported root: {default_prim.GetPath() if default_prim else imported_root}")
    stage.SetEditTarget(stage.GetRootLayer())

    capsules = _restore_urdf_capsules(stage, urdf_path, robot_hardware, Gf, Sdf, UsdGeom, UsdPhysics)
    expected_capsules = sum(
        1
        for link in ET.parse(urdf_path).getroot().findall("link")
        for collision in link.findall("collision")
        if collision.find("geometry/capsule") is not None
    )
    if len(capsules) != expected_capsules:
        app.close()
        raise RuntimeError(f"expected {expected_capsules} RB-Y1 capsules, restored {len(capsules)}")
    mobile_colliders = _add_mobile_base_colliders(stage, Gf, Sdf, UsdGeom, UsdPhysics)
    gripper_colliders = _add_gripper_colliders(stage, Gf, Sdf, UsdGeom, UsdPhysics)
    wheel_drives = _configure_wheel_drives(stage, UsdPhysics)
    hardware = robot_hardware.attach_robot_hardware(stage, _EXPECTED_ROOT)

    stage.GetRootLayer().Save()
    collision_paths = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    camera_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
    report = {
        "schema_version": 1,
        "asset": str(output_path),
        "source_urdf": str(urdf_path),
        "source_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "imported_root": str(default_prim.GetPath()),
        "import_settings": {
            "fix_base": False,
            "merge_fixed_joints": False,
            "self_collision": False,
            "import_inertia_tensor": True,
        },
        "restored_urdf_capsules": capsules,
        "mobile_base_colliders": mobile_colliders,
        "gripper_colliders": gripper_colliders,
        "wheel_drives": wheel_drives,
        "hardware_attachments": list(hardware.attachments),
        "removed_right_gripper_prims": list(hardware.removed_right_gripper_prims),
        "cameras": camera_paths,
        "cutting_surfaces": list(hardware.cutting_surfaces),
        "non_cutting_supports": list(hardware.non_cutting_supports),
        "collision_count": len(collision_paths),
        "collision_paths": collision_paths,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"robot:       {output_path}")
    print(f"root:        {default_prim.GetPath()}")
    print(f"capsules:    {len(capsules)} restored from URDF")
    print(f"base shapes: {len(mobile_colliders)}")
    print(f"EE shapes:   {len(gripper_colliders)}")
    print(f"right tool:  knife only ({len(hardware.removed_right_gripper_prims)} gripper scopes removed)")
    print(f"cameras:     {len(camera_paths)}")
    print(f"colliders:   {len(collision_paths)}")
    print(f"report:      {report_path}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
