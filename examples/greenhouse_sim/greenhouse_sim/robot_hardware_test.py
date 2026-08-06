"""Regression tests for the RBY1 deleafing hardware frames and semantics."""

from __future__ import annotations

import numpy as np
import pytest

from greenhouse_sim import robot_hardware


def test_knife_components_match_the_supplied_geometry() -> None:
    blade = robot_hardware.read_binary_stl(robot_hardware.DERIVED_DIR / "deleaf_knife_blade.stl")
    arc = robot_hardware.read_binary_stl(robot_hardware.DERIVED_DIR / "deleaf_knife_arc.stl")

    assert len(blade.triangles) == 1029
    assert len(arc.triangles) == 3811
    blade_min, blade_max = blade.bounds
    arc_min, arc_max = arc.bounds
    np.testing.assert_allclose(blade_max - blade_min, [0.03036205, 0.07147998, 0.013], atol=1e-7)
    np.testing.assert_allclose(arc_max - arc_min, [0.006, 0.06230074, 0.05091788], atol=1e-7)


def test_manifest_marks_only_the_flat_plate_as_cutting() -> None:
    manifest = robot_hardware.load_manifest()
    knife = next(part for part in manifest["parts"] if part["part"] == "deleaf_knife")
    semantics = {component["name"]: component["cutting_surface"] for component in knife["components"]}

    assert semantics == {"deleaf_knife_blade": True, "deleaf_knife_arc": False}
    assert "flat straight blade only" in knife["cut_semantics"]


def test_knife_projects_along_the_right_tool_axis() -> None:
    # CAD -Y is the blade length, and CAD +Z points toward the support arc.
    projection = robot_hardware.transform_direction(robot_hardware.KNIFE_ROTATION, (0.0, -1.0, 0.0))
    support_side = robot_hardware.transform_direction(robot_hardware.KNIFE_ROTATION, (0.0, 0.0, 1.0))

    np.testing.assert_allclose(projection, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(support_side, [1.0, 0.0, 0.0], atol=1e-12)


def test_wrist_cameras_look_down_and_outward() -> None:
    manifest = robot_hardware.load_manifest()
    camera_rotation = np.asarray(manifest["mounts"]["camera_bracket_to_d405"]["rotation_matrix"])
    camera_forward = np.array([0.0, 1.0, 0.0])
    left = robot_hardware.LEFT_CAMERA_ROTATION @ camera_rotation @ camera_forward
    right = robot_hardware.RIGHT_CAMERA_ROTATION @ camera_rotation @ camera_forward

    assert left[1] > 0.3 and left[2] < -0.9
    assert right[1] < -0.3 and right[2] < -0.9
    np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=1e-9)


def test_head_camera_faces_forward_through_bracket_window() -> None:
    head_forward = robot_hardware.HEAD_BRACKET_ROTATION @ robot_hardware.HEAD_CAMERA_ROTATION @ np.array(
        [0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(head_forward, [1.0, 0.0, 0.0], atol=1e-12)

    # The supplied body puts its optical origin 19.23 mm forward of its frame.
    # Rz(180) moves that point from y=+1.23 mm to the bracket's y=-18 mm face.
    optical_in_bracket = robot_hardware.HEAD_CAMERA_TRANSLATION_M + robot_hardware.HEAD_CAMERA_ROTATION @ np.array(
        [0.0, 0.01923, 0.0]
    )
    assert optical_in_bracket[1] == pytest.approx(-0.018, abs=1e-9)


def test_authored_stage_has_three_cameras_and_one_cutting_part() -> None:
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, robot_hardware.ROBOT_ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    for link in ("ee_left", "ee_right", "link_head_2"):
        UsdGeom.Xform.Define(stage, f"{robot_hardware.ROBOT_ROOT}/{link}")

    report = robot_hardware.attach_robot_hardware(stage)

    assert len(report.cameras) == 3
    assert all(stage.GetPrimAtPath(path).IsA(UsdGeom.Camera) for path in report.cameras)
    cutting = [
        prim
        for prim in stage.Traverse()
        if prim.HasAttribute("tomato:cuttingSurface") and prim.GetAttribute("tomato:cuttingSurface").Get()
    ]
    assert {str(prim.GetPath()) for prim in cutting} == set(report.cutting_surfaces)
    assert all(prim.GetAttribute("tomato:hardwareRole").Get() == "blade" for prim in cutting)
    assert all(stage.GetPrimAtPath(path).IsValid() for path in report.non_cutting_supports)
    assert all(not stage.GetPrimAtPath(path).GetAttribute("tomato:cuttingSurface").Get() for path in report.non_cutting_supports)
    assert Sdf.Path(report.attachments[-1]).IsAbsolutePath()

    camera_axes = {}
    for path in report.cameras:
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_axes[path] = (
            np.asarray(matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))),
            np.asarray(matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))),
        )
    head_forward, head_up = camera_axes[report.cameras[2]]
    left_forward, left_up = camera_axes[report.cameras[0]]
    right_forward, right_up = camera_axes[report.cameras[1]]
    np.testing.assert_allclose(head_forward, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(head_up, [0.0, 0.0, 1.0], atol=1e-9)
    assert left_forward[1] > 0.3 and left_forward[2] < -0.9
    assert right_forward[1] < -0.3 and right_forward[2] < -0.9
    assert left_up[0] > 0.9 and right_up[0] > 0.9
    np.testing.assert_allclose(left_up, right_up, atol=1e-9)
    assert stage.GetPrimAtPath(report.cameras[0]).GetAttribute("tomato:sensorRollDegrees").Get() == 0.0
    assert stage.GetPrimAtPath(report.cameras[1]).GetAttribute("tomato:sensorRollDegrees").Get() == 180.0

    blade_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(report.cutting_surfaces[0])).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    blade_projection = np.asarray(blade_matrix.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)))
    np.testing.assert_allclose(blade_projection, [0.0, 0.0, -1.0], atol=1e-9)
