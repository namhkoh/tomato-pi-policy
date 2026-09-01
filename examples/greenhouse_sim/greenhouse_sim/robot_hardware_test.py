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

    blade_centre, blade_half_extents = robot_hardware.knife_blade_box()
    arc_centre, arc_half_extents = robot_hardware.knife_support_box()
    np.testing.assert_allclose(
        2.0 * blade_half_extents,
        blade_max - blade_min,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        2.0 * arc_half_extents,
        arc_max - arc_min,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        blade_centre,
        0.5 * (blade_min + blade_max),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        arc_centre,
        0.5 * (arc_min + arc_max),
        atol=1e-7,
    )

    support_boxes = robot_hardware.knife_support_boxes()
    assert len(support_boxes) == 6
    for point in arc.points:
        assert any(
            np.all(np.abs(point - centre) <= half_extents + 1e-12)
            for centre, half_extents in support_boxes
        )


def test_manifest_marks_only_the_flat_plate_as_cutting() -> None:
    manifest = robot_hardware.load_manifest()
    knife = next(part for part in manifest["parts"] if part["part"] == "deleaf_knife")
    semantics = {component["name"]: component["cutting_surface"] for component in knife["components"]}

    assert semantics == {"deleaf_knife_blade": True, "deleaf_knife_arc": False}
    assert "flat straight blade only" in knife["cut_semantics"]


def test_knife_cutting_side_projects_along_the_right_tool_axis() -> None:
    cut_direction = robot_hardware.transform_direction(
        robot_hardware.KNIFE_ROTATION,
        robot_hardware.KNIFE_CUT_DIRECTION_LOCAL,
    )
    support_side = robot_hardware.transform_direction(
        robot_hardware.KNIFE_ROTATION, (0.0, 0.0, 1.0)
    )

    np.testing.assert_allclose(cut_direction, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(support_side, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(robot_hardware.KNIFE_TRANSLATION_M, [0.0, 0.0, 0.0], atol=0.0)


def test_right_wrist_camera_trails_the_flat_cutting_edge() -> None:
    blade_centre, blade_half = robot_hardware.knife_blade_box()
    leading_edge = blade_centre.copy()
    leading_edge[0] -= blade_half[0]
    leading_edge_ee = (
        robot_hardware.KNIFE_TRANSLATION_M
        + robot_hardware.KNIFE_ROTATION @ leading_edge
    )
    cut_direction = (
        robot_hardware.KNIFE_ROTATION
        @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
    )

    for centre, rotation, half_extents in (
        robot_hardware.wrist_d405_body_box(side="right"),
        robot_hardware.wrist_camera_bracket_box(side="right"),
    ):
        forward_centre = float(
            np.dot(centre - leading_edge_ee, cut_direction)
        )
        forward_radius = sum(
            abs(float(np.dot(rotation[:, axis], cut_direction)))
            * half_extents[axis]
            for axis in range(3)
        )
        assert forward_centre + forward_radius < 0.0


def test_flat_blade_long_cutting_side_is_outside_support() -> None:
    blade_centre, blade_half = robot_hardware.knife_blade_box()
    support_centre, support_half = robot_hardware.knife_support_box()
    blade_min_x = float(blade_centre[0] - blade_half[0])
    edge_max_x = blade_min_x + robot_hardware.CUTTING_EDGE_DEPTH_M
    support_min_x = float(support_centre[0] - support_half[0])

    assert edge_max_x < support_min_x


def test_runtime_mount_synchronization_repairs_a_stale_generated_asset() -> None:
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, robot_hardware.ROBOT_ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    for link in (
        "ee_left",
        "ee_right",
        "ee_finger_r1",
        "ee_finger_r2",
        "link_head_2",
    ):
        UsdGeom.Xform.Define(
            stage, f"{robot_hardware.ROBOT_ROOT}/{link}"
        )
    for link in robot_hardware.RIGHT_GRIPPER_LINKS:
        UsdGeom.Xform.Define(
            stage, f"{robot_hardware.ROBOT_ROOT}/{link}/visuals"
        )
        UsdGeom.Xform.Define(
            stage, f"{robot_hardware.ROBOT_ROOT}/{link}/collisions"
        )
    robot_hardware.attach_robot_hardware(stage)
    knife_path = (
        f"{robot_hardware.ROBOT_ROOT}/ee_right/attachments/DeleafKnife"
    )
    knife = UsdGeom.Xformable(stage.GetPrimAtPath(knife_path))
    stale_rotation = (
        robot_hardware.rotation_z(90.0)
        @ robot_hardware.rotation_x(90.0)
    )
    knife.GetOrderedXformOps()[0].Set(
        robot_hardware._gf_matrix(
            stale_rotation, robot_hardware.KNIFE_TRANSLATION_M
        )
    )

    result = robot_hardware.synchronize_fitted_hardware_mounts(
        stage, robot_hardware.ROBOT_ROOT
    )

    assert result["corrected_count"] == 1
    knife_record = next(
        record
        for record in result["mounts"]
        if record["name"] == "deleafing_knife"
    )
    assert knife_record["corrected"]
    assert np.isclose(
        knife_record["prior_orientation_error_rad"], np.pi
    )
    matrix = UsdGeom.Xformable(
        stage.GetPrimAtPath(knife_path)
    ).GetLocalTransformation()
    corrected_rotation = np.asarray(
        matrix, dtype=np.float64
    ).T[:3, :3]
    np.testing.assert_allclose(
        corrected_rotation, robot_hardware.KNIFE_ROTATION, atol=1e-12
    )
    second = robot_hardware.synchronize_fitted_hardware_mounts(
        stage, robot_hardware.ROBOT_ROOT
    )
    assert second["corrected_count"] == 0


def test_cut_aligned_knife_is_transverse_and_keeps_support_up() -> None:
    target = np.asarray([0.42, -0.64, -0.65])
    preferred = np.asarray([0.0, -0.966, 0.259])

    rotation = robot_hardware.cut_aligned_knife_rotation(target, preferred)
    edge_axis = rotation @ robot_hardware.KNIFE_EDGE_AXIS_LOCAL
    cut_direction = rotation @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
    support = rotation @ np.asarray([0.0, 0.0, 1.0])
    target /= np.linalg.norm(target)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) > 0.999999
    assert abs(float(np.dot(edge_axis, target))) < 1e-12
    assert abs(float(np.dot(cut_direction, target))) < 1e-12
    assert float(np.dot(cut_direction, preferred)) > 0.0
    assert support[2] > 0.0


def test_rotation_y_preserves_a_right_handed_tool_frame() -> None:
    rotation = robot_hardware.rotation_y(-35.0)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_wrist_camera_brackets_align_the_reference_m3_pair() -> None:
    right = (
        robot_hardware.RIGHT_CAMERA_ROTATION
        @ robot_hardware.WRIST_BRACKET_BOLT_CENTRES_M.T
    ).T + robot_hardware.RIGHT_CAMERA_TRANSLATION_M
    left = (
        robot_hardware.LEFT_CAMERA_ROTATION
        @ robot_hardware.WRIST_BRACKET_BOLT_CENTRES_M.T
    ).T + robot_hardware.LEFT_CAMERA_TRANSLATION_M

    np.testing.assert_allclose(
        right, robot_hardware.WRIST_REFERENCE_BOLT_CENTRES_M, atol=1e-12
    )
    np.testing.assert_allclose(
        left, robot_hardware.LEFT_WRIST_REFERENCE_BOLT_CENTRES_M, atol=1e-12
    )
    np.testing.assert_allclose(
        robot_hardware.LEFT_CAMERA_ROTATION,
        robot_hardware.rotation_z(180.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        robot_hardware.RIGHT_CAMERA_ROTATION, np.eye(3), atol=0.0
    )

def test_wrist_camera_mount_preserves_the_freecad_ee_component_frame() -> None:
    assert robot_hardware.WRIST_REFERENCE_BOLT_CENTRES_M[0, 2] == pytest.approx(
        0.0385,
        abs=1e-12,
    )
    assert robot_hardware.WRIST_REFERENCE_BOLT_CENTRES_M[0, 1] == pytest.approx(
        robot_hardware.WRIST_MOUNT_FACE_Y_M
        - robot_hardware.WRIST_BOLT_HEAD_STANDOFF_M,
        abs=1e-12,
    )


def test_wrist_camera_bracket_box_matches_authored_mount() -> None:
    right_centre, right_rotation, half_extents = (
        robot_hardware.wrist_camera_bracket_box(side="right")
    )
    left_centre, left_rotation, left_half_extents = (
        robot_hardware.wrist_camera_bracket_box(side="left")
    )

    np.testing.assert_allclose(right_rotation, np.eye(3), atol=0.0)
    np.testing.assert_allclose(
        left_rotation, robot_hardware.rotation_z(180.0), atol=1e-12
    )
    np.testing.assert_allclose(left_half_extents, half_extents, atol=0.0)
    np.testing.assert_allclose(
        2.0 * half_extents, [0.027, 0.05991954, 0.03461918], atol=1e-9
    )
    np.testing.assert_allclose(
        left_centre, robot_hardware.rotation_z(180.0) @ right_centre, atol=1e-12
    )
    assert right_centre[1] + half_extents[1] == pytest.approx(
        robot_hardware.WRIST_MOUNT_FACE_Y_M,
        abs=5e-9,
    )
    assert left_centre[1] - left_half_extents[1] == pytest.approx(
        -robot_hardware.WRIST_MOUNT_FACE_Y_M,
        abs=5e-9,
    )

    assert right_centre[2] + half_extents[2] == pytest.approx(
        robot_hardware.WRIST_SENSOR_PLATE_BASE_Z_M,
        abs=5e-9,
    )


def test_wrist_camera_cad_axes_match_the_reference_mount() -> None:
    manifest = robot_hardware.load_manifest()
    camera_rotation = np.asarray(
        manifest["mounts"]["camera_bracket_to_d405"]["rotation_matrix"]
    )
    camera_forward = np.array([0.0, 1.0, 0.0])
    left = robot_hardware.LEFT_CAMERA_ROTATION @ camera_rotation @ camera_forward
    right = robot_hardware.RIGHT_CAMERA_ROTATION @ camera_rotation @ camera_forward

    assert left[1] < -0.3 and left[2] < -0.9
    assert right[1] > 0.3 and right[2] < -0.9
    np.testing.assert_allclose(left[2], right[2], atol=1e-12)


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
    from pxr import UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, robot_hardware.ROBOT_ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    for link in ("ee_left", "ee_right", "ee_finger_r1", "ee_finger_r2", "link_head_2"):
        UsdGeom.Xform.Define(stage, f"{robot_hardware.ROBOT_ROOT}/{link}")
    for link in robot_hardware.RIGHT_GRIPPER_LINKS:
        UsdGeom.Xform.Define(stage, f"{robot_hardware.ROBOT_ROOT}/{link}/visuals")
        UsdGeom.Xform.Define(stage, f"{robot_hardware.ROBOT_ROOT}/{link}/collisions")

    report = robot_hardware.attach_robot_hardware(stage)

    assert len(report.cameras) == 3
    assert all(stage.GetPrimAtPath(path).IsA(UsdGeom.Camera) for path in report.cameras)
    cutting = [
        prim
        for prim in stage.Traverse()
        if prim.HasAttribute("tomato:cuttingSurface") and prim.GetAttribute("tomato:cuttingSurface").Get()
    ]
    assert {str(prim.GetPath()) for prim in cutting} == set(report.cutting_surfaces)
    assert len(cutting) == 1
    assert cutting[0].GetAttribute("tomato:hardwareRole").Get() == "cutting_edge"
    assert cutting[0].GetAttribute("tomato:edgeDepthMillimeters").Get() == 2.0
    assert cutting[0].GetAttribute("tomato:cuttingDirection").Get() == Gf.Vec3f(
        *robot_hardware.KNIFE_CUT_DIRECTION_LOCAL.tolist()
    )
    assert cutting[0].GetAttribute("tomato:edgeAxis").Get() == Gf.Vec3f(
        *robot_hardware.KNIFE_EDGE_AXIS_LOCAL.tolist()
    )
    assert all(stage.GetPrimAtPath(path).IsValid() for path in report.non_cutting_supports)
    assert all(not stage.GetPrimAtPath(path).GetAttribute("tomato:cuttingSurface").Get() for path in report.non_cutting_supports)
    arc_collision = stage.GetPrimAtPath(
        f"{robot_hardware.ROBOT_ROOT}/ee_right/attachments/DeleafKnife/ArcCollision"
    )
    assert (
        UsdPhysics.MeshCollisionAPI(arc_collision).GetApproximationAttr().Get()
        == UsdPhysics.Tokens.convexDecomposition
    )
    blade_collision = stage.GetPrimAtPath(
        f"{robot_hardware.ROBOT_ROOT}/ee_right/attachments/DeleafKnife/BladeCollision"
    )
    assert blade_collision.GetAttribute(
        "physxCollision:contactOffset"
    ).Get() == pytest.approx(robot_hardware.KNIFE_BLADE_CONTACT_OFFSET_M)
    assert blade_collision.GetAttribute(
        "physxCollision:restOffset"
    ).Get() == pytest.approx(
        robot_hardware.KNIFE_BLADE_REST_OFFSET_M
    )
    assert Sdf.Path(report.attachments[-1]).IsAbsolutePath()
    assert len(report.removed_right_gripper_prims) == 6
    assert all(not stage.GetPrimAtPath(path).IsActive() for path in report.removed_right_gripper_prims)
    assert stage.GetPrimAtPath(f"{robot_hardware.ROBOT_ROOT}/ee_right").GetAttribute(
        "tomato:toolConfiguration"
    ).Get() == "knife_only"

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
    assert left_forward[1] < -0.3 and left_forward[2] < -0.9
    assert right_forward[1] > 0.3 and right_forward[2] < -0.9
    np.testing.assert_allclose(left_forward[2], right_forward[2], atol=1e-9)
    assert left_up[0] < -0.9 and right_up[0] < -0.9
    np.testing.assert_allclose(left_up, right_up, atol=1e-9)
    assert stage.GetPrimAtPath(report.cameras[0]).GetAttribute("tomato:sensorRollDegrees").Get() == 0.0
    assert stage.GetPrimAtPath(report.cameras[1]).GetAttribute("tomato:sensorRollDegrees").Get() == 180.0
    for attachment in report.attachments[:2]:
        prim = stage.GetPrimAtPath(attachment)
        assert prim.GetAttribute("tomato:mountInterface").Get() == "rby1_wrist_m3_pair"
        assert prim.GetAttribute("tomato:mountBoltSpacingMillimeters").Get() == 18.0

    edge_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(report.cutting_surfaces[0])).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    blade_projection = np.asarray(
        edge_matrix.TransformDir(
            Gf.Vec3d(*robot_hardware.KNIFE_CUT_DIRECTION_LOCAL.tolist())
        ).GetNormalized()
    )
    np.testing.assert_allclose(blade_projection, [0.0, 1.0, 0.0], atol=1e-9)
