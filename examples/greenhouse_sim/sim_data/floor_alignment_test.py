"""Static placement regression tests; no physics or human review implied."""
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from sim_data.floor_alignment import PACKAGE_FLOOR, align_robot_to_floor, floor_triangles, surface_height


def stage_and_floor(z=.101):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Floor")
    mesh.CreatePointsAttr([(-5, -5, z), (5, -5, z), (5, 5, z), (-5, 5, z)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    return stage, mesh


def robot(stage, z=0):
    root = UsdGeom.Xform.Define(stage, "/World/Robot")
    root.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(.8, 0, z)))
    for name, position, scale in (
            ("wheel_l", (.228, .265, .1), (.1, .025, .1)),
            ("wheel_r", (.228, -.265, .1), (.1, .025, .1)),
            ("base", (0, 0, .15), (.3, .2, .13))):
        cube = UsdGeom.Cube.Define(stage, f"/World/Robot/{name}")
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3d(*scale))
    return root


def test_alignment_is_session_only_preserves_xy_and_is_idempotent():
    stage, _ = stage_and_floor()
    root = robot(stage)
    original = stage.GetRootLayer().ExportToString()
    first = align_robot_to_floor(stage, str(root.GetPath()), "/World/Floor")
    np.testing.assert_allclose(first["robot_position_world_m"], [.8, 0, .101], atol=1e-8)
    assert all(abs(w["clearance_m"]) < 1e-10 for w in first["wheel_supports"])
    assert first["base_minimum_clearance_m"] == pytest.approx(.02)
    second = align_robot_to_floor(stage, str(root.GetPath()), "/World/Floor")
    assert second["vertical_adjustment_m"] == pytest.approx(0, abs=1e-12)
    assert stage.GetRootLayer().ExportToString() == original
    assert not first["physics_validated"]


def test_reads_actual_triangles_not_distant_raised_strip_or_hidden_mesh():
    stage, mesh = stage_and_floor()
    strip = UsdGeom.Mesh.Define(stage, "/World/Floor/Raised")
    strip.CreatePointsAttr([(3, 3, 1), (4, 3, 1), (3, 4, 1)])
    strip.CreateFaceVertexCountsAttr([3])
    strip.CreateFaceVertexIndicesAttr([0, 1, 2])
    triangles = floor_triangles(stage, "/World/Floor")
    assert surface_height(triangles, .8, 0) == pytest.approx(.101)
    assert surface_height(triangles, 3.1, 3.1) == pytest.approx(1)
    strip.CreateVisibilityAttr("invisible")
    assert surface_height(floor_triangles(stage, "/World/Floor"), 3.1, 3.1) == pytest.approx(.101)
    mesh.CreateHoleIndicesAttr([0, 1])
    with pytest.raises(ValueError, match="No visible"):
        floor_triangles(stage, "/World/Floor")


def test_world_transforms_and_upward_or_downward_repositioning():
    stage, _ = stage_and_floor(.4)
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddTranslateOp().Set(Gf.Vec3d(1, 2, .2))
    root = robot(stage, z=.9)
    result = align_robot_to_floor(stage, str(root.GetPath()), "/World/Floor")
    np.testing.assert_allclose(result["robot_position_world_m"], [1.8, 2, .6], atol=1e-7)
    assert result["vertical_adjustment_m"] == pytest.approx(-.5)


@pytest.mark.parametrize("case", ["missing", "outside", "slope", "hidden_wheel", "wrong_units", "nontriangles"])
def test_invalid_support_fails_without_changing_robot(case):
    stage, mesh = stage_and_floor()
    root = robot(stage)
    if case == "missing":
        stage.RemovePrim("/World/Floor")
    elif case == "outside":
        mesh.AddTranslateOp().Set(Gf.Vec3d(100, 0, 0))
    elif case == "slope":
        mesh.AddRotateXOp().Set(10)
    elif case == "hidden_wheel":
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Robot/wheel_l")).CreateVisibilityAttr("invisible")
    elif case == "wrong_units":
        UsdGeom.SetStageMetersPerUnit(stage, .01)
    else:
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):
        align_robot_to_floor(stage, str(root.GetPath()), "/World/Floor")
    assert stage.GetSessionLayer().ExportToString() == before


def test_supplied_floor_and_v12_robot_regression():
    from sim_data.audit import DEFAULT_PACK
    from sim_data.robot_preview import add_robot_preview
    from greenhouse_sim.robot_model import DEFAULT_ASSET
    scene = DEFAULT_PACK / "house/green_house_base.usd"
    if not scene.is_file() or not DEFAULT_ASSET.is_file():
        pytest.skip("Local supplied package/generated v1.2 asset required")
    stage = Usd.Stage.Open(str(scene), Usd.Stage.LoadNone)
    stage.Load("/World/Environment/GreenHouse", Usd.LoadWithoutDescendants)
    stage.Load(PACKAGE_FLOOR)
    stage.SetEditTarget(stage.GetSessionLayer())
    before = {l.identifier: l.ExportToString() for l in stage.GetUsedLayers() if not l.anonymous}
    result = add_robot_preview(stage, gutter_x=-.2, floor_path=PACKAGE_FLOOR)
    alignment = result["floor_alignment"]
    np.testing.assert_allclose(alignment["robot_position_world_m"], [.8, 0, .101], atol=1e-7)
    assert all(abs(w["clearance_m"]) < 1e-8 for w in alignment["wheel_supports"])
    assert alignment["base_minimum_clearance_m"] >= 0
    assert len(result["camera_paths"]) == 3
    for layer in stage.GetUsedLayers():
        if layer.identifier in before:
            assert layer.ExportToString() == before[layer.identifier]
