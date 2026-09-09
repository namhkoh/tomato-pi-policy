"""Mounted POV is not a floating inspection camera or a visibility oracle."""
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from sim_data.review_camera import HEAD_CAMERA, attachment_world, has_head_camera, select_review_view, view_context


class Viewport:
    camera_path = HEAD_CAMERA
    resolution = (848, 408)

    def get_texture_resolution(self):
        return self.resolution

    def set_texture_resolution(self, resolution):
        self.resolution = resolution

    def set_active_camera(self, path):
        self.camera_path = path


def fixture(point):
    stage = Usd.Stage.CreateInMemory()
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    camera = UsdGeom.Camera.Define(stage, HEAD_CAMERA)
    camera.CreateFocalLengthAttr(10)
    camera.CreateHorizontalApertureAttr(20)
    camera.CreateVerticalApertureAttr(20 * 408 / 848)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.04, 10))
    UsdGeom.Xform.Define(stage, "/World/RBY1/base")
    UsdGeom.Xform.Define(stage, "/World/Plant")
    UsdGeom.Xform.Define(stage, "/World/Plant/Target")
    report = {"components": {"target": {"translation_plant_m": [0, 0, 0]}}}
    scene = SimpleNamespace(paths={"target": "/World/Plant/Target"}, report=report,
                            plant_root="/World/Plant", visibility_paths={})
    scene.restore_visibility = lambda: scene.visibility_paths.clear()
    target = {"component_id": "target", "attachment_plant_m": point}
    return stage, Viewport(), scene, target


@pytest.mark.parametrize("point,status", [([0, 0, -1], "inside_image_frustum"),
                                         ([2, 0, -1], "outside_image"),
                                         ([0, 0, 1], "behind_camera"),
                                         ([0, 0, -.01], "outside_clipping_range"),
                                         ([0, 0, -11], "outside_clipping_range")])
def test_projection_is_not_occlusion_or_reachability(point, status):
    stage, viewport, scene, target = fixture(point)
    context = view_context(stage, viewport, scene, target)
    assert context["projection_status"] == status
    assert context["view_kind"] == "robot_head"
    assert context["occlusion"] == "not_measured"
    assert not context["projection_is_visibility_measurement"]
    assert not context["training_input_allowed"]
    if status == "inside_image_frustum":
        np.testing.assert_allclose(context["attachment_pixel_xy"], [424, 204])


def test_head_selection_does_not_move_camera_or_robot():
    stage, viewport, scene, target = fixture([0, 0, -1])
    before = stage.GetSessionLayer().ExportToString()
    viewport.resolution = (1280, 720)
    scene.visibility_paths = {"fixture": True}
    select_review_view(stage, viewport, scene, target, "robot_head")
    assert stage.GetSessionLayer().ExportToString() == before
    assert viewport.camera_path == HEAD_CAMERA and viewport.resolution == (848, 408)
    assert not scene.visibility_paths


def test_component_transform_and_camera_offset_projection():
    stage, viewport, scene, target = fixture([0, 0, -1])
    UsdGeom.Xformable(stage.GetPrimAtPath(scene.paths["target"])).AddTranslateOp().Set(Gf.Vec3d(.1, 0, 0))
    np.testing.assert_allclose(attachment_world(stage, scene.paths, scene.report, target), [.1, 0, -1])
    camera = UsdGeom.Camera(stage.GetPrimAtPath(HEAD_CAMERA))
    camera.CreateHorizontalApertureOffsetAttr(1)
    camera.CreateVerticalApertureOffsetAttr(1)
    context = view_context(stage, viewport, scene, target)
    np.testing.assert_allclose(context["attachment_pixel_xy"], [424, 246.4], atol=1e-5)


def test_missing_head_has_no_silent_floating_camera_fallback():
    stage, viewport, scene, target = fixture([0, 0, -1])
    stage.RemovePrim(HEAD_CAMERA)
    assert not has_head_camera(stage)
    with pytest.raises(ValueError, match="unavailable"):
        select_review_view(stage, viewport, scene, target, "robot_head")
