"""Synthetic CPU geometry/material tests; no native worker or dataset writes."""
from copy import deepcopy
import inspect
import json
import sys

import numpy as np
import pytest

from . import visibility_shadow as s


def mesh(component, z, *, reasons=(), organ="sub_stem"):
    points = np.array([[-3., -2., z], [3., -2., z], [0., 4., z]])
    a, b, ids, count, bad = s.triangulate(points, [3], [0, 1, 2])
    assert not bad
    return s.Mesh(component, organ, "/"+component, s.frozen_array(points.min(axis=0)),
                  s.frozen_array(points.max(axis=0)), a, b, ids, count, tuple(reasons))


def snapshot(meshes):
    components = dict(target=dict(id="target", type="sub_stem", parent="parent", deleafed=False,
        translation_plant_m=[0., 0., 2.], attachment_plant_m=[0., 0., 2.], axis_plant=[1., 0., 0.],
        capsules_local_m=[[[0., 0., 0., .02], [.2, 0., 0., .02]]]),
        parent=dict(id="parent", type="main_stem", parent=None, translation_plant_m=[0., 0., 3.],
                    capsules_local_m=[[[0., 0., 0., .03], [0., .3, 0., .03]]]))
    return s.Snapshot(tuple(meshes), s.canonical(components), s.frozen_array(np.eye(4)), (), b"synthetic")


def camera():
    # USD -Z-forward transformed so these fixtures look down world +Z.
    return dict(resolution=[1696, 816], crop_resize=None, clipping_range_m=[.01, 10.],
        intrinsics=[[941., 0., 848.], [0., 941., 408.], [0., 0., 1.]],
        camera_to_world_usd_row_vectors=np.diag([1., -1., -1., 1.]).tolist())


def ray(context, allowed=("target",)):
    return s._ray(context, np.zeros(3), np.array([0., 0., 1.]), .01, 10., allowed)


def test_supported_opaque_foreground_is_shadow_prediction_only():
    context = snapshot([mesh("fruit", 1., organ="fruit"), mesh("target", 2.)])
    r = ray(context)
    assert r["status"] == "predicted_blocked" and r["blocker_component"] == "fruit"
    assert r["blocker_ray_distance_upper_m"] < r["allowed_ray_distance_lower_m"]
    result = s.inspect_interval(context, camera(), "target")
    assert result["status"] == "predicted_blocked"
    assert not result["skip_allowed"] and not result["ranking_allowed"]
    assert not result["native_depth_reconstructed"] and not result["training_approved"]
    assert result["full_scene_visibility"] == "not_established"
    assert result["unique_pixel_allowed_sets"] < len(result["interval"])+len(result["proximal"])
    json.dumps(result, allow_nan=False)


def test_missing_surroundings_never_becomes_visibility_pass():
    result = s.inspect_interval(snapshot([mesh("target", 2.)]), camera(), "target")
    assert result["status"] == "unknown"
    assert {r["status"] for r in result["interval"]} == {"no_blocker_found_in_partial_model"}
    assert result["policy"]["coverage"].startswith("generated_plant_only")


@pytest.mark.parametrize("reason", ["opacity_not_explicit_constant_one", "unsupported_polygon",
    "subdivision_not_native_tessellation", "material_displacement", "single_sided_material_geometry"])
def test_unsupported_foreground_is_unknown_not_opaque(reason):
    r = ray(snapshot([mesh("fruit", 1., reasons=[reason]), mesh("target", 2.)]))
    assert r["status"] == "unknown"
    assert reason in r["unknown_meshes"][0]["reasons"]


def test_unknown_allowed_surface_and_absent_surface_are_unknown():
    for meshes in [[mesh("target", 2., reasons=["unknown_material"])], [mesh("fruit", 1.)]]:
        assert ray(snapshot(meshes))["status"] == "unknown"


def test_background_unknown_does_not_invent_foreground_occlusion():
    r = ray(snapshot([mesh("target", 2.), mesh("fruit", 3., reasons=["unknown_material"])]))
    assert r["status"] == "no_blocker_found_in_partial_model"


def test_parent_allowed_only_for_native_proximal_subset():
    context = snapshot([mesh("target", 2.), mesh("parent", 1., organ="main_stem")])
    assert ray(context)["status"] == "predicted_blocked"
    assert ray(context, ("target", "parent"))["status"] == "no_blocker_found_in_partial_model"
    r = s.inspect_interval(context, camera(), "target")
    for row in r["proximal"]:
        assert ("parent" in row["allowed_components"]) == (row["arc_m"] < .008)
    assert all(row["allowed_components"] == ["target"] for row in r["interval"])


def test_grazing_triangle_edge_not_certified():
    m = mesh("target", 2.)
    d, strict = s.triangle_hits(m.triangles_a, np.array([-3., -2., 0.]), np.array([0., 0., 1.]), .01, 10.)
    assert np.isfinite(d[0]) and not strict[0]


def test_quad_requires_same_face_under_both_diagonals():
    hit = mesh("x", 1.).triangles_a[0]
    miss = hit + [10., 0., 0.]
    # All-A and all-B both hit, but neither physical face hits both choices.
    a, b = s.frozen_array([hit, miss]), s.frozen_array([miss, hit])
    m = s.Mesh("x", "fruit", "/x", s.frozen_array([-3., -2., 1.]), s.frozen_array([13., 4., 1.]),
               a, b, s.frozen_array([0, 1], int), 2, ())
    possible, certain = s.mesh_hits(m, np.zeros(3), np.array([0., 0., 1.]), .01, 10.)
    assert possible == 1. and np.isinf(certain)


def test_quad_holes_and_unsupported_ngons_are_explicit():
    points = np.array([[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.], [0., 2., 1.]])
    a, b, ids, count, unsupported = s.triangulate(points, [4, 5], [0, 1, 2, 3, 0, 1, 2, 3, 4])
    assert unsupported and count == 2 and ids.tolist() == [0, 0] and len(a) == len(b) == 2
    a, b, ids, _, unsupported = s.triangulate(points, [4], [0, 1, 2, 3], holes=[0])
    assert len(a) == len(b) == len(ids) == 0 and not unsupported


def test_ray_aabb_parallel_and_clipping():
    lower = [[-1, -1, 1], [2, 2, 1], [-1, -1, -2]]
    upper = [[1, 1, 2], [3, 3, 2], [1, 1, -1]]
    candidates, _ = s.ray_boxes(lower, upper, np.zeros(3), np.array([0., 0., 1.]), .1, 5.)
    assert candidates.tolist() == [True, False, False]


def test_empty_mesh_topology_can_be_retained_as_bounded_unknown():
    a, b, ids, count, unsupported = s.triangulate([[0., 0., 1.]], [], [])
    assert len(a) == len(b) == len(ids) == count == 0 and not unsupported
    m = s.Mesh("empty", "sub_stem", "/empty", s.frozen_array([-1., -1., 1.]),
        s.frozen_array([1., 1., 1.]), a, b, ids, count, ("empty_mesh_no_surface",))
    r = ray(snapshot([m, mesh("target", 2.)]))
    assert r["status"] == "unknown"
    assert r["unknown_meshes"][0]["reasons"] == ["empty_mesh_no_surface"]


@pytest.mark.parametrize("counts,indices", [([], [0]), ([3], [0, 1, 9])])
def test_malformed_topology_still_rejected(counts, indices):
    with pytest.raises(ValueError):
        s.triangulate([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]], counts, indices)


def test_pixel_ray_uses_native_intrinsics_and_metric_clipping():
    origin, direction, near, far = s._pixel_ray(camera(), [848., 408.])
    assert np.allclose(origin, 0) and np.allclose(direction, [0, 0, 1])
    assert near == .01 and far == 10.
    c = camera(); c["resolution"] = [848, 408]
    with pytest.raises(ValueError): s.inspect_interval(snapshot([mesh("target", 2.)]), c, "target")


def test_snapshot_arrays_cannot_be_mutated_or_made_writable():
    m = mesh("target", 2.)
    with pytest.raises(ValueError): m.triangles_a[0, 0, 0] = 9
    with pytest.raises(ValueError): m.triangles_a.setflags(write=True)


def test_predictor_cannot_receive_sensor_labels_or_decisions():
    assert list(inspect.signature(s.inspect_interval).parameters) == ["snapshot", "calibration", "component_id"]
    with pytest.raises(TypeError): s.inspect_interval(snapshot([]), camera(), "target", depth=np.zeros((2, 2)))
    assert "isaacsim" not in sys.modules and "omni.replicator.core" not in sys.modules


def usd_material():
    from pxr import Usd, UsdGeom, UsdShade, Sdf
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    material = UsdShade.Material.Define(stage, "/Material")
    shader = UsdShade.Shader.Define(stage, "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return stage, mesh, shader, material


@pytest.mark.parametrize("kind", ["opaque", "transparent", "connected_opacity", "time_opacity",
                                   "displacement", "unknown_shader", "display_opacity"])
def test_portable_usd_material_support_is_strict(kind):
    from pxr import UsdShade, UsdGeom, Sdf
    stage, mesh, shader, material = usd_material()
    if kind == "transparent": shader.GetInput("opacity").Set(.5)
    elif kind == "connected_opacity":
        texture = UsdShade.Shader.Define(stage, "/Material/Tex")
        texture.CreateIdAttr("UsdUVTexture")
        shader.GetInput("opacity").ConnectToSource(texture.ConnectableAPI(), "a")
    elif kind == "time_opacity": shader.GetInput("opacity").Set(1., 1.)
    elif kind == "displacement": shader.CreateInput("displacement", Sdf.ValueTypeNames.Float).Set(.01)
    elif kind == "unknown_shader": shader.CreateIdAttr("Unknown")
    elif kind == "display_opacity": UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([.5])
    reasons = s._material_reasons(mesh.GetPrim())
    assert bool(reasons) == (kind != "opaque")
