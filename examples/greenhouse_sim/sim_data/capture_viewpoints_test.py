import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from sim_data.capture_viewpoints import (candidate_specs, component_catalogue, scene_triangle_refiner,
                                         screen_bounds, select_diverse, triangles_intersect_box, visible_bounds)


def box(path,low,high):
    return {"path":path,"min":low,"max":high}


def test_catalogue_joins_draft_variant_identity_not_source_plant_id():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Plant/Stem")
    UsdGeom.Cube.Define(stage,"/World/Plant/Stem/Added")
    UsdGeom.Cube.Define(stage,"/World/Plant/Stem/Old").GetPrim().SetActive(False)
    record={"plant_root":"/World/Plant","component_paths":{"stem":"/World/Plant/Stem","old":"/World/Plant/Stem/Old"}}
    variant={"plant_root":"/World/Plant","variant_id":"source+variant","source_plant_id":"source",
             "split_group":"source","added_components":{"added":{"type":"sub_stem"}},
             "added_component_paths":{"added":"/World/Plant/Stem/Added"}}
    report={"plant_id":"source+variant","source_plant_id":"source",
            "components":{"stem":{"type":"main_stem"},"added":{"type":"sub_stem"}}}
    rows=component_catalogue(stage,[record],[report],[variant])
    assert [r["component_id"] for r in rows]==["stem","added"]
    assert all(r["split_group"]=="source" and r["variant_id"]=="source+variant" for r in rows)


def test_candidates_preserve_side_resolution_and_have_bounded_distinct_poses():
    specs=candidate_specs(.8,.03)
    assert len(specs)==12
    assert len({(s["root_x_m"],s["y_offset_m"]) for s in specs})==12
    assert min(s["root_x_m"] for s in specs)==.5
    assert {s["approach_from_original_m"] for s in specs}=={0,.1,.2,.3}
    assert len({tuple(s["desired_pixel_xy"]) for s in specs})==3
    assert all(s["root_x_m"]-.3>=.35 for s in candidate_specs(.8,.3))
    with pytest.raises(ValueError): candidate_specs(.03,.8)


def test_bound_screen_rejects_possible_overlap_and_margin_but_does_not_certify():
    robot=[box("robot",[0,0,0],[1,1,1])]
    obstacle=[box("leaf",[1.005,0,0],[2,1,1])]
    result=screen_bounds(robot,obstacle)
    assert not result["passed"] and result["possible_overlap_count"]==1
    obstacle[0]["min"][0]=1.1
    result=screen_bounds(robot,obstacle)
    assert result["passed"] and not result["collision_free_certified"]
    with pytest.raises(ValueError): screen_bounds([],obstacle)


def test_bound_collection_uses_own_visible_geometry_and_instance_proxies():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Leaf").CreateSizeAttr(.1)
    hidden=UsdGeom.Cube.Define(stage,"/World/Hidden")
    hidden.CreateVisibilityAttr("invisible")
    UsdGeom.Cube.Define(stage,"/World/Floor")
    rows=visible_bounds(stage,exclude=("/World/Floor",))
    assert [r["path"] for r in rows]==["/World/Leaf"]
    assert np.allclose(rows[0]["max"],[.05,.05,.05])


def test_selection_does_not_trade_passed_gate_for_failed_diversity():
    def record(name,x,passed):
        return {"candidate_id":name,"base_xy_m":[x,0],
                "quality":{"clear_view_gate_passed":passed,"estimated_petiole_diameter_px":4,
                           "projected_interval_length_px":5,"target_mask_dark_fraction":.1},
                "visibility":{"nominal":{"visible_target_evidence":True},
                              "sampled_interval_visible_pixel_fraction":1}}
    records=[record("a",0,True),record("b",.1,True),record("c",.3,False)]
    assert [r["candidate_id"] for r in select_diverse(records,2)]==["a","b"]


@pytest.mark.parametrize("vertices,intersects",[
    ([[2,0,0],[0,2,0],[2,2,0]],False),
    ([[-2,0,0],[2,0,0],[0,2,0]],True),
    ([[-2,-2,1],[2,-2,1],[0,2,1]],False),
    ([[.5,0,0],[.5,.2,0],[.5,0,.2]],True),
    ([[0,0,0],[0,0,0],[0,0,0]],True),
])
def test_triangle_box_sat_not_just_overlapping_aabbs(vertices,intersects):
    assert triangles_intersect_box([vertices],[-.5]*3,[.5]*3)==intersects


def test_merged_plant_empty_space_is_cleared_without_ignoring_real_leaf_surface():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Robot/Body").CreateSizeAttr(1.)
    mesh=UsdGeom.Mesh.Define(stage,"/World/PackPlants/Backdrop")
    mesh.CreatePointsAttr([Gf.Vec3f(2,0,0),Gf.Vec3f(0,2,0),Gf.Vec3f(2,2,0)])
    mesh.CreateFaceVertexCountsAttr([3]); mesh.CreateFaceVertexIndicesAttr([0,1,2])
    robot=visible_bounds(stage,"/World/Robot")
    obstacles=visible_bounds(stage,"/World/PackPlants")
    assert not screen_bounds(robot,obstacles)["passed"]
    refined=screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage))
    assert refined["passed"] and refined["pairs_cleared_by_triangle_refinement"]==1
    mesh.GetPointsAttr().Set([Gf.Vec3f(-2,0,0),Gf.Vec3f(2,0,0),Gf.Vec3f(0,2,0)])
    obstacles=visible_bounds(stage,"/World/PackPlants")
    assert not screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage))["passed"]


def test_mixed_quad_triangle_backdrop_refinement_and_holes():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Robot/Body").CreateSizeAttr(1.)
    mesh=UsdGeom.Mesh.Define(stage,"/World/PackPlants/Backdrop")
    mesh.CreatePointsAttr([Gf.Vec3f(2,0,0),Gf.Vec3f(0,2,0),Gf.Vec3f(2,2,0),Gf.Vec3f(3,1,0),
                          Gf.Vec3f(-2,0,0),Gf.Vec3f(2,0,0),Gf.Vec3f(0,2,0)])
    mesh.CreateFaceVertexCountsAttr([4,3]); mesh.CreateFaceVertexIndicesAttr([0,1,2,3,4,5,6])
    robot=visible_bounds(stage,"/World/Robot"); obstacles=visible_bounds(stage,"/World/PackPlants")
    assert not screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage))["passed"]
    mesh.CreateHoleIndicesAttr([1])
    assert screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage))["passed"]
