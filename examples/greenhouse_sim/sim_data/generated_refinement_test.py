"""Generated foliage gets the original mesh test, not a bypass of real contact."""
import numpy as np
import pytest
from pxr import Gf,Usd,UsdGeom
from .capture_viewpoints import visible_bounds,scene_triangle_refiner,screen_bounds
from .training_screen import StaticBoundScreen


def fixture(root="/World/GeneratedNativePilot",intersect=False):
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Robot/Body").CreateSizeAttr(1.)
    mesh=UsdGeom.Mesh.Define(stage,root+"/Plant/Leaf")
    vertices=[[-2,0,0],[2,0,0],[0,2,0]] if intersect else [[2,0,0],[0,2,0],[2,2,0]]
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0,1,2])
    return stage,visible_bounds(stage,"/World/Robot"),visible_bounds(stage,root)


def test_generated_empty_box_overlap_cleared_only_by_explicit_mesh_refinement():
    stage,robot,obstacles=fixture()
    assert not screen_bounds(robot,obstacles)["passed"]
    assert not screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage))["passed"]
    refiner=scene_triangle_refiner(stage,include_generated_plants=True)
    result=screen_bounds(robot,obstacles,refinement=refiner)
    assert result["passed"] and result["pairs_cleared_by_triangle_refinement"]==1
    assert result["margin_m"]==.01 and not result["collision_free_certified"]
    assert StaticBoundScreen(obstacles,refiner)(robot)==result


def test_real_generated_surface_intersection_still_rejected():
    stage,robot,obstacles=fixture(intersect=True)
    result=screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage,include_generated_plants=True))
    assert not result["passed"] and result["possible_overlap_count"]==1
    assert result["pairs_cleared_by_triangle_refinement"]==0


@pytest.mark.parametrize("root",["/World/Greenhouse","/World/GeneratedNativePilotExtra","/World/RobotProp"])
def test_other_structures_and_prefix_lookalikes_remain_conservative(root):
    stage,robot,obstacles=fixture(root)
    assert not screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage,include_generated_plants=True))["passed"]


def test_unsupported_generated_shapes_remain_conservative():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Robot/Body").CreateSizeAttr(1.)
    UsdGeom.Cube.Define(stage,"/World/GeneratedNativePilot/Plant/Unsupported").CreateSizeAttr(2.)
    robot=visible_bounds(stage,"/World/Robot")
    obstacles=visible_bounds(stage,"/World/GeneratedNativePilot")
    assert not screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage,include_generated_plants=True))["passed"]


@pytest.mark.parametrize("intersect",[False,True])
def test_original_plant_results_unchanged_with_new_option(intersect):
    stage,robot,obstacles=fixture("/World/PackPlants",intersect)
    a=screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage))
    b=screen_bounds(robot,obstacles,refinement=scene_triangle_refiner(stage,include_generated_plants=True))
    assert a==b


@pytest.mark.parametrize("flag",[None,0,1,"yes",[]])
def test_generated_scope_cannot_be_enabled_by_ambiguous_truthiness(flag):
    stage,_,_=fixture()
    with pytest.raises(ValueError):scene_triangle_refiner(stage,include_generated_plants=flag)


def test_generated_margin_preserves_near_surface_rejection():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,"/World/Robot/Body").CreateSizeAttr(1.)
    mesh=UsdGeom.Mesh.Define(stage,"/World/GeneratedNativePilot/Plant/Leaf")
    mesh.CreatePointsAttr([Gf.Vec3f(.505,-.2,-.2),Gf.Vec3f(.505,.2,-.2),Gf.Vec3f(.505,0,.2)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    robot=visible_bounds(stage,"/World/Robot");obs=visible_bounds(stage,"/World/GeneratedNativePilot")
    assert not screen_bounds(robot,obs,refinement=scene_triangle_refiner(stage,include_generated_plants=True))["passed"]
    assert screen_bounds(robot,obs,margin_m=0,refinement=scene_triangle_refiner(stage,include_generated_plants=True))["passed"]
