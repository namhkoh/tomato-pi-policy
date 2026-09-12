from types import SimpleNamespace
import pytest

from pxr import Gf,Usd,UsdGeom,UsdPhysics

from sim_physics.startup_screen import screen


def test_hidden_and_instanced_collisions_are_not_ignored():
    stage=Usd.Stage.CreateInMemory()
    robot=UsdGeom.Cube.Define(stage,'/World/R/arm/shape');robot.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(robot.GetPrim())
    obstacle=UsdGeom.Cube.Define(stage,'/World/Obstacle');obstacle.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim());obstacle.CreateVisibilityAttr('invisible')
    fixture=SimpleNamespace(root='/World/R',floor_root=None)
    assert not screen(stage,fixture)['passed']
    UsdGeom.Xformable(obstacle).AddTranslateOp().Set(Gf.Vec3d(1,0,0))
    assert screen(stage,fixture)['passed']
    group=UsdGeom.Xform.Define(stage,'/World/Group')
    child=UsdGeom.Cube.Define(stage,'/World/Group/shape');child.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(child.GetPrim())
    UsdGeom.Xformable(group).AddTranslateOp().Set(Gf.Vec3d(1,0,0))
    source=UsdGeom.Xform.Define(stage,'/World/Instance')
    source.GetPrim().GetReferences().AddInternalReference('/World/Group')
    source.GetPrim().SetInstanceable(True)
    UsdGeom.Xformable(source).ClearXformOpOrder()
    assert not screen(stage,fixture)['passed']


def test_quad_refinement_clears_empty_bbox_corner_but_not_solid_containment():
    import numpy as np
    stage=Usd.Stage.CreateInMemory()
    robot=UsdGeom.Cube.Define(stage,'/World/R/arm/shape');robot.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(robot.GetPrim())
    obstacle=UsdGeom.Mesh.Define(stage,'/World/Leaf')
    obstacle.CreatePointsAttr([(-1,1.3,-.1),(1.3,-1,-.1),(1.3,-1,.1),(-1,1.3,.1)])
    obstacle.CreateFaceVertexCountsAttr([4]);obstacle.CreateFaceVertexIndicesAttr([0,1,2,3])
    UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
    fixture=SimpleNamespace(root='/World/R',floor_root=None)
    assert screen(stage,fixture)['passed']
    points=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    obstacle.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    obstacle.GetFaceVertexCountsAttr().Set([3]);obstacle.GetFaceVertexIndicesAttr().Set([0,1,2])
    UsdPhysics.MeshCollisionAPI.Apply(obstacle.GetPrim()).CreateApproximationAttr('convexHull')
    assert not screen(stage,fixture)['passed']
    UsdGeom.Xformable(obstacle).AddTranslateOp().Set(Gf.Vec3d(3,0,0))
    assert screen(stage,fixture)['passed']


def test_capsule_narrow_phase_clears_only_empty_rotated_box_corners():
    stage=Usd.Stage.CreateInMemory()
    box=UsdGeom.Cube.Define(stage,'/World/R/finger/shape');box.CreateSizeAttr(1.)
    box.AddRotateZOp().Set(45.);box.AddScaleOp().Set(Gf.Vec3f(.1,.01,.01))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    stem=UsdGeom.Capsule.Define(stage,'/World/Stem');stem.CreateRadiusAttr(.002);stem.CreateHeightAttr(.03)
    op=stem.AddTranslateOp();op.Set(Gf.Vec3d(.03,-.03,0))
    UsdPhysics.CollisionAPI.Apply(stem.GetPrim())
    robot=SimpleNamespace(root='/World/R',floor_root=None)
    assert screen(stage,robot)['passed']
    op.Set(Gf.Vec3d(.02,.02,0));assert not screen(stage,robot)['passed']


@pytest.mark.parametrize('mesh',[False,True])
def test_actual_robot_capsule_clears_whole_unknown_scene_bound_not_just_vertices(mesh):
    stage=Usd.Stage.CreateInMemory()
    cap=UsdGeom.Capsule.Define(stage,'/World/R/arm/shape')
    cap.CreateRadiusAttr(.02);cap.CreateHeightAttr(.2);cap.CreateAxisAttr('X')
    cap.CreateExtentAttr([Gf.Vec3f(-.12,-.02,-.02),Gf.Vec3f(.12,.02,.02)])
    cap.AddRotateZOp().Set(45.);UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    if mesh:
        import itertools
        obstacle=UsdGeom.Mesh.Define(stage,'/World/UnknownCookedMesh')
        obstacle.CreatePointsAttr(list(itertools.product((-.005,.005),repeat=3)))
        obstacle.CreateFaceVertexCountsAttr([3]);obstacle.CreateFaceVertexIndicesAttr([0,1,2])
        UsdPhysics.MeshCollisionAPI.Apply(obstacle.GetPrim()).CreateApproximationAttr('convexDecomposition')
    else:
        obstacle=UsdGeom.Cube.Define(stage,'/World/Box');obstacle.CreateSizeAttr(.01)
    UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
    op=obstacle.AddTranslateOp();op.Set(Gf.Vec3d(.065,-.065,0))
    robot=SimpleNamespace(root='/World/R',floor_root=None)
    before=stage.GetRootLayer().ExportToString()
    result=screen(stage,robot)
    assert result['tested_broad_pairs']==1
    assert result['passed'] and result['robot_capsule_vs_whole_scene_bound_cleared']==1
    assert stage.GetRootLayer().ExportToString()==before
    op.Set(Gf.Vec3d(.02,.02,0))
    result=screen(stage,robot)
    assert not result['passed'] and result['robot_capsule_vs_whole_scene_bound_cleared']==0
    # A point inside an empty box corner must remain rejected when the actual
    # capsule representation is unsupported (nonuniformly scaled).
    op.Set(Gf.Vec3d(.065,-.065,0));cap.AddScaleOp().Set(Gf.Vec3f(2,1,1))
    result=screen(stage,robot)
    assert not result['passed'] and result['robot_capsule_vs_whole_scene_bound_cleared']==0


def test_capsule_scene_bound_margin_and_solid_containment_stay_rejected():
    stage=Usd.Stage.CreateInMemory()
    cap=UsdGeom.Capsule.Define(stage,'/World/R/arm/shape')
    cap.CreateRadiusAttr(.02);cap.CreateHeightAttr(.2);UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    cube=UsdGeom.Cube.Define(stage,'/World/Solid');cube.CreateSizeAttr(.01)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim());op=cube.AddTranslateOp()
    robot=SimpleNamespace(root='/World/R',floor_root=None)
    op.Set(Gf.Vec3d(.0245,0,0));assert not screen(stage,robot)['passed']
    op.Set(Gf.Vec3d(0,0,0));cube.GetSizeAttr().Set(1.)
    assert not screen(stage,robot)['passed']
