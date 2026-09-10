from types import SimpleNamespace

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
