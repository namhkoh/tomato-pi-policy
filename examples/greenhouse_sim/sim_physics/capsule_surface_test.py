import numpy as np
import pytest
from sim_physics.capsule_surface import segment_triangles_distance as distance,world_capsule


@pytest.mark.parametrize('start,end,expected',[
    ([.2,.2,1],[.2,.2,-1],0),
    ([.2,.2,1],[.6,.2,1],1),
    ([2,0,0],[2,1,0],1),
    ([.2,.2,1],[.2,.2,1],1),
    ([.2,.2,0],[.3,.2,0],0),
    ([1,1,0],[2,2,0],np.sqrt(.5)),
    ([.5,-1,1],[.5,1,1],1)])
def test_analytic_segment_triangle_cases(start,end,expected):
    tri=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
    assert distance(start,end,tri)==pytest.approx(expected,abs=1e-12)
    assert distance(end,start,tri[:,::-1])==pytest.approx(expected,abs=1e-12)


def test_degenerate_faces_and_rigid_transform_invariance():
    from scipy.spatial.transform import Rotation
    tri=np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[1,0,0],[2,0,0]]],float)
    assert distance([.5,1,0],[.5,2,0],tri)==pytest.approx(1)
    rng=np.random.default_rng(44)
    for _ in range(30):
        tri=rng.normal(size=(10,3,3));a,b=rng.normal(size=(2,3))
        r=Rotation.random(random_state=rng).as_matrix();t=rng.normal(size=3)
        assert distance(a@r.T+t,b@r.T+t,tri@r.T+t)==pytest.approx(distance(a,b,tri),abs=1e-10)
    with pytest.raises(ValueError): distance([0,0,np.nan],[0,0,0],tri)


def test_capsule_corner_refinement_retains_real_contacts():
    from types import SimpleNamespace
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from sim_physics.startup_screen import screen
    stage=Usd.Stage.CreateInMemory()
    cap=UsdGeom.Capsule.Define(stage,'/World/R/arm/shape')
    cap.CreateRadiusAttr(.1);cap.CreateHeightAttr(.4)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    mesh=UsdGeom.Mesh.Define(stage,'/World/Obstacle')
    mesh.CreatePointsAttr([(.09,.09,-.1),(.09,.09,.1),(.095,.095,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    robot=SimpleNamespace(root='/World/R',floor_root=None)
    assert screen(stage,robot)['passed']  # Empty corner of capsule's box.
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(-.025,-.025,0))
    assert not screen(stage,robot)['passed']
    assert world_capsule(cap.GetPrim(),np.diag([2,1,1,1])) is None
