import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from sim_physics.held_plant_screen import HeldPlantScreen
from sim_physics.held_plant_screen import TriangleIndex,collision_triangles


def fixture(shape_kind='capsule',blade=False):
    stage=Usd.Stage.CreateInMemory()
    paths=['/World/P/Support','/World/P/Branch']
    for path in paths: UsdGeom.Xform.Define(stage,path)
    capsule=UsdGeom.Capsule.Define(stage,paths[1]+'/StemCollider')
    capsule.CreateRadiusAttr(.002);capsule.CreateHeightAttr(.02);capsule.CreateAxisAttr('Z')
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    leaf=UsdGeom.Mesh.Define(stage,paths[1]+'/Leaf')
    leaf.CreatePointsAttr(list(itertools.product((.07,.13),(-.05,.05),(-.02,.02))))
    leaf.CreateFaceVertexCountsAttr([3]);leaf.CreateFaceVertexIndicesAttr([0,1,2])
    UsdPhysics.CollisionAPI.Apply(leaf.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(leaf.GetPrim()).CreateApproximationAttr('convexHull')
    rig=SimpleNamespace(stage=stage,body_paths=paths,cut_index=1,rest_frames=np.repeat(np.eye(4)[None],2,axis=0))
    shape=((np.array([.08,0,0]),np.array([.12,0,0]),.004) if shape_kind=='capsule'
           else (np.array([.1,0,0]),np.eye(3),np.full(3,.005)))
    path='/World/R/ee_right/Blade' if blade else '/World/R/link_right_arm_5/Shape'
    link='ee_right' if blade else 'link_right_arm_5'
    screen=HeldPlantScreen(rig,[(path,'/World/R/'+link,link,shape_kind,shape)],'/World/R/ee_right/Blade')
    screen.snapshot(rig.rest_frames)
    return screen,rig,{link:np.eye(4)}


@pytest.mark.parametrize('kind',['capsule','box'])
def test_native_snapshot_not_stale_usd_and_solid_containment(kind):
    screen,rig,robot=fixture(kind)
    before=rig.stage.GetRootLayer().ExportToString()
    assert not screen.check(robot)
    assert screen.last_failure['plant_collider'].endswith('/Leaf')
    frames=rig.rest_frames.copy();frames[1,0,3]=.5
    screen.snapshot(frames)
    assert screen.check(robot)
    assert rig.stage.GetRootLayer().ExportToString()==before
    with pytest.raises(ValueError): screen.check(robot,margin=0)
    with pytest.raises(ValueError): screen.snapshot(np.full_like(frames,float('nan')))


def test_only_blade_seam_pairs_can_be_expected_during_stroke():
    screen,rig,robot=fixture('box',blade=True)
    # Blade/leaf collisions remain rejected, including during the stroke.
    assert not screen.check(robot,stroke=True)
    # Move the same small blade box to the seam shaft.
    robot['ee_right'][0,3]=-.1
    assert not screen.check(robot)
    assert screen.check(robot,stroke=True)
    screen.blade_path='different_shape'
    assert not screen.check(robot,stroke=True)


def test_local_static_contacts_and_unchecked_region_are_not_ignored():
    screen,rig,robot=fixture()
    # Move the held plant away, retaining a hidden static object on the path.
    cube=UsdGeom.Cube.Define(rig.stage,'/World/StaticFruit')
    cube.CreateSizeAttr(.03);cube.CreateVisibilityAttr('invisible')
    from pxr import Gf
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(.1,0,0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    screen.include_static_scene(rig.stage,'/World/R','/World/P',np.zeros(3))
    frames=rig.rest_frames.copy();frames[1,0,3]=.5;screen.snapshot(frames)
    assert not screen.check(robot)
    assert screen.last_failure['plant_collider']=='/World/StaticFruit'
    robot['link_right_arm_5'][1,3]=.3
    assert screen.check(robot)
    robot['link_right_arm_5'][1,3]=2
    assert not screen.check(robot)
    assert screen.last_failure['reason']=='right_shape_left_cached_scene_region'


@pytest.mark.parametrize('link',['ee_finger_l1','ee_finger_l2','ee_left','link_left_arm_5'])
def test_left_grasp_allows_only_fingers_on_exact_selected_shaft(link):
    original,rig,_=fixture('box')
    shape=(np.zeros(3),np.eye(3),np.full(3,.005))
    shapes=[('/World/R/'+link+'/Shape','/World/R/'+link,link,'box',shape)]
    screen=HeldPlantScreen(rig,shapes,'unused',arm='left',grasp_path=rig.body_paths[1])
    screen.snapshot(rig.rest_frames);robot={link:np.eye(4)}
    before=rig.stage.GetRootLayer().ExportToString()
    assert not screen.check(robot)
    assert screen.check(robot,grasp=True)==link.startswith('ee_finger_l')
    # Even a finger hitting an attached leaf cannot stand in for shaft grasp.
    robot[link][0,3]=.1
    assert not screen.check(robot,grasp=True)
    assert screen.last_failure['plant_collider'].endswith('/Leaf')
    assert rig.stage.GetRootLayer().ExportToString()==before
    with pytest.raises(ValueError): original.check({'link_right_arm_5':np.eye(4)},grasp=True)


def test_left_grasp_does_not_allow_neighboring_shaft_or_invalid_grasp():
    original,rig,_=fixture('box')
    cap=UsdGeom.Capsule.Define(rig.stage,rig.body_paths[0]+'/StemCollider')
    cap.CreateRadiusAttr(.002);cap.CreateHeightAttr(.02)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    link='ee_finger_l1'
    shapes=[('/World/R/'+link+'/Shape','/World/R/'+link,link,'box',
        (np.zeros(3),np.eye(3),np.full(3,.005)))]
    screen=HeldPlantScreen(rig,shapes,'unused',arm='left',grasp_path=rig.body_paths[1])
    screen.snapshot(rig.rest_frames)
    assert not screen.check({link:np.eye(4)},grasp=True)
    assert screen.last_failure['plant_collider']==rig.body_paths[0]+'/StemCollider'
    with pytest.raises(ValueError): HeldPlantScreen(rig,shapes,'unused',arm='left',grasp_path=rig.body_paths[0])
    with pytest.raises(ValueError): HeldPlantScreen(rig,shapes,'unused',arm='right',grasp_path=rig.body_paths[1])
    with pytest.raises(ValueError): HeldPlantScreen(rig,shapes,'unused',arm='unknown')


def test_vectorized_collision_topology_keeps_both_quad_diagonals():
    points=np.arange(21,dtype=float).reshape(7,3)
    actual=collision_triangles(points,[3,4],np.arange(7))
    expected=points[[[0,1,2],[3,4,5],[3,5,6],[3,4,6],[4,5,6]]]
    np.testing.assert_array_equal(actual,expected)
    assert collision_triangles(points,[],[]).shape==(0,3,3)
    for counts,indices in (([5],[0,1,2,3,4]),([3],[0,1]),([3],[-1,1,2]),([3],[0,1,7])):
        with pytest.raises(ValueError): collision_triangles(points,counts,indices)


@pytest.mark.parametrize('count',[0,40,3000])
def test_triangle_lookup_matches_exhaustive_aabb_and_narrow_phase(count):
    from sim_data.capture_viewpoints import triangles_intersect_box
    from sim_physics.capsule_surface import segment_triangles_distance
    random=np.random.default_rng(431)
    triangles=random.uniform(-2,2,(count,1,3))+random.uniform(-.1,.1,(count,3,3))
    if count: triangles[0]=[[-5,0,0],[5,0,0],[0,5,0]]  # large triangle cannot be missed by centre lookup
    index=TriangleIndex(triangles)
    for _ in range(25):
        a=random.uniform(-1,1,3);b=a+random.uniform(-.1,.1,3);radius=.06
        low=np.minimum(a,b)-radius;high=np.maximum(a,b)+radius
        selected=index.query(low,high)
        brute=triangles[np.all(triangles.max(1)>=low-1e-9,axis=1)&np.all(triangles.min(1)<=high+1e-9,axis=1)]
        assert {tuple(v.ravel()) for v in selected}=={tuple(v.ravel()) for v in brute}
        assert triangles_intersect_box(selected,low,high)==triangles_intersect_box(triangles,low,high)
        assert (segment_triangles_distance(a,b,selected)<=radius)==(segment_triangles_distance(a,b,triangles)<=radius)
    with pytest.raises(ValueError): index.query([0,0,0],[-1,1,1])
