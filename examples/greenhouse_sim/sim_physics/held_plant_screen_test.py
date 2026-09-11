import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from sim_physics.held_plant_screen import HeldPlantScreen


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
