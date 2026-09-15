"""USD notice invalidation and equality with the uncached geometry screen."""
from unittest.mock import patch
import pytest
from pxr import Gf,Sdf,Usd,UsdGeom
from sim_data import static_geometry_cache as module
from sim_data.capture_viewpoints import static_obstacles,visible_bounds,scene_triangle_refiner
from sim_data.training_screen import StaticBoundScreen


def fixture():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World')
    UsdGeom.Xform.Define(stage,'/World/Robot')
    UsdGeom.Xform.Define(stage,'/World/Scene')
    robot=UsdGeom.Cube.Define(stage,'/World/Robot/Body');robot.CreateSizeAttr(1)
    obstacle=UsdGeom.Cube.Define(stage,'/World/Scene/Obstacle');obstacle.CreateSizeAttr(1)
    UsdGeom.XformCommonAPI(obstacle).SetTranslate(Gf.Vec3d(3,0,0))
    return stage,robot,obstacle


def reference(stage,margin=.01):
    return StaticBoundScreen(static_obstacles(stage,'/World/Robot'),
        scene_triangle_refiner(stage,include_generated_plants=True))(
            visible_bounds(stage,'/World/Robot'),margin)


def test_exact_matches_multiple_robot_poses_without_rebuilding_static_obstacles():
    stage,robot,_=fixture()
    with module.StaticGeometryScreenCache(stage,'/World/Robot',include_generated_plants=True) as cache:
        for x in [0,1,1.9,2.05,3,4.2,0]:
            UsdGeom.XformCommonAPI(robot).SetTranslate(Gf.Vec3d(x,0,0))
            assert cache()==reference(stage)
        assert cache.builds==1 and cache.hits==6
        assert cache.diagnostics()['cached_screen_result'] is False


@pytest.mark.parametrize('change',['transform','size','visibility','deactivate','new_prim','parent_transform','purpose'])
def test_obstacle_change_invalidates_and_preserves_reference_decision(change):
    stage,robot,obstacle=fixture()
    with module.StaticGeometryScreenCache(stage,'/World/Robot',include_generated_plants=True) as cache:
        cache()
        # Keep a second obstacle so inactive/hidden cases remain nonempty.
        extra=UsdGeom.Cube.Define(stage,'/World/Scene/Other')
        extra.CreateSizeAttr(.1);UsdGeom.XformCommonAPI(extra).SetTranslate(Gf.Vec3d(8,0,0))
        cache()
        builds=cache.builds
        if change=='transform':UsdGeom.XformCommonAPI(obstacle).SetTranslate(Gf.Vec3d(.5,0,0))
        elif change=='size':obstacle.GetSizeAttr().Set(8)
        elif change=='visibility':obstacle.CreateVisibilityAttr().Set('invisible')
        elif change=='deactivate':obstacle.GetPrim().SetActive(False)
        elif change=='new_prim':UsdGeom.Cube.Define(stage,'/World/Scene/New').CreateSizeAttr(1)
        elif change=='parent_transform':UsdGeom.XformCommonAPI(stage.GetPrimAtPath('/World/Scene')).SetTranslate(Gf.Vec3d(-2,0,0))
        else:obstacle.CreatePurposeAttr().Set('guide')
        assert cache()==reference(stage)
        assert cache.builds==builds+1


def test_generated_mesh_edits_invalidate_triangle_cache_not_just_bounds():
    stage,_,obstacle=fixture()
    mesh=UsdGeom.Mesh.Define(stage,'/World/GeneratedNativePilot/Plant/Mesh')
    mesh.CreatePointsAttr([(0,2,0),(.1,2,0),(0,2.1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    with module.StaticGeometryScreenCache(stage,'/World/Robot',include_generated_plants=True) as cache:
        assert cache()==reference(stage)
        mesh.GetPointsAttr().Set([(0,0,0),(.1,0,0),(0,.1,0)])
        assert cache()==reference(stage)
        assert cache.builds==2


def test_bookkeeping_does_not_invalidate_but_world_metadata_does():
    stage,_,_=fixture()
    with module.StaticGeometryScreenCache(stage,'/World/Robot') as cache:
        first=cache()
        UsdGeom.Xform.Define(stage,'/Render/Products')
        assert cache()==first and cache.builds==1
        stage.GetPrimAtPath('/World').CreateAttribute('reviewNote',Sdf.ValueTypeNames.String).Set('changed')
        cache()
        assert cache.builds==2


def test_notice_during_screen_fails_closed_and_next_call_rebuilds():
    stage,robot,_=fixture()
    with module.StaticGeometryScreenCache(stage,'/World/Robot') as cache:
        cache()
        original=module.visible_bounds
        def changing(*args,**kwargs):
            bounds=original(*args,**kwargs)
            UsdGeom.XformCommonAPI(robot).SetTranslate(Gf.Vec3d(3,0,0))
            return bounds
        with patch.object(module,'visible_bounds',changing),pytest.raises(ValueError,match='Scene changed'):
            cache()
        assert cache()==reference(stage)
        assert cache.builds==2


def test_closed_and_invalid_arguments_fail():
    stage,_,_=fixture()
    with pytest.raises(ValueError):module.StaticGeometryScreenCache(stage,'/Missing')
    with pytest.raises(ValueError):module.StaticGeometryScreenCache(stage,'/World/Robot',include_generated_plants=1)
    cache=module.StaticGeometryScreenCache(stage,'/World/Robot');cache.close();cache.close()
    with pytest.raises(ValueError):cache()


def test_native_profile_requires_exact_reference_equality():
    stage,_,_=fixture()
    with module.StaticGeometryScreenCache(stage,'/World/Robot',include_generated_plants=True) as cache:
        expected=reference(stage)
        proof=module.profile_against_reference(cache,expected)
        assert proof['both_results_equal_full_reference'] and cache.hits==1
        wrong=dict(expected,passed=not expected['passed'])
        with pytest.raises(ValueError,match='differs from full reference'):
            module.profile_against_reference(cache,wrong)
