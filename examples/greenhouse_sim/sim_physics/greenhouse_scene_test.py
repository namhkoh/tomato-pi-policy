import numpy as np
import pytest


def test_actual_package_placement_floor_and_collisions_are_session_only():
    from pxr import Usd,UsdGeom,UsdPhysics
    from sim_data.audit import DEFAULT_PACK
    from sim_physics.greenhouse_scene import prepare
    from sim_physics.plant import build
    from sim_physics.full_robot import FullRobotGripper
    scene=DEFAULT_PACK/'house/green_house_base.usd'
    if not scene.is_file(): pytest.skip('Supplied greenhouse package not installed')
    stage=Usd.Stage.Open(str(scene),load=Usd.Stage.LoadNone)
    before=stage.GetRootLayer().ExportToString()
    record,height,report=prepare(stage,DEFAULT_PACK,'seed101_full')
    assert report['gutters_in_asset']==75
    assert report['detailed_plants']==1 and report['distant_instanced_plants']==4
    assert not report['greenhouse_geometry_moved']
    np.testing.assert_allclose(report['plant_position_world_m'],[-.005,0,.9],atol=1e-6)
    assert report['active_collision_prims']>7000
    assert not stage.GetPrimAtPath('/PhysicsScene').IsActive()
    assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(report['selected_gutter']+'/Collision')).GetCollisionEnabledAttr().Get()
    rig=build(stage,record,'SubStem_41')
    robot=FullRobotGripper(stage,rig,ground_height=height,torso_degrees=[0.]*6,
        sparse_contacts=True,floor_root=report['floor_path'],finger_gravity=True,approach_tilt=10.)
    assert not stage.GetPrimAtPath('/World/RobotTestFloor')
    assert robot.base[2,3]==pytest.approx(height(*robot.base[:2,3])+.001)
    assert robot.minimum_interarm>.01
    assert all(v==0 for v in robot.report()['torso_degrees'])
    assert robot.report()['finger_gravity_compensation']
    assert report['demo_lights_adjusted']>0
    assert stage.GetRootLayer().ExportToString()==before
