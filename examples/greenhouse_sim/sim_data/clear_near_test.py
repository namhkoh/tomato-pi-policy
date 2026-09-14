import numpy as np
import pytest
from sim_data.training_plan import configuration,view_specs,prepared_view_specs
from greenhouse_sim.robot_model import DEFAULT_ASSET


@pytest.mark.parametrize('opposite',[False,True])
@pytest.mark.parametrize('orbit',[False,True])
def test_closer_proposals_are_explicit_bounded_and_native_reconstructible(opposite,orbit):
    opts=dict(vary_torso=True,clear_capture=True,near_clear=True,
              opposite_aisle=opposite,oblique_clear=orbit,orbit_clear=orbit)
    p=dict(target_world_m={'target':[0,0,1.3]},requested_views_per_target=3,**opts)
    rows=view_specs(.8,0,'target',3,**opts)
    assert list(prepared_view_specs(p,'target',.8).values())==rows
    assert rows==view_specs(.8,0,'target',3,**opts)
    old=view_specs(.8,0,'target',3,**{**opts,'near_clear':False})
    for r,o in zip(rows,old):
        assert r['near_clear'] and r['candidate_id'].startswith('near_')
        assert (r['root_x_m']<0)==opposite
        distance=np.hypot(r['root_x_m'],r['y_offset_m']) if orbit else abs(r['root_x_m'])
        assert .20<=distance<=.40
        assert r['desired_pixel_xy']==o['desired_pixel_xy'] and r['torso_bend_degrees']==o['torso_bend_degrees']
    for bad in (dict(near_clear=True),dict(near_clear=True,clear_capture=True,vary_torso=True,oblique_clear=True)):
        with pytest.raises(ValueError):configuration(views=3,**bad)


@pytest.mark.skipif(not DEFAULT_ASSET.is_file(),reason='Built robot required')
@pytest.mark.parametrize('opposite',[False,True])
@pytest.mark.parametrize('orbit',[False,True])
def test_closer_snapshot_uses_real_mount_fk_and_floor(opposite,orbit):
    from pxr import Usd,UsdGeom,Gf
    from sim_data.floor_alignment import PACKAGE_FLOOR
    from sim_data.robot_preview import add_robot_preview
    from sim_data.capture_scene import set_snapshot_pose,mounted_camera_to_head,calibration
    from sim_data.capture_contract import project
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    floor=UsdGeom.Mesh.Define(stage,PACKAGE_FLOOR)
    floor.CreatePointsAttr([Gf.Vec3f(-10,-10,.1),Gf.Vec3f(10,-10,.1),Gf.Vec3f(10,10,.1),Gf.Vec3f(-10,10,.1)])
    floor.CreateFaceVertexCountsAttr([3,3]);floor.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
    robot=add_robot_preview(stage,gutter_x=-.2,right_tool='gripper')
    mount=mounted_camera_to_head(stage).copy();target=[0,0,1.45]
    x=(-1 if opposite else 1)*(.15 if orbit else .25);y=.19 if orbit else 0
    yaw=float(np.rad2deg(np.arctan2(-y,-x))) if orbit else (0 if opposite else 180)
    if not opposite and yaw<0:yaw+=360
    kw=dict(root_x_m=x,root_yaw_degrees=yaw,opposite_aisle=opposite,orbit_clear=orbit,near_clear=True)
    pose=set_snapshot_pose(stage,robot,target,y,[424,204],**kw)
    assert np.allclose(mounted_camera_to_head(stage),mount,atol=1e-9)
    assert max(abs(w['clearance_m']) for w in pose['floor_alignment']['wheel_supports'])<1e-6
    assert np.allclose(project([target],calibration(stage))[0]['pixel_xy'],[424,204],atol=.02)
    assert pose['joint_limits_checked'] and not pose['motion_between_snapshots_validated']
    with pytest.raises(ValueError):set_snapshot_pose(stage,robot,target,y,[424,204],**{**kw,'root_x_m':x*4})
