import numpy as np
import pytest
from sim_data.training_plan import configuration,view_specs
from greenhouse_sim.robot_model import DEFAULT_ASSET


def test_target_facing_orbit_is_explicit_and_preserves_position_posture():
    with pytest.raises(ValueError):configuration(orbit_clear=True)
    with pytest.raises(ValueError):view_specs(.8,0,'a',3,orbit_clear=True)
    for opposite in (False,True):
        args=dict(vary_torso=True,clear_capture=True,oblique_clear=True,lean_clear=True,opposite_aisle=opposite)
        old=view_specs(.8,0,'a',3,**args);new=view_specs(.8,0,'a',3,orbit_clear=True,**args)
        assert new==view_specs(.8,0,'a',3,orbit_clear=True,**args)
        for a,b in zip(old,new):
            bearing=np.rad2deg(np.arctan2(-b['y_offset_m'],-b['root_x_m']))
            delta=(b['root_yaw_degrees']-bearing+180)%360-180
            assert abs(delta)<=30+1e-9 and b['orbit_clear']
            assert np.isclose(delta,a['root_yaw_degrees']-(0 if opposite else 180))
            assert all(a[k]==b[k] for k in ('root_x_m','y_offset_m','desired_pixel_xy','torso_bend_degrees','torso_lean_degrees'))


@pytest.mark.skipif(not DEFAULT_ASSET.is_file(),reason='Built robot required')
@pytest.mark.parametrize('opposite',[False,True])
def test_orbit_snapshot_preserves_mount_floor_and_real_head_ik(opposite):
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
    mount=mounted_camera_to_head(stage).copy();target=[0,0,1.3]
    x=-.25 if opposite else .25;y=.35
    yaw=float(np.rad2deg(np.arctan2(-y,-x)))
    if not opposite:yaw+=360
    kwargs=dict(root_x_m=x,root_yaw_degrees=yaw,opposite_aisle=opposite)
    with pytest.raises(ValueError):set_snapshot_pose(stage,robot,target,y,[424,204],**kwargs)
    pose=set_snapshot_pose(stage,robot,target,y,[424,204],orbit_clear=True,**kwargs)
    assert np.allclose(mounted_camera_to_head(stage),mount,atol=1e-9)
    assert pose['joint_limits_checked'] and not pose['motion_between_snapshots_validated']
    assert max(abs(w['clearance_m']) for w in pose['floor_alignment']['wheel_supports'])<1e-6
    assert np.allclose(project([target],calibration(stage))[0]['pixel_xy'],[424,204],atol=.02)
    for bad in (dict(root_x_m=x*3,root_yaw_degrees=yaw),dict(root_x_m=x,root_yaw_degrees=yaw+40)):
        with pytest.raises(ValueError):set_snapshot_pose(stage,robot,target,y,[424,204],opposite_aisle=opposite,orbit_clear=True,**bad)
