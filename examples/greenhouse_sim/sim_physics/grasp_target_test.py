import numpy as np
import pytest
from sim_physics.plant_test import native


def test_nearer_ground_truth_site_keeps_original_pad_and_cut_clearance(native):
    from pxr import Gf,UsdGeom
    from sim_physics.plant import build
    from sim_physics.full_robot import FullRobotGripper
    from sim_physics.grasp_target import finger_seam_clearance
    stage,record=native;stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005,0,.9))
    rig=build(stage,record,'SubStem_41')
    robot=FullRobotGripper(stage,rig,arc=.05,approach_tilt=10,grasp_roll=180,grasp_depth=.125,
        approach_distance=.02,station_offset=(.06,.24),station_yaw=60,torso_degrees=[0]*6,ground_height=lambda x,y:.101)
    target=robot.report()['ground_truth_grasp']
    assert target['target_petiole']=='seed101_full/SubStem_41'
    assert target['requested_arc_m']==.05
    assert target['selected_body_centre_arc_m']==pytest.approx(.0466755728)
    assert .020<target['placement_screen']['minimum_finger_to_cut_plane_m']<.021
    # A hand centred on the cut/junction cannot be legitimized as a grasp.
    unsafe=robot.goal.copy();axis=rig.rest_frames[rig.cut_index,:3,2]
    unsafe[:3,3]-=(robot.arc-.01)*axis
    with pytest.raises(ValueError,match='protected cut corridor'):
        finger_seam_clearance(stage,robot.root,unsafe,rig.chain_world[rig.cut_index],axis)


def test_markers_follow_native_frames_are_guides_and_have_no_physics(native):
    from pxr import Usd,UsdGeom,UsdPhysics
    from sim_physics.plant import build
    from sim_physics.grasp_target import TargetMarkers
    stage,record=native;rig=build(stage,record,'SubStem_41')
    before=stage.GetRootLayer().ExportToString()
    markers=TargetMarkers(stage,rig,2)
    assert not markers.visible and markers.root.GetPrim().GetAttribute('tomato:diagnosticOnly').Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    markers.toggle();assert markers.visible
    frames=rig.rest_frames.copy();frames[2,:3,3]+=[.001,.002,.003]
    markers.update(frames)
    np.testing.assert_allclose(markers.ops['Grasp'].Get(),frames[2,:3,3])
    np.testing.assert_allclose(markers.ops['Attachment'].Get(),rig.chain_world[0])
    assert all(not p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI)
        for p in Usd.PrimRange(markers.root.GetPrim()))
    markers.toggle();assert not markers.visible
    assert stage.GetRootLayer().ExportToString()==before
