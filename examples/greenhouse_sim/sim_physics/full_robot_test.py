import numpy as np
import pytest
from sim_physics.plant_test import native


def test_finger_gravity_is_inside_total_force_budget():
    from sim_physics.full_robot import finger_force_budget
    np.testing.assert_allclose(finger_force_budget([.31,-.31]),[.19,.19])
    np.testing.assert_allclose(finger_force_budget([0,0]),[.5,.5])
    with pytest.raises(ValueError): finger_force_budget([float('nan'),0])
    with pytest.raises(ValueError): finger_force_budget([.41,0])


def test_compliant_pad_parameters_are_force_based_and_reject_bad_mass():
    from sim_physics.full_robot import finger_compliance
    p=finger_compliance([.03,.03],.001)
    assert p['stiffness_n_m']==1000 and p['force_based'] and not p['calibrated']
    assert p['reduced_mass_kg']==pytest.approx(.03*.001/.031)
    assert p['damping_n_s_m']==pytest.approx(1.4*np.sqrt(1000*.03*.001/.031))
    with pytest.raises(ValueError): finger_compliance([0,.03],.001)
    with pytest.raises(ValueError): finger_compliance([.03,.03],float('nan'))


@pytest.mark.parametrize('extra',[
    [],['--diagnostic-detach'],['--physics-hz','480'],['--scene','package'],
    ['--finger-friction','nan'],['--grasp-arc-m','.001'],['--gripper-probe'],
    ['--approach-tilt','nan'],['--approach-tilt','31']])
def test_full_robot_configuration_fail_closed_before_kit(tmp_path,extra):
    from sim_physics.benchmark import main
    args=['--output',str(tmp_path/'unused'),'--full-robot-probe']
    if extra: args+=['--spring-mode','implicit_effort','--solver','PGS','--seconds','7']
    with pytest.raises(ValueError,match='Full robot probe requires'):
        main(args+extra)
    assert not (tmp_path/'unused').exists()


def test_interactive_robot_needs_visible_window(tmp_path):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError,match='Robot interactive requires'):
        main(['--output',str(tmp_path/'unused'),'--robot-interactive'])


@pytest.mark.parametrize('option',['--sparse-contacts','--finger-gravity','--profile','--local-wire-physics','--batch-gutter-visuals','--scene-profile','--bimanual-hold-control'])
def test_robot_options_are_not_silently_ignored(tmp_path,option):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError):
        main(['--output',str(tmp_path/'unused'),option])
    assert not (tmp_path/'unused').exists()


def test_capture_can_be_disabled_without_disabling_rendering():
    from sim_physics.benchmark import parser
    args=parser().parse_args(['--output','unused','--render-hz','30','--no-capture-milestones'])
    assert args.render_hz==30 and not args.capture_milestones


@pytest.mark.parametrize('offset',[['nan','0'],['.31','0'],['.25','.25']])
def test_station_offset_fails_before_kit(tmp_path,offset):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError,match='Station offset requires'):
        main(['--output',str(tmp_path/'unused'),'--full-robot-probe','--station-offset',*offset])
    assert not (tmp_path/'unused').exists()


def test_station_offset_requires_full_robot(tmp_path):
    from sim_physics.benchmark import main,parser
    with pytest.raises(ValueError,match='Station offset requires'):
        main(['--output',str(tmp_path/'unused'),'--station-offset','.2','.2'])
    assert parser().parse_args(['--output','unused']).station_offset is None


@pytest.mark.parametrize('option,value',[
    ('--torso-yaw','nan'),('--torso-yaw','46'),('--torso-yaw','20'),
    ('--station-yaw','nan'),('--station-yaw','91'),('--station-yaw','45'),
    ('--approach-distance','nan'),('--approach-distance','.009'),
    ('--approach-distance','.081'),('--approach-distance','.02'),('--grasp-roll','180')])
def test_initial_pose_options_fail_closed_without_qualified_full_robot(tmp_path,option,value):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError): main(['--output',str(tmp_path/'unused'),option,value])
    assert not (tmp_path/'unused').exists()


def test_default_initial_pose_is_preserved():
    from sim_physics.benchmark import parser
    args=parser().parse_args(['--output','unused'])
    assert args.grasp_roll==args.torso_yaw==args.station_yaw==0 and args.approach_distance==.08


def test_shorter_approach_does_not_move_base_or_grasp_and_roll_preserves_geometry(native):
    from pxr import Gf,UsdGeom
    from sim_physics.plant import build
    from sim_physics.full_robot import FullRobotGripper
    stage,record=native
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(0,0,.9))
    rig=build(stage,record,'SubStem_41')
    kwargs=dict(approach_tilt=10,torso_degrees=[0]*6,ground_height=lambda x,y:.101)
    first=FullRobotGripper(stage,rig,**kwargs)
    first_base=first.base.copy();first_goal=first.goal.copy()
    stage.RemovePrim(first.root)
    second=FullRobotGripper(stage,rig,approach_distance=.02,grasp_roll=180,**kwargs)
    np.testing.assert_allclose(first_base,second.base,atol=1e-9)
    np.testing.assert_allclose(first_goal[:3,3],second.goal[:3,3],atol=1e-9)
    np.testing.assert_allclose(first_goal[:3,2],second.goal[:3,2],atol=1e-9)
    np.testing.assert_allclose(first_goal[:3,:2],-second.goal[:3,:2],atol=1e-9)
    assert np.linalg.norm(second.start[:3,3]-second.goal[:3,3])==pytest.approx(.02)


def test_cutting_cannot_silently_run_without_native_contact_and_gripper_guards(tmp_path):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError,match='Bimanual cutting requires'):
        main(['--output',str(tmp_path/'unused'),'--bimanual-cut'])
    assert not (tmp_path/'unused').exists()


def test_complete_robot_native_joints_and_no_plant_weld_are_session_only(native):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    from sim_physics.plant import build
    from sim_physics.full_robot import FullRobotGripper
    stage,record=native
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(0,0,.35))
    rig=build(stage,record,'SubStem_41')
    before=stage.GetRootLayer().ExportToString()
    robot=FullRobotGripper(stage,rig,compliant_fingers=True)
    assert stage.GetRootLayer().ExportToString()==before
    assert not robot.report()['grasp_weld']
    pad=stage.GetPrimAtPath(robot.root+'/ProbeFingerMaterial')
    assert pad.GetAttribute('physxMaterial:compliantContactStiffness').Get()==1000
    assert pad.GetAttribute('physxMaterial:compliantContactAccelerationSpring').Get() is False
    assert robot.minimum_interarm>.01
    assert len(robot.collider_paths)>20
    for name in ('base','link_torso_3','link_left_arm_3','link_right_arm_3','ee_left','ee_finger_l1'):
        body=UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(robot.root+'/'+name))
        assert body.GetRigidBodyEnabledAttr().Get()
        assert not body.GetKinematicEnabledAttr().Get()
    joints=[UsdPhysics.Joint(p) for p in Usd.PrimRange(stage.GetPrimAtPath(robot.root)) if p.IsA(UsdPhysics.Joint)]
    assert len(joints)>30
    for joint in joints:
        targets=list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())
        assert all(str(path).startswith(robot.root+'/') for path in targets)
    assert stage.GetPrimAtPath(robot.anchor).HasAPI(UsdPhysics.ArticulationRootAPI)
    robot.restore_authored_state()
    assert stage.GetRootLayer().ExportToString()==before
    assert np.isfinite(robot.path_q).all()
