import numpy as np
import pytest
from sim_physics.plant_test import native


def test_finger_gravity_is_inside_total_force_budget():
    from sim_physics.full_robot import finger_force_budget
    np.testing.assert_allclose(finger_force_budget([.31,-.31]),[.19,.19])
    np.testing.assert_allclose(finger_force_budget([0,0]),[.5,.5])
    with pytest.raises(ValueError): finger_force_budget([float('nan'),0])
    with pytest.raises(ValueError): finger_force_budget([.41,0])


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


@pytest.mark.parametrize('option',['--sparse-contacts','--finger-gravity','--profile'])
def test_robot_options_are_not_silently_ignored(tmp_path,option):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError):
        main(['--output',str(tmp_path/'unused'),option])
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
    robot=FullRobotGripper(stage,rig)
    assert stage.GetRootLayer().ExportToString()==before
    assert not robot.report()['grasp_weld']
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
