import numpy as np
import pytest
from sim_physics.gripper_probe import ramp,palm_frame,bilateral
from sim_physics.plant_test import native


def test_palm_axes_and_nominal_jaw_centre():
    point=np.array([.1,.4,.7]);frame=palm_frame(point,[.2,.5,-.1])
    np.testing.assert_allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-12)
    np.testing.assert_allclose(frame@np.array([0,0,-.1025,1]),np.r_[point,1],atol=1e-12)
    assert np.isclose(np.linalg.det(frame[:3,:3]),1)


def test_opposed_contacts_require_both_fingers_and_nonzero_force():
    assert bilateral([[.1,0,0],[-.1,0,0]])
    assert not bilateral([[.1,0,0],[0,0,0]])
    assert not bilateral([[.1,0,0],[.1,0,0]])
    assert not bilateral([[float('nan'),0,0],[-.1,0,0]])


def test_fixture_ramp_never_jumps_or_exceeds_bounds():
    assert ramp(-1,0,1)==0 and ramp(2,0,1)==1
    assert ramp(.5,0,1)==.5


def test_camera_setup_can_be_repeated_without_duplicate_transform_ops():
    from pxr import Usd,UsdGeom
    from sim_physics.gripper_probe import setup_probe_camera
    stage=Usd.Stage.CreateInMemory()
    before=stage.GetRootLayer().ExportToString()
    for target in ([0.,0.,1.],[.1,.2,.9],[0.,0.,1.]):
        path=setup_probe_camera(stage,np.asarray(target))
        assert len(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps())==1
    assert stage.GetRootLayer().ExportToString()==before


def test_actual_gripper_is_session_only_dynamic_fingers_without_target_weld(native):
    from pxr import Usd,UsdPhysics
    from sim_physics.plant import build
    from sim_physics.gripper_probe import GripperFixture
    stage,record=native
    rig=build(stage,record,'SubStem_41')
    before=stage.GetRootLayer().ExportToString()
    fixture=GripperFixture(stage,rig,arc=.08,friction=.5)
    assert stage.GetRootLayer().ExportToString()==before
    assert not fixture.report()['arm_ik_executed']
    assert not fixture.report()['grasp_weld']
    assert fixture.arc>.04
    for i,path in enumerate(fixture.paths):
        body=UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
        assert body.GetKinematicEnabledAttr().Get()==(i==0)
        if i: assert np.isclose(UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get(),.03326999)
    joints=[UsdPhysics.Joint(p) for p in Usd.PrimRange(stage.GetPrimAtPath(fixture.root)) if p.IsA(UsdPhysics.Joint)]
    assert len(joints)==2
    for joint in joints:
        paths=list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())
        assert all(str(path).startswith(fixture.root+'/') for path in paths)
    fixture.close(1.)
    assert all(d.GetTargetPositionAttr().Get()==0 for d in fixture.drives)
    fixture.restore_authored_state()
    np.testing.assert_allclose([d.GetTargetPositionAttr().Get() for d in fixture.drives],[-.025,.025])
    assert stage.GetRootLayer().ExportToString()==before


@pytest.mark.parametrize('extra',[[],['--finger-friction','nan'],['--finger-friction','1.5'],
    ['--grasp-arc-m','.001'],['--scene','package'],['--seconds','6'],['--interactive']])
def test_probe_rejects_unsupported_configuration_before_kit(tmp_path,extra):
    from sim_physics.benchmark import main
    output=tmp_path/'absent'
    args=['--output',str(output),'--gripper-probe']
    if extra: args+=['--spring-mode','implicit_effort','--solver','PGS','--seconds','7']
    with pytest.raises(ValueError,match='Gripper probe requires'):
        main(args+extra)
    assert not output.exists()
