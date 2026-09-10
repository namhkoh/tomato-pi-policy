import numpy as np
import pytest

from sim_physics.knife import ShearGate,KnifeGeometry
from sim_physics.plant_test import native


def sample(gate,x,**extra):
    edge=np.eye(4);edge[0,3]=-x
    data=dict(dt=.005,edge=edge,centre=np.zeros(3),axis=np.array([0.,0.,1.]),
        points=[[-x,0,0]],impulses=[[.00125,0,0]],held=True,slip=.001)
    data.update(extra)
    return gate.observe(**data)


def test_force_grasp_transverse_measured_motion_required():
    g=ShearGate('petiole');events=[sample(g,i*.0001) for i in range(12)]
    assert sum(e is not None for e in events)==1
    e=next(e for e in events if e)
    assert e['flat_edge_contact_verified'] and not e['commanded_motion_used_as_evidence']
    assert not e['tissue_fracture_calibrated']


@pytest.mark.parametrize('extra',[
    {'held':False},{'slip':.0031},{'slip':float('nan')},{'impulses':[[.0001,0,0]]},
    {'impulses':[[.01,0,0]]},{'points':[[0,0,.004]]},
    {'axis':np.array([1.,0,0])},{'points':[],'impulses':[]}])
def test_negative_controls_cannot_cut(extra):
    g=ShearGate('petiole')
    assert all(sample(g,i*.0001,**extra) is None for i in range(20))


def test_stationary_loading_reversed_motion_and_separate_taps_do_not_cut():
    for motion in ([0.]*20,[-i*.0001 for i in range(20)]):
        g=ShearGate('petiole')
        assert all(sample(g,x) is None for x in motion)
    g=ShearGate('petiole')
    assert all(sample(g,i*.0001,held=i%3!=0) is None for i in range(20))


def test_original_knife_and_blade_release_preserve_source(native):
    from pxr import Gf,UsdGeom,UsdPhysics
    from sim_physics.plant import build
    from sim_physics.bimanual import BimanualRobot
    stage,record=native
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(0,0,.9))
    rig=build(stage,record,'SubStem_41')
    original=stage.GetRootLayer().ExportToString()
    robot=BimanualRobot(stage,rig,sparse_contacts=True,approach_tilt=10,torso_degrees=[0]*6,
        ground_height=lambda x,y:.101)
    k=robot.knife
    assert robot.knife_mount['changed']
    assert k.local[2,3]<0
    from sim_physics.knife import mount_forward
    assert not mount_forward(stage,robot.root)['changed']
    assert k.size==pytest.approx([.002,.07147998,.013])
    frame=k.frame(np.eye(4))
    assert k.on_edge(frame[:3,3],frame)
    assert not k.on_edge(frame[:3,3]+.01*frame[:3,0],frame)
    desired=k.wrist_for_edge(np.zeros(3),np.array([-1.,0,0]),np.array([0.,0,1.]))
    np.testing.assert_allclose(k.frame(desired),np.eye(4),atol=1e-8)
    with pytest.raises(ValueError): rig.release_from_blade({})
    gate=ShearGate(rig.source_target)
    event=next(e for i in range(12) if (e:=sample(gate,i*.0001)) is not None)
    result=rig.release_from_blade(event)
    assert result['event']=='blade_contact_joint_release'
    assert not result['physical_cut_verified']
    assert not UsdPhysics.Joint.Get(stage,rig.cut_joint_path).GetJointEnabledAttr().Get()
    rig.restore_authored_state()
    assert stage.GetRootLayer().ExportToString()==original


def test_bimanual_replan_cannot_skip_self_screen(monkeypatch):
    from sim_physics.bimanual import BimanualRobot
    from sim_physics.full_robot import FullRobotGripper
    robot=object.__new__(BimanualRobot)
    robot.self_screen=object();robot.right=np.zeros(7)
    monkeypatch.setattr(FullRobotGripper,'plan_approach',lambda self:setattr(self,'path_q',np.zeros((3,7))))
    monkeypatch.setattr(robot,'check_self',lambda *args:dict(passed=False,minimum_clearance_m=-.01))
    with pytest.raises(RuntimeError,match='path index 0'): robot.plan_approach()


def test_known_bad_torso_station_rejected_before_native_physics(native):
    from pxr import Gf,UsdGeom
    from sim_physics.plant import build
    from sim_physics.bimanual import BimanualRobot
    stage,record=native;stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005,0,.9))
    rig=build(stage,record,'SubStem_41')
    with pytest.raises(RuntimeError,match='Pregrasp self-collision screen'):
        BimanualRobot(stage,rig,sparse_contacts=True,approach_tilt=10,grasp_roll=180,
            station_offset=(.16,.22),torso_degrees=[0,0,0,0,0,20],ground_height=lambda x,y:.101)
