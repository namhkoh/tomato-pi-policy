"""Portable actuator/guard tests; fake tensor views and in-memory USD, no SimApp."""
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sim_physics.contact_events import ContactEvents
from sim_physics.full_robot import (
    FullRobotGripper, finger_force_budget, finger_contact_forces,
)
from sim_physics.gripper_probe import GripperFixture

ROOT='/World/RBY1'
TARGET='/World/InteractionPhysics/Target'
FINGERS=(ROOT+'/ee_finger_l1',ROOT+'/ee_finger_l2')
LEFT=('gripper_finger_l1','gripper_finger_l2')
RIGHT=('gripper_finger_r1','gripper_finger_r2')


def fixture(limit=.8,sparse=True):
    f=FullRobotGripper.__new__(FullRobotGripper)
    f.finger_actuator_limit_n=limit
    f.sparse_contacts=sparse
    f.root=ROOT
    f.paths=[ROOT+'/ee_left',*FINGERS]
    f.body_paths=f.paths.copy()
    f.body_index=0
    f.rig=SimpleNamespace(root=TARGET,body_paths=[TARGET+'/Branch/Segment_000'],
        chain_world=np.zeros((2,3)),rest_frames=np.array([np.eye(4)]),cut_index=0,arcs=[0,.02],
        source_target='SubStem_41')
    f.grasp_path=f.rig.body_paths[0]
    f.event_monitor=ContactEvents(robot_root=ROOT,target_root=TARGET,fingers=FINGERS,floor_root=None)
    f.event_monitor.native_full_contact_reporting=True
    f.event_monitor.native_friction_type='patch'
    f.index=np.array([0],dtype=np.uint32)
    f.pose={**{'left_arm_'+str(i):0. for i in range(7)},'torso_0':0.}
    f.slides=dict(zip((*LEFT,*RIGHT),(-.025,.025,-.025,.025)))
    f.effort={name:100. for name in f.pose}
    f.finger_gravity=True
    f.finger_compensation=np.zeros(2)
    f.floor_root=None
    f.anchor=ROOT+'/joints/probe_world_fixed'
    f.names=[*f.pose,*LEFT,*RIGHT]
    f.robot=Mock()
    f.robot.count=1
    f.robot.shared_metatype=SimpleNamespace(fixed_base=True,dof_names=f.names)
    f.robot.get_dof_positions.return_value=np.zeros((1,len(f.names)))
    f.robot_bodies=SimpleNamespace(prim_paths=f.body_paths,
        get_velocities=lambda:np.zeros((len(f.body_paths),6)))
    f.expected_palm=np.eye(4)
    f.start=np.eye(4)
    f.goal=np.eye(4)
    f.goal[0,3]=.02
    f.base=np.eye(4)
    f.fractions=np.linspace(0,1.15,47)
    f.path_q=np.zeros((47,7))
    f.kin=SimpleNamespace(forward=lambda *a:np.eye(4),
        default_torso_degrees=lambda:np.zeros(6),
        all_link_transforms=lambda *a,**k:{'ee_left':np.eye(4)})
    f.window=None
    f.asset='source-robot-asset'
    f.pregrasp_ik_attempts=1
    f.arc=f.requested_grasp_arc=.075
    f.grasp_clearance={}
    f.station_offset=np.zeros(2)
    f.station_yaw=0.
    f.explicit_station_pose=None
    f.collider_paths=[]
    f.mount_exclusions=[]
    f.finger_contact_compliance={'stiffness_n_m':1000.,'calibrated':False}
    f.approach_tilt=10.
    f.grasp_roll=180
    f.grasp_skew=0.
    f.grasp_depth=.125
    f.approach_distance=.02
    f.approach_side=1
    f.minimum_interarm=.1
    return f


def forces(pairs,dt=1.):
    return finger_contact_forces(pairs,dt,fingers=FINGERS)


@pytest.mark.parametrize('value',[True,False,np.bool_(True),None,'0.8',0,.6,1.,
    float('nan'),float('inf'),-float('inf'),np.array(.8),complex(.8),np.nextafter(.8,1.)])
def test_only_two_finite_nonbool_actuator_limits(value):
    with pytest.raises(ValueError,match='exactly 0.5 or 0.8'):
        FullRobotGripper(None,None,finger_actuator_limit_n=value)
    with pytest.raises(ValueError):
        finger_force_budget([0,0],value)


@pytest.mark.parametrize('flags',[v for v in product((False,True),repeat=3) if not all(v)])
def test_new_configuration_requires_all_three_flags_before_authoring(flags):
    with pytest.raises(ValueError,match='requires sparse_contacts'):
        FullRobotGripper(None,None,finger_actuator_limit_n=.8,
            sparse_contacts=flags[0],finger_gravity=flags[1],compliant_fingers=flags[2])


@pytest.mark.parametrize('flag',['sparse_contacts','finger_gravity','compliant_fingers'])
def test_new_configuration_does_not_accept_truthy_nonbooleans(flag):
    kwargs=dict(sparse_contacts=True,finger_gravity=True,compliant_fingers=True)
    kwargs[flag]=1
    with pytest.raises(ValueError,match='requires sparse_contacts'):
        FullRobotGripper(None,None,finger_actuator_limit_n=.8,**kwargs)


@pytest.mark.parametrize('limit',[.5,.8,np.float64(.8)])
def test_signed_gravity_and_both_drive_extremes_stay_inside_total(limit):
    for g in product((-.4,-.314,0.,.314,.4),repeat=2):
        budget=finger_force_budget(g,limit)
        np.testing.assert_allclose(budget,float(limit)-np.abs(g))
        for signs in product((-1.,1.),repeat=2):
            assert np.all(np.abs(np.array(g)+budget*signs)<=float(limit)+1e-15)
    np.testing.assert_allclose(finger_force_budget([-.314,-.314],.8),[.486,.486])


@pytest.mark.parametrize('gravity',[[.40000001,0],[-.40000001,0],[float('nan'),0],
    [0,float('inf')],[0],[[0,0]]])
def test_existing_gravity_envelope_is_not_relaxed(gravity):
    with pytest.raises(ValueError):
        finger_force_budget(gravity,.8)


@pytest.mark.parametrize('limit',[.5,.8])
def test_author_reset_bind_and_runtime_preserve_left_total_and_right_legacy(monkeypatch,limit):
    from pxr import Usd,UsdGeom,UsdPhysics
    f=fixture(limit)
    f.stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(f.stage,ROOT+'/ee_left')
    for name in (*f.pose,*f.slides):
        joint=UsdPhysics.PrismaticJoint if name in f.slides else UsdPhysics.RevoluteJoint
        joint.Define(f.stage,ROOT+'/joints/'+name)
    f._author_initial_joints()
    for name in (*LEFT,*RIGHT):
        drive=UsdPhysics.DriveAPI(f.stage.GetPrimAtPath(ROOT+'/joints/'+name),'linear')
        assert drive.GetMaxForceAttr().Get()==pytest.approx(limit if name in LEFT else .5)
        assert drive.GetStiffnessAttr().Get()==200
        assert drive.GetDampingAttr().Get()==5
        drive.GetMaxForceAttr().Set(.01)
    f.restore_authored_state()
    for name in (*LEFT,*RIGHT):
        drive=UsdPhysics.DriveAPI(f.stage.GetPrimAtPath(ROOT+'/joints/'+name),'linear')
        assert drive.GetMaxForceAttr().Get()==pytest.approx(limit if name in LEFT else .5)

    monkeypatch.setattr(GripperFixture,'bind',lambda *args:None)
    def subscribe(monitor):
        monitor.native_full_contact_reporting=True
        monitor.native_friction_type='patch'
    monkeypatch.setattr(ContactEvents,'subscribe',subscribe)
    view=SimpleNamespace(create_articulation_view=lambda *a:f.robot,
        create_rigid_body_view=lambda *a:f.robot_bodies)
    f.bind(view)
    original_caps=f.robot.set_dof_max_forces.call_args.args[0].copy()
    for name in (*LEFT,*RIGHT):
        i=f.names.index(name)
        assert original_caps[0,i]==pytest.approx(limit if name in LEFT else .5)
        assert f.robot.set_dof_stiffnesses.call_args.args[0][0,i]==200
        assert f.robot.set_dof_dampings.call_args.args[0][0,i]==5
    gravity=np.zeros_like(f.targets)
    gravity[0,f.finger_indices]=[-.314,.314]
    gravity[0,[f.names.index(n) for n in RIGHT]]=[-.2,.2]
    f.robot.get_gravity_compensation_forces.return_value=gravity
    f.target_palm(np.zeros(3))
    np.testing.assert_allclose(f.force_limits[0,f.finger_indices],limit-.314,atol=1e-7)
    for name in RIGHT:
        i=f.names.index(name)
        assert f.force_limits[0,i]==.5
        assert f.robot.set_dof_actuation_forces.call_args.args[0][0,i]==0.
    np.testing.assert_allclose(f.robot.set_dof_actuation_forces.call_args.args[0][0,f.finger_indices],
        [-.314,.314],atol=1e-7)
    report=f.report()
    assert report['left_finger_total_actuator_limit_n']==limit
    assert report['finger_max_drive_force_n']==report['total_finger_effort_limit_n']==limit
    assert report['right_finger_legacy_drive_limit_n']==.5
    protocol=report['finger_actuator_protocol']
    assert protocol['changed_protocol'] and protocol['normal_friction_contact_guard_enabled']
    assert protocol['actuator_prior_changed']==(limit==.8)
    assert protocol['per_finger_contact_limit_n']==.5
    assert not protocol['calibrated'] and not protocol['hardware_limit_claim']
    assert not protocol['overshoot_prevention_claim']


def test_all_objects_both_orders_and_exact_finger_path_boundaries():
    pairs={(FINGERS[0]+'/pad',TARGET+'/Branch/Segment_003/StemCollider'):.1,
        (TARGET+'/Branch/Segment_005/Leaf000',FINGERS[0]+'/pad'):.2,
        (FINGERS[0],TARGET+'/Support/StemCollider'):.05,
        (TARGET,FINGERS[1]):.12,
        (FINGERS[0]+'0/pad',TARGET+'/Leaf'):.9,
        (FINGERS[0]+'/pad',TARGET+'Extra/Stem'):.9,
        (ROOT+'/ee_left/palm',TARGET+'/Stem'):.9,
        (FINGERS[0]+'/pad',FINGERS[1]+'/pad'):.2}
    assert forces(pairs)==pytest.approx({FINGERS[0]:1.45,FINGERS[1]:.32})
    assert forces(pairs,.5)==pytest.approx({FINGERS[0]:2.9,FINGERS[1]:.64})


@pytest.mark.parametrize('limit',[.5,.8])
def test_opposing_normal_rows_and_friction_cannot_cancel_or_bypass(limit):
    f=fixture(limit)
    m=f.event_monitor
    m.consume(FINGERS[0]+'/pad',TARGET+'/Stem',
        [(+.2,0,0),(-.2,0,0)],friction_impulses=[(0,.101,0)])
    with pytest.raises(RuntimeError,match='reached 0.5 N'):
        f.check(1.,np.eye(4))
    assert f.last_fault['per_finger_contact_upper_bound_n'][FINGERS[0]]==pytest.approx(.501)
    assert len(f.last_fault['pairs'])==1


@pytest.mark.parametrize('limit',[.5,.8])
@pytest.mark.parametrize('load,fail',[(.499999,False),(.5,True),(.500001,True)])
def test_strict_contact_boundary_identical_in_both_modes(limit,load,fail):
    f=fixture(limit)
    f.event_monitor.consume(FINGERS[1]+'/pad',TARGET+'/Stem',[(load,0,0)])
    if fail:
        with pytest.raises(RuntimeError,match='reached 0.5 N'):
            f.check(1.,np.eye(4))
    else:
        result=f.check(1.,np.eye(4))
        assert result['per_finger_contact_upper_bound_n'][FINGERS[1]]==load


def test_sum_across_distinct_colliders_not_selected_only_or_jaw_combined():
    f=fixture()
    f.event_monitor.consume(FINGERS[0]+'/pad',TARGET+'/Selected',[(.24,0,0)])
    f.event_monitor.consume(TARGET+'/Leaf',FINGERS[0]+'/pad',[(.27,0,0)])
    with pytest.raises(RuntimeError,match='reached 0.5 N'): f.check(1.,np.eye(4))
    f.event_monitor.begin_step()
    for finger in FINGERS:
        f.event_monitor.consume(finger+'/pad',TARGET+'/Stem',[(.49,0,0)])
    assert sum(f.check(1.,np.eye(4))['per_finger_contact_upper_bound_n'].values())==pytest.approx(.98)
    f.event_monitor.begin_step()
    assert f.check(1.,np.eye(4))['per_finger_contact_upper_bound_n']==dict.fromkeys(FINGERS,0.)


@pytest.mark.parametrize('limit',[.5,.8])
@pytest.mark.parametrize('fault',['missing','normal_only','wrong_friction','callback_error'])
def test_missing_or_invalid_full_contact_stream_cannot_authorize(limit,fault):
    f=fixture(limit)
    if fault=='missing': f.event_monitor=None
    elif fault=='normal_only': f.event_monitor.native_full_contact_reporting=False
    elif fault=='wrong_friction': f.event_monitor.native_friction_type='two_directional'
    else: f.event_monitor.error='bad native callback'
    with pytest.raises(RuntimeError): f.check(1.,np.eye(4))


@pytest.mark.parametrize('kind',['other_plant','self','tracking','speed'])
def test_existing_robot_guards_unchanged(kind):
    f=fixture()
    if kind=='other_plant':
        f.event_monitor.consume(ROOT+'/link_left_arm_0/collider','/World/ContextPlants/Leaf',
            [],friction_impulses=[(.501,0,0)])
    elif kind=='self':
        f.event_monitor.consume(ROOT+'/link_left_arm_0/collider',ROOT+'/ee_left/palm',[(3.001,0,0)])
    elif kind=='tracking': f.expected_palm[0,3]=.012001
    else: f.robot_bodies.get_velocities=lambda:np.array([[3.001,0,0,0,0,0]])
    with pytest.raises(RuntimeError,match='Full robot contact/tracking guard'):
        f.check(1.,np.eye(4))


def test_non_sparse_legacy_unchanged_and_report_labels_it():
    f=fixture(.5,sparse=False)
    f.event_monitor=None
    f.plant_contacts=SimpleNamespace(sensor_paths=[FINGERS[0]],
        get_contact_force_matrix=lambda dt:np.array([[[.7,0,0]]]))
    f.self_contacts=SimpleNamespace(get_contact_force_matrix=lambda dt:np.zeros((1,1,3)))
    result=f.check(1.,np.eye(4))
    assert 'per_finger_contact_upper_bound_n' not in result
    protocol=f.report()['finger_actuator_protocol']
    assert not protocol['changed_protocol'] and not protocol['normal_friction_contact_guard_enabled']
    assert protocol['name']=='legacy_engineering_0p5'


@pytest.mark.parametrize('dt',[True,0.,-1.,float('nan'),float('inf'),'1'])
def test_invalid_timestep_rejected(dt):
    with pytest.raises(ValueError): forces({},dt)


@pytest.mark.parametrize('impulse',[-.1,float('nan'),float('inf'),True,[.1,0,0]])
def test_invalid_pair_impulse_rejected(impulse):
    with pytest.raises(ValueError):
        forces({(FINGERS[0]+'/pad',TARGET+'/Stem'):impulse})


def test_overflow_fails_closed():
    pairs={(FINGERS[0]+'/pad',TARGET+'/A'):1e308,
        (FINGERS[0]+'/pad',TARGET+'/B'):1e308}
    with pytest.raises(ValueError,match='Overflow'): forces(pairs)
    with pytest.raises(ValueError,match='Overflow'):
        forces({(FINGERS[0],TARGET):1.},1e-320)


@pytest.mark.parametrize('limit',[.5,.8])
@pytest.mark.parametrize('other',[TARGET+'/Leaf','/World/ContextPlants/Leaf',
    '/World/Gutters/Gutter_31','/World/Floor',ROOT+'/ee_left/palm'])
def test_all_contact_scope_rejects_friction_load_before_further_decision(limit,other):
    f=fixture(limit)
    dt=1/240
    f.event_monitor.consume(other,FINGERS[0]+'/pad',[],
        friction_impulses=[(0,.501*dt,0)])
    assert forces(f.event_monitor.pairs,dt)[FINGERS[0]]==pytest.approx(.501)
    with pytest.raises(RuntimeError,match='all-contact'):
        f.check(dt,np.eye(4))
    assert f.last_fault['max_finger_contact_upper_bound_n']==pytest.approx(.501)
    assert f.last_fault['finger_contact_guard_is_plant_only'] is False


def test_one_pair_once_per_finger_even_both_colliders_on_same_body():
    same={(FINGERS[0]+'/padA',FINGERS[0]+'/padB'):.3}
    assert forces(same)==pytest.approx({FINGERS[0]:.3,FINGERS[1]:0.})
    between={(FINGERS[0]+'/pad',FINGERS[1]+'/pad'):.3}
    assert forces(between)==pytest.approx(dict.fromkeys(FINGERS,.3))
    # Header ordering does not affect ContactEvents' single canonical pair.
    f=fixture()
    f.event_monitor.consume(FINGERS[0]+'/pad',FINGERS[1]+'/pad',[(.1,0,0)])
    f.event_monitor.consume(FINGERS[1]+'/pad',FINGERS[0]+'/pad',[(.2,0,0)])
    assert len(f.event_monitor.pairs)==1
    assert forces(f.event_monitor.pairs)==pytest.approx(dict.fromkeys(FINGERS,.3))


@pytest.mark.parametrize('path',[ROOT+'/ee_finger_l10/pad',ROOT+'/ee_finger_l1_extra/pad',
    ROOT+'/ee_finger_l2extra/pad','/World/RBY10/ee_finger_l1/pad'])
def test_finger_prefix_lookalikes_do_not_count(path):
    assert forces({(path,TARGET+'/Stem'):9.})==dict.fromkeys(FINGERS,0.)
