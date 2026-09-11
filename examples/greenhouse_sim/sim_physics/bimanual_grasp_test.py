"""Bimanual grasp integration with in-memory callbacks/views; never launch Kit."""
import json
import sys
from types import SimpleNamespace as S

import numpy as np
import pytest

from sim_physics.bimanual import BimanualRobot
from sim_physics.contact_events import ContactEvents
from sim_physics.full_robot import FullRobotGripper
from sim_physics.runtime import pose_matrices


def monitor():
    result=ContactEvents(robot_root='/R',target_root='/T',
        fingers=['/R/ee_finger_l1','/R/ee_finger_l2'],floor_root=None)
    result.native_full_contact_reporting=True
    return result


def contact_robot(*,bilateral=True,adapter_valid=True):
    robot=object.__new__(BimanualRobot);calls=[]
    # Native view order deliberately differs from authored finger order, with
    # different rotations as well as translations to catch position-only use.
    poses=np.array([[7.,8.,9.,0.,0.,1.,0.],[1.,2.,3.,0.,0.,0.,1.]])
    robot.order=[1,0]
    def transforms():calls.append(('finger_frames',));return poses.copy()
    robot.fingers=S(get_transforms=transforms)
    result=dict(bilateral=bilateral,adapter_valid=adapter_valid,stem_only=adapter_valid,
        counts=[1,1],forces=[[.03,0.,0.],[-.03,0.,0.]],points=[],min_separation=0.)
    def evaluate(dt,frames,fingers,*,frames_step_id):
        calls.append(('evaluate',dt,frames,fingers.copy(),frames_step_id))
        return result
    robot.grasp_observer=S(evaluate=evaluate,begin_step=lambda:calls.append(('begin',)),
        add_contact=lambda *args:None,close=lambda:calls.append(('close',)))
    robot.event_monitor=monitor()
    robot.event_monitor.normal_contact_observer=robot.grasp_observer
    return robot,calls,poses,result


@pytest.mark.parametrize('bilateral',[True,False])
def test_contact_passes_all_frames_exact_finger_order_and_clock_step(bilateral):
    robot,calls,poses,result=contact_robot(bilateral=bilateral)
    frames=np.repeat(np.eye(4)[None],5,axis=0);frames[:,2,3]=np.arange(5)*.02
    original=frames.copy()
    returned=robot.contact_with_frames(.004,frames,step_id=37)
    assert returned is result and robot.latest_finger_bilateral is bilateral
    assert calls[0]==('finger_frames',) and len(calls)==2
    name,dt,supplied,fingers,step=calls[1]
    assert name=='evaluate' and dt==.004 and supplied is frames and step==37
    np.testing.assert_allclose(fingers,pose_matrices(poses)[[1,0]],atol=0.,rtol=0.)
    np.testing.assert_array_equal(frames,original)


def test_full_native_stream_required_before_any_frame_or_adapter_read():
    robot,calls,_,_=contact_robot()
    robot.event_monitor.native_full_contact_reporting=False
    with pytest.raises(RuntimeError,match='full native contact stream'):
        robot.contact_with_frames(.01,np.eye(4)[None],step_id=1)
    assert calls==[]


def test_raw_diagnostic_fault_raises_before_any_grasp_or_pose_read():
    robot,calls,_,_=contact_robot();robot.diagnostic_grasp_contacts=True
    def fail(dt,*,step_id):
        calls.append(('pending_fault',dt,step_id))
        raise RuntimeError('captured full signed step')
    robot.grasp_observer.raise_pending_fault=fail
    with pytest.raises(RuntimeError,match='full signed step'):
        robot.contact_with_frames(.01,np.eye(4)[None],step_id=719)
    assert calls==[('pending_fault',.01,719)]


def test_direct_probe_cannot_bypass_raw_diagnostic_hold_only_restriction():
    from sim_physics.bimanual_probe import run
    fixture=S(diagnostic_grasp_contacts=True,bind=lambda *a:pytest.fail('No native binding'))
    with pytest.raises(ValueError,match='right-parked'):
        run(None,None,None,None,None,fixture,S(bimanual_hold_control=False),None)


@pytest.mark.parametrize('message',['normal observer failed',''])
def test_native_callback_fault_blocks_contact_before_frame_reads(monkeypatch,message):
    robot,calls,_,_=contact_robot()
    original=ValueError(message)
    def fail(*args):raise original
    robot.grasp_observer.add_contact=fail
    # Decode only synthetic collider IDs. The REAL callback catches the
    # observer exception, so contact_with_frames must check its latched fault.
    monkeypatch.setitem(sys.modules,'pxr',S(PhysicsSchemaTools=S(intToSdfPath=lambda x:x)))
    vector=lambda v:S(x=v[0],y=v[1],z=v[2])
    header=S(collider0='/R/ee_finger_l1/pad',collider1='/T/Branch/StemCollider',
        contact_data_offset=0,num_contact_data=1,friction_anchors_offset=0,num_friction_anchors_data=0)
    point=S(impulse=vector((.001,0.,0.)),position=vector((0.,0.,0.)),
        normal=vector((1.,0.,0.)),separation=0.)
    robot.event_monitor._callback([header],[point],[])
    assert robot.event_monitor.error is not None
    with pytest.raises(RuntimeError):robot.contact_with_frames(.01,np.eye(4)[None],step_id=1)
    assert calls==[]


def test_adapter_exception_propagates_without_legacy_contact_fallback():
    robot,calls,_,_=contact_robot();original=ValueError('stale adapter frames')
    def fail(*args,**kwargs):raise original
    robot.grasp_observer.evaluate=fail
    robot.contact=lambda *args:pytest.fail('Legacy selected-only contact fallback forbidden')
    with pytest.raises(ValueError,match='stale adapter') as caught:
        robot.contact_with_frames(.01,np.eye(4)[None],step_id=1)
    assert caught.value is original and calls==[('finger_frames',)]


def binding_robot(monkeypatch,*,factory_error=None,diagnostic=False):
    import sim_physics.shaft_grasp_native as module
    robot=object.__new__(BimanualRobot);events=[]
    robot.diagnostic_grasp_contacts=diagnostic
    robot.stage=object();robot.rig=S(source_target='seed/SubStem_41')
    robot.body_index=3;robot.paths=['/R/ee_left','/R/ee_finger_l1','/R/ee_finger_l2']
    robot.contact_views=[object(),object()]
    robot.names=[f'right_arm_{i}' for i in range(7)]
    robot.knife=S(wrist_path='/R/ee_right')
    old_monitor=monitor()
    def old_close():
        assert old_monitor.normal_contact_observer is None
        events.append('old_close')
    old=S(close=old_close);old_monitor.normal_contact_observer=old
    robot.event_monitor=old_monitor;robot.grasp_observer=old
    new_monitor=monitor()
    def parent_bind(self,view):
        assert self.grasp_observer is None and events==['old_close']
        events.append('parent_bind_and_subscribe')
        self.event_monitor=new_monitor
    monkeypatch.setattr(FullRobotGripper,'bind',parent_bind)
    new=S(close=lambda:events.append('new_close'))
    def factory(stage,rig,**kwargs):
        assert stage is robot.stage and rig is robot.rig
        assert robot.event_monitor is new_monitor and new_monitor.native_full_contact_reporting
        assert new_monitor.normal_contact_observer is None
        evidence=(dict(diagnostic_noncompressive_report=True) if diagnostic else
            dict(allow_signed_native_normals=True,sensor_contract=module.NATIVE37_SENSOR_CONTRACT))
        assert kwargs==dict(selected_index=3,finger_paths=robot.paths[1:],contact_views=robot.contact_views,**evidence)
        events.append('new_adapter')
        if factory_error is not None:raise factory_error
        return new
    monkeypatch.setattr(module,'ShaftGraspNative',factory)
    def create(path):
        assert path=='/R/ee_right';events.append('right_view');return S(count=1)
    view=S(create_rigid_body_view=create)
    return robot,view,events,new,new_monitor


def test_bind_releases_old_observer_before_parent_subscription_and_attaches_new(monkeypatch):
    robot,view,events,new,new_monitor=binding_robot(monkeypatch)
    robot.bind(view)
    assert events==['old_close','parent_bind_and_subscribe','right_view','new_adapter']
    assert robot.grasp_observer is new and new_monitor.normal_contact_observer is new
    assert new_monitor.tool_contact.__self__ is robot
    assert robot.right_indices==list(range(7))
    robot.release_grasp_observer();robot.release_grasp_observer()
    assert robot.grasp_observer is None and new_monitor.normal_contact_observer is None
    assert events.count('new_close')==1


def test_failed_rebind_never_reuses_old_grasp_observer(monkeypatch):
    original=ValueError('adapter binding rejected')
    robot,view,events,_,new_monitor=binding_robot(monkeypatch,factory_error=original)
    with pytest.raises(ValueError,match='binding rejected') as caught:robot.bind(view)
    assert caught.value is original and events[0]=='old_close'
    assert robot.grasp_observer is None and new_monitor.normal_contact_observer is None


def test_strict_contact_diagnostic_does_not_enable_signed_mode(monkeypatch):
    robot,view,events,new,new_monitor=binding_robot(monkeypatch,diagnostic=True)
    robot.bind(view)
    assert robot.grasp_observer is new and new_monitor.normal_contact_observer is new


def test_release_only_detaches_owned_observer_and_is_idempotent():
    robot,calls,_,_=contact_robot();other=object()
    robot.event_monitor.normal_contact_observer=other
    robot.release_grasp_observer();robot.release_grasp_observer()
    assert calls==[('close',)] and robot.grasp_observer is None
    assert robot.event_monitor.normal_contact_observer is other
    # Release also supports constructor/offline failure before any binding.
    object.__new__(BimanualRobot).release_grasp_observer()


def test_reset_releases_observer_before_parent_reset(monkeypatch):
    robot,calls,_,_=contact_robot()
    robot.rig=S(source_target='seed/SubStem_41')
    robot.held_plant_screen=S(workspace=(0,1),static=['old'])
    robot.kin=S(forward=lambda *args:np.eye(4));robot.right=np.zeros(7);robot.base=np.eye(4)
    robot.planning_slides={'old':1.}
    def parent_reset(self):
        assert calls==[('close',)] and self.grasp_observer is None
        assert self.event_monitor.normal_contact_observer is None
        calls.append(('parent_reset',))
    monkeypatch.setattr(FullRobotGripper,'restore_authored_state',parent_reset)
    robot.restore_authored_state()
    assert calls==[('close',),('parent_reset',)] and robot.plan is None
    assert not hasattr(robot,'planning_slides')


@pytest.mark.parametrize('operation',['bind','reset'])
def test_close_fault_stops_before_parent_rebind_or_reset(monkeypatch,operation):
    robot,calls,_,_=contact_robot();original=RuntimeError('notice release failed')
    def fail():raise original
    robot.grasp_observer.close=fail
    def forbidden(*args):pytest.fail('Parent operation must not follow failed release')
    method='bind' if operation=='bind' else 'restore_authored_state'
    monkeypatch.setattr(FullRobotGripper,method,forbidden)
    with pytest.raises(RuntimeError,match='notice release failed') as caught:
        if operation=='bind':robot.bind(object())
        else:robot.restore_authored_state()
    assert caught.value is original and robot.event_monitor.normal_contact_observer is None
    assert robot.grasp_observer is not None and calls==[]


@pytest.mark.parametrize('adapter_valid',[True,False])
def test_probe_records_crosscheck_before_hard_stop_and_releases_before_stop(monkeypatch,adapter_valid):
    import greenhouse_sim.physics_clock as clock_module
    from sim_physics.bimanual_probe import run
    robot,calls,_,contact=contact_robot(bilateral=adapter_valid,adapter_valid=adapter_valid)
    frames=np.repeat(np.eye(4)[None],4,axis=0)
    runtime=S(frames=frames,articulation=S(get_dof_positions=lambda:np.zeros((1,2)),
        get_dof_velocities=lambda:np.zeros((1,2))))
    events=[]
    def sample():events.append('sample_after_fetch');return frames,np.zeros((4,6))
    runtime.sample=sample
    rig=S(cut=False,cut_index=1,rest_frames=frames.copy(),chain_world=np.zeros((5,3)),
        body_paths=[f'/T/Branch/Segment_{i:03d}' for i in range(len(frames))])
    robot.body_index=2;robot.start=np.eye(4);robot.goal=np.eye(4)
    robot.stop_requested=False;robot.cut_contacts=0
    robot.bind=lambda view:events.append('bind')
    robot.report=lambda:{}
    robot.target_palm=lambda position:robot.event_monitor.begin_step()
    robot.close=lambda fraction:None
    robot.command_right=lambda phase,fraction:events.append('park')
    robot.prepare_step=lambda current:None
    robot.palm=S(get_transforms=lambda:np.array([[0.,0.,0.,0.,0.,0.,1.]]))
    robot.all_contacts=S(get_net_contact_forces=lambda dt:np.zeros((2,3)))
    robot.seam=lambda current:(np.zeros(3),np.array([0.,0.,1.]))
    robot.check_plant_window=lambda current:events.append('plant_guard')
    robot.check=lambda dt,palm:dict(allowed_tool_contact_n=0.,minimum_tool_separation_m=0.)
    robot.inspect_cut=lambda *args:events.append('knife_guard') or {}
    robot.on_sample=lambda record:events.append('sample_complete')
    class Clock:
        def __init__(self,sim,*,physics_hz,render_hz):
            assert physics_hz==240 and render_hz==0
            self.stamp=S(step=0,simulation_time_s=0.)
        def tick(self,*,before,after,before_render):
            before(self.stamp,1/240)
            events.append('mock_step_and_fetch')
            self.stamp=S(step=1,simulation_time_s=1/240)
            after(self.stamp,1/240)
        def report(self):return {'mock_clock':True}
    monkeypatch.setattr(clock_module,'PhysicsClock',Clock)
    stored={}
    class Output:
        def __truediv__(self,name):
            def write(text,*,encoding):
                assert encoding=='utf-8';stored[name]=json.loads(text)
                events.append('trajectory_recorded')
            return S(write_text=write)
    def stop():
        assert robot.grasp_observer is None and robot.event_monitor.normal_contact_observer is None
        assert calls[-1]==('close',);events.append('stop')
    sim=S(physics_sim_view=object(),stop=stop)
    springs=S(k=np.ones(2),step=lambda *args,**kwargs:None)
    args=S(physics_hz=240,render_hz=0,seconds=1/240,gui=False,capture_milestones=False)
    result=run(S(is_running=lambda:True),sim,rig,runtime,springs,robot,args,Output())
    rows=stored['bimanual_trajectory.json']
    assert len(rows)==1,result['error']
    assert rows[0]['contact']==contact
    evaluate_calls=[row for row in calls if row[0]=='evaluate']
    assert len(evaluate_calls)==1 and evaluate_calls[0][2] is frames and evaluate_calls[0][-1]==1
    assert events.index('mock_step_and_fetch')<events.index('sample_after_fetch')
    assert events[-2:]==['trajectory_recorded','stop'] and calls[-1]==('close',)
    if adapter_valid:
        assert result['error'] is None and 'knife_guard' in events
    else:
        assert 'reconciliation failed' in result['error'] and not result['gates']['bounded']
        assert 'plant_guard' not in events and 'knife_guard' not in events
        assert 'sample_complete' not in events
    assert not result['physical_cut_verified'] and not result['training_eligible']
