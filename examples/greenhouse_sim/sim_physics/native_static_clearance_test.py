from types import ModuleType, SimpleNamespace as S
import sys
import numpy as np
import pytest
from .native_static_clearance import NativeStaticClearance


def records():
    return [('/World/Stem','box',None,np.full(3,-.01),np.full(3,.01))]


def test_lazy_coverage_keeps_all_bounds_and_brackets_refinement_with_positive_controls():
    calls=[]
    def query(path,centre,axes,half):
        calls.append(path)
        return len(calls) in (1,3)
    extra=[('/World/Far','box',None,np.full(3,10.),np.full(3,11.))]
    n=NativeStaticClearance(query,records()+extra,lazy_coverage=True,max_queries=3)
    assert n.calls==0 and len(n.coverage_boxes)==2 and not n.covered
    assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert n.calls==2 and n.used_paths=={'/World/Stem'}
    n.validate();n.close()
    r=n.report()
    assert calls==['/World/Stem']*3 and r['final_validation_passed']
    assert r['unchecked_static_colliders']==['/World/Far']
    assert r['final_coverage_checked']==['/World/Stem']
    assert r['coverage_mode']=='lazy_before_exact_actor_refinement'


@pytest.mark.parametrize('failure',['missing','throw','budget','epoch','late_final','final_missing'])
def test_lazy_controls_fail_closed_before_any_false_clearance(failure,monkeypatch):
    import sim_physics.native_static_clearance as module
    calls=[]
    def query(*args):
        calls.append(args)
        if failure=='throw':raise RuntimeError('native fault')
        if failure=='missing':return False
        if len(calls)==1:return True
        if failure=='late_final' and len(calls)==3:
            monkeypatch.setattr(module.time,'perf_counter',lambda:n.started+9)
        return len(calls)==3 and failure!='final_missing'
    n=NativeStaticClearance(query,records(),lazy_coverage=True,max_queries=1 if failure=='budget' else 10)
    if failure=='epoch':n.guard=lambda:(_ for _ in ()).throw(RuntimeError('physics advanced'))
    try:
        clear=n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    except RuntimeError:
        clear=False
    if failure in ('late_final','final_missing'):
        assert clear
        with pytest.raises(RuntimeError):n.validate()
    else:
        assert not clear and n.used_paths==set()
    n.close()
    assert not n.report()['final_validation_passed']


def test_lazy_unknown_or_previously_missing_actor_never_refines():
    calls=[]
    n=NativeStaticClearance(lambda *args:calls.append(args) or False,records(),lazy_coverage=True)
    for path in ('/World/Unknown','/World/Stem','/World/Stem'):
        assert not n.clear_box(path,np.zeros(3),np.eye(3),np.ones(3),.001)
    assert len(calls)==1 and n.coverage_failed==['/World/Stem']


def test_lazy_flag_is_explicit_boolean():
    with pytest.raises(ValueError):NativeStaticClearance(lambda *a:True,records(),lazy_coverage=1)


def test_requires_positive_native_coverage_before_empty_query_can_clear():
    missing=NativeStaticClearance(lambda *args:False,records())
    assert not missing.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert missing.coverage_failed==['/World/Stem']
    calls=[]
    def query(path,centre,axes,half):
        calls.append((path,centre.copy(),axes.copy(),half.copy()))
        return len(calls)==1
    native=NativeStaticClearance(query,records())
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    assert calls[-1][3]==pytest.approx(np.full(3,.003))
    assert not native.clear_box('/World/Uncovered',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert native.report()['coarse_rejections_cleared']==1
    native.close()
    assert not native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)


def test_native_hit_keeps_rejection_and_does_not_change_margin():
    native=NativeStaticClearance(lambda *args:True,records())
    assert not native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    with pytest.raises(ValueError):native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.0009)


@pytest.mark.parametrize('defect',['throw','nonbool','guard','budget','time','nan','shear'])
def test_any_query_failure_blocks(defect,monkeypatch):
    import sim_physics.native_static_clearance as module
    calls=[]
    def query(*args):
        calls.append(args)
        if len(calls)==1:return True
        if defect=='throw':raise RuntimeError('native failed')
        if defect=='nonbool':return None
        return False
    native=NativeStaticClearance(query,records(),max_queries=1 if defect=='budget' else 20)
    centre=np.zeros(3);axes=np.eye(3)
    if defect=='guard':native.guard=lambda:(_ for _ in ()).throw(RuntimeError('step changed'))
    if defect=='time':monkeypatch.setattr(module.time,'perf_counter',lambda:native.started+9)
    if defect=='nan':centre[0]=np.nan
    if defect=='shear':axes[0,1]=.1
    assert not native.clear_box('/World/Stem',centre,axes,np.ones(3),.001)
    assert native.errors


def test_reflection_keeps_exact_symmetric_box():
    calls=[]
    def query(path,centre,axes,half):
        calls.append(axes.copy());return len(calls)==1
    native=NativeStaticClearance(query,records())
    assert native.clear_box('/World/Stem',np.zeros(3),np.diag([1,1,-1]),np.ones(3),.001)
    assert np.linalg.det(calls[-1])==pytest.approx(1.)


def test_integration_refines_only_static_box_against_tool_box():
    from .held_plant_screen_test import fixture
    screen,rig,world=fixture('box',blade=True)
    old,body,link,kind,data=screen.shapes[0]
    screen.shapes[0]=(body+'/attachments/DeleafKnife/BladePlateContact',body,link,kind,data)
    # Move local dynamic leaves/shaft away, insert one conservative static box.
    frames=rig.rest_frames.copy();frames[1,1,3]=.5
    static=('/World/Static','box',(np.array([.1,0,0]),np.eye(3),np.full(3,.03)),
            np.array([.07,-.03,-.03]),np.array([.13,.03,.03]))
    screen.static=[static];screen.snapshot(frames)
    assert not screen.check(world)
    requested=[]
    screen.native_static_query=S(clear_box_checked=lambda *args:requested.append(args) or True)
    assert screen.check(world)
    assert requested[0][0]=='/World/Static' and requested[0][-1]==.001
    # A dynamic target leaf remains an obstruction, regardless of query result.
    screen.snapshot(rig.rest_frames)
    assert not screen.check(world)
    assert screen.last_failure['plant_collider'].endswith('/Leaf')


@pytest.mark.parametrize('path,link', [
    ('/World/R/link_right_arm_5/Shape','link_right_arm_5'),
    ('/World/R/ee_right/Blade','ee_right'),
    ('/World/R/ee_right/attachments/Unfitted/Shape','ee_right'),
    ('/World/R/ee_right/attachments/DeleafKnifeBackup/Shape','ee_right'),
    ('/World/R/ee_right/attachments/RightWristCamera','ee_right'),
])
def test_refinement_never_expands_to_arm_or_unfitted_boxes(path,link):
    from .held_plant_screen_test import fixture
    screen,rig,world=fixture('box',blade=True)
    _,_,_,kind,data=screen.shapes[0]
    screen.shapes[0]=(path,'/World/R/'+link,link,kind,data)
    frames=rig.rest_frames.copy();frames[1,1,3]=.5
    screen.static=[('/World/Static','box',(np.array([.1,0,0]),np.eye(3),np.full(3,.03)),
                    np.array([.07,-.03,-.03]),np.array([.13,.03,.03]))]
    screen.snapshot(frames)
    def forbidden(*args):raise AssertionError('Non-tool box reached native refinement')
    screen.native_static_query=S(clear_box_checked=forbidden)
    assert not screen.check({link:np.eye(4)})


def test_checked_query_distinguishes_timeout_from_real_overlap(monkeypatch):
    import sim_physics.native_static_clearance as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    native=NativeStaticClearance(lambda *args:True,records())
    assert not native.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    clock[0]=9.
    calls=native.calls
    with pytest.raises(RuntimeError,match='unavailable; collision not determined'):
        native.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert native.calls==calls and not native.active


def test_checked_query_rejects_invalidation_during_native_call():
    calls=[]
    def query(*args):
        calls.append(args)
        if len(calls)>1:raise RuntimeError('native callback failed')
        return True
    native=NativeStaticClearance(query,records())
    with pytest.raises(RuntimeError,match='collision not determined'):
        native.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert not native.validation_passed


def test_post_call_deadline_discards_negative_result_and_prior_clearances(monkeypatch):
    import sim_physics.native_static_clearance as module
    clock=[0.];calls=[]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    def query(*args):
        calls.append(args)
        if len(calls)==1:return True
        if len(calls)==3:clock[0]=9.
        return False
    native=NativeStaticClearance(query,records())
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert not native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    with pytest.raises(RuntimeError,match='final validation'):native.validate()
    assert not native.report()['final_validation_passed']
    assert any('budget' in error for error in native.errors)


@pytest.mark.parametrize('budget',[0,20001,True])
def test_query_budget_cannot_exceed_twenty_thousand(budget):
    with pytest.raises(ValueError):
        NativeStaticClearance(lambda *args:True,records(),max_queries=budget)


def test_final_validation_repeats_only_used_coverage_and_copies_bounds():
    source=records()+[('/World/Unused','box',None,np.full(3,-.02),np.full(3,.02))]
    calls=[]
    def query(path,centre,axes,half):
        calls.append((path,half.copy()))
        return bool(half[0]>.01)
    native=NativeStaticClearance(query,source)
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    source[0][3][:]=100  # Caller mutation cannot move the positive control.
    native.validate()
    assert len(calls)==4 and calls[-1][0]=='/World/Stem'
    assert calls[-1][1]==pytest.approx(np.full(3,.011))
    assert native.final_coverage==['/World/Stem']
    assert native.validation_passed
    native.close()
    with pytest.raises(RuntimeError):native.validate()


def test_missing_actor_at_final_coverage_invalidates_prior_clearance():
    calls=[]
    def query(*args):
        calls.append(args);return len(calls)==1
    native=NativeStaticClearance(query,records())
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    with pytest.raises(RuntimeError,match='coverage missing'):native.validate()
    assert not native.active and not native.validation_passed


@pytest.fixture
def scene_runtime(monkeypatch):
    """Real CPU-only USD notices, mocked PhysX subscriptions and overlaps."""
    from pxr import Tf,Usd,UsdGeom,UsdPhysics
    import sim_physics.native_static_clearance as module
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
    prim=UsdGeom.Cube.Define(stage,'/World/Stem').GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    state=S(stage=stage,time=3.5,playing=True,stopped=False,releases=[],
            subscribed=[],object_callbacks={},step_callback=None,overlaps=[],
            present={'/World/Stem'},reported_delta=0,query_hook=None,subscribe_fault=None,
            cleanup_fault=None,cleanup_hook=None,notice_active=0,owned=[])
    real_register=Tf.Notice.Register
    def register(*args):
        key=real_register(*args);state.notice_active+=1
        def revoke():
            state.releases.append('usd');key.Revoke();state.notice_active-=1
            if state.cleanup_fault=='usd':raise RuntimeError('usd cleanup fault')
        return S(Revoke=revoke)
    monkeypatch.setattr(Tf.Notice,'Register',register)
    def subscribe_objects(**kwargs):
        state.subscribed.append(('objects',kwargs))
        if state.subscribe_fault=='objects':raise RuntimeError('objects subscription fault')
        state.object_callbacks=kwargs
        return 41
    def unsubscribe_objects(token):
        assert token==41;state.releases.append('objects')
        state.object_callbacks={}
        if state.cleanup_fault=='objects':raise RuntimeError('objects cleanup fault')
    def subscribe_step(**kwargs):
        state.subscribed.append(('step',kwargs))
        if state.subscribe_fault=='step':raise RuntimeError('step subscription fault')
        if state.subscribe_fault=='step_invalid':return S()
        state.step_callback=kwargs['fn']
        def unsubscribe():
            state.releases.append('step')
            if state.cleanup_hook is not None:state.cleanup_hook()
            state.step_callback=None
            if state.cleanup_fault=='step':raise RuntimeError('step cleanup fault')
        return S(unsubscribe=unsubscribe)
    def overlap(half,centre,quat,callback,anyhit):
        assert anyhit is False
        state.overlaps.append((half,centre,quat))
        if state.query_hook is not None:state.query_hook()
        paths=sorted(state.present) if half[0]>.01 else []
        for path in paths:assert callback(S(collision=path)) is True
        return len(paths)+state.reported_delta
    def mod(name,**attrs):
        value=ModuleType(name);value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,value)
        if '.' in name:
            parent,child=name.rsplit('.',1)
            monkeypatch.setattr(sys.modules[parent],child,value,raising=False)
    mod('omni')
    mod('omni.usd',get_context=lambda:S(get_stage=lambda:state.stage))
    mod('omni.timeline',get_timeline_interface=lambda:S(
        get_current_time=lambda:state.time,is_playing=lambda:state.playing,is_stopped=lambda:state.stopped))
    physx=S(subscribe_object_changed_notifications=subscribe_objects,
            unsubscribe_object_change_notifications=unsubscribe_objects,
            subscribe_physics_on_step_events=subscribe_step)
    mod('omni.physx',get_physx_interface=lambda:physx,
        get_physx_scene_query_interface=lambda:S(overlap_box=overlap))
    def create(source=None):
        native=module.current_scene_query(stage,records() if source is None else source)
        state.owned.append(native)
        return native
    state.create=create
    try:yield state
    finally:
        for native in state.owned:
            try:native.close()
            except RuntimeError:pass


def test_owned_subscription_signatures_cleanup_and_no_extra_steps(scene_runtime):
    state=scene_runtime;native=state.create()
    assert state.subscribed[0][1]['stop_callback_when_sim_stopped'] is False
    assert state.subscribed[1][1]['pre_step'] is True
    assert state.subscribed[1][1]['order']==0
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    native.validate()
    assert native.report()['epoch']['revision']==0 and state.time==3.5
    native.close();native.close()
    assert state.releases==['step','objects','usd'] and state.notice_active==0
    assert native.report()['closed'] and native.report()['final_validation_passed']
    assert not native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)


@pytest.mark.parametrize('event',[
    'usd_attribute','usd_resync','create','destroy','destroy_all','step','time','timeline','stage',
])
def test_epoch_event_invalidates_all_prior_clearances_without_unsubscribing_in_callback(scene_runtime,event):
    from pxr import Usd
    state=scene_runtime;native=state.create()
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    if event=='usd_attribute':state.stage.GetPrimAtPath('/World/Stem').GetAttribute('size').Set(3.)
    elif event=='usd_resync':state.stage.DefinePrim('/World/NewPrim')
    elif event=='create':state.object_callbacks['object_creation_fn'](1,2,3)
    elif event=='destroy':state.object_callbacks['object_destruction_fn'](1,2,3)
    elif event=='destroy_all':state.object_callbacks['all_objects_destruction_fn']()
    elif event=='step':state.step_callback(1/240)  # Timeline time deliberately unchanged.
    elif event=='time':state.time+=1/240
    elif event=='timeline':state.playing=False;state.stopped=True
    elif event=='stage':state.stage=Usd.Stage.CreateInMemory()
    assert state.releases==[]
    with pytest.raises(RuntimeError,match='epoch invalidated'):native.validate()
    assert not native.validation_passed
    native.close()
    assert state.releases==['step','objects','usd'] and state.notice_active==0


def test_event_inside_native_call_discards_its_negative_result(scene_runtime):
    state=scene_runtime;native=state.create()
    state.query_hook=lambda:state.step_callback(1/240)
    assert not native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    with pytest.raises(RuntimeError):native.validate()
    assert native.report()['epoch']['invalidation_reasons']==['physics_pre_step']


def test_lost_actor_without_notification_is_detected_by_final_positive_control(scene_runtime):
    state=scene_runtime;native=state.create()
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    state.present.clear()  # Simulate missing actor without a delivered object notification.
    with pytest.raises(RuntimeError,match='coverage missing'):native.validate()
    assert not native.validation_passed


def test_callback_count_mismatch_invalidates_transaction(scene_runtime):
    state=scene_runtime;native=state.create();state.reported_delta=1
    assert not native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    with pytest.raises(RuntimeError,match='callback coverage'):native.validate()


@pytest.mark.parametrize('fault,expected',[
    ('objects',['usd']),('step',['objects','usd']),('step_invalid',['objects','usd']),
])
def test_partial_subscription_failure_releases_every_acquired_handle(scene_runtime,fault,expected):
    state=scene_runtime;state.subscribe_fault=fault
    with pytest.raises(RuntimeError):state.create()
    assert state.releases==expected and state.notice_active==0


@pytest.mark.parametrize('fault',['step','objects','usd'])
def test_cleanup_error_still_attempts_every_owned_subscription(scene_runtime,fault):
    state=scene_runtime;native=state.create();state.cleanup_fault=fault
    with pytest.raises(RuntimeError,match='cleanup failed'):native.close()
    assert state.releases==['step','objects','usd'] and state.notice_active==0
    assert native.closed
    assert not native.report()['epoch']['subscriptions_closed']
    native.close()
    assert len(state.releases)==3


def test_event_during_subscription_release_invalidates_final_validation(scene_runtime):
    state=scene_runtime;native=state.create()
    assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    native.validate()
    state.cleanup_hook=lambda:state.step_callback(1/240)
    with pytest.raises(RuntimeError,match='epoch invalidated'):native.close()
    assert state.releases==['step','objects','usd'] and state.notice_active==0
    assert not native.validation_passed


def test_native_interface_acquisition_failure_releases_owned_handles(scene_runtime,monkeypatch):
    state=scene_runtime
    def fail():raise RuntimeError('query interface unavailable')
    monkeypatch.setattr(sys.modules['omni.physx'],'get_physx_scene_query_interface',fail)
    with pytest.raises(RuntimeError,match='interface unavailable'):state.create()
    assert state.releases==['step','objects','usd'] and state.notice_active==0


def test_preflight_event_and_bad_records_cleanup_subscriptions(scene_runtime):
    state=scene_runtime
    state.query_hook=lambda:state.step_callback(1/240)
    with pytest.raises(RuntimeError,match='epoch invalidated'):state.create()
    assert state.releases==['step','objects','usd'] and state.notice_active==0


def test_bad_record_after_subscription_setup_cleans_up(scene_runtime):
    state=scene_runtime
    bad=[('/World/Stem','box',None,[1,0,0],[0,0,0])]
    with pytest.raises(ValueError,match='bounds'):state.create(bad)
    assert state.releases==['step','objects','usd'] and state.notice_active==0


@pytest.mark.parametrize('kinematic',[False,True])
def test_enabled_body_ancestors_never_enter_native_refinement(scene_runtime,kinematic):
    from pxr import UsdGeom,UsdPhysics
    state=scene_runtime
    parent=UsdGeom.Xform.Define(state.stage,'/World/Actor').GetPrim()
    api=UsdPhysics.RigidBodyAPI.Apply(parent)
    api.CreateRigidBodyEnabledAttr(True);api.CreateKinematicEnabledAttr(kinematic)
    child=UsdGeom.Cube.Define(state.stage,'/World/Actor/Collider').GetPrim()
    UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
    source=[('/World/Actor/Collider','box',None,np.full(3,-.01),np.full(3,.01))]
    native=state.create(source)
    assert not native.covered and not state.overlaps
    assert not native.clear_box('/World/Actor/Collider',np.zeros(3),np.eye(3),np.ones(3),.001)


def test_non_metre_stage_rejected_before_native_subscriptions(scene_runtime):
    from pxr import UsdGeom
    state=scene_runtime;UsdGeom.SetStageMetersPerUnit(state.stage,.01)
    with pytest.raises(ValueError,match='metre-unit'):state.create()
    assert not state.subscribed and state.notice_active==0
