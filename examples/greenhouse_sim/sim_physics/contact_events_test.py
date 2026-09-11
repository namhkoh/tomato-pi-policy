import pytest
import sys
from types import SimpleNamespace as NS
from sim_physics.contact_events import ContactEvents


def monitor():
    return ContactEvents(robot_root='/World/R',target_root='/World/Target',
        fingers=['/World/R/finger'],floor_root='/World/Floor')


def test_grasp_observer_preserves_header_order_and_never_receives_friction():
    m=monitor();rows=[];steps=[]
    m.normal_contact_observer=NS(begin_step=lambda:steps.append(True),add_contact=lambda *row:rows.append(row))
    m.begin_step()
    m.consume('/World/Target/Stem','/World/R/finger/shape',[(.001,0,0)],
        [(1,2,3)],[(1,0,0)],[-.0001],friction_impulses=[(0,.002,0)])
    assert rows==[('/World/Target/Stem','/World/R/finger/shape',(1,2,3),(1,0,0),(.001,0,0),-.0001)]
    m.consume('/World/Target/Stem','/World/R/finger/shape',[],friction_impulses=[(0,.001,0)])
    assert len(rows)==1 and steps==[True]
    assert m.measurements(.01)['allowed_target_contact_n']==pytest.approx(.4)


def test_grasp_observer_fault_is_latched_and_cannot_make_loads_look_safe():
    m=monitor()
    def fail(*args): raise ValueError('stale normal evidence')
    m.normal_contact_observer=NS(begin_step=lambda:None,add_contact=fail)
    with pytest.raises(ValueError,match='stale'):
        m.consume('/World/Target/Stem','/World/R/finger/shape',[(.001,0,0)],[(0,0,0)],[(1,0,0)],[0.])
    m.begin_step()
    with pytest.raises(RuntimeError,match='stale'): m.measurements(.01)


def test_grasp_observer_requires_complete_normal_rows_and_releases_on_close():
    m=monitor();m.normal_contact_observer=NS(begin_step=lambda:None,add_contact=lambda *a:None)
    with pytest.raises(ValueError,match='full native'):
        m.consume('/World/Target/Stem','/World/R/finger/shape',[(.001,0,0)])
    m.close();assert m.normal_contact_observer is None


def test_grasp_step_failure_latches_in_main_safety_monitor():
    m=monitor()
    def fail(): raise ValueError('wrong step')
    m.normal_contact_observer=NS(begin_step=fail)
    with pytest.raises(ValueError,match='wrong step'): m.begin_step()
    with pytest.raises(RuntimeError,match='wrong step'): m.measurements(.01)


def test_empty_exception_message_is_still_a_latched_fault():
    m=monitor();m.error='';m.begin_step()
    with pytest.raises(RuntimeError): m.measurements(.01)


def test_contact_accounting_does_not_cancel_opposing_forces():
    m=monitor();m.consume('/World/R/arm/shape','/World/Gutter',[(.001,0,0),(-.001,0,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)
    m.begin_step();assert m.measurements(.01)['unwanted_contact_n']==0


def test_only_target_finger_and_floor_support_are_allowed():
    m=monitor()
    m.consume('/World/Target/Stem','/World/R/finger/shape',[(.001,0,0)])
    m.consume('/World/Floor/mesh','/World/R/wheel_l/shape',[(0,0,1)])
    assert m.measurements(.01)['unwanted_contact_n']==0
    assert m.measurements(.01)['allowed_target_contact_n']==pytest.approx(.1)
    m.consume('/World/Neighbor/Leaf','/World/R/finger/shape',[(.002,0,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)


def test_self_contact_and_malformed_data_fail_closed():
    m=monitor();m.consume('/World/R/left/shape','/World/R/right/shape',[(.01,0,0)])
    assert m.measurements(.01)['self_contact_n']==1.
    with pytest.raises(ValueError): m.consume('/World/R/a','/World/Gutter',[(float('nan'),0,0)])
    with pytest.raises(ValueError): m.measurements(0)
    m.error='bad native buffer'
    m.begin_step()  # A callback fault stays latched until explicit rebind.
    with pytest.raises(RuntimeError): m.measurements(.01)


def test_expected_tool_contact_is_point_specific_and_not_a_collision_disable():
    m=monitor()
    m.tool_contact=lambda robot,other,point,impulse,normal,separation: robot=='/World/R/knife/plate' and other=='/World/Target/Stem' and point[0]<.001
    m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0),(.002,0,0)],[(0,0,0),(.01,0,0)],[(1,0,0)]*2,[-.0001,0])
    assert m.measurements(.01)['allowed_tool_contact_n']==pytest.approx(.1)
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)
    assert m.measurements(.01)['minimum_tool_separation_m']==-.0001
    m.consume('/World/R/knife/arc','/World/Target/Stem',[(.003,0,0)],[(0,0,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.5)
    m.consume('/World/R/knife/plate','/World/Neighbor/Stem',[(.001,0,0)],[(0,0,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.6)


def test_missing_normal_is_not_edge_evidence_and_broad_face_remains_unwanted():
    from sim_physics.knife import leading_face_normal
    m=monitor();m.tool_contact=lambda r,o,p,i,n,s: leading_face_normal(n,[1,0,0])
    m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0)],[(0,0,0)])
    m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0)],[(0,0,0)],[(0,0,1)],[0.])
    assert m.measurements(.01)['allowed_tool_contact_n']==0.
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)
    with pytest.raises(ValueError):
        m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0)],[(0,0,0)],[],[0.])
    with pytest.raises(ValueError):
        m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0)],[(0,0,0)],[(float('nan'),0,0)],[0.])


def tool_normal(m,*,accepted=True,impulse=.001,reverse=False):
    pair=('/World/R/knife/plate','/World/Target/Stem')
    if reverse: pair=pair[::-1]
    m.consume(*pair,[(impulse,0,0)],[(0 if accepted else 1,0,0)],[(1,0,0)],[-.0001])


def test_normal_and_friction_opposition_never_cancels_within_or_across_pairs():
    m=monitor()
    m.consume('/World/R/arm/shape','/World/Gutter',[(.001,0,0),(-.001,0,0)],
        friction_impulses=[(0,.002,0),(0,-.002,0)])
    m.consume('/World/Neighbor','/World/R/arm/shape',[],friction_impulses=[(0,-.003,0)])
    result=m.measurements(.01)
    assert result['normal_contact_n']==pytest.approx(.2)
    assert result['friction_contact_n']==pytest.approx(.7)
    assert result['unwanted_contact_n']==pytest.approx(.9)
    assert result['total_contact_upper_bound_n']==pytest.approx(.9)
    assert m.pairs[tuple(sorted(('/World/R/arm/shape','/World/Gutter')))]==pytest.approx(.006)
    assert result['native_full_contact_reporting'] is False  # Pure inputs are not native coverage.


@pytest.mark.parametrize('robot,other,bucket',[
    ('finger/shape','/World/Target/Stem','allowed_target'),
    ('finger/shape','/World/Neighbor/Leaf','unwanted'),
    ('wheel_l/shape','/World/Floor/mesh','allowed_floor'),
    ('base/shape','/World/Floor','allowed_floor'),
    ('arm/shape','/World/Floor','unwanted'),
    ('arm/shape','/World/R/other/shape','self'),
])
def test_friction_uses_finger_floor_neighbor_and_self_pair_classification(robot,other,bucket):
    m=monitor()
    # Also covers friction-only events, reversed collider order and opposing anchors.
    m.consume(other,'/World/R/'+robot,[],friction_impulses=[(0,.001,0),(0,-.002,0)])
    result=m.measurements(.01)
    assert result[bucket+'_contact_n']==pytest.approx(.3)
    assert result[bucket+'_friction_n']==pytest.approx(.3)
    assert sum(result[k+'_contact_n'] for k in m._buckets)==pytest.approx(.3)


def test_tool_friction_counts_toward_existing_half_newton_guard_not_edge_evidence():
    m=monitor();seen=[]
    def accept(*args):
        seen.append(args)
        return True
    m.tool_contact=accept
    m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0)],[(0,0,0)],[(1,0,0)],[-.0001],
        friction_impulses=[(0,.005,0)],friction_points=[(.2,0,0)])
    result=m.measurements(.01)
    assert result['allowed_tool_contact_n']==pytest.approx(.6)
    assert result['allowed_tool_contact_n']>.5
    assert result['allowed_tool_friction_n']==pytest.approx(.5)
    assert m.allowed_tool_impulse==pytest.approx(.006)
    assert len(seen)==1 and seen[0][3]==(.001,0,0)
    assert result['minimum_tool_separation_m']==-.0001


@pytest.mark.parametrize('friction_first',[True,False])
def test_tool_friction_can_only_borrow_all_normal_acceptance_from_same_pair_and_step(friction_first):
    m=monitor();m.tool_contact=lambda r,o,p,i,n,s: p[0]==0
    def friction():
        m.consume('/World/Target/Stem','/World/R/knife/plate',[],friction_impulses=[(0,.002,0)])
    if friction_first: friction()
    tool_normal(m)
    if not friction_first: friction()
    assert m.measurements(.01)['allowed_tool_contact_n']==pytest.approx(.3)
    m.consume('/World/R/knife/plate','/World/Target/OtherStem',[],friction_impulses=[(0,.004,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.4)
    m.begin_step();friction()
    assert m.measurements(.01)['allowed_tool_contact_n']==0
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)


@pytest.mark.parametrize('rejected_impulse',[0.,.001])
def test_late_rejected_normal_revokes_all_pair_friction_even_if_zero_impulse(rejected_impulse):
    m=monitor();m.tool_contact=lambda r,o,p,i,n,s: p[0]==0
    tool_normal(m)
    m.consume('/World/R/knife/plate','/World/Target/Stem',[],friction_impulses=[(0,.003,0)])
    assert m.measurements(.01)['allowed_tool_friction_n']==pytest.approx(.3)
    tool_normal(m,accepted=False,impulse=rejected_impulse,reverse=True)
    assert m.measurements(.01)['allowed_tool_contact_n']==pytest.approx(.1)
    assert m.measurements(.01)['allowed_tool_friction_n']==0
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.3+rejected_impulse/.01)
    tool_normal(m)  # A later accepted point cannot undo a rejection this step.
    assert m.measurements(.01)['allowed_tool_friction_n']==0


def test_missing_normal_evidence_poisoning_and_other_tool_pair_cannot_be_rescued():
    m=monitor();m.tool_contact=lambda *args: True
    tool_normal(m)
    m.consume('/World/R/knife/plate','/World/Target/Stem',[(0,0,0)],
        friction_impulses=[(0,.002,0)])
    m.consume('/World/R/knife/arc','/World/Target/Stem',[],friction_impulses=[(0,.003,0)])
    assert m.measurements(.01)['allowed_tool_friction_n']==0
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.5)


@pytest.mark.parametrize('changes',[
    dict(impulses=[(float('nan'),0,0)]),
    dict(points=[(float('inf'),0,0)]),
    dict(normals=[(0,float('nan'),0)]),
    dict(separations=[float('inf')]),
    dict(friction_impulses=[(0,float('nan'),0)]),
    dict(friction_points=[(0,0,float('inf'))]),
    dict(friction_points=[]),dict(friction_impulses=[(0,1)]),
    dict(normals=[]),dict(separations=[]),
])
def test_all_geometry_and_friction_fields_are_validated_even_for_allowed_fingers(changes):
    m=monitor()
    payload=dict(impulses=[(.001,0,0)],points=[(0,0,0)],normals=[(1,0,0)],separations=[0.],
        friction_impulses=[(0,.002,0)],friction_points=[(0,0,0)])
    payload.update(changes)
    with pytest.raises(ValueError): m.consume('/World/R/finger/shape','/World/Target/Stem',**payload)
    assert m.total_events==0


def test_overflow_cannot_produce_a_finite_or_silent_contact_bound():
    m=monitor()
    with pytest.raises(ValueError,match='Overflowing'):
        m.consume('/World/R/arm/shape','/World/Neighbor',[],friction_impulses=[(1e308,0,0),(1e308,0,0)])
    m.consume('/World/R/arm/shape','/World/Neighbor',[(1.,0,0)])
    with pytest.raises(ValueError,match='Overflowing'): m.measurements(1e-320)


@pytest.fixture
def native_stub(monkeypatch):
    """Only Python stubs: never import or launch SimulationApp."""
    state=NS(callback=None,legacy_calls=0)
    scene_schema=object()
    state.scene=lambda mode: NS(IsA=lambda schema:schema is scene_schema,
        GetFrictionTypeAttr=lambda:NS(Get=lambda:mode))
    state.scenes=[state.scene('patch')]
    state.stage=NS(Traverse=lambda:iter(state.scenes))
    def subscribe(callback):
        state.callback=callback
        return object()
    def legacy(callback):
        state.legacy_calls+=1
        raise AssertionError('Normal-only subscription forbidden')
    state.interface=NS(subscribe_full_contact_report_events=subscribe,subscribe_contact_report_events=legacy)
    monkeypatch.setitem(sys.modules,'omni',NS())
    monkeypatch.setitem(sys.modules,'omni.physx',NS(get_physx_simulation_interface=lambda:state.interface))
    monkeypatch.setitem(sys.modules,'omni.usd',NS(get_context=lambda:NS(get_stage=lambda:state.stage)))
    monkeypatch.setitem(sys.modules,'pxr',NS(PhysicsSchemaTools=NS(intToSdfPath=lambda value:value),
        UsdPhysics=NS(Scene=scene_schema),PhysxSchema=NS(PhysxSceneAPI=lambda scene:scene)))
    return state


def native_vector(values): return NS(x=values[0],y=values[1],z=values[2])


def native_point(impulse=(.001,0,0),point=(0,0,0),normal=(1,0,0),separation=-.0001):
    return NS(impulse=native_vector(impulse),position=native_vector(point),
        normal=native_vector(normal),separation=separation)


def native_header(**changes):
    values=dict(collider0='/World/R/knife/plate',collider1='/World/Target/Stem',
        contact_data_offset=0,num_contact_data=1,friction_anchors_offset=0,num_friction_anchors_data=1)
    values.update(changes)
    return NS(**values)


def test_full_native_callback_uses_exact_spans_and_never_forwards_friction_to_tool(native_stub):
    m=monitor();seen=[];m.tool_contact=lambda *args:seen.append(args) or True
    m.subscribe()
    header=native_header(contact_data_offset=1,friction_anchors_offset=1)
    native_stub.callback([header],[native_point((99,0,0)),native_point()],
        [native_point((99,0,0)),native_point((0,.002,0))])
    result=m.measurements(.01)
    assert result['native_full_contact_reporting'] is True
    assert result['native_patch_friction'] is True
    assert result['allowed_tool_contact_n']==pytest.approx(.3)
    assert len(seen)==1 and seen[0][3]==(.001,0,0)
    assert native_stub.legacy_calls==0
    m.close();assert m.measurements(.01)['native_full_contact_reporting'] is False


def test_native_friction_only_header_then_normal_then_rejection_across_callbacks(native_stub):
    m=monitor();m.tool_contact=lambda r,o,p,i,n,s: p[0]==0;m.subscribe()
    native_stub.callback([native_header(num_contact_data=0)],[],[native_point((0,.002,0))])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)
    native_stub.callback([native_header(num_friction_anchors_data=0)],[native_point()],[])
    assert m.measurements(.01)['allowed_tool_contact_n']==pytest.approx(.3)
    native_stub.callback([native_header(num_friction_anchors_data=0)],[native_point(impulse=(0,0,0),point=(1,0,0))],[])
    assert m.measurements(.01)['allowed_tool_contact_n']==pytest.approx(.1)
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)


@pytest.mark.parametrize('field,value',[
    ('contact_data_offset',-1),('num_contact_data',-1),('num_contact_data',2),
    ('contact_data_offset',.5),('num_contact_data',True),
    ('friction_anchors_offset',-1),('num_friction_anchors_data',-1),
    ('friction_anchors_offset',2),('num_friction_anchors_data',2),
    ('friction_anchors_offset',.5),('num_friction_anchors_data',True),
])
def test_native_buffer_errors_fail_closed_and_stay_latched(native_stub,field,value):
    m=monitor();m.subscribe()
    native_stub.callback([native_header(**{field:value})],[native_point()],[native_point()])
    with pytest.raises(RuntimeError,match='buffer range'): m.measurements(.01)
    m.begin_step()
    with pytest.raises(RuntimeError,match='buffer range'): m.measurements(.01)


@pytest.mark.parametrize('field', ['impulse','position'])
def test_native_nonfinite_friction_is_not_hidden_by_allowed_floor_classification(native_stub,field):
    m=monitor();m.subscribe();anchor=native_point()
    setattr(anchor,field,native_vector((float('nan'),0,0)))
    native_stub.callback([native_header(collider0='/World/R/base/shape',collider1='/World/Floor',num_contact_data=0)],[],[anchor])
    with pytest.raises(RuntimeError,match='Nonfinite native friction'): m.measurements(.01)


@pytest.mark.parametrize('modes',[[],['oneDirectional'],[None],['patch','twoDirectional']])
def test_full_subscription_requires_known_patch_friction_in_every_scene(native_stub,modes):
    native_stub.scenes=[native_stub.scene(mode) for mode in modes]
    m=monitor()
    with pytest.raises(RuntimeError): m.subscribe()
    assert not m.native_full_contact_reporting
    assert native_stub.callback is None and native_stub.legacy_calls==0
    with pytest.raises(RuntimeError): m.measurements(.01)


def test_missing_full_native_api_never_falls_back_to_normal_only(native_stub):
    del native_stub.interface.subscribe_full_contact_report_events
    m=monitor()
    with pytest.raises(AttributeError): m.subscribe()
    assert not m.native_full_contact_reporting and native_stub.legacy_calls==0
    with pytest.raises(RuntimeError): m.measurements(.01)


def test_measurements_remain_numeric_for_existing_full_robot_finite_guard(native_stub):
    import numpy as np
    m=monitor()
    assert np.isfinite([0.,0.,*m.measurements(.01).values()]).all()
    m.subscribe()
    assert np.isfinite([0.,0.,*m.measurements(.01).values()]).all()
    assert m.measurements(.01)['contact_magnitude_is_upper_bound'] is True
    assert m.measurements(.01)['friction_in_edge_evidence'] is False


def test_friction_cannot_lift_subthreshold_normal_force_into_shear_gate():
    import numpy as np
    from sim_physics.knife import ShearGate,KNIFE_IMPULSE_CONTRACT
    m=monitor();gate=ShearGate('petiole');normal_points=[];normal_impulses=[]
    def accept(robot,other,point,impulse,normal,separation):
        normal_points.append(point);normal_impulses.append(impulse)
        return True
    m.tool_contact=accept
    for i in range(20):
        m.begin_step();normal_points.clear();normal_impulses.clear()
        edge=np.eye(4);edge[0,3]=-i*.0001
        point=tuple(edge[:3,3])
        m.consume('/World/R/knife/plate','/World/Target/Stem',[(.001,0,0)],[point],[(1,0,0)],[0.],
            friction_impulses=[(0,.002,0)],friction_points=[point])
        assert m.measurements(.01)['allowed_tool_contact_n']==pytest.approx(.3)
        assert gate.observe(dt=.01,edge=edge,centre=np.zeros(3),axis=np.array([0.,0.,1.]),
            points=normal_points,impulses=normal_impulses,held=True,slip=.001,
            normals=[(1,0,0)],impulse_contract=KNIFE_IMPULSE_CONTRACT,edge_contact_verified=True,
            tool_contact_upper_bound_n=m.measurements(.01)['allowed_tool_contact_n']) is None
    assert not gate.completed and gate.dwell==0.


def test_missing_stage_or_subscription_handle_fails_closed(native_stub):
    m=monitor();native_stub.stage=None
    with pytest.raises(RuntimeError,match='stage'): m.subscribe()
    assert not m.native_full_contact_reporting
    native_stub.stage=NS(Traverse=lambda:iter(native_stub.scenes))
    native_stub.interface.subscribe_full_contact_report_events=lambda callback:None
    m=monitor()
    with pytest.raises(RuntimeError,match='subscription'): m.subscribe()
    assert not m.native_full_contact_reporting


@pytest.mark.parametrize('sentinel',[-1,2**32-1,2**64-1])
@pytest.mark.parametrize('empty_stream',['normal','friction','both'])
def test_zero_count_native_spans_ignore_sentinels_without_losing_other_stream(native_stub,sentinel,empty_stream):
    m=monitor();seen=[];m.tool_contact=lambda *args:seen.append(args) or True;m.subscribe()
    normal_count=0 if empty_stream in ('normal','both') else 1
    friction_count=0 if empty_stream in ('friction','both') else 1
    header=native_header(num_contact_data=normal_count,num_friction_anchors_data=friction_count,
        contact_data_offset=0 if normal_count else sentinel,
        friction_anchors_offset=0 if friction_count else sentinel)
    native_stub.callback([header],[native_point()] if normal_count else [],
        [native_point((0,.002,0))] if friction_count else [])
    result=m.measurements(.01)
    assert result['normal_contact_n']==pytest.approx(normal_count*.1)
    assert result['friction_contact_n']==pytest.approx(friction_count*.2)
    assert result['allowed_tool_contact_n']==pytest.approx(normal_count*.1)
    assert result['unwanted_contact_n']==pytest.approx(friction_count*.2)
    assert len(seen)==normal_count
    assert m.error is None and result['native_full_contact_reporting'] is True


def test_empty_span_never_interprets_offset_but_negative_and_malformed_counts_fail():
    from sim_physics.contact_events import _span
    class UnusedOffset:
        def __index__(self): raise AssertionError('Unused offset must not be interpreted')
    assert _span(UnusedOffset(),0,0,'friction')==slice(0,0)
    for count in (-1,False,.0,float('nan')):
        with pytest.raises(ValueError,match='buffer range'):
            _span(UnusedOffset(),count,0,'friction')
