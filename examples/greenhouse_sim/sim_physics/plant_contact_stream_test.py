"""Passive full-contact hook tests; SDK callbacks are Python stubs only."""
import json
import pickle
from types import SimpleNamespace as S

import numpy as np
import pytest

from .contact_events import ContactEvents
from .contact_events_test import native_stub,native_header,native_point
from .plant_contact_stream import PlantContactStream,MAX_ROWS,MAX_PLANT_COLLIDERS,MAX_PATH_CHARS


STEM='/World/Target/Segment/StemCollider'
LEAF='/World/Target/Segment/Leaf/Mesh'


def normal(first=STEM,second='/World/Gutter/Collider'):
    return dict(collider0=first,collider1=second,kind='normal',point_world_m=[1.,2.,3.],
        normal_on_0=[1.,0.,0.],separation_m=-.0001,impulse_on_0_ns=[-.001,0.,0.])


def friction(first=STEM,second='/World/Gutter/Collider'):
    return dict(collider0=first,collider1=second,kind='friction',point_world_m=[4.,5.,6.],
        impulse_on_0_ns=[0.,-.002,0.])


def monitor(observer=None):
    return ContactEvents(robot_root='/World/R',target_root='/World/Target',
        fingers=['/World/R/finger'],floor_root='/World/Floor',full_contact_observer=observer)


def consume(m,first=STEM,second='/World/Gutter/Collider',**changes):
    payload=dict(impulses=[[-.001,0.,0.]],points=[[1.,2.,3.]],normals=[[1.,0.,0.]],
        separations=[-.0001],friction_impulses=[[0.,-.002,0.]],friction_points=[[4.,5.,6.]])
    payload.update(changes)
    m.consume(first,second,**payload)


@pytest.mark.parametrize('other',['/World/Gutter/Collider','/World/OtherPlant/Stem',
    '/World/R/knife/plate',LEAF])
def test_exact_inventory_keeps_any_partner_and_each_original_row_once(other):
    c=PlantContactStream([STEM,LEAF]);m=monitor(c)
    consume(m,second=other)
    assert c.rows==[normal(second=other),friction(second=other)]
    if not other.startswith('/World/R/'):
        assert m.pairs=={} and m.total_events==0  # Not dropped by robot-only _kind.
    consume(m,first=STEM+'Lookalike',second='/World/Gutter/Collider')
    consume(m,first=STEM+'/Child',second='/World/Gutter/Collider')
    assert len(c.rows)==2


def test_reversed_headers_keep_signed_vectors_without_normalization_or_abs():
    c=PlantContactStream([STEM]);m=monitor(c)
    consume(m)
    consume(m,first='/World/Gutter/Collider',second=STEM,impulses=[[.001,0,0]],
        normals=[[-1.,0,0]],friction_impulses=[[0,.002,0]])
    a,b=c.rows[0],c.rows[2]
    assert (b['collider0'],b['collider1'])==(a['collider1'],a['collider0'])
    np.testing.assert_array_equal(b['normal_on_0'],-np.array(a['normal_on_0']))
    np.testing.assert_array_equal(b['impulse_on_0_ns'],-np.array(a['impulse_on_0_ns']))
    assert np.dot(a['impulse_on_0_ns'],a['normal_on_0'])<0
    assert c.rows[3]['impulse_on_0_ns']==[0,.002,0]


def test_friction_only_callback_ignores_zero_count_sentinel_offset(native_stub):
    c=PlantContactStream([STEM]);m=monitor(c);m.subscribe()
    assert m.full_contact_observer is c
    seen=[];m.normal_contact_observer=S(begin_step=lambda:None,add_contact=lambda *args:seen.append(args))
    h=native_header(collider0=STEM,collider1='/World/Gutter/Collider',
        num_contact_data=0,contact_data_offset=object())
    anchor=native_point(impulse=(0,-.002,0),point=(4,5,6))
    native_stub.callback([h],[],[anchor])
    assert c.rows==[friction()] and seen==[] and m.pairs=={}
    m.close();assert m.full_contact_observer is None


def test_default_accounting_and_existing_normal_observer_are_bitwise_unchanged():
    a=monitor();c=PlantContactStream([STEM,LEAF]);b=monitor(c)
    seen=[[],[]]
    for m,rows in zip((a,b),seen):
        m.normal_contact_observer=S(begin_step=lambda:None,add_contact=lambda *args,rows=rows:rows.append(args))
        m.tool_contact=lambda r,o,p,i,n,s:p[0]<2
        for first,second in [('/World/R/finger/shape',STEM),('/World/R/knife/plate',STEM),
            (STEM,'/World/R/arm/shape'),('/World/Floor','/World/R/wheel_l/shape'),
            ('/World/R/a','/World/R/b'),(STEM,LEAF)]:
            consume(m,first,second,impulses=[[.001,0,0],[-.001,0,0]],
                points=[[1,2,3],[2,3,4]],normals=[[1,0,0],[-1,0,0]],separations=[-.0001,0.])
        consume(m,'/World/R/knife/plate',STEM,impulses=[],points=[],normals=[],separations=[])
    assert seen[0]==seen[1]
    def state(m):return (m.measurements(.01),m.pairs,m.normal_pairs,m.friction_pairs,m._pair_states)
    assert pickle.dumps(state(a))==pickle.dumps(state(b))
    a.begin_step();b.begin_step()
    assert c.rows==[] and pickle.dumps(state(a))==pickle.dumps(state(b))


def test_copies_inputs_and_outputs_and_does_not_claim_completeness_or_assign_step():
    c=PlantContactStream(plant_colliders={STEM});r=normal();c.add_contact(r);r['point_world_m'][0]=999
    rows=c.rows;rows[0]['impulse_on_0_ns'][0]=999
    s=c.snapshot();s['rows'][0]['normal_on_0'][0]=999
    assert c.rows==[normal()]
    assert s['native_completeness_verified'] is False and s['step_and_pose_binding_verified'] is False
    assert s['actuation_authorized'] is False
    assert 'step' not in s and 'sample_id' not in s and 'generation' not in s
    json.dumps(c.snapshot(),allow_nan=False)
    c.begin_step();assert c.rows==[] and c.error is None


@pytest.mark.parametrize('change',[
    {'impulses':[[float('nan'),0,0]]},{'points':None},{'points':[]},
    {'normals':None},{'normals':[[1,float('inf'),0]]},{'separations':None},
    {'separations':[float('nan')]},{'friction_points':None},{'friction_points':[]},
    {'friction_impulses':[[0,float('inf'),0]]}])
def test_missing_or_bad_native_geometry_latches_monitor_and_collector(change):
    c=PlantContactStream([STEM]);m=monitor(c)
    with pytest.raises(ValueError):consume(m,**change)
    assert m.error is not None and c.error is not None
    first_error=c.error
    with pytest.raises(RuntimeError):m.begin_step()
    with pytest.raises(RuntimeError):c.begin_step()
    with pytest.raises(RuntimeError):consume(m)
    with pytest.raises(RuntimeError):m.measurements(.01)
    assert c.error==first_error


@pytest.mark.parametrize('change',[
    {'kind':'force'},{'point_world_m':None},{'point_world_m':[0,1]},
    {'normal_on_0':[1,0,float('nan')]},{'separation_m':True},
    {'impulse_on_0_ns':[True,0,0]},{'collider0':'/World/Target/*'},
    {'collider1':STEM},{'collider0':'/World/Target/Stem.attr'},
    {'step_id':42}])
def test_bad_collector_rows_are_permanent_faults(change):
    c=PlantContactStream([STEM]);r=normal();r.update(change)
    with pytest.raises(ValueError):c.add_contact(r)
    assert c.error is not None and c.rows==[]
    with pytest.raises(RuntimeError):c.add_contact(normal())
    with pytest.raises(RuntimeError):c.begin_step()


def test_friction_row_cannot_borrow_normal_or_separation():
    c=PlantContactStream([STEM]);r=friction();r['normal_on_0']=[1,0,0]
    with pytest.raises(ValueError,match='fields'):c.add_contact(r)


def test_limit_counts_normal_and_friction_rows_and_preserves_partial_fault_audit():
    c=PlantContactStream([STEM]);m=monitor(c)
    for _ in range(MAX_ROWS//2):consume(m)
    assert len(c.rows)==MAX_ROWS
    with pytest.raises(ValueError,match='256'):consume(m)
    assert len(c.rows)==MAX_ROWS and m.error is not None and c.error is not None
    with pytest.raises(RuntimeError):m.begin_step()
    assert len(c.rows)==MAX_ROWS


@pytest.mark.parametrize('paths',[[],STEM,[STEM,STEM],['/'],['relative'],[STEM+'/'],
    ['/World/../Stem'],['/World/'+'a'*MAX_PATH_CHARS]])
def test_invalid_inventory(paths):
    with pytest.raises(ValueError):PlantContactStream(paths)


def test_inventory_iterator_is_consumed_only_to_bounded_limit():
    visited=[]
    def paths():
        for i in range(MAX_PLANT_COLLIDERS+2):
            visited.append(i);yield '/World/C'+str(i)
    with pytest.raises(ValueError):PlantContactStream(paths())
    assert len(visited)==MAX_PLANT_COLLIDERS+1


def test_begin_step_observer_error_latches_and_propagates():
    c=PlantContactStream([STEM]);m=monitor(c)
    def fail():raise ValueError('reset observer failed')
    c.begin_step=fail
    with pytest.raises(ValueError,match='reset'):m.begin_step()
    assert m.error and c.error
    with pytest.raises(RuntimeError):m.measurements(.01)


def test_callback_buffer_failure_invalidates_stream_and_throws(native_stub):
    c=PlantContactStream([STEM]);m=monitor(c)
    h=native_header(collider0=STEM,num_contact_data=-1)
    with pytest.raises(ValueError,match='range'):m._callback([h],[],[])
    assert m.error is not None and c.error is not None


def test_full_observer_cannot_mutate_existing_normal_evidence():
    c=PlantContactStream([STEM]);m=monitor(c);seen=[]
    def change(row):row['point_world_m']=(999,0,0)
    c.add_contact=change
    m.normal_contact_observer=S(begin_step=lambda:None,add_contact=lambda *a:seen.append(a))
    consume(m)
    assert seen[0][2]==[1.,2.,3.] and seen[0][4]==(-.001,0.,0.)
