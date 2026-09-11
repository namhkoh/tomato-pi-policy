"""CPU-only momentum ledger tests. No native runtime, artifacts or gate changes."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sim_physics.contact_momentum import step_ledger, trace_ledger, summarize
from sim_physics.contact_spring_probe import Coupon, layout, _ry


def coupon():
    return Coupon('synthetic-momentum-not-native', 'a'*64, (.3, .3), (.01, .01),
        (1., 2., 3.), tuple(np.diag([.2, .3, .4]) for _ in range(3)),
        1000., 1., .5)


def fixture():
    c = coupon(); data = layout(c)
    previous = dict(source_sha256=c.source_sha256, step_id=1, dt_s=c.dt,
        frames_world_m=data['frames'].tolist(), body_velocities_world=np.zeros((3, 6)).tolist())
    current = deepcopy(previous); current.update(step_id=2, contact_rows=[],
        complete_stream_caller_asserted=True)
    evidence = dict(source_sha256=c.source_sha256, step_id=2, dt_s=c.dt,
        normal_rows_complete=True, friction_rows_complete=True, contacts_are_all_external_loads=True)
    return dict(coupon=c, previous=previous, current=current, masses=c.masses,
        inertias_body=c.inertias, com_local=np.zeros((3, 3)),
        mass_properties_basis='source_authored', contact_evidence=evidence,
        origin_world_m=[0., 0., 0.])


def row(c, impulse, point, kind='normal', body=0, reversed_order=False):
    data = layout(c); a=data['collider_paths'][body]; b=next(iter(data['pads']))
    if reversed_order: a,b=b,a; impulse=-np.asarray(impulse)
    return dict(collider0=a, collider1=b, kind=kind,
                point_world_m=np.asarray(point).tolist(), impulse_on_0_ns=np.asarray(impulse).tolist())


def accelerated():
    a=fixture(); c=a['coupon']; F=np.asarray(a['previous']['frames_world_m'])
    R=F[0,:3,:3]; pos=F[0,:3,3]; p=pos+np.array([.03,.02,.01])
    j=np.array([.002,-.001,.003])
    a['current']['contact_rows']=[row(c,j,p)]
    V=np.zeros((3,6)); V[0,:3]=j/c.masses[0]
    V[0,3:]=np.linalg.solve(R@np.array(c.inertias[0])@R.T,np.cross(p-pos,j))
    a['current']['body_velocities_world']=V.tolist()
    return a


def test_isolated_import_does_not_load_native():
    env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
             PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',
        'import sys;import sim_physics.contact_momentum;'
        'assert not any(n.startswith(("pxr","omni","isaacsim")) for n in sys.modules)'],
        env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_steady_balanced_normal_and_friction_contacts():
    a=fixture(); c=a['coupon']; point=np.asarray(a['previous']['frames_world_m'])[0,:3,3]
    a['current']['contact_rows']=[
        row(c,[.001,0,0],point),row(c,[-.001,0,0],point),
        row(c,[0,.0002,0],point,'friction'),row(c,[0,-.0002,0],point,'friction')]
    a['contact_evidence'].update(expected_normal_rows=2,expected_friction_rows=2)
    r=step_ledger(**a)
    assert r['static_net_wrench']['passed']
    assert r['force_residual_norm_n']==r['torque_residual_norm_nm']==0.
    assert r['dynamic_pass'] is r['dynamic_residual_tolerance'] is None
    assert not r['native_qualified'] and not r['force_applied']
    assert r['contact_evidence']['assertion_not_authentication']


def test_acceleration_balances_nonzero_torque_not_static_zero():
    r=step_ledger(**accelerated())
    np.testing.assert_allclose(r['delta_p_dt_n'],r['contact_force_n'],atol=1e-15)
    np.testing.assert_allclose(r['delta_h_dt_nm'],r['contact_torque_nm'],atol=1e-15)
    assert r['force_residual_norm_n']<1e-14 and r['torque_residual_norm_nm']<1e-14
    assert not r['static_net_wrench']['passed']
    assert np.linalg.norm(r['body_angular_acceleration_rad_s2'])>0
    assert r['same_fixed_origin_both_endpoints']


def test_same_fixed_origin_both_endpoints_and_translation_covariance():
    a=accelerated(); r0=step_ledger(**a)
    origin=np.array([.3,-.2,.1]); a['origin_world_m']=origin
    r1=step_ledger(**a)
    np.testing.assert_allclose(r1['contact_torque_nm'],
        r0['contact_torque_nm']-np.cross(origin,r0['contact_force_n']),atol=1e-15)
    np.testing.assert_allclose(r1['delta_h_dt_nm'],r1['contact_torque_nm'],atol=1e-15)
    # Translate BOTH frames, the physical contact points and reference equally.
    shift=np.array([1.,2.,3.])
    for state in (a['previous'],a['current']):
        F=np.asarray(state['frames_world_m']);F[:,:3,3]+=shift;state['frames_world_m']=F.tolist()
    for cr in a['current']['contact_rows']:cr['point_world_m']=(np.asarray(cr['point_world_m'])+shift).tolist()
    a['origin_world_m']=origin+shift
    moved=step_ledger(**a)
    np.testing.assert_allclose(moved['delta_h_dt_nm'],r1['delta_h_dt_nm'],atol=1e-15)
    np.testing.assert_allclose(moved['contact_torque_nm'],r1['contact_torque_nm'],atol=1e-15)


def test_original_collider_order_reversal_and_signed_tensile_are_not_absed():
    a=accelerated(); first=step_ledger(**a)
    cr=a['current']['contact_rows'][0]
    cr['collider0'],cr['collider1']=cr['collider1'],cr['collider0']
    cr['impulse_on_0_ns']=(-np.asarray(cr['impulse_on_0_ns'])).tolist()
    reverse=step_ledger(**a)
    np.testing.assert_array_equal(reverse['contact_force_n'],first['contact_force_n'])
    np.testing.assert_array_equal(reverse['contact_torque_nm'],first['contact_torque_nm'])
    # An opposite signed impulse must subtract, not contribute its magnitude.
    opposite=deepcopy(cr);opposite['impulse_on_0_ns']=(-np.asarray(cr['impulse_on_0_ns'])).tolist()
    a['current']['contact_rows'].append(opposite)
    canceled=step_ledger(**a)
    np.testing.assert_array_equal(canceled['contact_force_n'],0)
    assert canceled['force_residual_norm_n']>0


def test_internal_contact_equal_opposite_cancels_global_wrench():
    a=fixture(); data=layout(a['coupon'])
    cr=row(a['coupon'],[1,2,3],[.01,.02,.03],'friction')
    cr['collider1']=data['collider_paths'][1]
    a['current']['contact_rows']=[cr]
    r=step_ledger(**a)
    assert r['contact_evidence']['internal_contact_rows']==1
    np.testing.assert_array_equal(r['contact_force_n'],0)
    np.testing.assert_array_equal(r['contact_torque_nm'],0)


def test_orientation_and_lever_changes_are_retained_in_exact_finite_identity():
    a=fixture(); h=a['coupon'].dt
    F0=np.asarray(a['previous']['frames_world_m']);F1=F0.copy()
    F1[0,:3,:3]=_ry(.02)@F0[0,:3,:3]
    F1[1,:3,3]+=[.001,0,0]
    V=np.zeros((3,6));V[0,3:]=[.2,.3,.4];V[1,:3]=[0,.2,0]
    a['previous']['body_velocities_world']=V.tolist()
    a['current']['body_velocities_world']=V.tolist()
    a['current']['frames_world_m']=F1.tolist()
    r=step_ledger(**a)
    I=np.array(a['coupon'].inertias[0])
    orientation=(F1[0,:3,:3]@I@F1[0,:3,:3].T-F0[0,:3,:3]@I@F0[0,:3,:3].T)@V[0,3:]/h
    lever=np.cross([.001/h,0,0],a['coupon'].masses[1]*V[1,:3])
    np.testing.assert_allclose(r['orientation_change_term_nm'],orientation,atol=1e-15)
    np.testing.assert_allclose(r['lever_change_term_nm'],lever,atol=1e-15)
    np.testing.assert_allclose(r['delta_h_dt_nm'],orientation+lever,atol=1e-15)
    # Algebraic FP cancellation bound, NOT a native/dynamic residual gate.
    roundoff=64*np.finfo(float).eps*(np.linalg.norm(r['previous_angular_momentum_nms'])
        +np.linalg.norm(r['current_angular_momentum_nms']))/h
    np.testing.assert_allclose(r['momentum_decomposition_error_nm'],0,atol=roundoff)
    # No measured contact accounts for this synthetic change: no dynamic pass.
    assert r['torque_residual_norm_nm']>1e-6
    assert r['static_net_wrench']['passed'] and r['dynamic_pass'] is None


@pytest.mark.parametrize('fault', ['normal_stream','friction_stream','other_loads','recorded_incomplete',
    'friction_count','normal_count','contact_step','contact_dt','contact_source'])
def test_incomplete_or_misbound_contact_evidence_rejected(fault):
    a=fixture(); e=a['contact_evidence']
    if fault=='normal_stream':e['normal_rows_complete']=False
    elif fault=='friction_stream':e['friction_rows_complete']=False
    elif fault=='other_loads':e['contacts_are_all_external_loads']=False
    elif fault=='recorded_incomplete':a['current']['complete_stream_caller_asserted']=False
    elif fault=='friction_count':e['expected_friction_rows']=1
    elif fault=='normal_count':e['expected_normal_rows']=1
    elif fault=='contact_step':e['step_id']=1
    elif fault=='contact_dt':e['dt_s']*=2
    else:e['source_sha256']='b'*64
    with pytest.raises(ValueError):step_ledger(**a)


def test_cannot_infer_missing_friction_from_absent_rows_without_count_evidence():
    a=fixture()
    r=step_ledger(**a)  # empty FREE stream and falsely asserted missing stream are indistinguishable
    assert r['contact_evidence']['observed_friction_rows']==0
    assert not r['contact_evidence']['missing_friction_inferred']
    assert r['contact_evidence']['assertion_not_authentication']
    assert not r['native_qualified']
    a['contact_evidence']['expected_friction_rows']=2
    with pytest.raises(ValueError,match='friction row count'):step_ledger(**a)


@pytest.mark.parametrize('fault',['mass','inertia','com','basis','scale','reflection','homogeneous',
    'frame_nan','velocity_nan','origin_inf','dt','dt_bool','previous_dt','previous_source',
    'current_source','gap','step_bool','prefetch_frame','inertia_singular'])
def test_invalid_source_state_mass_frame_dt_and_sequence_rejected(fault):
    a=fixture()
    if fault=='mass':a['masses']=[1.01,2,3]
    elif fault=='inertia':a['inertias_body']=np.asarray(a['inertias_body'])*1.01
    elif fault=='inertia_singular':a['inertias_body']=np.zeros((3,3,3))
    elif fault=='com':a['com_local'][0,0]=1e-12
    elif fault=='basis':a['mass_properties_basis']='assume_native'
    elif fault=='scale':a['current']['frames_world_m'][0][0][0]*=1.1
    elif fault=='reflection':a['current']['frames_world_m'][0][1][1]*=-1
    elif fault=='homogeneous':a['current']['frames_world_m'][0][3][0]=1e-10
    elif fault=='frame_nan':a['current']['frames_world_m'][0][0][0]=np.nan
    elif fault=='velocity_nan':a['previous']['body_velocities_world'][0][0]=np.nan
    elif fault=='origin_inf':a['origin_world_m']=[np.inf,0,0]
    elif fault=='dt':a['current']['dt_s']*=2
    elif fault=='dt_bool':a['current']['dt_s']=True
    elif fault=='previous_dt':a['previous']['dt_s']*=2
    elif fault=='previous_source':a['previous']['source_sha256']='b'*64
    elif fault=='current_source':a['current']['source_sha256']='b'*64
    elif fault=='gap':a['current']['step_id']=4
    elif fault=='step_bool':a['previous']['step_id']=True
    else:
        a['current']['native_prediction_before_step']=dict(frames_world_m=deepcopy(a['previous']['frames_world_m']))
        a['current']['native_prediction_before_step']['frames_world_m'][0][0][3]+=.001
    with pytest.raises(ValueError):step_ledger(**a)


@pytest.mark.parametrize('fault',['unknown','irrelevant','same','kind','nan_impulse','nan_point','overflow'])
def test_bad_contact_rows_rejected(fault):
    a=fixture();cr=row(a['coupon'],[1,2,3],[0,0,0])
    if fault=='unknown':cr['collider0']='/World/Unknown'
    elif fault=='irrelevant':
        cr['collider0'],cr['collider1']=list(layout(a['coupon'])['pads'])[:2]
    elif fault=='same':cr['collider1']=cr['collider0']
    elif fault=='kind':cr['kind']='unsigned_scalar'
    elif fault=='nan_impulse':cr['impulse_on_0_ns'][0]=np.nan
    elif fault=='nan_point':cr['point_world_m'][0]=np.nan
    a['current']['contact_rows']=[cr]*257 if fault=='overflow' else [cr]
    with pytest.raises(ValueError):step_ledger(**a)


def test_actual_supplied_native_properties_not_replaced_by_source_nominal():
    a=accelerated();a['mass_properties_basis']='native_readback'
    actual=np.asarray(a['masses'])*(1+1e-7);a['masses']=actual
    r=step_ledger(**a)
    np.testing.assert_array_equal(r['mass_properties']['masses_kg'],actual)
    assert r['force_residual_norm_n']>0  # must not silently restore nominal masses
    assert not r['mass_properties']['native_properties_authenticated']
    a['masses']=actual*1.01
    with pytest.raises(ValueError,match='source mismatch'):step_ledger(**a)


def test_finite_json_determinism_copy_and_no_input_mutation():
    a=accelerated();before=deepcopy(a)
    r=step_ledger(**a)
    assert json.dumps(r,sort_keys=True,allow_nan=False)==json.dumps(step_ledger(**a),sort_keys=True,allow_nan=False)
    assert a['previous']==before['previous'] and a['current']==before['current']
    r['contact_evidence']['normal_rows_complete']=False
    r['mass_properties']['masses_kg'][0]=99
    assert a['contact_evidence']['normal_rows_complete'] is True and a['masses'][0]==1.


def trace_fixture():
    a=accelerated();c=a['coupon']
    initial=dict(frames=a['previous']['frames_world_m'],
        native_body_velocities=a['previous']['body_velocities_world'])
    # Deliberately nonzero bootstrap velocity; the first ordinary interval must retain it.
    initial['native_body_velocities']=deepcopy(initial['native_body_velocities'])
    initial['native_body_velocities'][0][0]=.01
    first=deepcopy(a['current']);first['step_id']=1
    first['body_velocities_world'][0][0]+=.01
    second=deepcopy(first);second.update(step_id=2,contact_rows=[])
    stepping=dict(startup_physics_steps=1,startup_physics_dt_s=1e-6)
    opts={k:a[k] for k in ('masses','inertias_body','com_local','mass_properties_basis')}
    opts['contact_assertions']={k:True for k in
        ('normal_rows_complete','friction_rows_complete','contacts_are_all_external_loads')}
    return c,[first,second],initial,stepping,opts


def test_trace_uses_post_bootstrap_readback_not_authored_rest_or_zero_velocity():
    c,rows,initial,stepping,opts=trace_fixture()
    result=trace_ledger(c,rows,initial,stepping,**opts,use_current_middle_origin=True)
    first=result['records'][0]
    assert first['first_interval_initial_state_basis']=='post_bootstrap_native_readback'
    assert first['initial_bootstrap_dt_s']==1e-6 and first['dt_s']==c.dt
    assert not first['bootstrap_impulses_included']
    assert first['previous_linear_momentum_ns'][0]==.01
    assert first['force_residual_norm_n']<1e-15
    assert result['bootstrap_physics_steps_excluded']==1
    assert result['summary']['sample_count']==2


@pytest.mark.parametrize('fault',['missing_initial_basis','skip_first','gap','bad_boot_dt','bad_boot_count','initial_source'])
def test_initial_and_trace_contiguity_not_implicitly_repaired(fault):
    c,rows,initial,stepping,opts=trace_fixture()
    if fault=='missing_initial_basis':
        a=fixture();a['previous']['step_id']=0;a['current']['step_id']=1
        a['contact_evidence']['step_id']=1
        with pytest.raises(ValueError,match='initial readback'):step_ledger(**a)
        return
    if fault=='skip_first':rows=rows[1:]
    elif fault=='gap':rows[1]['step_id']=3
    elif fault=='bad_boot_dt':stepping['startup_physics_dt_s']=0.
    elif fault=='bad_boot_count':stepping['startup_physics_steps']=2
    else:initial['source_sha256']='b'*64
    with pytest.raises(ValueError):trace_ledger(c,rows,initial,stepping,**opts)


def test_summary_keeps_dynamic_residuals_separate_from_static_and_rejects_mixed_tail():
    c,rows,initial,stepping,opts=trace_fixture()
    ledgers=trace_ledger(c,rows,initial,stepping,**opts)['records']
    s=summarize(ledgers)
    assert s['static_wrench_failed_samples']==1 and s['dynamic_pass'] is None
    ledgers[1]['source_sha256']='b'*64
    with pytest.raises(ValueError):summarize(ledgers)


def test_declared_body_order_mismatch_rejected_and_numpy_step_counts_json_safe():
    a=fixture();a['previous']['body_paths']=list(reversed(layout(a['coupon'])['body_paths']))
    with pytest.raises(ValueError,match='body order'):step_ledger(**a)
    a['previous']['body_paths']=layout(a['coupon'])['body_paths']
    a['previous']['step_id']=np.int64(1);a['current']['step_id']=np.int64(2)
    a['contact_evidence']['step_id']=np.int64(2)
    a['contact_evidence']['expected_friction_rows']=np.int64(0)
    r=step_ledger(**a)
    json.dumps(r,allow_nan=False)
    assert type(r['step_id']) is int
    assert type(r['contact_evidence']['expected_friction_rows']) is int


def test_complex_values_not_silently_cast_and_string_dt_rejected():
    a=fixture();a['current']['body_velocities_world']=np.zeros((3,6),dtype=complex)
    with pytest.raises(ValueError,match='Real'):step_ledger(**a)
    a=fixture();a['current']['dt_s']=str(a['coupon'].dt)
    with pytest.raises(ValueError,match='dt'):step_ledger(**a)


def test_archived31_34_pure_cpu_momentum_ledger_regression():
    """Optional ignored captures: no native run and no physical requalification."""
    root=Path(__file__).resolve().parents[3]/'data'/'sim_physics'
    hashes={
        '31':'55334886f3dbcf6e7b9c4f908d394f4b6d248f4d7dd71db020881a81c4e91c3c',
        '34':'bbfcf3ab8c01cb90f10ef19503d2eecc7c1a24d3544fc721af30fdf3e7b1e9e4'}
    if any(not (root/('contact_spring_native_20260911_'+label)/'trace.json').is_file() for label in hashes):
        pytest.skip('Ignored native31/34 captures absent; other tests remain pure and mandatory')
    reference=None
    for label,digest in hashes.items():
        folder=root/('contact_spring_native_20260911_'+label)
        raw=(folder/'trace.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        samples=json.loads(raw);report=json.loads((folder/'report.json').read_bytes())
        c=Coupon(**{x.name:report['configuration'][x.name] for x in fields(Coupon)})
        r=trace_ledger(c,samples,report['initial_readback'],report['stepping'],
            masses=c.masses,inertias_body=c.inertias,com_local=np.zeros((3,3)),
            mass_properties_basis='source_authored',
            contact_assertions=dict(normal_rows_complete=True,friction_rows_complete=True,
                contacts_are_all_external_loads=True,expected_normal_rows=12,expected_friction_rows=12),
            use_current_middle_origin=True)
        assert len(r['records'])==1920
        if reference is None:reference=r['records']
        else:assert r['records']==reference
        tail=summarize(r['records'][-961:])
        # Regression of recorded metrics, NOT new runtime physical thresholds.
        assert tail['max_torque_residual_nm']==pytest.approx(2.970482563938636e-9,rel=1e-8)
        assert tail['rms_torque_residual_nm']==pytest.approx(9.456476409680286e-10,rel=1e-8)
        assert tail['static_torque_failed_samples']==704
        assert tail['dynamic_pass'] is None and not tail['native_qualified']
        for record,sample in zip(r['records'],samples):
            np.testing.assert_allclose(record['contact_torque_nm'],sample['net_contact_torque_nm'],
                                       rtol=0,atol=3e-18)
        assert r['records'][0]['first_interval_initial_state_basis']=='post_bootstrap_native_readback'
        assert r['bootstrap_dt_s_excluded']==1e-6
