from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from .grasp_contact_geometry_test import fixture
from .grasp_contact_shadow import predict,bind_source


def source(root=False):
    a=fixture();a['current']['step_id']=1
    for frame in (a['reference'],a['current']):
        frame['body_world_com_jacobians'][0][1][6]=1.
        frame.update(joint_names=['Joint:0'],q_rad=[.01],mass_matrix=np.eye(7).tolist(),
            native_known_noncontact_force=[0.]*7,
            external_root_support_enabled_caller_asserted=root)
    compliance=dict(stiffness_n_m=1000.,damping_n_s_m=1.,reduced_mass_kg=.001,
                    calibrated=False,force_based=True)
    binding=SimpleNamespace(chain=a['chain'],pads=a['pads'],source_target='p/t',
        plant_collider_paths=tuple(a['all_plant_colliders']),finger_friction=.5,
        finger_contact_compliance=compliance)
    previous=dict(before=a['reference'],step_id=1,dt_s=1/240,
        contacts=dict(rows=a['rows'],row_count=len(a['rows']),error=None,
            plant_collider_paths=list(binding.plant_collider_paths)),
        after_generalized_velocity=[0.]*7)
    current=dict(before=a['current'],step_id=2,dt_s=1/240)
    drives=dict(names=['Joint:0'],stiffness=[[3.]],damping=[[.2]],max_forces=[[1.]])
    return previous,current,binding,drives


@pytest.mark.parametrize('root',[False,True])
def test_resolved_counterfactual_preserves_full_or_conditional_root_and_source(root):
    a=source(root);before=deepcopy(a[:2]);result=predict(*a)
    assert result['result']['status']=='resolved'
    assert result['result']['root_dofs']==(0 if root else 6)
    assert len(result['result']['tau_joint'])==1
    assert result['configured_force_caps_passed'] is True
    assert result['native_velocity_prediction_error'] is None
    assert not result['actuation_authorized'] and not result['native_qualified']
    assert not result['measured_contact_forces_used']
    assert a[:2]==before


@pytest.mark.parametrize('fault',['gap','dt','stream','inventory','fetch_velocity','joint_order','source','root','compliance'])
def test_unbound_stale_or_incomplete_evidence_rejects(fault):
    p,c,b,d=source()
    if fault=='gap':c['before']['step_id']=2
    elif fault=='dt':c['dt_s']=float('nan')
    elif fault=='stream':p['contacts']['error']='overflow'
    elif fault=='inventory':p['contacts']['plant_collider_paths']=[]
    elif fault=='fetch_velocity':p['after_generalized_velocity'][0]=.01
    elif fault=='joint_order':d['names']=['different']
    elif fault=='source':p['before']['source_target']='wrong'
    elif fault=='root':c['before']['external_root_support_enabled_caller_asserted']=1
    elif fault=='compliance':b.finger_contact_compliance['force_based']=False
    with pytest.raises(ValueError):predict(p,c,b,d)


def test_source_force_cap_failure_is_reported_never_clamped_or_authorized():
    a=source();a[3]['max_forces']=[[1e-8]]
    r=predict(*a)
    assert r['result']['status']=='resolved'
    assert r['configured_force_caps_passed'] is False
    assert abs(r['result']['tau_joint'][0])>1e-8
    assert r['actuation_authorized'] is False


def test_conditional_root_retains_initial_cross_momentum_without_force_claim():
    p,c,b,d=source(True)
    for frame in (p['before'],c['before']):
        frame['mass_matrix'][0][6]=frame['mass_matrix'][6][0]=.2
        frame['generalized_velocity'][0]=.03
    p['after_generalized_velocity'][0]=.03
    r=predict(p,c,b,d);h=c['dt_s']
    expected=(.2*.03-h*3*.01)/(1+h*.2+h*h*3)
    np.testing.assert_allclose(r['result']['v_pred'],[expected],atol=1e-14)
    assert r['initial_root_to_joint_momentum']==[.006]
    assert not r['actual_external_support_response_verified']


def binding_case(authored=False):
    from .plant_contact_binding_test import fixture as bound_fixture,rehash
    from .plant_contact_binding import capture,AUTHORED_SCHEMA
    f=bound_fixture(signed=True);p=capture(f)
    trace=[dict(contact=dict(binding_sha256=p['binding_sha256'],
        source_target=p['source_target'],allow_signed_native_normals=True))]
    report=dict(rig=dict(source_target=p['source_target'],body_count=len(p['chain'])),
        configuration=dict(finger_friction=f.friction),robot_probe=dict(
            finger_contact_compliance=deepcopy(f.finger_contact_compliance),grasp_body=f.grasp_path))
    if authored:
        p.update(schema=AUTHORED_SCHEMA,binding_sha256=None,native_binding_extras={})
        p['claims']['geometry_basis']='usd_authored_local_shapes_not_native_binding_receipt'
        rehash(p)
    return p,trace,report


@pytest.mark.parametrize('authored',[False,True])
def test_export_binding_matches_recorded_source_without_native_cooking_claim(authored):
    p,t,r=binding_case(authored);b=bind_source(p,t,r)
    assert b.source_target==r['rig']['source_target']


@pytest.mark.parametrize('fault',['hash','target','friction','compliance','selected','count','mixed'])
def test_geometry_cannot_be_transferred_to_different_native_binding_or_material(fault):
    p,t,r=binding_case(True)
    if fault=='hash':t[0]['contact']['binding_sha256']='0'*64
    elif fault=='target':t[0]['contact']['source_target']='wrong'
    elif fault=='friction':r['configuration']['finger_friction']=.7
    elif fault=='compliance':r['robot_probe']['finger_contact_compliance']['stiffness_n_m']=2000.
    elif fault=='selected':r['robot_probe']['grasp_body']='/different'
    elif fault=='count':r['rig']['body_count']+=1
    elif fault=='mixed':
        t.append(deepcopy(t[0]));t[1]['contact']['binding_sha256']='0'*64
    with pytest.raises(ValueError):bind_source(p,t,r)
