"""Pure array contracts only; actual backend results live in separate receipts."""
import numpy as np
import pytest
from sim_physics.newton_coupon import damping_ratio,contact_rows
from sim_physics.contact_spring_probe import sample,layout
from sim_physics.contact_spring_probe_test import coupon,equilibrium


def test_rayleigh_conversion_preserves_physical_damping():
    k=np.array([.3,.4,1000.]);c=np.array([.009,.008,1.05])
    np.testing.assert_allclose(k*damping_ratio(k,c),c)


@pytest.mark.parametrize('k,c',[(0,1),(-1,1),(1,-1),(float('nan'),1),([1,2],[1]),(1,float('inf'))])
def test_invalid_damping_conversion_rejected(k,c):
    with pytest.raises(ValueError):damping_ratio(k,c)


def contact_fixture():
    return dict(shape0=[0],shape1=[1],points0=[[0,0,0]],points1=[[0,0,.01]],
        normals=[[1,0,0]],forces1=[[.2,.03,0]],shapes={0:'A',1:'B'},dt=1/240)


def test_native_force_on_one_reversed_and_split_once():
    args=contact_fixture();rows=contact_rows(**args)
    np.testing.assert_allclose(np.sum([r['impulse_on_0_ns'] for r in rows],axis=0),-np.array(args['forces1'][0])/240)
    assert rows[0]['normal_on_0']==[-1.,0.,0.]
    assert rows[1]['point_on_1_world_m']==[0.,0.,.01]
    assert rows[0]['kind']=='normal' and rows[1]['kind']=='friction'


@pytest.mark.parametrize('key,value',[('normals',[[2,0,0]]),('forces1',[[-.1,0,0]]),
    ('points1',[[float('nan'),0,0]]),('shape1',[9]),('shape1',[0]),('dt',0)])
def test_bad_native_contact_contract_rejected(key,value):
    args=contact_fixture();args[key]=value
    with pytest.raises(ValueError):contact_rows(**args)


def test_oracle_uses_second_application_point_for_second_dynamic_body():
    c=coupon();args=equilibrium(c);data=layout(c)
    args['contact_rows']=[dict(collider0=data['collider_paths'][0],collider1=data['collider_paths'][1],
        kind='friction',point_world_m=[0,0,0],point_on_1_world_m=[0,0,.01],impulse_on_0_ns=[c.dt,0,0])]
    row=sample(c,**args)
    wrench=np.array(row['contact_wrenches_at_body_com'])
    expected=np.cross(np.array([0,0,.01])-np.array(args['frames'][1,:3,3]),[-1,0,0])
    np.testing.assert_allclose(wrench[1,3:],expected)
    assert row['contact_rows'][0]['point_on_1_world_m']==[0.,0.,.01]


def test_bad_second_application_point_rejected():
    c=coupon();args=equilibrium(c);args['contact_rows'][0]['point_on_1_world_m']=[1,2]
    with pytest.raises(ValueError):sample(c,**args)
