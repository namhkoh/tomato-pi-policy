import numpy as np
import pytest
from sim_physics.grasp_wrench import gravity_wrench, static_capacity


def patch():
    return dict(points=[[-.003,-.016,0],[-.003,.016,0],[.003,-.016,0],[.003,.016,0]],
        normals_on_object=[[1,0,0],[1,0,0],[-1,0,0],[-1,0,0]], finger_ids=[0,0,1,1],
        applied_wrench=[0,0,-.08,0,0,0], origin=[0,0,0])


def test_gravity_uses_com_lever_and_preserves_translation():
    expected=[0,0,-.0981,0,.00981,0]
    np.testing.assert_allclose(gravity_wrench([.01],[[.1,0,0]],[0,0,0]), expected)
    np.testing.assert_allclose(gravity_wrench([.01],[[2.1,-1,4]],[2,-1,4]),expected,atol=1e-15)


def test_known_symmetric_friction_load_and_no_execution_claim():
    args=patch();result=static_capacity(**args)
    assert result['balance_found'] and result['within_contact_budgets']
    assert result['normal_load_per_finger_n']==pytest.approx([.08,.08],abs=1e-8)
    assert result['minimum_worst_finger_utilization']==pytest.approx(.24,abs=1e-8)
    assert not result['retention_verified'] and not result['execution_authorized']
    assert not result['input_native_provenance_verified']


def test_long_lever_flags_insufficient_patch_without_raising_force_caps():
    args=patch();args['applied_wrench']=[0,0,-.08,-.01,0,0]
    result=static_capacity(**args)
    assert result['balance_found'] and not result['within_contact_budgets']
    assert result['minimum_worst_finger_utilization']>1
    assert result['contact_budgets_n']==[.5,.5]


def test_zero_friction_cannot_support_transverse_load():
    result=static_capacity(**patch(),friction=0)
    assert not result['balance_found'] and not result['within_contact_budgets']
    assert result['minimum_worst_finger_utilization'] is None


@pytest.mark.parametrize('status,weights',[(1,None),(4,None),(0,[float('nan')]*33),(0,[1.]*33)])
def test_solver_failure_or_unbalanced_output_cannot_pass(monkeypatch,status,weights):
    import sim_physics.grasp_wrench as module
    from types import SimpleNamespace
    monkeypatch.setattr(module,'linprog',lambda *args,**kwargs:SimpleNamespace(
        status=status,x=None if weights is None else np.array(weights),message='injected failure'))
    result=static_capacity(**patch())
    assert not result['within_contact_budgets'] and not result['balance_found']
    assert not result['execution_authorized']


def test_rotation_translation_and_input_immutability():
    args=patch();base=static_capacity(**args)
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('xyz',[.4,.7,-.2]).as_matrix();t=np.array([.6,-.4,.7])
    points=np.array(args['points'])@r.T+t; normals=np.array(args['normals_on_object'])@r.T
    before=points.copy();load=np.r_[r@args['applied_wrench'][:3],r@args['applied_wrench'][3:]]
    result=static_capacity(points,normals,args['finger_ids'],load,origin=t)
    assert result['minimum_worst_finger_utilization']==pytest.approx(base['minimum_worst_finger_utilization'],rel=.09)
    # Polygon orientation may change under rotation; the continuous cone does not.
    np.testing.assert_array_equal(points,before)


@pytest.mark.parametrize('field,value',[
    ('points',[[0,0,0]]),('points',[[float('nan'),0,0]]*4),
    ('normals_on_object',[[2,0,0]]*4),('finger_ids',[0,0,0,0]),('finger_ids',[False,False,True,True]),
    ('finger_ids',[0.,0.,1.,1.]),('applied_wrench',[0]*5),('origin',[float('inf'),0,0]),
    ('friction',True),('friction',-1),('friction',1.1),('per_finger_budget',[.5,.51]),
    ('per_finger_budget',[0,.5])])
def test_bad_patch_rejected(field,value):
    args=patch();args[field]=value
    with pytest.raises(ValueError):static_capacity(**args)


@pytest.mark.parametrize('m,c', [([],[]),([0],[[0,0,0]]),([.1],[[float('nan'),0,0]]),([.1,.2],[[0,0,0]])])
def test_bad_mass_inventory_rejected(m,c):
    with pytest.raises(ValueError):gravity_wrench(m,c,[0,0,0])
