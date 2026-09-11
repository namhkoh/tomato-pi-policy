from types import SimpleNamespace as S
import numpy as np
import pytest
from .plant_prediction_snapshot import PlantPredictionSnapshot


def fixture():
    a=S(count=1,shared_metatype=S(fixed_base=False,dof_names=['a','b']),link_paths=[['/Root','/Child']])
    a.get_dof_positions=lambda:np.array([[.1,.2]])
    a.get_dof_velocities=lambda:np.zeros((1,2))
    a.get_root_velocities=lambda:np.zeros((1,6))
    a.get_generalized_mass_matrices=lambda:np.eye(8)[None]
    a.get_jacobians=lambda:np.zeros((1,2,6,8))
    a.get_link_transforms=lambda:np.tile([0,0,0,0,0,0,1],(1,2,1))
    a.get_link_velocities=lambda:np.zeros((1,2,6))
    a.get_coms=lambda:np.tile([.01,0,0,0,0,0,1],(1,2,1))
    a.get_gravity_compensation_forces=lambda:np.arange(8)[None]
    a.get_coriolis_and_centrifugal_compensation_forces=lambda:np.ones((1,8))
    return a


def observer(a):return PlantPredictionSnapshot(a,source_target='plant/petiole',expected_body_paths=['/Root','/Child'])


@pytest.mark.parametrize('constrained',[True,False])
def test_root_retained_nonzero_com_not_silently_zeroed(constrained):
    a=fixture();r=observer(a).read(step=7,root_constrained=constrained)
    assert np.array(r['mass_matrix']).shape==(8,8)
    assert r['native_com_local_poses'][0][0]==.01
    assert r['native_known_noncontact_force']==list(-np.arange(8)-1)
    assert r['external_root_support_enabled_caller_asserted'] is constrained
    assert r['contact_force_included'] is False


def test_finite_nonzero_com_jacobian_velocity_mapping():
    a=fixture();a.get_dof_velocities=lambda:np.array([[2.,0.]])
    j=np.zeros((1,2,6,8));j[0,1,1,6]=.5;a.get_jacobians=lambda:j
    v=np.zeros((1,2,6));v[0,1,1]=1;a.get_link_velocities=lambda:v
    assert observer(a).read(step=0,root_constrained=False)['point_velocity_max_error']==0
    v[0,1,1]=0
    with pytest.raises(ValueError,match='convention'):observer(a).read(step=0,root_constrained=False)


@pytest.mark.parametrize('change',[lambda a:setattr(a,'count',2),
    lambda a:setattr(a.shared_metatype,'fixed_base',True),
    lambda a:a.link_paths[0].reverse(),lambda a:a.shared_metatype.dof_names.append('extra')])
def test_inventory_change_fails_closed(change):
    a=fixture();o=observer(a);change(a)
    with pytest.raises(ValueError):o.read(step=0,root_constrained=False)


def test_bad_mass_and_step_cannot_become_prediction_evidence():
    a=fixture();o=observer(a)
    with pytest.raises(ValueError):o.read(step=True,root_constrained=False)
    a.get_generalized_mass_matrices=lambda:-np.eye(8)[None]
    with pytest.raises(np.linalg.LinAlgError):o.read(step=0,root_constrained=False)
