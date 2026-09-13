import numpy as np
import pytest
from .kinetic_consistency import check


def sample():
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    frame=np.eye(4);frame[:3,:3]=rot
    inertia=np.diag([.1,.2,.3]);world=rot@inertia@rot.T
    m=np.zeros((6,6));m[:3,:3]=np.eye(3)*2;m[3:,3:]=world
    v=np.arange(6)*.1
    return dict(body_world_com_jacobians=np.eye(6)[None],body_frames_world=frame[None],
        body_velocities_world=v[None],generalized_velocity=v,mass_matrix=m),[2.],inertia.reshape(1,9)


def test_rotated_body_mass_and_energy_match_without_second_principal_rotation():
    s,m,i=sample();r=check(s,m,i)
    assert r['mass_max_diagonal_scaled_difference']==pytest.approx(0)
    assert r['kinetic_difference_j']==pytest.approx(0)
    assert r['jacobian_velocity_max_error']==0


def test_report_disagreement_without_changing_native_arrays():
    s,m,i=sample();s['mass_matrix'][0,0]+=.01;before=s['mass_matrix'].copy()
    r=check(s,m,i)
    assert r['mass_max_diagonal_scaled_difference']>0
    np.testing.assert_array_equal(s['mass_matrix'],before)


@pytest.mark.parametrize('key',['mass_matrix','generalized_velocity','body_velocities_world','body_world_com_jacobians'])
def test_nonfinite_fails(key):
    s,m,i=sample();s[key].flat[0]=np.nan
    with pytest.raises(ValueError):check(s,m,i)
