import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim_physics.grasp_frame import align_to_axis, approach_rotation


def test_current_axis_updates_the_entire_palm_without_changing_roll():
    r=Rotation.from_euler('xyz',[13,27,41],degrees=True).as_matrix()
    delta=Rotation.from_rotvec(np.array([.01,-.02,.03])).as_matrix()
    a=r[:,1];b=delta@a
    aligned=align_to_axis(r,a,b)
    np.testing.assert_allclose(aligned[:,1],b,atol=1e-12)
    np.testing.assert_allclose(aligned.T@aligned,np.eye(3),atol=1e-12)
    assert np.linalg.det(aligned)==pytest.approx(1.)
    rolled=r@np.diag([-1.,-1.,1.])
    np.testing.assert_allclose(align_to_axis(rolled,a,b),aligned@np.diag([-1.,-1.,1.]),atol=1e-12)


def test_rest_orientation_is_exact_and_large_or_invalid_change_rejected():
    r=np.eye(3);axis=r[:,1]
    np.testing.assert_array_equal(align_to_axis(r,axis,axis),r)
    for b in ([0,-1,0],[0,0,1],[0,2,0],[0,np.nan,0]):
        with pytest.raises(ValueError):align_to_axis(r,axis,b)


def test_approach_rotation_reaches_goal_and_does_not_rotate_further_on_pull():
    start=np.eye(3);goal=Rotation.from_euler('x',4,degrees=True).as_matrix()
    np.testing.assert_array_equal(approach_rotation(start,start,.57),start)
    np.testing.assert_allclose(approach_rotation(start,goal,0),start,atol=1e-12)
    for fraction in (1,1.05,1.15):
        np.testing.assert_allclose(approach_rotation(start,goal,fraction),goal,atol=1e-12)
    np.testing.assert_allclose(approach_rotation(start,goal,.5),Rotation.from_euler('x',2,degrees=True).as_matrix(),atol=1e-12)


@pytest.mark.parametrize('fraction',[-.01,1.16,float('nan')])
def test_unchecked_approach_fraction_is_refused(fraction):
    with pytest.raises(ValueError):approach_rotation(np.eye(3),np.eye(3),fraction)
