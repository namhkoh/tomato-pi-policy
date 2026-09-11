import numpy as np
import pytest
from sim_physics.joint_path import connect_path


def test_bounded_search_goes_around_obstacle_and_checks_dense_path():
    valid=lambda q:not (-3<q[0]<3 and q[1]<10)
    args=(np.array([-20.,0]),np.array([20.,0]),np.full(2,-30.),np.full(2,30.),valid)
    path=connect_path(*args)
    assert path is not None
    np.testing.assert_array_equal(path[0],args[0]);np.testing.assert_array_equal(path[-1],args[1])
    assert all(valid(q) for q in path)
    assert np.max(np.abs(np.diff(path,axis=0)))<=1+1e-10
    np.testing.assert_array_equal(path,connect_path(*args))


def test_exhaustion_and_blocked_goal_do_not_return_unchecked_fallback():
    counter=[]
    def valid(q):
        counter.append(q.copy())
        return not -3<q[0]<3
    args=(np.array([-20.,0]),np.array([20.,0]),np.full(2,-30.),np.full(2,30.))
    assert connect_path(*args,valid,max_checks=100) is None
    assert len(counter)<=100
    assert connect_path(*args,lambda q:False) is None
    with pytest.raises(ValueError): connect_path(*args,valid,max_checks=1)
    with pytest.raises(ValueError): connect_path(np.array([float('nan'),0]),*args[1:],valid)
