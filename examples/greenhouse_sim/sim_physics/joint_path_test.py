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


def test_nearby_configurations_do_not_share_rounded_clearance():
    seen=[]
    def valid(q):seen.append(float(q[0]));return q[0]<2e-9
    assert connect_path(np.array([1e-9]),np.array([4e-9]),np.array([-1.]),np.array([1.]),valid) is None
    assert seen==[1e-9,4e-9]


def test_telemetry_does_not_change_path_or_claim_infeasibility():
    valid=lambda q:not (-3<q[0]<3 and q[1]<10)
    args=(np.array([-20.,0]),np.array([20.,0]),np.full(2,-30.),np.full(2,30.),valid)
    detail={'stale':True}
    expected=connect_path(*args);actual=connect_path(*args,diagnostics=detail)
    np.testing.assert_array_equal(actual,expected)
    assert detail['state']=='connected_path_found' and 'stale' not in detail
    assert detail['vertices_start']>1 and detail['vertices_goal']>1 and detail['checks']>2
    assert not detail['motion_authorized'] and not detail['exhaustion_is_infeasibility_proof']


def test_rejected_endpoint_telemetry():
    detail={}
    assert connect_path(np.zeros(1),np.ones(1),np.full(1,-2),np.full(1,2),lambda q:q[0]==0,
        diagnostics=detail) is None
    assert detail['state']=='goal_rejected' and detail['checks']==2
