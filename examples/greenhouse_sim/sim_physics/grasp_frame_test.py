import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim_physics.grasp_frame import align_to_axis, approach_rotation
from sim_physics.grasp_frame import checked_approach_retraction


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


def test_reobserved_retraction_stays_on_actual_path_not_old_palm_axis():
    start=np.eye(4);goal=np.eye(4)
    start[:3,3]=[.1,.2,1.]
    goal[:3,3]=start[:3,3]+[.007,0,-.02]
    goal[:3,:3]=Rotation.from_euler('y',5,degrees=True).as_matrix()
    delta=goal[:3,3]-start[:3,3]
    # Reproduce the old off-path command; the unchanged 0.5 mm guard rejects it.
    old=goal[:3,3]+.005*goal[:3,2]
    fraction=np.dot(old-start[:3,3],delta)/np.dot(delta,delta)
    assert np.linalg.norm(old-(start[:3,3]+fraction*delta))>.0005
    vector=checked_approach_retraction(start,goal,.005)
    assert np.linalg.norm(vector)==pytest.approx(.005)
    for t in np.linspace(0,1,241):
        point=goal[:3,3]+t*vector
        fraction=np.dot(point-start[:3,3],delta)/np.dot(delta,delta)
        assert 0<=fraction<=1+1e-12
        np.testing.assert_allclose(point,start[:3,3]+fraction*delta,atol=1e-12)


@pytest.mark.parametrize('distance',[0.,.001,.005,.01])
def test_unchanged_axis_aligned_approach_preserves_metric_request(distance):
    start=np.eye(4);goal=np.eye(4);start[2,3]=.02
    np.testing.assert_allclose(checked_approach_retraction(start,goal,distance),[0,0,distance])


@pytest.mark.parametrize('distance',[-.001,.010001,np.nan,np.inf,True,[.005]])
def test_invalid_retraction_rejected(distance):
    start=np.eye(4);start[2,3]=.02
    with pytest.raises(ValueError):checked_approach_retraction(start,np.eye(4),distance)


@pytest.mark.parametrize('fault',['too_short','zero','scale','reflection','bottom','nan'])
def test_retraction_requires_real_corridor_and_rigid_frames(fault):
    start=np.eye(4);goal=np.eye(4);start[2,3]=.02
    if fault=='too_short':start[2,3]=.012
    if fault=='zero':start[2,3]=0
    if fault=='scale':goal[0,0]=2
    if fault=='reflection':goal[0,0]=-1
    if fault=='bottom':goal[3,0]=1
    if fault=='nan':goal[1,3]=np.nan
    with pytest.raises(ValueError):checked_approach_retraction(start,goal,.005)
@pytest.mark.parametrize('degrees',[-60,-30,0,30,60])
def test_pad_pitch_preserves_closing_axis_and_material_point(degrees):
    from sim_physics.grasp_frame import pitch_for_pad_span
    r=np.array([[0.,1,0],[0,0,1],[1,0,0]])
    before=r.copy();result=pitch_for_pad_span(r,degrees)
    np.testing.assert_allclose(result.T@result,np.eye(3),atol=1e-12)
    np.testing.assert_array_equal(result[:,0],r[:,0])
    np.testing.assert_array_equal(r,before)
    assert result[:,2]@r[:,2]>=.5-1e-12
    point=np.array([.02,.5,1.4]);depth=.1025
    palm=point+depth*result[:,2]
    np.testing.assert_allclose(result.T@(point-palm),[0,0,-depth],atol=1e-14)


@pytest.mark.parametrize('degrees',[-61,61,float('nan'),True,[0]])
def test_bad_pad_pitch_rejected(degrees):
    from sim_physics.grasp_frame import pitch_for_pad_span
    with pytest.raises(ValueError):pitch_for_pad_span(np.eye(3),degrees)


def test_pad_pitch_rejects_nonrigid_rotation():
    from sim_physics.grasp_frame import pitch_for_pad_span
    for r in (np.ones((3,3)),np.diag([-1,1,1]),np.eye(4)):
        with pytest.raises(ValueError):pitch_for_pad_span(r,30)
