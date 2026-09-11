"""Pure geometry/derivative tests; no native, USD, scene changes or jobs."""
import json

import numpy as np
import pytest

from greenhouse_sim.robot_kinematics import _segment_aabb_distance
from .capsule_box_witness import closest_capsule_box as witness, MIN_AXIS_BOX_DISTANCE_M


def check_surface(result, start, end, half, radius):
    start, end, half = map(np.asarray, (start, end, half))
    axis = np.asarray(result['axis_point']); box = np.asarray(result['box_point'])
    point = np.asarray(result['capsule_point']); n = np.asarray(result['normal'])
    np.testing.assert_allclose(axis, (1-result['parameter'])*start+result['parameter']*end, atol=1e-14, rtol=1e-13)
    assert np.all(abs(box) <= half)
    np.testing.assert_allclose(np.linalg.norm(n), 1., atol=3e-15)
    np.testing.assert_allclose(axis-box, result['distance_m']*n, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(point-box, result['gap_m']*n, atol=2e-14, rtol=2e-13)
    delta = end-start
    t = .5 if np.array_equal(start, end) else np.clip((point-start)@delta/(delta@delta), 0., 1.)
    closest = start+t*delta
    np.testing.assert_allclose(np.linalg.norm(point-closest), radius, atol=2e-14, rtol=2e-12)
    assert result['witness_count'] == 1 and result['native_contact_observed'] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('start,end,cap_feature,box_feature', [
    ([.02,0,0],[.03,.002,0],'start_cap','face'),
    ([.03,.002,0],[.02,0,0],'end_cap','face'),
    ([.02,.02,0],[.03,.03,0],'start_cap','edge'),
    ([.02,.02,.02],[.03,.03,.03],'start_cap','corner'),
    ([.03,0,0],[0,.03,0],'cylinder','edge'),
    ([.02,0,-.02],[.02,0,.02],'cylinder','face'),
    ([.02,.02,0],[.02,.02,0],'sphere','edge'),
])
def test_true_cap_cylinder_sphere_and_box_features(start,end,cap_feature,box_feature):
    result = witness(start,end,[.01]*3,.003)
    assert result['feature']['capsule'] == cap_feature
    assert result['feature']['box'] == box_feature
    check_surface(result,start,end,[.01]*3,.003)


@pytest.mark.parametrize('radius,gap',[(.002,.001),(.003,0.),(.004,-.001)])
def test_signed_gap_does_not_add_offsets_or_discard_shallow_penetration(radius,gap):
    start=[.013,0,-.004];end=[.013,0,.004]
    result=witness(start,end,[.01]*3,radius)
    assert result['distance_m'] == pytest.approx(.003,abs=3e-18)
    assert result['gap_m'] == pytest.approx(gap,abs=3e-18)
    check_surface(result,start,end,[.01]*3,radius)


def test_flat_minimum_uses_clipped_interval_midpoint_and_reversal_same_witness():
    start=[.02,0,-.03];end=[.02,0,.01];half=[.01]*3
    result=witness(start,end,half,.003); reverse=witness(end,start,half,.003)
    np.testing.assert_allclose(result['minimizer_interval'],[.5,1.],atol=2e-16)
    assert result['parameter'] == pytest.approx(.75)
    np.testing.assert_allclose(result['axis_point'],[.02,0,0],atol=1e-17)
    np.testing.assert_allclose(result['axis_point'],reverse['axis_point'],atol=1e-17)
    np.testing.assert_allclose(result['box_point'],reverse['box_point'],atol=1e-17)
    assert reverse['parameter'] == pytest.approx(1.-result['parameter'])


def test_small_nonzero_tilt_is_not_a_flat_contact_manifold():
    result=witness([.02,0,-.004],[.02+1e-12,0,.004],[.01]*3,.003)
    assert result['parameter'] == 0. and result['minimizer_interval'] == [0.,0.]
    assert result['feature']['capsule'] == 'start_cap'


@pytest.mark.parametrize('start,end', [
    ([0,0,0],[0,0,0]), ([-.02,0,0],[.02,0,0]),
    ([.01,0,-.004],[.01,0,.004]), ([.02,0,0],[.01,0,0]),
    ([.01+.5e-12,0,0],[.01+.5e-12,0,0]),
])
def test_axis_in_box_touching_or_too_close_fails_without_normal(start,end):
    with pytest.raises(ValueError,match='distance zero or numerically unresolved'):
        witness(start,end,[.01]*3,.003)


def test_positive_distance_beyond_fixed_numerical_floor_is_not_changed():
    start=[.01+2e-12,0,0]
    result=witness(start,start,[.01]*3,.003)
    assert result['distance_m'] > MIN_AXIS_BOX_DISTANCE_M
    assert result['gap_m'] == result['distance_m']-.003
    assert result['numerical_distance_floor_m'] == MIN_AXIS_BOX_DISTANCE_M


@pytest.mark.parametrize('field,value', [
    ('start',[True,0.,0.]),('start',[float('nan'),0,0]),('end',[0,float('inf'),0]),
    ('start',['.02','0','0']),('start',[.02+1j,0,0]),('start',[[.02,0,0]]),
    ('end',[0,0]),('half',[.01,0,.01]),('half',[-.01,.01,.01]),
    ('half',[.01,False,.01]),('half',[float('inf'),.01,.01]),
    ('radius',True),('radius',np.bool_(False)),('radius',0.),('radius',-.01),
    ('radius',float('nan')),('radius',float('inf')),('radius','0.003'),
    ('radius',[.003]),('radius',1+1j),
])
def test_invalid_or_ambiguous_numeric_input(field,value):
    args=dict(start=[.02,0,0],end=[.03,0,0],half=[.01]*3,radius=.003)
    args[field]=value
    with pytest.raises(ValueError):witness(args['start'],args['end'],args['half'],args['radius'])


def test_overflow_or_unresolved_extents_fails_closed():
    with pytest.raises(ValueError):witness([1e308]*3,[1e308]*3,[.01]*3,.003)
    with pytest.raises(ValueError):witness([1e308,0,0],[1e308,0,0],[1e-320]*3,.003)


def test_copied_outputs_no_input_edits_or_aliases():
    start=np.array([.02,0.,0.]);end=np.array([.03,0.,0.]);half=np.array([.01]*3)
    originals=[v.copy() for v in (start,end,half)]
    result=witness(start,end,half,np.float64(.003));result['axis_point'][0]=999.
    for a,b in zip((start,end,half),originals):np.testing.assert_array_equal(a,b)
    assert witness(start,end,half,.003)['axis_point'][0] == .02


def test_existing_exact_distance_dense_samples_and_end_reversal():
    rng=np.random.default_rng(5811);tested=0
    for _ in range(180):
        half=rng.uniform(.002,.02,3);start=rng.uniform(-.05,.05,3);end=rng.uniform(-.05,.05,3)
        radius=.0017
        expected=_segment_aabb_distance(start,end,half)
        if expected <= 1e-10:continue
        result=witness(start,end,half,radius);reverse=witness(end,start,half,radius)
        assert result['distance_m'] == pytest.approx(expected,rel=2e-12,abs=2e-15)
        np.testing.assert_allclose(result['axis_point'],reverse['axis_point'],atol=3e-14,rtol=0)
        dense=start+np.linspace(0,1,20001)[:,None]*(end-start)
        distances=np.linalg.norm(dense-np.clip(dense,-half,half),axis=1)
        assert result['distance_m'] <= float(distances.min())+2e-15
        assert float(distances.min())-result['distance_m'] <= np.linalg.norm(end-start)/20000+2e-15
        check_surface(result,start,end,half,radius);tested+=1
    assert tested >= 100


def _rotation(axis,angle):
    axis=np.array(axis,dtype=float,copy=True);axis/=np.linalg.norm(axis)
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)


@pytest.mark.parametrize('start,end', [([.02,.003,.002],[.03,.009,.005]),
                                       ([.03,0,.003],[0,.03,.007])])
def test_rigid_motion_gap_finite_difference_uses_both_actual_witness_velocities(start,end):
    start=np.asarray(start);end=np.asarray(end);half=np.array([.01]*3);radius=.003
    result=witness(start,end,half,radius)
    n=np.array(result['normal']);a=np.array(result['capsule_point']);b=np.array(result['box_point'])
    vA=np.array([.013,-.007,.004]);wA=np.array([.4,.2,-.3]);cA=np.array([.012,.013,-.003])
    vB=np.array([-.005,.004,.006]);wB=np.array([-.2,.35,.1]);cB=np.array([-.01,.002,.004])
    analytical=n@(vA+np.cross(wA,a-cA)-vB-np.cross(wB,b-cB))
    def gap_at(h):
        RA=_rotation(wA,np.linalg.norm(wA)*h);RB=_rotation(wB,np.linalg.norm(wB)*h)
        endpoints=[cA+RA@(p-cA)+h*vA for p in (start,end)]
        origin=cB+RB@(-cB)+h*vB
        local=[RB.T@(p-origin) for p in endpoints]
        return witness(*local,half,radius)['gap_m']
    numerical=(gap_at(1e-6)-gap_at(-1e-6))/(2e-6)
    assert numerical == pytest.approx(analytical,rel=2e-7,abs=2e-10)


def test_external_world_to_box_transform_preserves_gap_points_and_normal():
    start=np.array([.03,0,.003]);end=np.array([0,.03,.007]);half=np.array([.01]*3)
    original=witness(start,end,half,.003)
    R=_rotation([1,2,-1],.73);translation=np.array([.2,-.8,1.4])
    world=[R@p+translation for p in (start,end)]
    local=[R.T@(p-translation) for p in world]
    actual=witness(*local,half,.003)
    assert actual['gap_m'] == pytest.approx(original['gap_m'],abs=3e-16)
    for key in ('axis_point','box_point','capsule_point'):
        np.testing.assert_allclose(R@actual[key]+translation,R@original[key]+translation,atol=3e-16,rtol=0)
    np.testing.assert_allclose(R@actual['normal'],R@original['normal'],atol=3e-14)
