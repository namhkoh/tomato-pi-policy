"""Boundary equivalence for arithmetic-only contact/evidence optimizations."""
import numpy as np
import pytest
from .shaft_grasp_test import setup,contact
from .grasp_dynamics_evidence import _frames


@pytest.mark.parametrize('length',[
    0.,1.,1.-1e-4,1.+1e-4,
    np.nextafter(1.-1e-4,0.),np.nextafter(1.-1e-4,1.),
    np.nextafter(1.+1e-4,1.),np.nextafter(1.+1e-4,2.),
    float('nan'),float('inf'),1e308])
def test_normal_tolerance_is_exactly_the_previous_rtol_zero_test(length):
    e,_,_=setup()
    with np.errstate(over='ignore',invalid='ignore'):
        expected=bool(np.isclose(np.linalg.norm([length,0,0]),1.,atol=1e-4,rtol=0))
        if expected:contact(e,0,normal=[-length,0,0])
        else:
            with pytest.raises(ValueError):contact(e,0,normal=[-length,0,0])


def test_rigid_evidence_boundaries_match_previous_allclose_without_repair():
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(323)
    for scale in (0.,1e-8,1e-7,1e-6,1e-5,1e-4,1.,1e200):
        value=np.repeat(np.eye(4)[None],20,axis=0)
        value[:,:3,:3]=Rotation.random(20,random_state=rng).as_matrix()
        value+=rng.uniform(-scale,scale,value.shape)
        r=value[:,:3,:3]
        with np.errstate(over='ignore',invalid='ignore'):
            expected=(np.allclose(value[:,3],[0,0,0,1],atol=1e-7,rtol=0)
                and np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-5,rtol=0)
                and np.allclose(np.linalg.det(r),1.,atol=1e-5,rtol=0))
            if expected:np.testing.assert_array_equal(_frames(value,20,'test'),value)
            else:
                with pytest.raises(ValueError):_frames(value,20,'test')


def test_material_point_allocation_retains_exact_witness_values():
    from .flat_shaft_contact import _current_side_witness_validated
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(324)
    for _ in range(500):
        q=rng.uniform(-.01,.01,3);radius=.003;height=.008;half=np.array([.002,.01,.05])
        sw,pw=np.eye(4),np.eye(4)
        sw[:3,:3],pw[:3,:3]=Rotation.random(2,random_state=rng).as_matrix()
        sw[:3,3],pw[:3,3]=rng.uniform(-.01,.01,(2,3))
        axis=int(rng.integers(0,3));sign=int(rng.choice([-1,1]));offset=.001
        # Original np.r_ construction and arithmetic, before allocation change.
        material=np.r_[q[:2]*(radius/float(np.linalg.norm(q[:2]))),np.clip(q[2],-height,height)]
        local=(sw[:3,:3]@material+sw[:3,3]-pw[:3,3])@pw[:3,:3]
        face=np.clip(local,-half,half);face[axis]=sign*half[axis]
        tangent=local-face;tangent[axis]=0
        centre=sw[:3,:3]@np.array([0.,0.,material[2]])+sw[:3,3]
        inner=float(sign*(((centre-pw[:3,3])@pw[:3,:3])[axis]-face[axis]))
        result=_current_side_witness_validated(q,radius,height,sw,pw,half,axis,sign,offset)
        assert result['surface_gap_m']==float(sign*(local[axis]-face[axis]))
        assert result['tangential_face_gap_m']==float(np.linalg.norm(tangent))
        assert result['minimum_spine_distance_on_inner_side_m']==inner
