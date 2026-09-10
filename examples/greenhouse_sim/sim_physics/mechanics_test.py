import numpy as np
import pytest
from sim_physics.mechanics import Material,beam_properties,resample_chain,segment_frames,SkinBinding,clip_mesh,lamina_mass_properties,combine_mass_properties


def test_beam_units_no_arbitrary_stiffness_or_inertia_floor():
    m=Material(); p=beam_properties(.002,.025,m)
    assert p['mass']==pytest.approx(950*np.pi*.002**2*.025)
    assert p['stiffness'][0]==pytest.approx(1.5e8*np.pi*.002**4/(4*.025))
    assert p['usd_stiffness'][0]==pytest.approx(p['stiffness'][0]*np.pi/180)
    assert p['inertia'].max()<1e-5
    loaded=beam_properties(.002,.025,m,supported_inertia=1e-5)
    assert np.all(loaded['damping']>p['damping'])


def test_lamina_integral_matches_rectangular_plate():
    vertices=np.array([[-.02,-.01,0],[.02,-.01,0],[.02,.01,0],[-.02,.01,0]])+[.1,.02,0]
    mass,center,inertia=lamina_mass_properties(vertices,[[0,1,2],[0,2,3]],.0005)
    assert np.allclose(center,[.1,.02,0])
    assert np.allclose(inertia,np.diag([mass*.02**2/12,mass*.04**2/12,mass*(.02**2+.04**2)/12]),atol=1e-15)


def test_combined_leaf_mass_has_correct_offset_com_and_principal_inertia():
    amount=.0005;rod=(amount,np.zeros(3),np.eye(3)*1e-8)
    leaf=(amount,np.array([.1,0,0]),np.diag([1e-8,2e-8,3e-8]))
    props=combine_mass_properties([rod,leaf])
    assert props['mass']==.001 and np.allclose(props['center'],[.05,0,0])
    axes=props['principal_axes']
    assert np.linalg.det(axes)==pytest.approx(1)
    assert np.allclose(axes@np.diag(props['inertia'])@axes.T,np.diag([2e-8,2.53e-6,2.54e-6]))


def test_exact_cut_knot_preserves_source_bends():
    original=[[0,0,0,.003],[.028,0,0,.002],[.05,.02,0,.001]]
    points,arcs=resample_chain(original)
    assert np.any(arcs==.01) and np.max(np.diff(arcs))<=.025+1e-10
    assert any(np.allclose(p,original[1]) for p in points)
    with pytest.raises(ValueError): resample_chain(original,cut_m=.2)


def test_skin_is_identity_at_rest_and_no_cut_cross_talk():
    chain=np.array([[0.,0,0],[.01,0,0],[.04,0,0],[.08,0,0]])
    rest=segment_frames(chain);points=np.array([[.01,.001,0],[.035,.001,0],[.07,0,0]])
    skin=SkinBinding(points,rest,chain,first_segment=1)
    assert np.allclose(skin.deform(rest),points)
    moved=rest.copy();moved[0,:3,3]+=[0,10,0]
    assert np.allclose(skin.deform(moved),points)
    moved=rest.copy();moved[1:,:3,3]+=[0,.2,0]
    assert np.allclose(skin.deform(moved),points+[0,.2,0])


def test_skin_responds_to_internal_bending_not_just_first_body():
    chain=np.array([[0.,0,0],[.01,0,0],[.04,0,0],[.08,0,0]])
    rest=segment_frames(chain);points=np.array([[.011,0,0],[.079,0,0]])
    skin=SkinBinding(points,rest,chain,first_segment=1)
    moved=rest.copy();moved[2,1,3]+=.02
    displacement=skin.deform(moved)-points
    assert displacement[1,1]>.019 and displacement[0,1]==0


def test_cut_geometry_and_texture_interpolation_share_seam():
    points=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
    uv=np.array([[0.,0],[1,0],[1,1],[0,1]])
    for positive in (True,False):
        xyz,tex=clip_mesh(points,[4],[0,1,2,3],np.zeros(3),np.array([1,0,0]),positive=positive,uv=uv)
        assert len(xyz)==6 and np.all(xyz[:,0]*(1 if positive else -1)>=0)
        assert np.allclose(tex[np.isclose(xyz[:,0],0),0],.5)


@pytest.mark.parametrize('params',[dict(youngs_modulus_pa=0),dict(leaf_mass_kg=-1),dict(poisson_ratio=.5)])
def test_invalid_material_rejected(params):
    with pytest.raises(ValueError): Material(**params)
