import copy
import numpy as np
import pytest
from sim_data.procedural_petiole_geometry import (CurveSpec, curved_centerline, transport_frames,
    sample_curve, sweep_mesh, mesh_qa, intrinsic_descriptor, rotation_between)


@pytest.mark.parametrize('length,bend',[(.06,-.65),(.12,0.),(.25,.6),(.38,.85)])
def test_exact_metric_curve_and_recomputed_cut(length,bend):
    c = curved_centerline(CurveSpec(length,.002,bend_normal_rad=bend))
    assert np.linalg.norm(np.diff(c['points'],axis=0),axis=1).sum() == pytest.approx(length,abs=1e-12)
    assert c['arc'][-1] == length
    assert sample_curve(c,.01)['arc_m'] == .01
    assert np.isfinite(intrinsic_descriptor(c)).all()


@pytest.mark.parametrize('direction',[(1,0,0),(-1,0,0),(0,0,1),(.1,.7,-.8)])
def test_parallel_transport_right_handed_no_roll_flip(direction):
    c = curved_centerline(CurveSpec(.25,.002),direction)
    frames = transport_frames(c['points'])
    assert np.allclose(np.swapaxes(frames,1,2)@frames,np.eye(3),atol=1e-10)
    assert np.allclose(np.linalg.det(frames),1)
    assert (np.einsum('ij,ij->i',frames[:-1,:,1],frames[1:,:,1]) > .99).all()


@pytest.mark.parametrize('radius,ratio,bend',[(.001,.3,-.5),(.002,.6,.8),(.0035,1.,0.)])
def test_surface_closed_outward_and_smooth(radius,ratio,bend):
    c = curved_centerline(CurveSpec(.2,radius,tip_radius_ratio=ratio,bend_normal_rad=bend))
    m = sweep_mesh(c)
    qa = mesh_qa(m)
    assert qa['passed'] and not qa['physics_validated']
    assert np.allclose(np.linalg.norm(m['normals'],axis=1),1.)
    assert m['extent'].shape == (2,3)


def test_rigid_and_scale_changes_do_not_create_intrinsic_novelty():
    c = curved_centerline(CurveSpec(.2,.002))
    b = copy.deepcopy(c)
    r = rotation_between([1,0,0],[0,1,0])
    b['points'] = c['points']@r.T*1.3+np.array([.8,-.7,2.])
    b['radius'] *= 1.3
    b['arc'] *= 1.3
    assert np.allclose(intrinsic_descriptor(c),intrinsic_descriptor(b),atol=1e-12)
    changed = curved_centerline(CurveSpec(.2,.002,bend_normal_rad=-.8,bend_binormal_rad=.5))
    assert np.max(abs(intrinsic_descriptor(c)-intrinsic_descriptor(changed))) > .01


def test_same_seedless_spec_is_exactly_reproducible():
    spec=CurveSpec(.21,.0018)
    a,b=sweep_mesh(curved_centerline(spec)),sweep_mesh(curved_centerline(spec))
    assert all(np.array_equal(a[k],b[k]) for k in a)


@pytest.mark.parametrize('kwargs',[dict(length_m=.01),dict(radius_m=-1),dict(sides=8),
    dict(spacing_m=.01),dict(bend_normal_rad=4),dict(radius_m=float('nan'))])
def test_bad_configuration_rejected(kwargs):
    values=dict(length_m=.2,radius_m=.002);values.update(kwargs)
    with pytest.raises(ValueError): curved_centerline(CurveSpec(**values))


def test_corrupt_winding_and_missing_faces_rejected():
    m=sweep_mesh(curved_centerline(CurveSpec(.1,.002)))
    bad=copy.deepcopy(m);bad['triangles'][0]=bad['triangles'][0][::-1]
    with pytest.raises(ValueError):mesh_qa(bad)
    bad=copy.deepcopy(m);bad['triangles']=bad['triangles'][:-1]
    with pytest.raises(ValueError):mesh_qa(bad)


def test_invalid_arc_and_degenerate_centerline_rejected():
    c=curved_centerline(CurveSpec(.1,.002))
    with pytest.raises(ValueError):sample_curve(c,-.1)
    with pytest.raises(ValueError):sample_curve(c,.101)
    with pytest.raises(ValueError):transport_frames(np.zeros((3,3)))
