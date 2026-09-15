import numpy as np
import pytest
from sim_data.procedural_petiole_geometry import CurveSpec,curved_centerline
from sim_data.procedural_petiole_warp import CurveWarp


def straight_warp():
    new=curved_centerline(CurveSpec(.22,.002,bend_normal_rad=0,bend_binormal_rad=0))
    return CurveWarp([[0,0,0],[.2,0,0]],new,[1,2,3],1.2)


def test_straight_warp_exact_affine_including_end_extension():
    warp=straight_warp()
    points=np.array([[-.01,.01,.02],[.07,-.03,.01],[.21,.04,-.02]])
    assert np.allclose(warp.map(points),points*[1.1,1.2,1.2]+[1,2,3],atol=1e-10)
    assert np.allclose(warp.jacobian(points),np.diag([1.1,1.2,1.2]),atol=1e-9)


def test_normals_use_inverse_transpose_not_point_transform():
    warp=straight_warp();p=np.array([[.1,.005,.003]])
    n=np.array([[1,1,0.]])/np.sqrt(2)
    mapped,qa=warp.normals(p,n)
    expected=n[0]/[1.1,1.2,1.2];expected/=np.linalg.norm(expected)
    assert np.allclose(mapped[0],expected,atol=1e-9)
    assert qa['minimum_jacobian_determinant']>0


def test_every_old_centerline_point_maps_to_new_metric_chain():
    new=curved_centerline(CurveSpec(.23,.002,bend_normal_rad=.6,bend_binormal_rad=.3))
    old=np.c_[np.linspace(0,.2,50),np.zeros((50,2))]
    warp=CurveWarp(old,new,[.4,.5,.6])
    samples=np.c_[new['arc']/new['arc'][-1]*.2,np.zeros((len(new['arc']),2))]
    assert np.allclose(warp.map(samples),new['points']+[.4,.5,.6],atol=1e-12)


def test_mesh_and_attachment_share_exact_deformation():
    new=curved_centerline(CurveSpec(.2,.002,bend_normal_rad=.7))
    warp=CurveWarp([[0,0,0],[.2,0,0]],new,[.01,.02,.03])
    attachment=np.array([[.12,.008,-.004]])
    child_mesh=np.vstack((attachment,attachment+[.02,.03,.04]))
    assert np.array_equal(warp.map(attachment)[0],warp.map(child_mesh)[0])
    assert np.allclose(warp.map([[0,0,0]])[0],[.01,.02,.03])


def test_zero_normals_preserved_and_nonzero_normalized():
    warp=straight_warp()
    n,qa=warp.normals([[.1,0,0],[.15,0,0]],[[0,0,0],[0,2,0]])
    assert np.array_equal(n[0],[0,0,0]) and np.allclose(n[1],[0,1,0])
    assert qa['preserved_zero_normals']==1


def test_invalid_scaling_and_points_rejected():
    new=curved_centerline(CurveSpec(.2,.002))
    with pytest.raises(ValueError):CurveWarp([[0,0,0],[0,0,0]],new,[0,0,0])
    with pytest.raises(ValueError):CurveWarp([[0,0,0],[.02,0,0]],new,[0,0,0])
    with pytest.raises(ValueError):straight_warp().map([[float('nan'),0,0]])
