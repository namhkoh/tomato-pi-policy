"""Rigid blade preservation, metadata and exact serialized source channels."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pytest
from sim_data.procedural_petiole_geometry import CurveSpec, curved_centerline
from sim_data.procedural_petiole_warp import CurveWarp
from sim_data.procedural_leaf_transport import RigidLeafTransport, component_transport
from sim_data.procedural_petiole_usd import transform_metadata, deform_new_copy
from sim_data.procedural_petiole_catalogue import check_component
from sim_data.plant_variant_usd import copy_component
from sim_data import procedural_petiole_usd_test as fixtures

Usd = fixtures.Usd


def warp():
    curve=curved_centerline(CurveSpec(.11,.002,bend_normal_rad=.7),[1,0,0])
    return CurveWarp([[0,0,0],[.1,0,0]],curve,[.01,0,.02],1.1)


def test_anchor_moves_but_all_leaf_distances_and_orientation_are_preserved():
    field=warp();anchor=np.array([.065,.002,0]);rigid=RigidLeafTransport(field,anchor)
    points=np.array([anchor,anchor+[.15,.12,-.08],anchor+[-.06,.3,.12]])
    moved=rigid.map(points)
    np.testing.assert_allclose(moved[0],field.map(anchor[None,:])[0])
    np.testing.assert_allclose(np.linalg.norm(moved[:,None]-moved[None,:],axis=2),
                               np.linalg.norm(points[:,None]-points[None,:],axis=2),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rigid.jacobian(points)),1,atol=1e-12)
    normals,qa=rigid.normals(points,[[0,0,1],[0,0,0],[1,0,0]])
    np.testing.assert_allclose(np.linalg.norm(normals,axis=1),[1,0,1],atol=1e-12)
    assert qa['leaf_shape_preserved'] and qa['preserved_zero_normals']==1


def test_rigid_map_has_analytic_jacobian_and_preserves_handedness():
    rigid=RigidLeafTransport(warp(),[.06,.002,0]);p=np.array([[.2,.3,.1]])
    numerical=np.stack([(rigid.map(p+e*1e-6)-rigid.map(p-e*1e-6))/(2e-6) for e in np.eye(3)],axis=2)
    np.testing.assert_allclose(numerical,rigid.jacobian(p),atol=1e-9)
    assert np.linalg.det(rigid.rotation)>0


@pytest.mark.parametrize('bad',[np.diag([-1,1,1]),np.diag([.001,1,1]),np.diag([100,1,1])])
def test_folded_or_ill_conditioned_attachment_is_still_rejected(bad):
    field=warp()
    with patch.object(field,'jacobian',return_value=bad[None,:]),pytest.raises(ValueError):
        RigidLeafTransport(field,[.06,.002,0])


@pytest.mark.parametrize('bad',[[1,2],[np.nan,0,0],[[0,0,0]]])
def test_invalid_attachment(bad):
    with pytest.raises(ValueError):RigidLeafTransport(warp(),bad)


def test_component_dispatch_and_nested_rejection():
    field=warp();change=dict(component_id='Petiole',warp=field)
    assert component_transport(dict(id='Petiole'),change) is field
    assert isinstance(component_transport(dict(id='Leaf',type='leaf',parent='Petiole',attach_point=[.06,.002,0]),change),RigidLeafTransport)
    for row in (dict(id='Fruit',type='fruit',parent='Petiole'),dict(id='Leaf',type='leaf',parent='Other')):
        with pytest.raises(ValueError):component_transport(row,change)


def test_leaf_metadata_uses_same_rigid_map_and_keeps_capsule_radius():
    field=warp();rigid=RigidLeafTransport(field,[.06,.002,0])
    raw=dict(id='Leaf',type='leaf',parent='Petiole',attach_point=[.06,.002,0],
             transform={'translate':[.06,.002,0]},axis=[1,0,0],
             capsules=[[[0,0,0,.003],[.05,0,0,.002]]],length=.05,radius=.003)
    row=transform_metadata(raw,dict(component_id='Petiole',warp=rigid))
    np.testing.assert_allclose(row['attach_point'],rigid.new_anchor)
    np.testing.assert_allclose(np.array(row['capsules'][0])[:,3],[.003,.002])
    assert row['length']==raw['length'] and row['radius']==raw['radius']
    np.testing.assert_allclose(np.array(row['capsules'][0])[:,:3]+row['transform']['translate'],
        rigid.map(np.array(raw['capsules'][0])[:,:3]+raw['transform']['translate']))


@pytest.mark.skipif(Usd is None,reason='USD unavailable')
@pytest.mark.parametrize('mode',['faceVarying','vertex','varying'])
def test_saved_leaf_matches_rigid_replay_and_preserves_uvs(mode):
    with TemporaryDirectory() as tmp:
        root=Path(tmp);fixtures.SerializedTests().triangle(root/'source.usda',mode)
        before=(root/'source.usda').read_bytes()
        copy_component(root/'source.usda',root/'copy.usda',np.eye(3),1,root,{})
        rigid=RigidLeafTransport(warp(),[.06,.002,0]);old=np.array([.06,.002,0]);new=rigid.map(old[None,:])[0]
        deform_new_copy(root/'copy.usda',old,new,rigid)
        qa=check_component(root/'source.usda',root/'copy.usda',old,new,rigid)
        assert all(q['leaf_shape_preserved'] and q['serialized_geometry_checked'] for q in qa)
        assert (root/'source.usda').read_bytes()==before
