import itertools
import numpy as np
import pytest
from pxr import Sdf,Usd,UsdGeom
from sim_physics.stem_envelope import capsule_span
from sim_physics.plant_test import native
from sim_physics.plant import build
from sim_physics.capsule_surface import world_capsule


@pytest.mark.parametrize('start,end', itertools.product((False,True),repeat=2))
def test_span_ends_reach_internal_anchors_but_never_cross_cut(start,end):
    length=.020;radius=.003
    span=capsule_span(length,radius,flush_start=start,flush_end=end)
    low=span['center_z_m']-span['height_m']/2
    high=span['center_z_m']+span['height_m']/2
    assert low==pytest.approx(-length/2+(radius if start else 0))
    assert high==pytest.approx(length/2-(radius if end else 0))
    if start:assert low-radius==pytest.approx(-length/2)
    if end:assert high+radius==pytest.approx(length/2)


def test_old_tip_meeting_capsules_have_a_neck_new_internal_envelope_does_not():
    radius=.003;length=.020
    point=np.array([.0025,0.,length/2])
    # This point belongs to the physical circular tube, but old caps miss it.
    old_top=np.array([0.,0.,length/2-radius])
    new_top=np.array([0.,0.,length/2])
    assert np.linalg.norm(point-old_top)>radius
    assert np.linalg.norm(point-new_top)<radius


@pytest.mark.parametrize('length,radius', [(0,.003),(.02,-.1),(float('nan'),.003),(.005,.003)])
def test_invalid_endpoint_shape_is_not_silently_thinned(length,radius):
    with pytest.raises(ValueError):capsule_span(length,radius,flush_start=True,flush_end=True)


def test_full_source_chain_has_joint_cross_sections_and_separate_cut_sides(native):
    stage,record=native;before=stage.GetRootLayer().ExportToString()
    rig=build(stage,record,'SubStem_41',stem_contact_model='continuous_internal_capsules_v1')
    assert stage.GetRootLayer().ExportToString()==before
    cache=UsdGeom.XformCache();caps=[]
    for path in rig.body_paths:
        prim=stage.GetPrimAtPath(path+'/StemCollider')
        caps.append(world_capsule(prim,np.asarray(cache.GetLocalToWorldTransform(prim)).T))
        assert UsdGeom.Capsule(prim).GetExtentAttr().HasAuthoredValueOpinion()
    for i in range(1,len(caps)):
        if i==rig.cut_index:continue
        np.testing.assert_allclose(caps[i-1][1],rig.chain_world[i],atol=1e-8)
        np.testing.assert_allclose(caps[i][0],rig.chain_world[i],atol=1e-8)
    axis=rig.rest_frames[rig.cut_index,:3,2];seam=rig.chain_world[rig.cut_index]
    before_end=caps[rig.cut_index-1][1]+caps[rig.cut_index-1][2]*axis
    after_start=caps[rig.cut_index][0]-caps[rig.cut_index][2]*axis
    np.testing.assert_allclose(before_end,seam,atol=1e-8)
    np.testing.assert_allclose(after_start,seam,atol=1e-8)
    assert rig.report()['stem_contact_model']=='continuous_internal_capsules_v1'


def test_contact_envelope_does_not_change_mass_material_topology_or_source_art(native):
    stage,record=native;before=stage.GetRootLayer().ExportToString()
    session=Sdf.Layer.CreateAnonymous()
    session.TransferContent(stage.GetSessionLayer())
    other=Usd.Stage.Open(stage.GetRootLayer(),session)
    original=build(stage,record,'SubStem_41')
    revised=build(other,record,'SubStem_41',stem_contact_model='continuous_internal_capsules_v1')
    assert original.body_paths==revised.body_paths
    assert original.cut_index==revised.cut_index
    for key in ('rest_frames','chain_world','arcs'):
        np.testing.assert_array_equal(getattr(original,key),getattr(revised,key))
    for a,b in zip(original.properties,revised.properties,strict=True):
        assert a.keys()==b.keys()
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    # All authored physical/visual properties outside the intended capsules
    # stay identical, including joint limits, K/C, friction and mass tensors.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(original.root)):
        if str(prim.GetPath()).endswith('/StemCollider'):continue
        peer=other.GetPrimAtPath(prim.GetPath())
        assert peer and prim.GetAppliedSchemas()==peer.GetAppliedSchemas()
        assert {a.GetName():str(a.Get()) for a in prim.GetAuthoredAttributes()}=={
            a.GetName():str(a.Get()) for a in peer.GetAuthoredAttributes()}
    assert stage.GetRootLayer().ExportToString()==before
    assert other.GetRootLayer().ExportToString()==before


def test_invalid_model_rejected_before_any_session_edit(native):
    stage,record=native;before=stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):build(stage,record,'SubStem_41',stem_contact_model='unknown')
    assert stage.GetSessionLayer().ExportToString()==before
