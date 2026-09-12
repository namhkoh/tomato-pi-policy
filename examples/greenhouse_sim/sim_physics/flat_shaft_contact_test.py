import numpy as np
import pytest
from pxr import Sdf,UsdGeom
from sim_physics.shaft_grasp import ShaftCylinder,ShaftGraspEvidence
from sim_physics.shaft_grasp_test import setup,contact,result
from sim_physics.shaft_grasp_native_test import fixture,bind
from sim_physics.flat_shaft_contact import current_side_witness
from sim_physics.plant_test import native
from sim_physics.plant import build
from sim_physics.benchmark import main


def cylinder_setup():
    old,frames,links=setup(allow_signed_native_normals=True)
    chain=[ShaftCylinder(s.body,s.collider,s.local_frame,s.radius_m,s.half_height_m,s.contact_offset_m) for s in old.chain]
    e=ShaftGraspEvidence(chain,old.pads,selected_index=2,cut_index=1,
        source_target=old.source_target,allow_signed_native_normals=True)
    e.begin_step(1)
    return e,frames,links


def test_cylinder_side_contact_is_signed_and_current():
    e,frames,links=cylinder_setup();contact(e,0);contact(e,1)
    r=result(e,frames,links,contact_body_frames=frames,contact_frames_step_id=0)
    assert r['bilateral'] and r['compressive_support_n']==pytest.approx([.03,.03])
    assert all(p['model']=='current_flat_cylinder_material_side_witness_v1' for p in r['current_contact_proximity'])


@pytest.mark.parametrize('mode',['withdraw','through_pad','axial_slide','penetration'])
def test_old_cylinder_contact_cannot_verify_current_hold(mode):
    e,pre,links=cylinder_setup();contact(e,0);contact(e,1)
    post={p:m.copy() for p,m in pre.items()}
    if mode=='withdraw':post[e.pads[0].body][0,3]-=.005
    elif mode=='through_pad':post[e.pads[0].body][0,3]+=.009
    elif mode=='axial_slide':post[e.pads[0].body][2,3]+=.06
    else:post[e.pads[0].body][0,3]+=.0015
    r=result(e,post,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert r['stem_only'] and not r['bilateral']


def test_round_cap_extension_is_not_flat_cylinder_material():
    e,pre,links=cylinder_setup()
    for i in range(2):contact(e,i,point=[(-1 if i==0 else 1)*.003,0,.009])
    r=result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert not r['bilateral'] and all(x['reason']=='point_beyond_flat_cylinder_side' for x in r['rejected'])


def test_flat_end_face_is_not_a_side_grasp():
    e,pre,links=cylinder_setup();contact(e,0,point=[0,0,.007],normal=[0,0,1]);contact(e,1)
    r=result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert not r['bilateral']


def test_negative_compliant_load_is_not_positive_hold_evidence():
    e,pre,links=cylinder_setup();contact(e,0,force=-.03);contact(e,1)
    r=result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert r['compressive_support_n'][0]<0 and not r['bilateral']


def test_current_witness_never_uses_rounded_enclosing_tip():
    # Pad lies axially beyond the flat end. An enclosing radius-3 mm capsule
    # would falsely overlap it, but this actual material witness is separated.
    pad=np.eye(4);pad[:3,3]=[-.005,0,.019]
    r=current_side_witness([-.003,0,.007],.003,.007,np.eye(4),pad,
        [.002,.01,.01],0,1,.001)
    assert not r['passed'] and r['tangential_face_gap_m']==pytest.approx(.002)


def test_exact_cylinder_adapter_requires_matching_geometry_and_zero_margin():
    f=fixture();f.rig.stem_contact_model='flat_cylinders_v1'
    with pytest.raises(ValueError):bind(f)
    for path in f.rig.body_paths:
        prim=f.stage.GetPrimAtPath(path+'/StemCollider');prim.SetTypeName('Cylinder')
        prim.CreateAttribute('physxConvexGeometry:margin',Sdf.ValueTypeNames.Float).Set(0.)
    a=bind(f)
    assert all(isinstance(x,ShaftCylinder) for x in a.core.chain)
    digest=a.binding_sha256;a.close()
    f.stage.GetPrimAtPath(f.rig.body_paths[2]+'/StemCollider').GetAttribute('physxConvexGeometry:margin').Set(.001)
    with pytest.raises(ValueError,match='zero geometry margin'):bind(f)
    assert len(digest)==64


def test_full_source_geometry_changes_only_contact_shapes(native):
    stage,record=native;source=stage.GetRootLayer().ExportToString()
    a=build(stage,record,'SubStem_41',root='/World/Control',cut_m=.02,max_segment_m=.02,
        stem_contact_model='continuous_internal_capsules_v1')
    b=build(stage,record,'SubStem_41',root='/World/Flat',cut_m=.02,max_segment_m=.02,
        stem_contact_model='flat_cylinders_v1')
    assert stage.GetRootLayer().ExportToString()==source
    np.testing.assert_array_equal(a.chain_world,b.chain_world)
    for p,q in zip(a.properties,b.properties,strict=True):
        for key in p:np.testing.assert_array_equal(p[key],q[key])
    for i,path in enumerate(b.body_paths):
        shape=UsdGeom.Cylinder(stage.GetPrimAtPath(path+'/StemCollider'))
        assert shape and shape.GetHeightAttr().Get()==pytest.approx(np.linalg.norm(b.chain_world[i+1]-b.chain_world[i]))
        assert shape.GetPrim().GetAttribute('physxConvexGeometry:margin').Get()==0.
    assert not b.cut


@pytest.mark.parametrize('extra',[[],['--isolate-station'],['--isolate-station','--bimanual-hold-control'],
    ['--isolate-station','--bimanual-hold-control','--bimanual-cut','--grasp-contact-frames','pre_solve_pgs_v1','--diagnostic-contact-prediction']])
def test_experiment_cannot_enter_unqualified_cut_or_prediction(tmp_path,extra):
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='HOLD ONLY'):
        main(['--output',str(output),'--stem-contact-model','flat_cylinders_v1',*extra])
    assert not output.exists()
