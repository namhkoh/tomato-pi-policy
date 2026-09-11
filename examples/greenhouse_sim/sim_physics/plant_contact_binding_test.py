"""Cached and anonymous-USD export tests; no native physics or source writes."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from types import SimpleNamespace as S

import numpy as np
import pytest

from .plant_contact_binding import capture,deserialize,capture_authored,deserialize_authored,AUTHORED_SCHEMA
from .shaft_grasp import ShaftCapsule,FingerPad,ShaftGraspEvidence
from .shaft_grasp_native import _sensor_contract_record,NATIVE37_SENSOR_CONTRACT,STRICT_SENSOR_CONTRACT


def digest(p):return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class NoQuery:
    def __getattr__(self,name):raise AssertionError('USD/native query forbidden: '+name)


def fixture(signed=False):
    chain=[ShaftCapsule('/T/'+('Support' if i==0 else 'Branch')+'/S'+str(i),
        '/T/'+('Support' if i==0 else 'Branch')+'/S'+str(i)+'/StemCollider',np.eye(4),.003,.01,.0005) for i in range(5)]
    pads=[FingerPad('/R/F'+str(i),'/R/F'+str(i)+'/Pad',np.eye(4),[.002,.01,.03],0,-1,.0005) for i in range(2)]
    core=ShaftGraspEvidence(chain,pads,selected_index=2,cut_index=1,source_target='seed/SubStem_41',allow_signed_native_normals=signed)
    joints=[('/T/Joint'+str(i),NoQuery(),(a.body,b.body)) for i,(a,b) in enumerate(zip(chain[1:],chain[2:]))]
    native=dict(source_target=core.source_target,
        shapes=[dict(body=s.body,collider=s.collider,local_frame=s.local_frame.tolist(),contact_offset_m=s.contact_offset_m,
            geometry=[s.radius_m,s.half_height_m] if isinstance(s,ShaftCapsule) else s.half_extents_m.tolist(),
            face=None if isinstance(s,ShaftCapsule) else [s.face_axis,s.face_sign]) for s in [*chain,*pads]],
        joints=[dict(path=p,body0=ab[0],body1=ab[1]) for p,_,ab in joints],
        selected_body=chain[2].body,selected_collider=chain[2].collider)
    if signed:native.update(allow_signed_native_normals=True,sensor_contract=_sensor_contract_record(NATIVE37_SENSOR_CONTRACT))
    observer=S(core=core,closed=False,binding_error=None,error=None,body_paths=tuple(s.body for s in chain),
        finger_paths=tuple(s.body for s in pads),pad_paths=tuple(s.collider for s in pads),selected_index=2,cut_index=1,
        selected_body=chain[2].body,selected_collider=chain[2].collider,source_target=core.source_target,
        allow_signed_native_normals=signed,sensor_contract=NATIVE37_SENSOR_CONTRACT if signed else STRICT_SENSOR_CONTRACT,
        binding_sha256=digest(native),joints=joints)
    local=[(s.collider,i,'capsule',NoQuery()) for i,s in enumerate(chain)]
    local.append((chain[4].body+'/Leaf/Mesh',4,'hull',NoQuery()))
    return S(stage=NoQuery(),robot=NoQuery(),grasp_observer=observer,
        rig=S(source_target=core.source_target,body_paths=observer.body_paths,cut_index=1),
        body_index=2,grasp_path=chain[2].body,paths=['/R/Palm',*observer.finger_paths],
        held_plant_screen=S(local=local),friction=.5,
        finger_contact_compliance=dict(stiffness_n_m=1000.,damping_n_s_m=.0123,reduced_mass_kg=.00008,calibrated=False,force_based=True))


def load(p,**kwargs):
    return deserialize(p,source_target=kwargs.pop('source_target','seed/SubStem_41'),
        binding_sha256=kwargs.pop('binding_sha256',p['binding_sha256']),**kwargs)


def rehash(p):
    p.pop('sha256',None);p['sha256']=digest(p);return p


@pytest.mark.parametrize('signed',[False,True])
def test_roundtrip_retains_full_chain_exact_parameters_and_unsupported_leaves(signed):
    f=fixture(signed);p=capture(f);r=load(json.loads(json.dumps(p)),sha256=p['sha256'])
    assert r.source_target==f.rig.source_target and r.binding_sha256==f.grasp_observer.binding_sha256
    assert len(r.chain)==5 and len(f.grasp_observer.core.candidates)==3
    assert len(r.plant_collider_paths)==6 and r.unsupported_plant_colliders==(f.held_plant_screen.local[-1][0],)
    for a,b in zip(r.chain+r.pads,f.grasp_observer.core.chain+f.grasp_observer.core.pads):
        assert type(a) is type(b) and a.collider==b.collider
        np.testing.assert_array_equal(a.local_frame,b.local_frame)
    assert dict(r.finger_contact_compliance)==f.finger_contact_compliance and r.finger_friction==.5
    assert p['claims']['physical_contact_combine']=='unknown' and p['claims']['native_geometry_verified'] is False
    assert p['claims']['training_eligible'] is False
    assert r.shafts_by_collider[r.chain[0].collider] is r.chain[0]


def test_export_and_loaded_shapes_are_independent_and_immutable():
    f=fixture();p=capture(f);r=load(p)
    p['chain'][0]['local_frame'][0][3]=999.;f.finger_contact_compliance['stiffness_n_m']=999.
    assert r.chain[0].local_frame[0,3]==0. and r.finger_contact_compliance['stiffness_n_m']==1000.
    with pytest.raises(ValueError):r.chain[0].local_frame[0,3]=1.
    with pytest.raises(ValueError):r.pads[0].half_extents_m[0]=1.
    with pytest.raises(TypeError):r.shafts_by_collider['new']=r.chain[0]
    with pytest.raises(FrozenInstanceError):r.selected_body='/other'


@pytest.mark.parametrize('field',['source_target','binding_sha256','sha256'])
def test_expected_source_or_hash_mismatch(field):
    p=capture(fixture());value='other/Stem' if field=='source_target' else '0'*64
    with pytest.raises(ValueError):load(p,**{field:value})


def test_rehashed_geometry_tamper_still_fails_original_binding_hash():
    p=capture(fixture());p['chain'][0]['radius_m']*=1.1;rehash(p)
    with pytest.raises(ValueError,match='source binding'):load(p)


def test_external_export_pin_covers_material_extras_not_in_original_binding():
    p=capture(fixture());original=p['sha256'];p['finger_friction']=.7;rehash(p)
    with pytest.raises(ValueError,match='Export hash'):load(p,sha256=original)


@pytest.mark.parametrize('fault',['unknown_shape','nonrigid','missing_shaft','extra_kind','wrong_body',
    'duplicate','face','nonfinite','claims','unknown_field','bad_compliance'])
def test_malformed_export_fails_even_with_recomputed_export_checksum(fault):
    p=capture(fixture())
    if fault=='unknown_shape':p['chain'][0]['type']='Sphere'
    elif fault=='nonrigid':p['chain'][0]['local_frame'][0][0]=2.
    elif fault=='missing_shaft':p['plant_colliders'].pop(0)
    elif fault=='extra_kind':p['plant_colliders'][-1]['kind']='unknown'
    elif fault=='wrong_body':p['plant_colliders'][-1]['body_index']=0
    elif fault=='duplicate':p['plant_colliders'].append(p['plant_colliders'][-1])
    elif fault=='face':p['pads'][0]['face_sign']=True
    elif fault=='nonfinite':p['finger_friction']=float('nan')
    elif fault=='claims':p['claims']['native_geometry_verified']=True
    elif fault=='unknown_field':p['native_verified']=True
    else:p['finger_contact_compliance']['damping_n_s_m']=-1.
    if fault=='nonfinite':
        with pytest.raises(ValueError):load(p)
    else:
        rehash(p)
        with pytest.raises(ValueError):load(p)


@pytest.mark.parametrize('fault',['closed','binding_error','core_error','selected','source','chain','pads','type','local'])
def test_capture_rejects_unbound_or_inconsistent_fixture(fault):
    f=fixture();o=f.grasp_observer
    if fault=='closed':o.closed=True
    elif fault=='binding_error':o.binding_error='changed'
    elif fault=='core_error':o.core.error='failed'
    elif fault=='selected':f.grasp_path=o.core.chain[1].body
    elif fault=='source':f.rig.source_target='other/Stem'
    elif fault=='chain':o.core.chain=o.core.chain[1:]
    elif fault=='pads':o.pad_paths=o.pad_paths[::-1]
    elif fault=='type':o.core.chain=(S(body=o.core.chain[0].body),*o.core.chain[1:])
    else:f.held_plant_screen.local.append(('invalid',0,'box',None))
    with pytest.raises(ValueError):capture(f)


def test_none_compliance_is_preserved_without_inventing_stiffness():
    f=fixture();f.finger_contact_compliance=None
    assert load(capture(f)).finger_contact_compliance is None


@pytest.mark.parametrize('authored',[False,True])
@pytest.mark.parametrize('value',[None,False,1,'true'])
def test_compliance_requires_explicit_true_force_based_contract(authored,value):
    f=authored_fixture() if authored else fixture()
    if value is None:f.finger_contact_compliance.pop('force_based')
    else:f.finger_contact_compliance['force_based']=value
    with pytest.raises(ValueError):(capture_authored if authored else capture)(f)


def test_authored_preserves_force_based_compliance_and_hashes_the_field():
    f=authored_fixture();p=capture_authored(f)
    assert p['finger_contact_compliance']==f.finger_contact_compliance
    decoded=deserialize_authored(p,source_target=f.rig.source_target,sha256=p['sha256'])
    assert decoded.finger_contact_compliance['force_based'] is True
    p['finger_contact_compliance']['force_based']=False
    with pytest.raises(ValueError,match='Export hash'):
        deserialize_authored(p,source_target=f.rig.source_target)


@pytest.mark.parametrize('fault',['mixed_bool_geometry','bool_claim_alias','bool_sensor_alias'])
def test_boolean_numeric_aliases_are_not_geometry_or_claims(fault):
    p=capture(fixture(signed=True))
    if fault=='mixed_bool_geometry':p['chain'][0]['local_frame'][0][0]=True
    elif fault=='bool_claim_alias':p['claims']['native_geometry_verified']=0
    else:p['native_binding_extras']['allow_signed_native_normals']=1
    rehash(p)
    with pytest.raises(ValueError):load(p)


def authored_fixture():
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,'/T')
    f=fixture();f.stage=stage;f.grasp_observer=NoQuery()
    def collision(prim):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0005)
        prim.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
    for body_path in f.rig.body_paths:
        body=UsdGeom.Xform.Define(stage,body_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        body.CreateAttribute('tomato:sourceTarget',Sdf.ValueTypeNames.String).Set(f.rig.source_target)
        cap=UsdGeom.Capsule.Define(stage,body_path+'/StemCollider')
        cap.CreateAxisAttr('Z');cap.CreateRadiusAttr(.003);cap.CreateHeightAttr(.02);collision(cap.GetPrim())
    for i,(a,b) in enumerate(zip(f.rig.body_paths[1:],f.rig.body_paths[2:])):
        joint=UsdPhysics.SphericalJoint.Define(stage,'/T/Joint'+str(i))
        joint.CreateBody0Rel().SetTargets([a]);joint.CreateBody1Rel().SetTargets([b])
    leaf=UsdGeom.Mesh.Define(stage,f.held_plant_screen.local[-1][0]);collision(leaf.GetPrim())
    leaf.CreatePointsAttr([(0,0,0),(.01,0,0),(0,.01,0),(0,0,.01)])
    leaf.CreateFaceVertexCountsAttr([3,3,3,3]);leaf.CreateFaceVertexIndicesAttr([0,1,2,0,1,3,0,2,3,1,2,3])
    UsdPhysics.MeshCollisionAPI.Apply(leaf.GetPrim()).CreateApproximationAttr('convexHull')
    for i,path in enumerate(f.paths[1:]):
        body=UsdGeom.Xform.Define(stage,path).GetPrim();UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        if i==1:UsdGeom.Xformable(body).AddRotateZOp().Set(180.)
        cube=UsdGeom.Cube.Define(stage,path+'/restored_collisions/contact_proxy');cube.CreateSizeAttr(1.);collision(cube.GetPrim())
        transform=np.diag([.016,.032,.062,1.]);transform[:3,3]=[.005,0,-.0295]
        UsdGeom.Xformable(cube).AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
    f.rig.root='/T'
    return f


def test_authored_stage_read_is_distinct_no_observers_no_native_or_source_writes():
    f=authored_fixture();before=f.stage.GetRootLayer().ExportToString()
    p=capture_authored(f);r=deserialize_authored(json.loads(json.dumps(p)),source_target=f.rig.source_target,sha256=p['sha256'])
    assert f.stage.GetRootLayer().ExportToString()==before
    assert p['schema']==AUTHORED_SCHEMA and p['binding_sha256'] is None and r.binding_sha256 is None
    assert len(r.chain)==5 and len(r.plant_collider_paths)==6 and len(r.unsupported_plant_colliders)==1
    assert [(s.face_axis,s.face_sign) for s in r.pads]==[(0,-1),(0,-1)]
    for pad in r.pads:
        np.testing.assert_array_equal(pad.half_extents_m,[.008,.016,.031])
        np.testing.assert_array_equal(pad.local_frame[:3,3],[.005,0,-.0295])
    with pytest.raises(ValueError,match='schema'):deserialize(p,source_target=f.rig.source_target,binding_sha256='0'*64)
    bound=capture(fixture())
    with pytest.raises(ValueError,match='schema'):deserialize_authored(bound,source_target=f.rig.source_target)


@pytest.mark.parametrize('fault',['units','axis','disabled','extra_pad','source','inventory','scale','extra_shape','native_receipt'])
def test_authored_rejects_unknown_or_inconsistent_geometry(fault):
    from pxr import UsdGeom,UsdPhysics
    f=authored_fixture();cap=UsdGeom.Capsule.Get(f.stage,f.rig.body_paths[0]+'/StemCollider')
    if fault=='units':UsdGeom.SetStageMetersPerUnit(f.stage,.01)
    elif fault=='axis':cap.CreateAxisAttr('X')
    elif fault=='disabled':UsdPhysics.CollisionAPI(cap).CreateCollisionEnabledAttr(False)
    elif fault=='extra_pad':
        extra=UsdGeom.Cube.Define(f.stage,f.paths[1]+'/Extra');UsdPhysics.CollisionAPI.Apply(extra.GetPrim())
    elif fault=='source':f.rig.source_target='other/Stem'
    elif fault=='inventory':f.held_plant_screen.local.pop()
    elif fault=='scale':UsdGeom.Xformable(cap).AddScaleOp().Set((1,2,1))
    elif fault=='extra_shape':
        extra=UsdGeom.Cube.Define(f.stage,f.rig.body_paths[-1]+'/Other');UsdPhysics.CollisionAPI.Apply(extra.GetPrim())
    else:
        p=capture_authored(f);p['binding_sha256']='0'*64;rehash(p)
        with pytest.raises(ValueError,match='native binding'):deserialize_authored(p,source_target=f.rig.source_target)
        return
    with pytest.raises(ValueError):capture_authored(f)
