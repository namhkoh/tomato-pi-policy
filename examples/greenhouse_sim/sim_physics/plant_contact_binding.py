"""Local geometry exports: cached binding or explicitly CPU-authored USD.

capture() reads cached fields only. capture_authored() reads the authored stage
without constructing native views/observers or claiming a native binding.

The existing shaft-grasp digest binds capsules, pads and cached joint identities.
The export digest ALSO binds the plant collider inventory and copied configured
finger parameters. Pin that digest externally when authenticating those extras;
a self-contained checksum alone is not an authenticity proof. Leaves remain
explicit unsupported geometry. No contact combination, calibration, current
connectivity, pose freshness or native reporting completeness is certified.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from .plant_contact_stream import _path,MAX_PLANT_COLLIDERS
from .shaft_grasp import ShaftCapsule,FingerPad
from .shaft_grasp_native import _sensor_contract_record,NATIVE37_SENSOR_CONTRACT


SCHEMA='plant_contact_binding_v1'
AUTHORED_SCHEMA='plant_contact_authored_v1'
_CLAIMS=dict(geometry_basis='cached_bound_local_shapes_not_current_world_poses',
    physical_contact_combine='unknown',material_calibration='unknown_engineering_only',
    partner_materials_exported=False,native_geometry_verified=False,
    current_connectivity_verified=False,training_eligible=False,actuation_authorized=False)


def _json(value):
    try:return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Finite JSON binding required') from exc


def _hash(value):return hashlib.sha256(_json(value).encode()).hexdigest()


def _digest(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Lowercase SHA256 required')
    return value


def _source(value):
    if not isinstance(value,str) or not 1<=len(value)<=256:raise ValueError('Bounded source target required')
    _path('/'+value)
    return value


def _scalar(value,*,positive=False):
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or value<0 or (positive and value==0):
        raise ValueError('Finite nonnegative physical scalar required')
    return value


def _index(value):
    if isinstance(value,np.integer):value=int(value)
    if type(value) is not int or value<0:raise ValueError('Nonnegative integer index required')
    return value


def _keys(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('Unknown or missing binding fields')


def _array(value,shape):
    raw=np.asarray(value)
    if (raw.shape!=shape or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all()
            or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat)):
        raise ValueError('Finite exact geometry array required')
    return raw


def _pack(shape):
    if type(shape) not in (ShaftCapsule,FingerPad):raise ValueError('Unknown bound shape type')
    result=dict(type=type(shape).__name__,body=_path(shape.body),collider=_path(shape.collider),
        local_frame=_array(shape.local_frame,(4,4)).tolist(),contact_offset_m=_scalar(shape.contact_offset_m))
    if type(shape) is ShaftCapsule:
        result.update(radius_m=_scalar(shape.radius_m,positive=True),half_height_m=_scalar(shape.half_height_m))
    else:
        result.update(half_extents_m=_array(shape.half_extents_m,(3,)).tolist(),
            face_axis=_index(shape.face_axis),face_sign=shape.face_sign)
    return result


def _unpack(row,kind):
    fields={'type','body','collider','local_frame','contact_offset_m'}
    fields|={'radius_m','half_height_m'} if kind is ShaftCapsule else {'half_extents_m','face_axis','face_sign'}
    _keys(row,fields)
    if row['type']!=kind.__name__:raise ValueError('Unknown or misplaced shape type')
    _path(row['body']);_path(row['collider']);_array(row['local_frame'],(4,4))
    _scalar(row['contact_offset_m'])
    if kind is ShaftCapsule:
        _scalar(row['radius_m'],positive=True);_scalar(row['half_height_m'])
    else:
        _array(row['half_extents_m'],(3,));_index(row['face_axis'])
        if type(row['face_sign']) is not int or row['face_sign'] not in (-1,1):raise ValueError('Exact signed pad face required')
    return kind(**{k:v for k,v in row.items() if k!='type'})


def _native_binding(payload,chain,pads):
    # Exactly the existing ShaftGraspNative hash representation, not a new
    # geometry interpretation. No cached USD Joint object is queried.
    shapes=[dict(body=s.body,collider=s.collider,local_frame=s.local_frame.tolist(),
        contact_offset_m=s.contact_offset_m,
        geometry=[s.radius_m,s.half_height_m] if type(s) is ShaftCapsule else s.half_extents_m.tolist(),
        face=None if type(s) is ShaftCapsule else [s.face_axis,s.face_sign]) for s in (*chain,*pads)]
    return dict(source_target=payload['source_target'],shapes=shapes,joints=payload['joints'],
        selected_body=payload['selected_body'],selected_collider=chain[payload['selected_index']].collider,
        **payload['native_binding_extras'])


@dataclass(frozen=True)
class PlantContactBinding:
    source_target: str
    binding_sha256: str | None
    sha256: str
    selected_body: str
    selected_index: int
    cut_index: int
    chain: tuple
    pads: tuple
    plant_collider_paths: tuple
    unsupported_plant_colliders: tuple
    shafts_by_collider: object
    pads_by_collider: object
    finger_contact_compliance: object
    finger_friction: float
    schema: str


def deserialize(payload,*,source_target,binding_sha256,sha256=None):
    """Validate source/digests and return copied existing immutable shape types.

    source_target and binding_sha256 are caller-owned expected provenance.
    Optional sha256 pins the WHOLE export, including inventory/material extras
    not covered by the original shaft-grasp binding. No files are opened.
    """
    return _deserialize(payload,source_target=source_target,binding_sha256=binding_sha256,sha256=sha256,authored=False)


def deserialize_authored(payload,*,source_target,sha256=None):
    """Decode source USD geometry, explicitly NOT a native binding receipt."""
    return _deserialize(payload,source_target=source_target,binding_sha256=None,sha256=sha256,authored=True)


def _deserialize(payload,*,source_target,binding_sha256,sha256,authored):
    _keys(payload,{'schema','source_target','binding_sha256','selected_body','selected_index','cut_index',
        'chain','pads','joints','native_binding_extras','plant_colliders','all_plant_collider_paths',
        'finger_contact_compliance','finger_friction','claims','sha256'})
    p=json.loads(_json(payload))
    claims=dict(_CLAIMS)
    if authored:claims['geometry_basis']='usd_authored_local_shapes_not_native_binding_receipt'
    if p['schema']!=(AUTHORED_SCHEMA if authored else SCHEMA) or _json(p['claims'])!=_json(claims):
        raise ValueError('Unknown export schema or claims')
    if _source(p['source_target'])!=_source(source_target):raise ValueError('Source target mismatch')
    if authored:
        if p['binding_sha256'] is not None or p['native_binding_extras']!={}:
            raise ValueError('Authored geometry must not claim a native binding receipt')
    elif _digest(p['binding_sha256'])!=_digest(binding_sha256):
        raise ValueError('Source or expected binding hash mismatch')
    expected=_digest(p.pop('sha256'))
    if _hash(p)!=expected or (sha256 is not None and _digest(sha256)!=expected):raise ValueError('Export hash mismatch')
    if (not isinstance(p['chain'],list) or not 1<=len(p['chain'])<=MAX_PLANT_COLLIDERS
            or not isinstance(p['pads'],list) or len(p['pads'])!=2):raise ValueError('Bounded full chain and two pads required')
    chain=tuple(_unpack(r,ShaftCapsule) for r in p['chain']);pads=tuple(_unpack(r,FingerPad) for r in p['pads'])
    cut,selected=_index(p['cut_index']),_index(p['selected_index'])
    bodies=[s.body for s in chain];pad_bodies=[s.body for s in pads]
    if (not cut<=selected<len(chain) or p['selected_body']!=bodies[selected]
            or len(set(bodies))!=len(bodies) or len(set(pad_bodies))!=2 or set(bodies)&set(pad_bodies)
            or any(b.rsplit('/',1)[0]!=bodies[selected].rsplit('/',1)[0] for b in bodies[cut:])):
        raise ValueError('Inconsistent complete chain, pad or selected-body inventory')
    extras=p['native_binding_extras']
    allowed=dict(allow_signed_native_normals=True,sensor_contract=_sensor_contract_record(NATIVE37_SENSOR_CONTRACT))
    if _json(extras) not in (_json({}),_json(allowed)):raise ValueError('Unknown source binding metadata')
    if not isinstance(p['joints'],list) or len(p['joints'])!=len(chain)-cut-1:raise ValueError('Incomplete cached joint identities')
    links={frozenset(pair) for pair in zip(bodies[cut:],bodies[cut+1:])};seen=set();joint_paths=set()
    for row in p['joints']:
        _keys(row,{'path','body0','body1'});path=_path(row['path'])
        pair=frozenset((_path(row['body0']),_path(row['body1'])))
        if pair not in links or pair in seen or path in joint_paths:raise ValueError('Unknown or duplicated cached shaft joint')
        seen.add(pair);joint_paths.add(path)
    if not authored and _hash(_native_binding(p,chain,pads))!=p['binding_sha256']:
        raise ValueError('Cached shapes differ from source binding hash')
    inventory=p['plant_colliders']
    if not isinstance(inventory,list) or not 1<=len(inventory)<=MAX_PLANT_COLLIDERS:raise ValueError('Bounded complete plant inventory required')
    paths=[];shaft_paths={s.collider for s in chain}
    for row in inventory:
        _keys(row,{'collider','body_index','kind'})
        path=_path(row['collider']);i=_index(row['body_index'])
        if i>=len(chain) or not path.startswith(bodies[i]+'/') or row['kind'] not in ('capsule','hull'):
            raise ValueError('Unknown cached plant shape or body inventory')
        if path in shaft_paths and (path!=chain[i].collider or row['kind']!='capsule'):
            raise ValueError('Shaft geometry/inventory mismatch')
        paths.append(path)
    if len(set(paths))!=len(paths) or not shaft_paths<=set(paths) or p['all_plant_collider_paths']!=sorted(paths):
        raise ValueError('Missing, duplicate or inconsistent plant collider inventory')
    compliance=p['finger_contact_compliance']
    if compliance is not None:
        _keys(compliance,{'stiffness_n_m','damping_n_s_m','reduced_mass_kg','calibrated','force_based'})
        _scalar(compliance['stiffness_n_m'],positive=True);_scalar(compliance['damping_n_s_m'])
        _scalar(compliance['reduced_mass_kg'],positive=True)
        if compliance['calibrated'] is not False:raise ValueError('Unsupported material calibration claim')
        if compliance['force_based'] is not True:raise ValueError('Explicit force-based compliance required')
    friction=_scalar(p['finger_friction'])
    if friction>1:raise ValueError('Unknown fixture friction range')
    return PlantContactBinding(p['source_target'],p['binding_sha256'],expected,p['selected_body'],selected,cut,
        chain,pads,tuple(sorted(paths)),tuple(sorted(set(paths)-shaft_paths)),
        MappingProxyType({s.collider:s for s in chain}),MappingProxyType({s.collider:s for s in pads}),
        None if compliance is None else MappingProxyType(compliance),friction,p['schema'])


def capture(fixture):
    """Read cached binding fields ONLY; reject stale/partial/unknown fixtures."""
    try:
        observer=fixture.grasp_observer;core=observer.core;rig=fixture.rig
        if (observer.closed is not False or observer.binding_error is not None or observer.error is not None
                or core.error is not None):raise ValueError('Healthy open cached grasp binding required')
        chain,pads=tuple(core.chain),tuple(core.pads)
        if not 1<=len(chain)<=MAX_PLANT_COLLIDERS or len(pads)!=2:raise ValueError('Bounded complete cached shapes required')
        if (tuple(s.body for s in chain)!=tuple(rig.body_paths) or tuple(rig.body_paths)!=tuple(observer.body_paths)
                or tuple(s.body for s in pads)!=tuple(observer.finger_paths)
                or tuple(s.collider for s in pads)!=tuple(observer.pad_paths)
                or tuple(fixture.paths[1:])!=tuple(observer.finger_paths)
                or core.source_target!=rig.source_target or observer.source_target!=rig.source_target
                or core.selected_index!=fixture.body_index or core.selected_index!=observer.selected_index
                or core.cut_index!=rig.cut_index or core.cut_index!=observer.cut_index
                or fixture.grasp_path!=observer.selected_body or observer.selected_collider!=fixture.grasp_path+'/StemCollider'
                or core.allow_signed_native_normals!=observer.allow_signed_native_normals):
            raise ValueError('Fixture, observer and full-chain identities differ')
        extras={}
        if observer.allow_signed_native_normals:
            if observer.sensor_contract!=NATIVE37_SENSOR_CONTRACT:raise ValueError('Unknown signed source binding contract')
            extras=dict(allow_signed_native_normals=True,sensor_contract=_sensor_contract_record(observer.sensor_contract))
        local=fixture.held_plant_screen.local
        if not isinstance(local,(list,tuple)) or not 1<=len(local)<=MAX_PLANT_COLLIDERS:raise ValueError('Bounded cached plant inventory required')
        inventory=[]
        for entry in local:
            if not isinstance(entry,(tuple,list)) or len(entry)!=4:raise ValueError('Unknown cached shape record')
            path,index,kind,_=entry  # Hull geometry remains explicitly unsupported, never reconstructed.
            inventory.append(dict(collider=_path(path),body_index=_index(index),kind=kind))
        joints=[]
        for entry in observer.joints:
            if len(entry)!=3 or len(entry[2])!=2:raise ValueError('Unknown cached joint identity')
            path,_,pair=entry
            joints.append(dict(path=path,body0=pair[0],body1=pair[1]))
        p=dict(schema=SCHEMA,source_target=rig.source_target,binding_sha256=observer.binding_sha256,
            selected_body=observer.selected_body,selected_index=_index(core.selected_index),cut_index=_index(core.cut_index),
            chain=[_pack(s) for s in chain],pads=[_pack(s) for s in pads],joints=joints,native_binding_extras=extras,
            plant_colliders=inventory,all_plant_collider_paths=sorted(row['collider'] for row in inventory),
            finger_contact_compliance=fixture.finger_contact_compliance,finger_friction=_scalar(fixture.friction),claims=dict(_CLAIMS))
        p=json.loads(_json(p));p['sha256']=_hash(p)
        deserialize(p,source_target=rig.source_target,binding_sha256=observer.binding_sha256,sha256=p['sha256'])
        return p
    except (AttributeError,TypeError,KeyError,IndexError) as exc:
        raise ValueError('Incomplete or malformed bound fixture') from exc


def capture_authored(fixture):
    """Read a CPU-authored fixture's stage; no observer, views or native calls.

    Main owns fixture construction, mounting/configuration and comparison with
    the recorded native run. Both actual RBY1 pad faces are local -X, matching
    ShaftGraspNative's default ((0,-1),(0,-1)); finger2's BODY is rotated.
    This exports source-authored shapes, not measured native/cooked geometry.
    """
    from pxr import Usd,UsdGeom,UsdPhysics
    from .shaft_grasp_native import _active,_owner,_local_geometry,_offset,_coverage,_body_world_basis
    try:
        stage,rig=fixture.stage,fixture.rig
        bodies=tuple(rig.body_paths);fingers=tuple(fixture.paths[1:])
        selected,cut=_index(fixture.body_index),_index(rig.cut_index)
        source=_source(rig.source_target)
        if (not 1<=len(bodies)<=MAX_PLANT_COLLIDERS or len(set(bodies))!=len(bodies)
                or len(fingers)!=2 or len(set(fingers))!=2 or not cut<=selected<len(bodies)
                or UsdGeom.GetStageMetersPerUnit(stage)!=1.
                or fixture.grasp_path!=bodies[selected]):raise ValueError('Complete metre-stage fixture identities required')
        chain=[];inventory=[]
        for i,path in enumerate(bodies):
            _path(path);body=stage.GetPrimAtPath(path)
            if not body or body.GetAttribute('tomato:sourceTarget').Get()!=source:
                raise ValueError('Every authored shaft must match source_target')
            _body_world_basis(body,UsdGeom)
            coverage=_coverage(stage,path,Usd,UsdPhysics)
            for collider_path in coverage:
                prim=stage.GetPrimAtPath(collider_path)
                if prim.IsA(UsdGeom.Capsule):kind='capsule'
                elif (prim.IsA(UsdGeom.Mesh)
                        and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='convexHull'):kind='hull'
                else:raise ValueError('Unknown authored plant collision shape: '+collider_path)
                inventory.append(dict(collider=_path(collider_path),body_index=i,kind=kind))
                if len(inventory)>MAX_PLANT_COLLIDERS:raise ValueError('Authored collider inventory exceeds bound')
            collider=stage.GetPrimAtPath(path+'/StemCollider')
            if (not _active(collider,UsdPhysics) or not collider.IsA(UsdGeom.Capsule)
                    or _owner(collider,UsdPhysics)!=path):raise ValueError('Exact enabled authored shaft capsule required')
            cap=UsdGeom.Capsule(collider)
            if cap.GetAxisAttr().Get()!='Z':raise ValueError('Authored shaft capsule must use Z axis')
            frame,scale=_local_geometry(collider,body,UsdGeom)
            if not np.allclose(scale,scale[0],atol=1e-12,rtol=1e-8):raise ValueError('Nonuniform capsule scale unsupported')
            chain.append(ShaftCapsule(path,str(collider.GetPath()),frame,
                float(cap.GetRadiusAttr().Get())*scale[0],float(cap.GetHeightAttr().Get())*scale[0]/2,_offset(collider)))
        pads=[]
        for path in fingers:
            _path(path);body=stage.GetPrimAtPath(path);collider_path=path+'/restored_collisions/contact_proxy'
            prim=stage.GetPrimAtPath(collider_path)
            if (not _active(prim,UsdPhysics) or not prim.IsA(UsdGeom.Cube)
                    or _owner(prim,UsdPhysics)!=path or _coverage(stage,path,Usd,UsdPhysics)!=[collider_path]):
                raise ValueError('Each authored finger must contain exactly its source Cube pad')
            _body_world_basis(body,UsdGeom)
            frame,scale=_local_geometry(prim,body,UsdGeom)
            pads.append(FingerPad(path,collider_path,frame,scale*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2,0,-1,_offset(prim)))
        # The cached inventory must match actual authored active collisions;
        # keep leaf meshes rather than replacing the fixture by bare capsules.
        cached=[]
        for entry in fixture.held_plant_screen.local:
            if len(entry)!=4:raise ValueError('Unknown cached collision record')
            cached.append(dict(collider=entry[0],body_index=entry[1],kind=entry[2]))
        if sorted(cached,key=lambda r:r['collider'])!=sorted(inventory,key=lambda r:r['collider']):
            raise ValueError('Cached/authored collider inventory mismatch')
        links={frozenset(pair) for pair in zip(bodies[cut:],bodies[cut+1:])};joints=[]
        root=stage.GetPrimAtPath(rig.root)
        if not root:raise ValueError('Missing authored plant root')
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):continue
            joint=UsdPhysics.Joint(prim);a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
            if len(a)==len(b)==1 and frozenset((str(a[0]),str(b[0]))) in links:
                joints.append(dict(path=str(prim.GetPath()),body0=str(a[0]),body1=str(b[0])))
        joints.sort(key=lambda row:row['path'])
        claims=dict(_CLAIMS,geometry_basis='usd_authored_local_shapes_not_native_binding_receipt')
        p=dict(schema=AUTHORED_SCHEMA,source_target=source,binding_sha256=None,
            selected_body=bodies[selected],selected_index=selected,cut_index=cut,
            chain=[_pack(s) for s in chain],pads=[_pack(s) for s in pads],joints=joints,native_binding_extras={},
            plant_colliders=inventory,all_plant_collider_paths=sorted(r['collider'] for r in inventory),
            finger_contact_compliance=fixture.finger_contact_compliance,finger_friction=_scalar(fixture.friction),claims=claims)
        p=json.loads(_json(p));p['sha256']=_hash(p)
        deserialize_authored(p,source_target=source,sha256=p['sha256'])
        return p
    except (AttributeError,TypeError,KeyError,IndexError) as exc:
        raise ValueError('Incomplete or malformed authored fixture') from exc
