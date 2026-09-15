"""Create-only curved/relocated plant variants using original detailed mesh assets.

New centerlines, new attachment locations, preserved topology/materials/leaf UVs.
Static inspection only: no source edits, no release novelty or physics approval.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import math

import numpy as np

from .audit import audit_manifest, descendants, safe_asset
from .cut_regions import _oriented_chain, propose_cut_region, load_rule
from .plant_variants import (load_training_sources, training_envelope, normalized,
    PHYSICS_FIELDS, rotation, file_hash, digest)
from .plant_variant_usd import copy_component, mesh_arrays, first_ray_hit, cut_surface_probe, write_json_new
from .procedural_petiole_geometry import CurveSpec, curved_centerline, intrinsic_descriptor, unit, require
from .procedural_petiole_warp import CurveWarp

VERSION='curved_relocated_petiole_static.v1'


def donor_curve(component):
    points,arc,_,_=_oriented_chain(component,1e-6)
    return dict(points=np.asarray(points)[:,:3],radius=np.asarray(points)[:,3],arc=np.asarray(arc))


def choose_attachment(parent, old_anchor, direction, offset_m, parent_triangles):
    """Move along parent skeleton, then find its ACTUAL authored mesh surface."""
    choices=[]
    origin=np.asarray(parent['translation_plant_m'])
    for chain in parent['capsules_local_m']:
        for a,b in zip(chain,chain[1:]):
            a,b=np.asarray(a[:3])+origin,np.asarray(b[:3])+origin
            length=np.linalg.norm(b-a)
            if length<.04:continue
            axis=(b-a)/length
            old_s=np.clip(np.dot(np.asarray(old_anchor)-a,axis),0,length)
            new_s=old_s+offset_m
            if not .007 <= new_s <= length-.007:continue
            center=a+new_s*axis
            hit=first_ray_hit(parent_triangles,center-origin,direction)
            if hit is None or not .0005 <= hit <= .03:continue
            anchor=center+hit*direction
            displacement=float(np.linalg.norm(anchor-old_anchor))
            if displacement<.01:continue
            choices.append((float(np.linalg.norm(a+old_s*axis-old_anchor)),anchor,axis,hit,displacement))
    require(choices,'No qualified relocated attachment on the actual parent surface')
    _,anchor,axis,hit,displacement=min(choices,key=lambda v:v[0])
    return anchor,dict(parent_surface_ray_hit_m=float(hit),anchor_displacement_m=displacement,
        parent_axis=axis.tolist(),method='source_parent_centerline_then_authored_triangle_surface',
        protected_organ_clearance_verified=False)


def plan_change(source,key,envelope,seed):
    from pxr import Usd
    report=source['report'];component=report['components'][key]
    parent=report['components'][component['parent']]
    require(component['type']=='sub_stem' and component['deleafed'] is False and parent['type']=='main_stem',
            'Intact direct-main-stem target required')
    members=descendants(report['components'],key)
    require(all(report['components'][m]['type'] in ('sub_stem','leaf') for m in members)
            and any(report['components'][m]['type']=='leaf' for m in members),'Protected/leafless subtree')
    old=donor_curve(component)
    old_absolute=old['points']+component['translation_plant_m']
    old_direction=unit(old_absolute[1]-old_absolute[0])
    parent_stage=Usd.Stage.Open(str(safe_asset(Path(report['manifest_path']).parent,parent['file'])))
    _,tri=mesh_arrays(parent_stage)
    parent_axis=unit(np.asarray(parent['capsules_local_m'][0][-1][:3])-parent['capsules_local_m'][0][0][:3])
    rng=np.random.default_rng(seed)
    failures=[]
    for attempt in range(32):
        direction=rotation(parent_axis,float(rng.uniform(-22,22)))@old_direction
        bend_axis=np.cross(direction,parent_axis)
        if np.linalg.norm(bend_axis)>1e-5:
            direction=rotation(bend_axis,float(rng.uniform(-10,10)))@direction
        angle=math.degrees(math.acos(float(np.clip(np.dot(direction,parent_axis),-1,1))))
        lo,hi=envelope['bounds']['attachment_angle_deg']
        if not max(25.,lo) <= angle <= min(110.,hi):continue
        offset=float(rng.choice([-1,1])*rng.uniform(.014,.035))
        try:anchor,surface=choose_attachment(parent,component['attachment_plant_m'],direction,offset,tri)
        except ValueError as exc:failures.append(str(exc));continue
        length=float(old['arc'][-1]*rng.uniform(.85,1.15))
        radius=float(old['radius'][0]*rng.uniform(.9,1.1))
        if not (envelope['bounds']['length_m'][0]<=length<=envelope['bounds']['length_m'][1]
                and envelope['bounds']['radius_m'][0]<=radius<=envelope['bounds']['radius_m'][1]):continue
        spec=CurveSpec(length,radius,tip_radius_ratio=float(np.clip(old['radius'][-1]/old['radius'][0],.3,.8)),
            bend_normal_rad=float(rng.choice([-1,1])*rng.uniform(.35,.85)),
            bend_binormal_rad=float(rng.uniform(-.5,.5)),bend_phase_rad=float(rng.uniform(-np.pi,np.pi)))
        new=curved_centerline(spec,direction)
        delta=intrinsic_descriptor(new)-intrinsic_descriptor(old)
        if np.max(abs(delta))<.012:continue
        warp=CurveWarp(old_absolute,new,anchor,radius/old['radius'][0])
        return dict(component_id=key,members=members,curve=new,warp=warp,seed=seed,attempt=attempt,
            parameters=asdict(spec),direction=direction.tolist(),new_anchor_m=anchor.tolist(),
            source_anchor_m=component['attachment_plant_m'],surface_attachment=surface,
            intrinsic_max_absolute_difference=float(np.max(abs(delta))),
            intrinsic_rms_difference=float(np.sqrt(np.mean(delta**2))),
            descriptor=intrinsic_descriptor(new).tolist(),
            source_target_id=source['job']['plant_family']+'/'+key,
            new_independent_donor_family=False,novelty_admission_approved=False)
    raise ValueError('No bounded curved/relocated recipe: '+str(failures[-3:]))


def transform_metadata(raw, change):
    result=deepcopy(raw);warp=change['warp']
    origin=np.asarray(raw['transform']['translate'])
    new_origin=warp.map(origin[None,:])[0]
    result['transform']={'translate':new_origin.tolist()}
    result['attach_point']=warp.map(np.asarray(raw['attach_point'])[None,:])[0].tolist()
    result['axis']=warp.direction(raw['attach_point'],raw['axis']).tolist()
    if raw['id']==change['component_id']:
        curve=change['curve']
        result['transform']={'translate':change['new_anchor_m']}
        result['attach_point']=change['new_anchor_m']
        result['capsules']=[np.c_[curve['points'],curve['radius']].tolist()]
        result['axis']=unit(curve['points'][1]-curve['points'][0]).tolist()
        result['length']=float(curve['arc'][-1]);result['radius']=float(curve['radius'][0])
    elif raw.get('capsules'):
        result['capsules']=[np.c_[warp.map(np.asarray(chain)[:,:3]+origin)-new_origin,
            np.asarray(chain)[:,3]*warp.radial_scale].tolist() for chain in raw['capsules']]
    for key in PHYSICS_FIELDS:result.pop(key,None)
    return result


def deform_new_copy(path, old_origin, new_origin, warp):
    """Only a newly copied file. Normals follow the inverse-transpose Jacobian."""
    from pxr import Gf, Usd, UsdGeom, Vt
    stage=Usd.Stage.Open(str(path));receipts=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim)
        points=np.asarray(mesh.GetPointsAttr().Get(),float)
        original_points=points.copy()
        mapped=warp.map(points+old_origin)-new_origin
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
        for attr in prim.GetAuthoredAttributes():
            role=attr.GetTypeName().role
            if role=='Vector':raise ValueError('Unqualified vector/tangent channel')
            if role!='Normal':continue
            require(attr.GetName()=='normals','Unqualified normal primvar')
            normals=np.asarray(attr.Get(),float)
            interpolation=mesh.GetNormalsInterpolation()
            if interpolation=='faceVarying':positions=original_points[indices]
            elif interpolation in ('vertex','varying'):positions=original_points
            else:raise ValueError('Unsupported source normal interpolation: '+str(interpolation))
            require(len(positions)==len(normals),'Normal cardinality mismatch')
            transformed,qa=warp.normals(positions+old_origin,normals)
            attr.Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in transformed.tolist()]))
            receipts.append(dict(mesh=str(prim.GetPath()),**qa))
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mapped.tolist()]))
        mesh.CreateExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*mapped.min(axis=0)),Gf.Vec3f(*mapped.max(axis=0))]))
    require(bool(receipts),'No transformed mesh normal evidence')
    stage.GetRootLayer().Save()
    return receipts


def generate(plan_path,family,targets,seed,output):
    from pxr import Usd
    output=Path(output).resolve();require(not output.exists(),'New output only')
    plan,sources=load_training_sources(plan_path)
    require(family in sources,'Frozen TRAIN family required')
    source=sources[family];envelope=training_envelope(plan,sources)
    require(targets and len(set(targets))==len(targets),'Explicit unique targets required')
    allowed={t['component_id'] for t in source['job']['targets']}
    require(set(targets)<=allowed,'Target outside frozen source catalogue')
    changes=[plan_change(source,key,envelope,seed+i*104729) for i,key in enumerate(targets)]
    owners={}
    for change in changes:
        for key in change['members']:
            require(key not in owners,'Overlapping target subtrees');owners[key]=change
    source_path=Path(source['report']['manifest_path'])
    require(not output.is_relative_to(source_path.parent),'No writes inside source assets')
    output.mkdir(parents=True)
    try:
        raw=deepcopy(source['raw'])
        raw.update(generator=VERSION,version='1.0.0',seed=seed,physics_supported=False,
                   intended_use='curved_relocated_static_geometry_pending_native_qualification')
        for k in PHYSICS_FIELDS:raw.pop(k,None)
        rows=[];textures={};output_hashes={};geometry=[]
        for old in source['raw']['components']:
            change=owners.get(old['id'])
            row=transform_metadata(old,change) if change else deepcopy(old)
            for k in PHYSICS_FIELDS:row.pop(k,None)
            destination=safe_asset(output,old['file']);destination.parent.mkdir(parents=True,exist_ok=True)
            copied=copy_component(safe_asset(source_path.parent,old['file']),destination,np.eye(3),1.,source_path.parent,textures)
            if change:
                evidence=deform_new_copy(destination,np.asarray(old['transform']['translate']),
                    np.asarray(row['transform']['translate']),change['warp'])
                geometry.extend(dict(component_id=old['id'],**q) for q in evidence)
            rows.append(row);output_hashes[old['file']]=file_hash(destination)
        raw['components']=rows;raw['component_count']=len(rows)
        write_json_new(output/'manifest.json',raw)
        report=audit_manifest(output/'manifest.json')
        require(report['status']!='blocked','Generated manifest structure failed')
        rules=load_rule();targets_out=[]
        for change in changes:
            key=change['component_id'];component=report['components'][key];parent=report['components'][component['parent']]
            proposal=propose_cut_region(component,parent,rules)
            require(proposal['status']=='proposed_geometry_only' and not proposal['geometry_warnings'],
                    'Generated cut interval lacks conservative parent clearance')
            stage=Usd.Stage.Open(str(output/component['file']))
            surface=cut_surface_probe(stage,component,parent)
            require(surface['passed'],'New centerline cut does not lie within the actual deformed surface')
            targets_out.append(dict(component_id=key,source_target_id=change['source_target_id'],
                cut_region_proposal=proposal,cut_surface_probe=surface,
                conservative_view_cap_group=change['source_target_id'],
                # Until a separate global shape/near-duplicate qualification exists,
                # this generator does not reset the original target's view budget.
                shape_novelty_pending=True,training_eligible=False))
        for path,sha in textures.items():output_hashes[path.relative_to(output).as_posix()]=sha
        output_hashes['manifest.json']=file_hash(output/'manifest.json')
        recipe=[{k:v for k,v in c.items() if k not in ('curve','warp')} for c in changes]
        source_bindings={str(source_path):file_hash(source_path)}
        for c in source['report']['components'].values():
            path=safe_asset(source_path.parent,c['file'])
            require(file_hash(path)==c['asset_sha256'],'Source changed while generating')
            source_bindings[str(path)]=c['asset_sha256']
        for relative in output_hashes:
            if relative.endswith(('.png','.jpg','.jpeg')):
                path=safe_asset(source_path.parent,relative)
                require(file_hash(path)==output_hashes[relative],'Source texture changed')
                source_bindings[str(path)]=output_hashes[relative]
        receipt=dict(version=VERSION,state='cpu_curved_geometry_pending_native_review',source_family=family,
            split='train',split_group=family,variant_id=output.name,source_plan_path=str(Path(plan_path).resolve()),
            source_plan_sha256=file_hash(plan_path),frozen_family_assignments=plan['family_assignments'],
            source_bindings=source_bindings,source_manifest_path=str(source_path),
            output_hashes=output_hashes,recipes=recipe,targets=targets_out,mesh_derivative_diagnostics=geometry,
            training_envelope_sha256=digest(envelope),source_assets_unchanged=True,
            training_eligible=False,native_capture_validated=False,physics_validated=False,
            independent_target_novelty_approved=False,new_donor_family_created=False,
            code_sha256={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('procedural_petiole_geometry.py'),
                Path(__file__).with_name('procedural_petiole_warp.py'),Path(__file__).with_name('plant_variant_usd.py'),
                Path(__file__).with_name('plant_variants.py'),Path(__file__).with_name('audit.py'),Path(__file__).with_name('cut_regions.py')]})
        write_json_new(output/'qualification.json',receipt)
        print('CURVED_PLANT_CPU_QUALIFIED',output.name,len(targets_out),flush=True)
        return receipt
    except BaseException as exc:
        write_json_new(output/'FAILED.json',dict(error=type(exc).__name__+': '+str(exc),training_eligible=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--family',required=True)
    p.add_argument('--targets',nargs='+',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.source_plan,a.family,a.targets,a.seed,a.output)


if __name__=='__main__':main()
