"""Deterministic donor-derived whole-plant morphology; static perception only.

Every organ shares one globally injective plant-space map. Existing component
writer/material-copy/normal authoring primitives remain the only USD writers.
Capsules bound warped donor proxies, NOT certified rendered or cutting geometry.
No native capture, inherited reviews, original-family reset or release admission.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from .audit import audit_manifest, safe_asset
from .plant_variants import PHYSICS_FIELDS, digest, file_hash, require, unit
from .plant_variant_usd import copy_component, mesh_arrays, write_json_new
from .procedural_petiole_usd import deform_new_copy

VERSION = 'donor_whole_plant_morphology.v1'
SCHEMA = 'greenhouse.donor_whole_plant_morphology_qualification.v1'
RADIUS_POLICY = 'conservative_warped_donor_capsule_bound_not_certified_mesh_radius'
MAX_MESH_INTERPOLATION_ERROR_M = .00025
MAX_CENTERLINE_INTERPOLATION_ERROR_M = .00001


def configuration(manifest, seed):
    require(type(seed) is int and 0 <= seed < 2**32, 'Seed must be uint32')
    roots = [c for c in manifest['components'] if c['parent'] is None]
    require(len(roots) == 1 and roots[0]['type'] == 'main_stem', 'One main-stem root required')
    base = np.asarray(roots[0]['transform']['translate'], float)
    endpoints = [np.asarray(c['transform']['translate']) + np.asarray(p[:3])
                 for c in manifest['components'] if c['type'] == 'main_stem'
                 for chain in c.get('capsules', []) for p in chain]
    require(endpoints, 'Main-stem centerlines required')
    height = float(np.max(np.asarray(endpoints)[:, 2]) - base[2])
    require(height > .5 and np.isfinite(base).all(), 'Qualified vertical stem extent required')
    rng = np.random.Generator(np.random.PCG64(seed))
    scale = 1 + float(rng.choice([-1, 1])) * float(rng.uniform(.008, .04))
    spacing = float(rng.choice([-1, 1])) * float(rng.uniform(.012, .025))
    angle = float(rng.uniform(-np.pi, np.pi))
    bend = float(rng.uniform(.015, .04))
    return dict(version=VERSION, seed=seed, base_plant_m=base.tolist(), height_extent_m=height,
                vertical_scale=scale, spacing_wave=spacing,
                horizontal_bend_m=[bend*np.cos(angle), bend*np.sin(angle)],
                radius_policy=RADIUS_POLICY, parameter_origin='bounded_engineering_not_fitted_biology')


class PlantWarp:
    """F(x,y,z)=(x+bx*(1-cos(pi*u)), y+by*(1-cos(pi*u)), z0+h(z-z0)).

    h(t)=s*t+a*L*sin(2*pi*t/L)/(2*pi). h'>=s-|a|>0 for all t.
    Hence F is globally one-to-one, with a fixed root and det(J)>0.
    ||J-I||<=delta gives global distance bounds 1-delta .. 1+delta.
    """
    def __init__(self, config):
        self.config = deepcopy(config)
        require(config['version'] == VERSION and config['radius_policy'] == RADIUS_POLICY,
                'Typed bounded morphology configuration required')
        self.base = np.asarray(config['base_plant_m'], float)
        self.height = float(config['height_extent_m'])
        self.scale = float(config['vertical_scale'])
        self.spacing = float(config['spacing_wave'])
        self.bend = np.asarray(config['horizontal_bend_m'], float)
        require(self.base.shape == (3,) and self.bend.shape == (2,)
                and np.isfinite(np.r_[self.base,self.bend,self.height,self.scale,self.spacing]).all(),
                'Finite warp parameters required')
        require(self.height > .5 and .96 <= self.scale <= 1.04
                and abs(self.spacing) <= .025 and np.linalg.norm(self.bend) <= .04000000001,
                'Warp outside bounded engineering envelope')
        self.delta = float(np.hypot(np.pi*np.linalg.norm(self.bend)/self.height,
                                    abs(self.scale-1)+abs(self.spacing)))
        require(self.delta < .3, 'Excessive global distortion')
        self.radius_upper_scale = 1+self.delta
        self.distance_lower_scale = 1-self.delta
        self.second_derivative_bound = float(np.hypot(
            np.pi**2*np.linalg.norm(self.bend)/self.height**2,
            2*np.pi*abs(self.spacing)/self.height))

    def map(self, points):
        p = np.asarray(points, float)
        require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), 'Finite Nx3 points required')
        t = p[:, 2]-self.base[2];u = t/self.height
        out = p.copy()
        out[:, :2] += (1-np.cos(np.pi*u))[:, None]*self.bend
        out[:, 2] = self.base[2]+self.scale*t+self.spacing*self.height*np.sin(2*np.pi*u)/(2*np.pi)
        return out

    def jacobian(self, points):
        p = np.asarray(points, float);u = (p[:, 2]-self.base[2])/self.height
        j = np.broadcast_to(np.eye(3), (len(p), 3, 3)).copy()
        j[:, :2, 2] = (np.pi*np.sin(np.pi*u)/self.height)[:, None]*self.bend
        j[:, 2, 2] = self.scale+self.spacing*np.cos(2*np.pi*u)
        return j

    def normals(self, points, normals):
        p,n = np.asarray(points,float),np.asarray(normals,float)
        require(p.shape == n.shape and np.isfinite(n).all(), 'Normal locations must match points')
        if not len(p):
            return n.copy(),dict(minimum_jacobian_determinant=None,maximum_jacobian_condition=None,
                preserved_zero_normals=0,global_injectivity_proven=True,empty_normal_channel=True)
        j = self.jacobian(p)
        transformed = np.linalg.solve(j.swapaxes(1,2), n[:, :, None])[:, :, 0]
        lengths = np.linalg.norm(transformed,axis=1);nonzero = lengths > 1e-12
        transformed[nonzero] /= lengths[nonzero,None];transformed[~nonzero] = 0
        return transformed,dict(minimum_jacobian_determinant=float(np.linalg.det(j).min()),
            maximum_jacobian_condition=float(np.linalg.cond(j).max()),
            preserved_zero_normals=int((~nonzero).sum()),global_injectivity_proven=True,
            proof='triangular_map_strictly_monotone_vertical_coordinate')

    def direction(self, point, direction):
        return unit(self.jacobian(np.asarray(point)[None,:])[0] @ unit(direction))

    def interpolation_bound(self, vertical_span):
        return self.second_derivative_bound*np.asarray(vertical_span)**2/8

    def chain(self, chain, old_origin, new_origin):
        raw = np.asarray(chain,float)
        require(raw.ndim == 2 and raw.shape[1] == 4 and len(raw) >= 2
                and np.isfinite(raw).all() and (raw[:,3] > 0).all(), 'Finite positive capsule chain required')
        samples=[];maximum_error=0.
        for i,(a,b) in enumerate(zip(raw[:-1],raw[1:])):
            length = float(np.linalg.norm(b[:3]-a[:3]))
            pieces = max(1,int(np.ceil(length/.02)),int(np.ceil(np.sqrt(
                self.interpolation_bound(b[2]-a[2])/MAX_CENTERLINE_INTERPOLATION_ERROR_M))))
            error = float(self.interpolation_bound((b[2]-a[2])/pieces))
            maximum_error=max(maximum_error,error)
            segment=np.linspace(a,b,pieces+1)
            samples.extend(segment if i == 0 else segment[1:])
        samples=np.asarray(samples)
        points=self.map(samples[:,:3]+old_origin)-new_origin
        # A mapped original capsule lies in this expanded polygonal proxy.
        radii=samples[:,3]*self.radius_upper_scale+maximum_error
        return np.c_[points,radii].tolist(),maximum_error


def transformed_manifest(source, warp, variant_id, family, split):
    out={k:deepcopy(v) for k,v in source.items() if k not in PHYSICS_FIELDS}
    out.update(generator=VERSION,version='1.0.0',seed=warp.config['seed'],
               donor_generator=source.get('generator'),donor_generator_version=source.get('version'),
               donor_seed=source.get('seed'),variant_id=variant_id,source_family=family,split_group=family,
               source_split=split,physics_supported=False,static_perception_only=True,
               new_independent_source_family=False,capsule_radius_policy=RADIUS_POLICY)
    rows=[];maximum_error=0.
    for old in source['components']:
        row={k:deepcopy(v) for k,v in old.items() if k not in PHYSICS_FIELDS}
        origin=np.asarray(old['transform']['translate'],float);new=warp.map(origin[None,:])[0]
        row['transform']={'translate':new.tolist()}
        row['attach_point']=warp.map(np.asarray(old['attach_point'])[None,:])[0].tolist()
        row['axis']=warp.direction(old['attach_point'],old['axis']).tolist()
        if old.get('capsules'):
            chains=[warp.chain(chain,origin,new) for chain in old['capsules']]
            row['capsules']=[c for c,e in chains]
            error=max(e for c,e in chains);maximum_error=max(maximum_error,error)
            row['radius']=max(p[3] for c,e in chains for p in c)
            first=np.asarray(row['capsules'][0])
            length=float(np.linalg.norm(np.diff(first[:,:3],axis=0),axis=1).sum())
            if length <= 1e-12:
                require(old.get('deleafed') is True,
                        'Zero-length intact petiole cannot qualify as morphology geometry')
                row.pop('length',None)
                row['inherited_degenerate_capsule_preserved']=True
                row['donor_legacy_length_m']=old.get('length')
                row['length_summary_status']='undefined_degenerate_donor_proxy_not_repaired'
            else:row['length']=length
            row['capsule_radius_policy']=RADIUS_POLICY
            row['centerline_interpolation_error_bound_m']=error
        # Remove stale height/stub summaries; transformed chains are authoritative.
        for key in ('height_fraction','stub_length'):row.pop(key,None)
        rows.append(row)
    out['components']=rows
    require([(r['id'],r['parent'],r.get('deleafed')) for r in rows] ==
            [(r['id'],r['parent'],r.get('deleafed')) for r in source['components']],
            'Full graph/deleaf state must remain unchanged')
    return out,maximum_error


def shape_metrics(source, generated):
    old=np.asarray([c['transform']['translate'] for c in source['components'] if c['type']=='main_stem'])
    new=np.asarray([c['transform']['translate'] for c in generated['components'] if c['type']=='main_stem'])
    require(len(old)>=3 and old.shape==new.shape, 'Multiple stem landmarks required')
    def distances(p):return np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2)
    a,b=distances(old),distances(new)
    na,nb=a/np.linalg.norm(a),b/np.linalg.norm(b)
    metrics=dict(purpose='trivial_duplicate_guard_not_diversity_admission',donor_stem_distance_hash=digest(np.round(a,6).tolist()),
                 generated_stem_distance_hash=digest(np.round(b,6).tolist()),
                 donor_normalized_stem_distance_hash=digest(np.round(na,9).tolist()),
                 generated_normalized_stem_distance_hash=digest(np.round(nb,9).tolist()),
                 maximum_normalized_stem_distance_change=float(np.abs(nb-na).max()),
                 maximum_stem_pair_distance_change_m=float(np.abs(b-a).max()),
                 maximum_landmark_displacement_m=float(np.linalg.norm(new-old,axis=1).max()),
                 original_family_identity_reset=False)
    require(metrics['donor_stem_distance_hash']!=metrics['generated_stem_distance_hash']
            and metrics['maximum_stem_pair_distance_change_m']>.001
            and metrics['maximum_normalized_stem_distance_change']>1e-5,
            'Renamed, rigid, uniform-scale or near-identity morphology cannot qualify')
    return metrics


def mesh_interpolation_bound(stage, origin, warp):
    from pxr import UsdGeom
    maximum=0.
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),float)+origin
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
        if not len(counts):continue
        require((counts>=3).all() and counts.sum()==len(indices), 'Valid face topology required')
        starts=np.r_[0,np.cumsum(counts)[:-1]];z=points[indices,2]
        spans=np.maximum.reduceat(z,starts)-np.minimum.reduceat(z,starts)
        maximum=max(maximum,float(np.max(warp.interpolation_bound(spans))))
    return maximum


def shared_location_check(old_points,new_points,owners,warp):
    """Compare coincident donor locations across components; no invented seam repair."""
    old=np.concatenate(old_points);new=np.concatenate(new_points);owners=np.concatenate(owners)
    keys=np.round(old/1e-6).astype(np.int64)
    _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(counts)]
    matched=0;oldmax=newmax=0.
    for group in np.flatnonzero(counts>1):
        ix=order[starts[group]:starts[group+1]]
        if len(np.unique(owners[ix]))<2:continue
        before=float(np.linalg.norm(np.ptp(old[ix],axis=0)))
        after=float(np.linalg.norm(np.ptp(new[ix],axis=0)))
        require(after<=warp.radius_upper_scale*before+2e-6, 'Shared component mesh seam displaced')
        matched+=1;oldmax=max(oldmax,before);newmax=max(newmax,after)
    require(matched>0,'No shared component mesh locations verified')
    return dict(shared_component_mesh_location_groups=matched,
                maximum_donor_group_spread_m=oldmax,maximum_generated_group_spread_m=newmax,
                comparison_quantization_m=1e-6,serialization_allowance_m=2e-6)


def generate(source_manifest, source_plan, source_plan_sha256, seed, output_root):
    from pxr import Usd
    started=time.perf_counter()
    mp,pp=Path(source_manifest).resolve(),Path(source_plan).resolve()
    require(file_hash(pp)==source_plan_sha256,'Source plan pin differs')
    plan=json.loads(pp.read_text(encoding='utf-8'));source=json.loads(mp.read_text(encoding='utf-8'))
    family=mp.parent.name;split=plan['family_assignments'][family]
    require(split in ('train','validation','test'), 'Frozen donor split required')
    jobs=[j for j in plan['jobs'] if j['plant_family']==family]
    require(len(jobs)==1 and jobs[0]['split']==split and Path(jobs[0]['source_manifest_path']).resolve()==mp,
            'Exact donor job/manifest required')
    require(plan['source_bindings_sha256'].get(str(mp))==file_hash(mp),'Donor manifest pin differs')
    root=Path(output_root).resolve()
    require(not root.is_relative_to(Path(plan['package']).resolve()),'Cannot write into source package')
    source_report=audit_manifest(mp);require(source_report['status']!='blocked','Blocked donor manifest')
    pins={str(pp):source_plan_sha256,str(mp):file_hash(mp)}
    for component in source['components']:
        path=safe_asset(mp.parent,component['file']);h=file_hash(path)
        require(plan['source_bindings_sha256'].get(str(path))==h,'Donor component pin differs')
        pins[str(path)]=h
    config=configuration(source,seed);warp=PlantWarp(config)
    variant=family+'_morph_'+str(seed)+'_'+digest(config)[:12];out=root/variant
    require(not out.exists(),'Create-only generated variant required')
    manifest,error=transformed_manifest(source,warp,variant,family,split)
    metrics=shape_metrics(source,manifest)
    out.mkdir(parents=True,exist_ok=False)
    try:
        textures={};receipts={};old_points=[];new_points=[];owners=[];maximum_mesh_error=0.
        for index,(old,new) in enumerate(zip(source['components'],manifest['components'])):
            path=safe_asset(mp.parent,old['file']);destination=safe_asset(out,new['file'])
            destination.parent.mkdir(parents=True,exist_ok=True)
            origin=np.asarray(old['transform']['translate']);new_origin=np.asarray(new['transform']['translate'])
            stage=Usd.Stage.Open(str(path));points,_=mesh_arrays(stage)
            bound=mesh_interpolation_bound(stage,origin,warp)
            require(bound<=MAX_MESH_INTERPOLATION_ERROR_M,'Source faces too coarse for qualified warp')
            maximum_mesh_error=max(maximum_mesh_error,bound)
            receipt=copy_component(path,destination,np.eye(3),1.,mp.parent,textures)
            normal_qa=deform_new_copy(destination,origin,new_origin,warp)
            reread=Usd.Stage.Open(str(destination));actual,_=mesh_arrays(reread)
            expected=warp.map(points+origin)-new_origin
            require(actual.shape==expected.shape and np.allclose(actual,expected,atol=2e-7,rtol=1e-6),
                    'Serialized component/plant frame warp mismatch')
            receipt.update(output_sha256=file_hash(destination),normals=normal_qa,
                           mesh_interpolation_error_bound_m=bound)
            receipts[old['id']]=receipt
            old_points.append(points+origin);new_points.append(actual+new_origin)
            owners.append(np.full(len(points),index,dtype=np.int32))
            print(json.dumps(dict(component=old['id'],completed=index+1,total=len(source['components']))),flush=True) if (index+1)%100==0 else None
        seam=shared_location_check(old_points,new_points,owners,warp)
        source_geometry_hash=digest([np.round(p,6).tolist() for p in old_points])
        generated_geometry_hash=digest([np.round(p,6).tolist() for p in new_points])
        require(source_geometry_hash!=generated_geometry_hash,'Geometry-identical renamed variant rejected')
        write_json_new(out/'manifest.json',manifest)
        report=audit_manifest(out/'manifest.json')
        require(report['status']!='blocked','Generated structural audit failed')
        require(set(report['components'])==set(source_report['components']),'Component set changed')
        for path,h in pins.items():require(file_hash(path)==h,'Source changed during generation')
        for path,h in textures.items():require(file_hash(path)==h,'Copied texture changed')
        # Copy-time source texture pins are checked again; old plans do not bind textures.
        texture_pins={str(safe_asset(mp.parent,relative)):h
                      for r in receipts.values() for relative,h in r['texture_hashes'].items()}
        for path,h in texture_pins.items():require(file_hash(path)==h,'Source texture changed during generation')
        implementation_paths=[Path(__file__),Path(__file__).with_name('plant_variant_usd.py'),
            Path(__file__).with_name('plant_variants.py'),Path(__file__).with_name('procedural_petiole_usd.py'),
            Path(__file__).with_name('audit.py')]
        result=dict(schema=SCHEMA,version=VERSION,variant_id=variant,
            state='cpu_generated_donor_morphology_pending_native_geometry_and_visual_review',
            source_family=family,split_group=family,source_split=split,
            original_source_family_cap_group=family,new_independent_source_family=False,
            independent_biological_novelty_approved=False,morphology_diversity_approved=False,derived_geometry_instance_count=1,
            configuration=config,configuration_sha256=digest(config),source_bindings=pins,
            texture_source_bindings=texture_pins,texture_binding_origin='generator_copy_time',
            source_assets_unchanged=True,full_component_graph_preserved=True,source_deleaf_state_preserved=True,
            source_component_count=len(source['components']),source_geometry_hash=source_geometry_hash,
            generated_geometry_hash=generated_geometry_hash,shape_metrics=metrics,mesh_seam_check=seam,
            maximum_mesh_interpolation_error_bound_m=maximum_mesh_error,
            maximum_centerline_interpolation_error_bound_m=error,
            global_distance_scale_bounds=[warp.distance_lower_scale,warp.radius_upper_scale],
            global_injectivity_proven=True,global_mesh_collision_validated=False,
            capsule_radius_policy=RADIUS_POLICY,capsules_are_certified_mesh_geometry=False,
            static_perception_only=True,physics_supported=False,rendered_geometry_qualified=False,
            native_capture_validated=False,review_decisions_inherited=False,training_approved=False,
            accepted_training_increment=0,collection_approved=False,
            structural_audit_status=report['status'],component_receipts=receipts,
            output_hashes={**{c['file']:receipts[c['id']]['output_sha256'] for c in manifest['components']},
                           **{p.relative_to(out).as_posix():h for p,h in textures.items()},
                           'manifest.json':file_hash(out/'manifest.json')},
            implementation_hashes={str(p.resolve()):file_hash(p) for p in implementation_paths},
            versions=dict(python=platform.python_version(),numpy=np.__version__,usd=list(Usd.GetVersion())),
            elapsed_seconds=time.perf_counter()-started)
        write_json_new(out/'qualification.json',result)
        return result
    except BaseException as exc:
        write_json_new(out/'FAILED.json',dict(state='failed_not_usable',error=repr(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--source-plan-sha256',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    result=generate(a.source_manifest,a.source_plan,a.source_plan_sha256,a.seed,a.output_root)
    print(json.dumps({k:result[k] for k in ('variant_id','state','source_family','source_split','elapsed_seconds',
                                         'shape_metrics','mesh_seam_check','accepted_training_increment')},indent=2))


if __name__=='__main__':main()

