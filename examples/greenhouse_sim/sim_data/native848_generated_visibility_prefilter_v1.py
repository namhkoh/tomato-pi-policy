"""Cheap authored-plant ray rejection only; never native labels or acceptance.

Preserves native848 9mm geometry/probe locations, radius+3mm depth tolerance,
all support/junction probes and >=.95 unique-pixel proximal visibility. Only
certain authored opaque mesh hits can reject. Missing/unsupported mesh evidence
stays unknown, so a non-reject result is not a visibility pass. No greenhouse,
robot, lighting, alpha-texture rendering or full local-clarity reconstruction.
"""
from pathlib import Path
import json,time
import numpy as np
from .audit import audit_manifest
from .dataset_review import require,verify_bindings
from .depth_preview import sha256
from .native848_all_petiole_9mm_v2 import geometry_9mm,POLICY
from .native_capture_v5 import visibility_shadow as mesh

SCHEMA='greenhouse.generated_authored_mesh_visibility_prefilter.v1'

def prepare(manifest_pin,plant_to_world,*,expected_bindings=None):
    path=Path(manifest_pin['path']).resolve()
    require(path.name=='manifest.json' and sha256(path)==manifest_pin['sha256'],'Manifest pin changed')
    report=audit_manifest(path);require(report['status']!='blocked','Invalid manifest')
    bindings={str(Path(p).resolve()):h for p,h in (expected_bindings or {}).items()}
    local={str(path):manifest_pin['sha256']}
    local.update({str((path.parent/c['file']).resolve()):c['asset_sha256'] for c in report['components'].values()})
    for p,h in local.items():require(p not in bindings or bindings[p]==h,'Conflicting source pin');bindings[p]=h
    snapshot,build=mesh.load_partial_plant(path.parent,plant_to_world,expected_bindings=bindings)
    return snapshot,report,build

def _cast(snapshot,calibration,pixel):
    origin,direction,near,far=mesh._pixel_ray(calibration,np.asarray(pixel)+.5)
    boxes,entries=mesh.ray_boxes([m.lower for m in snapshot.meshes],[m.upper for m in snapshot.meshes],origin,direction,near,far)
    transform=np.asarray(calibration['camera_to_world_usd_row_vectors'],float)
    optical_z_per_ray_m=float(direction@(-transform[2,:3]))
    require(optical_z_per_ray_m>0,'Invalid calibrated optical direction')
    hits=[];unknown=[]
    for i in np.flatnonzero(boxes):
        m=snapshot.meshes[i]
        if m.unknown_reasons:
            unknown.append(dict(component_id=m.component,reasons=list(m.unknown_reasons),box_optical_z_m=float(entries[i]*optical_z_per_ray_m)));continue
        possible,certain=mesh.mesh_hits(m,origin,direction,near,far)
        if np.isfinite(possible):
            hits.append(dict(component_id=m.component,prim_path=m.prim_path,possible_optical_z_m=float(possible*optical_z_per_ray_m),certain_optical_z_m=float(certain*optical_z_per_ray_m) if np.isfinite(certain) else None))
    return dict(hits=hits,unknown_meshes=unknown,ray_pixel_center_xy=(np.asarray(pixel)+.5).tolist(),aabb_candidates=int(boxes.sum()))

def probe(point,ray,target_id,allowed_near_joint):
    projected=point['projected']
    row=dict(arc_m=point['arc_m'],projected=projected,geometrically_rejected=False,reason=None)
    if projected['projection_status']!='in_frame':return dict(row,geometrically_rejected=True,reason='out_of_frame')
    axis=float(projected['camera_optical_xyz_m'][2]);tolerance=float(point['radius_m'])+.003
    certain=[h for h in ray['hits'] if h['certain_optical_z_m'] is not None]
    foreground=[h for h in certain if axis-h['certain_optical_z_m']>tolerance]
    row.update(pixel_xy=np.floor(projected['pixel_xy']).astype(int).tolist(),target_axis_depth_m=axis,tolerance_m=tolerance,ray=ray)
    if foreground:
        h=min(foreground,key=lambda h:h['certain_optical_z_m'])
        return dict(row,geometrically_rejected=True,reason='authored_surface_foreground_depth_gap',blocker=h,minimum_foreground_gap_m=axis-h['certain_optical_z_m'])
    # At >=8mm native component ownership must be target-only. Near the joint,
    # ancestor seam ownership is deferred to the unchanged native proof adapter.
    if not allowed_near_joint:
        target=[h for h in ray['hits'] if h['component_id']==target_id]
        others=[h for h in certain if h['component_id']!=target_id]
        if target:
            nearest_target=min(h['possible_optical_z_m'] for h in target)
            blockers=[h for h in others if h['certain_optical_z_m']<nearest_target-1e-8]
            if blockers:return dict(row,geometrically_rejected=True,reason='authored_other_component_in_front',blocker=min(blockers,key=lambda h:h['certain_optical_z_m']))
    return dict(row,reason='partial_model_no_certain_rejection_not_visibility_pass')

def aggregate(groups):
    reasons=[]
    for name in ('nominal','support','junction'):
        if any(r['geometrically_rejected'] for r in groups[name]):reasons.append(name+'_authored_geometry_rejection')
    pixels={}
    for r in groups['proximal']:
        if r['projected']['projection_status']=='in_frame':pixels.setdefault(tuple(r['pixel_xy']),[]).append(not r['geometrically_rejected'])
    upper=sum(all(v) for v in pixels.values())/len(pixels) if pixels else 0.
    if upper<POLICY['minimum_proximal_visible_fraction']:reasons.append('proximal_possible_visible_fraction_below_native_threshold')
    return dict(reject=bool(reasons),reasons=reasons,proximal_possible_visible_fraction_upper_bound=upper,proximal_unique_pixels=len(pixels))

def evaluate(snapshot,report,component_id,calibration):
    started=time.perf_counter()
    require(calibration['resolution']==[848,408] and calibration.get('crop_resize') is None,'Exact native camera dimensions required')
    geo=geometry_9mm(report,component_id,snapshot.plant_to_world,calibration)
    groups={};rays={}
    for name,points in [('nominal',[geo['nominal']]),('support',geo['visibility_support']),('proximal',geo['proximal']),('junction',geo['junction_to_cut'])]:
        rows=[]
        for p in points:
            ray=None
            if p['projected']['projection_status']=='in_frame':
                key=tuple(np.floor(p['projected']['pixel_xy']).astype(int))
                if key not in rays:rays[key]=_cast(snapshot,calibration,key)
                ray=rays[key]
            rows.append(probe(p,ray,component_id,p['arc_m']<.008))
        groups[name]=rows
    result=aggregate(groups)
    result.update(schema=SCHEMA,component_id=component_id,snapshot_sha256=snapshot.fingerprint,groups=groups,nominal_arc_m=.009,visibility_support_arc_m=[.009,.019],depth_tolerance='physical_proxy_radius_plus_0.003_m_same_native_formula',native_minimum_proximal_visible_fraction=POLICY['minimum_proximal_visible_fraction'],ray_count=len(rays),wall_seconds=time.perf_counter()-started,full_scene_visibility_established=False,native_visibility_pass=False,native_depth_reconstructed=False,training_approved=False,acceptance_predicates_changed=False,unknown_policy='unknown counts as potentially visible; only certain authored opaque surfaces can reject')
    snapshot.finish()
    return json.loads(mesh.canonical(result))

def check(manifest_pin,component_id,plant_to_world,calibration,*,expected_bindings=None):
    snapshot,report,build=prepare(manifest_pin,plant_to_world,expected_bindings=expected_bindings)
    try:result=evaluate(snapshot,report,component_id,calibration);result['build']=build;result['source_bindings']=dict(snapshot.source_bindings);return result
    finally:snapshot.finish()
