"""Source-bound mesh diagnostics for fixed controlled petiole variants.

This initial evidence version admits NO modified radius-dependent target. It
reports current 9 mm authored-surface measurements and shared-junction changes;
unchanged petioles retain original-source treatment only after full frozen
catalogue replay proves their visible geometry and anatomical metadata unchanged.
No new radius tolerance, radius substitution, native or training approval.
"""
from pathlib import Path
import math
import numpy as np
from pxr import Usd
from .audit import audit_manifest, safe_asset
from .cut_regions import _oriented_chain, _sample
from .dataset_review import read_json, require, verify_bindings
from .depth_preview import sha256
from .plant_variant_usd import mesh_arrays, first_ray_hit
from .procedural_petiole_controlled_catalogue_v2 import load_for_inspection
from .procedural_petiole_controlled_v2 import code_bindings
from .native848_joint_ownership_v1 import candidate_stub, mesh_seam, MESH_SEAM_TOLERANCE_M

SCHEMA='greenhouse.native848_controlled_9mm_mesh_diagnostic.v1'
VERSION='source_frame_controlled_rigid_leaf_static.v2'


def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def read_pin(value):
    path=Path(value['path']).resolve()
    require(sha256(path)==value['sha256'],'Changed diagnostic input')
    return read_json(path)


def points_and_triangles(report,key):
    c=report['components'][key];path=safe_asset(Path(report['manifest_path']).parent,c['file'])
    require(sha256(path)==c['asset_sha256'],'Changed measured component')
    stage=Usd.Stage.Open(str(path));require(bool(stage),'Unreadable measured USD')
    require(all(l.anonymous or Path(l.realPath).resolve()==path for l in stage.GetUsedLayers()),
            'Self-contained measured component required')
    points,triangles=mesh_arrays(stage)
    origin=np.asarray(c['translation_plant_m'])
    return points+origin,triangles+origin


def radial_samples(report,key):
    c=report['components'][key];_,tri=points_and_triangles(report,key)
    chain,arc,_,_=_oriented_chain(c,1e-6);rows=[]
    for distance in np.linspace(.009,.019,11):
        sample=_sample(chain,arc,float(distance),c['translation_plant_m'])
        center=np.asarray(sample['point_plant_m']);tangent=np.asarray(sample['tangent_plant'])
        a=np.cross(tangent,np.eye(3)[np.argmin(np.abs(tangent))]);a/=np.linalg.norm(a);b=np.cross(tangent,a)
        hits=[first_ray_hit(tri,center,math.cos(theta)*a+math.sin(theta)*b)
              for theta in np.linspace(0,2*math.pi,64,endpoint=False)]
        finite=[v for v in hits if v is not None]
        rows.append(dict(arc_m=float(distance),center_plant_m=center.tolist(),tangent_plant=tangent.tolist(),
            declared_radius_m=sample['petiole_radius_m'],radial_hits_m=hits,hit_count=len(finite),
            minimum_hit_m=min(finite) if finite else None,maximum_hit_m=max(finite) if finite else None))
    return dict(scope='64_radial_rays_per_arc_on_actual_authored_fan_triangulated_mesh',samples=rows,
        nominal_arc_m=.009,support_arc_m=[.009,.019],support_is_cut_interval=False,
        radius_is_measured_from_centerline=False,finite_rays_are_not_continuous_radius_certificate=True,
        native_renderer_tessellation_verified=False,physical_radius_admission_approved=False)


def shared_junction(donor,generated,key):
    old=donor['components'][key];new=generated['components'][key]
    source_points,_=points_and_triangles(donor,key);new_points,_=points_and_triangles(generated,key)
    chain,_,_,_=_oriented_chain(old,1e-6);center=np.asarray(old['attachment_plant_m'])
    tangent=np.asarray(chain[1][:3])-chain[0][:3];tangent/=np.linalg.norm(tangent);radius=chain[0][3]
    ring=dict(attachment_plant_m=center.tolist(),petiole_tangent_plant=tangent.tolist(),attachment_radius_m=radius)
    candidates=[old['parent']];ancestor=donor['components'][old['parent']]['parent']
    if ancestor is not None:candidates.append(ancestor)
    nearby=source_points[np.linalg.norm(source_points-center,axis=1)<=radius+.001];rows=[]
    for parent in candidates:
        a,_=points_and_triangles(donor,parent);current_parent,_=points_and_triangles(generated,parent)
        require(np.array_equal(a,current_parent),'Protected junction stem mesh changed')
        a=a[np.linalg.norm(a-center,axis=1)<=radius+.001]
        source_distance=np.linalg.norm(a[:,None,:]-nearby[None,:,:],axis=2).min(axis=1) if len(a) and len(nearby) else np.array([])
        shared=np.unique(np.round(a[source_distance<=MESH_SEAM_TOLERANCE_M],9),axis=0) if len(source_distance) else np.empty((0,3))
        distances=np.linalg.norm(shared[:,None,:]-new_points[None,:,:],axis=2).min(axis=1) if len(shared) else np.array([])
        rows.append(dict(parent_component_id=parent,source_shared_unique_vertices=len(shared),
            original_shared_vertices_plant_m=shared.tolist(),generated_nearest_vertex_distances_m=distances.tolist(),
            generated_retained_shared_vertex_count=int(np.sum(distances<=MESH_SEAM_TOLERANCE_M)),
            maximum_generated_shared_vertex_error_m=float(distances.max()) if len(distances) else None,
            minimum_generated_shared_vertex_error_m=float(distances.min()) if len(distances) else None,
            frozen_source_ring_proof=mesh_seam(a,source_points,ring),
            frozen_generated_ring_proof=mesh_seam(current_parent,new_points,ring)))
    usable=[r for r in rows if r['source_shared_unique_vertices']>=6]
    preserved=bool(usable) and all(r['generated_retained_shared_vertex_count']==r['source_shared_unique_vertices'] for r in usable)
    return dict(source_attachment_preserved=old['attachment_plant_m']==new['attachment_plant_m'],
        source_candidate_stub=candidate_stub(donor,key),generated_candidate_stub=candidate_stub(generated,key),
        shared_vertex_tolerance_m=MESH_SEAM_TOLERANCE_M,comparisons=rows,
        authored_shared_junction_vertex_correspondence_preserved=preserved,
        topology_scope='declared_parent_and_immediate_mainstem_ancestor_only',
        vertex_correspondence_is_not_watertightness_certificate=True,
        evidence_of_lost_shared_junction=bool(usable) and not preserved)


def reconstruct(qualification_pin):
    q=read_pin(qualification_pin);directory=Path(qualification_pin['path']).resolve().parent
    require(q['version']==VERSION and q['split']=='train' and q['source_family']==q['split_group'],
            'Typed unchanged TRAIN controlled receipt required')
    require(q['explicit_control']['amplitude_m']==.025 and len(q['recipes'])==1,'Exact fixed25mm control required')
    catalogue=load_for_inspection(directory,q['source_plan_path'])
    require(len(catalogue['rows'])==1 and catalogue['rejected']==[],'Frozen controlled catalogue replay held')
    generated=catalogue['report'];donor=audit_manifest(q['source_manifest_path'])
    key=q['recipes'][0]['component_id'];modified=set(catalogue['protected_attribute_proof']['changed_visible_components'])
    checks={r['component_id']:r for r in catalogue['protected_attribute_proof']['component_attribute_checks']}
    unchanged=sorted(cid for cid,c in generated['components'].items() if c['type']=='sub_stem'
        and cid not in modified and checks[cid]['protected_visible_attributes_exactly_preserved'] is True
        and checks[cid]['changed_visible_attributes']==[])
    bindings=dict(q['source_bindings'])
    bindings.update({str(safe_asset(directory,p)):h for p,h in q['output_hashes'].items()})
    bindings.update({str(Path(__file__).with_name(name).resolve()):h for name,h in code_bindings().items()})
    bindings.update({str(Path(qualification_pin['path']).resolve()):qualification_pin['sha256'],
        str(Path(q['source_plan_path']).resolve()):q['source_plan_sha256'],str(Path(__file__).resolve()):sha256(__file__),
        str(Path(__file__).with_name('native848_joint_ownership_v1.py').resolve()):sha256(Path(__file__).with_name('native848_joint_ownership_v1.py'))})
    evidence=dict(schema=SCHEMA,qualification=qualification_pin,source_family=q['source_family'],
        geometry_source_id=q['variant_id'],donor_manifest=pin(q['source_manifest_path']),generated_manifest=pin(directory/'manifest.json'),
        component_id=key,modified_component_ids=sorted(modified),radius_qualified_component_ids=unchanged,
        radius_qualification_basis='unmodified_petioles_only_exact_frozen_visible_attribute_and_metadata_replay',
        modified_radius_qualified_component_ids=[],declared_radius_substituted=False,
        donor_surface=radial_samples(donor,key),generated_surface=radial_samples(generated,key),
        attachment_seam=shared_junction(donor,generated,key),
        modified_current9mm_geometry_qualified=False,
        hold_reasons=['modified_physical_radius_not_continuously_certified'],
        source_bindings=bindings,native_launched=False,training_approved=False,accepted_training_increment=0)
    if evidence['attachment_seam']['evidence_of_lost_shared_junction']:
        evidence['hold_reasons'].append('authored_shared_junction_vertex_correspondence_lost')
    verify_bindings(bindings)
    return evidence,generated,donor


def authenticate_9mm_evidence(evidence_pin, *, generated_report, donor_report, qualification_pin):
    saved=read_pin(evidence_pin);expected,generated,donor=reconstruct(qualification_pin)
    require(saved==expected,'Current mesh evidence does not replay exactly')
    require(generated_report==generated and donor_report==donor,'Geometry reports differ from pinned generated/donor assets')
    bindings=dict(expected['source_bindings']);bindings[str(Path(evidence_pin['path']).resolve())]=evidence_pin['sha256']
    return dict(evidence=saved,modified_component_ids=expected['modified_component_ids'],
        radius_qualified_component_ids=expected['radius_qualified_component_ids'],source_bindings=bindings,
        modified_current9mm_geometry_qualified=False)
