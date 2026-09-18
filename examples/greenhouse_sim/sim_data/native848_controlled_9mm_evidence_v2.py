"""Authenticate source-identical physical geometry around the current9mm target.

Modified petioles are admitted to the unchanged numeric radius predicates only
when their entire proximal30mm authored surface, centerline/radius function and
source joint correspondence are preserved. This does not certify native
visibility, collisions, independent morphology novelty, or dataset acceptance.
"""
from pathlib import Path
import numpy as np
from .audit import audit_manifest,safe_asset
from .cut_regions import _oriented_chain,_sample
from .dataset_review import require,verify_bindings
from .depth_preview import sha256
from .procedural_petiole_controlled_v3 import VERSION,code_bindings
from .procedural_petiole_controlled_catalogue_v3 import load_for_inspection
from .native848_controlled_9mm_evidence_v1 import pin,read_pin,shared_junction

SCHEMA='greenhouse.native848_controlled_9mm_proximal_identity.v2'


def centerline_identity(donor,generated,key):
    old=donor['components'][key];new=generated['components'][key]
    require(old['translation_plant_m']==new['translation_plant_m']
        and old['attachment_plant_m']==new['attachment_plant_m']
        and old['axis_plant']==new['axis_plant']
        and old['radius_m']==new['radius_m'],'Authored proximal geometry metadata differs')
    a,aa,_,_=_oriented_chain(old,1e-6);b,bb,_,_=_oriented_chain(new,1e-6)
    aa=np.asarray(aa);bb=np.asarray(bb)
    require(aa[-1]>.030 and bb[-1]>.030,'Complete protected source/generated prefix required')
    # Point and radius equality at the union of breakpoints proves equality of
    # their piecewise-linear functions. Tangents at a source bend are one-sided;
    # compare each open interval midpoint plus the actual9/19mm probes instead.
    distances=np.unique(np.r_[aa[aa<.030],bb[bb<.030],0.,.009,.019,.030])
    rows=[]
    for distance in distances:
        x=_sample(a,aa,float(distance),old['translation_plant_m'])
        y=_sample(b,bb,float(distance),new['translation_plant_m'])
        point_error=float(np.max(np.abs(np.asarray(x['point_plant_m'])-y['point_plant_m'])))
        radius_error=abs(x['petiole_radius_m']-y['petiole_radius_m'])
        require(point_error<=1e-14 and radius_error<=1e-15,
            'Proximal centerline/radius function changed beyond arithmetic roundoff')
        rows.append(dict(source_arc_m=float(distance),donor=x,generated=y,maximum_point_error_m=point_error,
            radius_error_m=radius_error,tangent_at_breakpoint_not_used=True))
    breakpoints=np.unique(np.r_[aa[aa<.030],bb[bb<.030],0.,.030])
    # Coalesce duplicate representations of the same geometric knot differing
    # by <=1e-12m; those intervals have no meaningful open tangent sample.
    intervals=[(lo,hi) for lo,hi in zip(breakpoints[:-1],breakpoints[1:]) if hi-lo>1e-12]
    tangent_distances=np.unique(np.r_[[.5*(lo+hi) for lo,hi in intervals],.009,.019])
    tangents=[]
    for distance in tangent_distances:
        x=_sample(a,aa,float(distance),old['translation_plant_m'])
        y=_sample(b,bb,float(distance),new['translation_plant_m'])
        tangent_error=float(np.max(np.abs(np.asarray(x['tangent_plant'])-y['tangent_plant'])))
        require(tangent_error<=1e-13,'Protected open-segment tangent changed')
        tangents.append(dict(source_arc_m=float(distance),donor_tangent=x['tangent_plant'],
            generated_tangent=y['tangent_plant'],maximum_tangent_error=tangent_error))
    return dict(protected_source_arc_m=[0.,.030],all_piecewise_linear_breakpoints_checked=True,
        protected_source_centerline_identity=True,curved_source_prefix_supported=True,
        tangent_policy='each_open_segment_midpoint_plus_nominal9mm_and_support19mm',
        nominal_cut_arc_m=.009,support_end_arc_m=.019,source_radius_definition_unchanged=True,
        arithmetic_point_tolerance_m=1e-14,arithmetic_tangent_tolerance=1e-13,arithmetic_radius_tolerance_m=1e-15,
        rows=rows,tangent_rows=tangents)


def reconstruct(qualification_pin):
    q=read_pin(qualification_pin);directory=Path(qualification_pin['path']).resolve().parent
    require(q['version']==VERSION and q['split']=='train' and q['source_family']==q['split_group'],
        'Typed unchanged TRAIN proximal-preserving receipt required')
    require(q['explicit_control']['amplitude_m']==.025 and len(q['recipes'])==1,'Fixed25mm recipe required')
    catalogue=load_for_inspection(directory,q['source_plan_path'])
    require(len(catalogue['rows'])==1 and catalogue['rejected']==[],'Generated catalogue replay held')
    generated=catalogue['report'];donor=audit_manifest(q['source_manifest_path']);key=q['recipes'][0]['component_id']
    proof=catalogue['proximal_mesh_identity']
    require(proof==q['proximal_mesh_identity'] and proof['protected_source_arc_m']==[0.,.030]
        and proof['protected_source_centerline_identity_verified_separately'] is True
        and proof['complete_intersecting_faces_exact'] is True
        and proof['distal_faces_remain_outside_protected_halfspace'] is True
        and proof['source_mesh_surface_equality_continuous_in_protected_halfspace'] is True,
        'Exact complete proximal surface proof missing')
    center=centerline_identity(donor,generated,key);seam=shared_junction(donor,generated,key)
    require(seam['authored_shared_junction_vertex_correspondence_preserved'] is True
        and not seam['evidence_of_lost_shared_junction'],'Authored shared source joint correspondence lost')
    modified=set(catalogue['protected_attribute_proof']['changed_visible_components'])
    checks={r['component_id']:r for r in catalogue['protected_attribute_proof']['component_attribute_checks']}
    unchanged=sorted(cid for cid,c in generated['components'].items() if c['type']=='sub_stem'
        and cid not in modified and checks[cid]['protected_visible_attributes_exactly_preserved'] is True
        and checks[cid]['changed_visible_attributes']==[])
    require(key in modified and generated['components'][key]['type']=='sub_stem','Missing modified target')
    require([cid for cid in modified if generated['components'][cid]['type']=='sub_stem']==[key],
        'Unexpected second modified petiole')
    bindings=dict(q['source_bindings'])
    bindings.update({str(safe_asset(directory,p)):h for p,h in q['output_hashes'].items()})
    bindings.update({str(Path(__file__).with_name(name).resolve()):h for name,h in code_bindings().items()})
    bindings.update({str(Path(qualification_pin['path']).resolve()):qualification_pin['sha256'],
        str(Path(q['source_plan_path']).resolve()):q['source_plan_sha256'],str(Path(__file__).resolve()):sha256(__file__)})
    for name in ('native848_controlled_9mm_evidence_v1.py','native848_joint_ownership_v1.py'):
        path=Path(__file__).with_name(name).resolve();bindings[str(path)]=sha256(path)
    evidence=dict(schema=SCHEMA,status='qualified_proximal_mesh_identity',qualification=qualification_pin,
        source_family=q['source_family'],split_group=q['split_group'],source_split='train',geometry_source_id=q['variant_id'],
        donor_manifest=pin(q['source_manifest_path']),generated_manifest=pin(directory/'manifest.json'),component_id=key,
        modified_component_ids=sorted(modified),radius_qualified_component_ids=sorted(unchanged+[key]),
        modified_radius_qualified_component_ids=[key],
        radius_qualification_basis='source_identical_complete_proximal_mesh_and_centerline_radius_function',
        declared_radius_substituted=False,physical_radius_reestimated=False,
        proximal_mesh_exact_30mm=True,physical_9mm_source_correspondence=True,shared_junction_preserved=True,
        proximal_mesh_identity=proof,proximal_centerline_identity=center,attachment_seam=seam,
        modified_current9mm_geometry_qualified=True,hold_reasons=[],source_bindings=bindings,
        native_launched=False,training_approved=False,accepted_training_increment=0,
        morphology_diversity_approved=False,new_original_donor_family=False)
    verify_bindings(bindings)
    return evidence,generated,donor


def authenticate_9mm_evidence(evidence_pin,*,generated_report,donor_report,qualification_pin):
    saved=read_pin(evidence_pin);expected,generated,donor=reconstruct(qualification_pin)
    require(saved==expected,'Current proximal evidence does not replay exactly')
    require(generated_report==generated and donor_report==donor,'Generated/donor report mismatch')
    bindings=dict(expected['source_bindings']);bindings[str(Path(evidence_pin['path']).resolve())]=evidence_pin['sha256']
    return dict(evidence=saved,status=expected['status'],modified_component_ids=expected['modified_component_ids'],
        radius_qualified_component_ids=expected['radius_qualified_component_ids'],
        modified_radius_qualified_component_ids=expected['modified_radius_qualified_component_ids'],source_bindings=bindings,
        proximal_mesh_exact_30mm=True,physical_9mm_source_correspondence=True,shared_junction_preserved=True,
        modified_current9mm_geometry_qualified=True)
