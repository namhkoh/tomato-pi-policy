"""Physical-source correspondence for every modified petiole in a v4 variant.

All numeric9mm predicates remain unchanged. Each controlled subtree must have
the complete proximal mesh identity, point/radius/tangent replay, and original
shared-junction correspondence. Unchanged petioles require the full catalogue's
exact visible-attribute and manifest replay. Primary nomination is separate.
"""
from pathlib import Path
from .audit import audit_manifest,safe_asset
from .dataset_review import require,verify_bindings
from .depth_preview import sha256
from .procedural_petiole_controlled_v4 import VERSION,SCHEMA as QUALIFICATION_SCHEMA,code_bindings
from .procedural_petiole_controlled_catalogue_v4 import load_for_inspection
from .native848_controlled_9mm_evidence_v1 import pin,read_pin,shared_junction
from .native848_controlled_9mm_evidence_v2 import centerline_identity

SCHEMA='greenhouse.native848_multisubtree_9mm_proximal_identity.v4'


def reconstruct(qualification_pin):
    q=read_pin(qualification_pin);directory=Path(qualification_pin['path']).resolve().parent
    require(q['schema']==QUALIFICATION_SCHEMA and q['version']==VERSION and q['split']=='train'
        and q['source_family']==q['split_group'],'Typed original TRAIN v4 receipt required')
    catalogue=load_for_inspection(directory,q['source_plan_path'])
    targets=[r['component_id'] for r in q['recipes']]
    require(len(catalogue['rows'])==len(targets) and catalogue['rejected']==[],
        'All controlled targets must replay successfully')
    generated=catalogue['report'];donor=audit_manifest(q['source_manifest_path'])
    proximal=catalogue['proximal_mesh_identity_by_component']
    require(set(proximal)==set(targets) and proximal==q['proximal_mesh_identity_by_component'],
        'Missing per-component proximal proof')
    component_evidence={}
    for key in targets:
        proof=proximal[key]
        require(proof['protected_source_arc_m']==[0.,.030]
            and proof['protected_source_centerline_identity_verified_separately'] is True
            and proof['complete_intersecting_faces_exact'] is True
            and proof['distal_faces_remain_outside_protected_halfspace'] is True
            and proof['source_mesh_surface_equality_continuous_in_protected_halfspace'] is True,
            'Exact complete proximal surface proof missing:'+key)
        center=centerline_identity(donor,generated,key);seam=shared_junction(donor,generated,key)
        require(seam['authored_shared_junction_vertex_correspondence_preserved'] is True
            and not seam['evidence_of_lost_shared_junction'],'Original shared joint lost:'+key)
        component_evidence[key]=dict(proximal_mesh_identity=proof,proximal_centerline_identity=center,
            attachment_seam=seam,physical_9mm_source_correspondence=True,shared_junction_preserved=True,
            radius_qualified=True,primary_nominated=key==q['primary_component_id'])
    protection=catalogue['protected_attribute_proof']
    modified=set(protection['allowed_changed_components'])
    checks={r['component_id']:r for r in protection['component_attribute_checks']}
    require({cid for cid in modified if generated['components'][cid]['type']=='sub_stem'}==set(targets),
        'Modified semantic petiole population differs from evidence')
    unchanged=sorted(cid for cid,c in generated['components'].items() if c['type']=='sub_stem'
        and cid not in modified and checks[cid]['protected_visible_attributes_exactly_preserved'] is True
        and checks[cid]['changed_visible_attributes']==[])
    bindings=dict(q['source_bindings'])
    bindings.update({str(safe_asset(directory,p)):h for p,h in q['output_hashes'].items()})
    bindings.update({str(Path(__file__).with_name(name).resolve()):h for name,h in code_bindings().items()})
    bindings.update({str(Path(qualification_pin['path']).resolve()):qualification_pin['sha256'],
        str(Path(q['source_plan_path']).resolve()):q['source_plan_sha256'],str(Path(__file__).resolve()):sha256(__file__)})
    for name in ('native848_controlled_9mm_evidence_v1.py','native848_controlled_9mm_evidence_v2.py','native848_joint_ownership_v1.py'):
        path=Path(__file__).with_name(name).resolve();bindings[str(path)]=sha256(path)
    result=dict(schema=SCHEMA,status='qualified_proximal_mesh_identity',all_controlled_targets_qualified=True,qualification=qualification_pin,
        source_family=q['source_family'],split_group=q['split_group'],source_split='train',geometry_source_id=q['variant_id'],
        primary_component_id=q['primary_component_id'],primary_source_target_id=q['primary_source_target_id'],
        primary_nomination_is_not_unique_answer_claim=True,
        donor_manifest=pin(q['source_manifest_path']),generated_manifest=pin(directory/'manifest.json'),
        modified_component_ids=sorted(modified),modified_mesh_component_ids=protection['changed_visible_components'],
        radius_qualified_component_ids=sorted(unchanged+targets),modified_radius_qualified_component_ids=sorted(targets),
        radius_qualification_basis='per_target_source_identical_proximal_surface_centerline_radius_and_joint',
        component_evidence=component_evidence,proximal_mesh_exact_30mm=True,physical_9mm_source_correspondence=True,
        shared_junction_preserved=True,declared_radius_substituted=False,physical_radius_reestimated=False,
        modified_current9mm_geometry_qualified=True,hold_reasons=[],source_bindings=bindings,
        native_launched=False,training_approved=False,accepted_training_increment=0,
        morphology_diversity_approved=False,new_original_donor_family=False)
    verify_bindings(bindings)
    return result,generated,donor


def authenticate_9mm_evidence(evidence_pin,*,generated_report,donor_report,qualification_pin):
    saved=read_pin(evidence_pin);expected,generated,donor=reconstruct(qualification_pin)
    require(saved==expected,'Multi-subtree physical evidence does not replay exactly')
    require(generated_report==generated and donor_report==donor,'Generated/donor report mismatch')
    bindings=dict(expected['source_bindings']);bindings[str(Path(evidence_pin['path']).resolve())]=evidence_pin['sha256']
    return dict(evidence=saved,status=expected['status'],modified_component_ids=expected['modified_component_ids'],
        radius_qualified_component_ids=expected['radius_qualified_component_ids'],
        modified_radius_qualified_component_ids=expected['modified_radius_qualified_component_ids'],
        primary_component_id=expected['primary_component_id'],primary_source_target_id=expected['primary_source_target_id'],
        source_bindings=bindings,proximal_mesh_exact_30mm=True,physical_9mm_source_correspondence=True,
        shared_junction_preserved=True,modified_current9mm_geometry_qualified=True)
