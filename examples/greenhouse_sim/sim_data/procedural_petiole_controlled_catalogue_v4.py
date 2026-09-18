"""Replay signed multi-subtree variants without assigning native acceptance."""
from copy import deepcopy
from pathlib import Path
import json
from .audit import audit_manifest,safe_asset
from .cut_regions import load_rule,propose_cut_region
from .plant_variants import file_hash,digest,load_training_sources,training_envelope,dimensions
from .plant_variant_usd import cut_surface_probe
from .procedural_petiole_geometry import require
from .procedural_petiole_controlled_v4 import (VERSION,SCHEMA,STATE,plan_change_explicit,
    verify_components,code_bindings,component_owners)


def load_for_inspection(directory,source_plan_path):
    from pxr import Usd
    from .geometry import audit_geometry
    directory=Path(directory).resolve();source_plan_path=Path(source_plan_path).resolve()
    require(not (directory/'FAILED.json').exists(),'Held control cannot be loaded')
    receipt_path=directory/'qualification.json';q=json.loads(receipt_path.read_text(encoding='utf-8'))
    require(q['schema']==SCHEMA and q['version']==VERSION and q['state']==STATE
        and q['variant_id']==directory.name and q['visual_label'] is None,'Typed unqualified v4 control required')
    require(all(q[k] is False for k in ('training_eligible','native_capture_validated','physics_validated',
        'independent_target_novelty_approved','new_donor_family_created','source_cap_reset',
        'geometry_distances_computed','visual_review_performed')),'Unexpected approval claim')
    require(q['source_plan_sha256']==file_hash(source_plan_path) and q['code_sha256']==code_bindings(),
        'Source plan or generator implementation changed')
    plan,sources=load_training_sources(source_plan_path);family=q['source_family']
    require(family in sources and q['split']=='train' and q['split_group']==family
        and q['frozen_family_assignments']==plan['family_assignments'],'Original TRAIN donor lineage required')
    source=sources[family];source_path=Path(source['report']['manifest_path']).resolve()
    require(Path(q['source_manifest_path']).resolve()==source_path,'Wrong donor manifest')
    envelope=training_envelope(plan,sources)
    require(digest(envelope)==q['training_envelope_sha256'] and envelope['bounds']==q['training_envelope_bounds'],
        'TRAIN envelope changed')
    for path,sha in q['source_bindings'].items():
        require(Path(path).resolve().is_relative_to(source_path.parent) and file_hash(path)==sha,'Changed donor geometry/texture')
    require(q['source_bindings'].get(str(source_path))==file_hash(source_path),'Unbound donor manifest')
    for c in source['report']['components'].values():
        require(q['source_bindings'].get(str(safe_asset(source_path.parent,c['file'])))==c['asset_sha256'],'Unbound source component')
    for relative,sha in q['output_hashes'].items():
        require(file_hash(safe_asset(directory,relative))==sha,'Generated output changed:'+relative)
    require(q['output_hashes'].get('manifest.json')==file_hash(directory/'manifest.json'),'Unbound generated manifest')
    controls=q['explicit_controls'];require(isinstance(controls,list) and controls
        and all(set(c)=={'component_id','control'} for c in controls)
        and len({c['component_id'] for c in controls})==len(controls),'Explicit unique controls required')
    require(len(q['recipes'])==len(q['targets'])==len(controls),'Control/recipe/target cardinality changed')
    changes=[plan_change_explicit(source,c['component_id'],envelope,c['control']) for c in controls]
    owners=component_owners(changes)
    require([{k:v for k,v in c.items() if k not in ('curve','warp')} for c in changes]==q['recipes'],'Recipe replay differs')
    primary=q['primary_component_id']
    require(primary in {c['component_id'] for c in changes} and q['primary_source_target_id']==family+'/'+primary,
        'Primary nomination differs from controlled source identity')
    raw=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    require(raw['generator']==VERSION and raw['physics_supported'] is False
        and raw['leaf_transport_policy']==q['leaf_transport_policy']=='rigid_leaf_blade.v1','Changed static/leaf policy')
    report=audit_manifest(directory/'manifest.json');require(report['status']!='blocked','Generated anatomy audit blocked')
    require(all(q['output_hashes'].get(c['file'])==c['asset_sha256'] for c in report['components'].values()),'Unbound output component')
    proof,serialized,proximal=verify_components(source,changes,directory,raw,report)
    require(proof==q['protected_attribute_proof'] and serialized==q['serialized_geometry_checks']
        and proximal==q['proximal_mesh_identity_by_component'],'Serialized geometry/proximal replay differs')
    geometry=audit_geometry(deepcopy(report))
    require(geometry['maximum_translation_error_m']<=1e-6,'Assembly frame differs')
    rows=[];rejected=[]
    for change,target in zip(changes,q['targets']):
        key=change['component_id'];component=report['components'][key];parent=report['components'][component['parent']]
        measured,_,_=dimensions(component,parent,load_rule())
        require(measured==q['training_envelope_measurements'][key]
            and all(envelope['bounds'][k][0]<=v<=envelope['bounds'][k][1] for k,v in measured.items()),'Output left TRAIN envelope')
        require(target['component_id']==key and target['source_target_id']==target['conservative_view_cap_group']==family+'/'+key
            and target['shape_novelty_pending'] is True and target['training_eligible'] is False,'Original target identity changed')
        proposal=propose_cut_region(component,parent,load_rule())
        surface=cut_surface_probe(Usd.Stage.Open(str(directory/component['file'])),component,parent)
        require(proposal==target['cut_region_proposal'] and surface==target['cut_surface_probe'],'Stale actual cut geometry')
        reasons=list(proposal['geometry_warnings'])
        if not surface['passed']:reasons.append('actual_cut_surface_probe_failed')
        reasons.extend(sorted({w['code'] for w in geometry['warnings'] if w['component_id'] in change['members']}))
        if reasons:rejected.append(dict(component_id=key,reasons=reasons));continue
        rows.append(dict(draft_id='E_'+directory.name+'_'+key,target_id=directory.name+'/'+key,component_id=key,
            variant_id=directory.name,source_plant_id=family,split_group=family,primary_nominated=key==primary,
            label_origin='explicit_control_centerline_not_execution_authority',cut_region_proposal=proposal,
            expected_detached_component_ids=change['members'],attachment_plant_m=component['attachment_plant_m'],
            training_label_approved=False,human_review_performed=False,physical_executability='not_tested',
            conservative_view_cap_group=family+'/'+key))
    return dict(schema_version='greenhouse.multisubtree_controlled_inspection_catalogue.v4',directory=str(directory),
        qualification_sha256=file_hash(receipt_path),source_plan_sha256=file_hash(source_plan_path),report=report,
        rows=rows,rejected=rejected,geometry=geometry,serialized_geometry_checks=serialized,
        texture_bindings={p:h for p,h in q['source_bindings'].items() if Path(p).suffix.lower() in ('.png','.jpg','.jpeg')},
        texture_binding_origin='generator_copy_time',source_family=family,split_group=family,variant_id=directory.name,
        primary_component_id=primary,primary_source_target_id=family+'/'+primary,
        state='cpu_replayed_native_visibility_collision_review_pending',training_eligible=False,collision_validation='not_tested',
        independent_target_novelty_approved=False,source_cap_reset=False,explicit_controls=controls,
        protected_attribute_proof=proof,proximal_mesh_identity_by_component=proximal)
