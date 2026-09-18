"""Replay explicit source-frame controls and exact protected visible attributes."""
from copy import deepcopy
from pathlib import Path
import json
from .audit import audit_manifest, safe_asset
from .cut_regions import load_rule, propose_cut_region
from .plant_variants import file_hash, digest, load_training_sources, training_envelope, dimensions
from .plant_variant_usd import cut_surface_probe
from .procedural_petiole_geometry import require
from .procedural_petiole_controlled_v1 import VERSION, plan_change_explicit, verify_components, code_bindings


def load_for_inspection(directory, source_plan_path):
    from pxr import Usd
    from .geometry import audit_geometry
    directory = Path(directory).resolve(); source_plan_path = Path(source_plan_path).resolve()
    require(not (directory/'FAILED.json').exists(), 'Held explicit control cannot be loaded')
    receipt_path = directory/'qualification.json'; receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt['version'] == VERSION and receipt['state'] == 'cpu_explicit_control_rigid_leaf_pending_native_review'
        and receipt['variant_id'] == directory.name and receipt['visual_label'] is None
        and all(receipt[k] is False for k in ('training_eligible','native_capture_validated','physics_validated',
            'independent_target_novelty_approved','new_donor_family_created','source_cap_reset','geometry_distances_computed','visual_review_performed')),
        'Explicit unqualified control contract changed')
    require(receipt['source_plan_sha256'] == file_hash(source_plan_path) and receipt['code_sha256'] == code_bindings(), 'Source plan or controlled implementation changed')
    plan,sources = load_training_sources(source_plan_path); family = receipt['source_family']
    require(family in sources and receipt['split'] == 'train' and receipt['split_group'] == family
        and receipt['frozen_family_assignments'] == plan['family_assignments'], 'Frozen TRAIN derivation required')
    source = sources[family]; source_path = Path(source['report']['manifest_path']).resolve()
    require(Path(receipt['source_manifest_path']).resolve() == source_path, 'Wrong source family manifest')
    envelope = training_envelope(plan,sources)
    require(digest(envelope) == receipt['training_envelope_sha256'] and envelope['bounds'] == receipt['training_envelope_bounds'], 'Frozen TRAIN envelope differs')
    for path,sha in receipt['source_bindings'].items():
        require(Path(path).resolve().is_relative_to(source_path.parent) and file_hash(path) == sha, 'Source geometry/texture changed')
    require(receipt['source_bindings'].get(str(source_path)) == file_hash(source_path), 'Unbound donor manifest')
    for component in source['report']['components'].values():
        require(receipt['source_bindings'].get(str(safe_asset(source_path.parent, component['file']))) == component['asset_sha256'], 'Unbound donor mesh')
    for relative,sha in receipt['output_hashes'].items():
        require(file_hash(safe_asset(directory, relative)) == sha, 'Controlled output changed: '+relative)
    require(receipt['output_hashes'].get('manifest.json') == file_hash(directory/'manifest.json')
        and len(receipt['recipes']) == len(receipt['targets']) == 1, 'Exactly one bound target control required')
    saved = receipt['recipes'][0]; key = saved['component_id']
    require(saved['control'] == receipt['explicit_control'], 'Explicit control receipt differs')
    change = plan_change_explicit(source, key, envelope, receipt['explicit_control'])
    require({k:v for k,v in change.items() if k not in ('curve','warp')} == saved, 'Explicit source-frame recipe replay differs')
    raw = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    require(raw['generator'] == VERSION and raw['physics_supported'] is False
        and raw['leaf_transport_policy'] == receipt['leaf_transport_policy'] == 'rigid_leaf_blade.v1', 'Changed static/leaf policy')
    report = audit_manifest(directory/'manifest.json'); require(report['status'] != 'blocked', 'Controlled anatomy audit blocked')
    require(all(receipt['output_hashes'].get(c['file']) == c['asset_sha256'] for c in report['components'].values()), 'Unbound serialized mesh')
    proof,serialized = verify_components(source, change, directory, raw, report)
    require(proof == receipt['protected_attribute_proof'] and serialized == receipt['serialized_geometry_checks'], 'Serialized control/visible protection replay differs')
    geometry = audit_geometry(deepcopy(report))
    require(geometry['maximum_translation_error_m'] <= 1e-6, 'Component assembly frame differs')
    component = report['components'][key]; parent = report['components'][component['parent']]
    measured,_,_ = dimensions(component,parent,load_rule())
    require(measured == receipt['training_envelope_measurements']
        and all(envelope['bounds'][k][0] <= value <= envelope['bounds'][k][1] for k,value in measured.items()), 'Serialized control left TRAIN envelope')
    target = receipt['targets'][0]
    require(target['component_id'] == key and target['source_target_id'] == target['conservative_view_cap_group'] == family+'/'+key
        and target['shape_novelty_pending'] is True and target['training_eligible'] is False, 'Original target pool changed')
    proposal = propose_cut_region(component,parent,load_rule())
    surface = cut_surface_probe(Usd.Stage.Open(str(directory/component['file'])),component,parent)
    require(proposal == target['cut_region_proposal'] and surface == target['cut_surface_probe'], 'Stale actual cut geometry')
    reasons = list(proposal['geometry_warnings'])
    if not surface['passed']: reasons.append('actual_cut_surface_probe_failed')
    reasons.extend(sorted({w['code'] for w in geometry['warnings'] if w['component_id'] in change['members']}))
    row = dict(draft_id='E_'+directory.name+'_'+key, target_id=directory.name+'/'+key, component_id=key,
        variant_id=directory.name, source_plant_id=family, split_group=family,
        label_origin='explicit_control_centerline_not_execution_authority', cut_region_proposal=proposal,
        expected_detached_component_ids=change['members'], attachment_plant_m=component['attachment_plant_m'],
        training_label_approved=False, human_review_performed=False, physical_executability='not_tested',
        conservative_view_cap_group=family+'/'+key)
    textures = {p:h for p,h in receipt['source_bindings'].items() if Path(p).suffix.lower() in ('.png','.jpg','.jpeg')}
    return dict(schema_version='greenhouse.explicit_control_rigid_leaf_inspection_catalogue.v1', directory=str(directory),
        qualification_sha256=file_hash(receipt_path), source_plan_sha256=file_hash(source_plan_path), report=report,
        rows=[] if reasons else [row], rejected=[dict(component_id=key,reasons=reasons)] if reasons else [],
        geometry=geometry, serialized_geometry_checks=serialized, texture_bindings=textures,
        texture_binding_origin='generator_copy_time', source_family=family, split_group=family, variant_id=directory.name,
        state='explicit_control_cpu_replayed_native_visual_collision_review_pending', training_eligible=False,
        collision_validation='not_tested', independent_target_novelty_approved=False, source_cap_reset=False,
        explicit_control=receipt['explicit_control'], protected_attribute_proof=proof)
