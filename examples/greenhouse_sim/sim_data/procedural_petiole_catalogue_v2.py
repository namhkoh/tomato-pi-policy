"""Read-only qualification of curved/relocated static perception variants.

Replay the deterministic recipe, verify serialized geometry and source detail,
and recompute labels. This is NOT native visibility, collision or data approval.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import audit_manifest, safe_asset
from .cut_regions import load_rule, propose_cut_region
from .plant_variants import PHYSICS_FIELDS, digest, file_hash, load_training_sources, training_envelope
from .plant_variant_usd import cut_surface_probe, write_json_new
from .procedural_petiole_geometry import require
from .procedural_petiole_v2 import VERSION, plan_change, transform_metadata


from .procedural_petiole_catalogue import check_component
from .procedural_leaf_transport import component_transport


def load_for_inspection(directory, source_plan_path):
    from pxr import Usd
    from .geometry import audit_geometry
    directory, source_plan_path = Path(directory).resolve(), Path(source_plan_path).resolve()
    require(not (directory/'FAILED.json').exists(), 'Failed generation is not usable')
    receipt_path = directory/'qualification.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt.get('version') == VERSION and receipt.get('state') == 'cpu_curved_rigid_leaf_pending_native_review',
            'Unknown or incomplete curved geometry qualification')
    require(receipt['variant_id'] == directory.name, 'Directory identity differs')
    require(all(receipt.get(k) is False for k in ('training_eligible','native_capture_validated',
        'physics_validated','independent_target_novelty_approved','new_donor_family_created')), 'Unqualified approval claim')
    require(receipt['source_plan_sha256'] == file_hash(source_plan_path), 'Source plan changed')
    plan, sources = load_training_sources(source_plan_path)
    family = receipt['source_family']
    require(family in sources and receipt['split'] == 'train' and receipt['split_group'] == family,
            'Only frozen TRAIN source derivation supported')
    require(receipt['frozen_family_assignments'] == plan['family_assignments'], 'Frozen splits changed')
    source = sources[family]
    source_path = Path(source['report']['manifest_path'])
    require(Path(receipt['source_manifest_path']).resolve() == source_path.resolve(), 'Donor identity changed')
    envelope = training_envelope(plan, sources)
    require(digest(envelope) == receipt['training_envelope_sha256'], 'TRAIN-fitted envelope changed')
    for path, sha in receipt['source_bindings'].items():
        require(Path(path).resolve().is_relative_to(source_path.parent.resolve()) and file_hash(path) == sha,
                'Donor mesh/texture changed or escaped family')
    for relative, sha in receipt['output_hashes'].items():
        require(file_hash(safe_asset(directory, relative)) == sha, 'Generated output changed: '+relative)
    for name, sha in receipt['code_sha256'].items():
        require(Path(name).name == name and file_hash(Path(__file__).with_name(name)) == sha,
                'Recipe implementation changed; explicitly regenerate')
    require(receipt['output_hashes'].get('manifest.json') == file_hash(directory/'manifest.json'), 'Unbound manifest')
    raw = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    require(raw.get('generator') == VERSION and raw.get('physics_supported') is False
            and raw.get('leaf_transport_policy') == receipt.get('leaf_transport_policy') == 'rigid_leaf_blade.v1',
            'Unexpected generated manifest or leaf transport policy')
    report = audit_manifest(directory/'manifest.json')
    require(report['status'] != 'blocked', 'Generated anatomy audit blocked')
    old_rows = {c['id']: c for c in source['raw']['components']}
    new_rows = {c['id']: c for c in raw['components']}
    require(set(old_rows) == set(new_rows), 'Changed organ population')
    allowed = {t['component_id'] for t in source['job']['targets']}
    owners, changes = {}, {}
    for recipe in receipt['recipes']:
        key = recipe['component_id']
        require(key in allowed and key not in changes, 'Unexpected/repeated target')
        replay = plan_change(source, key, envelope, recipe['seed'])
        require({k:v for k,v in replay.items() if k not in ('curve','warp')} == recipe, 'Deterministic recipe replay differs')
        changes[key] = replay
        for member in replay['members']:
            require(member not in owners, 'Overlapping subtrees')
            owners[member] = replay
    targets = {t['component_id']: t for t in receipt['targets']}
    require(bool(changes) and set(targets) == set(changes) and len(targets) == len(receipt['targets']), 'Target receipt mismatch')
    serialized = []
    for key, old in old_rows.items():
        change = owners.get(key)
        expected = transform_metadata(old, change) if change else {k:v for k,v in old.items() if k not in PHYSICS_FIELDS}
        require(new_rows[key] == expected, 'Changed centerline/attachment metadata: '+key)
        asset = report['components'][key]
        require(receipt['output_hashes'].get(asset['file']) == asset['asset_sha256'], 'Unbound component')
        qa = check_component(safe_asset(source_path.parent,old['file']), safe_asset(directory,asset['file']),
            np.asarray(old['transform']['translate']), np.asarray(expected['transform']['translate']),
            component_transport(old, change) if change else None)
        serialized.extend(dict(component_id=key, **q) for q in qa)
    geometry = audit_geometry(deepcopy(report))
    require(geometry['maximum_translation_error_m'] <= 1e-6, 'Assembly frame mismatch')
    rows, rejected = [], []
    for key, change in changes.items():
        target = targets[key]
        require(target['source_target_id'] == target['conservative_view_cap_group'] == family+'/'+key
                and target['shape_novelty_pending'] is True and target['training_eligible'] is False,
                'Similarity cap or approval bypass')
        component = report['components'][key]
        parent = report['components'][component['parent']]
        proposal = propose_cut_region(component,parent,load_rule())
        probe = cut_surface_probe(Usd.Stage.Open(str(directory/component['file'])),component,parent)
        require(proposal == target['cut_region_proposal'] and probe == target['cut_surface_probe'], 'Stale cut geometry')
        reasons = list(proposal['geometry_warnings'])
        if not probe['passed']: reasons.append('actual_cut_surface_probe_failed')
        reasons.extend(sorted({w['code'] for w in geometry['warnings'] if w['component_id'] in change['members']}))
        if reasons:
            rejected.append(dict(component_id=key,reasons=reasons))
            continue
        rows.append(dict(draft_id='C_'+directory.name+'_'+key,target_id=directory.name+'/'+key,
            component_id=key,variant_id=directory.name,source_plant_id=family,split_group=family,
            label_origin='curved_generated_centerline_not_execution_authority',cut_region_proposal=proposal,
            expected_detached_component_ids=change['members'],attachment_plant_m=component['attachment_plant_m'],
            training_label_approved=False,human_review_performed=False,physical_executability='not_tested',
            conservative_view_cap_group=family+'/'+key))
    textures = {p:h for p,h in receipt['source_bindings'].items() if Path(p).suffix.lower() in ('.png','.jpg','.jpeg')}
    return dict(schema_version='greenhouse.curved_rigid_leaf_inspection_catalogue.v2',directory=str(directory),
        qualification_sha256=file_hash(receipt_path),source_plan_sha256=file_hash(source_plan_path),
        report=report,rows=rows,rejected=rejected,geometry=geometry,serialized_geometry_checks=serialized,
        texture_bindings=textures,texture_binding_origin='generator_copy_time',source_family=family,
        split_group=family,variant_id=directory.name,state='cpu_replayed_native_visual_collision_review_pending',
        training_eligible=False,collision_validation='not_tested',independent_target_novelty_approved=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=Path,required=True);p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'New receipt only')
    c=load_for_inspection(a.variant,a.source_plan)
    write_json_new(a.output,c)
    print('CURVED_CATALOGUE_CHECKED',c['variant_id'],len(c['rows']),c['rejected'],flush=True)


if __name__=='__main__':main()
