"""Actual saved geometry for the original-reference batch v2 producer.

This separate, read-only adapter replays the prepared plan and unchanged mesh
extractor. It does not modify a producer inventory, join samples, launch Kit,
select a novelty cutoff, or grant an additional source pool. The named lineage
extensions are derived from actual saved qualification/manifests/meshes. A later
native join must authenticate the batch receipt and exact observed rows first.
"""
from copy import deepcopy
from pathlib import Path

from ..native_original_capture import contracts as oc
from ..native_generated_reference import batch_prepare_v2 as prepare
from . import morphology, morphology_frame_v2 as frame

SCHEMA = 'greenhouse.original_reference_batch_actual_geometry.v2'
_CODE = {str(Path(p).resolve()): oc.sha256(p)
         for p in (__file__, oc.__file__, frame.__file__)}


def _digest(value):
    return oc.digest(oc.canonical(value))


def _derive(plan, packet, qualification, manifest, qualification_sha256):
    """Map an already replayed extraction to exact planned target cases."""
    body = dict(packet); seal = body.pop('sha256')
    oc.require(_digest(body) == seal and packet['schema'] == morphology.SCHEMA
        and packet['catalogue_replayed'] is True
        and packet['frozen_splits'] == oc.FROZEN_SPLITS
        and packet['frozen_splits_sha256'] == _digest(oc.FROZEN_SPLITS),
        'Fresh sealed extraction with frozen donor reservations required')
    family = plan['source_family']
    oc.require(oc.FROZEN_SPLITS.get(family) == 'train'
        and packet['source_family'] == qualification['source_family'] == family,
        'Exact TRAIN donor required')
    components = {c['id']: c for c in manifest['components']}
    oc.require(len(components) == len(manifest['components']), 'Repeated saved component')
    cases = plan['target_cases']; targets = [c['source_row']['target_id'] for c in cases]
    oc.require(targets and len(targets) == len(set(targets)), 'Unique source targets required')
    oc.require(all(r['source_target'] in targets for r in packet['records'])
        and all(h['source_target'] in targets for h in packet['holds']), 'Unexpected extracted target')
    results = []
    for case in cases:
        source, generated = case['source_row'], case['generated_row']
        key, target = source['component_id'], source['target_id']
        oc.require(target == family+'/'+key == case['conservative_view_cap_group']
            == generated['conservative_view_cap_group']
            and generated['target_id'] == case['target_id']
            == Path(plan['variant_directory']).name+'/'+key
            and generated['component_id'] == key, 'Changed target or original source cap')
        rows = [r for r in packet['records'] if r['source_target'] == target]
        holds = [h for h in packet['holds'] if h['source_target'] == target]
        item = dict(reference_entry_id=case['reference_entry_id'], source_target=target,
            target_id=case['target_id'], original_view_cap_group=target,
            associated_sample_ids=[], extraction_holds=deepcopy(holds),
            state='held_geometry_domain' if holds else 'described_not_native_joined',
            records=[], lineage_extension=None, qualified_geometry=False)
        oc.require((bool(holds) and not rows) or
            (not holds and len(rows) == 2 and {r['kind'] for r in rows} == {'original', 'generated'}),
            'Each target requires a complete original/generated pair or explicit hold')
        if rows:
            original = next(r for r in rows if r['kind'] == 'original')
            new = next(r for r in rows if r['kind'] == 'generated')
            component = components[key]
            oc.require(original['identity']['manifest_sha256'] == source['source_manifest_sha256']
                and new['source_context_id'] == original['context_id']
                and new['target_id'] == case['target_id']
                and new['identity']['mesh_sha256'][key]
                    == qualification['output_hashes'][component['file']],
                'Extracted geometry differs from prepared source/generated target')
            item['lineage_extension'] = dict(
                source_collection_plan=qualification['source_plan_path'],
                source_plan_sha256=qualification['source_plan_sha256'],
                source_manifest_sha256=original['identity']['manifest_sha256'],
                frozen_family_assignments_sha256=packet['frozen_splits_sha256'],
                generated_qualification_sha256=qualification_sha256,
                generated_row_sha256=_digest(generated),
                generated_component_sha256=_digest(component),
                generated_component_asset_sha256=new['identity']['mesh_sha256'][key],
                structural_lineage_checked=True, geometry_derivation_replayed=True)
            for row in rows:
                item['records'].append(dict(context_id=row['context_id'], kind=row['kind'],
                    source_context_id=row['source_context_id'],
                    input_geometry_sha256=row['input_geometry_sha256'],
                    frame_v2=frame.describe(row['input_geometry'])))
        results.append(item)
    return results


def extract_plan_geometry(plan_path, *, plan_sha256):
    """CPU-only full plan/catalogue/mesh replay; no native observations added."""
    oc.bind_all(_CODE)
    plan_path = oc.pin(plan_path, plan_sha256); plan = oc.read_json(plan_path)
    prepare.check_plan(plan)
    directory = Path(plan['variant_directory']).resolve()
    qpath = directory/'qualification.json'
    qpin = plan['source_bindings'][str(qpath)]
    qualification = oc.read_json(oc.pin(qpath, qpin))
    source = oc.pin(qualification['source_plan_path'], qualification['source_plan_sha256'])
    oc.require(plan['source_bindings'].get(str(source)) == qualification['source_plan_sha256'],
        'Source plan is not authenticated by prepared batch')
    manifest = oc.read_json(oc.pin(directory/'manifest.json', qualification['output_hashes']['manifest.json']))
    packet = morphology.extract_output(morphology.OutputPin(directory, qpin, source,
        qualification['source_plan_sha256']),
        component_ids=[c['source_row']['component_id'] for c in plan['target_cases']])
    cases = _derive(plan, packet, qualification, manifest, qpin)
    bindings = oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'],
        packet['bindings'], packet['code_bindings'], _CODE, {str(plan_path): plan_sha256})
    oc.bind_all(bindings)
    result = dict(schema=SCHEMA, state='actual_geometry_ready_for_explicit_native_join',
        plan_path=str(plan_path), plan_sha256=plan_sha256, variant_directory=str(directory),
        source_family=plan['source_family'], frozen_splits=deepcopy(oc.FROZEN_SPLITS),
        extracted_packet=packet, target_cases=cases, source_bindings=bindings,
        counts=dict(targets=len(cases), extracted_records=len(packet['records']),
            extraction_holds=len(packet['holds']),
            frame_holds=sum(r['frame_v2']['descriptor'] is None for c in cases for r in c['records'])),
        native_execution_verified=False, native_sample_associations_verified=False,
        native_capture_increment=0, accepted_training_increment=0, threshold_selected=None,
        training_approved=False, qualified_geometry=False, source_cap_reset=False,
        calibration_validated=False, original_files_modified=False)
    result['sha256'] = _digest(result)
    return result
