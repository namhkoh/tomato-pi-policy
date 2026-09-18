"""Pinned owned batch receipt -> common observed inventory, all decisions kept.

Generated names/content hashes never reset original source caps or prove
independent morphology. A new full duplicate graph and selection remain separate.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from . import batch_execution_v2 as ex
from ..native_original_capture import contracts as oc

POLICY = 'greenhouse.generated_original_batch_observed_adapter.v2'


def replay_receipt(receipt_path, receipt_sha256):
    from .batch_audit_v2 import audit_capture, verify_launch
    receipt_path = oc.pin(receipt_path, receipt_sha256); root = receipt_path.parent
    oc.require(receipt_path.name == 'launcher_receipt.json' and not (root/'owner_failure.json').exists(),
        'Completed known owner receipt required')
    receipt = oc.read_json(receipt_path)
    oc.require(receipt['schema'] == ex.RECEIPT and receipt['state'] == 'owned_exit0_and_independent_saved_native_replay'
        and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
        and receipt['process_history_independently_attested'] is False
        and all(receipt[k] is v for k, v in ex.FLAGS.items()), 'Incomplete or approved producer receipt')
    request, plan = ex.preflight_request(receipt['execution_request_path'], receipt['execution_request_sha256'])
    oc.require(Path(request['output']) == root and Path(receipt['capture']) == root/'capture'
        and receipt['source_bindings'] == request['source_bindings']
        and receipt['implementation_bindings'] == request['implementation_bindings'], 'Receipt targets another executor/output')
    verify_launch(root, receipt['execution_request_path'], receipt['execution_request_sha256'], request,
        receipt['owner_started_sha256'], receipt['owned_worker_sha256'])
    exited = oc.read_json(oc.pin(root/'exit.json', receipt['exit_sha256']))
    oc.require(type(exited['exit_code']) is int and exited['exit_code'] == 0
        and exited['owned_worker_sha256'] == receipt['owned_worker_sha256'], 'Owned exit chain changed')
    saved = oc.read_json(oc.pin(root/'postexit_audit.json', receipt['postexit_audit_sha256']))
    replay = audit_capture(root/'capture', result_sha256=receipt['result_sha256'])
    oc.require(replay == saved and replay['request_sha256'] == receipt['request_sha256'],
        'Independent native replay differs from owner receipt')
    return receipt_path, root, receipt, request, plan, replay


def build_inventory(receipt_path, *, receipt_sha256):
    from ..native_dataset import inventory as inv, original_inventory as oi
    from ..native_original_capture.prepare import case_plan
    from ..native_dataset.bundle import SampleReader
    receipt_path, root, receipt, request, plan, replay = replay_receipt(receipt_path, receipt_sha256)
    b = inv._Bindings(); b.mapping(replay['source_bindings']); b.mapping(ex.implementation_bindings())
    b.mapping({str(receipt_path): receipt_sha256, str(root/'postexit_audit.json'): receipt['postexit_audit_sha256'],
        str(root/'owned_worker.json'): receipt['owned_worker_sha256'], str(root/'exit.json'): receipt['exit_sha256'],
        str(root/'owner_started.json'): receipt['owner_started_sha256']})
    qualification = b.document(Path(plan['variant_directory'])/'qualification.json')
    generated_assets = sorted(h for name, h in qualification['output_hashes'].items() if Path(name).suffix in ('.usd', '.usda', '.usdc'))
    oc.require(generated_assets, 'Actual generated geometry assets required')
    bases = {}
    for case in plan['target_cases']:
        native = case['native_reference']
        original = b.document(native['native_plan']['path'], native['native_plan']['sha256'])
        sample = b.document(native['sample']['path'], native['sample']['sha256'])
        cases = [c for c in original['cases'] if c['case_id'] == sample['sample_id'] and c['source_row'] == case['source_row']]
        oc.require(len(cases) == 1, 'Exact authenticated original source case required')
        basis, lineage = oi._original_basis(b, case_plan(original, cases[0]))
        bases[case['reference_entry_id']] = (basis, lineage)
    result = b.document(root/'capture/result.json', receipt['result_sha256'])
    records = {r['candidate_id']: r for r in result['records']}
    reviewed = {r['candidate_id']: r for r in replay['records']}
    rows, held, contexts = [], [], {}
    for case, spec in ex.jobs(plan):
        name = spec['candidate_id']; review, record = reviewed[name], records[name]
        if review['decision'] == 'held_pre_render':
            held.append(deepcopy(review)); continue
        folder = root/'capture'/name
        reader = SampleReader(folder, expected_bindings={'sample.json': record['sample_sha256'],
            'supervision/label.json': record['label_sha256']}, expected_json=(
            {'supervision/query_trace.json': record['query_trace']} if record['query_trace'] is not None else {}))
        meta = reader.metadata; basis, lineage = bases[case['reference_entry_id']]
        actual = deepcopy(basis)
        actual.update(lighting=deepcopy(meta['lighting']), scene_counts=deepcopy(meta['scene_counts']), renderer=meta['renderer'])
        actual['target_geometry_sha256'] = inv._hash(generated_assets)
        old_root = actual['target_plant_root']
        actual['target_plant_root'] = '/World/GeneratedNativePilot/'+case['generated_row']['variant_id']
        matched = [p for p in actual['population_placements'] if p['plant_root'] == old_root]
        oc.require(len(matched) == 1, 'Missing/ambiguous replaced original population placement')
        matched[0].update(plant_root=actual['target_plant_root'], geometry_content_sha256=inv._hash(generated_assets), asset_count=len(generated_assets))
        scene, camera = oi._features(meta, actual); contexts[inv._hash(actual)] = actual
        target, logical, fresh = case['target_id'], review['logical_files'], meta['synchronization']['freshness']
        row = dict(sample_id=inv._hash(dict(capture=receipt['capture'], candidate_id=name)), candidate_id=name,
            sample_path=str(folder), capture_path=receipt['capture'], source_kind='generated_native',
            target_id=target, variant_target=target, source_family=plan['source_family'], original_donor_family=plan['source_family'],
            source_target=case['source_row']['target_id'], conservative_view_cap_group=case['conservative_view_cap_group'],
            split='train', resolution=list(oc.RESOLUTION), decision=review['decision'], context_id='generated:'+target,
            geometry_sha256=actual['target_geometry_sha256'], image_bytes=logical['inputs/rgb.png']['bytes'],
            encoded_rgb_sha256=review['rgb_sha256'], decoded_rgb_sha256=review['decoded_rgb_sha256'],
            decoded_rgb_bytes=1696*816*3, callback_rgb_sha256=fresh['rgb_sha256'], background_preserved=True,
            annotation_review=dict(passed=review['decision'] == inv.STRICT, method='automatic', evidence_id=receipt_sha256,
                evidence_basis='independently_replayed_new_batch_native_buffers', audit_execution_independently_verified=True,
                prior_audit_execution_verified_from_pin_alone=False),
            provenance=dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha256, plan_path=request['plan_path'],
                plan_sha256=request['plan_sha256'], request_sha256=receipt['request_sha256'], result_sha256=receipt['result_sha256'],
                sample_sha256=record['sample_sha256'], label_sha256=record['label_sha256'], logical_files=logical,
                lineage=deepcopy(lineage), original_source_row=deepcopy(case['source_row']),
                generated_source_row=deepcopy(case['generated_row']), reference_entry_id=case['reference_entry_id'],
                native_reference=deepcopy(case['native_reference']), requested_spec=deepcopy(spec),
                scene_evidence=meta['scene_evidence'], callback=meta['synchronization'],
                native_instance_backend=meta['native_instance_backend'], geometry_screen=meta['geometry_screen'],
                rendered_camera_params=meta['rendered_camera_params'], scene_identity_policy=oi.SCENE_POLICY,
                adapter_policy=POLICY, renderer_settings=dict(status='unknown_not_fully_recorded', value=None),
                process_history_independently_attested=False), **ex.FLAGS)
        oi._identities(row, scene, camera); rows.append(row)
    rows.sort(key=lambda r: r['sample_id'])
    coverage = dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha256,
        capture_path=receipt['capture'], storage_root=receipt['capture'], plan_path=request['plan_path'],
        source_kind='generated_native', planned_rows=plan['maximum_native_frames'], captured_rows=len(rows),
        noncaptured_rows=len(held), held_pre_render=held, requested_target_ids=[c['target_id'] for c in plan['target_cases']],
        historical_training_rows=0, independent_replay_sha256=inv._hash(replay))
    b.protected_roots.update((str(root), str(Path(request['plan_path']).parent), plan['scene_authority']['package'], plan['variant_directory']))
    b.finish(); oc.pin(receipt_path, receipt_sha256)
    oc.require(not (root/'owner_failure.json').exists() and not (root/'capture/failure.json').exists(), 'Failure appeared during inventory')
    packet = dict(schema=inv.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        state='observed_inventory_only_not_global_admission', adapter_policy=POLICY, records=rows, scene_contexts=contexts,
        counts=dict(captured_rows=len(rows), receipts=1, decisions=dict(Counter(r['decision'] for r in rows)),
            source_kinds=dict(Counter(r['source_kind'] for r in rows)),
            original_donor_families=dict(Counter(r['source_family'] for r in rows)),
            conservative_view_cap_groups=dict(Counter(r['source_target'] for r in rows)),
            variant_targets=dict(Counter(r['target_id'] for r in rows))), exact_duplicates=inv._duplicates(rows),
        scope=dict(explicit_receipts=[coverage], extra_observed_roots=[], receipt_discovery_performed=False,
            capture_roots_scanned=False, global_complete=False, prerequisite_observations_implicitly_added=False,
            qualification_controls_implicitly_added=False, observed_split='train', heldout_inventory_included=False,
            historical_training_rows=0, near_image_comparison_performed=False, morphology_equivalence_finalized=False,
            scene_identity_policy=oi.SCENE_POLICY, renderer_settings_used_to_grant_view_identity=False,
            audit_execution_independently_verified=True, label_derivation_replayed=True, trace_derivation_replayed=True,
            callback_validation_scope='independent native ID/RGB/Z/mounted camera replay'),
        provenance=dict(source_bindings=dict(sorted(b.hashes.items())), read_only_source_roots=sorted(b.protected_roots),
            implementation_bindings=ex.implementation_bindings()), training_approved=False, source_cap_reset=False,
        admission_performed=False, original_reviews_modified=False, captures_modified=False, images_generated=False,
        depth_recomputed=False, compact_qualification_granted=False)
    packet['sha256'] = inv._hash(packet)
    return packet
