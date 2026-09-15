"""CPU-only original-native audit adapter and mixed OBSERVED inventory.

OriginalCapturePin(capture_path, result_sha256, launcher_receipt_path,
                   launcher_receipt_sha256)
original_observed_rows(pin) -> (rows, capture_summary, source_bindings)
build_inventory(originals, *, generated_receipts=(), extra_observed_roots=())

Only the pinned campaign_queue.py ``owned_original_worker_exited_and_reaudited``
receipt is supported: exit_code=0, exact request/result/plan, owned_worker
{worker_pid, command, event_path, event_sha256}, post-exit audit and producer
source_bindings. This authenticates that producer's declaration, not an OS
attestation. The original audit is independently replayed again here. No exit
inference from result.json, generic human exit declaration, audit bypass, job,
historical training image, generated qualification or new source budget.

Generated receipts go through inventory._receipt unchanged. Both kinds receive
one versioned, conservative physical comparison key (neutral plant_to_world,
current lighting, assets/population, actual robot/camera, renderer MODE). Full
renderer settings are not recorded by the original collector; they remain
explicitly unknown. Settings/budget/backend never grant a new view identity;
the generated legacy signatures and all recorded settings remain provenance.
These keys are NOT proof of complete render equality or morphology novelty.

All native_captured decisions survive, including holds/exclusions. Pre-render
holds stay in coverage only. Existing common schema/false approvals are retained;
the new inventory digest requires a NEW complete near-image graph. No global
coverage, selection, cap reset, geometry qualification or release occurs here.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import inventory as inv
from .bundle import SampleReader, safe_path
from ..native_original_capture import audit as original_audit
from ..native_original_capture import contracts as oc


POLICY = 'greenhouse.original_native_observed_adapter.v1'
SCENE_POLICY = 'greenhouse.neutral_observed_physical_context.v1'
LINEAGE = 'greenhouse.original_native_inventory_lineage.v1'
LAUNCH_STATE = 'owned_original_worker_exited_and_reaudited'
_ROOT = Path(__file__).resolve().parent.parent
_PRODUCER = [_ROOT / 'native_capture_v5/campaign_queue.py', _ROOT / 'native_capture_v4/campaign.py',
             _ROOT / 'native_dataset/native_process_guard.py']
_LOADED_PRODUCER = {str(p.resolve()): inv._file_sha(p) for p in _PRODUCER}
_LOADED_CODE = dict(oc.LOADED_IMPLEMENTATION, **inv._LOADED_CODE, **_LOADED_PRODUCER,
                   **{str(Path(__file__).resolve()): inv._file_sha(__file__)})


@dataclass(frozen=True)
class OriginalCapturePin:
    capture_path: str
    result_sha256: str
    launcher_receipt_path: str
    launcher_receipt_sha256: str


def implementation_bindings():
    """Pins for this loaded adapter, unchanged readers and original replay code."""
    oc.bind_all(_LOADED_CODE)
    return dict(_LOADED_CODE)


def _launcher(bindings, pin):
    inv._require(isinstance(pin, OriginalCapturePin), 'Explicit OriginalCapturePin required')
    capture = inv._path(pin.capture_path)
    receipt_path = inv._path(pin.launcher_receipt_path)
    inv._require(capture.is_dir() and not (capture / 'failure.json').exists(), 'Missing/failed original capture')
    inv._require(receipt_path == capture.parent / 'original_launcher_receipt.json',
                 'Known original launcher receipt location required')
    receipt = bindings.document(receipt_path, pin.launcher_receipt_sha256)
    inv._require(receipt['state'] == LAUNCH_STATE and type(receipt['exit_code']) is int
                 and receipt['exit_code'] == 0 and receipt['training_approved'] is False
                 and receipt['source_cap_reset'] is False, 'Completed exit-0 original launcher receipt required')
    inv._require(inv._path(receipt['capture']) == capture
                 and receipt['result_sha256'] == pin.result_sha256, 'Launcher/capture result pin mismatch')
    bindings.mapping(receipt['source_bindings'])
    for path, expected in _LOADED_PRODUCER.items():
        inv._require(receipt['source_bindings'].get(path) == expected,
                     'Known launcher producer code binding required')
    bindings.mapping(_LOADED_PRODUCER)
    result = bindings.document(capture / 'result.json', pin.result_sha256)
    request = bindings.document(capture / 'request.json', receipt['request_sha256'])
    plan_path = inv._path(receipt['plan_path'])
    inv._require(result['schema'] == request['schema'] == oc.RESULT_SCHEMA
                 and result['state'] == 'original_native_capture_complete_automatically_audited'
                 and result['admission'] == oc.ADMISSION
                 and result['request_sha256'] == receipt['request_sha256']
                 and result['plan_sha256'] == request['plan_sha256'] == receipt['plan_sha256']
                 and inv._path(request['plan_path']) == plan_path, 'Original request/result/plan mismatch')
    inv._require(receipt['source_bindings'].get(str(plan_path)) == receipt['plan_sha256'],
                 'Launcher did not bind the exact original plan')
    plan = bindings.document(plan_path, receipt['plan_sha256'])
    worker = receipt['owned_worker']
    inv._require(type(worker['worker_pid']) is int and worker['worker_pid'] > 0, 'Owned worker PID required')
    command = worker['command']
    inv._require(isinstance(command, list) and len(command) == 9
                 and isinstance(command[0], str) and Path(command[0]).is_absolute()
                 and command[1:] == ['-m', 'sim_data.native_original_capture.collector',
                     '--plan', str(plan_path), '--plan-sha256', receipt['plan_sha256'], '--output', str(capture)],
                 'Owned worker command differs from original capture')
    event_path = inv._path(worker['event_path'])
    inv._require(event_path.parent == capture.parent and inv.re.fullmatch(r'queue_\d{6}.json', event_path.name),
                 'Known owned-worker event location required')
    event = bindings.document(event_path, worker['event_sha256'])
    inv._require(event['state'] == 'original_pilot_running' and event['worker_pid'] == worker['worker_pid']
                 and event['command'] == command and event['bindings'] == receipt['source_bindings']
                 and event['training_approved'] is False and event['source_cap_reset'] is False,
                 'Owned-worker event differs from launcher receipt')
    audit_path = inv._path(receipt['audit_path'])
    inv._require(audit_path == capture.parent / 'original_postexit_audit.json', 'Known post-exit audit location required')
    postexit = bindings.document(audit_path, receipt['audit_sha256'])
    automatic = bindings.document(capture / 'automatic_audit.json', result['automatic_audit_sha256'])
    bindings.mapping(plan['source_bindings'])
    bindings.mapping(plan['implementation_bindings'])
    inv._require(all(receipt['source_bindings'].get(p) == h for p, h in plan['implementation_bindings'].items()),
                 'Original replay/collector code missing from launcher bindings')
    for name, expected in result['smoke_bindings'].items():
        bindings.bind(safe_path(capture, name), expected)
    # The independent implementation verifies 24 splits, original anatomy/pose
    # ancestry, complete ordered cases, all native masks/labels/traces and smoke.
    replay = original_audit.audit_capture(capture, result_sha256=pin.result_sha256)
    inv._require(replay == automatic == postexit, 'Original post-exit audit differs from independent replay')
    inv._require(plan['schema'] == oc.SCHEMA and plan['split'] == 'train'
                 and plan['family_assignments'] == oc.FROZEN_SPLITS
                 and oc.FROZEN_SPLITS.get(plan['source_family']) == 'train'
                 and plan['geometry_mode'] == 'unmodified_original' and plan['admission'] == oc.ADMISSION,
                 'Frozen original TRAIN-only ancestry required')
    bindings.protected_roots.update((str(capture), str(capture.parent), str(plan_path.parent), plan['package']))
    bindings.protected_roots.update(str(Path(p).parent) for p in plan['source_bindings'])
    return capture, receipt, result, request, plan, replay


def _neutral_basis(basis):
    """Normalize authenticated legacy generated evidence, without relabeling it."""
    names = ('background_scene_asset_sha256', 'background_asset_content_sha256', 'background_asset_count',
             'material_texture_content_sha256', 'material_texture_count', 'population_placements',
             'excluded_external_prop_roots')
    return dict(schema=SCENE_POLICY, **{k: deepcopy(basis[k]) for k in names},
        target_plant_root=basis['replaced_plant_root'], target_geometry_sha256=basis['generated_geometry_sha256'],
        lighting=deepcopy(basis['original_lighting']), scene_counts=deepcopy(basis['original_scene_counts']),
        renderer=basis['original_renderer'])


def _original_basis(bindings, plan):
    prior = bindings.document(Path(plan['pose_prior']['source_capture']) / 'manifest.json')
    package = inv._path(plan['package'])
    assets = prior['source_usd_sha256']
    bindings.mapping(assets)
    clear = bindings.document(plan['source_collection_plan'])
    jobs = [j for j in clear['jobs'] if j['plant_family'] == plan['source_family']]
    inv._require(len(jobs) == 1, 'Exact original donor job required')
    manifest_path = inv._path(jobs[0]['source_manifest_path'])
    inv._require(manifest_path == package / 'plants/components' / plan['source_family'] / 'manifest.json',
                 'Original component manifest escaped donor')
    geometry = bindings.document(manifest_path)
    components = inv._unique(geometry['components'], 'id')
    hashes = {key: bindings.hashes[str(safe_path(manifest_path.parent, c['file']))] for key, c in components.items()}
    physical = []
    for key, c in components.items():
        inv._require(c.get('parent') is None or c['parent'] in components, 'Unknown original component parent')
        physical.append(dict(asset_sha256=hashes[key], parent_asset_sha256=hashes.get(c.get('parent')),
            geometry={k: v for k, v in c.items() if k not in ('id', 'parent', 'file')}))
    placements = []
    for row in plan['scene_variants']:
        root = package / 'plants/components' / row['source_plant_id']
        placed = sorted(h for p, h in assets.items() if inv._path(p).is_relative_to(root))
        inv._require(placed and not row['added_components'] and not row['added_component_paths'],
                     'Complete unmodified original population required')
        placements.append(dict(plant_root=row['plant_root'], geometry_content_sha256=inv._hash(placed), asset_count=len(placed)))
    scene = inv._path(prior['scene'])
    inv._require(str(scene) in assets and placements, 'Pinned full background scene required')
    # Same suffix/content accounting as the frozen generated inventory; any
    # additional appearance files remain fully bound, not silently discarded.
    textures = sorted(h for p, h in plan['source_bindings'].items()
        if Path(p).suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr', '.hdr', '.dds', '.mdl', '.mtl')
        and Path(p).is_relative_to(package))
    basis = dict(schema=SCENE_POLICY, background_scene_asset_sha256=assets[str(scene)],
        background_asset_content_sha256=inv._hash(sorted(assets.values())), background_asset_count=len(assets),
        material_texture_content_sha256=inv._hash(textures), material_texture_count=len(textures),
        population_placements=sorted(placements, key=inv._canonical), target_plant_root=plan['original_variant']['plant_root'],
        target_geometry_sha256=inv._hash(sorted(physical, key=inv._canonical)),
        excluded_external_prop_roots=sorted(plan['excluded_external_roots']))
    ancestry = dict(schema=LINEAGE, source_collection_plan=plan['source_collection_plan'],
        source_plan_sha256=plan['source_collection_plan_sha256'], original_job_id=plan['collection_job_id'],
        frozen_family_assignments=deepcopy(plan['family_assignments']),
        frozen_family_assignments_sha256=inv._hash(plan['family_assignments']),
        source_manifest_path=str(manifest_path), source_manifest_sha256=bindings.hashes[str(manifest_path)],
        original_component_asset_sha256=hashes, original_component_parents={k: c.get('parent') for k, c in components.items()},
        geometry_mode='unmodified_original', historical_training_rows=0)
    return basis, ancestry


def _features(meta, basis):
    # Reuse the frozen camera/pose extraction; no generated plan/qualification
    # or source identity is fabricated. Only its matrix field name is neutralized.
    legacy = dict(basis, original_lighting=meta['lighting'], original_scene_counts=meta['scene_counts'],
                  original_renderer=meta['renderer'])
    scene, camera = inv._features(meta, legacy)
    scene['static_scene_sha256'] = inv._hash(basis)
    scene['plant_to_world'] = scene.pop('generated_plant_to_world')
    return scene, camera


def _identities(row, scene, camera):
    row.update(scene_identity_basis=scene, camera_identity_basis=camera,
               scene_sha256=inv._hash(scene), camera_sha256=inv._hash(camera))
    row['scene_camera_sha256'] = inv._hash(dict(scene=row['scene_sha256'], camera=row['camera_sha256']))


def _original(bindings, pin, absent):
    capture, receipt, result, request, plan, replay = _launcher(bindings, pin)
    absent.add(capture / 'failure.json')
    cases, captured, audited = plan['cases'], result['capture']['records'], replay['records']
    ids = [c['case_id'] for c in cases]
    inv._require(ids == list(inv._unique(cases, 'case_id')) and len(ids) == plan['sample_count_limit']
                 and ids == [r['case_id'] for r in captured] == [r['case_id'] for r in audited],
                 'Missing/reordered/duplicate original cases')
    inv._require(replay['counts'] == dict(Counter(r['decision'] for r in audited)), 'Original audit counts differ')
    basis, ancestry = _original_basis(bindings, plan)
    rows, held, contexts, sequences, rgb_seen = [], [], {}, [], set()
    for case, record, review in zip(cases, captured, audited):
        target = case['target_id']
        inv._require(case['source_row']['target_id'] == case['conservative_view_cap_group'] == target
                     and record['target_id'] == review['target_id'] == target
                     and target.startswith(plan['source_family'] + '/'), 'Requested original target/cap differs')
        folder = safe_path(capture, case['case_id'])
        if record['state'] == 'held_pre_render':
            inv._require(review['decision'] == 'held_pre_render' and review['reason'] == record['reason']
                         and isinstance(record['reason'], str) and record['reason'] and not folder.exists(),
                         'Unexplained pre-render hold')
            absent.add(folder)
            held.append(deepcopy(review))
            continue
        inv._require(record['state'] == 'native_captured', 'Unexpected original capture state')
        absent.add(folder / 'bundle.json')
        inv._require(not (folder / 'bundle.json').exists(), 'Original raw-only audit cannot authenticate compact storage')
        expected = {'sample.json': record['sample_sha256'], 'supervision/label.json': record['label_sha256']}
        if record['trace_sha256'] is not None:
            expected['supervision/query_trace.json'] = record['trace_sha256']
        else:
            absent.add(folder / 'supervision/query_trace.json')
        reader = SampleReader(folder, expected_bindings=expected)
        logical = {}
        for name in sorted(set(reader.metadata['files']) | expected.keys()):
            raw = reader.read(name)
            if name.endswith('.json'):
                inv._parse(raw)
            logical[name] = dict(sha256=inv.digest(raw), bytes=len(raw))
            bindings.bind(safe_path(folder, name), inv.digest(raw))
        meta, label = reader.metadata, reader.json('supervision/label.json')
        trace = reader.json('supervision/query_trace.json') if record['trace_sha256'] is not None else None
        inv._require(meta['schema_version'] == oc.SAMPLE_SCHEMA and meta['sample_id'] == case['case_id']
                     and meta['pose_prior'] == case['pose_prior'] and meta['historical_labels_inherited'] is False
                     and meta['training_sample_approved'] is False and label['training_approved'] is False,
                     'Fresh original sample/pose-prior provenance required')
        inv._require(type(label['eligible']) is bool and (trace is None or type(trace['passed']) is bool),
                     'Boolean native label/trace decisions required')
        passed = label['eligible'] and trace is not None and trace['passed']
        decision = inv.STRICT if passed else inv.HOLD if label['eligible'] else inv.EXCLUDE
        inv._require(decision == review['decision'] and review['label_replayed_exact'] is True
                     and review['trace_replayed_exact'] is (trace is not None)
                     and review['native_callback_hashes_verified'] is True and review['native_ID_masks_replayed'] is True,
                     'Original independently replayed annotation decision differs')
        rgb = reader.image('inputs/rgb.png')
        inv._require(rgb.dtype == np.uint8 and rgb.shape == (816, 1696, 3), 'Native RGB1696 required')
        fresh = meta['synchronization']['freshness']
        inv._require(type(fresh['callback_sequence']) is int and fresh['callback_sequence'] > 0
                     and fresh['rgb_sha256'] == inv.digest(rgb.tobytes())
                     and fresh['camera_sha256'] == inv.fingerprint(meta['calibration']), 'Stale original callback')
        sequences.append(fresh['callback_sequence'])
        decoded = inv.decoded_rgb_digest(rgb.tobytes(), width=1696, height=816)
        inv._require(decoded not in rgb_seen, 'Repeated native RGB within original batch')
        rgb_seen.add(decoded)
        actual_basis = dict(basis, lighting=deepcopy(meta['lighting']), scene_counts=deepcopy(meta['scene_counts']),
                            renderer=meta['renderer'])
        scene, camera = _features(meta, actual_basis)
        contexts[inv._hash(actual_basis)] = actual_basis
        lineage = dict(deepcopy(ancestry), original_source_row=deepcopy(case['source_row']),
            original_source_row_sha256=inv._hash(case['source_row']), requested_target_id=target,
            conservative_view_cap_group=target, pose_prior=deepcopy(case['pose_prior']))
        row = dict(sample_id=inv._hash(dict(capture=str(capture), candidate_id=case['case_id'])),
            candidate_id=case['case_id'], sample_path=str(folder), capture_path=str(capture),
            source_kind='original_native', target_id=target, variant_target=target,
            source_family=plan['source_family'], original_donor_family=plan['source_family'],
            source_target=target, conservative_view_cap_group=target, split='train', resolution=list(oc.RESOLUTION),
            decision=decision, context_id='original:' + target, geometry_sha256=basis['target_geometry_sha256'],
            image_bytes=logical['inputs/rgb.png']['bytes'], encoded_rgb_sha256=logical['inputs/rgb.png']['sha256'],
            decoded_rgb_sha256=decoded, decoded_rgb_bytes=1696 * 816 * 3, callback_rgb_sha256=fresh['rgb_sha256'],
            annotation_review=dict(passed=passed, method='automatic', evidence_id=pin.launcher_receipt_sha256,
                evidence_basis='pinned_owned_exit0_and_independent_original_native_audit_replay',
                audit_execution_independently_verified=True),
            provenance=dict(receipt_path=str(inv._path(pin.launcher_receipt_path)), receipt_sha256=pin.launcher_receipt_sha256,
                request_sha256=receipt['request_sha256'], result_sha256=pin.result_sha256,
                plan_path=receipt['plan_path'], plan_sha256=receipt['plan_sha256'],
                sample_sha256=record['sample_sha256'], label_sha256=record['label_sha256'], trace_sha256=record['trace_sha256'],
                logical_files=logical, callback=deepcopy(meta['synchronization']), lineage=lineage,
                native_instance_backend=meta['native_instance_backend'], native_instance_sha256=meta['native_instance_sha256'],
                native_mapping_sha256=meta['native_mapping_sha256'], rendered_camera_params=deepcopy(meta['rendered_camera_params']),
                geometry_screen=deepcopy(meta['geometry_screen']), owned_worker=deepcopy(receipt['owned_worker']),
                process_admission=deepcopy(request['process_admission']), audit_review_method=replay['review_method'],
                automatic_audit_sha256=result['automatic_audit_sha256'], postexit_audit_sha256=receipt['audit_sha256'],
                independent_replay_sha256=inv._hash(replay), audit_declarations=deepcopy(review),
                trace_evidence_basis='fresh_original_saved_native_query_trace_replay',
                renderer_settings=dict(status='unknown_not_recorded_by_original_collector', value=None),
                scene_identity_policy=SCENE_POLICY, adapter_policy=POLICY, admission=deepcopy(oc.ADMISSION)),
            training_approved=False, source_cap_reset=False, background_preserved=True,
            independent_geometry_qualification=False, depth_recomputed=False)
        _identities(row, scene, camera)
        rows.append(row)
    inv._require(sequences == sorted(set(sequences)), 'Stale/repeated original callback sequence')
    source = dict(receipt_path=str(inv._path(pin.launcher_receipt_path)), receipt_sha256=pin.launcher_receipt_sha256,
        capture_path=str(capture), storage_root=str(capture), plan_path=receipt['plan_path'], source_kind='original_native',
        planned_rows=len(cases), captured_rows=len(rows), noncaptured_rows=len(held), held_pre_render=held,
        requested_target_ids=[c['target_id'] for c in cases], plan_coverage=deepcopy(plan['coverage']),
        historical_training_rows=0, independent_replay_sha256=inv._hash(replay), scene_contexts=contexts)
    return rows, source


def _finish(bindings, absent):
    bindings.finish()
    inv._require(all(not p.exists() for p in absent), 'Failure, held capture or unsupported storage appeared during inventory')


def original_observed_rows(pin):
    """Authenticate one completed original capture; return rows, summary, pins."""
    try:
        bindings, absent = inv._Bindings(), set()
        bindings.mapping(implementation_bindings())
        rows, source = _original(bindings, pin, absent)
        _finish(bindings, absent)
        return rows, source, dict(sorted(bindings.hashes.items()))
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed/missing original native evidence: ' + str(exc)) from exc


def build_inventory(originals, *, generated_receipts=(), extra_observed_roots=()):
    """Return a new common inventory; no I/O writes or external-audit bypass."""
    try:
        return _build(list(originals), list(generated_receipts), list(extra_observed_roots))
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed/missing mixed native evidence: ' + str(exc)) from exc


def _build(originals, generated, extra_roots):
    bindings, absent, rows, coverage, contexts, extras = inv._Bindings(), set(), [], [], {}, []
    implementation = implementation_bindings()
    bindings.mapping(implementation)
    for declared in extra_roots:
        inv._require(isinstance(declared, inv.ObservedRoot), 'Explicit ObservedRoot required')
        root = inv._path(declared.path)
        inv._require(root.is_dir() and str(root) not in {r['path'] for r in extras}, 'Missing/duplicate extra root')
        for pin in declared.receipts:
            inv._require(isinstance(pin, inv.ReceiptPin) and inv._path(pin.storage_root or pin.capture_path).is_relative_to(root),
                         'Extra generated receipt outside declared observed root')
        generated.extend(declared.receipts)
        extras.append(dict(path=str(root), receipt_count=len(declared.receipts),
            status='explicit_receipts_only' if declared.receipts else 'unindexed_no_receipts',
            recursively_scanned=False, complete=False))
    inv._require(originals or generated, 'At least one explicit original or generated capture required')
    seen = set()
    for pin in [*originals, *generated]:
        inv._require(isinstance(pin, (OriginalCapturePin, inv.ReceiptPin)), 'Explicit capture pin required')
        key = str(inv._path(pin.capture_path))
        inv._require(key not in seen, 'Duplicate capture would double-count rows')
        seen.add(key)
    for pin in originals:
        part, source = _original(bindings, pin, absent)
        contexts.update(source.pop('scene_contexts'))
        rows.extend(part); coverage.append(source)
    for pin in generated:
        inv._require(isinstance(pin, inv.ReceiptPin), 'Explicit generated ReceiptPin required')
        part, source = inv._receipt(bindings, pin)  # Frozen generated authentication unchanged.
        legacy = source.pop('scene_context_basis')
        if legacy is not None:
            basis = _neutral_basis(legacy)
            contexts[inv._hash(basis)] = basis
            for row in part:
                row['provenance']['legacy_scene_identity'] = dict(scene_context_basis=deepcopy(legacy), **{
                    k: deepcopy(row[k]) for k in ('scene_identity_basis', 'camera_identity_basis',
                                                'scene_sha256', 'camera_sha256', 'scene_camera_sha256')})
                scene = deepcopy(row['scene_identity_basis'])
                scene['static_scene_sha256'] = inv._hash(basis)
                scene['plant_to_world'] = scene.pop('generated_plant_to_world')
                _identities(row, scene, row['camera_identity_basis'])
                row.update(source_kind='generated_native')
                row['provenance'].update(adapter_policy=POLICY, scene_identity_policy=SCENE_POLICY,
                    renderer_settings=dict(status='recorded_by_generated_source_contract', value=deepcopy(legacy['renderer_settings'])))
        source['source_kind'] = 'generated_native'
        absent.add(inv._path(pin.capture_path) / 'failure.json')
        rows.extend(part); coverage.append(source)
    rows.sort(key=lambda r: r['sample_id'])
    inv._require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate observation identity')
    _finish(bindings, absent)
    result = dict(schema=inv.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        state='observed_inventory_only_not_global_admission', adapter_policy=POLICY, records=rows, scene_contexts=contexts,
        counts=dict(captured_rows=len(rows), receipts=len(coverage), decisions=dict(Counter(r['decision'] for r in rows)),
            source_kinds=dict(Counter(r['source_kind'] for r in rows)),
            original_donor_families=dict(Counter(r['source_family'] for r in rows)),
            conservative_view_cap_groups=dict(Counter(r['source_target'] for r in rows)),
            variant_targets=dict(Counter(r['target_id'] for r in rows))), exact_duplicates=inv._duplicates(rows),
        scope=dict(explicit_receipts=coverage, extra_observed_roots=extras, receipt_discovery_performed=False,
            capture_roots_scanned=False, global_complete=False, observed_split='train', heldout_inventory_included=False,
            historical_training_rows=0, near_image_comparison_performed=False, morphology_equivalence_finalized=False,
            scene_identity_policy=SCENE_POLICY, renderer_settings_used_to_grant_view_identity=False,
            callback_validation_scope='original: independently replayed native ID/RGB/Z/camera; generated: frozen inventory structural checks',
            audit_execution_independently_verified=not generated, label_derivation_replayed=not generated,
            trace_derivation_replayed=not generated, trace_probe_counters_independently_verified=not generated,
            audit_decision_counts_basis='per-row original independent replay or unchanged generated receipt authentication'),
        provenance=dict(source_bindings=dict(sorted(bindings.hashes.items())),
            read_only_source_roots=sorted(bindings.protected_roots), implementation_bindings=implementation),
        training_approved=False, source_cap_reset=False, admission_performed=False, original_reviews_modified=False,
        captures_modified=False, images_generated=False, depth_recomputed=False, compact_qualification_granted=False)
    result['sha256'] = inv._hash(result)
    return result
