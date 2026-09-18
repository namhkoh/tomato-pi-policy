"""Independent saved-buffer replay of the distinct native batch producer.

Every planned pose has an ordered decision. Captures require full native
RGB/ID/optical-Z, mounted FK, hierarchy, metric anatomy, labels and trace replay.
Pre-render rejections remain coverage declarations and never become images.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from . import batch_execution_v2 as ex
from .audit_v1 import world_from_row, expected_catalogue
from ..native_original_capture import contracts as oc


def verify_launch(root, request_path, request_sha256, execution, owner_pin, worker_pin):
    from ..native_dataset.reference_bridge_owner_v1 import verify_owner_row
    owner = oc.read_json(oc.pin(root/'owner_started.json', owner_pin))
    worker = oc.read_json(oc.pin(root/'owned_worker.json', worker_pin))
    oc.require(owner['execution_request_path'] == request_path
        and owner['execution_request_sha256'] == request_sha256
        and type(owner['owner_pid']) is int and owner['owner_pid'] > 0
        and owner['owner_pid'] == worker['owner_pid']
        and type(worker['launcher_pid']) is int and worker['launcher_pid'] > 0
        and worker['command'] == ex.worker_command(request_path, request_sha256, execution),
        'Worker request differs from owned launch declaration')
    resource = owner['resources']; memory, process = resource['memory'], resource['process_inventory']
    oc.require(memory['checked'] is True and memory['allowed'] is True
        and type(memory['commit_headroom_bytes']) is int and memory['commit_headroom_bytes'] >= 20*2**30
        and type(resource['disk_free_bytes']) is int and resource['disk_free_bytes'] >= 60*2**30
        and process['classifications'] and process['blockers'] == [] and process['no_blockers_observed'] is True
        and resource['exclusive_launch_guaranteed'] is False, 'Insufficient declared launch resources')
    raw = resource['raw_owner_classification']; row = raw['metadata']
    oc.require(row['ProcessId'] == owner['owner_pid'] and raw == verify_owner_row(row,
        pid=owner['owner_pid'], executable=row['ExecutablePath']), 'Changed raw owner classification')
    return owner, worker


def review_sample(capture, plan, case, spec, record, reports, generated, bindings):
    from ..native_dataset import inventory as inv
    from ..native_dataset.bundle import SampleReader
    from ..capture_contract import fingerprint, project, depth_evidence
    from ..capture_visibility import interval_visibility, view_quality, component_masks, ORGAN_IDS
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    from .batch_scene_v2 import current_scene_evidence
    from .batch_pose_v2 import verify_pose

    name, row = spec['candidate_id'], case['generated_row']
    folder = oc.safe_file(capture, name)
    reader = SampleReader(folder, expected_bindings={'sample.json': record['sample_sha256'],
        'supervision/label.json': record['label_sha256']}, expected_json=(
        {'supervision/query_trace.json': record['query_trace']} if record['query_trace'] is not None else {}))
    oc.require(reader.manifest is not None, 'Qualified compact storage is required by this executor')
    audited = dict(sample_sha256=record['sample_sha256'], label_sha256=record['label_sha256'],
        rgb_sha256=reader.metadata['files']['inputs/rgb.png']['sha256'])
    meta, label, trace, logical, decoded = inv._read_sample(bindings, folder, record, audited)
    oc.require(('supervision/query_trace.json' in logical) is (trace is not None), 'Unexpected/missing trace payload')
    oc.require(meta['schema_version'] == ex.SAMPLE and meta['state'] == 'fresh_native_reference_batch_pending_replay'
        and meta['sample_id'] == name and meta['training_sample_approved'] is False
        and meta['historical_labels_inherited'] is False and meta['native_instance_backend'] == 'legacy'
        and meta['scene_counts'] == plan['scene_authority']['expected_scene_counts']
        and type(meta['synchronization']['render_budget_subframes']) is int
        and meta['synchronization']['render_budget_subframes'] == 56
        and type(meta['synchronization']['actual_orchestrator_requests']) is int
        and meta['synchronization']['actual_orchestrator_requests'] == 7
        and meta['generated_plant_triangle_refinement'] is True
        and meta['input_policy'] == plan['input_policy'] and meta['requested_spec'] == spec
        and meta['robot_snapshot'] == record['robot_snapshot'] and meta['calibration'] == record['calibration'],
        'Sample changed bounded native camera/scene policy')
    expected_scene = current_scene_evidence(plan, lighting=meta['lighting'], counts=meta['scene_counts'], renderer=meta['renderer'])
    oc.require(meta['scene_evidence'] == expected_scene and meta['geometry_screen'] == record['screen']
        and meta['geometry_screen']['passed'] is True, 'Actual scene/geometry record mismatch')
    sup, world = meta['supervision'], world_from_row(case, row)
    oc.require(sup['target_id'] == row['target_id'] == record['target_id']
        and sup['source_target_id'] == sup['conservative_view_cap_group'] == case['conservative_view_cap_group']
        and sup['split_group'] == plan['source_family'] and sup['cut_region_proposal'] == row['cut_region_proposal']
        and sup['cut_safety_validated'] is False, 'Different generated target ancestry')
    for key, value in world.items():
        oc.require(np.allclose(sup[key], value, atol=1e-9, rtol=0), 'Metric anatomy/placement changed: '+key)
    verify_pose(meta, case, spec, world['nominal_world_m'])
    ids, identities = reader.array('supervision/renderer_instance_id.npy'), reader.json('supervision/identities.json')
    catalogue = identities['component_catalogue']
    oc.require(catalogue == expected_catalogue(plan, 'generated_variant', reports, generated)
        and len(catalogue) == meta['scene_counts']['components'] and identities['organ_ids'] == ORGAN_IDS,
        'Full generated/background hierarchy differs')
    mapping = {int(k): v for k, v in identities['renderer_id_to_prim'].items()}
    oc.require(meta['native_instance_sha256'] == oc.digest(ids.tobytes())
        and meta['native_mapping_sha256'] == fingerprint(mapping), 'Same-callback identity fingerprints differ')
    components, organs, owners = component_masks(ids, mapping, catalogue)
    rgb, depth, valid = reader.image('inputs/rgb.png'), reader.array('inputs/depth_m.npy'), reader.image('inputs/depth_valid.png') != 0
    target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
    nominal, interval = project([world['nominal_world_m']], meta['calibration'])[0], project(world['interval_world_m'], meta['calibration'])
    radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
    visibility, mask = interval_visibility(nominal, interval, depth, valid, ids, mapping, owners, target, radius)
    quality = view_quality(meta['calibration'], nominal, interval, radius, visibility, rgb, mask)
    old_root = plan['scene_authority']['original_variant']['plant_root']
    old_ids = [i for i, path in mapping.items() if path == old_root or path.startswith(old_root+'/')]
    oc.require(sup['nominal_projected'] == nominal and sup['projected_interval'] == interval
        and sup['depth_evidence'] == depth_evidence(nominal, depth, valid, radius)
        and sup['visibility_evidence'] == visibility and meta['quality'] == quality
        and meta['native_target_pixels'] == int(mask.sum())
        and meta['old_plant_native_pixels'] == int(np.isin(ids, old_ids).sum()) == 0,
        'Stored native visibility/quality/substitution evidence differs')
    expected_label = derive(meta, generated['report'], rgb, depth, valid, components, catalogue)
    expected_trace = trace_review(meta, generated['report'], expected_label, rgb, depth, valid, components, catalogue) if expected_label['eligible'] else None
    oc.require(label == expected_label and trace == expected_trace, 'Saved label/trace differs from fresh replay')
    inv._trace_consistency(label, trace)
    decision = inv.EXCLUDE if not label['eligible'] else inv.STRICT if trace['passed'] else inv.HOLD
    oc.require(record['decision'] == decision, 'Stored annotation decision differs')
    return dict(candidate_id=name, target_id=row['target_id'], decision=decision, reason=label['reason'],
        clarity_reasons=label.get('clarity', {}).get('reasons'), trace_reasons=trace['reasons'] if trace else None,
        sample_sha256=record['sample_sha256'], label_sha256=record['label_sha256'],
        rgb_sha256=logical['inputs/rgb.png']['sha256'], decoded_rgb_sha256=decoded, logical_files=logical,
        label_replayed_exact=True, trace_replayed_exact=trace is not None, native_callback_hashes_verified=True,
        native_ID_masks_replayed=True, **ex.FLAGS), meta


def verify_sequence(samples):
    for a, b in zip(samples, samples[1:]):
        x, y = a['synchronization'], b['synchronization']
        oc.require(y['freshness']['callback_sequence'] > x['freshness']['callback_sequence']
            and all(x['freshness'][k] != y['freshness'][k] for k in ('camera_sha256', 'rgb_sha256', 'depth_sha256')),
            'Stale/repeated native multiview callback')
    return True


def audit_capture(capture, *, result_sha256):
    from .batch_prepare_v2 import check_plan
    from ..collection_plan import load_plan
    from ..native_dataset import inventory as inv
    capture = Path(capture).resolve()
    oc.require(not (capture/'failure.json').exists() and not (capture.parent/'owner_failure.json').exists(),
        'Failed worker/owner cannot become a completed capture')
    result = oc.read_json(oc.pin(capture/'result.json', result_sha256))
    request_path = oc.pin(capture/'request.json', result['request_sha256'])
    request = oc.read_json(request_path)
    execution, plan = ex.preflight_request(request['execution_request_path'], request['execution_request_sha256'])
    verify_launch(capture.parent, request['execution_request_path'], request['execution_request_sha256'], execution,
        request['owner_started_sha256'], request['owned_worker_sha256'])
    oc.require(capture == Path(execution['output'])/'capture'
        and result['schema'] == ex.RESULT and result['state'] == 'bounded_batch_capture_complete_pending_owned_exit'
        and request['schema'] == ex.RESULT and result['plan_sha256'] == request['plan_sha256'] == execution['plan_sha256']
        and result['implementation_bindings'] == execution['implementation_bindings']
        and request['training_started'] is False and result['native_exit_status_must_be_checked_by_owner'] is True
        and request['process_admission']['no_unrelated_kit_process_at_admission'] is True
        and request['host_memory_preflight']['checked'] is True and request['host_memory_preflight']['allowed'] is True
        and request['host_memory_preflight']['commit_headroom_bytes'] >= 20*2**30
        and result['source_assets_unchanged'] is True and result['stage_count'] == result['render_product_count'] == 1
        and all(result[k] is v for k, v in ex.FLAGS.items()), 'Wrong producer/result/admission scope')
    proof = ex.checked_storage(execution)
    oc.require(result['storage_qualification'] == proof and result['storage_backend'] == execution['storage_backend'],
        'Native result storage qualification differs')
    generated = check_plan(plan)
    _, all_reports = load_plan(plan['scene_authority']['clear_plan']['path'])
    reports = {r['plant_id']: r for r in all_reports}
    jobs, records = ex.jobs(plan), result['records']
    oc.require([r['candidate_id'] for r in records] == [s['candidate_id'] for c, s in jobs],
        'Incomplete/reordered batch decisions')
    bindings = inv._Bindings(); bindings.mapping(execution['source_bindings'])
    bindings.mapping({str(capture/'result.json'): result_sha256, str(request_path): result['request_sha256'],
        request['execution_request_path']: request['execution_request_sha256'],
        str(capture.parent/'owner_started.json'): request['owner_started_sha256'],
        str(capture.parent/'owned_worker.json'): request['owned_worker_sha256']})
    reviewed, samples = [], []
    for (case, spec), record in zip(jobs, records):
        name = spec['candidate_id']
        oc.require(record['target_id'] == case['target_id'] and record['requested_spec'] == spec
            and all(record[k] is v for k, v in ex.FLAGS.items()), 'Different batch target/proposal/scope')
        decision_path = capture/(name+'_decision.json')
        oc.require(bindings.document(decision_path, oc.sha256(decision_path)) == record, 'Decision file differs from result')
        if record['state'] in ('rejected_pose', 'rejected_possible_geometry_overlap'):
            oc.require(not (capture/name).exists(), 'Rejected pose unexpectedly contains image payload')
            if record['state'] == 'rejected_pose':
                oc.require(isinstance(record['reason'], str) and record['reason'], 'Missing pose rejection reason')
            else:
                oc.require(record['screen']['passed'] is False, 'Successful geometry cannot be declared rejected')
            reviewed.append(dict(candidate_id=name, target_id=case['target_id'], decision='held_pre_render',
                producer_state=record['state'], reason=record.get('reason', 'static_geometry_screen_failed'),
                geometry_pose_evidence='producer_declaration_not_offline_recomputed', **ex.FLAGS))
        else:
            oc.require(record['state'] == 'native_captured', 'Unknown batch decision')
            item, meta = review_sample(capture, plan, case, spec, record, reports, generated, bindings)
            reviewed.append(item); samples.append(meta)
    verify_sequence(samples)
    oc.require(type(result['captured_frames']) is int and result['captured_frames'] == len(samples), 'Capture count differs')
    actual_folders = sorted(p.name for p in capture.iterdir() if p.is_dir())
    expected_folders = sorted(r['candidate_id'] for r in records if r['state'] == 'native_captured')
    oc.require(actual_folders == expected_folders, 'Unplanned/partial native sample directories')
    bindings.finish(); ex.implementation_bindings(); oc.pin(capture/'result.json', result_sha256)
    return dict(schema=ex.AUDIT, capture=str(capture), result_sha256=result_sha256,
        request_sha256=result['request_sha256'], plan_sha256=execution['plan_sha256'],
        execution_request_path=request['execution_request_path'], execution_request_sha256=request['execution_request_sha256'],
        records=reviewed, counts=dict(Counter(r['decision'] for r in reviewed)),
        planned_count=len(jobs), captured_count=len(samples), ordered_native_callbacks_verified=True,
        source_bindings=dict(sorted(bindings.hashes.items())), implementation_bindings=ex.implementation_bindings(),
        review_method='independent_full_native_ID_Z_RGB_anatomy_bounded_pose_camera_label_trace_replay', **ex.FLAGS)
