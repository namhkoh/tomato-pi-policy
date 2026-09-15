"""Post-owned-exit CPU observer processor. No native launch or capture control.

The owner's new receipt extension is a pinned declaration, NOT OS attestation.
Native annotation replay remains the owner's independent audit responsibility.
"""
from collections import Counter
from pathlib import Path
import argparse
import sys
import time

from . import shadow_observer_v1 as p


def _owned_exit(exit_pin, ledger):
    receipt = p.read_pin(exit_pin, ledger)
    p.keys(receipt, 'schema state exit_code worker launch_event manifest native_request native_result native_failure native_audit outcomes source_bindings flags')
    p.require(receipt['schema'] == p.PREFIX+'owned_exit' and receipt['state'] == 'owned_worker_exited'
        and type(receipt['exit_code']) is int and receipt['flags'] == p.FLAGS, 'Explicit owned exit declaration required')
    p.flags(receipt['flags'])
    p.keys(receipt['worker'], 'pid command')
    command = receipt['worker']['command']
    p.require(type(receipt['worker']['pid']) is int and receipt['worker']['pid'] > 0
        and type(command) is list and len(command) >= 3 and all(type(x) is str for x in command)
        and Path(command[0]).is_absolute() and command[1:3] == ['-m', p.WORKER_MODULE], 'Unknown worker identity/command')
    launch = p.read_pin(receipt['launch_event'], ledger)
    p.keys(launch, 'schema state worker spec native_request producer_bindings flags')
    p.require(launch['schema'] == p.PREFIX+'owned_launch' and launch['state'] == 'owned_worker_running'
        and launch['worker'] == receipt['worker'] and launch['native_request'] == receipt['native_request']
        and launch['flags'] == p.FLAGS, 'Launcher/exit identity mismatch')
    p.flags(launch['flags'])
    spec, contexts, candidates, manifest = p.read_journal(receipt['manifest'], ledger)
    p.require(launch['spec'] == manifest['spec'] and launch['producer_bindings'] == spec['producer_bindings'], 'Launcher used another observer/producer')
    p.bindings_shape(receipt['source_bindings']); p.merge(ledger, receipt['source_bindings'])
    implementation = p.implementation_bindings()
    p.require(spec['implementation_bindings'] == implementation, 'Observer implementation changed since preflight')
    for bindings in (implementation, spec['producer_bindings']):
        p.require(all(receipt['source_bindings'].get(path) == h for path, h in bindings.items()), 'Owner did not bind observer/producer source')
        p.merge(ledger, bindings)
    for key in ('native_request', 'native_result', 'native_failure', 'native_audit', 'outcomes'):
        pin = receipt[key]
        if pin is not None:
            p.pin_shape(pin); p.merge(ledger, {pin['path']: pin['sha256']})
    p.require(receipt['native_request'] is not None, 'Native request required')
    if receipt['exit_code'] == 0:
        p.require(all(receipt[k] is not None for k in ('native_result', 'native_audit', 'outcomes'))
                  and receipt['native_failure'] is None, 'Successful owned exit requires result/audit/outcomes')
    else:
        p.require(receipt['native_failure'] is not None, 'Nonzero exit must preserve failure receipt')
    request = p.read_pin(receipt['native_request'], ledger)
    p.require(request.get('shadow_observer_spec') == manifest['spec']
        and request.get('plan_path') == spec['plan']['path']
        and request.get('plan_sha256') == spec['plan']['sha256'], 'Native request/observer plan mismatch')
    plan = p.read_pin(spec['plan'], ledger)
    p.require(plan.get('schema') == 'greenhouse.generated_from_original_reference_plan.v1'
        and plan.get('split') == 'train', 'Named TRAIN clear-authority bridge plan required')
    p.require([c['mode'] for c in spec['candidates']] == plan['modes']
        and len(spec['candidates']) == plan['sample_count_limit'], 'Observer changed planned candidate modes/count')
    for c in spec['candidates']:
        row = plan['source_row'] if c['mode'] == 'original_control' else plan['generated_row']
        p.require(c['target_id'] == row['target_id'] and c['component_id'] == row['component_id'], 'Observer changed planned target')
    p.require(plan['scene_authority']['clear_plan'] == spec['clear_plan'], 'Clear authority changed')
    proof = p.read_pin(plan['anchor_evidence'], ledger)
    p.require(proof['bank'] == spec['bank'] and proof['entry_id'] == spec['anchor_entry_id'], 'Original bank proof changed')
    p.merge(ledger, plan['source_bindings'])
    for key in ('bank', 'clear_plan'):
        p.merge(ledger, {spec[key]['path']: spec[key]['sha256']})
    for context in contexts.values():
        p.require(Path(context['variant_directory']).resolve() == Path(plan['variant_directory']).resolve(), 'Foreign variant context')
        p.require(all(plan['source_bindings'].get(path) == h for path, h in context['geometry_bindings'].items()), 'Unbound geometry context')
        actual = context['plant_to_world_usd_row_vectors']
        expected = plan['expected_original_world']['plant_to_world_usd_row_vectors']
        p.rigid(expected)
        p.require(all(abs(actual[i][j]-expected[i][j]) < 1e-9 for i in range(4) for j in range(4)), 'Native plant placement changed')
    p.verify(ledger)  # full disk reread before importing any predictor or USD
    return receipt, spec, contexts, candidates, manifest


def _load(context):
    from . import visibility_shadow
    return visibility_shadow.load_partial_plant(context['variant_directory'],
        context['plant_to_world_usd_row_vectors'], expected_bindings=context['geometry_bindings'])


def _predict(snapshot, calibration, component_id):
    from . import visibility_shadow
    return visibility_shadow.inspect_interval(snapshot, calibration, component_id)


def _join(receipt, spec, events, predictions, ledger):
    """Only called AFTER the complete predictions manifest has been published."""
    if receipt['exit_code'] != 0:
        return [dict(candidate_id=c['candidate_id'], native_state='native_failed', comparison='unavailable') for c in spec['candidates']]
    result = p.read_pin(receipt['native_result'], ledger)
    audit = p.read_pin(receipt['native_audit'], ledger)
    p.require(result.get('shadow_observer_manifest') == receipt['manifest']
        and result.get('request_sha256') == receipt['native_request']['sha256'], 'Result/event receipt mismatch')
    p.require(audit.get('shadow_observer_outcomes') == receipt['outcomes']
        and audit.get('result_sha256') == receipt['native_result']['sha256'], 'Post-exit native audit binding mismatch')
    outcomes = p.read_pin(receipt['outcomes'], ledger)
    p.keys(outcomes, 'schema manifest records flags')
    p.require(outcomes['schema'] == p.PREFIX+'native_outcomes' and outcomes['manifest'] == receipt['manifest']
        and outcomes['flags'] == p.FLAGS
        and [r['candidate_id'] for r in outcomes['records']] == [c['candidate_id'] for c in spec['candidates']], 'Outcome inventory mismatch')
    p.flags(outcomes['flags'])
    rows = []
    for candidate, truth, prediction in zip(spec['candidates'], outcomes['records'], predictions):
        p.keys(truth, 'candidate_id state event_seal first_render_request_index sample calibration_sha256 robot_snapshot_sha256 old_decision native_foreground_components')
        event = events[candidate['candidate_id']]
        p.require(truth['event_seal'] == event['seal'], 'Native observation does not bind the pre-render seal')
        p.require(truth['state'] in ('captured', 'geometry_hold', 'capture_error'), 'Unknown native disposition')
        payload = event['payload']
        if truth['state'] == 'captured':
            p.require(payload['kind'] in ('pre_render', 'observer_error'), 'Captured sample conflicts with geometry hold')
            p.pin_shape(truth['sample']); p.merge(ledger, {truth['sample']['path']: truth['sample']['sha256']})
            p.hash_value(truth['calibration_sha256']); p.hash_value(truth['robot_snapshot_sha256'])
            p.require(type(truth['first_render_request_index']) is int, 'Native first payload counter required')
            if payload['kind'] == 'pre_render':
                p.require(truth['first_render_request_index'] == payload['writer_request_index_before']+1
                    and truth['calibration_sha256'] == p.digest(p.canonical(payload['calibration']))
                    and truth['robot_snapshot_sha256'] == payload['robot_snapshot_sha256'], 'Pre-render/native pose or chronology mismatch')
            p.require(truth['old_decision'] in ('accept_strict_automatic_annotation_candidate', 'hold_visual_clarity',
                                               'exclude_geometry_or_visibility'), 'Unknown saved native decision')
        else:
            p.require(truth['sample'] is None and truth['old_decision'] is None, 'Uncaptured row cannot inherit a label')
            if truth['state'] == 'geometry_hold':
                p.require(payload['kind'] in ('geometry_hold', 'observer_error'), 'Native hold/event mismatch')
        components = truth['native_foreground_components']
        p.require(type(components) is list and len(components) == len(set(components)), 'Invalid native component set')
        for component in components: p.identifier(component)
        predicted = set()
        if prediction.get('evidence'):
            evidence = p.read_pin(prediction['evidence'], ledger)
            for probe in evidence['interval']:
                if probe['status'] == 'predicted_blocked':
                    predicted.update(ray['blocker_component'] for ray in probe['rays'] if ray['status']=='predicted_blocked')
        native = set(components)
        rows.append(dict(candidate_id=candidate['candidate_id'], native_state=truth['state'],
            old_decision=truth['old_decision'], prediction=prediction['status'], native_foreground_components=components,
            predicted_blocker_components=sorted(predicted), identity_overlap=sorted(native & predicted),
            identity_exact=bool(predicted) and predicted == native,
            false_block_against_saved_strict=prediction['status']=='predicted_blocked'
                and truth['old_decision']=='accept_strict_automatic_annotation_candidate'))
    return rows


def process(exit_path, *, exit_sha256, output):
    """Only source-bound post-exit input. No producer/audit/predictor callback injection."""
    p.require(not any(k.split('.')[0] in ('omni', 'isaacsim') for k in sys.modules), 'Standalone CPU process required')
    started, cpu_started = time.perf_counter(), time.process_time()
    ledger = {}; exit_pin = dict(path=str(Path(exit_path).resolve()), sha256=exit_sha256)
    receipt, spec, contexts, events, manifest = _owned_exit(exit_pin, ledger)
    target = Path(output).resolve()
    p.require(not target.exists() and target != Path(target.anchor), 'New post-exit output required')
    journal = Path(receipt['manifest']['path']).resolve().parent
    p.require(not target.is_relative_to(journal) and not journal.is_relative_to(target), 'Output overlaps native journal')
    for path in ledger:
        p.require(not Path(path).is_relative_to(target), 'Output overlaps protected source')
    target.mkdir(parents=True, exist_ok=False)
    snapshots, builds, failures, predictions = {}, [], [], []
    try:
        for candidate in spec['candidates']:
            event = events[candidate['candidate_id']]['payload']
            row = dict(candidate_id=candidate['candidate_id'], status='unknown', evidence=None)
            if event['kind'] == 'geometry_hold': row['status'] = 'not_run_geometry_hold'
            elif event['kind'] == 'observer_error': row.update(status='observer_error', code=event['code'])
            elif candidate['mode'] == 'original_control': row['status'] = 'not_run_original_control'
            else:
                context_id = event['context_id']; context = contexts[context_id]
                if context_id not in snapshots:
                    try:
                        snapshot, build = _load(context)
                        snapshots[context_id] = snapshot; builds.append(dict(context_id=context_id, **build))
                    except Exception as exc:
                        snapshots[context_id] = None
                        failures.append(dict(context_id=context_id, phase='geometry_load', error_type=type(exc).__name__, message=str(exc)))
                if snapshots[context_id] is None: row['status'] = 'geometry_error_unknown'
                else:
                    before, cpu_before = time.perf_counter(), time.process_time()
                    try:
                        # Deliberately no RGB/depth/mask/outcome/label or entire metadata argument.
                        evidence = _predict(snapshots[context_id], event['calibration'], candidate['component_id'])
                        row.update(status=evidence['status'], inspect_wall_seconds=time.perf_counter()-before,
                                   inspect_cpu_seconds=time.process_time()-cpu_before)
                        path = target/(candidate['candidate_id']+'.json'); p.write_json_new(path, evidence)
                        row['evidence'] = p.file_pin(path); row['evidence_bytes'] = path.stat().st_size
                        p.merge(ledger, {str(path): row['evidence']['sha256']})
                    except Exception as exc:
                        row['status'] = 'prediction_error_unknown'
                        failures.append(dict(candidate_id=candidate['candidate_id'], phase='prediction', error_type=type(exc).__name__, message=str(exc)))
            predictions.append(row)
        for snapshot in snapshots.values():
            if snapshot is not None: snapshot.finish()
        p.verify(ledger)
        sealed = target/'predictions.json'
        p.write_json_new(sealed, dict(schema=p.PREFIX+'predictions', exit=exit_pin, manifest=receipt['manifest'],
            native_outcomes_joined=False, predictions=predictions, builds=builds, failures=failures, flags=p.FLAGS))
        seal_pin = p.file_pin(sealed); p.merge(ledger, {str(sealed): seal_pin['sha256']})
        comparisons = _join(receipt, spec, events, predictions, ledger)
        p.verify(ledger)
        report = dict(schema=p.PREFIX+'report', state='post_exit_shadow_only', exit=exit_pin,
            predictions=seal_pin, candidate_count=len(predictions), comparison=comparisons,
            prediction_counts=dict(Counter(row['status'] for row in predictions)), failures=failures,
            journal_faults=manifest['faults'], native_exit_code=receipt['exit_code'],
            wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu_started,
            source_and_evidence_bindings=ledger, owned_exit_is_producer_declaration_not_os_attestation=True,
            native_audit_reexecuted=False, native_launched=False, algorithm_tuned=False, **p.FLAGS)
        p.write_json_new(target/'report.json', report)
        return p.file_pin(target/'report.json')
    except Exception as exc:
        p.write_json_new(target/'failure.json', dict(schema=p.PREFIX+'processor_failure',
            error_type=type(exc).__name__, message=str(exc), candidate_count=len(spec['candidates']),
            no_completed_report=True, flags=p.FLAGS))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exit-receipt', required=True); parser.add_argument('--exit-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(p.canonical(process(args.exit_receipt, exit_sha256=args.exit_sha256, output=args.output)).decode())


if __name__ == '__main__': main()
