"""Finite read-only recovery of one completed slot from one failed pilot parent.

This does not turn the failed parent into a successful run or approve an image.
Only the exact 14-frame completed schedule can proceed to new annotation and QA.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import re
import sys
from . import native848_bulk_sibling_gate_v5 as gate
from . import native848_data_roots_v1 as roots
from . import native848_pilot_parallel_predecessor_v4 as retirement_validation
from .depth_preview import sha256
from .dataset_review import require

ROOT = Path('D:/research/tomato-pi-policy')
H = ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'
PARENT = Path('C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_unique9mm_pilot_parallel_20260917_v1/four_legacy_ranked_64_candidates_v4')
LAUNCH = ROOT/'data/sim_data/diagnostics/native848_unique9mm_pilot_parallel_launch_20260917_v1/four_legacy_ranked_64_candidates_v4'
OWNER = H/'run_native848_pilot_parallel_v4.py'
OWNER_SHA = '91d6f9456d23768fd152f6ea2982735561849fed64839812df79fb05741a0d4a'
WORKER = Path(__file__).with_name('native848_pilot_sibling_worker_v4.py')
WORKER_SHA = 'd068688e007976bbe22d05bd0d2b41b4d5e9ba26b7a1d9db4b4531c68c468af1'
RESULT_SHA = 'ff1987fbca75d7fcb72e90b0ef6731d5ccd344ade3f242f6a043bbc3107121bb'
METADATA_SHA = 'a44bc48ef690253aabdd0546bbb7e371d3a6a557b40a471e2455513516df59c4'
SCHEMA = 'greenhouse.native848_pilot_failed_parent_recovery.v4'
RECOVERABLE = [0]
EXPECTED_COUNTS = [14, 20, 10, 20]
FAILURE_SHA = 'fd55c5bf42b0fac2d4b12e0e5a769b729107ca20e02ba1df7b62e24a9d490f3f'
SNAPSHOT_NAME = 'process_admission_failure_84892_c7e5eb8f417a421e83ded9f9c098ad8a.json'
SNAPSHOT_SHA = '50f910ca05205a462da9fc851103c2ac03d82b27d82517c428e978b4a1a07c85'
RETIREMENT = Path(__file__).with_name('native848_owned_retirement_v3.py')
RETIREMENT_SHA = 'a340b93b167013e33f4d4346cf872a0b8d81142a2b53fe2580013e0272aa33ef'
RETIREMENT_VALIDATION_SHA = '2e8384109456ace741e1f87912b41663847834c4c70d294c0075e3bdcb69df37'


def check_timeout_snapshot(value, slot, coordinator, recorded_slots):
    require(value['schema'] == 'greenhouse.native848_owned_process_failure_snapshot.v1'
            and value['phase'] == 'completed_slot_retirement'
            and value['error'] == 'Timed out awaiting exit of all exact recorded owned identities'
            and value['bounded_retirement_wait_seconds'] == 60
            and value['coordinator_identity'] == coordinator and value['recorded_slots'] == recorded_slots
            and [s['slot'] for s in recorded_slots] == [0,1]
            and value['process_snapshot_available'] is True and value['process_control_performed'] is False
            and value['same_snapshot_classification'] is True and value['automatic_retry'] is False,
            'Exact finite60 retirement timeout snapshot required')
    expected=gate.owned_allowlist(coordinator,recorded_slots)
    rows=value['process_snapshot'];by={r['ProcessId']:r for r in rows}
    require(len(rows)==len(by), 'Duplicate failure snapshot identity')
    require(gate.same_identity(gate.one(rows,coordinator['ProcessId']),coordinator), 'Snapshot coordinator differs')
    require(slot['command_identity']['ProcessId'] not in by and slot['native_identity']['ProcessId'] not in by
            and len(slot['auxiliary_identities'])==1, 'Timeout must follow command/native departure')
    aux=slot['auxiliary_identities'][0]
    require(gate.same_identity(gate.one(rows,aux['ProcessId']),aux), 'Exact recorded auxiliary must remain at timeout')
    polls=value['deferred_poll_evidence'];previous=-1.
    require(len(polls)==50 and polls[-1]['process_snapshot']==rows
            and polls[-1]['process_inventory']==value['process_inventory'], 'All50 actual timeout polls required')
    for poll in polls:
        process=poll['process_inventory'];sample=poll['process_snapshot'];resources=poll['resource_check']
        require(type(poll['elapsed_seconds']) in (int,float) and poll['elapsed_seconds']>=previous
                and poll['admission_deferred'] is True and poll['pending_owned_pids']==[aux['ProcessId']]
                and process['admission_deferred'] is True and process['exact_recorded_identities_only'] is True
                and process['owned_identities']==expected and process['no_blockers_observed'] is True
                and process['blockers']==[] and len({r['ProcessId'] for r in sample})==len(sample),
                'Every timeout poll must defer admission with exact recorded identities')
        checked=gate.classify_owned(sample,expected,process,require_all=False)
        require(checked['no_blockers_observed'] is True and not checked['blockers']
                and gate.same_identity(gate.one(sample,aux['ProcessId']),aux), 'Timeout poll process identity differs')
        require(resources['memory']['allowed'] is True and resources['memory']['commit_headroom_bytes']>=20*2**30
                and resources['disk_free_bytes']>=60*2**30 and resources['gpu']['free_mib']>=4096,
                'Every timeout poll must retain actual resource-floor evidence')
        previous=poll['elapsed_seconds']
    require(previous>=60, 'Actual timeout polling budget must have expired')
    return dict(snapshot=pin(PARENT/SNAPSHOT_NAME),captured_utc=value['captured_utc'],
        pending_exact_auxiliary=aux,command_and_native_absent_at_timeout=True,deferred_poll_count=len(polls),
        first_poll_elapsed_seconds=polls[0]['elapsed_seconds'],last_poll_elapsed_seconds=previous,
        cause='explicit_monotonic_retirement_budget_expiry',individual_query_timing_recorded=False)


def check_retirement_receipt(value, slot, coordinator):
    require(value['owned_returncode']==0 and value['bounded_retirement_wait_seconds']==60,
            'Exact actual successful child wait and finite60 retirement required')
    checked_slot=dict(slot)
    own=[s for s in value['recorded_slots'] if s['slot']==slot['slot']]
    require(len(own)==1, 'Exact recorded retirement source slot required')
    # Frozen parallel4 serializes an empty auxiliary list in its final record,
    # but omits the field in slot1's earlier snapshot. Both must be exactly empty.
    if 'auxiliary_identities' not in own[0]:
        require(slot['slot']==1 and slot.get('auxiliary_identities')==[]
                and value['auxiliary_identities']==[], 'Only exact empty auxiliary serialization may differ')
        checked_slot.pop('auxiliary_identities')
    retirement_validation.validate_retirement_evidence(value,checked_slot,coordinator,dict(returncode=0),
        maximum_wait_seconds=60,require_deferred_polls=True)


def check_excluded_partial(manifest,captured,plan):
    require(captured['committed_frames']==manifest['committed_frames']==3
            and len(manifest['observations'])==3 and len(manifest['decisions'])==4
            and manifest['production_native_requests']==3 and manifest['warmup_request_count']==7
            and manifest['native_request_count']==manifest['native_callback_count']==10
            and [d['observation_id'] for d in manifest['decisions']]==[s['observation_id'] for s in plan['schedule'][:4]]
            and [o['observation_id'] for o in manifest['observations']]==[s['observation_id'] for s in plan['schedule'][:3]]
            and all(d['state']=='queued_lossless_callback_pending_cpu_admission' and d['native_requests']==1
                    for d in manifest['decisions'][:3])
            and manifest['decisions'][-1]['state']=='not_requested_user_stop'
            and manifest['decisions'][-1]['native_requests']==0,
            'Exact stopped3 prefix required; entire slot remains excluded')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


class Evidence:
    def __init__(self):
        self.bindings = {}

    def path(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        require(expected is None or actual == expected, 'Changed recovery evidence')
        require(str(path) not in self.bindings or self.bindings[str(path)] == actual,
                'Conflicting or changing recovery evidence')
        self.bindings[str(path)] = actual
        return path

    def spec(self, spec, expected_path=None):
        path = roots.resolve_evidence(spec['path'])
        require(expected_path is None or path == Path(expected_path).resolve(), 'Wrong recovery evidence path')
        return self.path(path, spec['sha256'])

    def obj(self, path, expected=None):
        return read(self.path(path, expected))

    def merge(self, bindings):
        # External pinned runtime sources may be outside the data roots.
        for path, expected in bindings.items():
            self.path(path, expected)

    def close(self):
        for path, expected in self.bindings.items():
            require(sha256(path) == expected, 'Recovery source changed before closure')


def assert_absent(rows, identities):
    present = {row['ProcessId']: row for row in rows}
    reused = []
    for old in identities:
        if old['ProcessId'] not in present:
            continue
        current = gate.identity(present[old['ProcessId']])
        old_birth = re.fullmatch(r'/Date\((\d+)\)/', old['CreationDate'])
        new_birth = re.fullmatch(r'/Date\((\d+)\)/', current['CreationDate'])
        require(old_birth and new_birth and int(new_birth[1]) > int(old_birth[1])
                and not gate.same_identity(current, old),
                'Failed-parent identity still present or PID reuse is unproven')
        reused.append(dict(recorded_identity=old,current_later_identity=current,
                           conclusion='recorded_process_absent_pid_reused_after_its_birth'))
    return reused


def _validate(*, rows=None, require_idle=False):
    e = Evidence()
    e.path(__file__); e.path(OWNER, OWNER_SHA); e.path(WORKER, WORKER_SHA); e.path(RETIREMENT, RETIREMENT_SHA); e.path(retirement_validation.__file__,RETIREMENT_VALIDATION_SHA)
    e.merge(roots.bindings())
    metadata = e.obj(LAUNCH/'metadata.json', METADATA_SHA)
    result = e.obj(PARENT/'result.json', RESULT_SHA)
    failure = e.obj(PARENT/'failure.json', FAILURE_SHA)
    require(not (PARENT/'owner_complete.json').exists(), 'Failed parent must not acquire a successful terminal')
    require(Path(metadata['trial_path']).resolve() == PARENT and result['schema'] == 'greenhouse.native848_pilot_parallel.v4'
            and result['requested_worker_count'] == 4 and result['frame_counts'] == EXPECTED_COUNTS
            and result['planned_frames'] == 64 and result['committed_frames'] == 14
            and result['training_approved'] is False and result['accepted_training_increment'] == 0,
            'Wrong finite failed-parent scope')
    require(result['all_recorded_owned_identities_absent'] is True and not result['unretired_owned_slots']
            and failure['all_recorded_identities_absent'] is True, 'Every actually started slot must be retired')
    require(failure['all_owned_command_children_waited'] is True and failure['whole_parallel_owner_complete'] is True
            and failure['training_approved'] is False and failure['accepted_training_increment'] == 0,
            'Actual failed-parent closure and all waits required')
    e.spec(failure['result'], PARENT/'result.json')
    require(failure['result']['sha256'] == RESULT_SHA and failure['incomplete_slots'] == result['incomplete_slots']
            and sorted(r['slot'] for r in result['incomplete_slots']) == [1]
            and not result['skipped_admissions'] and not result['harmless_bootstrap_failures']
            and not result['remaining_slots_not_attempted'], 'Failed/incomplete slot inventory differs')
    coordinator = gate.identity(result['coordinator_identity'])
    birth = re.fullmatch(r'/Date\((\d+)\)/', coordinator['CreationDate'])
    require(birth and int(birth[1]) == metadata['creation_unix_ms'] and coordinator['ProcessId'] == metadata['pid']
            and Path(coordinator['ExecutablePath']).resolve() == Path(metadata['executable']).resolve(),
            'Exact failed owner identity differs')
    argv = gate.command_argv(coordinator['CommandLine'])
    recorded = gate.command_argv(metadata['command_line'])
    require(Path(argv[0]).resolve() == Path(recorded[0]).resolve() and argv[1:] == recorded[1:], 'Owner command differs')
    outer, launch = e.obj(LAUNCH/'owned_exit.json'), e.obj(LAUNCH/'launch.json')
    require(outer['returncode'] == 1 and outer['method'] == 'subprocess_wait_on_owned_process'
            and outer['pid'] == launch['pid'] == metadata['pid']
            and outer['launch_sha256'] == e.bindings[str((LAUNCH/'launch.json').resolve())]
            and Path(launch['command'][0]).resolve() == Path(argv[0]).resolve() and launch['command'][1:] == argv[1:],
            'Actual failed owner subprocess wait/launch required')
    intent = read(e.spec(result['intent'], PARENT/'intent.json'))
    request_path, review_path = e.spec(intent['request']), e.spec(intent['launch_review'])
    request, review = read(request_path), read(review_path)
    require(intent['coordinator_identity'] == coordinator and intent['native_workers'] == request['worker_count'] == 4
            and intent['plans'] == request['plans'] and intent['scene_preflights'] == request['scene_preflights']
            and intent['frame_counts'] == request['frame_counts'] == EXPECTED_COUNTS
            and intent['planned_frames'] == 64, 'Actual parent intent/request scope differs')
    require(review['owner_sha256'] == OWNER_SHA and review['worker_sha256'] == WORKER_SHA
            and review['request_sha256'] == intent['request']['sha256']
            and review['gate_sha256'] == sha256(gate.__file__) and review['native_launch_review_passed'] is True
            and review['blocking_findings'] == [] and review['native_worker_count'] == 4
            and review['training_approved'] is False, 'Original exact independent launch review required')
    for flag, wanted in [('--request',str(request_path)),('--request-sha256',intent['request']['sha256']),
                         ('--review',str(review_path)),('--review-sha256',intent['launch_review']['sha256']),
                         ('--output',str(PARENT))]:
        require(argv.count(flag) == 1 and argv[argv.index(flag)+1] == wanted, 'Owner arguments differ from evidence')
    require(str(OWNER) in argv, 'Wrong owner source in actual command')
    e.merge(result['source_bindings']); e.merge(intent['source_bindings'])
    slots = {row['slot']:row for row in result['completed_slots']}
    successful = {row['slot']:row for row in result['results']}
    require(len(result['completed_slots']) == len(slots) == 2 and set(slots) == {0,1}
            and len(result['results']) == 1 and set(successful) == {0}, 'Exact all-child and successful-slot inventories required')
    identities = [coordinator]
    # Import the frozen owner's pure completed-slot validator, never its main.
    if str(H) not in sys.path: sys.path.insert(0, str(H))
    module_spec = importlib.util.spec_from_file_location('_failed_parent_frozen_owner', OWNER)
    owner = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(owner)
    require('Timed out awaiting exit of all exact recorded owned identities' in failure['error']
            and 'wait_completed_slot' in failure['error'] and SNAPSHOT_NAME in failure['error']
            and SNAPSHOT_SHA in failure['error'], 'Wrong finite retirement failure')
    failure_snapshot = e.obj(PARENT/SNAPSHOT_NAME,SNAPSHOT_SHA)
    for index in (2,3):
        require(not (PARENT/f'slot_{index}').exists(), 'Unlaunched slot must remain uncreated')
    require(not (PARENT/'owned_auxiliary_registration_2.json').exists()
            and not (PARENT/'before_slot_2_resource_admission.json').exists(),
            'Failed registration must not acquire a successful admission')
    require(not (PARENT/'before_slot_2_warmup.json').exists(), 'Second worker did not qualify for next launch')
    registration = e.obj(PARENT/'owned_auxiliary_registration_1.json')
    require(registration['exact_owned_direct_children_only'] is True
            and registration['generic_descendant_exemption'] is False
            and [r['slot'] for r in registration['slots']] == [0], 'Exact recorded auxiliary scope differs')
    e.merge(registration['source_bindings'])
    auxiliaries = []
    for slot in registration['slots']:
        recorded = slots[slot['slot']]
        require(slot['command_identity'] == recorded['command_identity']
                and slot['native_identity'] == recorded['native_identity'], 'Auxiliary parent identity differs')
        for aux in slot.get('auxiliary_identities',[]):
            auxiliaries.append(gate.checked_telemetry(aux,slot['native_identity']))
    require(len(auxiliaries) == 1, 'Exact recorded telemetry inventory required')
    identities.extend(auxiliaries)
    timeout_slot = {k:slots[0][k] for k in ('slot','command_identity','native_identity','auxiliary_identities')}
    timeout_records = [timeout_slot,{k:slots[1][k] for k in ('slot','command_identity','native_identity')}]
    timeout_proof = check_timeout_snapshot(failure_snapshot,timeout_slot,coordinator,timeout_records)
    require(slots[0]['auxiliary_identities'] == auxiliaries and slots[1].get('auxiliary_identities',[]) == [],
            'Exact retained recorded auxiliary inventory differs')
    slot_proofs = {}; excluded_partial_ids = []
    for index in range(2):
        slot, folder = slots[index], PARENT/f'slot_{index}'
        command, native = gate.identity(slot['command_identity']), gate.identity(slot['native_identity'])
        require(not {command['ProcessId'],native['ProcessId']} & {r['ProcessId'] for r in identities}
                and command['ProcessId'] != native['ProcessId'], 'Duplicate child PID')
        identities.extend([command,native])
        child_launch = read(e.spec(slot['launch'], folder/'launch.json'))
        child_exit = read(e.spec(slot['owned_exit'], folder/'owned_exit.json'))
        require(child_exit['returncode'] == 0
                and child_exit['method'] == 'subprocess_wait_on_owned_process'
                and child_exit['pid'] == child_launch['pid'] == command['ProcessId']
                and child_exit['launch_sha256'] == slot['launch']['sha256'], 'Actual bounded child exit differs')
        retirement = read(e.spec(slot['retirement_receipt'],folder/'all_recorded_identities_exited.json'))
        require(slot['all_recorded_identities_absent'] is True, 'Actual slot retirement required')
        check_retirement_receipt(retirement,slot,coordinator); e.merge(retirement['source_bindings'])
        gate.command_child(command,coordinator,child_launch['command'])
        gate.native_child(native,command,child_launch['command'][1:])
        ready, ticket = e.obj(folder/'native_ready.json'), e.obj(folder/'ticket.json')
        require(ready['slot'] == index and ready['native_identity'] == native and ready['command_identity'] == command
                and ready['ticket_sha256'] == sha256(folder/'ticket.json')
                and ticket['schema'] == 'greenhouse.native848_pilot_parallel_ticket.v4'
                and ticket['coordinator_identity'] == coordinator and ticket['slot'] == index
                and ticket['worker_count'] == 4 and ticket['frames_per_worker'] == EXPECTED_COUNTS[index]
                and Path(ticket['output']).resolve() == folder/'capture'
                and ticket['plan'] == intent['plans'][index] and ticket['scene_preflight'] == intent['scene_preflights'][index],
                'Actual registration/ticket scope differs')
        command_receipt = e.obj(folder/'command_identity.json')
        require(command_receipt['identity'] == command and command_receipt['command'] == child_launch['command']
                and command_receipt['ticket_sha256'] == sha256(folder/'ticket.json'), 'Command receipt differs')
        admission = e.obj(folder/'actual_process_admission.json')
        require(admission['same_snapshot_classification'] is True and admission['require_all_live_identities'] is True
                and admission['process_inventory']['no_blockers_observed'] is True, 'Actual v2 process admission required')
        plan_path = e.spec(ticket['plan']); plan = read(plan_path)
        e.spec(ticket['scene_preflight'])
        release = e.obj(folder/'release.json')
        require(release['ticket_sha256'] == sha256(folder/'ticket.json')
                and release['resource_admission']['allowed'] is True
                and release['resource_admission']['process_inventory']['no_blockers_observed'] is True,
                'Original release resource admission required')
        if index not in RECOVERABLE:
            require(not (folder/'owner_complete.json').exists() and not (folder/'result.json').exists(),
                    'Incomplete slots must remain unadmitted')
            partial_manifest = e.obj(folder/'capture/batch_manifest.json')
            partial_result = e.obj(folder/'capture/result.json')
            check_excluded_partial(partial_manifest,partial_result,plan)
            require((folder/'capture/STOP_AFTER_CURRENT_FRAME').is_file()
                    and partial_result['manifest_sha256']==sha256(folder/'capture/batch_manifest.json')
                    and partial_result['native_identity']==native and partial_result['wrapper_sha256']==WORKER_SHA
                    and partial_result['bounded_parallel_slot']==index
                    and partial_manifest['source_assets_unchanged'] is True,
                    'Exact stopped partial capture binding required; it is excluded')
            e.path(folder/'capture/STOP_AFTER_CURRENT_FRAME')
            partial_context = e.obj(folder/'capture/batch_context.json',partial_manifest['context_sha256'])
            e.merge(partial_context['source_bindings'])
            for partial_index,obs_pin in enumerate(partial_manifest['observations']):
                partial_obs = read(e.spec(obs_pin))
                require(partial_obs['source_pose_id']==plan['selected_sample_ids'][partial_index]
                        and partial_obs['observation_id']==plan['schedule'][partial_index]['observation_id'],
                        'Excluded partial camera identity differs')
                for asset in [*partial_obs['files'].values(),partial_obs['geometry_proof'],partial_obs['mapping']]:
                    require(e.spec(asset).is_relative_to(folder/'capture'), 'Partial evidence escaped original slot')
                excluded_partial_ids.append(partial_obs['source_pose_id'])
            continue
        row = successful[index]
        receipt = read(e.spec(row['owner_result'],folder/'result.json'))
        require(receipt['source_bindings'] == result['source_bindings'], 'Slot source closure differs from actual parent')
        terminal = e.obj(folder/'owner_complete.json')
        require(terminal['result_sha256'] == row['owner_result']['sha256'] and terminal['parallel_slot_only'] is True
                and receipt['bounded_parallel_slot'] == index and receipt['all_owned_native_children_waited'] is True,
                'Actual completed slot-only terminal required')
        fresh = owner.check_completed_slot(folder,plan_path,plan)
        require(all(receipt.get(k) == v for k,v in fresh.items()) and fresh['committed_frames'] == EXPECTED_COUNTS[index]
                and row['committed_frames'] == EXPECTED_COUNTS[index], 'Full immutable schedule did not complete')
        context = e.obj(folder/'capture/batch_context.json',receipt['context_sha256'])
        manifest = e.obj(folder/'capture/batch_manifest.json',receipt['manifest_sha256'])
        captured = e.obj(folder/'capture/result.json',receipt['capture_result_sha256'])
        require(captured['native_identity'] == native and captured['wrapper_sha256'] == WORKER_SHA
                and captured['bounded_parallel_slot'] == index, 'Native capture identity differs')
        e.merge(context['source_bindings'])
        for name in ('ticket.json','command_identity.json','native_ready.json','release.json','actual_process_admission.json'):
            require(context['source_bindings'].get(str((folder/name).resolve())) == sha256(folder/name),
                    'Context lacks its actual process receipt')
        require(manifest['native_request_count'] == manifest['native_callback_count']
                == manifest['warmup_request_count'] + manifest['production_native_requests']
                and manifest['source_assets_unchanged'] is True and manifest['stage_count'] == manifest['render_product_count'] == 1,
                'Native callback totals or stage ownership differ')
        expected = [r['observation_id'] for r in plan['schedule']]
        require(plan['capture_split'] == 'train' and len(expected) == len(set(expected)) == 14
                and [r['observation_id'] for r in manifest['decisions']] == expected
                and [r['observation_id'] for r in manifest['observations']] == expected
                and manifest['native_request_count'] == manifest['native_callback_count'] == 21
                and manifest['warmup_request_count'] == 7 and manifest['production_native_requests'] == 14,
                'Exact full14 TRAIN11 schedule and callback totals required')
        warmup = e.obj(context['warmup_evidence_path'],context['warmup_evidence_sha256'])
        require(len(warmup['requests']) == 7 and warmup['control_frames_excluded'] is True,
                'Seven actual excluded warmups required')
        profile = read(e.spec(plan['profile_evidence']))
        seen = set()
        for frame_index,(scheduled,decision,obs_pin) in enumerate(zip(plan['schedule'],manifest['decisions'],manifest['observations'])):
            observation_path = e.spec(obs_pin)
            require(observation_path.parent.parent.parent == folder/'capture' and observation_path.name == 'observation.json',
                    'Observation outside exact slot capture')
            obs = read(observation_path); sync, req = obs['synchronization'], obs['request_evidence']
            require(obs['schema'] == 'greenhouse.native848_unique9mm_pilot_observation.v1'
                    and obs['observation_id'] == decision['observation_id'] == obs_pin['observation_id']
                    and obs['source_pose_id'] == decision['source_pose_id'] and obs['capture_index'] == decision['capture_index']
                    and obs['observation_id'] not in seen and obs['context_sha256'] == receipt['context_sha256']
                    and Path(obs['context_path']).resolve() == folder/'capture/batch_context.json'
                    and obs['calibration']['resolution'] == [848,408]
                    and req['native_requests'] == req['callback_count'] == 1 and req['wait_for_render'] is True
                    and req['callback_sequence_after'] == req['callback_sequence_before']+1 == sync['callback_sequence']
                    and req['request_index'] == obs_pin['request_index'] == sync['request_index']
                    and sync['static_before'] == sync['static_after'], 'Observation/callback identity or freshness differs')
            require(decision['state'] == 'queued_lossless_callback_pending_cpu_admission' and decision['native_requests'] == 1
                    and obs['observation_id'] == scheduled['observation_id']
                    and obs['source_pose_id'] == scheduled['source_pose_id'] == plan['selected_sample_ids'][frame_index]
                    and obs['capture_index'] == frame_index and obs['capture_role'] == scheduled['capture_role']
                    and obs['split'] == 'train' and obs['source_family'] == 'seed11_full'
                    and obs['target_id'] == 'seed11_full/SubStem_44'
                    and req['callback_sequence_after'] == sync['callback_sequence'] == frame_index+8
                    and req['request_index'] == frame_index+8
                    and len(req['callbacks']) == 1
                    and req['callbacks'][0]['callback_sequence'] == sync['callback_sequence']
                    and req['callbacks'][0]['request_index'] == req['request_index']
                    and sync['freshness']['callback_sequence'] == sync['callback_sequence']
                    and req['requested_subframes'] == profile['request_subframes'] == 8
                    and req['delta_time_seconds'] == profile['delta_time_seconds'] == 0
                    and req['timeline_after_seconds'] == req['timeline_before_seconds']
                    and req['render_settings'] == profile['render_settings']
                    and req['reset_returned_without_exception'] is True,
                    'Exact scheduled TRAIN11 single fresh reset8 callback required')
            seen.add(obs['observation_id'])
            for asset in list(obs['files'].values())+[obs['geometry_proof'],obs['mapping']]:
                asset_path = e.spec(asset)
                require(asset_path.is_relative_to(folder/'capture'), 'Captured asset outside exact slot')
        for asset in [*manifest['geometry_proofs'],*manifest['renderer_mappings']]:
            require(e.spec(asset).is_relative_to(folder/'capture'), 'Manifest asset outside exact slot')
        slot_proofs[str(index)] = dict(owner_result=row['owner_result'],plan=ticket['plan'],scene_preflight=ticket['scene_preflight'],
                committed_frames=fresh['committed_frames'],context=pin(folder/'capture/batch_context.json'),
                manifest=pin(folder/'capture/batch_manifest.json'),native_owned_exit=slot['owned_exit'])
    planned = []
    for index, spec in enumerate(intent['plans']):
        plan = read(e.spec(spec)); e.spec(intent['scene_preflights'][index])
        require(plan['max_frames'] == EXPECTED_COUNTS[index]
                and len(plan['selected_sample_ids']) == EXPECTED_COUNTS[index], 'Finite planned camera scope differs')
        planned.append(plan['selected_sample_ids'])
    flat = [pose for group in planned for pose in group]
    require(len(flat) == len(set(flat)) == 64 and len(excluded_partial_ids) == 3
            and set(excluded_partial_ids) <= set(planned[1]), 'Duplicate or unexpected planned/partial camera ID')
    remaining = [pose for group in planned[1:] for pose in group]
    never_captured = [pose for pose in remaining if pose not in excluded_partial_ids]
    require(len(remaining) == 50 and len(never_captured) == 47
            and not set(planned[0]) & set(remaining), 'Continuation camera scope overlaps recovery')
    continuation = dict(completed_recoverable_camera_ids=planned[0],excluded_or_unlaunched_camera_ids=remaining,
            excluded_partial_raw_camera_ids=excluded_partial_ids,never_captured_camera_ids=never_captured,
            never_captured_camera_ids_by_slot={str(i):[p for p in planned[i] if p not in excluded_partial_ids] for i in (1,2,3)},
            complete_recoverable_count=14,excluded_partial_raw_count=3,never_captured_count=47,
            duplicate_camera_ids=[],automatic_recapture_authorized=False)
    rows = gate.snapshot() if rows is None else rows
    pid_reuse_observations = assert_absent(rows,identities)
    process_report = gate.inventory([],rows=rows) if require_idle else None
    if process_report is not None:
        require(process_report['no_blockers_observed'] is True and not process_report['blockers'],
                'Unknown native/relevant process blocks recovery audit')
    e.close()
    return dict(metadata=metadata,coordinator_identity=coordinator,parent_failed=True,parent_returncode=1,
            recoverable_slots=RECOVERABLE,excluded_slots=[1,2,3],completed_raw_frames=14,
            actual_started_slots=[0,1],incomplete_started_slots=[1],unlaunched_slots=[2,3],
            continuation_inventory=continuation,recorded_auxiliary_identities=auxiliaries,timeout_departure_proof=timeout_proof,
            successful_slot_results={k:v['owner_result'] for k,v in slot_proofs.items()},slot_proofs=slot_proofs,
            source_bindings=e.bindings,outer_exit=pin(LAUNCH/'owned_exit.json'),outer_launch=pin(LAUNCH/'launch.json'),
            parent_result=pin(PARENT/'result.json'),parent_failure=pin(PARENT/'failure.json'),
            metadata_pin=pin(LAUNCH/'metadata.json'),all_recorded_native_children_absent=True,
            absence_scope='complete_recorded_identities_not_bare_PID_numbers',pid_reuse_observations=pid_reuse_observations,
            process_inventory=process_report,completion_method='actual_failed_parent_and_owned_child_waits',
            synthetic_root_native_exit=False,whole_parent_success=False,throughput_qualified=False,
            per_frame_geometry_reset_profile_replay_pending=True,complete_census_ambiguity_and_visual_QA_pending=True,
            training_approved=False,accepted_training_increment=0)


def run(output):
    output = roots.diagnostic(output)
    require(not output.exists(), 'Recovery audit output must be create-only')
    proof = _validate(require_idle=True)
    proof.update(schema=SCHEMA,implementation=pin(__file__),created_utc=datetime.now(timezone.utc).isoformat(),
                 recovery_scope='only_completed_slots_pending_new_complete_census_9mm_ambiguity_and_actual_QA')
    output.mkdir(parents=True)
    path = output/'result.json'
    with path.open('x',encoding='utf-8') as f: json.dump(proof,f,indent=2,allow_nan=False)
    return pin(path)


def check(spec,rows=None):
    path = roots.resolve_evidence(spec['path'])
    require(sha256(path) == spec['sha256'], 'Changed recovery audit')
    audit = read(path)
    require(audit['schema'] == SCHEMA and audit['implementation'] == pin(__file__)
            and audit['parent_failed'] is True and audit['whole_parent_success'] is False
            and audit['training_approved'] is False and audit['accepted_training_increment'] == 0
            and audit['recoverable_slots'] == RECOVERABLE and audit['excluded_slots'] == [1,2,3]
            and audit['process_inventory']['no_blockers_observed'] is True,
            'Wrong finite recovery audit/implementation')
    proof = _validate(rows=rows)
    for key in ('metadata','outer_exit','outer_launch','parent_result','parent_failure','slot_proofs','source_bindings',
                'successful_slot_results','parent_returncode','completed_raw_frames','continuation_inventory',
                'actual_started_slots','incomplete_started_slots','unlaunched_slots','recorded_auxiliary_identities','timeout_departure_proof'):
        require(audit[key] == proof[key], 'Recovery audit differs from actual frozen closure')
    require(sha256(path) == spec['sha256'], 'Recovery audit changed during validation')
    proof['source_bindings'][str(path)] = spec['sha256']
    proof['recovery_audit'] = spec
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output)))
