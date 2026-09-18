"""Explicit original short-capture admission; controls never become candidates."""
from pathlib import Path
from datetime import datetime, timezone
import traceback
import numpy as np
from . import native848_clear_labels_v1 as labels, native848_query_selection_v1 as queries
from .native848_pair_audit_v2 import image_array
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256


SCHEMA = 'greenhouse.original848_short_admission.v1'
REQUEST_SCHEMA = 'greenhouse.original848_short_admission_request.v1'
STATE = 'original848_short_labels_trace_workspace_complete_pending_review_and_global_admission'
OWNED_STATE = 'owned_original848_short_completed_and_audited'
PLAN_SCHEMA = 'greenhouse.original848_short_camera_batch_plan.v1'
SAMPLE_SCHEMA = 'greenhouse.original848_short_camera_sample.v1'
AUDIT_SCHEMA = 'greenhouse.original848_short_buffer_geometry_audit.v1'
WORKER = 'sim_data.native848_original_short_worker_v1'
PROFILE = 'original848_cached_reset8_after_product56_native_quality.v1'
BUDGET = 8


def _argument(command, flag):
    require(isinstance(command, list) and command.count(flag) == 1,
            'One explicit worker argument required: ' + flag)
    index = command.index(flag) + 1
    require(index < len(command), 'Missing worker argument: ' + flag)
    return command[index]


def qualification_binding(plan):
    binding = plan['qualification_evidence']
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}
        and Path(binding['path']).is_absolute()
        and isinstance(binding['sha256'], str) and len(binding['sha256']) == 64
        and all(c in '0123456789abcdef' for c in binding['sha256']),
        'Explicit sealed qualification path/SHA256 required')
    require(plan['schema'] == PLAN_SCHEMA and plan['profile'] == PROFILE
        and plan['mode'] in ('adaptation', 'production')
        and type(plan['render_budget_subframes']) is int and plan['render_budget_subframes'] == BUDGET
        and plan['resolution'] == [848, 408]
        and plan['generated_geometry_used'] is False and plan['source_cap_reset'] is False
        and plan['training_approved'] is False, 'Explicit original reset8 plan required')
    return binding


def check_owned_route(trial, receipt, owned, launch, intent, plan):
    trial = Path(trial).resolve()
    qualification_binding(plan)
    require(receipt['state'] == OWNED_STATE and receipt['training_approved'] is False
        and owned['returncode'] == 0 and owned['method'] == 'subprocess_wait_on_owned_process'
        and launch['pid'] == owned['pid'], 'Owned short-capture completion required')
    require(launch['plan_sha256'] == intent['plan_sha256'] == receipt['plan_sha256']
        and _argument(launch['command'], '-m') == WORKER
        and Path(_argument(launch['command'], '--plan')).resolve() == Path(intent['plan_path']).resolve()
        and _argument(launch['command'], '--plan-sha256') == intent['plan_sha256']
        and Path(_argument(launch['command'], '--output')).resolve() == trial/'capture'
        and launch['command'][-1] == str(trial/'capture'), 'Changed explicit short producer route')
    require(receipt['profile'] == plan['profile']
        and receipt['render_budget_subframes'] == BUDGET, 'Owned short budget/profile differs')


def verify_short_audit_contract(plan, actual):
    """Check framing of an already independently recomputed raw native audit."""
    binding = qualification_binding(plan)
    path = str(Path(binding['path']).resolve())
    require(actual['schema'] == AUDIT_SCHEMA and actual['training_approved'] is False
        and actual['source_cap_reset'] is False and actual['geometry_novelty_qualified'] is False
        and actual['profile'] == PROFILE and actual['render_budget_subframes'] == BUDGET
        and actual['source_bindings'].get(path) == binding['sha256'],
        'Raw short audit lacks exact qualification/profile bindings')
    controls, records, adaptation = actual['adaptation_records'], actual['records'], actual['runtime_adaptation']
    require(isinstance(controls, list) and isinstance(records, list)
        and adaptation['passed'] is True and adaptation['sequence'] == ['A', 'B', 'A'],
        'Independently replayed original A-B-A adaptation required')
    control_ids = adaptation['control_sample_ids']
    pose_ids = adaptation['source_pose_ids']
    require(len(control_ids) == len(set(control_ids)) == 3
        and len(pose_ids) == 3 and pose_ids[0] == pose_ids[2] and pose_ids[0] != pose_ids[1],
        'Actual distinct A-B-A source pose sequence required')
    if plan['mode'] == 'adaptation':
        require(len(controls) == 3 and records == []
            and [r['sample_id'] for r in controls] == control_ids
            and [r['source_pose_id'] for r in controls] == pose_ids
            and adaptation['evidence_path'] is None and adaptation['evidence_sha256'] is None
            and plan.get('original_adaptation_evidence') is None,
            'Adaptation controls cannot be production or self-sealed qualification')
    else:
        evidence = plan['original_adaptation_evidence']
        require(isinstance(evidence, dict) and set(evidence) == {'path', 'sha256'}
            and Path(evidence['path']).is_absolute()
            and isinstance(evidence['sha256'], str) and len(evidence['sha256']) == 64
            and all(c in '0123456789abcdef' for c in evidence['sha256'])
            and Path(adaptation['evidence_path']).resolve() == Path(evidence['path']).resolve()
            and adaptation['evidence_sha256'] == evidence['sha256']
            and actual['source_bindings'].get(str(Path(evidence['path']).resolve())) == evidence['sha256']
            and controls == [], 'Production requires separately sealed and replayed original adaptation evidence')
    production_ids = [r['sample_id'] for r in records]
    require(len(set(production_ids)) == len(production_ids)
        and not set(control_ids).intersection(production_ids)
        and not set(control_ids).intersection(plan['selected_sample_ids'])
        and {r['source_pose_id'] for r in records} <= set(plan['selected_sample_ids']),
        'Adaptation controls cannot enter production inventory')
    for row in controls:
        require(row['capture_role'] == 'adaptation_control', 'Changed adaptation control role')
    for row in records:
        require(row['capture_role'] == 'production' and row['render_profile_qualified'] is True
            and row['profile'] == PROFILE and row['render_budget_subframes'] == BUDGET
            and Path(row['qualification_evidence_path']).resolve() == Path(path)
            and row['qualification_evidence_sha256'] == binding['sha256'],
            'Production frame lacks explicit qualified short-profile evidence')
    for row in controls + records:
        sample_path = Path(row['sample_path']).resolve()
        require(actual['source_bindings'].get(str(sample_path)) == row['sample_sha256']
            and actual['source_bindings'].get(str(sample_path.parent/'inputs/rgb.png')) == row['rgb_sha256'],
            'Raw adaptation/production sample and RGB bindings must be retained')
    return binding


def verify_production_sample(meta, record, plan, actual):
    binding = qualification_binding(plan)
    require(meta['schema_version'] == SAMPLE_SCHEMA and meta['sample_id'] == record['sample_id']
        and meta['capture_role'] == record['capture_role'] == 'production'
        and plan['mode'] == 'production'
        and meta['direct_camera']['cached_sample_id'] == record['source_pose_id']
        and meta['original_geometry_only'] is True and meta['generated_plant_native_pixels'] == 0
        and meta['synchronization']['profile'] == PROFILE
        and meta['synchronization']['render_budget_subframes'] == BUDGET
        and actual['runtime_adaptation']['passed'] is True,
        'Sample is not independently audited original short production')
    require(record['qualification_evidence_sha256'] == binding['sha256']
        and Path(record['qualification_evidence_path']).resolve() == Path(binding['path']).resolve(),
        'Sample record qualification differs')


def run(trial, *, result_sha256, output):
    from . import native848_original_short_plan_v1 as api
    from . import native848_original_short_audit_v1 as audit_api
    trial, output = Path(trial).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(trial) and not trial.is_relative_to(output), 'New disjoint annotation output required')
    receipt = read_json(trial/'result.json')
    require(sha256(trial/'result.json') == result_sha256 and not (trial/'failure.json').exists()
        and receipt['state'] == OWNED_STATE
        and receipt['training_approved'] is False, 'Owned completed direct batch required')
    intent, owned, launch = (read_json(trial/n) for n in ('intent.json', 'owned_exit.json', 'launch.json'))
    require(owned['returncode'] == 0 and owned['method'] == 'subprocess_wait_on_owned_process'
        and sha256(trial/'owned_exit.json') == receipt['owned_exit_sha256']
        and sha256(trial/'launch.json') == owned['launch_sha256'] and launch['pid'] == owned['pid'], 'Changed owned exit')
    require(launch['plan_sha256'] == intent['plan_sha256'] == receipt['plan_sha256']
        and WORKER in launch['command']
        and launch['command'][-1] == str(trial/'capture'), 'Changed direct worker route')
    verify_bindings(intent['runtime_owner_bindings'])
    plan = read_json(intent['plan_path'])
    require(sha256(intent['plan_path']) == intent['plan_sha256'], 'Changed short plan')
    check_owned_route(trial, receipt, owned, launch, intent, plan)
    cache, anchor, original_report = api.check(plan, full=True)
    saved = read_json(trial/'audit.json')
    require(sha256(trial/'audit.json') == receipt['audit_sha256'], 'Changed saved direct audit')
    actual = audit_api.audit_capture(trial/'capture', plan_path=intent['plan_path'],
        plan_sha256=intent['plan_sha256'], result_sha256=sha256(trial/'capture/result.json'))
    require(actual == saved, 'Independent short buffer replay differs')
    qualification = verify_short_audit_contract(plan, actual)
    verify_bindings({str(Path(qualification['path']).resolve()): qualification['sha256']})
    pins = {**actual['source_bindings'], **{str(trial/n): sha256(trial/n)
        for n in ('result.json', 'audit.json', 'intent.json', 'owned_exit.json', 'launch.json')}}
    checker = WorkspaceChecker()
    output.mkdir(parents=True)
    write_json(output/'request.json', dict(schema=REQUEST_SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        mode=plan['mode'], profile=PROFILE, render_budget_subframes=BUDGET, qualification_evidence=qualification,
        original_adaptation_evidence=plan.get('original_adaptation_evidence'),
        adaptation_control_sample_ids=actual['runtime_adaptation']['control_sample_ids'],
        runtime_adaptation_audit_sha256=sha256(trial/'audit.json'),
        trial=str(trial), trial_result_sha256=result_sha256, source_bindings=pins,
        implementation_bindings=plan['implementation_bindings'], workspace_implementation_bindings=checker.bindings,
        annotation_epoch=queries.ANNOTATION_EPOCH, resolution=[848, 408], training_approved=False))
    records = []
    try:
        for record in actual['records']:
            name = record['sample_id']
            folder = trial/'capture'/name
            source_row = next(r['source_row'] for r in cache['records'] if r['sample_id'] == record['source_pose_id'])
            meta = read_json(folder/'sample.json')
            verify_production_sample(meta, record, plan, actual)
            rgb = image_array(folder/'inputs/rgb.png')
            depth = np.load(folder/'inputs/depth_m.npy', allow_pickle=False)
            valid = image_array(folder/'inputs/depth_valid.png') == 255
            components = np.load(folder/'supervision/component_id.npy', allow_pickle=False)
            catalogue = read_json(folder/'supervision/identities.json')['component_catalogue']
            mask = image_array(folder/'supervision/target_visible.png')
            baseline = labels.derive(meta, original_report, rgb, depth, valid, components, catalogue)
            label, trace, selection = queries.annotate_v1(meta, original_report, rgb, depth, valid,
                components, catalogue, target_mask=mask, expected_label=baseline)
            proof = checker.check_sample(folder/'sample.json', record['sample_sha256'])
            require(proof['target_id'] == label['target_id']
                and np.allclose(proof['nominal_world_m'], label['nominal_world_m'], atol=1e-8, rtol=0), 'Workspace target mismatch')
            dest = output/name
            dest.mkdir()
            for filename, value in (('baseline_label.json', baseline), ('label.json', label),
                ('query_selection.json', selection), ('workspace.json', proof)):
                write_json(dest/filename, value)
            if trace is not None:
                write_json(dest/'query_trace.json', trace)
            local, traced, workspace = label['eligible'] is True, bool(trace and trace['passed']), proof['result']['workspace_passed'] is True
            render = record['render_profile_qualified']
            decision = ('hold_render_profile' if not render else 'exclude_local_clarity' if not local
                else 'hold_full_trace' if not traced else 'hold_workspace' if not workspace
                else 'candidate_pending_individual_visual_review_and_global_grouping')
            out = dict(sample_id=name, source_pose_id=record['source_pose_id'], capture_role='production', profile=PROFILE,
                render_budget_subframes=BUDGET, qualification_evidence_path=qualification['path'],
                qualification_evidence_sha256=qualification['sha256'],
                target_id=label['target_id'], source_target=source_row['target_id'],
                source_family=anchor['source_family'], split='train', source_sample_path=str(folder/'sample.json'),
                source_sample_sha256=record['sample_sha256'], rgb_path=str(folder/'inputs/rgb.png'), rgb_sha256=record['rgb_sha256'],
                label_path=str(dest/'label.json'), label_sha256=sha256(dest/'label.json'),
                workspace_path=str(dest/'workspace.json'), workspace_sha256=sha256(dest/'workspace.json'),
                query_selection_sha256=sha256(dest/'query_selection.json'),
                query_trace_sha256=sha256(dest/'query_trace.json') if trace else None,
                local_clarity_passed=local, full_trace_and_anchored_grid_passed=traced,
                workspace_passed=workspace, render_profile_qualified=render, decision=decision,
                candidate_for_individual_visual_review=local and traced and workspace and render,
                source_cap_reset=False, geometry_novelty_qualified=False, training_approved=False, accepted_training_increment=0)
            records.append(out)
            print('DIRECT848_ADMISSION', name, decision, flush=True)
        checker.finish()
        verify_bindings(pins)
        verify_bindings(plan['implementation_bindings'])
        write_json(output/'result.json', dict(schema=SCHEMA, state=STATE,
            mode=plan['mode'], profile=PROFILE, render_budget_subframes=BUDGET, qualification_evidence=qualification,
            original_adaptation_evidence=plan.get('original_adaptation_evidence'),
            adaptation_control_sample_ids=actual['runtime_adaptation']['control_sample_ids'],
            adaptation_controls_excluded_from_candidates=True,
            runtime_adaptation_audit_sha256=sha256(trial/'audit.json'),
            records=records, candidates_for_individual_visual_review=sum(r['candidate_for_individual_visual_review'] for r in records),
            source_bindings=pins, visual_review_performed=False, geometry_novelty_qualified=False,
            near_duplicate_graph_rebuilt=False, source_cap_reset=False, training_approved=False, accepted_training_increment=0))
    except BaseException:
        write_json(output/'failure.json', dict(error=traceback.format_exc(), completed=records, training_approved=False))
        raise

