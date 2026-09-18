"""Fresh native848 labels, full trace, and workspace for audited direct batches."""
from pathlib import Path
from datetime import datetime, timezone
import traceback
import numpy as np
from . import native848_direct_plan_v1 as api
from . import native848_direct_audit_v1 as audit_api
from . import native848_clear_labels_v1 as labels, native848_query_selection_v1 as queries
from .native848_pair_audit_v2 import image_array
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256


def run(trial, *, result_sha256, output):
    trial, output = Path(trial).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(trial) and not trial.is_relative_to(output), 'New disjoint annotation output required')
    receipt = read_json(trial/'result.json')
    require(sha256(trial/'result.json') == result_sha256 and not (trial/'failure.json').exists()
        and receipt['state'] == 'owned_direct_native848_completed_and_audited'
        and receipt['training_approved'] is False, 'Owned completed direct batch required')
    intent, owned, launch = (read_json(trial/n) for n in ('intent.json', 'owned_exit.json', 'launch.json'))
    require(owned['returncode'] == 0 and owned['method'] == 'subprocess_wait_on_owned_process'
        and sha256(trial/'owned_exit.json') == receipt['owned_exit_sha256']
        and sha256(trial/'launch.json') == owned['launch_sha256'] and launch['pid'] == owned['pid'], 'Changed owned exit')
    require(launch['plan_sha256'] == intent['plan_sha256'] == receipt['plan_sha256']
        and 'sim_data.native848_direct_worker_v1' in launch['command']
        and launch['command'][-1] == str(trial/'capture'), 'Changed direct worker route')
    verify_bindings(intent['runtime_owner_bindings'])
    plan = read_json(intent['plan_path'])
    _, plans, generated = api.check(plan, full=True)
    saved = read_json(trial/'audit.json')
    require(sha256(trial/'audit.json') == receipt['audit_sha256'], 'Changed saved direct audit')
    actual = audit_api.audit_capture(trial/'capture', plan_path=intent['plan_path'],
        plan_sha256=intent['plan_sha256'], result_sha256=sha256(trial/'capture/result.json'))
    require(actual == saved, 'Independent direct buffer replay differs')
    pins = {**actual['source_bindings'], **{str(trial/n): sha256(trial/n)
        for n in ('result.json', 'audit.json', 'intent.json', 'owned_exit.json', 'launch.json')}}
    checker = WorkspaceChecker()
    output.mkdir(parents=True)
    write_json(output/'request.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trial=str(trial), trial_result_sha256=result_sha256, source_bindings=pins,
        implementation_bindings=plan['implementation_bindings'], workspace_implementation_bindings=checker.bindings,
        annotation_epoch=queries.ANNOTATION_EPOCH, resolution=[848, 408], training_approved=False))
    records = []
    try:
        for record in actual['records']:
            name = record['sample_id']
            folder = trial/'capture'/name
            pair = plans[record['pair_plan_path']]
            meta = read_json(folder/'sample.json')
            rgb = image_array(folder/'inputs/rgb.png')
            depth = np.load(folder/'inputs/depth_m.npy', allow_pickle=False)
            valid = image_array(folder/'inputs/depth_valid.png') == 255
            components = np.load(folder/'supervision/component_id.npy', allow_pickle=False)
            catalogue = read_json(folder/'supervision/identities.json')['component_catalogue']
            mask = image_array(folder/'supervision/target_visible.png')
            baseline = labels.derive(meta, generated['report'], rgb, depth, valid, components, catalogue)
            label, trace, selection = queries.annotate_v1(meta, generated['report'], rgb, depth, valid,
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
            out = dict(sample_id=name, target_id=label['target_id'], source_target=pair['conservative_view_cap_group'],
                source_family=pair['source_family'], split='train', source_sample_path=str(folder/'sample.json'),
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
        write_json(output/'result.json', dict(state='direct_native848_labels_trace_workspace_complete_pending_review_and_global_admission',
            records=records, candidates_for_individual_visual_review=sum(r['candidate_for_individual_visual_review'] for r in records),
            source_bindings=pins, visual_review_performed=False, geometry_novelty_qualified=False,
            near_duplicate_graph_rebuilt=False, source_cap_reset=False, training_approved=False, accepted_training_increment=0))
    except BaseException:
        write_json(output/'failure.json', dict(error=traceback.format_exc(), completed=records, training_approved=False))
        raise
