"""Read-only label/trace/workspace admission for the explicit native848 pair.

Consumes an owned completed pair and replays its saved-buffer audit. Produces
new annotations only. Individual visual review, global geometry/near-image
qualification and source caps remain downstream requirements; zero accepted
training images are claimed here.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,traceback
import numpy as np
from PIL import Image
from . import native848_pair_plan_v1 as plan_api,native848_pair_audit_v1 as audit_api
from . import native848_clear_labels_v1 as labels,native848_query_selection_v1 as queries
from .dataset_review import require,read_json,write_json,verify_bindings
from .depth_preview import sha256
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker

def implementation_bindings():
    root=Path(__file__).resolve().parent
    names=['native848_pair_admission_v1.py','native848_clear_labels_v1.py','native848_query_selection_v1.py',
        'native_clear_labels.py','native_clear_contract.py','native_query_visibility.py','automated_native_review.py',
        'query_visibility.py','cut_regions.py','capture_contract.py','capture_sensor.py','clear_cutpoint_contract.py',
        'dataset_review.py','depth_preview.py','qwen_coordinates.py']
    return {**plan_api.implementation_bindings(),**{str(root/name):sha256(root/name) for name in names}}

def run(trial,*,result_sha256,output):
    from .collection_plan import load_plan
    trial=Path(trial).resolve();output=Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(trial) and not trial.is_relative_to(output),'New disjoint admission output required')
    receipt=read_json(trial/'result.json')
    require(sha256(trial/'result.json')==result_sha256 and not (trial/'failure.json').exists()
        and receipt['state']=='owned_native848_pair_completed_and_buffers_replayed_pending_admission'
        and receipt['native_frames']==2 and receipt['training_approved'] is False,'Owned completed native848 pair required')
    intent=read_json(trial/'intent.json');owned=read_json(trial/'owned_exit.json');launch=read_json(trial/'launch.json')
    require(owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process'
        and sha256(trial/'owned_exit.json')==receipt['owned_exit_sha256']
        and sha256(trial/'launch.json')==owned['launch_sha256'] and launch['pid']==owned['pid'],'Owned native exit differs')
    require(launch['plan_sha256']==intent['plan_sha256']==receipt['plan_sha256'],'Plan receipt mismatch')
    verify_bindings(intent['runtime_owner_bindings']);verify_bindings(intent['implementation_bindings'])
    plan_path=Path(intent['plan_path']);plan=read_json(plan_path)
    generated=plan_api.check(plan,full=True)
    saved_audit=read_json(trial/'audit.json')
    require(sha256(trial/'audit.json')==receipt['audit_sha256'],'Changed native audit')
    audit=audit_api.audit_capture(trial/'capture',plan_path=plan_path,plan_sha256=intent['plan_sha256'],result_sha256=sha256(trial/'capture/result.json'))
    require(audit==saved_audit,'Independent848 buffer/geometry replay differs')
    require(launch['command'][-1]==str(trial/'capture') and 'sim_data.native848_pair_worker_v1' in launch['command'],'Wrong native worker route')
    implementation=implementation_bindings();checker=WorkspaceChecker()
    pins={**audit['source_bindings'],**{str(trial/name):sha256(trial/name) for name in ('result.json','audit.json','intent.json','owned_exit.json','launch.json')}}
    _,all_reports=load_plan(plan['source_collection_plan']);reports={r['plant_id']:r for r in all_reports}
    output.mkdir(parents=True)
    write_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),trial=str(trial),trial_result_sha256=result_sha256,
        native_plan_sha256=intent['plan_sha256'],source_bindings=pins,implementation_bindings=implementation,
        workspace_implementation_bindings=checker.bindings,annotation_epoch=queries.ANNOTATION_EPOCH,resolution=[848,408],
        training_approved=False,generated_target_novelty_qualified=False))
    records=[]
    try:
        for record in audit['records']:
            mode=record['mode'];folder=trial/'capture'/mode;meta=read_json(folder/'sample.json')
            require(meta['schema_version']==plan_api.SAMPLE_SCHEMA and meta['geometry_screen']['passed'] is True,'Explicit848 producer/current geometry required')
            rgb=np.asarray(Image.open(folder/'inputs/rgb.png'));depth=np.load(folder/'inputs/depth_m.npy',allow_pickle=False)
            valid=np.asarray(Image.open(folder/'inputs/depth_valid.png'))==255
            components=np.load(folder/'supervision/component_id.npy',allow_pickle=False)
            catalogue=read_json(folder/'supervision/identities.json')['component_catalogue']
            target_mask=np.asarray(Image.open(folder/'supervision/target_visible.png'))
            report=reports[plan['source_family']] if mode=='original_control' else generated['report']
            baseline=labels.derive(meta,report,rgb,depth,valid,components,catalogue)
            label,trace,selection=queries.annotate_v1(meta,report,rgb,depth,valid,components,catalogue,target_mask=target_mask,expected_label=baseline)
            proof=checker.check_sample(folder/'sample.json',record['sample_sha256'])
            require(proof['target_id']==label['target_id'] and np.allclose(proof['nominal_world_m'],label['nominal_world_m'],atol=1e-8,rtol=0),'Workspace refers to another target')
            path=output/mode;path.mkdir()
            write_json(path/'baseline_label.json',baseline);write_json(path/'label.json',label);write_json(path/'query_selection.json',selection)
            write_json(path/'workspace.json',proof)
            if trace is not None:write_json(path/'query_trace.json',trace)
            local=label['eligible'] is True;trace_pass=bool(trace and trace['passed'] is True);workspace=proof['result']['workspace_passed'] is True
            decision='exclude_local_clarity' if not local else 'hold_full_trace' if not trace_pass else 'hold_workspace' if not workspace else 'candidate_pending_individual_visual_review_and_global_grouping'
            out=dict(mode=mode,target_id=label['target_id'],source_target=plan['conservative_view_cap_group'],source_family=plan['source_family'],split='train',
                source_sample_path=str(folder/'sample.json'),source_sample_sha256=record['sample_sha256'],rgb_path=str(folder/'inputs/rgb.png'),rgb_sha256=record['rgb_sha256'],
                label_path=str(path/'label.json'),label_sha256=sha256(path/'label.json'),query_trace_sha256=sha256(path/'query_trace.json') if trace else None,
                workspace_path=str(path/'workspace.json'),workspace_sha256=sha256(path/'workspace.json'),query_selection_sha256=sha256(path/'query_selection.json'),
                local_clarity_passed=local,full_trace_and_anchored_grid_passed=trace_pass,workspace_passed=workspace,decision=decision,
                candidate_for_individual_visual_review=local and trace_pass and workspace,original_control_is_new_target_diversity=False,
                source_cap_reset=False,geometry_novelty_qualified=False,training_approved=False,accepted_training_increment=0)
            records.append(out);print('NATIVE848_PAIR_ADMISSION',mode,decision,flush=True)
        checker.finish();verify_bindings(implementation);verify_bindings(pins)
        require(implementation_bindings()==implementation,'Admission code closure changed')
        write_json(output/'result.json',dict(state='explicit_native848_pair_fresh_labels_trace_workspace_complete_pending_individual_review_and_global_admission',records=records,
            candidates_for_individual_visual_review=sum(r['candidate_for_individual_visual_review'] for r in records),source_bindings=pins,
            visual_review_performed=False,geometry_novelty_qualified=False,near_duplicate_graph_rebuilt=False,source_cap_reset=False,training_approved=False,accepted_training_increment=0))
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc(),completed=records,training_approved=False));raise
