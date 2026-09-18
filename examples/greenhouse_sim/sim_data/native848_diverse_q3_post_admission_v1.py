"""Q3 post-admission for owned original short/direct captures; never exports."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import traceback
import numpy as np
from .dataset_review import read_json, write_json, require, verify_bindings
from .depth_preview import sha256
from .audit import audit_manifest
from .training_export import view_signature
from .native848_pair_audit_v2 import image_array
from . import native848_clear_labels_v1 as labels
from . import native848_query_selection_v3 as queries

SCHEMA = 'greenhouse.native848_diverse_q3_post_admission.v1'
STATE = 'original_native_q3_post_admission_complete_pending_individual_review'
Q3_SHA = '69bc1babbe462d7dee107c1a70832faaa688579cebe3493928119e98a7b5a462'
Q1_SHA = '896c81e5900ed31eb5a52989e15517a47d367b9aacb452bd8580e47260fe3964'
ROOT = Path('D:/research/tomato-pi-policy')
POLICY_REVIEW = ROOT/'data/sim_data/diagnostics/native848_future_query_route_root_review_20260916_v1/result.json'
POLICY_SHA = '7a6479039d976f87a27fc08a03c00d5dff77ba5baa5473f1314ee8e9b13f68f4'
ROUTES = {
 'original848_short_labels_trace_workspace_complete_pending_review_and_global_admission': (
  'owned_original848_short_completed_and_audited', 'sim_data.native848_original_short_worker_v1',
  'greenhouse.original848_short_buffer_geometry_audit.v1', 'greenhouse.original848_short_camera_sample.v1'),
 'original848_reference_direct_labels_trace_workspace_complete_pending_review_and_global_admission': (
  'owned_original848_reference_direct_completed_and_audited', 'sim_data.native848_original_direct_worker_v2',
  'greenhouse.original848_reference_direct_buffer_geometry_audit.v2', 'greenhouse.original848_reference_direct_camera_sample.v2'),
}
CANDIDATE = 'candidate_pending_individual_visual_review_and_global_grouping'


def artifact(path, pins, expected=None):
    path = Path(path).resolve()
    require(path.is_file(), 'Missing artifact: '+str(path))
    actual = sha256(path)
    require(expected is None or actual == expected, 'Changed artifact: '+str(path))
    require(str(path) not in pins or pins[str(path)] == actual, 'Conflicting artifact pin')
    pins[str(path)] = actual
    return dict(path=str(path), sha256=actual)


def bound_json(path, pins, expected=None):
    return read_json(artifact(path, pins, expected)['path'])


def argument(command, flag):
    require(isinstance(command, list) and command.count(flag) == 1, 'One exact worker argument required')
    return command[command.index(flag)+1]


def prior_candidate(row):
    flags = ('local_clarity_passed', 'full_trace_and_anchored_grid_passed',
             'workspace_passed', 'render_profile_qualified')
    all_pass = all(row.get(k) is True for k in flags)
    require(row.get('candidate_for_individual_visual_review') is
            (all_pass and row.get('decision') == CANDIDATE), 'Inconsistent old candidate decision')
    return all_pass and row.get('decision') == CANDIDATE


def q3_pass(label, trace, selection):
    chosen = selection.get('selected_query')
    return bool(label.get('eligible') is True and trace and trace.get('passed') is True
        and trace.get('schema') == 'greenhouse.native848_query_trace.v3'
        and trace.get('annotation_epoch') == queries.ANNOTATION_EPOCH
        and trace.get('legacy_trace', {}).get('passed') is True
        and trace.get('fixed_grid', {}).get('passed') is True
        and trace.get('route_separation', {}).get('passed') is True
        and label.get('annotation_epoch') == queries.ANNOTATION_EPOCH
        and selection.get('schema') == queries.ANNOTATION_EPOCH
        and chosen and chosen.get('both_passed') is True and chosen.get('all_checks_passed') is True
        and chosen.get('route_separation', {}).get('passed') is True
        and not trace.get('original_query_retained_as_failed_fallback'))


def load_owned(admission_path, admission_sha256):
    admission_path = Path(admission_path).resolve()
    pins = {}
    old = bound_json(admission_path, pins, admission_sha256)
    require(old.get('state') in ROUTES and old.get('training_approved') is False,
            'Unsupported original admission route')
    require(not (admission_path.parent/'failure.json').exists(), 'Old admission failed')
    request = bound_json(admission_path.parent/'request.json', pins)
    trial = Path(request['trial']).resolve()
    complete = bound_json(trial/'owner_complete.json', pins)
    require(complete.get('training_approved') is False
        and Path(complete['admission_path']).resolve() == admission_path
        and complete['admission_sha256'] == admission_sha256, 'Whole owner did not complete this admission')
    receipt = bound_json(trial/'result.json', pins, request['trial_result_sha256'])
    require(complete.get('result_sha256', complete.get('trial_result_sha256')) == sha256(trial/'result.json'),
            'Changed owner completion')
    state, worker, audit_schema, sample_schema = ROUTES[old['state']]
    require(receipt['state'] == state and receipt['training_approved'] is False
        and not (trial/'failure.json').exists() and not (trial/'capture/failure.json').exists(),
        'Owned original capture failed')
    owned = bound_json(trial/'owned_exit.json', pins, receipt['owned_exit_sha256'])
    launch = bound_json(trial/'launch.json', pins, owned['launch_sha256'])
    intent = bound_json(trial/'intent.json', pins)
    require(owned['returncode'] == 0 and owned['method'] == 'subprocess_wait_on_owned_process'
        and launch['pid'] == owned['pid'] and argument(launch['command'], '-m') == worker
        and Path(argument(launch['command'], '--output')).resolve() == trial/'capture',
        'Wrong owned native route or exit')
    plan = bound_json(intent['plan_path'], pins, intent['plan_sha256'])
    require(launch['plan_sha256'] == receipt['plan_sha256'] == intent['plan_sha256']
        and Path(argument(launch['command'], '--plan')).resolve() == Path(intent['plan_path']).resolve()
        and argument(launch['command'], '--plan-sha256') == intent['plan_sha256']
        and plan['resolution'] == [848,408] and plan['generated_geometry_used'] is False
        and plan['training_approved'] is False, 'Changed original plan')
    if worker.endswith('short_worker_v1'):
        require(plan['mode'] == 'production' and plan['render_budget_subframes'] == 8,
                'Adaptation controls are never production')
    audit = bound_json(trial/'audit.json', pins, receipt['audit_sha256'])
    capture_result = trial/'capture/result.json'
    capture_hash = receipt.get('capture_result_sha256', audit['source_bindings'].get(str(capture_result)))
    require(isinstance(capture_hash, str) and len(capture_hash) == 64, 'Native result lacks owned audit pin')
    bound_json(capture_result, pins, capture_hash)
    require(audit['schema'] == audit_schema and audit['training_approved'] is False,
            'Wrong independently completed native audit')
    require(old['source_bindings'].get(str(trial/'audit.json')) == receipt['audit_sha256'],
            'Old admission not bound to native audit')
    for source in (intent['runtime_owner_bindings'], plan['implementation_bindings'],
                   request['source_bindings'], old['source_bindings'], audit['source_bindings']):
        for path, digest in source.items():
            require(path not in pins or pins[path] == digest, 'Conflicting old source binding')
            pins[path] = digest
    verify_bindings(pins)
    cache = bound_json(plan['cache_path'], pins, plan['cache_sha256'])
    anchor = bound_json(cache['anchor_reference_path'], pins, cache['anchor_reference_sha256'])
    source_plan = bound_json(cache['source_collection_plan'], pins, cache['source_collection_plan_sha256'])
    jobs = [j for j in source_plan['jobs'] if j['plant_family'] == anchor['source_family'] and j['split'] == 'train']
    require(anchor['split'] == 'train' and jobs, 'Original source family is not TRAIN')
    paths = {str(Path(j['source_manifest_path']).resolve()) for j in jobs}
    require(len(paths) == 1, 'Ambiguous family source manifest')
    manifest = next(iter(paths))
    require(manifest in cache['source_bindings'], 'Family manifest lacks original pin')
    artifact(manifest, pins, cache['source_bindings'][manifest])
    report = audit_manifest(Path(manifest))
    require(report['plant_id'] == anchor['source_family'], 'Wrong source report')
    rows = {r['target_id']:r for j in jobs for r in j['targets']}
    records = {r['sample_id']:r for r in audit['records']}
    require(len(records) == len(audit['records']) == len(old['records'])
        and set(records) == {r['sample_id'] for r in old['records']},
        'Incomplete or duplicate native/admission population')
    for r in old['records']:
        source_id = records[r['sample_id']].get('source_pose_id', r['sample_id'])
        source = next(x for x in cache['records'] if x['sample_id'] == source_id)
        require(source['source_row'] == rows[r['source_target']]
            and r['split'] == 'train' and r['source_family'] == anchor['source_family'],
            'Changed original source target/split')
    return old, trial, plan, anchor, report, records, sample_schema, pins


def run(admission_result, *, admission_sha256, output):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),
            'New workspace diagnostic output required')
    old, trial, plan, anchor, report, audited, sample_schema, pins = load_owned(admission_result, admission_sha256)
    require(not output.is_relative_to(trial) and not trial.is_relative_to(output)
        and not output.is_relative_to(Path(admission_result).parent), 'Disjoint post-admission output required')
    artifact(__file__, pins)
    artifact(queries.__file__, pins, Q3_SHA)
    artifact(Path(queries.__file__).with_name('native848_query_selection_v1.py'), pins, Q1_SHA)
    artifact(POLICY_REVIEW, pins, POLICY_SHA)
    output.mkdir(parents=True)
    records, excluded = [], []
    write_json(output/'request.json', dict(schema=SCHEMA, admission_result=str(Path(admission_result).resolve()),
        admission_sha256=admission_sha256, trial=str(trial), annotation_epoch=queries.ANNOTATION_EPOCH,
        source_bindings=pins, training_approved=False))
    try:
        for prior in old['records']:
            name = prior['sample_id']
            require(Path(name).name == name and name not in ('.','..'), 'Unsafe sample ID')
            audit = audited[name]
            folder = trial/'capture'/name
            source = Path(prior['source_sample_path']).resolve()
            require(source == folder/'sample.json' and audit['sample_path'] == str(source)
                and prior['source_sample_sha256'] == audit['sample_sha256']
                and prior['rgb_sha256'] == audit['rgb_sha256'], 'Changed audited native identity')
            assets = {}
            for key, path, digest in [
                ('sample',source,prior['source_sample_sha256']),
                ('rgb',folder/'inputs/rgb.png',prior['rgb_sha256']),
                ('original_label',prior['label_path'],prior['label_sha256']),
                ('workspace',prior['workspace_path'],prior['workspace_sha256']),
                ('depth',folder/'inputs/depth_m.npy',None),('validity',folder/'inputs/depth_valid.png',None),
                ('target_mask',folder/'supervision/target_visible.png',None),
                ('renderer_instance_id',folder/'supervision/renderer_instance_id.npy',None),
                ('component_id',folder/'supervision/component_id.npy',None),
                ('identities',folder/'supervision/identities.json',None)]:
                if key not in ('original_label', 'workspace'):
                    require(str(Path(path).resolve()) in pins, 'Native input missing audit pin')
                assets[key] = artifact(path,pins,digest)
            meta, original_label, workspace = (read_json(assets[k]['path']) for k in ('sample','original_label','workspace'))
            require(meta['schema_version'] == sample_schema and meta['sample_id'] == name
                and meta['original_geometry_only'] is True and meta['generated_plant_native_pixels'] == 0
                and meta['calibration']['resolution'] == [848,408]
                and meta['geometry_screen']['passed'] is True
                and audit['same_callback_buffers_replayed'] is True
                and audit['full_native_identity_masks_replayed'] is True
                and audit['metric_cut_geometry_recomputed'] is True,
                'Original native geometry/buffer audit missing')
            require(workspace['target_id'] == original_label['target_id'] == prior['target_id']
                and Path(workspace['sample_path']).resolve() == source
                and workspace['sample_sha256'] == prior['source_sample_sha256']
                and np.allclose(workspace['nominal_world_m'],original_label['nominal_world_m'],atol=1e-8,rtol=0),
                'Workspace target or cut differs')
            verify_bindings(workspace['source_bindings'])
            rgb = image_array(assets['rgb']['path'])
            require(rgb.shape == (408,848,3), 'Native848 RGB required')
            out = dict(prior, id=name, original_decision=prior['decision'],
                source_plant_family=prior['source_family'], source_family=prior['source_family'],
                annotation_epoch=queries.ANNOTATION_EPOCH, individual_visual_review=False,
                artifacts=assets, source_artifacts=dict(assets),
                decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                conservative_camera_signature=view_signature(meta,prior['source_family'],None),
                original_label_path=prior['label_path'],original_label_sha256=prior['label_sha256'],
                original_admission_result=artifact(admission_result,pins,admission_sha256),
                original_audit=artifact(trial/'audit.json',pins),
                original_owner_complete=artifact(trial/'owner_complete.json',pins),
                old_automated_candidate=prior_candidate(prior),
                prior_candidate_for_individual_visual_review=prior_candidate(prior), q3_passed=False,
                source_cap_reset=False, geometry_novelty_qualified=False,
                training_approved=False,accepted_training_increment=0)
            if not out['old_automated_candidate']:
                out.update(decision='preserved_old_'+prior['decision'],candidate_for_individual_visual_review=False)
                excluded.append(out)
                continue
            require(workspace['result']['workspace_passed'] is True, 'Old candidate workspace no longer passes')
            depth=np.load(assets['depth']['path'],allow_pickle=False)
            valid=image_array(assets['validity']['path'])==255
            components=np.load(assets['component_id']['path'],allow_pickle=False)
            identities=read_json(assets['identities']['path'])
            mask=image_array(assets['target_mask']['path'])
            renderer=np.load(assets['renderer_instance_id']['path'],allow_pickle=False)
            mapping={int(k):v for k,v in identities['renderer_id_to_prim'].items()}
            baseline=labels.derive(meta,report,rgb,depth,valid,components,identities['component_catalogue'])
            label,trace,selection=queries.annotate_v3(meta,report,rgb,depth,valid,components,
                identities['component_catalogue'],target_mask=mask,renderer_ids=renderer,
                renderer_mapping=mapping,expected_label=baseline)
            require(label['answer']==original_label['answer'] and label['target_id']==prior['target_id'],
                'Q3 changed cut answer or target')
            dest=output/name;dest.mkdir()
            for key,value in [('label',label),('query_trace',trace),('query_selection',selection),('calibration',meta['calibration'])]:
                path=dest/(key+'.json');write_json(path,value);assets[key]=artifact(path,pins)
                out[key+'_path']=str(path);out[key+'_sha256']=assets[key]['sha256']
            passed=q3_pass(label,trace,selection)
            out.update(q3_passed=passed, candidate_for_individual_visual_review=passed,
                local_clarity_passed=label['eligible'] is True,
                full_trace_and_anchored_grid_passed=bool(trace and trace['legacy_trace']['passed'] and trace['fixed_grid']['passed']),
                route_separation_passed=bool(trace and (trace.get('route_separation') or {}).get('passed')),
                decision='candidate_pending_individual_visual_review' if passed else 'hold_q3_route_separation')
            (records if passed else excluded).append(out)
        require(len(records)+len(excluded)==len(old['records']), 'Lost original population member')
        verify_bindings(pins)
        result=dict(schema=SCHEMA,state=STATE,created_utc=datetime.now(timezone.utc).isoformat(),
            implementation=artifact(__file__,pins),
            original_admission=artifact(admission_result,pins,admission_sha256),
            original_audit=artifact(trial/'audit.json',pins),
            owner_complete=artifact(trial/'owner_complete.json',pins),
            annotation_epoch=queries.ANNOTATION_EPOCH,query_selector_sha256=Q3_SHA,
            query_selector_dependencies={str(Path(queries.__file__).with_name('native848_query_selection_v1.py').resolve()):Q1_SHA},
            original_admission_result=str(Path(admission_result).resolve()),original_admission_sha256=admission_sha256,
            original_trial=str(trial),original_record_count=len(old['records']),records=records,excluded_records=excluded,
            candidate_count=len(records),excluded_count=len(excluded),source_bindings=pins,
            old_geometry_workspace_and_audit_gates_preserved=True,old_exclusions_or_holds_revived=False,
            visual_review_performed=False,all_pilot_candidates_require_individual_review=True,
            dataset_export_performed=False,training_approved=False,accepted_training_increment=0)
        write_json(output/'result.json',result)
        return result
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc(),records=records,excluded_records=excluded,
            training_approved=False,accepted_training_increment=0))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission-result',type=Path,required=True)
    parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=run(a.admission_result,admission_sha256=a.admission_sha256,output=a.output)
    print('DIVERSE_Q3_POST_ADMISSION_COMPLETE',r['candidate_count'],r['excluded_count'])

if __name__=='__main__':
    main()

