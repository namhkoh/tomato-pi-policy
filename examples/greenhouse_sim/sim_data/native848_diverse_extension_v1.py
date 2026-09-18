"""Extend only the exact user-approved477-image diverse base; never ship rejected bulk.

Creates a new cumulative registry referencing the portable base and copying only
fresh, independently admitted and individually reviewed, non-dominant images.
No cap removes existing base images, no reannotation, and no training launch.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
from PIL import Image
from .training_export import view_signature
from .clear_cutpoint_contract import crop_box, crop_image
SCHEMA='greenhouse.native848_diverse_extension.v1'
REQUEST_SCHEMA='greenhouse.native848_diverse_extension_request.v1'
REVIEW_SCHEMA='greenhouse.native848_diverse_individual_review.v1'
EPOCH='greenhouse.native848_query_selection.v3'
Q4_EPOCH='greenhouse.native848_query_selection.v4'
WORKSPACE=Path(__file__).resolve().parents[3]
OUTPUT_ROOT=WORKSPACE/'data/sim_data/dataset_checkpoints'
BASE=OUTPUT_ROOT/'tomato_cutpoint_848x408_diverse_base_20260917_v1'
BASE_RESULT_SHA='9b2c0433dc56e2868f196a4222fdbda25aca622e420c75a8a55cf602c7b452e3'
EXCLUDED_TARGET='seed41_full/SubStem_38'
ALIASES=('rgb_sha256','decoded_rgb_sha256','conservative_camera_signature')
HELPER_MODULES=[Path(__file__).resolve().with_name(name) for name in ('training_export.py','clear_cutpoint_contract.py','capture_contract.py')]
FRESH_NATIVE_ROOT=WORKSPACE/'data/sim_data/diagnostics/native848_diverse_noon_native_20260917_v1'
FROZEN_Q3_ADAPTER_SHA='cb43398abec73122d570589de3600c9ca6081a57c16eeb00132169e86043addc'


def require(value, message):
    if not value:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def rows_at(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def aliases(row):
    return {(key, row[key]) for key in ALIASES if row.get(key)}

class Pins:
    def __init__(self):
        self.values = {}

    def add(self, spec):
        require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'Exact path/SHA256 pin required')
        path = Path(spec['path'])
        require(path.is_absolute(), 'Absolute evidence path required')
        path = path.resolve()
        require(path.is_file(), 'Missing pinned file: ' + str(path))
        expected = spec['sha256']
        require(isinstance(expected, str) and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'Invalid SHA256')
        key = str(path)
        if key in self.values:
            require(self.values[key] == expected, 'Conflicting evidence pin')
        else:
            require(digest(path) == expected, 'Changed pinned file: ' + key)
            self.values[key] = expected
        return path

    def verify(self):
        for path, expected in self.values.items():
            require(digest(path) == expected, 'Source changed during snapshot: ' + path)

def family_target(row):
    family, target = row['source_plant_family'], row['source_target']
    require(target.startswith(family + '/'), 'Physical source target must include its frozen family')
    return family, target

def individually_reviewed(row):
    return (row.get('individual_visual_review') is True or
            (row.get('review', {}).get('decision') == 'accept' and
             row.get('review', {}).get('full_native_image_inspected') is True))

def distribution(rows):
    train = [row for row in rows if row['split'] == 'train']
    return dict(counts=dict(Counter(row['split'] for row in rows)),
        train_families=len({family_target(row)[0] for row in train}),
        train_physical_targets=len({family_target(row) for row in train}),
        train_by_family=dict(sorted(Counter(row['source_plant_family'] for row in train).items())),
        train_by_source_target=dict(sorted(Counter(row['source_target'] for row in train).items())))

def validate_review(actual, row):
    require(actual['sample_id'] == row['sample_id'] and actual['rgb_sha256'] == row['rgb_sha256']
        and actual['label_sha256'] == row['artifacts']['label']['sha256']
        and actual['source_sample_sha256'] == row['source_sample_sha256'], 'Individual review artifact identity differs')
    require(actual['decision'] in ('accept', 'hold', 'reject') and actual['reviewer_type'] in ('assistant', 'human')
        and bool(actual['reviewer'].strip()) and bool(actual['reason'].strip()), 'Attributed individual review required')
    require(actual['full_native_image_inspected'] is True
        and actual['unscaled_lossless_association_crop_inspected'] is True, 'Actual full native and exact unscaled crop review required')
    if actual['decision'] == 'accept':
        require(actual['obvious_ghosting'] is False, 'Visible ghosting cannot be accepted')

def check_q4_proof(proof):
    require(proof['passed'] is True and proof['Q1passed'] is True
        and proof['protected_foreground_passed'] is True and proof['conditional_background_passed'] is True
        and proof['q3_passed'] is False and proof['background_exception_applied'] is True
        and proof['all_route_probes_recorded'] is True and proof['all_violating_pixels_at_each_probe_checked'] is True
        and proof['actual_individual_visual_review_required'] is True and proof['training_approved'] is False,
        'Explicit preserved Q1/foreground and conditional background proof required')
    require(len(proof['probes'])==proof['probe_count'] and len(proof['probes'])>0,
            'All actual Q4 trace probes required')
    endpoint=proof['endpoint']
    require(endpoint['q3_probe']['passed'] is True and endpoint['background_exception_applied'] is False
        and endpoint['protected_foreground_passed'] is True and endpoint['conditional_background_passed'] is True,
        'Query endpoint cannot receive a background exception')
    exceptions=0
    for probe in proof['probes']:
        require(probe['protected_foreground_passed'] is True and probe['conditional_background_passed'] is True,
                'A route probe failed protected gates')
        if probe['q3_probe']['passed']:
            require(probe['background_exception_applied'] is False,'Unnecessary exception')
        else:
            pixel=probe['background_pixels']; exceptions+=1
            require(probe['q3_probe']['reasons']==['visible_foreign_plant_too_close_to_route']
                and probe['background_exception_applied'] is True and pixel['passed'] is True
                and pixel['all_violating_pixels_checked'] is True and pixel['violating_pixel_count']>0
                and pixel['minimum_centerline_distance_px']>=4.0 and pixel['minimum_depth_gap_m']>=.20,
                'Q4 background exception omitted or relaxed a violating pixel')
    require(exceptions==proof['exception_probe_count'] and exceptions>0,'Q4 exception count differs')


def check_candidate(row, pins, frozen):
    route = row.get('input_route', 'original_q3')
    require(route in ('original_q3', 'bulk_q3', 'bulk_q4'), 'Unknown native admission route')
    epoch=Q4_EPOCH if route=='bulk_q4' else EPOCH
    require(row['candidate_for_individual_visual_review'] is True
        and row['render_profile_qualified'] is True and row['workspace_passed'] is True
        and row['annotation_epoch'] == epoch and row['split'] == 'train'
        and row['source_cap_reset'] is False and row['geometry_novelty_qualified'] is False,
        'Only source-preserving native TRAIN candidates allowed')
    if route=='bulk_q4':
        require(row['prior_candidate_for_individual_visual_review'] is False and row['q3_passed'] is False
            and row['q4_passed'] is True and row['Q1passed'] is True
            and row['protected_foreground_passed'] is True and row['conditional_background_passed'] is True
            and row['background_exception_applied'] is True
            and row['original_decision']=='hold_query_trace_or_route_separation', 'Explicit fresh Q4-only status required')
    else:
        require(row['prior_candidate_for_individual_visual_review'] is True,'Q3 cannot revive a prior exclusion')
    family = row['source_family']
    require(frozen.get(family) == 'train' and row['source_target'].startswith(family + '/')
        and row['target_id'] == row['source_target'], 'Changed original source target/family/split')
    required = {'rgb', 'target_mask', 'sample', 'workspace', 'label', 'query_trace', 'query_selection'}
    required |= ({'depth', 'validity', 'renderer_instance_id', 'component_id', 'identities'}
                 if route == 'original_q3' else {'buffers', 'observation', 'context', 'mapping', 'workspace_replay'})
    if route=='bulk_q4':required|={'original_q3_label','original_q3_query_trace','original_q3_query_selection'}
    require(required <= set(row['artifacts']), 'Missing actual native or annotation artifacts')
    paths = {key: pins.add(value) for key, value in row['artifacts'].items()}
    require(row['rgb_sha256'] == row['artifacts']['rgb']['sha256']
        and row['source_sample_sha256'] == row['artifacts']['sample']['sha256'], 'Row native source hashes differ')
    label, trace, selection, workspace, meta = [read(paths[key]) for key in ('label','query_trace','query_selection','workspace','sample')]
    require(label['task'] == 'greenhouse.native848_curved_clear_cutpoint_rgb.v1'
        and label['eligible'] is True and label['target_id'] == row['target_id']
        and label['annotation_epoch'] == epoch and label['answer']['status'] == 'localized'
        and label['answer']['visibility'] == 'clear', 'Fresh clear cut label required')
    chosen = selection['selected_query']
    require(selection['schema'] == epoch and selection['annotation_epoch'] == epoch and chosen is not None
        and chosen['both_passed'] is True and trace['passed'] is True
        and trace['legacy_trace']['passed'] is True and trace['fixed_grid']['passed'] is True
        and trace['annotation_epoch'] == epoch and trace['original_query_retained_as_failed_fallback'] is False
        and label['query_pixel_uv'] == trace['query_pixel_uv'] == chosen['query_pixel_uv'], 'Full unchanged grid/trace gates required')
    if route=='bulk_q4':
        require(trace['schema']=='greenhouse.native848_query_trace.v4' and trace['original_q3_passed'] is False
            and chosen['conditional_checks_passed'] is True
            and chosen['conditional_background']==trace['conditional_background'], 'Conditional Q4 trace differs')
        check_q4_proof(trace['conditional_background'])
        old_label,old_trace,old_selection=[read(paths['original_q3_'+k]) for k in ('label','query_trace','query_selection')]
        require(old_trace['passed'] is False and old_trace['annotation_epoch']==EPOCH
            and old_label['answer']==label['answer'] and old_label['target_id']==label['target_id']
            and old_label['nominal_world_m']==label['nominal_world_m']
            and old_label['accepted_interval_uv']==label['accepted_interval_uv'], 'Original Q3 false/cut answer must be preserved')
        from .capture_contract import fingerprint
        require(trace['original_q3_trace_sha256']==fingerprint(old_trace)
            and selection['original_q3_selection_sha256']==fingerprint(old_selection), 'Q4 original Q3 content differs')
    else:
        require(chosen['all_checks_passed'] is True and chosen['route_separation']['passed'] is True
            and trace['schema']=='greenhouse.native848_query_trace.v3' and trace['route_separation']['passed'] is True,
            'Full Q3 route separation required')
    require(workspace['result']['workspace_passed'] is True and workspace['target_id'] == row['target_id']
        and workspace['nominal_world_m'] == label['nominal_world_m'], 'Bound passing workspace proof required')
    if route == 'original_q3':
        require(workspace['schema'] == 'greenhouse.clear848_kinematic_workspace.v2'
            and workspace['sample_sha256'] == row['source_sample_sha256'], 'Original workspace sample binding differs')
    else:
        replay = row['bulk_workspace_replay']
        require(read(paths['workspace_replay']) == replay, 'Saved bulk workspace replay receipt differs')
        require(workspace['schema'] == 'greenhouse.native848_bulk_workspace.v1'
            and workspace['per_frame_camera_FK_verified'] is True
            and workspace['per_frame_cached_solution_FK_verified'] is True
            and replay['sample_sha256'] == row['source_sample_sha256']
            and replay['workspace_sha256'] == row['artifacts']['workspace']['sha256']
            and replay['solve_input_sha256'] == workspace['solve_input_sha256']
            and replay['per_frame_camera_FK_verified'] is True
            and replay['per_frame_cached_solution_FK_verified'] is True
            and meta['observation_sha256'] == row['artifacts']['observation']['sha256'],
            'Bulk workspace input/FK/observation binding differs')
    with Image.open(paths['rgb']) as image:
        require(image.size == (848, 408) and image.mode == 'RGB' and image.format == 'PNG', 'Lossless native848 RGB PNG required')
        decoded = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    camera = view_signature(meta, family, None)
    require(row['decoded_rgb_sha256'] == decoded and row['conservative_camera_signature'] == camera, 'Actual decoded RGB or source-camera identity mismatch')
    box = crop_box(label['query_pixel_uv'])
    interval = np.asarray(label['accepted_interval_uv'], dtype=float)
    require(interval.ndim == 2 and interval.shape[1] == 2 and len(interval) >= 2 and np.isfinite(interval).all()
        and ((interval >= box[:2]) & (interval < box[2:])).all(), 'Query-only training crop omits accepted cut interval')
    return label, paths

def load_base(request,pins):
    path=pins.add(request['base_result'])
    require(path==BASE/'result.json' and request['base_result']['sha256']==BASE_RESULT_SHA,
            'Only the exact user-approved477 base is permitted')
    result=read(path)
    require(result['schema']=='greenhouse.native848_diverse_base_export.v1'
        and result['state']=='exact_user_selected_diverse_base_exported'
        and result['counts']==dict(train=477,validation=48,test=90)
        and result['dominant_target_images']==0, 'Wrong diverse base')
    for name in ('index.jsonl','status.json','global_exclusion_aliases.json','manifest.json'):
        p=pins.add(result['output_bindings'][name]); require(p==BASE/name,'Base metadata path differs')
    rows=rows_at(BASE/'index.jsonl'); status=read(BASE/'status.json')
    require(dict(Counter(r['split'] for r in rows))==result['counts']
        and len({r['id'] for r in rows})==len(rows),'Base population changed')
    frozen=status['frozen_family_splits']
    for row in rows:
        require(row['source_target']!=EXCLUDED_TARGET and row['target_id']!=EXCLUDED_TARGET
            and frozen.get(row['source_plant_family'])==row['split'],'Rejected target or changed base split')
        row.update(asset_root=str(BASE),registry_origin='user_selected_diverse_base')
    global_aliases=read(BASE/'global_exclusion_aliases.json')
    require(global_aliases['schema']=='greenhouse.native848_global_exclusion_aliases.v1'
        and global_aliases['contains_images'] is False and global_aliases['contains_training_records'] is False,
        'Anonymous global exclusion scope changed')
    seen={(r['kind'],r['value']) for r in global_aliases['accepted_identity_aliases']}
    holds={(r['kind'],r['value']) for r in global_aliases['visual_hold_aliases']}
    require(not set().union(*(aliases(r) for r in rows)) & holds,'Existing base now held')
    return rows,frozen,seen,holds


def read_hold_snapshot(path):
    raw=Path(path).read_bytes()
    require(not raw or raw.endswith(b'\n'),'Incomplete visual hold ledger')
    rows=[json.loads(line) for line in raw.splitlines() if line]
    return set().union(*(aliases(row) for row in rows)) if rows else set()


def inspect(request_path,request_sha256):
    pins=Pins(); request_path=pins.add(dict(path=str(Path(request_path).resolve()),sha256=request_sha256))
    request=read(request_path)
    require(request['schema']==REQUEST_SCHEMA and request['preserve_all477_base_train'] is True
        and request['permanent_excluded_source_targets']==[EXCLUDED_TARGET]
        and request['training_approved'] is False,'Explicit unchanged477 base and permanent target exclusion required')
    require(request['implementation_bindings'].get(str(Path(__file__).resolve()))==digest(__file__)
        and all(request['implementation_bindings'].get(str(p))==digest(p) for p in HELPER_MODULES),
        'Exact extension and shared helper implementation pins required')
    for path,expected in request['implementation_bindings'].items():pins.add(dict(path=path,sha256=expected))
    base,frozen,seen,holds=load_base(request,pins)
    previous=request.get('previous_extension')
    if previous is not None:
        prior=read(pins.add(previous))
        require(prior['schema']==SCHEMA and prior['base_result']==request['base_result'], 'Previous extension has a different base')
        prior_request=read(pins.add(prior['output_bindings']['request.json']))
        canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
        require(set(map(canonical,prior_request['admissions']))<=set(map(canonical,request['admissions'])),
                'Cumulative extension omitted prior admissions or review holds')
    candidates,excluded,new_holds=[],[],[]


    for spec in request['admissions']:
        result_path=pins.add(spec['result']);result=read(result_path)
        require(result['training_approved'] is False,'Adapter cannot approve training')
        is_q4=result['schema']=='greenhouse.native848_diverse_bulk_q4_background_post.v1'
        normalized=result
        if is_q4:
            route='bulk_q4';expected_impl='native848_diverse_bulk_q4_post_v1.py'
            require(result['state']=='fresh_original_bulk_background_exception_pending_individual_review'
                and result['annotation_epoch']==Q4_EPOCH and result['input_route']==route
                and result['q3_outputs_unchanged'] is True and result['manual_or_historical_holds_revived'] is False
                and result['all_candidates_require_individual_visual_review'] is True,
                'Separate fresh conditional Q4 result required')
            require(result['root_policy_review'] is not None
                and result['root_policy_review']==request.get('approved_q4_policy'), 'Sealed exact root Q4 approval required')
            policy=read(pins.add(result['root_policy_review']))
            require(policy['background_exception_policy_approved'] is True and policy['training_approved'] is False
                and policy['blocking_findings']==[] and policy['selector_sha256']==result['selector']['sha256']
                and policy['post_wrapper_sha256']==result['implementation']['sha256'], 'Q4 policy/source approval differs')
            selector_path=pins.add(result['selector'])
            require(selector_path.name=='native848_query_selection_v4.py'
                and request['implementation_bindings'].get(str(selector_path))==result['selector']['sha256'], 'Q4 selector pin missing')
            normalized=read(pins.add(result['q3_normalization']))
            require(normalized['schema']=='greenhouse.native848_diverse_bulk_q3_adapter.v1'
                and normalized['implementation']['sha256']==FROZEN_Q3_ADAPTER_SHA
                and normalized['annotation_epoch']==EPOCH and normalized['old_holds_revived'] is False
                and normalized['labels_recomputed'] is False, 'Genuine unchanged Q3 normalization required')
            adapter_path=pins.add(normalized['implementation'])
            require(request['implementation_bindings'].get(str(adapter_path))==FROZEN_Q3_ADAPTER_SHA,'Frozen Q3 adapter binding missing')
            trial=Path(normalized['native_result']['path']).resolve().parent
            require(trial.is_relative_to(FRESH_NATIVE_ROOT),'Q4 cannot reannotate historical captures')
            require(result['unchanged_q3_candidates']==normalized['records'], 'Existing Q3 candidates changed or lost')
            original_excluded={r['sample_id']:r for r in normalized['excluded_records']}
            require(len(original_excluded)==len(normalized['excluded_records'])
                and len(result['records'])+len(result['excluded_records'])==len(original_excluded)
                and {r['sample_id'] for r in result['records']+result['excluded_records']}==set(original_excluded),
                'Q4 must account for the complete original Q3 hold population')
            snapshot=read_hold_snapshot(pins.add(result['preserved_hold_snapshot']))
            holds.update(snapshot)
            # Freeze the live ledger for this serialized export and recheck before commit.
            ledger=Path(result['original_hold_ledger_path']).resolve()
            require(ledger==OUTPUT_ROOT/'tomato_cutpoint_848x408_tonight_bulk_v1/visual_holds.jsonl',
                    'Unknown historical hold ledger')
            live=pins.add(dict(path=str(ledger),sha256=digest(ledger)));holds.update(read_hold_snapshot(live))
        elif result['schema']=='greenhouse.native848_diverse_q3_post_admission.v1':
            route='original_q3';expected_impl='native848_diverse_q3_post_admission_v1.py'
            require(result['state']=='original_native_q3_post_admission_complete_pending_individual_review'
                and result['annotation_epoch']==EPOCH and result['old_exclusions_or_holds_revived'] is False
                and result['old_geometry_workspace_and_audit_gates_preserved'] is True,
                'Dedicated original Q3 post-admission result required')
        else:
            route='bulk_q3';expected_impl='native848_diverse_bulk_adapter_v1.py'
            require(result['schema']=='greenhouse.native848_diverse_bulk_q3_adapter.v1'
                and result['state']=='original_bulk_q3_normalized_pending_individual_review'
                and result['annotation_epoch']==EPOCH and result['old_holds_revived'] is False
                and result['labels_recomputed'] is False, 'Unsupported canonical adapter')
        wrapper_path=pins.add(result['implementation'])
        require(wrapper_path.name==expected_impl
            and request['implementation_bindings'].get(str(wrapper_path))==result['implementation']['sha256'],
            'Exact reviewed adapter implementation binding required')
        evidence_keys=(('original_admission','original_audit','owner_complete') if route=='original_q3' else
            ('original_admission','native_result','owner_complete','cpu_owner_result','context','manifest','native_owned_exit'))
        for evidence in ([normalized,result] if is_q4 else [result]):
            for key in evidence_keys:
                path=pins.add(evidence[key])
                require(evidence['source_bindings'].get(str(path))==evidence[key]['sha256'],'Unbound original route evidence')
                if is_q4:require(evidence[key]==normalized[key],'Q4 changed a genuine native/CPU receipt')
            for path,expected in evidence['source_bindings'].items():pins.add(dict(path=path,sha256=expected))
        original=read(pins.add(normalized['original_admission']))
        if route=='original_q3':
            prior_key='sample_id'
            require(original['state'] in (
                'original848_short_labels_trace_workspace_complete_pending_review_and_global_admission',
                'original848_reference_direct_labels_trace_workspace_complete_pending_review_and_global_admission'),
                'Unsupported original admission route')
        else:
            prior_key='observation_id'
            require(original['schema']=='greenhouse.original848_bulk_admission.v4'
                and original['annotation_epoch']==EPOCH
                and original['state']=='automated_frame_checks_complete_pending_batch_QA_and_owned_capture_completion',
                'Actual Q3 bulk admission required')
        prior_records={r[prior_key]:r for r in original['records']}
        require(len(prior_records)==len(original['records']),'Repeated original admission sample ID')
        names=[r['sample_id'] for r in result['records']]
        if names:
            review=read(pins.add(spec['review']))
            require(review['schema']==REVIEW_SCHEMA and review['wrapper_result_sha256']==spec['result']['sha256'],
                    'Review must bind exact canonical wrapper result')
        else:
            require(spec['review'] is None,'No invented visual review for an empty eligible population');review=dict(reviews=[])
        reviews={r['sample_id']:r for r in review['reviews']}
        require(len(names)==len(set(names)) and len(reviews)==len(review['reviews']) and set(names)==set(reviews),
                'Every eligible new frame requires exactly one actual individual review')
        # Replay population from the genuine normalization; Q4 intentionally leaves Q3 candidates unchanged.
        all_names=[r['sample_id'] for r in normalized['records']+normalized['excluded_records']]
        geometry_names=set()
        if route!='original_q3':
            manifest=read(normalized['manifest']['path']);decisions={r['observation_id']:r for r in manifest['decisions']}
            require(len(decisions)==len(manifest['decisions'])==normalized['proposed_population'],'Changed complete native schedule')
            for er in normalized['excluded_records']:
                if er['decision']=='preserved_native_geometry_hold':
                    name=er['sample_id'];native=decisions[name]
                    require(name not in prior_records and native['state']=='rejected_native_geometry'
                        and native['native_requests']==0,'Geometry-only exclusion has an admitted image')
                    geometry_names.add(name)
            require(set(all_names)==set(decisions),'Lost native pilot population member')
        require(len(all_names)==len(set(all_names)) and set(all_names)-geometry_names==set(prior_records),
                'Normalization lost or duplicated an actual prior record')
        excluded.extend(dict(source_result_sha256=spec['result']['sha256'],record=r,reason='post_admission_excluded')
                        for r in result['excluded_records'])
        for row in result['records']:
            require(row['source_target']!=EXCLUDED_TARGET and row['target_id']!=EXCLUDED_TARGET,
                    'Permanently excluded dominant target cannot be appended')
            prior=prior_records[row['sample_id']]
            prior_pass=(prior['candidate_for_individual_visual_review'] if route=='original_q3' else prior['automated_pass'])
            sample_hash=prior['source_sample_sha256' if route=='original_q3' else 'sample_sha256']
            require(row.get('input_route','original_q3')==route and sample_hash==row['source_sample_sha256']
                and all(prior[k]==row[k] for k in ('rgb_sha256','source_family','source_target','target_id','split','workspace_sha256')),
                'Candidate differs from original actual source/workspace')
            if is_q4:
                require(prior_pass is False and prior['decision']=='hold_query_trace_or_route_separation',
                        'Q4 may only reconsider fresh Q3 route holds')
                previous_row=original_excluded[row['sample_id']]
                require(previous_row['q3_passed'] is False,'Q4 cannot relabel a passing Q3 record')
                for key,pin in previous_row['artifacts'].items():
                    new_key='original_q3_'+key if key in ('label','query_trace','query_selection') else key
                    require(row['artifacts'][new_key]==pin,'Q4 changed preserved source/Q3 artifacts')
                require(row['bulk_workspace_replay']==previous_row['bulk_workspace_replay'],'Q4 workspace replay changed')
            else:require(prior_pass is True,'Q3 candidate was previously excluded')
            require(row['artifacts']['workspace']['sha256']==prior['workspace_sha256'],'Workspace proof changed')
            label,paths=check_candidate(row,pins,frozen);actual=reviews[row['sample_id']];validate_review(actual,row)
            sid='diverse_'+hashlib.sha256(json.dumps([row['source_family'],row['source_target'],row['sample_id'],row['rgb_sha256']],
                                                  separators=(',',':')).encode()).hexdigest()[:24]
            normalized_row=dict(row,id=sid,source_plant_family=row['source_family'],label_sha256=row['artifacts']['label']['sha256'],
                registry_origin='new_supplement',individual_visual_review=actual['decision']=='accept',
                review_state=('conditional_q4_pass' if is_q4 else 'automated_q3_pass')+'_and_individual_visual_'+actual['decision'],
                review=actual,answer=label['answer'],annotation_schema=label['task'],query_pixel_uv=label['query_pixel_uv'],
                accepted_interval_uv=label['accepted_interval_uv'],
                lineage=dict(wrapper_result=spec['result'],individual_review=spec['review'],
                    q3_normalization=result.get('q3_normalization'),root_policy_review=result.get('root_policy_review')),
                kind='original_native848_diverse_supplement',training_approved=False)
            if actual['decision']!='accept':
                new_holds.append(normalized_row);holds.update(aliases(normalized_row))
                excluded.append(dict(id=sid,reason='actual_individual_visual_hold_or_reject'))
            else:candidates.append((normalized_row,paths))

    base_aliases=set().union(*(aliases(row) for row in base))
    require(not base_aliases & holds,'A new visual hold conflicts with the fixed base; explicit review resolution required')
    accepted=[]; ids={r['id'] for r in base}
    for row,paths in sorted(candidates,key=lambda pair:pair[0]['id']):
        reason=('preserved_visual_hold_identity' if aliases(row)&holds else
            'duplicate_any_old_corpus_or_new_RGB_or_camera' if aliases(row)&seen or row['id'] in ids else None)
        if reason:excluded.append(dict(id=row['id'],reason=reason));continue
        seen.update(aliases(row));ids.add(row['id']);accepted.append((row,paths))
    pins.verify()
    return dict(request=request,pins=pins,base=base,accepted=accepted,excluded=excluded,
                new_holds=new_holds,seen=seen,holds=holds,frozen=frozen)


def build(request_path,*,request_sha256,output):
    output=Path(output).resolve()
    require(output.parent==OUTPUT_ROOT and not output.exists(),'New sibling cumulative checkpoint required')
    checked=inspect(request_path,request_sha256)
    require(checked['accepted'],'No fresh individually accepted nonduplicate image to append')
    output.mkdir()
    try:
        supplements=[];portable={}
        for spec in checked['request']['admissions']:
            adapter=read(spec['result']['path'])
            primary=[adapter[k] for k in ('original_admission','original_audit','owner_complete','native_result',
                'cpu_owner_result','context','manifest','native_owned_exit','q3_normalization','root_policy_review',
                'preserved_hold_snapshot') if adapter.get(k) is not None]
            for item in [spec['result']]+([spec['review']] if spec['review'] is not None else [])+primary:
                relative='evidence/'+item['sha256']+'.json';dest=output/relative
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item['path'],dest)
                    require(digest(dest)==item['sha256'],'Copied primary evidence changed')
                portable[relative]=dict(source=item['path'],sha256=item['sha256'])
        for row,paths in checked['accepted']:
            sid=hashlib.sha256(row['id'].encode()).hexdigest()[:24];files={}
            for key,source in paths.items():
                require(key.replace('_','').isalnum(),'Unsafe artifact key')
                relative=Path('supplement')/sid/(key+source.suffix);dest=output/relative
                dest.parent.mkdir(parents=True,exist_ok=True);require(not dest.exists(),'Asset overwrite forbidden')
                shutil.copy2(source,dest);require(digest(dest)==row['artifacts'][key]['sha256'],'Copied artifact differs')
                files[key]=relative.as_posix()
            relative=Path('supplement')/sid/'crop.png'
            with Image.open(paths['rgb']) as image:crop_image(image,row['query_pixel_uv']).save(output/relative,compress_level=1)
            files['crop']=relative.as_posix()
            row.update(asset_root=str(output),files=files,native_calibration_file=files['sample'],
                crop_box_xyxy=crop_box(row['query_pixel_uv']),crop_source_dimensions=[384,384],crop_output_dimensions=[768,768],
                crop_depends_only_on_query=True,crop_full_interval_contained=True,crop_sha256=digest(output/relative))
            supplements.append(row)
        rows=checked['base']+supplements
        require(not any(r['source_target']==EXCLUDED_TARGET or r['target_id']==EXCLUDED_TARGET for r in rows),
                'Rejected target leaked into combined registry')
        require(sum(r['registry_origin']=='user_selected_diverse_base' and r['split']=='train' for r in rows)==477,
                'User base was reduced')
        jsonl(output/'index.jsonl',rows);jsonl(output/'supplement_index.jsonl',supplements)
        for split in ('train','validation','test'):jsonl(output/(split+'_index.jsonl'),[r for r in rows if r['split']==split])
        write(output/'request.json',checked['request']);write(output/'portable_evidence.json',portable)
        write(output/'exclusions.json',dict(records=checked['excluded']))
        write(output/'global_exclusion_aliases.json',dict(schema='greenhouse.native848_global_exclusion_aliases.v1',
            accepted_identity_aliases=[dict(kind=k,value=v) for k,v in sorted(checked['seen'])],
            visual_hold_aliases=[dict(kind=k,value=v) for k,v in sorted(checked['holds'])],
            contains_images=False,contains_training_records=False))
        status=dict(schema=SCHEMA,base_counts=dict(train=477,validation=48,test=90),
            **distribution(rows),new_individually_accepted_images=len(supplements),dominant_target_images=0,
            base_rows_removed=0,base_annotations_changed=False,new_annotation_epochs=dict(Counter(r['annotation_epoch'] for r in supplements)),
            new_capture_scope='Actual individually reviewed Q3 or explicitly approved conditional Q4 native captures; distinct epochs and original Q3 results preserved',
            frozen_family_splits=checked['frozen'],training_started=False,training_approved=False)
        write(output/'status.json',status)
        readme=('# Diverse extension of the exact 477-image base\n\n'
            'The complete 477 TRAIN base plus 48 validation and 90 test records are retained unchanged. '
            'Every new TRAIN image passed its explicit Q3 or approved conditional Q4 policy, workspace and individual full-image/crop review. '
            'seed41_full/SubStem_38 is permanently excluded. No cap discards base images, and no rejected bulk corpus is shipped.\n\n'
            'Each index row resolves assets as Path(row["asset_root"])/row["files"]["rgb"]. '
            'Copy the approved diverse_base package alongside this directory for portability; remap asset_root when moving them. '
            'Only fresh supplement assets are copied here. Old annotation/review scopes remain unchanged; '
            'New annotation epochs and original Q3 evidence are explicit per row. Primary provenance receipts are copied under evidence; transitive replay sources remain pinned external references. '
            'status.json gives exact counts by physical source target and family. No training job is started.\n')
        with (output/'README.md').open('x',encoding='utf-8') as stream:stream.write(readme)
        checked['pins'].verify()
        previous=checked['request'].get('previous_extension')
        previous_new=read(previous['path'])['accepted_new_images'] if previous else 0
        result=dict(schema=SCHEMA,state='diverse_base_plus_individually_reviewed_extension_complete',
            created_utc=datetime.now(timezone.utc).isoformat(),base_result=checked['request']['base_result'],
            accepted_new_images=len(supplements),added_since_previous_extension=len(supplements)-previous_new,
            counts=status['counts'],train_physical_targets=status['train_physical_targets'],
            train_families=status['train_families'],dominant_target_images=0,base_rows_removed=0,
            no_rejected_dominant_target_images_included=True,source_bindings=checked['pins'].values,
            output_bindings={name:dict(path=str(output/name),sha256=digest(output/name)) for name in
                ('index.jsonl','supplement_index.jsonl','train_index.jsonl','validation_index.jsonl','test_index.jsonl',
                 'status.json','request.json','portable_evidence.json','exclusions.json','global_exclusion_aliases.json','README.md')},
            previous_extension=previous,training_started=False,training_approved=False)
        write(output/'result.json',result);return result
    except BaseException as error:
        write(output/'failure.json',dict(error=repr(error),partial_output_not_admissible=True));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    result=build(a.request,request_sha256=a.request_sha256,output=a.output)
    print(json.dumps(dict(counts=result['counts'],accepted_new_images=result['accepted_new_images'])))


if __name__=='__main__':main()

