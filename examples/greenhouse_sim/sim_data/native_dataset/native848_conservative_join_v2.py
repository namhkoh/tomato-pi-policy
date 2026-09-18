"""Conservative CPU consumer adding the explicit ORIGINAL-only direct producer.

V1 and all capture/review files remain unchanged. Original-direct captures must
replay their own original-only audit and use the unchanged original target pool.
No native capture or review is fabricated. `inspect` never creates a selection.
The portable record/selection contract remains v1; consumer provenance is v2.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
from . import native848_conservative_join_v1 as base
from .native848_conservative_join_v1 import (SCHEMA, REQUEST_SCHEMA, CAP, Pins,
    read, save, digest, normalized_prior, held_aliases, identities,
    filter_identities, shared_cap_selection, select_views, view_signature)
from ..dataset_review import require

ORIGINAL_STATE = 'original848_direct_labels_trace_workspace_complete_pending_review_and_global_admission'
ORIGINAL_AUDIT = 'greenhouse.original848_direct_buffer_geometry_audit.v1'
ORIGINAL_SAMPLE = 'greenhouse.original848_direct_camera_sample.v1'


def original_identity_context(anchor, source_row, rec):
    """A label identity context, never represented as an authored capture plan."""
    require(anchor['split']=='train' and rec['split']=='train'
        and rec['source_family']==anchor['source_family'], 'Original source split/family differs')
    require(rec['target_id']==rec['source_target']==source_row['target_id']
        and source_row['target_id'].startswith(anchor['source_family']+'/')
        and source_row['variant_id']==anchor['source_family'], 'Generated or remapped target in original-only input')
    return dict(anchor, source_row=source_row, conservative_view_cap_group=source_row['target_id'])


def replay_admission(spec, pins):
    from .. import native848_original_direct_plan_v1 as api
    from .. import native848_original_direct_audit_v1 as audit_api
    result_path=Path(pins.binding(spec['result']));result=read(result_path)
    if result['state']!=ORIGINAL_STATE:
        return base.replay_admission(spec,pins)
    request_path=Path(pins.binding(spec['request']));request=read(request_path)
    require(request_path==result_path.with_name('request.json') and not result_path.with_name('failure.json').exists(),
        'Mismatched or failed original-only admission')
    require(all(result[k] is False for k in ('training_approved','geometry_novelty_qualified','source_cap_reset'))
        and request['resolution']==[848,408], 'Changed original-only admission scope')
    for bindings in (result['source_bindings'],request['source_bindings'],request['implementation_bindings'],request['workspace_implementation_bindings']):
        pins.extend(bindings)
    trial=Path(request['trial']);pins.add(trial/'result.json',request['trial_result_sha256'])
    receipt,owned,launch,intent=(read(trial/n) for n in ('result.json','owned_exit.json','launch.json','intent.json'))
    require(not (trial/'failure.json').exists() and not (trial/'capture/failure.json').exists()
        and receipt['state']=='owned_original848_direct_completed_and_audited'
        and receipt['training_approved'] is False,'Actual successful original-only completion required')
    for name in ('owned_exit.json','launch.json','intent.json','audit.json'):
        pins.add(trial/name,result['source_bindings'][str((trial/name).resolve())])
    require(owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process'
        and digest(trial/'owned_exit.json')==receipt['owned_exit_sha256']
        and digest(trial/'launch.json')==owned['launch_sha256'] and launch['pid']==owned['pid']
        and launch['plan_sha256']==intent['plan_sha256']==receipt['plan_sha256']
        and 'sim_data.native848_original_direct_worker_v1' in launch['command']
        and launch['command'][-1]==str(trial/'capture'), 'Original-only owned producer route differs')
    pins.extend(intent['runtime_owner_bindings'])
    if 'implementation_bindings' in intent:pins.extend(intent['implementation_bindings'])
    plan_path=pins.add(intent['plan_path'],intent['plan_sha256']);plan=read(plan_path)
    cache,anchor,original_report=api.check(plan,full=True)
    require(plan['generated_geometry_used'] is False and cache['generated_geometry_used'] is False
        and cache['frozen_original_assets'] is True and request['implementation_bindings']==plan['implementation_bindings'],
        'Unmodified original geometry contract required')
    pins.add(plan['cache_path'],plan['cache_sha256']);pins.extend(cache['source_bindings'])
    pins.add(cache['anchor_pair_plan_path'],cache['anchor_pair_plan_sha256'])
    actual=audit_api.audit_capture(trial/'capture',plan_path=plan_path,plan_sha256=intent['plan_sha256'],
        result_sha256=digest(trial/'capture/result.json'))
    require(actual['schema']==ORIGINAL_AUDIT and digest(trial/'audit.json')==receipt['audit_sha256']
        and actual==read(trial/'audit.json'),'Independent original-only native replay differs')
    pins.extend(actual['source_bindings'])
    reviews=read(pins.binding(spec['reviews']));require(isinstance(reviews,list),'Explicit actual review list required')
    keys=[(r['rgb_sha256'],r['label_sha256'],r['source_sample_sha256']) for r in reviews]
    require(len(keys)==len(set(keys)) and len({r['id'] for r in reviews})==len(reviews),'Ambiguous original-direct reviews')
    by_key=dict(zip(keys,reviews));used=set();candidates=[];negatives=[];excluded=[]
    audits={r['sample_id']:r for r in actual['records']};cached={r['sample_id']:r for r in cache['records']}
    require(len(audits)==len(actual['records']) and len({r['sample_id'] for r in result['records']})==len(result['records'])
        and set(audits)=={r['sample_id'] for r in result['records']},'Missing or repeated admitted original samples')
    for rec in result['records']:
        key=(rec['rgb_sha256'],rec['label_sha256'],rec['source_sample_sha256']);review=by_key.get(key)
        if review is None:
            excluded.append(dict(target_id=rec['target_id'],reason='not_individually_reviewed',rgb_sha256=rec['rgb_sha256']));continue
        used.add(key);audit=audits[rec['sample_id']];cached_row=cached[rec['sample_id']]
        source_row=cached_row['source_row'];context=original_identity_context(anchor,source_row,rec)
        meta=read(rec['source_sample_path'])
        require(meta['schema_version']==ORIGINAL_SAMPLE and meta['original_geometry_only'] is True
            and meta['generated_plant_native_pixels']==0
            and meta['supervision']['target_id']==meta['supervision']['source_target_id']==source_row['target_id'],
            'Actual metadata is not original-only')
        require(audit['target_id']==audit['source_target']==source_row['target_id']
            and audit['anchor_pair_plan_sha256']==cache['anchor_pair_plan_sha256'], 'Original audit ancestry differs')
        # Reuse only label/workspace/review logic; actual producer audit above is
        # original-only. The context is not passed to a pair producer or checker.
        row=base.verify_fresh_row(rec,review,audit,context,None,'original_control',pins)
        row['kind']='original_direct';row['original_geometry_only']=True
        row['lineage']=dict(admission_result=spec['result'],admission_request=spec['request'],reviews=spec['reviews'],
            native_plan=dict(path=plan_path,sha256=intent['plan_sha256']),
            original_pose_cache=dict(path=plan['cache_path'],sha256=plan['cache_sha256']),
            scene_anchor_pair_plan=dict(path=cache['anchor_pair_plan_path'],sha256=cache['anchor_pair_plan_sha256']),
            original_target=source_row['target_id'],source_family=anchor['source_family'],
            source_collection_plan=cached_row['source_collection_plan'],source_collection_plan_sha256=cached_row['source_collection_plan_sha256'],
            generated_geometry_used=False,generated_target=None,generator_version=None,
            scene_anchor_is_not_generated_capture_authority=True)
        if review['decision']!='accept':negatives.append(row)
        elif not rec['candidate_for_individual_visual_review']:excluded.append(dict(id=row['id'],reason=rec['decision']))
        else:candidates.append(row)
    require(used==set(by_key),'Review does not match an actual original-direct row')
    return candidates,negatives,excluded


def run(request_path, *, request_sha256, output, operation):
    require(operation in ('inspect','join'), 'Explicit inspect or join required')
    output = Path(output).resolve(); require(not output.exists(), 'New immutable output required')
    pins = Pins(); request_path = pins.add(request_path, request_sha256); request = read(request_path)
    require(request['schema'] == REQUEST_SCHEMA, 'Wrong request schema')
    pins.add(__file__); pins.add(base.__file__); pins.add(Path(select_views.__code__.co_filename)); pins.add(Path(view_signature.__code__.co_filename))
    prior_path = pins.binding(request['prior']); prior = read(prior_path)
    require(prior['training_approved'] is False and prior['max_views_per_original_target'] == CAP
        and prior['near_duplicate_graph_rebuilt'] is False and prior['global_geometry_novelty_qualified'] is False, 'Unsupported prior policy')
    pins.extend(prior['source_bindings'])
    original = [normalized_prior(r, pins) for r in prior['records']]
    holds, hold_keys = held_aliases(prior, request.get('hold_source_manifests', []), pins)
    incoming = []; excluded = []; new_holds = []
    for spec in request['admissions']:
        rows, negative, other = replay_admission(spec, pins)
        incoming.extend(rows); new_holds.extend(negative); excluded.extend(other)
    for row in new_holds: hold_keys.update(identities(row))
    holds.extend(new_holds)
    frozen = dict(prior['frozen_family_splits'])
    eligible, duplicate_exclusions = filter_identities(original + sorted(incoming, key=lambda r:r['id']), hold_keys, frozen)
    excluded.extend(duplicate_exclusions)
    prior_ids = {r['id'] for r in original}; new_ids = {r['id'] for r in incoming}
    pools = defaultdict(lambda:dict(prior=0, eligible_prior=0, eligible_incoming=0))
    for r in original: pools[r['source_target']]['prior'] += 1
    for r in eligible: pools[r['source_target']]['eligible_prior' if r['id'] in prior_ids else 'eligible_incoming'] += 1
    impact = {k:dict(v, cap=CAP, free_slots_before=max(0,CAP-v['prior']),
        maximum_pool_count_after=min(CAP,v['eligible_prior']+v['eligible_incoming']),
        maximum_net_pool_increment=min(CAP,v['eligible_prior']+v['eligible_incoming'])-v['prior'])
        for k,v in sorted(pools.items()) if v['eligible_incoming'] or any(r['source_target']==k for r in incoming)}
    result = dict(schema='greenhouse.native848_conservative_join_diagnostic.v1', operation=operation, consumer_module='native848_conservative_join_v2',
        created_utc=datetime.now(timezone.utc).isoformat(), prior_counts=prior['counts'],
        candidate_records_verified=len(incoming), identity_eligible_incoming=[r['id'] for r in eligible if r['id'] in new_ids and r['id'] not in prior_ids],
        cap_pressure=impact, exclusions=excluded, preserved_visual_holds=len(holds),
        source_cap_reset=False, max_views_per_original_target=CAP, geometry_novelty_qualified=False,
        near_duplicate_graph_rebuilt=False, global_geometry_novelty_qualified=False, training_approved=False,
        accepted_training_increment=0, existing_selection_modified=False, selection_created=False,
        source_bindings=pins.values)
    selection = None
    if operation == 'join':
        chosen = shared_cap_selection(eligible); selected = {r['id'] for r in chosen}
        exclusions = excluded + [dict(id=r['id'],reason='shared_original_target_view_cap') for r in eligible if r['id'] not in selected]
        counts = dict(Counter(r['split'] for r in chosen))
        selection = dict(schema=SCHEMA, state='reviewed_native848_conservative_mixed_provisional_not_release',
            records=chosen, counts=counts, excluded=exclusions, frozen_family_splits=frozen,
            preserved_visual_holds=[r['review'] for r in holds], preserved_hold_identity_rows=holds,
            source_bindings=pins.values, max_views_per_original_target=CAP, source_cap_reset=False,
            geometry_novelty_qualified=False, global_geometry_novelty_qualified=False,
            near_duplicate_graph_rebuilt=False, training_approved=False,
            generated_variants_receive_new_target_budget=False, original_exporter_compatible=False,
            crop_policy='existing_original_crops_preserved; fresh_rows_have_no_crop_until_query_only_export',
            selection_policy='unchanged_farthest_view_selector_with_original_source_target_group_keys',
            duplicate_policy='encoded_and_decoded_rgb_sample_existing_view_and_source_family_camera_intrinsics',
            independent_human_heldout_review_claimed=False, physical_execution_approved=False)
        result.update(selection_created=True, provisional_counts=counts,
            provisional_net_train_increment=counts.get('train',0)-prior['counts'].get('train',0),
            selected_new_ids=sorted(selected-prior_ids), removed_prior_ids=sorted(prior_ids-selected))
    for source in pins.values:
        require(not output.is_relative_to(Path(source).parent) and not Path(source).is_relative_to(output), 'Output must be disjoint from source folders')
    pins.finish(); output.mkdir(parents=True)
    if selection is not None:
        save(output/'selection.json',selection); result['selection_sha256']=digest(output/'selection.json')
    save(output/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('inspect','join'))
    p.add_argument('--request',type=Path,required=True); p.add_argument('--request-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=run(a.request,request_sha256=a.request_sha256,output=a.output,operation=a.operation)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_bindings','exclusions')},indent=2))


if __name__ == '__main__':
    main()
