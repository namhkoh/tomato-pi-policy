"""Explicit original reset8 consumer; frozen v1-v3 routes remain unchanged.

This CPU-only route replays sealed qualification and raw original adaptation,
then production labels/query/workspace and actual reviews. Controls get no
training credit and every original target retains its shared 12-view pool.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import numpy as np
from . import native848_conservative_join_v1 as base
from . import native848_conservative_join_v3 as previous
from . import native848_conservative_join_v2 as original_context
from .. import native848_original_short_admission_v1 as short_admission
from .native848_conservative_join_v1 import (SCHEMA, REQUEST_SCHEMA, CAP, Pins,
    read, save, digest, normalized_prior, held_aliases, identities,
    filter_identities, shared_cap_selection, select_views, view_signature)
from ..dataset_review import require

REFERENCE_STATE = short_admission.STATE
REFERENCE_AUDIT = short_admission.AUDIT_SCHEMA
REFERENCE_SAMPLE = short_admission.SAMPLE_SCHEMA
REFERENCE_PLAN = short_admission.PLAN_SCHEMA
REFERENCE_ANCHOR = 'greenhouse.original_native848_bank_reference.v2'
REFERENCE_WORKER = short_admission.WORKER
REFERENCE_RECEIPT = short_admission.OWNED_STATE


def _argument(command, flag):
    require(isinstance(command, list) and command.count(flag) == 1,
            'Unique explicit worker argument required: ' + flag)
    index = command.index(flag) + 1
    require(index < len(command), 'Missing worker argument: ' + flag)
    return command[index]


def check_owned_route(trial, receipt, owned, launch, intent, plan):
    return short_admission.check_owned_route(trial, receipt, owned, launch, intent, plan)


def original_reference_identity_context(anchor, source_row, rec, meta, audit, cache):
    """Bind the captured stem, which may differ from the scene reference stem."""
    require(anchor['schema_version'] == REFERENCE_ANCHOR
        and anchor['generated_geometry_used'] is False and anchor['source_cap_reset'] is False,
        'Authenticated unmodified original reference required')
    context = original_context.original_identity_context(anchor, source_row, rec)
    target = source_row['target_id']
    sup = meta['supervision']
    require(meta['schema_version'] == REFERENCE_SAMPLE and meta['sample_id'] == rec['sample_id']
        and meta['original_geometry_only'] is True and meta['generated_plant_native_pixels'] == 0
        and meta['capture_role'] == rec['capture_role'] == audit['capture_role'] == 'production'
        and rec['source_pose_id'] == audit['source_pose_id'] == meta['direct_camera']['cached_sample_id']
        and sup['target_id'] == sup['source_target_id'] == sup['conservative_view_cap_group'] == target
        and sup['split_group'] == anchor['source_family'], 'Actual metadata is not original-reference geometry')
    require(audit['sample_id'] == rec['sample_id'] and audit['target_id'] == audit['source_target'] == target
        and Path(audit['sample_path']).resolve() == Path(rec['source_sample_path']).resolve()
        and audit['anchor_reference_sha256'] == cache['anchor_reference_sha256']
        and Path(audit['anchor_reference_path']).resolve() == Path(cache['anchor_reference_path']).resolve(),
        'Original-reference audit ancestry differs')
    return context


replay_saved_workspace = previous.replay_saved_workspace


def replay_admission(spec, pins):
    result_path = Path(pins.binding(spec['result'])); result = read(result_path)
    if result['state'] != REFERENCE_STATE:
        return previous.replay_admission(spec, pins)
    from .. import native848_original_short_plan_v1 as api
    from .. import native848_original_short_audit_v1 as audit_api
    from .clear848_workspace_v2 import WorkspaceChecker
    request_path = Path(pins.binding(spec['request'])); request = read(request_path)
    require(request_path == result_path.with_name('request.json')
        and not result_path.with_name('failure.json').exists(), 'Mismatched or failed reference admission')
    require(result['schema'] == short_admission.SCHEMA and request['schema'] == short_admission.REQUEST_SCHEMA
        and all(result[k] is False for k in ('training_approved', 'geometry_novelty_qualified', 'source_cap_reset'))
        and request['resolution'] == [848, 408], 'Changed reference admission scope')
    for bindings in (result['source_bindings'], request['source_bindings'],
                     request['implementation_bindings'], request['workspace_implementation_bindings']):
        pins.extend(bindings)
    trial = Path(request['trial']).resolve(); pins.add(trial/'result.json', request['trial_result_sha256'])
    receipt, owned, launch, intent = (read(trial/n) for n in ('result.json', 'owned_exit.json', 'launch.json', 'intent.json'))
    require(not (trial/'failure.json').exists() and not (trial/'capture/failure.json').exists(),
        'Successful reference capture required')
    for name in ('owned_exit.json', 'launch.json', 'intent.json', 'audit.json'):
        pins.add(trial/name, result['source_bindings'][str((trial/name).resolve())])
    require(digest(trial/'owned_exit.json') == receipt['owned_exit_sha256']
        and digest(trial/'launch.json') == owned['launch_sha256'], 'Changed reference owned completion')
    pins.extend(intent['runtime_owner_bindings'])
    if 'implementation_bindings' in intent:
        pins.extend(intent['implementation_bindings'])
    plan_path = pins.add(intent['plan_path'], intent['plan_sha256']); plan = read(plan_path)
    check_owned_route(trial, receipt, owned, launch, intent, plan)
    cache, anchor, _ = api.check(plan, full=True)
    require(cache['generated_geometry_used'] is False and cache['frozen_original_assets'] is True
        and cache['source_cap_reset'] is False
        and request['implementation_bindings'] == plan['implementation_bindings'],
        'Unmodified original-reference closure required')
    pins.extend(plan['implementation_bindings']); pins.extend(cache['source_bindings'])
    pins.add(plan['cache_path'], plan['cache_sha256'])
    pins.add(cache['anchor_reference_path'], cache['anchor_reference_sha256'])
    actual = audit_api.audit_capture(trial/'capture', plan_path=plan_path,
        plan_sha256=intent['plan_sha256'], result_sha256=digest(trial/'capture/result.json'))
    require(actual['schema'] == REFERENCE_AUDIT and digest(trial/'audit.json') == receipt['audit_sha256']
        and actual == read(trial/'audit.json'), 'Independent original-reference native replay differs')
    pins.extend(actual['source_bindings'])
    qualification = short_admission.verify_short_audit_contract(plan, actual)
    pins.binding(qualification)
    adaptation_evidence = plan.get('original_adaptation_evidence')
    if adaptation_evidence is not None:
        pins.binding(adaptation_evidence)
    for packet in (request, result):
        require(packet['mode'] == plan['mode'] and packet['profile'] == short_admission.PROFILE
            and packet['render_budget_subframes'] == short_admission.BUDGET
            and packet['qualification_evidence'] == qualification
            and packet['original_adaptation_evidence'] == adaptation_evidence
            and packet['adaptation_control_sample_ids'] == actual['runtime_adaptation']['control_sample_ids']
            and packet['runtime_adaptation_audit_sha256'] == digest(trial/'audit.json'),
            'Short admission qualification/adaptation/profile differs')
    require(result['adaptation_controls_excluded_from_candidates'] is True,
        'Adaptation controls cannot become candidates')
    reviews = read(pins.binding(spec['reviews'])); require(isinstance(reviews, list), 'Explicit actual review list required')
    keys = [(r['rgb_sha256'], r['label_sha256'], r['source_sample_sha256']) for r in reviews]
    require(len(keys) == len(set(keys)) and len({r['id'] for r in reviews}) == len(reviews), 'Ambiguous reference reviews')
    by_key = dict(zip(keys, reviews)); used = set(); candidates = []; negatives = []; excluded = []
    audits = {r['sample_id']: r for r in actual['records']}; cached = {r['sample_id']: r for r in cache['records']}
    require(len(audits) == len(actual['records']) and len({r['sample_id'] for r in result['records']}) == len(result['records'])
        and set(audits) == {r['sample_id'] for r in result['records']}, 'Missing or repeated admitted reference samples')
    checker = WorkspaceChecker()
    try:
        require(checker.bindings == request['workspace_implementation_bindings'], 'Workspace implementation differs')
        pins.extend(checker.bindings)
        for rec in result['records']:
            key = (rec['rgb_sha256'], rec['label_sha256'], rec['source_sample_sha256']); review = by_key.get(key)
            if review is None:
                excluded.append(dict(target_id=rec['target_id'], reason='not_individually_reviewed', rgb_sha256=rec['rgb_sha256']))
                continue
            used.add(key); audit = audits[rec['sample_id']]; cached_row = cached[audit['source_pose_id']]
            source_row = cached_row['source_row']
            meta = read(pins.add(rec['source_sample_path'], rec['source_sample_sha256']))
            short_admission.verify_production_sample(meta, audit, plan, actual)
            require(all(rec[k] == audit[k] for k in ('capture_role', 'source_pose_id', 'profile',
                'render_budget_subframes', 'qualification_evidence_path', 'qualification_evidence_sha256')),
                'Admitted short-profile evidence differs from actual audit')
            context = original_reference_identity_context(anchor, source_row, rec, meta, audit, cache)
            # Only label/query/workspace-binding/review logic is reused; this
            # identity context is never passed to an authored capture checker.
            row = base.verify_fresh_row(rec, review, audit, context, None, 'original_control', pins)
            workspace_replay = replay_saved_workspace(read(rec['workspace_path']), meta, checker)
            require(workspace_replay['workspace_passed'] == rec['workspace_passed'], 'Workspace replay decision differs')
            row['kind'] = 'original_reference_short'; row['original_geometry_only'] = True
            row['workspace_certificate_replay'] = workspace_replay
            row['lineage'] = dict(capture_mode=plan['mode'], render_profile=short_admission.PROFILE,
                render_budget_subframes=short_admission.BUDGET,
                qualification_evidence=qualification, original_adaptation_evidence=adaptation_evidence,
                raw_native_audit=dict(path=str(trial/'audit.json'), sha256=digest(trial/'audit.json')),
                adaptation_control_frames_excluded=True, source_pose_id=audit['source_pose_id'],
                admission_result=spec['result'], admission_request=spec['request'], reviews=spec['reviews'],
                native_plan=dict(path=plan_path, sha256=intent['plan_sha256']),
                original_pose_cache=dict(path=plan['cache_path'], sha256=plan['cache_sha256']),
                original_scene_reference=dict(path=cache['anchor_reference_path'], sha256=cache['anchor_reference_sha256']),
                original_target=source_row['target_id'], source_family=anchor['source_family'],
                source_collection_plan=cached_row['source_collection_plan'],
                source_collection_plan_sha256=cached_row['source_collection_plan_sha256'],
                generated_geometry_used=False, generated_target=None, generator_version=None,
                source_cap_reset=False, reference_review_is_not_new_frame_acceptance=True)
            if review['decision'] != 'accept':
                negatives.append(row)
            elif not rec['candidate_for_individual_visual_review']:
                excluded.append(dict(id=row['id'], reason=rec['decision']))
            else:
                candidates.append(row)
        require(used == set(by_key), 'Review does not match an actual original-reference row')
    finally:
        checker.finish()
    return candidates, negatives, excluded


def run(request_path, *, request_sha256, output, operation):
    require(operation in ('inspect', 'join'), 'Explicit inspect or join required')
    output = Path(output).resolve(); require(not output.exists(), 'New immutable output required')
    pins = Pins(); request_path = pins.add(request_path, request_sha256); request = read(request_path)
    require(request['schema'] == REQUEST_SCHEMA, 'Wrong request schema')
    for path in (__file__, previous.__file__, original_context.__file__, base.__file__, short_admission.__file__, select_views.__code__.co_filename, view_signature.__code__.co_filename):
        pins.add(path)
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
    impact = {k:dict(v, cap=CAP, free_slots_before=max(0, CAP-v['prior']),
        maximum_pool_count_after=min(CAP, v['eligible_prior']+v['eligible_incoming']),
        maximum_net_pool_increment=min(CAP, v['eligible_prior']+v['eligible_incoming'])-v['prior'])
        for k,v in sorted(pools.items()) if v['eligible_incoming'] or any(r['source_target']==k for r in incoming)}
    result = dict(schema='greenhouse.native848_conservative_join_diagnostic.v1', operation=operation, consumer_module='native848_conservative_join_v4',
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
        exclusions = excluded + [dict(id=r['id'], reason='shared_original_target_view_cap') for r in eligible if r['id'] not in selected]
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
            provisional_net_train_increment=counts.get('train', 0)-prior['counts'].get('train', 0),
            selected_new_ids=sorted(selected-prior_ids), removed_prior_ids=sorted(prior_ids-selected))
    for source in pins.values:
        require(not output.is_relative_to(Path(source).parent) and not Path(source).is_relative_to(output), 'Output must be disjoint from source folders')
    pins.finish(); output.mkdir(parents=True)
    if selection is not None:
        save(output/'selection.json', selection); result['selection_sha256'] = digest(output/'selection.json')
    save(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('inspect', 'join'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.request, request_sha256=args.request_sha256, output=args.output, operation=args.operation)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_bindings', 'exclusions')}, indent=2))


if __name__ == '__main__':
    main()
