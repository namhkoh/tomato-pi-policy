"""Explicitly invoked provisional native848 join; generated novelty is withheld.

`inspect` authenticates inputs and reports eligibility/cap pressure without
running the selector or creating a selection. `join` writes a NEW selection.
Neither mode changes captures, annotations, reviews, prior selections or releases.
The existing farthest-view selector is reused with ORIGINAL source-target keys.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image

from ..clear_cutpoint_contract import select_views
from ..dataset_review import require
from ..training_export import view_signature

SCHEMA = 'greenhouse.native848_conservative_mixed_selection.v1'
REQUEST_SCHEMA = 'greenhouse.native848_conservative_join_request.v1'
PAIR_STATE = 'explicit_native848_pair_fresh_labels_trace_workspace_complete_pending_individual_review_and_global_admission'
DIRECT_STATE = 'direct_native848_labels_trace_workspace_complete_pending_review_and_global_admission'
CAP = 12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'Duplicate JSON key: ' + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class Pins:
    def __init__(self):
        self.values = {}

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        observed = self.values.get(path)
        if observed is None:
            observed = digest(path)
            self.values[path] = observed
        require(expected is None or observed == expected, 'Hash mismatch: ' + path)
        return path

    def binding(self, binding):
        return self.add(binding['path'], binding['sha256'])

    def extend(self, bindings):
        for p, h in bindings.items():
            self.add(p, h)

    def finish(self):
        for p, h in self.values.items():
            require(digest(p) == h, 'Input changed during join: ' + p)


def image_identity(path):
    with Image.open(path) as im:
        require(im.size == (848, 408) and im.mode == 'RGB', 'Native848 RGB required')
        rgb = np.asarray(im)
    require(rgb.dtype == np.uint8 and rgb.shape == (408, 848, 3), 'Wrong native pixels')
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def identities(row):
    return {('id', row['id']), ('rgb', row['rgb_sha256']),
        ('decoded_rgb', row['decoded_rgb_sha256']), ('sample', row['source_sample_sha256']),
        ('view', row['view_signature']), ('conservative_camera', row['conservative_camera_signature'])}


def check_review(review, row, *, fresh):
    require(review['decision'] in ('accept', 'hold', 'reject')
        and review['reviewer_type'] in ('assistant', 'human')
        and bool(review['reviewer'].strip()) and bool(review['reason'].strip()), 'Explicit attributed visual review required')
    require(review['rgb_sha256'] == row['rgb_sha256'], 'Review RGB differs')
    if fresh:
        require(review['label_sha256'] == row['label_sha256']
            and review['source_sample_sha256'] == row['source_sample_sha256'], 'Fresh review label/sample differs')


def check_workspace(path, expected, row, label, pins, *, require_pass=True):
    pins.add(path, expected)
    proof = read(path)
    require(proof['schema'] == 'greenhouse.clear848_kinematic_workspace.v2'
        and (not require_pass or proof['result']['workspace_passed'] is True)
        and proof['target_id'] == row['target_id']
        and proof['sample_sha256'] == row['source_sample_sha256']
        and proof['nominal_world_m'] == label['nominal_world_m'], 'Workspace target/sample/nominal differs')
    pins.add(proof['sample_path'], proof['sample_sha256'])
    pins.extend(proof['source_bindings'])
    return read(proof['sample_path'])


def normalized_prior(row, pins):
    if 'artifacts' in row:
        require(row['source_cap_reset'] is False and row['geometry_novelty_qualified'] is False, 'Prior mixed flags changed')
        result = dict(row)
        for value in row['artifacts'].values():
            pins.binding(value)
        label = read(row['artifacts']['label']['path'])
        meta = check_workspace(row['artifacts']['workspace']['path'], row['artifacts']['workspace']['sha256'], row, label, pins)
        rgb_path = row['artifacts']['rgb']['path']
    else:
        folder = Path(row['source_dataset'])
        rgb_path = pins.add(folder / row['files']['rgb'], row['rgb_sha256'])
        label_path = pins.add(folder / row['files']['label'], row['source_label_sha256'])
        label = read(label_path)
        meta = check_workspace(row['workspace_proof_path'], row['workspace_proof_sha256'], row, label, pins)
        require(row['target_id'].startswith(row['source_plant_family'] + '/'), 'Original prior required')
        artifacts = {k: dict(path=pins.add(folder/v), sha256=digest(folder/v)) for k, v in row['files'].items()}
        artifacts['workspace'] = dict(path=str(Path(row['workspace_proof_path']).resolve()), sha256=row['workspace_proof_sha256'])
        result = dict(id=row['id'], split=row['split'], source_plant_family=row['source_plant_family'],
            target_id=row['target_id'], source_target=row['target_id'], kind='reviewed_original',
            rgb_sha256=row['rgb_sha256'], label_sha256=row['source_label_sha256'],
            source_sample_sha256=row['source_sample_sha256'], review=row['review'],
            view_signature=row['view_signature'], selection_features=row['selection_features'],
            artifacts=artifacts, prior_record=row, source_cap_reset=False,
            geometry_novelty_qualified=False, novelty_decision='original_source_target', training_approved=False)
    check_review(result['review'], result, fresh=False)
    require(result['review']['decision'] == 'accept' and label['eligible'] is True
        and label['target_id'] == result['target_id'], 'Prior row is not locally accepted')
    result['decoded_rgb_sha256'] = image_identity(rgb_path)
    result['conservative_camera_signature'] = view_signature(meta, result['source_plant_family'], None)
    return result


def replay_admission(spec, pins):
    """CPU-only replay of owned native arrays; never starts a native worker."""
    from .. import native848_pair_plan_v2 as pp, native848_pair_audit_v2 as pa
    from .. import native848_direct_plan_v2 as dp, native848_direct_audit_v2 as da
    from ..native848_pair_admission_v2 import implementation_bindings
    result_path = Path(pins.binding(spec['result']))
    request_path = Path(pins.binding(spec['request']))
    require(request_path == result_path.with_name('request.json')
        and not result_path.with_name('failure.json').exists(), 'Unsuccessful or mismatched admission')
    result, request = read(result_path), read(request_path)
    require(result['state'] in (PAIR_STATE, DIRECT_STATE) and result['training_approved'] is False
        and result['geometry_novelty_qualified'] is False and result['source_cap_reset'] is False
        and request['resolution'] == [848, 408], 'Unsupported admission scope')
    for value in (result['source_bindings'], request['source_bindings'], request['implementation_bindings'], request['workspace_implementation_bindings']):
        pins.extend(value)
    trial = Path(request['trial'])
    receipt_path = pins.add(trial/'result.json', request['trial_result_sha256'])
    receipt, owned, launch, intent = (read(trial/n) for n in ('result.json', 'owned_exit.json', 'launch.json', 'intent.json'))
    for n in ('owned_exit.json', 'launch.json', 'intent.json', 'audit.json'):
        pins.add(trial/n, result['source_bindings'][str((trial/n).resolve())])
    require(owned['returncode'] == 0 and owned['method'] == 'subprocess_wait_on_owned_process'
        and digest(trial/'owned_exit.json') == receipt['owned_exit_sha256']
        and digest(trial/'launch.json') == owned['launch_sha256'] and launch['pid'] == owned['pid']
        and launch['plan_sha256'] == intent['plan_sha256'] == receipt['plan_sha256']
        and launch['command'][-1] == str(trial/'capture'), 'Owned completion differs')
    pins.extend(intent['runtime_owner_bindings'])
    if 'implementation_bindings' in intent:
        pins.extend(intent['implementation_bindings'])
    plan_path = pins.add(intent['plan_path'], intent['plan_sha256'])
    plan = read(plan_path)
    pair = result['state'] == PAIR_STATE
    if pair:
        require(receipt['state'] == 'owned_native848_pair_completed_and_buffers_replayed_pending_admission'
            and receipt['native_frames'] == 2 and 'sim_data.native848_pair_worker_v2' in launch['command'], 'Wrong pair producer')
        require(request['implementation_bindings'] == implementation_bindings(), 'Pair annotation closure differs')
        generated = pp.check(plan, full=True)
        plans = {plan_path: plan}
        actual = pa.audit_capture(trial/'capture', plan_path=plan_path, plan_sha256=intent['plan_sha256'], result_sha256=digest(trial/'capture/result.json'))
    else:
        require(receipt['state'] == 'owned_direct_native848_completed_and_audited'
            and 'sim_data.native848_direct_worker_v2' in launch['command'], 'Wrong direct producer')
        _, plans, generated = dp.check(plan, full=True)
        require(request['implementation_bindings'] == plan['implementation_bindings'], 'Direct annotation closure differs')
        actual = da.audit_capture(trial/'capture', plan_path=plan_path, plan_sha256=intent['plan_sha256'], result_sha256=digest(trial/'capture/result.json'))
    require(digest(trial/'audit.json') == receipt['audit_sha256'] and actual == read(trial/'audit.json'), 'Independent native buffer replay differs')
    pins.extend(actual['source_bindings'])
    reviews = read(pins.binding(spec['reviews']))
    require(isinstance(reviews, list), 'Separate explicit individual review list required')
    keys = [(r['rgb_sha256'], r['label_sha256'], r['source_sample_sha256']) for r in reviews]
    require(len(keys) == len(set(keys)) and len({r['id'] for r in reviews}) == len(reviews), 'Ambiguous review identities')
    by_key = dict(zip(keys, reviews)); used = set(); candidates = []; negatives = []; excluded = []
    audits = {r['mode'] if pair else r['sample_id']: r for r in actual['records']}
    for rec in result['records']:
        key = (rec['rgb_sha256'], rec['label_sha256'], rec['source_sample_sha256'])
        review = by_key.get(key)
        if review is None:
            excluded.append(dict(target_id=rec['target_id'], reason='not_individually_reviewed', rgb_sha256=rec['rgb_sha256']))
            continue
        used.add(key)
        audit = audits[rec['mode'] if pair else rec['sample_id']]
        pair_path = plan_path if pair else audit['pair_plan_path']
        pair_plan = plans[pair_path]
        mode = rec['mode'] if pair else 'generated_variant'
        row = verify_fresh_row(rec, review, audit, pair_plan, generated, mode, pins)
        row['lineage'] = dict(admission_result=spec['result'], admission_request=spec['request'], reviews=spec['reviews'],
            native_plan=dict(path=plan_path, sha256=intent['plan_sha256']),
            pair_plan=dict(path=pair_path, sha256=digest(pair_path)), variant_directory=pair_plan['variant_directory'],
            generated_target=pair_plan['generated_row']['target_id'] if mode == 'generated_variant' else None,
            original_target=pair_plan['source_row']['target_id'], source_family=pair_plan['source_family'],
            generator_version=pair_plan['generator_version'] if mode == 'generated_variant' else None,
            generator_code_bindings=pair_plan['generator_code_bindings'] if mode == 'generated_variant' else {})
        if review['decision'] != 'accept':
            negatives.append(row)
        elif not rec['candidate_for_individual_visual_review']:
            excluded.append(dict(id=row['id'], reason=rec['decision']))
        else:
            candidates.append(row)
    require(used == set(by_key), 'Review does not match an admitted native row')
    return candidates, negatives, excluded


def verify_fresh_row(rec, review, audit, plan, generated, mode, pins):
    from .. import native848_clear_labels_v1 as labels, native848_query_selection_v1 as queries
    from ..collection_plan import load_plan
    require(mode in ('original_control', 'generated_variant') and rec['split'] == plan['split'] == 'train'
        and rec['source_family'] == plan['source_family'] and rec['source_target'] == plan['conservative_view_cap_group'] == plan['source_row']['target_id']
        and rec['source_cap_reset'] is False and rec['geometry_novelty_qualified'] is False
        and rec['training_approved'] is False, 'Native lineage/cap/split differs')
    expected_target = plan['source_row' if mode == 'original_control' else 'generated_row']['target_id']
    require(rec['target_id'] == expected_target and rec['source_sample_sha256'] == audit['sample_sha256']
        and rec['rgb_sha256'] == audit['rgb_sha256'], 'Native target/buffer identity differs')
    sp = Path(pins.add(rec['source_sample_path'], rec['source_sample_sha256'])); meta = read(sp)
    folder = sp.parent; lp = Path(pins.add(rec['label_path'], rec['label_sha256']))
    pins.add(rec['rgb_path'], rec['rgb_sha256'])
    require(Path(rec['rgb_path']).resolve() == (folder/'inputs/rgb.png').resolve(), 'RGB path not owned sample')
    artifacts = {}
    for name, rel in [('rgb', 'inputs/rgb.png'), ('depth', 'inputs/depth_m.npy'), ('validity', 'inputs/depth_valid.png'), ('target_mask', 'supervision/target_visible.png')]:
        path = pins.add(folder/rel, meta['files'][rel]['sha256'])
        artifacts[name] = dict(path=path, sha256=meta['files'][rel]['sha256'])
    rgb = np.asarray(Image.open(rec['rgb_path']))
    depth = np.load(folder/'inputs/depth_m.npy', allow_pickle=False)
    valid = np.asarray(Image.open(folder/'inputs/depth_valid.png')) == 255
    components = np.load(folder/'supervision/component_id.npy', allow_pickle=False)
    catalogue = read(folder/'supervision/identities.json')['component_catalogue']
    mask = np.asarray(Image.open(folder/'supervision/target_visible.png'))
    if mode == 'original_control':
        _, reports = load_plan(plan['source_collection_plan'])
        report = next(r for r in reports if r['plant_id'] == plan['source_family'])
    else:
        report = generated['report']
    baseline = labels.derive(meta, report, rgb, depth, valid, components, catalogue)
    label, trace, query = queries.annotate_v1(meta, report, rgb, depth, valid, components, catalogue, target_mask=mask, expected_label=baseline)
    require(label == read(lp), 'Fresh label replay differs')
    for name, value, expected in [('query_selection', query, rec['query_selection_sha256']), ('query_trace', trace, rec['query_trace_sha256'])]:
        if value is not None:
            path = pins.add(lp.with_name(name+'.json'), expected)
            require(read(path) == value, 'Fresh query replay differs')
            artifacts[name] = dict(path=path, sha256=expected)
        else:
            require(expected is None, 'Missing query trace was concealed')
    check_workspace(rec['workspace_path'], rec['workspace_sha256'], rec, label, pins, require_pass=False)
    workspace_passed = read(rec['workspace_path'])['result']['workspace_passed'] is True
    require(label['target_id'] == expected_target and label['source_plant_family'] == plan['source_family']
        and label['conservative_view_cap_group'] == rec['source_target'] and label['resolution'] == [848, 408], 'Fresh label ancestry differs')
    passed = label['eligible'] is True and bool(trace and trace['passed'] is True) and workspace_passed
    if 'render_profile_qualified' in rec:
        require(rec['render_profile_qualified'] == audit['render_profile_qualified'], 'Render qualification differs')
        passed = passed and audit['render_profile_qualified'] is True
    require(rec['local_clarity_passed'] == label['eligible'] and rec['full_trace_and_anchored_grid_passed'] == bool(trace and trace['passed'])
        and rec['workspace_passed'] == workspace_passed and rec['candidate_for_individual_visual_review'] == passed, 'Admission eligibility differs')
    camera = np.asarray(meta['calibration']['camera_to_world_usd_row_vectors'], float)
    features = ([*camera[3,:3], *camera[2,:3], *(np.asarray(label['query_pixel_uv'])/[848, 408])]
        if label.get('query_pixel_uv') is not None else None)
    for name, path, value in [('label', str(lp), rec['label_sha256']), ('sample', str(sp), rec['source_sample_sha256']), ('workspace', rec['workspace_path'], rec['workspace_sha256'])]:
        artifacts[name] = dict(path=str(Path(path).resolve()), sha256=value)
    row = dict(id=review['id'], split='train', source_plant_family=rec['source_family'], target_id=expected_target,
        source_target=rec['source_target'], kind=mode, rgb_sha256=rec['rgb_sha256'], label_sha256=rec['label_sha256'],
        source_sample_sha256=rec['source_sample_sha256'], review=review, artifacts=artifacts,
        decoded_rgb_sha256=image_identity(rec['rgb_path']), view_signature=view_signature(meta, rec['source_family'], None),
        conservative_camera_signature=view_signature(meta, rec['source_family'], None), selection_features=features,
        source_cap_reset=False, geometry_novelty_qualified=False, training_approved=False,
        novelty_decision='novelty_withheld' if mode == 'generated_variant' else 'original_source_target')
    check_review(review, row, fresh=True)
    return row


def held_aliases(prior, supplemental, pins):
    """Resolve all old holds; absence of an alias binding fails closed."""
    wanted = {r['id']: r for r in prior['preserved_visual_holds']}
    found = {}; aliases = set()
    for row in prior.get('preserved_hold_identity_rows', []):
        require(row['review']['decision'] in ('hold', 'reject'), 'Invalid saved hold')
        for value in row['artifacts'].values(): pins.binding(value)
        found[row['id']] = row
    manifests = [(Path(p), read(p)) for p in (pins.binding(s) for s in supplemental)]
    # Later original joins directly pin held labels; early holds need an explicit
    # source manifest. The new request pins that manifest instead of trusting a
    # directory search or quietly weakening identity propagation.
    for sid, review in wanted.items():
        if sid in found: continue
        choices = []
        for p, h in list(pins.values.items()):
            if Path(p).name == sid+'.json' and Path(p).parent.name == 'labels':
                rp = Path(p).parent.parent/'images'/ (sid+'.png')
                if rp.is_file(): choices.append((Path(p), h, rp, review['rgb_sha256']))
        for mp, manifest in manifests:
            files = manifest['files_sha256']; rel = 'labels/'+sid+'.json'; rgb = 'images/'+sid+'.png'
            if rel in files and rgb in files:
                choices.append((mp.parent/rel, files[rel], mp.parent/rgb, files[rgb]))
        require(choices, 'Cannot resolve preserved hold identity: '+sid)
        lp, lh, rp, rh = choices[0]; pins.add(lp, lh); pins.add(rp, rh)
        require(rh == review['rgb_sha256'], 'Held image changed')
        label = read(lp); family = label['source_plant_family']; meta = {'calibration': label['calibration']}
        found[sid] = dict(id=sid, rgb_sha256=rh, source_sample_sha256=label['source_sample_sha256'],
            decoded_rgb_sha256=image_identity(rp), view_signature=view_signature(meta, family, None),
            conservative_camera_signature=view_signature(meta, family, None), review=review,
            artifacts={'rgb':dict(path=str(rp.resolve()),sha256=rh), 'label':dict(path=str(lp.resolve()),sha256=lh)})
    for row in found.values(): aliases.update(identities(row))
    require(set(wanted) <= set(found), 'Unresolved held identity')
    return list(found.values()), aliases


def filter_identities(rows, hold_keys, frozen):
    # Alias components propagate holds transitively, including excluded rows.
    seen = {}; parents = list(range(len(rows))); groups = defaultdict(list)
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for i, row in enumerate(rows):
        family, split = row['source_plant_family'], row['split']
        require(split in ('train','validation','test') and frozen.setdefault(family, split) == split, 'Family split changed')
        require(row['source_target'].startswith(family+'/'), 'Variant ID cannot be a source cap group')
        for key in identities(row):
            if key in seen:
                j = seen[key]
                if key[0] == 'id':
                    require(row['rgb_sha256'] == rows[j]['rgb_sha256'] and row['label_sha256'] == rows[j]['label_sha256'], 'ID reused for different content')
                parents[root(i)] = root(j)
            else:
                seen[key] = i
    for i in range(len(rows)): groups[root(i)].append(i)
    eligible = []; excluded = []
    for indices in groups.values():
        group = [rows[i] for i in indices]
        require(len({r['split'] for r in group}) == 1, 'Exact identity appears across splits')
        held = set().union(*(identities(r) for r in group)) & hold_keys
        if held:
            excluded.extend(dict(id=r['id'], reason='preserved_visual_hold_identity', matching_kinds=sorted({k for k,v in held})) for r in group)
        else:
            eligible.append(group[0])
            excluded.extend(dict(id=r['id'], reason='exact_identity_or_shared_source_camera', matching_ids=[group[0]['id']]) for r in group[1:])
    return eligible, excluded


def shared_cap_selection(rows):
    """Preserve actual target IDs; change only the selector's temporary group key."""
    by_id = {r['id']:r for r in rows}
    require(len(by_id) == len(rows), 'Nonunique selection IDs')
    adapters = [dict(r, target_id=r['source_target']) for r in rows]
    selected = [by_id[r['id']] for r in select_views(adapters)]
    require(max(Counter((r['source_plant_family'],r['source_target']) for r in selected).values(), default=0) <= CAP, 'Source cap exceeded')
    return selected


def run(request_path, *, request_sha256, output, operation):
    require(operation in ('inspect','join'), 'Explicit inspect or join required')
    output = Path(output).resolve(); require(not output.exists(), 'New immutable output required')
    pins = Pins(); request_path = pins.add(request_path, request_sha256); request = read(request_path)
    require(request['schema'] == REQUEST_SCHEMA, 'Wrong request schema')
    pins.add(__file__); pins.add(Path(select_views.__code__.co_filename)); pins.add(Path(view_signature.__code__.co_filename))
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
    result = dict(schema='greenhouse.native848_conservative_join_diagnostic.v1', operation=operation,
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
