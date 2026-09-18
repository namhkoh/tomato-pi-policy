"""Create-only diverse supplement, capped training registry and sampling weights.

Old package bytes and annotation epochs are preserved.  New records require the
separate q3 post-admission wrapper and individual visual review; this module never
captures, annotates, changes a hold ledger, or starts training.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil

import numpy as np
from PIL import Image
from .training_export import view_signature
from .clear_cutpoint_contract import crop_box, crop_image

SCHEMA = 'greenhouse.native848_diverse_supplement.v1'
REQUEST_SCHEMA = 'greenhouse.native848_diverse_supplement_request.v1'
REVIEW_SCHEMA = 'greenhouse.native848_diverse_individual_review.v1'
EPOCH = 'greenhouse.native848_query_selection.v3'
CAP = 12
WORKSPACE = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = WORKSPACE/'data/sim_data/dataset_checkpoints'
ALIASES = ('rgb_sha256', 'decoded_rgb_sha256', 'conservative_camera_signature')
HELPER_MODULES = [Path(__file__).resolve().with_name(name) for name in ('training_export.py', 'clear_cutpoint_contract.py')]


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


def balanced(rows, holds):
    groups = defaultdict(list)
    excluded = []
    for row in rows:
        if row['split'] != 'train':
            continue
        if aliases(row) & holds:
            excluded.append(dict(id=row['id'], reason='preserved_visual_hold_identity'))
        else:
            groups[family_target(row)].append(row)
    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: (
            row.get('registry_origin') != 'new_supplement',
            not individually_reviewed(row),
            hashlib.sha256(row['id'].encode('utf-8')).hexdigest(), row['id']))
        selected.extend(ranked[:CAP])
        excluded.extend(dict(id=row['id'], reason='physical_source_target_cap12') for row in ranked[CAP:])
    require(max(Counter(family_target(row) for row in selected).values(), default=0) <= CAP, 'Target cap failed')
    return selected, excluded


def weights(rows, holds):
    eligible = [row for row in rows if row['split'] == 'train' and not aliases(row) & holds]
    groups = defaultdict(list)
    for row in eligible:
        groups[family_target(row)].append(row)
    targets = Counter(family for family, target in groups)
    out = []
    for (family, target), group in sorted(groups.items()):
        weight = 1. / (len(targets) * targets[family] * len(group))
        out.extend(dict(id=row['id'], source_plant_family=family, source_target=target,
                        weight=weight) for row in sorted(group, key=lambda row: row['id']))
    total = math.fsum(row['weight'] for row in out)
    require(not out or math.isclose(total, 1., abs_tol=1e-12, rel_tol=0), 'Weights do not sum to one')
    return out, dict(eligible_train_images=len(out), families=len(targets), physical_targets=len(groups),
        normalized_sum=total, policy='equal_family_then_equal_physical_source_target_then_equal_image',
        effective_family_weights={family: 1./len(targets) for family in sorted(targets)})


def distribution(rows):
    train = [row for row in rows if row['split'] == 'train']
    return dict(counts=dict(Counter(row['split'] for row in rows)),
        train_families=len({family_target(row)[0] for row in train}),
        train_physical_targets=len({family_target(row) for row in train}),
        train_by_family=dict(sorted(Counter(row['source_plant_family'] for row in train).items())),
        train_by_source_target=dict(sorted(Counter(row['source_target'] for row in train).items())))


def load_old(request, pins):
    spec = request['old_package']
    require(set(spec) == {'path', 'index_sha256', 'status_sha256', 'visual_holds_sha256', 'baseline_sha256'}, 'Exact old package snapshot required')
    root = Path(spec['path']).resolve()
    for name, key in [('index.jsonl', 'index_sha256'), ('status.json', 'status_sha256'),
                      ('visual_holds.jsonl', 'visual_holds_sha256'), ('baseline.json', 'baseline_sha256')]:
        pins.add(dict(path=str(root/name), sha256=spec[key]))
    old = rows_at(root/'index.jsonl')
    status = read(root/'status.json')
    require(status['schema'] == 'greenhouse.original848_bulk_checkpoint.v1'
        and status['native_resolution'] == [848,408] and status['training_approved'] is False, 'Expected frozen native848 old package')
    counts = Counter(row['split'] for row in old)
    require(dict(counts) == status['counts'], 'Old status/index counts differ')
    require(len({row['id'] for row in old}) == len(old), 'Repeated old image ID')
    baseline = read(root/'baseline.json')
    selection_path = pins.add(request['frozen_selection'])
    require(selection_path == (Path(baseline['prior_checkpoint'])/'evidence/selection.json').resolve(),
            'Frozen split/hold selection must be the old package baseline source')
    pins.add(dict(path=str(Path(baseline['prior_checkpoint'])/'index.jsonl'), sha256=baseline['prior_index_sha256']))
    selection = read(selection_path)
    frozen = selection['frozen_family_splits']
    hold_rows = rows_at(root/'visual_holds.jsonl') + selection['preserved_hold_identity_rows']
    holds = set().union(*(aliases(row) for row in hold_rows)) if hold_rows else set()
    seen = set()
    for row in old:
        family, target = family_target(row)
        require(frozen.get(family) == row['split'], 'Old family split differs from frozen selection')
        require(all(row.get(key) for key in ALIASES), 'Old row lacks saved duplicate identity')
        require(not seen & aliases(row), 'Old package contains repeated RGB/camera identity')
        seen.update(aliases(row))
        require(row.get('source_cap_reset') is not True and row.get('geometry_novelty_qualified') is not True, 'Unqualified source cap reset')
        for relative in row['files'].values():
            path = Path(relative)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe old relative asset path')
        row.update(asset_root=str(root), registry_origin='retained_old_corpus',
                   retained_without_reannotation=True)
    return root, old, frozen, hold_rows, holds, seen


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


def check_candidate(row, pins, frozen):
    require(row['candidate_for_individual_visual_review'] is True
        and row['render_profile_qualified'] is True and row['workspace_passed'] is True
        and row['prior_candidate_for_individual_visual_review'] is True
        and row['annotation_epoch'] == EPOCH and row['split'] == 'train'
        and row['source_cap_reset'] is False and row['geometry_novelty_qualified'] is False,
        'Only previously eligible, source-preserving q3 TRAIN candidates allowed')
    family = row['source_family']
    require(frozen.get(family) == 'train' and row['source_target'].startswith(family + '/')
        and row['target_id'] == row['source_target'], 'Changed original source target/family/split')
    route = row.get('input_route', 'original_q3')
    require(route in ('original_q3', 'bulk_q3'), 'Unknown native admission route')
    required = {'rgb', 'target_mask', 'sample', 'workspace', 'label', 'query_trace', 'query_selection'}
    required |= ({'depth', 'validity', 'renderer_instance_id', 'component_id', 'identities'}
                 if route == 'original_q3' else {'buffers', 'observation', 'context', 'mapping', 'workspace_replay'})
    require(required <= set(row['artifacts']), 'Missing actual native or q3 artifacts')
    paths = {key: pins.add(value) for key, value in row['artifacts'].items()}
    require(row['rgb_sha256'] == row['artifacts']['rgb']['sha256']
        and row['source_sample_sha256'] == row['artifacts']['sample']['sha256'], 'Row native source hashes differ')
    label, trace, selection, workspace, meta = [read(paths[key]) for key in ('label','query_trace','query_selection','workspace','sample')]
    require(label['task'] == 'greenhouse.native848_curved_clear_cutpoint_rgb.v1'
        and label['eligible'] is True and label['target_id'] == row['target_id']
        and label['annotation_epoch'] == EPOCH and label['answer']['status'] == 'localized'
        and label['answer']['visibility'] == 'clear', 'Fresh clear q3 cut label required')
    chosen = selection['selected_query']
    require(selection['schema'] == EPOCH and selection['annotation_epoch'] == EPOCH and chosen is not None
        and chosen['both_passed'] is True and chosen['all_checks_passed'] is True
        and chosen['route_separation']['passed'] is True
        and trace['schema'] == 'greenhouse.native848_query_trace.v3' and trace['passed'] is True
        and trace['route_separation']['passed'] is True
        and trace['legacy_trace']['passed'] is True and trace['fixed_grid']['passed'] is True
        and trace['annotation_epoch'] == EPOCH and trace['original_query_retained_as_failed_fallback'] is False
        and label['query_pixel_uv'] == trace['query_pixel_uv'] == chosen['query_pixel_uv'], 'Full q3 route/grid/trace gates required')
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


def inspect(request_path, request_sha256):
    pins = Pins()
    request_path = pins.add(dict(path=str(Path(request_path).resolve()), sha256=request_sha256))
    request = read(request_path)
    require(request['schema'] == REQUEST_SCHEMA and request['per_target_training_cap'] == CAP
        and request['preserve_old_corpus'] is True and request['training_approved'] is False,
        'Explicit separate supplement and cap12 contract required')
    require(request['implementation_bindings'].get(str(Path(__file__).resolve())) == digest(__file__),
            'Exact reviewed exporter implementation binding required')
    require(all(request['implementation_bindings'].get(str(path)) == digest(path) for path in HELPER_MODULES),
            'Exact camera fingerprint and query-only crop helper bindings required')
    for path, expected in request['implementation_bindings'].items():
        pins.add(dict(path=path, sha256=expected))
    old_root, old, frozen, hold_rows, holds, seen = load_old(request, pins)
    candidates, excluded, new_holds = [], [], []
    for spec in request['admissions']:
        result_path = pins.add(spec['result']); result = read(result_path)
        require(result['annotation_epoch'] == EPOCH and result['training_approved'] is False,
                'Explicit completed q3-only adapter required')
        if result['schema'] == 'greenhouse.native848_diverse_q3_post_admission.v1':
            route = 'original_q3'
            require(result['state'] == 'original_native_q3_post_admission_complete_pending_individual_review'
                and result['old_exclusions_or_holds_revived'] is False
                and result['old_geometry_workspace_and_audit_gates_preserved'] is True,
                'Dedicated original q3 post-admission result required')
            expected_impl = 'native848_diverse_q3_post_admission_v1.py'
            evidence_keys = ('original_admission', 'original_audit', 'owner_complete')
            prior_key = 'sample_id'
        else:
            require(result['schema'] == 'greenhouse.native848_diverse_bulk_q3_adapter.v1'
                and result['state'] == 'original_bulk_q3_normalized_pending_individual_review'
                and result['old_holds_revived'] is False and result['labels_recomputed'] is False,
                'Unsupported canonical adapter')
            route = 'bulk_q3'
            expected_impl = 'native848_diverse_bulk_adapter_v1.py'
            evidence_keys = ('original_admission', 'native_result', 'owner_complete', 'cpu_owner_result',
                             'context', 'manifest', 'native_owned_exit')
            prior_key = 'observation_id'
        wrapper_path = pins.add(result['implementation'])
        require(wrapper_path.name == expected_impl
            and request['implementation_bindings'].get(str(wrapper_path)) == result['implementation']['sha256'],
            'Exact reviewed q3 adapter implementation binding required')
        original = read(pins.add(result['original_admission']))
        if route == 'original_q3':
            require(original['state'] in (
                'original848_short_labels_trace_workspace_complete_pending_review_and_global_admission',
                'original848_reference_direct_labels_trace_workspace_complete_pending_review_and_global_admission'),
                'Unsupported original admission route')
        else:
            require(original['schema'] == 'greenhouse.original848_bulk_admission.v4'
                and original['annotation_epoch'] == EPOCH
                and original['state'] == 'automated_frame_checks_complete_pending_batch_QA_and_owned_capture_completion',
                'Actual q3 bulk admission required')
        prior_records = {row[prior_key]: row for row in original['records']}
        require(len(prior_records) == len(original['records']), 'Repeated original admission sample ID')
        for key in evidence_keys:
            path = pins.add(result[key])
            require(result['source_bindings'].get(str(path)) == result[key]['sha256'], 'Unbound original route evidence')
        for path, expected in result['source_bindings'].items():
            pins.add(dict(path=path, sha256=expected))
        names = [row['sample_id'] for row in result['records']]
        if names:
            review_path = pins.add(spec['review']); review = read(review_path)
            require(review['schema'] == REVIEW_SCHEMA and review['wrapper_result_sha256'] == spec['result']['sha256'],
                    'Review must bind exact q3 wrapper result')
        else:
            require(spec['review'] is None, 'No invented visual review for an empty eligible population')
            review = dict(reviews=[])
        reviews = {row['sample_id']: row for row in review['reviews']}
        require(len(names) == len(set(names)) and len(reviews) == len(review['reviews'])
            and set(names) == set(reviews), 'Every eligible new frame requires exactly one actual individual review')
        all_names = names + [row['sample_id'] for row in result['excluded_records']]
        geometry_names = set()
        if route == 'bulk_q3':
            manifest = read(result['manifest']['path'])
            decisions = {row['observation_id']: row for row in manifest['decisions']}
            require(len(decisions) == len(manifest['decisions']) == result['proposed_population'],
                    'Changed complete native schedule')
            for excluded_row in result['excluded_records']:
                if excluded_row['decision'] == 'preserved_native_geometry_hold':
                    name = excluded_row['sample_id']; native = decisions[name]
                    require(name not in prior_records and native['state'] == 'rejected_native_geometry'
                        and native['native_requests'] == 0, 'Geometry-only exclusion has an admitted image')
                    geometry_names.add(name)
            require(set(all_names) == set(decisions), 'Lost native pilot population member')
        require(len(all_names) == len(set(all_names)) and set(all_names)-geometry_names == set(prior_records),
                'Post-admission lost or duplicated an actual prior record')
        excluded.extend(dict(source_result_sha256=spec['result']['sha256'], record=row, reason='post_admission_excluded')
                        for row in result['excluded_records'])
        for row in result['records']:
            prior = prior_records[row['sample_id']]
            prior_pass = (prior['candidate_for_individual_visual_review'] if route == 'original_q3' else prior['automated_pass'])
            sample_hash = prior['source_sample_sha256' if route == 'original_q3' else 'sample_sha256']
            require(prior_pass is True and row.get('input_route', 'original_q3') == route
                and sample_hash == row['source_sample_sha256']
                and all(prior[key] == row[key] for key in ('rgb_sha256',
                    'source_family', 'source_target', 'target_id', 'split', 'workspace_sha256')),
                'Q3 candidate differs from original actual candidate/source/workspace')
            require(row['artifacts']['workspace']['sha256'] == prior['workspace_sha256'], 'Workspace proof changed')
            label, paths = check_candidate(row, pins, frozen)
            actual = reviews[row['sample_id']]
            validate_review(actual, row)
            sid = 'diverse_' + hashlib.sha256(json.dumps([row['source_family'], row['source_target'],
                row['sample_id'], row['rgb_sha256']], separators=(',', ':')).encode()).hexdigest()[:24]
            normalized = dict(row, id=sid,
                source_plant_family=row['source_family'], label_sha256=row['artifacts']['label']['sha256'],
                registry_origin='new_supplement', individual_visual_review=actual['decision'] == 'accept',
                review_state='automated_q3_pass_and_individual_visual_' + actual['decision'],
                review=actual, answer=label['answer'], annotation_schema=label['task'], query_pixel_uv=label['query_pixel_uv'],
                accepted_interval_uv=label['accepted_interval_uv'],
                lineage=dict(wrapper_result=spec['result'], individual_review=spec['review']),
                kind='original_native848_diverse_supplement', training_approved=False)
            if actual['decision'] != 'accept':
                new_holds.append(normalized); holds.update(aliases(normalized))
                excluded.append(dict(id=normalized['id'], reason='actual_individual_visual_hold_or_reject'))
            else:
                candidates.append((normalized, paths))
    accepted = []
    ids = {row['id'] for row in old}
    # Collect every new HOLD first: result ordering cannot revive another alias.
    for row, paths in sorted(candidates, key=lambda pair: pair[0]['id']):
        reason = ('preserved_visual_hold_identity' if aliases(row) & holds else
                  'duplicate_old_or_new_RGB_or_camera' if aliases(row) & seen or row['id'] in ids else None)
        if reason:
            excluded.append(dict(id=row['id'], reason=reason)); continue
        seen.update(aliases(row)); ids.add(row['id']); accepted.append((row, paths))
    all_rows = old + [row for row, paths in accepted]
    chosen, cap_excluded = balanced(all_rows, holds)
    weighted, weight_summary = weights(all_rows, holds)
    pins.verify()
    return dict(request=request, pins=pins, old_root=old_root, old=old, accepted=accepted,
        holds=holds, hold_rows=hold_rows + new_holds, excluded=excluded, chosen=chosen,
        cap_excluded=cap_excluded, weighted=weighted, weight_summary=weight_summary)


def build(request_path, *, request_sha256, output):
    output = Path(output).resolve()
    require(output.parent == OUTPUT_ROOT and not output.exists(), 'New separate checkpoint output required')
    checked = inspect(request_path, request_sha256)
    require(checked['accepted'], 'At least one actual fresh individually accepted supplement image required')
    require(output != checked['old_root'] and not output.is_relative_to(checked['old_root']), 'Old corpus cannot be changed')
    output.mkdir(parents=True)
    try:
        evidence = {}
        for spec in checked['request']['admissions']:
            wrapper = read(spec['result']['path'])
            primary = [wrapper[key] for key in ('original_admission', 'original_audit', 'owner_complete',
                'native_result', 'cpu_owner_result', 'context', 'manifest', 'native_owned_exit') if key in wrapper]
            for item in [spec['result']] + ([spec['review']] if spec['review'] is not None else []) + primary:
                relative = 'evidence/' + item['sha256'] + '.json'
                destination = output/relative
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item['path'], destination)
                    require(digest(destination) == item['sha256'], 'Copied provenance differs')
                evidence[relative] = dict(source=item['path'], sha256=item['sha256'])
        write(output/'portable_evidence.json', evidence)
        supplements = []
        for row, paths in checked['accepted']:
            sid = hashlib.sha256(row['id'].encode()).hexdigest()[:24]
            files = {}
            for key, source in paths.items():
                require(key.replace('_', '').isalnum(), 'Unsafe artifact key')
                relative = Path('supplement')/sid/(key + source.suffix)
                target = output/relative; target.parent.mkdir(parents=True, exist_ok=True)
                require(not target.exists(), 'Artifact destination already exists')
                shutil.copy2(source, target)
                require(digest(target) == row['artifacts'][key]['sha256'], 'Copied artifact differs')
                files[key] = relative.as_posix()
            relative = Path('supplement')/sid/'crop.png'
            with Image.open(paths['rgb']) as image:
                crop = crop_image(image, row['query_pixel_uv'])
                crop.save(output/relative, compress_level=1)
            files['crop'] = relative.as_posix()
            row.update(asset_root=str(output), files=files, native_calibration_file=files['sample'],
                crop_box_xyxy=crop_box(row['query_pixel_uv']), crop_source_dimensions=[384,384],
                crop_output_dimensions=[768,768], crop_depends_only_on_query=True, crop_full_interval_contained=True,
                crop_sha256=digest(output/relative))
            supplements.append(row)
        all_rows = checked['old'] + supplements
        chosen, cap_excluded = balanced(all_rows, checked['holds'])
        weighted, weight_summary = weights(all_rows, checked['holds'])
        jsonl(output/'full_index.jsonl', all_rows)
        jsonl(output/'supplement_index.jsonl', supplements)
        jsonl(output/'balanced_train_index.jsonl', chosen)
        jsonl(output/'sampling_weights.jsonl', weighted)
        jsonl(output/'validation_index.jsonl', [row for row in all_rows if row['split'] == 'validation'])
        jsonl(output/'test_index.jsonl', [row for row in all_rows if row['split'] == 'test'])
        write(output/'preserved_holds.json', dict(records=checked['hold_rows'], mutable_ledger=False))
        write(output/'exclusions.json', dict(admission=checked['excluded'], balanced=cap_excluded))
        write(output/'request.json', checked['request'])
        checked['pins'].verify()
        index_pins = {name: dict(path=str(output/name), sha256=digest(output/name)) for name in
            ('full_index.jsonl','supplement_index.jsonl','balanced_train_index.jsonl','sampling_weights.jsonl',
             'validation_index.jsonl','test_index.jsonl','preserved_holds.json','exclusions.json','request.json','portable_evidence.json')}
        old_targets = {family_target(row) for row in checked['old'] if row['split'] == 'train'}
        new_targets = {family_target(row) for row in supplements}
        result = dict(schema=SCHEMA, state='separate_supplement_and_balanced_indices_complete',
            created_utc=datetime.now(timezone.utc).isoformat(), old_package_untouched=True,
            old_assets_rehashed=False, old_annotations_recomputed=False,
            portable_supplement_assets=True, portable_primary_provenance=True,
            transitive_replay_dependencies_remain_external=True,
            old_package=checked['request']['old_package'], accepted_new_images=len(supplements),
            added_physical_targets=len(new_targets-old_targets),
            added_train_families=len({f for f,t in new_targets}-{f for f,t in old_targets}),
            old_corpus=distribution(checked['old']), full_registry=distribution(all_rows),
            supplement=distribution(supplements), balanced_training=distribution(chosen),
            balanced_per_physical_target_cap=CAP, sampling_weights=weight_summary,
            source_bindings=checked['pins'].values, output_bindings=index_pins,
            implementation=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
            new_images_individually_reviewed=True, old_review_scope_preserved=True,
            annotation_epochs_preserved=True, training_approved=False, training_started=False,
            quality_scope='New supplement individual QA; old corpus retains its original mixed individual/sampled evidence. Cap and weights change sampling, not biological diversity.')
        usage = (
            '# Diverse supplement\n\nFull registry retains every old record without relabeling. '
            'Its asset_root points to the immutable old package; supplement files are copied here. '
            'Copy both packages for portability and remap asset_root, or materialize referenced files.\n\n'
            'balanced_train_index.jsonl caps each original physical source target at 12, preferring new individually reviewed frames, then old individually reviewed frames, with a stable ID hash tie-break. '
            'Validation/test keep their frozen family splits. Full-corpus weights give equal family, then target, then image mass and omit held aliases; use their exact IDs with full_index TRAIN rows only. '
            'Weights and caps do not create new biological diversity or upgrade historical QA. Primary new review/admission/audit receipts are copied under evidence; transitive source/scene dependencies remain pinned external references for replay.\n\n'
            '```python\nimport json\nfrom pathlib import Path\n'
            "rows = [json.loads(x) for x in Path('balanced_train_index.jsonl').read_text().splitlines()]\n"
            "rgb_path = Path(rows[0]['asset_root']) / rows[0]['files']['rgb']\n```\n"
            )
        with (output/'USAGE.md').open('x', encoding='utf-8') as stream:
            stream.write(usage)
        result['output_bindings']['USAGE.md'] = dict(path=str(output/'USAGE.md'), sha256=digest(output/'USAGE.md'))
        write(output/'result.json', result)
        return result
    except BaseException as error:
        write(output/'failure.json', dict(error=repr(error), partial_output_not_admissible=True, training_approved=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(args.request, request_sha256=args.request_sha256, output=args.output)
    print(json.dumps(dict(state=result['state'], accepted_new_images=result['accepted_new_images'],
                          balanced_training=result['balanced_training']['counts'])))


if __name__ == '__main__':
    main()
