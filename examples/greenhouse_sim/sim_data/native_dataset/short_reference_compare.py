"""Offline evidence, NOT short-budget qualification or a native-V2 relabel.

Consume Rawls's externally SHA-pinned serial_phases receipt containing
audit.matched_comparison (greenhouse.native_query_reference_comparison_handoff.v1).
Its old completion is pinned separately; the outer receipt pins new completion.
No registry imports, launches, source writes, image extraction, or thresholds.

API: compare_receipt(path, *, expected_sha256, roi_padding_px=64) -> JSON dict.
write_report(report, destination) writes ONE new JSON with an embedded review
index; parent must exist, destination must be outside protected source roots.
CLI: --receipt PATH --receipt-sha256 SHA --output NEW_JSON [--roi-padding-px 64].
Padding is a recorded DISPLAY region, never a qualification tolerance. Full RGB
paths and native pixel boxes are supplied; no review image/model input is made.
Reference stored V2 must replay exactly. Old native legacy is replayed first,
then ONLY annotation_epoch/policy are overlaid in RAM; canonicalization removes
only files. Full comparison-V2 evidence is retained, with original provenance.
The fixed grid and legacy trace start at 8mm: junction-before-8mm and continuous
visibility remain unverified, even when all sampled probes pass.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path

import numpy as np

from ..automated_native_review import check_native_evidence, trace_review
from ..capture_contract import fingerprint, project, transform_points
from ..cut_regions import _oriented_chain, _sample
from ..dataset_review import require, verify_bindings
from ..depth_preview import sha256
from ..native_budget import REFERENCE, TRIAL, evidence as budget_evidence
from ..native_clear_contract import crop_box
from ..native_clear_labels import derive
from ..plant_variant_catalogue import load_for_inspection
from .bundle import SampleReader, digest, read_bounded, safe_path
from . import query_audit_v2 as v2

SCHEMA = 'greenhouse.native_short_reference_comparison.v1'
HANDOFF = 'greenhouse.native_query_reference_comparison_handoff.v1'
LEGACY_SCHEMA = 'greenhouse.generated_native_multiview_sample.v1'
CAPTURED = 'native_captured_pending_review'
FLAGS = dict(native_profile_qualified=False, training_approved=False,
    diversity_increment=0, source_cap_reset=False, visual_review_performed=False,
    photometric_equivalence_claimed=False, continuous_visibility_verified=False,
    junction_before_8mm_verified=False, native_depth_reconstructed=False)
_SELF_PIN = sha256(__file__)


def _path(value):
    require(isinstance(value, (str, Path)) and Path(value).is_absolute(), 'Absolute path required')
    return Path(value).resolve()


def _pin(path, expected, pins):
    path = _path(path)
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'SHA256 required')
    require(str(path) not in pins or pins[str(path)] == expected, 'Conflicting path pin')
    require(path.is_file() and sha256(path) == expected, 'Changed pinned file: ' + str(path))
    pins[str(path)] = expected
    return path


def _json(path, expected, pins):
    path = _pin(path, expected, pins)
    raw = read_bounded(path)
    require(digest(raw) == expected, 'Changed JSON during read')
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    def bad_number(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_number)


def _keys_changed(a, b):
    return sorted(k for k in a.keys() | b.keys() if k not in a or k not in b or a[k] != b[k])


def _rows(result, plan):
    planned = {}
    for case in plan['target_cases']:
        for spec in case['views']:
            name = spec['candidate_id']
            require(isinstance(name, str) and name not in ('', '.', '..')
                    and not any(c in name for c in '/\\:'), 'Unsafe candidate ID')
            require(name not in planned, 'Duplicate planned candidate')
            planned[name] = (case['target_id'], spec, case['conservative_view_cap_group'])
    rows = {}
    for row in result['records']:
        name = row['candidate_id']
        require(name in planned and name not in rows, 'Unplanned/duplicate capture decision')
        require((row['target_id'], row['requested_spec']) == planned[name][:2], 'Wrong target/spec')
        require(row['state'] in (CAPTURED, 'rejected_pose', 'rejected_possible_geometry_overlap'),
                'Unknown capture decision')
        rows[name] = row
    require(sum(r['state'] == CAPTURED for r in rows.values()) == result['captured_frames'],
            'Wrong captured count')
    return rows, planned  # Missing decisions remain explicit report rows, never silently dropped.


def _capture(side, plan, pins, *, reference):
    root = _path(side['capture_path'])
    require(not (root/'failure.json').exists() and not (root.parent/'failure.json').exists(),
            'Failed capture/completion')
    request = _json(root/'request.json', side['request_sha256'], pins)
    result = _json(root/'result.json', side['result_sha256'], pins)
    audit = _json(side['audit_path'], side['audit_sha256'], pins)
    require(_path(request['plan_path']) == _path(side['plan_path'])
            and request['plan_sha256'] == side['plan_sha256'], 'Request/plan mismatch')
    require(request['training_started'] is False and result['training_approved'] is False
            and result['source_assets_unchanged'] is True and result['source_cap_reset'] is False,
            'Unsafe or incomplete capture')
    profile = REFERENCE if reference else TRIAL
    require(all(d['native_instance_backend'] == 'fast' and d['render_budget_profile'] == profile
                and d['render_profile_experiment'] is False for d in (request, result)),
            'Only fixed fast matched profiles supported')
    if reference:
        require(result['plan_sha256'] == side['plan_sha256']
                and result['request_sha256'] == side['request_sha256'], 'Reference result pins differ')
        v2.verify_epoch_documents(request, result)
    require(result['state'] == (v2.CAPTURE_STATE if reference else
            'native_generated_multiview_pilot_complete_pending_review'), 'Incomplete result')
    require(audit['state'] == (v2.AUDIT_STATE if reference else 'completed_automatic_annotation_replay')
            and _path(audit['capture']) == root and audit['plan_sha256'] == side['plan_sha256']
            and audit['request_sha256'] == side['request_sha256']
            and audit['result_sha256'] == side['result_sha256']
            and audit['training_approved'] is False and audit['original_reviews_modified'] is False,
            'Audit does not bind capture')
    rows, planned = _rows(result, plan)
    audited = {}
    for item in audit['records']:
        sample = _path(item['sample'])
        require(sample.parent == root and sample.name not in audited, 'Wrong/duplicate audited sample')
        require(item['target_id'] == planned[sample.name][0], 'Audit target mismatch')
        audited[sample.name] = item
    require(set(audited) == {k for k, r in rows.items() if r['state'] == CAPTURED}
            and dict(Counter(r['decision'] for r in audited.values())) == audit['counts']
            and sum(r['decision'] == 'accept_strict_automatic_annotation_candidate' for r in audited.values())
                == result['automatically_clear_annotation_candidates'],
            'Audit membership/count mismatch')
    return dict(root=root, request=request, result=result, rows=rows, planned=planned,
                audited=audited, side=side, reference=reference)


def _budget(meta, row, profile, captured_index):
    expected = budget_evidence(profile, captured_index)
    actual = meta['render_budget']
    require(actual == row['render_budget'] and all(actual.get(k) == v for k, v in expected.items()),
            'Row/sample/profile budget mismatch')
    requests = actual['actual_orchestrator_requests']
    require(type(requests) is int and requests >= expected['native_step_calls']
            and meta['synchronization']['render_budget_subframes'] == expected['requested_subframes'],
            'Actual request/subframe evidence mismatch')
    # Retries are reported, never silently classified as exact 8/56 exposure.
    return dict(requested_subframes=expected['requested_subframes'],
        actual_orchestrator_requests=requests, subframes_per_request=8,
        actual_requested_subframes=requests*8,
        retry_requests=requests-expected['native_step_calls'],
        exact_nominal_request_count=requests == expected['native_step_calls'])


def _freshness(meta, previous):
    freshness = meta['synchronization']['freshness']
    require(type(freshness['callback_sequence']) is int and freshness['callback_sequence'] > 0,
            'Missing callback sequence')
    if previous is not None:
        require(freshness['callback_sequence'] > previous['callback_sequence']
                and all(freshness[k] != previous[k] for k in ('camera_sha256','rgb_sha256','depth_sha256')),
                'Within-run stale camera/callback evidence')
    return deepcopy(freshness)


def replay_frame(reader, row, geometry_report, *, reference):
    """Verified saved buffers -> original replay + private comparison V2; no writes.

    reference=False ONLY accepts legacy sample schema; True ONLY saved native V2.
    Caller owns request/plan/audit/completion bindings (compare_receipt does this).
    """
    require(type(reference) is bool, 'Explicit reference boolean required')
    reader.verify_all()
    meta = reader.metadata
    require(meta['schema_version'] == (v2.SAMPLE_SCHEMA if reference else LEGACY_SCHEMA),
            'Unexpected native source epoch')
    require(meta['sample_id'] == row['candidate_id'] and meta['supervision']['target_id'] == row['target_id']
            and meta['robot_snapshot'] == row['robot_snapshot'] and meta['geometry_screen'] == row['screen'],
            'Sample/row mismatch')
    require(digest(reader.read('sample.json')) == row['sample_sha256']
            and digest(reader.read('supervision/label.json')) == row['label_sha256'], 'Changed sample/label')
    require(meta['training_sample_approved'] is False and meta['native_instance_backend'] == 'fast'
            and meta['calibration']['resolution'] == [1696, 816]
            and meta['calibration']['crop_resize'] is None
            and meta['calibration']['depth_convention'] == 'optical_axis_z_metres_not_ray_range'
            and meta['synchronization']['engine_frame_id_verified'] is False, 'Unsupported native contract')
    rgb, depth = reader.image('inputs/rgb.png'), reader.array('inputs/depth_m.npy')
    validity = reader.image('inputs/depth_valid.png')
    require(set(np.unique(validity)) <= {0, 255}, 'Invalid native validity values')
    valid, components = validity != 0, reader.array('supervision/component_id.npy')
    target = reader.image('supervision/target_visible.png')
    require(rgb.shape == (816,1696,3) and rgb.dtype == np.uint8
            and depth.shape == valid.shape == components.shape == (816,1696)
            and depth.dtype == np.float32 and components.dtype.kind in 'ui', 'Invalid native buffers')
    identities = reader.json('supervision/identities.json')
    catalogue = identities['component_catalogue']
    check_native_evidence(meta, rgb, depth, components, target, catalogue)
    args = (meta, geometry_report, rgb, depth, valid, components, catalogue)
    if reference:
        label, trace = v2.annotate_for_storage(*args, target_mask=target)
    else:
        require(not any(k in meta for k in v2.annotation_fields()), 'Legacy source already carries V2 epoch')
        label = derive(*args)
        trace = trace_review(meta, geometry_report, label, *args[2:]) if label['eligible'] else None
    require(label == reader.json('supervision/label.json'), 'Original label replay differs')
    require(trace == row.get('query_trace') and (trace is None or
            trace == reader.json('supervision/query_trace.json')), 'Original trace replay differs')
    if trace is None:
        require('supervision/query_trace.json' not in reader.manifest['files'] if reader.manifest else
                not (reader.root/'supervision/query_trace.json').exists(), 'Excluded sample carries unexpected trace')
    require(row['eligible_annotation'] == label['eligible'] and row['label_reason'] == label['reason']
            and row['automatic_annotation_eligible'] == bool(trace is not None and trace['passed']),
            'Original annotation decision differs')
    overlay = {} if reference else v2.annotation_fields()
    private = deepcopy(meta)
    private.update(overlay)
    if reference:
        compared_label, compared_trace = label, trace
    else:
        compared_label, compared_trace = v2.annotate_for_storage(private, *args[1:], target_mask=target)
    proof = dict(native_source_schema=meta['schema_version'], native_v2_capture=reference,
        comparison_only=not reference, metadata_overlay=overlay,
        original_sample_sha256=row['sample_sha256'], original_metadata_sha256=fingerprint(meta),
        comparison_input_metadata_sha256=fingerprint(v2.canonical_annotation_metadata(private)),
        original_label_replayed_exact=True, original_trace_replayed_exact=trace is not None,
        original_label=label, original_trace=trace, v2_label=compared_label, v2_trace=compared_trace)
    rgb_path = safe_path(reader.root, reader.manifest['files']['inputs/rgb.png']['stored_path']
                         if reader.manifest else 'inputs/rgb.png')
    return dict(meta=meta, rgb=rgb, depth=depth, valid=valid, components=components,
        instances=reader.array('supervision/renderer_instance_id.npy'), identities=identities,
        target=target != 0, proof=proof, rgb_path=str(rgb_path),
        rgb_sha256=digest(reader.read('inputs/rgb.png')))


def _semantic_codes(array, mapping, vocabulary, *, reserved):
    require(array.ndim == 2 and array.dtype.kind in 'ui', 'Integer semantic raster required')
    parsed = {}
    for key, value in mapping.items():
        require(isinstance(key, str) and key.isdecimal() and str(int(key)) == key,
                'Noncanonical semantic ID')
        require(isinstance(value, str) and value.startswith('/'), 'Invalid semantic prim identity')
        parsed[int(key)] = ('prim', value)
    ids, inverse = np.unique(array, return_inverse=True)
    codes = []
    for ident in ids:
        ident = int(ident)
        require(ident in parsed or ident in reserved, 'Unmapped native semantic ID')
        token = parsed.get(ident, ('unmapped_reserved', ident))
        codes.append(vocabulary.setdefault(token, len(vocabulary)))
    return np.asarray(codes, dtype=np.int32)[inverse].reshape(array.shape)


def semantic_changes(a, mapping_a, b, mapping_b):
    """Per-frame renderer ID resolution. Unmapped 0/1 remain explicitly reserved."""
    require(a.shape == b.shape, 'Semantic raster shape differs')
    vocabulary = {}
    aa = _semantic_codes(a, mapping_a, vocabulary, reserved={0, 1})
    bb = _semantic_codes(b, mapping_b, vocabulary, reserved={0, 1})
    return aa != bb


def _core_changes(a, b):
    # Explicitly allowed run/observation differences; every other field is core.
    skip = {'files', 'schema_version', 'annotation_epoch', 'annotation_policy_sha256',
            'render_budget', 'synchronization', 'supervision', 'quality', 'native_target_pixels'}
    changes = _keys_changed({k:v for k,v in a.items() if k not in skip},
                            {k:v for k,v in b.items() if k not in skip})
    for group, ignored in (
        ('supervision', {'depth_evidence', 'visibility_evidence'}),
        ('quality', {'target_mask_dark_fraction', 'clear_view_gate_passed', 'clear_view_rejection_reasons'}),
        ('synchronization', {'freshness', 'reference_time', 'native_render_frame', 'render_budget_subframes'})):
        changes += [group+'.'+k for k in _keys_changed(
            {k:v for k,v in a[group].items() if k not in ignored},
            {k:v for k,v in b[group].items() if k not in ignored})]
    return changes


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
        p95=float(np.percentile(values, 95)) if values.size else None,
        maximum=float(values.max()) if values.size else None)


def _box(points, padding):
    points = np.asarray(points, dtype=float)
    require(points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all(), 'Invalid ROI points')
    lo = np.floor(points.min(axis=0)).astype(int)-padding
    hi = np.ceil(points.max(axis=0)).astype(int)+padding+1
    box = [max(0, int(lo[0])), max(0, int(lo[1])), min(1696, int(hi[0])), min(816, int(hi[1]))]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _regions(frame, other, report, padding):
    meta = frame['meta']
    regions = dict(full=[0, 0, 1696, 816],
        cut=_box([meta['supervision']['nominal_projected']['pixel_xy']], padding))
    component = report['components'][meta['supervision']['target_id'].split('/')[1]]
    chain, lengths, _, error = _oriented_chain(component, 1e-6)
    require(error <= 1e-6, 'Invalid oriented anatomy')
    labels = [f['proof'][k] for f in (frame, other) for k in ('original_label', 'v2_label')]
    ends = [label['query_evidence']['arc_m'] for label in labels if label['eligible']]
    end = max(ends, default=min(.045, float(lengths[-1])))
    arcs = sorted({0., min(.008, end), end, *(float(x) for x in lengths if 0 < x < end)})
    points = [_sample(chain, lengths, arc, component['translation_plant_m'])['point_plant_m'] for arc in arcs]
    projections = project(transform_points(points, meta['supervision']['plant_to_world_usd_row_vectors']),
                          meta['calibration'])
    regions['junction'] = (_box([projections[0]['pixel_xy']], padding)
                           if projections[0]['projection_status'] == 'in_frame' else None)
    regions['junction_to_query_bbox'] = (_box([p['pixel_xy'] for p in projections], padding)
        if all(p['projection_status'] == 'in_frame' for p in projections) else None)
    for side, f in (('old', frame), ('new', other)):
        for epoch in ('original_label', 'v2_label'):
            label = f['proof'][epoch]
            regions[side+'_'+epoch+'_query_crop'] = list(crop_box(label['query_pixel_uv'])) if label['eligible'] else None
    return regions


def _measure(a, b, regions):
    changed = semantic_changes(a['instances'], a['identities']['renderer_id_to_prim'],
                               b['instances'], b['identities']['renderer_id_to_prim'])
    component_maps = []
    for frame in (a,b):
        mapping, identities = {}, set()
        for item in frame['identities']['component_catalogue']:
            key, identity = str(item['component_index']), '/'+item['variant_id']+'/'+item['component_id']
            require(key not in mapping and identity not in identities and key != '0', 'Ambiguous component catalogue')
            mapping[key] = identity
            identities.add(identity)
        component_maps.append(mapping)
    vocabulary = {}
    component_changed = (_semantic_codes(a['components'], component_maps[0], vocabulary, reserved={0}) !=
                         _semantic_codes(b['components'], component_maps[1], vocabulary, reserved={0}))
    rgb = np.abs(a['rgb'].astype(np.int16)-b['rgb'].astype(np.int16))
    common = a['valid'] & b['valid'] & np.isfinite(a['depth']) & np.isfinite(b['depth'])
    delta = np.zeros(a['depth'].shape, dtype=np.float64)
    delta[common] = np.abs(a['depth'][common].astype(np.float64)-b['depth'][common].astype(np.float64))
    result = {}
    for name, box in regions.items():
        if box is None:
            result[name] = dict(available=False)
            continue
        x0,y0,x1,y1 = box
        roi = np.s_[y0:y1,x0:x1]
        result[name] = dict(available=True, native_box_xyxy=box, pixels=(x1-x0)*(y1-y0),
            rgb_absolute_delta_8bit=_stats(rgb[roi]),
            native_z_absolute_delta_m=_stats(delta[roi][common[roi]]),
            validity_changed_pixels=int(np.count_nonzero((a['valid'] != b['valid'])[roi])),
            valid_nonfinite_old=int(np.count_nonzero((a['valid'] & ~np.isfinite(a['depth']))[roi])),
            valid_nonfinite_new=int(np.count_nonzero((b['valid'] & ~np.isfinite(b['depth']))[roi])),
            semantic_prim_changed_pixels=int(np.count_nonzero(changed[roi])),
            semantic_component_changed_pixels=int(np.count_nonzero(component_changed[roi])),
            reserved_unmapped_old_pixels=int(np.count_nonzero(np.isin(a['instances'][roi],
                [k for k in (0,1) if str(k) not in a['identities']['renderer_id_to_prim']]))),
            reserved_unmapped_new_pixels=int(np.count_nonzero(np.isin(b['instances'][roi],
                [k for k in (0,1) if str(k) not in b['identities']['renderer_id_to_prim']]))),
            target_mask_changed_pixels=int(np.count_nonzero((a['target'] != b['target'])[roi])))
    return result


def _selection_summary(proof):
    label, trace = proof['v2_label'], proof['v2_trace']
    evidence = label['query_selection_evidence']
    return dict(eligible=label['eligible'], reason=label['reason'],
        answer=label.get('answer'), nominal_pixel_uv=label.get('nominal_pixel_uv'),
        accepted_interval_uv=label.get('accepted_interval_uv'), query_pixel_uv=label.get('query_pixel_uv'),
        passing_candidates=[r['candidate_index'] for r in evidence['candidates'] if r['both_passed']],
        candidates=[dict(candidate_index=r['candidate_index'], arc_m=r['arc_m'],
            local_passed=r['local_passed'], both_passed=r['both_passed'], rejections=r['rejections'])
            for r in evidence['candidates']], trace_passed=trace['passed'] if trace else None)


def compare_receipt(path, *, expected_sha256, roi_padding_px=64):
    """Read-only, sequential per-frame CPU comparison of one pinned matched job."""
    require(type(roi_padding_px) is int and 0 <= roi_padding_px <= 816, 'Bounded display padding required')
    pins = v2.implementation_bindings()
    _pin(Path(__file__).resolve(), _SELF_PIN, pins)
    receipt = _json(path, expected_sha256, pins)
    require(receipt['schema'] == 'greenhouse.native_serial_phases.job.v1'
            and receipt['state'] == 'owned_exit0_postexit_audited_pending_admission'
            and receipt['worker_kind'] == 'query_v2_qualification.v1'
            and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
            and receipt['training_approved'] is False and receipt['source_cap_reset'] is False,
            'Pinned completed reference receipt required')
    handoff = receipt['audit']['matched_comparison']
    require(handoff['schema'] == HANDOFF and handoff['backend'] == 'fast'
            and handoff['new_render_budget'] == REFERENCE
            and handoff['old_annotation_epoch_unchanged'] is True
            and handoff['comparisons_performed'] is False and handoff['equivalence_cutoff'] is None
            and all(handoff[k] is False for k in ('native_query_qualification_granted',
                'short_budget_qualified', 'source_cap_reset', 'training_approved'))
            and handoff['training_diversity_increment'] == 0, 'Unsupported handoff contract')
    old, new = handoff['old'], handoff['new']
    require(old['plan_sha256'] == new['plan_sha256'] == receipt['plan_sha256'], 'Different plans')
    require(_path(new['capture_path']) == _path(receipt['capture'])
            and _path(new['plan_path']) == _path(receipt['submitted_plan_path'])
            and all(new[k] == receipt['audit'][k] for k in
                    ('request_sha256', 'result_sha256', 'audit_path', 'audit_sha256')),
            'Outer completion does not bind reference')
    plan = _json(old['plan_path'], old['plan_sha256'], pins)
    require(_json(new['plan_path'], new['plan_sha256'], pins) == plan, 'Changed submitted plan')
    require(plan['split'] == 'train' and plan['resolution'] == [1696,816]
            and all(plan[k] is False for k in ('training_approved', 'source_cap_reset',
                'physical_motion_commanded', 'hidden_cut_coordinates_executable')), 'TRAIN-only plan required')
    for field in ('source_bindings', 'implementation_bindings', 'prerequisite_bindings'):
        require(isinstance(plan[field], dict) and plan[field], 'Missing plan source pins')
        for p, pin in plan[field].items():
            _pin(p, pin, pins)
    anchor_path = str(_path(plan['anchor_pair_plan']))
    require(anchor_path in pins, 'Unbound anchor plan')
    anchor = _json(anchor_path, pins[anchor_path], pins)
    generated = load_for_inspection(anchor['variant_directory'], anchor['source_collection_plan'])
    captures = [_capture(s, plan, pins, reference=r) for s,r in ((old,False),(new,True))]
    done = _json(old['completion_path'], old['completion_sha256'], pins)
    old_result = captures[0]['result']
    require(done['state'] == 'native_complete_pending_review' and type(done['native_exit_code']) is int
            and done['native_exit_code'] == 0 and _path(done['job']) == captures[0]['root'].parent
            and done['source_family'] == plan['source_family'] and done['training_approved'] is False
            and done['captured'] == old_result['captured_frames']
            and done['automatic'] == old_result['automatically_clear_annotation_candidates']
            and done['audit_counts'] == dict(Counter(r['decision'] for r in captures[0]['audited'].values())),
            'Old completion does not bind saved result/audit')
    require(handoff['planned_proposals'] == len(captures[0]['planned']) == plan['maximum_native_frames']
            and handoff['captured_frames'] == captures[1]['result']['captured_frames'], 'Handoff counts differ')
    # First CAPTURED image uses 56; rejected proposals do not advance the counter.
    for capture in captures:
        capture['capture_indices'] = {name:i for i,name in enumerate(
            name for name,row in capture['rows'].items() if row['state'] == CAPTURED)}
        capture['previous'] = None
    frames = {}
    # Validate freshness in each run's original result order, not the join order.
    # Retain only metadata/proofs; load paired image arrays later, bounded to one pair.
    for index,capture in enumerate(captures):
        for name,row in capture['rows'].items():
            if row['state'] != CAPTURED:
                continue
            reader = SampleReader(safe_path(capture['root'], name), expected_bindings={
                'sample.json':row['sample_sha256'], 'supervision/label.json':row['label_sha256']},
                expected_json={'supervision/query_trace.json':row['query_trace']} if row.get('query_trace') is not None else {})
            if reader.manifest:
                p = reader.root/'bundle.json'
                _pin(p, sha256(p), pins)  # Snapshot derived from externally bound logical sample, not external authentication.
            frame = replay_frame(reader, row, generated['report'], reference=capture['reference'])
            if reader.manifest:
                physical = [(safe_path(reader.root,e['stored_path']),e['stored_sha256'])
                            for e in reader.manifest['files'].values()]
            else:
                names = set(reader.metadata['files']) | {'sample.json','supervision/label.json'}
                if row.get('query_trace') is not None:
                    names.add('supervision/query_trace.json')
                physical = [(safe_path(reader.root,n),digest(reader.read(n))) for n in names]
            for p, pin in physical:
                _pin(p, pin, pins)
            meta = frame['meta']
            require(meta['supervision']['conservative_view_cap_group'] == capture['planned'][name][2]
                    and meta['supervision']['split_group'] == plan['source_family'], 'Wrong source ancestry')
            audit_row = capture['audited'][name]
            require(all(audit_row[k] == row[k] for k in ('sample_sha256','label_sha256'))
                    and audit_row['rgb_sha256'] == frame['rgb_sha256'], 'Audit/sample pins disagree')
            expected_decision = ('accept_strict_automatic_annotation_candidate' if row['automatic_annotation_eligible']
                else 'hold_visual_clarity' if row['eligible_annotation'] else 'exclude_geometry_or_visibility')
            require(audit_row['decision'] == expected_decision, 'Audit/replay disposition differs')
            capture['previous'] = _freshness(meta, capture['previous'])
            frame['budget'] = _budget(meta, row, REFERENCE if capture['reference'] else TRIAL,
                                      capture['capture_indices'][name])
            frame['folder'] = str(reader.root)
            for key in ('rgb','depth','valid','components','instances','identities','target'):
                frame.pop(key)
            frames[index,name] = frame
    records, review = [], []
    for name, (target, spec, cap) in captures[0]['planned'].items():
        row = dict(plan_sha256=old['plan_sha256'], target_id=target, candidate_id=name,
                   requested_spec=spec, conservative_view_cap_group=cap)
        decisions = [c['rows'].get(name) for c in captures]
        row['decisions'] = [deepcopy(d) for d in decisions]
        if not all(d is not None and d['state'] == CAPTURED for d in decisions):
            row['state'] = ('matched_rejection' if all(d is not None for d in decisions)
                            and decisions[0]['state'] == decisions[1]['state'] else 'missing_or_asymmetric_decision')
            if row['state'] == 'matched_rejection':
                row['decision_differences'] = _keys_changed(*[
                    {k:v for k,v in d.items() if k not in ('elapsed_seconds','geometry_screen_seconds')} for d in decisions])
                if row['decision_differences']:
                    row['state'] = 'rejection_evidence_mismatch_hold'
            records.append(row)
            continue
        a,b = [deepcopy(frames[i,name]) for i in (0,1)]
        core = _core_changes(a['meta'], b['meta'])
        ba,bb = a['budget'], b['budget']
        profile = ('56_vs_56_warmup' if ba['requested_subframes'] == 56 else '8_vs_56')
        if not all(x['exact_nominal_request_count'] for x in (ba,bb)):
            profile = 'retry_confounded_'+profile
        regions = _regions(a,b,generated['report'],roi_padding_px)
        summaries = [_selection_summary(f['proof']) for f in (a,b)]
        for f in (a,b):
            reader = SampleReader(f['folder'], expected_bindings={'sample.json':f['proof']['original_sample_sha256']},
                expected_json=dict({'supervision/label.json':f['proof']['original_label']}, **(
                    {'supervision/query_trace.json':f['proof']['original_trace']} if f['proof']['original_trace'] is not None else {})))
            reader.verify_all()
            f.update(rgb=reader.image('inputs/rgb.png'), depth=reader.array('inputs/depth_m.npy'),
                valid=reader.image('inputs/depth_valid.png') != 0, target=reader.image('supervision/target_visible.png') != 0,
                components=reader.array('supervision/component_id.npy'),
                instances=reader.array('supervision/renderer_instance_id.npy'), identities=reader.json('supervision/identities.json'))
        def catalogue_core(f):
            return sorted(({k:v for k,v in c.items() if k != 'component_index'}
                for c in f['identities']['component_catalogue']), key=lambda c:(c['variant_id'],c['component_id']))
        if catalogue_core(a) != catalogue_core(b):
            core.append('component_catalogue_semantic_ancestry')
        row.update(state='core_mismatch_hold' if core else 'compared_unqualified',
            budget_class=profile, budgets=[ba,bb], core_differences=core,
            interpret_as_budget_only=not core and not profile.startswith('retry'),
            original_replay_and_comparison=[a['proof'], b['proof']],
            selection_summary=summaries, selection_differences=_keys_changed(*summaries),
            regions=regions, metrics=_measure(a,b,regions),
            camera_freshness=dict(within_run_verified=True, cross_run_tokens_required_equal=False,
                engine_frame_id_verified=False), **FLAGS)
        review.append(dict(join=[old['plan_sha256'],target,name], budget_class=profile,
            rgb=[dict(path=f['rgb_path'], sha256=f['rgb_sha256']) for f in (a,b)],
            native_regions_xyxy=regions, missing_regions=[k for k,v in regions.items() if v is None],
            coordinate_system='native_1696x816_half_open_xyxy_no_resize', visual_review_performed=False))
        records.append(row)
    verify_bindings(pins)
    return dict(schema=SCHEMA, **FLAGS, equivalence_cutoff=None, receipt_sha256=expected_sha256,
        source_family=plan['source_family'], records=records, review_index=review,
        state_counts=dict(Counter(r['state'] for r in records)),
        budget_counts=dict(Counter(r['budget_class'] for r in records if 'budget_class' in r)),
        display_roi_padding_px=roi_padding_px, annotation_policy=v2.annotation_policy(),
        bindings=dict(sorted(pins.items())), protected_roots=sorted({
            *(str(c['root'].parent) for c in captures),
            str(_path(old['plan_path']).parent), str(_path(new['plan_path']).parent),
            str(_path(anchor['variant_directory']))}),
        completion_evidence='pinned_saved_exit0_declarations_not_independent_OS_attestation',
        limitations=['No release or short-profile qualification threshold selected.',
            'Fixed V2 grid and legacy trace begin at 8mm; junction-before-8mm remains unverified.',
            'Sampled checks do not certify continuous visibility; native junction/trace boxes need explicit visual review.',
            'Unmapped renderer IDs 0/1 are reserved unknown identities, not resolved anatomical labels.',
            'Timings include CPU orchestration; no isolated GPU throughput claim.',
            'Exact core mismatches or extra orchestrator requests confound render-budget interpretation.',
            'Completed TRAIN rerenders add zero independent biological diversity; no heldout tuning.'])


def write_report(report, destination):
    """Create-only single JSON, no images copied/extracted and no source edits."""
    require(report.get('schema') == SCHEMA and all(report.get(k) == v for k,v in FLAGS.items())
            and report.get('equivalence_cutoff') is None, 'Unqualified comparison report required')
    destination = _path(destination)
    require(not destination.exists() and destination.parent.is_dir(), 'New file in existing directory required')
    require(not any(destination.is_relative_to(_path(root)) for root in report['protected_roots']),
            'Output inside protected capture/plan root')
    verify_bindings(report['bindings'])
    raw = json.dumps(report, indent=2, allow_nan=False).encode('utf-8')
    with destination.open('xb') as stream:
        stream.write(raw)
    return dict(path=str(destination), sha256=digest(raw), bytes=len(raw))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roi-padding-px', type=int, default=64)
    args = parser.parse_args(argv)
    report = compare_receipt(args.receipt, expected_sha256=args.receipt_sha256, roi_padding_px=args.roi_padding_px)
    print(json.dumps(write_report(report, args.output), indent=2))


if __name__ == '__main__':
    main()
