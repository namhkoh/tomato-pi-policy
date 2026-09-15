"""Read-only original-native references with TWO authentic provenance chains.

Historical848 supplies a pose, never current lighting/labels or a legacy pair.
The clear plan plus completed original1696 capture supplies scene/native evidence.
V1 supports ONLY original_inventory's explicit V5 pilot producer. Serial receipts
need their own named adapter; no generic launcher/callback bypass is accepted.

build_bank(captures, reviews=()) returns deterministic JSON; check_bank replays it.
verify_anchor(path, bank_sha256=..., entry_id=...) requires an external byte pin
and replays the saved native audit before returning scoped1696 evidence. No Kit,
generation, image writes, depth reconstruction, grouping approval or cap reset.
Pins prove bytes/declared provenance, NOT reviewer identity or historical OS state.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json

from . import inventory as inv, original_inventory as oi
from ..native_original_capture import contracts as oc

SCHEMA = 'greenhouse.original_native_reference_bank.v1'
PROOF_SCHEMA = 'greenhouse.original_native_1696_anchor_evidence.v1'
REVIEW_SCHEMA = 'greenhouse.main_assistant_native_original_inspection.v1'
SUPPORTED_REVIEW = 'supports_clear_static_annotation_pending_global_admission'
OriginalCapturePin = oi.OriginalCapturePin
_SELF = {str(Path(__file__).resolve()): oc.sha256(__file__)}
_OPTICS = ('camera_path', 'resolution', 'intrinsics', 'focal_length_mm',
           'apertures_mm', 'aperture_offsets_mm', 'clipping_range_m',
           'depth_convention', 'crop_resize')
_FLAGS = dict(training_approved=False, source_cap_reset=False,
    independent_geometry_qualification=False, global_complete=False,
    historical_training_rows=0, new_biological_families=0,
    images_generated=False, depth_recomputed=False, native_launched=False)


@dataclass(frozen=True)
class ReviewPin:
    path: str
    sha256: str


def implementation_bindings():
    result = oc.merge_bindings(oi.implementation_bindings(), _SELF)
    oc.bind_all(result)
    return result


def _file(bindings, path):
    path = inv._path(path)
    return dict(path=str(path), sha256=bindings.hashes[str(path)])


def _chains(bindings, plan, case):
    """Additional explicit joins; frozen original audit already reconstructs FK/anatomy."""
    prior = case['pose_prior']
    root = inv._path(prior['source_capture'])
    manifest_path = root / 'manifest.json'
    sample_path = oc.safe_file(root, prior['sample_id'] + '/sample.json')
    manifest = bindings.document(manifest_path, prior['source_manifest_sha256'])
    sample = bindings.document(sample_path, prior['source_sample_sha256'])
    old_path = inv._path(prior['prior_plan'])
    old = bindings.document(old_path, manifest['source_collection_plan_sha256'])
    clear_path = inv._path(plan['source_collection_plan'])
    clear = bindings.document(clear_path, plan['source_collection_plan_sha256'])
    family, target = plan['source_family'], case['target_id']
    oc.require(plan['scene_policy'] == oc.SCENE_POLICY
        and clear['configuration']['clear_capture'] == oc.SCENE_POLICY['profile']
        and clear['family_assignments'] == old['family_assignments'] == oc.FROZEN_SPLITS
        and oc.FROZEN_SPLITS.get(family) == 'train', 'Clear authority/frozen TRAIN mismatch')
    job = inv._unique(clear['jobs'], 'job_id')[plan['collection_job_id']]
    old_job = inv._unique(old['jobs'], 'job_id')[manifest['collection_job_id']]
    source = inv._unique(job['targets'], 'target_id')[target]
    old_row = inv._unique(old_job['targets'], 'target_id')[prior['prior_target_id']]
    oc.require(job['plant_family'] == old_job['plant_family'] == family
        and job['split'] == old_job['split'] == manifest['target_family_split'] == 'train'
        and source == case['source_row']
        and source['target_id'] == case['conservative_view_cap_group'] == target
        and source['source_plant_id'] == source['split_group'] == source['variant_id'] == family,
        'Original source row/cap/plan mismatch')
    oc.require(inv._path(manifest['source_collection_plan_path']) == old_path
        and manifest['state'] == 'pilot_ready_for_review'
        and manifest['source_assets_unchanged'] is True
        and manifest['source_geometry_modified'] is False
        and prior['role'] == 'historical_pose_only_never_training_observation'
        and prior['historical_review_inherited'] is False
        and prior['prior_lighting'] == manifest['lighting'], 'Historical pose chain mismatch')
    sup = sample['supervision']
    recorded = inv._unique(manifest['samples'], 'sample_id')[prior['sample_id']]
    oc.require(sample['schema_version'] == 'greenhouse.rgbd_pilot_sample.v2'
        and sample['sample_id'] == prior['sample_id']
        and sample['calibration'] == case['prior_calibration']
        and sample['calibration']['resolution'] == [848, 408]
        and sample['robot_snapshot']['joint_degrees'] == case['expected_robot_snapshot']['joint_degrees']
        and all(sample['robot_snapshot'][k] == case['expected_robot_snapshot'][k] for k in
                ('robot_root_to_world_usd_row_vectors', 'camera_to_head_column_vectors'))
        and sup['target_id'] == old_row['target_id']
        and sup['cut_region_proposal'] == old_row['cut_region_proposal']
        and sup['review_id'] == old_row['draft_id'] == recorded['target_review_id'],
        'Historical sample/pose/source-row mismatch')
    pose = dict(role=prior['role'], manifest=_file(bindings, manifest_path),
        sample=_file(bindings, sample_path), collection_plan=_file(bindings, old_path),
        source_row=deepcopy(old_row), calibration=deepcopy(case['prior_calibration']),
        robot_snapshot=deepcopy(case['expected_robot_snapshot']),
        declaration=deepcopy(prior), historical_labels_or_reviews_inherited=False)
    scene = dict(clear_plan=_file(bindings, clear_path), source_row=deepcopy(source),
        source_manifest=_file(bindings, job['source_manifest_path']),
        family_assignments=deepcopy(oc.FROZEN_SPLITS), package=plan['package'],
        policy=deepcopy(plan['scene_policy']), scene_variants=deepcopy(plan['scene_variants']),
        original_variant=deepcopy(plan['original_variant']),
        plant_to_world=deepcopy(plan['expected_plant_to_world']),
        expected_scene_counts=deepcopy(plan['expected_scene_counts']),
        excluded_external_roots=deepcopy(plan['excluded_external_roots']),
        source_asset_manifest=_file(bindings, manifest_path),
        source_asset_bindings_sha256=inv._hash(manifest['source_usd_sha256']),
        source_and_appearance_bindings_sha256=inv._hash(plan['source_bindings']),
        historical_lighting_reproduction_claimed=False,
        actual_lighting=None, actual_renderer=None,
        actual_renderer_settings=dict(status='unknown_not_recorded_by_original_collector', value=None))
    return pose, scene


def _reviews(bindings, pins, entries):
    used = set()
    for pin in pins:
        oc.require(type(pin) is ReviewPin, 'Explicit ReviewPin required')
        path = str(inv._path(pin.path))
        oc.require(path not in used, 'Duplicate visual review pin')
        used.add(path)
        review = bindings.document(path, pin.sha256)
        oc.require(review['schema'] == REVIEW_SCHEMA and review['training_approved'] is False
            and review['training_count_increment'] == 0 and type(review['human_review_performed']) is bool
            and type(review['visual_review_performed']) is bool
            and isinstance(review['reviewer'], str) and review['reviewer'].strip()
            and isinstance(review['decision'], str) and review['decision'].strip()
            and isinstance(review['observations'], str) and review['observations'].strip(),
            'Unsupported/malformed visual review; no approval translation')
        bindings.mapping(review['source_bindings'])
        matches = [e for e in entries if e['native_observation'] is not None
            and review['source_bindings'].get(e['native_observation']['sample']['path'])
                == e['native_observation']['sample']['sha256']]
        oc.require(len(matches) == 1, 'Review must match exactly one included native sample')
        entry = matches[0]
        evidence = entry['native_observation']
        required = [evidence[k] for k in ('sample', 'rgb', 'label', 'result', 'postexit_audit', 'launcher')]
        if evidence['trace'] is not None:
            required.append(evidence['trace'])
        oc.require(review['target_id'] == entry['target_id'] and all(
            review['source_bindings'].get(p['path']) == p['sha256'] for p in required),
            'Visual review must bind the exact native receipt/label/trace chain')
        if review['decision'] == SUPPORTED_REVIEW:
            oc.require(review['visual_review_performed'] is True, 'Positive review without visual inspection')
        entry['visual_reviews'].append(dict(pin=_file(bindings, path), document=deepcopy(review)))


def _status(entry):
    reasons = []
    if entry['native_decision'] != inv.STRICT:
        reasons.append('native_case_not_strict')
    prior = entry['pose_prior']['declaration']
    if prior['allow_other_target_pose'] is not False or prior['prior_target_id'] != entry['target_id']:
        reasons.append('other_target_pose_not_supported_by_reference_v1')
    if entry['pose_request'] != dict(mode='exact_prior', native_pixel_xy=None):
        reasons.append('reframed_pose_not_supported_by_reference_v1')
    if not entry['visual_reviews']:
        reasons.append('missing_pinned_visual_support')
    elif any(r['document']['decision'] != SUPPORTED_REVIEW for r in entry['visual_reviews']):
        reasons.append('visual_hold_or_negative_history')
    return dict(usable_as_original_reference=not reasons, reasons=reasons,
        scope='exact_original_pose_current_clear_scene_not_generated_visibility_or_training_approval')


def build_bank(captures, *, reviews=()):
    """Replay exact pinned captures; retain EVERY planned case, including all holds."""
    try:
        return _build(list(captures), list(reviews))
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed original-reference evidence: ' + str(exc)) from exc


def _build(captures, reviews):
    oc.require(captures and all(type(p) is OriginalCapturePin for p in captures),
               'V1 requires explicit OriginalCapturePin objects, not generic/serial receipts')
    oc.require(all(type(p) is ReviewPin for p in reviews), 'Explicit ReviewPin objects required')
    captures.sort(key=lambda p: str(inv._path(p.capture_path)))
    reviews.sort(key=lambda p: str(inv._path(p.path)))
    roots = [str(inv._path(p.capture_path)) for p in captures]
    oc.require(len(set(roots)) == len(roots), 'Duplicate capture pin')
    bindings, entries, summaries, absent = inv._Bindings(), [], [], set()
    implementation = implementation_bindings()
    bindings.mapping(implementation)
    for pin in captures:
        # No callback parameter: public original adapter performs genuine saved-
        # buffer/source/FK/smoke replay, NOT just a receipt-declared passed flag.
        rows, summary, hashes = oi.original_observed_rows(pin)
        bindings.mapping(hashes)
        summaries.append(summary)
        capture = inv._path(pin.capture_path)
        absent.add(capture / 'failure.json')
        launcher = bindings.document(pin.launcher_receipt_path, pin.launcher_receipt_sha256)
        result = bindings.document(capture / 'result.json', pin.result_sha256)
        plan = bindings.document(launcher['plan_path'], launcher['plan_sha256'])
        audit = bindings.document(launcher['audit_path'], launcher['audit_sha256'])
        audited = inv._unique(audit['records'], 'case_id')
        by_case = inv._unique(rows, 'candidate_id')
        for case in plan['cases']:
            pose, scene = _chains(bindings, plan, case)
            cid = case['case_id']
            review = audited[cid]
            entry = dict(id=inv._hash(dict(capture=str(capture), result_sha256=pin.result_sha256, case_id=cid)),
                target_id=case['target_id'], source_family=plan['source_family'], split='train',
                conservative_view_cap_group=case['conservative_view_cap_group'],
                native_decision=review['decision'], native_audit=deepcopy(review),
                pose_prior=pose, pose_request=deepcopy(case['pose_request']), scene_authority=scene,
                native_observation=None, visual_reviews=[], compatibility_group=None, **_FLAGS)
            if cid not in by_case:
                oc.require(review['decision'] == 'held_pre_render', 'Missing captured reference row')
                absent.add(capture / cid)
            else:
                row = by_case[cid]
                folder = inv._path(row['sample_path'])
                meta = bindings.document(folder / 'sample.json')
                label = bindings.document(folder / 'supervision/label.json')
                trace_path = folder / 'supervision/query_trace.json'
                trace = bindings.document(trace_path) if row['provenance']['trace_sha256'] else None
                inv._trace_consistency(label, trace)
                if review['decision'] == inv.STRICT:
                    oc.require(label['eligible'] is True and trace is not None and trace['passed'] is True
                        and label['clarity']['reasons'] == review['clarity_reasons'] == []
                        and trace['reasons'] == review['trace_reasons'] == [], 'Contradictory strict native evidence')
                scene.update(actual_lighting=deepcopy(meta['lighting']), actual_renderer=meta['renderer'])
                evidence = dict(sample=_file(bindings, folder / 'sample.json'),
                    rgb=_file(bindings, folder / 'inputs/rgb.png'), label=_file(bindings, folder / 'supervision/label.json'),
                    trace=_file(bindings, trace_path) if trace is not None else None,
                    request=_file(bindings, capture / 'request.json'), result=_file(bindings, capture / 'result.json'),
                    native_plan=_file(bindings, launcher['plan_path']), launcher=_file(bindings, pin.launcher_receipt_path),
                    postexit_audit=_file(bindings, launcher['audit_path']),
                    automatic_audit=_file(bindings, capture / 'automatic_audit.json'),
                    calibration=deepcopy(meta['calibration']), robot_snapshot=deepcopy(meta['robot_snapshot']),
                    callback=deepcopy(meta['synchronization']), geometry_screen=deepcopy(meta['geometry_screen']),
                    decoded_rgb_sha256=row['decoded_rgb_sha256'], smoke_bindings={
                        str(oc.safe_file(capture, p)): h for p, h in result['smoke_bindings'].items()},
                    audit_replayed_by_this_build=True, launcher_exit_code=launcher['exit_code'],
                    launcher_exit_evidence='authenticated_producer_declaration_not_OS_attestation',
                    visual_truth_independently_verified=False,
                    paired_848_1696_proof=False, legacy_sensor_prerequisite_satisfied=False)
                entry['native_observation'] = evidence
                mount = evidence['robot_snapshot']['camera_to_head_column_vectors']
                group = dict(clear_plan=scene['clear_plan'], source_family=plan['source_family'],
                    source_manifest=scene['source_manifest'], policy=scene['policy'],
                    source_asset_bindings_sha256=scene['source_asset_bindings_sha256'],
                    appearance_bindings_sha256=inv._hash({p: h for p, h in plan['source_bindings'].items()
                        if Path(p).suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr', '.hdr', '.mdl', '.mtlx')
                        and Path(p).is_relative_to(Path(plan['package']))}),
                    scene_variants=scene['scene_variants'], plant_to_world=scene['plant_to_world'],
                    counts=scene['expected_scene_counts'], exclusions=scene['excluded_external_roots'],
                    lighting=scene['actual_lighting'], renderer=scene['actual_renderer'],
                    optics={k: deepcopy(evidence['calibration'][k]) for k in _OPTICS},
                    mount_index_only=[[round(float(v), 10) or 0.0 for v in r] for r in mount],
                    mount_recheck_tolerance=1e-9, full_renderer_settings_unknown=True)
                entry['compatibility_group'] = inv._hash(group)
                entry['compatibility_basis'] = group
                absent.add(folder / 'bundle.json')
                if trace is None:
                    absent.add(trace_path)
            entries.append(entry)
    _reviews(bindings, reviews, entries)
    for entry in entries:
        entry['reference_status'] = _status(entry)
    entries.sort(key=lambda e: e['id'])
    eligible = [e for e in entries if e['reference_status']['usable_as_original_reference']]
    bindings.finish()
    oc.require(all(not p.exists() for p in absent), 'Failure or unsupported data appeared during bank build')
    bank = dict(schema=SCHEMA, state='replayed_original_references_not_generated_or_training_approval',
        inputs=dict(captures=[asdict(p) for p in captures], reviews=[asdict(p) for p in reviews]),
        entries=entries, capture_coverage=summaries,
        counts=dict(planned_cases=len(entries), captured_cases=sum(e['native_observation'] is not None for e in entries),
            native_decisions=dict(Counter(e['native_decision'] for e in entries)),
            target_ids=len({e['target_id'] for e in entries}), reference_ready_cases=len(eligible),
            reference_ready_targets=len({e['target_id'] for e in eligible})),
        source_bindings=dict(sorted(bindings.hashes.items())), implementation_bindings=implementation,
        proof_scope='1696_native_original_plus_known_surfaces_not_legacy_paired_resolution',
        review_scope='all_explicit_pins_retained_no_review_discovery_or_identity_attestation',
        generated_prepare_implemented=False, original_reviews_modified=False, **_FLAGS)
    bank['sha256'] = inv._hash(bank)
    return bank


def check_bank(bank):
    """Full deterministic source/native replay, not an approval or hash-only check."""
    oc.require(isinstance(bank, dict) and bank.get('schema') == SCHEMA, 'Unknown original-reference bank schema')
    request = bank['inputs']
    rebuilt = build_bank([OriginalCapturePin(**p) for p in request['captures']],
                         reviews=[ReviewPin(**p) for p in request['reviews']])
    oc.require(bank == rebuilt, 'Bank differs from independently replayed inputs')
    return rebuilt


def verify_anchor(bank_path, *, bank_sha256, entry_id):
    """Require caller-pinned BANK BYTES; return only exact original1696 proof."""
    path = oc.pin(bank_path, bank_sha256)
    bank = check_bank(oc.read_json(path))
    matches = [e for e in bank['entries'] if e['id'] == entry_id]
    oc.require(len(matches) == 1 and matches[0]['reference_status']['usable_as_original_reference'],
               'Anchor absent, held, unreviewed, or unsupported')
    entry = matches[0]
    oc.pin(path, bank_sha256)
    return dict(schema=PROOF_SCHEMA, bank=dict(path=str(path), sha256=bank_sha256),
        entry_id=entry_id, target_id=entry['target_id'], compatibility_group=entry['compatibility_group'],
        pose_prior=deepcopy(entry['pose_prior']), scene_authority=deepcopy(entry['scene_authority']),
        native_observation=deepcopy(entry['native_observation']), visual_reviews=deepcopy(entry['visual_reviews']),
        binding_scope='exact_original_native1696_camera_scene_not_generated_target_visibility',
        consumer_must_recheck_case_mount_optics_scene_and_fresh_native_geometry=True,
        legacy_sensor_prerequisite_satisfied=False, paired_848_1696_proof=False, **_FLAGS)


def write_bank(output, captures, *, reviews=()):
    path = oc.new_destination(output, ())
    bank = build_bank(captures, reviews=reviews)
    protected = [p['capture_path'] for p in bank['inputs']['captures']]
    protected += [e['scene_authority']['package'] for e in bank['entries']]
    protected += [e['native_observation']['native_plan']['path'] for e in bank['entries']
                  if e['native_observation'] is not None]
    # Protect exact bound files, not their broad diagnostics parent: an ignored
    # queue script may live directly beside the NEW diagnostic output directory.
    protected += list(bank['source_bindings'])
    path = oc.new_destination(path, protected)
    path.parent.mkdir(parents=True, exist_ok=True)
    oc.write_new(path, bank)
    return dict(path=str(path), sha256=oc.sha256(path), counts=bank['counts'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build')
    build.add_argument('--capture', nargs=4, action='append', required=True,
                       metavar=('CAPTURE', 'RESULT_SHA', 'LAUNCH_RECEIPT', 'LAUNCH_SHA'))
    build.add_argument('--review', nargs=2, action='append', default=[], metavar=('PATH', 'SHA'))
    build.add_argument('--output', required=True)
    check = sub.add_parser('check')
    check.add_argument('--bank', required=True)
    check.add_argument('--bank-sha256', required=True)
    check.add_argument('--entry-id', help='Also verify this exact anchor')
    args = parser.parse_args(argv)
    if args.action == 'build':
        result = write_bank(args.output, [OriginalCapturePin(*v) for v in args.capture],
                            reviews=[ReviewPin(*v) for v in args.review])
    elif args.entry_id:
        result = verify_anchor(args.bank, bank_sha256=args.bank_sha256, entry_id=args.entry_id)
    else:
        path = oc.pin(args.bank, args.bank_sha256)
        result = check_bank(oc.read_json(path))['counts']
        oc.pin(path, args.bank_sha256)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
