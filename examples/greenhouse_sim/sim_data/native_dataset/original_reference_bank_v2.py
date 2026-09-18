"""Original reference bank v2 with fixed, named producer-specific readers.

Accepts the original V5 pilot, serial39, and serial_phases original batches.
Each producer is independently authenticated by its exact inventory adapter.
Review/pose/scene/optics and source-cap rules reuse the frozen V1 bank helpers.
V2 is a new schema; it cannot masquerade as V1 proof to a frozen consumer.
No launch, source edit, independent geometry credit or training approval.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from . import inventory as inv, original_reference_bank as v1
from . import original_inventory as pilot, original_inventory_v2 as serial
from . import original_phase_inventory as phase
from ..native_original_capture import contracts as oc

SCHEMA = 'greenhouse.original_native_reference_bank.v2'
PROOF_SCHEMA = 'greenhouse.original_native_1696_anchor_evidence.v2'
REVIEW_SCHEMA = v1.REVIEW_SCHEMA
SUPPORTED_REVIEW = v1.SUPPORTED_REVIEW
ReviewPin = v1.ReviewPin
OriginalCapturePin = pilot.OriginalCapturePin
SerialCapturePin = serial.SerialCapturePin
PhaseCapturePin = phase.PhaseCapturePin
_KINDS = {OriginalCapturePin: 'v5_original_pilot.v1', SerialCapturePin: 'serial39_original.v2',
          PhaseCapturePin: 'serial_phases_original.v1'}
_TYPES = {v: k for k, v in _KINDS.items()}
_FLAGS, _OPTICS = v1._FLAGS, v1._OPTICS
_file, _chains, _reviews, _status = v1._file, v1._chains, v1._reviews, v1._status
_SELF = {str(Path(__file__).resolve()): oc.sha256(__file__)}


def implementation_bindings():
    result = oc.merge_bindings(v1.implementation_bindings(), serial.implementation_bindings(),
        phase.implementation_bindings(), _SELF)
    oc.bind_all(result)
    return result


def _decode(record):
    oc.require(isinstance(record, dict) and record.get('producer') in _TYPES, 'Unregistered original producer')
    value = dict(record)
    cls = _TYPES[value.pop('producer')]
    oc.require(set(value) == {'capture_path', 'result_sha256', 'launcher_receipt_path', 'launcher_receipt_sha256'},
        'Exact typed original capture pin required')
    return cls(**value)


def _observed(pin):
    # Fixed registry, never caller-supplied loaders or relabelled receipt schemas.
    if type(pin) is OriginalCapturePin:
        return pilot.original_observed_rows(pin)
    if type(pin) is SerialCapturePin:
        return serial.original_observed_rows(pin)
    if type(pin) is PhaseCapturePin:
        return phase.original_observed_rows(pin)
    raise ValueError('Unregistered original capture producer')


def _locations(pin, launcher):
    if type(pin) is PhaseCapturePin:
        return dict(plan_path=launcher['submitted_plan_path'], audit_path=launcher['audit']['audit_path'],
                    audit_sha256=launcher['audit']['audit_sha256'])
    return {k: launcher[k] for k in ('plan_path', 'audit_path', 'audit_sha256')}


def build_bank(captures, *, reviews=()):
    """Replay exact pinned captures; retain EVERY planned case, including all holds."""
    try:
        return _build(list(captures), list(reviews))
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed original-reference evidence: ' + str(exc)) from exc


def _build(captures, reviews):
    oc.require(captures and all(type(p) in _KINDS for p in captures),
               'Explicit registered original producer pins required')
    oc.require(all(type(p) is ReviewPin for p in reviews), 'Explicit ReviewPin objects required')
    captures.sort(key=lambda p: str(inv._path(p.capture_path)))
    reviews.sort(key=lambda p: str(inv._path(p.path)))
    roots = [str(inv._path(p.capture_path)) for p in captures]
    oc.require(len(set(roots)) == len(roots), 'Duplicate capture pin')
    bindings, entries, summaries, absent = inv._Bindings(), [], [], set()
    implementation = implementation_bindings()
    bindings.mapping(implementation)
    for pin in captures:
        # Fixed producer-specific public adapter performs genuine saved-
        # buffer/source/FK/smoke replay, NOT just a receipt-declared passed flag.
        rows, summary, hashes = _observed(pin)
        bindings.mapping(hashes)
        summaries.append(summary)
        capture = inv._path(pin.capture_path)
        absent.add(capture / 'failure.json')
        launcher = bindings.document(pin.launcher_receipt_path, pin.launcher_receipt_sha256)
        result = bindings.document(capture / 'result.json', pin.result_sha256)
        locations = _locations(pin, launcher)
        plan = bindings.document(locations['plan_path'], launcher['plan_sha256'])
        audit = bindings.document(locations['audit_path'], locations['audit_sha256'])
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
                native_observation=None, visual_reviews=[], compatibility_group=None,
                launcher_producer=_KINDS[type(pin)], **_FLAGS)
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
                    native_plan=_file(bindings, locations['plan_path']), launcher=_file(bindings, pin.launcher_receipt_path),
                    postexit_audit=_file(bindings, locations['audit_path']),
                    automatic_audit=_file(bindings, capture / 'automatic_audit.json'),
                    calibration=deepcopy(meta['calibration']), robot_snapshot=deepcopy(meta['robot_snapshot']),
                    callback=deepcopy(meta['synchronization']), geometry_screen=deepcopy(meta['geometry_screen']),
                    decoded_rgb_sha256=row['decoded_rgb_sha256'], smoke_bindings={
                        str(oc.safe_file(capture, p)): h for p, h in result['smoke_bindings'].items()},
                    audit_replayed_by_this_build=True, launcher_exit_code=launcher['exit_code'],
                    launcher_exit_evidence='authenticated_producer_declaration_not_OS_attestation',
                    launcher_producer=_KINDS[type(pin)], launcher_schema=launcher.get('schema'),
                    launcher_state=launcher['state'],
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
        inputs=dict(captures=[dict(producer=_KINDS[type(p)], **asdict(p)) for p in captures], reviews=[asdict(p) for p in reviews]),
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
    rebuilt = build_bank([_decode(p) for p in request['captures']],
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


