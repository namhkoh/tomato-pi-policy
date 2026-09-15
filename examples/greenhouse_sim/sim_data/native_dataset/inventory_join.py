"""Opt-in immutable union of externally attested OBSERVED inventory packets.

join_inventories([AttestedInventoryPin(...)]) is CPU/read-only. It accepts only
the common inventory v1 schema, including original_inventory/v2 outputs. Every
component needs caller-pinned inventory AND replay-attestation files and an
explicit external authority/audit-basis declaration. Hashes authenticate bytes,
NOT execution or the honesty/independence of that authority. We do not replay
native audits, decode buffers, qualify storage, discover receipts, select rows,
normalize physical identities, or grant approvals. Supply only disjoint capture
shards; even identical overlapping captures/row IDs are rejected, not discounted.

All source/implementation and attestation binding maps share one single-use V4
Reader: initial AND final full SHA256 per unique file, fresh original-path and
link-topology checks. No cross-run or stat-only byte cache. This is not an atomic
filesystem snapshot. Rows, contexts and coverage retain their canonical JSON
bytes; original packet bytes remain untouched and pinned. Whitespace is not a
row identity. Each component retains its exact inventory metadata/scope and the
attestation's claims (binding maps are represented by their seal/count plus the
checked union). Claims are never rewritten to say this join reran an audit.

Output has its OWN schema, deliberately rejected by frozen provisional.py.
Different physical-context policies remain different; an explicit downstream
adapter must handle that before selection. A new complete global near graph
must bind this join's seal and ALL row IDs, including holds/exclusions. No old
graph or provisional selection is inherited. graph_requirement() describes that
obligation; it is not graph execution/validation or permission to select.

write_join(output, pins) exclusively creates one new file in an existing parent
outside source/capture/review roots. CLI: repeat --component INVENTORY FILE_SHA
ATTESTATION FILE_SHA EXPECTED_STATE EXTERNAL_AUTHORITY AUDIT_BASIS --output NEW.
To extend a join, pass its original component pins plus the new disjoint shard;
join outputs are not silently accepted as independent replay attestations.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re

from ..native_capture_v4 import io
from ..native_capture_v3 import reference_bank

SCHEMA = 'greenhouse.immutable_observed_inventory_join.v1'
INPUT_SCHEMA = 'greenhouse.pinned_native_observed_inventory.v1'
STATE = 'attested_observed_inventory_join_only_not_global_admission'
INPUT_STATE = 'observed_inventory_only_not_global_admission'
DECISIONS = {'accept_strict_automatic_annotation_candidate', 'hold_visual_clarity',
             'exclude_geometry_or_visibility'}
FALSE_FLAGS = ('training_approved', 'source_cap_reset', 'admission_performed',
               'original_reviews_modified', 'captures_modified', 'images_generated',
               'depth_recomputed', 'compact_qualification_granted')
BINDING_MAPS = ('source_bindings', 'implementation_bindings', 'bindings',
                'artifact_file_sha256', 'script_file_sha256')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


_LOADED_CODE = {str(Path(p).resolve()): _file_sha(p)
                for p in (__file__, io.__file__, reference_bank.__file__)}


def implementation_bindings():
    """Loaded metadata-only joiner, content reader and strict JSON parser."""
    return dict(_LOADED_CODE)


@dataclass(frozen=True)
class AttestedInventoryPin:
    inventory_path: str
    inventory_file_sha256: str
    attestation_path: str
    attestation_sha256: str
    attestation_state: str
    external_authority: str
    audit_basis: str


@dataclass(frozen=True)
class UnindexedRoot:
    """Caller-declared omitted root; no discovery, row estimate or completeness."""
    path: str
    reason: str


def _text(value):
    _require(isinstance(value, str) and bool(value.strip()), 'Explicit nonempty identity/declaration required')
    return value


def _path(reader, value):
    _text(value)
    _require(Path(value).is_absolute(), 'Explicit absolute path required')
    return reader.resolve(value)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def _sealed(packet):
    _require(isinstance(packet, dict), 'Inventory mapping required')
    _require(_digest({k: v for k, v in packet.items() if k != 'sha256'}) == _sha(packet['sha256']),
             'Changed inventory seal')


def _counts(records, coverage):
    return dict(captured_rows=len(records), receipts=len(coverage),
        decisions=dict(Counter(r['decision'] for r in records)),
        original_donor_families=dict(Counter(r['source_family'] for r in records)),
        conservative_view_cap_groups=dict(Counter(r['source_target'] for r in records)),
        variant_targets=dict(Counter(r['target_id'] for r in records)))


def _attestation(reader, pin, packet):
    path = _path(reader, pin.attestation_path)
    _require(path != _path(reader, pin.inventory_path), 'Inventory cannot attest itself')
    report = reader.json(pin.attestation_path, pin.attestation_sha256)
    _text(pin.external_authority); _text(pin.audit_basis); _text(pin.attestation_state)
    _require(report['state'] == pin.attestation_state and report['training_approved'] is False
             and report['global_complete'] is False, 'Different/unsafe external attestation')
    _require(not any(word in pin.attestation_state.lower() for word in
                     ('failed', 'failure', 'incomplete', 'running', 'waiting', 'error')),
             'Non-completed external attestation')
    for key in (*FALSE_FLAGS, 'release_approved'):
        if key in report:
            _require(report[key] is False, 'Attestation grants approval or mutations: ' + key)
    for key in ('failure', 'failures', 'error'):
        _require(not report.get(key), 'Failed external attestation')
    # The two existing execution-receipt spellings; never infer an inventory
    # from directory names or accept a report merely because it has a seal.
    found = False
    for prefix in ('inventory', 'observed_inventory'):
        names = (prefix+'_path', prefix+'_file_sha256', prefix+'_sha256')
        if any(k in report for k in names):
            found = True
            _require(all(k in report for k in names), 'Partial attestation inventory binding')
            _require(_path(reader, report[names[0]]) == _path(reader, pin.inventory_path)
                     and report[names[1]] == pin.inventory_file_sha256
                     and report[names[2]] == packet['sha256'], 'Attestation/inventory pin mismatch')
            reader.bind(report[names[0]], report[names[1]])
    _require(found, 'Attestation must bind exact inventory path, file hash and content seal')
    binding_maps = {}
    for key in BINDING_MAPS:
        if key in report:
            reader.bindings_from(report[key])
            binding_maps[key] = dict(sha256=_digest(report[key]), entries=len(report[key]))
    return dict(pin=asdict(pin),
        trust_basis='caller_designated_external_attestation_not_self_authenticating',
        external_execution_claims_independently_verified_by_joiner=False,
        attestation_claims={k: deepcopy(v) for k, v in report.items() if k not in BINDING_MAPS},
        attestation_binding_maps=binding_maps)


def _component(reader, packet, contexts, captures, receipts, sample_ids, samples, splits, absent):
    _sealed(packet)
    _require(packet['schema'] == INPUT_SCHEMA and packet['state'] == INPUT_STATE,
             'Only explicit common observed inventory components supported')
    _require(all(packet[k] is False for k in FALSE_FLAGS), 'Approved or mutated inventory forbidden')
    _require(packet['scope']['global_complete'] is False, 'Observed scope only')
    provenance = packet['provenance']
    own_bindings = provenance['source_bindings']
    reader.bindings_from(own_bindings)
    reader.bindings_from(provenance['implementation_bindings'])
    own = {_path(reader, k): v for k, v in own_bindings.items()}
    for path, sha in provenance['implementation_bindings'].items():
        _require(own.get(_path(reader, path)) == sha, 'Unbound inventory implementation')
    for name, context in packet['scene_contexts'].items():
        _require(_digest(context) == _sha(name), 'Changed scene context content key')
        _require(name not in contexts or _canonical(contexts[name]) == _canonical(context),
                 'Conflicting scene context')
        contexts[name] = deepcopy(context)
    coverage = packet['scope']['explicit_receipts']
    _require(isinstance(coverage, list) and coverage, 'Explicit capture coverage required')
    current = {}
    for entry in coverage:
        capture = _path(reader, entry['capture_path'])
        receipt = _path(reader, entry['receipt_path'])
        plan = _path(reader, entry['plan_path'])
        storage = _path(reader, entry['storage_root'])
        _require(capture not in captures and receipt not in receipts, 'Duplicate/conflicting capture or receipt')
        captures.add(capture); receipts.add(receipt); current[capture] = entry
        _require(own.get(receipt) == entry['receipt_sha256'] and plan in own,
                 'Coverage receipt/plan not bound by component')
        for field in ('planned_rows', 'captured_rows', 'noncaptured_rows'):
            _require(type(entry[field]) is int and entry[field] >= 0, 'Invalid coverage count')
        _require(entry['planned_rows'] == entry['captured_rows'] + entry['noncaptured_rows'],
                 'Incomplete capture coverage accounting')
        for root in (capture, storage):
            failure = reader.resolve(root/'failure.json')
            _require(not failure.exists(), 'Capture failure marker exists')
            absent.add(failure)
    per_capture = Counter()
    _require(isinstance(packet['records'], list), 'Inventory row list required')
    for row in packet['records']:
        key = _text(row['sample_id'])
        capture = _path(reader, row['capture_path'])
        sample = _path(reader, row['sample_path'])
        _require(key not in sample_ids and sample not in samples, 'Duplicate/conflicting row ID or sample path')
        sample_ids.add(key); samples.add(sample)
        _require(capture in current, 'Row outside component capture coverage')
        entry = current[capture]
        _require(sample.parent == _path(reader, entry['storage_root']) and sample.name == row['candidate_id'],
                 'Row outside captured sample storage')
        per_capture[capture] += 1
        family, target = _text(row['source_family']), _text(row['source_target'])
        _text(row['target_id'])
        _require(row['original_donor_family'] == family and target.startswith(family+'/')
                 and len(target) > len(family)+1 and row['conservative_view_cap_group'] == target,
                 'Changed original ancestry/cap identity')
        split = row['split']
        _require(split in ('train', 'validation', 'test') and
                 (family not in splits or splits[family] == split), 'Conflicting donor split')
        splits[family] = split
        _require(row['resolution'] == [1696, 816] and row['decision'] in DECISIONS
                 and row['training_approved'] is False and row['source_cap_reset'] is False
                 and row['depth_recomputed'] is False, 'Unsafe observation')
        for field in ('encoded_rgb_sha256', 'decoded_rgb_sha256', 'geometry_sha256'):
            _sha(row[field])
        for basis, field in (('scene_identity_basis', 'scene_sha256'), ('camera_identity_basis', 'camera_sha256')):
            _require(_digest(row[basis]) == row[field], 'Changed recorded physical identity')
        _require(_digest(dict(scene=row['scene_sha256'], camera=row['camera_sha256'])) ==
                 row['scene_camera_sha256'], 'Changed scene-camera identity')
        _require(row['scene_identity_basis']['static_scene_sha256'] in packet['scene_contexts'],
                 'Missing component scene context')
        proof = row['provenance']
        for name, expected in ((proof['receipt_path'], entry['receipt_sha256']),
                               (proof['plan_path'], proof['plan_sha256']),
                               (str(capture/'result.json'), proof['result_sha256']),
                               (str(capture/'request.json'), proof['request_sha256'])):
            _require(own.get(_path(reader, name)) == expected, 'Row provenance missing/conflicting pin')
        _require(_path(reader, proof['receipt_path']) == _path(reader, entry['receipt_path'])
                 and proof['receipt_sha256'] == entry['receipt_sha256']
                 and _path(reader, proof['plan_path']) == _path(reader, entry['plan_path']),
                 'Row differs from capture receipt/plan')
        review = row['annotation_review']
        _require(type(review['passed']) is bool and review['evidence_id'] == proof['receipt_sha256']
                 and review['passed'] is (row['decision'] == 'accept_strict_automatic_annotation_candidate'),
                 'Changed annotation review evidence')
        # A later unbound compact marker must not change a raw source's meaning.
        marker = reader.resolve(sample/'bundle.json')
        _require(marker.is_file() if marker in own else not marker.exists(), 'Unbound/missing compact marker')
        if marker not in own:
            absent.add(marker)
    _require(all(per_capture[p] == e['captured_rows'] for p, e in current.items()),
             'Captured row membership/coverage mismatch')
    expected_counts = _counts(packet['records'], coverage)
    for name, value in expected_counts.items():
        _require(_canonical(packet['counts'][name]) == _canonical(value), 'Component count mismatch: '+name)
    if 'source_kinds' in packet['counts']:
        _require(packet['counts']['source_kinds'] == dict(Counter(r['source_kind'] for r in packet['records'])),
                 'Component source-kind count mismatch')


def join_inventories(components, *, unindexed_roots=()):
    """No audit replay. Accept externally attested disjoint shards, fully rehash union."""
    try:
        pins = list(components)
        _require(pins and all(isinstance(p, AttestedInventoryPin) for p in pins),
                 'Explicit externally attested inventory pins required')
        reader = io.Reader()
        reader.bindings_from(implementation_bindings())
        records, coverage, extras, details, contexts = [], [], [], [], {}
        captures, receipts, sample_ids, samples, splits, absent = set(), set(), set(), set(), {}, set()
        inventories = set()
        for pin in sorted(pins, key=lambda p: p.inventory_path):
            path = _path(reader, pin.inventory_path)
            _require(path not in inventories, 'Duplicate inventory component')
            inventories.add(path)
            packet = reader.json(pin.inventory_path, pin.inventory_file_sha256)
            detail = _attestation(reader, pin, packet)
            _component(reader, packet, contexts, captures, receipts, sample_ids, samples, splits, absent)
            # Preserve even unknown metadata; never project/change claims or rows.
            detail['inventory_metadata'] = deepcopy({k: v for k, v in packet.items()
                                                    if k not in ('records', 'scene_contexts')})
            detail['row_ids'] = [r['sample_id'] for r in packet['records']]
            detail['rows_sha256'] = _digest(packet['records'])
            detail['scene_contexts_sha256'] = _digest(packet['scene_contexts'])
            detail['scene_context_ids'] = list(packet['scene_contexts'])
            details.append(detail)
            records.extend(deepcopy(packet['records']))
            coverage.extend(deepcopy(packet['scope']['explicit_receipts']))
            # Do not coalesce conflicting observed-root completeness declarations.
            extras.extend(dict(component_inventory_sha256=packet['sha256'], declaration=deepcopy(r))
                          for r in packet['scope']['extra_observed_roots'])
        omitted = set()
        for declaration in unindexed_roots:
            _require(isinstance(declaration, UnindexedRoot), 'Explicit UnindexedRoot required')
            root = _path(reader, declaration.path)
            _text(declaration.reason)
            _require(root.is_dir() and root not in omitted, 'Missing/duplicate unindexed root')
            _require(all(not (c.is_relative_to(root) or root.is_relative_to(c)) for c in captures),
                     'Unindexed root overlaps included capture')
            omitted.add(root)
            extras.append(dict(component_inventory_sha256=None,
                declaration=dict(path=declaration.path, reason=declaration.reason,
                    status='caller_declared_unindexed_no_rows_added', recursively_scanned=False, complete=False)))
        records.sort(key=lambda r: r['sample_id'])
        reader.finish()
        _require(all(not p.exists() for p in absent), 'Late failure or unbound compact marker')
        result = dict(schema=SCHEMA, state=STATE, records=records, scene_contexts=contexts,
            counts=_counts(records, coverage), components=details,
            scope=dict(explicit_receipts=coverage, extra_observed_roots=extras,
                global_complete=False, receipt_discovery_performed=False, capture_roots_scanned=False,
                audit_execution_independently_verified=False, label_derivation_replayed=False,
                trace_derivation_replayed=False, native_arrays_decoded=False,
                external_audit_attestations_reused=True, per_component_audit_basis_preserved=True,
                physical_contexts_normalized=False, mixed_context_policy_requires_explicit_downstream_adapter=True,
                near_image_comparison_performed=False, fresh_global_pair_coverage_required=True,
                previous_graphs_or_selections_inherited=False, morphology_equivalence_finalized=False),
            provenance=dict(source_bindings=dict(sorted(reader.bindings.items())),
                implementation_bindings=implementation_bindings(),
                verification=reader.diagnostics(),
                trust_basis='explicit_external_attestations_not_self_authenticating_or_reexecuted'),
            **{key: False for key in FALSE_FLAGS})
        result['sha256'] = _digest(result)
        return result
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
        raise ValueError('Malformed/missing attested inventory evidence: '+str(exc)) from exc


def graph_requirement(packet):
    """Return only a fresh graph's required inventory coverage, NOT graph approval."""
    _sealed(packet)
    _require(packet['schema'] == SCHEMA and packet['state'] == STATE, 'Joined inventory required')
    ids = sorted(r['sample_id'] for r in packet['records'])
    _require(len(ids) == len(set(ids)), 'Duplicate joined row ID')
    return dict(inventory_sha256=packet['sha256'], sample_ids=ids,
                total_pairs=len(ids)*(len(ids)-1)//2,
                include_holds_exclusions_and_all_components=True,
                compact_provenance_must_be_preserved=True, complete=True,
                metric='max_full_and_junction_patch_rgb_mae_0to1_v1',
                threshold=0.0391332848665376, dimensions=[1696, 816], patch_radius=64,
                patch_anchor='nominal_10mm_cut_point_not_anatomical_attachment',
                graph_execution_verified=False, training_approved=False)


def write_join(output, components, *, unindexed_roots=()):
    """Exclusive fresh output; builds/checks before publishing, never overwrites."""
    output = Path(output).resolve()
    _require(not output.exists() and output.parent.is_dir(), 'Fresh output in existing parent required')
    packet = join_inventories(components, unindexed_roots=unindexed_roots)
    files = {Path(p).resolve() for p in packet['provenance']['source_bindings']}
    roots = []
    for entry in packet['scope']['explicit_receipts']:
        roots.extend(Path(entry[k]).resolve() for k in ('capture_path', 'storage_root'))
        roots.extend(Path(entry[k]).resolve().parent for k in ('receipt_path', 'plan_path'))
    for component in packet['components']:
        metadata = component['inventory_metadata']
        roots.extend(Path(p).resolve() for p in metadata['provenance']['read_only_source_roots'])
        roots.extend(Path(r['path']).resolve() for r in metadata['scope']['extra_observed_roots'])
    roots.extend(Path(r['declaration']['path']).resolve() for r in packet['scope']['extra_observed_roots'])
    _require(output not in files and all(not output.is_relative_to(r) for r in roots),
             'Join output overlaps protected sources/captures/reviews')
    with output.open('xb') as stream:
        stream.write(json.dumps(packet, indent=2, allow_nan=False).encode()+b'\n')
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', nargs=7, action='append', required=True)
    parser.add_argument('--unindexed-root', nargs=2, action='append', default=[], metavar=('ROOT', 'REASON'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    packet = write_join(args.output, [AttestedInventoryPin(*p) for p in args.component],
                        unindexed_roots=[UnindexedRoot(*p) for p in args.unindexed_root])
    print(json.dumps(dict(sha256=packet['sha256'], counts=packet['counts'],
                         global_complete=False, training_approved=False)))


if __name__ == '__main__':
    main()
