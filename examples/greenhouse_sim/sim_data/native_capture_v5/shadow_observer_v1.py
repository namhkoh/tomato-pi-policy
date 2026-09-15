"""Write-only, stdlib-only pre-render shadow journal. Never a capture decision.

The native owner publishes events; only a separate CPU process AFTER its owned
exit may predict. This module imports neither USD, the predictor nor a collector.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

PREFIX = 'greenhouse.shadow_observer.v1.'
WORKER_MODULE = 'sim_data.native_generated_reference.worker_v1'
MAX_BYTES = 8 * 1024 * 1024
MAX_CANDIDATES = 1024  # serialization/resource bound, not a geometry/training cap
_HERE = Path(__file__).resolve().parent
FROZEN = {
    str(_HERE / 'visibility_shadow.py'): '1887e36d7a24eb7f6cadb82df120e2be271524948f8412bce5a52c0eb6e4bfb8',
    str(_HERE / 'test_visibility_shadow.py'): '113143d279d49dc69f850ed52d72e87b6c71bcf4a18129533b31184a1fb68172',
}
FLAGS = dict(skip_adopted=False, ranking_adopted=False, training_approved=False)


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def decode(raw):
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, 'Bounded immutable JSON bytes required')
    value = json.loads(raw, object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
    canonical(value)
    return value


def keys(value, fields):
    require(type(value) is dict and set(value) == set(fields.split()), 'Unknown or missing schema fields')


def flags(value):
    require(type(value) is dict and canonical(value) == canonical(FLAGS), 'Exact false observer flags required')


def identifier(value):
    require(type(value) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', value), 'Invalid identifier')


def hash_value(value):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value), 'Explicit lowercase SHA256 required')


def pin_shape(value):
    keys(value, 'path sha256'); hash_value(value['sha256'])
    require(type(value['path']) is str and Path(value['path']).is_absolute(), 'Absolute pinned path required')


def file_pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def read_pin(pin, ledger=None):
    pin_shape(pin)
    path = Path(pin['path']).resolve(strict=True)
    require(path.is_file() and path.stat().st_size <= MAX_BYTES, 'Missing/oversized JSON')
    raw = path.read_bytes()
    require(digest(raw) == pin['sha256'], 'Pinned JSON bytes changed: ' + str(path))
    if ledger is not None: merge(ledger, {str(path): pin['sha256']})
    return decode(raw)


def bindings_shape(value):
    require(type(value) is dict and value, 'Nonempty explicit bindings required')
    for path, h in value.items():
        require(Path(path).is_absolute() and str(Path(path).resolve()) == path, 'Noncanonical source path')
        hash_value(h)


def merge(target, source):
    for path, h in source.items():
        path = str(Path(path).resolve())
        require(path not in target or target[path] == h, 'Conflicting source hash')
        target[path] = h


def verify(bindings):
    for path, h in bindings.items():
        require(sha(path) == h, 'Full byte hash changed: ' + path)


def implementation_bindings():
    """Hash the static repository-local Python closure without importing it."""
    root = _HERE.parent
    pending = [_HERE / 'shadow_observer_v1.py', _HERE / 'shadow_observer_cpu_v1.py']
    paths = set()
    while pending:
        path = pending.pop().resolve()
        if path in paths: continue
        require(path.is_file(), 'Missing observer implementation')
        paths.add(path)
        package = ['sim_data', *path.relative_to(root).parts[:-1]]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = []
            if isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                base += node.module.split('.') if node.module else []
                names = [base, *[base+[a.name] for a in node.names]]
            elif isinstance(node, ast.Import): names = [a.name.split('.') for a in node.names]
            for name in names:
                if not name or name[0] != 'sim_data': continue
                candidate = root.joinpath(*name[1:])
                for local in (candidate.with_suffix('.py'), candidate / '__init__.py'):
                    if local.is_file(): pending.append(local)
    verify(FROZEN)
    return {**{str(p): sha(p) for p in sorted(paths)}, **FROZEN}


def _numbers(value, count):
    require(type(value) is list and len(value) == count
        and all(type(x) in (float, int) and math.isfinite(x) for x in value), 'Invalid numeric vector')


def matrix(value, size):
    require(type(value) is list and len(value) == size, 'Invalid matrix')
    for row in value: _numbers(row, size)


def rigid(value):
    matrix(value, 4)
    require(all(abs(value[i][3] - (i == 3)) < 1e-9 for i in range(4)), 'Non-affine row transform')
    require(all(abs(sum(value[i][k]*value[j][k] for k in range(3))-(i == j)) < 1e-7
                for i in range(3) for j in range(3)), 'Non-rigid transform')


def calibration(value):
    required = {'resolution', 'crop_resize', 'clipping_range_m', 'intrinsics', 'camera_to_world_usd_row_vectors'}
    optional = {'schema_version', 'camera_path', 'optical_frame', 'usd_camera_frame', 'pixel_convention',
        'focal_length_mm', 'apertures_mm', 'aperture_offsets_mm', 'depth_convention', 'sensor_model'}
    require(type(value) is dict and required <= value.keys() <= required | optional, 'Unexpected calibration fields')
    require(value['resolution'] == [1696, 816] and value['crop_resize'] is None, 'Native uncropped calibration required')
    matrix(value['intrinsics'], 3); rigid(value['camera_to_world_usd_row_vectors'])
    _numbers(value['clipping_range_m'], 2)
    require(0 < value['clipping_range_m'][0] < value['clipping_range_m'][1], 'Invalid clipping range')
    for k in optional & value.keys():
        if k in ('apertures_mm', 'aperture_offsets_mm'): _numbers(value[k], 2)
        elif k == 'focal_length_mm': _numbers([value[k]], 1)
        else: require(type(value[k]) is str and len(value[k]) <= 256, 'Invalid calibration descriptor')


def validate_spec(value):
    keys(value, 'schema plan bank clear_plan anchor_entry_id producer_module producer_bindings implementation_bindings candidates flags')
    require(value['schema'] == PREFIX+'spec' and value['producer_module'] == WORKER_MODULE
        and value['flags'] == FLAGS, 'Unsupported observer/producer policy')
    flags(value['flags'])
    for k in ('plan', 'bank', 'clear_plan'): pin_shape(value[k])
    identifier(value['anchor_entry_id'])
    for k in ('producer_bindings', 'implementation_bindings'): bindings_shape(value[k])
    require(all(value['implementation_bindings'].get(p) == h for p, h in FROZEN.items()), 'Wrong frozen algorithm')
    require(any(Path(p).as_posix().endswith('/native_generated_reference/worker_v1.py')
                for p in value['producer_bindings']), 'Named collector source pin required')
    candidates = value['candidates']
    require(type(candidates) is list and 0 < len(candidates) <= MAX_CANDIDATES, 'Bounded explicit candidate inventory required')
    seen = set()
    for c in candidates:
        keys(c, 'candidate_id mode target_id component_id')
        identifier(c['candidate_id']); identifier(c['component_id'])
        require(c['candidate_id'] not in seen and c['mode'] in ('original_control', 'generated_variant'), 'Duplicate/unknown candidate')
        require(type(c['target_id']) is str and c['target_id'].count('/') == 1
            and c['target_id'].split('/')[1] == c['component_id'], 'Target/component mismatch')
        identifier(c['target_id'].split('/')[0]); seen.add(c['candidate_id'])
    return value


def validate_context(value):
    keys(value, 'kind context_id variant_directory geometry_bindings plant_to_world_usd_row_vectors scene_evidence_sha256')
    require(value['kind'] == 'generated_context', 'Only generated plant context is supported')
    identifier(value['context_id']); bindings_shape(value['geometry_bindings'])
    require(type(value['variant_directory']) is str and Path(value['variant_directory']).is_absolute(), 'Absolute variant required')
    directory = Path(value['variant_directory']).resolve()
    require(str(directory / 'manifest.json') in value['geometry_bindings'], 'Generated manifest must be pinned')
    rigid(value['plant_to_world_usd_row_vectors']); hash_value(value['scene_evidence_sha256'])
    return value


def validate_candidate(value, candidate, contexts):
    kind = value.get('kind') if type(value) is dict else None
    if kind == 'pre_render':
        keys(value, 'kind candidate_id context_id calibration robot_snapshot_sha256 writer_request_index_before')
        calibration(value['calibration']); hash_value(value['robot_snapshot_sha256'])
        require(type(value['writer_request_index_before']) is int and value['writer_request_index_before'] >= 0, 'Native request counter required')
        if candidate['mode'] == 'original_control':
            require(value['context_id'] is None, 'Original control must not use generated geometry')
        else:
            require(value['context_id'] in contexts, 'Missing sealed generated context')
            context = contexts[value['context_id']]
            require(Path(context['variant_directory']).name == candidate['target_id'].split('/')[0], 'Foreign generated context')
    else:
        keys(value, 'kind candidate_id code')
        require(kind in ('geometry_hold', 'observer_error'), 'Unexpected candidate event')
        identifier(value['code'])
    require(value['candidate_id'] == candidate['candidate_id'], 'Candidate inventory/order changed')
    return value


def write_bytes_new(path, raw):
    """Exclusive publication + fsync; caller returns only after the seal is durable."""
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def write_json_new(path, value):
    write_bytes_new(path, canonical(value))


class EventWriter:
    """One synchronous journal in the single native owner; every public method returns None.

    Validate/create at setup. Per-event validation/I/O failures become sanitized
    observer errors, not capture exceptions. Persistent disk failure is not
    magically recoverable: owner must record missing manifest in its exit receipt.
    """
    def __init__(self, output, spec_bytes):
        self._spec = validate_spec(decode(spec_bytes))
        self._root = Path(output).resolve()
        require(not self._root.exists() and self._root != Path(self._root.anchor), 'New non-root journal required')
        for p in [*(self._spec[k]['path'] for k in ('plan', 'bank', 'clear_plan')),
                  *self._spec['producer_bindings'], *self._spec['implementation_bindings']]:
            require(not Path(p).resolve().is_relative_to(self._root), 'Journal would contain protected input')
        self._root.mkdir(parents=True, exist_ok=False)
        write_json_new(self._root/'spec.json', self._spec)
        self._contexts, self._context_records, self._records, self._faults = {}, [], [], []
        self._previous, self._sequence, self._closed = None, 0, False

    def _publish(self, value):
        sequence = self._sequence; self._sequence += 1
        event = dict(schema=PREFIX+'event', sequence=sequence, previous_seal_sha256=self._previous,
                     monotonic_ns=time.monotonic_ns(), payload=value)
        payload = self._root / ('event_%06d.json' % sequence)
        seal = self._root / ('seal_%06d.json' % sequence)
        write_json_new(payload, event)
        write_json_new(seal, dict(schema=PREFIX+'event_seal', event=file_pin(payload)))
        result = file_pin(seal); self._previous = result['sha256']
        return result

    def bind_context(self, context_bytes) -> None:
        try:
            require(not self._closed, 'Closed journal')
            context = validate_context(decode(context_bytes))
            require(context['context_id'] not in self._contexts, 'Repeated context')
            seal = self._publish(context)
            self._contexts[context['context_id']] = context
            self._context_records.append(seal)
        except (ValueError, TypeError, OSError, KeyError, RecursionError): self._faults.append('context_publication_failed')

    def record_candidate(self, candidate_bytes) -> None:
        if self._closed or len(self._records) >= len(self._spec['candidates']):
            self._faults.append('extra_or_late_candidate'); return None
        candidate = self._spec['candidates'][len(self._records)]
        try:
            value = validate_candidate(decode(candidate_bytes), candidate, self._contexts)
        except (ValueError, TypeError, KeyError, RecursionError):
            value = dict(kind='observer_error', candidate_id=candidate['candidate_id'], code='invalid_event')
        try:
            seal = self._publish(value)
            self._records.append(dict(candidate_id=candidate['candidate_id'], seal=seal, error=None))
        except OSError:
            self._records.append(dict(candidate_id=candidate['candidate_id'], seal=None, error='event_io_error'))

    def seal(self) -> None:
        if self._closed: return None
        self._closed = True
        for c in self._spec['candidates'][len(self._records):]:
            self._records.append(dict(candidate_id=c['candidate_id'], seal=None, error='missing_event'))
        try:
            manifest = dict(schema=PREFIX+'manifest', spec=file_pin(self._root/'spec.json'),
                contexts=self._context_records, candidates=self._records, faults=self._faults,
                final_seal_sha256=self._previous, flags=FLAGS)
            write_json_new(self._root/'manifest.json', manifest)
        except OSError: self._faults.append('manifest_io_error')


def read_journal(manifest_pin, ledger):
    manifest = read_pin(manifest_pin, ledger)
    keys(manifest, 'schema spec contexts candidates faults final_seal_sha256 flags')
    require(manifest['schema'] == PREFIX+'manifest' and manifest['flags'] == FLAGS, 'Wrong journal manifest')
    flags(manifest['flags'])
    require(type(manifest['faults']) is list and all(type(x) is str and len(x) < 128 for x in manifest['faults']), 'Invalid journal faults')
    root = Path(manifest_pin['path']).resolve().parent
    require(Path(manifest['spec']['path']).resolve() == root/'spec.json', 'Foreign spec path')
    spec = validate_spec(read_pin(manifest['spec'], ledger))
    require([r['candidate_id'] for r in manifest['candidates']] == [c['candidate_id'] for c in spec['candidates']], 'Candidate omission/order change')
    events = []
    for pin in [*manifest['contexts'], *[r['seal'] for r in manifest['candidates'] if r['seal']]]:
        require(Path(pin['path']).resolve().parent == root, 'Seal escaped journal')
        seal = read_pin(pin, ledger); keys(seal, 'schema event')
        require(seal['schema'] == PREFIX+'event_seal' and Path(seal['event']['path']).resolve().parent == root, 'Foreign event seal')
        event = read_pin(seal['event'], ledger)
        keys(event, 'schema sequence previous_seal_sha256 monotonic_ns payload')
        require(event['schema'] == PREFIX+'event' and type(event['sequence']) is int
                and type(event['monotonic_ns']) is int, 'Invalid event sequence')
        events.append((event, pin))
    events.sort(key=lambda pair: pair[0]['sequence'])
    previous, last_sequence, last_ns, contexts, candidates = None, -1, -1, {}, {}
    index = {c['candidate_id']: i for i, c in enumerate(spec['candidates'])}
    last_candidate = -1
    for event, pin in events:
        require(event['sequence'] > last_sequence and event['monotonic_ns'] > last_ns
                and event['previous_seal_sha256'] == previous, 'Broken seal chain/order')
        payload = event['payload']
        if payload.get('kind') == 'generated_context':
            validate_context(payload)
            require(payload['context_id'] not in contexts, 'Duplicate context')
            contexts[payload['context_id']] = payload
        else:
            candidate_index = index[payload['candidate_id']]
            require(candidate_index > last_candidate and all(manifest['candidates'][i]['seal'] is None
                    for i in range(last_candidate+1, candidate_index)), 'Candidate event order/omission')
            candidate = spec['candidates'][candidate_index]
            validate_candidate(payload, candidate, contexts)
            candidates[candidate['candidate_id']] = dict(payload=payload, seal=pin)
            last_candidate = candidate_index
        previous, last_sequence, last_ns = pin['sha256'], event['sequence'], event['monotonic_ns']
    require(previous == manifest['final_seal_sha256'], 'Wrong terminal seal')
    # Missing/I/O-failed candidates are explicit manifest rows, never silently dropped.
    for row in manifest['candidates']:
        keys(row, 'candidate_id seal error')
        if row['seal'] is None:
            require(row['error'] in ('missing_event', 'event_io_error'), 'Unknown missing-event state')
            candidates[row['candidate_id']] = dict(payload=dict(kind='observer_error',
                candidate_id=row['candidate_id'], code=row['error']), seal=None)
        else:
            require(row['error'] is None and candidates[row['candidate_id']]['seal'] == row['seal'], 'Candidate seal mismatch')
    return spec, contexts, candidates, manifest
