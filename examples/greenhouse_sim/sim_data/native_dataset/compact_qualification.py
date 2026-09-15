"""CPU-only admission of the frozen, queued SAME-CALLBACK storage diagnostic.

The caller must pin the original proof. Its pre-exit marker alone is insufficient:
require the zero-exit queue receipt, both post-exit audits, all paired observations
and unchanged source bindings. This qualifies serialization, never camera poses,
render budgets, biology, visual clarity or training release. No simulator imports.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from ..dataset_review import require
from ..depth_preview import sha256
from .bundle import ARRAYS, REQUIRED, SCHEMA as STORAGE_BACKEND, SampleReader, digest, read_bounded, safe_path

SCHEMA = 'greenhouse.compact_storage_qualification_binding.v1'
ORIGINAL_COLLECTOR_SHA256 = '573054884fa48da986b2926815f6fd225b591500ac247dbd95d0a2fc974fc8b7'
WORKER_SHA256 = 'aa0e434727a104511c0a931e8e4fda871acdbb08949be52c47f36fc3184827ee'
QUEUE_SHA256 = '50b458e946c7f71c356f4dd9378ed6330845892dbc4fa656da90a11ca341fe91'
DIAGNOSTIC_PLAN_SHA256 = '5c0063e4cb72a8730ad8949abbfbfac5f3ce90cfd964b994f236e5bb86413c23'
_HERE = Path(__file__).resolve().parent
_SIM = _HERE.parent
_STORAGE_PATHS = (_HERE/'capture_storage.py', _HERE/'bundle.py', _SIM/'native_lossless_codec.py',
                  _HERE/'dual_storage.py', _SIM/'capture_contract.py', _SIM/'capture_visibility.py')
_REVIEW_PATHS = (_HERE/'audit.py', _HERE/'bundle.py')
_CODE_PATHS = (*_STORAGE_PATHS, *_REVIEW_PATHS, _HERE/'__init__.py', Path(__file__),
               _SIM/'native_generated_views.py', _SIM/'dataset_review.py', _SIM/'depth_preview.py')
_LOADED = {str(p.resolve()): sha256(p) for p in _CODE_PATHS}


def implementation_bindings():
    """Additional storage/proof code; native plan bindings remain mandatory."""
    require(_LOADED[str(_SIM/'native_generated_views.py')] == ORIGINAL_COLLECTOR_SHA256,
            'Frozen original collector changed')
    for path, expected in _LOADED.items():
        require(sha256(Path(path)) == expected, 'Compact implementation changed: ' + path)
    return dict(_LOADED)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate qualification JSON key')
        result[key] = value
    return result


class _Files:
    def __init__(self):
        self.bindings, self.paths = {}, {}

    def bind(self, path, expected=None):
        original = str(Path(path).absolute())
        resolved = Path(original).resolve(strict=True)
        require(original not in self.paths or self.paths[original] == str(resolved), 'Retargeted proof path')
        self.paths[original] = str(resolved)
        if expected is not None:
            require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid SHA256 pin')
        actual = sha256(resolved)
        require(expected is None or actual == expected, 'Qualification hash mismatch: ' + original)
        key = str(resolved)
        require(key not in self.bindings or self.bindings[key] == actual, 'Changed/conflicting proof binding')
        self.bindings[key] = actual
        return resolved

    def document(self, path, expected=None):
        path = self.bind(path, expected)
        raw = read_bounded(path)
        require(digest(raw) == self.bindings[str(path)], 'Qualification metadata changed')
        def invalid(value):
            raise ValueError('Nonfinite qualification JSON: ' + value)
        result = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid)
        require(isinstance(result, dict), 'Qualification JSON object required')
        return result

    def mapping(self, bindings):
        require(isinstance(bindings, dict) and bindings, 'Missing qualification source bindings')
        for path, expected in bindings.items():
            self.bind(path, expected)

    def finish(self):
        _verify_files(self.bindings, self.paths)


def _verify_files(bindings, paths):
    require(bool(bindings) and bool(paths), 'Missing checked qualification bindings')
    for path, expected in bindings.items():
        require(sha256(Path(path)) == expected, 'Changed qualification binding: ' + path)
    for original, resolved in paths.items():
        require(str(Path(original).resolve(strict=True)) == resolved, 'Retargeted qualification path')


def _no_failure(root):
    require(not any((root/name).exists() for name in
                    ('failure.json', 'capture/failure.json', 'raw_same_callback/failure.json')),
            'Failed storage diagnostic')


def _integer(value, expected=None):
    require(type(value) is int and value >= 0 and (expected is None or value == expected),
            'Invalid qualification count')
    return value


def _sample(files, directory, row, *, compact):
    reader = SampleReader(directory, expected_bindings={'sample.json': row['sample_sha256'],
        'supervision/label.json': row['label_sha256']}, expected_json=(
            {'supervision/query_trace.json': row['query_trace']} if row.get('query_trace') is not None else {}))
    require((reader.manifest is not None) is compact, 'Wrong qualification storage format')
    reader.verify_all()
    if compact:
        require(files.document(directory/'bundle.json') == reader.manifest, 'Compact manifest changed during check')
        for entry in reader.manifest['files'].values():
            files.bind(safe_path(directory, entry['stored_path']), entry['stored_sha256'])
    else:
        files.bind(directory/'sample.json', row['sample_sha256'])
        files.bind(directory/'supervision/label.json', row['label_sha256'])
        for name, entry in reader.metadata['files'].items():
            files.bind(safe_path(directory, name), entry['sha256'])
        if row.get('query_trace') is not None:
            require(files.document(directory/'supervision/query_trace.json') == row['query_trace'],
                    'Changed raw query trace')
    require(reader.metadata['calibration']['resolution'] == [1696, 816], 'Native1696 qualification required')
    return reader


def _audits(files, root, folder, result, plan_hash, request_hash, proof):
    prefix = 'compact' if folder == 'capture' else 'raw'
    initial = files.document(root/(prefix+'_audit.json'), proof[prefix+'_audit_sha256'])
    post = files.document(root/(folder+'_postexit_audit.json'))
    require(initial == post, 'Post-exit audit differs from bound original replay')
    require(post['state'] == 'completed_automatic_annotation_replay'
            and post['training_approved'] is False and post['source_assets_unchanged'] is True
            and post['original_reviews_modified'] is False, 'Incomplete post-exit audit')
    require(post['capture'] == str(root/folder) and post['plan_sha256'] == plan_hash
            and post['request_sha256'] == request_hash
            and post['result_sha256'] == proof['result_sha256' if folder == 'capture' else 'raw_result_sha256'],
            'Post-exit audit source mismatch')
    expected_code = {str(p): _LOADED[str(p)] for p in _REVIEW_PATHS}
    require(post['review_code_bindings'] == expected_code, 'Wrong audit implementation')
    files.mapping(post['review_code_bindings'])
    captured = {r['candidate_id']: r for r in result['records'] if r['state'] == 'native_captured_pending_review'}
    require(len(post['records']) == len(captured), 'Partial post-exit audit')
    seen = set()
    for record in post['records']:
        name = Path(record['sample']).name
        require(name in captured and name not in seen and record['sample'] == str(root/folder/name),
                'Duplicate/unrelated post-exit sample')
        seen.add(name)
        row = captured[name]
        for field in ('sample_sha256', 'label_sha256', 'target_id'):
            require(record[field] == row[field], 'Post-exit sample binding mismatch')
        decision = ('accept_strict_automatic_annotation_candidate' if row['automatic_annotation_eligible'] else
                    'hold_visual_clarity' if row['eligible_annotation'] else 'exclude_geometry_or_visibility')
        require(record['decision'] == decision and record['reason'] == row['label_reason'],
                'Post-exit annotation decision mismatch')
        pair = next(p for p in result['same_callback_storage'] if Path(p['compact_directory']).name == name)
        require(record['rgb_sha256'] == pair['exact_observations']['inputs/rgb.png']['sha256'],
                'Post-exit RGB binding mismatch')
        for field in ('label_replayed_exact', 'native_callback_hashes_verified', 'source_and_file_hashes_verified'):
            require(record[field] is True, 'Incomplete post-exit evidence')
        require(record['trace_replayed_exact'] is (row.get('query_trace') is not None), 'Missing trace replay')
        for field in ('training_approved', 'source_cap_reset', 'physical_execution_approved', 'visual_review_performed'):
            require(record[field] is False, 'Unexpected audit approval')
    require(post['counts'] == dict(Counter(r['decision'] for r in post['records'])), 'Audit counts mismatch')
    return post


def check_qualification(path, *, expected_sha256):
    """Return a new full-file binding receipt, or fail before any output/native work.

    Caller pins same_callback_qualification.json. The exact queued producer and
    plan versions are pinned here; replacing them requires a reviewed code change.
    Rehashes source files and all paired physical payloads; not a stat-only cache.
    """
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256),
            'Caller storage qualification SHA256 required')
    path = Path(path)
    require(path.name == 'same_callback_qualification.json', 'Original same-callback proof required')
    files = _Files()
    proof = files.document(path, expected_sha256)
    root = path.resolve(strict=True).parent
    _no_failure(root)
    require(proof['state'] == 'native_same_callback_storage_and_annotation_replay_passed', 'Failed qualification')
    count = _integer(proof['captured_frames'])
    require(0 < count <= 36, 'Empty/unbounded qualification')
    _integer(proof['exact_native_npy_files'], 3*count)
    for field in ('full_prim_identity_tables_equal', 'caller_buffers_unchanged'):
        require(proof[field] is True, 'Incomplete same-callback proof')
    require(proof['training_approved'] is False and proof['source_removed'] is False, 'Invalid proof scope')
    _integer(proof['training_diversity_increment'], 0)
    require(proof['worker_sha256'] == WORKER_SHA256, 'Wrong diagnostic worker')
    code = implementation_bindings()
    files.mapping(code)
    expected_storage = {str(p): _LOADED[str(p)] for p in _STORAGE_PATHS}
    require(proof['implementation_bindings'] == expected_storage, 'Wrong qualified storage implementation')

    queue_paths = sorted(root.glob('queue_*.json'))
    require(queue_paths, 'Missing post-exit queue completion')
    queue = files.document(queue_paths[-1])
    require(queue['state'] == 'native_diagnostic_complete_no_default_change', 'Diagnostic queue not complete')
    _integer(queue['exit_code'], 0)
    require(queue['training_approved'] is False, 'Invalid queue approval')
    _integer(queue['training_diversity_increment'], 0)
    queue_bindings = queue['bindings']
    files.mapping(queue_bindings)
    for filename, pin in (('native_same_callback_worker_20260915_v1.py', WORKER_SHA256),
                           ('queue_same_callback_native_20260916_v1.py', QUEUE_SHA256)):
        matches = [p for p in queue_bindings if Path(p).name == filename]
        require(len(matches) == 1 and queue_bindings[matches[0]] == pin, 'Wrong queue source binding')
    for p in (_HERE/'dual_storage.py', _HERE/'capture_storage.py', *_REVIEW_PATHS):
        require(queue_bindings.get(str(p)) == code[str(p)], 'Missing queued storage/review code')

    compact = files.document(root/'capture/result.json', proof['result_sha256'])
    raw = files.document(root/'raw_same_callback/result.json', proof['raw_result_sha256'])
    request = files.document(root/'capture/request.json')
    raw_request = files.document(root/'raw_same_callback/request.json')
    expected_request = dict(request, compact_native_storage=False, same_callback_compact_capture=str(root/'capture'))
    require(raw_request == expected_request, 'Raw/compact request mismatch')
    require(request['same_callback_raw_compact_diagnostic'] is True and request['compact_native_storage'] is True
            and request['training_started'] is False and request['render_profile_experiment'] is False,
            'Not a same-callback native diagnostic')
    _integer(request['training_diversity_increment'], 0)
    require(request['plan_sha256'] == DIAGNOSTIC_PLAN_SHA256
            and queue_bindings.get(request['plan_path']) == DIAGNOSTIC_PLAN_SHA256, 'Wrong queued diagnostic plan')
    plan = files.document(request['plan_path'], request['plan_sha256'])
    require(plan['split'] == 'train' and plan['resolution'] == [1696, 816]
            and plan['maximum_native_frames'] == 36 and len(plan['target_cases']) == 4, 'Wrong diagnostic plan bounds')
    for field in ('training_approved', 'source_cap_reset', 'physical_motion_commanded', 'hidden_cut_coordinates_executable'):
        require(plan[field] is False, 'Invalid diagnostic plan scope')
    for field in ('source_bindings', 'implementation_bindings', 'prerequisite_bindings'):
        files.mapping(plan[field])
    for document in (request, compact, raw):
        require(document['experiment_script_sha256'] == WORKER_SHA256
                and document['storage_implementation_bindings'] == expected_storage, 'Changed diagnostic implementation')
    for result, is_compact in ((compact, True), (raw, False)):
        require(result['state'] == 'native_generated_multiview_pilot_complete_pending_review'
                and result['training_approved'] is False and result['source_assets_unchanged'] is True
                and result['compact_native_storage'] is is_compact, 'Incomplete native diagnostic result')
        _integer(result['captured_frames'], count)
    planned = {s['candidate_id']: (case['target_id'], s) for case in plan['target_cases'] for s in case['views']}
    require(len(planned) == 36 and len(compact['records']) == 36, 'Incomplete diagnostic decisions')
    seen = set()
    for row in compact['records']:
        name = row['candidate_id']
        require(name in planned and name not in seen and (row['target_id'], row['requested_spec']) == planned[name],
                'Repeated/unplanned diagnostic record')
        require(row['state'] in ('native_captured_pending_review', 'rejected_pose', 'rejected_possible_geometry_overlap'),
                'Invalid diagnostic decision')
        safe_path(root/'capture', name)
        require('/' not in name, 'Nested diagnostic sample')
        seen.add(name)
    captured = {r['candidate_id']: r for r in compact['records'] if r['state'] == 'native_captured_pending_review'}
    require(len(captured) == count and len(compact['same_callback_storage']) == count, 'Partial callback evidence')
    expected_raw = deepcopy(compact)
    expected_raw['compact_native_storage'] = False
    pairs = {}
    for pair in compact['same_callback_storage']:
        name = Path(pair['compact_directory']).name
        require(name in captured and name not in pairs, 'Duplicate/unrelated callback pair')
        require(pair['compact_directory'] == str(root/'capture'/name)
                and pair['raw_directory'] == str(root/'raw_same_callback'/name), 'Wrong callback directories')
        pairs[name] = pair
        require(pair == files.document(root/'raw_same_callback'/name/'same_callback_storage.json'), 'Changed pair receipt')
        require(pair['schema'] == 'greenhouse.same_callback_dual_storage.v1'
                and pair['state'] == 'same_callback_serialization_exact_not_native_qualification_by_itself', 'Wrong pair state')
        for field in ('complete_prim_identity_table_equal', 'rgb_png_bytes_exact', 'annotation_and_trace_equal',
                      'metadata_equal_except_review_files_and_json_newlines', 'caller_unchanged'):
            require(pair[field] is True, 'Failed callback comparison')
        for field in ('source_removed', 'depth_recomputed', 'training_approved'):
            require(pair[field] is False, 'Invalid callback scope')
        _integer(pair['training_diversity_increment'], 0)
        _integer(pair['native_npy_bytes_exact'], len(ARRAYS))
        require(set(pair['exact_observations']) == REQUIRED-{'supervision/identities.json'}, 'Partial native array proof')
        row = captured[name]
        require(pair['compact_result'] == {k: row[k] for k in ('sample_sha256', 'label_sha256', 'query_trace')},
                'Callback result binding mismatch')
        left = _sample(files, root/'raw_same_callback'/name, pair['raw_result'], compact=False)
        right = _sample(files, root/'capture'/name, pair['compact_result'], compact=True)
        for logical, evidence in pair['exact_observations'].items():
            a, b = left.read(logical), right.read(logical)
            require(a == b and evidence == dict(sha256=digest(a), logical_bytes=len(a)), 'Different callback observation')
        for logical in ('supervision/identities.json', 'supervision/label.json'):
            require(left.json(logical) == right.json(logical), 'Different callback identity/annotation')
        require(pair['raw_result']['query_trace'] == pair['compact_result']['query_trace'], 'Different callback trace')
        expected_meta = deepcopy(left.metadata)
        expected_meta['files'] = {k: v for k, v in expected_meta['files'].items() if k in REQUIRED}
        expected_meta['files']['supervision/identities.json']['sha256'] = digest(right.read('supervision/identities.json'))
        require(expected_meta == right.metadata, 'Different callback metadata')
    for row in expected_raw['records']:
        if row['state'] == 'native_captured_pending_review':
            row.update(pairs[row['candidate_id']]['raw_result'])
    require(raw == expected_raw, 'Raw result not the same callback decisions')
    audits = [_audits(files, root, folder, result, request['plan_sha256'],
                     files.bindings[str(root/folder/'request.json')], proof)
              for folder, result in (('capture', compact), ('raw_same_callback', raw))]
    require(audits[0]['counts'] == audits[1]['counts'], 'Different post-exit annotation counts')
    files.finish()
    _no_failure(root)
    require(sorted(root.glob('queue_*.json')) == queue_paths, 'Queue changed during qualification check')
    return dict(schema=SCHEMA, storage_backend=STORAGE_BACKEND, path=str(path.resolve()),
        sha256=expected_sha256, captured_frames=count, exact_native_npy_files=3*count,
        bindings=files.bindings, original_paths=files.paths, queue_completion=str(queue_paths[-1]),
        training_approved=False, training_diversity_increment=0)


def verify_checked_qualification(receipt):
    """Rehash a receipt returned in this process before publishing a capture result."""
    require(receipt['schema'] == SCHEMA and receipt['storage_backend'] == STORAGE_BACKEND, 'Wrong qualification receipt')
    _verify_files(receipt['bindings'], receipt['original_paths'])
    root = Path(receipt['path']).parent
    _no_failure(root)
    require(str(sorted(root.glob('queue_*.json'))[-1]) == receipt['queue_completion'], 'Queue changed after qualification')
    require(sha256(Path(receipt['path'])) == receipt['sha256'], 'Qualification pin changed')
    return True


def qualify_storage(proof_path, *, expected_sha256=None):
    """Controller API: validate, then return all full SHA256 bindings to freeze.

    If no previous pin is supplied, establish it from the current proof and
    return it in this map. The controller must persist/pin that map and pass its
    proof entry as --storage-qualification-sha256; the worker never self-pins.
    """
    pin = sha256(Path(proof_path)) if expected_sha256 is None else expected_sha256
    return dict(check_qualification(proof_path, expected_sha256=pin)['bindings'])
