"""Create-only portable native RGB releases from EXTERNALLY admitted selection.

No discovery, raw conversion, admission/calibration, training or model downloads.
The existing native contract's training-release hold is not removed: this new
packaging schema is NOT a qualification of the legacy exporter/H200 pipeline.

API: FilePin(path, sha256); build(destination, *, selection, admission,
frozen_splits, sources); validate(root, *, manifest_sha256); model_messages
(root, record, *, supervised=False, crop=True). sources maps sample_id to an
explicit absolute compact root. FilePin hashes are SHA256 of exact file bytes.
validate requires an OUT-OF-BAND manifest pin, not a self-asserted checksum.

Input JSON contracts (unknown fields rejected except in opaque evidence):
  selection: {schema: SELECTION_SCHEMA, rows: [ROW_FIELDS dictionaries]}
  frozen_splits: the complete original {donor_family: train|validation|test}
                mapping, exactly the original 24 seed names and reservations
                pinned by FROZEN_SPLITS_SHA256, not merely any 16/4/4 partition.
  admission: {schema: ADMISSION_SCHEMA, state: 'final_selection_admitted',
    authority: nonempty external identity, selection_sha256, frozen_splits_sha256,
    counts: {train: >=20000, validation: >0, test: >0},
    gates: {each name in GATES: {passed: true, evidence_sha256: SHA256}}}
Counts are exact frozen selection counts, not targets or defaults. These gates
are externally asserted, hash-bound references, NOT independently re-audited or
cryptographically signed here. Do not synthesize this handoff from candidates.
No claim that row counts measure biological independence. No action approval.

Selection rows bind global sample_id AND capture_sample_id (the saved local
name), sample/label/bundle, encoded AND decoded RGB, calibration, target and
original donor/target identities. Calibration uses json_sha256, i.e.
sorted compact UTF-8 JSON with finite numbers only. All source/native sidecar
bytes stay unchanged; RGB must already be identity-encoded PNG. Query crops are
768x768, never resized. Labels, optical-Z depth, validity, identities, masks,
calibration and robot metadata are sidecars only, never user model inputs.

Output: provenance/{selection,admission,frozen_splits}.json (exact input bytes),
samples/00000000/{bundle.json,payload/...} (exact selected compact bytes),
crops/00000000.png, splits/{train,validation,test}.jsonl, manifest.json (last).
Indices, not source names, form paths. No source path is needed after relocation;
absolute strings inside original metadata remain inert provenance. No code is
automatically bundled: validation requires this module and its pinned sim_data
dependencies plus NumPy/Pillow/SciPy/zstandard, not Isaac, USD or model weights.
Python >=3.11; each metadata/index file is bounded to 64 MiB by SampleReader's
existing bounded-read helper. This is a directory package, not an archive/code
kit. Data-byte validation does not independently replay projection or review.
Interrupted builds retain an incomplete new directory, never delete originals.
Build validates all inputs before creating destination, checks copied bytes and
rechecks sources at publication; callers must keep inputs quiescent (no locking).
Only synthetic temporary releases belong in tests. No public small-release mode.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import stat

import numpy as np
from PIL import Image

from . import bundle
from .admission import decoded_rgb_digest
from .. import native_clear_contract as contract


SCHEMA = 'greenhouse.portable_native_clear_rgb_release.v1'
SELECTION_SCHEMA = 'greenhouse.native_clear_final_selection.v1'
ADMISSION_SCHEMA = 'greenhouse.native_clear_external_admission.v1'
SPLITS = ('train', 'validation', 'test')
# Portable canonical-JSON baseline, verified from native_original_capture/
# contracts.py FROZEN_SPLITS (source SHA256 6e41ad1a052d74a2fbdda4a4af09639d1da2abd641d98c652451d5111a143388).
# No import or runtime dependency on native-original collection code.
# TRAIN: 101,103,11,17,19,23,41,43,47,53,67,71,73,7,83,89;
# validation: 13,29,37,97; test: 31,59,61,79. Names are seed{n}_full.
FROZEN_SPLITS_SHA256 = '65b06cdded97a06c163aedc954656629dd548749b6cc6fb9f383e40a061c9e1c'
GATES = ('native_capture_audit', 'individual_annotation_review',
         'geometry_calibration', 'near_image_calibration', 'global_inventory_and_duplicates')
ROW_FIELDS = {'sample_id', 'capture_sample_id', 'target_id', 'source_family', 'source_target', 'split',
              'bundle_sha256', 'sample_sha256', 'label_sha256', 'encoded_rgb_sha256',
              'decoded_rgb_sha256', 'calibration_sha256'}
POLICY = dict(resolution=[1696, 816], crop_size=[768, 768], crop_resize=False,
              coordinates='normalized_1000_original_full_frame_2dp',
              model_input='RGB_full_and_optional_query_crop_plus_distal_query',
              sidecars='unchanged_native_optical_Z_validity_calibration_and_supervision',
              admission='externally_attested_not_recomputed_or_authenticated_here',
              training_qualification_performed=False, calibration_performed=False,
              metric_action_approved=False, biological_independence_claimed=False)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'Explicit SHA256 required')
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def json_sha256(value):
    """Canonical JSON pin for calibration only; file pins hash original bytes."""
    return bundle.digest(_json(value))


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _json(value)  # Also reject exponent overflow to infinity.
    return value


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


_CODE_ROOT = Path(__file__).resolve().parent.parent
_CODE_NAMES = ('native_dataset/native_clear_export.py', 'native_dataset/bundle.py',
               'native_dataset/admission.py', 'native_lossless_codec.py',
               'native_clear_contract.py', 'capture_sensor.py', 'clear_cutpoint_contract.py',
               'native_query_visibility.py', 'query_visibility.py', 'qwen_coordinates.py', 'dataset_review.py')
_LOADED_CODE = {name: _file_sha(_CODE_ROOT / name) for name in _CODE_NAMES}


def _check_code(expected):
    _require(expected == _LOADED_CODE and all(_file_sha(_CODE_ROOT / name) == pin
             for name, pin in _LOADED_CODE.items()), 'Implementation bindings changed')


@dataclass(frozen=True)
class FilePin:
    path: str
    sha256: str


def _pinned(pin):
    _require(isinstance(pin, FilePin) and Path(pin.path).is_absolute(), 'Absolute FilePin required')
    raw = bundle.read_bounded(Path(pin.path))
    _require(bundle.digest(raw) == _sha(pin.sha256), 'External document pin changed')
    return raw


def _counts(counts):
    _require(isinstance(counts, dict) and set(counts) == set(SPLITS)
             and all(type(n) is int and n > 0 for n in counts.values())
             and counts['train'] >= 20000, 'Exact TRAIN >=20000 and positive heldout counts required')


def _handoff(documents):
    selection, admission, splits = (_parse(documents[k]) for k in ('selection', 'admission', 'frozen_splits'))
    _require(isinstance(selection, dict) and set(selection) == {'schema', 'rows'}
             and selection['schema'] == SELECTION_SCHEMA and isinstance(selection['rows'], list),
             'Explicit final selection schema required; candidates unsupported')
    _require(isinstance(splits, dict) and len(splits) == 24
             and all(isinstance(k, str) and k.strip() for k in splits)
             and Counter(splits.values()) == {'train': 16, 'validation': 4, 'test': 4},
             'Complete frozen 24-donor 16/4/4 split map required')
    _require(json_sha256(splits) == FROZEN_SPLITS_SHA256,
             'Exact original donor names and frozen split reservations required')
    _require(isinstance(admission, dict) and set(admission) == {'schema', 'state', 'authority',
             'selection_sha256', 'frozen_splits_sha256', 'counts', 'gates'}
             and admission['schema'] == ADMISSION_SCHEMA
             and admission['state'] == 'final_selection_admitted'
             and isinstance(admission['authority'], str) and admission['authority'].strip(),
             'External final admission required; no candidate promotion')
    for name in ('selection', 'frozen_splits'):
        _require(admission[name + '_sha256'] == bundle.digest(documents[name]), 'Admission binding mismatch')
    _counts(admission['counts'])
    _require(isinstance(admission['gates'], dict) and set(admission['gates']) == set(GATES),
             'All external admission gates required')
    for gate in admission['gates'].values():
        _require(isinstance(gate, dict) and set(gate) == {'passed', 'evidence_sha256'}
                 and gate['passed'] is True, 'External passed evidence required')
        _sha(gate['evidence_sha256'])
    seen, rgbs, bundles = set(), set(), set()
    rows = selection['rows']
    for row in rows:
        _require(isinstance(row, dict) and set(row) == ROW_FIELDS, 'Exact selection row fields required')
        for field in ('sample_id', 'capture_sample_id', 'target_id', 'source_family', 'source_target'):
            _require(isinstance(row[field], str) and row[field].strip(), 'Nonempty row identity required')
        for field in ROW_FIELDS:
            if field.endswith('_sha256'):
                _sha(row[field])
        _require(row['source_family'] in splits and row['split'] == splits[row['source_family']],
                 'Original donor frozen split mismatch')
        _require(row['source_target'].startswith(row['source_family'] + '/'), 'Original target ancestry mismatch')
        _require(row['sample_id'] not in seen and row['decoded_rgb_sha256'] not in rgbs
                 and row['bundle_sha256'] not in bundles, 'Duplicate selected identity or RGB')
        seen.add(row['sample_id']); rgbs.add(row['decoded_rgb_sha256']); bundles.add(row['bundle_sha256'])
    _require(dict(Counter(row['split'] for row in rows)) == admission['counts'], 'Frozen selection counts mismatch')
    return rows, admission, splits


def _regular_tree(root):
    """No symlinks/junctions or undeclared physical files in the portable tree."""
    def link(path):
        # Path.is_junction is unavailable on the runbook's Python 3.11.
        return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0)
                                        & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
    root = Path(root)
    _require(root.is_dir() and not link(root), 'Regular directory required')
    files = set()
    for path in root.rglob('*'):
        _require(not link(path), 'Links/junctions unsupported')
        _require(path.is_dir() or path.is_file(), 'Special filesystem entry unsupported')
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _png(raw, size, mode):
    with Image.open(io.BytesIO(raw)) as image:
        _require(image.format == 'PNG' and image.mode == mode and image.size == size, 'Native PNG layout required')
        image.load()
        return image.copy()


def _sample(root, row):
    """Full local byte/buffer contract check; NOT a geometry or visual audit."""
    root = Path(root)
    marker = root / 'bundle.json'
    _require(marker.is_file(), 'Raw input unsupported; verified compact source required')
    _require(_file_sha(marker) == row['bundle_sha256'], 'Compact bundle pin changed')
    reader = bundle.SampleReader(root, expected_bindings={'sample.json': row['sample_sha256'],
        'supervision/label.json': row['label_sha256'], 'inputs/rgb.png': row['encoded_rgb_sha256']})
    _require(reader.manifest is not None, 'Compact source became raw')
    physical = {'bundle.json': row['bundle_sha256']}
    for name, entry in reader.manifest['files'].items():
        path = entry['stored_path']
        _require(path.startswith('payload/') and path not in physical, 'Compact payload path required')
        physical[path] = entry['stored_sha256']
        reader.read(name)  # Check both encoded and logical bytes, including every sidecar.
    _require(_regular_tree(root) == set(physical), 'Undeclared compact files')
    metadata, label = reader.metadata, reader.json('supervision/label.json')
    supervision = metadata['supervision']
    _require(metadata['sample_id'] == row['capture_sample_id']
             and supervision['target_id'] == label['target_id'] == row['target_id']
             and supervision['split_group'] == label['source_plant_family'] == row['source_family']
             and supervision.get('source_target_id', supervision['target_id'])
             == label['conservative_view_cap_group'] == row['source_target'], 'Sample ancestry mismatch')
    cal = metadata['calibration']
    _require(json_sha256(cal) == row['calibration_sha256'] and cal['resolution'] == [1696, 816]
             and cal['crop_resize'] is None and cal['depth_convention'] == 'optical_axis_z_metres_not_ray_range',
             'Native calibration binding/layout mismatch')
    clip = cal['clipping_range_m']
    _require(len(clip) == 2 and all(type(v) in (int, float) and np.isfinite(v) for v in clip)
             and 0 < clip[0] < clip[1], 'Invalid native clipping range')
    _require(label['task'] == contract.TASK and label['contract_sha256'] == contract.contract_hash()
             and label['resolution'] == [1696, 816] and label['eligible'] is True
             and label['native_depth_reconstructed'] is False
             and label['hidden_cut_coordinates_executable'] is False, 'Eligible native label required')
    contract.point(label['query_pixel_uv']); contract.point(label['nominal_pixel_uv'])
    answer = contract.validate_answer(label['answer'])
    _require(answer['status'] == 'localized' and answer['visibility'] == 'clear'
             and answer['cut_point_uv'] == [round(float(v), 2) for v in label['nominal_pixel_uv']],
             'Clear canonical nominal answer required')
    rgb = _png(reader.read('inputs/rgb.png'), (1696, 816), 'RGB')
    _require(decoded_rgb_digest(rgb.tobytes(), width=1696, height=816) == row['decoded_rgb_sha256'],
             'Decoded RGB pin changed')
    depth = reader.array('inputs/depth_m.npy')
    _require(depth.shape == (816, 1696) and depth.dtype == np.float32, 'Native float32 optical-Z required')
    valid = np.asarray(_png(reader.read('inputs/depth_valid.png'), (1696, 816), 'L'))
    _require(np.isin(valid, [0, 255]).all() and np.array_equal(valid != 0,
             np.isfinite(depth) & (depth > 0) & (depth >= clip[0]) & (depth <= clip[1])),
             'Native depth validity mismatch')
    for name in ('supervision/renderer_instance_id.npy', 'supervision/component_id.npy'):
        array = reader.array(name)
        _require(array.shape == (816, 1696) and array.dtype.kind in 'iu', 'Native identity layout required')
    for name in ('supervision/target_visible.png', 'supervision/organ_type.png'):
        _png(reader.read(name), (1696, 816), 'L')
    _require(_file_sha(marker) == row['bundle_sha256'], 'Compact changed during validation')
    return reader, label, rgb, physical


def _record(index, row, reader, label, rgb):
    base = f'samples/{index:08d}'
    full = base + '/' + reader.manifest['files']['inputs/rgb.png']['stored_path']
    crop = f'crops/{index:08d}.png'
    chat = contract.model_messages(rgb, label['query_pixel_uv'], answer=label['answer'], crop=True)
    for item, path in zip(chat[1]['content'][:2], (full, crop)):
        item['image'] = path
    return dict(sample_id=row['sample_id'], split=row['split'], compact_root=base,
                query_pixel_uv=label['query_pixel_uv'], crop_box=contract.crop_box(label['query_pixel_uv']),
                full_rgb=full, query_crop=crop, messages=chat)


def _put(root, relative, raw, files):
    path = bundle.safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    files[relative] = bundle.digest(raw)


def build(destination, *, selection, admission, frozen_splits, sources):
    """Create a NEW package only; returns manifest_sha256 and the manifest.

    No override of counts, gates or frozen splits. Nothing is created until all
    selected sources pass preflight. On failure never publishes manifest.json.
    Destination and sources/documents must be nonnested and already quiescent.
    """
    destination = Path(destination).absolute()
    _require(not destination.exists(), 'Create-only destination required')
    pins = dict(selection=selection, admission=admission, frozen_splits=frozen_splits)
    documents = {key: _pinned(pin) for key, pin in pins.items()}
    rows, _, _ = _handoff(documents)
    _require(isinstance(sources, dict) and set(sources) == {row['sample_id'] for row in rows},
             'Exact selected source mapping required')
    roots = {}
    for key, value in sources.items():
        _require(Path(value).is_absolute(), 'Absolute compact source required')
        roots[key] = Path(value).absolute()
    dest = destination.resolve()
    for path in [*roots.values(), *(Path(p.path) for p in pins.values())]:
        resolved = path.resolve()
        _require(not dest.is_relative_to(resolved) and not resolved.is_relative_to(dest),
                 'Separate nonnested destination required')
    _check_code(_LOADED_CODE)
    for row in rows:
        _sample(roots[row['sample_id']], row)
    destination.mkdir(parents=True, exist_ok=False)
    files, chats, source_files = {}, {split: [] for split in SPLITS}, {}
    for name, raw in documents.items():
        _put(destination, f'provenance/{name}.json', raw, files)
    for index, row in enumerate(rows):
        reader, label, rgb, physical = _sample(roots[row['sample_id']], row)
        source_files[row['sample_id']] = physical
        record = _record(index, row, reader, label, rgb)
        for name, pin in physical.items():
            raw = bundle.read_bounded(bundle.safe_path(reader.root, name))
            _require(bundle.digest(raw) == pin, 'Source changed while copying')
            _put(destination, record['compact_root'] + '/' + name, raw, files)
        stream = io.BytesIO()
        rgb.crop(record['crop_box']).save(stream, format='PNG')
        _put(destination, record['query_crop'], stream.getvalue(), files)
        chats[row['split']].append(record)
    for split in SPLITS:
        _put(destination, f'splits/{split}.jsonl', b''.join(_json(r) + b'\n' for r in chats[split]), files)
    manifest = dict(schema=SCHEMA, state='packaged_external_selection', policy=deepcopy(POLICY),
                    native_contract_sha256=contract.contract_hash(), implementation_bindings=dict(_LOADED_CODE),
                    provenance={name: bundle.digest(raw) for name, raw in documents.items()}, files=files)
    _validate(destination, manifest)
    # Recheck every source physical byte and external document at publication.
    for row in rows:
        root = roots[row['sample_id']]
        expected = source_files[row['sample_id']]
        _require(_regular_tree(root) == set(expected) and all(_file_sha(bundle.safe_path(root, name)) == pin
                 for name, pin in expected.items()), 'Source changed before publication')
    _require(all(_pinned(pins[name]) == raw for name, raw in documents.items()), 'Handoff changed before publication')
    _check_code(_LOADED_CODE)
    raw = _json(manifest)
    _put(destination, 'manifest.json', raw, {})
    return dict(manifest_sha256=bundle.digest(raw), manifest=manifest)


def _validate(root, manifest):
    _require(set(manifest) == {'schema', 'state', 'policy', 'native_contract_sha256',
             'implementation_bindings', 'provenance', 'files'} and manifest['schema'] == SCHEMA
             and manifest['state'] == 'packaged_external_selection' and manifest['policy'] == POLICY
             and manifest['native_contract_sha256'] == contract.contract_hash(), 'Native release schema/policy changed')
    _check_code(manifest['implementation_bindings'])
    files = manifest['files']
    _require(isinstance(files, dict) and set(manifest['provenance']) == {'selection', 'admission', 'frozen_splits'},
             'Release file/provenance inventory required')
    for name, pin in files.items():
        _require(_file_sha(bundle.safe_path(root, name)) == _sha(pin), 'Release file pin changed: ' + name)
    _require(_regular_tree(root) - {'manifest.json'} == set(files), 'Release contains undeclared files')
    documents = {name: bundle.read_bounded(bundle.safe_path(root, f'provenance/{name}.json'))
                 for name in manifest['provenance']}
    _require(all(bundle.digest(raw) == manifest['provenance'][name] for name, raw in documents.items()),
             'Provenance pin changed')
    rows, admission, splits = _handoff(documents)
    records = {split: [_parse(line) for line in bundle.read_bounded(
        bundle.safe_path(root, f'splits/{split}.jsonl')).splitlines()] for split in SPLITS}
    _require({key: len(value) for key, value in records.items()} == admission['counts'], 'Chat counts changed')
    positions = Counter()
    expected_files = {f'provenance/{name}.json' for name in documents} | {f'splits/{s}.jsonl' for s in SPLITS}
    for index, row in enumerate(rows):
        base = f'samples/{index:08d}'
        reader, label, rgb, physical = _sample(bundle.safe_path(root, base), row)
        expected = _record(index, row, reader, label, rgb)
        actual = records[row['split']][positions[row['split']]]
        positions[row['split']] += 1
        _require(actual == expected, 'Portable chat/query/answer binding changed')
        crop = _png(bundle.read_bounded(bundle.safe_path(root, expected['query_crop'])), (768, 768), 'RGB')
        _require(crop.tobytes() == rgb.crop(expected['crop_box']).tobytes(), 'Native crop pixels changed')
        expected_files.update(base + '/' + name for name in physical)
        expected_files.add(expected['query_crop'])
    _require(set(files) == expected_files, 'Unexpected packaged content')
    return dict(counts=admission['counts'], frozen_splits=splits, records=records,
                admission_scope=POLICY['admission'], training_qualification_performed=False)


def validate(root, *, manifest_sha256):
    """Read-only relocation validator; requires trusted external manifest pin."""
    root = Path(root)
    raw = bundle.read_bounded(root / 'manifest.json')
    _require(bundle.digest(raw) == _sha(manifest_sha256), 'External manifest pin changed')
    result = _validate(root, _parse(raw))
    _require(bundle.read_bounded(root / 'manifest.json') == raw, 'Manifest changed during validation')
    return result


def model_messages(root, record, *, supervised=False, crop=True):
    """Prepare RGB-only messages from a record returned by validate().

    Validation must precede loading, and package must remain immutable. This is
    not a substitute for validate(). Regenerates text using the native contract;
    never loads depth/masks/world labels. Optional answer is assistant-only.
    """
    rgb = _png(bundle.read_bounded(bundle.safe_path(root, record['full_rgb'])), (1696, 816), 'RGB')
    answer = None
    if supervised:
        # Preserve the original normalized answer exactly (no inverse/re-round).
        answer = contract.validate_answer(_parse(record['messages'][2]['content'][0]['text']), normalized=True)
    messages = contract.model_messages(rgb, record['query_pixel_uv'], crop=crop)
    if supervised:
        messages.append(dict(role='assistant', content=[dict(type='text', text=json.dumps(answer, separators=(',', ':')))]))
    return messages
