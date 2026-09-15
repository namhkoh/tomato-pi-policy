"""Synthetic CPU-only exporter tests; no native/release qualification evidence.

Policy tests exercise the real >=20k floor using in-memory rows only. Packaging
integration tests monkeypatch ONLY the private count checker for three synthetic
temporary samples (one per split). No public bypass, real release, or actual
admission declaration is produced. Actual compact reading is tested separately,
read-only, when NATIVE_CLEAR_EXPORT_SAMPLE names an explicitly supplied fixture.
"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

from . import bundle, native_clear_export as export
from .admission import decoded_rgb_digest
from .. import native_clear_contract as contract


def write(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def write_json(root, name, value):
    return write(root, name, export._json(value))


def frozen():
    return {'seed' + str(seed) + '_full': split for split, seeds in (
        ('train', (101, 103, 11, 17, 19, 23, 41, 43, 47, 53, 67, 71, 73, 7, 83, 89)),
        ('validation', (13, 29, 37, 97)), ('test', (31, 59, 61, 79))) for seed in seeds}


def documents(rows, counts=None):
    selection = export._json(dict(schema=export.SELECTION_SCHEMA, rows=rows))
    splits = export._json(frozen())
    admission = dict(schema=export.ADMISSION_SCHEMA, state='final_selection_admitted',
                     authority='SYNTHETIC_TEST_ONLY_NOT_AN_APPROVAL',
                     selection_sha256=bundle.digest(selection), frozen_splits_sha256=bundle.digest(splits),
                     counts=counts or dict(train=1, validation=1, test=1),
                     gates={k: dict(passed=True, evidence_sha256=bundle.digest(k.encode())) for k in export.GATES})
    return dict(selection=selection, admission=export._json(admission), frozen_splits=splits)


def synthetic(root, index):
    family = ('seed101_full', 'seed13_full', 'seed31_full')[index]
    split = frozen()[family]
    # Capture-local IDs can repeat across independent capture roots.
    local = 'SubStem_42_wide_006'
    raw, compact = root / str(index) / 'raw', root / str(index) / 'compact'
    yy, xx = np.indices((816, 1696), dtype=np.uint16)
    rgb = np.stack(((xx + index) % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    depth = np.ones((816, 1696), dtype=np.float32)
    depth[0, :4] = [0., -0., np.nan, np.inf]
    valid = np.isfinite(depth) & (depth > 0)
    for name, array in {
        'inputs/depth_m.npy': depth,
        'supervision/component_id.npy': np.zeros((816, 1696), dtype=np.uint32),
        'supervision/renderer_instance_id.npy': np.zeros((816, 1696), dtype=np.uint32),
    }.items():
        stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
        write(raw, name, stream.getvalue())
    for name, array in {
        'inputs/rgb.png': rgb, 'inputs/depth_valid.png': valid.astype(np.uint8) * 255,
        'supervision/target_visible.png': np.zeros((816, 1696), dtype=np.uint8),
        'supervision/organ_type.png': np.zeros((816, 1696), dtype=np.uint8),
    }.items():
        stream = io.BytesIO(); Image.fromarray(array).save(stream, format='PNG')
        write(raw, name, stream.getvalue())
    write_json(raw, 'supervision/identities.json', dict(component_catalogue=[]))
    files = {p.relative_to(raw).as_posix(): dict(sha256=export._file_sha(p),
        role='observation' if p.parent.name == 'inputs' else 'ground_truth_supervision')
        for p in raw.rglob('*') if p.is_file()}
    target, original = family + '_generated/SubStem_42', family + '/SubStem_42'
    cal = dict(resolution=[1696, 816], crop_resize=None,
               depth_convention='optical_axis_z_metres_not_ray_range', clipping_range_m=[.04, 10.],
               intrinsics=[[900., 0., 848.], [0., 900., 408.], [0., 0., 1.]])
    write_json(raw, 'sample.json', dict(sample_id=local, files=files, calibration=cal,
        training_sample_approved=False, original_source='Z:/unavailable/capture.usd',
        supervision=dict(target_id=target, source_target_id=original, split_group=family)))
    nominal = [1053.07395794031, 250.64042443424904]
    label = dict(task=contract.TASK, contract_sha256=contract.contract_hash(), eligible=True,
                 resolution=[1696, 816], training_approved=False, source_plant_family=family,
                 target_id=target, conservative_view_cap_group=original,
                 native_depth_reconstructed=False, hidden_cut_coordinates_executable=False,
                 query_pixel_uv=([1207.6, 322.1], [0., 0.], [1695.99, 815.99])[index],
                 nominal_pixel_uv=nominal, answer=dict(status='localized', visibility='clear',
                 next_action='inspect_cut_region', cut_point_uv=[round(v, 2) for v in nominal]))
    write_json(raw, 'supervision/label.json', label)
    trace = dict(passed=True, fixture='synthetic_only')
    write_json(raw, 'supervision/query_trace.json', trace)
    sample_pin, label_pin = (export._file_sha(raw / name) for name in ('sample.json', 'supervision/label.json'))
    bundle.pack_sample(raw, compact, sample_sha256=sample_pin, label_sha256=label_pin, query_trace=trace)
    row = dict(sample_id=f'global-{index}', capture_sample_id=local, target_id=target, source_family=family,
               source_target=original, split=split, bundle_sha256=export._file_sha(compact / 'bundle.json'),
               sample_sha256=sample_pin, label_sha256=label_pin,
               encoded_rgb_sha256=export._file_sha(raw / 'inputs/rgb.png'),
               decoded_rgb_sha256=decoded_rgb_digest(rgb.tobytes(), width=1696, height=816),
               calibration_sha256=export.json_sha256(cal))
    return compact, row


@pytest.fixture(scope='module')
def samples(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_native_export_sources')
    return [synthetic(root, i) for i in range(3)]


@pytest.fixture
def tiny_gate(monkeypatch):
    # Test isolation seam only; the public API has no bypass/minimum argument.
    def counts(value):
        assert value == dict(train=1, validation=1, test=1)
    monkeypatch.setattr(export, '_counts', counts)


def pins(root, docs):
    return {name: export.FilePin(str(write(root, name + '.json', raw)), bundle.digest(raw))
            for name, raw in docs.items()}


def build_tiny(tmp_path, samples):
    docs = documents([row for _, row in samples])
    result = export.build(tmp_path / 'package', **pins(tmp_path / 'handoff', docs),
                          sources={row['sample_id']: str(root) for root, row in samples})
    return tmp_path / 'package', result, docs


def tree(root):
    return {p.relative_to(root).as_posix(): export._file_sha(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('counts', [dict(train=19999, validation=1, test=1),
    dict(train=20000, validation=0, test=1), dict(train=20000, validation=1, test=-1),
    dict(train=20000, validation=True, test=1), dict(train=20000, validation=1),
    dict(train=20000.0, validation=1, test=1)])
def test_real_count_floor_rejects(counts):
    with pytest.raises(ValueError, match='positive heldout'):
        export._counts(counts)


@pytest.mark.parametrize('val,test', [(1, 1), (2000, 2000), (713, 99)])
def test_heldout_counts_are_explicit_not_invented(val, test):
    export._counts(dict(train=20000, validation=val, test=test))


def test_real_handoff_20002_in_memory_only(samples):
    rows = []
    for i in range(20002):
        row = deepcopy(samples[0 if i < 20000 else i - 19999][1])
        row['sample_id'] = str(i)
        row['bundle_sha256'] = bundle.digest(f'bundle{i}'.encode())
        row['decoded_rgb_sha256'] = bundle.digest(f'rgb{i}'.encode())
        rows.append(row)
    actual, admission, splits = export._handoff(documents(rows, dict(train=20000, validation=1, test=1)))
    assert actual == rows and len(splits) == 24 and admission['counts']['train'] == 20000


@pytest.mark.parametrize('mutation', ['candidate', 'pending', 'missing_gate', 'unpassed', 'bad_evidence',
    'admission_pin', 'split_pin', 'missing_donor', 'split_ratio', 'row_split', 'duplicate', 'extra_field',
    'source_target', 'bad_sha', 'count_mismatch'])
def test_handoff_rejects_invalid_before_output(tmp_path, samples, tiny_gate, mutation):
    docs = documents([r for _, r in samples])
    sel, adm, splits = (export._parse(docs[k]) for k in ('selection', 'admission', 'frozen_splits'))
    if mutation == 'candidate': sel['schema'] = 'greenhouse.native_admission_candidates.v1'
    if mutation == 'pending': adm['state'] = 'candidate_selection_only'
    if mutation == 'missing_gate': del adm['gates'][export.GATES[0]]
    if mutation == 'unpassed': adm['gates'][export.GATES[0]]['passed'] = False
    if mutation == 'bad_evidence': adm['gates'][export.GATES[0]]['evidence_sha256'] = 'missing'
    if mutation == 'missing_donor': del splits['seed79_full']
    if mutation == 'split_ratio': splits['seed79_full'] = 'train'
    if mutation == 'row_split': sel['rows'][0]['split'] = 'validation'
    if mutation == 'duplicate': sel['rows'][1]['decoded_rgb_sha256'] = sel['rows'][0]['decoded_rgb_sha256']
    if mutation == 'extra_field': sel['rows'][0]['training_approved'] = True
    if mutation == 'source_target': sel['rows'][0]['source_target'] = 'new_donor/SubStem_42'
    if mutation == 'bad_sha': sel['rows'][0]['sample_sha256'] = 'x' * 64
    if mutation == 'count_mismatch': sel['rows'].pop()
    docs['selection'], docs['frozen_splits'] = export._json(sel), export._json(splits)
    adm['selection_sha256'] = bundle.digest(docs['selection'])
    adm['frozen_splits_sha256'] = bundle.digest(docs['frozen_splits'])
    if mutation == 'admission_pin': adm['selection_sha256'] = '0' * 64
    if mutation == 'split_pin': adm['frozen_splits_sha256'] = '0' * 64
    docs['admission'] = export._json(adm)
    with pytest.raises(ValueError):
        export.build(tmp_path / 'never_created', **pins(tmp_path / 'docs', docs), sources={})
    assert not (tmp_path / 'never_created').exists()


def test_exact_portable_build_and_messages(tmp_path, samples, tiny_gate):
    before = [tree(p) for p, _ in samples]
    root, result, docs = build_tiny(tmp_path, samples)
    checked = export.validate(root, manifest_sha256=result['manifest_sha256'])
    assert checked['frozen_splits'] == frozen()
    assert checked['counts'] == dict(train=1, validation=1, test=1)
    assert checked['training_qualification_performed'] is False
    assert before == [tree(p) for p, _ in samples]
    for name, raw in docs.items():
        assert (root / 'provenance' / (name + '.json')).read_bytes() == raw
    for i, (source, row) in enumerate(samples):
        copied = root / 'samples' / f'{i:08d}'
        assert tree(copied) == tree(source)
        assert bundle.SampleReader(copied).read('inputs/depth_m.npy') == bundle.SampleReader(source).read('inputs/depth_m.npy')
        record = checked['records'][row['split']][0]
        train = export.model_messages(root, record, supervised=True)
        infer = export.model_messages(root, record)
        assert train[0] == infer[0] and train[1]['content'][-1] == infer[1]['content'][-1]
        assert len(infer) == 2 and len(train) == 3
        assert [item['image'].size for item in infer[1]['content'][:2]] == [(1696, 816), (768, 768)]
        assert all(a['image'].tobytes() == b['image'].tobytes() for a, b in zip(train[1]['content'][:2], infer[1]['content'][:2]))
        full, crop = [item['image'] for item in infer[1]['content'][:2]]
        assert full.crop(record['crop_box']).tobytes() == crop.tobytes()
        assert np.array_equal(np.asarray(Image.open(root / record['query_crop'])), np.asarray(crop))
        answer = export._parse(train[2]['content'][0]['text'])
        assert answer == {'status': 'localized', 'visibility': 'clear', 'next_action': 'inspect_cut_region',
                          'cut_point_uv': [620.91, 307.16]}
        canonical = contract.canonical_answer(answer)['cut_point_uv']
        assert np.max(np.abs(np.array(canonical) - [1053.07, 250.64])) <= .01696
        full_only = export.model_messages(root, record, crop=False)
        assert len(full_only[1]['content']) == 2
        assert 'depth' not in json.dumps(record['messages']).lower()
        assert 'world_m' not in json.dumps(record['messages'])
    relocated = tmp_path / 'relocated'
    root.rename(relocated)
    # Original metadata contains a nonexistent Windows path; validator does not follow it.
    assert export.validate(relocated, manifest_sha256=result['manifest_sha256']) == checked
    with pytest.raises(ValueError, match='Create-only'):
        export.build(relocated, selection=None, admission=None, frozen_splits=None, sources={})
    result['manifest']['policy']['resolution'][0] = 1
    result['manifest']['implementation_bindings'].clear()
    assert export.POLICY['resolution'] == [1696, 816] and export._LOADED_CODE
    assert export.validate(relocated, manifest_sha256=result['manifest_sha256']) == checked


@pytest.mark.parametrize('field', ['sample_id', 'capture_sample_id', 'target_id', 'source_target',
    'source_family', 'sample_sha256', 'label_sha256', 'bundle_sha256', 'encoded_rgb_sha256',
    'decoded_rgb_sha256', 'calibration_sha256'])
def test_source_identity_pins(samples, field):
    source, row = samples[0]
    row = dict(row)
    row[field] = '0' * 64 if field.endswith('_sha256') else 'wrong'
    if field == 'sample_id':
        # Global inventory ID is intentionally independent of capture-local name.
        export._sample(source, row)
    else:
        with pytest.raises(ValueError):
            export._sample(source, row)


@pytest.mark.parametrize('name', ['bundle.json', 'payload/inputs/rgb.png', 'payload/inputs/depth_m.npy.ghn',
                                  'payload/sample.json', 'payload/supervision/query_trace.json'])
def test_compact_tampering(tmp_path, samples, name):
    source, row = samples[0]
    target = tmp_path / 'mutated'; shutil.copytree(source, target)
    path = target / name
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError):
        export._sample(target, row)


def test_raw_explicitly_unsupported(samples):
    source, row = samples[0]
    with pytest.raises(ValueError, match='Raw input unsupported'):
        export._sample(source.parent / 'raw', row)


def test_unlisted_file_held(tmp_path, samples):
    source, row = samples[0]
    target = tmp_path / 'extra'; shutil.copytree(source, target)
    write(target, 'executable_gt.py', b'not executable model input')
    with pytest.raises(ValueError, match='Undeclared'):
        export._sample(target, row)


@pytest.mark.parametrize('mutation', ['chat', 'crop', 'extra', 'code', 'manifest'])
def test_relocation_validator_rejects_rehashed_tampering(tmp_path, samples, tiny_gate, mutation):
    root, result, _ = build_tiny(tmp_path, samples)
    manifest = deepcopy(result['manifest'])
    if mutation == 'chat':
        name = 'splits/train.jsonl'
        row = export._parse((root / name).read_bytes())
        row['messages'][2]['content'][0]['text'] = row['messages'][2]['content'][0]['text'].replace('620.91', '310.45')
        write(root, name, export._json(row) + b'\n')
        manifest['files'][name] = export._file_sha(root / name)
    if mutation == 'crop':
        name = 'crops/00000000.png'
        stream = io.BytesIO(); Image.new('RGB', (768, 768)).save(stream, format='PNG')
        write(root, name, stream.getvalue()); manifest['files'][name] = export._file_sha(root / name)
    if mutation == 'extra':
        write(root, 'unsafe.py', b'not included'); manifest['files']['unsafe.py'] = export._file_sha(root / 'unsafe.py')
    if mutation == 'code': manifest['implementation_bindings']['native_dataset/bundle.py'] = '0' * 64
    raw = export._json(manifest)
    write(root, 'manifest.json', raw)
    pin = result['manifest_sha256'] if mutation == 'manifest' else bundle.digest(raw)
    if mutation == 'manifest':
        write(root, 'manifest.json', raw + b' ')
    with pytest.raises(ValueError):
        export.validate(root, manifest_sha256=pin)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): export._parse(raw)


def test_source_changed_while_copying_never_publishes(tmp_path, samples, tiny_gate, monkeypatch):
    source, row = samples[0]
    changed = tmp_path / 'mutable'; shutil.copytree(source, changed)
    copies = [(changed, row), *samples[1:]]
    original_put = export._put
    def mutate(root, name, raw, files):
        original_put(root, name, raw, files)
        if name == 'splits/train.jsonl':
            with (changed / 'bundle.json').open('ab') as stream: stream.write(b' ')
    monkeypatch.setattr(export, '_put', mutate)
    with pytest.raises(ValueError, match='Source changed before publication'):
        build_tiny(tmp_path, copies)
    assert not (tmp_path / 'package' / 'manifest.json').exists()


def test_external_pin_and_missing_source_mapping_fail_without_writes(tmp_path, samples, tiny_gate):
    docs = documents([r for _, r in samples])
    args = pins(tmp_path / 'handoff', docs)
    with pytest.raises(ValueError, match='source mapping'):
        export.build(tmp_path / 'absent', **args, sources={})
    args['selection'] = export.FilePin(args['selection'].path, '0' * 64)
    with pytest.raises(ValueError, match='pin changed'):
        export.build(tmp_path / 'absent', **args, sources={})
    assert not (tmp_path / 'absent').exists()


def test_actual_compact_read_only_if_explicitly_supplied():
    value = os.environ.get('NATIVE_CLEAR_EXPORT_SAMPLE')
    if not value: pytest.skip('Explicit actual compact source not provided')
    root = Path(value)
    before = tree(root)
    reader = bundle.SampleReader(root)
    metadata, label = reader.metadata, reader.json('supervision/label.json')
    rgb = reader.image('inputs/rgb.png')
    # Locally computed pins here test extractor compatibility ONLY, not admission.
    row = dict(sample_id='read-only-diagnostic', capture_sample_id=metadata['sample_id'],
        target_id=label['target_id'], source_family=label['source_plant_family'],
        source_target=label['conservative_view_cap_group'], split='train',
        bundle_sha256=export._file_sha(root / 'bundle.json'), sample_sha256=bundle.digest(reader.read('sample.json')),
        label_sha256=bundle.digest(reader.read('supervision/label.json')),
        encoded_rgb_sha256=bundle.digest(reader.read('inputs/rgb.png')),
        decoded_rgb_sha256=decoded_rgb_digest(rgb.tobytes(), width=1696, height=816),
        calibration_sha256=export.json_sha256(metadata['calibration']))
    verified, checked_label, image, _ = export._sample(root, row)
    record = export._record(0, row, verified, checked_label, image)
    assert record['crop_box'] == contract.crop_box(label['query_pixel_uv'])
    assert image.size == (1696, 816) and tree(root) == before


@pytest.mark.parametrize('mutation', ['ineligible', 'wrong_task', 'hidden_action', 'depth_reconstructed',
    'partial', 'wrong_answer', 'query_outside', 'resolution', 'resize', 'ray_range', 'invalid_clip',
    'depth_dtype', 'depth_shape', 'validity', 'rgba', 'identity_dtype', 'mask_mode'])
def test_rebound_invalid_native_semantics_still_rejected(tmp_path, samples, mutation):
    source, row = samples[0]
    raw = tmp_path / 'raw'; shutil.copytree(source.parent / 'raw', raw)
    metadata = export._parse((raw / 'sample.json').read_bytes())
    label = export._parse((raw / 'supervision/label.json').read_bytes())
    if mutation == 'ineligible': label['eligible'] = False
    if mutation == 'wrong_task': label['task'] = 'legacy'
    if mutation == 'hidden_action': label['hidden_cut_coordinates_executable'] = True
    if mutation == 'depth_reconstructed': label['native_depth_reconstructed'] = True
    if mutation == 'partial': label['answer']['visibility'] = 'partial'
    if mutation == 'wrong_answer': label['answer']['cut_point_uv'] = [1., 2.]
    if mutation == 'query_outside': label['query_pixel_uv'] = [1696., 400.]
    if mutation == 'resolution': metadata['calibration']['resolution'] = [848, 408]
    if mutation == 'resize': metadata['calibration']['crop_resize'] = [1696, 816]
    if mutation == 'ray_range': metadata['calibration']['depth_convention'] = 'ray_range'
    if mutation == 'invalid_clip': metadata['calibration']['clipping_range_m'] = [10., .04]
    changed = None
    if mutation in ('depth_dtype', 'depth_shape', 'identity_dtype'):
        changed = 'supervision/component_id.npy' if mutation == 'identity_dtype' else 'inputs/depth_m.npy'
        array = np.load(raw / changed, allow_pickle=False)
        array = array[:4] if mutation == 'depth_shape' else array.astype(np.float64)
        stream = io.BytesIO(); np.save(stream, array, allow_pickle=False); write(raw, changed, stream.getvalue())
    if mutation in ('validity', 'rgba', 'mask_mode'):
        changed = {'validity': 'inputs/depth_valid.png', 'rgba': 'inputs/rgb.png',
                   'mask_mode': 'supervision/target_visible.png'}[mutation]
        with Image.open(raw / changed) as image:
            image = image.copy()
        if mutation == 'validity': image.putpixel((10, 10), 0)
        else: image = image.convert('RGBA' if mutation == 'rgba' else 'RGB')
        stream = io.BytesIO(); image.save(stream, format='PNG'); write(raw, changed, stream.getvalue())
    if changed: metadata['files'][changed]['sha256'] = export._file_sha(raw / changed)
    write_json(raw, 'sample.json', metadata); write_json(raw, 'supervision/label.json', label)
    compact = tmp_path / 'compact'
    row = dict(row, sample_sha256=export._file_sha(raw / 'sample.json'),
               label_sha256=export._file_sha(raw / 'supervision/label.json'),
               encoded_rgb_sha256=export._file_sha(raw / 'inputs/rgb.png'),
               calibration_sha256=export.json_sha256(metadata['calibration']))
    bundle.pack_sample(raw, compact, sample_sha256=row['sample_sha256'], label_sha256=row['label_sha256'],
                       query_trace=export._parse((raw / 'supervision/query_trace.json').read_bytes()))
    row['bundle_sha256'] = export._file_sha(compact / 'bundle.json')
    with pytest.raises(ValueError): export._sample(compact, row)


def test_model_loading_no_sidecars_or_filesystem_writes(tmp_path, samples, tiny_gate):
    root, result, _ = build_tiny(tmp_path, samples)
    record = export.validate(root, manifest_sha256=result['manifest_sha256'])['records']['train'][0]
    active = [True]
    reads = []
    def audit(event, args):
        if not active[0]: return
        if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate'):
            raise AssertionError('Model input preparation attempted a write')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0]).replace('\\', '/')
            mode, flags = args[1:3]
            assert not (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            assert not (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            assert name.endswith('/inputs/rgb.png'), name
            reads.append(name)
    sys.addaudithook(audit)
    try:
        for supervised in (False, True):
            for crop in (False, True):
                export.model_messages(root, record, supervised=supervised, crop=crop)
    finally:
        active[0] = False
    assert len(reads) == 4


def test_nested_source_destination_rejected(tmp_path, samples, tiny_gate):
    docs = documents([r for _, r in samples])
    args = pins(tmp_path / 'handoff', docs)
    sources = {row['sample_id']: str(root) for root, row in samples}
    destination = samples[0][0] / 'cannot_create'
    with pytest.raises(ValueError, match='Separate nonnested'):
        export.build(destination, **args, sources=sources)
    assert not destination.exists()


def test_manifest_path_escape_rejected(tmp_path, samples):
    source, row = samples[0]
    target = tmp_path / 'mutated'; shutil.copytree(source, target)
    manifest = export._parse((target / 'bundle.json').read_bytes())
    manifest['files']['inputs/rgb.png']['stored_path'] = '../outside.png'
    write_json(target, 'bundle.json', manifest)
    row = dict(row, bundle_sha256=export._file_sha(target / 'bundle.json'))
    with pytest.raises(ValueError): export._sample(target, row)


@pytest.mark.parametrize('mutation', ['train_val_swap', 'val_test_swap', 'train_test_swap',
                                     'rename_train', 'rename_validation', 'rename_test'])
def test_exact_frozen_reservations_even_when_counts_and_external_pins_match(samples, tiny_gate, mutation):
    docs = documents([row for _, row in samples])
    splits = frozen()
    # Use donors absent from the three selected rows so no row/split mismatch
    # can accidentally catch these: the former count-only gate admitted them.
    if mutation.endswith('_swap'):
        a, b = {'train_val_swap': ('seed103_full', 'seed29_full'),
                'val_test_swap': ('seed29_full', 'seed59_full'),
                'train_test_swap': ('seed103_full', 'seed59_full')}[mutation]
        splits[a], splits[b] = splits[b], splits[a]
    else:
        name = {'rename_train': 'seed89_full', 'rename_validation': 'seed97_full',
                'rename_test': 'seed79_full'}[mutation]
        splits['seed9999_full'] = splits.pop(name)
    assert len(splits) == 24 and export.Counter(splits.values()) == {'train': 16, 'validation': 4, 'test': 4}
    docs['frozen_splits'] = export._json(splits)
    adm = export._parse(docs['admission'])
    adm['frozen_splits_sha256'] = bundle.digest(docs['frozen_splits'])
    docs['admission'] = export._json(adm)
    with pytest.raises(ValueError, match='Exact original donor names'):
        export._handoff(docs)


def test_exact_baseline_accepts_order_and_whitespace_not_reassignment(samples, tiny_gate):
    assert export.json_sha256(frozen()) == export.FROZEN_SPLITS_SHA256 == (
        '65b06cdded97a06c163aedc954656629dd548749b6cc6fb9f383e40a061c9e1c')
    docs = documents([row for _, row in samples])
    docs['frozen_splits'] = json.dumps(dict(reversed(list(frozen().items()))), indent=2).encode()
    adm = export._parse(docs['admission'])
    adm['frozen_splits_sha256'] = bundle.digest(docs['frozen_splits'])
    docs['admission'] = export._json(adm)
    assert export._handoff(docs)[2] == frozen()
