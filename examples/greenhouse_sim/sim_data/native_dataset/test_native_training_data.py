"""CPU fake-processor tests, not admission or real Qwen/grid qualification.

Only tmp_path synthetic packages are written. Reuse exporter fixtures and its
private three-row count seam; production NativeRows has no validation bypass.
No AutoProcessor, model, network, renderer or training calls are made.
"""
from copy import deepcopy
import json
import io
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pytest
import torch

from . import native_training_data as data
from . import native_clear_export as export
from . import native_clear_processor_preflight as preflight
from .test_native_clear_export import samples, build_tiny, tree
from .test_native_clear_processor_preflight import Qwen3VLProcessor as NumpyProcessor
from .test_native_clear_processor_preflight import snapshot


KEYS = ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids')


class Qwen3VLProcessor(NumpyProcessor):
    """Same tiny fake grids; PT branch uses real CPU torch.from_numpy only."""

    def apply_chat_template(self, messages, *, return_tensors, **kwargs):
        assert return_tensors in ('np', 'pt')
        output = super().apply_chat_template(messages, return_tensors='np', **kwargs)
        if self.mode == 'longer':
            for key, value in (('input_ids', 4), ('attention_mask', 1), ('mm_token_type_ids', 0)):
                output[key] = np.concatenate((np.array([[value]], dtype=np.int64), output[key]), axis=1)
        if return_tensors == 'pt':
            if self.mode == 'pt_pixels': output['pixel_values'] = output['pixel_values'] + 1
            if self.mode == 'pt_dtype': output['input_ids'] = output['input_ids'].astype(np.int32)
            if self.mode == 'pt_types': output['mm_token_type_ids'][0, -1] += 1
            if self.mode == 'pt_not_tensors': return output
            output = {k: torch.from_numpy(v.copy()) for k, v in output.items()}
            if self.mode == 'pt_grad': output['pixel_values'].requires_grad_(True)
        return output


@pytest.fixture(autouse=True)
def counts(monkeypatch):
    # Synthetic packaging only; no public adapter bypass for the >=20k floor.
    def tiny(value):
        assert value == dict(train=1, validation=1, test=1)
    monkeypatch.setattr(export, '_counts', tiny)


@pytest.fixture(scope='module')
def package(tmp_path_factory, samples):
    root = tmp_path_factory.mktemp('native_training_synthetic')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export, '_counts', lambda v: None)
        package_root, result, _ = build_tiny(root, samples)
    return package_root, result['manifest_sha256']


@pytest.fixture(scope='module')
def validated_rows(package):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export, '_counts', lambda v: None)
        result = data.NativeRows(package[0], manifest_sha256=package[1])
    yield result
    result.close()


@pytest.fixture
def rows(validated_rows):
    result = validated_rows.for_split('train')
    yield result
    result.close()


@pytest.fixture
def binding(rows, snapshot, tmp_path, monkeypatch):
    processor = Qwen3VLProcessor()
    class NeverAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert Path(path) == snapshot[0]
            assert kwargs == dict(local_files_only=True, trust_remote_code=False)
            return processor
    monkeypatch.setattr(preflight, '_auto_processor', lambda: NeverAutoProcessor)
    report = preflight.run(rows.root, manifest_sha256=rows.manifest_sha256,
        model_snapshot=snapshot[0], processor_files=snapshot[1], sample_ids=['global-0'], maximum_tokens=10000)
    path = tmp_path / 'preflight.json'
    path.write_bytes(export._json(report))
    return dict(processor=processor, preflight_report=export.FilePin(str(path), export._file_sha(path)),
                model_snapshot=snapshot[0], maximum_tokens=10000, model_input_keys=KEYS)


def ready(rows, binding):
    collator = data.NativeCollator(rows, **binding)
    report = data.check_numpy_torch_parity(collator, sample_ids=['global-0'])
    return collator, report


def test_validation_is_mandatory_and_not_caller_attested(package, monkeypatch):
    called = []
    def reject(*args, **kwargs):
        called.append(kwargs)
        raise ValueError('upstream validation failure')
    monkeypatch.setattr(export, 'validate', reject)
    with pytest.raises(ValueError, match='upstream validation'):
        data.NativeRows(package[0], manifest_sha256=package[1])
    assert called == [dict(manifest_sha256=package[1])]


def test_lazy_rows_no_revalidation_or_image_read_after_construction(rows, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('not a lazy row read')
    monkeypatch.setattr(export, 'validate', forbidden)
    monkeypatch.setattr(export, 'model_messages', forbidden)
    assert rows._stream is None
    assert len(rows) == 1 and rows[0] == rows[-1] == rows.by_id('global-0')
    assert rows._stream is not None
    assert rows.for_split('validation')[0]['sample_id'] == 'global-1'
    assert rows.receipt['frozen_splits']['seed13_full'] == 'validation'
    assert 'records' not in rows.__dict__ and 'messages' not in repr(rows._indices)
    changed = rows[0]; changed['query_pixel_uv'][0] = 1
    assert rows[0]['query_pixel_uv'][0] == 1207.6
    with pytest.raises(ValueError, match='sealed'): rows.for_split('test')
    assert rows.for_split('test', allow_test=True)[0]['sample_id'] == 'global-2'
    with pytest.raises(ValueError): rows.by_id('global-1')
    for index in (1, -2):
        with pytest.raises(IndexError): rows[index]
    for index in (True, '0', slice(None)):
        with pytest.raises(TypeError): rows[index]


def test_pickle_and_pid_reopen_use_independent_handles(rows, monkeypatch):
    expected = rows[0]
    first = rows._stream
    clone = pickle.loads(pickle.dumps(rows))
    assert clone._stream is None and not first.closed
    assert clone[0] == expected and clone._stream is not first
    original_pid = os.getpid()
    monkeypatch.setattr(data.os, 'getpid', lambda: original_pid + 1)
    assert rows[0] == expected and first.closed and rows._stream is not first
    clone.close()


def test_worker_retains_parent_source_pins(rows):
    clone = pickle.loads(pickle.dumps(rows))
    clone._bindings['adapter_source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Parent/worker source bindings'): clone[0]


def test_loaded_row_digest_not_only_stat_or_worker_pin(rows):
    record = rows[0]
    rows.close()
    raw = export._json(record).replace(b'global-0', b'global-X') + b'\n'
    rows._stream, rows._pid = io.BytesIO(raw), os.getpid()
    with pytest.raises(ValueError, match='Loaded JSONL row pin'): rows[0]


def identity_batch(batch):
    return batch


def test_real_spawn_workers_read_lazy_rows(rows):
    loader = torch.utils.data.DataLoader(rows, batch_size=1, num_workers=2,
        multiprocessing_context='spawn', collate_fn=identity_batch)
    assert list(loader) == [[rows[0]]]


@pytest.mark.parametrize('mutation', ['manifest', 'split', 'rgb', 'sidecar', 'extra'])
def test_release_source_tampering_held(package, tmp_path, mutation):
    root = tmp_path / 'copy'; shutil.copytree(package[0], root)
    rows = data.NativeRows(root, manifest_sha256=package[1])
    record = rows[0]
    if mutation == 'manifest': path = root / 'manifest.json'
    elif mutation == 'split': path = root / 'splits/train.jsonl'
    elif mutation == 'rgb': path = root / record['full_rgb']
    elif mutation == 'sidecar': path = next((root / record['compact_root']).rglob('*.ghn'))
    else: path = root / 'unexpected.txt'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): rows.verify_sources()
    if mutation in ('manifest', 'split'):
        with pytest.raises(ValueError): rows[0]
    rows.close()


def test_row_hash_independent_of_stat_guard(package, tmp_path, monkeypatch):
    root = tmp_path / 'copy'; shutil.copytree(package[0], root)
    rows = data.NativeRows(root, manifest_sha256=package[1])
    assert rows[0]['sample_id'] == 'global-0'
    path = root / 'splits/train.jsonl'
    raw = path.read_bytes().replace(b'global-0', b'global-X')
    path.write_bytes(raw)
    monkeypatch.setattr(rows, '_unchanged', lambda: None)  # Force row-hash branch.
    rows.close()
    # First worker-open check also independently hashes the whole split.
    with pytest.raises(ValueError, match='split pin'): rows[0]


def test_manifest_and_source_code_pins_required(package, monkeypatch):
    with pytest.raises(ValueError): data.NativeRows(package[0], manifest_sha256='0' * 64)
    with pytest.raises(ValueError): data.NativeRows('.', manifest_sha256=package[1])
    monkeypatch.setattr(data, '_CODE_SHA', '0' * 64)
    with pytest.raises(ValueError, match='source changed'):
        data.NativeRows(package[0], manifest_sha256=package[1])


def test_real_numpy_torch_parity_masks_coords_and_replay(rows, binding):
    before = tree(rows.root)
    collator, report = ready(rows, binding)
    tensors = collator([rows[0]])
    assert set(tensors) == set(KEYS) | {'labels'}
    assert all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' for t in tensors.values())
    count = report['rows'][0]['prompt_tokens']
    assert torch.all(tensors['labels'][:, :count] == -100)
    assert torch.equal(tensors['labels'][:, count:], tensors['input_ids'][:, count:])
    answer = binding['processor'].tokenizer.decode(tensors['labels'][tensors['labels'] != -100].tolist(),
                skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert data.decode_answer(answer)['cut_point_uv'] == pytest.approx([1053.06336, 250.64256])
    assert report['rows'][0]['image_grid_thw'] == [[1, 4, 8], [1, 4, 4]]  # Fake only.
    again = collator([rows[0]])
    assert all(torch.equal(tensors[k], again[k]) for k in tensors)
    assert tree(rows.root) == before
    assert not report['model_loaded'] and not report['training_approved']


def test_parity_required_and_failed_retry_disarms(rows, binding):
    collator = data.NativeCollator(rows, **binding)
    with pytest.raises(ValueError, match='parity'): collator([rows[0]])
    data.check_numpy_torch_parity(collator, sample_ids=['global-0'])
    with pytest.raises(ValueError): data.check_numpy_torch_parity(collator, sample_ids=[])
    with pytest.raises(ValueError, match='parity'): collator([rows[0]])


@pytest.mark.parametrize('mode', ['pt_pixels', 'pt_dtype', 'pt_types', 'pt_not_tensors', 'pt_grad'])
def test_cpu_torch_divergence_held(rows, binding, mode):
    binding['processor'].mode = mode
    collator = data.NativeCollator(rows, **binding)
    with pytest.raises(ValueError): data.check_numpy_torch_parity(collator, sample_ids=['global-0'])
    assert not collator._ready


@pytest.mark.parametrize('mode', ['wrong_prefix', 'padding', 'float_ids', 'batch_two', 'missing_grid',
    'one_grid', 'video', 'fractional_grid', 'zero_grid', 'unmerged_grid', 'bad_pixels', 'nan_pixels',
    'prefix_pixels', 'prefix_types', 'wrong_image_tokens', 'image_in_answer', 'bad_decode',
    'missing_system', 'answer_leak', 'mutate_image', 'truncate'])
def test_hot_path_fake_processor_fail_closed(rows, binding, mode):
    collator, _ = ready(rows, binding)
    collator.processor.mode = mode
    collator.processor.tokenizer.mode = mode
    with pytest.raises((ValueError, KeyError)): collator([rows[0]])


def test_validation_collator_uses_train_controls_only(rows, binding):
    validation = rows.for_split('validation')
    collator, report = ready(validation, binding)
    assert report['rows'][0]['sample_id'] == 'global-0'
    assert collator([validation[0]])['labels'].shape[0] == 1
    with pytest.raises(ValueError, match='bound dataset'): collator([rows[0]])


def test_collator_rgb_only_hot_path_and_no_writes(rows, binding, monkeypatch):
    collator, _ = ready(rows, binding)
    record = rows[0]
    original = data.read_bounded
    reads = []
    def checked_read(path):
        reads.append(str(path))
        assert str(path).endswith('rgb.png')
        return original(path)
    monkeypatch.setattr(export.bundle, 'read_bounded', checked_read)
    active = [True]
    def audit(event, args):
        if not active[0]: return
        if event in ('socket.connect', 'socket.getaddrinfo', 'os.mkdir', 'os.remove', 'os.rename'):
            raise AssertionError('Unexpected side effect')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1:3]
            assert not (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            assert not str(args[0]).endswith(('.ghn', 'label.json', 'safetensors'))
    sys.addaudithook(audit)
    try:
        collator([record])
    finally:
        active[0] = False
    assert len(reads) == 1


@pytest.mark.parametrize('field', ['query_pixel_uv', 'crop_box', 'full_rgb', 'messages', 'sample_id'])
def test_passed_record_cannot_be_rebound(rows, binding, field):
    collator, _ = ready(rows, binding)
    record = rows[0]
    if field == 'query_pixel_uv': record[field] = [0., 0.]
    if field == 'crop_box': record[field] = [0, 0, 768, 768]
    if field == 'full_rgb': record[field] = '../outside.png'
    if field == 'messages': record[field][2]['content'][0]['text'] = '{}'
    if field == 'sample_id': record[field] = 'missing'
    with pytest.raises(ValueError): collator([record])


def test_collator_single_example_and_declared_keys(rows, binding):
    collator, _ = ready(rows, binding)
    for batch in ([], [rows[0], rows[0]], rows[0]):
        with pytest.raises(ValueError, match='one native'): collator(batch)
    other = data.NativeCollator(rows, **{**binding, 'model_input_keys': tuple(data._CORE)})
    with pytest.raises(ValueError, match='model-input keys'):
        data.check_numpy_torch_parity(other, sample_ids=['global-0'])
    for keys in (None, ['input_ids'], [*KEYS, 'depth'], [*KEYS, 'input_ids']):
        with pytest.raises(ValueError): data.NativeCollator(rows, **{**binding, 'model_input_keys': keys})


@pytest.mark.parametrize('budget', [None, False, 0, -1, '10000', 10000., 9999])
def test_explicit_bound_budget(rows, binding, budget):
    with pytest.raises(ValueError): data.NativeCollator(rows, **{**binding, 'maximum_tokens': budget})


def rebound_report(binding, mutate):
    path = Path(binding['preflight_report'].path)
    report = export._parse(path.read_bytes())
    mutate(report)
    path.write_bytes(export._json(report))
    return {**binding, 'preflight_report': export.FilePin(str(path), export._file_sha(path))}


def test_exact_budget_and_one_token_overflow(rows, binding):
    report = export._parse(Path(binding['preflight_report'].path).read_bytes())
    total = report['rows'][0]['total_tokens']
    def exact(r):
        r['maximum_tokens'] = total
        r['rows'][0]['maximum_tokens'] = total
    exact_binding = rebound_report(binding, exact)
    collator, _ = ready(rows, {**exact_binding, 'maximum_tokens': total})
    assert collator([rows[0]])['input_ids'].shape[1] == total
    # A longer otherwise valid processor response may NOT be silently truncated.
    collator.processor.mode = 'longer'
    with pytest.raises(ValueError, match='never truncate'): collator([rows[0]])


def test_budget_and_model_keys_cannot_change_after_parity(rows, binding):
    collator, _ = ready(rows, binding)
    collator.maximum_tokens += 1
    with pytest.raises(ValueError, match='Bound budget'): collator([rows[0]])
    collator.maximum_tokens -= 1
    collator.model_input_keys = frozenset(data._CORE)
    with pytest.raises(ValueError, match='Bound budget'): collator([rows[0]])


def test_rgb_source_checked_in_collation(package, binding, tmp_path):
    root = tmp_path / 'rgb_change'; shutil.copytree(package[0], root)
    copied = data.NativeRows(root, manifest_sha256=package[1])
    collator, _ = ready(copied, binding)
    record = copied[0]
    (root / record['full_rgb']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='RGB source pin'): collator([record])
    copied.close()


@pytest.mark.parametrize('field,value', [
    ('schema', 'legacy'), ('release_manifest_sha256', '0' * 64),
    ('preflight_source_sha256', '0' * 64), ('release_implementation_bindings', {}),
    ('processor_files_sha256', '0' * 64), ('runtime_versions', {}), ('python_version', 'other'),
    ('effective_tokenizer_sha256', '0' * 64), ('chat_template_sha256', '0' * 64),
    ('image_processor_config', {}), ('processor_implementation_bindings', {}),
    ('weights_loaded', True), ('training_approved', True), ('rows', []), ('scope', 'all20k')])
def test_rebound_report_incompatible_fields_held(rows, binding, field, value):
    changed = rebound_report(binding, lambda r: r.__setitem__(field, value))
    with pytest.raises(ValueError): data.NativeCollator(rows, **changed)


def test_report_and_snapshot_byte_tampering(rows, binding):
    path = Path(binding['preflight_report'].path)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='pin changed'): data.NativeCollator(rows, **binding)


def test_snapshot_and_effective_tokenizer_reaudit(rows, binding):
    collator, _ = ready(rows, binding)
    collator.processor.tokenizer.backend_tokenizer = None
    with pytest.raises(ValueError, match='tokenizer'): collator.verify_sources()
    assert not collator._ready
    collator.processor.tokenizer.backend_tokenizer = collator.processor.tokenizer
    (binding['model_snapshot'] / 'tokenizer.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='snapshot pin'): collator.verify_sources()


def test_rebound_evidence_detected_by_fresh_numpy(rows, binding):
    changed = rebound_report(binding, lambda r: r['rows'][0].__setitem__('labels_sha256', '0' * 64))
    collator = data.NativeCollator(rows, **changed)
    with pytest.raises(ValueError, match='Fresh NumPy'): data.check_numpy_torch_parity(collator, sample_ids=['global-0'])


@pytest.mark.parametrize('ids', [[], ['global-1'], ['global-2'], ['global-0', 'global-0'], None, 'global-0'])
def test_parity_requires_explicit_report_train_ids(rows, binding, ids):
    collator = data.NativeCollator(rows, **binding)
    with pytest.raises(ValueError): data.check_numpy_torch_parity(collator, sample_ids=ids)


@pytest.mark.parametrize('raw', ['{}', '```json\n{}\n```', '{"status":1,"status":2}',
    '{"status":"localized","visibility":"clear","next_action":"inspect_cut_region","cut_point_uv":[1000,1]}',
    '{"status":"localized","visibility":"clear","next_action":"inspect_cut_region","cut_point_uv":[NaN,1]}'])
def test_native_decode_strict(raw):
    with pytest.raises(ValueError): data.decode_answer(raw)


def test_native_decode_axes_and_abstention():
    answer = dict(status='localized', visibility='clear', next_action='inspect_cut_region', cut_point_uv=[500., 500.])
    assert data.decode_answer(json.dumps(answer))['cut_point_uv'] == [848., 408.]
    abstain = dict(status='abstain', visibility='occluded', next_action='change_viewpoint', cut_point_uv=None)
    assert data.decode_answer(json.dumps(abstain)) == abstain


def test_20k_offset_metadata_only(tmp_path, monkeypatch):
    # Scaling exercise at the validator I/O seam, NOT a real release/admission.
    root = tmp_path / 'metadata_only'; (root / 'splits').mkdir(parents=True)
    records = {s: [dict(sample_id=f'{s}-{i}', split=s) for i in range(n)]
               for s, n in (('train', 20000), ('validation', 2), ('test', 2))}
    files = {}
    for split, values in records.items():
        path = root / f'splits/{split}.jsonl'
        path.write_bytes(b''.join(export._json(v) + b'\n' for v in values))
        files[f'splits/{split}.jsonl'] = export._file_sha(path)
    (root / 'manifest.json').write_bytes(export._json(dict(files=files)))
    from .test_native_clear_export import frozen
    calls = []
    def validation(*args, **kwargs):
        calls.append(True)
        return dict(records=records, counts={s: len(v) for s, v in records.items()}, frozen_splits=frozen())
    monkeypatch.setattr(export, 'validate', validation)
    rows = data.NativeRows(root, manifest_sha256=export._file_sha(root / 'manifest.json'))
    assert len(rows) == 20000 and rows[19999] == records['train'][19999]
    assert rows[42] == records['train'][42] and len(calls) == 1
    assert 'records' not in rows.__dict__ and rows._indices['train'][42][0] > 0
    # No adapter-side sharding: normal sampler decides coverage, order, padding.
    a = list(torch.utils.data.DistributedSampler(rows, num_replicas=2, rank=0, shuffle=False))
    b = list(torch.utils.data.DistributedSampler(rows, num_replicas=2, rank=1, shuffle=False))
    assert not set(a) & set(b) and sorted(a + b) == list(range(20000))
    rows.close()
