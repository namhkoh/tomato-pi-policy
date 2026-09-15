"""Fake processor tests ONLY: no AutoProcessor/model snapshot is loaded here.

Reuse the frozen exporter's synthetic native-buffer fixture without editing it.
run() tests stub release validation and AutoProcessor at their I/O boundaries;
they do not fabricate an admitted 20k release. The processor fake intentionally
has tiny grids, unrelated to actual Qwen counts. Network denial tests intercept
Python audit events before any network operation. All files are tmp_path data.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from . import native_clear_processor_preflight as preflight
from . import native_clear_export as export
from .bundle import SampleReader
from .test_native_clear_export import synthetic


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(export._json(value))


def tree(root):
    return {p.relative_to(root).as_posix(): export._file_sha(p) for p in root.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    task = tmp_path_factory.mktemp('processor_fake_native')
    compact, row = synthetic(task / 'source', 0)
    root = task / 'package'
    shutil.copytree(compact, root / 'samples/00000000')
    reader = SampleReader(compact)
    label = reader.json('supervision/label.json')
    rgb = export._png(reader.read('inputs/rgb.png'), (1696, 816), 'RGB')
    record = export._record(0, row, reader, label, rgb)
    (root / 'crops').mkdir()
    rgb.crop(record['crop_box']).save(root / record['query_crop'])
    # Deliberately NOT a real release; only a post-check file inventory fixture.
    manifest = dict(fixture='fake_processor_not_admission', files=tree(root),
                    implementation_bindings=dict(export._LOADED_CODE))
    write_json(root / 'manifest.json', manifest)
    return root, record, export._file_sha(root / 'manifest.json')


class FakeTokenizer:
    def __init__(self, mode='ok'):
        self.mode = mode
        self.backend_tokenizer = self

    def to_str(self):
        return '{"fixture":"character_fake_not_a_Qwen_tokenizer"}'

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        if self.mode == 'bad_decode' and skip_special_tokens:
            return '{}'
        return ''.join(chr(i - 1000) if i >= 1000 else '' if skip_special_tokens else f'<special:{i}>' for i in ids)


class FakeImageProcessor:
    patch_size = 2
    temporal_patch_size = 2
    merge_size = 2

    def __init__(self, mode='ok'):
        self.mode, self.calls = mode, 0

    def to_dict(self):
        return dict(patch_size=self.patch_size, temporal_patch_size=self.temporal_patch_size,
                    merge_size=self.merge_size, fixture='tiny_grids_not_actual_Qwen',
                    changed=self.mode == 'settings_changed' and self.calls > 0)

    def __call__(self, *, images, return_tensors):
        assert return_tensors == 'np'
        assert [image.size for image in images] == [(1696, 816), (768, 768)]
        assert all(image.mode == 'RGB' for image in images)
        self.calls += 1
        pixels = np.concatenate([np.full((count, 24), np.asarray(image)[0, 0].sum() / 255., dtype=np.float32)
                                 for image, count in zip(images, (32, 16))])
        if self.mode == 'direct_mismatch' and self.calls % 3 == 0:
            pixels[0, 0] += 1
        return dict(pixel_values=pixels, image_grid_thw=np.array([[1, 4, 8], [1, 4, 4]], dtype=np.int64))


def encode(text):
    return [1000 + ord(c) for c in text]


class Qwen3VLProcessor:
    """Name matches loader's class guard; fake is installed ONLY by tests."""
    image_token_id = 100
    chat_template = 'SYNTHETIC TEST TEMPLATE'

    def __init__(self, mode='ok'):
        self.mode = mode
        self.image_processor = FakeImageProcessor(mode)
        self.tokenizer = FakeTokenizer(mode)

    def apply_chat_template(self, messages, *, tokenize, return_dict, return_tensors,
                            padding, truncation, add_generation_prompt):
        assert tokenize and return_dict and return_tensors == 'np'
        assert padding is False and truncation is False
        assert [m['role'] for m in messages] == (['system', 'user'] if add_generation_prompt else ['system', 'user', 'assistant'])
        assert [p['type'] for p in messages[1]['content']] == ['image', 'image', 'text']
        images = [item['image'] for item in messages[1]['content'][:2]]
        assert 'world_m' not in messages[1]['content'][-1]['text']
        assert 'depth' not in messages[1]['content'][-1]['text']
        image_data = self.image_processor(images=images, return_tensors='np')
        system = '' if self.mode == 'missing_system' else messages[0]['content'][0]['text']
        user = messages[1]['content'][-1]['text']
        answer = export._json(dict(status='localized', cut_point_uv=[620.91, 307.16],
                    visibility='clear', next_action='inspect_cut_region')).decode()
        if self.mode == 'answer_leak':
            # Match exact answer string/order used by exporter, in both calls.
            answer = json.dumps(dict(cut_point_uv=[620.91, 307.16], next_action='inspect_cut_region',
                                status='localized', visibility='clear'), separators=(',', ':'))
            user += answer
        blocks = [10] + [100] * 8 + [11, 10] + [100] * 4 + [11]
        if self.mode == 'wrong_image_tokens': blocks = [10] + [100] * 12 + [11]
        values = [1] + encode(system) + blocks + encode(user) + [2]
        if not add_generation_prompt:
            values += encode(messages[2]['content'][0]['text']) + [3]
            if self.mode == 'image_in_answer': values.append(100)
        ids = np.array([values], dtype=np.int64)
        if self.mode == 'wrong_prefix' and add_generation_prompt: ids[0, 0] += 1
        if self.mode == 'truncate' and not add_generation_prompt: ids = ids[:, :-10]
        output = dict(input_ids=ids, attention_mask=np.ones_like(ids),
                      mm_token_type_ids=(ids == 100).astype(np.int64), **image_data)
        if self.mode == 'padding': output['attention_mask'][0, -1] = 0
        if self.mode == 'float_ids': output['input_ids'] = ids.astype(float)
        if self.mode == 'batch_two': output['input_ids'] = np.repeat(ids, 2, axis=0)
        if self.mode == 'missing_grid': del output['image_grid_thw']
        if self.mode == 'one_grid': output['image_grid_thw'] = output['image_grid_thw'][:1]
        if self.mode == 'video': output['video_grid_thw'] = np.array([[1, 4, 4]])
        if self.mode == 'fractional_grid': output['image_grid_thw'] = output['image_grid_thw'].astype(float)
        if self.mode == 'zero_grid': output['image_grid_thw'][0, 0] = 0
        if self.mode == 'unmerged_grid': output['image_grid_thw'][0, 1] = 3
        if self.mode == 'bad_pixels': output['pixel_values'] = output['pixel_values'][:, :-1]
        if self.mode == 'nan_pixels': output['pixel_values'][0, 0] = np.nan
        if self.mode == 'prefix_pixels' and add_generation_prompt: output['pixel_values'][0, 0] += 1
        if self.mode == 'prefix_types' and add_generation_prompt: output['mm_token_type_ids'][0, 0] = 1
        if self.mode == 'mutate_image': images[0].putpixel((0, 0), (123, 124, 125))
        return output


def check(native, mode='ok', budget=10000):
    root, record, _ = native
    return preflight.check_record(root, record, Qwen3VLProcessor(mode), maximum_tokens=budget)


def test_native_grids_coordinate_budget_and_assistant_mask(native):
    root, record, _ = native
    assert SampleReader(root / record['compact_root']).metadata['supervision']['split_group'] == 'seed101_full'
    before = tree(root)
    result = check(native)
    assert result['image_grid_thw'] == [[1, 4, 8], [1, 4, 4]]
    assert result['image_tokens'] == [8, 4]
    assert result['native_sizes'] == [[1696, 816], [768, 768]]
    assert result['processed_sizes_wh'] == [[16, 8], [8, 8]]  # Fake internal resize, not new label coordinates.
    assert result['normalized_answer']['cut_point_uv'] == [620.91, 307.16]
    assert result['crop_box'] == [823, 0, 1591, 768]
    assert max(result['answer_roundtrip_error_px']) == pytest.approx(.00664)
    assert result['total_tokens'] == result['prompt_tokens'] + result['assistant_tokens']
    assert result['assistant_span'] == [result['prompt_tokens'], result['total_tokens']]
    assert result['assistant_mask_verified'] and result['independent_image_processing_verified']
    assert result['record_sha256'] == export.json_sha256(record)
    exact = check(native, budget=result['total_tokens'])
    assert exact['labels_sha256'] == result['labels_sha256']
    with pytest.raises(ValueError, match='never truncate'):
        check(native, budget=result['total_tokens'] - 1)
    assert tree(root) == before


@pytest.mark.parametrize('budget', [None, 0, -1, True, 2048.0, '4096'])
def test_explicit_budget_required(native, budget):
    with pytest.raises(ValueError, match='maximum_tokens'):
        check(native, budget=budget)


@pytest.mark.parametrize('mode', ['wrong_prefix', 'padding', 'float_ids', 'batch_two', 'missing_grid',
    'one_grid', 'video', 'fractional_grid', 'zero_grid', 'unmerged_grid', 'bad_pixels', 'nan_pixels',
    'prefix_pixels', 'prefix_types', 'wrong_image_tokens', 'image_in_answer', 'bad_decode',
    'missing_system', 'answer_leak', 'direct_mismatch', 'mutate_image', 'truncate'])
def test_fake_processors_fail_closed(native, mode):
    with pytest.raises(ValueError): check(native, mode)


@pytest.mark.parametrize('change', ['heldout', 'crop', 'query', 'legacy_answer'])
def test_native_binding_errors(native, change):
    root, record, _ = native
    record = deepcopy(record)
    if change == 'heldout': record['split'] = 'test'
    if change == 'crop': record['crop_box'] = [0, 0, 768, 768]
    if change == 'query': record['query_pixel_uv'] = [12., 12.]
    if change == 'legacy_answer':
        answer = export._parse(record['messages'][2]['content'][0]['text'])
        answer['cut_point_uv'] = [310.45, 153.58]
        record['messages'][2]['content'][0]['text'] = export._json(answer).decode()
    with pytest.raises(ValueError):
        preflight.check_record(root, record, Qwen3VLProcessor(), maximum_tokens=10000)


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / 'local_processor'
    write_json(root / 'config.json', dict(model_type='qwen3_vl', image_token_id=100,
        text_config=dict(hidden_size=4096, num_hidden_layers=36)))
    write_json(root / 'tokenizer.json', dict(fixture='not a real tokenizer'))
    write_json(root / 'tokenizer_config.json', dict(chat_template='SYNTHETIC TEST TEMPLATE'))
    write_json(root / 'preprocessor_config.json', FakeImageProcessor().to_dict())
    return root, tree(root)


def fake_runtime(monkeypatch, native, processor=None, callback=None):
    root, record, manifest_sha = native
    calls = []
    def validation(given_root, *, manifest_sha256):
        assert Path(given_root) == root and manifest_sha256 == manifest_sha
        return dict(records=dict(train=[deepcopy(record)], validation=[], test=[]))
    monkeypatch.setattr(export, 'validate', validation)
    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert Path(path).is_absolute()
            assert kwargs == dict(local_files_only=True, trust_remote_code=False)
            assert all(os.environ.get(k) == v for k, v in preflight._OFFLINE.items())
            calls.append(path)
            if callback: callback()
            return processor or Qwen3VLProcessor()
    monkeypatch.setattr(preflight, '_auto_processor', lambda: AutoProcessor)
    return calls


def run(native, snapshot, **overrides):
    root, record, sha = native
    model, pins = snapshot
    args = dict(manifest_sha256=sha, model_snapshot=str(model), processor_files=pins,
                sample_ids=[record['sample_id']], maximum_tokens=10000)
    args.update(overrides)
    return preflight.run(str(root), **args)


def test_offline_run_report_and_provenance(native, snapshot, monkeypatch):
    calls = fake_runtime(monkeypatch, native)
    before = [tree(native[0]), tree(snapshot[0])]
    environment = {name: os.environ.get(name) for name in preflight._OFFLINE}
    bytecode = sys.dont_write_bytecode
    report = run(native, snapshot)
    assert len(calls) == 1
    assert report['schema'] == preflight.SCHEMA
    assert report['release_manifest_sha256'] == native[2]
    assert report['release_implementation_bindings']['native_dataset/native_clear_export.py'] == export._file_sha(export.__file__)
    assert report['processor_files'] == snapshot[1]
    assert report['processor_files_sha256'] == export.json_sha256(snapshot[1])
    assert report['chat_template_sha256'] == hashlib.sha256(Qwen3VLProcessor.chat_template.encode()).hexdigest()
    assert report['effective_tokenizer_sha256'] == hashlib.sha256(FakeTokenizer().to_str().encode()).hexdigest()
    assert all(report[name] is False for name in ('weights_loaded', 'weight_integrity_verified',
               'training_performed', 'depth_is_model_input', 'route_integration_complete', 'training_approved'))
    assert report['maximum_observed_total_tokens'] == report['rows'][0]['total_tokens']
    assert len(report['processor_implementation_bindings']) == 6
    assert {name: os.environ.get(name) for name in preflight._OFFLINE} == environment
    assert sys.dont_write_bytecode == bytecode
    assert [tree(native[0]), tree(snapshot[0])] == before


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'wrong_hash', 'weight_pin', 'remote_id',
    'wrong_model', 'wrong_size', 'quantized', 'no_image_id', 'nested_template', 'external_tokenizer'])
def test_snapshot_failures_before_loader(native, snapshot, monkeypatch, mutation):
    calls = fake_runtime(monkeypatch, native)
    root, pins = snapshot
    pins = dict(pins)
    if mutation == 'missing': del pins['tokenizer.json']
    if mutation == 'extra': write_json(root / 'chat_template.json', dict(template='unbound'))
    if mutation == 'wrong_hash': pins['config.json'] = '0' * 64
    if mutation == 'weight_pin': pins['model.safetensors'] = '0' * 64
    if mutation in ('wrong_model', 'wrong_size', 'quantized', 'no_image_id'):
        cfg = export._parse((root / 'config.json').read_bytes())
        if mutation == 'wrong_model': cfg['model_type'] = 'qwen3_vl_moe'
        if mutation == 'wrong_size': cfg['text_config']['num_hidden_layers'] = 40
        if mutation == 'quantized': cfg['quantization_config'] = dict(quant_method='fp8')
        if mutation == 'no_image_id': del cfg['image_token_id']
        write_json(root / 'config.json', cfg); pins['config.json'] = export._file_sha(root / 'config.json')
    if mutation == 'nested_template': (root / 'chat_templates').mkdir()
    if mutation == 'external_tokenizer':
        write_json(root / 'tokenizer_config.json', dict(tokenizer_file='../outside.json'))
        pins['tokenizer_config.json'] = export._file_sha(root / 'tokenizer_config.json')
    if mutation == 'remote_id': root = 'Qwen/Qwen3-VL-8B-Instruct'
    with pytest.raises(ValueError): run(native, (root, pins))
    assert not calls


@pytest.mark.parametrize('ids', [[], ['missing'], ['global-0', 'global-0'], None, 'global-0'])
def test_explicit_train_ids_required_before_loading(native, snapshot, monkeypatch, ids):
    calls = fake_runtime(monkeypatch, native)
    with pytest.raises(ValueError): run(native, snapshot, sample_ids=ids)
    assert not calls


@pytest.mark.parametrize('mode', ['network', 'weight', 'unpinned', 'write', 'outside_write', 'mkdir'])
def test_loader_guard_fails_before_access(native, snapshot, monkeypatch, mode):
    root, _ = snapshot
    if mode == 'weight': (root / 'model.safetensors').write_bytes(b'FAKE SENTINEL NOT MODEL WEIGHTS')
    if mode == 'unpinned': (root / 'unbound.txt').write_text('fixture')
    before = tree(root)
    def attempt():
        if mode == 'network': sys.audit('socket.connect', None, ('must-not-connect.invalid', 443))
        if mode == 'weight': (root / 'model.safetensors').read_bytes()
        if mode == 'unpinned': (root / 'unbound.txt').read_bytes()
        if mode == 'write': (root / 'tokenizer.json').write_bytes(b'changed')
        if mode == 'outside_write': (root.parent / 'forbidden.txt').write_text('changed')
        if mode == 'mkdir': (root.parent / 'forbidden_directory').mkdir()
    fake_runtime(monkeypatch, native, callback=attempt)
    environment = {name: os.environ.get(name) for name in preflight._OFFLINE}
    with pytest.raises(RuntimeError): run(native, snapshot)
    assert {name: os.environ.get(name) for name in preflight._OFFLINE} == environment
    assert tree(root) == before


@pytest.mark.parametrize('mode', ['class', 'token', 'template', 'settings_changed', 'slow_tokenizer'])
def test_runtime_processor_identity_and_settings_held(native, snapshot, monkeypatch, mode):
    processor = Qwen3VLProcessor(mode)
    if mode == 'class':
        class Wrong(Qwen3VLProcessor): pass
        processor = Wrong()
    if mode == 'token': processor.image_token_id = 101
    if mode == 'template': processor.chat_template = {'default': 'ambiguous'}
    if mode == 'slow_tokenizer': processor.tokenizer.backend_tokenizer = None
    fake_runtime(monkeypatch, native, processor)
    with pytest.raises(ValueError): run(native, snapshot)


def test_cli_has_no_implicit_budget_or_model_load_path(capsys):
    with pytest.raises(SystemExit) as failure: preflight.main([])
    assert failure.value.code == 2
    with pytest.raises(SystemExit) as help_exit: preflight.main(['--help'])
    assert help_exit.value.code == 0
    assert '--maximum-tokens' in capsys.readouterr().out
