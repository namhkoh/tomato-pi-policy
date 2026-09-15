"""CPU boundary tests only, NOT a model/processor/Trainer/GPU qualification.

Fake release documents below are deliberately invalid production releases;
NativeRows is stubbed only at the explicit validation I/O seam. Tests assert
that the seam is mandatory. Existing exporter/adapter suites own real semantic
validation tests. Fake 'weights' are tiny text sentinels in tmp_path and are
NEVER deserialized. The simulated execution test uses a fake Trainer with no
optimization, torchrun, process group, networking, model or device calls.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from . import native_h200_train as runner
from . import native_training_data as data
from . import native_clear_export as export
from . import native_clear_processor_preflight as preflight


KEYS = ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else export._json(value))
    return export._file_sha(path)


class FakeRows:
    def __init__(self, root, *, manifest_sha256, split='train'):
        assert split == 'train'
        self.root, self.manifest_sha256, self.split = root, manifest_sha256, split
        self.closed = False
    @property
    def receipt(self):
        return dict(release_implementation_bindings=dict(export._LOADED_CODE),
                    fixture='FAKE_VALIDATION_SEAM_NOT_AN_ADMISSION')
    def close(self): self.closed = True
    def for_split(self, split):
        assert split == 'validation'
        result = FakeRows(self.root, manifest_sha256=self.manifest_sha256)
        result.split = split
        return result


@pytest.fixture
def inputs(tmp_path):
    root, model, pinsdir = (tmp_path / n for n in ('dataset', 'model', 'pins'))
    admission = write(root / 'provenance/admission.json', dict(fixture='NOT_AN_ADMISSION'))
    manifest = write(root / 'manifest.json', dict(fixture='NOT_A_RELEASE', provenance=dict(admission=admission)))
    config = dict(model_type='qwen3_vl', architectures=['Qwen3VLForConditionalGeneration'], image_token_id=100,
        text_config=dict(hidden_size=4096, num_hidden_layers=36, max_position_embeddings=32768))
    processor_files = {
        'config.json': write(model / 'config.json', config),
        'tokenizer_config.json': write(model / 'tokenizer_config.json', {}),
        'tokenizer.json': write(model / 'tokenizer.json', dict(fixture='FAKE_TOKENIZER')),
        'preprocessor_config.json': write(model / 'preprocessor_config.json', dict(fixture='FAKE_PROCESSOR'))}
    files = {'model.safetensors': write(model / 'model.safetensors', b'NOT WEIGHTS - TEST SENTINEL')}
    filemap = write(pinsdir / 'model-files.json', files)
    report = dict(schema=preflight.SCHEMA, state='checked_rows_processor_only', release_manifest_sha256=manifest,
        maximum_tokens=10000, preflight_source_sha256=preflight._CODE_SHA,
        release_implementation_bindings=dict(export._LOADED_CODE), runtime_versions=data._versions(),
        python_version=runner.sys.version, rows=[dict(sample_id='train-control')],
        processor_files=processor_files, processor_files_sha256=export.json_sha256(processor_files),
        offline=True, weights_loaded=False, weight_integrity_verified=False, training_performed=False,
        depth_is_model_input=False, route_integration_complete=False, training_approved=False,
        scope='explicit_train_rows_known_answers_not_unchecked_rows_or_future_generations',
        fixture='FAKE_PREFLIGHT_NOT_QUALIFICATION')
    report_sha = write(pinsdir / 'preflight.json', report)
    zero = write(pinsdir / 'zero2.json', runner.ZERO2)
    argv = ['--dataset', str(root), '--model', str(model), '--output', str(tmp_path / 'new-run'),
        '--manifest-sha256', manifest, '--admission-sha256', admission,
        '--model-files', str(pinsdir / 'model-files.json'), '--model-files-sha256', filemap,
        '--preflight-report', str(pinsdir / 'preflight.json'), '--preflight-sha256', report_sha,
        '--deepspeed', str(pinsdir / 'zero2.json'), '--deepspeed-sha256', zero,
        '--maximum-tokens', '10000', '--control-id', 'train-control', '--mode', 'smoke']
    for key in KEYS: argv += ['--model-input-key', key]
    return runner.arguments(argv), argv


@pytest.fixture
def prepared(inputs, monkeypatch):
    monkeypatch.setattr(data, 'NativeRows', FakeRows)
    result = runner.prepare(inputs[0])
    yield result
    result.rows.close()


def test_defaults_and_readonly_cli_boundary(inputs, monkeypatch, capsys):
    args, argv = inputs
    assert args.expected_world_size == 4 and args.gradient_accumulation == 8 and args.epochs == 3
    assert args.learning_rate == 1e-5 and args.vision_learning_rate == 1e-6
    assert not args.execute_training
    monkeypatch.setattr(data, 'NativeRows', FakeRows)
    def forbidden(*a, **k): raise AssertionError('runtime/model boundary crossed')
    monkeypatch.setattr(runner, '_runtime', forbidden)
    monkeypatch.setattr(runner, 'execute', forbidden)
    runner.main(argv)
    report = json.loads(capsys.readouterr().out)
    assert report['state'] == 'configuration_checked_only'
    assert not report['model_loaded'] and not report['training_performed'] and not report['training_approved']
    assert report['effective_examples_per_step'] == 32
    assert not args.output.exists()


def test_cli_requires_budget_and_pins_and_has_no_legacy_switches(inputs):
    _, argv = inputs
    for flag in ('--maximum-tokens', '--manifest-sha256', '--admission-sha256', '--preflight-sha256', '--model-files-sha256'):
        changed = list(argv); i = changed.index(flag); del changed[i:i+2]
        with pytest.raises(SystemExit): runner.arguments(changed)
    for extra in (['--depth-input'], ['--method', 'lora'], ['--resume'], ['--wandb-project', 'x'], ['--coordinates', 'pixels']):
        with pytest.raises(SystemExit): runner.arguments(argv + extra)


@pytest.mark.parametrize('field,value', [('maximum_tokens', 0), ('maximum_tokens', True),
    ('gradient_accumulation', 0), ('gradient_accumulation', False), ('expected_world_size', -1),
    ('epochs', float('nan')), ('epochs', 0), ('learning_rate', float('inf')),
    ('vision_learning_rate', 0), ('seed', -1), ('execute_training', 1), ('mode', 'evaluate'),
    ('control_id', []), ('control_id', ['x', 'x']), ('model_input_key', KEYS + ['depth']),
    ('model_input_key', KEYS + ['labels']), ('model_input_key', ['input_ids'])])
def test_argument_failures(inputs, field, value):
    args, _ = inputs
    setattr(args, field, value)
    with pytest.raises(ValueError): runner.validate_arguments(args)


@pytest.mark.parametrize('target', ['existing', 'dataset', 'inside_dataset', 'above_dataset', 'model', 'pins', 'relative'])
def test_output_and_input_paths_fail_closed(inputs, target):
    args, _ = inputs
    if target == 'existing': args.output.mkdir()
    if target == 'dataset': args.output = args.dataset
    if target == 'inside_dataset': args.output = args.dataset / 'new'
    if target == 'above_dataset': args.output = args.dataset.parent
    if target == 'model': args.output = args.model / 'new'
    if target == 'pins': args.output = args.preflight_report
    if target == 'relative': args.dataset = Path('.')
    with pytest.raises(ValueError): runner.check_paths(args)


def test_actual_native_validation_not_bypassed_even_for_smoke(inputs, monkeypatch):
    args, _ = inputs
    # The fake package is invalid: use the REAL NativeRows/export validator.
    with pytest.raises(ValueError): runner.prepare(args)
    assert not args.output.exists()
    calls = []
    def reject(*a, **kw):
        calls.append(kw)
        raise ValueError('native admission held')
    monkeypatch.setattr(data, 'NativeRows', reject)
    monkeypatch.setattr(runner, 'check_model_files', lambda *a: pytest.fail('weights read before admission'))
    with pytest.raises(ValueError, match='admission held'): runner.prepare(args)
    assert calls == [dict(manifest_sha256=args.manifest_sha256, split='train')]


@pytest.mark.parametrize('field', ['manifest_sha256', 'admission_sha256', 'preflight_sha256', 'model_files_sha256', 'deepspeed_sha256'])
def test_all_external_pins_required(inputs, monkeypatch, field):
    args, _ = inputs; setattr(args, field, '0' * 64)
    monkeypatch.setattr(data, 'NativeRows', FakeRows)
    with pytest.raises(ValueError): runner.prepare(args)
    assert not args.output.exists()


@pytest.mark.parametrize('field,value', [('maximum_tokens', 9999), ('schema', 'legacy'),
    ('release_manifest_sha256', '0' * 64), ('preflight_source_sha256', '0' * 64),
    ('release_implementation_bindings', {}), ('runtime_versions', {}), ('python_version', 'other'), ('rows', []),
    ('state', 'approved'), ('training_approved', True), ('weights_loaded', True), ('scope', 'all20k'),
    ('processor_files_sha256', '0' * 64)])
def test_rebound_preflight_headers_held(inputs, monkeypatch, field, value):
    args, _ = inputs
    report = export._parse(args.preflight_report.read_bytes()); report[field] = value
    args.preflight_sha256 = write(args.preflight_report, report)
    monkeypatch.setattr(data, 'NativeRows', FakeRows)
    with pytest.raises(ValueError): runner.prepare(args)


@pytest.mark.parametrize('mutation', ['architecture', 'remote_code', 'context_missing', 'context_small', 'wrong_size', 'quantized'])
def test_model_config_boundary(inputs, monkeypatch, mutation):
    args, _ = inputs
    cfg = export._parse((args.model / 'config.json').read_bytes())
    if mutation == 'architecture': cfg['architectures'] = ['OtherModel']
    if mutation == 'remote_code': cfg['auto_map'] = {'AutoModel': 'custom.code'}
    if mutation == 'context_missing': del cfg['text_config']['max_position_embeddings']
    if mutation == 'context_small': cfg['text_config']['max_position_embeddings'] = 9999
    if mutation == 'wrong_size': cfg['text_config']['hidden_size'] = 1024
    if mutation == 'quantized': cfg['quantization_config'] = {'method': 'fp8'}
    config_sha = write(args.model / 'config.json', cfg)
    report = export._parse(args.preflight_report.read_bytes())
    report['processor_files']['config.json'] = config_sha
    report['processor_files_sha256'] = export.json_sha256(report['processor_files'])
    args.preflight_sha256 = write(args.preflight_report, report)
    monkeypatch.setattr(data, 'NativeRows', FakeRows)
    with pytest.raises(ValueError): runner.prepare(args)


@pytest.mark.parametrize('mutation', ['stage3', 'offload', 'batch', 'bf16_off', 'fp16'])
def test_zero2_config_is_explicit_and_narrow(inputs, monkeypatch, mutation):
    args, _ = inputs; value = deepcopy(runner.ZERO2)
    if mutation == 'stage3': value['zero_optimization']['stage'] = 3
    if mutation == 'offload': value['zero_optimization']['offload_optimizer'] = {'device': 'cpu'}
    if mutation == 'batch': value['train_micro_batch_size_per_gpu'] = 2
    if mutation == 'bf16_off': value['bf16']['enabled'] = False
    if mutation == 'fp16': value['fp16'] = {'enabled': True}
    args.deepspeed_sha256 = write(args.deepspeed, value)
    monkeypatch.setattr(data, 'NativeRows', FakeRows)
    with pytest.raises(ValueError, match='ZeRO-2'): runner.prepare(args)


@pytest.mark.parametrize('mutation', ['changed', 'extra', 'pickle', 'custom', 'adapter', 'generation_unpinned', 'escape'])
def test_model_file_inventory_failures(inputs, mutation):
    args, _ = inputs
    if mutation == 'changed': write(args.model / 'model.safetensors', b'changed sentinel')
    if mutation == 'extra': write(args.model / 'unlisted.safetensors', b'unlisted')
    if mutation == 'pickle': write(args.model / 'pytorch_model.bin', b'not pickle')
    if mutation == 'custom': write(args.model / 'custom.py', b'not executable')
    if mutation == 'adapter': write(args.model / 'adapter_config.json', {})
    if mutation == 'generation_unpinned': write(args.model / 'generation_config.json', {})
    if mutation == 'escape': args.model_files_sha256 = write(args.model_files, {'../model.safetensors': '0' * 64})
    with pytest.raises(ValueError): runner.check_model_files(args.model, export.FilePin(str(args.model_files), args.model_files_sha256))


def test_sharded_safetensors_exact_coverage(tmp_path):
    root = tmp_path / 'sharded'
    a, b = 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'
    files = {a: write(root / a, b'FAKE SHARD A'), b: write(root / b, b'FAKE SHARD B')}
    index = root / 'model.safetensors.index.json'
    files[index.name] = write(index, dict(weight_map={'a': a, 'b': b}))
    files['generation_config.json'] = write(root / 'generation_config.json', {})
    pins = tmp_path / 'pins.json'; sha = write(pins, files)
    assert runner.check_model_files(root, export.FilePin(str(pins), sha)) == files
    files[index.name] = write(index, dict(weight_map={'a': a}))
    sha = write(pins, files)
    with pytest.raises(ValueError, match='coverage'): runner.check_model_files(root, export.FilePin(str(pins), sha))


class ExplicitForward:
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels): pass


def test_forward_signature_does_not_trust_kwargs():
    runner.check_forward_keys(ExplicitForward, KEYS)
    with pytest.raises(ValueError): runner.check_forward_keys(ExplicitForward, KEYS + ['mm_token_type_ids'])
    class Ambiguous:
        def forward(self, input_ids, **kwargs): pass
    with pytest.raises(ValueError): runner.check_forward_keys(Ambiguous, KEYS)


@pytest.mark.parametrize('mutation', ['missing', 'unexpected', 'mismatched', 'errors', 'conversion', 'unknown', 'absent', 'none'])
def test_model_loading_diagnostics_fail_closed(mutation):
    info = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
    runner.check_loading_info(info)
    runner.check_loading_info(dict(info, conversion_errors={}))
    if mutation == 'missing': info['missing_keys'] = ['layer']
    if mutation == 'unexpected': info['unexpected_keys'] = ['layer']
    if mutation == 'mismatched': info['mismatched_keys'] = ['layer']
    if mutation == 'errors': info['error_msgs'] = ['error']
    if mutation == 'conversion': info['conversion_errors'] = {'layer': 'error'}
    if mutation == 'unknown': info['unknown_warning'] = []
    if mutation == 'absent': del info['missing_keys']
    if mutation == 'none': info['missing_keys'] = None
    with pytest.raises(ValueError): runner.check_loading_info(info)


def test_native_collators_bind_parity_for_train_and_validation(prepared, monkeypatch):
    prepared.args.mode = 'train'
    calls = []
    class Collator:
        def __init__(self, rows, processor, **kwargs):
            calls.append(('bind', rows.split, kwargs)); self.split, self.rows = rows.split, rows
    def parity(collator, *, sample_ids):
        calls.append(('parity', collator.split, sample_ids)); return {'fixture': True}
    monkeypatch.setattr(data, 'NativeCollator', Collator)
    monkeypatch.setattr(data, 'check_numpy_torch_parity', parity)
    dispatch, reports = runner.bind_processor(prepared, object(), ExplicitForward)
    assert set(dispatch.collators) == set(reports) == {'train', 'validation'}
    assert [c[0] for c in calls] == ['bind', 'parity', 'bind', 'parity']
    assert calls[0][2]['maximum_tokens'] == 10000
    assert calls[1][2] == ['train-control']
    dispatch.close()
    assert all(c.rows.closed for c in dispatch.collators.values())


@pytest.mark.parametrize('where', ['train_constructor', 'train_parity', 'validation_constructor', 'validation_parity', 'forward_keys'])
def test_bind_failure_closes_all_runner_owned_rows(prepared, monkeypatch, where):
    prepared.args.mode = 'train'
    opened = [prepared.rows]
    original_split = prepared.rows.for_split
    def split(name):
        result = original_split(name); opened.append(result); return result
    monkeypatch.setattr(prepared.rows, 'for_split', split)
    class Collator:
        def __init__(self, rows, processor, **kwargs):
            self.rows = rows
            if where == rows.split + '_constructor': raise ValueError('constructor held')
    def parity(collator, **kwargs):
        if where == collator.rows.split + '_parity': raise ValueError('parity held')
        return {}
    monkeypatch.setattr(data, 'NativeCollator', Collator)
    monkeypatch.setattr(data, 'check_numpy_torch_parity', parity)
    if where == 'forward_keys': prepared.args.model_input_key += ['mm_token_type_ids']
    with pytest.raises(ValueError): runner.bind_processor(prepared, object(), ExplicitForward)
    assert opened and all(rows.closed for rows in opened)


def test_split_dispatch_holds_test_and_multirow():
    dispatch = runner.SplitCollator({'train': lambda batch: 'train', 'validation': lambda batch: 'validation'})
    assert dispatch([{'split': 'train'}]) == 'train'
    assert dispatch([{'split': 'validation'}]) == 'validation'
    for batch in ([], [{'split': 'test'}], [{'split': 'train'}] * 2):
        with pytest.raises(ValueError): dispatch(batch)


@pytest.mark.parametrize('mode,steps,eval_strategy', [('smoke', 2, 'no'), ('overfit', 100, 'no'), ('train', -1, 'epoch')])
def test_training_options_are_unweighted_native_defaults(inputs, mode, steps, eval_strategy):
    args, _ = inputs; args.mode = mode
    config = runner.training_options(args, runner.ZERO2)
    assert config['per_device_train_batch_size'] == config['per_device_eval_batch_size'] == 1
    assert config['gradient_accumulation_steps'] == 8 and config['max_steps'] == steps
    assert config['bf16'] and not config['fp16'] and config['eval_strategy'] == eval_strategy
    assert config['label_names'] == ['labels'] and not config['remove_unused_columns']
    assert config['report_to'] == [] and not config['push_to_hub'] and config['dataloader_num_workers'] == 0
    assert config['logging_steps'] == 1 and not config['logging_nan_inf_filter']


def test_overfit_selection_lazy_deterministic_and_train_only():
    rows = [{'sample_id': str(i), 'split': 'train'} for i in range(100)]
    selected, ids = runner.training_rows(rows, 'overfit')
    assert len(selected) == len(ids) == 32 and len(set(ids)) == 32
    again, again_ids = runner.training_rows(list(reversed(rows)), 'overfit')
    assert ids == again_ids and list(selected) == list(again)
    assert runner.training_rows(rows, 'smoke') == (rows, None)
    assert runner.training_rows(rows, 'train') == (rows, None)
    with pytest.raises(ValueError): runner.training_rows(rows[:31], 'overfit')
    with pytest.raises(ValueError): runner.training_rows(rows, 'unknown')
    rows[42]['split'] = 'test'
    with pytest.raises(ValueError): runner.training_rows(rows, 'overfit')


class Parameter:
    def __init__(self): self.requires_grad = False
    def requires_grad_(self, flag): self.requires_grad = flag


class FakeParameterModel:
    config = SimpleNamespace(use_cache=True)
    def __init__(self):
        self.named = [(n, Parameter()) for n in ('model.visual.blocks.0.weight', 'model.visual.merger.weight',
            'model.visual.deepstack_merger.weight', 'model.language_model.layers.0.weight', 'lm_head.weight')]
    def named_parameters(self): return iter(self.named)


def test_full_ft_parameter_partition():
    model = FakeParameterModel()
    groups = runner.parameter_groups(model, 1e-5, 1e-6)
    assert [g['lr'] for g in groups] == [1e-5, 1e-6]
    assert [len(g['params']) for g in groups] == [4, 1]
    assert all(p.requires_grad for _, p in model.named_parameters())
    assert not set(map(id, groups[0]['params'])) & set(map(id, groups[1]['params']))
    model.named = model.named[:-1]
    with pytest.raises(ValueError): runner.parameter_groups(model, 1e-5, 1e-6)


def test_finite_loss_uses_existing_assistant_labels_and_no_item_count():
    finite, guard = runner.trainer_types(object, object, torch)
    obj = finite(); received = []
    inputs = {'labels': torch.tensor([[-100, 3, 4]])}
    def model(**kwargs): received.append(kwargs); return SimpleNamespace(loss=torch.tensor(2.))
    assert obj.compute_loss(model, inputs, num_items_in_batch=999).item() == 2
    assert received[0] == inputs and 'num_items_in_batch' not in received[0]
    for loss in (torch.tensor(float('nan')), torch.tensor(float('inf')), torch.tensor([1., 2.])):
        with pytest.raises(RuntimeError): obj.compute_loss(lambda **kw: SimpleNamespace(loss=loss), inputs)
    g = guard()
    with pytest.raises(RuntimeError): g.on_train_end(None, None, None)
    for norm in (0, float('nan'), float('inf')):
        with pytest.raises(RuntimeError): g.on_log(None, None, None, logs={'grad_norm': norm})
    g.on_log(None, None, None, logs={'grad_norm': 1.2}); g.on_train_end(None, None, None)


@pytest.mark.parametrize('local_bad,peer_bad', [(False, False), (False, True), (True, False)])
def test_coordinated_failure_mock_collective_no_process_group(monkeypatch, local_bad, peer_bad):
    calls = []
    def reduce(flag, op):
        calls.append(flag.item())
        if peer_bad: flag.fill_(1)
    dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
        get_backend=lambda: 'gloo', all_reduce=reduce, ReduceOp=SimpleNamespace(MAX='MAX'))
    monkeypatch.setattr(torch, 'distributed', dist)
    if local_bad or peer_bad:
        with pytest.raises(RuntimeError, match='one or more ranks'):
            runner.coordinated_failure(local_bad, 'fixture', torch)
    else:
        runner.coordinated_failure(False, 'fixture', torch)
    assert calls == [int(local_bad)]
    if peer_bad:
        finite, guard = runner.trainer_types(object, object, torch)
        with pytest.raises(RuntimeError, match='assistant loss'):
            finite().compute_loss(lambda **kw: SimpleNamespace(loss=torch.tensor(1.)), {})
        with pytest.raises(RuntimeError, match='gradient norm'):
            guard().on_log(None, None, None, logs={'grad_norm': 1.})


def test_optimizer_groups_are_not_replaced_by_base_defaults():
    class Base:
        @staticmethod
        def get_optimizer_cls_and_kwargs(args, model):
            return lambda groups, **kw: (groups, kw), {'lr': 123, 'eps': 1e-8}
    finite, _ = runner.trainer_types(Base, object, torch)
    obj = finite(); obj.optimizer = None; obj.args = object(); obj.model = object()
    obj.native_parameter_groups = [{'params': ['vision'], 'lr': 1e-6}]
    value = obj.create_optimizer()
    assert value == (obj.native_parameter_groups, {'eps': 1e-8})
    assert obj.create_optimizer() is value


def test_offline_environment_restores_on_failure(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', 'original')
    with pytest.raises(RuntimeError):
        with runner.offline():
            assert all(os.environ[k] == v for k, v in preflight._OFFLINE.items())
            assert os.environ['WANDB_MODE'] == 'disabled'
            raise RuntimeError('fixture')
    assert os.environ['HF_HUB_OFFLINE'] == 'original'


def test_execute_requires_explicit_linux_gate_before_runtime(inputs, monkeypatch):
    args, _ = inputs
    monkeypatch.setattr(runner, '_runtime', lambda: pytest.fail('runtime imported'))
    with pytest.raises(ValueError): runner.execute(args)
    args.execute_training = True
    monkeypatch.setattr(runner.sys, 'platform', 'win32')
    with pytest.raises(ValueError): runner.execute(args)


def test_collective_stage_propagates_local_and_peer_failure():
    state = SimpleNamespace(process_index=0)
    runtime = SimpleNamespace(gather_object=lambda status: status)
    assert runner.collective_stage(runtime, state, 'fixture', lambda: 'local') == 'local'
    with pytest.raises(RuntimeError, match='local failure'):
        runner.collective_stage(runtime, state, 'fixture', lambda: (_ for _ in ()).throw(ValueError('local failure')))
    runtime.gather_object = lambda status: status + [dict(rank=1, error='peer failure')]
    with pytest.raises(RuntimeError, match='peer failure'):
        runner.collective_stage(runtime, state, 'fixture', lambda: 'local success')


class FakeCuda:
    def is_available(self): return True
    def device_count(self): return 4
    def set_device(self, rank): assert rank == 0
    def is_bf16_supported(self): return True
    def get_device_properties(self, rank):
        return SimpleNamespace(name='FAKE_CPU_TEST_DEVICE', total_memory=123, major=0, minor=0)


@pytest.mark.parametrize('mutation', ['world', 'multinode', 'rank', 'cuda', 'bf16'])
def test_hardware_topology_fails_before_preparation(inputs, monkeypatch, mutation):
    args, _ = inputs; args.execute_training = True
    monkeypatch.setattr(runner.sys, 'platform', 'linux')
    for k, v in {'WORLD_SIZE': '4', 'LOCAL_WORLD_SIZE': '4', 'LOCAL_RANK': '0'}.items(): monkeypatch.setenv(k, v)
    cuda = FakeCuda()
    if mutation == 'world': monkeypatch.setenv('WORLD_SIZE', '2')
    if mutation == 'multinode': monkeypatch.setenv('LOCAL_WORLD_SIZE', '2')
    if mutation == 'rank': monkeypatch.setenv('LOCAL_RANK', '-1')
    if mutation == 'cuda': cuda.is_available = lambda: False
    if mutation == 'bf16': cuda.is_bf16_supported = lambda: False
    runtime = SimpleNamespace(torch=SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runner, '_runtime', lambda: runtime)
    monkeypatch.setattr(runner, 'prepare', lambda *a: pytest.fail('preparation reached'))
    with pytest.raises(ValueError): runner.execute(args)


@pytest.mark.parametrize('hold', ['admission', 'parity', 'loading_info', None])
def test_fake_execution_order_and_no_qualification_claim(inputs, monkeypatch, hold):
    """Pure orchestration simulation: FakeTrainer.train performs NO training."""
    args, _ = inputs; args.execute_training = True
    monkeypatch.setattr(runner.sys, 'platform', 'linux')
    for k, v in {'WORLD_SIZE': '4', 'LOCAL_WORLD_SIZE': '4', 'LOCAL_RANK': '0'}.items(): monkeypatch.setenv(k, v)
    events = []
    class Rows(FakeRows):
        def __init__(self, *a, **kw):
            events.append('admission')
            if hold == 'admission': raise ValueError('held admission')
            super().__init__(*a, **kw)
    monkeypatch.setattr(data, 'NativeRows', Rows)
    class Processor:
        def save_pretrained(self, path): events.append('save_processor')
    class Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs == dict(local_files_only=True, trust_remote_code=False)
            assert os.environ['HF_HUB_OFFLINE'] == '1'
            events.append('processor'); return Processor()
    class Model(FakeParameterModel):
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs == dict(local_files_only=True, trust_remote_code=False, use_safetensors=True,
                dtype='FAKE_BF16', attn_implementation='sdpa', output_loading_info=True)
            events.append('weights')
            info = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
            if hold == 'loading_info': info['missing_keys'] = ['broken']
            return Model(), info
    class Arguments:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def to_dict(self): return self.kwargs
    class Trainer:
        def __init__(self, **kwargs):
            events.append('trainer'); self.kwargs = kwargs
            assert kwargs['eval_dataset'] is None
        def train(self):
            events.append('fake_train_no_optimization')
            assert self.model_accepts_loss_kwargs is False
            assert len(self.native_parameter_groups) == 2
            return SimpleNamespace(metrics=dict(fixture='NOT_TRAINING'))
        def save_model(self, path): Path(path).mkdir(); events.append('save_model')
        def save_state(self): events.append('save_state')
    state = SimpleNamespace(process_index=0, is_main_process=True, num_processes=4)
    runtime = SimpleNamespace(torch=SimpleNamespace(cuda=FakeCuda(), bfloat16='FAKE_BF16'),
        AutoProcessor=Auto, Model=Model, Trainer=Trainer, TrainingArguments=Arguments,
        TrainerCallback=object, PartialState=lambda **kw: state, gather_object=lambda value: value,
        set_seed=lambda seed: events.append('seed'))
    monkeypatch.setattr(runner, '_runtime', lambda: runtime)
    def bind(*unused):
        events.append('parity')
        if hold == 'parity': raise ValueError('held parity')
        return SimpleNamespace(verify_sources=lambda: events.append('audit'),
                               close=lambda: events.append('close_collators')), {'fixture': 'FAKE_PARITY'}
    monkeypatch.setattr(runner, 'bind_processor', bind)
    if hold:
        with pytest.raises(RuntimeError): runner.execute(args)
        assert 'fake_train_no_optimization' not in events
        if hold in ('admission', 'parity'): assert 'weights' not in events and not args.output.exists()
        assert not (args.output / 'completed.json').exists()
    else:
        runner.execute(args)
        assert events.index('admission') < events.index('processor') < events.index('parity') < events.index('weights')
        assert events.index('weights') < events.index('fake_train_no_optimization') < events.index('save_model')
        completed = export._parse((args.output / 'completed.json').read_bytes())
        assert completed['state'] == 'recipe_finished_not_qualified'
        assert all(completed[k] is False for k in ('gpu_qualified', 'checkpoint_reload_verified',
            'generation_accuracy_measured', 'test_evaluated', 'simulator_actions_validated', 'training_approved'))
        contract = export._parse((args.output / 'model/native_grounding_contract.json').read_bytes())
        assert contract['native_contract']['resolution'] == [1696, 816]
        assert contract['native_contract']['crop_size'] == [768, 768]
