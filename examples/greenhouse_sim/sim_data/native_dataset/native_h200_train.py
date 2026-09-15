"""Unqualified, offline native full-FT server recipe; NEVER invoked by import.

Normal CLI invocation is a read-only configuration/data/file-pin check. Actual
server execution additionally requires --execute-training, Linux, BF16 CUDA and
an explicit single-node torchrun topology (expected world size defaults to 4).
There is no model downloader, resume, tracker, legacy data route or test scorer.
Do NOT execute here: CPU tests/fakes are not processor, GPU or training evidence.

Required CLI inputs: --dataset ABS --manifest-sha256 SHA --admission-sha256 SHA,
--model ABS --model-files ABS_JSON --model-files-sha256 SHA,
--preflight-report ABS_JSON --preflight-sha256 SHA,
--deepspeed ABS_JSON --deepspeed-sha256 SHA, --output NEW_ABS,
--maximum-tokens N, repeated --control-id TRAIN_ID and --model-input-key KEY,
--mode smoke|overfit|train. Then, ONLY on the separately authorized server, add
--execute-training. Model-input keys must explicitly appear in model.forward;
**kwargs is not evidence that an optional token-type field is supported.

model-files is an externally authenticated {basename: SHA256} JSON map: either
model.safetensors, OR model.safetensors.index.json and every referenced shard;
include generation_config.json iff present. Processor/config files are already
pinned by the preflight report. No pickle weights, adapters or custom code.
Hashes verify supplied bytes, NOT their origin, tensor correctness or accuracy.
All actual loaded tensors must match the dense config without loading warnings.

Frozen NativeRows performs admission/source/count/exact-donor-map validation on
each rank; only small error/status objects cross collectives, never chat lists.
NativeCollator and fresh NumPy/Torch parity run before weights. Microbatch=1,
accumulation=8 by default in EVERY mode; effective batch=world_size*accumulation.
BF16/SDPA/ZeRO-2, language+merger LR1e-5, vision LR1e-6, seed41, train3epochs are
starting hypotheses. Smoke uses the ordinary TRAIN sampler for 2 optimizer
steps; overfit uses 32 deterministic TRAIN IDs for 100 steps. Repeats/padding
are sampler behavior, not new independent biological examples. No LR search.

Only TRAIN controls are preflight-qualified. Every consumed row checks its own
budget/mask and fails rather than truncating. No claim that all TRAIN/validation
lengths were audited before loading weights. Validation loss only in train mode;
test files are integrity-validated by the exporter but NEVER used for evaluation.
Native1696 full + native768 query crop; outputs normalized1000/full-frame/2dp.
No depth, masks, identities or world coordinates enter model inputs.

Keep dataset/code/local snapshot/processor immutable throughout execution.
Full source audits precede loading and follow training; no filesystem locks.
Loss and logged-gradient checks reduce a failure flag across initialized ranks.
CUDA faults/uneven callback execution still require real torchrun qualification;
the supervisor/timeouts, not this helper, handle arbitrary process/kernel faults.
Model and processor load local-only/trust_remote_code=False, safetensors only;
HF offline/telemetry-off and report_to=[] prohibit hub/tracker workflows. Local
distributed CUDA communications are required; this is not a network sandbox.
An interrupted NEW output is left incomplete, never removed or resumed.
Completion records do not claim reload, generation accuracy, GPU qualification,
calibration, independent admission, biological independence or action approval.
No native generation, checkpoint evaluation or archive code is included.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from . import native_training_data as data
from . import native_clear_export as export
from . import native_clear_processor_preflight as preflight
from .bundle import read_bounded, safe_path


SCHEMA = 'greenhouse.native_h200_full_ft_recipe.v1'
_SOURCE_SHA = export._file_sha(__file__)
ZERO2 = dict(bf16=dict(enabled=True), zero_optimization=dict(stage=2, overlap_comm=True,
    contiguous_gradients=True), gradient_accumulation_steps='auto',
    train_micro_batch_size_per_gpu='auto', train_batch_size='auto', gradient_clipping='auto')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'model', 'model-files', 'preflight-report', 'deepspeed', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'admission-sha256', 'model-files-sha256', 'preflight-sha256', 'deepspeed-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--maximum-tokens', type=int, required=True)
    parser.add_argument('--control-id', action='append', required=True)
    parser.add_argument('--model-input-key', action='append', required=True)
    parser.add_argument('--mode', choices=('smoke', 'overfit', 'train'), required=True)
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--expected-world-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=3.)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--vision-learning-rate', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args(argv)
    validate_arguments(args)
    return args


def validate_arguments(args):
    preflight._budget(args.maximum_tokens)
    for name in ('manifest_sha256', 'admission_sha256', 'model_files_sha256', 'preflight_sha256', 'deepspeed_sha256'):
        export._sha(getattr(args, name))
    _require(args.mode in ('smoke', 'overfit', 'train') and type(args.execute_training) is bool, 'Explicit execution mode required')
    for name in ('expected_world_size', 'gradient_accumulation'):
        _require(type(getattr(args, name)) is int and getattr(args, name) > 0, 'Positive world size/accumulation required')
    _require(type(args.seed) is int and 0 <= args.seed < 2**32, 'Valid reproducible seed required')
    _require(all(type(getattr(args, k)) in (int, float) and math.isfinite(getattr(args, k))
                 and getattr(args, k) > 0 for k in ('epochs', 'learning_rate', 'vision_learning_rate')),
             'Positive finite epochs and learning rates required')
    _require(isinstance(args.control_id, list) and args.control_id
             and all(isinstance(s, str) and s for s in args.control_id)
             and len(set(args.control_id)) == len(args.control_id), 'Explicit unique control IDs required')
    keys = args.model_input_key
    _require(isinstance(keys, list) and len(set(keys)) == len(keys)
             and data._CORE <= set(keys) <= data._CORE | data._OPTIONAL, 'Explicit supported model-input keys required')


def check_paths(args):
    paths = [Path(getattr(args, n)) for n in ('dataset', 'model', 'model_files', 'preflight_report', 'deepspeed', 'output')]
    _require(all(p.is_absolute() for p in paths), 'All paths must be explicit absolute local paths')
    root, model, weights, report, zero, output = [p.resolve() for p in paths]
    _require(root.is_dir() and model.is_dir() and all(p.is_file() for p in (weights, report, zero)), 'Existing local input paths required')
    _require(not output.exists(), 'Choose a NEW output; no overwrite or resume')
    protected = (root, model, weights, report, zero)
    _require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected), 'Separate output from every input')
    _require(not root.is_relative_to(model) and not model.is_relative_to(root), 'Separate release and model required')
    return root, model, output


def _document(path, sha):
    return export._parse(export._pinned(export.FilePin(str(path), sha)))


def check_model_files(model, pin):
    """Byte-only safetensors inventory check; never deserializes a tensor."""
    files = export._parse(export._pinned(pin))
    _require(isinstance(files, dict) and files, 'Explicit model-file pin mapping required')
    for name, sha in files.items():
        _require(isinstance(name, str) and '/' not in name and '\\' not in name and name not in ('.', '..'), 'Top-level model filenames required')
        export._sha(sha)
    allowed = {'model.safetensors', 'model.safetensors.index.json', 'generation_config.json'}
    _require(all(name in allowed or re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', name) for name in files), 'Unsupported model-file format')
    model = Path(model)
    _require(not any(model.glob('*.bin')) and not any(model.glob('*.py'))
             and not (model / 'adapter_config.json').exists(), 'Pickle weights/adapters/custom model code unsupported')
    actual = {p.name for p in model.glob('*.safetensors')}
    for name in ('model.safetensors.index.json', 'generation_config.json'):
        if (model / name).exists(): actual.add(name)
    _require(actual == set(files), 'Unpinned or missing model files')
    for name, sha in files.items():
        _require(export._file_sha(safe_path(model, name)) == sha, 'Model file pin changed: ' + name)
    if 'model.safetensors.index.json' in files:
        index = export._parse(read_bounded(model / 'model.safetensors.index.json'))
        mapping = index.get('weight_map')
        _require(isinstance(mapping, dict) and mapping and all(isinstance(k, str) and k for k in mapping)
                 and all(isinstance(v, str) for v in mapping.values()), 'Nonempty safetensors weight_map required')
        _require(set(mapping.values()) == set(files) - {'model.safetensors.index.json', 'generation_config.json'}
                 and 'model.safetensors' not in files, 'Exact shard/index coverage required')
    else:
        _require(set(files) - {'generation_config.json'} == {'model.safetensors'}, 'Single safetensors file or complete shard index required')
    return files


@dataclass
class Prepared:
    args: object
    rows: object
    report: dict
    zero2: dict
    model_files: dict
    summary: dict


def prepare(args):
    """Read-only CPU preparation. No processor import/load, model or output write."""
    validate_arguments(args)
    root, model, output = check_paths(args)
    _require(export._file_sha(__file__) == _SOURCE_SHA, 'Runner source changed')
    manifest = _document(root / 'manifest.json', args.manifest_sha256)
    _require(manifest.get('provenance', {}).get('admission') == args.admission_sha256,
             'Explicit admission pin differs from release')
    _document(root / 'provenance/admission.json', args.admission_sha256)
    # This is the real gate: no private count override, candidate or smoke bypass.
    rows = data.NativeRows(root, manifest_sha256=args.manifest_sha256, split='train')
    try:
        report = _document(args.preflight_report, args.preflight_sha256)
        _require(report.get('schema') == preflight.SCHEMA
                 and report.get('state') == 'checked_rows_processor_only'
                 and report.get('release_manifest_sha256') == args.manifest_sha256
                 and report.get('maximum_tokens') == args.maximum_tokens, 'Preflight release/budget mismatch')
        _require(report.get('offline') is True and all(report.get(k) is False for k in (
            'weights_loaded', 'weight_integrity_verified', 'training_performed', 'depth_is_model_input',
            'route_integration_complete', 'training_approved'))
            and report.get('scope') == 'explicit_train_rows_known_answers_not_unchecked_rows_or_future_generations',
            'Explicit processor-only preflight scope/flags required')
        _require(export.json_sha256(report['processor_files']) == report['processor_files_sha256'],
                 'Processor file-map binding changed')
        _require(report.get('preflight_source_sha256') == preflight._CODE_SHA
                 and report.get('release_implementation_bindings') == rows.receipt['release_implementation_bindings']
                 and report.get('runtime_versions') == data._versions() and report.get('python_version') == sys.version,
                 'Preflight source/runtime mismatch; rerun on the server stack')
        _require(set(args.control_id) <= {r['sample_id'] for r in report['rows']}, 'Controls not present in pinned preflight')
        _, config = preflight._snapshot(model, report['processor_files'])
        _require(not config.get('auto_map') and config.get('architectures') == ['Qwen3VLForConditionalGeneration'],
                 'Explicit standard dense Qwen3-VL model architecture required')
        context = config['text_config'].get('max_position_embeddings')
        _require(type(context) is int and 0 < args.maximum_tokens <= context, 'Token budget exceeds or lacks configured text context')
        zero2 = _document(args.deepspeed, args.deepspeed_sha256)
        _require(zero2 == ZERO2, 'Only the explicit BF16 ZeRO-2 starting configuration is supported')
        files = check_model_files(model, export.FilePin(str(args.model_files), args.model_files_sha256))
        summary = dict(schema=SCHEMA, state='configuration_checked_only', release=rows.receipt,
            admission_sha256=args.admission_sha256, preflight_report_sha256=args.preflight_sha256,
            model_files_manifest_sha256=args.model_files_sha256, model_files=files,
            deepspeed_sha256=args.deepspeed_sha256, runner_source_sha256=_SOURCE_SHA,
            maximum_tokens=args.maximum_tokens, model_input_keys=sorted(args.model_input_key),
            mode=args.mode, microbatch=1, accumulation=args.gradient_accumulation,
            expected_world_size=args.expected_world_size,
            effective_examples_per_step=args.expected_world_size * args.gradient_accumulation,
            counts_are_not_biological_independence=True, model_loaded=False, training_performed=False,
            real_processor_checked=False, gpu_qualified=False, training_approved=False, test_evaluated=False)
        return Prepared(args, rows, report, zero2, files, summary)
    except Exception:
        rows.close()
        raise


def check_forward_keys(model_class, keys):
    parameters = inspect.signature(model_class.forward).parameters
    named = {k for k, v in parameters.items() if v.kind in (v.POSITIONAL_OR_KEYWORD, v.KEYWORD_ONLY)}
    _require(set(keys) | {'labels'} <= named, 'Model.forward does not explicitly support the declared native inputs/labels')


def check_loading_info(info):
    required = {'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'}
    _require(isinstance(info, dict) and required <= set(info) <= required | {'conversion_errors'}
             and all(isinstance(v, (list, tuple, set, dict)) and not v for v in info.values()),
             'Model loading has missing/unknown/nonempty tensor diagnostics')


class SplitCollator:
    def __init__(self, collators):
        self.collators = collators

    def __call__(self, batch):
        _require(isinstance(batch, (list, tuple)) and len(batch) == 1, 'Native microbatch is exactly one example')
        split = batch[0].get('split')
        _require(split in self.collators and split != 'test', 'No test or unknown split may enter Trainer')
        return self.collators[split](batch)

    def verify_sources(self):
        for collator in self.collators.values(): collator.verify_sources()

    def close(self):
        for collator in self.collators.values(): collator.rows.close()


def bind_processor(prepared, processor, model_class):
    args = prepared.args
    collators, parity = {}, {}
    owned = [prepared.rows]
    try:
        check_forward_keys(model_class, args.model_input_key)
        splits = ('train', 'validation') if args.mode == 'train' else ('train',)
        for split in splits:
            rows = prepared.rows if split == 'train' else prepared.rows.for_split(split)
            if split != 'train': owned.append(rows)
            collator = data.NativeCollator(rows, processor,
                preflight_report=export.FilePin(str(args.preflight_report), args.preflight_sha256),
                model_snapshot=args.model, maximum_tokens=args.maximum_tokens, model_input_keys=args.model_input_key)
            parity[split] = data.check_numpy_torch_parity(collator, sample_ids=args.control_id)
            collators[split] = collator
    except BaseException:
        for rows in owned: rows.close()
        raise
    return SplitCollator(collators), parity


class RowSubset:
    def __init__(self, rows, indices): self.rows, self.indices = rows, tuple(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.rows[self.indices[index]]


def training_rows(rows, mode):
    _require(mode in ('smoke', 'overfit', 'train') and getattr(rows, 'split', 'train') == 'train',
             'Explicit TRAIN dataset and supported mode required')
    if mode != 'overfit': return rows, None
    _require(len(rows) >= 32, 'Overfit requires 32 TRAIN rows')
    ranked = []
    for i in range(len(rows)):
        row = rows[i]
        _require(row.get('split') == 'train', 'Overfit selection is TRAIN only')
        ranked.append((hashlib.sha256(row['sample_id'].encode()).hexdigest(), i, row['sample_id']))
    ranked = sorted(ranked)[:32]
    return RowSubset(rows, [i for _, i, _ in ranked]), [key for _, _, key in ranked]


def training_options(args, zero2):
    return dict(output_dir=str(args.output), num_train_epochs=args.epochs,
        max_steps={'smoke': 2, 'overfit': 100, 'train': -1}[args.mode],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation, bf16=True, fp16=False,
        learning_rate=args.learning_rate, weight_decay=0., optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, ddp_timeout=7200, dataloader_num_workers=0,
        dataloader_drop_last=False, remove_unused_columns=False, label_names=['labels'],
        prediction_loss_only=True, eval_strategy='epoch' if args.mode == 'train' else 'no',
        save_strategy='epoch' if args.mode == 'train' else 'no', save_only_model=True,
        save_total_limit=2,
        save_safetensors=True, deepspeed=deepcopy(zero2), logging_steps=1,
        logging_nan_inf_filter=False, report_to=[], push_to_hub=False,
        seed=args.seed, data_seed=args.seed, max_grad_norm=1.,
        lr_scheduler_type='cosine' if args.mode == 'train' else 'constant',
        warmup_ratio=.03 if args.mode == 'train' else 0.)


def parameter_groups(model, language_lr, vision_lr):
    """Full FT only. Merger projections follow language LR; no freeze/LoRA path."""
    named = list(model.named_parameters())
    _require(named and any('.language_model.' in n for n, _ in named)
             and any('lm_head' in n for n, _ in named) and any('merger' in n for n, _ in named),
             'Expected language/head/merger parameter names required')
    vision, rest = [], []
    for name, parameter in named:
        parameter.requires_grad_(True)
        is_vision = name.startswith('model.visual.') and '.merger' not in name and 'deepstack_merger' not in name
        (vision if is_vision else rest).append(parameter)
    _require(vision and rest, 'Both vision tower and language/merger groups required')
    return [dict(params=rest, lr=language_lr, weight_decay=0.), dict(params=vision, lr=vision_lr, weight_decay=0.)]


def coordinated_failure(failed, message, torch):
    """Small synchronous flag, not an arbitrary-fault/distributed safety proof."""
    dist = getattr(torch, 'distributed', None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        device = 'cuda' if dist.get_backend() == 'nccl' else 'cpu'
        flag = torch.tensor(int(failed), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        failed = bool(flag.item())
    if failed: raise RuntimeError(message + ' (one or more ranks)')


def trainer_types(base, callback, torch):
    """Injectable class factory for CPU tests; does not instantiate a model."""
    class FiniteTrainer(base):
        native_parameter_groups = None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = getattr(outputs, 'loss', None)
            bad = not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss))
            coordinated_failure(bad, 'Missing/nonfinite scalar assistant loss', torch)
            return (loss, outputs) if return_outputs else loss

        def create_optimizer(self):
            if self.optimizer is None:
                _require(self.native_parameter_groups is not None, 'Explicit full-FT parameter groups required')
                optimizer, kwargs = base.get_optimizer_cls_and_kwargs(self.args, self.model)
                self.optimizer = optimizer(self.native_parameter_groups, **{k: v for k, v in kwargs.items() if k != 'lr'})
            return self.optimizer

    class GradientGuard(callback):
        def __init__(self): self.seen = 0
        def on_log(self, args, state, control, logs=None, **kwargs):
            norm = (logs or {}).get('grad_norm')
            try: bad = norm is not None and (not math.isfinite(float(norm)) or float(norm) <= 0)
            except (TypeError, ValueError, OverflowError): bad = True
            coordinated_failure(bad, 'Nonfinite or zero global gradient norm', torch)
            if norm is not None: self.seen += 1
        def on_train_end(self, args, state, control, **kwargs):
            coordinated_failure(self.seen == 0, 'Global gradient norm never observed', torch)
    return FiniteTrainer, GradientGuard


@contextmanager
def offline():
    values = {**preflight._OFFLINE, 'WANDB_MODE': 'disabled'}
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try: yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def _runtime():
    import torch
    from accelerate import PartialState
    from accelerate.utils import gather_object
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, set_seed
    return SimpleNamespace(torch=torch, PartialState=PartialState, gather_object=gather_object,
        AutoProcessor=AutoProcessor, Model=Qwen3VLForConditionalGeneration, Trainer=Trainer,
        TrainingArguments=TrainingArguments, TrainerCallback=TrainerCallback, set_seed=set_seed)


def check_server(args, runtime):
    _require(args.execute_training and sys.platform.startswith('linux'), 'Explicit execution on a Linux training server required')
    world = int(os.environ.get('WORLD_SIZE', '0'))
    local = int(os.environ.get('LOCAL_WORLD_SIZE', '0'))
    rank = int(os.environ.get('LOCAL_RANK', '-1'))
    _require(world == local == args.expected_world_size and 0 <= rank < local, 'Explicit matching single-node torchrun topology required')
    cuda = runtime.torch.cuda
    _require(cuda.is_available() and cuda.device_count() >= local, 'Required CUDA devices unavailable')
    cuda.set_device(rank)
    _require(cuda.is_bf16_supported(), 'BF16 CUDA support required')


def runtime_provenance(runtime):
    versions = dict(data._versions())
    for name in ('accelerate', 'deepspeed'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    sources = {}
    for name in ('Model', 'Trainer', 'TrainingArguments'):
        source = inspect.getsourcefile(getattr(runtime, name))
        _require(source is not None, 'Installed training implementation source unavailable')
        sources[name] = export._file_sha(source)
    device = runtime.torch.cuda.get_device_properties(int(os.environ['LOCAL_RANK']))
    return dict(versions=versions, implementation_sources=sources, python_version=sys.version,
                device=dict(name=device.name, total_memory=device.total_memory,
                            compute_capability=[device.major, device.minor]), gpu_qualified=False)


def collective_stage(runtime, state, label, operation):
    value, error = None, None
    try: value = operation()
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    statuses = runtime.gather_object([dict(rank=state.process_index, error=error)])
    failures = [s for s in statuses if s['error'] is not None]
    if failures: raise RuntimeError(label + ' failed on ranks: ' + json.dumps(failures))
    return value


def _write(path, value):
    with Path(path).open('xb') as stream: stream.write(export._json(value) + b'\n')


def execute(args):
    """Only explicit future server execution; never called by CPU config mode."""
    _require(args.execute_training and sys.platform.startswith('linux'), 'Explicit Linux server execution required')
    validate_arguments(args)
    with offline():
        runtime = _runtime()
        check_server(args, runtime)
        state = runtime.PartialState(timeout=timedelta(hours=2))
        _require(state.num_processes == args.expected_world_size, 'Distributed state differs from requested world size')
        prepared = collective_stage(runtime, state, 'native admission/configuration', lambda: prepare(args))
        collator = None
        try:
            runtime.set_seed(args.seed)
            def processor_stage():
                with preflight._offline(Path(args.model).resolve(), prepared.report['processor_files']):
                    processor = runtime.AutoProcessor.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
                collator, parity = bind_processor(prepared, processor, runtime.Model)
                return processor, collator, parity
            processor, collator, parity = collective_stage(runtime, state, 'native processor parity', processor_stage)
            def create_output():
                if state.is_main_process:
                    check_paths(args)
                    args.output.mkdir(parents=True, exist_ok=False)
                    _write(args.output / 'prepared.json', dict(prepared.summary,
                        state='processor_parity_checked_before_weights', real_processor_checked=True, processor_parity=parity))
            collective_stage(runtime, state, 'new output', create_output)
            training = collective_stage(runtime, state, 'Trainer configuration',
                lambda: runtime.TrainingArguments(**training_options(args, prepared.zero2)))
            def load_model():
                # Recheck all byte pins immediately before the sole weight-loader call.
                collator.verify_sources()
                check_model_files(args.model, export.FilePin(str(args.model_files), args.model_files_sha256))
                model, info = runtime.Model.from_pretrained(str(args.model), local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, dtype=runtime.torch.bfloat16,
                    attn_implementation='sdpa', output_loading_info=True)
                check_loading_info(info)
                model.config.use_cache = False
                return model
            model = collective_stage(runtime, state, 'pinned local weights', load_model)
            groups = parameter_groups(model, args.learning_rate, args.vision_learning_rate)
            train_rows, subset_ids = training_rows(prepared.rows, args.mode)
            finite, guard = trainer_types(runtime.Trainer, runtime.TrainerCallback, runtime.torch)
            trainer = finite(model=model, args=training, train_dataset=train_rows,
                eval_dataset=collator.collators['validation'].rows if args.mode == 'train' else None,
                data_collator=collator, callbacks=[guard()])
            trainer.native_parameter_groups = groups
            # Keep gradient accumulation as mean of single-example assistant-token means.
            trainer.model_accepts_loss_kwargs = False
            if state.is_main_process:
                _write(args.output / 'run_contract.json', dict(prepared.summary, state='server_training_started',
                    model_loaded=True, real_processor_checked=True, processor_parity=parity, subset_train_ids=subset_ids,
                    runtime=runtime_provenance(runtime),
                    training_arguments=training.to_dict(), loss_normalization='mean_of_single_example_assistant_token_means',
                    native_contract=export.POLICY, checkpoint_reload_verified=False))
            result = trainer.train()
            def audit():
                collator.verify_sources()
                _require(export._file_sha(__file__) == _SOURCE_SHA, 'Runner source changed during training')
                _document(args.deepspeed, args.deepspeed_sha256)
                check_model_files(args.model, export.FilePin(str(args.model_files), args.model_files_sha256))
            collective_stage(runtime, state, 'post-training source audit', audit)
            artifact = args.output / 'model'
            trainer.save_model(str(artifact))
            trainer.save_state()
            def finish():
                if state.is_main_process:
                    processor.save_pretrained(str(artifact))
                    _write(artifact / 'native_grounding_contract.json', dict(schema=SCHEMA,
                        native_contract=export.POLICY, release_manifest_sha256=args.manifest_sha256,
                        preflight_report_sha256=args.preflight_sha256, maximum_tokens=args.maximum_tokens,
                        model_input_keys=args.model_input_key, checkpoint_reload_verified=False))
                    _write(args.output / 'completed.json', dict(schema=SCHEMA, state='recipe_finished_not_qualified',
                        metrics=result.metrics, training_performed=True, checkpoint_reload_verified=False,
                        generation_accuracy_measured=False, gpu_qualified=False, test_evaluated=False,
                        simulator_actions_validated=False, training_approved=False))
            collective_stage(runtime, state, 'save native provenance', finish)
        finally:
            try:
                if collator is not None: collator.close()
            finally:
                prepared.rows.close()


def main(argv=None):
    args = arguments(argv)
    if args.execute_training:
        execute(args)
    else:
        prepared = prepare(args)
        try: print(json.dumps(prepared.summary, indent=2, allow_nan=False))
        finally: prepared.rows.close()


if __name__ == '__main__':
    main()
