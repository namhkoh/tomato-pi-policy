"""Offline, CPU processor-only preflight; NEVER loads weights or trains.

run(root, *, manifest_sha256, model_snapshot, processor_files, sample_ids,
    maximum_tokens) returns a JSON-compatible report, writes nothing.
root/model_snapshot are explicit absolute directories. processor_files is an
externally supplied {relative processor filename: SHA256} map, not inferred
trust. Include every present name in PROCESSOR_FILES; no weight files. Required:
config.json, tokenizer_config.json, tokenizer.json, plus preprocessor_config.json
or processor_config.json. A single nonempty runtime chat template is required.
Nested chat-template directories/custom remote code are explicitly unsupported.
Explicit tokenizer-file references outside the pinned snapshot are held.
Requires the standard fast tokenizer's serializable backend for an effective
runtime tokenizer hash (including normalization, merges and added tokens).
Architecture checks establish dense Qwen3-VL-8B config only, NOT weight identity
or Instruct-vs-Thinking provenance. The caller authenticates the snapshot pins.

Requires a fully validated native_clear_export package, explicit unique TRAIN
sample IDs and a positive token budget (no 2048 default). Checks two native RGB
images per row, full-frame normalized coordinates, actual image grids/expanded
image tokens, independent image preprocessing, exact generation-prefix match,
un-padded/un-truncated encoding, and assistant-only labels decoding to the known
answer. Budget includes prompt + the actual supervised answer/closing tokens;
it is NOT a reserve for arbitrary future generated answers or unchecked rows.
No prediction, calibration, admission, biological-independence or action claim.

AutoProcessor is imported lazily inside an offline context, local_files_only=True
and trust_remote_code=False. Python audit hooks additionally deny network events,
weight-file opens and unpinned snapshot-file reads while active. This is a guard
for trusted installed libraries, not a security sandbox for malicious native
code. Run in a dedicated process: offline environment and audit guard are process
wide during run(), restored/deactivated afterward. No CUDA/model API is invoked.
Release validation reads native sidecars; processor calls receive RGB/query only.
The tokenizer/image processor are exercised with return_tensors='np', not a
qualification of the trainer's Torch batching, GPU memory or model execution.

The frozen root H200 scripts still reject this native profile/use legacy
coordinate routing and the 2048 collator limit. ROUTE INTEGRATION IS MISSING;
this module deliberately does not patch or invoke them. Check actual server
processor behavior after transfer; fake tests are not server-grid evidence.

CLI (server only, existing local snapshot, no download command):
python -B -m sim_data.native_dataset.native_clear_processor_preflight \
  --dataset ABS_RELEASE --manifest-sha256 SHA --model-snapshot ABS_SNAPSHOT \
  --processor-pins ABS_JSON_MAP --sample-id TRAIN_ID [--sample-id TRAIN_ID] \
  --maximum-tokens EXPLICIT_BUDGET
Prints report to stdout only. Python >=3.11; existing server Transformers stack.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import native_clear_export as export
from .bundle import SampleReader, read_bounded, safe_path


SCHEMA = 'greenhouse.native_clear_processor_preflight.v1'
PROCESSOR_FILES = frozenset(('config.json', 'processor_config.json', 'preprocessor_config.json',
    'video_preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json',
    'merges.txt', 'added_tokens.json', 'special_tokens_map.json', 'chat_template.json', 'chat_template.jinja'))
_CODE_SHA = export._file_sha(__file__)
_OFFLINE = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                TOKENIZERS_PARALLELISM='false')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _budget(value):
    _require(type(value) is int and value > 0, 'Explicit positive maximum_tokens required')


def _snapshot(root, pins):
    _require(Path(root).is_absolute() and Path(root).is_dir(), 'Explicit existing absolute local snapshot required')
    root = Path(root).resolve()
    _require(isinstance(pins, dict) and set(pins) <= PROCESSOR_FILES
             and {'config.json', 'tokenizer_config.json', 'tokenizer.json'} <= set(pins)
             and bool({'preprocessor_config.json', 'processor_config.json'} & set(pins)),
             'Explicit processor-only file pins required')
    _require({name for name in PROCESSOR_FILES if (root / name).exists()} == set(pins)
             and not (root / 'chat_templates').exists(), 'Unpinned/unsupported processor configuration')
    for name, pin in pins.items():
        _require(export._file_sha(root / name) == export._sha(pin), 'Processor snapshot pin changed: ' + name)
    tokenizer_config = export._parse(read_bounded(root / 'tokenizer_config.json'))
    for key, filename in (('tokenizer_file', 'tokenizer.json'), ('vocab_file', 'vocab.json'), ('merges_file', 'merges.txt')):
        value = tokenizer_config.get(key)
        if value is not None:
            _require(isinstance(value, str) and filename in pins
                     and (root / value).resolve() == (root / filename).resolve(), 'External tokenizer-file reference unsupported')
    config = export._parse(read_bounded(root / 'config.json'))
    text = config.get('text_config', {})
    _require(config.get('model_type') == 'qwen3_vl' and text.get('hidden_size') == 4096
             and text.get('num_hidden_layers') == 36 and not config.get('quantization_config'),
             'Dense Qwen3-VL-8B configuration required; no weights checked')
    _require(type(config.get('image_token_id')) is int and config['image_token_id'] >= 0,
             'Configured image token identity required')
    return root, config


@contextmanager
def _offline(root, pins):
    active = [True]
    allowed = {(root / name).resolve() for name in pins}
    previous = {name: os.environ.get(name) for name in _OFFLINE}
    previous_bytecode = sys.dont_write_bytecode
    def audit(event, args):
        if not active[0]:
            return
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo',
                     'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.sendto', 'http.client.connect'):
            raise RuntimeError('Processor preflight forbids network access')
        if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink',
                     'os.truncate', 'os.chmod', 'os.utime'):
            raise RuntimeError('Processor preflight is read-only')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            requested = Path(os.fsdecode(args[0])).absolute()
            path = requested.resolve()
            if path.suffix.lower() in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf'):
                raise RuntimeError('Processor preflight forbids weight-file access')
            if (path.is_relative_to(root) or requested.is_relative_to(root)) and path not in allowed:
                raise RuntimeError('Unpinned snapshot file access: ' + str(path))
            mode, flags = args[1:3]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                raise RuntimeError('Processor preflight is read-only')
    sys.addaudithook(audit)
    os.environ.update(_OFFLINE)
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        active[0] = False
        sys.dont_write_bytecode = previous_bytecode
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _auto_processor():
    from transformers import AutoProcessor
    return AutoProcessor


def _array(value, name):
    result = np.asarray(value)
    _require(result.dtype.kind in 'iuf' and np.isfinite(result).all(), 'Finite numeric processor output required: ' + name)
    return result


def _array_sha(value):
    array = np.ascontiguousarray(value)
    header = export._json(dict(dtype=array.dtype.str, shape=list(array.shape)))
    return hashlib.sha256(header + b'\n' + array.tobytes()).hexdigest()


def _image_output(value, settings):
    grid = _array(value['image_grid_thw'], 'image_grid_thw')
    _require(grid.shape == (2, 3) and grid.dtype.kind in 'iu' and (grid > 0).all()
             and (grid[:, 0] == 1).all(), 'Exactly two still-image THW grids required')
    patch, temporal, merge = (settings[k] for k in ('patch_size', 'temporal_patch_size', 'merge_size'))
    _require(all(type(v) is int and v > 0 for v in (patch, temporal, merge)), 'Explicit positive patch settings required')
    _require((grid[:, 1:] % merge == 0).all(), 'Image grid must divide into spatial merge blocks')
    pixels = _array(value['pixel_values'], 'pixel_values')
    _require(pixels.dtype.kind == 'f' and pixels.shape == (int(np.prod(grid, axis=1).sum()),
             3 * temporal * patch * patch), 'Two RGB image patch buffers required')
    return grid, pixels, (np.prod(grid, axis=1) // (merge * merge)).tolist()


def _sequence(value):
    _require(set(value) <= {'input_ids', 'attention_mask', 'image_grid_thw', 'pixel_values',
             'token_type_ids', 'mm_token_type_ids'}
             and {'input_ids', 'attention_mask', 'image_grid_thw', 'pixel_values'} <= set(value),
             'Unsupported/missing processor outputs; no video or depth inputs')
    ids = _array(value['input_ids'], 'input_ids')
    mask = _array(value['attention_mask'], 'attention_mask')
    _require(ids.dtype.kind in 'iu' and ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > 0
             and (ids >= 0).all() and mask.shape == ids.shape and (mask == 1).all(),
             'One nonempty unpadded token sequence required')
    for name in ('token_type_ids', 'mm_token_type_ids'):
        if name in value:
            _require(_array(value[name], name).shape == ids.shape, 'Token-type shape mismatch')
    return ids


def check_record(root, record, processor, *, maximum_tokens):
    """CPU check for a record from export.validate(); not standalone admission.

    Public test seam accepts a fake processor; run() alone loads AutoProcessor.
    Returns evidence, not training tensors. No unchecked-record qualification.
    """
    _budget(maximum_tokens)
    _require(record.get('split') == 'train', 'Preflight uses explicit TRAIN rows only')
    supervised = export.model_messages(root, record, supervised=True, crop=True)
    inference = export.model_messages(root, record, supervised=False, crop=True)
    label = SampleReader(safe_path(root, record['compact_root'])).json('supervision/label.json')
    expected = export.contract.model_answer(label['answer'])
    answer_text = supervised[-1]['content'][0]['text']
    _require(export._parse(answer_text) == expected and record['query_pixel_uv'] == label['query_pixel_uv']
             and record['crop_box'] == export.contract.crop_box(label['query_pixel_uv']), 'Native coordinate binding changed')
    error = np.abs(np.asarray(export.contract.canonical_answer(expected)['cut_point_uv'])
                   - label['answer']['cut_point_uv'])
    # Full one-quantum bound includes format_point's upper-edge 999.99 clamp.
    bound = np.asarray([1696., 816.]) * .01 / 1000.
    _require((error <= bound).all(), 'Native full-frame coordinate round-trip error')
    _require(supervised[0] == inference[0] and supervised[1]['content'][-1] == inference[1]['content'][-1],
             'Train/inference prompt mismatch')
    images = [item['image'] for item in supervised[1]['content'][:2]]
    _require([image.size for image in images] == [(1696, 816), (768, 768)]
             and all(image.mode == 'RGB' for image in images)
             and images[0].crop(record['crop_box']).tobytes() == images[1].tobytes()
             and all(image.tobytes() == item['image'].tobytes()
                     for image, item in zip(images, inference[1]['content'][:2])), 'Native full/crop pixels changed')
    before = [image.tobytes() for image in images]
    settings = {k: getattr(processor.image_processor, k, None)
                for k in ('patch_size', 'temporal_patch_size', 'merge_size')}
    kwargs = dict(tokenize=True, return_dict=True, return_tensors='np', padding=False, truncation=False)
    full = processor.apply_chat_template(supervised, add_generation_prompt=False, **kwargs)
    prefix = processor.apply_chat_template(inference, add_generation_prompt=True, **kwargs)
    ids, prompt = _sequence(full), _sequence(prefix)
    count, total = prompt.shape[1], ids.shape[1]
    _require(0 < count < total and np.array_equal(ids[:, :count], prompt), 'Generation prefix/answer boundary mismatch')
    _require(total <= maximum_tokens,
             f'Token budget exceeded: prompt={count}, answer={total-count}, required={total}, budget={maximum_tokens}; never truncate')
    full_grid, pixels, image_tokens = _image_output(full, settings)
    prefix_grid, prefix_pixels, _ = _image_output(prefix, settings)
    direct = processor.image_processor(images=images, return_tensors='np')
    direct_grid, direct_pixels, _ = _image_output(direct, settings)
    _require(np.array_equal(full_grid, prefix_grid) and np.array_equal(full_grid, direct_grid)
             and np.array_equal(pixels, prefix_pixels) and np.array_equal(pixels, direct_pixels),
             'Full/prefix/independent RGB processing mismatch')
    _require(before == [image.tobytes() for image in images], 'Processor mutated native inputs')
    for name in ('token_type_ids', 'mm_token_type_ids'):
        _require((name in full) == (name in prefix), 'Token-type outputs changed')
        if name in full:
            _require(np.array_equal(np.asarray(full[name])[:, :count], prefix[name]), 'Token-type prefix mismatch')
    token = processor.image_token_id
    _require(type(token) is int and token >= 0, 'Image token identity required')
    positions = np.flatnonzero(prompt[0] == token)
    runs = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    _require([len(run) for run in runs] == image_tokens and not (ids[0, count:] == token).any(),
             'Expanded image tokens do not match the two ordered grids')
    labels = ids.astype(np.int64, copy=True); labels[:, :count] = -100
    decoded = processor.tokenizer.decode(labels[labels != -100].tolist(), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).strip()
    _require(export._parse(decoded) == expected, 'Assistant tokens do not decode to the exact native answer')
    decoded_prompt = processor.tokenizer.decode(prompt[0].tolist(), skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
    _require(all(turn['content'][-1]['text'] in decoded_prompt for turn in inference)
             and answer_text not in decoded_prompt, 'System/query missing or answer leaked into prefix')
    return dict(sample_id=record['sample_id'], record_sha256=export.json_sha256(record),
        native_sizes=[[1696, 816], [768, 768]], crop_box=record['crop_box'],
        query_full_pixel_uv=record['query_pixel_uv'], normalized_answer=expected,
        answer_roundtrip_error_px=error.tolist(), answer_roundtrip_bound_px=bound.tolist(),
        image_grid_thw=full_grid.tolist(), image_tokens=image_tokens,
        processed_sizes_wh=(full_grid[:, [2, 1]] * settings['patch_size']).tolist(),
        patch_settings=settings, pixel_values_shape=list(pixels.shape), pixel_values_sha256=_array_sha(pixels),
        prompt_tokens=count, assistant_tokens=total-count, total_tokens=total, maximum_tokens=maximum_tokens,
        assistant_span=[count, total], prompt_mask_value=-100, labels_sha256=_array_sha(labels),
        input_ids_sha256=_array_sha(ids), prompt_ids_sha256=_array_sha(prompt),
        assistant_mask_verified=True, independent_image_processing_verified=True)


def _implementation(processor):
    objects = (processor, processor.image_processor, processor.tokenizer)
    files = {}
    for obj in objects:
        cls = type(obj)
        path = inspect.getsourcefile(cls)
        _require(path is not None, 'Installed processor implementation source required')
        files[cls.__module__ + '.' + cls.__qualname__] = export._file_sha(path)
    for name, method in (('apply_chat_template', processor.apply_chat_template),
                         ('image_processor_call', processor.image_processor.__call__),
                         ('tokenizer_decode', processor.tokenizer.decode)):
        path = inspect.getsourcefile(method)
        _require(path is not None, 'Installed processor method source required')
        files[name] = export._file_sha(path)
    return files


def _tokenizer_sha(processor):
    backend = getattr(processor.tokenizer, 'backend_tokenizer', None)
    _require(backend is not None and callable(getattr(backend, 'to_str', None)),
             'Serializable fast tokenizer backend required')
    text = backend.to_str()
    _require(isinstance(text, str) and text, 'Nonempty tokenizer serialization required')
    return hashlib.sha256(text.encode()).hexdigest()


def run(root, *, manifest_sha256, model_snapshot, processor_files, sample_ids, maximum_tokens):
    """Read-only server entry point. Known-answer checks only; no model APIs."""
    _budget(maximum_tokens)
    _require(Path(root).is_absolute(), 'Explicit absolute release required')
    _require(isinstance(sample_ids, (list, tuple)) and sample_ids
             and all(isinstance(s, str) and s for s in sample_ids)
             and len(set(sample_ids)) == len(sample_ids), 'Explicit unique TRAIN sample IDs required')
    root = Path(root).resolve()
    _require(isinstance(processor_files, dict), 'Explicit processor file pin mapping required')
    pins = dict(processor_files)
    snapshot, config = _snapshot(model_snapshot, pins)
    _require(not root.is_relative_to(snapshot) and not snapshot.is_relative_to(root), 'Separate release and snapshot required')
    with _offline(snapshot, pins):
        validated = export.validate(root, manifest_sha256=manifest_sha256)
        records = {record['sample_id']: record for record in validated['records']['train']}
        _require(set(sample_ids) <= set(records), 'Requested IDs must be validated TRAIN rows')
        processor = _auto_processor().from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        _require(type(processor).__name__ == 'Qwen3VLProcessor'
                 and processor.image_token_id == config['image_token_id'], 'Unexpected processor/image-token identity')
        template = getattr(processor, 'chat_template', None)
        _require(isinstance(template, str) and bool(template.strip()), 'Single explicit processor chat template required')
        implementation = _implementation(processor)
        tokenizer_sha = _tokenizer_sha(processor)
        image_config = processor.image_processor.to_dict()
        image_config_sha = export.json_sha256(image_config)
        checked = [check_record(root, records[key], processor, maximum_tokens=maximum_tokens) for key in sample_ids]
        _snapshot(snapshot, pins)
        _require(processor.chat_template == template and _implementation(processor) == implementation
                 and _tokenizer_sha(processor) == tokenizer_sha
                 and export.json_sha256(processor.image_processor.to_dict()) == image_config_sha,
                 'Processor configuration/implementation changed during preflight')
        manifest_raw = read_bounded(root / 'manifest.json')
        _require(hashlib.sha256(manifest_raw).hexdigest() == manifest_sha256, 'Release manifest changed')
        manifest = export._parse(manifest_raw)
        for name, pin in manifest['files'].items():
            _require(export._file_sha(safe_path(root, name)) == pin, 'Release bytes changed during preflight')
        export._check_code(manifest['implementation_bindings'])
        _require(export._file_sha(__file__) == _CODE_SHA, 'Preflight implementation changed')
    versions = {}
    for package in ('transformers', 'tokenizers', 'numpy', 'Pillow', 'torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(schema=SCHEMA, state='checked_rows_processor_only', release_manifest_sha256=manifest_sha256,
        processor_snapshot_path=str(snapshot), processor_files=pins, processor_files_sha256=export.json_sha256(pins),
        processor_implementation_bindings=implementation, preflight_source_sha256=_CODE_SHA,
        release_implementation_bindings=manifest['implementation_bindings'], runtime_versions=versions,
        python_version=sys.version, image_processor_config=image_config,
        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(), rows=checked,
        effective_tokenizer_sha256=tokenizer_sha,
        maximum_tokens=maximum_tokens, maximum_observed_total_tokens=max(row['total_tokens'] for row in checked),
        scope='explicit_train_rows_known_answers_not_unchecked_rows_or_future_generations',
        offline=True, weights_loaded=False, weight_integrity_verified=False, training_performed=False,
        depth_is_model_input=False, route_integration_complete=False, training_approved=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--model-snapshot', required=True)
    parser.add_argument('--processor-pins', required=True)
    parser.add_argument('--sample-id', action='append', required=True)
    parser.add_argument('--maximum-tokens', type=int, required=True)
    args = parser.parse_args(argv)
    _require(Path(args.processor_pins).is_absolute(), 'Absolute processor pin-map file required')
    result = run(args.dataset, manifest_sha256=args.manifest_sha256, model_snapshot=args.model_snapshot,
                 processor_files=export._parse(read_bounded(Path(args.processor_pins))),
                 sample_ids=args.sample_id, maximum_tokens=args.maximum_tokens)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
