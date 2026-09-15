"""Pinned, lazy native RGB data adapter; no trainer, loader, download or writes.

API (all paths absolute; caller authenticates the out-of-band SHA256 pins):
  rows = NativeRows(root, manifest_sha256=SHA, split='train')
  validation = rows.for_split('validation')  # no repeated full validation
  collate = NativeCollator(rows, processor, preflight_report=export.FilePin(...),
      model_snapshot=LOCAL_DIR, maximum_tokens=N, model_input_keys=KEYS)
  evidence = check_numpy_torch_parity(collate, sample_ids=[CHECKED_TRAIN_ID])
  tensors = collate([rows[0]])  # one example, two native images, assistant loss
  pixels_answer = decode_answer(generated_json)

NativeRows always calls the frozen export validator, including its exact donor
map/count/source gates. That validator materializes metadata and reads sidecars
ONCE. Only offsets, hashes, identities and manifest pins survive here. Each
worker opens its own JSONL handle (pickle/fork safe); no images/tensors are cached.
Use normal map-style distributed sampling; this adapter does not shard, shuffle,
drop rows or hide sampler padding/repeats. test access requires allow_test=True.

Full RGB and native768 crop use export.model_messages, never legacy coordinates.
No depth/label sidecars are read in the collation hot path. Processor resampling
is its pinned configuration, not a coordinate transform. Unknown processor keys
are held; model_input_keys explicitly declares the downstream model's accepted
keys (including optional token types). This does NOT verify model.forward.

The pinned preflight report binds the release, processor files, implementation,
versions, effective tokenizer, image config and template. CPU NumPy/Torch parity
must pass on explicit report TRAIN controls before collation, including when the
dataset split is validation. It never qualifies untested rows or GPU/model runs.
Every row still checks prefix/mask, grids, answer decoding and explicit budget;
no truncation, packing, automatic budget, multiprocessing processor loader or
model APIs. Torch is imported lazily for CPU tensor checking only.

Keep release, code, snapshot and processor immutable throughout use. Stat guards
and loaded-row/RGB hashes detect ordinary changes; they are NOT filesystem locks
or a malicious-concurrent-writer sandbox. Call verify_sources() on the collator
at run boundaries to rehash all release files and processor/source bindings.
Tokenizers/source files are not repeatedly hashed every training step. Row
handles are process-local, not intended for concurrent threads on one instance.
No approval, biological-independence, weight-integrity or training claim.
"""
from copy import copy, deepcopy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys

import numpy as np

from . import native_clear_export as export
from . import native_clear_processor_preflight as preflight
from .bundle import MAX_FILE_BYTES, digest, read_bounded, safe_path


SCHEMA = 'greenhouse.native_training_data.v1'
PARITY_SCHEMA = 'greenhouse.native_numpy_torch_parity.v1'
_CODE_SHA = export._file_sha(__file__)
_CORE = frozenset(('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'))
_OPTIONAL = frozenset(('token_type_ids', 'mm_token_type_ids'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _stat(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _bindings():
    return dict(adapter_source_sha256=_CODE_SHA, preflight_source_sha256=preflight._CODE_SHA,
                release_implementation_bindings=dict(export._LOADED_CODE))


def _code(expected):
    _require(expected == _bindings(), 'Parent/worker source bindings changed')
    export._check_code(expected['release_implementation_bindings'])
    _require(export._file_sha(__file__) == expected['adapter_source_sha256']
             and export._file_sha(preflight.__file__) == expected['preflight_source_sha256'],
             'Adapter/preflight source changed')


class NativeRows:
    """Map-style lazy JSONL; full validation is mandatory, never a caller boolean."""

    def __init__(self, root, *, manifest_sha256, split='train', allow_test=False):
        self._split_check(split, allow_test)
        _require(Path(root).is_absolute(), 'Explicit absolute release required')
        self.root = Path(root).resolve()
        self.manifest_sha256 = export._sha(manifest_sha256)
        self._bindings = _bindings()
        _code(self._bindings)
        checked = export.validate(self.root, manifest_sha256=self.manifest_sha256)
        raw = read_bounded(self.root / 'manifest.json')
        _require(digest(raw) == self.manifest_sha256, 'Manifest changed after validation')
        manifest = export._parse(raw)
        self._files = dict(manifest['files'])
        self._counts = dict(checked['counts'])
        self._frozen_splits = dict(checked['frozen_splits'])
        self._indices, self._ids, self._index_stats = {}, {}, {}
        for name in export.SPLITS:
            path = safe_path(self.root, f'splits/{name}.jsonl')
            before = _stat(path)
            entries, ids, sha = [], {}, hashlib.sha256()
            with path.open('rb') as stream:
                for expected in checked['records'][name]:
                    offset = stream.tell()
                    line = stream.readline(MAX_FILE_BYTES + 1)
                    _require(0 < len(line) <= MAX_FILE_BYTES and line.endswith(b'\n'), 'Invalid bounded JSONL line')
                    record = export._parse(line)
                    _require(record == expected and record['split'] == name, 'Validated JSONL row changed')
                    key = record['sample_id']
                    _require(key not in ids, 'Duplicate row identity')
                    ids[key] = len(entries)
                    entries.append((offset, len(line), digest(line), export.json_sha256(record), key))
                    sha.update(line)
                _require(not stream.read(1), 'Unexpected extra JSONL rows')
            _require(sha.hexdigest() == self._files[f'splits/{name}.jsonl'] and before == _stat(path),
                     'Split source pin changed')
            _require(len(entries) == self._counts[name], 'Validated split count changed')
            self._indices[name], self._ids[name], self._index_stats[name] = tuple(entries), ids, before
        self._manifest_stat = _stat(self.root / 'manifest.json')
        _require(export._file_sha(self.root / 'manifest.json') == self.manifest_sha256, 'Manifest changed')
        self.split, self._stream, self._pid = split, None, None
        self._checked_pid = os.getpid()
        # checked (including every materialized chat) is intentionally not retained.

    @staticmethod
    def _split_check(split, allow_test):
        _require(split in export.SPLITS and type(allow_test) is bool, 'Explicit supported split required')
        _require(split != 'test' or allow_test, 'Test split sealed; explicit allow_test=True required')

    @property
    def receipt(self):
        return dict(schema=SCHEMA, release_manifest_sha256=self.manifest_sha256,
            **deepcopy(self._bindings), counts=dict(self._counts),
            frozen_splits=dict(self._frozen_splits), training_approved=False)

    def for_split(self, split, *, allow_test=False):
        self._split_check(split, allow_test)
        result = copy(self)
        result.split, result._stream, result._pid = split, None, None
        return result

    def __len__(self):
        return len(self._indices[self.split])

    def _unchanged(self):
        if self._checked_pid != os.getpid():
            _code(self._bindings)
            self._checked_pid = os.getpid()
        _require(_stat(self.root / 'manifest.json') == self._manifest_stat, 'Manifest stat changed; revalidate')
        _require(_stat(safe_path(self.root, f'splits/{self.split}.jsonl')) == self._index_stats[self.split],
                 'Split stat changed; revalidate')

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError('One integer row index required')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._unchanged()
        if self._stream is None or self._pid != os.getpid():
            self.close()
            path = safe_path(self.root, f'splits/{self.split}.jsonl')
            _require(export._file_sha(path) == self._files[f'splits/{self.split}.jsonl'], 'Worker split pin changed')
            self._stream, self._pid = path.open('rb'), os.getpid()
        offset, length, sha, _, _ = self._indices[self.split][index]
        self._stream.seek(offset)
        raw = self._stream.read(length)
        _require(len(raw) == length and digest(raw) == sha, 'Loaded JSONL row pin changed')
        self._unchanged()
        return export._parse(raw)

    def by_id(self, sample_id):
        _require(sample_id in self._ids[self.split], 'ID outside selected split')
        return self[self._ids[self.split][sample_id]]

    def _bound_record(self, record):
        _require(isinstance(record, dict) and record.get('split') == self.split
                 and record.get('sample_id') in self._ids[self.split], 'Row outside bound dataset split')
        entry = self._indices[self.split][self._ids[self.split][record['sample_id']]]
        _require(export.json_sha256(record) == entry[3], 'Row content differs from validated source')
        self._unchanged()

    def verify_sources(self):
        """Explicit full rehash, not a replacement for initial semantic validation."""
        _code(self._bindings)
        _require(export._file_sha(self.root / 'manifest.json') == self.manifest_sha256, 'Manifest pin changed')
        _require(export._regular_tree(self.root) == set(self._files) | {'manifest.json'}, 'Release tree changed')
        for name, pin in self._files.items():
            _require(export._file_sha(safe_path(self.root, name)) == pin, 'Release source pin changed: ' + name)
        return self.receipt

    def close(self):
        if self._stream is not None:
            self._stream.close()
        self._stream, self._pid = None, None

    def __getstate__(self):
        return {**self.__dict__, '_stream': None, '_pid': None, '_checked_pid': None}

    def __del__(self):
        stream = getattr(self, '_stream', None)
        if stream is not None:
            stream.close()


def decode_answer(raw):
    """Strict JSON -> original 1696x816 pixels; no repair or crop offset."""
    _require(isinstance(raw, (str, bytes)), 'Raw JSON answer required')
    return export.contract.canonical_answer(export._parse(raw))


def _versions():
    result = {}
    for name in ('transformers', 'tokenizers', 'numpy', 'Pillow', 'torch'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _arrays(output, kind):
    _require(isinstance(output, dict) or hasattr(output, 'items'), 'Processor mapping required')
    if kind == 'pt':
        import torch
        _require(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and not v.requires_grad
                     for v in output.values()), 'CPU Torch tensors without gradients required')
        return {k: v.detach().numpy() for k, v in output.items()}
    _require(all(isinstance(v, np.ndarray) for v in output.values()), 'NumPy arrays required')
    return dict(output)


class NativeCollator:
    """Single-example CPU collator; requires explicit parity qualification first."""

    def __init__(self, rows, processor, *, preflight_report, model_snapshot,
                 maximum_tokens, model_input_keys):
        _require(isinstance(rows, NativeRows), 'Validated NativeRows required')
        preflight._budget(maximum_tokens)
        _require(isinstance(model_input_keys, (set, frozenset, tuple, list))
                 and len(set(model_input_keys)) == len(model_input_keys)
                 and _CORE <= set(model_input_keys) <= _CORE | _OPTIONAL,
                 'Explicit supported downstream model_input_keys required')
        self.rows, self.processor = rows, processor
        self.maximum_tokens, self.model_input_keys = maximum_tokens, frozenset(model_input_keys)
        self._bound_keys = self.model_input_keys
        self._report_pin = preflight_report
        self._report = export._parse(export._pinned(preflight_report))
        report = self._report
        _require(report.get('schema') == preflight.SCHEMA and report.get('state') == 'checked_rows_processor_only'
                 and report.get('release_manifest_sha256') == rows.manifest_sha256
                 and report.get('maximum_tokens') == maximum_tokens, 'Preflight release/schema/budget mismatch')
        _require(report.get('offline') is True and all(report.get(k) is False for k in (
            'weights_loaded', 'weight_integrity_verified', 'training_performed', 'depth_is_model_input',
            'route_integration_complete', 'training_approved')), 'Processor-only preflight flags required')
        _require(report.get('scope') == 'explicit_train_rows_known_answers_not_unchecked_rows_or_future_generations',
                 'Unsupported preflight scope')
        controls = report.get('rows')
        _require(isinstance(controls, list) and controls, 'Nonempty preflight controls required')
        self._controls = {}
        train = rows.for_split('train')
        for control in controls:
            record = train.by_id(control['sample_id'])
            _require(control['sample_id'] not in self._controls
                     and control['record_sha256'] == export.json_sha256(record), 'Preflight control row binding changed')
            self._controls[control['sample_id']] = deepcopy(control)
        train.close()
        _require(report.get('maximum_observed_total_tokens') == max(c['total_tokens'] for c in controls)
                 and all(0 < c['total_tokens'] <= maximum_tokens and c['maximum_tokens'] == maximum_tokens
                         for c in controls), 'Preflight control budget mismatch')
        self._snapshot = Path(model_snapshot)
        self._ready = False
        self._parity = None
        self.verify_sources()

    def _settings(self):
        p, r = self.processor, self._report
        _require(self.maximum_tokens == r['maximum_tokens'] and self.model_input_keys == self._bound_keys,
                 'Bound budget/model-input configuration changed')
        _require(type(p).__name__ == 'Qwen3VLProcessor' and p.image_token_id == self._image_token,
                 'Processor/image-token identity changed')
        template = getattr(p, 'chat_template', None)
        _require(isinstance(template, str) and bool(template.strip())
                 and digest(template.encode()) == r['chat_template_sha256']
                 and p.image_processor.to_dict() == r['image_processor_config'], 'Processor settings changed')

    def verify_sources(self):
        """Recheck full release, pinned report, local processor files/runtime/code."""
        try:
            return self._verify_sources()
        except Exception:
            self._ready, self._parity = False, None
            raise

    def _verify_sources(self):
        self.rows.verify_sources()
        _require(export._parse(export._pinned(self._report_pin)) == self._report, 'Preflight report changed')
        r = self._report
        _require(r.get('preflight_source_sha256') == preflight._CODE_SHA
                 and r.get('release_implementation_bindings') == export._LOADED_CODE,
                 'Preflight source bindings changed')
        _require(r.get('runtime_versions') == _versions() and r.get('python_version') == sys.version,
                 'Preflight runtime versions changed; rerun on this runtime')
        pins = r['processor_files']
        _require(export.json_sha256(pins) == r['processor_files_sha256'], 'Processor pin-map changed')
        snapshot, config = preflight._snapshot(self._snapshot, pins)
        _require(not snapshot.is_relative_to(self.rows.root) and not self.rows.root.is_relative_to(snapshot),
                 'Separate release and processor snapshot required')
        self._image_token = config['image_token_id']
        self._settings()
        _require(preflight._implementation(self.processor) == r['processor_implementation_bindings']
                 and preflight._tokenizer_sha(self.processor) == r['effective_tokenizer_sha256'],
                 'Effective processor implementation/tokenizer changed')
        return self.rows.receipt

    def _encode(self, rows, record, kind):
        rows._bound_record(record)
        self._settings()
        path = safe_path(rows.root, record['full_rgb'])
        pin = rows._files[record['full_rgb']]
        _require(export._file_sha(path) == pin, 'RGB source pin changed')
        messages = export.model_messages(rows.root, record, supervised=True, crop=True)
        _require(export._file_sha(path) == pin, 'RGB changed while loading')
        inference = messages[:2]
        images = [item['image'] for item in messages[1]['content'][:2]]
        before = [image.tobytes() for image in images]
        _require([im.size for im in images] == [(1696, 816), (768, 768)]
                 and all(im.mode == 'RGB' for im in images)
                 and images[0].crop(record['crop_box']).tobytes() == before[1], 'Native full/crop pixels changed')
        kwargs = dict(tokenize=True, return_dict=True, return_tensors=kind, padding=False, truncation=False)
        full = self.processor.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
        prefix = self.processor.apply_chat_template(inference, add_generation_prompt=True, **kwargs)
        values, prompt_values = _arrays(full, kind), _arrays(prefix, kind)
        _require(set(values) == set(prompt_values) == self.model_input_keys, 'Downstream model-input keys mismatch')
        ids, prompt = preflight._sequence(values), preflight._sequence(prompt_values)
        count, total = prompt.shape[1], ids.shape[1]
        _require(0 < count < total and np.array_equal(ids[:, :count], prompt), 'Generation prefix/answer boundary mismatch')
        _require(total <= self.maximum_tokens, f'Token budget exceeded: required={total}, budget={self.maximum_tokens}; never truncate')
        settings = {k: getattr(self.processor.image_processor, k, None)
                    for k in ('patch_size', 'temporal_patch_size', 'merge_size')}
        grid, pixels, image_tokens = preflight._image_output(values, settings)
        pg, pp, _ = preflight._image_output(prompt_values, settings)
        _require(np.array_equal(grid, pg) and np.array_equal(pixels, pp), 'Full/prefix image processing mismatch')
        for name in self.model_input_keys & _OPTIONAL:
            _require(values[name].dtype.kind in 'iu' and prompt_values[name].dtype.kind in 'iu'
                     and np.array_equal(values[name][:, :count], prompt_values[name]), 'Token-type prefix mismatch')
        positions = np.flatnonzero(prompt[0] == self._image_token)
        runs = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
        _require([len(run) for run in runs] == image_tokens and not (ids[0, count:] == self._image_token).any(),
                 'Expanded image tokens do not match the two ordered grids')
        labels = ids.astype(np.int64, copy=True)
        labels[:, :count] = -100
        answer_text = messages[-1]['content'][0]['text']
        decoded = self.processor.tokenizer.decode(labels[labels != -100].tolist(), skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False).strip()
        _require(export._parse(decoded) == export._parse(answer_text), 'Assistant decoding changed')
        decoded_prompt = self.processor.tokenizer.decode(prompt[0].tolist(), skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
        _require(all(turn['content'][-1]['text'] in decoded_prompt for turn in inference)
                 and answer_text not in decoded_prompt, 'System/query missing or answer leaked into prefix')
        _require(before == [image.tobytes() for image in images], 'Processor mutated native inputs')
        self._settings()
        rows._bound_record(record)
        if kind == 'pt':
            import torch
            full = dict(full, labels=torch.from_numpy(labels))
        else:
            full = dict(full, labels=labels)
        return full, dict(prompt_tokens=count, total_tokens=total, image_grid_thw=grid.tolist(),
            input_ids_sha256=preflight._array_sha(ids), prompt_ids_sha256=preflight._array_sha(prompt),
            labels_sha256=preflight._array_sha(labels), pixel_values_sha256=preflight._array_sha(pixels))

    def __call__(self, batch):
        _require(self._ready, 'Explicit NumPy/Torch parity controls must pass before collation')
        _require(isinstance(batch, (tuple, list)) and len(batch) == 1, 'Exactly one native example per microbatch')
        output, _ = self._encode(self.rows, batch[0], 'pt')
        return output


def check_numpy_torch_parity(collator, *, sample_ids):
    """CPU qualification seam; fresh frozen NumPy controls vs actual PT arrays.

    Returns a serializable, hash-bindable report. No artifact is written. A
    failed/restarted check disarms the collator, even after a previous success.
    """
    _require(isinstance(collator, NativeCollator), 'Bound NativeCollator required')
    collator._ready, collator._parity = False, None
    _require(isinstance(sample_ids, (list, tuple)) and sample_ids
             and all(isinstance(k, str) and k in collator._controls for k in sample_ids)
             and len(set(sample_ids)) == len(sample_ids), 'Explicit unique preflight TRAIN control IDs required')
    collator.verify_sources()
    rows = collator.rows.for_split('train')
    checked = []
    try:
        for key in sample_ids:
            record = rows.by_id(key)
            evidence = preflight.check_record(rows.root, record, collator.processor,
                                             maximum_tokens=collator.maximum_tokens)
            _require(evidence == collator._controls[key], 'Fresh NumPy evidence differs from pinned preflight')
            numpy_output, _ = collator._encode(rows, record, 'np')
            torch_output, observed = collator._encode(rows, record, 'pt')
            arrays = _arrays(torch_output, 'pt')
            _require(set(arrays) == set(numpy_output) and all(arrays[k].dtype == numpy_output[k].dtype
                     and np.array_equal(arrays[k], numpy_output[k]) for k in arrays), 'NumPy/Torch parity mismatch')
            _require(all(evidence[k] == value for k, value in observed.items()), 'Torch/preflight evidence mismatch')
            checked.append(dict(sample_id=key, record_sha256=export.json_sha256(record), **observed,
                output_sha256={k: preflight._array_sha(v) for k, v in arrays.items()}))
        collator.verify_sources()
    finally:
        rows.close()
    result = dict(schema=PARITY_SCHEMA, release=collator.rows.receipt,
        preflight_report_sha256=collator._report_pin.sha256, maximum_tokens=collator.maximum_tokens,
        model_input_keys=sorted(collator.model_input_keys), rows=checked,
        scope='explicit_train_controls_only_not_unchecked_rows_or_model_forward',
        training_performed=False, model_loaded=False, depth_is_model_input=False, training_approved=False)
    collator._parity, collator._ready = deepcopy(result), True
    return result
