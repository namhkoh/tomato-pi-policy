"""Compact-source adapter for the unchanged near_image_index.build_graph.

Compact RGB currently uses identity encoding: its stored payload IS the native
PNG, not a compressed archive member. Authenticate it with a fresh SampleReader,
then pass its existing physical path to the original ImagePin/index. No image
writes, extraction, re-encoding, monkeypatching or graph-algorithm fork.
The original index still checks PNG format/mode, encoded and decoded RGB hashes,
patch bounds, pruning and exact scores. Its v1 schema/code/results are unchanged.

API: CompactImagePin(sample_id, root, bundle_sha256, sample_sha256, label_sha256,
encoded_rgb_sha256, decoded_rgb_sha256, nominal_uv); build_graph(pins, **options).
Pins may mix CompactImagePin and the original near_image_index.ImagePin.
``root`` is an explicit absolute compact SampleReader root, not a raw directory
or a cached reader. All hashes are caller-supplied external pins. nominal_uv is
caller-supplied original label['nominal_pixel_uv']; we verify equality, never
derive it from anatomical attachment, pixels, depth or camera geometry.

The v2 graph keeps the original flat graph fields and adds compact source pins,
sample/bundle/label/eligible-trace bindings, and adapter implementation bindings.
Compact sources are reopened and all bound bytes rechecked at completion. This
is RGB/metadata validation, NOT a full native-buffer or annotation audit. Raw
ImagePins retain their original caller-owned label/inventory provenance contract.
Inventory coverage, sample/target ancestry, nominal-label semantics, calibration
and release approval remain caller responsibilities. No threshold is selected.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path

from . import admission, bundle, near_image, near_image_index
from .. import dataset_review, native_lossless_codec

SCHEMA = 'greenhouse.native_near_image_graph.v2'
SOURCE_ADAPTER = 'compact_identity_png_sample_reader.v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _verify(bindings):
    for path, expected in bindings.items():
        _require(_sha(path) == expected, 'Source changed during compact graph construction: ' + path)


def _merge(destination, bindings):
    for path, expected in bindings.items():
        _require(path not in destination or destination[path] == expected, 'Conflicting source pins: ' + path)
        destination[path] = expected


_LOADED_CODE = {str(Path(path).resolve()): _sha(path) for path in (
    __file__, near_image_index.__file__, near_image.__file__, bundle.__file__,
    admission.__file__, dataset_review.__file__, native_lossless_codec.__file__)}


@dataclass(frozen=True)
class CompactImagePin:
    sample_id: str
    root: str
    bundle_sha256: str
    sample_sha256: str
    label_sha256: str
    encoded_rgb_sha256: str
    decoded_rgb_sha256: str
    nominal_uv: tuple


def _snapshot(pin):
    _require(isinstance(pin, (CompactImagePin, near_image_index.ImagePin)),
             'Explicit CompactImagePin or original ImagePin required')
    _require(isinstance(pin.nominal_uv, (tuple, list)) and len(pin.nominal_uv) == 2
             and all(type(v) in (int, float) and math.isfinite(v) for v in pin.nominal_uv),
             'Explicit finite nominal pixel coordinates required')
    if isinstance(pin, CompactImagePin):
        for field in ('bundle_sha256', 'sample_sha256', 'label_sha256',
                      'encoded_rgb_sha256', 'decoded_rgb_sha256'):
            value = getattr(pin, field)
            _require(isinstance(value, str) and len(value) == 64
                     and all(c in '0123456789abcdef' for c in value), 'Explicit SHA256 required: ' + field)
        _require(isinstance(pin.root, str) and Path(pin.root).is_absolute(),
                 'Explicit absolute compact SampleReader root required')
    return replace(pin, nominal_uv=tuple(pin.nominal_uv))


def _resolve(pin):
    """Use only authenticated, already-existing identity payload paths."""
    root = Path(pin.root)
    marker = root / 'bundle.json'
    _require(marker.is_file(), 'Explicit complete compact SampleReader source required')
    _require(bundle.digest(bundle.read_bounded(marker)) == pin.bundle_sha256, 'Compact bundle pin changed')
    expected = {'sample.json': pin.sample_sha256, 'supervision/label.json': pin.label_sha256,
                'inputs/rgb.png': pin.encoded_rgb_sha256}
    reader = bundle.SampleReader(root, expected_bindings=expected)
    _require(reader.manifest is not None, 'Compact source became raw')
    label = reader.json('supervision/label.json')
    uv = label.get('nominal_pixel_uv')
    _require(isinstance(uv, (list, tuple)) and len(uv) == 2
             and all(type(v) in (int, float) and math.isfinite(v) for v in uv)
             and tuple(uv) == pin.nominal_uv, 'Caller nominal coordinates differ from pinned original label')
    names = list(expected)
    if label.get('eligible') is True:
        names.append('supervision/query_trace.json')  # Also required by SampleReader.
    bindings = {str(marker.resolve()): pin.bundle_sha256}
    paths = {}
    for name in names:
        raw = reader.read(name)
        entry = reader.manifest['files'][name]
        _require(entry['encoding'] == 'identity' and entry['stored_sha256'] == bundle.digest(raw),
                 'Existing identity-encoded metadata/RGB payload required')
        path = bundle.safe_path(root, entry['stored_path'])
        _merge(bindings, {str(path): entry['stored_sha256']})
        paths[name] = path
    _require(bundle.digest(bundle.read_bounded(marker)) == pin.bundle_sha256, 'Compact bundle changed while resolving')
    _verify(bindings)
    image = near_image_index.ImagePin(pin.sample_id, str(paths['inputs/rgb.png']),
        pin.encoded_rgb_sha256, pin.decoded_rgb_sha256, pin.nominal_uv)
    return image, bindings


def build_graph(pins, *, inventory_sha256, threshold, tile=32, cache_images=16, progress=None):
    """Same index options/metric, with explicit compact pins and a new schema.

    Never supply a bare SampleReader or infer its pins from mutable content.
    The caller authenticates sample IDs/target ancestry against inventory_sha256.
    Exceptions abort construction; no partially complete graph is returned.
    """
    sources = [_snapshot(pin) for pin in pins]
    _require(sources and all(isinstance(p.sample_id, str) and p.sample_id for p in sources)
             and len({p.sample_id for p in sources}) == len(sources), 'Unique nonempty sample IDs required')
    sources.sort(key=lambda pin: pin.sample_id)
    _verify(_LOADED_CODE)
    images, compact, bindings = [], [], {}
    for source in sources:
        if isinstance(source, CompactImagePin):
            image, bound = _resolve(source)
            _merge(bindings, bound)
            compact.append((source, image, bound))
        else:
            image = source
        images.append(image)
    # The entire existing pruning/cache/exact-score implementation is reused.
    base = near_image_index.build_graph(images, inventory_sha256=inventory_sha256,
        threshold=threshold, tile=tile, cache_images=cache_images, progress=progress)
    for source, image, bound in compact:
        final_image, final_bound = _resolve(source)
        _require(final_image == image and final_bound == bound, 'Compact source path or identity changed')
    _merge(bindings, base['input_bindings'])
    code = dict(_LOADED_CODE)
    _merge(code, base['implementation_bindings'])
    _verify(bindings)
    _verify(code)
    result = dict(base, schema=SCHEMA, source_adapter=SOURCE_ADAPTER,
        base_graph_schema=base['schema'], base_graph_sha256=base['sha256'],
        compact_image_pins=[asdict(source) for source, _, _ in compact],
        input_bindings=bindings, implementation_bindings=code,
        compact_validation_scope='manifest_sample_label_eligible_trace_and_rgb_not_full_native_audit',
        compact_sources_rechecked=True, source_nominal_coordinates='caller_supplied_bound_original_label',
        full_native_audit_performed=False, images_extracted=False, images_reencoded=False)
    result.pop('sha256')
    result['sha256'] = near_image_index._digest(result)
    return result
