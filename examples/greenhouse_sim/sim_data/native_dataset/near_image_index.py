"""Exact observed-inventory near-image graph with conservative block-mean pruning.

No resize, image writes, learned embeddings, depth, labels, or training approval.
The triangle inequality gives a lower bound on native MAE from weighted block
means. A bound can only reject a duplicate pair; all remaining pairs receive the
unchanged full-frame/nominal-cut-patch MAE test. This is not a novelty classifier.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import near_image
from .admission import decoded_rgb_digest

SCHEMA = 'greenhouse.native_near_image_graph.v1'
BOUND_GUARD = 1e-10  # Conservative float64 roundoff margin, never subtracted.


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ImagePin:
    sample_id: str
    path: str
    sha256: str
    decoded_rgb_sha256: str
    nominal_uv: tuple


def _load(pin):
    path = Path(pin.path)
    _require(path.is_absolute() and path.is_file(), 'Explicit existing native PNG required')
    _require(_sha(path) == pin.sha256, 'Native PNG changed: '+str(path))
    with Image.open(path) as image:
        _require(image.format == 'PNG' and image.mode == 'RGB', 'Unconverted native RGB PNG required')
        rgb = np.asarray(image).copy()
    near_image._rgb(rgb)
    _require(decoded_rgb_digest(rgb.tobytes(), width=rgb.shape[1], height=rgb.shape[0])
             == pin.decoded_rgb_sha256, 'Decoded native RGB identity changed')
    return rgb


def _features(rgb, tile):
    """Cell/channel means and exact-area weights; partial edge cells retained."""
    ys, xs = np.arange(0, rgb.shape[0], tile), np.arange(0, rgb.shape[1], tile)
    counts = np.outer(np.diff(np.r_[ys, rgb.shape[0]]), np.diff(np.r_[xs, rgb.shape[1]]))
    sums = np.add.reduceat(np.add.reduceat(rgb.astype(np.uint64), ys, axis=0), xs, axis=1)
    means = sums.astype(np.float64) / counts[..., None] / 255.0
    weights = np.repeat((counts / (rgb.shape[0]*rgb.shape[1]*3))[..., None], 3, axis=2)
    return means.ravel(), weights.ravel()


def _patch(rgb, uv):
    a = near_image._patch_bounds(rgb, uv, 64)
    return rgb[a[1]:a[3], a[0]:a[2]]


def _mae(left, right):
    delta = np.abs(left.astype(np.int16)-right.astype(np.int16))
    return float(delta.mean()) / 255.0


def exact_score(left, right, left_uv, right_uv):
    """Same score as near_image.compare, without unused percentiles/sorting."""
    near_image._rgb(left); near_image._rgb(right)
    _require(left.shape == right.shape, 'Mixed native resolutions')
    return max(_mae(left, right), _mae(_patch(left, left_uv), _patch(right, right_uv)))


def build_graph(pins, *, inventory_sha256, threshold, tile=32, cache_images=16, progress=None):
    """Resolve every pair of EXPLICIT pins; missing/unreadable sources fail.

    inventory_sha256 identifies the caller's observed snapshot, not a declaration
    of global collection coverage. Every pin's RGB bytes are checked initially
    and again before completion. The returned graph never assigns approval.
    """
    pins = sorted(list(pins), key=lambda p: p.sample_id)
    _require(pins and all(isinstance(p, ImagePin) for p in pins), 'Explicit ImagePins required')
    pins = [ImagePin(p.sample_id, p.path, p.sha256, p.decoded_rgb_sha256, tuple(p.nominal_uv)) for p in pins]
    _require(all(isinstance(p.sample_id, str) and p.sample_id for p in pins)
             and len({p.sample_id for p in pins}) == len(pins), 'Unique nonempty sample IDs required')
    _require(isinstance(inventory_sha256, str) and len(inventory_sha256) == 64
             and all(c in '0123456789abcdef' for c in inventory_sha256), 'Inventory SHA256 required')
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1,
             'Finite empirical threshold in [0,1] required')
    _require(type(tile) is int and 1 <= tile <= 256 and type(cache_images) is int
             and 1 <= cache_images <= 128, 'Bounded positive feature/cache sizes required')
    code = {str(Path(p).resolve()): _sha(p) for p in (__file__, near_image.__file__)}
    inputs, full_features, local_features, dimensions, cache = {}, [], [], None, OrderedDict()

    def pixels(index):
        if index not in cache:
            cache[index] = _load(pins[index])
            if len(cache) > cache_images:
                cache.popitem(last=False)
        cache.move_to_end(index)
        return cache[index]

    for index, pin in enumerate(pins):
        path = str(Path(pin.path).resolve())
        _require(path not in inputs or inputs[path] == pin.sha256, 'Conflicting native file pins')
        inputs[path] = pin.sha256
        rgb = pixels(index)
        if dimensions is None:
            dimensions = rgb.shape
        _require(rgb.shape == dimensions, 'Mixed native resolutions require separate calibrated graphs')
        full, weights = _features(rgb, tile)
        local, local_weights = _features(_patch(rgb, pin.nominal_uv), tile)
        full_features.append(full); local_features.append(local)
        if progress is not None:
            progress(dict(phase='features', completed=index+1, total=len(pins)))
    full_features, local_features = np.asarray(full_features), np.asarray(local_features)
    edges, scores, pruned, compared = [], [], 0, 0
    for left in range(len(pins)):
        for start in range(left+1, len(pins), 256):
            indices = np.arange(start, min(start+256, len(pins)))
            low = np.sum(np.abs(full_features[indices]-full_features[left])*weights, axis=1)
            local_low = np.sum(np.abs(local_features[indices]-local_features[left])*local_weights, axis=1)
            rejected = np.maximum(low, local_low) > threshold+BOUND_GUARD
            pruned += int(rejected.sum())
            for right in indices[~rejected]:
                right = int(right)
                score = exact_score(pixels(left), pixels(right), pins[left].nominal_uv, pins[right].nominal_uv)
                compared += 1
                if score <= threshold:
                    edges.append([pins[left].sample_id, pins[right].sample_id]); scores.append(score)
        if progress is not None:
            progress(dict(phase='pairs', completed_rows=left+1, total_rows=len(pins),
                          exact_compared_pairs=compared, bound_pruned_pairs=pruned))
    for path, expected in {**inputs, **code}.items():
        _require(_sha(path) == expected, 'Source changed during graph construction: '+path)
    for pin in pins:
        _require(_sha(pin.path) == pin.sha256, 'Original image path changed during graph construction')
    total = len(pins)*(len(pins)-1)//2
    _require(compared+pruned == total, 'Incomplete pair coverage')
    result = dict(schema=SCHEMA, inventory_sha256=inventory_sha256,
        sample_ids=[p.sample_id for p in pins], image_pins=[asdict(p) for p in pins],
        threshold=threshold, metric=near_image.METRIC, patch_radius=64,
        patch_anchor='nominal_10mm_cut_point_not_anatomical_attachment',
        dimensions=[dimensions[1], dimensions[0]], near_image_edges=edges, edge_scores=scores,
        counts=dict(total_pairs=total, exact_compared_pairs=compared, bound_pruned_pairs=pruned,
                    resolved_pairs=compared+pruned), complete=True,
        pruning=dict(method='weighted_native_block_mean_MAE_lower_bound', tile=tile, guard=BOUND_GUARD),
        input_bindings=inputs, implementation_bindings=code,
        scope='all_pairs_of_explicit_observed_pins_only', global_collection_complete=False,
        threshold_calibrated=False, training_approved=False, depth_recomputed=False, images_modified=False)
    result['sha256'] = _digest(result)
    return result
