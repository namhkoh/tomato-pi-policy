"""Native RGB similarity measurements; never an automatic novelty approval.

Compare full frames at their original pixel positions AND equal-size patches
centred on the projected junction. The patch uses integer indexing only: no
resizing, interpolation, brightness normalization, depth calculation or image
output. Whole-frame and local evidence are retained separately. Exact decoded
RGB duplicate detection remains independent of this photometric comparator.

Known matched rerenders are duplicate controls, not new training observations.
``separation_report`` reports empirical TRAIN-control separation only. It does
not choose a release threshold, attest control identities, or validate globally.
"""
import math
import numpy as np

SCHEMA = 'greenhouse.native_rgb_similarity.v1'
METRIC = 'max_full_and_junction_patch_rgb_mae_0to1_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rgb(value):
    _require(isinstance(value, np.ndarray) and value.dtype == np.uint8,
             'Native uint8 RGB array required')
    _require(value.ndim == 3 and value.shape[2] == 3 and min(value.shape[:2]) > 0,
             'Nonempty HxWx3 RGB required; no alpha or inferred conversion')
    return value


def _patch_bounds(rgb, uv, radius):
    _require(type(radius) is int and radius >= 1, 'Positive integer patch radius required')
    _require(isinstance(uv, (list, tuple)) and len(uv) == 2,
             'Explicit projected junction [u,v] required')
    _require(all(type(v) in (int, float) and math.isfinite(v) for v in uv),
             'Finite numeric junction coordinates required')
    height, width = rgb.shape[:2]
    _require(0 <= uv[0] < width and 0 <= uv[1] < height, 'Junction outside native image')
    # Edge-origin camera convention: pixel i has centre i+0.5. Do not use round().
    u, v = (math.floor(value) for value in uv)
    bounds = [u-radius, v-radius, u+radius+1, v+radius+1]
    _require(bounds[0] >= 0 and bounds[1] >= 0
             and bounds[2] <= width and bounds[3] <= height,
             'Complete native junction patch required; no padding or clipping')
    return bounds


def _measure(left, right):
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return dict(mae=float(delta.mean())/255.0,
                p95=float(np.percentile(delta, 95))/255.0,
                max_delta=int(delta.max()), changed_channels=int(np.count_nonzero(delta)))


def compare(left, right, *, left_junction_uv, right_junction_uv, patch_radius=64):
    """Measure a pair without mutating arrays or declaring duplicate/novelty.

    A complete 129x129 patch is the default. Different resolutions or clipped
    patches must be held by the caller, not implicitly resized or padded.
    Junction coordinates are supervision for offline dedup only; never expose
    hidden junction/cut coordinates to the execution policy through this API.
    """
    left, right = _rgb(left), _rgb(right)
    _require(left.shape == right.shape, 'Native image dimensions must match')
    a = _patch_bounds(left, left_junction_uv, patch_radius)
    b = _patch_bounds(right, right_junction_uv, patch_radius)
    full = _measure(left, right)
    local = _measure(left[a[1]:a[3], a[0]:a[2]], right[b[1]:b[3], b[0]:b[2]])
    return dict(schema=SCHEMA, metric=METRIC, dimensions=[left.shape[1], left.shape[0]],
                patch_radius=patch_radius, left_patch_xyxy=a, right_patch_xyxy=b,
                full_frame=full, junction_patch=local,
                score=max(full['mae'], local['mae']),
                decoded_pixels_exact=bool(np.array_equal(left, right)),
                resized=False, interpolated=False, depth_computed=False,
                threshold_calibrated=False, training_approved=False)


def separation_report(controls):
    """Summarize measured, externally adjudicated TRAIN controls.

    Each control declares pair_id, source_family, source_target, split='train',
    control_kind in {'duplicate','meaningful_change'}, evidence_id and a compare
    result. Those declarations remain UNVERIFIED here. The report can diagnose
    overlap but cannot certify the supplied control labels or unseen donors.
    """
    _require(isinstance(controls, list) and len(controls) > 0, 'Nonempty control list required')
    by_kind = {'duplicate': [], 'meaningful_change': []}
    ids, donors, targets, definitions = set(), set(), set(), set()
    for row in controls:
        for key in ('pair_id', 'source_family', 'source_target', 'evidence_id'):
            _require(isinstance(row.get(key), str) and bool(row[key].strip()), 'Missing control identity')
        _require(row['pair_id'] not in ids, 'Repeated control pair identity')
        ids.add(row['pair_id'])
        _require(row.get('split') == 'train', 'Held-out controls must not tune calibration')
        kind = row.get('control_kind')
        _require(kind in by_kind, 'Unsupported control kind')
        value = row.get('measurement', {})
        _require(value.get('schema') == SCHEMA and value.get('metric') == METRIC,
                 'Mixed/unknown comparator metric')
        _require(value.get('training_approved') is False and value.get('threshold_calibrated') is False,
                 'Unapproved measurement required')
        score = value.get('score')
        _require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                 'Bounded finite measured score required')
        _require(score == max(value['full_frame']['mae'], value['junction_patch']['mae']),
                 'Inconsistent measured score')
        definitions.add((tuple(value['dimensions']), value['patch_radius']))
        by_kind[kind].append(score)
        donors.add(row['source_family']); targets.add((row['source_family'], row['source_target']))
    _require(len(definitions) == 1, 'Do not mix patch/resolution definitions')
    duplicate_max = max(by_kind['duplicate'], default=None)
    changed_min = min(by_kind['meaningful_change'], default=None)
    both = duplicate_max is not None and changed_min is not None
    separated = both and duplicate_max < changed_min
    return dict(schema='greenhouse.native_rgb_control_separation.v1', metric=METRIC,
                pair_count=len(ids), source_donor_count=len(donors), source_target_count=len(targets),
                counts={k: len(v) for k,v in by_kind.items()}, duplicate_max_score=duplicate_max,
                meaningful_change_min_score=changed_min,
                empirical_controls_separate=bool(separated),
                candidate_interval=([duplicate_max, changed_min] if separated else None),
                interval_convention='duplicate <= lower; change >= upper; upper excluded for duplicate threshold',
                control_labels_verified=False, threshold_selected=None, calibration_validated=False,
                training_approved=False,
                limitation='Control separation only; require authenticated identities, independent donor validation and global inventory')
