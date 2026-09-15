'''Uncalibrated distance on V2 fixed features and unordered leaf FEATURE SETS.

distance = max(fixed L-infinity, symmetric Hausdorff leaf-set L-infinity).
Multiplicity is ignored: coincident/repeated feature rows add zero distance.
Matching is NOT injective; different leaves may share a nearest neighbour.
This deliberately can merge distinct arrangements. Features themselves only
summarize geometry (including vertex covariance), not full authored meshes.

This is a separately versioned pseudometric on descriptor records, not the
old equal-cardinality bottleneck metric. No old cutoff/calibration transfers.
No extraction, I/O, novelty threshold, qualification, or native-image claim.
Both empty sets have zero leaf distance; one empty set is unsupported, not
infinite novelty. Actual supported extractor outputs have nonempty leaves.
'''
import numpy as np

from .morphology_frame_v2 import validate

METRIC = 'v2_fixed_linf_symmetric_leaf_feature_set_hausdorff.v1'


def compare(left, right):
    validate(left); validate(right)
    fixed = float(np.max(np.abs(np.asarray(left['fixed']) - right['fixed'])))
    n, m = left['leaf_count'], right['leaf_count']
    if bool(n) != bool(m):
        raise ValueError('Empty/nonempty leaf sets are unsupported, not novelty')
    forward = reverse = 0.
    if n:
        costs = np.max(np.abs(np.asarray(left['leaves'])[:, None, :]
                             - np.asarray(right['leaves'])[None, :, :]), axis=2)
        forward = float(np.max(np.min(costs, axis=1)))
        reverse = float(np.max(np.min(costs, axis=0)))
    value = max(fixed, forward, reverse)
    if not np.isfinite(value):
        raise ValueError('Unrepresentable feature distance')
    return dict(metric=METRIC, distance=value, fixed_linf=fixed,
                leaf_forward_linf=forward, leaf_reverse_linf=reverse,
                leaf_hausdorff_linf=max(forward, reverse))


def distance(left, right):
    return compare(left, right)['distance']
