# Synthetic feature sets; no novelty/calibration evidence.
from copy import deepcopy

import numpy as np
import pytest

from . import morphology_frame_v2 as frame, morphology_set_v1 as metric


def descriptor(values, fixed=0.):
    return dict(schema=frame.SCHEMA, frame_version=frame.FRAME_VERSION,
                fixed=[fixed]*104, leaves=[[v]*15 for v in values], leaf_count=len(values))


def test_duplicate_addition_order_and_multiplicity_do_not_create_distance():
    a, b = descriptor([0., 1.]), descriptor([1., 0., 1., 0.])
    assert metric.distance(a, b) == metric.distance(b, a) == 0.
    with pytest.raises(ValueError):frame.distance(a, b)


def test_unmatched_leaf_distance_comes_from_features_not_count():
    assert metric.distance(descriptor([0., 1.]), descriptor([0., 1., 1.2])) == pytest.approx(.2)
    assert metric.distance(descriptor([0., 1.]), descriptor([0., 1., 8.])) == 7.


def test_fixed_and_both_directed_leaf_terms():
    result = metric.compare(descriptor([0., 2.], fixed=3), descriptor([0.]))
    assert result['fixed_linf'] == 3 and result['leaf_forward_linf'] == 2
    assert result['leaf_reverse_linf'] == 0 and result['distance'] == 3


def test_same_count_hausdorff_can_merge_more_than_bottleneck():
    a, b = descriptor([0., 0., 10.]), descriptor([0., 10., 10.])
    assert metric.distance(a, b) == 0. and frame.distance(a, b) == 10.


def test_symmetry_triangle_and_lower_than_same_count_bottleneck():
    rng = np.random.default_rng(729)
    for _ in range(30):
        a, b, c = [descriptor(rng.normal(size=n).tolist()) for n in (3, 3, 5)]
        assert metric.distance(a, b) == metric.distance(b, a)
        assert metric.distance(a, c) <= metric.distance(a, b) + metric.distance(b, c) + 1e-12
        assert metric.distance(a, b) <= frame.distance(a, b)


def test_inputs_and_return_are_detached():
    a, b = descriptor([0., 1.]), descriptor([0., 2.])
    before = deepcopy([a, b]); result = metric.compare(a, b); result['distance'] = 123
    assert [a, b] == before and metric.distance(a, b) == 1.


def test_empty_sets_are_explicitly_partial_not_infinite_novelty():
    assert metric.distance(descriptor([]), descriptor([], fixed=2)) == 2
    with pytest.raises(ValueError, match='unsupported'):metric.distance(descriptor([]), descriptor([0.]))


@pytest.mark.parametrize('field,value', [('schema','v1'), ('leaf_count', True), ('fixed',[0.]*103)])
def test_descriptor_validation_stays_v2(field, value):
    a = descriptor([0.]); a[field] = value
    with pytest.raises(ValueError):metric.distance(a, descriptor([0.]))


def test_nonfinite_features_rejected():
    with pytest.raises(ValueError):metric.distance(descriptor([float('nan')]), descriptor([0.]))
