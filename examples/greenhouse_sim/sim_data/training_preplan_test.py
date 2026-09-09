"""Offline proposal comparison is strict about structure and numeric drift."""
import pytest

from .training_preplan import same_tree


@pytest.mark.parametrize("left,right,expected", [
    ({"pose": [1, 2.0], "approved": False}, {"approved": False, "pose": [1.0, 2]}, True),
    ({"x": 0.1}, {"x": 0.1 + 1e-10}, True),
    ({"x": 0.1}, {"x": 0.1 + 1e-7}, False),
    ({"pose": [1]}, {"pose": [1], "extra": None}, False),
    ([1, 2], [2, 1], False),
    ([1, 2], [1], False),
    ([1, 2], (1, 2), False),
    (False, 0, False),
    (True, 1, False),
    ("1", 1, False),
    (None, None, True),
    (float("nan"), float("nan"), False),
])
def test_plan_comparison(left, right, expected):
    assert same_tree(left, right) is expected
