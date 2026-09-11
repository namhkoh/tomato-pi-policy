"""Pure synthetic vertex-hull/Cube tests: no assets, native runtime or search."""
from decimal import Decimal, localcontext
import itertools
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from sim_physics import convex_clearance as cc


def transform(linear=None, translation=(0., 0., 0.)):
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) if linear is None else linear
    matrix[:3, 3] = translation
    return matrix


def vertices(half=(.5, .5, .5), centre=(0., 0., 0.)):
    return np.array(list(itertools.product((-1., 1.), repeat=3))) * half + centre


def cube(half=(.5, .5, .5), matrix=None):
    return cc.ConvexUnion.exact_cube(
        half, world_from_local=matrix, source_id="synthetic exact Cube")


def native(parts, matrix=None):
    return cc.ConvexUnion.from_native_vertices(
        parts, world_from_local=matrix, source_id="synthetic copied convex test fixture")


def check(a, b, margin=0., **kwargs):
    return cc.convex_union_clearance(a, b, margin_m=margin, **kwargs)


def assert_brackets(result, distance, slack=2e-13):
    assert 0 <= result.lower_m <= distance + slack
    assert result.upper_m >= distance - slack
    assert result.lower_m <= result.upper_m
    assert not result.native_certified
    if result.separated:
        assert result.lower_m > result.margin_m + cc.TOLERANCE_M


def rod_pair(scale=1., common=None):
    # Perpendicular centre lines, separated along their common normal. Each
    # has width .04 along that normal, so the exact gap is .2 - .04 = .16.
    u = np.array([1., 1., 0.]) / np.sqrt(2)
    v = np.array([-1., 1., 2.]) / np.sqrt(6)
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    ra = np.column_stack((u, n, np.cross(u, n)))
    rb = np.column_stack((v, n, np.cross(v, n)))
    ta = transform(ra, (0, 0, 0))
    tb = transform(rb, n * (.2 * scale))
    if common is not None:
        ta, tb = common @ ta, common @ tb
    half = np.array([2., .02, .02]) * scale
    return cube(half, ta), cube(half, tb)


def test_import_isolated_without_native_runtime():
    script = """
import importlib.abc
import sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'omni', 'isaacsim', 'carb', 'pxr'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockNative())
sys.path[:0] = sys.argv[1:]
import sim_physics.convex_clearance
"""
    run = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script,
         str(Path(cc.__file__).resolve().parents[1]),
         str(Path(np.__file__).resolve().parents[1])],
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("gap,margin,passed", [
    (.1, 0., True), (.1, .099, True), (.1, .1, False),
    (0., 0., False), (-.2, 0., False),
    (1e-8, 0., False), (1.5e-8, 0., True),
])
def test_boxes_threshold_is_strict(gap, margin, passed):
    result = check(cube(), cube(matrix=transform(translation=(1 + gap, 0, 0))), margin)
    assert result.separated == passed
    assert_brackets(result, max(0., gap))


def test_exact_threshold_binary_gap_never_passes():
    point = native([np.zeros((1, 3))])
    other = native([np.array([[cc.TOLERANCE_M, 0, 0]])])
    result = check(point, other)
    assert not result.separated
    assert_brackets(result, cc.TOLERANCE_M, slack=0.)


def test_large_margin_rounding_cannot_weaken_strict_threshold():
    point = native([[[0., 0., 0.]]])
    other = native([[[1e10, 0., 0.]]])
    assert not check(point, other, 1e10).separated


def test_overlapping_boxes_are_unresolved_not_penetration():
    result = check(cube(), cube(matrix=transform(translation=(.2, .1, .1))))
    assert result.status == "unresolved"
    assert result.upper_m < cc.TOLERANCE_M
    assert "penetr" not in result.reason
    assert_brackets(result, 0.)


def test_native_union_preserves_nonconvex_hole():
    # A square frame around an empty 2 x 2 hole, not its filled overall hull.
    frame = native([
        vertices((.5, 2., .5), (-1.5, 0, 0)),
        vertices((.5, 2., .5), (1.5, 0, 0)),
        vertices((1., .5, .5), (0, -1.5, 0)),
        vertices((1., .5, .5), (0, 1.5, 0)),
    ])
    inside = cube((.25, .25, .25))
    result = check(frame, inside, .7)
    assert result.separated
    assert result.part_pairs_total == 4
    assert result.part_pairs_aabb_pruned == 4
    assert result.part_pairs_evaluated == 0
    assert_brackets(result, .75)
    filled_hull = native([np.concatenate(frame.parts)])
    assert not check(filled_hull, inside, .7).separated


def test_crossed_rods_have_overlapping_aabbs_but_support_separates():
    a, b = rod_pair()
    result = check(a, b, .159)
    assert result.separated
    assert result.part_pairs_aabb_pruned == 0
    assert 1 <= result.part_pairs_evaluated == 1
    assert 1 <= result.iterations <= 80
    assert_brackets(result, .16)
    assert result.lower_m > .159


def test_rod_bounds_resolve_distance_when_threshold_requires_it():
    a, b = rod_pair()
    result = check(a, b, .16)
    assert not result.separated
    assert_brackets(result, .16)
    assert result.upper_m - result.lower_m < 1e-8


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6, 1e100])
def test_scale_does_not_change_certificate_meaning(scale):
    a, b = rod_pair(scale)
    result = check(a, b, .14 * scale)
    assert result.separated
    assert_brackets(result, .16 * scale, slack=1e-13 * scale)
    assert result.iterations <= 80


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("translation", [(0., 0., 0.), (13., -24., 53.)])
def test_rotation_translation_reflection_and_swapped_pairs(reflection, translation):
    angle = .63
    linear = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    if reflection:
        linear = linear @ np.diag([-1., 1., 1.])
    a, b = rod_pair(common=transform(linear, translation))
    ab, ba = check(a, b, .159), check(b, a, .159)
    assert ab.separated and ba.separated
    assert_brackets(ab, .16)
    assert_brackets(ba, .16)
    # Bounds may differ by direction/rounding, but both enclose the same gap.
    assert max(ab.lower_m, ba.lower_m) <= min(ab.upper_m, ba.upper_m)


def test_column_transform_matches_explicit_vertices():
    linear = np.array([[0, -2., 0], [3., 0, 0], [0, 0, -4.]])
    ta = transform(linear, (7, 8, 9))
    tb = transform(linear, (7, 12, 9))
    a, b = cube(matrix=ta), cube(matrix=tb)
    baked_a = native([a.parts[0] @ ta[:3, :3].T + ta[:3, 3]])
    baked_b = native([b.parts[0] @ tb[:3, :3].T + tb[:3, 3]])
    transformed, baked = check(a, b, .9), check(baked_a, baked_b, .9)
    assert transformed.separated and baked.separated
    assert_brackets(transformed, 1.)
    assert_brackets(baked, 1.)


def test_large_common_translation_does_not_destroy_small_local_gap():
    matrix = transform(translation=(1e100, -1e100, 1e100))
    a = native([vertices()], matrix)
    b = native([vertices(centre=(1.01, 0, 0))], matrix)
    result = check(a, b, .009)
    assert result.separated
    assert_brackets(result, .01)


@pytest.mark.parametrize("points", [
    [[0., 0., 0.]],
    [[0., 0., 0.]] * 6,
    [[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
    [[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]],
    [[-1., -1., 0.], [1., -1., 0.], [1., 1., 1e-16], [-1., 1., 0.]],
])
def test_degenerate_hulls_are_conservative(points):
    a = native([points])
    b = native([np.asarray(points) + [0., 0., .1]])
    result = check(a, b, .09)
    assert result.separated
    assert_brackets(result, .1)
    overlap = check(a, a)
    assert not overlap.separated
    assert overlap.upper_m < 1e-8


def test_degenerate_support_simplex_does_not_crash_with_aabb_overlap():
    # Parallel diagonal segments with overlapping coordinate ranges.
    a = native([[[-1., -1., 0.], [1., 1., 0.], [1., 1., 0.]]])
    b = native([[[-1., -.8, 0.], [1., 1.2, 0.]]])
    result = check(a, b, .13)
    assert result.part_pairs_aabb_pruned == 0
    assert result.separated
    assert_brackets(result, .2 / np.sqrt(2))


@pytest.mark.parametrize("iterations", [0, 1])
def test_iteration_exhaustion_fails_closed_deterministically(iterations):
    a, b = rod_pair()
    results = [check(a, b, .159, max_iterations=iterations) for _ in range(3)]
    assert results[0] == results[1] == results[2]
    assert not results[0].separated
    assert results[0].reason == "iteration_limit"
    assert results[0].iterations == iterations
    assert_brackets(results[0], .16)


def test_zero_pair_budget_is_unresolved_for_non_aabb_certificate():
    a, b = rod_pair()
    result = check(a, b, .159, max_pair_evaluations=0)
    assert not result.separated
    assert result.reason == "pair_budget"
    assert result.part_pairs_evaluated == result.iterations == 0


def test_aabb_can_safely_pass_without_any_solver_budget():
    a, b = cube(), cube(matrix=transform(translation=(3., 0, 0)))
    with patch.object(cc, "_convex_pair", side_effect=AssertionError("unneeded solver")):
        result = check(a, b, 1., max_iterations=0, max_pair_evaluations=0)
    assert result.separated and result.reason == "aabb_separation"
    assert result.part_pairs_evaluated == result.iterations == 0
    assert_brackets(result, 2.)


def test_pruned_pairs_and_unprocessed_pairs_remain_in_union_bounds():
    a, b = rod_pair()
    far = b.parts[0] + [30, 0, 0]
    b_union = native([b.parts[0], far], b.world_from_local)
    result = check(a, b_union, .159)
    assert result.separated
    assert result.part_pairs_total == 2
    assert result.part_pairs_aabb_pruned == 1
    assert result.part_pairs_evaluated == 1
    assert_brackets(result, .16)


def test_far_first_part_cannot_hide_close_later_part():
    a = native([vertices(centre=(10, 0, 0)), vertices()])
    result = check(a, cube(), .01)
    assert not result.separated
    assert result.upper_m < 1e-8
    assert result.part_pairs_total == 2


def test_union_stops_on_feasible_witness_without_claiming_exact_distance():
    a = native([vertices(), vertices(centre=(10, 0, 0))])
    with patch.object(cc, "_convex_pair", side_effect=AssertionError("unneeded solver")):
        result = check(a, cube(), .01)
    assert result.reason == "feasible_witness_within_threshold"
    assert not result.separated
    assert result.part_pairs_evaluated == 0


def test_copied_geometry_and_transform_are_immutable():
    raw = vertices()
    matrix = np.eye(4)
    shape = native([raw], matrix)
    raw[:] = 100.
    matrix[:] = 100.
    np.testing.assert_array_equal(shape.parts[0], vertices())
    np.testing.assert_array_equal(shape.world_from_local, np.eye(4))
    with pytest.raises(ValueError):
        shape.parts[0].setflags(write=True)
    with pytest.raises(ValueError):
        shape.world_from_local.setflags(write=True)


@pytest.mark.parametrize("bad", [
    [], [np.empty((0, 3))], [np.zeros((3, 2))], [np.zeros(3)],
    [[[np.nan, 0, 0]]], [[[np.inf, 0, 0]]], [[[complex(1, 2), 0, 0]]],
])
def test_invalid_vertices_rejected(bad):
    with pytest.raises(ValueError):
        native(bad)


@pytest.mark.parametrize("bad", [
    [0, .5, .5], [-1, 1, 1], [1, 1], [np.nan, 1, 1], [np.inf, 1, 1],
])
def test_invalid_cube_rejected(bad):
    with pytest.raises(ValueError):
        cube(bad)


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4)), np.eye(3), np.diag([1., 1., 0., 1.]),
    np.diag([1., 1., 1e-16, 1.]), np.full((4, 4), np.nan),
    np.full((4, 4), np.inf),
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1e-15, 0, 0, 1]]),
])
def test_invalid_singular_or_projective_transforms_rejected(bad):
    with pytest.raises(ValueError):
        cube(matrix=bad)


@pytest.mark.parametrize("margin", [-1., np.nan, np.inf, [1.], 1+2j])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        check(cube(), cube(), margin)


@pytest.mark.parametrize("name,value", [
    ("max_iterations", -1), ("max_iterations", 1.5), ("max_iterations", True),
    ("max_pair_evaluations", -1), ("max_pair_evaluations", 2.5),
    ("max_pair_evaluations", False),
])
def test_invalid_budgets_rejected(name, value):
    with pytest.raises(ValueError):
        check(cube(), cube(), **{name: value})


def test_source_reference_and_supported_representation_required():
    with pytest.raises(TypeError, match="use from_native_vertices"):
        cc.ConvexUnion()
    with pytest.raises(ValueError):
        cc.ConvexUnion.from_native_vertices([vertices()], source_id="")
    with pytest.raises(TypeError):
        check(vertices(), cube())
    shape = cc.ConvexUnion._make([vertices()], None, "visual_triangles", "bad")
    with pytest.raises(ValueError, match="unsupported"):
        check(shape, cube())


def test_native_vertex_hull_result_is_not_an_occupancy_certificate():
    assert "NOT a native occupancy certificate" in cc.ConvexUnion.from_native_vertices.__doc__
    # Synthetic counterexample: the supplied vertex hull is [-.5,.5]^3,
    # while fitted halfspaces could describe [-.6,.6]^3. A point at x=.55 is
    # outside the former and inside the latter. This is not native-run evidence.
    point = np.array([.55, 0., 0.])
    fitted_planes = np.column_stack((np.vstack((np.eye(3), -np.eye(3))),
                                     np.full(6, -.6)))
    assert np.all(fitted_planes[:, :3] @ point + fitted_planes[:, 3] < 0)
    target = native([[point]])
    for label in ("unverified", "caller-claims-native-sha256"):
        shape = cc.ConvexUnion.from_native_vertices([vertices()], source_id=label)
        result = check(shape, target)
        assert result.separated  # Only the specified mathematical vertex hulls.
        assert not result.native_certified
        assert_brackets(result, .05)


def test_arithmetic_overflow_fails_closed():
    huge = native([vertices((1e308, 1e308, 1e308))], transform(np.eye(3)*10))
    result = check(huge, cube())
    assert result.status == "unresolved"
    assert result.reason == "arithmetic_range_or_inconsistent_bounds"
    assert result.lower_m == 0 and result.upper_m == np.inf


def test_subnormal_points_never_produce_spurious_clearance():
    tiny = np.nextafter(0., 1.)
    result = check(native([[[tiny, 0, 0]]]), native([[[0, 0, 0]]]))
    assert not result.separated
    assert_brackets(result, tiny, slack=0.)


def test_corrupt_solver_bound_cannot_produce_pass():
    a, b = rod_pair()
    # Defense against accidental future loss of bound consistency.
    with patch.object(cc, "_support_lower", return_value=1e6):
        result = check(a, b, .159)
    assert not result.separated
    assert result.reason == "arithmetic_range_or_inconsistent_bounds"


def test_deterministic_axis_aligned_boxes_against_analytic_oracle():
    rng = np.random.default_rng(117)
    for _ in range(40):
        half_a, half_b = rng.uniform(.001, 2., (2, 3))
        centre_a, centre_b = rng.uniform(-4., 4., (2, 3))
        gap = np.maximum(np.abs(centre_a-centre_b) - half_a-half_b, 0.)
        exact = float(np.linalg.norm(gap))
        a = cube(half_a, transform(translation=centre_a))
        b = cube(half_b, transform(translation=centre_b))
        for aa, bb in ((a, b), (b, a)):
            result = check(aa, bb, max(0., exact - .001))
            assert_brackets(result, exact, slack=5e-13)
            assert result.separated == (exact > 1e-8)


def test_fixed_absolute_tolerance_is_not_rescaled_with_geometry():
    a, b = rod_pair(1e-6)
    result = check(a, b, .15e-6)
    # Gap-minus-margin equals the fixed 1e-8 tolerance: strict pass forbidden.
    assert not result.separated
    assert_brackets(result, .16e-6)


def _decimal_point_distance(pa, ta, pb, tb):
    # Independent high-precision oracle for the REAL values of binary64 inputs,
    # not a repeat of this module's rounded transform/norm implementation.
    def world(point, matrix):
        return [sum((Decimal.from_float(float(matrix[i, j])) *
                     Decimal.from_float(float(point[j])) for j in range(3)),
                    Decimal.from_float(float(matrix[i, 3]))) for i in range(3)]
    with localcontext() as ctx:
        ctx.prec = 400
        aa, bb = world(pa, ta), world(pb, tb)
        return sum((x-y)**2 for x, y in zip(aa, bb)).sqrt()


@pytest.mark.parametrize("scale", [1e-200, 1e-6, 1., 1e100])
def test_outward_bounds_against_decimal_transformed_point_oracle(scale):
    rng = np.random.default_rng(311)
    for _ in range(10):
        pa, pb = rng.uniform(-2., 2., (2, 3)) * scale
        linear_a = np.eye(3) + rng.uniform(-.2, .2, (3, 3))
        linear_b = np.eye(3) + rng.uniform(-.2, .2, (3, 3))
        linear_b[:, 0] *= -1
        ta = transform(linear_a, rng.uniform(-2., 2., 3) * scale)
        tb = transform(linear_b, rng.uniform(-2., 2., 3) * scale)
        exact = _decimal_point_distance(pa, ta, pb, tb)
        for a, b in ((native([[pa]], ta), native([[pb]], tb)),
                     (native([[pb]], tb), native([[pa]], ta))):
            result = check(a, b)
            assert Decimal.from_float(result.lower_m) <= exact
            assert exact <= Decimal.from_float(result.upper_m)
            assert result.separated == (exact > Decimal("1e-8"))
