"""Advisory distance certificates for specified vertex-hull unions and Cubes.

Each supplied vertex part means its mathematical convex hull; a union is NOT
replaced by the hull of all vertices. Copying vertices from a native cooker does
NOT make that hull a native occupancy certificate: native fitted polygon planes
can define a solid extending outside the reported vertex hull. This discrepancy
was observed in the source-bound cooked-query assessment of 2026-09-11.
The source_id is a caller-supplied label, not verified provenance, enclosure,
actor identity, or physical proof. Neither native origin nor a separation
result establishes native clearance. Any use as an actor bound needs separate
enclosure/equivalence evidence; this module does not read native polygon planes.
Exact Cube dimensions/transforms likewise require independent actor binding.
No native contact, penetration, whole-robot clearance, or safe execution is
established here.

Matrices act on homogeneous COLUMN vectors. Full nonsingular affine transforms
preserve authored scale/reflection; nothing is fitted, shrunk, or normalized.
No visual triangles, USD/PhysX imports, file access, or screen wiring.

The solver's approximate simplex only proposes directions and feasible weights.
Certificates are computed separately: outward-rounded support/AABB lower bounds
and feasible convex-combination upper bounds. A pass requires a lower bound
STRICTLY greater than margin + 1e-8 m. Nonconvergence never means penetration.
Arithmetic assumes IEEE-754 binary64 basic operations and gradual underflow
(not flush-to-zero). Arithmetic-range failures return unresolved.
"""
from dataclasses import dataclass
from itertools import combinations, product
from numbers import Integral

import numpy as np

TOLERANCE_M = 1e-8
_EPS = np.finfo(np.float64).eps


def _array(value, name):
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex coordinates are not supported")
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be real finite numeric data") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _frozen(value):
    # bytes-backed arrays cannot have writeability re-enabled by the caller.
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def _matrix(value):
    matrix = _array(np.eye(4) if value is None else value, "world_from_local")
    if matrix.shape != (4, 4) or not np.array_equal(matrix[3], [0, 0, 0, 1]):
        raise ValueError("world_from_local must be an affine 4x4 column matrix")
    try:
        singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("invalid transform") from exc
    if (not np.isfinite(singular).all() or singular[0] == 0
            or singular[-1] / singular[0] <= 64 * _EPS):
        raise ValueError("singular or numerically singular transform")
    return _frozen(matrix)


@dataclass(frozen=True, init=False)
class ConvexUnion:
    """Immutable vertex-hull snapshot, not a native occupancy certificate.

    from_native_vertices preserves separate copied parts, but native fitted
    polygon-plane solids may extend outside those vertex hulls. source_id is
    only a caller-supplied label; construction verifies neither provenance nor
    enclosure/actor equivalence. exact_cube represents the supplied dimensions,
    not independently verified actor geometry. Degenerate/duplicate vertices
    are allowed: distance is always between the specified mathematical hulls.
    """
    parts: tuple
    world_from_local: np.ndarray
    representation: str
    source_id: str

    def __init__(self):
        raise TypeError("use from_native_vertices or exact_cube")

    @classmethod
    def _make(cls, parts, world_from_local, representation, source_id):
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("nonempty source_id/provenance reference required")
        copied = []
        for part in parts:
            vertices = _array(part, "convex vertices")
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
                raise ValueError("each convex part must have shape (N, 3), N >= 1")
            copied.append(_frozen(vertices))
        if not copied:
            raise ValueError("a nonempty convex union is required")
        result = object.__new__(cls)
        object.__setattr__(result, "parts", tuple(copied))
        object.__setattr__(result, "world_from_local", _matrix(world_from_local))
        object.__setattr__(result, "representation", representation)
        object.__setattr__(result, "source_id", source_id)
        return result

    @classmethod
    def from_native_vertices(cls, parts, *, source_id, world_from_local=None):
        """Copy vertex-hull parts; NOT a native occupancy certificate.

        Native fitted polygon planes can define occupancy outside the convex
        hull of their reported vertices. This constructor ignores those planes,
        so a solver separation result concerns ONLY the supplied vertex hulls,
        not the native plane-defined solids or their conservative enclosure.
        Native origin alone is insufficient. source_id is an unverified label,
        not physical proof; callers need independent actor/geometry binding and
        enclosure evidence before treating these hulls as actor bounds.
        """
        return cls._make(parts, world_from_local, "native_copied_convex_vertices", source_id)

    @classmethod
    def exact_cube(cls, half_extents_m, *, source_id, world_from_local=None):
        """Exact centered Cube, with authored scale included in the full matrix."""
        half = _array(half_extents_m, "half_extents_m")
        if half.shape != (3,) or np.any(half <= 0):
            raise ValueError("three strictly positive half extents required")
        vertices = np.asarray(list(product((-1., 1.), repeat=3))) * half
        return cls._make([vertices], world_from_local, "exact_cube", source_id)


@dataclass(frozen=True)
class ClearanceResult:
    """Bounds between the supplied mathematical vertex-hull parts, in meters.

    These are not bounds on native occupancy without separate enclosure proof.
    'unresolved' means the requested strict clearance was not certified, even if
    an upper bound demonstrates proximity. It NEVER asserts penetration.
    Bounds may be loose after an early exit. +inf is an uninformative upper
    bound on arithmetic failure; lower=0 is always valid.
    """
    status: str
    reason: str
    lower_m: float
    upper_m: float
    margin_m: float
    tolerance_m: float
    part_pairs_total: int
    part_pairs_evaluated: int
    part_pairs_aabb_pruned: int
    iterations: int
    closest_upper_pair: tuple | None
    native_certified: bool = False

    @property
    def separated(self):
        return self.status == "separated"


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _norm_bound(nonnegative, *, upper):
    """Outward Euclidean norm without squaring dimensional large/small values."""
    x = np.maximum(np.asarray(nonnegative, dtype=float), 0.)
    scale = float(np.max(x))
    if scale == 0:
        return 0.
    if not np.isfinite(scale):
        raise FloatingPointError("nonfinite norm")
    rounding = _up if upper else _down
    ratios = np.maximum(rounding(x / scale), 0.)
    squares = np.maximum(rounding(ratios * ratios), 0.)
    total = squares[0]
    for value in squares[1:]:
        total = max(0., float(rounding(total + value)))
    value = float(rounding(rounding(np.sqrt(total)) * scale))
    return max(0., value)


def _interval_norm_upper(lo, hi):
    return _norm_bound(np.maximum(np.abs(lo), np.abs(hi)), upper=True)


def _world_intervals(shape, origin):
    """Enclose transformed coordinates relative to a common translation.

    Subtract translations BEFORE adding the linear vertex transforms, avoiding
    needless loss of small clearances under large common world translations.
    Each elementary multiplication/addition is separately rounded outwards.
    """
    matrix = shape.world_from_local
    delta = matrix[:3, 3] - origin
    equal = matrix[:3, 3] == origin
    delta_lo = np.where(equal, 0., _down(delta))
    delta_hi = np.where(equal, 0., _up(delta))
    result = []
    for vertices in shape.parts:
        lo = np.zeros_like(vertices)
        hi = np.zeros_like(vertices)
        for k in range(3):
            term = vertices[:, k, None] * matrix[None, :3, k]
            lo = _down(lo + _down(term))
            hi = _up(hi + _up(term))
        lo = _down(lo + delta_lo)
        hi = _up(hi + delta_hi)
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise FloatingPointError("transformed coordinate range")
        nominal = lo * .5 + hi * .5
        result.append((lo, hi, nominal))
    return result


def _aabb_lower(a, b):
    alo, ahi, _ = a
    blo, bhi, _ = b
    gaps = np.maximum(0., np.maximum(
        _down(alo.min(axis=0) - bhi.max(axis=0)),
        _down(blo.min(axis=0) - ahi.max(axis=0))))
    return _norm_bound(gaps, upper=False)


def _dot_intervals(lo, hi, direction):
    lower = np.zeros(len(lo))
    upper = np.zeros(len(lo))
    for k in range(3):
        p = lo[:, k] * direction[k]
        q = hi[:, k] * direction[k]
        lower = _down(lower + _down(np.minimum(p, q)))
        upper = _up(upper + _up(np.maximum(p, q)))
    return lower, upper


def _support_lower(a, b, direction):
    # Any direction gives a lower bound. The approximate support vertex chosen
    # for the next simplex is NOT used as a certificate.
    # For every x in A, y in B:
    # ||x-y|| >= dot(direction, x-y)/||direction||
    #          >= (min_A dot(direction,x)-max_B dot(direction,y))/||direction||.
    # Extremal projections over a convex hull occur at its copied vertices.
    norm_upper = _norm_bound(np.abs(direction), upper=True)
    if norm_upper == 0:
        return 0.
    a_lower, _ = _dot_intervals(a[0], a[1], direction)
    _, b_upper = _dot_intervals(b[0], b[1], direction)
    gap = float(_down(np.min(a_lower) - np.max(b_upper)))
    return max(0., float(_down(max(0., gap) / norm_upper)))


def _witness_upper(a, b, ids, weights):
    """Enclose a feasible point in A-B with exactly normalized positive weights.

    Float weights are interpreted as exact nonnegative coefficients, divided
    by their real sum. Interval division avoids assuming float sum(weights)=1.
    """
    lo = np.zeros(3)
    hi = np.zeros(3)
    sum_lo = 0.
    sum_hi = 0.
    for (i, j), weight in zip(ids, weights):
        if weight <= 0:
            continue
        diff_lo = _down(a[0][i] - b[1][j])
        diff_hi = _up(a[1][i] - b[0][j])
        lo = _down(lo + _down(diff_lo * weight))
        hi = _up(hi + _up(diff_hi * weight))
        sum_lo = float(_down(sum_lo + weight))
        sum_hi = float(_up(sum_hi + weight))
    if sum_lo <= 0 or not np.isfinite(sum_hi):
        raise FloatingPointError("invalid feasible weights")
    lower = _down(np.minimum(lo / sum_lo, lo / sum_hi))
    upper = _up(np.maximum(hi / sum_lo, hi / sum_hi))
    return _interval_norm_upper(lower, upper)


def _closest_simplex(points):
    """Approximate projection only; all returned weights are actually feasible."""
    best = None
    for count in range(1, min(4, len(points)) + 1):
        for ids in combinations(range(len(points)), count):
            selected = points[list(ids)]
            if count == 1:
                weights = np.ones(1)
            else:
                edges = (selected[1:] - selected[0]).T
                try:
                    tail = np.linalg.lstsq(edges, -selected[0], rcond=64*_EPS)[0]
                except np.linalg.LinAlgError:
                    continue
                weights = np.r_[1. - tail.sum(), tail]
                if not np.isfinite(weights).all() or np.min(weights) < -64*_EPS:
                    continue
                weights = np.maximum(weights, 0.)
                weights /= weights.sum()
            vector = weights @ selected
            score = float(vector @ vector)
            if best is None or score < best[0]:
                best = score, ids, weights
    # Singleton subsets always provide a feasible answer, including degeneracy.
    _, ids, weights = best
    active = weights > 0
    weights = weights[active]
    weights /= weights.sum()
    return np.asarray(ids)[active], weights


@dataclass(frozen=True)
class _PairResult:
    lower: float
    upper: float
    iterations: int
    reason: str


def _convex_pair(a, b, lower, limit, max_iterations):
    scale = max(float(np.max(np.abs(a[2]))), float(np.max(np.abs(b[2]))))
    scale = scale if scale > 0 else 1.
    an, bn = a[2] / scale, b[2] / scale
    ids = [(0, 0)]
    upper = _witness_upper(a, b, ids, [1.])
    if upper <= limit:
        return _PairResult(lower, upper, 0, "feasible_witness_within_threshold")
    for iteration in range(1, max_iterations + 1):
        points = np.asarray([an[i] - bn[j] for i, j in ids])
        selected, weights = _closest_simplex(points)
        ids = [ids[i] for i in selected]
        points = points[selected]
        upper = min(upper, _witness_upper(a, b, ids, weights))
        vector = weights @ points
        length = float(np.linalg.norm(vector))
        if length > 0:
            direction = vector / length
            lower = max(lower, _support_lower(a, b, direction))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise FloatingPointError("inconsistent distance bounds")
        if lower > limit:
            return _PairResult(lower, upper, iteration, "support_separation")
        if upper <= limit:
            return _PairResult(lower, upper, iteration, "feasible_witness_within_threshold")
        if upper - lower <= TOLERANCE_M or length == 0:
            return _PairResult(lower, upper, iteration, "distance_bounds_no_strict_clearance")
        following = (int(np.argmin(an @ direction)), int(np.argmax(bn @ direction)))
        if following in ids:
            return _PairResult(lower, upper, iteration, "support_stagnation")
        ids.append(following)
    return _PairResult(lower, upper, max_iterations, "iteration_limit")


def _budget(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def convex_union_clearance(a, b, *, margin_m, max_iterations=80,
                           max_pair_evaluations=None):
    """Bound supplied vertex-hull distance, not native occupancy or penetration.

    Work is bounded by the finite part-pair count and max_iterations per pair.
    max_pair_evaluations can impose a smaller budget. Zero budgets still permit
    an AABB separation certificate. Safe early returns are threshold-oriented:
    all pairs must certify separation; one unresolved pair blocks the union.

    The fixed 1e-8 m tolerance is additional to, never a relaxation of, margin.
    AABB-pruned pairs remain represented in the union's minimum lower bound.
    """
    if not isinstance(a, ConvexUnion) or not isinstance(b, ConvexUnion):
        raise TypeError("ConvexUnion inputs required")
    for shape in (a, b):
        if getattr(shape, "representation", None) not in {
                "native_copied_convex_vertices", "exact_cube"}:
            raise ValueError("unsupported geometry representation")
    margin = _array(margin_m, "margin_m")
    if margin.shape != () or float(margin) < 0:
        raise ValueError("margin_m must be a nonnegative scalar")
    margin = float(margin)
    iterations_limit = _budget(max_iterations, "max_iterations")
    total = len(a.parts) * len(b.parts)
    pair_budget = total if max_pair_evaluations is None else _budget(
        max_pair_evaluations, "max_pair_evaluations")
    evaluated = iterations = pruned = 0
    closest = None

    def result(lower, upper, reason):
        # Rounded-up threshold is at least the mathematical margin + tolerance.
        passed = lower > limit
        return ClearanceResult(
            "separated" if passed else "unresolved", reason, float(lower),
            float(upper), margin, TOLERANCE_M, total, evaluated, pruned,
            iterations, closest)

    # Underflow is deliberately allowed; each nextafter enclosure includes it.
    # Overflow/invalid/divide failures cannot yield an optimistic certificate.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            limit = float(_up(margin + TOLERANCE_M))
            origin = a.world_from_local[:3, 3]
            aparts = _world_intervals(a, origin)
            bparts = _world_intervals(b, origin)
            bounds = []
            upper = np.inf
            for i, aa in enumerate(aparts):
                for j, bb in enumerate(bparts):
                    lower = _aabb_lower(aa, bb)
                    bounds.append([lower, i, j])
                    candidate = _witness_upper(aa, bb, [(0, 0)], [1.])
                    if candidate < upper:
                        upper, closest = candidate, (i, j)
            pruned = sum(lower > limit for lower, _, _ in bounds)
            lower = min(row[0] for row in bounds)
            if lower > limit:
                return result(lower, upper, "aabb_separation")
            if upper <= limit:
                return result(lower, upper, "feasible_witness_within_threshold")
            # Stable deterministic order, with all untouched bounds retained.
            for row in sorted(bounds, key=lambda item: tuple(item)):
                if row[0] > limit:
                    continue
                if evaluated >= pair_budget:
                    return result(min(x[0] for x in bounds), upper, "pair_budget")
                _, i, j = row
                pair = _convex_pair(aparts[i], bparts[j], row[0], limit, iterations_limit)
                evaluated += 1
                iterations += pair.iterations
                row[0] = max(row[0], pair.lower)
                if pair.upper < upper:
                    upper, closest = pair.upper, (i, j)
                lower = min(x[0] for x in bounds)
                if lower > upper:
                    raise FloatingPointError("inconsistent union bounds")
                if row[0] <= limit:
                    return result(lower, upper, pair.reason)
            return result(min(x[0] for x in bounds), upper, "convex_union_separation")
        except (FloatingPointError, OverflowError):
            # No assertion of penetration, and no partial positive result.
            limit = np.inf
            return result(0., np.inf, "arithmetic_range_or_inconsistent_bounds")
