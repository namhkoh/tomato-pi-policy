"""Versioned maximum-transverse DONOR frame; no extraction or release approval.

Public API: describe(input_geometry), validate(descriptor),
distance(left_descriptor, right_descriptor), equivalent(left, right, tolerance).
Input is morphology.extract_output's input_geometry, not assets or capture paths.
No imports of / changes to the frozen extractor or morphology_context module.

Only reference_petiole and parent determine the frame. The nearest parent
segment supplies z; the original donor displacement with largest ||z cross v||
supplies y=unit(z cross v), x=unit(y cross z). No world-axis fallback. Current
petiole, leaves, names, seeds and ordering do not select the donor frame.

Numerics: u=eps64/2, gamma(n)=n*u/(1-n*u). Displacement, normalization and cross
product estimates propagate cancellation/conditioning; overlapping selection
intervals, unresolved axes or estimated linear-feature error >1e-9 cause holds.
These deliberately conservative operation-budget ESTIMATES are not a formal
certificate: they assume rounded float64 inputs, normal-range arithmetic, and
omit unknown upstream errors and a verified LAPACK/eigensolver error bound.
In particular the covariance square-root conditioning is not certified here.
The estimate is a one-input screening diagnostic, not a pairwise error bound;
actual paired nuisance checks at 1e-9 remain necessary. Extreme coordinate
offsets/scales, tied donor witnesses and nearest-parent joints may be held.
The unchanged 1e-9 is a numerical nuisance test, NOT a geometry novelty cutoff.

The 104 fixed / 15-per-leaf features otherwise retain the prior meanings. V2
axes change descriptor values: v1 descriptors cannot be compared as v2, and
v1 thresholds/calibration/rounded fingerprints cannot be promoted or reused.
Ill-conditioned/ambiguous domains return state='held' with descriptor=None.
Malformed data raises ValueError. No I/O, asset generation, learned models,
global groups, cap reset, calibration validation or biological-family claims.
"""
import hashlib
import json

import numpy as np

SCHEMA = "greenhouse.normalized_morphology_context.v2"
RESULT_SCHEMA = "greenhouse.morphology_frame_v2_result.v1"
FRAME_VERSION = "original_donor_maximum_transverse.v2"
NUMERICAL_TOLERANCE = 1e-9
_U = np.finfo(np.float64).eps / 2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _gamma(n):
    return n * _U / (1 - n * _U)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class _Hold(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details or {}


def _array(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    _require(np.isfinite(a).all() and (shape is None or a.shape == shape), "Finite correctly shaped geometry required")
    return a.copy()


def _chain(value):
    a = _array(value)
    _require(a.ndim == 2 and a.shape[1] == 4 and len(a) >= 2 and np.all(a[:, 3] > 0),
             "Positive-radius xyz/r chain required")
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1))]
    if np.any(np.diff(arc) <= 1e-12):
        raise _Hold("degenerate_chain")
    return a, arc


def _delta(a, b):
    # Operation budget includes initially rounded coordinates and subtraction.
    d = a - b
    error = float(_gamma(3) * np.linalg.norm(np.abs(a) + np.abs(b)))
    return d, error


def _unit(d, error, reason):
    norm = float(np.linalg.norm(d))
    error += _gamma(5) * norm
    if norm <= error:
        raise _Hold(reason, dict(norm=norm, error_estimate=error))
    return d / norm, float(2 * error / (norm - error) + _gamma(8))


def _frame(donor, parent):
    origin = donor[0, :3]
    candidates = []
    for index, (a, b) in enumerate(zip(parent, parent[1:])):
        d, ed = _delta(b[:3], a[:3])
        z, ez = _unit(d, ed, "unresolved_parent_axis")
        offset, eo = _delta(origin, a[:3])
        t = np.clip(np.dot(offset, d) / np.dot(d, d), 0., 1.)
        dist = float(np.linalg.norm(offset - t * d))
        # Closest-point sensitivity budget, not an interval-arithmetic proof.
        err = float(eo + ed + 4 * np.linalg.norm(offset) * ez
                    + _gamma(24) * (np.linalg.norm(offset) + np.linalg.norm(d)))
        candidates.append(dict(index=index, distance=dist, error=err, z=z, ez=ez))
    candidates.sort(key=lambda c: (c["distance"], c["index"]))
    chosen = candidates[0]
    if any(chosen["distance"] + chosen["error"] >= c["distance"] - c["error"] for c in candidates[1:]):
        raise _Hold("ambiguous_nearest_parent_segment",
                    dict(candidates=[{k: c[k] for k in ("index", "distance", "error")} for c in candidates]))
    z, ez = chosen["z"], chosen["ez"]
    witnesses = []
    for index, point in enumerate(donor[1:, :3], 1):
        v, ev = _delta(point, origin)
        cross = np.cross(z, v)
        norm = float(np.linalg.norm(cross))
        ec = float(ev + np.linalg.norm(v) * ez + _gamma(8) * np.linalg.norm(v))
        score_error = ec + _gamma(5) * norm
        witnesses.append(dict(index=index, norm=norm, error=score_error, cross=cross, ec=ec, v=v))
    witnesses.sort(key=lambda w: (-w["norm"], w["index"]))
    best = witnesses[0]
    if best["norm"] <= best["error"]:
        raise _Hold("degenerate_or_unresolved_donor_transverse",
                    dict(transverse_norm=best["norm"], error_estimate=best["error"]))
    if any(best["norm"] - best["error"] <= w["norm"] + w["error"] for w in witnesses[1:]):
        raise _Hold("ambiguous_maximum_donor_transverse",
                    dict(candidates=[{k: w[k] for k in ("index", "norm", "error")} for w in witnesses]))
    y, ey = _unit(best["cross"], best["ec"], "unresolved_donor_transverse")
    x, ex = _unit(np.cross(y, z), ey + ez + _gamma(8), "unresolved_cross_frame")
    basis = np.column_stack((x, y, z))
    # Frobenius estimate bounds the operator perturbation in the linear model.
    frame_error = float(np.linalg.norm([ex, ey, ez]))
    evidence = dict(version=FRAME_VERSION, basis_columns=basis.tolist(), origin=origin.tolist(),
        parent_segment_index=chosen["index"], donor_witness_index=best["index"],
        parent_distance_m=chosen["distance"], parent_distance_error_estimate_m=chosen["error"],
        transverse_m=best["norm"], transverse_error_estimate_m=best["error"],
        witness_sin_angle=float(best["norm"] / np.linalg.norm(best["v"])),
        selection_gap_m=float(best["norm"] - witnesses[1]["norm"]) if len(witnesses) > 1 else None,
        frame_error_estimate=float(frame_error))
    return basis, evidence


def _arc_error_estimate(chain):
    lengths = np.linalg.norm(np.diff(chain[:, :3], axis=0), axis=1)
    return float(sum(_delta(b[:3], a[:3])[1] + _gamma(5) * segment
                     for a, b, segment in zip(chain, chain[1:], lengths))
                 + _gamma(len(chain)) * sum(lengths))


def _linear_error_estimate(donor, parent, current, leaves, basis_error, length):
    origin = donor[0, :3]
    length_error = _arc_error_estimate(donor)
    points = [*donor[:, :3], *parent[:, :3], *current[:, :3]]
    points += [l[k] for l in leaves for k in ("attachment", "centroid")]
    extent = max(float(np.linalg.norm(p - origin) / length) for p in points)
    subtraction = max(_delta(p, origin)[1] / length for p in points)
    radius_extent = max(float(np.max(c[:, 3]) / length) for c in (donor, parent, current))
    amplification = max(1., extent, radius_extent)
    # Piecewise linear xyz has unit arclength speed. Radius speed can exceed 1.
    # Perturbation of both a knot and a sample location contributes arc error.
    sampling = max(2 * _arc_error_estimate(c) / length * max(1., float(np.max(
        np.abs(np.diff(c[:, 3])) / np.linalg.norm(np.diff(c[:, :3], axis=0), axis=1))))
        for c in (parent, current))
    estimate = subtraction + sampling + amplification * (length_error / length + basis_error + _gamma(32))
    return dict(unit_roundoff=float(_U), model="float64_operation_budget_conditioning_estimate",
        formal_certificate=False, scope="frame_and_linear_features_not_covariance_eigensolver",
        unknown_upstream_error_included=False, length_error_estimate_m=float(length_error),
        sampling_arc_error_estimate=float(sampling), maximum_normalized_radius=radius_extent,
        maximum_normalized_extent=extent, linear_feature_error_estimate=float(estimate),
        numerical_tolerance=NUMERICAL_TOLERANCE)


def _sample(chain, count):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(chain[:, :3], axis=0), axis=1))]
    s = np.linspace(0., arc[-1], count)
    return np.column_stack([np.interp(s, arc, chain[:, i]) for i in range(4)])


def describe(input_geometry):
    """Return a detached versioned descriptor or explicit numerical/domain hold.

    Input hash binds used geometry, NOT source provenance. Authenticate the
    extractor packet separately; this function does not verify donor ancestry.
    No adjustable frame-error limit, novelty threshold, or approval option.
    """
    _require(isinstance(input_geometry, dict), "input_geometry mapping required")
    _require({"reference_petiole", "parent", "current_petiole", "leaves"} <= input_geometry.keys(),
             "Complete extracted input_geometry required")
    result = dict(schema=RESULT_SCHEMA, frame_version=FRAME_VERSION, state="held",
        descriptor=None, holds=[], frame=None, conditioning=None, training_approved=False,
        qualified_geometry=False, source_cap_reset=False, calibration_validated=False,
        global_inventory_complete=False, numerical_nuisance_tolerance=NUMERICAL_TOLERANCE)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            donor, arc = _chain(input_geometry["reference_petiole"])
            parent, _ = _chain(input_geometry["parent"])
            current, _ = _chain(input_geometry["current_petiole"])
            _require(isinstance(input_geometry["leaves"], (list, tuple)), "Leaf list required")
            leaves = []
            for leaf in input_geometry["leaves"]:
                _require(isinstance(leaf, dict) and {"attachment", "centroid", "axis", "covariance"} <= leaf.keys(),
                         "Complete leaf mapping required")
                row = {k: _array(leaf[k], shape) for k, shape in
                       (("attachment", (3,)), ("centroid", (3,)), ("axis", (3,)), ("covariance", (3, 3)))}
                _require(np.linalg.norm(row["axis"]) > 1e-12, "Nondegenerate leaf axis required")
                cov = row["covariance"]
                # Retain the old covariance input-domain checks; no PSD/quality relaxation.
                _require(np.allclose(cov, cov.T, atol=1e-12, rtol=0)
                         and np.min(np.linalg.eigvalsh(cov)) >= -1e-12, "Symmetric PSD leaf covariance required")
                leaves.append(row)
            q = dict(reference_petiole=donor.tolist(), parent=parent.tolist(), current_petiole=current.tolist(),
                     leaves=[{k: v.tolist() for k, v in row.items()} for row in leaves])
            result["input_geometry_sha256"] = _digest(q)
            result["frame_reference_sha256"] = _digest({k: q[k] for k in ("reference_petiole", "parent")})
            basis, evidence = _frame(donor, parent)
            length = float(arc[-1])
            evidence["donor_arc_length_m"] = length
            evidence["normalized_transverse"] = evidence["transverse_m"] / length
            result["frame"] = evidence
            conditioning = _linear_error_estimate(donor, parent, current, leaves,
                                                  evidence["frame_error_estimate"], length)
            result["conditioning"] = conditioning
            if conditioning["linear_feature_error_estimate"] > NUMERICAL_TOLERANCE:
                raise _Hold("conditioning_estimate_exceeds_numerical_tolerance", conditioning)
            origin = donor[0, :3]
            def mapped(chain, count):
                local = chain.copy()
                local[:, :3] -= origin  # Interpolate displacements, not offset world coordinates.
                samples = _sample(local, count)
                return np.column_stack((samples[:, :3] @ basis / length, samples[:, 3] / length)).ravel()
            fixed = np.r_[mapped(current, 21), mapped(parent, 5)]
            vectors = []
            for leaf in leaves:
                local_cov = basis.T @ leaf["covariance"] @ basis / (length * length)
                eig, rot = np.linalg.eigh(local_cov)
                root = (rot * np.sqrt(np.maximum(eig, 0))) @ rot.T
                axis = leaf["axis"] / np.linalg.norm(leaf["axis"])
                vectors.append(np.r_[(leaf["attachment"] - origin) @ basis / length,
                    (leaf["centroid"] - origin) @ basis / length, axis @ basis,
                    root[np.triu_indices(3)]].tolist())
            vectors.sort(key=lambda row: tuple(np.round(row, 12)))
            value = dict(schema=SCHEMA, frame_version=FRAME_VERSION, fixed=fixed.tolist(),
                         leaves=vectors, leaf_count=len(vectors))
            validate(value)
            result.update(state="descriptor_available_not_calibrated", descriptor=value,
                          descriptor_sha256=_digest(value))
    except _Hold as exc:
        result["holds"].append(dict(reason=str(exc), details=exc.details))
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        result["holds"].append(dict(reason="unsupported_numerical_range_or_solver", details=str(exc)))
    return result


def validate(value):
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA
             and value.get("frame_version") == FRAME_VERSION, "V2 descriptor required; v1 calibration is incompatible")
    _require({"fixed", "leaves", "leaf_count"} <= value.keys(), "Complete V2 descriptor required")
    _array(value["fixed"], (104,))
    n = value["leaf_count"]
    _require(type(n) is int and n >= 0 and isinstance(value["leaves"], list) and len(value["leaves"]) == n,
             "Leaf count mismatch")
    if n:
        _array(value["leaves"], (n, 15))


def _costs(a, b):
    validate(a)
    validate(b)
    _require(a["leaf_count"] == b["leaf_count"], "Different leaf cardinality is outside equivalence domain, not novelty")
    fixed = float(np.max(np.abs(np.asarray(a["fixed"]) - b["fixed"])))
    if not a["leaf_count"]:
        return fixed, np.empty((0, 0))
    costs = np.max(np.abs(np.asarray(a["leaves"])[:, None, :] - np.asarray(b["leaves"])[None, :, :]), axis=2)
    return fixed, costs


def _matching(costs, tolerance):
    neighbours = [np.flatnonzero(row <= tolerance).tolist() for row in costs]
    matched = [-1] * len(neighbours)
    def augment(i, seen):
        for j in neighbours[i]:
            if j in seen:
                continue
            seen.add(j)
            if matched[j] < 0 or augment(matched[j], seen):
                matched[j] = i
                return True
        return False
    return all(augment(i, set()) for i in sorted(range(len(neighbours)), key=lambda i: len(neighbours[i])))


def equivalent(a, b, tolerance):
    """Pure V2 numerical comparison. True/False conveys NO novelty approval."""
    _require(type(tolerance) in (int, float) and np.isfinite(tolerance) and tolerance >= 0,
             "Finite nonnegative explicit tolerance required")
    fixed, costs = _costs(a, b)
    return fixed <= tolerance and _matching(costs, tolerance)


def distance(a, b):
    """Minimum V2 tolerance yielding a perfect unordered leaf matching."""
    fixed, costs = _costs(a, b)
    if not costs.size:
        return fixed
    values = np.unique(np.maximum(costs.ravel(), fixed))
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _matching(costs, values[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(values[lo])
