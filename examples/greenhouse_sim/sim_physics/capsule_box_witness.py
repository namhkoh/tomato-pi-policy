"""One fresh finite capsule/AABB witness; geometry only, not native contact.

Inputs and outputs are in the SAME origin-centred box-local frame, in metres.
The capsule is a closed medial segment swept by a sphere of radius_m. Neither
shape is enlarged by contact offsets, margins, guessed normals or PCM caches.

The squared segment/box distance is convex and piecewise quadratic, split where
the segment crosses box slab planes. Each interval's endpoints and stationary
point are tested. A truly constant minimum interval uses its midpoint; this
does NOT create a two-point manifold or turn a small tilt into parallel contact.
This is the witness form of robot_kinematics._segment_aabb_distance, with no
finite difference/search approximation and no dependency on robot/native APIs.

For d > 0: n=(axis-box)/d, capsule_point=axis-radius*n, gap=d-radius.
These are actual capsule/box surface witnesses (including caps), not projected
PCM representative points. gap<0 permits surface overlap while the axis stays
outside the box. If the axis touches/intersects the box, unsigned distance
cannot provide penetration direction/depth: fail closed, never invent a normal.
The numerical exclusion is max(1e-12 m, 64*float64_epsilon*input_scale); it only
REJECTS ill-conditioned geometry and never modifies a returned gap or shape.
Radius/box extents unresolved at 64*epsilon*input_scale also fail closed.

Zero-length segments are spheres, represented at t=0.5. Endpoint cap labels
include the cap/cylinder rim. At nonunique minima, the chosen witness is one
valid subgradient, not a claim that distance is differentiable under rotation.
Caller owns measured frame provenance, pair discovery, material/force law and
all force/collision gates. No force, friction, contact completeness, native
parity, calibration or actuation authority is supplied here.
"""
import math
from numbers import Real

import numpy as np


MODEL = 'fresh_finite_capsule_box_single_witness_v1'
MIN_AXIS_BOX_DISTANCE_M = 1e-12
_EPS = np.finfo(np.float64).eps


def _vector(value, name):
    try:
        raw = np.asarray(value)
        if (raw.shape != (3,) or raw.dtype.kind not in 'fiu'
                or any(isinstance(v, (bool, np.bool_))
                       for v in np.asarray(value, dtype=object).flat)):
            raise ValueError('Three real numeric components required: ' + name)
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, OverflowError) as exc:
        raise ValueError('Finite numeric vector required: ' + name) from exc
    if not np.isfinite(result).all():
        raise ValueError('Finite numeric vector required: ' + name)
    return result


def _radius(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('Positive finite radius_m required')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Positive finite radius_m required') from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Positive finite radius_m required')
    return result


def _minimum(a, b, half):
    """Normalized coordinates; return minimizing t and its exact flat interval."""
    delta = b-a
    breaks = [0., 1.]
    for axis in range(3):
        if delta[axis] == 0:
            continue
        low, high = sorted((a[axis], b[axis]))
        for boundary in (-half[axis], half[axis]):
            # Only divide for an actual crossing; avoids huge ratios for a
            # nearly parallel segment. No angular/length cutoff is applied.
            if low < boundary < high:
                breaks.append(float((boundary-a[axis])/delta[axis]))
    breaks = sorted(set(breaks))
    candidates = set(breaks)
    flats = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        middle = a + ((lo+hi)*.5)*delta
        active = (middle < -half) | (middle > half)
        direction = np.where(active, delta, 0.)
        magnitude = float(np.max(abs(direction)))
        if magnitude == 0:
            # Derivative is identically zero on this interval. Convexity
            # makes any such interval a GLOBAL minimum, not a local plateau.
            flats.append((lo, hi))
            continue
        direction = direction/magnitude
        p0, p1 = a+lo*delta, a+hi*delta
        derivative0 = float(direction @ (p0-np.clip(p0, -half, half)))
        derivative1 = float(direction @ (p1-np.clip(p1, -half, half)))
        if derivative0 < 0 < derivative1:
            fraction = -derivative0/(derivative1-derivative0)
            candidates.add(float(lo+(hi-lo)*fraction))
    if flats:
        lo = min(v[0] for v in flats); hi = max(v[1] for v in flats)
        return (lo+hi)*.5, (lo, hi)

    def score(t):
        point = a+t*delta
        difference = point-np.clip(point, -half, half)
        return math.hypot(*difference), abs(t-.5), t
    t = min(candidates, key=score)
    return t, (t, t)


def closest_capsule_box(start, end, half_extents, radius):
    """Return copied JSON-ready geometry for ONE pair, or raise ValueError.

    normal acts on the capsule; point identities and gap use
    the same orientation regardless of any external native header ordering.
    minimizer_interval is nontrivial only for a genuine constant-distance
    interval. feature['box'] labels the surface containing box_point.
    No raw contact rows, material parameters or world-frame guesses are read.
    """
    start = _vector(start, 'start'); end = _vector(end, 'end')
    half = _vector(half_extents, 'half_extents'); radius = _radius(radius)
    if np.any(half <= 0):
        raise ValueError('Strictly positive source box half extents required')
    scale = max(float(np.max(abs(start))), float(np.max(abs(end))),
                float(np.max(half)), radius)
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            a, b, h = start/scale, end/scale, half/scale
            if radius/scale <= 64*_EPS or np.any(h <= 64*_EPS):
                raise ValueError('Radius or box extents numerically unresolved at input scale')
            t, interval = _minimum(a, b, h)
            # Recompute using ORIGINAL extents: do not return a resized box.
            axis = (1.-t)*start+t*end
            box = np.clip(axis, -half, half)
            difference = axis-box
            distance = math.hypot(*difference)
            floor = max(MIN_AXIS_BOX_DISTANCE_M, 64*_EPS*scale)
            if not math.isfinite(distance) or distance <= floor:
                raise ValueError('Axis/box distance zero or numerically unresolved; penetration normal unavailable')
            normal = difference/distance
            capsule = axis-radius*normal
            gap = distance-radius
            if not np.isfinite(capsule).all() or not math.isfinite(gap):
                raise ValueError('Witness outside finite numeric range')
            # Convex squared-distance optimality, in normalized coordinates.
            # Use the unnormalized gradient: dividing by a small distance
            # needlessly amplifies witness-coordinate roundoff.
            derivative = float((difference/scale) @ (b-a))
            roundoff = 128*_EPS*max(1., math.hypot(*(b-a)))
            if ((t == 0 and derivative < -roundoff)
                    or (t == 1 and derivative > roundoff)
                    or (0 < t < 1 and abs(derivative) > roundoff)):
                raise ValueError('Closest witness numerically fails segment optimality')
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('Numerically unresolved capsule/box geometry') from exc

    sphere = np.array_equal(start, end)
    feature = ('sphere' if sphere else 'start_cap' if t == 0 else
               'end_cap' if t == 1 else 'cylinder')
    axes = np.flatnonzero(abs(box) == half).tolist()
    if not axes:
        raise ValueError('No finite box surface witness')
    return dict(model=MODEL, frame='box_local', parameter=float(t),
        minimizer_interval=list(interval), axis_point=axis.tolist(),
        box_point=box.tolist(), capsule_point=capsule.tolist(),
        distance_m=float(distance), gap_m=float(gap),
        normal=normal.tolist(),
        feature=dict(capsule=feature, box={1:'face', 2:'edge', 3:'corner'}[len(axes)]),
        box_boundary_axes=axes, numerical_distance_floor_m=float(floor),
        witness_count=1, feature_basis='computed_source_surface_geometry',
        native_contact_observed=False, native_contact_law_parity=False,
        actuation_authorized=False, training_eligible=False)
