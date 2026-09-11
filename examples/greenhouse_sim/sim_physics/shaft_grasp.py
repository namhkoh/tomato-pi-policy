"""Pure, normal-only evidence for a discretized shaft; no control or cut authority.

Caller supplies authored collider geometry/chain identity, live intact links and
post-fetch native body frames. No stage walk, subscriptions, dwell or slip state.
Contact points/impulses must come ONLY from full-report normal contact data, never
friction anchors. PhysX normals point collider1 -> collider0; impulse acts on 0:
https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_contact_pair_point.html
The caller remains responsible for native stream coverage and truthful step IDs.
"""
from dataclasses import dataclass
import operator

import numpy as np


_EPS_M = 1e-6  # Floating-point geometry roundoff, not a fitted shaft radius.


def _array(value, shape):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite array with shape ' + str(shape) + ' required')
    result.setflags(write=False)
    return result


def _pose(value):
    result = _array(value, (4, 4)); r = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
        raise ValueError('Rigid unscaled transform required')
    return result


def _number(value, low, high):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError('Scalar outside validated range')
    return float(value)


def _index(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Integer required')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError('Integer required') from exc
    if result < 0:
        raise ValueError('Nonnegative integer required')
    return result


@dataclass(frozen=True)
class ShaftCapsule:
    body: str
    collider: str
    local_frame: np.ndarray  # Collider -> native body; capsule axis is local Z.
    radius_m: float
    half_height_m: float  # HALF cylinder height, excluding spherical caps.
    contact_offset_m: float

    def __post_init__(self):
        if not self.body.startswith('/') or self.collider != self.body + '/StemCollider':
            raise ValueError('Exact body/StemCollider identity required')
        object.__setattr__(self, 'local_frame', _pose(self.local_frame))
        _number(self.radius_m, np.finfo(float).tiny, 1.)
        _number(self.half_height_m, 0., 1.)
        _number(self.contact_offset_m, 0., .001)


@dataclass(frozen=True)
class FingerPad:
    body: str
    collider: str
    local_frame: np.ndarray  # Exact authored BOX -> native finger body.
    half_extents_m: np.ndarray
    face_axis: int  # Inner contact face, outward from pad toward the shaft.
    face_sign: int
    contact_offset_m: float

    def __post_init__(self):
        if not self.body.startswith('/') or not self.collider.startswith(self.body + '/'):
            raise ValueError('Exact finger pad collider required')
        object.__setattr__(self, 'local_frame', _pose(self.local_frame))
        half = _array(self.half_extents_m, (3,))
        if np.any(half <= 0) or np.any(half > 1):
            raise ValueError('Positive exact box half extents required')
        object.__setattr__(self, 'half_extents_m', half)
        if _index(self.face_axis) > 2 or isinstance(self.face_sign, bool) or self.face_sign not in (-1, 1):
            raise ValueError('Signed principal inner pad face required')
        _number(self.contact_offset_m, 0., .001)


class ShaftGraspEvidence:
    """begin_step -> add_contact (normal rows) -> evaluate once after fetch.

    chain is the authored ordered shaft, INCLUDING support bodies before
    cut_index, not a list of all colliders carried by its articulation. Pads are
    ordered finger1, finger2. Only selected +/- one, on the detachable side,
    linked to selected through supplied live connected_pairs, can qualify.
    Callbacks must propagate errors; faults stay latched until the next step.
    """
    def __init__(self, chain, pads, *, selected_index, cut_index, source_target, max_rows=256):
        self.chain = tuple(chain); self.pads = tuple(pads)
        self.selected_index = _index(selected_index); self.cut_index = _index(cut_index)
        self.max_rows = _index(max_rows)
        if (not self.chain or len(self.pads) != 2 or not 1 <= self.max_rows <= 4096
                or not self.cut_index <= self.selected_index < len(self.chain)
                or not isinstance(source_target, str) or not source_target):
            raise ValueError('Ordered shaft, two pads and detached selection required')
        if not all(isinstance(s, ShaftCapsule) for s in self.chain) or not all(isinstance(p, FingerPad) for p in self.pads):
            raise ValueError('Validated exact capsule and pad geometry required')
        bodies = [s.body for s in self.chain]
        branch = bodies[self.selected_index].rsplit('/', 1)[0]
        if (len(set(bodies)) != len(bodies) or len({p.body for p in self.pads}) != 2
                or set(bodies) & {p.body for p in self.pads}
                or any(s.body.rsplit('/', 1)[0] != branch for s in self.chain[self.cut_index:])):
            raise ValueError('Unique bodies from one detachable shaft branch required')
        self.source_target = source_target
        self.candidates = self.chain[max(self.cut_index, self.selected_index - 1):self.selected_index + 2]
        self.by_collider = {s.collider: s for s in self.candidates}
        self.links = {frozenset((a.body, b.body)) for a, b in zip(self.chain[self.cut_index:], self.chain[self.cut_index + 1:])}
        self.step_id = None; self.rows = []; self.error = None; self.evaluated = False

    def begin_step(self, step_id):
        step_id = _index(step_id)
        if self.step_id is not None and step_id <= self.step_id:
            raise ValueError('Strictly newer step required; recreate after native reset')
        self.step_id = step_id; self.rows = []; self.error = None; self.evaluated = False

    def add_contact(self, collider0, collider1, point, normal, impulse, separation):
        """One raw normal ContactData row; preserve original header collider order."""
        try:
            if self.step_id is None or self.evaluated or self.error:
                raise ValueError('Contact outside an open healthy step')
            matches = [(i, 1 if collider0 == p.collider else -1)
                       for i, p in enumerate(self.pads) if p.collider in (collider0, collider1)]
            # Unrelated events remain the responsibility of the existing guards.
            if not matches:
                return
            if len(matches) != 1 or len(self.rows) >= self.max_rows:
                raise ValueError('Ambiguous pair or normal contact buffer overflow')
            i, sign = matches[0]; other = collider1 if sign == 1 else collider0
            p = _array(point, (3,)); n = _array(normal, (3,)); j = _array(impulse, (3,))
            length = float(np.linalg.norm(n))
            if not np.isclose(length, 1., atol=1e-4, rtol=0):
                raise ValueError('Unit native normal required')
            n = n / length; scalar = float(np.dot(j, n)); magnitude = float(np.linalg.norm(j))
            if (not np.isfinite([scalar, magnitude]).all() or scalar < 0
                    or np.linalg.norm(j - scalar * n) > 1e-12 + 1e-5 * magnitude):
                raise ValueError('Normal impulse must be compressive and collinear; no friction')
            separation = _number(separation, -1., 1.)
            self.rows.append((i, other, p, sign * n, sign * scalar * n, separation))
        except (ValueError, TypeError, OverflowError) as exc:
            self.error = str(exc)
            raise

    def evaluate(self, *, step_id, frames_step_id, dt, body_frames, connected_pairs):
        """No dwell update: caller retains existing 20 mN / cos < -0.5 window.

        connected_pairs are exact native body IDs for intact adjacent shaft
        links, from live topology. frames_step_id explicitly binds post-fetch
        frames to the contact step; this pure helper cannot verify the producer.
        Surface tolerance uses only authored contact offsets, reported separation
        and 1 um roundoff. Penetration >1 mm rejects; geometry is never inflated.
        """
        if (self.error or self.step_id is None or self.evaluated
                or _index(step_id) != self.step_id or _index(frames_step_id) != self.step_id):
            raise ValueError('Faulted, stale or already consumed native evidence: ' + str(self.error))
        self.evaluated = True
        dt = _number(dt, np.finfo(float).tiny, 1.)
        links = set()
        for pair in connected_pairs:
            if len(pair) != 2 or frozenset(pair) not in self.links:
                raise ValueError('Connectivity must use exact adjacent detachable body IDs')
            links.add(frozenset(pair))
        selected = self.chain[self.selected_index].body
        eligible = {s.collider for s in self.candidates
                    if s.body == selected or frozenset((selected, s.body)) in links}
        shapes = (*self.candidates, *self.pads)
        world = {s.collider: _pose(_pose(body_frames[s.body]) @ s.local_frame) for s in shapes}
        forces = np.zeros((2, 3)); loads = np.zeros(2); counts = [0, 0]
        pairs = {}; rejected = []; points = []; separations = []
        for row, (i, other, point, normal, impulse, separation) in enumerate(self.rows):
            shaft = self.by_collider.get(other); pad = self.pads[i]
            reason = None
            if shaft is None or other not in eligible:
                rejected.append(dict(row=row, finger=pad.body, collider=other, reason='not_connected_detachable_shaft'))
                continue
            force = impulse / dt
            forces[i] += force; loads[i] += np.linalg.norm(force); counts[i] += 1
            key = (i, other)
            pair = pairs.setdefault(key, dict(finger=pad.body, body=shaft.body, collider=other,
                force_n=np.zeros(3), normal_load_upper_n=0., count=0))
            pair['force_n'] += force; pair['normal_load_upper_n'] += float(np.linalg.norm(force)); pair['count'] += 1
            points.append(point.tolist()); separations.append(separation)
            sw, pw = world[other], world[pad.collider]
            q = _array((point - sw[:3, 3]) @ sw[:3, :3], (3,))
            axis_point = np.array([0., 0., np.clip(q[2], -shaft.half_height_m, shaft.half_height_m)])
            radial = q - axis_point; distance = float(np.linalg.norm(radial))
            qp = _array((point - pw[:3, 3]) @ pw[:3, :3], (3,))
            face = np.clip(qp, -pad.half_extents_m, pad.half_extents_m)
            face[pad.face_axis] = pad.face_sign * pad.half_extents_m[pad.face_axis]
            offset = shaft.contact_offset_m + pad.contact_offset_m
            tolerance = max(abs(separation), offset) + _EPS_M
            pad_outward = pad.face_sign * pw[:3, pad.face_axis]
            if not np.isfinite(distance) or not np.isfinite(np.linalg.norm(qp - face)):
                raise ValueError('Overflowing native contact geometry')
            if separation < -.001 or separation > offset + _EPS_M:
                reason = 'separation_outside_native_contact_guards'
            elif abs(distance - shaft.radius_m) > tolerance or np.linalg.norm(qp - face) > tolerance:
                reason = 'point_off_capsule_or_inner_pad_face'
            elif distance == 0 or np.dot(normal, sw[:3, :3] @ radial) <= 0 or np.dot(normal, -pad_outward) <= 0:
                reason = 'normal_not_compressive_on_capsule_and_pad'
            if reason:
                rejected.append(dict(row=row, finger=pad.body, collider=other, reason=reason))
        if not np.isfinite(forces).all() or not np.isfinite(loads).all():
            raise ValueError('Overflowing native force totals')
        norms = np.linalg.norm(forces, axis=1)
        if not np.isfinite(norms).all():
            raise ValueError('Overflowing native resultant')
        cosine = float(np.dot(forces[0] / norms[0], forces[1] / norms[1])) if np.all(norms > 0) else None
        for pair in pairs.values():
            pair['force_n'] = pair['force_n'].tolist()
        return dict(step_id=self.step_id, source_target=self.source_target,
            selected_body=selected, eligible_colliders=sorted(eligible), forces=forces.tolist(),
            counts=counts, points=points, min_separation=min(separations, default=0.),
            normal_load_upper_n=loads.tolist(), pairs=list(pairs.values()), rejected=rejected,
            forces_scope='eligible_identity_normal_rows_including_geometry_rejections',
            stem_only=not rejected, opposition_cosine=cosine,
            bilateral=bool(not rejected and np.all(norms >= .02) and cosine is not None and cosine < -.5),
            normal_only=True, friction_used_for_grasp=False, training_eligible=False)
