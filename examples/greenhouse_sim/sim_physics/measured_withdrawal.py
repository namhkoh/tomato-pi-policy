"""Pure, opt-in measured-start withdrawal; NEVER sends native commands.

Paths are BimanualRobot.plan['stroke'/'approach'] arrays in degrees (N,7).
wrist_fk(q_degrees) can wrap kin.forward('right', q, base); edge_in_wrist is
knife.frame(identity). Native snapshots/hold targets are supplied by the caller.
Only an existing backward stroke prefix and reversed approach are considered.

Validator(request) must synchronously screen the EXACT supplied sampled path,
including the actual native start, all robot bodies, current plant geometry and
existing grasp allowances. It owns native query creation/final validation/close.
Receipts are structurally checked and hash/step bound, NOT authenticated native
evidence. A dishonest callback cannot be made trustworthy by a pure helper.
Callbacks must not step physics or send commands. Late results fail closed;
Python cannot preempt a blocking callback. No continuous collision certificate,
full forward cut, fracture calibration, or physical safety claim is made.

construct -> initial_command() (measured joints) -> advance(next fetched sample).
advance returns no target on failure, permanently latching blocked. Caller MUST
stop execution on blocked and preserve ALL existing contact/slip/force guards.
Left hold targets never change. dt/rates are explicit caller-owned protocol
values; rates must come from the existing controller/source limits, not tuning.

Budgets: each constructor/advance synchronous operation gets a NEW 8 wall-second
deadline. Waiting between fetched samples consumes none of the next deadline.
At most 4800 advance calls (4800*dt simulation seconds), 4801 validation calls,
4096 samples per path, and 20000 TOTAL native queries are permitted. Query
exhaustion blocks, not a reason to expand the budget or reuse stale clearance.
"""
from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from .withdrawal_evidence import _pose, _sample_id, withdrawal_evidence


MAX_SAMPLES = 4096
MAX_QUERIES = 20000
MAX_UPDATES = 4800
MAX_SECONDS = 8.
JOINT_SAMPLE_DEG = 1.
JOINT_ATTAINMENT_DEG = .1  # Additional engineering posture check, not calibration.
MARGINS = dict(self_m=.003, interarm_m=.01, scene_m=.001)
CHECKS = ('native_inventory', 'measured_start', 'joint_limits', 'self', 'interarm',
          'left_scene', 'right_scene', 'tool', 'native_static', 'hold_unchanged',
          'native_guards', 'bilateral_grasp')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('Numeric, nonboolean '+name+' required')
    a = np.array(raw, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite '+name+' with shape '+str(shape)+' required')
    return a


def _paths(value):
    if (type(value) is not dict or not 1 <= len(value) <= MAX_SAMPLES
            or any(type(p) is not str or not p.startswith('/') or p.endswith('/') for p in value)):
        raise ValueError('Complete path-labelled native body snapshot required')
    return {p: _pose(m).tolist() for p, m in value.items()}


def _measurement(value):
    m = deepcopy(value)
    if type(m) is not dict:
        raise ValueError('Explicit fetched measurement required')
    m['sample_id'] = list(_sample_id(m['sample_id']))
    m['right_q_deg'] = _array(m['right_q_deg'], (7,), 'right joints').tolist()
    m['right_wrist_world'] = _pose(m['right_wrist_world']).tolist()
    hold = m['hold_targets']
    if type(hold) is not dict or set(hold) != {'left_goal_world', 'finger_targets_m'}:
        raise ValueError('Exact unchanged left goal and two finger targets required')
    hold['left_goal_world'] = _pose(hold['left_goal_world']).tolist()
    hold['finger_targets_m'] = _array(hold['finger_targets_m'], (2,), 'finger targets').tolist()
    scene = m['scene']
    if type(scene) is not dict or list(_sample_id(scene['sample_id'])) != m['sample_id']:
        raise ValueError('Scene snapshot must have the fetched physics sample ID')
    for name in ('robot_body_world', 'plant_body_world'):
        scene[name] = _paths(scene[name])
    scene['left_q_deg'] = _array(scene['left_q_deg'], (7,), 'measured left joints').tolist()
    slides = scene['finger_slides_m']
    if type(slides) is not dict or not slides or any(type(k) is not str for k in slides):
        raise ValueError('Exact native finger slide inventory required')
    scene['finger_slides_m'] = {k: _array(v, (), 'finger slide').item() for k,v in slides.items()}
    path = m['right_wrist_path']
    if path not in scene['robot_body_world'] or not np.allclose(
            scene['robot_body_world'][path], m['right_wrist_world'], atol=1e-7, rtol=0):
        raise ValueError('Native wrist/body snapshot mismatch')
    return json.loads(_json(m))


def _attained(actual, desired):
    return withdrawal_evidence(actual, desired, clearance_verified=False,
        measured_sample_id=(0,0), clearance_sample_id=(0,0))['endpoint_attained']


def _dense(a, b):
    count = max(2, int(np.ceil(np.max(abs(b-a))/JOINT_SAMPLE_DEG))+1)
    if count > MAX_SAMPLES:
        raise ValueError('Segment sample budget exceeded')
    return np.linspace(a,b,count)


class MeasuredWithdrawal:
    """One release snapshot, no search/retry, no command-fraction authority.

Receipt schema is exercised in measured_withdrawal_test.validator. Request
hash includes snapshot, hold targets, joint samples, phases and fixed margins.
Native receipt uses NativeStaticClearance.report() fields. The adapter must
add exact coverage checks/counts; merely returning passed=True is insufficient.
"""

    def __init__(self, stroke, approach, *, measurement, lower_deg, upper_deg,
                 max_speed_deg_s, dt, wrist_fk, edge_in_wrist, stroke_direction, validate,
                 quantize_command=None):
        if not callable(wrist_fk) or not callable(validate):
            raise ValueError('Explicit FK and complete path validator required')
        self._fk, self._validate = wrist_fk, validate
        if quantize_command is not None and not callable(quantize_command):
            raise ValueError('Explicit command quantizer must be callable')
        self._quantize = quantize_command
        self._deadline = time.monotonic()+MAX_SECONDS
        self._edge = _pose(edge_in_wrist)
        self._direction = _array(stroke_direction, (3,), 'world stroke direction')
        if abs(np.linalg.norm(self._direction)-1) > 1e-8:
            raise ValueError('Unit stroke direction required; no normalization')
        self._lower = _array(lower_deg, (7,), 'lower limits')
        self._upper = _array(upper_deg, (7,), 'upper limits')
        self._speed = _array(max_speed_deg_s, (7,), 'explicit command rates')
        self._dt = _array(dt, (), 'dt').item()
        if (np.any(self._lower >= self._upper) or np.any(self._speed <= 0)
                or not 0 < self._dt <= .1):
            raise ValueError('Valid limits, positive rates and bounded dt required')
        self._queries = 0; self._updates = 0; self._validation_calls = 0
        self._blocked = None; self._busy = False
        m = _measurement(measurement)
        self._sample = tuple(m['sample_id']); self._hold = deepcopy(m['hold_targets'])
        self._inventory = self._inventory_of(m)
        self._wrist_path = m['right_wrist_path']
        self._q = self._joints(m['right_q_deg'])
        if not np.array_equal(self._packet(self._q,self._q),self._q):
            raise ValueError('Initial command packet must preserve exact native measured joints')
        self._measured_fk(m)
        paths = []
        for name, value in (('stroke',stroke), ('approach',approach)):
            a = np.asarray(value)
            if a.ndim != 2 or a.shape[1] != 7 or not 2 <= len(a) <= MAX_SAMPLES:
                raise ValueError('Bounded (N,7) '+name+' required')
            paths.append(np.array([self._joints(row) for row in a]))
        stroke, approach = paths
        if not _attained(self._pose(stroke[0]), self._pose(approach[-1])):
            raise ValueError('Existing stroke/approach pose junction mismatch')
        progress = np.array([self._progress(self._pose(q)) for q in stroke])
        observed = self._progress(m['right_wrist_world'])
        if np.any(np.diff(progress) <= 0) or not progress[0] <= observed <= progress[-1]:
            raise ValueError('Measured edge outside monotone existing stroke; no clamp')
        index = min(int(np.searchsorted(progress, observed, side='right'))-1, len(stroke)-2)
        alpha = (observed-progress[index])/(progress[index+1]-progress[index])
        projected = stroke[index]*(1-alpha)+stroke[index+1]*alpha
        if not _attained(m['right_wrist_world'], self._pose(projected)):
            raise ValueError('Measured pose is off the existing stroke; no snap')
        # Include a source node only at/before OBSERVED progress, never sent fraction.
        last = int(np.searchsorted(progress, observed, side='right'))-1
        vertices = [(q,'extract') for q in stroke[:last+1][::-1]]
        vertices += [(q,'return') for q in approach[::-1]]
        samples = [self._q.copy()]; phases = []
        for q, phase in vertices:
            chunk = _dense(samples[-1],q)
            if len(samples)+len(chunk)-1 > MAX_SAMPLES:
                raise ValueError('Withdrawal sample budget exceeded')
            samples.extend(chunk[1:]); phases.extend([phase]*(len(chunk)-1))
        self._path = np.array(samples); self._phases = phases; self._cursor = 1
        self._initial_receipt = self._screen(m, self._path, phases, 'release_path')
        self._initial = self._result('ready', self._q, self._initial_receipt, m)

    @staticmethod
    def _inventory_of(m):
        s = m['scene']
        return tuple(tuple(sorted(s[k])) for k in ('robot_body_world','plant_body_world','finger_slides_m'))

    def _joints(self, q):
        q = _array(q, (7,), 'right joints')
        if np.any(q <= self._lower) or np.any(q >= self._upper):
            raise ValueError('Joint limit violation; no clipping')
        return q

    def _pose(self, q):
        if time.monotonic() >= self._deadline:
            raise ValueError('FK/validation deadline exceeded')
        result = _pose(self._fk(np.array(q, copy=True)))
        if time.monotonic() >= self._deadline:
            raise ValueError('FK/validation deadline exceeded')
        return result

    def _packet(self, measured, desired):
        """Validate the actual representable command BEFORE geometry/receipt."""
        if self._quantize is None:return desired.copy()
        packet=self._joints(self._quantize(measured.copy(),desired.copy()))
        delta=desired-measured;actual=packet-measured
        if np.any(actual*delta<0) or np.any(abs(actual)>abs(delta)):
            raise ValueError('Command quantization exceeds requested measured-to-goal interval')
        return packet

    def _progress(self, pose):
        return float((np.asarray(pose)@self._edge)[:3,3]@self._direction)

    def _measured_fk(self, m):
        if not _attained(m['right_wrist_world'], self._pose(m['right_q_deg'])):
            raise ValueError('Native pose/joint FK inconsistency')

    def _screen(self, m, path, phases, kind):
        started = time.monotonic()
        if self._queries >= MAX_QUERIES or self._validation_calls >= MAX_UPDATES+1:
            raise ValueError('Total validation/query budget exhausted')
        if len(path) > MAX_SAMPLES or len(phases) != len(path)-1:
            raise ValueError('Invalid bounded path coverage')
        poses = [self._pose(q) for q in path]
        # Use the ACTUAL native start, not FK/command progress, at the splice.
        progress = [self._progress(m['right_wrist_world'])]+[self._progress(p) for p in poses[1:]]
        if any(p == 'extract' and progress[i+1] > progress[i]+1e-12 for i,p in enumerate(phases)):
            raise ValueError('Extraction connector would command forward edge travel')
        request = dict(kind=kind, sample_id=m['sample_id'], snapshot_sha256=_hash(m),
            measurement=deepcopy(m), path_q_deg=np.asarray(path).tolist(),
            path_wrist_world=[p.tolist() for p in poses], phases=list(phases),
            seam_allowance=[p == 'extract' for p in phases], margins_m=dict(MARGINS),
            maximum_joint_sample_step_deg=JOINT_SAMPLE_DEG,
            max_queries=MAX_QUERIES-self._queries, max_seconds=MAX_SECONDS,
            left_hold_fixed=True, full_forward_cutstroke_verified=False)
        request['request_sha256'] = _hash(request)
        expected = deepcopy(request)
        self._validation_calls += 1
        receipt = self._validate(request)
        if _hash(request) != _hash(expected):
            raise ValueError('Validator mutated its exact path request')
        if time.monotonic()-started >= MAX_SECONDS or time.monotonic() >= self._deadline:
            raise ValueError('Synchronous validation deadline exceeded')
        # JSON detaches mutable callback objects; NaN and unsupported values fail.
        r = json.loads(_json(receipt))
        if type(r) is dict:
            _sample_id(r.get('sample_id'))
        if (type(r) is not dict or r.get('passed') is not True
                or r.get('request_sha256') != expected['request_sha256']
                or r.get('snapshot_sha256') != expected['snapshot_sha256']
                or r.get('sample_id') != expected['sample_id']
                or r.get('margins_m') != MARGINS or r.get('errors') != []
                or type(r.get('checked_samples')) is not int or r['checked_samples'] != len(path)
                or type(r.get('checked_segments')) is not int or r['checked_segments'] != len(phases)
                or type(r.get('checks')) is not dict
                or any(r['checks'].get(k) is not True for k in CHECKS)):
            raise ValueError('Incomplete, stale or failed path validation receipt')
        n = r.get('native_static', {}); e = n.get('epoch', {})
        if (n.get('final_validation_passed') is not True or n.get('closed') is not True
                or n.get('errors') != [] or type(n.get('query_count')) is not int
                or not 1 <= n['query_count'] <= MAX_QUERIES-self._queries
                or type(e.get('revision')) is not int or e['revision'] != 0
                or e.get('subscriptions_closed') is not True or e.get('cleanup_errors') != []
                or e.get('invalidation_reasons') != []):
            raise ValueError('Unvalidated native epoch/coverage/budget/cleanup')
        used = n.get('used_static_colliders'); covered = n.get('final_coverage_checked')
        if (type(used) is not list or type(covered) is not list or not covered
                or any(type(p) is not str or not p.startswith('/') for p in used+covered)
                or len(set(covered)) != len(covered) or not set(used) <= set(covered)):
            raise ValueError('Native final positive coverage missing')
        self._queries += n['query_count']
        return r

    def _result(self, status, target, receipt, m):
        return dict(status=status, target_right_q_deg=None if target is None else np.asarray(target).tolist(),
            hold_targets=deepcopy(self._hold), sample_id=list(m['sample_id']), receipt=deepcopy(receipt),
            waypoint_index=self._cursor, path_samples=len(self._path), native_query_count=self._queries,
            validation_calls=self._validation_calls, max_validation_seconds=MAX_SECONDS,
            maximum_total_queries=MAX_QUERIES, maximum_advance_calls=MAX_UPDATES,
            maximum_episode_simulation_seconds=MAX_UPDATES*self._dt, cut_authorized=False,
            right_withdrawal_completed=status == 'complete', native_provenance_verified=False,
            whole_path_certified=False, full_forward_cutstroke_verified=False,
            physical_cut_verified=False, tissue_fracture_calibrated=False)

    def initial_command(self):
        """Same release-sample command only; caller must not reuse on later ticks."""
        if self._updates or self._blocked:
            raise ValueError('Initial command expired')
        return deepcopy(self._initial)

    def advance(self, measurement):
        """Exactly next fetched step; no timer advances a waypoint or completion."""
        if self._blocked is not None:
            return dict(status='blocked', reason=self._blocked, target_right_q_deg=None,
                        right_withdrawal_completed=False)
        if self._busy:
            self._blocked = 'Reentrant validation is forbidden'
            raise RuntimeError(self._blocked)
        self._busy = True
        try:
            self._deadline = time.monotonic()+MAX_SECONDS
            self._updates += 1
            if self._updates > MAX_UPDATES:
                raise ValueError('Withdrawal update budget exceeded')
            m = _measurement(measurement)
            if tuple(m['sample_id']) != (self._sample[0],self._sample[1]+1):
                raise ValueError('Exactly next same-episode fetched sample required')
            if (_hash(m['hold_targets']) != _hash(self._hold)
                    or self._inventory_of(m) != self._inventory or m['right_wrist_path'] != self._wrist_path):
                raise ValueError('Hold targets or native body inventory changed')
            q = self._joints(m['right_q_deg']); self._measured_fk(m)
            goal = self._path[self._cursor]
            arrived = bool(np.max(abs(q-goal)) < JOINT_ATTAINMENT_DEG
                           and _attained(m['right_wrist_world'], self._pose(goal)))
            cursor = self._cursor
            if arrived and cursor < len(self._path)-1:
                cursor += 1; goal = self._path[cursor]
            complete = arrived and self._cursor == len(self._path)-1
            if complete:
                path = np.array([q]); phases = []
            else:
                delta = goal-q
                moving = abs(delta) > 0
                alpha = min(1., float(np.min(self._speed[moving]*self._dt/abs(delta[moving])))) if np.any(moving) else 1.
                target = self._packet(q,self._joints(q+alpha*delta))
                path = _dense(q,target); phases = [self._phases[cursor-1]]*(len(path)-1)
            receipt = self._screen(m,path,phases,'endpoint' if complete else 'next_segment')
            if self._blocked:
                raise ValueError(self._blocked)
            self._cursor = cursor; self._sample = tuple(m['sample_id'])
            return self._result('complete' if complete else 'moving', None if complete else path[-1], receipt, m)
        except BaseException as exc:
            self._blocked = type(exc).__name__+': '+str(exc)
            if not isinstance(exc, Exception):
                raise
            return dict(status='blocked', reason=self._blocked, target_right_q_deg=None,
                        right_withdrawal_completed=False)
        finally:
            self._busy = False
