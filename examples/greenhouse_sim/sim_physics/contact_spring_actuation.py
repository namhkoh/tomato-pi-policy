"""Experimental HOLD-ONLY fresh-contact spring effort submission.

Only intrinsic generalized efforts are written. Native contacts remain active;
predicted contact forces, root wrench, poses and velocities are NEVER submitted.
The caller authenticates the geometry/material export, supplies same-episode
snapshots synchronously (no intervening physics), and verifies the real external
support and all native guards. The support flag is not independently observable
through this adapter. No bootstrap, release mode, clipping or legacy fallback.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from . import grasp_contact_shadow
from .implicit_springs import ImplicitJointSprings
from .plant_contact_binding import PlantContactBinding, SCHEMA, _digest, _pack
from .plant_contact_stream import MAX_ROWS


DOFS = 39
MODEL = 'experimental_fresh_box_hold_springs_v1'


def _array(value, shape, name):
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in 'fiu'
            or any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat)
            or not np.isfinite(raw).all()):
        raise ValueError('Finite numeric '+name+' of shape '+str(shape)+' required')
    return np.array(raw, dtype=float, copy=True)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _names(value):
    names = list(value)
    if (len(names) != DOFS or len(set(names)) != DOFS
            or any(type(n) is not str or not 1 <= len(n) <= 256 for n in names)):
        raise ValueError('Exactly 39 unique original intrinsic DOF names required')
    return names


def _binding_record(binding):
    if type(binding) is not PlantContactBinding or binding.schema != SCHEMA:
        raise ValueError('Deserialized cached-bound geometry required, not authored-only geometry')
    _digest(binding.sha256); _digest(binding.binding_sha256)
    return dict(source_target=binding.source_target, sha256=binding.sha256,
        binding_sha256=binding.binding_sha256, cut_index=binding.cut_index,
        selected_body=binding.selected_body,
        chain=[_pack(s) for s in binding.chain], pads=[_pack(p) for p in binding.pads],
        plant_colliders=list(binding.plant_collider_paths),
        compliance=dict(binding.finger_contact_compliance or {}), friction=binding.finger_friction)


class FreshHoldSprings:
    """One instance per native episode; ANY step fault latches without recovery.

    The explicit keyword is required, not an enabled default. A rejected solve
    issues no writes. A submission/readback failure may occur AFTER a write:
    it is latched and propagated; caller must stop, never advance physics using
    the preceding effort or retry this instance. Readback verifies a command
    buffer, NOT physical realized torque or numerical/native qualification.
    """
    def __init__(self, springs, binding, drive_parameters, *, experimental_hold_only):
        if experimental_hold_only is not True or not isinstance(springs, ImplicitJointSprings):
            raise ValueError('Explicit experimental_hold_only=True and existing ImplicitJointSprings required')
        self.springs = springs
        self.a = springs.articulation
        self.binding = binding
        self.error = None
        self.last_step = None
        self._busy = False
        self._binding_hash = _hash(_binding_record(binding))
        self.names = _names(drive_parameters['names'])
        self.paths = [s.body for s in binding.chain[binding.cut_index:]]
        if (len(self.paths) != 14 or len(set(self.paths)) != 14
                or binding.selected_body not in self.paths or len(binding.pads) != 2
                or not {s.collider for s in binding.chain} <= set(binding.plant_collider_paths)
                or len(set(binding.plant_collider_paths)) != len(binding.plant_collider_paths)):
            raise ValueError('Full 14-link/39-DOF source chain and complete collider inventory required')
        self._drive = dict(names=list(self.names))
        for key in ('stiffness', 'damping', 'max_forces', 'targets'):
            x = _array(drive_parameters[key], (1, DOFS), 'original '+key)
            if ((key == 'targets' and np.any(x != 0))
                    or (key != 'targets' and np.any(x < 0))
                    or (key == 'max_forces' and np.any(x <= 0))):
                raise ValueError('Original nonnegative K/C, positive caps and zero rest targets required')
            self._drive[key] = x.tolist()
        self._k = np.array(self._drive['stiffness'][0])
        self._c = np.array(self._drive['damping'][0])
        self._caps = np.array(self._drive['max_forces'][0])
        self._drive_hash = _hash(self._drive)
        self._indices = np.array([0], dtype=np.uint32)
        self._native_contract()

    def _native_contract(self):
        a = self.a
        if (self.springs.articulation is not a or a.count != 1
                or a.shared_metatype.fixed_base or self.springs.fixed_base
                or _names(a.shared_metatype.dof_names) != self.names
                or list(a.link_paths[0]) != self.paths
                or _hash(_binding_record(self.binding)) != self._binding_hash
                or _hash(self._drive) != self._drive_hash
                or not np.array_equal(_array(self.springs.k, (DOFS,), 'retained K'), self._k)
                or not np.array_equal(_array(self.springs.c, (DOFS,), 'retained C'), self._c)):
            raise ValueError('Original source, body/DOF order or retained K/C changed')
        for method in ('get_dof_stiffnesses', 'get_dof_dampings',
                       'get_dof_position_targets', 'get_dof_velocity_targets'):
            if np.any(_array(getattr(a, method)(), (1, DOFS), method) != 0):
                raise ValueError('Native K/C and rest/velocity targets must remain exactly zero')
        if (not np.array_equal(_array(a.get_dof_max_forces(), (1, DOFS), 'native caps')[0], self._caps)
                or np.any(_array(a.get_drive_types(), (1, DOFS), 'Force drive types') != 1)):
            raise ValueError('Original native force caps / Force drive type changed')

    def _state_matches(self, now):
        self._native_contract()
        q = _array(self.a.get_dof_positions(), (1, DOFS), 'live q')[0]
        v = np.r_[_array(self.a.get_root_velocities(), (1, 6), 'live root velocity')[0],
                  _array(self.a.get_dof_velocities(), (1, DOFS), 'live joint velocity')[0]]
        if (not np.array_equal(q, _array(now['q_rad'], (DOFS,), 'snapshot q'))
                or not np.array_equal(v, _array(now['generalized_velocity'], (DOFS+6,), 'snapshot v'))):
            raise ValueError('Actual pre-solve q/v differ from current.before')

    def _records(self, previous, current):
        if not isinstance(previous, dict) or not isinstance(current, dict):
            raise ValueError('Adjacent previous/current records required; no bootstrap fallback')
        for row in (previous, current):
            now = row['before']
            if (type(row['step_id']) is not int or type(now['step_id']) is not int
                    or now['step_id'] < 0 or row['step_id'] != now['step_id']+1
                    or type(row['dt_s']) not in (int, float)
                    or not np.isfinite(row['dt_s']) or not 0 < row['dt_s'] <= .02
                    or now.get('schema') != 'full_plant_native_prediction_snapshot_v1'
                    or now.get('external_root_support_enabled_caller_asserted') is not True
                    or now.get('contact_force_included') is not False
                    or now.get('constraint_forces_included') is not False
                    or now.get('native_com_jacobian_velocity_check_passed') is not True
                    or now['source_target'] != self.binding.source_target
                    or now['joint_names'] != self.names or now['body_paths'] != self.paths):
                raise ValueError('Exact source-bound supported HOLD-only snapshot required')
        step = current['before']['step_id']
        stream = previous['contacts']
        if (step < 1 or previous['step_id'] != step or previous['dt_s'] != current['dt_s']
                or (self.last_step is not None and step != self.last_step+1)
                or stream.get('error') is not None or not isinstance(stream['rows'], list)
                or type(stream['row_count']) is not int or not 0 <= stream['row_count'] <= MAX_ROWS
                or stream['row_count'] != len(stream['rows'])
                or stream['plant_collider_paths'] != list(self.binding.plant_collider_paths)
                or not np.array_equal(_array(previous['after_generalized_velocity'], (DOFS+6,), 'previous fetched v'),
                                      _array(current['before']['generalized_velocity'], (DOFS+6,), 'current v'))):
            raise ValueError('Stale/repeated/incomplete or different-inventory contact generation')
        return step


    def _predicted_finger_loads(self, result, geometry):
        # One unscaled normal witness and one 2-D friction anchor per fresh pair.
        # Never sum vectors across opposing contacts: every magnitude contributes.
        pairs = geometry['observed_pairs']; count = len(pairs)
        normal = _array(result.get('normal_forces_n', []) if count == 0 else result['normal_forces_n'],
                        (count,), 'predicted pair normal loads')
        friction = result['friction_forces_n']
        if (not isinstance(friction, list) or len(friction) != count
                or result['anchor_counts'] != [1]*count
                or geometry['patches']['normal_indices'] != [[i] for i in range(count)]):
            raise ValueError('Exact fresh pair / anchor force ordering required')
        loads = {pad.collider: 0. for pad in self.binding.pads}; seen = set()
        for i, pair in enumerate(pairs):
            identity = (pair['cap'], pair['pad'])
            if (identity in seen or pair['pad'] not in loads
                    or pair['cap'] not in self.binding.shafts_by_collider
                    or self.binding.shafts_by_collider[pair['cap']].body not in self.paths
                    or pair['model_feature_count'] != 1 or pair['model_anchor_count'] != 1
                    or geometry['features'][i]['cap'] != pair['cap']
                    or geometry['features'][i]['pad'] != pair['pad']):
                raise ValueError('Exact unique source shaft / pad pair required for load guard')
            seen.add(identity)
            f = _array(friction[i], (1, 2), 'predicted fresh friction anchor')[0]
            magnitude = float(np.hypot(f[0], f[1]))
            loads[pair['pad']] += abs(float(normal[i]))+magnitude
        if any(not np.isfinite(v) or v >= .5 for v in loads.values()):
            raise ValueError('Predicted target per-finger normal+friction upper bound >=0.5 N; no clipping')
        return loads

    def _command_report(self, prediction, step, previous):
        r = prediction['result']; g = prediction['geometry']
        if (r['status'] != 'resolved' or prediction['configured_force_caps_passed'] is not True
                or prediction['geometry_model'] != 'fresh_box'
                or prediction['root_next_velocity_prescribed_zero'] is not True
                or prediction['measured_contact_forces_used'] is not False
                or prediction['original_finger_compliance'] != dict(self.binding.finger_contact_compliance)
                or r['root_dofs'] != 0 or r['generalized_coordinates'] != DOFS
                or r['root_actuated'] is not False or r['contact_force_applied'] is not False
                or r['friction_force_applied'] is not False
                or g['source_target'] != self.binding.source_target or g['current_step'] != step
                or g['reference_step'] != step-1):
            raise ValueError('Unresolved or inconsistent HOLD-only spring prediction')
        effort = _array(r['tau_joint'], (DOFS,), 'predicted intrinsic effort')
        if (not np.array_equal(effort, _array(r['tau_spring'], (DOFS,), 'spring effort'))
                or np.any(abs(effort) > self._caps)):
            raise ValueError('Intrinsic source effort cap exceeded; no clipping')
        with np.errstate(over='ignore'):
            command = effort.astype(np.float32)
        if not np.isfinite(command).all() or np.any(abs(command.astype(float)) > self._caps):
            raise ValueError('Float32 effort exceeds original cap; no clipping')
        residuals = {}
        for key in ('relative_equilibrium_residual', 'relative_fixed_point_residual',
                    'normal_law_residual_n', 'friction_fixed_point_residual_n', 'cone_max_violation_n'):
            if key in r:
                value = float(_array(r[key], (), key))
                if value < 0: raise ValueError('Negative algebra residual')
                residuals[key] = value
        if 'relative_equilibrium_residual' not in residuals:
            raise ValueError('Missing resolved equilibrium evidence')
        iterations = r['iterations']
        if type(iterations) is not int or not 0 <= iterations <= 64:
            raise ValueError('Bounded solver iteration count required')
        pairs = g['observed_pairs']; features = g['features']
        if (not isinstance(pairs, list) or not isinstance(features, list)
                or not 0 <= len(pairs) <= 16 or len(features) != len(pairs)
                or g['feature_covered'] != [True]*len(features)):
            raise ValueError('Bounded covered fresh geometry required')
        predicted_loads = self._predicted_finger_loads(r, g)
        solve_s = float(_array(prediction['solve_wall_s'], (), 'solve time'))
        if solve_s < 0: raise ValueError('Nonnegative solve time required')
        report = dict(model=MODEL, experimental_hold_only=True, before_step=step,
            expected_after_step=step+1, contact_generation_step=previous['before']['step_id'],
            dt_s=previous['dt_s'], source_target=self.binding.source_target,
            geometry_sha256=self.binding.sha256, binding_sha256=self.binding.binding_sha256,
            original_drive_parameters_sha256=self._drive_hash, dof_names_sha256=_hash(self.names),
            dof_count=DOFS, geometry_model='fresh_box', observed_pair_count=len(pairs),
            generated_normal_point_count=len(features), input_contact_row_count=previous['contacts']['row_count'],
            ignored_outside_dynamic_row_count=len(g['ignored_outside_dynamic_system']),
            algebra_residuals=residuals, solver_iterations=iterations, solve_wall_s=solve_s,
            effort_dtype='float32', effort_shape=[1, DOFS],
            submitted_effort_sha256=hashlib.sha256(command.astype('<f4').tobytes()).hexdigest(),
            maximum_abs_effort_nm=float(np.max(abs(command))),
            maximum_source_cap_fraction=float(np.max(abs(command.astype(float))/self._caps)),
            maximum_float32_rounding_nm=float(np.max(abs(command.astype(float)-effort))),
            predicted_target_per_finger_contact_upper_bound_n=predicted_loads,
            predicted_target_finger_limit_n=.5, predicted_target_finger_limit_reject_at_or_above=True,
            predicted_load_scope='discovered_target_shaft_pad_pairs_only_not_all_contacts',
            actual_all_contact_guard_still_required=True,
            source_force_caps_are_mechanical_certification=False,
            maximum_original_source_effort_cap_nm=float(np.max(self._caps)),
            configured_force_caps_passed=True, original_K_C_unchanged=True,
            native_K_C_zero_verified=True, actual_pre_solve_q_v_exact=True,
            actual_effort_write_verified=False, write_attempted=False,
            effort_basis='actuation_command_readback_not_measured_torque',
            root_constrained_caller_asserted=True, root_support_independently_verified=False,
            geometry_epoch_caller_owned=True, physical_calibration=False,
            native_cooked_geometry_verified=False, native_contact_completeness_verified=False,
            native_qualified=False, native_contact_law_parity=False, training_eligible=False,
            contact_force_submitted=False, friction_force_submitted=False, root_actuated=False,
            bootstrap_or_release_supported=False, fallback_used=False)
        return command, report

    def step(self, previous, current):
        """Return (exact submitted float32[39], bounded report), or latch/raise."""
        if self.error is not None:
            raise RuntimeError('FreshHoldSprings latched: '+self.error)
        if self._busy:
            self.error = 'Reentrant FreshHoldSprings.step'
            raise RuntimeError(self.error)
        self._busy = True
        try:
            step = self._records(previous, current)
            # Copy caller evidence so the predictor cannot mutate their trace.
            p, c = deepcopy(previous), deepcopy(current)
            self._state_matches(c['before'])
            prediction = grasp_contact_shadow.predict(p, c, self.binding, deepcopy(self._drive),
                                                       geometry_model='fresh_box')
            command, report = self._command_report(prediction, step, p)
            self._state_matches(c['before'])  # No physics / parameter change during solve.
            if self.error is not None: raise RuntimeError(self.error)
            report['write_attempted'] = True
            self.a.set_dof_actuation_forces(command[None, :].copy(), self._indices.copy())
            readback = np.array(self.a.get_dof_actuation_forces(), copy=True)
            _array(readback, (1, DOFS), 'actuation readback')
            if (readback.dtype != np.float32
                    or not np.array_equal(readback[0].view(np.uint32), command.view(np.uint32))):
                raise RuntimeError('Exact float32 intrinsic effort readback failed; write may have occurred')
            self.last_step = step
            report['actual_effort_write_verified'] = True
            return command.copy(), report
        except BaseException as exc:
            self.error = (type(exc).__name__+': '+str(exc))[:512]
            raise
        finally:
            self._busy = False
