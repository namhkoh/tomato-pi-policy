"""Guarded spring-effort submission for the diagnostic coupon ONLY.

Native contacts/friction remain active. The predictor's contact response is
never submitted. No root wrench, state setters, constraint or force clipping.
This is a numerical experiment, not a qualified greenhouse spring backend.
"""
import numpy as np

from .contact_coupled_prediction import MaterialLaw, compile_geometry, solve


def _array(value, shape, name):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite '+name+' with shape '+str(shape)+' required')
    return a


class ContactCoupledCouponSpring:
    """Construct through coupon bind after original native coefficient checks."""

    def __init__(self, articulation, coupon, *, law, patch_friction=False):
        if not isinstance(law, MaterialLaw):
            raise ValueError('Explicit diagnostic material law required')
        if type(patch_friction) is not bool or (patch_friction and law.response != 'unilateral_kv_v1'):
            raise ValueError('Explicit boolean patch friction requires unilateral_kv_v1')
        self.a = articulation; self.coupon = coupon; self.law = law
        self.patch_friction = patch_friction; self.anchor_binding = None
        self.predicted_seed = None
        self.indices = np.array([0], dtype=np.uint32)
        self.paths = list(articulation.link_paths[0])
        self.names = list(articulation.shared_metatype.dof_names)
        self.k = _array(articulation.get_dof_stiffnesses(), (1,2), 'original K')[0]
        self.c = _array(articulation.get_dof_dampings(), (1,2), 'original C')[0]
        self.caps = _array(articulation.get_dof_max_forces(), (1,2), 'native caps')[0]
        self.last_step = 0
        if (articulation.count != 1 or articulation.shared_metatype.fixed_base
                or np.any(self.k <= 0) or np.any(self.c <= 0) or np.any(self.caps <= 0)
                or np.any(_array(articulation.get_drive_types(), (1,2), 'drive types') != 1)
                or np.any(_array(articulation.get_dof_velocity_targets(), (1,2), 'targets') != 0)):
            raise ValueError('Floating Force drives with positive original K/C/caps required')
        zeros = np.zeros((1,2), dtype=np.float32)
        articulation.set_dof_stiffnesses(zeros, self.indices)
        articulation.set_dof_dampings(zeros, self.indices)
        self._check()

    def _check(self):
        a = self.a
        if (a.count != 1 or a.shared_metatype.fixed_base
                or list(a.link_paths[0]) != self.paths
                or list(a.shared_metatype.dof_names) != self.names
                or np.any(_array(a.get_dof_stiffnesses(), (1,2), 'K') != 0)
                or np.any(_array(a.get_dof_dampings(), (1,2), 'C') != 0)
                or not np.array_equal(_array(a.get_dof_max_forces(), (1,2), 'caps')[0], self.caps)
                or np.any(_array(a.get_drive_types(), (1,2), 'drive types') != 1)
                or np.any(_array(a.get_dof_velocity_targets(), (1,2), 'targets') != 0)):
            raise ValueError('Contact-coupled native contract changed')

    def step(self, dt, *, state, frames, velocities, native_rows,
             reference_frames, step_id, reference_step_id):
        if (isinstance(dt, (bool,np.bool_)) or not np.isscalar(dt)
                or not np.isfinite(dt) or abs(dt-self.coupon.dt) > 1e-12
                or isinstance(step_id, (bool,np.bool_)) or not isinstance(step_id, (int,np.integer))
                or step_id != self.last_step+1):
            raise ValueError('Exact timestep and contiguous pre-step evidence required')
        self._check()
        q = _array(self.a.get_dof_positions(), (1,2), 'current q')[0]
        if np.any(abs(q) >= .05):
            raise ValueError('Coupon small-angle guard; no clipping')
        native_v = np.r_[_array(self.a.get_root_velocities(), (1,6), 'root velocity')[0],
                         _array(self.a.get_dof_velocities(), (1,2), 'joint velocity')[0]]
        if (state.get('native_coordinate_convention_verified') is not True
                or state.get('contact_force_included') is not False
                or not np.array_equal(_array(state['generalized_velocity'], (8,), 'snapshot velocity'), native_v)):
            raise ValueError('Current verified native snapshot required')
        compiler = compile_geometry; prediction_solver = solve; geometry_options = {}; solve_options = {}
        if self.patch_friction:
            from .contact_patch_prediction import compile_patches, solve as patch_solve
            compiler = compile_patches; prediction_solver = patch_solve
            geometry_options['anchor_binding'] = self.anchor_binding
        J,g,s,geometry = compiler(self.coupon, frames, velocities,
            state['body_world_com_jacobians'], native_rows=native_rows,
            generalized_velocity=native_v, step_id=step_id,
            native_geometry_step_id=reference_step_id, native_geometry_frames=reference_frames,
            **geometry_options)
        if self.patch_friction:
            solve_options['patches'] = geometry['patches']
            solve_options['previous_predicted_seed'] = self.predicted_seed
        prediction = prediction_solver(state['mass_matrix'], np.r_[np.zeros(6),q], native_v,
            np.r_[np.zeros(6),self.k], np.r_[np.zeros(6),self.c], J,g,s,
            np.full(len(g),self.coupon.contact_stiffness), np.full(len(g),self.coupon.contact_damping),
            dt, state['native_known_external_force'], law=self.law,
            feature_observed=geometry['feature_observed'], **solve_options)
        if prediction['status'] != 'resolved':
            raise ValueError('Unresolved contact-coupled prediction: '+prediction['reason'])
        effort = _array(prediction['tau_joint'], (2,), 'predicted spring effort')
        if np.any(abs(effort) > self.caps):
            raise ValueError('Explicit spring effort cap exceeded; no clipping')
        command = effort.astype(np.float32)
        if not np.isfinite(command).all() or np.any(abs(command.astype(float)) > self.caps):
            raise ValueError('Float32 spring command outside bounds')
        self.a.set_dof_actuation_forces(command[None,:], self.indices)
        if self.patch_friction:
            self.anchor_binding = geometry['anchor_binding']
            self.predicted_seed = prediction['next_predicted_seed']
        self.last_step = step_id
        return command.astype(float), dict(geometry=geometry, prediction=prediction,
            submitted_spring_effort_nm=command.tolist(), effort_is_measured=False,
            contact_effort_submitted=False, root_actuated=False)

    def report(self):
        return dict(original_native_k=self.k.tolist(), original_native_c=self.c.tolist(),
            native_drive_caps=self.caps.tolist(), native_angular_K_C_zero_verified=True,
            material_law=self.law.response, native_contact_law_parity=False,
            contact_force_applied=False, root_actuated=False, friction_in_prediction=self.patch_friction,
            measured_friction_used=False, friction_force_applied=False,
            friction_model='two_anchor_circular_coulomb_v1' if self.patch_friction else None,
            predictor_seed='previous_resolved_prediction_only' if self.patch_friction else None,
            native_qualified=False, scope='three-link planar coupon only')
