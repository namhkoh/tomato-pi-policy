"""Diagnostic original native-C / explicit-K split after checked free release.

Keeps the original material constants. Native damping participates in native
contact solving; no damping torque is also added externally. The inequality
C >= h K/2 is only a constant-M consistent-implicit-C passivity condition,
NOT proof of the installed solver or a nonlinear beam/contact qualification.
No pose/root actuation, force replay, stiffness fitting or effort clipping.
"""
import numpy as np


class NativeDampedSprings:
    def __init__(self,springs,*,dt=1/240):
        a=springs.articulation
        self.articulation=a;self.fixed_base=bool(a.shared_metatype.fixed_base)
        self.names=tuple(a.shared_metatype.dof_names)
        self.k=np.array(springs.k,dtype=float,copy=True)
        self.c=np.array(springs.c,dtype=float,copy=True)
        self.indices=np.array([0],dtype=np.uint32);self.dt=dt
        self.caps=np.array(a.get_dof_max_forces(),dtype=float,copy=True)
        n=len(self.names)
        if (type(dt) is not float or dt!=1/240 or a.count!=1 or self.fixed_base
                or springs.fixed_base or n<1 or len(set(self.names))!=n
                or self.k.shape!=(n,) or self.c.shape!=(n,) or self.caps.shape!=(1,n)
                or not np.isfinite(np.r_[self.k,self.c,self.caps[0]]).all()
                or np.any(self.k<=0) or np.any(self.c<dt*self.k/2) or np.any(self.caps<=0)):
            raise ValueError('Original finite free-root K/C satisfying C >= h K/2 required')
        for method in ('get_dof_stiffnesses','get_dof_dampings','get_dof_velocity_targets','get_dof_position_targets'):
            value=np.asarray(getattr(a,method)())
            if value.shape!=(1,n) or np.any(value):
                raise ValueError('Original implicit zero-native-gain/rest-target state required')
        if np.any(np.asarray(a.get_drive_types())!=1):
            raise ValueError('Original force drive type required')
        q=np.asarray(a.get_dof_positions(),float)
        if q.shape!=(1,n) or not np.isfinite(q).all() or np.max(abs(q))>=.25:
            raise ValueError('Bounded pre-switch native spring coordinates required')
        a.set_dof_dampings(self.c[None].astype(np.float32),self.indices)
        self._check()
        self.receipt=dict(model='post_release_original_native_C_explicit_K_v1',
            native_damping_restored_to_original=True,native_stiffness_zero=True,
            minimum_c_minus_half_dt_k=float(np.min(self.c-dt*self.k/2)),
            native_damping_effort_measured=False,root_actuated=False,
            state_setters_used=False,contact_force_reapplied=False,
            diagnostic_maximum_abs_joint_angle_rad=.25,material_strain_calibrated=False,
            native_physics_qualified=False,training_eligible=False)

    def _check(self):
        a=self.articulation;n=len(self.names)
        if (a.count!=1 or a.shared_metatype.fixed_base or self.fixed_base
                or tuple(a.shared_metatype.dof_names)!=self.names):
            raise RuntimeError('Native damping split coordinate topology changed')
        for method,expected in (
            ('get_dof_stiffnesses',np.zeros((1,n))),('get_dof_dampings',self.c[None]),
            ('get_dof_max_forces',self.caps),('get_dof_position_targets',np.zeros((1,n))),
            ('get_dof_velocity_targets',np.zeros((1,n))),('get_drive_types',np.ones((1,n)))):
            if not np.array_equal(getattr(a,method)(),expected):
                raise RuntimeError('Native damping split material/drive contract changed: '+method)

    def step(self,dt,*,root_constrained=False):
        if root_constrained is not False or dt!=self.dt:
            raise ValueError('Fixed 240 Hz unconstrained root required')
        self._check()
        q=np.asarray(self.articulation.get_dof_positions(),dtype=float)
        if q.shape!=(1,len(self.k)) or not np.isfinite(q).all() or np.max(abs(q))>=.25:
            raise RuntimeError('Native damping split diagnostic angle bound exceeded')
        command=(-self.k*q[0]).astype(np.float32)[None]
        if not np.isfinite(command).all() or np.any(abs(command)>self.caps):
            raise RuntimeError('Native damping split original effort budget exceeded; no clipping')
        self.articulation.set_dof_actuation_forces(command,self.indices)
        if not np.array_equal(self.articulation.get_dof_actuation_forces(),command):
            raise RuntimeError('Native damping split effort readback mismatch')
        return command[0].astype(float)
