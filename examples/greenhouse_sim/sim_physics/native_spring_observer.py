"""Read-only native drive contract for isolated comparisons.

Unlike ImplicitJointSprings this does NOT disable drives or submit torques.
PhysX owns the spring/contact solve. A returned None is explicitly NOT a
measured drive torque or a zero-work claim. Explicit release-mode observation
does not authorize release or establish cutting/retention qualification.
"""
import numpy as np


def restore_after_release(springs):
    """Opt-in diagnostic handoff, no topology edit or motion/state override.

    Restore the cached original physical K/C, not tuned gains. The external
    implicit effort is explicitly removed to avoid double spring actuation.
    This changes the numerical integrator; it is not yet native qualification.
    Any failure after a setter requires the caller to stop without stepping.
    """
    from .implicit_springs import ImplicitJointSprings
    from .root_transition import snapshot, _FIELDS
    if type(springs) is not ImplicitJointSprings or springs.fixed_base:
        raise ValueError('Original implicit springs with checked free-root release required')
    a=springs.articulation;before=snapshot(a);n=len(before['names'])
    k=np.array(springs.k,dtype=float,copy=True);c=np.array(springs.c,dtype=float,copy=True)
    if (before['fixed_base'] or k.shape!=(n,) or c.shape!=(n,)
            or not np.isfinite(np.r_[k,c]).all() or np.any(k<=0) or np.any(c<0)
            or any(np.any(before[name]) for name in ('stiffness','damping','targets','velocity_targets'))
            or np.any(np.asarray(a.get_drive_types())!=1)):
        raise ValueError('Exact original physical spring cache, zero native gains/rest targets and force drives required')
    index=np.array([0],dtype=np.uint32)
    a.set_dof_actuation_forces(np.zeros((1,n),np.float32),index)
    a.set_dof_stiffnesses(k[None].astype(np.float32),index)
    a.set_dof_dampings(c[None].astype(np.float32),index)
    after=snapshot(a)
    if (before['names']!=after['names'] or before['links']!=after['links'] or after['fixed_base']
            or any(not np.array_equal(before[key],after[key])
                for key in _FIELDS if key not in ('stiffness','damping','efforts'))
            or not np.array_equal(after['stiffness'][0],k)
            or not np.array_equal(after['damping'][0],c) or np.any(after['efforts'])):
        raise RuntimeError('Native drive handoff changed state/material or failed exact K/C/effort readback')
    result=NativeSpringObserver(a,allow_release=True)
    result.handoff_receipt=dict(model='original_native_springs_after_checked_release_v1',
        original_k_c_restored=True,explicit_spring_effort_removed=True,
        mass_inertia_poses_velocities_targets_unchanged=True,
        numerical_integrator_changed=True,native_contact_unchanged=True,
        state_setters_used=False,physics_steps_during_handoff=0,
        native_drive_effort_measured=False,physical_model_qualified=False,training_eligible=False)
    return result


class NativeSpringObserver:
    def __init__(self,articulation,*,allow_release=False):
        if type(allow_release) is not bool:raise ValueError('Explicit native release comparison flag required')
        self.allow_release=allow_release
        if articulation.count!=1:raise ValueError('Exactly one native plant articulation required')
        self.articulation=articulation
        self.names=tuple(articulation.shared_metatype.dof_names)
        self.k=np.asarray(articulation.get_dof_stiffnesses(),float)[0].copy()
        self.c=np.asarray(articulation.get_dof_dampings(),float)[0].copy()
        if (not self.names or len(set(self.names))!=len(self.names)
                or self.k.shape!=(len(self.names),) or self.c.shape!=self.k.shape
                or not np.isfinite(np.r_[self.k,self.c]).all()
                or np.any(self.k<=0) or np.any(self.c<0)):
            raise ValueError('Finite positive native stiffness and nonnegative damping required')
        self.step(1/240,root_constrained=True)

    def step(self,dt,*,root_constrained):
        if (isinstance(dt,(bool,np.bool_)) or not np.isfinite(dt) or abs(dt-1/240)>1e-12
                or type(root_constrained) is not bool or not root_constrained and not self.allow_release):
            raise ValueError('Native drive comparison requires 240 Hz and explicit release diagnostic')
        a=self.articulation;n=len(self.names)
        if tuple(a.shared_metatype.dof_names)!=self.names:raise RuntimeError('Native joint inventory changed')
        for method,expected in (('get_dof_stiffnesses',self.k),('get_dof_dampings',self.c),
                ('get_dof_position_targets',np.zeros(n)),('get_dof_velocity_targets',np.zeros(n)),
                ('get_dof_actuation_forces',np.zeros(n))):
            value=np.asarray(getattr(a,method)(),float)
            if value.shape!=(1,n) or not np.array_equal(value[0],expected):
                raise RuntimeError('Native drive comparison contract changed: '+method)
        return None
