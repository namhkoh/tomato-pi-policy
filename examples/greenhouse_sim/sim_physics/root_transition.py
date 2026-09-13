"""Isolated, no-step fixed-root release with native state continuity checks.

No reset, pose/velocity setter, grasp weld, drive rewrite or implicit warmup.
This is topology bookkeeping, NOT tissue fracture or a cutting decision.
Only PlantRig's validated blade-evidence path may invoke release(). A partial
failure is latched and must stop the caller before another physics step.
"""
import numpy as np


_FIELDS={
    'q':'get_dof_positions', 'v':'get_dof_velocities',
    'poses':'get_link_transforms', 'velocities':'get_link_velocities',
    'masses':'get_masses', 'inertias':'get_inertias', 'coms':'get_coms',
    'stiffness':'get_dof_stiffnesses', 'damping':'get_dof_dampings',
    'targets':'get_dof_position_targets', 'velocity_targets':'get_dof_velocity_targets',
    'efforts':'get_dof_actuation_forces',
    'force_limits':'get_dof_max_forces'}
_STATE=('q','v','poses','velocities')


def snapshot(view):
    if view.count!=1 or view.shared_metatype is None:
        raise RuntimeError('One live articulation required for topology continuity')
    value=dict(fixed_base=bool(view.shared_metatype.fixed_base),
        names=tuple(view.shared_metatype.dof_names),links=tuple(view.link_paths[0]))
    if (not value['names'] or len(set(value['names']))!=len(value['names'])
            or not value['links'] or len(set(value['links']))!=len(value['links'])):
        raise RuntimeError('Unique native coordinate and link inventory required')
    for key,method in _FIELDS.items():
        array=np.array(getattr(view,method)(),dtype=float,copy=True)
        if not array.size or not np.isfinite(array).all():
            raise RuntimeError('Nonfinite or missing native transition '+key)
        value[key]=array
    n=len(value['names']);links=len(value['links'])
    if (value['q'].shape!=(1,n) or value['v'].shape!=(1,n)
            or value['poses'].shape!=(1,links,7) or value['velocities'].shape!=(1,links,6)):
        raise RuntimeError('Native transition dimensions disagree with inventory')
    return value


def continuity(before,after,*,release):
    if type(release) is not bool:
        raise ValueError('Explicit topology transition type required')
    if (before['names']!=after['names'] or before['links']!=after['links']
            or not before['fixed_base'] or after['fixed_base'] is release):
        raise RuntimeError('Unexpected root topology or coordinate inventory')
    jumps={}
    for key in _FIELDS:
        a,b=before[key],after[key]
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError('Changed native transition shape/finite '+key)
        delta=float(np.max(np.abs(a-b)))
        # State envelope only for float32 readback; command/material must be exact.
        if delta>(1e-7 if key in _STATE else 0.):
            raise RuntimeError('Native state/command/material changed across no-step transition: '+key)
        jumps[key]=delta
    return jumps


def verify_free_mass(view,coordinate_count):
    if view.shared_metatype.fixed_base:
        raise RuntimeError('Stale fixed-base metadata; free-root spring solve refused')
    matrix=np.array(view.get_generalized_mass_matrices(),dtype=float,copy=True)
    n=coordinate_count+6
    if matrix.shape!=(1,n,n) or not np.isfinite(matrix).all():
        raise RuntimeError('Free-root mass matrix must include six root coordinates')
    m=matrix[0]
    if np.max(np.abs(m-m.T))>1e-5*np.max(np.abs(m))+1e-15:
        raise RuntimeError('Asymmetric free-root mass matrix')
    np.linalg.cholesky((m+m.T)/2)
    return list(matrix.shape)


def jacobian_velocity_audit(view,state):
    """Report both sides of the topology edit; never repair native velocity."""
    skip=int(state['fixed_base']);n=len(state['names'])+(0 if skip else 6)
    j=np.array(view.get_jacobians(),dtype=float,copy=True)
    shape=(1,len(state['links'])-skip,6,n)
    if j.shape!=shape or not np.isfinite(j).all():
        raise RuntimeError('Exact finite Jacobian topology required')
    v=state['v'][0] if skip else np.r_[state['velocities'][0,0],state['v'][0]]
    error=np.einsum('bcj,j->bc',j[0],v)-state['velocities'][0,skip:]
    return j[0],dict(max_linear_velocity_error_m_s=float(np.max(abs(error[:,:3]))),
        max_angular_velocity_error_rad_s=float(np.max(abs(error[:,3:]))),
        native_velocity_replaced=False,physical_consistency_certified=False)


class FixedRootTransition:
    """Single-use diagnostic adapter. Failures never rewind or resume physics."""
    def __init__(self,runtime,springs,fixture):
        self.runtime=runtime;self.springs=springs;self.fixture=fixture
        self.rig=runtime.rig;self.attempted=False;self.receipt=None;self.error=None
        if (self.rig.constraint_mode!='fixed_articulation' or self.rig.cut
                or not runtime.articulation.shared_metatype.fixed_base
                or springs.articulation is not runtime.articulation or not springs.fixed_base
                or not fixture.sparse_contacts or fixture.rig is not self.rig):
            raise ValueError('Exact isolated fixed-root implicit full-robot binding required')

    def release(self):
        if self.attempted or self.rig.cut:
            raise RuntimeError('Fixed-root transition cannot be retried or replayed')
        self.attempted=True
        try:
            from pxr import Usd,UsdPhysics
            import omni.usd
            import omni.physics.tensors as tensors
            from omni.physx import get_physx_simulation_interface
            r=self.runtime;s=self.springs;f=self.fixture
            if s.articulation is not r.articulation or not s.fixed_base:
                raise RuntimeError('Spring binding changed before release')
            plant=snapshot(r.articulation);robot=snapshot(f.robot)
            fixed_jac,fixed_jac_audit=jacobian_velocity_audit(r.articulation,plant)
            body_paths=tuple(r.bodies.prim_paths)
            bodies=np.array(r.bodies.get_transforms(),copy=True)
            velocities=np.array(r.bodies.get_velocities(),copy=True)
            commands=np.array(f.targets,copy=True)
            if (not np.array_equal(robot['targets'],commands)
                    or np.any(plant['stiffness']) or np.any(plant['damping'])
                    or s.k.shape!=(len(plant['names']),) or s.c.shape!=s.k.shape):
                raise RuntimeError('Cached physical springs and live unchanged robot targets required')
            context=omni.usd.get_context()
            if context.get_stage()!=self.rig.stage or context.get_stage_id()<=0:
                raise RuntimeError('Explicit current native stage identity required')
            anchor=UsdPhysics.Joint.Get(self.rig.stage,self.rig.cut_joint_path)
            if anchor.GetJointEnabledAttr().Get() is not True:
                raise RuntimeError('Attached seam required before transition')
            with Usd.EditContext(self.rig.stage,self.rig.stage.GetSessionLayer()):
                anchor.GetJointEnabledAttr().Set(False)
            if anchor.GetJointEnabledAttr().Get() is not False:
                raise RuntimeError('Composed release opinion not applied')
            self.rig.cut=True  # Even a later failure must not report an intact seam.
            get_physx_simulation_interface().flush_changes()
            fresh=tensors.create_simulation_view('numpy',stage_id=context.get_stage_id())
            fresh.set_subspace_roots('/')
            plant_view=fresh.create_articulation_view(self.rig.cut_joint_path)
            robot_view=fresh.create_articulation_view(f.anchor)
            free_state=snapshot(plant_view)
            plant_jumps=continuity(plant,free_state,release=True)
            free_jac,free_jac_audit=jacobian_velocity_audit(plant_view,free_state)
            robot_jumps=continuity(robot,snapshot(robot_view),release=False)
            old_robot_jumps=continuity(robot,snapshot(f.robot),release=False)
            shape=verify_free_mass(plant_view,len(plant['names']))
            # Keep native contact buffers and callback history. No global reset.
            # Existing body and robot views must independently remain readable.
            if (tuple(r.bodies.prim_paths)!=body_paths
                    or not np.array_equal(r.bodies.get_transforms(),bodies)
                    or not np.array_equal(r.bodies.get_velocities(),velocities)
                    or not np.array_equal(f.targets,commands)):
                raise RuntimeError('Retained native body/robot view continuity failed')
            r.simulation=fresh;r.articulation=plant_view
            s.articulation=plant_view;s.fixed_base=False
            self.receipt=dict(model='fixed_root_no_step_tensor_refresh_v1',
                plant_max_abs_changes=plant_jumps,robot_max_abs_changes=robot_jumps,
                retained_robot_view_max_abs_changes=old_robot_jumps,free_mass_matrix_shape=shape,
                stage_id=context.get_stage_id(),state_setters_used=False,global_tensor_reset=False,
                simulation_steps_during_transition=0,drive_reinitialization=False,
                physical_spring_cache_preserved=True,controller_contact_history_preserved=True,
                native_manifold_cache_preservation_verified=False,
                fixed_before_jacobian_velocity_audit=fixed_jac_audit,
                free_after_jacobian_velocity_audit=free_jac_audit,
                intrinsic_jacobian_max_no_step_change=float(np.max(abs(fixed_jac-free_jac[1:,:,6:]))),
                next_step_contact_reconciliation_required=True,training_eligible=False,
                tissue_fracture_calibrated=False)
            return dict(self.receipt)
        except Exception as exc:
            self.error=type(exc).__name__+': '+str(exc)
            raise
