"""Read-only discrete spring-work accounting; no pass/fail or control authority.

The constant submitted generalized effort does work tau dot (q_after-q_before).
Add the change of the declared quadratic spring potential. A positive sum is
an apparent constitutive energy source, not proof of an accurate tissue model.
Native split integration, coordinate precision and unobserved effort limiting
remain limitations. This is NOT contact work or whole-system energy balance.
"""
import numpy as np


def _vector(value, name):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.ndim!=1 or not len(raw):
        raise ValueError('Numeric nonempty '+name+' required')
    value=np.array(raw,dtype=float,copy=True)
    if not np.isfinite(value).all():raise ValueError('Finite '+name+' required')
    return value


def account(before,after,stiffness,effort,*,before_step,after_step,dt):
    """Same episode caller asserted; exactly adjacent integer physics steps."""
    if (type(before_step) is not int or before_step<0 or type(after_step) is not int
            or after_step!=before_step+1 or isinstance(dt,(bool,np.bool_))
            or not np.isscalar(dt) or not np.isfinite(dt) or dt<=0):
        raise ValueError('Adjacent steps and positive finite dt required')
    q0,q1,k,tau=(_vector(x,n) for x,n in zip((before,after,stiffness,effort),
        ('pre-step q','post-step q','stiffness','submitted effort')))
    if any(x.shape!=q0.shape for x in (q1,k,tau)) or np.any(k<0):
        raise ValueError('Matching coordinates and nonnegative stiffness required')
    dq=q1-q0
    work=tau*dq
    # Algebraic difference avoids subtracting two large total energies.
    potential=.5*k*dq*(q1+q0)
    if not np.isfinite([work,potential,work+potential]).all():
        raise ValueError('Unrepresentable spring work')
    return dict(before_step=before_step,after_step=after_step,dt_s=float(dt),
        q_before_rad=q0.tolist(),q_after_rad=q1.tolist(),
        stiffness_nm_rad=k.tolist(),submitted_effort_nm=tau.tolist(),
        joint_displacement_rad=dq.tolist(),effort_work_j=float(np.sum(work)),
        quadratic_potential_change_j=float(np.sum(potential)),
        apparent_constitutive_energy_j=float(np.sum(work+potential)),
        per_joint_apparent_constitutive_energy_j=(work+potential).tolist(),
        effort_basis='caller_supplied_constant_generalized_effort_not_authenticated',
        native_integrator_work_verified=False,whole_system_energy_balance=False,
        tissue_calibrated=False,training_eligible=False)


def capture(articulation,command,*,step):
    """After effort submission, before physics; never calls a native setter."""
    names=list(articulation.shared_metatype.dof_names)
    q=np.array(articulation.get_dof_positions(),dtype=float,copy=True)
    effort=np.array(articulation.get_dof_actuation_forces(),copy=True)
    expected=np.asarray(command,dtype=np.float32)
    if (type(step) is not int or step<0 or not names or len(set(names))!=len(names)
            or q.shape!=(1,len(names)) or effort.shape!=q.shape or expected.shape!=(len(names),)
            or not np.isfinite(q).all() or not np.isfinite(effort).all()
            or not np.array_equal(effort[0],expected)):
        raise ValueError('Exact native coordinate/actuation readback required')
    return dict(step=step,names=names,q=q[0].tolist(),effort=effort[0].astype(float).tolist())


def finish(snapshot,articulation,stiffness,*,step,dt):
    if list(articulation.shared_metatype.dof_names)!=snapshot['names']:
        raise ValueError('Native spring coordinate inventory changed')
    q=np.array(articulation.get_dof_positions(),dtype=float,copy=True)
    if q.shape!=(1,len(snapshot['names'])):raise ValueError('Exact native coordinate count required')
    result=account(snapshot['q'],q[0],stiffness,snapshot['effort'],
        before_step=snapshot['step'],after_step=step,dt=dt)
    result['native_joint_names']=list(snapshot['names'])
    result['effort_basis']='constant_actuation_command_readback_not_measured_joint_torque'
    return result
