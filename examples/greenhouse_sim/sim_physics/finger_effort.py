"""Isolated explicit left-finger PD comparison, not a grasp sensor.

Uses the same 200 N/m, 5 N s/m controller and bounded drive/gravity budget.
Only two native finger drives are disabled; physical contacts remain native.
Submitted efforts are logged separately from unknown actual internal forces.
Cut diagnostics require the separate complete isolated fixture CLI opt-in.
"""
import numpy as np


def command(position,velocity,target,gravity,caps,*,dt):
    q,v,g,ff,limit=[np.asarray(x,float) for x in (position,velocity,target,gravity,caps)]
    if (any(x.shape!=(2,) or not np.isfinite(x).all() for x in (q,v,g,ff,limit))
            or isinstance(dt,(bool,np.bool_)) or not np.isfinite(dt) or abs(dt-1/240)>1e-12
            or np.any(limit<=0) or np.any(limit>.15+1e-8) or np.any(abs(ff)>.4)
            or np.any(abs(g)>.025+1e-8) or g[0]>0 or g[1]<0):
        raise ValueError('Finite bounded two-finger 240 Hz effort command required')
    raw=200*(g-q)-5*v
    if not np.isfinite(raw).all():
        raise ValueError('Unrepresentable explicit PD request')
    pd=np.clip(raw,-limit,limit)
    total=ff+pd
    if not np.isfinite(total).all() or np.any(abs(total)>.8):
        raise ValueError('Explicit finger total budget exceeded')
    return total,dict(model='explicit_finger_PD_hold_v1',
        pre_positions_m=q.tolist(),pre_velocities_m_s=v.tolist(),targets_m=g.tolist(),
        gravity_feedforward_n=ff.tolist(),unclipped_pd_n=raw.tolist(),
        submitted_pd_n=pd.tolist(),submitted_total_n=total.tolist(),
        remaining_drive_caps_n=limit.tolist(),native_drive_gains_zero=True,
        measured_contact_used_as_effort=False,actual_drive_effort_measured=False,
        grasp_verified=False,cutting_qualified=False)


class FingerEffort:
    def __init__(self,fixture):
        a=fixture.robot
        self.names=tuple(a.shared_metatype.dof_names)
        self.indices=tuple(fixture.finger_indices)
        if (a.count!=1 or len(self.indices)!=2 or len(set(self.indices))!=2
                or tuple(self.names[i] for i in self.indices)!=('gripper_finger_l1','gripper_finger_l2')):
            raise ValueError('Exact left-finger native joint inventory required')
        k=np.array(a.get_dof_stiffnesses(),copy=True)
        c=np.array(a.get_dof_dampings(),copy=True)
        if (k.shape!=(1,len(self.names)) or c.shape!=k.shape
                or not np.array_equal(k[0,list(self.indices)],[200.,200.])
                or not np.array_equal(c[0,list(self.indices)],[5.,5.])):
            raise ValueError('Expected original left-finger PD gains before comparison')
        k[0,list(self.indices)]=0;c[0,list(self.indices)]=0
        a.set_dof_stiffnesses(k,fixture.index);a.set_dof_dampings(c,fixture.index)
        self.k=k;self.c=c;self.last_step=-1;self.receipt=None
        self._verify(a)

    def _verify(self,a):
        if (tuple(a.shared_metatype.dof_names)!=self.names
                or not np.array_equal(a.get_dof_stiffnesses(),self.k)
                or not np.array_equal(a.get_dof_dampings(),self.c)):
            raise RuntimeError('Finger effort native inventory/gains changed')

    def apply(self,fixture,*,step,dt):
        if type(step) is not int or step!=self.last_step+1:
            raise RuntimeError('Contiguous finger effort command required')
        a=fixture.robot;self._verify(a);idx=list(self.indices)
        before=np.array(a.get_dof_actuation_forces(),dtype=float,copy=True)
        targets=np.asarray(a.get_dof_position_targets(),float)
        if (before.shape!=(1,len(self.names)) or not np.isfinite(before).all()
                or targets.shape!=before.shape or not np.array_equal(targets,fixture.targets)
                or not np.allclose(before[0,idx],fixture.finger_compensation,atol=1e-8,rtol=0)):
            raise RuntimeError('Fresh gravity-only actuation and latest targets required')
        total,receipt=command(a.get_dof_positions()[0,idx],a.get_dof_velocities()[0,idx],
            targets[0,idx],before[0,idx],fixture.force_limits[0,idx],dt=dt)
        before[0,idx]=total
        submitted=before.astype(np.float32)
        a.set_dof_actuation_forces(submitted,fixture.index)
        if not np.array_equal(a.get_dof_actuation_forces(),submitted):
            raise RuntimeError('Native explicit effort submission readback mismatch')
        receipt.update(command_step=step,observed_pre_step=step,
            submitted_tensor_total_n=submitted[0,idx].astype(float).tolist(),
            submission_readback_verified=True)
        self.last_step=step;self.receipt=receipt
