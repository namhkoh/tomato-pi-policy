"""Opt-in measured withdrawal orchestration for the bounded robot diagnostic.

Only right joint DRIVE TARGETS are submitted, after a fresh validated fetched
sample. No body/joint positions or velocities are overwritten. The harness
owns all native force, grasp, slip and contact checks before observe().
"""
from copy import deepcopy

import numpy as np

from .measured_withdrawal import MeasuredWithdrawal, _hash
from .measured_withdrawal_native import NativeWithdrawalAdapter, RIGHT


def native_command_packet(measured_deg,desired_deg):
    """Float32 radians, rounded toward measured state if nearest overshoots.

At most one ULP adjustment, never a joint-limit clamp. The actual resulting
packet must pass the helper's original limits/rates/geometry before submission.
"""
    q=np.asarray(measured_deg,dtype=float);desired=np.asarray(desired_deg,dtype=float)
    if q.shape!=(7,) or desired.shape!=(7,) or not np.isfinite([q,desired]).all():
        raise ValueError('Finite measured and desired seven-joint vectors required')
    origin=np.radians(q).astype(np.float32)
    if not np.array_equal(np.degrees(origin.astype(float)),q):
        raise ValueError('Measured joints must retain their exact native float32 provenance')
    packet=np.radians(desired).astype(np.float32)
    decoded=np.degrees(packet.astype(float))
    too_far=(decoded-desired)*(desired-q)>0
    packet[too_far]=np.nextafter(packet[too_far],origin[too_far])
    return np.degrees(packet.astype(float))


def measured_completion(record,sample_id):
    evidence=record.get('measured_withdrawal',{});receipt=evidence.get('receipt') or {}
    return bool(evidence.get('status')=='complete'
        and evidence.get('right_withdrawal_completed') is True
        and evidence.get('sample_id')==list(sample_id)
        and receipt.get('passed') is True and receipt.get('sample_id')==list(sample_id))


def reference_rates(plan, cut_fraction, native_limits_rad_s):
    """Existing reverse-stroke(2s)/approach(4s) smoothstep speed envelope.

Constant reference joints may correct measured tracking error, bounded by the
largest ORIGINAL path reference rate and their native velocity limit. This is
a diagnostic controller rule, not a measured hardware speed or tuned gain.
"""
    if isinstance(cut_fraction,(bool,np.bool_)) or not np.isfinite(cut_fraction) or not 0<=cut_fraction<=1:
        raise ValueError('Actual sent release fraction required for rate envelope only')
    native=np.array(native_limits_rad_s,dtype=float,copy=True)
    if native.shape!=(7,) or not np.isfinite(native).all() or np.any(native<=0):
        raise ValueError('Positive finite native right-joint velocity limits required')
    rates=[]
    for phase,duration,fraction in (('stroke',2.,cut_fraction),('approach',4.,1.)):
        path=np.asarray(plan[phase],dtype=float)
        if path.ndim!=2 or path.shape[1]!=7 or len(path)<2 or not np.isfinite(path).all():
            raise ValueError('Finite existing sampled right path required')
        rates.append(np.max(abs(np.diff(path,axis=0)),axis=0)*(len(path)-1)*1.5*fraction/duration)
    rate=np.maximum(*rates); envelope=float(np.max(rate))
    if envelope<=0:raise ValueError('Existing withdrawal reference has no motion')
    result=np.minimum(np.where(rate>0,rate,envelope),np.degrees(native))
    if not np.isfinite(result).all() or np.any(result<=0):raise ValueError('Invalid rate envelope')
    return result


class WithdrawalController:
    def __init__(self, fixture, runtime, clock, *, cut_fraction):
        self.f=fixture;self.clock=clock;self.cut_fraction=cut_fraction
        self.adapter=NativeWithdrawalAdapter(fixture,runtime.bodies,
            lambda:(clock.stamp.episode,clock.stamp.step))
        self.controller=None;self.pending=None;self.last_command_step=None
        self._pending_hash=None;self._blocked=None
        self.hold=dict(left_goal_world=np.asarray(fixture.expected_palm).tolist(),
            finger_targets_m=fixture.targets[0,fixture.finger_indices].tolist())

    def observe(self,stamp,record):
        """Call only after ALL original post-fetch guards and inspect_cut pass."""
        if self._blocked:raise RuntimeError(self._blocked)
        sample=(stamp.episode,stamp.step)
        if (self.pending is not None or record.get('native_guards_passed') is not True
                or record.get('cut') is not True or not isinstance(record.get('robot'),dict)):
            raise RuntimeError('Fresh guarded cut record required, with prior command consumed')
        m=self.adapter.snapshot(sample,hold_targets=self.hold,
            guard_evidence=dict(sample_id=list(sample),passed=True,errors=[]),
            grasp_evidence=dict(sample_id=list(sample),result=record['contact']))
        if self.controller is None:
            f=self.f
            if f.plan is None:raise RuntimeError('Existing cut/approach plan required')
            low,high=f.kin.arm_limits_degrees('right')
            native=np.asarray(f.robot.get_dof_max_velocities(),dtype=float)
            if native.shape!=(1,len(f.names)):raise ValueError('Exact native velocity-limit inventory required')
            rate=reference_rates(f.plan,self.cut_fraction,native[0,f.right_indices])
            self.controller=MeasuredWithdrawal(f.plan['stroke'],f.plan['approach'],
                measurement=m,lower_deg=low,upper_deg=high,max_speed_deg_s=rate,
                dt=self.clock.dt,wrist_fk=self.adapter.wrist_fk,edge_in_wrist=f.knife.frame(np.eye(4)),
                stroke_direction=f.plan['direction'],validate=self.adapter.validate,
                quantize_command=native_command_packet)
            result=self.controller.initial_command()
            result['reference_rate_deg_s']=rate.tolist()
        else:
            result=self.controller.advance(m)
        record['measured_withdrawal']=deepcopy(result)
        record['withdrawal_native_snapshot']=m
        if result.get('status') not in ('ready','moving','complete'):
            raise RuntimeError('Measured withdrawal blocked: '+str(result.get('reason')))
        self.pending=deepcopy(result)
        self._pending_hash=_hash(self.pending)

    def command(self,stamp):
        """Consume exactly one same-sample decision before the next physics step."""
        if self._blocked:raise RuntimeError(self._blocked)
        try:return self._command(stamp)
        except BaseException as exc:
            self._blocked=type(exc).__name__+': '+str(exc)
            raise

    def _command(self,stamp):
        result=self.pending;sample=(stamp.episode,stamp.step)
        if (result is None or tuple(result.get('sample_id',()))!=sample
                or self.last_command_step==sample or result.get('cut_authorized') is not False):
            raise RuntimeError('Missing, reused or stale measured-withdrawal decision')
        try:unchanged=_hash(result)==self._pending_hash
        except Exception as exc:raise RuntimeError('Invalid withdrawal target decision') from exc
        if not unchanged:raise RuntimeError('Withdrawal target decision mutated after validation')
        # Re-read actual state/targets; buffer reset is not a physics advance.
        self.adapter._fresh()
        f=self.f
        if tuple(f.names[i] for i in f.right_indices)!=RIGHT:
            raise RuntimeError('Right command index inventory changed')
        before=f.targets.copy()
        q=result.get('target_right_q_deg')
        if q is not None:
            q=np.asarray(q,dtype=float);low,high=f.kin.arm_limits_degrees('right')
            if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<=low) or np.any(q>=high):
                raise RuntimeError('Invalid withdrawal target; no clipping')
            proposed=before.copy();proposed[0,f.right_indices]=np.radians(q)
            sent=np.degrees(proposed[0,f.right_indices].astype(float))
            if not np.isfinite(sent).all() or np.any(sent<=low) or np.any(sent>=high):
                raise RuntimeError('Native command precision violates joint limits')
            if not np.array_equal(sent,q):
                raise RuntimeError('Submitted native packet differs from validated path target')
            expected=self.adapter.wrist_fk(sent)
            f.robot.set_dof_position_targets(proposed,f.index)
            actual=np.asarray(f.robot.get_dof_position_targets())
            if actual.shape!=proposed.shape or not np.array_equal(actual,proposed):
                raise RuntimeError('Native right-drive target readback mismatch')
            f.targets=proposed;f.expected_right=expected
        elif result['status']!='complete':
            raise RuntimeError('Absent target without measured completion')
        f.cut_authorized=False
        self.last_command_step=sample;self.pending=None
        self._pending_hash=None
        return dict(phase='measured_withdrawal',status=result['status'],
            right_target_degrees=None if q is None else q.tolist(),sample_id=list(sample))
