"""Bounded measured preload dwell before the independent retention audit.

No motion, effort, material, cut or grasp setter. A pass is not retention.
"""
import numpy as np


class PreloadSettle:
    def __init__(self):
        self.last_step=None;self.start_time=None;self.stable_steps=0

    def observe(self,record,*,step,time_s,start_time_s):
        d=record.get('grasp_dynamics',{});c=record.get('contact',{});f=record.get('force_closure',{})
        if (type(step) is not int or step<1 or not np.isfinite([time_s,start_time_s]).all()
                or not 3.5<=start_time_s<=13.5 or not start_time_s<time_s
                or self.last_step is not None and step!=self.last_step+1
                or self.start_time is not None and start_time_s!=self.start_time
                or record.get('t')!=time_s or abs(time_s-step/240)>1e-10
                or d.get('step_id')!=step or d.get('dt_s')!=1/240
                or d.get('model')!='grasp_dynamics_post_fetch_telemetry_v1' or c.get('step_id')!=step
                or record.get('native_guards_passed') is not True or record.get('cut') is not False
                or c.get('adapter_valid') is not True
                or f.get('load_profile')!='retention_preload_v1'
                or f.get('desired_support_n')!=.24 or f.get('symmetric_aperture_command') is not True):
            raise ValueError('Fresh original-profile attached native preload observation required')
        support=np.array(c.get('compressive_support_n'),float)
        velocity=np.array(d.get('finger_velocities_m_s'),float)
        slip=record.get('slip_m')
        if (support.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite([support,velocity]).all()
                or type(slip) not in (float,int) or not np.isfinite(slip) or not 0<=slip<.003):
            raise ValueError('Finite current support, finger velocity and bounded slip required')
        stable=(c.get('bilateral') is True and c.get('stem_only') is True
            and np.all(support>=.02) and abs(float(np.mean(support))-.24)<=.03+1e-12
            and np.all(abs(velocity)<=.002))
        self.stable_steps=self.stable_steps+1 if stable else 0
        self.last_step=step;self.start_time=start_time_s
        ready=self.stable_steps>=48
        # Use the existing feedback acquisition deadline, not an arbitrary
        # shorter timer that expires while the unchanged slow preload grows.
        deadline=13.5
        return dict(model='measured_original_preload_settle_v1',step_id=step,time_s=float(time_s),
            state='ready' if ready else 'timeout' if time_s>=deadline-1e-10 else 'waiting',
            consecutive_stable_steps=self.stable_steps,required_steps=48,required_dwell_s=.2,
            desired_support_n=.24,support_tolerance_n=.03,measured_support_n=support.tolist(),
            regulated_quantity='mean_support_original_symmetric_controller',
            finger_velocity_limit_m_s=.002,measured_finger_velocities_m_s=velocity.tolist(),
            deadline_s=deadline,retention_verified=False,cut_authorized=False,
            force_limits_changed=False)
