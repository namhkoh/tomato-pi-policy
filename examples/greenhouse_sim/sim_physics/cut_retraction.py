"""Contact-paced reverse of the already screened cut stroke, after severance.

The old two-second reversal could drag the retained shaft while the blade
still supported/touched it. Use the SAME 0.3 mm/s contact rate as insertion;
only after measured unloading may the existing approach path be reversed.
No grasp reset, pose setter, force-limit increase or release authority.
"""
import math
import numpy as np


class CutRetraction:
    def __init__(self,offsets,*,fraction,endpoint,direction,step,section):
        a=np.asarray(offsets,float);p=np.asarray(endpoint,float);d=np.asarray(direction,float)
        if (a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or not np.all(np.diff(a)>0)
                or not -.025<=a[0]<0<a[-1]<=.02 or isinstance(fraction,bool)
                or not np.isfinite(fraction) or not 0<fraction<1
                or p.shape!=(3,) or d.shape!=(3,) or not np.isfinite([p,d]).all()
                or abs(np.linalg.norm(d)-1)>1e-5 or d[2]>-np.cos(np.radians(5))
                or type(step) is not int or step<1):
            raise ValueError('Existing downward stroke and actual submitted fraction required')
        self.start=float(a[0]);self.span=float(a[-1]-a[0]);self.offset=self.start+fraction*self.span
        self.endpoint=p.copy();self.direction=d.copy();self.section=dict(section)
        self.first=step;self.step=step-1;self.consumed=None;self.complete=False
        self.dwell=0;self.receipt=None;self.upper=0.;self.allowed=False;self.origin=None

    def observe(self,record,*,step,support_ready):
        from .cut_section import cut_face_contact
        from .knife import KNIFE_IMPULSE_CONTRACT
        if (type(step) is not int or step!=self.step+1 or type(support_ready) is not bool
                or step>self.first and self.consumed!=self.step
                or record.get('cut') is not True or record.get('native_guards_passed') is not True):
            raise RuntimeError('Fresh guarded post-severance support sample required')
        k=record['knife'];edge=np.asarray(k['edge_frame'],float);upper=k['tool_contact_upper_bound_n']
        if (k.get('force_contract')!=KNIFE_IMPULSE_CONTRACT or edge.shape!=(4,4)
                or not np.isfinite(edge).all() or not np.allclose(edge[3],[0,0,0,1],rtol=0,atol=1e-8)
                or not np.allclose(edge[:3,:3].T@edge[:3,:3],np.eye(3),rtol=0,atol=1e-5)
                or np.linalg.det(edge[:3,:3])<=0 or edge[2,0]<np.cos(np.radians(5))
                or isinstance(upper,bool) or not math.isfinite(upper) or not 0<=upper<=.5):
            raise RuntimeError('Complete finite current knife frame and bounded full load required')
        delta=edge[:3,3]-self.endpoint;along=float(delta@self.direction)
        lateral=float(np.linalg.norm(delta-along*self.direction))
        if lateral>.003 or along<-.0005:raise RuntimeError('Measured reverse stroke left its original corridor')
        if self.origin is None:
            if along<=.0003:raise RuntimeError('Retraction must start at the measured completed cut, not the park endpoint')
            self.origin=edge[:3,3].copy()
        reverse=float((self.origin-edge[:3,3])@self.direction)
        self.allowed=cut_face_contact(record,knife=self.section['knife'],faces=self.section['faces']) and support_ready
        at_end=abs(along)<=.00025 and upper<.01 and self.allowed and self.offset==self.start and reverse>.0003
        self.dwell=self.dwell+1 if at_end else 0;self.complete=self.dwell>=12
        self.step=step;self.upper=float(upper)
        self.receipt=dict(model='contact_paced_cut_retraction_v1',step=step,complete=self.complete,
            remaining_m=along,lateral_error_m=lateral,full_knife_load_n=upper,
            net_measured_reverse_m=reverse,
            cut_face_and_support_verified=self.allowed,unloaded_endpoint_dwell_s=self.dwell/480,
            cut_authorized=False,commanded_motion_used_as_completion=False,force_limits_changed=False)
        return dict(self.receipt)

    def command(self,*,step,dt):
        if step!=self.step or self.consumed==step or isinstance(dt,bool) or not math.isfinite(dt) or abs(dt-1/480)>1e-12:
            raise RuntimeError('Fresh unused retraction sample required')
        if not self.complete and (step-self.first)/480>=35:raise RuntimeError('Contact-paced retraction timed out')
        if self.complete:speed=0.;state='complete'
        elif not self.allowed or self.upper>.40:speed=0.;state='hold_contact_or_support'
        elif self.upper>.01:speed=.0003;state='loaded_reverse'
        else:speed=.002;state='unloaded_reverse'
        self.offset=max(self.start,self.offset-speed*dt);self.consumed=step
        self.receipt.update(command_state=state,command_speed_m_s=speed,command_offset_m=self.offset)
        return (self.offset-self.start)/self.span
