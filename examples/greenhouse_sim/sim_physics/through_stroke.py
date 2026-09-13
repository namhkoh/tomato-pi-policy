"""Measured post-release follow-through on the already screened knife path.

This does not model a kerf or authorize a release. Solid cut faces can still
block the blade: stop on contact/tracking limits, never disable collisions.
"""
import math
import numpy as np


class ThroughStroke:
    def __init__(self, offsets, *, fraction, endpoint, direction, physics_hz, step, material_section=None, maximum_feed_m_s=.0003):
        from .postrelease_feed import validate
        self.maximum_feed=validate(maximum_feed_m_s)
        offsets=np.asarray(offsets,float)
        endpoint=np.asarray(endpoint,float);direction=np.asarray(direction,float)
        if (offsets.ndim!=1 or len(offsets)<2 or not np.isfinite(offsets).all()
                or not np.all(np.diff(offsets)>0) or not -.025<=offsets[0]<0<offsets[-1]<=.02
                or isinstance(fraction,bool) or not np.isfinite(fraction) or not 0<=fraction<1
                or endpoint.shape!=(3,) or direction.shape!=(3,)
                or not np.isfinite([endpoint,direction]).all()
                or abs(np.linalg.norm(direction)-1)>1e-6 or direction[2]>-np.cos(np.radians(5))
                or physics_hz!=480 or type(step) is not int or step<1):
            raise ValueError('Exact screened downward stroke and release sample required')
        self.start=float(offsets[0]);self.end=float(offsets[-1]);self.span=self.end-self.start
        self.offset=self.start+float(fraction)*self.span
        self.endpoint=endpoint.copy();self.direction=direction.copy();self.hz=physics_hz
        self.first_step=step;self.step=step-1;self.consumed=None;self.origin=None
        self.complete=False;self.dwell=0;self.receipt=None;self.upper=0.;self.normal=0.
        self.material_section=None if material_section is None else dict(material_section)
        self.face_sliding=False

    def observe(self, record, *, step, arc_up, support_ready, section_pose=None):
        from .knife import KNIFE_IMPULSE_CONTRACT
        if (type(step) is not int or step!=self.step+1
                or step>self.first_step and self.consumed!=self.step
                or record.get('native_guards_passed') is not True or record.get('cut') is not True
                or type(support_ready) is not bool):
            raise RuntimeError('Fresh guarded released-target sample required for follow-through')
        k=record['knife'];edge=np.asarray(k['edge_frame'],float);arc=np.asarray(arc_up,float)
        if (k.get('force_contract')!=KNIFE_IMPULSE_CONTRACT or k.get('raw_normal_rows_complete') is not True
                or edge.shape!=(4,4) or arc.shape!=(3,) or not np.isfinite(np.r_[edge.flat,arc]).all()
                or not np.allclose(edge[3],[0,0,0,1],atol=1e-8,rtol=0)
                or not np.allclose(edge[:3,:3].T@edge[:3,:3],np.eye(3),atol=1e-5,rtol=0)
                or np.linalg.det(edge[:3,:3])<=0 or abs(np.linalg.norm(arc)-1)>1e-5
                or min(edge[2,0],arc[2])<np.cos(np.radians(5))):
            raise RuntimeError('Actual arc-up, edge-down complete contact observation required')
        upper=k['tool_contact_upper_bound_n'];normal=k['edge_signed_resistance_n']
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in (upper,normal))
                or not np.isfinite([upper,normal]).all() or not 0<=upper<=.5 or abs(normal)>upper+1e-7):
            raise RuntimeError('Invalid/unsafe native follow-through load')
        p=edge[:3,3]
        if self.origin is None:self.origin=p.copy()
        remaining=float(np.dot(self.endpoint-p,self.direction))
        lateral=float(np.linalg.norm((self.endpoint-p)-remaining*self.direction))
        # Same 3 mm metric target-motion/axial envelope as the existing cut
        # trial, NOT a new 0.25 mm full-arm tracking requirement. The native
        # force, penetration and unintended-contact guards still bound every
        # actual pose; this radius is not a collision-clearance certificate.
        if lateral>.003 or remaining<-.0005:
            raise RuntimeError('Measured follow-through left the screened stroke corridor')
        advance=float(np.dot(p-self.origin,self.direction));world_down=float(self.origin[2]-p[2])
        section=None
        if self.material_section is not None:
            from .cut_section import clearance,cut_face_contact
            if not isinstance(section_pose,dict) or set(section_pose)!={'centre','axis'}:
                raise RuntimeError('Fresh native proximal material-section pose required')
            spec=self.material_section
            section=clearance(edge,section_pose['centre'],section_pose['axis'],
                radius=spec['radius'],tip_offset=spec['tip_offset'],half_span=spec['half_span'])
            self.face_sliding=cut_face_contact(record,knife=spec['knife'],faces=spec['faces'])
        at_end=(abs(remaining)<=.00025 and upper<.01 and support_ready
                and min(advance,world_down)>.0003)
        if section is not None:
            # The sharp edge must traverse the full material cross-section.
            # Light side-face sliding after verified release is not a new cut
            # and need not be unloaded until the separate withdrawal check.
            at_end=(section['sharp_edge_cleared'] and self.face_sliding and upper<=.40
                    and support_ready and min(advance,world_down)>.0003)
        self.dwell=self.dwell+1 if at_end else 0
        self.complete=self.complete or self.dwell>=math.ceil(.025*self.hz)
        self.step=step;self.upper=float(upper);self.normal=float(normal);self.support=support_ready
        self.receipt=dict(model='guarded_measured_post_release_stroke_v1',step=step,
            complete=self.complete,remaining_m=remaining,lateral_error_m=lateral,
            net_world_down_m=world_down,net_stroke_advance_m=advance,
            endpoint_unloaded_dwell_s=self.dwell/self.hz,tool_contact_upper_bound_n=upper,
            endpoint_axial_tolerance_m=.00025,lateral_tracking_envelope_m=.003,
            cut_authorized=False,collision_geometry_changed=False,tissue_fracture_calibrated=False)
        if section is not None:
            self.receipt.update(model='guarded_native_material_section_traversal_v1',section=section,
                completion_basis='sharp_edge_beyond_full_native_shaft_section_then_withdraw',
                post_release_cut_face_sliding_verified=self.face_sliding,
                section_clearance_dwell_s=self.dwell/self.hz,
                endpoint_unloaded_dwell_s=None,withdrawal_unloading_still_required=True)
        return dict(self.receipt)

    def command(self, *, step, dt):
        if (step!=self.step or self.consumed==step or not np.isfinite(dt)
                or isinstance(dt,bool) or abs(dt-1/self.hz)>1e-12):
            raise RuntimeError('Follow-through decision is missing, stale or reused')
        if not self.complete and (step-self.first_step)/self.hz>=35:
            raise RuntimeError('Measured follow-through timed out; solid-face/clearance model unresolved')
        if self.complete: speed=0.;state='complete'
        elif not self.support: speed=0.;state='hold_support'
        elif self.upper>.40 or self.normal>.28: speed=-.0005;state='backoff_load'
        elif self.material_section is not None and not self.face_sliding: speed=0.;state='hold_unverified_cut_face'
        elif self.upper>.10 and self.normal<.01 and not self.face_sliding: speed=0.;state='hold_nonleading_contact'
        elif self.face_sliding:
            from .postrelease_feed import loaded_speed
            speed=loaded_speed(self.upper,self.maximum_feed)*min(1.,max(0.,(.40-self.upper)/.10));state='guarded_cut_face_sliding'
        else: speed=.0003;state='forward'
        self.offset=float(np.clip(self.offset+speed*dt,self.start,self.end))
        self.consumed=step
        self.receipt.update(command_state=state,command_offset_m=self.offset,
            command_speed_m_s=speed,maximum_contact_feed_m_s=self.maximum_feed,commanded_motion_used_as_completion=False)
        return (self.offset-self.start)/self.span
