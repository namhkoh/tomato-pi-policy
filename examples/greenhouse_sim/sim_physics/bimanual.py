"""Guarded full-robot knife test. Privileged fixture, not a learned controller."""
import time
import math
from dataclasses import asdict
import numpy as np

from .full_robot import FullRobotGripper
from .knife import (KnifeGeometry,ShearGate,ShearParameters,mount_forward,cut_plane_normal,
    transverse_stroke_offsets,knife_normal_impulse,resisting_face_normal,
    LEGACY_CUT_MODEL,KNIFE_IMPULSE_CONTRACT)
from .contact_events import NativeNormalContact,original_order_tool_contact,NATIVE_NORMAL_ROW_CONTRACT


class BimanualRobot(FullRobotGripper):
    def __init__(self,*args,**kwargs):
        self.cut_model=kwargs.pop('cut_model',LEGACY_CUT_MODEL)
        cut_parameters=ShearParameters(model=self.cut_model)
        proposal_path=kwargs.pop('cut_proposal_json',None)
        fixed=kwargs.pop('right_ik_fixed_joint',None)
        if fixed is not None:
            values=np.asarray(fixed)
            if (values.shape!=(2,) or values.dtype.kind not in 'iuf'
                    or not np.isfinite(values).all() or float(values[0])!=int(values[0])
                    or not 0<=int(values[0])<7):
                raise ValueError('Right IK fixed-joint proposal requires joint index 0..6 and finite degrees')
            fixed=(int(values[0]),float(values[1]))
        self.right_ik_fixed_joint=fixed
        self.native_static_clearance=kwargs.pop('native_static_clearance',False)
        self.native_static_planning_seconds=kwargs.pop('native_static_planning_seconds',8.)
        if (isinstance(self.native_static_planning_seconds,(bool,np.bool_))
                or not np.isfinite(self.native_static_planning_seconds)
                or not 0<self.native_static_planning_seconds<=60
                or (self.native_static_planning_seconds!=8 and not self.native_static_clearance)):
            raise ValueError('Explicit native planning budget must be within (0,60] seconds')
        self.diagnostic_grasp_contacts=kwargs.pop('diagnostic_grasp_contacts',False)
        if type(self.diagnostic_grasp_contacts) is not bool: raise ValueError('Explicit contact diagnostic flag required')
        if type(self.native_static_clearance) is not bool:raise ValueError('Explicit native clearance flag required')
        standoff=kwargs.pop('cut_standoff',.025)
        self.grasp_compression=kwargs.pop('grasp_compression',.0005)
        if not np.isfinite(self.grasp_compression) or not .00025<=self.grasp_compression<=.001:
            raise ValueError('Diagnostic pad compression must be finite 0.25..1 mm')
        kwargs.setdefault('station_offset',(0.,0.))
        kwargs.setdefault('approach_side',1)
        super().__init__(*args,**kwargs)
        if fixed is not None:
            low,high=self.kin.arm_limits_degrees('right')
            if not low[fixed[0]]<fixed[1]<high[fixed[0]]:
                raise ValueError('Right IK fixed-joint proposal is outside exact URDF limits')
        from .cut_proposal import load_cut_proposal
        self.cut_proposal=load_cut_proposal(proposal_path,source_target=self.rig.source_target)
        if not self.sparse_contacts: raise ValueError('Bimanual test requires sparse native contacts')
        self.knife_mount=mount_forward(self.stage,self.root)
        from .blade_contacts import refine_blade_contacts
        self.blade_contacts=refine_blade_contacts(self.stage,self.root)
        old=self.root+'/ee_right/attachments/DeleafKnife/BladeCollision'
        self.collider_paths=[p for p in self.collider_paths if p!=old]+self.blade_contacts['collider_paths']
        from .arc_contacts import refine_arc_contacts
        self.arc_contacts=refine_arc_contacts(self.stage,self.root)
        old_arc=self.root+'/ee_right/attachments/DeleafKnife/ArcCollision'
        self.collider_paths=[p for p in self.collider_paths if p!=old_arc]+self.arc_contacts['collider_paths']
        self.knife=KnifeGeometry(self.stage,self.root)
        from pxr import UsdGeom
        radius=max(float(UsdGeom.Capsule.Get(self.stage,self.rig.body_paths[i]+'/StemCollider').GetRadiusAttr().Get())
            for i in (self.rig.cut_index-1,self.rig.cut_index))
        self.stroke_offsets=transverse_stroke_offsets(radius,self.knife.size[0],standoff)
        self.cut_gate=ShearGate(self.rig.source_target,cut_parameters)
        self.cut_authorized=False;self.edge_points=[];self.edge_impulses=[]
        self.edge_normals=[];self.edge_contact_rows=[];self.edge_contact_error=None
        self.cut_contacts=0;self.cut_event=None;self.plan=None
        self.plan_diagnostics=None
        self.expected_right=self.kin.forward('right',self.right,self.base)
        from .self_screen import SelfCapsuleScreen
        self.self_screen=SelfCapsuleScreen(self.stage,self.root,include_tool_boxes=True)
        from .held_plant_screen import HeldPlantScreen
        self.held_plant_screen=HeldPlantScreen(self.rig,self.self_screen.shapes,self.knife.collider)
        # This includes arm-versus-torso, which an inter-arm-only check misses.
        # Check initial and dense grasp path before any native physics starts.
        self.check_grasp_path()

    def plan_approach(self):
        super().plan_approach()
        # The parent also calls this before the cached screen is constructed.
        # Later gravity-settled replans must receive the same screening.
        if hasattr(self,'self_screen'): self.check_grasp_path()

    def check_grasp_path(self):
        minimum=float('inf')
        for index,q in enumerate(self.path_q):
            result=self.check_self(q,self.right)
            if not result['passed']:
                raise RuntimeError(f'Pregrasp self-collision screen at path index {index}: '+str(result))
            minimum=min(minimum,result['minimum_clearance_m'])
        self.minimum_grasp_self_clearance=minimum

    def body_world(self,left,right):
        pose=dict(self.pose)
        pose.update({f'left_arm_{i}':float(v) for i,v in enumerate(left)})
        pose.update({f'right_arm_{i}':float(v) for i,v in enumerate(right)})
        return {link:self.base@frame for link,frame in
            self.kin.all_link_transforms(pose,prismatic_m=getattr(self,'planning_slides',self.slides)).items()}

    def check_self(self,left,right,*,include_clearances=False):
        return self.self_screen.check(self.body_world(left,right),include_clearances=include_clearances)

    def check_held_plant(self,left,right,*,stroke=False):
        return (not hasattr(self,'held_plant_screen') or
            self.held_plant_screen.check(self.body_world(left,right),stroke=stroke))

    def solve_right_pose(self,desired,seed):
        """Optional redundancy constraint; never bypass pose/path validation.

        Keep the selected joint fixed for the WHOLE tool stroke, not just the
        endpoint. Transit from the unchanged parked arm is separately checked.
        This proposes joint-drive targets; it never sets native body poses.
        """
        fixed=getattr(self,'right_ik_fixed_joint',None)
        if fixed is None:
            return self.kin.solve_pose('right',desired,seed,self.base,maximum_evaluations=250)
        from .redundant_ik import solve_fixed_joint
        return solve_fixed_joint(self.kin,'right',desired,seed,self.base,
            joint_index=fixed[0],joint_degrees=fixed[1],maximum_evaluations=250)

    def screen_grasp_scene(self,frames):
        """Conservative left approach/closure screen; never certifies a grasp.

        The caller supplies current native plant frames before moving. Only
        finger/selected-shaft pairs are expected; every leaf remains checked.
        Reuse the local static cache, never traverse USD in each physics tick.
        """
        from .held_plant_screen import HeldPlantScreen
        if self.held_plant_screen.workspace is None:
            self.held_plant_screen.include_static_scene(self.stage,self.root,self.rig.root,
                self.body_world(self.initial_q,self.right)['link_right_arm_0'][:3,3])
        screen=HeldPlantScreen(self.rig,self.self_screen.shapes,self.knife.collider,
            arm='left',grasp_path=self.grasp_path)
        screen.static=self.held_plant_screen.static
        screen.static_indices=self.held_plant_screen.static_indices
        screen.workspace=self.held_plant_screen.workspace
        screen.snapshot(frames)
        previous=getattr(self,'planning_slides',None);checks=0
        result=dict(passed=False,training_eligible=False,native_grasp_verified=False)
        try:
            self.planning_slides=self.slides.copy()
            for fraction,q in zip(self.fractions,self.path_q):
                if fraction>1.+1e-8: break
                checks+=1
                if not screen.check(self.body_world(q,self.right),grasp=True):
                    result['failure']={**screen.last_failure,'phase':'approach','fraction':float(fraction)}
                    return result
            q=self.path_q[int(np.argmin(abs(self.fractions-1.)))]
            aperture=max(0.,self.radius-self.grasp_compression)
            for gap in np.linspace(.025,aperture,max(2,int(np.ceil((.025-aperture)/.001))+1)):
                self.planning_slides={'gripper_finger_l1':-gap,'gripper_finger_l2':gap}
                checks+=1
                if not screen.check(self.body_world(q,self.right),grasp=True):
                    result['failure']={**screen.last_failure,'phase':'closure','aperture_m':float(gap)}
                    return result
            result['passed']=True
            return result
        finally:
            if previous is None: del self.planning_slides
            else: self.planning_slides=previous
            result['checks']=checks
            self.grasp_scene_screen=result

    def bind(self,simulation_view):
        self.release_grasp_observer()
        super().bind(simulation_view)
        self.right_indices=[self.names.index(f'right_arm_{i}') for i in range(7)]
        self.right_palm=simulation_view.create_rigid_body_view(self.knife.wrist_path)
        if self.right_palm.count!=1: raise RuntimeError('Missing native right wrist')
        self.event_monitor.tool_contact=self._tool_contact
        from .shaft_grasp_native import ShaftGraspNative,NATIVE37_SENSOR_CONTRACT
        # Physical support uses signed callbacks, not unsigned tensor magnitudes.
        # Keep the historical strict fault-capture mode separately reproducible.
        evidence_options=(dict(diagnostic_noncompressive_report=True)
            if getattr(self,'diagnostic_grasp_contacts',False) else
            dict(allow_signed_native_normals=True,sensor_contract=NATIVE37_SENSOR_CONTRACT))
        self.grasp_observer=ShaftGraspNative(self.stage,self.rig,
            selected_index=self.body_index,finger_paths=self.paths[1:],
            contact_views=self.contact_views,**evidence_options)
        self.event_monitor.normal_contact_observer=self.grasp_observer

    def release_grasp_observer(self):
        observer=getattr(self,'grasp_observer',None)
        if observer is not None:
            monitor=getattr(self,'event_monitor',None)
            if monitor is not None and monitor.normal_contact_observer is observer:
                monitor.normal_contact_observer=None
            observer.close()
            self.grasp_observer=None

    def contact_with_frames(self,dt,frames,*,step_id):
        """Same-step exact shaft/pad contacts, reconciled with selected tensors.

        No motion command, FK pose or friction impulse is grasp evidence.
        Existing controller dwell/slip windows remain in bimanual_probe.
        """
        from .runtime import pose_matrices
        if getattr(self,'diagnostic_grasp_contacts',False):
            self.grasp_observer.raise_pending_fault(dt,step_id=step_id)
        if self.event_monitor.error is not None: raise RuntimeError(self.event_monitor.error)
        if not self.event_monitor.native_full_contact_reporting:
            raise RuntimeError('Missing full native contact stream for shaft grasp')
        fingers=pose_matrices(self.fingers.get_transforms())[self.order]
        result=self.grasp_observer.evaluate(dt,frames,fingers,frames_step_id=step_id)
        self.latest_finger_bilateral=result['bilateral']
        return result

    def close(self,fraction):
        # Geometry-bounded closure for this privileged shaft fixture. Driving
        # to a zero-width aperture keeps compressing a ~6 mm stem after grasp.
        # Default stop is 0.5 mm inside radius; the bounded diagnostic bias
        # can be qualified separately. Force and penetration guards
        # remain unchanged and actual opposing contact still verifies grasp.
        if not np.isfinite(fraction) or not 0<=fraction<=1: raise ValueError('Invalid finger closure')
        aperture=max(0.,self.radius-getattr(self,'grasp_compression',.0005))
        super().close(fraction*(1-aperture/.025))

    def seam(self,frames):
        i=self.rig.cut_index
        half=np.linalg.norm(self.rig.chain_world[i+1]-self.rig.chain_world[i])/2
        return frames[i,:3,3]-half*frames[i,:3,2],frames[i,:3,2]

    @original_order_tool_contact
    def _tool_contact(self,row):
        # Only the flat leading strip contacting the two seam-adjacent shaft
        # capsules is an expected tool load. Arc, camera, main stem and leaves
        # remain unwanted contacts. Positions come from the native callback.
        if not isinstance(row,NativeNormalContact):
            self.edge_contact_error='Explicit original-order native blade row required'
            raise ValueError(self.edge_contact_error)
        if self.edge_contact_error is not None: raise RuntimeError(self.edge_contact_error)
        if len(self.edge_contact_rows)>=256:
            self.edge_contact_error='Blade normal-row trace overflow (256); evidence incomplete'
            raise RuntimeError(self.edge_contact_error)
        record=dict(asdict(row),raw_contract=NATIVE_NORMAL_ROW_CONTRACT,eligible=False)
        self.edge_contact_rows.append(record)
        try:
            first=row.collider0==self.knife.collider;second=row.collider1==self.knife.collider
            if first==second:
                record['rejection']='not_exactly_one_knife_collider';return False
            other=row.collider1 if first else row.collider0
            sign=1 if first else -1
            impulse=sign*np.asarray(row.impulse);normal=sign*np.asarray(row.normal)
            record.update(knife_header_index=0 if first else 1,knife_sign=sign,
                impulse_on_knife=impulse.tolist(),normal_on_knife=normal.tolist(),
                signed_resistance_impulse_ns=-float(np.dot(impulse,-self.edge_frame[:3,0])))
            eligible=[self.rig.body_paths[i]+'/StemCollider'
                for i in (self.rig.cut_index-1,self.rig.cut_index)]
            reason=None
            if not self.cut_authorized: reason='cut_not_authorized'
            elif other not in eligible: reason='not_seam_adjacent_shaft'
            elif not self.knife.on_edge(row.point,self.edge_frame): reason='outside_exact_leading_strip'
            elif not resisting_face_normal(normal,-self.edge_frame[:3,0]): reason='wrong_oriented_leading_normal'
            else:
                centre,axis=self.cut_frame
                if abs(np.dot(np.asarray(row.point)-centre,axis))>.003: reason='outside_seam_axial_window'
            if reason is not None:
                record['rejection']=reason;return False
            _,scalar=knife_normal_impulse(normal,impulse)
            record['signed_normal_impulse_ns']=scalar
            # Negative compliant rows remain eligible geometry and SUBTRACT
            # from signed resistance. Their magnitudes still contribute caps.
            self.edge_points.append(row.point);self.edge_impulses.append(impulse)
            self.edge_normals.append(normal);record['eligible']=True
            self.cut_contacts+=1
            return True
        except Exception as exc:
            record['error']=str(exc);self.edge_contact_error=str(exc)
            raise

    def prepare_step(self,frames):
        from .runtime import pose_matrices
        if self.edge_contact_error is not None: raise RuntimeError(self.edge_contact_error)
        self.edge_frame=self.knife.frame(pose_matrices(self.right_palm.get_transforms())[0])
        self.cut_frame=self.seam(frames)
        self.edge_points=[];self.edge_impulses=[]
        self.edge_normals=[];self.edge_contact_rows=[]

    def right_transit(self,left_q,goal):
        """Bounded shoulder and joint-space detours, not a continuous scene certificate.

        A clear endpoint can require opening the right shoulder before raising
        the forearm. Try the direct path first, then fixed URDF-valid outward
        shoulder waypoints, then a seeded bounded bidirectional search. Every
        <=1-degree sample retains arm/tool and scene checks; no filter is changed.
        """
        lower,upper=self.kin.arm_limits_degrees('right')
        goal=np.asarray(goal,dtype=float)
        if (goal.shape!=(7,) or not np.isfinite(goal).all()
                or np.any(goal<=lower) or np.any(goal>=upper)):
            raise ValueError('Invalid right transit goal')
        waypoints=[None]
        for angle in (-30.,-60.,-90.):
            waypoint=self.right.copy();waypoint[1]=angle
            if np.all((waypoint>lower)&(waypoint<upper)): waypoints.append(waypoint)
        for waypoint in waypoints:
            vertices=[self.right,goal] if waypoint is None else [self.right,waypoint,goal]
            chunks=[]
            for start,end in zip(vertices[:-1],vertices[1:]):
                count=max(2,int(np.ceil(np.max(np.abs(end-start))))+1)
                chunk=np.linspace(start,end,count)
                chunks.append(chunk if not chunks else chunk[1:])
            path=np.concatenate(chunks);minimum=float('inf')
            for row in path:
                clearance=self.kin.inter_arm_clearance(left_q,row,self.base).clearance_m
                if (clearance<.01 or not self.check_self(left_q,row)['passed']
                        or not self.check_held_plant(left_q,row)): break
                minimum=min(minimum,clearance)
            else:
                return path,minimum,dict(method='direct' if waypoint is None else 'outward_shoulder_waypoint',
                    waypoint_degrees=None if waypoint is None else waypoint.tolist(),
                    maximum_joint_sample_step_degrees=1.,held_plant_snapshot_screened=hasattr(self,'held_plant_screen'),
                    whole_scene_certified=False)
        from .joint_path import connect_path
        checked=0
        def valid(q):
            nonlocal checked
            checked+=1
            return (self.kin.inter_arm_clearance(left_q,q,self.base).clearance_m>=.01
                and self.check_self(left_q,q)['passed'] and self.check_held_plant(left_q,q))
        path=connect_path(self.right,goal,lower,upper,valid)
        if path is not None:
            minimum=min(self.kin.inter_arm_clearance(left_q,q,self.base).clearance_m for q in path)
            return path,minimum,dict(method='bounded_bidirectional_joint_search',seed=0,
                collision_checks=checked,maximum_joint_sample_step_degrees=1.,
                held_plant_snapshot_screened=hasattr(self,'held_plant_screen'),whole_scene_certified=False)
        return None

    def plan_cut(self,frames,left_q):
        # Include failed planning attempts: PhysicsClock does not complete its
        # tick timing when a planner raises before the next native step.
        start=time.perf_counter()
        planning_error=None
        try:
            self.plan=None
            result=self._plan_cut(frames,left_q)
            candidate=self.plan
            self.plan=None  # Provisional geometry is not an accepted plan.
            screen=getattr(self,'held_plant_screen',None)
            native=getattr(screen,'native_static_query',None)
            if native is not None:native.validate()
            self.plan=candidate
            return result
        except BaseException as exc:
            self.plan=None;planning_error=exc
            raise
        finally:
            cleanup_errors=[]
            screen=getattr(self,'held_plant_screen',None)
            native=getattr(screen,'native_static_query',None)
            if native is not None:
                try:native.close()
                except Exception as exc:cleanup_errors.append(exc)
                try:
                    diagnostics=getattr(self,'plan_diagnostics',None)
                    if diagnostics is not None:
                        reported=native.report()
                        evidence=diagnostics.setdefault('native_static_clearance',{})
                        evidence.update(reported)
                        evidence['query_count_known']=type(reported.get('query_count')) is int
                except Exception as exc:cleanup_errors.append(exc)
                finally:screen.native_static_query=None
            self.planning_wall_seconds=time.perf_counter()-start
            if cleanup_errors:
                self.plan=None
                detail='Native refinement cleanup failed: '+str([str(e) for e in cleanup_errors])
                if planning_error is not None:planning_error.add_note(detail)
                else:raise RuntimeError(detail) from cleanup_errors[0]

    def _plan_cut(self,frames,left_q):
        """Bounded orientation search, then dense arm-pair screened IK paths.

        Native whole-scene guards remain essential: arm capsules alone do not
        certify the knife, cameras, foliage or gutter swept volume.
        """
        # Reset before any snapshot/query setup: a failed new plan must never
        # publish a prior plan's diagnostics or omit a zero-attempt failure.
        attempts=[];failures=[]
        self.plan=None
        self.plan_diagnostics=dict(endpoint_attempts=attempts,path_failures=failures,
            minimum_required_interarm_m=.01,held_plant_margin_m=.001,
            held_plant_native_snapshot_screened=False,
            complete_tool_stroke_screened_before_IK=False,
            whole_scene_path_certified=False)
        native_requested=getattr(self,'native_static_clearance',False)
        if native_requested:
            native_evidence=dict(initialization_status='not_started',query_count=0,
                query_count_known=True,final_validation_passed=False,errors=[],
                current_plan_only=True,whole_path_certified=False,
                all_simulation_shape_query_coverage_proved=False)
            self.plan_diagnostics['native_static_clearance']=native_evidence
        centre,axis=self.seam(frames)
        # Context plants/gutters are populated after robot construction. Cache
        # only once the complete scene exists, not in __init__.
        if self.held_plant_screen.workspace is None:
            self.held_plant_screen.include_static_scene(self.stage,self.root,self.rig.root,
                self.body_world(self.initial_q,self.right)['link_right_arm_0'][:3,3])
        self.held_plant_screen.snapshot(frames)
        self.plan_diagnostics.update(held_plant_native_snapshot_screened=True,
            local_static_colliders=len(self.held_plant_screen.static),
            local_scene_bounds_m=[v.tolist() for v in self.held_plant_screen.workspace])
        if native_requested:
            try:
                if not hasattr(self,'robot'):raise RuntimeError('Native static refinement requires an initialized robot scene')
                from .native_static_clearance import current_scene_query
                # The factory can perform coverage queries before raising. If
                # it never returns, its internal count is unknown, not zero.
                native_evidence.update(initialization_status='in_progress',
                    query_count=None,query_count_known=False)
                query_options={}
                if getattr(self,'native_static_planning_seconds',8.)!=8.:
                    query_options['wall_limit_s']=self.native_static_planning_seconds
                self.held_plant_screen.native_static_query=current_scene_query(
                    self.stage,self.held_plant_screen.static,**query_options)
                native_evidence['initialization_status']='ready'
            except BaseException as exc:
                native_evidence['initialization_status']='failed'
                native_evidence['errors'].append(type(exc).__name__+': '+str(exc))
                raise
        # Finger geometry must use the held aperture, not the initial open hand.
        if hasattr(self,'robot'):
            positions=self.robot.get_dof_positions()[0]
            self.planning_slides={name:float(positions[self.names.index(name)]) for name in self.slides}
        else:
            # Offline fixture-only plan; native execution always snapshots above.
            self.planning_slides={'gripper_finger_l1':-self.radius,'gripper_finger_l2':self.radius}
        from .rigid_tool_screen import RigidToolScreen
        rigid_screen=RigidToolScreen(self,left_q)
        self.plan_diagnostics['complete_tool_stroke_screened_before_IK']=True
        direction=-self.goal[:3,2];direction-=axis*np.dot(direction,axis)
        direction_norm=float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm):raise ValueError('Finite legacy palm approach required')
        direction=direction/direction_norm if direction_norm>1e-12 else None
        single=None
        if getattr(self,'cut_proposal',None) is not None:
            from .cut_proposal import resolve_cut_proposal
            single=resolve_cut_proposal(self.cut_proposal,source_target=self.rig.source_target,
                stem_axis_world=axis,edge_width_m=self.knife.size[1])
            self.plan_diagnostics['single_world_proposal']=single
            single_direction=np.asarray(single['direction_world'],float)
            # A world-space proposal does not need a legacy transverse basis.
            # When that basis is undefined, retain the real direction and use
            # JSON null for the diagnostic relative angle; invent no heading.
            single_angle=None if direction is None else float(np.degrees(np.arctan2(
                np.dot(np.cross(direction,single_direction),axis),np.dot(direction,single_direction))))
            single['relative_angle_defined']=direction is not None
            if direction is None:
                single['relative_angle_unavailable_reason']='legacy_approach_parallel_to_stem_axis'
        elif direction is None:
            raise ValueError('Legacy palm approach is parallel to stem axis; no orientation-grid basis')
        # Try the original plane first; only then small oblique planes within
        # the existing measured angular gate. Never replace actual stem truth
        # with the proposed blade normal when evaluating native contact.
        clear_endpoints=0
        for tilt in ((single['plane_tilt_degrees'],) if single is not None else (0.,-10.,10.)):
            usable_wing=float(self.knife.size[1]/2-.005)
            proposals=([(single_angle,single['normal_sign'],single['wing_m'])] if single is not None else
                [(a,s,w) for w in (0.,-usable_wing/2,usable_wing/2,-.9*usable_wing,.9*usable_wing,-usable_wing,usable_wing) for s in (1,-1)
                    for a in (0,15,-15,30,-30,45,-45,60,-60,90,-90,120,-120,135,-135,150,-150,180)])
            for degrees,normal_sign,wing in proposals:
                if single is not None:d=single_direction.copy()
                else:
                    angle=np.radians(degrees)
                    d=direction*np.cos(angle)+np.cross(axis,direction)*np.sin(angle)
                normal=cut_plane_normal(d,normal_sign*axis,tilt)
                # Mounting roll is fixed on the wrist. Both signs of a
                # transverse cutting plane are valid wrist poses; global
                # "arc up" is not a cut-contact criterion. Scene/tool checks
                # still reject the support hitting the main stem or left hand.
                desired=self.knife.wrist_for_edge(centre+self.stroke_offsets[0]*d,d,normal,wing)
                attempt=dict(angle=degrees,normal_sign=normal_sign,wing_m=wing,plane_tilt_degrees=tilt,
                    ik_attempted=False,ik_succeeded=False,evaluations=0)
                attempts.append(attempt)
                # Constant orientation: translate the actual wrist frame for
                # every <=0.5 mm stroke sample, including the final endpoint.
                # Reject local hardware/plant conflicts before costly arm IK.
                wrist_frames=np.repeat(desired[None],len(self.stroke_offsets),axis=0)
                wrist_frames[:,:3,3]+=(self.stroke_offsets-self.stroke_offsets[0])[:,None]*d
                subset=rigid_screen.check(wrist_frames)
                attempt['rigid_tool_corridor']=subset
                if not subset['passed']:
                    attempt['rejection']='rigid_tool_corridor';continue
                solution=self.solve_right_pose(desired,self.right)
                attempt.update(ik_attempted=True,
                    position_error_m=solution.position_error_m,orientation_error_rad=solution.orientation_error_rad,
                    evaluations=solution.evaluations,ik_succeeded=solution.succeeded)
                if not solution.succeeded:
                    attempt['rejection']='endpoint_IK';continue
                q=np.array(solution.joint_degrees)
                clearance=self.kin.inter_arm_clearance(left_q,q,self.base).clearance_m
                attempt['interarm_clearance_m']=clearance
                if clearance<.01: attempt['rejection']='endpoint_arm_clearance';continue
                check=self.check_self(left_q,q)
                attempt['self_capsule_screen']=check
                if check['passed']:
                    if not self.check_held_plant(left_q,q):
                        attempt['rejection']='endpoint_held_plant'
                        attempt['plant_screen']=self.held_plant_screen.last_failure
                        continue
                    clear_endpoints+=1
                    candidate=(np.linalg.norm(q-self.right),degrees,d,q,normal_sign,wing,normal)
                    # Check the useful path NOW. Enumerating the rest of the
                    # grid first spent the native epoch budget after finding a
                    # usable endpoint (native45). This is first fully checked
                    # feasibility, not an optimal/shortest-path search.
                    if self._try_cut_candidate(left_q,centre,axis,candidate,tilt,failures):return
                else: attempt['rejection']='endpoint_self_collision'
        ik=sum(a['ik_succeeded'] for a in attempts)
        raise RuntimeError(f'No bimanual arm-clearance path: endpoints={len(attempts)}, '
            f'IK_attempted={sum(a["ik_attempted"] for a in attempts)}, IK_converged={ik}, '
            f'arm_clear_endpoints={clear_endpoints}, path_failures={failures}')

    def _try_cut_candidate(self,left_q,centre,axis,candidate,tilt,failures):
        """Full sampled stroke, then bounded transit; no endpoint-only success."""
        _,angle,d,q,normal_sign,wing,normal=candidate
        failure=dict(angle=angle,plane_tilt_degrees=tilt,normal_sign=normal_sign,wing_m=wing)
        minimum=float('inf');stroke=[];seed=q
        for offset in self.stroke_offsets:
            desired=self.knife.wrist_for_edge(centre+offset*d,d,normal,wing)
            solution=self.solve_right_pose(desired,seed)
            failure.update(offset_m=float(offset),rejection='stroke_IK')
            if not solution.succeeded:break
            seed=np.asarray(solution.joint_degrees)
            clearance=self.kin.inter_arm_clearance(left_q,seed,self.base).clearance_m
            minimum=min(minimum,clearance)
            failure.update(rejection='stroke_arm_clearance',interarm_clearance_m=clearance)
            if clearance<.01:break
            check=self.check_self(left_q,seed)
            failure.update(rejection='stroke_self_collision',self_screen=check)
            if not check['passed']:break
            if not self.check_held_plant(left_q,seed,stroke=True):
                failure.update(rejection='stroke_held_plant',plant_screen=self.held_plant_screen.last_failure)
                break
            stroke.append(seed)
        if len(stroke)!=len(self.stroke_offsets):
            failures.append(failure);return False
        transit=self.right_transit(left_q,q)
        if transit is None:
            failures.append(dict(failure,rejection='bounded_transit_arm_self_or_plant_clearance'));return False
        approach,transit_minimum,transit_evidence=transit
        minimum=min(minimum,transit_minimum)
        self.plan=dict(approach=approach,stroke=np.asarray(stroke),direction=d,
            centre=centre.copy(),axis=axis.copy(),angle=angle,normal_sign=normal_sign,wing_m=wing,
            blade_plane_normal=normal.copy(),plane_tilt_degrees=tilt,
            stroke_offset_range_m=[float(self.stroke_offsets[0]),float(self.stroke_offsets[-1])],
            stroke_samples=len(self.stroke_offsets),stroke_end_basis='shaft_radius_plus_half_edge_strip_plus_1mm',
            minimum_interarm_m=minimum,transit=transit_evidence)
        return True

    def command_right(self,phase,fraction):
        if phase=='park': q=self.right
        else:
            if self.plan is None or not 0<=fraction<=1: raise RuntimeError('Unplanned right command')
            path=self.plan[phase];where=fraction*(len(path)-1)
            low=min(int(where),len(path)-2);alpha=where-low
            q=(1-alpha)*path[low]+alpha*path[low+1]
        self.expected_right=self.kin.forward('right',q,self.base)
        self.targets[0,self.right_indices]=np.radians(q)
        self.robot.set_dof_position_targets(self.targets,self.index)

    def inspect_cut(self,dt,frames,held,slip):
        from .runtime import pose_matrices
        if self.edge_contact_error is not None: raise RuntimeError(self.edge_contact_error)
        if not self.event_monitor.native_full_contact_reporting:
            raise RuntimeError('Full original-order native contact stream required for blade evidence')
        loads=self.event_monitor.measurements(dt)
        if loads['allowed_tool_contact_n']>.5:
            raise RuntimeError('Unsigned normal-plus-friction tool load exceeds 0.5 N')
        actual=pose_matrices(self.right_palm.get_transforms())[0]
        error=float(np.linalg.norm(actual[:3,3]-self.expected_right[:3,3]))
        if error>.012: raise RuntimeError(f'Right wrist tracking error {error:.5f} m')
        edge=self.knife.frame(actual);centre,axis=self.seam(frames)
        decision=self.cut_gate.observe(dt=dt,edge=edge,centre=centre,axis=axis,
            points=self.edge_points,impulses=self.edge_impulses,normals=self.edge_normals,
            impulse_contract=KNIFE_IMPULSE_CONTRACT,edge_contact_verified=True,
            tool_contact_upper_bound_n=loads['allowed_tool_contact_n'],
            held=held and self.cut_authorized,slip=slip)
        if decision:
            self.cut_event=self.rig.release_from_blade(decision)
        return dict(edge_frame=edge.tolist(),right_tracking_error_m=error,
            edge_contact_count=len(self.edge_points),edge_force_n=math.fsum(math.hypot(*v) for v in self.edge_impulses)/dt,
            edge_signed_resistance_n=self.cut_gate.signed_resistance_n,
            edge_unsigned_projection_n=self.cut_gate.unsigned_projection_n,
            tool_contact_upper_bound_n=loads['allowed_tool_contact_n'],
            force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows=[dict(r) for r in self.edge_contact_rows],
            raw_normal_row_limit=256,raw_normal_rows_complete=True,
            cut_model=self.cut_gate.parameters.model,
            loading_travel_required=self.cut_gate.parameters.model==LEGACY_CUT_MODEL,
            gate_dwell_s=self.cut_gate.dwell,gate_travel_m=self.cut_gate.travel,cut_event=self.cut_event)

    def restore_authored_state(self):
        self.release_grasp_observer()
        super().restore_authored_state()
        self.cut_gate=ShearGate(self.rig.source_target,ShearParameters(model=getattr(self,'cut_model',LEGACY_CUT_MODEL)))
        self.cut_authorized=False
        self.edge_points=[];self.edge_impulses=[];self.edge_normals=[];self.edge_contact_rows=[];self.edge_contact_error=None
        self.cut_event=None;self.plan=None;self.plan_diagnostics=None;self.cut_contacts=0
        self.planning_wall_seconds=None
        self.held_plant_screen.workspace=None;self.held_plant_screen.static=[]
        if hasattr(self,'planning_slides'): del self.planning_slides
        self.expected_right=self.kin.forward('right',self.right,self.base)

    def setup_views(self,viewport):
        from pxr import Gf,UsdGeom
        super().setup_views(viewport)
        edge=self.knife.frame(self.kin.forward('right',self.right,self.base))
        centre=edge[:3,3]
        eye=centre+.23*edge[:3,2]+.18*edge[:3,1]-.12*edge[:3,0]
        path='/World/RightKnifeMountInspection'
        camera=UsdGeom.Camera.Define(self.stage,path)
        camera.CreateFocalLengthAttr(28.);camera.CreateClippingRangeAttr(Gf.Vec2f(.003,10))
        xf=UsdGeom.Xformable(camera);xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*centre),Gf.Vec3d(0,0,1)).GetInverse())
        self.views['Right knife mount']=path

    def report(self):
        result=super().report()
        result.update(right_arm='original_fitted_knife_guarded_native_joint_drives',
            grasp_closure=dict(mode='ground_truth_shaft_width_stop_with_native_contact_verification',
                commanded_half_aperture_m=max(0.,self.radius-self.grasp_compression),
                nominal_pad_compression_m=self.grasp_compression,material_calibrated=False),
            grasp_evidence_model='exact_connected_detached_shaft_inner_pad_normal_contacts_with_selected_tensor_crosscheck',
            diagnostic_grasp_contacts=getattr(self,'diagnostic_grasp_contacts',False),
            minimum_grasp_self_capsule_clearance_m=self.minimum_grasp_self_clearance,
            knife_mount=self.knife_mount,
            blade_contact_geometry=self.blade_contacts,
            arc_contact_geometry=self.arc_contacts,
            planning_wall_seconds=getattr(self,'planning_wall_seconds',None),
            cut_plan_diagnostics=self.plan_diagnostics,
            right_ik_fixed_joint=getattr(self,'right_ik_fixed_joint',None),
            right_ik_policy='first_fully_screened_path_not_shortest_path',
            cut_model=getattr(self,'cut_model',LEGACY_CUT_MODEL),
            cut_model_class='measured_contact_seam_failure_not_calibrated_tissue_cutting')
        if self.plan is not None:
            result['cut_plan']={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                for k,v in self.plan.items() if k not in ('approach','stroke')}
        return result
