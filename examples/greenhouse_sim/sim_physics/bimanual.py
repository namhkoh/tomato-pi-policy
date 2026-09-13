"""Guarded full-robot knife test. Privileged fixture, not a learned controller."""
import time
import math
from dataclasses import asdict
import numpy as np

from .full_robot import FullRobotGripper
from .knife import (KnifeGeometry,ShearGate,ShearParameters,mount_forward,cut_plane_normal,
    transverse_stroke_offsets,knife_normal_impulse,resisting_face_normal,
    LEGACY_CUT_MODEL,DOWNWARD_CUT_MODEL,KNIFE_IMPULSE_CONTRACT)
from .contact_events import NativeNormalContact,original_order_tool_contact,NATIVE_NORMAL_ROW_CONTRACT


class BimanualRobot(FullRobotGripper):
    retention_preload=False
    symmetric_finger_closure=False
    physical_grasp_span=False
    effort_bounded_grasp_target=False
    preload_force_servo=False
    staged_downward_transit=False
    diagnostic_physics_hz=240
    def __init__(self,*args,**kwargs):
        from .cut_strategy import mode
        self.cut_strategy=mode(kwargs.pop('cut_strategy','bimanual'))
        from .diagnostic_rate import frequency
        self.diagnostic_physics_hz=frequency(kwargs.pop('diagnostic_physics_hz',240))
        self.cut_model=kwargs.pop('cut_model',LEGACY_CUT_MODEL)
        from .blade_contacts import SIDE_EDGE,LOWER_EDGE,CROSSBAR_EDGE,DOWNWARD_EDGES,EDGE_MODES
        self.knife_edge_mode=kwargs.pop('knife_edge_mode',SIDE_EDGE)
        source_wrist_contacts=kwargs.pop('source_wrist_contacts',False)
        if type(source_wrist_contacts) is not bool:raise ValueError('Explicit wrist partition option required')
        self.cut_style=kwargs.pop('cut_style','legacy')
        self.cut_priority=kwargs.pop('cut_priority',None)
        if (self.knife_edge_mode not in EDGE_MODES or
                (self.knife_edge_mode in DOWNWARD_EDGES)!=(self.cut_model==DOWNWARD_CUT_MODEL) or
                self.knife_edge_mode in DOWNWARD_EDGES and self.cut_style!='downward'):
            raise ValueError('Lower rim requires the measured downward motion model and downward planner')
        self.staged_downward_transit=kwargs.pop('staged_downward_transit',False)
        self.screened_transit_modes=()
        if type(self.staged_downward_transit) is not bool or self.staged_downward_transit and self.cut_style!='downward':
            raise ValueError('Staged wrist approach requires explicit downward policy')
        self.blade_axial_aim_offset_m=kwargs.pop('blade_axial_aim_offset_m',0.)
        from .blade_aim import edge_centre
        edge_centre(np.zeros(3),np.array([0.,0.,1.]),self.blade_axial_aim_offset_m)
        if self.blade_axial_aim_offset_m and self.cut_style!='downward':
            raise ValueError('Non-default blade aim requires the downward diagnostic')
        self.knife_alignment=kwargs.pop('knife_alignment','legacy')
        self.force_closure_enabled=kwargs.pop('force_closure',False)
        self.physical_grasp_span=kwargs.pop('physical_grasp_span',False)
        self.effort_bounded_grasp_target=kwargs.pop('effort_bounded_grasp_target',False)
        self.preload_force_servo=kwargs.pop('preload_force_servo',False)
        if type(self.preload_force_servo) is not bool or self.preload_force_servo and not self.effort_bounded_grasp_target:
            raise ValueError('Preload force servo requires explicit effort-bounded grasp target')
        if type(self.effort_bounded_grasp_target) is not bool:
            raise ValueError('Explicit effort-bounded grasp target required')
        if type(self.physical_grasp_span) is not bool:
            raise ValueError('Explicit physical grasp-span mode required')
        self.finger_target_antiwindup=kwargs.pop('finger_target_antiwindup',False)
        if type(self.finger_target_antiwindup) is not bool or self.finger_target_antiwindup and not self.force_closure_enabled:
            raise ValueError('Finger target antiwindup requires explicit feedback closure')
        self.native_capsule_sphere_cover=kwargs.pop('native_capsule_sphere_cover',False)
        if type(self.native_capsule_sphere_cover) is not bool:
            raise ValueError('Explicit boolean native sphere cover flag required')
        self.explicit_finger_effort=kwargs.pop('explicit_finger_effort',False)
        if type(self.explicit_finger_effort) is not bool:
            raise ValueError('Explicit boolean finger effort flag required')
        self.retention_preload=kwargs.pop('retention_preload',False)
        self.symmetric_finger_closure=kwargs.pop('symmetric_finger_closure',False)
        if (type(self.symmetric_finger_closure) is not bool or self.symmetric_finger_closure
                and not (self.force_closure_enabled and self.explicit_finger_effort and self.finger_target_antiwindup)):
            raise ValueError('Symmetric closure requires explicit feedback fingers and antiwindup')
        if (type(self.retention_preload) is not bool or self.retention_preload
                and not (self.force_closure_enabled and self.explicit_finger_effort and self.finger_target_antiwindup)):
            raise ValueError('Retention preload requires explicit feedback fingers and antiwindup')
        self.grasp_contact_frames=kwargs.pop('grasp_contact_frames','post_fetch_legacy')
        if self.effort_bounded_grasp_target and not (self.force_closure_enabled and self.retention_preload
                and self.symmetric_finger_closure and self.explicit_finger_effort and self.finger_target_antiwindup):
            raise ValueError('Effort-bounded reference requires unchanged explicit symmetric feedback with antiwindup')
        if self.grasp_contact_frames not in ('post_fetch_legacy','pre_solve_pgs_v1'):
            raise ValueError('Unknown grasp contact frame contract')
        if type(self.force_closure_enabled) is not bool:
            raise ValueError('Explicit native force closure flag required')
        if self.cut_style not in ('legacy','downward'):
            raise ValueError('Unknown cut style')
        cut_parameters=ShearParameters(model=self.cut_model)
        proposal_path=kwargs.pop('cut_proposal_json',None)
        if self.cut_style=='downward' and proposal_path is not None:
            raise ValueError('Downward policy cannot use an unrelated world-cut proposal')
        fixed=kwargs.pop('right_ik_fixed_joint',None)
        if fixed is not None:
            values=np.asarray(fixed)
            if (values.shape!=(2,) or values.dtype.kind not in 'iuf'
                    or not np.isfinite(values).all() or float(values[0])!=int(values[0])
                    or not 0<=int(values[0])<7):
                raise ValueError('Right IK fixed-joint proposal requires joint index 0..6 and finite degrees')
            fixed=(int(values[0]),float(values[1]))
        self.right_ik_fixed_joint=fixed
        if self.cut_style=='downward' and fixed is not None:
            raise ValueError('Downward policy cannot use legacy fixed-joint constraints')
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
        self.rest_grasp_rotation=self.goal[:3,:3].copy()
        if self.force_closure_enabled:
            from .force_closure import ForceClosure
            self.force_closer=ForceClosure(self.radius,self.grasp_compression,retention_preload=self.retention_preload,
                symmetric=self.symmetric_finger_closure,pregrasp_half_aperture=self.pregrasp_half_aperture,
                effort_bounded_target=self.effort_bounded_grasp_target,preload_force_servo=self.preload_force_servo,
                physics_hz=self.diagnostic_physics_hz)
        if fixed is not None:
            low,high=self.kin.arm_limits_degrees('right')
            if not low[fixed[0]]<fixed[1]<high[fixed[0]]:
                raise ValueError('Right IK fixed-joint proposal is outside exact URDF limits')
        from .cut_proposal import load_cut_proposal
        self.cut_proposal=load_cut_proposal(proposal_path,source_target=self.rig.source_target)
        if not self.sparse_contacts: raise ValueError('Bimanual test requires sparse native contacts')
        self.knife_mount=mount_forward(self.stage,self.root,alignment=self.knife_alignment)
        from .blade_contacts import refine_blade_contacts
        self.blade_contacts=refine_blade_contacts(self.stage,self.root,
            edge_mode=LOWER_EDGE if self.knife_edge_mode==CROSSBAR_EDGE else self.knife_edge_mode)
        old=self.root+'/ee_right/attachments/DeleafKnife/BladeCollision'
        self.collider_paths=[p for p in self.collider_paths if p!=old]+self.blade_contacts['collider_paths']
        from .arc_contacts import refine_arc_contacts
        arc_options={'crossbar_edge':True} if self.knife_edge_mode==CROSSBAR_EDGE else {}
        self.arc_contacts=refine_arc_contacts(self.stage,self.root,**arc_options)
        if self.knife_edge_mode==CROSSBAR_EDGE:
            self.blade_contacts['physical_role']='mounting_plate_not_cutting_edge'
        old_arc=self.root+'/ee_right/attachments/DeleafKnife/ArcCollision'
        self.collider_paths=[p for p in self.collider_paths if p!=old_arc]+self.arc_contacts['collider_paths']
        self.knife=KnifeGeometry(self.stage,self.root)
        self.wrist_contacts=None
        if source_wrist_contacts:
            from .wrist_contacts import refine
            self.wrist_contacts=refine(self.stage,self.root)
            self.collider_paths=[p for p in self.collider_paths if p not in self.wrist_contacts['replaced_colliders']]+self.wrist_contacts['collider_paths']
        from pxr import UsdGeom
        radius=max(float(self.stage.GetPrimAtPath(self.rig.body_paths[i]+'/StemCollider').GetAttribute('radius').Get())
            for i in (self.rig.cut_index-1,self.rig.cut_index))
        self.stroke_offsets=transverse_stroke_offsets(radius,self.knife.size[0],standoff)
        self.cut_gate=ShearGate(self.rig.source_target,cut_parameters,strategy=self.cut_strategy)
        self.cut_authorized=False;self.edge_points=[];self.edge_impulses=[]
        self.edge_normals=[];self.edge_contact_rows=[];self.edge_contact_error=None
        self.cut_contacts=0;self.cut_event=None;self.plan=None
        self.plan_diagnostics=None
        self.expected_right=self.kin.forward('right',self.right,self.base)
        from .self_screen import SelfCapsuleScreen
        self.self_screen=SelfCapsuleScreen(self.stage,self.root,include_tool_boxes=True,
            fit_plate=self.cut_style=='downward')
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

    def refresh_grasp_goal(self,frames):
        if getattr(self,'force_closure_enabled',False):
            from .grasp_frame import align_to_axis
            self.goal[:3,:3]=align_to_axis(self.rest_grasp_rotation,
                self.rig.rest_frames[self.body_index,:3,2],frames[self.body_index,:3,2])
        self.goal[:3,3]=self.grasp_point(frames)+self.grasp_depth*self.goal[:3,2]
        from .grasp_target import finger_seam_clearance
        centre,axis=self.seam(frames)
        self.live_grasp_placement=finger_seam_clearance(self.stage,self.root,self.goal,centre,axis)

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
            options=({'joint_limit_margin_degrees':3.}
                if getattr(self,'cut_style','legacy')=='downward' else {})
            result=self.kin.solve_pose('right',desired,seed,self.base,maximum_evaluations=250,**options)
            if getattr(self,'cut_style','legacy')=='downward' and not result.succeeded:
                for wrist in (-120.,0.,120.):
                    proposal=np.array(seed,float,copy=True);proposal[6]=wrist
                    result=self.kin.solve_pose('right',desired,proposal,self.base,maximum_evaluations=250,**options)
                    if result.succeeded:break
            return result
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
        native=None;planning_error=None
        try:
            # Internal capsule overlap and segment refinement can put more
            # than one neighboring link beneath the SAME physical finger.
            # Derive expected contact identity from the final pad footprint,
            # never from an arbitrary segment count. Native guards unchanged.
            if getattr(self.rig,'stem_contact_model','flush_capsules_v1') in ('continuous_internal_capsules_v1','flat_cylinders_v1'):
                aperture=max(0.,self.radius-self.grasp_compression)
                self.planning_slides={'gripper_finger_l1':-aperture,'gripper_finger_l2':aperture}
                final_q=self.path_q[int(np.argmin(abs(self.fractions-1.)))]
                result['physical_grasp_span']=screen.set_physical_grasp_span(
                    frames,self.body_world(final_q,self.right),self.body_index,self.grasp_point(frames))
            if getattr(self,'native_static_clearance',False):
                if not hasattr(self,'robot'):
                    raise RuntimeError('Native grasp refinement requires an initialized robot scene')
                from .native_static_clearance import current_scene_query
                result['native_static_clearance']=dict(initialization_status='in_progress',
                    query_count=None,final_validation_passed=False)
                native=current_scene_query(self.stage,screen.static,lazy_coverage=True,
                    wall_limit_s=self.native_static_planning_seconds)
                screen.native_static_query=native
            # The left-arm cache does not cover the torso/base or parked right
            # arm. These can hit distal leaves even when the hand path clears.
            # Keep that bounded static-context screen AND independently check
            # every robot collider against the full current dynamic target.
            from .whole_robot_target import WholeRobotTargetScreen
            target_screen=WholeRobotTargetScreen(screen,self.self_screen.shapes)
            def check_proposal(q):
                world=self.body_world(q,self.right)
                complete=target_screen.check(world,grasp=True)
                result['complete_robot_target']=complete
                if not complete['passed']:
                    screen.last_failure=complete['failure']
                    return False
                return screen.check(world,grasp=True)
            self.planning_slides=self.slides.copy()
            for fraction,q in zip(self.fractions,self.path_q):
                if fraction>1.+1e-8: break
                checks+=1
                if not check_proposal(q):
                    result['failure']={**screen.last_failure,'phase':'approach','fraction':float(fraction)}
                    return result
            q=self.path_q[int(np.argmin(abs(self.fractions-1.)))]
            aperture=max(0.,self.radius-self.grasp_compression)
            from .pregrasp_aperture import closure_samples
            if self.effort_bounded_grasp_target:aperture=self.force_closer.minimum
            for gap in closure_samples(self.pregrasp_half_aperture,aperture):
                self.planning_slides={'gripper_finger_l1':-gap,'gripper_finger_l2':gap}
                checks+=1
                if not check_proposal(q):
                    result['failure']={**screen.last_failure,'phase':'closure','aperture_m':float(gap)}
                    return result
            result['passed']=True
            return result
        except BaseException as exc:
            result['passed']=False;planning_error=exc
            raise
        finally:
            if previous is None:
                if hasattr(self,'planning_slides'):del self.planning_slides
            else: self.planning_slides=previous
            result['checks']=checks
            self.grasp_scene_screen=result
            if native is not None:
                cleanup_errors=[]
                if result['passed']:
                    try:native.validate()
                    except Exception as exc:cleanup_errors.append(exc)
                try:native.close()
                except Exception as exc:cleanup_errors.append(exc)
                try:result['native_static_clearance']=native.report()
                except Exception as exc:cleanup_errors.append(exc)
                finally:screen.native_static_query=None
                if cleanup_errors:
                    result['passed']=False
                    detail='Native grasp clearance validation/cleanup failed: '+str([str(e) for e in cleanup_errors])
                    if planning_error is not None:planning_error.add_note(detail)
                    else:raise RuntimeError(detail) from cleanup_errors[0]

    def bind(self,simulation_view):
        self.release_grasp_observer()
        self._released_grasp_contacts=None
        super().bind(simulation_view)
        if getattr(self,'force_closure_enabled',False):
            from .force_closure import ForceClosure
            self.force_closer=ForceClosure(self.radius,self.grasp_compression,retention_preload=self.retention_preload,
                symmetric=self.symmetric_finger_closure,pregrasp_half_aperture=self.pregrasp_half_aperture,
                effort_bounded_target=self.effort_bounded_grasp_target,preload_force_servo=self.preload_force_servo,
                physics_hz=self.diagnostic_physics_hz)
        self.right_indices=[self.names.index(f'right_arm_{i}') for i in range(7)]
        if getattr(self,'finger_coupling_trial',None) is not None:
            self.force_closer.physical_gear_coupling_modeled=True
        if getattr(self,'explicit_finger_effort',False):
            from .finger_effort import FingerEffort
            self.finger_effort=FingerEffort(self)
        self.right_palm=simulation_view.create_rigid_body_view(self.knife.wrist_path)
        if self.right_palm.count!=1: raise RuntimeError('Missing native right wrist')
        self.event_monitor.tool_contact=self._tool_contact
        from .shaft_grasp_native import ShaftGraspNative,NATIVE37_SENSOR_CONTRACT
        # Physical support uses signed callbacks, not unsigned tensor magnitudes.
        # Keep the historical strict fault-capture mode separately reproducible.
        evidence_options=(dict(diagnostic_noncompressive_report=True)
            if getattr(self,'diagnostic_grasp_contacts',False) else
            dict(allow_signed_native_normals=True,sensor_contract=NATIVE37_SENSOR_CONTRACT))
        if self.physical_grasp_span:evidence_options['physical_grasp_span']=True
        if self.diagnostic_physics_hz!=240:evidence_options['physics_hz']=self.diagnostic_physics_hz
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
        options={}
        if getattr(self,'grasp_contact_frames','post_fetch_legacy')=='pre_solve_pgs_v1':
            options['require_pre_step_frames']=True
        result=self.grasp_observer.evaluate(dt,frames,fingers,frames_step_id=step_id,**options)
        if self.rig.cut and getattr(self,'cut_strategy','bimanual')=='bimanual':
            if getattr(self,'_released_grasp_contacts',None) is None:
                from .released_grasp import ReleasedGraspContacts
                self._released_grasp_contacts=ReleasedGraspContacts(self)
            result=self._released_grasp_contacts.classify(result)
        self.latest_finger_bilateral=result['bilateral']
        return result

    def close(self,fraction,*,step=None,dt=None):
        # Geometry-bounded closure for this privileged shaft fixture. Driving
        # to a zero-width aperture keeps compressing a ~6 mm stem after grasp.
        # Default stop is 0.5 mm inside radius; the bounded diagnostic bias
        # can be qualified separately. Force and penetration guards
        # remain unchanged and actual opposing contact still verifies grasp.
        if not np.isfinite(fraction) or not 0<=fraction<=1: raise ValueError('Invalid finger closure')
        if getattr(self,'force_closure_enabled',False):
            gaps=self.force_closer.command(fraction,step=step,dt=dt)
            self.force_limits[0,self.finger_indices]=np.minimum(
                self.force_limits[0,self.finger_indices],self.force_closer.drive_limit_n)
            self.robot.set_dof_max_forces(self.force_limits,self.index)
            if getattr(self,'finger_target_antiwindup',False):
                from .finger_target_antiwindup import project
                gaps,receipt=project(gaps,self.robot.get_dof_positions()[0,self.finger_indices],
                    self.robot.get_dof_velocities()[0,self.finger_indices],self.force_limits[0,self.finger_indices],
                    minimum=self.force_closer.minimum,retention_preload=self.force_closer.retention_preload,
                    symmetric=self.force_closer.symmetric,maximum=self.force_closer.opening)
                self.force_closer.gaps=gaps.copy()
                self.force_closer.receipt.update(half_gaps_m=gaps.tolist(),antiwindup=receipt)
            self.targets[0,self.finger_indices]=np.array([-1.,1.])*gaps
            self.robot.set_dof_position_targets(self.targets,self.index)
            return
        aperture=max(0.,self.radius-getattr(self,'grasp_compression',.0005))
        super().close(fraction*(1-aperture/self.pregrasp_half_aperture))

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
        if getattr(self,'cut_style','legacy')=='downward' and not getattr(self,'_joint_transit_proposal',False):
            from .downward_cut import cartesian_transit
            self.last_transit_rejection={}
            if self.staged_downward_transit:
                modes=getattr(self,'screened_transit_modes',())
                if not modes:raise RuntimeError('No current screened wrist transit schedule')
                failures=[]
                for mode in modes:
                    detail={}
                    result=cartesian_transit(self,left_q,goal,replan_stroke_from_endpoint=True,
                        diagnostics=detail,mode=mode)
                    if result is not None:return result
                    failures.append(dict(mode=mode,detail=detail))
                self.last_transit_rejection=dict(reason='screened_wrist_schedules_failed_full_arm_path',attempts=failures)
                return None
            return cartesian_transit(self,left_q,goal,replan_stroke_from_endpoint=True,
                diagnostics=self.last_transit_rejection)
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
        search={};options=({'diagnostics':search} if getattr(self,'_joint_transit_proposal',False) else {})
        try:path=connect_path(self.right,goal,lower,upper,valid,**options)
        finally:
            if options:self.last_transit_rejection=dict(reason='bounded_joint_search_result',search=dict(search),
                last_plant_failure=getattr(self.held_plant_screen,'last_failure',None))
            if options and isinstance(getattr(self,'plan_diagnostics',None),dict):
                self.plan_diagnostics['last_joint_search']=self.last_transit_rejection
        if path is not None:
            minimum=min(self.kin.inter_arm_clearance(left_q,q,self.base).clearance_m for q in path)
            return path,minimum,dict(method='bounded_bidirectional_joint_search',seed=0,search=search,
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
        from .blade_aim import edge_centre
        aim=edge_centre(centre,axis,getattr(self,'blade_axial_aim_offset_m',0.))
        self.plan_diagnostics.update(actual_seam_world=centre.tolist(),blade_aim_world=aim.tolist(),
            blade_axial_aim_offset_m=getattr(self,'blade_axial_aim_offset_m',0.),release_seam_or_tolerance_changed=False)
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
                if getattr(self,'native_capsule_sphere_cover',False):query_options['capsule_sphere_cover']=True
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
        downward=getattr(self,'cut_style','legacy')=='downward'
        if downward:
            from .downward_cut import downward_direction,downward_angles,vertical_cut_frame
            direction=downward_direction(axis)
            self.plan_diagnostics['downward_direction_world']=direction.tolist()
            self.plan_diagnostics['minimum_right_arm_extension']=.8
            self.plan_diagnostics['maximum_right_arm_extension']=.98
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
        tilts=(single['plane_tilt_degrees'],) if single is not None else ((0.,-10.,10.,-15.,15.) if downward else (0.,-10.,10.))
        from .cut_priority import tilt_order,proposal_order
        priority=getattr(self,'cut_priority',None)
        tilts=tilt_order(tilts,priority)
        for tilt in tilts:
            usable_wing=float(self.knife.size[1]/2-.005)
            proposals=([(single_angle,single['normal_sign'],single['wing_m'])] if single is not None else
                [(a,s,w) for w in (0.,-usable_wing/2,usable_wing/2,-.9*usable_wing,.9*usable_wing,-usable_wing,usable_wing) for s in (1,-1)
                    for a in (0,15,-15,30,-30,45,-45,60,-60,90,-90,120,-120,135,-135,150,-150,180)])
            if downward:
                wings=(0.,-usable_wing/2,usable_wing/2,-usable_wing,usable_wing)
                # First try a true vertical stroke within the EXISTING native
                # angular gate. The transverse fan can place the waiting
                # plate back into the parent on an upward-sloping petiole.
                proposals=[(None,s,w) for w in wings for s in (1,-1)
                    if vertical_cut_frame(axis,s,tilt) is not None]
                if getattr(self,'cut_model',LEGACY_CUT_MODEL)!=DOWNWARD_CUT_MODEL:
                    proposals += [(a,s,w) for w in wings
                        for s in (1,-1) for a in downward_angles(axis)]
            proposals=proposal_order(proposals,tilt,priority)
            for degrees,normal_sign,wing in proposals:
                vertical=downward and degrees is None
                if vertical:d,normal=vertical_cut_frame(axis,normal_sign,tilt)
                elif single is not None:d=single_direction.copy()
                else:
                    angle=np.radians(degrees)
                    d=direction*np.cos(angle)+np.cross(axis,direction)*np.sin(angle)
                if not vertical:normal=cut_plane_normal(d,normal_sign*axis,tilt)
                # The lower-rim frame maps source +Z (arc) to edge +X (up).
                # Both horizontal headings are proposals, not contact approval.
                desired=self.knife.wrist_for_edge(aim+self.stroke_offsets[0]*d,d,normal,wing)
                attempt=dict(angle=degrees,normal_sign=normal_sign,wing_m=wing,plane_tilt_degrees=tilt,
                    ik_attempted=False,ik_succeeded=False,evaluations=0)
                if downward:
                    attempt.update(stroke_basis='world_vertical' if vertical else 'stem_transverse',
                        planned_joint_limit_reserve_degrees=3.,
                        direction_world=d.tolist(),stroke_axis_dot_stem=float(abs(d@axis)),
                        edge_axis_dot_stem=float(abs(np.cross(normal,-d)@axis)))
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
                if self.staged_downward_transit:
                    from .wrist_transit import try_modes
                    self.screened_transit_modes=()
                    start=self.kin.forward('right',self.right,self.base)
                    detail={};solution=None;full_modes=[]
                    attempt['rigid_tool_transits']=detail
                    attempt['full_path_modes_attempted']=full_modes
                    def accept_mode(mode):
                        nonlocal solution
                        if solution is None:
                            solution=self.solve_right_pose(desired,self.right)
                            attempt.update(ik_attempted=True,ik_succeeded=solution.succeeded,
                                position_error_m=solution.position_error_m,
                                orientation_error_rad=solution.orientation_error_rad,
                                evaluations=solution.evaluations,endpoint_is_pose_seed_only=True)
                        if not solution.succeeded:return False
                        self.screened_transit_modes=(mode,);full_modes.append(mode)
                        candidate=(float(np.linalg.norm(np.asarray(solution.joint_degrees)-self.right)),
                            degrees,d,np.asarray(solution.joint_degrees),normal_sign,wing,normal)
                        return self._try_cut_candidate(left_q,centre,axis,candidate,tilt,failures)
                    if try_modes(rigid_screen,start,desired,accept_mode,detail):
                        self.plan['stroke_basis']=attempt['stroke_basis'];return
                    if getattr(self,'joint_transit_fallback',False):
                        # Cartesian template failure is not proof that no
                        # collision-free joint path exists. Change APPROACH
                        # search only: rebuild and validate the same extended,
                        # correctly oriented downward stroke afterward.
                        if solution is None:solution=self.solve_right_pose(desired,self.right)
                        attempt.update(ik_attempted=True,ik_succeeded=solution.succeeded,
                            evaluations=solution.evaluations)
                        attempt['joint_fallback_attempts']=[]
                        if solution.succeeded:
                            from itertools import chain
                            from .redundant_ik import pose_family
                            family=chain((solution,),pose_family(self.kin,'right',desired,
                                np.asarray(solution.joint_degrees),self.base,steps_per_direction=32,
                                joint_limit_margin_degrees=3.))
                            self._joint_transit_proposal=True
                            try:
                                for member in family:
                                    q=np.asarray(member.joint_degrees)
                                    entry=dict(joint_degrees=q.tolist(),endpoint_clear=False)
                                    attempt['joint_fallback_attempts'].append(entry)
                                    entry['interarm_clearance_m']=self.kin.inter_arm_clearance(left_q,q,self.base).clearance_m
                                    if entry['interarm_clearance_m']<.01:entry['rejection']='interarm';continue
                                    checked=self.check_self(left_q,q)
                                    if not checked['passed']:
                                        entry.update(rejection='robot_self',self_screen=checked);continue
                                    if not self.check_held_plant(left_q,q):
                                        entry.update(rejection='plant_or_scene',plant_screen=self.held_plant_screen.last_failure);continue
                                    entry['endpoint_clear']=True
                                    candidate=(float(np.linalg.norm(q-self.right)),degrees,d,q,normal_sign,wing,normal)
                                    if self._try_cut_candidate(left_q,centre,axis,candidate,tilt,failures):
                                        self.plan['stroke_basis']=attempt['stroke_basis']
                                        self.plan['approach_joint_fallback']=True
                                        return
                            finally:self._joint_transit_proposal=False
                    attempt['rejection']=('rigid_tool_transit' if solution is None else
                        'endpoint_IK' if not solution.succeeded else 'actual_transit_or_rebuilt_stroke')
                    continue
                solution=self.solve_right_pose(desired,self.right)
                attempt.update(ik_attempted=True,
                    position_error_m=solution.position_error_m,orientation_error_rad=solution.orientation_error_rad,
                    evaluations=solution.evaluations,ik_succeeded=solution.succeeded)
                if not solution.succeeded:
                    attempt['rejection']='endpoint_IK';continue
                family=(solution,)
                if downward:
                    from itertools import chain
                    from .redundant_ik import pose_family
                    # One converged IK branch can intersect a held leaf while
                    # the identical wrist pose has a clear elbow configuration.
                    # These are proposals only: every member must pass ALL
                    # original endpoint, stroke, transit and native checks.
                    family=chain((solution,),pose_family(self.kin,'right',desired,
                        np.asarray(solution.joint_degrees),self.base,
                        steps_per_direction=32,joint_limit_margin_degrees=3.))
                base_attempt=dict(attempt)
                for family_index,solution in enumerate(family):
                    if family_index:
                        attempt=dict(base_attempt);attempts.append(attempt)
                    q=np.array(solution.joint_degrees)
                    attempt.update(endpoint_family_index=family_index,joint_degrees=q.tolist(),
                        position_error_m=solution.position_error_m,orientation_error_rad=solution.orientation_error_rad,
                        evaluations=solution.evaluations,ik_succeeded=solution.succeeded)
                    if downward:
                        from .downward_cut import arm_extension
                        attempt['right_arm_extension']=arm_extension(self.body_world(left_q,q))
                        if not .8<=attempt['right_arm_extension']<=.98:
                            attempt['rejection']='folded_or_fully_extended_right_arm';continue
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
                        if self._try_cut_candidate(left_q,centre,axis,candidate,tilt,failures):
                            if downward:self.plan['stroke_basis']=attempt['stroke_basis']
                            return
                    else: attempt['rejection']='endpoint_self_collision'
        ik=sum(a['ik_succeeded'] for a in attempts)
        raise RuntimeError(f'No bimanual arm-clearance path: endpoints={len(attempts)}, '
            f'IK_attempted={sum(a["ik_attempted"] for a in attempts)}, IK_converged={ik}, '
            f'arm_clear_endpoints={clear_endpoints}, path_failures={failures}')

    def _try_cut_candidate(self,left_q,centre,axis,candidate,tilt,failures):
        """Complete transit/stroke validation; no endpoint-only success.

        Downward Cartesian transit selects its own redundant-arm branch. Rebuild
        the stroke from that exact terminal state, not a disconnected IK branch.
        Legacy joint-space transit retains the original stroke-first protocol.
        """
        _,angle,d,q,normal_sign,wing,normal=candidate
        from .blade_aim import edge_centre
        aim=edge_centre(centre,axis,getattr(self,'blade_axial_aim_offset_m',0.))
        failure=dict(angle=angle,plane_tilt_degrees=tilt,normal_sign=normal_sign,wing_m=wing)
        minimum=float('inf');stroke=[];seed=q;transit=None
        downward=getattr(self,'cut_style','legacy')=='downward'
        if downward:
            transit=self.right_transit(left_q,q)
            if transit is None:
                failures.append(dict(failure,rejection='bounded_transit_arm_self_or_plant_clearance',
                    transit_detail=getattr(self,'last_transit_rejection',None)));return False
            approach,_,_=transit
            seed=np.asarray(approach[-1],float).copy()
        for sample_index,offset in enumerate(self.stroke_offsets):
            desired=self.knife.wrist_for_edge(aim+offset*d,d,normal,wing)
            if downward and sample_index==0:
                from scipy.spatial.transform import Rotation
                actual=self.kin.forward('right',seed,self.base)
                error=float(np.linalg.norm(actual[:3,3]-desired[:3,3]))
                angle_error=float(np.linalg.norm(Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec()))
                if not np.isfinite([error,angle_error]).all() or error>.0005 or angle_error>.005:
                    failures.append(dict(failure,rejection='transit_terminal_cut_pose',position_error_m=error,
                        orientation_error_rad=angle_error));return False
                # Reuse the very same joint vector: no second IK solve, snap,
                # or unchecked interpolation at the approach-to-cut boundary.
                from types import SimpleNamespace
                solution=SimpleNamespace(succeeded=True,joint_degrees=seed)
            else:solution=self.solve_right_pose(desired,seed)
            failure.update(offset_m=float(offset),rejection='stroke_IK')
            if not solution.succeeded:
                # Exact failed solve, not the preceding sample's collision checks.
                failure.update(sample_index=sample_index,ik_result=dict(
                    joint_degrees=np.asarray(solution.joint_degrees).tolist(),
                    position_error_m=float(solution.position_error_m),
                    orientation_error_rad=float(solution.orientation_error_rad),
                    evaluations=None if solution.evaluations is None else int(solution.evaluations),
                    succeeded=bool(solution.succeeded)))
                break
            seed=np.asarray(solution.joint_degrees)
            if getattr(self,'cut_style','legacy')=='downward':
                from .downward_cut import arm_extension
                if not .8<=arm_extension(self.body_world(left_q,seed))<=.98:
                    failure.update(rejection='folded_or_fully_extended_right_arm');break
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
        if transit is None:transit=self.right_transit(left_q,q)
        if transit is None:
            failures.append(dict(failure,rejection='bounded_transit_arm_self_or_plant_clearance'));return False
        approach,transit_minimum,transit_evidence=transit
        minimum=min(minimum,transit_minimum)
        self.plan=dict(approach=approach,stroke=np.asarray(stroke),direction=d,
            centre=centre.copy(),axis=axis.copy(),angle=angle,normal_sign=normal_sign,wing_m=wing,
            blade_aim_centre=aim.copy(),blade_axial_aim_offset_m=getattr(self,'blade_axial_aim_offset_m',0.),
            release_seam_or_tolerance_changed=False,
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

    def inspect_cut(self,dt,frames,held,slip,*,cut_only_ready=False):
        from .runtime import pose_matrices
        if self.edge_contact_error is not None: raise RuntimeError(self.edge_contact_error)
        if not self.event_monitor.native_full_contact_reporting:
            raise RuntimeError('Full original-order native contact stream required for blade evidence')
        loads=self.event_monitor.measurements(dt)
        from .knife_load import physical_load
        physical=physical_load(self.event_monitor.pairs,self.edge_contact_rows,root=self.knife.root,dt=dt)
        tool_upper=max(loads['allowed_tool_contact_n'],physical['upper_bound_n'])
        if tool_upper>.5:
            raise RuntimeError('Unsigned normal-plus-friction tool load exceeds 0.5 N')
        if physical['minimum_separation_m']<-.001:
            raise RuntimeError('Physical knife penetration exceeds 1 mm, including noncutting contacts')
        actual=pose_matrices(self.right_palm.get_transforms())[0]
        error=float(np.linalg.norm(actual[:3,3]-self.expected_right[:3,3]))
        if error>.012: raise RuntimeError(f'Right wrist tracking error {error:.5f} m')
        edge=self.knife.frame(actual);centre,axis=self.seam(frames)
        if getattr(self,'cut_style','legacy')=='downward' and self.cut_authorized:
            if (-edge[:3,0])[2]>-np.cos(np.radians(30)):
                raise RuntimeError('Measured cutting direction is not downward; release refused')
        orientation={}
        if self.cut_gate.parameters.model==DOWNWARD_CUT_MODEL:
            orientation=dict(arc_up=self.knife.arc_up(actual),edge_mode=self.knife.edge_mode)
        decision=self.cut_gate.observe(dt=dt,edge=edge,centre=centre,axis=axis,
            points=self.edge_points,impulses=self.edge_impulses,normals=self.edge_normals,
            impulse_contract=KNIFE_IMPULSE_CONTRACT,edge_contact_verified=True,
            tool_contact_upper_bound_n=tool_upper,
            held=held and self.cut_authorized,slip=slip,
            cut_only_ready=cut_only_ready and self.cut_authorized,**orientation)
        if decision:
            transition=getattr(self,'root_transition',None)
            options={}
            if transition is not None:options['transition']=transition
            if getattr(self,'cut_strategy','bimanual')=='right_only':options['strategy']='right_only'
            self.cut_event=self.rig.release_from_blade(decision,**options)
        return dict(edge_frame=edge.tolist(),right_tracking_error_m=error,
            edge_contact_count=len(self.edge_points),edge_force_n=math.fsum(math.hypot(*v) for v in self.edge_impulses)/dt,
            edge_signed_resistance_n=self.cut_gate.signed_resistance_n,
            edge_unsigned_projection_n=self.cut_gate.unsigned_projection_n,
            tool_contact_upper_bound_n=tool_upper,physical_knife_contact=physical,
            force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows=[dict(r) for r in self.edge_contact_rows],
            raw_normal_row_limit=256,raw_normal_rows_complete=True,
            cut_model=self.cut_gate.parameters.model,
            loading_geometry_verified=self.cut_gate.diagnostic.get('loading_geometry_verified',False),
            loading_travel_required=self.cut_gate.parameters.travel_required,
            gate_diagnostic=dict(self.cut_gate.diagnostic),
            gate_dwell_s=self.cut_gate.dwell,gate_travel_m=self.cut_gate.travel,cut_event=self.cut_event)

    def restore_authored_state(self):
        self._released_grasp_contacts=None
        self.release_grasp_observer()
        super().restore_authored_state()
        self.cut_gate=ShearGate(self.rig.source_target,ShearParameters(model=getattr(self,'cut_model',LEGACY_CUT_MODEL)),
            strategy=getattr(self,'cut_strategy','bimanual'))
        self.cut_authorized=False
        self.edge_points=[];self.edge_impulses=[];self.edge_normals=[];self.edge_contact_rows=[];self.edge_contact_error=None
        self.cut_event=None;self.plan=None;self.plan_diagnostics=None;self.cut_contacts=0
        self.screened_transit_modes=()
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
        # Opposed external inspection angles expose the cutting plane when
        # the original wrist/main-stem view is occluded. These are diagnostic
        # cameras only; existing robot D405 views/calibration stay unchanged.
        for sign,label in ((1.,'front'),(-1.,'back')):
            eye=centre+sign*.22*edge[:3,2]+.045*edge[:3,1]+.05*edge[:3,0]
            path='/World/BladePlaneInspection_'+label
            camera=UsdGeom.Camera.Define(self.stage,path)
            camera.CreateFocalLengthAttr(28.);camera.CreateClippingRangeAttr(Gf.Vec2f(.003,10))
            xf=UsdGeom.Xformable(camera);xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*centre),Gf.Vec3d(0,0,1)).GetInverse())
            self.views['Blade plane '+label]=path

    def report(self):
        result=super().report()
        result['staged_downward_transit']=self.staged_downward_transit
        result['cut_candidate_priority']=getattr(self,'cut_priority',None)
        result.update(right_arm='original_fitted_knife_guarded_native_joint_drives',
            grasp_closure=dict(mode='native_force_closure_v1' if getattr(self,'force_closure_enabled',False) else 'ground_truth_shaft_width_stop_with_native_contact_verification',
                commanded_half_aperture_m=None if getattr(self,'force_closure_enabled',False) else max(0.,self.radius-self.grasp_compression),
                minimum_commanded_half_aperture_m=self.force_closer.minimum if getattr(self,'force_closure_enabled',False) else max(0.,self.radius-self.grasp_compression),
                nominal_pad_compression_m=self.grasp_compression,
                maximum_nominal_target_bias_m=self.force_closer.nominal_target_bias if getattr(self,'force_closure_enabled',False) else self.grasp_compression,
                effort_bounded_position_reference=self.effort_bounded_grasp_target,
                preload_force_servo=self.preload_force_servo,
                actual_native_penetration_guard_m=.001,material_calibrated=False),
            grasp_evidence_model='exact_connected_detached_shaft_inner_pad_normal_contacts_with_selected_tensor_crosscheck',
            physical_grasp_span=self.physical_grasp_span,
            diagnostic_grasp_contacts=getattr(self,'diagnostic_grasp_contacts',False),
            minimum_grasp_self_capsule_clearance_m=self.minimum_grasp_self_clearance,
            knife_mount=self.knife_mount,
            blade_contact_geometry=self.blade_contacts,
            wrist_contact_geometry=getattr(self,'wrist_contacts',None),
            arc_contact_geometry=self.arc_contacts,
            planning_wall_seconds=getattr(self,'planning_wall_seconds',None),
            cut_plan_diagnostics=self.plan_diagnostics,
            right_ik_fixed_joint=getattr(self,'right_ik_fixed_joint',None),
            right_ik_policy='first_fully_screened_path_not_shortest_path',
            cut_style=getattr(self,'cut_style','legacy'),
            closure_control='native_force_closure_v1' if getattr(self,'force_closure_enabled',False) else 'geometric_compression',
            finger_target_antiwindup=getattr(self,'finger_target_antiwindup',False),
            retention_preload=self.retention_preload,symmetric_finger_closure=self.symmetric_finger_closure,
            live_grasp_placement=getattr(self,'live_grasp_placement',None),
            cut_model=getattr(self,'cut_model',LEGACY_CUT_MODEL),
            cut_model_class='measured_contact_seam_failure_not_calibrated_tissue_cutting')
        if self.plan is not None:
            result['cut_plan']={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                for k,v in self.plan.items() if k not in ('approach','stroke')}
        return result
