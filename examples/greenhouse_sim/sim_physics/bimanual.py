"""Guarded full-robot knife test. Privileged fixture, not a learned controller."""
import time
import numpy as np

from .full_robot import FullRobotGripper
from .knife import KnifeGeometry,ShearGate,mount_forward,cut_plane_normal,transverse_stroke_offsets,leading_face_normal


class BimanualRobot(FullRobotGripper):
    def __init__(self,*args,**kwargs):
        standoff=kwargs.pop('cut_standoff',.025)
        self.grasp_compression=kwargs.pop('grasp_compression',.0005)
        if not np.isfinite(self.grasp_compression) or not .00025<=self.grasp_compression<=.001:
            raise ValueError('Diagnostic pad compression must be finite 0.25..1 mm')
        kwargs.setdefault('station_offset',(0.,0.))
        kwargs.setdefault('approach_side',1)
        super().__init__(*args,**kwargs)
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
        self.cut_gate=ShearGate(self.rig.source_target)
        self.cut_authorized=False;self.edge_points=[];self.edge_impulses=[]
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
        super().bind(simulation_view)
        self.right_indices=[self.names.index(f'right_arm_{i}') for i in range(7)]
        self.right_palm=simulation_view.create_rigid_body_view(self.knife.wrist_path)
        if self.right_palm.count!=1: raise RuntimeError('Missing native right wrist')
        self.event_monitor.tool_contact=self._tool_contact

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

    def _tool_contact(self,robot,other,point,impulse,normal,separation):
        # Only the flat leading strip contacting the two seam-adjacent shaft
        # capsules is an expected tool load. Arc, camera, main stem and leaves
        # remain unwanted contacts. Positions come from the native callback.
        eligible=[self.rig.body_paths[i]+'/StemCollider'
            for i in (self.rig.cut_index-1,self.rig.cut_index)]
        if (not self.cut_authorized or robot!=self.knife.collider or other not in eligible
                or not self.knife.on_edge(point,self.edge_frame)
                or not leading_face_normal(normal,-self.edge_frame[:3,0])):
            return False
        centre,axis=self.cut_frame
        if abs(np.dot(np.asarray(point)-centre,axis))>.003: return False
        self.edge_points.append(point);self.edge_impulses.append(impulse)
        self.cut_contacts+=1
        return True

    def prepare_step(self,frames):
        from .runtime import pose_matrices
        self.edge_frame=self.knife.frame(pose_matrices(self.right_palm.get_transforms())[0])
        self.cut_frame=self.seam(frames)
        self.edge_points=[];self.edge_impulses=[]

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
        try: return self._plan_cut(frames,left_q)
        finally: self.planning_wall_seconds=time.perf_counter()-start

    def _plan_cut(self,frames,left_q):
        """Bounded orientation search, then dense arm-pair screened IK paths.

        Native whole-scene guards remain essential: arm capsules alone do not
        certify the knife, cameras, foliage or gutter swept volume.
        """
        centre,axis=self.seam(frames)
        # Context plants/gutters are populated after robot construction. Cache
        # only once the complete scene exists, not in __init__.
        if self.held_plant_screen.workspace is None:
            self.held_plant_screen.include_static_scene(self.stage,self.root,self.rig.root,
                self.body_world(self.initial_q,self.right)['link_right_arm_0'][:3,3])
        self.held_plant_screen.snapshot(frames)
        # Finger geometry must use the held aperture, not the initial open hand.
        if hasattr(self,'robot'):
            positions=self.robot.get_dof_positions()[0]
            self.planning_slides={name:float(positions[self.names.index(name)]) for name in self.slides}
        else:
            # Offline fixture-only plan; native execution always snapshots above.
            self.planning_slides={'gripper_finger_l1':-self.radius,'gripper_finger_l2':self.radius}
        from .rigid_tool_screen import RigidToolScreen
        rigid_screen=RigidToolScreen(self,left_q)
        direction=-self.goal[:3,2];direction-=axis*np.dot(direction,axis);direction/=np.linalg.norm(direction)
        attempts=[];failures=[]
        self.plan=None
        self.plan_diagnostics=dict(endpoint_attempts=attempts,path_failures=failures,
            minimum_required_interarm_m=.01,held_plant_margin_m=.001,
            held_plant_native_snapshot_screened=True,local_static_colliders=len(self.held_plant_screen.static),
            local_scene_bounds_m=[v.tolist() for v in self.held_plant_screen.workspace],
            complete_tool_stroke_screened_before_IK=True,
            whole_scene_path_certified=False)
        # Try the original plane first; only then small oblique planes within
        # the existing measured angular gate. Never replace actual stem truth
        # with the proposed blade normal when evaluating native contact.
        clear_endpoints=0
        for tilt in (0.,-10.,10.):
            candidates=[]
            usable_wing=float(self.knife.size[1]/2-.005)
            for degrees,normal_sign,wing in [(a,s,w) for w in (0.,-usable_wing/2,usable_wing/2,-.9*usable_wing,.9*usable_wing,-usable_wing,usable_wing) for s in (1,-1)
                    for a in (0,15,-15,30,-30,45,-45,60,-60,90,-90,120,-120,135,-135,150,-150,180)]:
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
                solution=self.kin.solve_pose('right',desired,self.right,self.base,maximum_evaluations=250)
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
                    candidates.append((np.linalg.norm(q-self.right),degrees,d,q,normal_sign,wing,normal))
                    clear_endpoints+=1
                else: attempt['rejection']='endpoint_self_collision'
            for _,angle,d,q,normal_sign,wing,normal in sorted(candidates,key=lambda v:v[0]):
                failure=dict(angle=angle,plane_tilt_degrees=tilt,normal_sign=normal_sign,wing_m=wing)
                minimum=float('inf')
                stroke=[];seed=q
                for offset in self.stroke_offsets:
                    desired=self.knife.wrist_for_edge(centre+offset*d,d,normal,wing)
                    solution=self.kin.solve_pose('right',desired,seed,self.base,maximum_evaluations=250)
                    failure.update(offset_m=float(offset),rejection='stroke_IK')
                    if not solution.succeeded: break
                    seed=np.asarray(solution.joint_degrees)
                    clearance=self.kin.inter_arm_clearance(left_q,seed,self.base).clearance_m
                    minimum=min(minimum,clearance)
                    failure.update(rejection='stroke_arm_clearance',interarm_clearance_m=clearance)
                    if clearance<.01: break
                    check=self.check_self(left_q,seed)
                    failure.update(rejection='stroke_self_collision',self_screen=check)
                    if not check['passed']: break
                    if not self.check_held_plant(left_q,seed,stroke=True):
                        failure.update(rejection='stroke_held_plant',plant_screen=self.held_plant_screen.last_failure)
                        break
                    stroke.append(seed)
                if len(stroke)!=len(self.stroke_offsets):
                    failures.append(failure);continue
                # Reject unusable stroke endpoints before spending the bounded
                # joint-space transit budget. Both use the same native snapshot.
                transit=self.right_transit(left_q,q)
                if transit is None:
                    failures.append(dict(failure,rejection='bounded_transit_arm_self_or_plant_clearance'));continue
                approach,transit_minimum,transit_evidence=transit
                minimum=min(minimum,transit_minimum)
                self.plan=dict(approach=approach,stroke=np.asarray(stroke),direction=d,
                    centre=centre.copy(),axis=axis.copy(),angle=angle,normal_sign=normal_sign,wing_m=wing,
                    blade_plane_normal=normal.copy(),plane_tilt_degrees=tilt,
                    stroke_offset_range_m=[float(self.stroke_offsets[0]),float(self.stroke_offsets[-1])],
                    stroke_samples=len(self.stroke_offsets),stroke_end_basis='shaft_radius_plus_half_edge_strip_plus_1mm',
                    minimum_interarm_m=minimum,transit=transit_evidence)
                return
        ik=sum(a['ik_succeeded'] for a in attempts)
        raise RuntimeError(f'No bimanual arm-clearance path: endpoints={len(attempts)}, '
            f'IK_attempted={sum(a["ik_attempted"] for a in attempts)}, IK_converged={ik}, '
            f'arm_clear_endpoints={clear_endpoints}, path_failures={failures}')

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
        actual=pose_matrices(self.right_palm.get_transforms())[0]
        error=float(np.linalg.norm(actual[:3,3]-self.expected_right[:3,3]))
        if error>.012: raise RuntimeError(f'Right wrist tracking error {error:.5f} m')
        edge=self.knife.frame(actual);centre,axis=self.seam(frames)
        decision=self.cut_gate.observe(dt=dt,edge=edge,centre=centre,axis=axis,
            points=self.edge_points,impulses=self.edge_impulses,held=held and self.cut_authorized,slip=slip)
        if decision:
            self.cut_event=self.rig.release_from_blade(decision)
        return dict(edge_frame=edge.tolist(),right_tracking_error_m=error,
            edge_contact_count=len(self.edge_points),edge_force_n=sum(np.linalg.norm(v) for v in self.edge_impulses)/dt,
            gate_dwell_s=self.cut_gate.dwell,gate_travel_m=self.cut_gate.travel,cut_event=self.cut_event)

    def restore_authored_state(self):
        super().restore_authored_state()
        self.cut_gate=ShearGate(self.rig.source_target);self.cut_authorized=False
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
            minimum_grasp_self_capsule_clearance_m=self.minimum_grasp_self_clearance,
            knife_mount=self.knife_mount,
            blade_contact_geometry=self.blade_contacts,
            arc_contact_geometry=self.arc_contacts,
            planning_wall_seconds=getattr(self,'planning_wall_seconds',None),
            cut_plan_diagnostics=self.plan_diagnostics,
            cut_model='measured_contact_seam_failure_not_calibrated_tissue_cutting')
        if self.plan is not None:
            result['cut_plan']={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                for k,v in self.plan.items() if k not in ('approach','stroke')}
        return result
