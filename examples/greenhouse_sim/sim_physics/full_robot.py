"""Full dynamic RB-Y1 diagnostic. Joint-drive IK, native finger contact, no weld.

This is a privileged test station (isolated or in the supplied greenhouse),
not observation-driven execution, general plant manipulation, tissue cutting
or a training-data collector.
"""
import json
import time
import xml.etree.ElementTree as ET

import numpy as np

from .gripper_probe import GripperFixture


def finger_force_budget(gravity):
    """Reserve native gravity effort inside, not in addition to, the 0.5 N cap."""
    gravity=np.asarray(gravity,dtype=float)
    if gravity.shape!=(2,) or not np.isfinite(gravity).all() or np.any(np.abs(gravity)>.4):
        raise ValueError('Finger gravity leaves insufficient bounded grasp effort')
    return .5-np.abs(gravity)


class FullRobotGripper(GripperFixture):
    def __init__(self,stage,rig,*,arc=.08,friction=.5,ground_height=None,
                 torso_degrees=None,sparse_contacts=False,floor_root=None,finger_gravity=False,approach_tilt=0.,station_offset=(0.,0.),approach_side=1,approach_vector=(1.,-1.,.2)):
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
        from greenhouse_sim.robot_model import DEFAULT_ASSET,DEFAULT_URDF
        from greenhouse_sim.robot_kinematics import Rby1Kinematics,base_transform
        from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES
        from .plant import matrix_attr,physics_schema
        self.stage,self.rig,self.asset=stage,rig,DEFAULT_ASSET
        self.root='/World/RBY1'
        self.kin=Rby1Kinematics()
        if torso_degrees is not None: self.kin.set_default_torso_degrees(torso_degrees)
        self.sparse_contacts=sparse_contacts;self.event_monitor=None;self.floor_root=floor_root
        self.finger_gravity=finger_gravity;self.finger_compensation=np.zeros(2)
        self.window=None
        self.body_index=int(np.argmin(np.abs((rig.arcs[:-1]+rig.arcs[1:])/2-arc)))
        if self.body_index<rig.cut_index or rig.arcs[self.body_index]<.035:
            raise ValueError('Grasp must leave clearance from the diagnostic seam')
        self.arc=float((rig.arcs[self.body_index]+rig.arcs[self.body_index+1])/2)
        self.grasp_path=rig.body_paths[self.body_index]
        self.half_length=float(np.linalg.norm(rig.chain_world[self.body_index+1]-rig.chain_world[self.body_index])/2)
        self.radius=float(UsdGeom.Capsule.Get(stage,self.grasp_path+'/StemCollider').GetRadiusAttr().Get())
        # Side entry avoids the fixed foliage above this shaft. A kinematic
        # palm fixture does not reveal palm-vs-static contacts; the full dynamic
        # arm does, so do not reuse its top-down approach blindly.
        point=rig.rest_frames[self.body_index,:3,3]
        y=rig.rest_frames[self.body_index,:3,2]
        z=np.asarray(approach_vector,dtype=float).copy()
        if z.shape!=(3,) or not np.isfinite(z).all(): raise ValueError('Invalid approach vector')
        z-=y*np.dot(y,z)
        if np.linalg.norm(z)<.1: raise ValueError('Approach vector nearly parallel to stem')
        z/=np.linalg.norm(z)
        if approach_side not in (-1,1): raise ValueError('Approach side must be -1 or 1')
        self.approach_side=approach_side;z*=approach_side
        if not np.isfinite(approach_tilt) or abs(approach_tilt)>30:
            raise ValueError('Diagnostic approach tilt must be within 30 degrees')
        self.approach_tilt=float(approach_tilt)
        angle=np.radians(approach_tilt)
        z=np.cos(angle)*z+np.sin(angle)*np.cross(y,z)
        self.goal=np.eye(4);self.goal[:3,:3]=np.column_stack([np.cross(y,z),y,z])
        self.goal[:3,3]=point+.1025*z
        self.start=self.goal.copy();self.start[:3,3]+=.08*self.goal[:3,2]
        # Approach from the open side. The +Y station overlapped the fixed
        # upper canopy with the torso; never disable those plant contacts.
        yaw=float(np.degrees(np.arctan2(-z[1],-z[0])))
        angle=np.radians(yaw);forward=np.array([np.cos(angle),np.sin(angle),0.])
        left=np.array([-np.sin(angle),np.cos(angle),0.])
        self.base=base_transform(self.start[:3,3]-.4*forward-.22*left,yaw);self.base[2,3]=.001
        self.station_offset=np.asarray(station_offset,dtype=float)
        if self.station_offset.shape!=(2,) or not np.isfinite(self.station_offset).all() or np.linalg.norm(self.station_offset)>.3:
            raise ValueError('Initial station adjustment must be finite and within 0.3 m')
        self.base[:3,3]+=self.station_offset[0]*forward+self.station_offset[1]*left
        if ground_height is not None: self.base[2,3]+=ground_height(*self.base[:2,3])
        self.pose=dict(SDK_READY_POSE_DEGREES)
        self.pose.update({f'torso_{i}':float(v) for i,v in enumerate(self.kin.default_torso_degrees())})
        self.right=np.array([self.pose[f'right_arm_{i}'] for i in range(7)])
        result=self.kin.solve_pose('left',self.start,[self.pose[f'left_arm_{i}'] for i in range(7)],self.base)
        if not result.succeeded: raise ValueError('Full-robot pregrasp IK failed: '+str(result))
        self.initial_q=np.array(result.joint_degrees)
        self.pose.update({f'left_arm_{i}':v for i,v in enumerate(self.initial_q)})
        self.slides={'gripper_finger_l1':-.025,'gripper_finger_l2':.025}
        self.paths=[self.root+'/'+n for n in ('ee_left','ee_finger_l1','ee_finger_l2')]
        self.collider_paths=[];self.body_paths=[];self.drives=[];self.friction=friction
        self.stop_requested=False;self.on_sample=lambda record:None
        self.expected_palm=self.start.copy()
        self.effort={j.attrib['name']:float(j.find('limit').attrib['effort'])
            for j in ET.parse(DEFAULT_URDF).getroot().findall('joint')
            if j.find('limit') is not None and 'effort' in j.find('limit').attrib}
        self.anchor=self.root+'/joints/probe_world_fixed'
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            root=UsdGeom.Xform.Define(stage,self.root).GetPrim()
            root.GetReferences().AddReference(str(self.asset));stage.Load(self.root)
            matrix_attr(root,self.base)
            for link,frame in self.kin.all_link_transforms(self.pose,prismatic_m=self.slides).items():
                prim=stage.GetPrimAtPath(self.root+'/'+link)
                if not prim: raise ValueError('Missing full robot link '+link)
                matrix_attr(prim,frame)
            material=UsdShade.Material.Define(stage,self.root+'/ProbeFingerMaterial')
            api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(friction);api.CreateDynamicFrictionAttr(friction);api.CreateRestitutionAttr(0.)
            physics_schema(material.GetPrim(),'PhysxMaterialAPI',[
                ('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token,'min')])
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI): prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    self.body_paths.append(str(prim.GetPath()))
                    body=UsdPhysics.RigidBodyAPI(prim);body.CreateRigidBodyEnabledAttr(True);body.CreateKinematicEnabledAttr(False)
                    body.CreateVelocityAttr(Gf.Vec3f(0));body.CreateAngularVelocityAttr(Gf.Vec3f(0))
                    # Anchored wheels can sit inside the floor's contact offset
                    # with zero load. Report meaningful support contacts, not
                    # zero-impulse proximity; collision itself remains enabled.
                    report_threshold=.001 if prim.GetName() in ('base','wheel_l','wheel_r') else 0.
                    physics_schema(prim,'PhysxContactReportAPI',[('physxContactReport:threshold',Sdf.ValueTypeNames.Float,report_threshold)])
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    # Preserve source-active arm, chassis and tool colliders.
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
                    physics_schema(prim,'PhysxCollisionAPI',[
                        ('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0005),
                        ('physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)])
                    self.collider_paths.append(str(prim.GetPath()))
                    if str(prim.GetPath()).startswith(tuple(p+'/' for p in self.paths[1:])):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
                if prim.IsA(UsdPhysics.Joint):
                    joint=UsdPhysics.Joint(prim);joint.CreateJointEnabledAttr(True);joint.CreateCollisionEnabledAttr(False)
            base_prim=stage.GetPrimAtPath(self.root+'/base')
            world=UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            anchor=UsdPhysics.FixedJoint.Define(stage,self.anchor)
            anchor.CreateBody1Rel().SetTargets([self.root+'/base'])
            anchor.CreateLocalPos0Attr(Gf.Vec3f(world.ExtractTranslation()))
            anchor.CreateLocalRot0Attr(Gf.Quatf(world.ExtractRotationQuat()))
            anchor.CreateLocalPos1Attr(Gf.Vec3f(0));anchor.CreateLocalRot1Attr(Gf.Quatf(1))
            UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
            physics_schema(anchor.GetPrim(),'PhysxArticulationAPI',[
                ('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,True),
                ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
                ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,8)])
            # Explicit overlapping mounting proxies, NOT a global self-collision
            # disable. Torso is held fixed in this test; this exclusion list is
            # not a validated collision model for arbitrary torso trajectories.
            self.mount_exclusions=[
                ('base','link_torso_1'),('link_torso_2','link_torso_4'),
                ('link_right_arm_5','ee_right'),('link_left_arm_5','ee_left'),
                ('link_right_arm_3','link_right_arm_5'),('link_left_arm_3','link_left_arm_5'),
                ('link_right_arm_4','ee_right'),('link_left_arm_4','ee_left')]
            for first,second in self.mount_exclusions:
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(self.root+'/'+first)).CreateFilteredPairsRel().AddTarget(self.root+'/'+second)
            self._author_initial_joints()
            if ground_height is None:
                floor=UsdGeom.Cube.Define(stage,'/World/RobotTestFloor')
                floor.CreateSizeAttr(1.)
                xf=UsdGeom.Xformable(floor);xf.AddTranslateOp().Set(Gf.Vec3d(0,0,-.04));xf.AddScaleOp().Set(Gf.Vec3f(8,8,.08))
                floor.CreateDisplayColorAttr([(0.18,0.20,0.22)])
                UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
                self.floor_root='/World/RobotTestFloor'
            self.plant_filter_paths=list(rig.body_paths)+[
                str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Plant'))
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        self.plan_approach()

    def _author_initial_joints(self):
        from pxr import Sdf,UsdPhysics
        from .plant import physics_schema
        self.drives=[]
        for name,value in {**self.pose,**self.slides}.items():
            prim=self.stage.GetPrimAtPath(self.root+'/joints/'+name)
            axis='linear' if name in self.slides else 'angular'
            drive=UsdPhysics.DriveAPI.Apply(prim,axis)
            drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(float(value));drive.CreateTargetVelocityAttr(0.)
            linear=axis=='linear'
            drive.CreateStiffnessAttr(200. if linear else 6000.*np.pi/180)
            drive.CreateDampingAttr(5. if linear else 160.*np.pi/180)
            drive.CreateMaxForceAttr(.5 if linear else .7*self.effort[name])
            physics_schema(prim,'PhysxJointStateAPI:'+axis,[
                (f'state:{axis}:physics:position',Sdf.ValueTypeNames.Float,float(value)),
                (f'state:{axis}:physics:velocity',Sdf.ValueTypeNames.Float,0.)])
            if linear: self.drives.append(drive)

    def plan_approach(self):
        """Solve before motion; interpolate only a dense, checked local IK path."""
        self.fractions=np.linspace(0,1.15,47)
        joints=[];seed=self.initial_q.copy();minimum=float('inf')
        delta=self.goal[:3,3]-self.start[:3,3]
        for fraction in self.fractions:
            desired=self.start.copy();desired[:3,3]+=fraction*delta
            solution=self.kin.solve_pose('left',desired,seed,self.base)
            if not solution.succeeded: raise RuntimeError('Approach IK fails at '+str(fraction))
            seed=np.asarray(solution.joint_degrees)
            clearance=self.kin.inter_arm_clearance(seed,self.right,self.base).clearance_m
            if clearance<.01: raise RuntimeError('Arm-to-arm capsule clearance fails')
            minimum=min(minimum,clearance);joints.append(seed)
        self.path_q=np.array(joints);self.minimum_interarm=minimum
        self.expected_palm=self.start.copy()

    def report(self):
        return dict(asset=str(self.asset),scope='full_dynamic_robot_native_joint_drives',
            arm_ik_solved=True,grasp_weld=False,plant_pose_override=False,base_fixed=True,
            robot_base_world=self.base.tolist(),grasp_body=self.grasp_path,grasp_arc_m=self.arc,
            initial_station_forward_left_offset_m=self.station_offset.tolist(),
            source_colliders_retained=len(self.collider_paths),finger_max_drive_force_n=.5,
            mounting_proxy_exclusions=self.mount_exclusions,self_collision_enabled=True,
            contact_monitor='sparse_native_events' if self.sparse_contacts else 'dense_pair_matrices',
            finger_gravity_compensation=self.finger_gravity,total_finger_effort_limit_n=.5,
            torso_degrees=self.kin.default_torso_degrees().tolist(),floor_root=self.floor_root,
            approach_tilt_degrees=self.approach_tilt,
            approach_side=self.approach_side,
            minimum_planned_interarm_capsule_clearance_m=self.minimum_interarm,
            right_arm='parked_with_original_fitted_knife_not_cutting',
            target_source='privileged_test_fixture_not_perception_verified',
            whole_path_collision_certified=False,training_eligible=False)

    def restore_authored_state(self):
        from pxr import Gf,Usd,UsdPhysics
        from .plant import matrix_attr
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for link,frame in self.kin.all_link_transforms(self.pose,prismatic_m=self.slides).items():
                prim=self.stage.GetPrimAtPath(self.root+'/'+link);matrix_attr(prim,frame)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    body=UsdPhysics.RigidBodyAPI(prim)
                    body.CreateVelocityAttr(Gf.Vec3f(0));body.CreateAngularVelocityAttr(Gf.Vec3f(0))
            self._author_initial_joints()
        self.expected_palm=self.start.copy();self.stop_requested=False
        self.finger_compensation=np.zeros(2)

    def bind(self,simulation_view):
        super().bind(simulation_view)
        self.robot=simulation_view.create_articulation_view(self.anchor)
        if self.robot.count!=1 or not self.robot.shared_metatype.fixed_base:
            raise RuntimeError('Expected one fixed-base native full robot articulation')
        self.names=list(self.robot.shared_metatype.dof_names)
        self.left_indices=[self.names.index(f'left_arm_{i}') for i in range(7)]
        self.finger_indices=[self.names.index(f'gripper_finger_l{i}') for i in (1,2)]
        targets=np.zeros((1,len(self.names)),dtype=np.float32)
        k=np.zeros_like(targets);d=np.zeros_like(targets);force=np.zeros_like(targets)
        self.feedforward_limit=np.zeros_like(targets)
        for i,name in enumerate(self.names):
            is_finger=name.startswith('gripper_finger')
            targets[0,i]=self.slides.get(name,0.) if is_finger else np.radians(self.pose.get(name,0.))
            k[0,i]=200 if is_finger else (25000 if name.startswith('torso') else 6000)
            d[0,i]=5 if is_finger else (500 if name.startswith('torso') else 160)
            limit=self.effort.get(name,100)
            force[0,i]=.5 if is_finger else .7*limit
            self.feedforward_limit[0,i]=0 if is_finger else .3*limit
        self.targets=targets
        self.force_limits=force
        self.robot.set_dof_stiffnesses(k,self.index);self.robot.set_dof_dampings(d,self.index)
        self.robot.set_dof_max_forces(force,self.index)
        self.robot.set_dof_position_targets(targets,self.index)
        # Filter only plant loads, so wheel support does not look like damage.
        if self.event_monitor is not None: self.event_monitor.close()
        if self.sparse_contacts:
            from .contact_events import ContactEvents
            self.event_monitor=ContactEvents(robot_root=self.root,target_root=self.rig.root,
                fingers=self.paths[1:],floor_root=self.floor_root)
            self.event_monitor.subscribe()
        else:
            self.plant_contacts=simulation_view.create_rigid_contact_view(self.body_paths,
                filter_patterns=[self.plant_filter_paths]*len(self.body_paths))
            self.self_contacts=simulation_view.create_rigid_contact_view(self.body_paths,
                filter_patterns=[self.body_paths]*len(self.body_paths))
            if self.plant_contacts.sensor_count!=len(self.body_paths): raise RuntimeError('Incomplete robot contact sensors')
        self.robot_bodies=simulation_view.create_rigid_body_view(self.body_paths)
        if len(self.robot_bodies.prim_paths)!=len(self.body_paths) or set(self.robot_bodies.prim_paths)!=set(self.body_paths):
            raise RuntimeError('Incomplete native robot body coverage')
        self.body_order=[self.body_paths.index(p) for p in self.robot_bodies.prim_paths]

    def target_palm(self,position):
        if self.event_monitor is not None: self.event_monitor.begin_step()
        delta=self.goal[:3,3]-self.start[:3,3]
        fraction=float(np.dot(np.asarray(position)-self.start[:3,3],delta)/np.dot(delta,delta))
        if not -.01<=fraction<=1.15: raise RuntimeError('Command outside checked IK corridor')
        q=np.array([np.interp(fraction,self.fractions,self.path_q[:,i]) for i in range(7)])
        expected=self.kin.forward('left',q,self.base)
        if np.linalg.norm(expected[:3,3]-position)>.0005: raise RuntimeError('IK interpolation exceeds 0.5 mm')
        self.targets[0,self.left_indices]=np.radians(q)
        self.expected_palm=expected
        self.robot.set_dof_position_targets(self.targets,self.index)
        gravity=self.robot.get_gravity_compensation_forces()
        compensation=np.clip(gravity,-self.feedforward_limit,self.feedforward_limit)
        if self.finger_gravity:
            self.finger_compensation=np.array(gravity[0,self.finger_indices])
            self.force_limits[0,self.finger_indices]=finger_force_budget(self.finger_compensation)
            self.robot.set_dof_max_forces(self.force_limits,self.index)
            compensation[0,self.finger_indices]=self.finger_compensation
        self.robot.set_dof_actuation_forces(compensation.astype(np.float32),self.index)

    def close(self,fraction):
        if not np.isfinite(fraction) or not 0<=fraction<=1: raise ValueError('Invalid finger closure')
        self.targets[0,self.finger_indices]=np.array([-.025,.025])*(1-fraction)
        self.robot.set_dof_position_targets(self.targets,self.index)

    def check(self,dt,palm):
        error=float(np.linalg.norm(palm[:3,3]-self.expected_palm[:3,3]))
        speed=float(np.linalg.norm(self.robot_bodies.get_velocities()[:,:3],axis=1).max())
        if self.window is not None:
            from .collision_window import inside_window
            if not inside_window(self.robot_bodies.get_transforms()[:,:3],
                    self.window_robot_radii[self.body_order],**self.window):
                raise RuntimeError('Robot collision geometry left the validated local wire window')
        if self.event_monitor is not None:
            metrics=self.event_monitor.measurements(dt)
            # Positive native finger tensor loads must be observed by the
            # sparse stream too; a disconnected callback must not look safe.
            if getattr(self,'latest_finger_bilateral',False) and metrics['allowed_target_contact_n']<.01:
                raise RuntimeError('Sparse contact stream missed native bilateral grasp load')
            if (not np.isfinite([error,speed,*metrics.values()]).all()
                    or error>.012 or speed>3 or metrics['unwanted_contact_n']>.5 or metrics['self_contact_n']>3):
                self.last_fault=dict(palm_error_m=error,speed_m_s=speed,**metrics,
                    pairs=[[a,b,v/dt] for (a,b),v in self.event_monitor.pairs.items()])
                print('FULL_ROBOT_GUARD '+json.dumps(self.last_fault),flush=True)
                raise RuntimeError('Full robot contact/tracking guard: '+json.dumps(self.last_fault))
            return dict(palm_tracking_error_m=error,max_body_speed_m_s=speed,**metrics,
                finger_gravity_effort_n=self.finger_compensation.tolist(),
                joint_positions_rad=self.robot.get_dof_positions()[0].tolist())
        # get_net_contact_forces includes all contacts regardless of filters;
        # the force matrix, below, is the plant-specific measurement.
        matrix=np.asarray(self.plant_contacts.get_contact_force_matrix(dt))
        loads=np.linalg.norm(matrix,axis=-1).sum(axis=1)
        paths=list(self.plant_contacts.sensor_paths)
        unwanted=max((float(f) for path,f in zip(paths,loads) if path not in self.paths[1:]),default=0.)
        self_matrix=np.linalg.norm(self.self_contacts.get_contact_force_matrix(dt),axis=-1)
        row,col=np.unravel_index(np.argmax(self_matrix),self_matrix.shape)
        self_force=float(self_matrix[row,col])
        if not np.isfinite([error,speed,unwanted,self_force]).all() or error>.012 or speed>3 or unwanted>.5 or self_force>3:
            def top_pairs(view,values):
                filters=np.asarray(view.filter_paths).reshape(view.sensor_count,view.filter_count)
                pairs=[]
                for flat in np.argsort(values.ravel())[-6:][::-1]:
                    i,j=np.unravel_index(flat,values.shape)
                    pairs.append([view.sensor_paths[i],str(filters[i,j]),float(values[i,j])])
                return pairs
            self.last_fault=dict(palm_error_m=error,speed_m_s=speed,
                self_pairs=top_pairs(self.self_contacts,self_matrix),
                plant_pairs=top_pairs(self.plant_contacts,np.linalg.norm(matrix,axis=-1)),
                dofs=self.names,positions=self.robot.get_dof_positions().tolist(),targets=self.targets.tolist())
            print('FULL_ROBOT_GUARD '+json.dumps(self.last_fault),flush=True)
            raise RuntimeError(f'Full robot guard: palm error {error:.4f} m, speed {speed:.3f} m/s, non-finger plant load {unwanted:.3f} N, self load {self_force:.3f} N')
        return dict(palm_tracking_error_m=error,max_body_speed_m_s=speed,
            non_finger_plant_load_n=unwanted,joint_positions_rad=self.robot.get_dof_positions()[0].tolist())

    def contact(self,dt,frame):
        result=super().contact(dt,frame)
        self.latest_finger_bilateral=result['bilateral']
        return result

    def check_plant_window(self,frames):
        if self.window is not None:
            from .collision_window import inside_window
            if not inside_window(frames[:,:3,3],self.window_plant_radii,**self.window):
                raise RuntimeError('Dynamic plant left the validated local wire window')

    def setup_views(self,viewport):
        from pxr import Gf,UsdGeom
        self.viewport=viewport
        target=self.rig.rest_frames[self.body_index,:3,3]
        centre=(target+self.base[:3,3])/2;centre[2]=.9
        greenhouse=self.floor_root!='/World/RobotTestFloor'
        wide_eye=self.base[:3,3]+np.array([.2,-2.2,1.25]) if greenhouse else centre+np.array([2.5,-3.,1.3])
        close_eye=target+np.array([.5,-.7,.3]) if greenhouse else target+np.array([.60,.68,.35])
        self.views={}
        for name,eye,at in (
            ('Full robot',wide_eye,centre),
            ('Grasp close-up',close_eye,target)):
            path='/World/RobotProbe'+('Wide' if name=='Full robot' else 'Close')
            camera=UsdGeom.Camera.Define(self.stage,path)
            camera.CreateFocalLengthAttr(12. if greenhouse and name=='Full robot' else 24.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.005,100))
            xf=UsdGeom.Xformable(camera);xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*at),Gf.Vec3d(0,0,1)).GetInverse())
            self.views[name]=path
        self.views.update({
            'Head D405':self.root+'/link_head_2/attachments/HeadCamera/D405/DepthCamera',
            'Left wrist D405':self.root+'/ee_left/attachments/LeftWristCamera/D405/DepthCamera',
            'Right wrist D405':self.root+'/ee_right/attachments/RightWristCamera/D405/DepthCamera'})
        self.select_view('Full robot')

    def select_view(self,name):
        self.viewport.set_texture_resolution((848,408) if 'D405' in name else (1280,720))
        self.viewport.set_active_camera(self.views[name])


def interactive(app,sim,rig,fixture,args,output):
    import omni.ui as ui
    from pxr import UsdLux
    from omni.kit.viewport.utility import get_active_viewport
    from .gripper_probe import run
    cutting=bool(getattr(args,'bimanual_cut',False))
    if cutting:
        from .bimanual_probe import run
    from .runtime import PlantRuntime
    from .implicit_springs import ImplicitJointSprings
    if args.scene=='isolated':
        UsdLux.DomeLight.Define(rig.stage,'/World/RobotProbeLight').CreateIntensityAttr(1400.)
    fixture.setup_views(get_active_viewport())
    request={'run':True,'reset':False};runs=[]
    def command(name): request[name]=True
    window=ui.Window('Full RB-Y1 - physical '+('grasp + cut' if cutting else 'grasp test'),width=390,height=610)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label('FULL DYNAMIC RBY1-A v1.2',height=25)
            ui.Label('IK-driven left arm, force-limited fingers. Original plant; only the target petiole is compliant. No grasp weld.',word_wrap=True,height=52)
            status=ui.Label('Ready. Run approach / grasp / 10 mm motion.',word_wrap=True,height=65)
            ui.Button('Run left grasp + right knife' if cutting else 'Run grasp + 10 mm movement',height=30,clicked_fn=lambda:command('run'))
            ui.Button('Stop test',height=26,clicked_fn=lambda:setattr(fixture,'stop_requested',True))
            ui.Button('Reset',height=26,clicked_fn=lambda:command('reset'))
            with ui.HStack(height=23):
                captures=ui.CheckBox()
                captures.model.set_value(bool(getattr(args,'capture_milestones',True)))
                captures.model.add_value_changed_fn(lambda m:setattr(args,'capture_milestones',m.get_value_as_bool()))
                ui.Label('Save milestone PNGs (pauses next trial)')
            for name in fixture.views:
                ui.Button(name,height=23,clicked_fn=lambda n=name:fixture.select_view(n))
            ui.Label('Diagnostic: privileged target; seam release is not calibrated tissue fracture. No deposit qualification, training approval or hardware commands.',word_wrap=True,height=48)
    def on_sample(record):
        if round(record['t']*240)%8: return
        t=record['t'];phase='Settle' if t<1 else 'IK approach' if t<2 else 'Close fingers' if t<3 else 'Grasp hold' if t<3.5 else 'Move if grasp verified' if t<4.5 else 'Hold' if t<5.5 else 'Open'
        phase=record.get('phase',phase)
        slip=record['slip_m']
        status.text=f"{phase} | {t:.2f} s\nOpposing shaft contact: {record['contact']['bilateral']}\nSlip: {'n/a' if slip is None else format(slip*1000,'.2f')+' mm'}"
    fixture.on_sample=on_sample
    property_window=ui.Workspace.get_window('Property')
    if property_window: window.dock_in(property_window,ui.DockPosition.SAME)
    def reset():
        sim.stop();rig.restore_authored_state();fixture.restore_authored_state()
        sim.reset();sim.step(render=False)
        runtime=PlantRuntime(rig,sim.physics_sim_view)
        springs=ImplicitJointSprings(runtime.articulation)
        fixture.bind(sim.physics_sim_view)
        runtime.sync_visuals()
        return runtime,springs
    runtime,springs=reset()
    print('FULL_ROBOT_DEMO_READY '+str(output),flush=True)
    (output/'live_status.json').write_text(json.dumps(dict(state='ready',robot=fixture.report()),indent=2),encoding='utf-8')
    while app.is_running():
        if request['reset'] or request['run']:
            run_requested=request['run'];request.update(run=False,reset=False)
            runtime,springs=reset()
            if run_requested:
                folder=output/f'trial_{len(runs)+1:03d}';folder.mkdir()
                result=run(app,sim,rig,runtime,springs,fixture,args,folder)
                (folder/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
                runs.append(result)
                status.text=('Limited mechanism test PASSED' if all(result['gates'].values()) else 'Test did NOT pass')+'\n'+str(result['error'] or result['gates'])
                (output/'live_status.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
                print('FULL_ROBOT_TRIAL '+json.dumps(dict(state=result['state'],error=result['error'],gates=result['gates'])),flush=True)
            else: status.text='Reset. Ready for a new grasp trial.'
        sim.render();time.sleep(.01)
    return dict(state='interactive_full_robot_diagnostic',trials=runs,training_eligible=False)
