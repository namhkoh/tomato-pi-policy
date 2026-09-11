"""Bounded left-gripper mechanism test with native contacts and no grasp weld.

The palm follows a kinematic fixture trajectory, NOT a solved robot-arm path.
Fingers are dynamic force-limited prismatic joints from the current RBY1 asset.
Plant geometry is never moved to simulate grasping. All evidence is non-training.
"""
import asyncio
import json
import time

import numpy as np


def ramp(t, low, high):
    x=float(np.clip((t-low)/(high-low),0.,1.))
    return x*x*(3.-2.*x)


def palm_frame(point, tangent):
    """Jaw Y runs along the petiole; palm Z points out of the foliage."""
    y=np.asarray(tangent,dtype=float); y=y/np.linalg.norm(y)
    z=np.array([1.,1.,.7]);z-=y*np.dot(z,y)
    if np.linalg.norm(z)<.1:
        z=np.array([0.,0.,1.]);z-=y*np.dot(z,y)
    z/=np.linalg.norm(z); x=np.cross(y,z)
    frame=np.eye(4);frame[:3,:3]=np.column_stack([x,y,z])
    frame[:3,3]=np.asarray(point)+.1025*z
    return frame


def bilateral(forces, *, minimum=.02):
    forces=np.asarray(forces,dtype=float)
    if forces.shape!=(2,3) or not np.isfinite(forces).all(): return False
    magnitudes=np.linalg.norm(forces,axis=1)
    return bool(np.all(magnitudes>=minimum) and np.dot(forces[0],forces[1])<-.5*np.prod(magnitudes))


class GripperFixture:
    def __init__(self,stage,rig,*,arc=.12,friction=.5):
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
        from greenhouse_sim.robot_model import DEFAULT_ASSET,ROBOT_ROOT
        from .plant import matrix_attr,physics_schema
        self.stage,self.rig,self.asset=stage,rig,DEFAULT_ASSET
        self.root='/World/GripperProbe'
        self.body_index=int(np.argmin(np.abs((rig.arcs[:-1]+rig.arcs[1:])/2-arc)))
        if self.body_index<rig.cut_index or rig.arcs[self.body_index]<.035:
            raise ValueError('Grasp fixture must be on distal material with seam clearance')
        self.arc=float((rig.arcs[self.body_index]+rig.arcs[self.body_index+1])/2)
        self.grasp_path=rig.body_paths[self.body_index]
        self.half_length=float(np.linalg.norm(rig.chain_world[self.body_index+1]-rig.chain_world[self.body_index])/2)
        self.radius=float(UsdGeom.Capsule.Get(stage,self.grasp_path+'/StemCollider').GetRadiusAttr().Get())
        self.goal=palm_frame(rig.rest_frames[self.body_index,:3,3],rig.rest_frames[self.body_index,:3,2])
        self.start=self.goal.copy();self.start[:3,3]+=.08*self.goal[:3,2]
        self.paths=[self.root+'/'+n for n in ('ee_left','ee_finger_l1','ee_finger_l2')]
        self.drives=[];self.collider_paths=[];self.local_finger_frames=[]
        self.friction=friction
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            UsdGeom.Xform.Define(stage,self.root)
            material=UsdShade.Material.Define(stage,self.root+'/ContactMaterial')
            api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(friction);api.CreateDynamicFrictionAttr(friction);api.CreateRestitutionAttr(0.)
            physics_schema(material.GetPrim(),'PhysxMaterialAPI',[
                ('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token,'min')])
            for i,path in enumerate(self.paths):
                name=path.rsplit('/',1)[1]
                prim=UsdGeom.Xform.Define(stage,path).GetPrim()
                prim.GetReferences().AddReference(str(DEFAULT_ASSET),Sdf.Path(ROBOT_ROOT+'/'+name));stage.Load(path)
                if i==0:
                    frame=self.start
                else:
                    local=np.eye(4);sign=1 if i==1 else -1
                    local[:3,3]=[sign*(.003+.025),0,-.075]
                    if i==2: local[:3,:3]=np.diag([-1.,-1.,1.])
                    self.local_finger_frames.append(local)
                    frame=self.start@local
                matrix_attr(prim,frame)
                UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(i==0)
                UsdPhysics.RigidBodyAPI(prim).CreateVelocityAttr(Gf.Vec3f(0))
                UsdPhysics.RigidBodyAPI(prim).CreateAngularVelocityAttr(Gf.Vec3f(0))
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI): prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                physics_schema(prim,'PhysxContactReportAPI',[('physxContactReport:threshold',Sdf.ValueTypeNames.Float,0.)])
                for child in Usd.PrimRange(prim):
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(True)
                        physics_schema(child,'PhysxCollisionAPI',[
                            ('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0005),
                            ('physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)])
                        UsdShade.MaterialBindingAPI.Apply(child).Bind(material,materialPurpose='physics')
                        self.collider_paths.append(str(child.GetPath()))
            for i in (1,2):
                name='gripper_finger_l'+str(i)
                prim=stage.DefinePrim(self.root+'/joints/'+name)
                prim.GetReferences().AddReference(str(DEFAULT_ASSET),Sdf.Path(ROBOT_ROOT+'/joints/'+name))
                joint=UsdPhysics.PrismaticJoint(prim)
                joint.CreateBody0Rel().SetTargets([self.paths[0]]);joint.CreateBody1Rel().SetTargets([self.paths[i]])
                joint.CreateJointEnabledAttr(True);joint.CreateCollisionEnabledAttr(False)
                drive=UsdPhysics.DriveAPI.Apply(prim,'linear');drive.CreateTypeAttr('force')
                drive.CreateStiffnessAttr(200.);drive.CreateDampingAttr(5.);drive.CreateMaxForceAttr(.5)
                drive.CreateTargetPositionAttr((-.025 if i==1 else .025));drive.CreateTargetVelocityAttr(0.)
                self.drives.append(drive)

    def report(self):
        return dict(asset=str(self.asset),scope='actual_left_gripper_on_kinematic_palm_fixture',
            arm_ik_executed=False,grasp_weld=False,plant_pose_override=False,
            grasp_body=self.grasp_path,grasp_arc_m=self.arc,requested_friction=self.friction,
            friction_calibration='engineering_prior_not_measured',finger_max_drive_force_n=.5,
            collider_paths=self.collider_paths,training_eligible=False)

    def restore_authored_state(self):
        from pxr import Gf,Usd,UsdPhysics
        from .plant import matrix_attr
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for i,path in enumerate(self.paths):
                frame=self.start if i==0 else self.start@self.local_finger_frames[i-1]
                prim=self.stage.GetPrimAtPath(path);matrix_attr(prim,frame)
                api=UsdPhysics.RigidBodyAPI(prim)
                api.CreateVelocityAttr(Gf.Vec3f(0));api.CreateAngularVelocityAttr(Gf.Vec3f(0))
            self.close(0.)

    def bind(self, simulation_view):
        self.palm=simulation_view.create_rigid_body_view(self.paths[0])
        self.fingers=simulation_view.create_rigid_body_view(self.root+'/ee_finger_l*')
        if self.palm.count!=1 or self.fingers.count!=2: raise RuntimeError('Incomplete native gripper fixture')
        self.order=[list(self.fingers.prim_paths).index(p) for p in self.paths[1:]]
        self.contact_views=[simulation_view.create_rigid_contact_view(p,filter_patterns=[self.grasp_path],max_contact_data_count=128) for p in self.paths[1:]]
        self.all_contacts=simulation_view.create_rigid_contact_view(self.root+'/ee_*')
        self.index=np.array([0],dtype=np.uint32)

    def target_palm(self, position):
        from pxr import Gf
        frame=self.start.copy();frame[:3,3]=position
        quat=Gf.Matrix4d(frame.T.tolist()).ExtractRotationQuat()
        value=np.r_[position,list(quat.GetImaginary()),quat.GetReal()].astype(np.float32)[None,:]
        self.palm.set_kinematic_targets(value,self.index)

    def close(self, fraction):
        from pxr import Usd
        if not np.isfinite(fraction) or not 0<=fraction<=1:
            raise ValueError('Closure fraction must be finite and in [0,1]')
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for i,drive in enumerate(self.drives):
                drive.GetTargetPositionAttr().Set(float((-.025 if i==0 else .025)*(1-fraction)))

    def contact(self,dt,frame):
        forces=[];separations=[];points=[];counts=[];stem_only=True
        half,radius=self.half_length,self.radius
        for view in self.contact_views:
            f,p,n,d,c,s=view.get_contact_data(dt)
            count=int(c[0,0]);start=int(s[0,0]);sl=slice(start,start+count)
            if start<0 or count<0 or start+count>=128:
                raise RuntimeError('Native contact detail buffer exhausted or invalid')
            force=np.sum(np.asarray(f)[sl]*np.asarray(n)[sl],axis=0)
            forces.append(force);counts.append(count)
            if count:
                local=(np.asarray(p)[sl]-frame[:3,3])@frame[:3,:3]
                radial=np.linalg.norm(local[:,:2],axis=1)
                stem_only=stem_only and bool(np.all(radial<radius+.001) and np.all(np.abs(local[:,2])<half+.001))
                separations.extend(np.asarray(d)[sl].reshape(-1).tolist());points.extend(np.asarray(p)[sl].tolist())
        return dict(forces=np.asarray(forces).tolist(),counts=counts,stem_only=stem_only,
            bilateral=bilateral(forces) and stem_only,min_separation=min(separations,default=0.),points=points)


def setup_probe_camera(stage,target):
    """Session-only and repeatable across resets/replays."""
    from pxr import Gf,Usd,UsdGeom
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        camera=UsdGeom.Camera.Define(stage,'/World/GraspProbeCamera')
        camera.CreateFocalLengthAttr(24.);camera.CreateClippingRangeAttr(Gf.Vec2f(.005,100.))
        eye=np.asarray(target)+np.array([.65,.85,.45])
        xform=UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
    return str(camera.GetPath())


def run(app,sim,rig,runtime,springs,fixture,args,output):
    from greenhouse_sim.physics_clock import PhysicsClock
    from .runtime import pose_matrices
    fixture.bind(sim.physics_sim_view)
    full_robot=bool(getattr(args,'full_robot_probe',False))
    capture_milestones=bool(getattr(args,'capture_milestones',True))
    step_context=sim
    if getattr(args,'step_profile',False):
        from .step_profile import MeasuredStep
        step_context=MeasuredStep(sim)
    clock=PhysicsClock(step_context,physics_hz=args.physics_hz,render_hz=args.render_hz)
    records=[];events=[];stable=0;grasp_local=None;baseline=None;goal_set=False
    moved=False;fault=None;release_authorized=False;captures={}
    viewport=None
    if args.render_hz:
        from pxr import UsdLux
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport();viewport.set_texture_resolution((1280,720))
        if args.scene=='isolated':
            UsdLux.DomeLight.Define(rig.stage,'/World/ProbeLight').CreateIntensityAttr(1400.)
        target=rig.rest_frames[fixture.body_index,:3,3]
        viewport.set_active_camera(setup_probe_camera(rig.stage,target))
        if full_robot: fixture.setup_views(viewport)

    def capture(name):
        from omni.kit.viewport.utility import capture_viewport_to_file
        runtime.sync_visuals();before=runtime.frames.copy()
        for _ in range(20): sim.render()
        future=asyncio.ensure_future(capture_viewport_to_file(viewport,str(output/(name+'.png'))).wait_for_result())
        deadline=time.monotonic()+30
        while not future.done():
            if not app.is_running() or time.monotonic()>deadline: raise RuntimeError('Probe capture failed')
            sim.render()
        future.result();runtime.sample()
        if not np.allclose(before,runtime.frames,atol=1e-8,rtol=0): raise RuntimeError('Rendering moved physical bodies')
        captures[name]=name+'.png'

    def before(stamp,dt):
        nonlocal goal_set,baseline,grasp_local,moved,release_authorized
        t=stamp.simulation_time_s
        if t>=.9 and not goal_set:
            depth=fixture.grasp_depth if full_robot else .1025
            fixture.goal[:3,3]=runtime.frames[fixture.body_index,:3,3]+depth*fixture.goal[:3,2]
            if full_robot: fixture.plan_approach()
            goal_set=True
        goal=fixture.start[:3,3]+ramp(t,1.,2.)*(fixture.goal[:3,3]-fixture.start[:3,3])
        closing=ramp(t,2.,3.)*(1-ramp(t,5.5,6.5))
        if t>=3.5 and baseline is None:
            baseline=runtime.frames[fixture.body_index,:3,3].copy()
            palm=pose_matrices(fixture.palm.get_transforms())[0]
            grasp_local=(baseline-palm[:3,3])@palm[:3,:3]
            moved=stable>=int(.1*args.physics_hz)
            events.append(dict(t=t,event='grasp_gate',passed=moved,consecutive_bilateral_steps=stable))
        if moved:
            goal=goal+.01*ramp(t,3.5,4.5)*fixture.goal[:3,2]
        if args.diagnostic_detach and t>=4.75 and not rig.cut and not release_authorized:
            release_authorized=True
            if moved and stable>=int(.1*args.physics_hz):
                events.append(dict(t=t,**rig.diagnostic_release()))
            else: events.append(dict(t=t,event='diagnostic_release_blocked_no_grasp'))
        fixture.target_palm(goal);fixture.close(closing)
        # PhysX applies actual contacts. Do not invent/reapply contact forces or
        # attach the target by a weld. Contact response is what this probe tests.
        springs.step(dt,root_constrained=not rig.cut)

    def after(stamp,dt):
        nonlocal stable
        frames,velocities=runtime.sample();frame=frames[fixture.body_index]
        c=fixture.contact(dt,frame)
        stable=stable+1 if c['bilateral'] else 0
        palm=pose_matrices(fixture.palm.get_transforms())[0]
        positions=pose_matrices(fixture.fingers.get_transforms())[fixture.order,:3,3]
        speed=float(np.linalg.norm(velocities[:,:3],axis=1).max())
        total=float(np.linalg.norm(fixture.all_contacts.get_net_contact_forces(dt),axis=1).max())
        slip=None if grasp_local is None else float(np.linalg.norm((frame[:3,3]-palm[:3,3])@palm[:3,:3]-grasp_local))
        support=float(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1).max())
        displacement=float(np.linalg.norm(runtime.tip()-rig.chain_world[-1]))
        records.append(dict(t=stamp.simulation_time_s,grasp_point=frame[:3,3].tolist(),
            tip=runtime.tip().tolist(),palm=palm.tolist(),fingers=positions.tolist(),
            contact=c,slip_m=slip,max_speed_m_s=speed,max_gripper_net_contact_n=total,
            support_error_m=support,tip_deflection_m=displacement,cut=rig.cut))
        if full_robot:
            fixture.check_plant_window(frames)
            records[-1]['robot']=fixture.check(dt,palm)
            fixture.on_sample(records[-1])
        if speed>20 or total>3 or support>1e-5 or (not rig.cut and displacement>.12):
            raise RuntimeError('Contact probe exceeded velocity/force/support/deflection bounds')

    print('GRIPPER_PROBE_READY '+json.dumps(fixture.report()),flush=True)
    try:
        if viewport and capture_milestones: capture('initial')
        for _ in range(int(args.seconds*args.physics_hz)):
            tick_start=time.monotonic()
            if not app.is_running(): raise RuntimeError('Probe closed before completion')
            if full_robot and fixture.stop_requested: raise RuntimeError('Stopped by user; reset before replay')
            clock.tick(before=before,after=after,before_render=lambda _:runtime.sync_visuals())
            if viewport and capture_milestones:
                for name,t in (('approach',1.9),('closed',3.4),('moved',4.5),('hold_after_diagnostic_release',5.3),('opened',6.6)):
                    if clock.stamp.simulation_time_s>=t and name not in captures:
                        capture(name)
                        if name=='closed' and full_robot and not args.gui:
                            previous=str(viewport.camera_path)
                            fixture.select_view('Grasp close-up');capture('closed_detail')
                            viewport.set_active_camera(previous)
            if full_robot and args.gui: time.sleep(max(0.,1/args.physics_hz-(time.monotonic()-tick_start)))
    except Exception as exc:
        fault=str(exc)
        if viewport and app.is_running():
            try: capture('stopped_on_fault')
            except Exception as capture_error: captures['fault_capture_error']=str(capture_error)
    movement=[r for r in records if 3.5<=r['t']<=4.5]
    retained=[r for r in records if r['cut'] and 4.8<=r['t']<=5.4]
    contact_records=[r for r in records if 3.0<=r['t']<=3.5]
    min_bilateral=float(np.mean([r['contact']['bilateral'] for r in contact_records])) if contact_records else 0.
    max_slip=max((r['slip_m'] for r in movement if r['slip_m'] is not None),default=None)
    actual_move=(float(np.dot(np.array(movement[-1]['grasp_point'])-baseline,fixture.goal[:3,2])) if movement and baseline is not None else 0.)
    gates=dict(bounded=fault is None,completed=len(records)==int(args.seconds*args.physics_hz),
        bilateral_stem_grasp=moved,maintains_contact_while_moving=bool(movement) and bool(np.mean([r['contact']['bilateral'] for r in movement])>.9),
        target_follows=actual_move>.005,slip_below_3mm=max_slip is not None and max_slip<.003,
        limited_penetration=all(r['contact']['min_separation']>-.001 for r in records))
    if args.diagnostic_detach:
        gates['diagnostic_retention']=bool(retained) and all(r['contact']['bilateral'] and r['slip_m']<.003 for r in retained)
    measurements=dict(contact_fraction_before_move=min_bilateral,movement_along_command_m=actual_move,
        max_slip_during_move_m=max_slip,max_gripper_contact_n=max((r['max_gripper_net_contact_n'] for r in records),default=0.),
        max_speed_m_s=max((r['max_speed_m_s'] for r in records),default=0.),
        maximum_penetration_m=-min((r['contact']['min_separation'] for r in records),default=0.))
    # A fresh parse must restore both plant and gripper; no stale tensor handles.
    from .runtime import PlantRuntime
    from .implicit_springs import ImplicitJointSprings
    sim.stop();rig.restore_authored_state();fixture.restore_authored_state();sim.reset();sim.step(render=False)
    reset_runtime=PlantRuntime(rig,sim.physics_sim_view)
    reset_springs=ImplicitJointSprings(reset_runtime.articulation);fixture.bind(sim.physics_sim_view)
    measurements['reset_body_error_m']=float(np.linalg.norm(reset_runtime.frames[:,:3,3]-rig.rest_frames[:,:3,3],axis=1).max())
    replay=[]
    for i in range(min(len(records),int(.5*args.physics_hz))):
        fixture.target_palm(fixture.start[:3,3]);fixture.close(0.)
        reset_springs.step(1/args.physics_hz,root_constrained=True)
        sim.step(render=False);reset_runtime.sample()
        replay.append(float(np.linalg.norm(reset_runtime.tip()-np.array(records[i]['tip']))))
    measurements['reset_replay_error_m']=max(replay,default=None)
    gates['reset_replays']=len(replay)==int(.5*args.physics_hz) and measurements['reset_body_error_m']<.005 and max(replay)<.001
    if viewport and capture_milestones:
        runtime=reset_runtime
        capture('reset_replay')
    result=dict(state='passed_gripper_mechanism_not_robot_task' if all(gates.values()) else 'failed_gripper_qualification',
        gates=gates,measurements=measurements,error=fault,events=events,timing=clock.report(),images=captures,
        paused_milestone_captures=capture_milestones,
        grasp_fixture_only=True,full_robot_ik_verified=False,physical_cut_verified=False,training_eligible=False,
        unknown_contact_load_prediction='not_included_in_elastic_predictor_native_contact_response_under_test')
    if full_robot:
        result.update(grasp_fixture_only=False,full_robot_ik_executed=True,
            robot=fixture.report(),full_robot_ik_verified=fault is None,
            full_deleafing_task_verified=False)
    if step_context is not sim: result['step_profile']=step_context.report()
    (output/'gripper_trajectory.json').write_text(json.dumps(records,allow_nan=False),encoding='utf-8')
    sim.stop()
    return result
