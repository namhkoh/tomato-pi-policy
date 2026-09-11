"""Bounded real-PhysX qualification, not dataset collection or robot cutting.

Uses an isolated plant by default; --scene package retains the supplied building,
plants, v1.2 static robot and head/wrist cameras. Never takes over a running Kit.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',choices=('isolated','package'),default='isolated')
    p.add_argument('--plant',default='seed101_full')
    p.add_argument('--target',default='SubStem_41')
    p.add_argument('--physics-hz',type=int,choices=(120,240,480,1920),default=240)
    p.add_argument('--solver',choices=('TGS','PGS'),default='TGS')
    p.add_argument('--gravity',type=float,choices=(0.,9.81),default=9.81)
    p.add_argument('--constraint-mode',choices=('articulation','maximal','fixed_articulation'),default='articulation')
    p.add_argument('--attached-only',action='store_true')
    p.add_argument('--spring-mode',choices=('native','implicit_effort'),default='native')
    p.add_argument('--force-newton',type=float,default=.02)
    p.add_argument('--physics-threads',type=int,choices=(1,2,4,8,16))
    p.add_argument('--fabric',action='store_true')
    p.add_argument('--render-hz',type=int,choices=(0,15,30,60),default=0)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--max-segment-m',type=float,default=.025)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--interactive',action='store_true',help='Keep an isolated physics demo open with pull/release/reset controls')
    p.add_argument('--gripper-probe',action='store_true',help='Bounded actual left-gripper contact fixture, not full-arm IK')
    p.add_argument('--finger-friction',type=float,default=.5)
    p.add_argument('--grasp-arc-m',type=float,default=.12)
    p.add_argument('--cut-arc-m',type=float,default=.01,
        help='Explicit diagnostic seam within agreed 10..20 mm petiole interval; original 10 mm default unchanged')
    p.add_argument('--diagnostic-detach',action='store_true')
    p.add_argument('--full-robot-probe',action='store_true',help='Full dynamic v1.2 robot with an IK-driven left arm')
    p.add_argument('--bimanual-cut',action='store_true',help='Guarded native left grasp and original right knife seam-release qualification')
    p.add_argument('--bimanual-hold-control',action='store_true',
        help='Negative control: hold left grasp with right arm parked; never qualifies as cutting')
    p.add_argument('--robot-interactive',action='store_true',help='Keep the full-robot test window open with replay controls')
    p.add_argument('--robot-auto-run',action=argparse.BooleanOptionalAction,default=True,
        help='In robot interactive mode, run immediately; disable to inspect the mounting/target before Run')
    p.add_argument('--sparse-contacts',action='store_true',help='Native event accounting including all greenhouse/neighbor contacts')
    p.add_argument('--finger-gravity',action='store_true',help='Compensate native finger weight inside the original 0.5 N total effort budget')
    p.add_argument('--compliant-fingers',action='store_true',help='Experimental native force-based finger-pad compliance; unchanged masses/effort/guard limits')
    p.add_argument('--approach-tilt',type=float,default=0.,help='Bounded diagnostic wrist tilt around the shaft, in degrees')
    p.add_argument('--station-offset',type=float,nargs=2,metavar=('FORWARD_M','LEFT_M'),
        help='Initial fixed-base station offset only (norm <=0.3 m); never moves a running robot')
    p.add_argument('--station-yaw',type=float,default=0.,
        help='Initial station heading relative to palm approach, within +/-90 degrees; no live base motion')
    p.add_argument('--grasp-roll',type=int,choices=(0,180),default=0,
        help='Initial equivalent finger orientation about palm approach axis; native grasp must be requalified')
    p.add_argument('--grasp-depth-m',type=float,default=.1025,
        help='Shaft distance from the palm within the original pads: 90..125 mm; no live base or plant override')
    p.add_argument('--torso-yaw',type=float,default=0.,
        help='Fixed initial torso_5 yaw in package robot tests, bounded to +/-45 degrees')
    p.add_argument('--approach-distance',type=float,default=.08,
        help='Initial palm approach distance 0.01..0.08 m; leaves the selected fixed base unchanged')
    p.add_argument('--profile',action='store_true',help='Save diagnostic Python/native call timing alongside the non-training report')
    p.add_argument('--step-profile',action='store_true',help='Time the installed physics-only step phases without bypassing physics manager events')
    p.add_argument('--no-physics-profiler',action='store_true',help='Disable optional native profiling instrumentation in this process only')
    p.add_argument('--local-wire-physics',action='store_true',help='Guarded fixed-base 4 m collision window; all wire visuals retained')
    p.add_argument('--context-gutters',type=int,choices=(1,3,5),help='Restore original preview planting density, with static mesh contacts near the fixed robot')
    p.add_argument('--scene-profile',action='store_true',help='Non-qualifying root-removal timing controls; never use as demo evidence')
    p.add_argument('--batch-gutter-visuals',action='store_true',help='Batch all identical static gutter visuals; retain every original gutter collider')
    p.add_argument('--capture-milestones',action=argparse.BooleanOptionalAction,default=True,
        help='Save paused diagnostic screenshots during a trial; disable for smoother interactive playback')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if not math.isfinite(args.cut_arc_m) or not .01<=args.cut_arc_m<=.02 or (
            args.cut_arc_m!=.01 and not args.bimanual_cut):
        raise ValueError('Non-default cut arc requires bimanual qualification within 10..20 mm')
    if not math.isfinite(args.grasp_depth_m) or not .09<=args.grasp_depth_m<=.125 or (
            args.grasp_depth_m!=.1025 and not args.full_robot_probe):
        raise ValueError('Grasp depth requires a full robot and finite 90..125 mm')
    if not math.isfinite(args.station_yaw) or abs(args.station_yaw)>90 or (args.station_yaw and not args.full_robot_probe):
        raise ValueError('Station yaw requires a full robot and finite +/-90 degrees')
    if args.bimanual_hold_control and not args.bimanual_cut:
        raise ValueError('Bimanual hold control requires the guarded bimanual harness')
    if args.station_offset is not None and (not args.full_robot_probe
            or not all(math.isfinite(x) for x in args.station_offset)
            or math.hypot(*args.station_offset)>.3):
        raise ValueError('Station offset requires a full robot and finite norm <=0.3 m')
    if args.grasp_roll and not args.full_robot_probe:
        raise ValueError('Grasp roll requires a full robot')
    if not math.isfinite(args.approach_distance) or not .01<=args.approach_distance<=.08 or (
            args.approach_distance!=.08 and not args.full_robot_probe):
        raise ValueError('Approach distance requires a full robot and finite 10..80 mm')
    if not math.isfinite(args.torso_yaw) or abs(args.torso_yaw)>45 or (
            args.torso_yaw and not (args.full_robot_probe and args.scene=='package')):
        raise ValueError('Torso yaw requires a package full robot and finite +/-45 degrees')
    if args.bimanual_cut and (not args.full_robot_probe or not args.sparse_contacts or not args.finger_gravity or args.seconds<20):
        raise ValueError('Bimanual cutting requires full robot, sparse contacts, finger gravity and >=20 seconds')
    if not args.robot_auto_run and not args.robot_interactive:
        raise ValueError('Disabling robot auto-run requires robot interactive mode')
    if args.robot_interactive and (not args.full_robot_probe or not args.gui or not args.render_hz):
        raise ValueError('Robot interactive requires full-robot probe, GUI and rendering')
    if args.full_robot_probe and (args.gripper_probe or args.interactive or (args.scene=='package' and not args.sparse_contacts)
            or args.constraint_mode!='articulation' or args.spring_mode!='implicit_effort'
            or args.solver!='PGS' or args.physics_hz!=240 or args.gravity!=9.81 or args.seconds<7
            or args.diagnostic_detach or not -30<=args.approach_tilt<=30
            or not 0<=args.finger_friction<=1 or not .04<=args.grasp_arc_m<=.25):
        raise ValueError('Full robot probe requires implicit articulation, PGS 240 Hz, gravity, >=7 s, bounded grasp/friction, no diagnostic detach and sparse contacts for package scenes')
    if (args.sparse_contacts or args.finger_gravity or args.approach_tilt or args.compliant_fingers) and not args.full_robot_probe:
        raise ValueError('Robot contact/gravity/approach options require the full robot probe')
    if args.local_wire_physics and not (args.full_robot_probe and args.scene=='package'):
        raise ValueError('Local wire physics requires the fixed full robot in the supplied package')
    if args.context_gutters and not args.local_wire_physics:
        raise ValueError('Dense context requires a guarded local collision window')
    if args.batch_gutter_visuals and not (args.full_robot_probe and args.scene=='package'):
        raise ValueError('Gutter batching requires a full-robot package scene')
    if args.scene_profile and not (args.full_robot_probe and args.scene=='package' and not args.robot_interactive and not args.gui):
        raise ValueError('Scene profiling requires a bounded headless full-robot package diagnostic')
    if args.profile and (not (args.gripper_probe or args.full_robot_probe) or args.robot_interactive):
        raise ValueError('Profiling requires a bounded gripper or full-robot probe')
    if args.gripper_probe and (args.interactive or args.scene!='isolated'
            or args.constraint_mode!='articulation' or args.spring_mode!='implicit_effort'
            or args.solver!='PGS' or args.gravity!=9.81 or args.seconds<7
            or not 0<=args.finger_friction<=1 or not .04<=args.grasp_arc_m<=.25):
        raise ValueError('Gripper probe requires isolated articulation, implicit_effort, PGS, gravity, >=7 s and bounded friction/grasp location')
    if args.interactive and (not args.gui or args.scene!='isolated' or args.render_hz==0
            or args.spring_mode!='implicit_effort' or args.constraint_mode!='articulation'
            or args.gravity!=9.81 or args.solver!='PGS' or args.physics_hz!=240):
        raise ValueError('Interactive demo requires GUI, isolated scene, rendering, implicit_effort, articulation, Earth gravity, PGS and 240 Hz')
    if args.constraint_mode=='fixed_articulation' and not args.attached_only:
        raise ValueError('Fixed-base comparison is attached-only until topology transition is qualified')
    if args.spring_mode=='implicit_effort' and args.constraint_mode=='maximal':
        raise ValueError('Implicit spring diagnostic requires an articulation')
    if not 0<=args.force_newton<=.2:
        raise ValueError('Diagnostic force must be between zero and 0.2 N')
    from sim_data.audit import DEFAULT_PACK
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(DEFAULT_PACK.resolve()):
        raise ValueError('Choose a NEW output outside the source package')
    if not 4<=args.seconds<=30 or not .005<=args.max_segment_m<=.05:
        raise ValueError('Qualification must be bounded to 4-30 seconds and 5-50 mm segments')
    output.mkdir(parents=True)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':not args.gui,'width':1280 if args.interactive else 848,'height':720 if args.interactive else 408,'multi_gpu':False,
                       'sync_loads':False,'renderer':'RaytracedLighting'})
    import carb.settings
    process_settings=carb.settings.get_settings()
    thread_setting='/persistent/physics/numThreads'
    previous_threads=process_settings.get(thread_setting)
    profiler_setting='/physics/exposeProfilerData'
    previous_profiler=process_settings.get(profiler_setting)
    if args.no_physics_profiler: process_settings.set_bool(profiler_setting,False)
    if args.physics_threads is not None:
        process_settings.set_int(thread_setting,args.physics_threads)
    report=dict(state='initializing',started_utc=datetime.now(timezone.utc).isoformat(),
                configuration={**vars(args),'output':str(output)},training_eligible=False)
    try:
        import numpy as np
        import omni.usd
        from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema
        from isaacsim.core.api import SimulationContext
        from greenhouse_sim.physics_clock import PhysicsClock
        from sim_data.audit import audit_manifest
        from sim_data.geometry import assemble_plant
        from .plant import build
        from .runtime import PlantRuntime
        context=omni.usd.get_context()
        manifest=DEFAULT_PACK/f'plants/components/{args.plant}/manifest.json'
        audit=audit_manifest(manifest)
        source_hashes={manifest:hashlib.sha256(manifest.read_bytes()).hexdigest()}
        for component in audit['components'].values():
            path=manifest.parent/component['file'];source_hashes[path]=component['asset_sha256']
        robot_options=dict(sparse_contacts=args.sparse_contacts,finger_gravity=args.finger_gravity,
            approach_tilt=args.approach_tilt,grasp_roll=args.grasp_roll,approach_distance=args.approach_distance,
            compliant_fingers=args.compliant_fingers,station_yaw=args.station_yaw,grasp_depth=args.grasp_depth_m)
        if args.station_offset is not None: robot_options['station_offset']=args.station_offset
        if args.scene=='package' and args.full_robot_probe:
            from .greenhouse_scene import prepare
            from sim_data.floor_alignment import PACKAGE_FLOOR
            scene=DEFAULT_PACK/'house/green_house_base.usd'
            source_hashes[scene]=hashlib.sha256(scene.read_bytes()).hexdigest()
            if not context.open_stage(str(scene),load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE):
                raise RuntimeError('Cannot open supplied greenhouse')
            stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
            record,height,scene_report=prepare(stage,DEFAULT_PACK,args.plant,sparse_backdrop=not args.context_gutters)
            report['greenhouse']=scene_report
            robot_options.update(ground_height=height,torso_degrees=[0.,0.,0.,0.,0.,args.torso_yaw],floor_root=PACKAGE_FLOOR)
        elif args.scene=='package':
            from launch_sim_data import load_local_payloads,populate
            from sim_data.robot_preview import add_robot_preview,select_camera
            from sim_data.floor_alignment import PACKAGE_FLOOR
            scene=DEFAULT_PACK/'house/green_house_base.usd'
            source_hashes[scene]=hashlib.sha256(scene.read_bytes()).hexdigest()
            context.open_stage(str(scene),load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE)
            stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
            load_local_payloads(stage);records=[]
            cx,counts=populate(stage,DEFAULT_PACK,app,records,args.plant)
            record=records[0]
            robot=add_robot_preview(stage,gutter_x=cx,floor_path=PACKAGE_FLOOR,right_tool='knife_only')
            report.update(scene_counts=counts,robot=robot)
            from omni.kit.viewport.utility import get_active_viewport
            viewport=get_active_viewport()
            if viewport: select_camera(viewport,robot['camera_paths']['Robot head D405'])
        else:
            context.new_stage();stage=context.get_stage()
            UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
            stage.SetEditTarget(stage.GetSessionLayer())
            paths=assemble_plant(stage,'/World/Plant',audit)
            record=dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant')
            if args.full_robot_probe:
                # Explicit test-station placement BEFORE building physics, never
                # a running plant pose override or source-package edit.
                UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp(opSuffix='testStation').Set(Gf.Vec3d(0,0,.35))
        rig=build(stage,record,args.target,max_segment_m=args.max_segment_m,constraint_mode=args.constraint_mode,cut_m=args.cut_arc_m)
        report['rig']=rig.report()
        fixture=None
        if args.full_robot_probe:
            from .full_robot import FullRobotGripper
            robot_class=FullRobotGripper
            if args.bimanual_cut:
                from .bimanual import BimanualRobot
                robot_class=BimanualRobot
            fixture=robot_class(stage,rig,arc=args.grasp_arc_m,friction=args.finger_friction,**robot_options)
            source_hashes[fixture.asset]=hashlib.sha256(fixture.asset.read_bytes()).hexdigest()
            report['robot_probe']=fixture.report()
            if args.local_wire_physics:
                from .collision_window import configure
                report['collision_window']=configure(stage,fixture)
            if args.context_gutters:
                from .greenhouse_context import populate as populate_context
                report['context_plants']=populate_context(stage,DEFAULT_PACK,fixture,args.context_gutters)
                source_hashes.update({Path(p):h for p,h in report['context_plants']['source_sha256'].items()})
            if args.batch_gutter_visuals:
                from .gutter_instances import batch
                report['gutter_visual_batch']=batch(stage)
        if args.gripper_probe:
            from .gripper_probe import GripperFixture
            fixture=GripperFixture(stage,rig,arc=args.grasp_arc_m,friction=args.finger_friction)
            source_hashes[fixture.asset]=hashlib.sha256(fixture.asset.read_bytes()).hexdigest()
            report['gripper_fixture']=fixture.report()
        if args.bimanual_cut:
            from .startup_screen import screen
            report['startup_collision_screen']=screen(stage,fixture)
            if not report['startup_collision_screen']['passed']:
                raise RuntimeError('Bimanual spawn has possible collision overlaps; inspect startup_collision_screen before any physics motion')
        physics=UsdPhysics.Scene.Define(stage,'/World/QualificationPhysics')
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(args.gravity)
        settings=PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
        settings.CreateSolverTypeAttr(args.solver);settings.CreateEnableGPUDynamicsAttr(False)
        settings.CreateBroadphaseTypeAttr('MBP')
        # One explicit scene, fixed dt. Rendering is independently scheduled.
        sim=SimulationContext(physics_dt=1/args.physics_hz,rendering_dt=1/60,
                              stage_units_in_meters=1,physics_prim_path='/World/QualificationPhysics',
                              set_defaults=False,backend='numpy')
        if args.fabric:
            sim.get_physics_context().enable_fabric(True)
        report['effective_scene']=dict(gravity_m_s2=float(physics.GetGravityMagnitudeAttr().Get()),
            solver=settings.GetSolverTypeAttr().Get(),physics_dt=sim.get_physics_dt())
        if (report['effective_scene']['solver']!=args.solver
                or not np.isclose(report['effective_scene']['gravity_m_s2'],args.gravity,rtol=1e-6,atol=1e-8)):
            raise RuntimeError('Simulation initialization changed explicit scene configuration')
        sim.reset()
        report['effective_scene_after_reset']=dict(gravity_m_s2=float(physics.GetGravityMagnitudeAttr().Get()),
            solver=settings.GetSolverTypeAttr().Get(),physics_dt=sim.get_physics_dt(),
            gpu_dynamics=settings.GetEnableGPUDynamicsAttr().Get(),
            update_to_usd=process_settings.get('/physics/updateToUsd'),
            physics_threads=process_settings.get(thread_setting))
        report['native_diagnostics_settings']={k:process_settings.get(k) for k in (
            '/physics/enableSynchronousKernelLaunches','/physics/exposeProfilerData',
            '/persistent/physics/pvdEnabled','/physics/omniPvdOutputEnabled','/physics/omniPvdIsRecording',
            '/physics/physxDispatcher','/physics/updateVelocitiesToUsd')}
        effective=report['effective_scene_after_reset']
        if (effective['solver']!=args.solver or effective['gpu_dynamics']
                or not np.isclose(effective['gravity_m_s2'],args.gravity,rtol=1e-6,atol=1e-8)
                or not np.isclose(effective['physics_dt'],1/args.physics_hz,rtol=1e-6)):
            raise RuntimeError('Physics reset changed explicit configuration')
        sim.step(render=False)
        runtime=PlantRuntime(rig,sim.physics_sim_view)
        report['native_drive_parameters']=runtime.drive_diagnostics
        springs=None
        if args.spring_mode=='implicit_effort':
            from .implicit_springs import ImplicitJointSprings,NativeBodyLoads
            springs=ImplicitJointSprings(runtime.articulation)
            loads=NativeBodyLoads(runtime.articulation,[0,0,-args.gravity]) if args.force_newton else None
            if loads:
                report['force_mapping']=dict(reference=loads.reference,gravity_relative_errors=loads.reference_errors)
        if fixture is not None:
            if args.scene_profile:
                from .scene_profile import run as profile_scene
                report.update(profile_scene(sim,rig,fixture,output))
                (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
                return 0
            from .gripper_probe import run
            if args.bimanual_cut:
                from .bimanual_probe import run
            if args.robot_interactive:
                from .full_robot import interactive
                report.update(interactive(app,sim,rig,fixture,args,output))
            else:
                if args.profile:
                    import cProfile
                    profile=cProfile.Profile();profile.enable()
                    try: report.update(run(app,sim,rig,runtime,springs,fixture,args,output))
                    finally:
                        profile.disable();profile.dump_stats(str(output/'profile.pstats'))
                else: report.update(run(app,sim,rig,runtime,springs,fixture,args,output))
            report['source_assets_unchanged']=all(hashlib.sha256(path.read_bytes()).hexdigest()==h for path,h in source_hashes.items())
            if not report['source_assets_unchanged']: raise RuntimeError('Source asset changed during probe')
            (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            print('GRIPPER_PROBE_RESULT '+json.dumps({k:report.get(k) for k in ('state','gates','measurements')}),flush=True)
            return 0 if report['state'] in ('passed_gripper_mechanism_not_robot_task','passed_bimanual_mechanism_not_robot_task','interactive_full_robot_diagnostic') else 2
        if args.interactive:
            from .demo import run
            report['state']='interactive_demo_not_qualification'
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            report['demo']=run(app,sim,rig,runtime,springs,args,output)
            report['source_assets_unchanged']=all(hashlib.sha256(path.read_bytes()).hexdigest()==h for path,h in source_hashes.items())
            if not report['source_assets_unchanged']: raise RuntimeError('Source asset changed during demo')
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            return 0
        frames,initial_velocity=runtime.sample()
        report['initial_body_state']=dict(paths=rig.body_paths,frames=frames.tolist(),
            rest_frames=rig.rest_frames.tolist(),velocities=initial_velocity.tolist(),
            masses=[p['mass'] for p in rig.properties],
            inertia=[p['inertia'].tolist() for p in rig.properties])
        clock=PhysicsClock(sim,physics_hz=args.physics_hz,render_hz=args.render_hz)
        samples=[];events=[];start=time.perf_counter()
        release_step=int(args.seconds*(1 if args.attached_only else .75)*args.physics_hz)
        def before(stamp,dt):
            t=stamp.simulation_time_s
            if stamp.step==release_step: events.append(rig.diagnostic_release())
            force=[0,args.force_newton,0] if 1<=t<2 else [0,0,0]
            runtime.apply_tip_force(force)
            if springs is not None:
                springs.step(dt,loads.at_com(rig.body_paths[-1],force) if loads else None,
                             root_constrained=not rig.cut)
        def after(stamp,dt):
            frames,velocity=runtime.sample()
            if np.linalg.norm(velocity[:,:3],axis=1).max()>20:
                report['failed_body_state']=dict(step=stamp.step,frames=frames.tolist(),velocities=velocity.tolist())
                raise RuntimeError('Unstable plant velocity; qualification stopped')
            samples.append(dict(t=stamp.simulation_time_s,tip=runtime.tip().tolist(),
                dof_positions=runtime.articulation.get_dof_positions().tolist() if runtime.articulation is not None else None,
                root_rotation=frames[rig.cut_index,:3,:3].tolist(),
                contact_force_n=float(np.linalg.norm(runtime.contacts.get_net_contact_forces(dt),axis=1).max()),
                max_speed=float(np.linalg.norm(velocity[:,:3],axis=1).max()),
                support_error=float(np.max(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1))),
                anchor_gap=float(np.linalg.norm(
                    frames[rig.cut_index,:3,3]-frames[rig.cut_index,:3,2]*np.linalg.norm(rig.chain_world[rig.cut_index+1]-rig.chain_world[rig.cut_index])/2
                    -rig.chain_world[rig.cut_index]))))
        print('PHYSICS_QUALIFICATION_READY '+json.dumps(rig.report()),flush=True)
        for step in range(int(args.seconds*args.physics_hz)):
            if not app.is_running(): raise RuntimeError('Kit closed before bounded qualification completed')
            clock.tick(before=before,after=after,before_render=lambda stamp:runtime.sync_visuals())
            if step%args.physics_hz==0: print('PHYSICS_SECOND',step//args.physics_hz,flush=True)
        runtime.sync_visuals()
        report['timing']=clock.report();report['run_wall_seconds']=time.perf_counter()-start
        report['events']=events
        report['measurements']=dict(max_speed_m_s=max(s['max_speed'] for s in samples),
            max_native_contact_force_n=max(s['contact_force_n'] for s in samples),
            max_support_error_m=max(s['support_error'] for s in samples),
            max_attached_anchor_gap_m=max(s['anchor_gap'] for s in samples[:release_step]),
            released_anchor_gap_m=samples[-1]['anchor_gap'])
        tips=np.asarray([s['tip'] for s in samples])
        baseline=tips[int(.9*args.physics_hz)]
        report['measurements']['max_attached_tip_displacement_m']=float(np.linalg.norm(
            tips[:release_step]-rig.chain_world[-1],axis=1).max())
        report['measurements']['force_pulse_tip_change_m']=float(np.linalg.norm(
            tips[int(1.9*args.physics_hz)]-baseline))
        report['measurements']['recovery_tip_residual_m']=float(np.linalg.norm(
            tips[int(2.9*args.physics_hz)]-baseline))
        if args.attached_only:
            tail=samples[-int(.5*args.physics_hz):]
            report['measurements']['settled_tip_displacement_m']=float(np.linalg.norm(tips[-1]-rig.chain_world[-1]))
            report['measurements']['tail_tip_variation_m']=float(np.linalg.norm(np.ptp(tips[-len(tail):],axis=0)))
            report['measurements']['tail_max_body_speed_m_s']=max(s['max_speed'] for s in tail)
        # Reset is a separate parse boundary; no stale tensor handles reused.
        sim.stop();rig.restore_authored_state();sim.reset();sim.step(render=False)
        restored=PlantRuntime(rig,sim.physics_sim_view);frames,_=restored.sample()
        report['reset_max_body_error_m']=float(np.max(np.linalg.norm(frames[:,:3,3]-rig.rest_frames[:,:3,3],axis=1)))
        if springs is not None:
            report['reset_controller_parameters_restored']=bool(
                np.allclose(restored.drive_diagnostics['stiffness'],report['native_drive_parameters']['stiffness'])
                and np.allclose(restored.drive_diagnostics['damping'],report['native_drive_parameters']['damping']))
            reset_springs=ImplicitJointSprings(restored.articulation)
            replay_errors=[]
            for i in range(int(.5*args.physics_hz)):
                restored.apply_tip_force([0,0,0])
                reset_springs.step(1/args.physics_hz,root_constrained=True)
                sim.step(render=False);restored.sample()
                replay_errors.append(float(np.linalg.norm(restored.tip()-tips[i])))
            report['reset_replay_tip_error_m']=max(replay_errors)
        sim.stop()
        report['source_assets_unchanged']=all(hashlib.sha256(path.read_bytes()).hexdigest()==h for path,h in source_hashes.items())
        report['gates']=dict(finite_bounded_motion=report['measurements']['max_speed_m_s']<20,
            fixed_support=report['measurements']['max_support_error_m']<1e-5,
            connected_before_release=report['measurements']['max_attached_anchor_gap_m']<.005,
            bounded_attached_deflection=report['measurements']['max_attached_tip_displacement_m']<.08,
            reset_restored=report['reset_max_body_error_m']<.005,
            source_assets_unchanged=report['source_assets_unchanged'])
        if args.force_newton>0:
            report['gates']['responds_to_force']=report['measurements']['force_pulse_tip_change_m']>.0001
        if args.attached_only:
            report['gates']['settles_after_load']=(report['measurements']['tail_tip_variation_m']<.001
                and report['measurements']['tail_max_body_speed_m_s']<.01)
        if springs is not None:
            # A pulse is a known external load, not a contact qualification.
            report['gates']['no_unqualified_contact_loads']=report['measurements']['max_native_contact_force_n']<1e-5
            report['gates']['reset_controller_replay']=(report['reset_controller_parameters_restored']
                and report['reset_replay_tip_error_m']<.001)
        if not args.attached_only:
            report['gates']['detached_after_release']=report['measurements']['released_anchor_gap_m']>.02
        report['qualification_scope']='attached_only' if args.attached_only else 'attached_and_diagnostic_release'
        report['state']='passed_mechanism_qualification_not_robot_task' if all(report['gates'].values()) else 'failed_qualification'
        (output/'trajectory.json').write_text(json.dumps(samples,allow_nan=False),encoding='utf-8')
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('PHYSICS_QUALIFICATION_RESULT '+json.dumps(report),flush=True)
        return 0 if all(report['gates'].values()) else 2
    except Exception:
        report['state']='error';report['error']=traceback.format_exc()
        if 'samples' in locals():
            (output/'trajectory.json').write_text(json.dumps(samples,allow_nan=False),encoding='utf-8')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    finally:
        if args.no_physics_profiler:
            if previous_profiler is None: process_settings.destroy_item(profiler_setting)
            else: process_settings.set(profiler_setting,previous_profiler)
        if args.physics_threads is not None:
            if previous_threads is None: process_settings.destroy_item(thread_setting)
            else: process_settings.set(thread_setting,previous_threads)
        # Fast Kit shutdown otherwise exits with zero even after an exception.
        app.close(exit_code=0 if report['state'] in ('passed_mechanism_qualification_not_robot_task','interactive_demo_not_qualification','passed_gripper_mechanism_not_robot_task','passed_bimanual_mechanism_not_robot_task','interactive_full_robot_diagnostic','scene_ablation_diagnostic_not_qualification') else 2)


if __name__=='__main__':
    raise SystemExit(main())
