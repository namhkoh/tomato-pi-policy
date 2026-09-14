"""Bounded grasp -> blade load -> withdraw -> retention qualification.

Outputs are diagnostic evidence only, never automatically training-approved.
Grasp is native opposing finger contact; no weld or pose override is used.
"""
import asyncio
import json
import time

import numpy as np

from .gripper_probe import ramp,setup_probe_camera
from .runtime import pose_matrices


def diagnostic_milestones(*,grasp_verified,grasp_time,hold_control,stroke_start,stroke_end,delay,cut_time,
                          through_time=None,retraction_time=None):
    """Observed grasp/cut events, not a nominal time mislabeled as contact."""
    milestones=[('grasp',grasp_time)] if grasp_verified else []
    milestones+=([('hold_10s',10),('hold_20s',20)] if hold_control else [
        ('knife_precontact',stroke_start-.1),('stroke_midpoint',(stroke_start+stroke_end)/2),
        ('late_sequence',17.5+delay)])
    if cut_time is not None:milestones+=[('severed',cut_time),('post_cut_2s',cut_time+2)]
    if through_time is not None:milestones.append(('full_forward_stroke',through_time))
    if retraction_time is not None:milestones.append(('knife_unloaded',retraction_time))
    return milestones


def diagnostic_detail_views(name):
    if name in ('grasp','hold_20s','final'):
        return [('Grasp close-up','close'),('Grasp plant-side','plant_side')]
    if name in ('knife_precontact','severed','post_cut_2s','full_forward_stroke','knife_unloaded'):
        return [('Right knife mount','knife'),('Grasp plant-side','plant_side'),
                ('Blade plane front','blade_front'),('Blade plane back','blade_back')]
    return []


def capture_milestone(name,capture,fixture,viewport):
    """Paused diagnostic views only; restore the user's selected camera."""
    capture(name)
    previous=str(viewport.camera_path)
    try:
        if diagnostic_detail_views(name) and hasattr(fixture,'refresh_inspection_views'):
            fixture.refresh_inspection_views()
        for view,suffix in diagnostic_detail_views(name):
            fixture.select_view(view);capture(name+'_'+suffix)
    finally:
        viewport.set_active_camera(previous)


def released_stroke_fraction(last_command):
    """Latch the sent command, never infer a later command from fetch time."""
    phase,fraction=last_command
    if (phase!='stroke' or isinstance(fraction,(bool,np.bool_))
            or not isinstance(fraction,(int,float,np.integer,np.floating))
            or not np.isfinite(fraction) or not 0<=fraction<=1):
        raise RuntimeError('Blade release without an actually commanded stroke')
    return float(fraction)


def sequence_times(reposition,force_closure=False):
    """Keep legacy times unchanged; allow one second pull +0.5 second settle.

    These schedule motion, never authorize a cut. Native contact, grasp and
    travel evidence still determine the seam release.
    """
    if not np.isfinite(reposition) or not 0<=reposition<=.01:
        raise ValueError('Bounded finite reposition distance required')
    if type(force_closure) is not bool: raise ValueError('Explicit force closure phase required')
    closure_delay=5. if force_closure else 0.
    delay=(1.5 if reposition else 0.)+closure_delay
    return dict(delay=delay,plan=3.5+delay,approach=4.+delay,stroke=8.+delay,end=14.+delay)


def grasp_acquisition_state(t,stable_steps,physics_hz,feedback):
    """Bounded acquisition timing, NEVER an alternative grasp detector.

    stable_steps counts consecutive native, guard-accepted bilateral samples.
    Keep the original 100 ms requirement. Feedback may finish early or spend
    bounded time backing off/re-closing; a scheduled deadline is not contact.
    """
    if (type(feedback) is not bool or type(stable_steps) is not int or stable_steps<0
            or type(physics_hz) is not int or physics_hz<=0
            or not np.isfinite(t) or t<0):raise ValueError('Valid acquisition clock and native dwell required')
    deadline=13.5 if feedback else 3.5
    if t>deadline+1e-9:return 'timeout'
    if t<3.5:return 'waiting'
    if stable_steps>=int(np.ceil(.1*physics_hz)):return 'verified'
    return 'timeout' if t>=deadline else 'waiting'


def schedule_after_grasp(t,reposition):
    """Shift the original complete sequence from the actual verified grasp."""
    if not np.isfinite(t) or not 3.5<=t<=13.5+1e-9:raise ValueError('Bounded verified grasp time required')
    schedule=sequence_times(reposition)
    return {key:value+t-3.5 for key,value in schedule.items()}


def experimental_spring_phase(enabled,grasp_verified,started):
    """Matched hold experiment starts from the existing verified grasp.

    Initialization is explicitly legacy physics, NOT fresh-model qualification.
    Once experimental actuation starts it cannot fall back on loss of a grasp.
    """
    if any(type(v) is not bool for v in (enabled,grasp_verified,started)):
        raise ValueError('Explicit spring experiment phase booleans required')
    if started and (not enabled or not grasp_verified):
        raise RuntimeError('Experimental springs cannot fall back after activation')
    return 'contact' if enabled and grasp_verified else 'initialize' if enabled else 'legacy'


def run(app,sim,rig,runtime,springs,fixture,args,output):
    from greenhouse_sim.physics_clock import PhysicsClock
    ready=None
    if getattr(args,'neutral_ready_start',False):
        from .neutral_ready import run as approach_from_ready
        ready=approach_from_ready(app,sim,rig,runtime,springs,fixture,args,output)
        if not ready['execution_passed']:
            return dict(state='failed_neutral_ready_approach',error=ready['error'],neutral_ready=ready,
                gates={'neutral_ready_approach':False},cut_action={'passed':False},training_eligible=False)
    if (getattr(fixture,'diagnostic_grasp_contacts',False)
            and not getattr(args,'bimanual_hold_control',False)):
        raise ValueError('Raw contact diagnostic is restricted to a right-parked hold control')
    if getattr(fixture,'diagnostic_physics_hz',240)!=args.physics_hz:
        raise ValueError('Robot control/sensing clock must match the physics clock')
    fixture.bind(sim.physics_sim_view)
    coupling_monitor=None
    if getattr(args,'coupled_fingers_trial',False):
        from .finger_coupling import CouplingMonitor
        coupling_monitor=CouplingMonitor()
    cut_only=bool(getattr(args,'right_only_cut_trial',False))
    if getattr(fixture,'cut_strategy','bimanual')!=('right_only' if cut_only else 'bimanual'):
        raise ValueError('Fixture and execution cut strategy disagree')
    if getattr(args,'fixed_root_cut_trial',False):
        from .root_transition import FixedRootTransition
        fixture.root_transition=FixedRootTransition(runtime,springs,fixture)
    step_context=sim
    if getattr(args,'step_profile',False):
        from .step_profile import MeasuredStep
        step_context=MeasuredStep(sim)
    render_options={'wall_render_hz':15} if getattr(args,'watch_cut_trial',False) else {}
    clock=PhysicsClock(step_context,physics_hz=args.physics_hz,render_hz=args.render_hz,**render_options)
    records=[];events=[];captures={};capture_receipts={};fault=None;stable=0;lost=0
    waiting_history=[]  # Only populated by the explicit pre-grasp diagnostic.
    if getattr(args,'stream_trajectory',False):
        from .probe_records import ProbeRecords
        records=ProbeRecords(output/'bimanual_trajectory.jsonl.gz',
            background_compression=getattr(args,'background_evidence_compression',False))
    grasp_local=None;goal_set=False;planned=False;grasp_verified=False;cut_time=None;cut_fraction=0.
    last_right_command=('park',0.)
    through_requested=bool(getattr(args,'through_stroke_trial',False))
    through=None;through_time=None;through_fraction=None;section_binding=None
    separation=None;separation_verified=False
    retraction=None;retraction_time=None
    blade_feed=None
    seam_yield=None
    if getattr(args,'seam_contact_yield',False):
        from .seam_yield import NativeEdgeYield
        seam_yield=NativeEdgeYield(rig,fixture)
    if getattr(args,'blade_force_feed',False):
        from .blade_feed import BladeFeed
        from .knife import DOWNWARD_CUT_MODEL
        feed_options=({'loaded_advance':True}
            if getattr(fixture,'cut_model',None)==DOWNWARD_CUT_MODEL else {})
        # Radius follows the exact existing stroke endpoint construction.
        blade_feed=BladeFeed(fixture.stroke_offsets,
            radius=float(fixture.stroke_offsets[-1])-fixture.knife.size[0]/2-.001,
            dwell_feedback=getattr(args,'blade_dwell_feedback',False),
            compliant_rate=getattr(args,'compliant_blade_rate',False),
            friction_budget=getattr(args,'blade_friction_budget',False),physics_hz=args.physics_hz,**feed_options)
    strain_probe=None
    if getattr(args,'diagnostic_grasp_dynamics',False):
        from .rod_strain import RodStrain
        strain_probe=RodStrain(rig.rest_frames,np.linalg.norm(np.diff(rig.chain_world,axis=0),axis=1),
            [float(rig.stage.GetPrimAtPath(p+'/StemCollider').GetAttribute('radius').Get()) for p in rig.body_paths],
            [p['stiffness'] for p in rig.properties],int(rig.cut_index))
    spring_snapshot=None
    free_root_snapshot=None;free_root_reader=None
    prediction_before=None;prediction_reader=None;contact_stream=None
    previous_prediction=None;contact_springs=None;spring_control_record=None
    contact_springs_started=False
    if getattr(args,'diagnostic_contact_prediction',False):
        from .plant_contact_stream import PlantContactStream
        from .plant_prediction_snapshot import PlantPredictionSnapshot
        from .plant_contact_binding import capture as capture_contact_binding
        if fixture.event_monitor.full_contact_observer is not None:
            raise ValueError('Existing full-contact observer must not be replaced')
        contact_stream=PlantContactStream(plant_colliders=[v[0] for v in fixture.held_plant_screen.local])
        prediction_reader=PlantPredictionSnapshot(runtime.articulation,
            source_target=rig.source_target,expected_body_paths=rig.body_paths[rig.cut_index:])
        # Once per diagnostic run, copy the actual bound local geometry and
        # original material contract. No stage traversal in the physics loop.
        contact_binding=capture_contact_binding(fixture)
        (output/'plant_contact_binding.json').write_text(
            json.dumps(contact_binding,allow_nan=False,indent=2),encoding='utf-8')
        fixture.event_monitor.full_contact_observer=contact_stream
        if getattr(args,'experimental_contact_springs',False):
            if not getattr(args,'bimanual_hold_control',False):
                raise ValueError('Experimental spring actuation is HOLD ONLY')
            from .plant_contact_binding import deserialize
            from .contact_spring_actuation import FreshHoldSprings
            bound=deserialize(contact_binding,source_target=rig.source_target,
                binding_sha256=fixture.grasp_observer.binding_sha256,sha256=contact_binding['sha256'])
            contact_springs=FreshHoldSprings(springs,bound,runtime.drive_diagnostics,
                experimental_hold_only=True)
    release_contacts=None
    if getattr(args,'fixed_root_cut_trial',False) and getattr(args,'diagnostic_grasp_dynamics',False):
        from .release_contact_trace import ReleaseContactTrace
        release_contacts=ReleaseContactTrace(fixture.event_monitor,
            [v[0] for v in fixture.held_plant_screen.local])
    measured_withdrawal=bool(getattr(args,'measured_withdrawal',False));withdrawal=None
    hold_control=bool(getattr(args,'bimanual_hold_control',False))
    reposition=float(getattr(args,'bimanual_reposition_m',0.))
    reposition_vector=None
    force_closure=bool(getattr(fixture,'force_closure_enabled',False))
    preload_settle=None;preload_wait_logged=False
    if getattr(args,'settle_retention_preload',False):
        if not (getattr(args,'require_retention_screen',False) and force_closure
                and fixture.retention_preload and fixture.symmetric_finger_closure and reposition==0.):
            raise ValueError('Preload settling requires original no-reposition feedback retention trial')
        from .retention_settle import PreloadSettle
        preload_settle=PreloadSettle(physics_hz=args.physics_hz)
    times=sequence_times(reposition);delay=times['delay']
    grasp_time=3.5;acquisition_wait_logged=False
    plan_time=times['plan'];approach_start=times['approach'];stroke_start=times['stroke'];stroke_end=times['end']
    viewport=None
    if args.render_hz:
        from pxr import UsdLux
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        previous_view=str(viewport.camera_path)
        if args.scene=='isolated': UsdLux.DomeLight.Define(rig.stage,'/World/ProbeLight').CreateIntensityAttr(1400.)
        viewport.set_active_camera(setup_probe_camera(rig.stage,rig.rest_frames[fixture.body_index,:3,3]))
        fixture.setup_views(viewport)
        if (getattr(args,'robot_interactive',False) or getattr(args,'watch_cut_trial',False)) and previous_view in fixture.views.values():
            viewport.set_active_camera(previous_view)

    def capture(name):
        from omni.kit.viewport.utility import capture_viewport_to_file
        runtime.sync_visuals();before=runtime.frames.copy()
        for _ in range(20): sim.render()
        future=asyncio.ensure_future(capture_viewport_to_file(viewport,str(output/(name+'.png'))).wait_for_result())
        deadline=time.monotonic()+30
        while not future.done():
            if not app.is_running() or time.monotonic()>deadline: raise RuntimeError('Capture failed')
            sim.render()
        future.result();runtime.sample()
        if not np.allclose(before,runtime.frames,atol=1e-8,rtol=0): raise RuntimeError('Capture advanced physics')
        captures[name]=name+'.png'
        capture_receipts[name]=dict(step_id=clock.stamp.step,t=clock.stamp.simulation_time_s,
            camera_path=str(viewport.camera_path),cut=rig.cut,left_grasp_verified=grasp_verified,
            bilateral_contact=records[-1]['contact']['bilateral'] if records else None,
            slip_m=records[-1]['slip_m'] if records else None,
            full_forward_stroke_verified=through_time is not None,
            stroke_retraction_unloaded=retraction_time is not None,
            final_withdrawal_verified=bool(records and records[-1].get('withdrawal',{}).get('right_withdrawal_completed') is True),
            native_guards_passed=records[-1].get('native_guards_passed') if records else None,
            plant_pose_unchanged_during_capture=True,synchronized_rgbd_observation=False,
            purpose='paused_native_viewport_diagnostic_not_training_data')

    def before(stamp,dt):
        nonlocal goal_set,grasp_local,planned,cut_fraction,grasp_verified,last_right_command
        nonlocal reposition_vector
        nonlocal preload_wait_logged
        nonlocal spring_snapshot,prediction_before
        nonlocal springs
        nonlocal free_root_snapshot,free_root_reader
        nonlocal spring_control_record
        nonlocal contact_springs_started
        nonlocal times,grasp_time,delay,plan_time,approach_start,stroke_start,stroke_end,acquisition_wait_logged
        if getattr(args,'watch_cut_trial',False) and not sim.is_playing():
            raise RuntimeError('Timeline changed during watched trial; close and relaunch, no stale-step continuation')
        t=stamp.simulation_time_s
        if getattr(args,'screen_settled_waiting',False) and stamp.step:
            if (not records or records[-1].get('native_guards_passed') is not True
                    or records[-1].get('t')!=t or len(waiting_history)!=int(stamp.step)-1
                    or stamp.step>1024):
                raise RuntimeError('Incomplete native settling history; no proposals')
            waiting_history.append((int(stamp.step),runtime.frames.copy()))
        if seam_yield is not None:seam_yield.apply(step=int(stamp.step))
        if (rig.cut and getattr(args,'native_drives_after_cut',False)
                and not hasattr(springs,'handoff_receipt')):
            if fixture.cut_event is None or fixture.root_transition.receipt is None:
                raise RuntimeError('Native spring handoff requires successful evidence-gated topology transition')
            from .native_spring_observer import restore_after_release
            springs=restore_after_release(springs,physics_hz=args.physics_hz)
            events.append(dict(t=t,event='native_spring_handoff_after_cut',
                step=int(stamp.step),receipt=dict(springs.handoff_receipt)))
        if t>=.9 and not goal_set:
            if not cut_only:fixture.refresh_grasp_goal(runtime.frames)
            fixture.plan_approach()
            grasp_screen=fixture.screen_grasp_scene(runtime.frames) if not cut_only else None
            events.append(dict(t=t,event='left_park_maintained_no_grasp_approach' if cut_only else 'left_grasp_corridor_screened',result=grasp_screen))
            if getattr(args,'screen_settled_waiting',False):
                from .settled_waiting_search import search as waiting_search
                if not records:raise RuntimeError('Missing settled native observation')
                frozen=runtime.frames.copy();native_time=sim.current_time;step=int(stamp.step)
                def frozen_guard():
                    fresh=pose_matrices(runtime.bodies.get_transforms()[runtime.order])
                    if (sim.current_time!=native_time or clock.stamp.step!=step
                            or not np.array_equal(fresh,frozen)):
                        raise RuntimeError('Settled waiting sample changed; no restart proposals')
                proposals=waiting_search(fixture,frozen,records[-1],step=step,time_s=t,
                    physics_hz=args.physics_hz,guard=frozen_guard,history=waiting_history)
                events.append(dict(t=t,event='settled_waiting_restart_proposals',result=proposals))
                raise RuntimeError('Settled waiting diagnostic completed; relaunch and revalidate, no grasp/cut executed')
            if grasp_screen is not None and not grasp_screen['passed']:
                raise RuntimeError('Left grasp corridor intersects unintended plant geometry')
            goal_set=True
        goal=fixture.start[:3,3]+(0. if cut_only else ramp(t,1,2))*(fixture.goal[:3,3]-fixture.start[:3,3])
        if grasp_verified and reposition:
            if reposition_vector is None:raise RuntimeError('Missing checked retraction vector')
            goal+=ramp(t,grasp_time,grasp_time+1)*reposition_vector
        # Keep the target held in place until measured knife withdrawal and a
        # fresh clearance screen authorize a separate transport/reposition.
        # A release+1 s timer cannot establish a clear blade corridor.
        if cut_only:fixture.hold_left_park()
        elif separation is not None:
            q=separation.command(step=int(stamp.step),dt=dt)
            if fixture.event_monitor is not None:fixture.event_monitor.begin_step()
            fixture._command_left_drives(q,fixture.kin.forward('left',q,fixture.base))
            records[-1]['retained_separation_command']=dict(separation.receipt)
        else:fixture.target_palm(goal)
        if force_closure: fixture.close(0. if cut_only else ramp(t,2,3),step=int(stamp.step),dt=dt)
        else: fixture.close(ramp(t,2,3))
        if t>=3.5 and not grasp_verified and not cut_only:
            acquisition=grasp_acquisition_state(t,stable,args.physics_hz,force_closure)
            if acquisition=='timeout':
                # Preserve which actual shapes blocked closure, including a
                # neighboring leaf that cannot count as opposing shaft contact.
                events.append(dict(t=t,event='left_grasp_verification_failed',
                    consecutive_bilateral_steps=stable,
                    last_contact=records[-1]['contact'] if records else None,
                    native_contact_pairs_n=records[-1]['native_contact_pairs_n'] if records else []))
                raise RuntimeError('Left grasp was not stable before knife planning')
            if acquisition=='waiting':
                if not acquisition_wait_logged:
                    events.append(dict(t=t,event='left_grasp_waiting_for_native_contact',
                        deadline_s=13.5,required_bilateral_dwell_s=.1,right_motion_authorized=False))
                    acquisition_wait_logged=True
            else:
                grasp_verified=True;grasp_time=t
                if reposition:
                    from .grasp_frame import checked_approach_retraction
                    reposition_vector=checked_approach_retraction(fixture.start,fixture.goal,reposition)
                    events.append(dict(t=t,event='held_retraction_corridor_selected',
                        direction_source='refreshed_checked_approach_translation_not_palm_axis',
                        requested_distance_m=reposition,displacement_world_m=reposition_vector.tolist(),
                        orientation_source='existing_checked_approach_path',
                        native_grasp_retention_verified=False,current_plant_collision_certified=False))
                times=schedule_after_grasp(t,reposition);delay=times['delay']
                plan_time=times['plan'];approach_start=times['approach']
                stroke_start=times['stroke'];stroke_end=times['end']
                if getattr(args,'require_retention_screen',False):
                    # grasp_local is established below. Fetch one real step
                    # before assessing its slip; never substitute a fake zero.
                    plan_time+=1/args.physics_hz
                events.append(dict(t=t,event='left_grasp_verified',grasp_body=fixture.grasp_path,
                    consecutive_bilateral_steps=stable))
                palm=pose_matrices(fixture.palm.get_transforms())[0]
                grasp_local=(fixture.grasp_point(runtime.frames)-palm[:3,3])@palm[:3,:3]
        if grasp_verified and not planned and preload_settle is not None and t+1e-10>=plan_time:
            settled=preload_settle.observe(records[-1],step=int(stamp.step),time_s=t,start_time_s=grasp_time)
            if settled['state']!='waiting' or not preload_wait_logged:
                events.append(dict(t=t,event='native_preload_settle',result=settled))
                preload_wait_logged=True
            if settled['state']=='timeout':
                raise RuntimeError('Original preload did not settle within bounded hold; no knife motion')
            if settled['state']=='waiting':plan_time=(int(stamp.step)+1)/args.physics_hz
            else:
                # Preserve the entire original approach/stroke duration. A
                # delayed grasp must never jump forward into the cut schedule.
                times=schedule_after_grasp(t,0.)
                plan_time=t;approach_start=times['approach']
                stroke_start=times['stroke'];stroke_end=times['end']
        if (grasp_verified or cut_only) and not planned and not hold_control and t>=plan_time:
            if not cut_only and stable<int(.1*args.physics_hz):
                raise RuntimeError('Held target not stable after reposition; no right-arm execution')
            if cut_only:
                from .cut_only import parked_left
                if not records or records[-1].get('native_guards_passed') is not True:
                    raise RuntimeError('Cut-only requires preceding complete native guards')
                parked_left(fixture,records[-1],step=int(stamp.step),physics_hz=args.physics_hz)
            if getattr(args,'require_retention_screen',False):
                from .retention_preflight import assess as assess_retention
                if not records:raise RuntimeError('No current retention observation')
                capacity=assess_retention(records[-1],step=int(stamp.step),time_s=t,
                    body_paths=rig.body_paths,cut_index=int(rig.cut_index),
                    masses=np.asarray(runtime.bodies.get_masses())[runtime.order],
                    local_coms=np.asarray(runtime.bodies.get_coms())[runtime.order,:3],
                    current_frames=runtime.frames,friction=fixture.friction,physics_hz=args.physics_hz)
                events.append(dict(t=t,event='precut_static_retention_screen',result=capacity))
                from .retention_policy import evaluate as retention_policy
                decision=retention_policy(capacity,native_trial=getattr(args,'native_retention_trial',False))
                events.append(dict(t=t,event='retention_assessment_policy',result=decision))
                if not decision['continue_to_independent_cut_planning']:
                    raise RuntimeError('Static retention capacity not established; knife planning/execution refused')
            events.append(dict(t=t,event='unheld_target_reobserved_before_cut_plan' if cut_only else 'native_grasp_reobserved_before_cut_plan',
                requested_reposition_m=reposition,grasp_body=fixture.grasp_path,
                grasp_world_m=fixture.grasp_point(runtime.frames).tolist(),
                seam_world_m=fixture.seam(runtime.frames)[0].tolist()))
            q=np.degrees(fixture.robot.get_dof_positions()[0,fixture.left_indices])
            positions=fixture.robot.get_dof_positions()[0]
            snapshot=dict(native_body_frames=runtime.frames.tolist(),left_joint_degrees=q.tolist(),
                finger_slides_m={name:float(positions[fixture.names.index(name)]) for name in fixture.slides},
                simulation_time_s=t,target=rig.source_target,training_eligible=False)
            (output/'bimanual_planning_snapshot.json').write_text(json.dumps(snapshot,allow_nan=False),encoding='utf-8')
            fixture.plan_cut(runtime.frames,q);planned=True
            events.append(dict(t=t,event='parked_left_and_right_IK_planned' if cut_only else 'grasp_verified_and_right_IK_planned',plan=fixture.report()['cut_plan']))
        if grasp_verified and t>=grasp_time+.5 and lost>int(.05*args.physics_hz):
            raise RuntimeError('Left grasp lost during bimanual sequence')
        phase='park';fraction=0.
        if planned and t>=approach_start:
            phase='approach';fraction=ramp(t,approach_start,stroke_start)
            if t>=stroke_start:
                phase='stroke';fraction=ramp(t,stroke_start,stroke_end)
                # Planned centre must still match the native seam; no stale
                # target execution following an unmodeled plant movement.
                if cut_time is None and np.linalg.norm(fixture.seam(runtime.frames)[0]-fixture.plan['centre'])>.003:
                    raise RuntimeError('Cut target moved >3 mm since verified plan; reobserve/replan required')
                if cut_time is None and blade_feed is not None:
                    fraction=blade_feed.command(step=int(stamp.step),dt=dt)
            if cut_time is not None and not measured_withdrawal:
                reverse_time=cut_time if not through_requested else through_time
                if through_requested and through_time is None:
                    if through is None:raise RuntimeError('No measured through-stroke release binding')
                    phase='stroke';fraction=through.command(step=int(stamp.step),dt=dt,
                        hold_forward=separation is not None and not separation_verified)
                elif getattr(args,'material_clearance_trial',False):
                    if retraction is None:raise RuntimeError('Missing guarded post-severance retraction')
                    if retraction_time is None:
                        phase='stroke';fraction=retraction.command(step=int(stamp.step),dt=dt)
                    else:
                        if getattr(args,'postcut_egress_trial',False):
                            phase='egress';fraction=ramp(t,retraction_time,
                                retraction_time+fixture.postcut_egress_evidence['duration_s'])
                        else:phase='approach';fraction=1-ramp(t,retraction_time,retraction_time+4)
                elif t<reverse_time+2:
                    phase='stroke';fraction=(through_fraction if through_requested else cut_fraction)*(1-ramp(t,reverse_time,reverse_time+2))
                else:
                    phase='approach';fraction=1-ramp(t,reverse_time+2,reverse_time+6)
            elif cut_time is None and t>=stroke_end and blade_feed is None:
                raise RuntimeError('Cut stroke ended without qualified blade contact; no timed release')
        if cut_time is not None and measured_withdrawal:
            if withdrawal is None:raise RuntimeError('No measured release snapshot; withdrawal refused')
            withdrawal.command(stamp)
        else:
            fixture.cut_authorized=phase=='stroke'
            fixture.command_right(phase,fraction)
            if through is not None and through_time is None and phase=='stroke':
                # Action computed from the preceding accepted observation;
                # never confuse commanded feed with measured material travel.
                records[-1]['through_stroke_command']={key:through.receipt[key] for key in
                    ('step','command_state','command_speed_m_s','command_offset_m','commanded_motion_used_as_completion')}
            last_right_command=(phase,fraction)
        if getattr(fixture,'explicit_finger_effort',False):
            fixture.finger_effort.apply(fixture,step=int(stamp.step),dt=dt)
        fixture.prepare_step(runtime.frames)
        if getattr(fixture,'grasp_contact_frames','post_fetch_legacy')=='pre_solve_pgs_v1':
            fixture.grasp_observer.capture_contact_frames(runtime.frames,
                pose_matrices(fixture.fingers.get_transforms())[fixture.order],step_id=stamp.step)
        if prediction_reader is not None:
            prediction_before=prediction_reader.read(step=stamp.step,root_constrained=not rig.cut)
            # Native body paths and COM velocities, not the desired/FK pose.
            prediction_before['robot_body_paths']=list(fixture.robot_bodies.prim_paths)
            prediction_before['robot_body_frames_world']=pose_matrices(
                fixture.robot_bodies.get_transforms()).tolist()
            prediction_before['robot_body_velocities_world']=np.array(
                fixture.robot_bodies.get_velocities(),dtype=float,copy=True).tolist()
            prediction_before['robot_com_local_poses']=np.array(
                fixture.robot_bodies.get_coms(),dtype=float,copy=True).tolist()
        free_root_snapshot=None
        if rig.cut and getattr(args,'diagnostic_free_root_dynamics',False):
            from .plant_prediction_snapshot import PlantPredictionSnapshot
            from .kinetic_consistency import check as kinetic_check
            if free_root_reader is None:
                free_root_reader=PlantPredictionSnapshot(runtime.articulation,
                    source_target=rig.source_target,expected_body_paths=rig.body_paths[rig.cut_index:])
            try:
                free_root_snapshot=free_root_reader.read(step=int(stamp.step),root_constrained=False)
            except Exception:
                rejected=free_root_reader.last_failure
                if rejected is not None:
                    rejected['kinetic_consistency']=kinetic_check(rejected,
                        runtime.articulation.get_masses(),runtime.articulation.get_inertias())
                    rejected['native_masses']=np.asarray(runtime.articulation.get_masses()).tolist()
                    rejected['native_inertias']=np.asarray(runtime.articulation.get_inertias()).tolist()
                    (output/'free_root_snapshot_fault.json').write_text(
                        json.dumps(rejected,allow_nan=False),encoding='utf-8')
                raise
            free_root_snapshot['kinetic_consistency']=kinetic_check(free_root_snapshot,
                runtime.articulation.get_masses(),runtime.articulation.get_inertias())
        spring_snapshot=None  # Native drive steps do not yield measured/explicit work.
        spring_control_record=None
        spring_phase=experimental_spring_phase(contact_springs is not None,grasp_verified,contact_springs_started)
        if spring_phase=='contact':
            if previous_prediction is None:
                raise RuntimeError('No preceding native contact snapshot; no spring fallback')
            if not contact_springs_started:
                events.append(dict(t=t,event='experimental_springs_activate_after_verified_grasp',
                    initialization_model='legacy_implicit_effort',native_qualified=False))
            try:
                spring_effort,spring_control_record=contact_springs.step(previous_prediction,
                    dict(before=prediction_before,step_id=stamp.step+1,dt_s=dt))
                contact_springs_started=True
            except Exception as exc:
                events.append(dict(t=t,event='experimental_spring_step_rejected',
                    step_id=stamp.step,error=type(exc).__name__+': '+str(exc)))
                raise
        else:
            spring_effort=springs.step(dt,root_constrained=not rig.cut)
            if contact_springs is not None:
                spring_control_record=dict(mode='legacy_initialization_until_verified_grasp',
                    predicted_effort_applied=False,native_qualified=False,training_eligible=False)
        if getattr(args,'diagnostic_grasp_dynamics',False) and spring_effort is not None:
            from .spring_work import capture as spring_capture
            spring_snapshot=spring_capture(runtime.articulation,spring_effort,step=stamp.step)

    def after(stamp,dt):
        nonlocal stable,lost,cut_time,cut_fraction,withdrawal
        nonlocal through,through_time,through_fraction,section_binding
        nonlocal retraction,retraction_time
        nonlocal separation,separation_verified
        nonlocal previous_prediction
        frames,velocity=runtime.sample();frame=frames[fixture.body_index]
        prediction_record=None
        if contact_stream is not None:
            prediction_record=dict(before=prediction_before,contacts=contact_stream.snapshot(),
                step_id=stamp.step,dt_s=dt,after_body_paths=list(rig.body_paths),
                after_body_frames_world=frames.tolist(),after_body_velocities_world=velocity.tolist(),
                after_generalized_velocity=np.r_[runtime.articulation.get_root_velocities()[0],
                    runtime.articulation.get_dof_velocities()[0]].astype(float).tolist(),
                contact_generation_basis='caller_bound_pre_solve_frame_not_native_generation_timestamp',
                contact_impulses_trace_only=True,
                predicted_effort_applied=bool(spring_control_record is not None
                    and spring_control_record.get('actual_effort_write_verified',False)),
                experimental_spring_control=spring_control_record,training_eligible=False)
            previous_prediction=prediction_record
        spring_work=None
        if getattr(args,'diagnostic_grasp_dynamics',False) and spring_snapshot is not None:
            from .spring_work import finish as spring_finish
            spring_work=spring_finish(spring_snapshot,runtime.articulation,
                getattr(springs,'work_stiffness',springs.k),step=stamp.step,dt=dt)
            if hasattr(springs,'work_stiffness'):
                spring_work['scope']='explicit_bending_only_native_torsion_work_unmeasured'
        try:
            c=fixture.contact_with_frames(dt,frames,step_id=stamp.step)
        except Exception as exc:
            events.append(dict(t=stamp.simulation_time_s,event='post_fetch_contact_fault',
                error=type(exc).__name__+': '+str(exc),spring_work=spring_work,
                contact_prediction=prediction_record))
            raise
        stable=stable+1 if c['bilateral'] else 0
        lost=0 if c['bilateral'] else lost+1
        palm=pose_matrices(fixture.palm.get_transforms())[0]
        grasp_point=fixture.grasp_point(frames)
        slip=None if grasp_local is None else float(np.linalg.norm((grasp_point-palm[:3,3])@palm[:3,:3]-grasp_local))
        speed=float(np.linalg.norm(velocity[:,:3],axis=1).max())
        total=float(np.linalg.norm(fixture.all_contacts.get_net_contact_forces(dt),axis=1).max())
        support=float(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1).max())
        seam,_=fixture.seam(frames)
        record=dict(t=stamp.simulation_time_s,contact=c,slip_m=slip,palm=palm.tolist(),
            grasp_point=grasp_point.tolist(),max_speed_m_s=speed,max_gripper_net_contact_n=total,
            support_error_m=support,seam_world=seam.tolist(),
            detached_seam_gap_m=float(np.linalg.norm(seam-rig.chain_world[rig.cut_index])),cut=rig.cut)
        record['attached_root_translation_error_m']=(None if rig.cut else float(np.linalg.norm(
            frames[rig.cut_index,:3,3]-rig.rest_frames[rig.cut_index,:3,3])))
        if getattr(args,'fixed_root_contact_hold',False) or getattr(args,'fixed_root_cut_trial',False) and not rig.cut:
            if rig.cut or not runtime.articulation.shared_metatype.fixed_base:
                raise RuntimeError('Fixed root HOLD topology changed')
            if record['attached_root_translation_error_m']>1e-6:
                raise RuntimeError('Fixed root HOLD attachment moved beyond 1 micrometre')
        # Snapshot after the native step. target_palm() clears the event stream
        # at the next tick; inspecting it in before() loses the blocking pair.
        record['native_contact_pairs_n']=[[a,b,v/dt] for (a,b),v in fixture.event_monitor.pairs.items()]
        # Diagnostic dynamics, not camera observations or training approval.
        plant_q=np.asarray(runtime.articulation.get_dof_positions(),dtype=float)[0]
        plant_v=np.asarray(runtime.articulation.get_dof_velocities(),dtype=float)[0]
        record['plant_dynamics']=dict(joint_positions_rad=plant_q.tolist(),
            joint_velocities_rad_s=plant_v.tolist(),elastic_energy_j=.5*float(np.dot(springs.k*plant_q,plant_q)),
            fastest_body=rig.body_paths[int(np.argmax(np.linalg.norm(velocity[:,:3],axis=1)))])
        records.append(record)
        if release_contacts is not None:
            record['release_contacts']=release_contacts.measured_snapshot(
                step=int(stamp.step),dt=dt,cut=bool(rig.cut))
        if free_root_snapshot is not None:
            record['free_root_prediction_before_step']=free_root_snapshot
        # Preserve failure-tick contact data without invoking release logic
        # before the guards. No sensor/force evidence is manufactured here.
        if blade_feed is not None:
            record['blade_feed']=None if blade_feed.receipt is None else dict(blade_feed.receipt)
            record['native_blade_normal_rows_before_guards']=[dict(r) for r in fixture.edge_contact_rows]
        if getattr(fixture,'explicit_finger_effort',False):
            record['finger_effort_control']=dict(fixture.finger_effort.receipt)
        if prediction_record is not None:record['contact_prediction']=prediction_record
        if getattr(args,'diagnostic_grasp_dynamics',False):
            record['plant_dynamics']['frame_based_strain']=strain_probe.evaluate(frames)
            record['spring_work']=spring_work
            from .grasp_dynamics_evidence import grasp_dynamics_evidence
            positions=fixture.robot.get_dof_positions()[0]
            joint_velocity=fixture.robot.get_dof_velocities()[0]
            if coupling_monitor is not None:
                record['finger_mechanism']=coupling_monitor.observe(
                    positions[fixture.finger_indices],joint_velocity[fixture.finger_indices],step=int(stamp.step))
            fingers=pose_matrices(fixture.fingers.get_transforms())[fixture.order]
            record['grasp_dynamics']=grasp_dynamics_evidence(
                frames,rig.body_paths,fingers,fixture.paths[1:],fixture.grasp_observer.core.rows,
                qf=positions[fixture.finger_indices],qdotf=joint_velocity[fixture.finger_indices],
                targetf=fixture.targets[0,fixture.finger_indices],gravityf=fixture.finger_compensation,
                drivecapsf=fixture.force_limits[0,fixture.finger_indices],dt=dt,step_id=stamp.step)
            record['grasp_dynamics']['right_wrist_world_m']=pose_matrices(fixture.right_palm.get_transforms())[0].tolist()
            record['grasp_dynamics']['robot_joint_names']=fixture.names
            record['grasp_dynamics']['robot_joint_velocities']=joint_velocity.tolist()
            record['grasp_dynamics']['robot_joint_velocity_units']=[
                'm/s' if name.startswith('gripper_finger') else 'rad/s' for name in fixture.names]
        if not c['adapter_valid']:
            raise RuntimeError('Native shaft grasp callback/tensor force reconciliation failed')
        fixture.check_plant_window(frames)
        record['robot']=fixture.check(dt,palm)
        from .bimanual_limits import violations
        failures=violations(record)
        if failures:
            record['execution_limit_failures']=failures
            events.append(dict(t=stamp.simulation_time_s,event='native_execution_limit',failures=failures))
            raise RuntimeError('Bimanual execution limit: '+json.dumps(failures,allow_nan=False))
        if cut_only:
            from .cut_only import parked_left
            record['cut_only_park']=parked_left(fixture,record,step=int(stamp.step),physics_hz=args.physics_hz)
            record['knife']=fixture.inspect_cut(dt,frames,False,None,cut_only_ready=planned)
        else:
            record['knife']=fixture.inspect_cut(dt,frames,stable>=int(.025*args.physics_hz),slip)
        record['cut']=rig.cut
        # True only here: every existing callback, robot, force, penetration,
        # support, slip and knife guard above has returned without exception.
        record['native_guards_passed']=True
        if seam_yield is not None:seam_yield.observe(record,step=int(stamp.step))
        if blade_feed is not None:
            blade_feed.observe(record['knife'],step=int(stamp.step),guards_passed=True,released=bool(rig.cut))
        if force_closure:
            record['force_closure']=dict(fixture.force_closer.receipt)
            fixture.force_closer.observe(c,
                [record['robot']['per_finger_contact_upper_bound_n'][p] for p in fixture.paths[1:]],
                step=int(stamp.step),guards_passed=True)
        if rig.cut and cut_time is None:
            cut_time=stamp.simulation_time_s
            # The release timestamp is AFTER fetch; it is one step later than
            # the command that loaded the blade. Re-evaluating the ramp here
            # would add an unrequested forward increment after release.
            cut_fraction=released_stroke_fraction(last_right_command)
            events.append(dict(t=cut_time,**fixture.cut_event))
            if getattr(args,'cut_action_trial',False):
                from .released_debris import ReleasedDebris
                if fixture.event_monitor.released_debris_policy is not None:
                    raise RuntimeError('Debris policy cannot replace an existing release binding')
                policy=ReleasedDebris(fixture)
                fixture.event_monitor.released_debris_policy=policy
                events.append(dict(t=cut_time,event='released_debris_landing_deferred',policy=policy.report()))
        if rig.cut and measured_withdrawal:
            from .withdrawal_controller import WithdrawalController
            if withdrawal is None:
                withdrawal=WithdrawalController(fixture,runtime,clock,cut_fraction=cut_fraction)
            try:
                withdrawal.observe(stamp,record)
            except Exception as exc:
                record['withdrawal_failure']=dict(error=type(exc).__name__+': '+str(exc),
                    native_receipt=withdrawal.adapter.last_receipt)
                raise
        if rig.cut and getattr(args,'retained_separation_trial',False):
            from .retained_separation import plan as plan_separation,recheck as recheck_separation
            if separation is None:
                separation=plan_separation(fixture,frames,record=record,step=int(stamp.step),
                    read_frames=lambda:pose_matrices(runtime.bodies.get_transforms()[runtime.order]))
                events.append(dict(t=stamp.simulation_time_s,event='retained_separation_planned',
                    evidence=fixture.retained_separation_evidence))
            record['retained_separation']=separation.observe(record,step=int(stamp.step),target=frames[fixture.body_index])
            if separation.complete and not separation_verified:
                evidence=recheck_separation(fixture,frames,record=record,step=int(stamp.step),
                    read_frames=lambda:pose_matrices(runtime.bodies.get_transforms()[runtime.order]))
                separation_verified=True
                events.append(dict(t=stamp.simulation_time_s,event='retained_separation_measured_and_reobserved',
                    evidence=evidence,measurement=dict(separation.receipt)))
        if rig.cut and through_requested and through_time is None:
            from .through_stroke import ThroughStroke
            if through is None:
                if fixture.cut_event is None or fixture.root_transition.receipt is None:
                    raise RuntimeError('Through-stroke requires validated native release transition')
                endpoint=fixture.knife.frame(fixture.kin.forward('right',fixture.plan['stroke'][-1],fixture.base))[:3,3]
                options={'maximum_feed_m_s':getattr(args,'postrelease_feed_m_s',.0003)}
                if getattr(args,'support_aware_feed_trial',False):options['support_aware_feed']=True
                if getattr(args,'proportional_face_backoff_trial',False):options['proportional_face_backoff']=True
                if getattr(args,'material_clearance_trial',False):
                    parent_index=rig.cut_index-1
                    centre,axis=fixture.seam(frames)
                    parent=frames[parent_index]
                    section_binding=(parent_index,(centre-parent[:3,3])@parent[:3,:3],axis@parent[:3,:3])
                    faces=[rig.body_paths[i]+'/StemCollider' for i in (parent_index,rig.cut_index)]
                    radius=max(float(rig.stage.GetPrimAtPath(p).GetAttribute('radius').Get()) for p in faces)
                    options['material_section']=dict(radius=radius,tip_offset=float(fixture.knife.size[0]/2),
                        half_span=float(fixture.knife.size[1]/2),knife=fixture.knife.collider,faces=faces)
                    events.append(dict(t=stamp.simulation_time_s,event='proximal_material_section_bound',
                        body=rig.body_paths[parent_index],source_target=rig.source_target,
                        reference='current_post_fetch_proximal_body_not_falling_detached_branch',
                        section=options['material_section'],fracture_calibrated=False))
                through=ThroughStroke(fixture.stroke_offsets,fraction=cut_fraction,endpoint=endpoint,
                    direction=fixture.plan['direction'],physics_hz=args.physics_hz,step=int(stamp.step),**options)
            support=bool(c['bilateral'] and slip is not None and slip<.003) if not cut_only else 'cut_only_park' in record
            actual=pose_matrices(fixture.right_palm.get_transforms())[0]
            options={}
            if section_binding is not None:
                i,local_centre,local_axis=section_binding;parent=frames[i]
                options['section_pose']=dict(centre=parent[:3,3]+parent[:3,:3]@local_centre,
                    axis=parent[:3,:3]@local_axis)
            try:
                record['through_stroke']=through.observe(record,step=int(stamp.step),
                    arc_up=fixture.knife.arc_up(actual),support_ready=support,**options)
            except Exception as exc:
                record['through_stroke_failure']=dict(error=type(exc).__name__+': '+str(exc),
                    endpoint_world_m=through.endpoint.tolist(),direction_world=through.direction.tolist(),
                    edge_frame=record['knife']['edge_frame'],arc_up=fixture.knife.arc_up(actual).tolist())
                raise
            if through.complete:
                through_time=stamp.simulation_time_s
                through_fraction=released_stroke_fraction(last_right_command)
                events.append(dict(t=through_time,event='measured_forward_stroke_complete',evidence=dict(through.receipt)))
        if through_time is not None and getattr(args,'material_clearance_trial',False) and retraction_time is None:
            from .cut_retraction import CutRetraction
            from .postrelease_feed import reverse_maximum
            if retraction is None:
                endpoint=fixture.knife.frame(fixture.kin.forward('right',fixture.plan['stroke'][0],fixture.base))[:3,3]
                retraction=CutRetraction(fixture.stroke_offsets,fraction=through_fraction,endpoint=endpoint,
                    direction=fixture.plan['direction'],step=int(stamp.step),section=through.material_section,
                    # Qualify faster INSERTION separately. Native435's faster
                    # reverse recontacted during egress; retain the original
                    # reverse profile in the support-aware insertion trial.
                    maximum_feed_m_s=reverse_maximum(getattr(args,'postrelease_feed_m_s',.0003),
                        support_aware_insertion=getattr(args,'support_aware_feed_trial',False)))
            support=bool(c['bilateral'] and slip is not None and slip<.003) if not cut_only else 'cut_only_park' in record
            record['cut_retraction']=retraction.observe(record,step=int(stamp.step),support_ready=support)
            if retraction.complete:
                retraction_time=stamp.simulation_time_s
                events.append(dict(t=retraction_time,event='measured_stroke_retraction_unloaded',evidence=dict(retraction.receipt)))
                if getattr(args,'postcut_egress_trial',False):
                    from .postcut_egress import plan as plan_egress
                    path,evidence=plan_egress(fixture,frames,step=int(stamp.step),record=record,
                        retraction=retraction.receipt)
                    fixture.plan['egress']=path
                    events.append(dict(t=retraction_time,event='postcut_egress_planned',evidence=evidence))
        if rig.cut and stamp.step==int(args.seconds*args.physics_hz):
            from .withdrawal_native_check import check as check_withdrawal
            options={}
            if getattr(args,'native_station_park_reference',False):options['native_station_park']=True
            if getattr(args,'postcut_egress_trial',False):options['postcut_egress']=True
            record['withdrawal']=check_withdrawal(fixture,frames,stamp.step,**options)
            events.append(dict(t=stamp.simulation_time_s,event='final_native_withdrawal_endpoint',
                evidence=record['withdrawal']))
        if retraction_time is not None and getattr(args,'postcut_egress_trial',False):
            from .seam_escape import execution_guard
            execution_guard(record)
            record['postcut_egress_unloaded_guard_passed']=True
        record['phase']='Cut-only / withdraw' if cut_only and rig.cut else 'Left parked / right cut' if cut_only else 'Left hold negative control' if hold_control else 'Retain / withdraw' if rig.cut else 'Blade stroke' if planned and stamp.simulation_time_s>=stroke_start else 'Right approach' if planned and stamp.simulation_time_s>=approach_start else 'Held target reposition / reobserve' if grasp_verified and reposition else 'Left grasp'
        if rig.cut and through_requested and through_time is None:record['phase']='Measured downward follow-through'
        fixture.on_sample(record)

    def render_state(_):
        runtime.sync_visuals()
        if hasattr(fixture,'target_markers'): fixture.target_markers.update(runtime.frames)

    print('BIMANUAL_PROBE_READY '+json.dumps(fixture.report()),flush=True)
    try:
        if viewport and args.capture_milestones:
            capture('initial')
            previous=str(viewport.camera_path)
            fixture.select_view('Right knife mount');capture('knife_mount')
            viewport.set_active_camera(previous)
        for _ in range(int(args.seconds*args.physics_hz)):
            tick=time.monotonic()
            if not app.is_running() or fixture.stop_requested: raise RuntimeError('Stopped; reset required')
            clock.tick(before=before,after=after,before_render=render_state)
            if clock.stamp.step%args.physics_hz==0 and records:
                latest=records[-1]
                status=dict(t=latest['t'],phase=latest['phase'],
                    bilateral=latest['contact']['bilateral'],slip_m=latest['slip_m'],cut=rig.cut)
                if seam_yield is not None:
                    status.update(yield_state=latest['seam_yield']['state'],
                        seam_stiffness_n_m=latest['seam_yield']['applied_stiffness_n_m'],
                        edge_resistance_n=latest['knife']['edge_signed_resistance_n'],
                        qualified_travel_m=latest['knife']['gate_travel_m'])
                if latest.get('through_stroke') is not None:status['through_stroke']=latest['through_stroke']
                print('BIMANUAL_SECOND '+json.dumps(status),flush=True)
            if viewport and args.capture_milestones:
                milestones=diagnostic_milestones(grasp_verified=grasp_verified,grasp_time=grasp_time,
                    hold_control=hold_control,stroke_start=stroke_start,stroke_end=stroke_end,
                    delay=delay,cut_time=cut_time,through_time=through_time,retraction_time=retraction_time)
                for name,t in milestones:
                    if clock.stamp.simulation_time_s>=t and name not in captures:
                        capture_milestone(name,capture,fixture,viewport)
                if clock.stamp.step==int(args.seconds*args.physics_hz):
                    capture_milestone('final',capture,fixture,viewport)
            if args.gui: time.sleep(max(0,1/args.physics_hz-(time.monotonic()-tick)))
    except Exception as exc:
        fault=str(exc)
        if viewport and app.is_running():
            try: capture('stopped_on_fault')
            except Exception as error: captures['fault_capture_error']=str(error)
    finally:
        if getattr(args,'stream_trajectory',False):records.close()
    retained=[r for r in records if cut_time is not None and r['t']>=cut_time+2]
    from .withdrawal_controller import measured_completion
    helper_complete=bool(records and measured_completion(records[-1],
        (clock.stamp.episode,clock.stamp.step))) if measured_withdrawal else None
    gates=dict(bounded=fault is None,completed=len(records)==int(args.seconds*args.physics_hz),
        left_grasp_verified=grasp_verified,collision_clear_right_plan=planned,blade_contact_release=rig.cut,
        native_retention=bool(retained) and all(r['contact']['bilateral'] and r['slip_m']<.003 for r in retained),
        released_material_separates=bool(retained) and max(r['detached_seam_gap_m'] for r in retained)>.003,
        right_withdrawal_completed=bool(records
            and records[-1].get('withdrawal',{}).get('right_withdrawal_completed') is True
            and (not measured_withdrawal or helper_complete)))
    if cut_only:
        gates.pop('left_grasp_verified');gates.pop('native_retention')
        gates['left_parked_open_unloaded']=bool(records) and all('cut_only_park' in r for r in records)
    if through_requested:gates['full_forward_cut_stroke_verified']=through_time is not None
    if getattr(args,'material_clearance_trial',False):
        gates['contact_paced_retraction_unloaded']=retraction_time is not None
    if ready is not None:gates['neutral_ready_approach']=ready['execution_passed']
    if getattr(args,'retained_separation_trial',False):
        gates['retained_separation_reobserved']=separation_verified
    passed_state='passed_cut_only_mechanism_not_robot_task' if cut_only else 'passed_bimanual_mechanism_not_robot_task'
    result=dict(state=passed_state if all(gates.values()) else 'failed_cut_only_qualification' if cut_only else 'failed_bimanual_qualification',
        cut_strategy='right_only' if cut_only else 'bimanual',left_grasp_verified=grasp_verified,
        retention_assessment_mode=('native_outcome_experiment' if getattr(args,'native_retention_trial',False)
            else 'not_required_right_only' if cut_only else 'static_prerequisite_and_native_outcome'),
        native_retention_trial_requested=bool(getattr(args,'native_retention_trial',False)),
        retention_expected=not cut_only,dropped_material_expected=cut_only,drop_corridor_certified=False,
        gates=gates,error=fault,events=events,images=captures,image_evidence=capture_receipts,
        timing=clock.report(),robot=fixture.report(),
        measurements=dict(native_edge_contact_count=fixture.cut_contacts,cut_time_s=cut_time,
            bilateral_contact_fraction_before_cut_plan=float(np.mean([r['contact']['bilateral'] for r in records if grasp_time-.5<=r['t']<=grasp_time])) if any(grasp_time-.5<=r['t']<=grasp_time for r in records) else None,
            maximum_slip_m=max((r['slip_m'] for r in records if r['slip_m'] is not None),default=None)),
        physical_cut_verified=False,tissue_fracture_calibrated=False,deposit_verified=False,
        right_withdrawal_schedule_elapsed=(through_time if through_requested else cut_time) is not None
            and records[-1]['t']>=(through_time if through_requested else cut_time)+6,
        right_withdrawal_verification='final_post_fetch_native_endpoint_and_fresh_clearance_not_elapsed_time',
        full_forward_cut_stroke_verified=through_time is not None,
        through_stroke_requested=through_requested,
        through_stroke=None if through is None or through.receipt is None else dict(through.receipt),
        contact_paced_retraction=None if retraction is None or retraction.receipt is None else dict(retraction.receipt),
        contact_paced_retraction_completed_time_s=retraction_time,
        measured_withdrawal_requested=measured_withdrawal,
        measured_withdrawal_completed=helper_complete,
        negative_control_no_right_motion=hold_control,
        requested_pre_cut_reposition_m=reposition,
        retained_separation=getattr(fixture,'retained_separation_evidence',None),
        retained_separation_verified=separation_verified,
        measured_finger_mechanism=None if not records else records[-1].get('finger_mechanism'),
        postcut_egress=getattr(fixture,'postcut_egress_evidence',None),
        grasp_acquisition=dict(feedback_event_driven=force_closure,
            earliest_verification_s=3.5,deadline_s=13.5 if force_closure else 3.5,
            required_native_bilateral_dwell_s=.1,verified_time_s=grasp_time if grasp_verified else None,
            sequence_schedule=times if grasp_verified else None),
        target_source='privileged_test_fixture_not_perception_verified',training_eligible=False)
    if ready is not None:
        result['neutral_ready']=ready
        result['cut_protocol_native_time_origin_s']=ready['native_end_time_s']
    if getattr(args,'material_clearance_trial',False):
        result['right_withdrawal_schedule_elapsed']=bool(retraction_time is not None
            and records and records[-1]['t']>=retraction_time+4)
    if step_context is not sim: result['step_profile']=step_context.report()
    if getattr(args,'cut_action_trial',False):
        from .cut_action import assess
        action=assess(cut_only=cut_only,grasp_verified=grasp_verified,planned=planned,
            cut_event=fixture.cut_event,cut_time=cut_time,records=records,fault=fault)
        result['full_sequence_state']=result['state']
        result['full_sequence_qualified']=all(gates.values())
        result['cut_action']=action
        result['state']='passed_cut_action_not_complete_robot_task' if action['passed'] else 'failed_cut_action'
        if through_requested and not all(gates.values()):result['state']='failed_through_stroke_qualification'
        result['measurements']['maximum_released_debris_contact_n']=max(
            (r.get('robot',{}).get('released_debris_contact_n',0.) for r in records),default=0.)
    if getattr(args,'stream_trajectory',False):
        records.close();result['trajectory_storage']=records.report()
    else:
        (output/'bimanual_trajectory.json').write_text(json.dumps(records,allow_nan=False),encoding='utf-8')
    if contact_stream is not None:fixture.event_monitor.full_contact_observer=None
    fixture.release_grasp_observer()
    if getattr(args,'watch_cut_trial',False):sim.pause()
    else:sim.stop()
    return result
