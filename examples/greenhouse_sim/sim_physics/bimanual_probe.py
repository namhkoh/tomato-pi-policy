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


def sequence_times(reposition):
    """Keep legacy times unchanged; allow one second pull +0.5 second settle.

    These schedule motion, never authorize a cut. Native contact, grasp and
    travel evidence still determine the seam release.
    """
    if not np.isfinite(reposition) or not 0<=reposition<=.01:
        raise ValueError('Bounded finite reposition distance required')
    delay=1.5 if reposition else 0.
    return dict(delay=delay,plan=3.5+delay,approach=4.+delay,stroke=8.+delay,end=14.+delay)


def run(app,sim,rig,runtime,springs,fixture,args,output):
    from greenhouse_sim.physics_clock import PhysicsClock
    fixture.bind(sim.physics_sim_view)
    clock=PhysicsClock(sim,physics_hz=args.physics_hz,render_hz=args.render_hz)
    records=[];events=[];captures={};fault=None;stable=0;lost=0
    grasp_local=None;goal_set=False;planned=False;grasp_verified=False;cut_time=None;cut_fraction=0.
    hold_control=bool(getattr(args,'bimanual_hold_control',False))
    reposition=float(getattr(args,'bimanual_reposition_m',0.))
    times=sequence_times(reposition);delay=times['delay']
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
        if getattr(args,'robot_interactive',False) and previous_view in fixture.views.values():
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

    def before(stamp,dt):
        nonlocal goal_set,grasp_local,planned,cut_fraction,grasp_verified
        t=stamp.simulation_time_s
        if t>=.9 and not goal_set:
            fixture.goal[:3,3]=runtime.frames[fixture.body_index,:3,3]+fixture.grasp_depth*fixture.goal[:3,2]
            fixture.plan_approach()
            grasp_screen=fixture.screen_grasp_scene(runtime.frames)
            events.append(dict(t=t,event='left_grasp_corridor_screened',result=grasp_screen))
            if not grasp_screen['passed']:
                raise RuntimeError('Left grasp corridor intersects unintended plant geometry')
            goal_set=True
        goal=fixture.start[:3,3]+ramp(t,1,2)*(fixture.goal[:3,3]-fixture.start[:3,3])
        if grasp_verified and reposition:
            goal+=reposition*ramp(t,3.5,4.5)*fixture.goal[:3,2]
        if cut_time is not None:
            goal+=.008*ramp(t,cut_time+1,cut_time+2)*fixture.goal[:3,2]
        fixture.target_palm(goal);fixture.close(ramp(t,2,3))
        if t>=3.5 and not grasp_verified:
            if stable<int(.1*args.physics_hz):
                # Preserve which actual shapes blocked closure, including a
                # neighboring leaf that cannot count as opposing shaft contact.
                events.append(dict(t=t,event='left_grasp_verification_failed',
                    consecutive_bilateral_steps=stable,
                    last_contact=records[-1]['contact'] if records else None,
                    native_contact_pairs_n=records[-1]['native_contact_pairs_n'] if records else []))
                raise RuntimeError('Left grasp was not stable before knife planning')
            grasp_verified=True
            events.append(dict(t=t,event='left_grasp_verified',grasp_body=fixture.grasp_path,
                consecutive_bilateral_steps=stable))
            palm=pose_matrices(fixture.palm.get_transforms())[0]
            grasp_local=(runtime.frames[fixture.body_index,:3,3]-palm[:3,3])@palm[:3,:3]
        if grasp_verified and not planned and not hold_control and t>=plan_time:
            if stable<int(.1*args.physics_hz):
                raise RuntimeError('Held target not stable after reposition; no right-arm execution')
            events.append(dict(t=t,event='native_grasp_reobserved_before_cut_plan',
                requested_reposition_m=reposition,grasp_body=fixture.grasp_path,
                grasp_world_m=runtime.frames[fixture.body_index,:3,3].tolist(),
                seam_world_m=fixture.seam(runtime.frames)[0].tolist()))
            q=np.degrees(fixture.robot.get_dof_positions()[0,fixture.left_indices])
            positions=fixture.robot.get_dof_positions()[0]
            snapshot=dict(native_body_frames=runtime.frames.tolist(),left_joint_degrees=q.tolist(),
                finger_slides_m={name:float(positions[fixture.names.index(name)]) for name in fixture.slides},
                simulation_time_s=t,target=rig.source_target,training_eligible=False)
            (output/'bimanual_planning_snapshot.json').write_text(json.dumps(snapshot,allow_nan=False),encoding='utf-8')
            fixture.plan_cut(runtime.frames,q);planned=True
            events.append(dict(t=t,event='grasp_verified_and_right_IK_planned',plan=fixture.report()['cut_plan']))
        if t>=4 and lost>int(.05*args.physics_hz):
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
            if cut_time is not None:
                if t<cut_time+2:
                    phase='stroke';fraction=cut_fraction*(1-ramp(t,cut_time,cut_time+2))
                else:
                    phase='approach';fraction=1-ramp(t,cut_time+2,cut_time+6)
            elif t>=stroke_end:
                raise RuntimeError('Cut stroke ended without qualified blade contact; no timed release')
        fixture.cut_authorized=phase=='stroke'
        fixture.command_right(phase,fraction)
        fixture.prepare_step(runtime.frames)
        springs.step(dt,root_constrained=not rig.cut)

    def after(stamp,dt):
        nonlocal stable,lost,cut_time,cut_fraction
        frames,velocity=runtime.sample();frame=frames[fixture.body_index]
        c=fixture.contact(dt,frame);stable=stable+1 if c['bilateral'] else 0
        lost=0 if c['bilateral'] else lost+1
        palm=pose_matrices(fixture.palm.get_transforms())[0]
        slip=None if grasp_local is None else float(np.linalg.norm((frame[:3,3]-palm[:3,3])@palm[:3,:3]-grasp_local))
        speed=float(np.linalg.norm(velocity[:,:3],axis=1).max())
        total=float(np.linalg.norm(fixture.all_contacts.get_net_contact_forces(dt),axis=1).max())
        support=float(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1).max())
        seam,_=fixture.seam(frames)
        record=dict(t=stamp.simulation_time_s,contact=c,slip_m=slip,palm=palm.tolist(),
            grasp_point=frame[:3,3].tolist(),max_speed_m_s=speed,max_gripper_net_contact_n=total,
            support_error_m=support,seam_world=seam.tolist(),
            detached_seam_gap_m=float(np.linalg.norm(seam-rig.chain_world[rig.cut_index])),cut=rig.cut)
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
        fixture.check_plant_window(frames)
        record['robot']=fixture.check(dt,palm)
        if (speed>20 or total>3 or support>1e-5 or c['min_separation']<-.001
                or record['robot']['allowed_tool_contact_n']>.5
                or record['robot']['minimum_tool_separation_m']<-.001
                or (slip is not None and slip>.003)):
            raise RuntimeError('Bimanual force/slip/penetration/support guard')
        record['knife']=fixture.inspect_cut(dt,frames,stable>=int(.025*args.physics_hz),slip)
        record['cut']=rig.cut
        if rig.cut and cut_time is None:
            cut_time=stamp.simulation_time_s;cut_fraction=ramp(cut_time,stroke_start,stroke_end)
            events.append(dict(t=cut_time,**fixture.cut_event))
        record['phase']='Left hold negative control' if hold_control else 'Retain / withdraw' if rig.cut else 'Blade stroke' if stamp.simulation_time_s>=stroke_start else 'Right approach' if stamp.simulation_time_s>=approach_start else 'Held target reposition / reobserve' if grasp_verified and reposition else 'Left grasp'
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
                print('BIMANUAL_SECOND '+json.dumps(dict(t=latest['t'],phase=latest['phase'],
                    bilateral=latest['contact']['bilateral'],slip_m=latest['slip_m'],cut=rig.cut)),flush=True)
            if viewport and args.capture_milestones:
                milestones=(('grasp',3.4),('hold_10s',10),('hold_20s',20)) if hold_control else (
                    ('grasp',3.4),('knife_precontact',stroke_start-.1),('stroke_midpoint',(stroke_start+stroke_end)/2),('late_sequence',17.5+delay))
                for name,t in milestones:
                    if clock.stamp.simulation_time_s>=t and name not in captures:
                        capture(name)
                        if name in ('grasp','hold_20s'):
                            previous=str(viewport.camera_path)
                            fixture.select_view('Grasp close-up');capture(name+'_close')
                            fixture.select_view('Grasp plant-side');capture(name+'_plant_side')
                            viewport.set_active_camera(previous)
                if rig.cut and 'severed' not in captures: capture('severed')
            if args.gui: time.sleep(max(0,1/args.physics_hz-(time.monotonic()-tick)))
    except Exception as exc:
        fault=str(exc)
        if viewport and app.is_running():
            try: capture('stopped_on_fault')
            except Exception as error: captures['fault_capture_error']=str(error)
    retained=[r for r in records if cut_time is not None and r['t']>=cut_time+2]
    gates=dict(bounded=fault is None,completed=len(records)==int(args.seconds*args.physics_hz),
        left_grasp_verified=grasp_verified,collision_clear_right_plan=planned,blade_contact_release=rig.cut,
        native_retention=bool(retained) and all(r['contact']['bilateral'] and r['slip_m']<.003 for r in retained),
        released_material_separates=bool(retained) and max(r['detached_seam_gap_m'] for r in retained)>.003,
        right_withdrawal_completed=cut_time is not None and records[-1]['t']>=cut_time+6)
    result=dict(state='passed_bimanual_mechanism_not_robot_task' if all(gates.values()) else 'failed_bimanual_qualification',
        gates=gates,error=fault,events=events,images=captures,timing=clock.report(),robot=fixture.report(),
        measurements=dict(native_edge_contact_count=fixture.cut_contacts,cut_time_s=cut_time,
            bilateral_contact_fraction_before_cut_plan=float(np.mean([r['contact']['bilateral'] for r in records if 3<=r['t']<=3.5])) if any(3<=r['t']<=3.5 for r in records) else None,
            maximum_slip_m=max((r['slip_m'] for r in records if r['slip_m'] is not None),default=None)),
        physical_cut_verified=False,tissue_fracture_calibrated=False,deposit_verified=False,
        negative_control_no_right_motion=hold_control,
        requested_pre_cut_reposition_m=reposition,
        target_source='privileged_test_fixture_not_perception_verified',training_eligible=False)
    (output/'bimanual_trajectory.json').write_text(json.dumps(records,allow_nan=False),encoding='utf-8')
    sim.stop()
    return result
