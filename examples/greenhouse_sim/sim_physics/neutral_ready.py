"""Explicit neutral-arm approach before the privileged grasp/cut fixture.

No torso motion, native pose reset, attachment release or contact allowance.
Initialization is pre-physics only. Execution uses ordinary effort-limited
joint drives, fresh contact guards and a separately reported phase clock.
"""
import json
import time
import numpy as np


# Planning reserve only, not a changed collision shape/contact tolerance.
# Native438 tracked the wrist 1.35 mm off a path screened at only 1 mm.
TRANSIT_MARGIN_M=.005


def contact_diagnostic(monitor,dt):
    """Copy exact native pair loads for a rejected step; no contact authority."""
    if isinstance(dt,bool) or not np.isfinite(dt) or dt<=0:
        raise ValueError('Finite positive native contact timestep required')
    rows=[]
    for pair,impulse in monitor.pairs.items():
        normal=monitor.normal_pairs[pair];friction=monitor.friction_pairs[pair]
        if len(pair)!=2 or not np.isfinite([impulse,normal,friction]).all() or min(impulse,normal,friction)<0:
            raise ValueError('Finite nonnegative original contact accounting required')
        rows.append(dict(collider0=pair[0],collider1=pair[1],
            normal_upper_bound_n=normal/dt,friction_upper_bound_n=friction/dt,
            total_upper_bound_n=impulse/dt))
    return rows


def ready_arms():
    from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES as ready
    return np.array([ready[f'{arm}_arm_{i}'] for arm in ('left','right') for i in range(7)])


def apply_open_finger_effort(fixture,*,step,dt):
    """Keep the existing open target, with the SAME closure-PD effort ceiling."""
    fixture.force_limits[0,fixture.finger_indices]=np.minimum(
        fixture.force_limits[0,fixture.finger_indices],fixture.force_closer.drive_limit_n)
    fixture.robot.set_dof_max_forces(fixture.force_limits,fixture.index)
    fixture.finger_effort.apply(fixture,step=step,dt=dt)


def initialize(fixture):
    if hasattr(fixture,'robot') or hasattr(fixture,'neutral_ready'):
        raise RuntimeError('Neutral start must be authored once BEFORE native binding')
    initial=ready_arms();goal=np.r_[fixture.initial_q,fixture.right]
    fixture.neutral_goal_pose=dict(fixture.pose)
    for j,arm in enumerate(('left','right')):
        fixture.pose.update({f'{arm}_arm_{i}':float(initial[7*j+i]) for i in range(7)})
    fixture.restore_authored_state()
    fixture.neutral_ready=dict(model='sdk_arms_ready_to_fixture_v1',
        initial_arm_degrees=initial.tolist(),goal_arm_degrees=goal.tolist(),
        torso_degrees=[fixture.pose[f'torso_{i}'] for i in range(6)],
        torso_moved=False,base_moved=False,authored_before_native_parse=True,
        execution_passed=False,cut_authorized=False,training_eligible=False)
    return dict(fixture.neutral_ready)


def time_path(path, *, speed=15.):
    """Smooth each checked interval; <=15 deg/s, no geometric shortcut."""
    q=np.asarray(path,float)
    if (q.ndim!=2 or q.shape[1]!=14 or len(q)<2 or not np.isfinite(q).all()
            or isinstance(speed,bool) or not np.isfinite(speed) or not 0<speed<=15.):
        raise ValueError('Finite two-arm path and bounded speed required')
    distance=np.max(abs(np.diff(q,axis=0)),axis=1)
    # Smoothstep peak derivative is 1.5. Minimum segment duration prevents
    # zero-time edges; this timing does NOT authorize a collision-free path.
    duration=np.maximum(.02,1.5*distance/speed)
    return np.r_[0.,np.cumsum(duration)]


def interpolate(path,knots,t):
    q=np.asarray(path,float);knots=np.asarray(knots,float)
    if (q.ndim!=2 or q.shape[1]!=14 or len(q)<2 or knots.shape!=(len(q),)
            or not np.isfinite(q).all() or not np.isfinite(knots).all()
            or knots[0]!=0 or not np.all(np.diff(knots)>0)
            or isinstance(t,bool) or not np.isfinite(t) or t<0):
        raise ValueError('Finite bounded neutral transit clock required')
    i=min(int(np.searchsorted(knots,t,side='right')-1),len(q)-2)
    f=float(np.clip((t-knots[i])/(knots[i+1]-knots[i]),0,1));f=f*f*(3-2*f)
    return (1-f)*q[i]+f*q[i+1]


def plan(fixture,frames,start):
    """Bounded current-scene proposal, never evidence of executed motion."""
    from .held_plant_screen import HeldPlantScreen
    from .whole_robot_target import WholeRobotTargetScreen
    from .native_static_clearance import current_scene_query
    from .joint_path import connect_path
    goal=np.asarray(fixture.neutral_ready['goal_arm_degrees'],float)
    screens=[HeldPlantScreen(fixture.rig,fixture.self_screen.shapes,fixture.knife.collider,
        arm=arm) for arm in ('left','right')]
    centre=fixture.body_world(start[:7],start[7:])['link_right_arm_0'][:3,3]
    screens[0].include_static_scene(fixture.stage,fixture.root,fixture.rig.root,centre)
    for s in screens:
        s.static=screens[0].static;s.static_indices=screens[0].static_indices
        s.workspace=screens[0].workspace;s.snapshot(frames)
    complete=WholeRobotTargetScreen(screens[0],fixture.self_screen.shapes)
    evidence=dict(model='neutral_ready_current_scene_path_v1',passed=False,
        no_intended_contact=True,scene_margin_m=TRANSIT_MARGIN_M,self_margin_m=.003,
        tracking_reserve_is_not_continuous_collision_certificate=True,
        interarm_margin_m=.01,maximum_joint_sample_degrees=1.,motion_authorized=False)
    native=None
    try:
        options={}
        if getattr(fixture,'planning_heartbeat',None) is not None:options['heartbeat']=fixture.planning_heartbeat
        native=current_scene_query(fixture.stage,screens[0].static,lazy_coverage=True,
            wall_limit_s=60.,**options)
        for s in screens:s.native_static_query=native
        def valid(q):
            world=fixture.body_world(q[:7],q[7:])
            if not fixture.self_screen.check(world)['passed']:
                evidence['last_rejection']='self';return False
            if fixture.kin.inter_arm_clearance(q[:7],q[7:],fixture.base).clearance_m<.01:
                evidence['last_rejection']='interarm';return False
            target=complete.check(world,margin=TRANSIT_MARGIN_M)
            if not target['passed']:
                evidence['last_rejection']=target['failure'];return False
            for s in screens:
                if not s.check(world,margin=TRANSIT_MARGIN_M):
                    evidence['last_rejection']=s.last_failure;return False
            return True
        limits=[fixture.kin.arm_limits_degrees(a) for a in ('left','right')]
        search={}
        path=connect_path(start,goal,np.r_[limits[0][0],limits[1][0]],
            np.r_[limits[0][1],limits[1][1]],valid,iterations=150,max_checks=1800,diagnostics=search)
        evidence['search']=search
        native.validate()
        if path is None:raise RuntimeError('No complete neutral-ready path within bounded search')
        evidence.update(passed=True,path_points=len(path),joint_degrees=path.tolist(),
            sampled_geometry_only=True,continuous_collision_certified=False)
        return path,evidence
    finally:
        if native is not None:
            try:native.close()
            finally:evidence['native_static']=native.report()
        fixture.neutral_ready['planning']=evidence


def run(app,sim,rig,runtime,springs,fixture,args,output):
    from greenhouse_sim.physics_clock import PhysicsClock
    from .runtime import pose_matrices
    from .finger_coupling import CouplingMonitor
    receipt=fixture.neutral_ready
    clock=PhysicsClock(sim,physics_hz=args.physics_hz,render_hz=0,
        wall_render_hz=15 if args.render_hz else 0)
    begun=float(sim.current_time);last_status=0.;stable=0;last=None
    fixture.bind(sim.physics_sim_view)
    # The neutral approach has NO grasp-contact authorization. A fresh shaft
    # observer and closure controller are bound after this phase, without reset.
    fixture.release_grasp_observer();fixture.cut_authorized=False
    coupling=CouplingMonitor() if args.coupled_fingers_trial else None
    idx=fixture.left_indices+fixture.right_indices
    q=np.degrees(fixture.robot.get_dof_positions()[0,idx])
    try:
        if np.max(abs(q-ready_arms()))>np.degrees(.005):
            raise RuntimeError('Native initial arms do not match SDK ready pose')
        path=None;knots=None;motion_start=None;planning_frames=None
        receipt['native_start_time_s']=begun
        def before(stamp,dt):
            nonlocal path,knots,motion_start,planning_frames
            if not app.is_running() or fixture.stop_requested or not sim.is_playing():
                raise RuntimeError('Neutral approach stopped; no cut continuation')
            if path is None and stamp.simulation_time_s>=1.:
                # Reobserve AFTER the initial native hold. The first second
                # of gravity response must not silently stale the cut corridor.
                frames,_=runtime.sample();planning_frames=frames.copy()
                current=np.degrees(fixture.robot.get_dof_positions()[0,idx])
                path,evidence=plan(fixture,frames,current)
                knots=time_path(path)
                if knots[-1]>45:raise RuntimeError('Neutral approach exceeds bounded 45 second transit')
                motion_start=stamp.simulation_time_s
                receipt.update(planned_motion_seconds=float(knots[-1]),
                    motion_planning=evidence,
                    planning_native_time_s=float(sim.current_time),motion_start_phase_time_s=motion_start,
                    reobserved_after_initial_hold=True)
                print('NEUTRAL_READY_APPROACH_READY '+json.dumps(evidence),flush=True)
            desired=q if path is None else interpolate(path,knots,max(0.,stamp.simulation_time_s-motion_start))
            fixture.event_monitor.begin_step()
            fixture.targets[0,fixture.right_indices]=np.radians(desired[7:])
            fixture.expected_right=fixture.kin.forward('right',desired[7:],fixture.base)
            fixture._command_left_drives(desired[:7],fixture.kin.forward('left',desired[:7],fixture.base))
            if getattr(fixture,'explicit_finger_effort',False):
                apply_open_finger_effort(fixture,step=stamp.step,dt=dt)
            fixture.prepare_step(runtime.frames)
            springs.step(dt,root_constrained=True)
        def after(stamp,dt):
            nonlocal stable,last,last_status
            frames,vel=runtime.sample();palm=pose_matrices(fixture.palm.get_transforms())[0]
            metrics=fixture.check(dt,palm)
            positions=fixture.robot.get_dof_positions()[0];velocities=fixture.robot.get_dof_velocities()[0]
            measured=np.degrees(positions[idx]);target=np.degrees(fixture.targets[0,idx])
            right=pose_matrices(fixture.right_palm.get_transforms())[0]
            right_error=float(np.linalg.norm(right[:3,3]-fixture.expected_right[:3,3]))
            error=float(np.max(abs(positions[idx]-fixture.targets[0,idx])))
            # Preserve the CURRENT rejected sample, not only the preceding
            # accepted one (native438 previously lost the offending pair).
            last=dict(phase='Neutral ready -> pregrasp / knife waiting',t=stamp.simulation_time_s,
                absolute_native_time_s=float(sim.current_time),step=stamp.step,
                measured_arm_degrees=measured.tolist(),commanded_arm_degrees=target.tolist(),
                joint_error_rad=error,right_tracking_error_m=right_error,robot=metrics,
                contact={'bilateral':False},slip_m=None,cut=bool(rig.cut),training_eligible=False,
                native_guards_passed=False,contact_pairs=contact_diagnostic(fixture.event_monitor,dt),
                target_translation_from_plan_m=None if planning_frames is None else
                    float(np.max(np.linalg.norm(frames[:,:3,3]-planning_frames[:,:3,3],axis=1))))
            if (rig.cut or fixture.cut_event is not None or error>.012 or right_error>.012
                    or not np.isfinite(frames).all() or np.max(np.linalg.norm(vel[:,:3],axis=1))>20
                    or np.max(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1))>1e-5
                    or not fixture.check_self(measured[:7],measured[7:])['passed']
                    or fixture.kin.inter_arm_clearance(measured[:7],measured[7:],fixture.base).clearance_m<.01):
                raise RuntimeError('Neutral approach native tracking/plant/self guard failed')
            loads=metrics['per_finger_contact_upper_bound_n']
            if max(loads.values())>.005 or metrics['allowed_tool_contact_n']>.005 or metrics['unwanted_contact_n']>.005:
                raise RuntimeError('Unexpected plant/tool contact during neutral approach')
            if coupling is not None:coupling.observe(positions[fixture.finger_indices],velocities[fixture.finger_indices],step=stamp.step)
            final=(motion_start is not None and stamp.simulation_time_s>=motion_start+knots[-1] and error<=.005
                and np.max(abs(velocities[idx]))<.02)
            stable=stable+1 if final else 0
            last['native_guards_passed']=True
            stream.write(json.dumps(last,allow_nan=False)+'\n')
            if time.monotonic()-last_status>.25:
                fixture.on_sample(last);last_status=time.monotonic()
        def render(_):runtime.sync_visuals()
        with (output/'neutral_ready_trajectory.jsonl').open('w',encoding='utf-8') as stream:
            for _ in range(49*args.physics_hz):
                clock.tick(before=before,after=after,before_render=render)
                if stable>=int(.1*args.physics_hz):break
                if motion_start is not None and clock.stamp.simulation_time_s>motion_start+knots[-1]+3:
                    raise RuntimeError('Neutral approach endpoint did not settle')
        if stable<int(.1*args.physics_hz):raise RuntimeError('Neutral approach endpoint did not settle')
        # Recheck the CURRENT target/scene at the reached endpoint. No archived
        # prelude snapshot can certify the later grasp or cut corridor.
        _,endpoint=plan(fixture,runtime.frames,np.degrees(fixture.robot.get_dof_positions()[0,idx]))
        receipt['endpoint_screen']=endpoint
        receipt.update(execution_passed=True,native_end_time_s=float(sim.current_time),
            native_state_reset=False,last_sample=last)
        fixture.pose=dict(fixture.neutral_goal_pose)  # subsequent DRIVE references only
    except Exception as exc:
        receipt.update(execution_passed=False,error=type(exc).__name__+': '+str(exc),last_sample=last)
    finally:
        receipt['timing']=clock.report()
        (output/'neutral_ready_result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
    return dict(receipt)
