"""Visible, bounded original-petiole demo. Not robot control or training capture."""
import asyncio
import json
import time

import numpy as np


def pulse_force(simulation_time, magnitude=.02):
    if not np.isfinite(magnitude) or not 0 <= magnitude <= .2:
        raise ValueError('Demo force must be between 0 and 0.2 N')
    return np.array([0., magnitude if 1. <= simulation_time < 2. else 0., 0.])


def run(app, sim, rig, runtime, springs, args, output):
    import omni.ui as ui
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, Usd, UsdGeom, UsdLux
    from greenhouse_sim.physics_clock import PhysicsClock
    from .implicit_springs import ImplicitJointSprings, NativeBodyLoads
    from .runtime import PlantRuntime

    stage = rig.stage
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('Interactive demo needs a visible viewport')
    viewport.set_texture_resolution((1280, 720))
    dome = UsdLux.DomeLight.Define(stage, '/World/DemoLight')
    dome.CreateIntensityAttr(1400.)
    sun = UsdLux.DistantLight.Define(stage, '/World/DemoKey')
    sun.CreateIntensityAttr(2200.)
    sun.CreateAngleAttr(10.)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, 25, -30))
    camera = UsdGeom.Camera.Define(stage, '/World/PhysicsDemoCamera')
    camera.CreateFocalLengthAttr(25.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100.))
    camera_op = UsdGeom.Xformable(camera).AddTransformOp()
    viewport.set_active_camera(str(camera.GetPath()))
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    views = {}
    for name, path in (('Target close-up', rig.root+'/Branch'), ('Whole plant', '/World/Plant')):
        box = bounds.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        low, high = np.array(box.GetMin()), np.array(box.GetMax())
        centre = (low+high)/2
        distance = max(.55, float(np.linalg.norm(high-low))*1.5)
        direction = np.array([.65, 1., .3]); direction /= np.linalg.norm(direction)
        eye = centre+distance*direction
        views[name] = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*centre), Gf.Vec3d(0,0,1)).GetInverse()

    def view(name):
        camera_op.Set(views[name])
        viewport.set_active_camera(str(camera.GetPath()))

    view('Target close-up')
    clock = PhysicsClock(sim, physics_hz=240, render_hz=args.render_hz)
    loads = NativeBodyLoads(runtime.articulation, [0,0,-9.81])
    state = dict(command=None, running=False, paused=False, fault=None, released_at=None,
                 cycles=0, tip_mm=0., contact=0., speed=0.)
    auto = ui.SimpleBoolModel(True)
    evidence = dict(training_eligible=False, physical_cut_verified=False, robot_grasp_verified=False,
                    view_kind='diagnostic_closeup_not_robot_camera', images={}, cycles=[])
    samples = []

    def command(name):
        # UI callbacks never mutate a live articulation; handle at a tick boundary.
        state['command'] = name

    window = ui.Window('Plant Physics - experimental demo', width=370, height=475)
    with window.frame:
        with ui.VStack(spacing=7):
            ui.Label('ORIGINAL PETIOLE | native PhysX', height=24)
            ui.Label('Only this petiole and its leaves are dynamic. The rest of the plant is fixed. Diagnostic view, NOT a robot camera.', word_wrap=True, height=54)
            status = ui.Label('Loading materials...', word_wrap=True, height=60)
            metrics = ui.Label('', word_wrap=True, height=42)
            ui.Button('Run pull + recover (20 mN, 6 seconds)', clicked_fn=lambda:command('pull'), height=28)
            ui.Button('Pause / resume', clicked_fn=lambda:command('pause'), height=28)
            ui.Button('Release seam (diagnostic, NOT blade cutting)', clicked_fn=lambda:command('release'), height=28)
            ui.Button('Reset / reattach', clicked_fn=lambda:command('reset'), height=28)
            with ui.HStack(height=24):
                ui.CheckBox(auto, width=24)
                ui.Label('Repeat pull/recover, with an explicit reset')
            with ui.HStack(height=28):
                for name in views:
                    ui.Button(name, clicked_fn=lambda n=name:view(n))
            ui.Label('No grasp/contact qualification. Release is joint separation, not tissue cutting. No floor in this isolated test. No dataset collection.', word_wrap=True, height=62)
    other = ui.Workspace.get_window('Property')
    if other:
        window.dock_in(other, ui.DockPosition.SAME)

    def capture(name):
        # Native viewport evidence only; freeze the physical state while the
        # renderer settles. Never synthesize RGB or depth, or advance via app.update.
        runtime.sync_visuals()
        before_frames = runtime.frames.copy()
        for _ in range(12):
            if not app.is_running(): return
            sim.render()
        task = asyncio.ensure_future(capture_viewport_to_file(viewport, str(output/(name+'.png'))).wait_for_result())
        deadline = time.monotonic()+30
        while not task.done():
            if not app.is_running() or time.monotonic()>deadline:
                task.cancel()
                raise RuntimeError('Native demo capture timed out or window closed')
            sim.render()
        task.result()
        frames, _ = runtime.sample()
        error = float(np.max(np.abs(frames-before_frames)))
        if error > 1e-7: raise RuntimeError('Rendering advanced the supposedly frozen plant')
        evidence['images'][name] = dict(file=name+'.png', simulation_time=clock.stamp.simulation_time_s,
            tip=runtime.tip().tolist(), render_only_body_error=error, training_eligible=False)

    def save():
        (output/'demo_status.json').write_text(json.dumps({**evidence, 'state':state,
            'timing':clock.report()}, indent=2, allow_nan=False), encoding='utf-8')

    def reset():
        nonlocal runtime, springs, loads
        sim.stop(); rig.restore_authored_state(); sim.reset(); sim.step(render=False)
        runtime = PlantRuntime(rig, sim.physics_sim_view)
        springs = ImplicitJointSprings(runtime.articulation)
        loads = NativeBodyLoads(runtime.articulation, [0,0,-9.81])
        clock.reset_epoch(); samples.clear()
        state.update(running=False, paused=False, fault=None, released_at=None)
        runtime.sync_visuals()

    def before(stamp, dt):
        force = pulse_force(stamp.simulation_time_s, args.force_newton) if not rig.cut else np.zeros(3)
        runtime.apply_tip_force(force)
        springs.step(dt, loads.at_com(rig.body_paths[-1], force), root_constrained=not rig.cut)

    def after(stamp, dt):
        frames, velocity = runtime.sample()
        state['tip_mm'] = float(np.linalg.norm(runtime.tip()-rig.chain_world[-1])*1000)
        state['speed'] = float(np.linalg.norm(velocity[:,:3],axis=1).max())
        state['contact'] = float(np.linalg.norm(runtime.contacts.get_net_contact_forces(dt),axis=1).max())
        support = float(np.max(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1)))
        if state['speed']>20 or state['contact']>1e-5 or support>1e-5 or (not rig.cut and state['tip_mm']>80):
            raise RuntimeError('Motion/contact exceeded isolated-demo qualification. Paused; reset required.')
        if state['cycles']==0:
            samples.append(dict(t=stamp.simulation_time_s, tip=runtime.tip().tolist(), **{k:state[k] for k in ('speed','contact')}))

    runtime.sync_visuals()
    for _ in range(45):
        if not app.is_running(): return evidence
        sim.render()
    capture('initial')
    save()
    print('PHYSICS_DEMO_READY '+str(output), flush=True)
    start_at = time.monotonic()+3.
    last_ui = 0.
    while app.is_running():
        wall_start = time.monotonic()
        requested = state['command']; state['command'] = None
        if requested in ('reset', 'pull'):
            reset()
            if requested=='reset': auto.set_value(False)
            state['running'] = requested=='pull'
            start_at = wall_start+2.
        elif requested=='pause':
            state['paused'] = not state['paused']; auto.set_value(False)
        elif requested=='release' and not state['fault'] and not rig.cut:
            auto.set_value(False)
            rig.diagnostic_release()
            state.update(running=True, paused=False, released_at=clock.stamp.simulation_time_s)
        if not state['running'] and not state['fault'] and auto.as_bool and wall_start>=start_at:
            reset(); state['running'] = True
        if state['running'] and not state['paused'] and not state['fault']:
            try:
                clock.tick(before=before, after=after, before_render=lambda _:runtime.sync_visuals())
                t = clock.stamp.simulation_time_s
                if state['cycles']==0 and not rig.cut:
                    for name, at in (('settled', .9), ('pulled', 1.9), ('recovered', 2.9)):
                        if t>=at and name not in evidence['images']:
                            capture(name)
                finished = (t>=6. if not rig.cut else t-state['released_at']>=.3)
                if finished:
                    runtime.sync_visuals()
                    state['running']=False; state['cycles']+=1; start_at=time.monotonic()+2.
                    evidence['cycles'].append(dict(episode=clock.episode, released=rig.cut, final_tip=runtime.tip().tolist(), timing=clock.report()))
                    evidence['cycles']=evidence['cycles'][-10:]
                    if samples:
                        (output/'demo_first_cycle.json').write_text(json.dumps(samples,allow_nan=False),encoding='utf-8')
                        samples.clear()
                    save()
                    print('PHYSICS_DEMO_CYCLE '+str(state['cycles']), flush=True)
            except Exception as exc:
                state.update(fault=str(exc), running=False)
                auto.set_value(False); save()
                print('PHYSICS_DEMO_PAUSED '+str(exc), flush=True)
        else:
            # SimulationContext.render disables automatic physics during Kit's
            # UI update. Pausing this demo cannot bypass elastic control.
            sim.render()
        if wall_start-last_ui>.1:
            t=clock.stamp.simulation_time_s
            phase = ('FAULT: '+state['fault'] if state['fault'] else 'Detached (diagnostic release)' if rig.cut else
                     'Paused' if state['paused'] else 'PULL: 20 mN along world +Y' if state['running'] and 1<=t<2 else
                     'Gravity / recover' if state['running'] else 'Ready / waiting for replay')
            status.text = phase+f'\nPhysics time {t:.2f} s | cycle {state["cycles"]+1}'
            metrics.text = f'Tip displacement from authored rest: {state["tip_mm"]:.1f} mm\nMax native contact: {state["contact"]:.6f} N'
            last_ui=wall_start
        # No unbounded catch-up. Cap to real-time physics; slow renders reduce
        # playback speed but never skip control or increase the physics step.
        period=1/240 if state['running'] and not state['paused'] and not state['fault'] else 1/30
        time.sleep(max(0., period-(time.monotonic()-wall_start)))
    save()
    return evidence
