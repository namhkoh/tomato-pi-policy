"""One native trial in a visible window, without unqualified reset/replay.

This observer changes no material, collision, gain, force or time-step settings.
It uses sim.render (no physics advance) while waiting and pauses after the run.
"""
import json
import time
import numpy as np


def validate(args):
    if not getattr(args,'watch_cut_trial',False):return
    if not (args.cut_action_trial and args.fixed_root_cut_trial and args.bimanual_cut
            and args.full_robot_probe and ((args.isolate_station and args.branch_contact_fixture)
                or getattr(args,'greenhouse_cut_trial',False))
            and args.physics_hz==480 and args.render_hz==15
            and not args.gui and not args.robot_interactive and not args.interactive
            and not args.profile and not args.scene_profile):
        raise ValueError('Watch requires the isolated 480 Hz cut-action profile, 15 Hz rendering and no reset/profiling mode')


class OneShot:
    def __init__(self):self.state='ready'
    def request(self):
        if self.state=='ready':self.state='requested'
    def take(self):
        if self.state!='requested':return False
        self.state='running';return True
    def finish(self):
        if self.state!='running':raise RuntimeError('Only a started trial can finish')
        self.state='finished'


def check_idle(playing,plant_before,plant_now,q_before,q_now):
    if (not playing or not np.allclose(plant_before,plant_now,atol=1e-8,rtol=0)
            or not np.allclose(q_before,q_now,atol=1e-8,rtol=0)):
        raise RuntimeError('Scene/timeline changed before Run; close and relaunch the watched trial')


def idle_robot_view(sim,fixture):
    # The trial deliberately binds its controls only when Run is pressed.
    # Observe the native articulation here without an early controller bind.
    view=sim.physics_sim_view.create_articulation_view(fixture.anchor)
    if view.count!=1 or not view.shared_metatype.fixed_base:
        raise RuntimeError('One fixed-base native robot required for idle observation')
    return view


def watch(app,sim,rig,runtime,springs,fixture,args,output,run):
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport
    validate(args)
    mode='right_only' if args.right_only_cut_trial else 'bimanual'
    control=OneShot();result=None;last_update=0.
    fixture.setup_views(get_active_viewport());fixture.select_view('Grasp close-up')
    runtime.sample();runtime.sync_visuals()
    idle_robot=idle_robot_view(sim,fixture)
    initial_frames=runtime.frames.copy();initial_q=idle_robot.get_dof_positions().copy()
    def publish(value):
        (output/'live_status.json').write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    window=ui.Window('Cut action watch - '+mode,width=430,height=650)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label('FULL RB-Y1 A v1.2 - '+('INTACT GREENHOUSE' if getattr(args,'greenhouse_cut_trial',False) else 'ISOLATED ORIGINAL BRANCH'),word_wrap=True,height=38)
            ui.Label('Live native physics, not recorded playback. Diagnostic only: no tissue calibration or greenhouse qualification.',word_wrap=True,height=48)
            status=ui.Label('Ready. Choose a view, then Run once. Use this panel, not the timeline controls.',word_wrap=True,height=70)
            start=ui.Button('Run once: right-only cut' if args.right_only_cut_trial else 'Run once: left grasp + right cut',height=32,clicked_fn=control.request)
            ui.Button('Stop trial (relaunch required)',height=26,clicked_fn=lambda:setattr(fixture,'stop_requested',True))
            for name in fixture.views:
                ui.Button(name,height=23,clicked_fn=lambda n=name:fixture.select_view(n))
            ui.Label('Experimental isolated fixture. Full through-stroke requires its separate measured trial. Falling material may contact torso; reset/replay disabled.',word_wrap=True,height=58)
            ui.Label(f'{args.seconds:g} simulated seconds may take several minutes. Close Isaac to end. Do not transform the robot or plant.',word_wrap=True,height=44)
    def on_sample(record):
        nonlocal last_update
        now=time.monotonic()
        if now-last_update<.25:return
        last_update=now;slip=record['slip_m']
        status.text=(f"{record['phase']} | {record['t']:.2f} / {args.seconds:g} s\n"
            f"Bilateral contact: {record['contact']['bilateral']} | Cut: {record['cut']}\n"
            f"Slip: {'n/a' if slip is None else format(slip*1000,'.2f')+' mm'}")
        publish(dict(state='running',strategy=mode,t=record['t'],cut=record['cut'],
            bilateral=record['contact']['bilateral'],slip_m=slip,training_eligible=False))
    fixture.on_sample=on_sample
    property_window=ui.Workspace.get_window('Property')
    if property_window:window.dock_in(property_window,ui.DockPosition.SAME)
    publish(dict(state='ready',strategy=mode,one_shot=True,training_eligible=False))
    print('CUT_WATCH_READY '+str(output),flush=True)
    while app.is_running():
        if control.state in ('ready','requested'):
            runtime.sample()
            check_idle(sim.is_playing(),initial_frames,runtime.frames,
                       initial_q,idle_robot.get_dof_positions())
        if control.take():
            start.enabled=False
            result=run(app,sim,rig,runtime,springs,fixture,args,output)
            control.finish()
            (output/'watch_result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            publish(result)
            passed=result.get('cut_action',{}).get('passed') is True
            status.text=(('Cut action PASSED' if passed else 'Cut action FAILED')+
                '\nFull withdrawal: '+str(result.get('gates',{}).get('right_withdrawal_completed'))+
                '\n'+str(result.get('error') or 'Paused for inspection. Close and relaunch for another trial.'))
            print('CUT_WATCH_FINISHED '+str(result.get('state')),flush=True)
        sim.render();time.sleep(.01)
    return result or dict(state='watch_closed_without_trial',training_eligible=False)
