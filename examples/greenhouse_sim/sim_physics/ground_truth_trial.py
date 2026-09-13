"""Reproducible isolated ground-truth actions, NOT a full greenhouse qualification.

From examples/greenhouse_sim, using Isaac's interpreter:
  -B -m sim_physics.ground_truth_trial --output NEW_DIRECTORY --mode bimanual
  -B -m sim_physics.ground_truth_trial --output NEW_DIRECTORY --mode right_only

Add --milestone cut_action for the limited user-requested grasp/cut or unheld
cut result, with post-release torso/base landing recorded separately. The
default full_sequence keeps stricter historical gates. Add --capture for
event-bound native diagnostic PNGs (not synchronized training observations).

Both modes retain original assets/guards and default to headless. --watch opens
a separate visible Run-once panel and keeps the result paused for inspection.
The public launcher uses the actual source crossbar and measured downward
travel law. Its corrected approach is NOT yet qualified end to end. The old
mounting-plate contact recipe is available only with the explicit
--historical-mounting-plate comparison flag; it is not intended-blade evidence.
No hardware, training, automatic fallback, existing-GUI takeover, arbitrary
target selection, reset/replay qualification or OS changes.

--process-zone-trial explicitly selects the experimental actual-crossbar
precontact pose and local yielding contact law. Historical native274/275
passes predate the corrected ALL-knife load accounting and do not qualify
the current controller. Native279/283 stop before release. This is not an
extended-approach, full-through-stroke, tissue or greenhouse qualification.
"""
import argparse
import json
from pathlib import Path


PROFILE = Path(__file__).with_name('ground_truth_trial.json')


def arguments(output, mode, milestone='full_sequence', capture=False, watch=False):
    """Historical recipe vector for reproducible comparisons.

    The public CLI upgrades this vector to the corrected source crossbar.
    Keep the base vector stable for prior recorded diagnostic reproduction.
    """
    if mode not in ('bimanual','right_only'):
        raise ValueError('Explicit diagnostic mode required')
    if milestone not in ('full_sequence','cut_action'):
        raise ValueError('Explicit full-sequence or limited cut-action milestone required')
    if watch and milestone!='cut_action':
        raise ValueError('Watch requires the explicit cut_action milestone')
    data=json.loads(PROFILE.read_text(encoding='utf-8'))
    if data['schema']!='isolated_ground_truth_cut_trial_v1':
        raise ValueError('Unknown diagnostic profile')
    args=list(data['argv'])
    if mode=='right_only':
        remove={'--cut-convergence-trial','--require-retention-screen','--physical-grasp-span',
                '--settle-retention-preload','--effort-bounded-grasp-target','--preload-force-servo'}
        args=[v for v in args if v not in remove]
        args+=['--right-only-cut-trial']
        args[args.index('--approach-distance')+1]='.08'
    if milestone=='cut_action':args+=['--cut-action-trial']
    if capture:
        args.remove('--no-capture-milestones')
        args[args.index('--render-hz')+1]='15'
        args+=['--capture-milestones']
    if watch:
        args+=['--watch-cut-trial']
        args[args.index('--render-hz')+1]='15'
    return ['--output',str(output),*args]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',required=True,choices=('bimanual','right_only'))
    p.add_argument('--milestone',choices=('full_sequence','cut_action'),default='full_sequence')
    p.add_argument('--capture',action='store_true',help='Paused native milestone PNGs; headless, not synchronized training RGB-D')
    p.add_argument('--watch',action='store_true',help='Open a visible Run-once panel; no automatic run, reset or hardware commands')
    p.add_argument('--historical-mounting-plate',action='store_true',
        help='Reproduce the superseded mounting-plate contact test, NOT the physical knife edge')
    p.add_argument('--process-zone-trial',action='store_true',
        help='Explicit experimental corrected-edge/yielding contact recipe; isolated only, not frozen')
    p.add_argument('--through-stroke-trial',action='store_true',
        help='Experimental measured follow-through and withdrawal; 85 s, no solid-face bypass')
    p.add_argument('--stream-trajectory',action='store_true',help='Lossless full-rate compressed diagnostic evidence')
    p.add_argument('--joint-transit-fallback',action='store_true',help='Screened whole-arm joint search for approach only')
    p.add_argument('--postrelease-feed-m-s',type=float,default=.0003,help='Explicit post-release contact feed comparison')
    p.add_argument('--postcut-egress-trial',action='store_true',help='Fresh fully screened post-cut withdrawal goal')
    p.add_argument('--material-clearance-trial',action='store_true',
        help='Measure sharp-edge clearance of the shaft section; bounded cut-face sliding only after verified release')
    p.add_argument('--rectilinear-floor-contacts',action='store_true',help='Exact source-floor collision solid as native boxes; no visual changes')
    p.add_argument('--greenhouse-trial',action='store_true',help='Restore intact original scene and three-gutter preview planting; fresh native qualification required')
    p.add_argument('--screen-ready-pose',action='store_true',help='Zero-motion native elbow search only; saves proposals and stops before physics')
    p.add_argument('--screen-approach-start',action='store_true',help='Zero-motion higher/lateral waiting-pose search only')
    p.add_argument('--screen-tool-heading',action='store_true',help='Zero-motion arc-up complete-tool heading search only')
    p.add_argument('--screen-station',action='store_true',help='Zero-motion base/two-arm proposal search; no base-motion or path authority')
    p.add_argument('--station-proposal-report',type=Path,help='Explicit new initial station in greenhouse or isolation; all native startup/path checks run again')
    p.add_argument('--coupled-fingers-trial',action='store_true',help='Experimental physical jaw coupling; bimanual only, unchanged force/slip limits')
    p.add_argument('--right-ready-lift-m',type=float,default=0.,help='Initial world-up waiting-pose lift; fresh IK and original native guards required')
    p.add_argument('--right-ready-retreat-m',type=float,default=0.,help='Initial waiting pose withdrawn along wrist +Z; fresh IK and all native guards required')
    p.add_argument('--profile',action='store_true',help='Diagnostic cProfile and native-step timing; overhead means this is not a latency comparison')
    p.add_argument('--cut-priority-report',type=Path,help='Try a prior native cut-frame family first; no pose/path replay or inherited clearance')
    p.add_argument('--blade-aim-offset-m',type=float,default=None,
        help='Explicit bounded contact-placement comparison, +/-1.5 mm; never changes the seam or release gates')
    p.add_argument('--grasp-arc-m',type=float,default=None,
        help='Explicit bimanual grasp proposal 60..120 mm from attachment; all native clearance and retention checks remain')
    p.add_argument('--physics-threads',type=int,choices=(1,2,4,8),default=None,
        help='Controlled scheduler comparison; unchanged physics rate, solver, iterations and geometry')
    args=p.parse_args(argv)
    if args.process_zone_trial and args.historical_mounting_plate:
        p.error('Process-zone trial cannot use the historical mounting plate')
    if args.through_stroke_trial and not (args.process_zone_trial and args.milestone=='cut_action'):
        p.error('Through-stroke requires explicit process-zone cut_action trial')
    if args.material_clearance_trial and not args.through_stroke_trial:
        p.error('Material-clearance trial requires --through-stroke-trial')
    if args.greenhouse_trial and not (args.process_zone_trial and args.milestone=='cut_action'):
        p.error('Greenhouse trial requires explicit process-zone cut_action trial')
    if (args.screen_ready_pose or args.screen_approach_start or args.screen_tool_heading or args.screen_station) and not (args.process_zone_trial and not args.watch):
        p.error('Startup screening requires process-zone trial without watched execution')
    if sum((args.screen_ready_pose,args.screen_approach_start,args.screen_tool_heading,args.screen_station))>1:p.error('Select one startup search')
    if args.blade_aim_offset_m is not None:
        import math
        if not (args.process_zone_trial and math.isfinite(args.blade_aim_offset_m)
                and abs(args.blade_aim_offset_m)<=.0015):
            p.error('Contact-placement comparison requires process-zone trial and finite +/-1.5 mm offset')
    from .benchmark import main as run
    if args.grasp_arc_m is not None:
        import math
        if not (args.process_zone_trial and args.mode=='bimanual' and math.isfinite(args.grasp_arc_m)
                and .06<=args.grasp_arc_m<=.12):
            p.error('Grasp comparison requires bimanual process-zone trial and finite 60..120 mm arc')
    options=arguments(args.output,args.mode,args.milestone,args.capture,args.watch)
    if not args.historical_mounting_plate:
        from .blade_contacts import CROSSBAR_EDGE
        from .knife import DOWNWARD_CUT_MODEL
        options+=['--knife-edge-mode',CROSSBAR_EDGE,'--cut-model',DOWNWARD_CUT_MODEL,'--source-wrist-contacts']
    if args.process_zone_trial:
        recipe=json.loads(PROFILE.with_name('crossbar_process_zone_trial.json').read_text(encoding='utf-8'))
        if recipe.get('schema')!='experimental_crossbar_process_zone_trial_v1':
            raise ValueError('Unknown explicit process-zone recipe')
        options+=['--seam-contact-yield','--right-ready-degrees',*map(str,recipe['right_ready_degrees'])]
    if args.through_stroke_trial:
        options+=['--through-stroke-trial','--seconds','85']
    if args.material_clearance_trial:options+=['--material-clearance-trial']
    if args.postcut_egress_trial:options+=['--postcut-egress-trial']
    if args.joint_transit_fallback:options+=['--joint-transit-fallback']
    if args.postrelease_feed_m_s!=.0003:options+=['--postrelease-feed-m-s',str(args.postrelease_feed_m_s)]
    if args.rectilinear_floor_contacts:options+=['--rectilinear-floor-contacts']
    if args.stream_trajectory or args.through_stroke_trial:options+=['--stream-trajectory']
    if args.greenhouse_trial:
        options=[v for v in options if v not in ('--isolate-station','--branch-contact-fixture')]
        options+=['--greenhouse-cut-trial','--local-wire-physics','--context-gutters','3',
                  '--batch-gutter-visuals','--stream-trajectory']
    if args.screen_ready_pose:options+=['--native-startup-pose-search']
    if args.screen_approach_start:options+=['--native-startup-approach-search']
    if args.screen_tool_heading:options+=['--native-startup-heading-search']
    if args.screen_station:options+=['--native-startup-station-search']
    if args.station_proposal_report:options+=['--station-proposal-report',str(args.station_proposal_report)]
    if args.coupled_fingers_trial:options+=['--coupled-fingers-trial']
    if args.right_ready_lift_m:options+=['--right-ready-lift-m',str(args.right_ready_lift_m)]
    if args.right_ready_retreat_m:options+=['--right-ready-retreat-m',str(args.right_ready_retreat_m)]
    if args.profile:options+=['--profile','--step-profile']
    if args.cut_priority_report:options+=['--cut-priority-report',str(args.cut_priority_report)]
    if args.blade_aim_offset_m is not None:
        options+=['--blade-axial-aim-offset-m',str(args.blade_aim_offset_m)]
    if args.grasp_arc_m is not None:options+=['--grasp-arc-m',str(args.grasp_arc_m)]
    if args.physics_threads is not None:options+=['--physics-threads',str(args.physics_threads)]
    return run(options)


if __name__=='__main__':
    raise SystemExit(main())
