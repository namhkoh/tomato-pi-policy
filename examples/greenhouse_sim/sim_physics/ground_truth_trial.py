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
    args=p.parse_args(argv)
    from .benchmark import main as run
    options=arguments(args.output,args.mode,args.milestone,args.capture,args.watch)
    if not args.historical_mounting_plate:
        from .blade_contacts import CROSSBAR_EDGE
        from .knife import DOWNWARD_CUT_MODEL
        options+=['--knife-edge-mode',CROSSBAR_EDGE,'--cut-model',DOWNWARD_CUT_MODEL,'--source-wrist-contacts']
    return run(options)


if __name__=='__main__':
    raise SystemExit(main())
