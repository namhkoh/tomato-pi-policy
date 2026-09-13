"""Reproducible isolated ground-truth diagnostics, NOT a qualified demo preset.

From examples/greenhouse_sim, using Isaac's interpreter:
  -B -m sim_physics.ground_truth_trial --output NEW_DIRECTORY --mode bimanual
  -B -m sim_physics.ground_truth_trial --output NEW_DIRECTORY --mode right_only

Both modes retain original assets/guards and run headless. No hardware, training,
automatic fallback, GUI takeover, arbitrary target selection or OS changes.
"""
import argparse
import json
from pathlib import Path


PROFILE = Path(__file__).with_name('ground_truth_trial.json')


def arguments(output, mode):
    if mode not in ('bimanual','right_only'):
        raise ValueError('Explicit diagnostic mode required')
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
    return ['--output',str(output),*args]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',required=True,choices=('bimanual','right_only'))
    args=p.parse_args(argv)
    from .benchmark import main as run
    return run(arguments(args.output,args.mode))


if __name__=='__main__':
    raise SystemExit(main())
