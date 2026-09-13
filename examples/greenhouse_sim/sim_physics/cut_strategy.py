"""Explicit simulator task modes; never turn a missing grasp into a fake one."""
import math
from numbers import Real


MODES = ('bimanual', 'right_only')


def mode(value):
    if value not in MODES:
        raise ValueError('Explicit bimanual or right_only cut strategy required')
    return value


def support_ready(strategy, held, slip, cut_only_ready=False, *, maximum_slip=.003):
    mode(strategy)
    if type(held) is not bool or type(cut_only_ready) is not bool:
        raise ValueError('Explicit measured support state required')
    if strategy == 'right_only':
        return not held and slip is None and cut_only_ready
    return (not cut_only_ready and held and not isinstance(slip, bool)
            and isinstance(slip, Real) and math.isfinite(slip) and 0 <= slip < maximum_slip)


def validate_evidence(evidence, strategy):
    mode(strategy)
    if evidence.get('cut_strategy', 'bimanual') != strategy:
        raise ValueError('Cut evidence cannot cross execution strategies')
    if not support_ready(strategy, evidence.get('stable_left_grasp'),
                         evidence.get('grasp_slip_m'), evidence.get('cut_only_ready', False)):
        raise ValueError('Missing strategy-specific grasp or cut-only readiness evidence')


def choose(*, bimanual, right_only, drop_allowed):
    """Planning disposition, NOT permission to move or bypass a runtime fault.

    A leftward petiole is only a proposal heuristic. Measured planning results
    decide. Unknown/error is not a geometric rejection and cannot trigger cut.
    A cut-only retry needs reset/reobservation and a separate complete plan.
    """
    states = ('clear', 'blocked', 'unknown')
    if bimanual not in states or right_only not in states or type(drop_allowed) is not bool:
        raise ValueError('Explicit plan states and drop policy required')
    if bimanual == 'clear':
        return 'bimanual'
    if bimanual == 'unknown' or right_only == 'unknown':
        return 'inspect_or_retry'
    if right_only == 'clear' and drop_allowed:
        return 'right_only_from_checked_park'
    return 'skip'
