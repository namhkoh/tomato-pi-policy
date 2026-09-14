"""Bounded command-rate comparison; never changes force/contact thresholds."""
import math


def validate(maximum):
    if isinstance(maximum,bool) or not isinstance(maximum,(int,float)) or not math.isfinite(maximum) or not .0003<=maximum<=.002:
        raise ValueError('Post-release feed must be finite within 0.3..2 mm/s')
    return float(maximum)


def reverse_maximum(maximum, *, support_aware_insertion):
    """Qualify faster insertion without also accelerating the loaded reverse.

    The old explicit rate comparison remains reproducible when the insertion
    trial is disabled. Neither profile is evidence of physical completion.
    """
    maximum=validate(maximum)
    if type(support_aware_insertion) is not bool:
        raise ValueError('Explicit support-aware insertion profile required')
    return .0003 if support_aware_insertion else maximum


def loaded_speed(load,maximum):
    maximum=validate(maximum)
    if isinstance(load,bool) or not isinstance(load,(int,float)) or not math.isfinite(load) or not 0<=load<=.5:
        raise ValueError('Finite full-tool load within original limit required')
    # Retain the existing slow contact speed above 0.2 N. A lighter measured
    # load can use the explicit faster cap; the caller still holds/backs off
    # at 0.4 N and guards the original absolute 0.5 N limit every step.
    return .0003+(maximum-.0003)*min(1.,max(0.,(.20-load)/.10))
