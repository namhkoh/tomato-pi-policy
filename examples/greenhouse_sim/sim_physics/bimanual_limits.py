"""Unchanged native execution bounds with explicit, fail-closed diagnostics.

These engineering limits are not calibrated plant-damage thresholds. A pass
does not replace collision, grasp, withdrawal or blade-evidence validation.
"""
import math
from numbers import Real


def violations(record):
    """Identify each failed bound; never modify observations or clamp state."""
    measurements=(
        ('plant_speed',record['max_speed_m_s'],20.,'maximum','m/s'),
        ('gripper_net_contact',record['max_gripper_net_contact_n'],3.,'maximum','N'),
        ('support_displacement',record['support_error_m'],1e-5,'maximum','m'),
        ('finger_separation',record['contact']['min_separation'],-.001,'minimum','m'),
        ('tool_contact',record['robot']['allowed_tool_contact_n'],.5,'maximum','N'),
        ('tool_separation',record['robot']['minimum_tool_separation_m'],-.001,'minimum','m'),
    )
    if record['slip_m'] is not None:
        measurements+= (('grasp_slip',record['slip_m'],.003,'maximum','m'),)
    failed=[]
    for name,value,limit,direction,unit in measurements:
        finite=isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)
        valid=finite and (direction=='minimum' or value>=0)
        if not valid or (value>limit if direction=='maximum' else value<limit):
            failed.append(dict(metric=name,measured=float(value) if finite else None,
                               limit=limit,direction=direction,unit=unit,
                               reason='limit_exceeded' if valid else 'invalid_measurement'))
    return failed
