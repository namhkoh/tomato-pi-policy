"""A commanded pre-grasp opening, not a change to gripper geometry or limits."""
import numpy as np


DEFAULT_HALF_APERTURE_M = .025


def validate(radius, opening=DEFAULT_HALF_APERTURE_M):
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in (radius, opening))
            or not np.isfinite([radius, opening]).all()
            or not 0 < radius < .02 or not radius+.002 <= opening <= DEFAULT_HALF_APERTURE_M):
        raise ValueError('Pre-grasp half aperture must leave at least 2 mm shaft clearance and be <=25 mm')
    return float(opening)


def closure_samples(opening, minimum):
    """Screen the complete commanded closure, at intervals no larger than 1 mm."""
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in (opening, minimum))
            or not np.isfinite([opening, minimum]).all() or not 0 <= minimum < opening <= DEFAULT_HALF_APERTURE_M):
        raise ValueError('Invalid commanded closure interval')
    return np.linspace(opening, minimum, max(2, int(np.ceil((opening-minimum)/.001))+1))
