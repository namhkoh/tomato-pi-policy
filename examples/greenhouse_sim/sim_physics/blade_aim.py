"""Bounded contact-placement proposal; NEVER a change to the release seam.

The native loaded edge must still satisfy the original 3 mm axial window,
leading-strip, force, dwell, grasp and collision guards. Engineering diagnostic
only: a command offset is not evidence that a cut occurred.
"""
import numpy as np


def edge_centre(seam, axis, offset_m=0.):
    if (isinstance(offset_m,(bool,np.bool_)) or not isinstance(offset_m,(float,int,np.floating,np.integer))
            or not np.isfinite(offset_m) or abs(offset_m)>.0015):
        raise ValueError('Blade contact aim offset must be finite and within +/-1.5 mm')
    seam,axis=np.asarray(seam,float),np.asarray(axis,float)
    if (seam.shape!=(3,) or axis.shape!=(3,) or not np.isfinite([seam,axis]).all()
            or abs(np.linalg.norm(axis)-1)>1e-5):
        raise ValueError('Measured finite seam and unit shaft axis required')
    return seam+float(offset_m)*axis
