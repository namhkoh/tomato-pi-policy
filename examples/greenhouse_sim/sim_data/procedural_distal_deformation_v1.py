"""Signed transverse deformation sharing the frozen proximal identity machinery.

Amplitude is an explicit finite recipe input. The unchanged determinant and
condition-number guards determine admissible strength for each mesh face guard;
there is no special25mm value or automatic amplitude adjustment.
"""
import math
import numpy as np
from .procedural_petiole_geometry import require,unit
from .procedural_proximal_shear_v1 import ProximalShear,PROTECTED_M,DISTAL_M


class SignedProximalShear(ProximalShear):
    def __init__(self,anchor,tangent,direction,amplitude,guard):
        self.anchor=np.asarray(anchor,float);self.tangent=unit(tangent);self.direction_vector=unit(direction)
        self.amplitude=float(amplitude);self.guard=float(guard)
        require(self.anchor.shape==(3,) and np.isfinite(self.anchor).all(),'Finite attachment required')
        require(math.isfinite(self.amplitude) and self.amplitude!=0.,'Finite nonzero signed amplitude required')
        require(abs(float(self.tangent@self.direction_vector))<1e-12,'Shear must be transverse')
        require(PROTECTED_M<self.guard<DISTAL_M,'Authored proximal faces must leave a supported blend')
        maximum_gradient=abs(self.amplitude)*1.5/(DISTAL_M-self.guard)
        worst=np.eye(3)+maximum_gradient*np.outer(self.direction_vector,self.tangent)
        require(np.isfinite(worst).all() and np.linalg.det(worst)>.05 and np.linalg.cond(worst)<30,
            'Explicit amplitude violates unchanged Jacobian guard; no adjustment')


def rotated_source_direction(parent_axis,tangent,angle_degrees):
    require(type(angle_degrees) in (int,float) and math.isfinite(angle_degrees)
        and -180.<=float(angle_degrees)<=180.,'Finite canonical source-frame angle required')
    tangent=unit(tangent);first=unit(np.cross(unit(parent_axis),tangent));second=unit(np.cross(tangent,first))
    angle=math.radians(float(angle_degrees))
    return unit(math.cos(angle)*first+math.sin(angle)*second)
