"""Read-only native drive contract for isolated hold comparisons.

Unlike ImplicitJointSprings this does NOT disable drives or submit torques.
PhysX owns the spring/contact solve. A returned None is explicitly NOT a
measured drive torque or a zero-work claim. No cutting/release qualification.
"""
import numpy as np


class NativeSpringObserver:
    def __init__(self,articulation):
        if articulation.count!=1:raise ValueError('Exactly one native plant articulation required')
        self.articulation=articulation
        self.names=tuple(articulation.shared_metatype.dof_names)
        self.k=np.asarray(articulation.get_dof_stiffnesses(),float)[0].copy()
        self.c=np.asarray(articulation.get_dof_dampings(),float)[0].copy()
        if (not self.names or len(set(self.names))!=len(self.names)
                or self.k.shape!=(len(self.names),) or self.c.shape!=self.k.shape
                or not np.isfinite(np.r_[self.k,self.c]).all()
                or np.any(self.k<=0) or np.any(self.c<0)):
            raise ValueError('Finite positive native stiffness and nonnegative damping required')
        self.step(1/240,root_constrained=True)

    def step(self,dt,*,root_constrained):
        if (isinstance(dt,(bool,np.bool_)) or not np.isfinite(dt) or abs(dt-1/240)>1e-12
                or root_constrained is not True):
            raise ValueError('Native drive comparison is 240 Hz intact HOLD ONLY')
        a=self.articulation;n=len(self.names)
        if tuple(a.shared_metatype.dof_names)!=self.names:raise RuntimeError('Native joint inventory changed')
        for method,expected in (('get_dof_stiffnesses',self.k),('get_dof_dampings',self.c),
                ('get_dof_position_targets',np.zeros(n)),('get_dof_velocity_targets',np.zeros(n)),
                ('get_dof_actuation_forces',np.zeros(n))):
            value=np.asarray(getattr(a,method)(),float)
            if value.shape!=(1,n) or not np.array_equal(value[0],expected):
                raise RuntimeError('Native drive comparison contract changed: '+method)
        return None
