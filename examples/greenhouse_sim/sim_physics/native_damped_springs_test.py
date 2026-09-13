from types import SimpleNamespace
import numpy as np
import pytest

from .native_damped_springs import NativeDampedSprings


class View:
    def __init__(self):
        self.count=1;self.shared_metatype=SimpleNamespace(fixed_base=False,dof_names=['x','y'])
        self.k=np.zeros((1,2),np.float32);self.c=self.k.copy();self.q=self.k.copy()
        self.targets=self.k.copy();self.vtargets=self.k.copy();self.efforts=self.k.copy()
        self.caps=np.ones((1,2),np.float32);self.types=np.ones((1,2),np.float32)
        self.writes=[]
    def get_dof_stiffnesses(self):return self.k
    def get_dof_dampings(self):return self.c
    def get_dof_max_forces(self):return self.caps
    def get_dof_positions(self):return self.q
    def get_dof_velocity_targets(self):return self.vtargets
    def get_dof_position_targets(self):return self.targets
    def get_drive_types(self):return self.types
    def get_dof_actuation_forces(self):return self.efforts
    def set_dof_dampings(self,value,indices):self.writes.append('damping');self.c=value.copy()
    def set_dof_actuation_forces(self,value,indices):self.writes.append('effort');self.efforts=value.copy()


def source():
    return SimpleNamespace(articulation=View(),fixed_base=False,
        k=np.array([.2,.1],np.float32).astype(float),c=np.array([.01,.01],np.float32).astype(float))


def test_original_damping_only_then_explicit_stiffness_no_double_damping():
    original=source();v=original.articulation;v.q[:]=[.01,-.02]
    solver=NativeDampedSprings(original)
    assert v.writes==['damping']
    np.testing.assert_array_equal(v.c[0],original.c)
    effort=solver.step(1/240,root_constrained=False)
    np.testing.assert_array_equal(effort,(-original.k*v.q[0]).astype(np.float32))
    assert v.writes==['damping','effort']
    assert not solver.receipt['native_physics_qualified']


@pytest.mark.parametrize('mutation',['fixed','small_c','nan_k','bad_cap','nonzero_k','nonzero_c',
    'target','vtarget','wrong_type','large_angle','names'])
def test_invalid_activation_writes_nothing(mutation):
    s=source();v=s.articulation
    if mutation=='fixed':v.shared_metatype.fixed_base=True
    elif mutation=='small_c':s.c[:]=1e-9
    elif mutation=='nan_k':s.k[0]=np.nan
    elif mutation=='bad_cap':v.caps[0,0]=0
    elif mutation=='nonzero_k':v.k[0,0]=1
    elif mutation=='nonzero_c':v.c[0,0]=1
    elif mutation=='target':v.targets[0,0]=1
    elif mutation=='vtarget':v.vtargets[0,0]=1
    elif mutation=='wrong_type':v.types[:]=0
    elif mutation=='large_angle':v.q[:]=.25
    else:v.shared_metatype.dof_names=['x','x']
    with pytest.raises(ValueError):NativeDampedSprings(s)
    assert not v.writes


@pytest.mark.parametrize('mutation',['c','k','q','cap','dt','root'])
def test_changed_contract_cannot_submit(mutation):
    s=source();v=s.articulation;solver=NativeDampedSprings(s)
    if mutation=='c':v.c[:]=0
    elif mutation=='k':v.k[:]=1
    elif mutation=='q':v.q[:]=np.nan
    elif mutation=='cap':v.caps[:]=2
    with pytest.raises((ValueError,RuntimeError)):
        solver.step(1/120 if mutation=='dt' else 1/240,root_constrained=mutation=='root')
    assert v.writes==['damping']
