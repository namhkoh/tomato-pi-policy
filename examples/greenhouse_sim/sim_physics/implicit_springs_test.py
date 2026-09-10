import numpy as np
import pytest
from types import SimpleNamespace

from sim_physics.implicit_springs import elastic_effort, ImplicitJointSprings, NativeBodyLoads


def test_exact_linear_backward_euler_and_energy_decay():
    m=np.array([[2.,.7],[.7,1.]])*1e-8
    k=np.array([.3,.1]);c=np.array([.001,.002])
    q=np.array([.1,-.2]);v=np.array([3.,-1.]);dt=1/120
    energy=lambda q,v: .5*(v@m@v+np.dot(k*q,q))
    previous=energy(q,v)
    for _ in range(50):
        effort=elastic_effort(m,q,v,k,c,np.zeros(2),dt)
        v=v+dt*np.linalg.solve(m,effort)
        q=q+dt*v
        current=energy(q,v)
        assert current<=previous+1e-12
        previous=current
    assert previous<1e-8


def test_constant_load_has_correct_stiffness_without_mass_inflation():
    m=np.array([[1e-8]]);k=np.array([.1]);c=np.array([.001]);f=np.array([.0004905])
    q=np.zeros(1);v=np.zeros(1);dt=1/240
    for _ in range(720):
        effort=elastic_effort(m,q,v,k,c,f,dt)
        v+=dt*np.linalg.solve(m,effort+f)
        q+=dt*v
    np.testing.assert_allclose(q,f/k,atol=1e-10)
    np.testing.assert_allclose(v,0,atol=1e-10)


@pytest.mark.parametrize('dt',[0,-1,float('nan')])
def test_bad_dt_rejected(dt):
    with pytest.raises(ValueError):
        elastic_effort(np.eye(1),[0],[0],[1],[1],[0],dt)


def test_bad_mass_not_silently_regularized():
    with pytest.raises(np.linalg.LinAlgError):
        elastic_effort(-np.eye(1),[0],[0],[1],[1],[0],.01)


def test_tiny_off_diagonal_roundoff_uses_matrix_scale():
    m=np.diag([1e-4,1e-8]);m[0,1]=2e-12
    assert np.isfinite(elastic_effort(m,[0,0],[0,0],[1,1],[1,1],[0,0],.01)).all()
    m[0,1]=1e-6
    with pytest.raises(ValueError,match='symmetric'):
        elastic_effort(m,[0,0],[0,0],[1,1],[1,1],[0,0],.01)


class SpringArticulation:
    count=1
    shared_metatype=SimpleNamespace(fixed_base=False)
    def __init__(self):
        self.m=np.diag([.1,.1,.1,.001,.001,.001,1e-5])
        self.m[0,6]=self.m[6,0]=5e-5
    def get_dof_stiffnesses(self): return np.array([[.1]])
    def get_dof_dampings(self): return np.array([[.001]])
    def get_generalized_mass_matrices(self): return self.m[None,:]
    def get_dof_positions(self): return np.array([[.02]])
    def get_dof_velocities(self): return np.array([[0.]])
    def get_root_velocities(self): return np.zeros((1,6))
    def get_gravity_compensation_forces(self): return np.zeros((1,7))
    def get_coriolis_and_centrifugal_compensation_forces(self): return np.zeros((1,7))
    def set_dof_stiffnesses(self, values, indices): assert not np.any(values)
    def set_dof_dampings(self, values, indices): assert not np.any(values)
    def set_dof_actuation_forces(self, values, indices): self.applied=values.copy()


@pytest.mark.parametrize('constrained',[False,True])
def test_free_root_is_coupled_not_actuated_and_attached_root_is_conditional(constrained):
    a=SpringArticulation();springs=ImplicitJointSprings(a)
    effort=springs.step(.004,root_constrained=constrained)
    if constrained:
        expected=elastic_effort(a.m[6:,6:],[.02],[0],[.1],[.001],[0],.004)
    else:
        expected=elastic_effort(a.m,np.r_[np.zeros(6),.02],np.zeros(7),
            np.r_[np.zeros(6),.1],np.r_[np.zeros(6),.001],np.zeros(7),.004)
        np.testing.assert_array_equal(expected[:6],0)
        expected=expected[6:]
    np.testing.assert_allclose(effort,expected)
    assert a.applied.shape==(1,1)  # no root forces or state setters


class LoadArticulation:
    shared_metatype=SimpleNamespace(fixed_base=True)
    link_paths=[['/Root','/Tip']]
    def __init__(self,reference):
        self.reference=reference
    def get_coms(self):
        return np.array([[[0,0,0,0,0,0,1],[.02,0,0,0,0,0,1]]])
    def get_masses(self): return np.array([[1.,.001]])
    def get_link_transforms(self):
        return np.array([[[0,0,0,0,0,0,1],[.05,0,0,0,0,0,1]]])
    def get_jacobians(self):
        j=np.zeros((1,1,6,1));j[0,0,4,0]=1
        j[0,0,2,0]=-.07 if self.reference=='center_of_mass' else -.05
        return j
    def get_gravity_compensation_forces(self): return np.array([[-.001*9.81*.07]])


@pytest.mark.parametrize('reference',['center_of_mass','prim_origin'])
def test_com_load_reference_is_verified_against_native_gravity(reference):
    loads=NativeBodyLoads(LoadArticulation(reference),[0,0,-9.81])
    assert loads.reference==reference
    np.testing.assert_allclose(loads.at_com('/Tip',[0,0,-1]),[.07])
    with pytest.raises(ValueError,match='fixed'):
        loads.at_com('/Root',[0,0,-1])


def test_inconsistent_load_mapping_fails_closed():
    with pytest.raises(RuntimeError,match='disagrees'):
        NativeBodyLoads(LoadArticulation('prim_origin'),[0,0,-1])
