from types import SimpleNamespace
import numpy as np
import pytest

from .implicit_springs import elastic_effort
from .split_springs import SplitJointSprings


class Articulation:
    count = 1

    def __init__(self):
        self.shared_metatype = SimpleNamespace(fixed_base=False,
            dof_names=[f'Joint_{i:03}:{j}' for i in (2, 3) for j in range(3)])
        self.k = np.array([[.4, .4, .3, .2, .2, .15]], np.float32)
        self.c = np.array([[.01, .01, .008, .005, .005, .004]], np.float32)
        self.cap = np.full((1, 6), 10., np.float32)
        self.q = np.array([[.01, -.02, .1, -.02, .03, -.2]])
        self.v = self.q/10
        self.target = np.zeros((1, 6))
        self.velocity_target = np.zeros((1, 6))
        self.force = np.zeros((1, 6), np.float32)
        self.types = np.ones((1, 6))
        self.m = np.diag([.1]*3+[.01]*3+[1e-5]*6)
        self.m[6, 8] = self.m[8, 6] = 2e-6
        self.m[1, 8] = self.m[8, 1] = 1e-5
        self.writes = []

    def get_dof_stiffnesses(self): return self.k
    def get_dof_dampings(self): return self.c
    def get_dof_max_forces(self): return self.cap
    def get_dof_positions(self): return self.q
    def get_dof_velocities(self): return self.v
    def get_dof_position_targets(self): return self.target
    def get_dof_velocity_targets(self): return self.velocity_target
    def get_dof_actuation_forces(self): return self.force
    def get_drive_types(self): return self.types
    def get_generalized_mass_matrices(self): return self.m[None]
    def get_root_velocities(self): return np.zeros((1, 6))
    def get_gravity_compensation_forces(self): return np.array([[0.]*6+[.001]*6])
    def get_coriolis_and_centrifugal_compensation_forces(self): return np.zeros((1, 12))
    def set_dof_stiffnesses(self, values, indices): self.k = values.copy(); self.writes.append('K')
    def set_dof_dampings(self, values, indices): self.c = values.copy(); self.writes.append('C')
    def set_dof_actuation_forces(self, values, indices): self.force = values.copy(); self.writes.append('effort')


@pytest.mark.parametrize('root_constrained', [True, False])
def test_full_original_predictor_but_exactly_one_torsional_spring(root_constrained):
    a = Articulation(); k = a.k[0].astype(float).copy(); c = a.c[0].astype(float).copy()
    q = a.q.copy(); mass = a.m.copy(); caps = a.cap.copy()
    s = SplitJointSprings(a)
    np.testing.assert_array_equal(a.k[0], np.where(s.torsion, k, 0))
    np.testing.assert_array_equal(a.c[0], np.where(s.torsion, c, 0))
    if root_constrained:
        expected = elastic_effort(a.m[6:, 6:], a.q[0], a.v[0], k, c, [-.001]*6, 1/240)
    else:
        expected = elastic_effort(a.m, np.r_[np.zeros(6), a.q[0]], np.r_[np.zeros(6), a.v[0]],
            np.r_[np.zeros(6), k], np.r_[np.zeros(6), c], np.r_[np.zeros(6), [-.001]*6], 1/240)[6:]
    command = s.step(1/240, root_constrained=root_constrained)
    np.testing.assert_allclose(command, np.where(s.torsion, 0, expected))
    np.testing.assert_array_equal(a.force[0, s.torsion], 0)
    assert a.force.shape == (1, 6) and a.writes == ['K', 'C', 'effort']
    np.testing.assert_array_equal(a.m, mass); np.testing.assert_array_equal(a.q, q)
    np.testing.assert_array_equal(a.cap, caps)
    np.testing.assert_array_equal(s.work_stiffness, np.where(s.torsion, 0, k))
    assert not s.receipt['production_qualified']
    assert 'unmeasured' in s.receipt['work_scope']


@pytest.mark.parametrize('bad', ['inventory', 'fixed', 'target', 'velocity_target', 'type', 'effort', 'nan', 'negative_k'])
def test_invalid_initial_contract_does_not_write(bad):
    a = Articulation()
    if bad == 'inventory': a.shared_metatype.dof_names[2] = 'Joint_002:1'
    elif bad == 'fixed': a.shared_metatype.fixed_base = True
    elif bad == 'target': a.target[0, 0] = 1
    elif bad == 'velocity_target': a.velocity_target[0, 0] = 1
    elif bad == 'type': a.types[0, 2] = 2
    elif bad == 'effort': a.force[0, 1] = .1
    elif bad == 'nan': a.k[0, 0] = np.nan
    elif bad == 'negative_k': a.k[0, 0] = -1
    with pytest.raises((ValueError, RuntimeError)): SplitJointSprings(a)
    assert not a.writes


@pytest.mark.parametrize('bad', ['K', 'C', 'cap', 'target', 'type', 'effort', 'inventory'])
def test_contract_drift_fails_before_new_effort(bad):
    a = Articulation(); s = SplitJointSprings(a)
    if bad == 'K': a.k[0, 2] *= 2
    elif bad == 'C': a.c[0, 0] = 1
    elif bad == 'cap': a.cap[0, 0] *= 2
    elif bad == 'target': a.target[0, 2] = 1
    elif bad == 'type': a.types[0, 1] = 2
    elif bad == 'effort': a.force[0, 2] = 1
    elif bad == 'inventory': a.shared_metatype.dof_names[0] = 'Other:0'
    with pytest.raises(RuntimeError): s.step(1/240, root_constrained=True)
    assert a.writes == ['K', 'C']


@pytest.mark.parametrize('dt', [0., 1/120, True, float('nan')])
def test_wrong_timestep_rejected(dt):
    a = Articulation(); s = SplitJointSprings(a)
    with pytest.raises(ValueError): s.step(dt, root_constrained=True)
    assert a.writes == ['K', 'C']


def test_no_external_load_and_no_implicit_configuration_opt_in(tmp_path):
    a = Articulation(); s = SplitJointSprings(a)
    with pytest.raises(ValueError): s.step(1/240, np.zeros(12), root_constrained=True)
    from .benchmark import main
    with pytest.raises(ValueError, match='complete isolated'):
        main(['--output', str(tmp_path/'unused'), '--native-torsion-trial'])
    assert not (tmp_path/'unused').exists()
