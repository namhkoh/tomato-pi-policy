"""Analytic CPU tests only; no native jobs, USD authoring or material tuning."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from .seam_loads import MAX_BODIES, gravity_seam_loads


def inputs(count=1):
    return dict(native_frames=np.repeat(np.eye(4)[None], count, axis=0),
                masses_kg=np.ones(count), local_coms_m=np.zeros((count, 3)),
                seam_world_m=np.zeros(3), unit_axis_world=np.array([0., 0., 1.]),
                radius_m=1., normal_strength_pa=10.,
                gravity_world_m_s2=np.array([0., 0., -10.]))


def test_point_mass_force_moment_and_com():
    data = inputs()
    data['masses_kg'][0] = 2.
    data['native_frames'][0, :3, 3] = [3., 4., 5.]
    result = gravity_seam_loads(**data)
    assert result['total_mass_kg'] == 2.
    assert result['combined_com_world_m'] == [3., 4., 5.]
    assert result['gravity_force_world_n'] == [0., 0., -20.]
    assert result['gravity_moment_world_nm'] == [-80., 60., 0.]
    assert result['bending_moment_nm'] == 100.
    assert result['axial_force_n'] == -20.
    assert result['transverse_force_n'] == 0.


def test_rotated_body_local_com_and_shifted_seam():
    data = inputs()
    data['native_frames'][0, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    data['native_frames'][0, :3, 3] = [1., 2., 3.]
    data['local_coms_m'][0] = [4., 0., 0.]
    data['seam_world_m'] = [1., 1., 1.]
    data['masses_kg'][0] = .2
    result = gravity_seam_loads(**data)
    np.testing.assert_allclose(result['combined_com_world_m'], [1., 6., 3.])
    np.testing.assert_allclose(result['gravity_moment_world_nm'], [-10., 0., 0.])
    data['unit_axis_world'] = [1., 0., 0.]
    result = gravity_seam_loads(**data)
    assert result['torsional_moment_nm'] == -10.
    assert result['bending_moment_nm'] == 0.
    assert result['transverse_force_n'] == 2.
    assert not result['circular_intact_elastic_screen']['shear_or_torsional_failure_screened']


def test_multiple_bodies_match_direct_individual_cross_products():
    data = inputs(3)
    data['native_frames'][:, :3, 3] = [[1., 2., 3.], [-4., 5., 6.], [0., -2., 8.]]
    data['masses_kg'] = [2., 3., 4.]
    data['local_coms_m'] = [[.1, .2, .3], [-.5, 0., 1.], [0., 1., -2.]]
    data['seam_world_m'] = [5., 6., 7.]
    data['gravity_world_m_s2'] = [1., -2., -9.]
    result = gravity_seam_loads(**data)
    coms = data['native_frames'][:, :3, 3] + data['local_coms_m']
    forces = np.asarray(data['masses_kg'])[:, None] * data['gravity_world_m_s2']
    np.testing.assert_allclose(result['gravity_force_world_n'], forces.sum(axis=0))
    np.testing.assert_allclose(result['gravity_moment_world_nm'],
                               np.cross(coms - data['seam_world_m'], forces).sum(axis=0))
    np.testing.assert_allclose(result['combined_com_world_m'],
                               np.average(coms, axis=0, weights=data['masses_kg']))


@pytest.mark.parametrize('sign,expected', [(1, 9 / math.pi), (-1, 7 / math.pi)])
def test_signed_axial_and_bending_stress_combination(sign, expected):
    data = inputs()
    data['gravity_world_m_s2'] = [0., 0., sign]
    data['native_frames'][0, :3, 3] = [0., 2., 0.]
    screen = gravity_seam_loads(**data)['circular_intact_elastic_screen']
    assert screen['maximum_tensile_stress_pa'] == pytest.approx(expected)
    assert screen['first_tensile_damage_utilization'] == pytest.approx(expected / 10)


def test_pure_compression_has_no_tensile_damage_claim_or_authorization():
    result = gravity_seam_loads(**inputs())
    screen = result['circular_intact_elastic_screen']
    assert screen['axial_stress_pa'] < 0
    assert screen['maximum_tensile_stress_pa'] == 0
    assert not screen['at_or_above_first_tensile_damage']
    assert not result['scope']['activation_permission']
    assert not result['scope']['constitutive_success']


@pytest.mark.parametrize('factor,at_onset', [(.5, False), (1., True), (2., True)])
def test_pure_tensile_onset_and_equality(factor, at_onset):
    data = inputs()
    data['masses_kg'][0] = 10 * math.pi * factor
    data['gravity_world_m_s2'] = [0., 0., 1.]
    screen = gravity_seam_loads(**data)['circular_intact_elastic_screen']
    assert screen['first_tensile_damage_utilization'] == pytest.approx(factor)
    assert screen['at_or_above_first_tensile_damage'] is at_onset


def test_full_three_mm_toy_disk_analytic_capacity_not_material_default():
    data = inputs()
    data.update(radius_m=.003, normal_strength_pa=10000.)
    screen = gravity_seam_loads(**data)['circular_intact_elastic_screen']
    assert screen['area_m2'] == pytest.approx(2.8274333882308137e-5)
    assert screen['second_moment_m4'] == pytest.approx(6.361725123519332e-11)
    assert screen['uniform_tensile_onset_force_n'] == pytest.approx(.2827433388230814)
    assert screen['pure_bending_first_tensile_onset_moment_nm'] == pytest.approx(.00021205750411731105)


def test_pure_bending_screen_and_axis_reversal():
    data = inputs()
    data['unit_axis_world'] = [1., 0., 0.]
    data['native_frames'][0, :3, 3] = [2., 0., 0.]
    result = gravity_seam_loads(**data)
    screen = result['circular_intact_elastic_screen']
    assert result['axial_force_n'] == 0
    assert screen['maximum_tensile_stress_pa'] == pytest.approx(20 / (math.pi / 4))
    data['unit_axis_world'] = [-1., 0., 0.]
    assert gravity_seam_loads(**data)['circular_intact_elastic_screen'] == screen


def test_zero_gravity():
    data = inputs(2)
    data['gravity_world_m_s2'] = [0., 0., 0.]
    data['native_frames'][1, :3, 3] = [7., 8., 9.]
    result = gravity_seam_loads(**data)
    assert result['gravity_force_world_n'] == [0., 0., 0.]
    assert result['gravity_moment_world_nm'] == [0., 0., 0.]
    assert result['circular_intact_elastic_screen']['first_tensile_damage_utilization'] == 0


@pytest.mark.parametrize('factor,at_onset', [(.5, False), (1., True), (2., True)])
def test_pure_bending_onset_boundary(factor, at_onset):
    data = inputs()
    data['native_frames'][0, 0, 3] = 1.
    data['masses_kg'][0] = 10 * math.pi / 4 * factor
    data['gravity_world_m_s2'] = [0., 0., -1.]
    data['unit_axis_world'] = [1., 0., 0.]
    result = gravity_seam_loads(**data)
    screen = result['circular_intact_elastic_screen']
    assert result['axial_force_n'] == 0
    assert screen['first_tensile_damage_utilization'] == pytest.approx(factor)
    assert screen['at_or_above_first_tensile_damage'] is at_onset
    assert result['scope']['activation_permission'] is False


def test_normal_reversal_changes_signed_axial_not_bending():
    data = inputs()
    data['native_frames'][0, 0, 3] = 1.
    forward = gravity_seam_loads(**data)
    data['unit_axis_world'] *= -1
    reverse = gravity_seam_loads(**data)
    assert reverse['axial_force_n'] == -forward['axial_force_n']
    assert reverse['bending_moment_nm'] == forward['bending_moment_nm']
    a = forward['circular_intact_elastic_screen']
    b = reverse['circular_intact_elastic_screen']
    assert b['maximum_tensile_stress_pa'] - a['maximum_tensile_stress_pa'] == pytest.approx(20 / math.pi)


def proper_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q[:, 0] *= np.linalg.det(q)
    return q


def test_body_local_origin_and_basis_change_preserves_world_loads():
    data = inputs()
    rng = np.random.default_rng(81)
    r, q, shift = proper_rotation(rng), proper_rotation(rng), np.array([2., -3., 4.])
    data['native_frames'][0, :3, :3] = r
    data['native_frames'][0, :3, 3] = [5., 4., -6.]
    data['local_coms_m'][0] = [7., 8., 9.]
    original = gravity_seam_loads(**data)
    data['native_frames'][0, :3, 3] += r @ shift
    data['native_frames'][0, :3, :3] = r @ q
    data['local_coms_m'][0] = q.T @ (data['local_coms_m'][0] - shift)
    transformed = gravity_seam_loads(**data)
    for key in ('combined_com_world_m', 'gravity_force_world_n', 'gravity_moment_world_nm'):
        np.testing.assert_allclose(transformed[key], original[key], atol=1e-12)


def test_joint_body_permutation_preserves_aggregate_loads():
    data = inputs(3)
    data['masses_kg'] = np.array([2., 3., 4.])
    data['native_frames'][:, :3, 3] = [[1., 2., 3.], [-4., 5., 6.], [0., -2., 8.]]
    data['local_coms_m'] = np.array([[.1, .2, .3], [-.5, 0., 1.], [0., 1., -2.]])
    original = gravity_seam_loads(**data)
    for key in ('masses_kg', 'native_frames', 'local_coms_m'):
        data[key] = data[key][[2, 0, 1]]
    assert gravity_seam_loads(**data) == original


def test_rigid_coordinate_equivariance():
    rng = np.random.default_rng(714)
    for _ in range(24):
        data = inputs(4)
        data['native_frames'][:, :3, :3] = [proper_rotation(rng) for _ in range(4)]
        data['native_frames'][:, :3, 3] = rng.normal(size=(4, 3))
        data['local_coms_m'] = rng.normal(size=(4, 3))
        data['masses_kg'] = rng.uniform(.01, 3., size=4)
        data['seam_world_m'] = rng.normal(size=3)
        data['unit_axis_world'] = proper_rotation(rng)[:, 0]
        data['gravity_world_m_s2'] = rng.normal(size=3)
        original = gravity_seam_loads(**data)
        q, t = proper_rotation(rng), rng.normal(size=3) * 20
        changed = deepcopy(data)
        changed['native_frames'][:, :3, :3] = q @ data['native_frames'][:, :3, :3]
        changed['native_frames'][:, :3, 3] = data['native_frames'][:, :3, 3] @ q.T + t
        changed['seam_world_m'] = q @ data['seam_world_m'] + t
        changed['unit_axis_world'] = q @ data['unit_axis_world']
        changed['gravity_world_m_s2'] = q @ data['gravity_world_m_s2']
        transformed = gravity_seam_loads(**changed)
        for key in ('gravity_force_world_n', 'gravity_moment_world_nm', 'seam_to_com_world_m',
                    'transverse_force_world_n', 'bending_moment_world_nm'):
            np.testing.assert_allclose(transformed[key], q @ original[key], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(transformed['combined_com_world_m'], q @ original['combined_com_world_m'] + t)
        for key in ('axial_force_n', 'torsional_moment_nm', 'bending_moment_nm', 'transverse_force_n'):
            assert transformed[key] == pytest.approx(original[key], abs=1e-11)
        assert transformed['circular_intact_elastic_screen']['maximum_tensile_stress_pa'] == pytest.approx(
            original['circular_intact_elastic_screen']['maximum_tensile_stress_pa'], abs=1e-11)


@pytest.mark.parametrize('bad_rotation', [np.diag([2., 1., 1.]), np.diag([-1., 1., 1.]),
    np.zeros((3, 3)), [[1., .01, 0.], [0., 1., 0.], [0., 0., 1.]], np.full((3, 3), 1e308)])
def test_invalid_frames(bad_rotation):
    data = inputs()
    data['native_frames'][0, :3, :3] = bad_rotation
    with pytest.raises(ValueError): gravity_seam_loads(**data)


@pytest.mark.parametrize('row', [[0., 0., 0., 2.], [.000001, 0., 0., 1.], [0., 0., 0., 0.]])
def test_nonhomogeneous_frames(row):
    data = inputs()
    data['native_frames'][0, 3] = row
    with pytest.raises(ValueError): gravity_seam_loads(**data)


@pytest.mark.parametrize('key,bad', [
    ('native_frames', np.eye(4)), ('native_frames', np.ones((1, 3, 4))),
    ('native_frames', np.ones((1, 4, 4), dtype=bool)),
    ('masses_kg', [[1.]]), ('masses_kg', [1., 2.]), ('masses_kg', [0.]),
    ('masses_kg', [-1.]), ('masses_kg', [True]), ('masses_kg', ['1']),
    ('masses_kg', [1j]), ('masses_kg', np.array([1.], dtype=object)),
    ('local_coms_m', [0., 0., 0.]), ('seam_world_m', [[0., 0., 0.]]),
    ('seam_world_m', [True, 0., 0.]), ('gravity_world_m_s2', -9.81),
    ('gravity_world_m_s2', ['0', '0', '-10']), ('unit_axis_world', [0., 0., 0.]),
    ('unit_axis_world', [0., 0., 2.]), ('unit_axis_world', [0., 0., 1.00001]),
    ('unit_axis_world', [1e308, 1e308, 1e308]), ('local_coms_m', [[0., 0.], [0.]]),
])
def test_invalid_array_inputs(key, bad):
    data = inputs(); data[key] = bad
    with pytest.raises(ValueError): gravity_seam_loads(**data)


@pytest.mark.parametrize('key', ['native_frames', 'masses_kg', 'local_coms_m',
                               'seam_world_m', 'unit_axis_world', 'gravity_world_m_s2'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_array_inputs(key, bad):
    data = inputs(); data[key].flat[0] = bad
    with pytest.raises(ValueError): gravity_seam_loads(**data)


@pytest.mark.parametrize('key', ['radius_m', 'normal_strength_pa'])
@pytest.mark.parametrize('bad', [0., -1., True, np.bool_(False), '1', [1.], np.array(1.),
                               1j, None, float('nan'), float('inf'), 10**400])
def test_invalid_scalar_inputs(key, bad):
    data = inputs(); data[key] = bad
    with pytest.raises(ValueError): gravity_seam_loads(**data)


@pytest.mark.parametrize('key,bad', [
    ('radius_m', 1e100), ('radius_m', 1e-100), ('normal_strength_pa', 1e308),
    ('masses_kg', [1e308]), ('local_coms_m', [[1e308, 0., 0.]]),
    ('gravity_world_m_s2', [0., 0., 1e308]),
])
def test_unrepresentable_derived_values_fail_without_nonfinite_json(key, bad, recwarn):
    data = inputs(); data[key] = bad
    data['native_frames'][0, :3, 3] = [10., 10., 10.]
    with pytest.raises(ValueError): gravity_seam_loads(**data)
    assert not recwarn


def test_mass_sum_overflow_and_force_underflow_fail_closed():
    data = inputs(2); data['masses_kg'][:] = 1e308
    with pytest.raises(ValueError): gravity_seam_loads(**data)
    data = inputs(); data['masses_kg'][0] = 1e-200
    data['gravity_world_m_s2'] = [0., 0., 1e-200]
    with pytest.raises(ValueError): gravity_seam_loads(**data)


def test_body_count_bound_including_empty():
    for count in (0, MAX_BODIES + 1):
        with pytest.raises(ValueError): gravity_seam_loads(**inputs(count))
    data = inputs(MAX_BODIES); data['gravity_world_m_s2'][:] = 0.
    assert gravity_seam_loads(**data)['body_count'] == MAX_BODIES


def test_float32_rotations_and_axis_roundoff_are_explicit():
    data = inputs()
    data['native_frames'][0, :3, :3] = proper_rotation(np.random.default_rng(8)).astype(np.float32)
    data['unit_axis_world'] = [0., 0., 1.0000005]
    result = gravity_seam_loads(**data)
    assert result['unit_axis_world'] == [0., 0., 1.]
    assert result['provenance']['unit_axis_roundoff_normalized']


@pytest.mark.parametrize('offset', [0., 1e4])
def test_advisory_flags_json_and_input_immutability(offset):
    data = inputs(); data['native_frames'][0, 0, 3] = offset
    before = deepcopy(data)
    result = gravity_seam_loads(**data)
    assert result == gravity_seam_loads(**data)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    for key in ('actual_seam_reaction', 'equilibrium_verified', 'material_calibration_verified',
                'constitutive_success', 'activation_permission', 'physical_cut_verified'):
        assert result['scope'][key] is False
    for key in ('gravity_only', 'gripper_reaction_excluded', 'other_contact_reactions_excluded',
                'inertial_reaction_excluded', 'other_external_loads_excluded'):
        assert result['scope'][key] is True
    for key in ('body_selection_verified', 'common_snapshot_verified',
                'native_frames_masses_COMs_verified', 'source_hashes_verified',
                'proximal_to_distal_axis_orientation_verified'):
        assert result['provenance'][key] is False
    result['seam_world_m'][0] = 300.
    for key, value in data.items(): np.testing.assert_array_equal(value, before[key])


def test_all_inputs_required_no_physical_defaults():
    for key in inputs():
        data = inputs(); del data[key]
        with pytest.raises(TypeError): gravity_seam_loads(**data)


def test_import_and_calculation_need_no_native_modules():
    script = '''
import importlib.abc, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('carb', 'omni', 'isaacsim', 'pxr'):
            raise AssertionError('Unexpected native import: '+fullname)
sys.meta_path.insert(0, BlockNative())
sys.path.insert(0, sys.argv[1])
from sim_physics.seam_loads import gravity_seam_loads
import numpy as np
gravity_seam_loads(native_frames=np.eye(4)[None], masses_kg=[1.], local_coms_m=[[0.,0.,0.]],
    seam_world_m=[0.,0.,0.], unit_axis_world=[0.,0.,1.], radius_m=1., normal_strength_pa=1.,
    gravity_world_m_s2=[0.,0.,-1.])
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script,
                             str(Path(__file__).resolve().parents[1])],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
