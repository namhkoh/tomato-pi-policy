"""Pure NumPy constitutive checks; toy material values are NOT tissue fits."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from sim_physics.cohesive import (
    ENGINEERING_STATUS, CohesiveParameters, CohesiveState, cohesive_response,
)


@pytest.fixture
def material():
    # Deliberately dimensioned generic values, unrelated to native force guards.
    # r0=0.1 mm, rf=1 mm; elastic onset energy=0.5 J/m^2, Gc=5 J/m^2.
    return CohesiveParameters(1e8, 4e8, 1e4, 5.0)


def response(state, jump, area=2e-6, normal=(1., 0., 0.)):
    return cohesive_response(state, jump, normal, area_m2=area)


def test_units_derived_modes_and_no_material_defaults(material):
    assert material.initiation_separation_m == pytest.approx(1e-4)
    assert material.final_separation_m == pytest.approx(1e-3)
    assert material.shear_strength_pa == pytest.approx(2e4)
    with pytest.raises(TypeError):
        CohesiveParameters()


def test_zero_and_absent_connectivity_do_not_create_energy_or_contact(material):
    virgin = CohesiveState(material)
    zero = response(virgin, [0., 0., 0.])
    assert zero.state.damage == zero.stored_energy_j == zero.dissipated_energy_j == 0
    assert zero.force_a_n == zero.force_b_n == (0., 0., 0.)
    unbonded = CohesiveState(material, bonded=False)
    for jump in ([0., 0., 0.], [.002, .01, 0.], [-.002, .01, 0.]):
        empty = response(unbonded, jump)
        assert empty.state is unbonded
        assert not empty.state.fully_separated
        assert empty.force_a_n == (0., 0., 0.)
        assert empty.constitutive_work_increment_j == empty.dissipated_energy_j == 0
    # A stretched BOND carries traction without any rigid-contact input/callback.
    assert response(virgin, [5e-5, 0., 0.]).force_a_n[0] > 0


def test_exact_separation_endpoint_not_rounded_damage(material):
    virgin = CohesiveState(material)
    almost = response(virgin, [np.nextafter(material.final_separation_m, 0.), 0., 0.])
    assert not almost.state.fully_separated
    assert almost.state.retained_stiffness > 0 and almost.traction_a_pa[0] > 0
    complete = response(almost.state, [material.final_separation_m, 0., 0.])
    assert complete.state.fully_separated and complete.force_a_n == (0., 0., 0.)
    assert response(complete.state, [0., 0., 0.]).state.fully_separated


def test_subcritical_unload_reload_is_elastic_and_recovers_work(material):
    state = CohesiveState(material)
    work = 0.
    for x in (0., 5e-5, 2e-5, 5e-5, 0.):
        result = response(state, [x, 0., 0.])
        state = result.state
        assert state.damage == result.dissipation_increment_j == 0
        assert result.traction_a_pa[0] == pytest.approx(material.normal_stiffness_pa_m * x)
        work += result.constitutive_work_increment_j
    assert work == pytest.approx(0., abs=1e-18)


def test_damage_unloading_reloading_irreversibility_and_no_input_mutation(material):
    virgin = CohesiveState(material)
    peak = response(virgin, [4e-4, 0., 0.])
    assert 0 < peak.state.damage < 1
    assert virgin.damage == 0
    previous = peak.state
    for x in (2e-4, 0., -2e-4, 1e-4, 4e-4):
        result = response(previous, [x, 0., 0.])
        assert result.state.damage == peak.state.damage
        assert result.dissipation_increment_j == 0
        assert result.traction_a_pa[0] == pytest.approx(
            peak.state.retained_stiffness * material.normal_stiffness_pa_m * max(x, 0.))
        previous = result.state
    assert response(previous, [6e-4, 0., 0.]).state.damage > peak.state.damage
    with pytest.raises(FrozenInstanceError):
        peak.state.maximum_effective_separation_m = 0


@pytest.mark.parametrize("direction", [(1., 0., 0.), (0., 1., 0.), (0., 0., -1.), (.6, .8, 0.)])
def test_bilinear_envelope_complete_separation_and_no_healing(material, direction):
    # Unit directions in energy-normalized coordinates.
    direction = np.array(direction) / [1., material.shear_weight, material.shear_weight]
    state = CohesiveState(material)
    r0, rf = material.initiation_separation_m, material.final_separation_m
    for r in (0., r0 / 2, r0, (r0 + rf) / 2, rf, 2 * rf):
        result = response(state, r * direction)
        radial = np.dot(result.traction_a_pa, direction)
        expected = (material.normal_stiffness_pa_m * r if r <= r0
                    else material.normal_strength_pa * max(0., (rf-r)/(rf-r0)))
        assert radial == pytest.approx(expected, abs=1e-10)
        state = result.state
    assert state.damage == 1
    assert result.stored_energy_j == 0
    assert result.dissipated_energy_j == pytest.approx(2e-6 * material.fracture_energy_j_m2)
    for jump in ([0., 0., 0.], [1e-5, 0., 0.], [-.002, .01, -.01]):
        result = response(state, jump)
        state = result.state
        assert state.damage == 1 and result.force_a_n == (0., 0., 0.)
        assert result.constitutive_work_increment_j == result.dissipation_increment_j == 0


def test_compression_is_separate_and_does_not_fracture(material):
    state = CohesiveState(material)
    for x in (-1e-6, -.002, -.01, 0.):
        result = response(state, [x, 0., 0.])
        state = result.state
        assert result.compression_m == -x
        assert state.damage == 0 and result.force_a_n == (0., 0., 0.)
        assert result.stored_energy_j == result.dissipated_energy_j == 0
    # Compression does not erase real shear separation or supply Coulomb friction.
    shear = response(state, [0., 2e-4, 0.])
    compressed_shear = response(state, [-.001, 2e-4, 0.])
    assert compressed_shear.state == shear.state
    np.testing.assert_allclose(compressed_shear.force_a_n, shear.force_a_n)


@pytest.mark.parametrize("subdivisions", [1, 3, 37, 1000])
def test_monotonic_subdivision_and_holds_do_not_change_state_or_energy(material, subdivisions):
    virgin = CohesiveState(material)
    target = np.array([.00048, .00032, 0.])  # r=0.8 mm, mixed mode.
    direct = response(virgin, target)
    state = virgin
    work = 0.
    for fraction in np.linspace(0., 1., subdivisions + 1):
        result = response(state, fraction * target)
        state = result.state
        work += result.constitutive_work_increment_j
        held = response(state, fraction * target)
        assert held.state == state and held.constitutive_work_increment_j == 0
    assert state == direct.state
    assert work == pytest.approx(direct.constitutive_work_increment_j, rel=1e-12)
    np.testing.assert_allclose(result.force_a_n, direct.force_a_n)


def test_nonproportional_path_energy_balance_against_independent_traction_work(material):
    # Includes softening, mode changes, shear reversal, compression and closure.
    waypoints = np.array([[0., 0., 0.], [.0004, 0., 0.], [0., .0002, 0.],
                          [-.0002, .0002, 0.], [0., 0., -.0003],
                          [.0008, .0004, 0.], [0., 0., 0.]])
    state = CohesiveState(material)
    previous_jump = np.zeros(3)
    previous_force = np.zeros(3)
    actual_integral = budget = 0.
    previous_stored = previous_dissipated = 0.
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        for jump in np.linspace(start, end, 1001)[1:]:
            result = response(state, jump)
            force = np.array(result.force_a_n)
            actual_integral += .5 * np.dot(previous_force + force, jump - previous_jump)
            budget += result.constitutive_work_increment_j
            assert result.dissipation_increment_j >= 0
            assert result.dissipated_energy_j >= previous_dissipated
            assert result.constitutive_work_increment_j == pytest.approx(
                result.stored_energy_j - previous_stored + result.dissipation_increment_j,
                abs=1e-19)
            state, previous_jump, previous_force = result.state, jump, force
            previous_stored, previous_dissipated = result.stored_energy_j, result.dissipated_energy_j
    assert state.damage == 1 and result.stored_energy_j == 0
    assert budget == pytest.approx(material.fracture_energy_j_m2 * result.area_m2, rel=1e-12)
    assert actual_integral == pytest.approx(budget, rel=1e-5)


def test_tractions_are_potential_gradient_at_fixed_damage(material):
    state = response(CohesiveState(material), [.0008, 0., 0.]).state
    jump = np.array([.0002, .0001, -.0001])
    result = response(state, jump, area=1.)
    gradient = []
    for axis in np.eye(3):
        hi = response(state, jump + 1e-9 * axis, area=1.)
        lo = response(state, jump - 1e-9 * axis, area=1.)
        assert hi.state.damage == lo.state.damage == state.damage
        gradient.append((hi.stored_energy_j - lo.stored_energy_j) / 2e-9)
    np.testing.assert_allclose(gradient, result.traction_a_pa, rtol=1e-9, atol=1e-7)


def test_action_reaction_face_swap_and_rotational_covariance(material):
    normal = np.array([1., 0., 0.])
    angle = .71
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    # Compose another axis so all three world components participate.
    rotation = rotation @ np.array([[1., 0., 0.], [0., .6, -.8], [0., .8, .6]])
    original = rotated = swapped = CohesiveState(material)
    for jump in np.array([[.0003, .0001, -.0002], [.0001, 0., .0001], [-.0002, .0001, 0.]]):
        a = response(original, jump)
        b = response(rotated, rotation @ jump, normal=rotation @ normal)
        c = response(swapped, -jump, normal=-normal)
        np.testing.assert_array_equal(np.array(a.force_a_n) + a.force_b_n, np.zeros(3))
        np.testing.assert_array_equal(np.array(a.traction_a_pa) + a.traction_b_pa, np.zeros(3))
        np.testing.assert_allclose(b.force_a_n, rotation @ a.force_a_n, atol=1e-15)
        np.testing.assert_allclose(c.force_a_n, a.force_b_n, atol=1e-15)
        assert b.state.damage == pytest.approx(a.state.damage)
        assert b.dissipated_energy_j == pytest.approx(a.dissipated_energy_j)
        assert c.state == a.state
        original, rotated, swapped = a.state, b.state, c.state


@pytest.mark.parametrize("pieces", [2, 7, 31])
def test_area_refinement_preserves_force_energy_and_material_history(material, pieces):
    whole = CohesiveState(material)
    children = [whole] * pieces
    # Unequal reference areas, not an equal-count special case.
    areas = np.arange(1., pieces + 1)
    areas *= 2e-6 / areas.sum()
    for jump in ([0., 0., 0.], [.0003, .0001, 0.], [.0001, 0., 0.], [.002, 0., 0.]):
        coarse = response(whole, jump)
        fine = [response(state, jump, area) for state, area in zip(children, areas)]
        np.testing.assert_allclose(np.sum([r.force_a_n for r in fine], axis=0), coarse.force_a_n)
        for field in ("stored_energy_j", "dissipated_energy_j", "dissipation_increment_j",
                      "constitutive_work_increment_j"):
            assert sum(getattr(r, field) for r in fine) == pytest.approx(getattr(coarse, field), abs=1e-18)
        assert all(r.state == coarse.state for r in fine)
        whole, children = coarse.state, [r.state for r in fine]


@pytest.mark.parametrize("field", ["normal_stiffness_pa_m", "shear_stiffness_pa_m",
                                   "normal_strength_pa", "fracture_energy_j_m2"])
@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), True, "1"])
def test_invalid_material_values_rejected(material, field, bad):
    with pytest.raises(ValueError):
        replace(material, **{field: bad})


def test_energy_inequality_and_unrepresentable_material_rejected(material):
    for gc in (.1, .5):
        with pytest.raises(ValueError, match="Gc"):
            replace(material, fracture_energy_j_m2=gc)
    for kwargs in (dict(normal_stiffness_pa_m=1e-310), dict(normal_strength_pa=1e-310),
                   dict(fracture_energy_j_m2=1e308, normal_strength_pa=1.)):
        with pytest.raises(ValueError):
            replace(material, **kwargs)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), True, [1.], "1"])
def test_invalid_area_rejected_even_when_unbonded(material, bad):
    with pytest.raises(ValueError):
        response(CohesiveState(material, bonded=False), [0., 0., 0.], bad)


@pytest.mark.parametrize("bad", [[0., 0.], [0., 0., 0., 0.], [0., float("nan"), 0.],
                                 [float("inf"), 0., 0.], [1j, 0., 0.], [True, False, True],
                                 ["1", "0", "0"]])
def test_invalid_vectors_rejected(material, bad):
    with pytest.raises(ValueError):
        response(CohesiveState(material), bad)
    with pytest.raises(ValueError):
        response(CohesiveState(material), [0., 0., 0.], normal=bad)


@pytest.mark.parametrize("normal", [[0., 0., 0.], [2., 0., 0.], [1e300, 0., 0.]])
def test_nonunit_normals_rejected(material, normal):
    with pytest.raises(ValueError, match="unit"):
        response(CohesiveState(material), [0., 0., 0.], normal=normal)


def test_invalid_history_and_overflow_fail_closed(material):
    with pytest.raises(ValueError):
        replace(material, normal_stiffness_pa_m=10**1000)
    for kwargs in (dict(bonded=1), dict(maximum_effective_separation_m=-1.),
                   dict(effective_separation_m=float("nan")),
                   dict(effective_separation_m=.1),
                   dict(bonded=False, maximum_effective_separation_m=.1)):
        with pytest.raises(ValueError):
            CohesiveState(material, **kwargs)
    with pytest.raises(ValueError):
        cohesive_response(None, [0., 0., 0.], [1., 0., 0.], area_m2=1.)
    with pytest.raises(ValueError):
        response(CohesiveState(material), [5e-5, 0., 0.], area=1e308)
    with pytest.raises(ValueError):
        response(CohesiveState(material), [0., 1e308, 1e308])


def test_report_never_claims_native_work_calibration_or_cut(material):
    report = response(CohesiveState(material), [.002, 0., 0.]).report()
    assert report["status"] == ENGINEERING_STATUS
    assert report["damage"] == 1
    assert report["fully_separated"] and report["bonded"]
    assert not report["material_calibrated"] and not report["physical_cut_verified"]
    assert not report["seam_release_authorized"]
    assert not report["compression_response_included"] and not report["friction_included"]
    assert report["work_is_constitutive_not_native_measurement"]
