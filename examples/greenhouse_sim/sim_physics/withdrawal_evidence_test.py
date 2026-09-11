"""Pure endpoint tests only: no native physics, runtime import or model download."""
import json
import math

import numpy as np
import pytest

from sim_physics.withdrawal_evidence import (
    ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M, withdrawal_evidence,
)


def pose(angle=0., translation=(0., 0., 0.), axis=(0., 0., 1.)):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    result = np.eye(4)
    result[:3, :3] = np.eye(3) + math.sin(angle)*skew + (1.-math.cos(angle))*(skew @ skew)
    result[:3, 3] = translation
    return result


def evidence(measured=None, park=None, **kwargs):
    options = dict(clearance_verified=True, measured_sample_id=(3, 719),
                   clearance_sample_id=(3, 719))
    options.update(kwargs)
    return withdrawal_evidence(np.eye(4) if measured is None else measured,
                               np.eye(4) if park is None else park, **options)


def test_exact_endpoint_needs_explicit_same_sample_clearance():
    result = evidence()
    assert result['right_withdrawal_completed'] and result['endpoint_attained']
    assert result['same_native_sample']
    assert not evidence(clearance_verified=False)['right_withdrawal_completed']
    assert evidence(clearance_verified=False)['endpoint_attained']
    assert not result['whole_path_certified']
    assert not result['full_forward_cutstroke_verified']
    assert not result['physical_cut_verified']
    assert not result['native_provenance_verified_by_helper']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}, np.bool_(True)])
def test_no_truthy_or_missing_clearance(value):
    with pytest.raises(ValueError, match='boolean'):
        evidence(clearance_verified=value)


def test_clearance_keyword_is_required():
    with pytest.raises(TypeError):
        withdrawal_evidence(np.eye(4), np.eye(4), measured_sample_id=(0, 1),
                            clearance_sample_id=(0, 1))


@pytest.mark.parametrize('translation,passed', [
    ((.00049, 0., 0.), True), ((.0005, 0., 0.), False),
    ((.005, 0., 0.), False), ((.0004, .0004, 0.), False),
])
def test_position_uses_half_millimetre_euclidean_strict_limit(translation, passed):
    result = evidence(pose(translation=translation))
    assert POSITION_TOLERANCE_M == .0005
    assert result['right_withdrawal_completed'] is passed
    assert result['position_error_m'] == pytest.approx(math.hypot(*translation))


@pytest.mark.parametrize('angle,passed', [
    (0., True), (1e-12, True), (.0049, True), (.005, False),
    (.0051, False), (-.0051, False), (math.pi-1e-12, False), (math.pi, False),
])
def test_rotation_geodesic_including_pi(angle, passed):
    result = evidence(pose(angle))
    assert ORIENTATION_TOLERANCE_RAD == .005
    assert result['orientation_error_rad'] == pytest.approx(abs(angle), abs=1e-14)
    assert result['right_withdrawal_completed'] is passed


@pytest.mark.parametrize('axis', [(1., 0., 0.), (0., 1., 0.), (1., 2., 3.)])
def test_large_rotation_never_looks_like_zero_skew(axis):
    result = evidence(pose(math.pi, axis=axis))
    assert result['orientation_error_rad'] == pytest.approx(math.pi)
    assert not result['endpoint_attained']


def test_world_frame_invariance_symmetry_float32_and_no_input_mutation():
    world = pose(.7, (1., -2., .8), (1., 2., -1.))
    measured = pose(.004, (.0001, -.0002, .0002))
    original = measured.copy()
    baseline = evidence(measured)
    moved = evidence(world @ measured, world)
    swapped = evidence(park=measured)
    for other in (moved, swapped):
        assert other['position_error_m'] == pytest.approx(baseline['position_error_m'], abs=1e-14)
        assert other['orientation_error_rad'] == pytest.approx(baseline['orientation_error_rad'], abs=1e-14)
        assert other['right_withdrawal_completed']
    assert evidence((world @ measured).astype(np.float32), world.astype(np.float32))['right_withdrawal_completed']
    np.testing.assert_array_equal(measured, original)


@pytest.mark.parametrize('sample', [(3, 718), (3, 720), (2, 719)])
def test_old_future_or_other_episode_clearance_cannot_complete(sample):
    result = evidence(clearance_sample_id=sample)
    assert result['endpoint_attained'] and result['clearance_verified']
    assert not result['same_native_sample']
    assert not result['right_withdrawal_completed']


@pytest.mark.parametrize('key', ['measured_sample_id', 'clearance_sample_id'])
@pytest.mark.parametrize('sample', [None, 719, (0,), (0, 1, 2), (True, 1),
                                   (0, False), (-1, 1), (0, -1), (0, 1.), ('0', 1)])
def test_invalid_sample_ids_rejected(key, sample):
    with pytest.raises(ValueError):
        evidence(**{key: sample})


def test_integer_sample_ids_are_serializable_and_success_does_not_latch():
    result = evidence(measured_sample_id=[np.int64(3), np.int64(719)])
    assert result['right_withdrawal_completed']
    json.dumps(result, allow_nan=False)
    assert not evidence(pose(translation=(.01, 0., 0.)))['right_withdrawal_completed']
    assert not evidence(clearance_verified=False)['right_withdrawal_completed']


def invalid_poses():
    result = [np.eye(3), np.zeros((4, 4))]
    for index, value in [((0, 3), np.nan), ((2, 1), np.inf), ((3, 0), .001),
                         ((3, 3), 2.), ((0, 0), -1.), ((0, 0), 1.01), ((0, 1), .01)]:
        bad = np.eye(4)
        bad[index] = value
        result.append(bad)
    return result


@pytest.mark.parametrize('bad', invalid_poses())
@pytest.mark.parametrize('key', ['measured', 'park'])
def test_nonfinite_nonrigid_reflected_and_scaled_inputs_fail_closed(key, bad):
    with pytest.raises(ValueError):
        evidence(**{key: bad})


def test_unrepresentable_finite_translation_difference_fails_closed():
    with pytest.raises(ValueError, match='Unrepresentable'):
        evidence(pose(translation=(1.7e308, 0., 0.)),
                 pose(translation=(-1.7e308, 0., 0.)))
