import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from .rod_strain import RodStrain


def frames(n=4):
    f = np.tile(np.eye(4), (n, 1, 1))
    f[:, 2, 3] = np.arange(n)*.02
    return f


def audit(rest=None, cut=1):
    rest = frames() if rest is None else rest
    return RodStrain(rest, [.02]*4, [.003]*4, [[.4, .4, .3]]*4, cut)


def test_rest_and_rigid_motion_no_strain():
    rest = frames()
    rest[2:, :3, :3] = Rotation.from_euler('xyz', [.3, .2, .1]).as_matrix()
    probe = audit(rest)
    moved = rest.copy()
    rot = Rotation.from_euler('xyz', [.7, -.5, .2]).as_matrix()
    moved[:, :3, :3] = rot@moved[:, :3, :3]
    moved[:, :3, 3] = moved[:, :3, 3]@rot.T+[2, 1, -3]
    assert probe.evaluate(moved)['quadratic_elastic_energy_estimate_j'] < 1e-25


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_known_internal_deformation(axis):
    f = frames(); vector = np.zeros(3); vector[axis] = .02
    f[2:, :3, :3] = Rotation.from_rotvec(vector).as_matrix()
    r = audit().evaluate(f)
    assert r['internal_child_indices'] == [2, 3]
    np.testing.assert_allclose(r['relative_rotation_vectors_rad'][0], vector, atol=1e-14)
    assert r['quadratic_elastic_energy_estimate_j'] == pytest.approx(.5*([.4, .4, .3][axis])*.02**2)
    if axis == 2:
        assert r['maximum_torsional_surface_shear_estimate'] == pytest.approx(.003)
        assert r['maximum_bending_surface_strain_estimate'] == 0
    else:
        assert r['maximum_bending_surface_strain_estimate'] == pytest.approx(.003)
        assert r['maximum_torsional_surface_shear_estimate'] == 0


def test_release_rotation_excludes_seam():
    f = frames(); f[1:, :3, :3] = Rotation.from_euler('x', 1.).as_matrix()
    assert audit().evaluate(f)['quadratic_elastic_energy_estimate_j'] == 0


def test_half_turn_explicitly_ambiguous_not_damage_certificate():
    f = frames(); f[2:, :3, :3] = Rotation.from_euler('x', np.pi).as_matrix()
    r = audit().evaluate(f)
    assert not r['principal_rotation_branch_clear']
    assert not r['tissue_damage_calibrated'] and not r['execution_authority']


@pytest.mark.parametrize('kind', ['nan', 'scale', 'reflection', 'bad_row', 'missing'])
def test_invalid_native_frames(kind):
    f = frames()
    if kind == 'nan': f[0, 0, 0] = np.nan
    elif kind == 'scale': f[1, :3, :3] *= 2
    elif kind == 'reflection': f[0, 0, 0] = -1
    elif kind == 'bad_row': f[0, 3, 1] = 1
    else: f = f[:-1]
    with pytest.raises(ValueError): audit().evaluate(f)


@pytest.mark.parametrize('kind', ['cut_bool', 'cut_end', 'negative_length', 'zero_radius', 'bad_stiffness'])
def test_invalid_model(kind):
    lengths = [.02]*4; radii = [.003]*4; stiffness = [[.4]*3]*4; cut = 1
    if kind == 'cut_bool': cut = True
    elif kind == 'cut_end': cut = 3
    elif kind == 'negative_length': lengths[0] = -.02
    elif kind == 'zero_radius': radii[1] = 0
    else: stiffness = [[.4]*2]*4
    with pytest.raises(ValueError): RodStrain(frames(), lengths, radii, stiffness, cut)
