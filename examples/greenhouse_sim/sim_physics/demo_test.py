import numpy as np
import pytest
from sim_physics.demo import pulse_force


@pytest.mark.parametrize('time, expected', [(0,0),(.999,0),(1,.02),(1.999,.02),(2,0),(6,0)])
def test_pull_is_bounded_one_second_only(time, expected):
    np.testing.assert_array_equal(pulse_force(time), [0,expected,0])


@pytest.mark.parametrize('force', [-1, .201, float('nan'), float('inf')])
def test_invalid_force_rejected(force):
    with pytest.raises(ValueError,match='between'):
        pulse_force(1,force)
