import numpy as np
import pytest

from .bimanual_probe import released_stroke_fraction
from .gripper_probe import ramp


def test_release_latches_sent_command_not_one_tick_later_fetch_time():
    fetch=11.866666666666667
    sent=ramp(fetch-1/240,8,14)
    assert released_stroke_fraction(('stroke',sent))==sent
    assert released_stroke_fraction(('stroke',sent))<ramp(fetch,8,14)


@pytest.mark.parametrize('phase',['park','approach','withdraw',None])
def test_only_actually_sent_stroke_can_seed_reversal(phase):
    with pytest.raises(RuntimeError):released_stroke_fraction((phase,.5))


@pytest.mark.parametrize('fraction',[-.001,1.001,float('nan'),float('inf'),True,np.bool_(False),'0.5'])
def test_invalid_fraction_never_silently_clamps(fraction):
    with pytest.raises(RuntimeError):released_stroke_fraction(('stroke',fraction))


@pytest.mark.parametrize('fraction',[0.,.5,1.,np.float32(.25)])
def test_valid_sent_endpoints_remain_exact(fraction):
    assert released_stroke_fraction(('stroke',fraction))==fraction
