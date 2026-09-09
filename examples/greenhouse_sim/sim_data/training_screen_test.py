import numpy as np
import pytest

from sim_data.training_screen import StaticBoundScreen
from sim_data.capture_viewpoints import screen_bounds


def boxes(rng,count,scale):
    low=rng.uniform(-scale,scale,(count,3)); high=low+rng.uniform(0,.3,(count,3))
    return [dict(path=str(i),min=a.tolist(),max=b.tolist()) for i,(a,b) in enumerate(zip(low,high))]


@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('margin',[0,.01,.5])
def test_cached_screen_matches_reference_exactly(seed,margin):
    rng=np.random.default_rng(seed); robot=boxes(rng,40,1); scene=boxes(rng,3000,10)
    refine=lambda a,b,m:int(b['path'])%3!=0
    assert StaticBoundScreen(scene,refine)(robot,margin)==screen_bounds(robot,scene,margin,refine)


def test_invalid_empty_bounds_cannot_bypass_validation():
    with pytest.raises(ValueError): StaticBoundScreen([])
    scene=[dict(path='s',min=[0,0,0],max=[1,1,1])]
    s=StaticBoundScreen(scene)
    with pytest.raises(ValueError): s([])
    with pytest.raises(ValueError): s(scene,-1)
    with pytest.raises(ValueError): s([dict(path='r',min=[float('nan'),0,0],max=[1,1,1])])
