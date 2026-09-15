"""Fast native mapping must preserve every visible pixel at both resolutions."""
from copy import deepcopy
import numpy as np
import pytest
from sim_data.native_instance_adapter import normalize,LEGACY,FAST

def payload(resolution):
    w,h=resolution;pixels=np.full((h,w),2,np.uint32);pixels[-1,-1]=3
    return {LEGACY:dict(data=pixels.copy(),info={'idToLabels':{'2':'/Stem','3':'/Leaf','4':'/Unused'}}),
            FAST:dict(data=pixels.copy(),info={'ids':np.array([2,3,4],np.uint32),
                                             'labels':['/Stem','/Leaf','/Unused']}),
            'rgb_marker':'unchanged'}

@pytest.mark.parametrize('resolution',[(848,408),(1696,816)])
def test_every_pixel_and_observed_identity_is_exact(resolution):
    p=payload(resolution);before=deepcopy(p)
    result=normalize(p,'compare',resolution)
    assert result['native_instance_equivalence']['passed']
    assert result['native_instance_equivalence']['compared_pixels']==resolution[0]*resolution[1]
    assert result[LEGACY]['info']['idToLabels']=={2:'/Stem',3:'/Leaf'}
    assert result['rgb_marker']=='unchanged' and FAST not in result
    assert np.array_equal(p[FAST]['data'],before[FAST]['data'])
    assert p[FAST]['info']['labels']==before[FAST]['info']['labels']

@pytest.mark.parametrize('kind',['pixel','mapping','shape','dtype','duplicate','missing'])
def test_invalid_or_different_native_data_fails(kind):
    p=payload((1696,816))
    if kind=='pixel':p[FAST]['data'][0,0]=3
    if kind=='mapping':p[FAST]['info']['labels'][0]='/Fruit'
    if kind=='shape':p[FAST]['data']=p[FAST]['data'][:408,:848]
    if kind=='dtype':p[FAST]['data']=p[FAST]['data'].astype(np.uint16)
    if kind=='duplicate':p[FAST]['info']['ids'][0]=3
    if kind=='missing':p.pop(FAST)
    with pytest.raises(ValueError):normalize(p,'compare',(1696,816))

def test_fast_mode_does_not_depend_on_legacy_node():
    p=payload((1696,816));p.pop(LEGACY)
    result=normalize(p,'fast',(1696,816))
    assert 'native_instance_equivalence' not in result
    assert result['native_instance_backend']=='fast'
    assert set(result[LEGACY]['info']['idToLabels'])=={2,3}

