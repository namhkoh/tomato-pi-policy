"""Native resolution fidelity comparison and fresh reference selection."""
from copy import deepcopy
import numpy as np
import pytest
from sim_data import native_render_probe as probe


@pytest.fixture
def buffers(monkeypatch):
    p=dict(rgb=np.full((816,1696,3),100,np.uint8),
        depth=np.ones((816,1696),np.float32), valid=np.ones((816,1696),bool),
        ids=np.full((816,1696),2,np.uint32),mapping={2:'/World/Stem'})
    monkeypatch.setattr(probe,'validate_native_payload',lambda p,c:(p['rgb'],p['depth'],p['valid'],[0,1]))
    monkeypatch.setattr(probe,'decode_native_instances',lambda p,r:(p['ids'],p['mapping']))
    return p


CAL={'resolution':[1696,816]}


def test_native_roi_beyond_legacy_bounds_is_checked(buffers):
    b=deepcopy(buffers);b['rgb'][600:730,1250:1380]=150
    result=probe.comparison(buffers,b,CAL,roi_pixel=[1310,660])
    assert not result['passed'] and result['target_roi_radius_px']==64
    assert result['rgb']['roi_mean']>40 and result['rgb']['mean']<3


@pytest.mark.parametrize('kind',['depth','valid','identity','rgb'])
def test_changed_buffers_reject(buffers,kind):
    b=deepcopy(buffers)
    if kind=='depth':b['depth'][700,1400]+=.01
    if kind=='valid':b['valid'][700,1400]=False
    if kind=='identity':b['mapping'][2]='/World/Leaf'
    if kind=='rgb':b['rgb']+=10
    assert not probe.comparison(buffers,b,CAL,roi_pixel=[1400,700])['passed']


def test_id_renumbering_preserves_resolved_identity(buffers):
    b=deepcopy(buffers);b['ids'][:]=17;b['mapping']={17:'/World/Stem',18:'/Unused'}
    assert probe.comparison(buffers,b,CAL,roi_pixel=[1600,800])['passed']


def test_mismatch_location_and_counts_are_not_hidden(buffers):
    b=deepcopy(buffers);b['ids'][700,1400]=3;b['mapping'][3]='/World/Leaf'
    result=probe.comparison(buffers,b,CAL,roi_pixel=[1400,700])
    assert not result['passed']
    assert result['identity_differences']['changed_pixels']==1
    assert result['identity_differences']['target_roi_changed_pixels']==1
    assert result['identity_differences']['largest_changed_prim_pairs'][0]['current']=='/World/Leaf'


@pytest.mark.parametrize('roi',[None,[1700,0],[0,816],[-1,0],[float('nan'),0],[1,2,3]])
def test_invalid_roi_not_silently_replaced_by_full_image(buffers,roi):
    with pytest.raises(ValueError):probe.comparison(buffers,buffers,CAL,roi_pixel=roi)


def test_probe_returns_last_reference_even_when_candidate_is_poor(buffers,monkeypatch):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes):
        calls.append(subframes)
        return {**buffers,'rgb':buffers['rgb']+(20 if len(calls)==1 else 0),
                'sequence':len(calls)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    result,evidence=probe.noise_probe(None,None,CAL,roi_pixel=[1400,700])
    assert calls==[56]+[8]*14 and result['sequence']==15
    assert evidence['requested_subframes_total']==sum(calls)==168
    assert not evidence['experimental_candidate_approved']
    assert not evidence['comparisons']['candidate_reference']['passed']
    assert evidence['comparisons']['reference_repeat']['passed']
    assert evidence['training_diversity_increment']==0
    assert np.array_equal(result['rgb'],buffers['rgb'])
