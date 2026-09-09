from copy import deepcopy
import numpy as np
import pytest

from sim_data import static_render


@pytest.fixture
def buffers(monkeypatch):
    a={'rgb':np.full((408,848,3),100,np.uint8),'depth':np.full((408,848),1.,np.float32),
       'valid':np.ones((408,848),bool),'ids':np.full((408,848),2,np.uint32),'mapping':{2:'/World/Stem'}}
    monkeypatch.setattr(static_render,'validate_payload',lambda p,c:(p['rgb'],p['depth'],p['valid'],[0,1]))
    monkeypatch.setattr(static_render,'decode_instances',lambda p:(p['ids'],p['mapping']))
    return a


def test_identical_native_buffers_converge(buffers):
    assert static_render.convergence(buffers,deepcopy(buffers),{},roi_pixel=[424,204])['passed']


def test_renderer_ID_renumbering_is_not_a_geometry_change(buffers):
    other=deepcopy(buffers)
    other['ids'][:]=52
    other['mapping']={52:'/World/Stem',100:'/World/UnusedChangedEntry'}
    result=static_render.convergence(buffers,other,{})
    assert result['passed'] and result['visible_prim_identity_equal']
    assert not result['renderer_local_ID_buffers_equal']


def test_changed_unused_mapping_does_not_change_visible_identity(buffers):
    other=deepcopy(buffers)
    other['mapping'][100]='/World/NotInImage'
    assert static_render.convergence(buffers,other,{})['passed']


@pytest.mark.parametrize('kind',['depth','ids','mapping','rgb','target_roi'])
def test_stale_or_unsettled_buffers_do_not_converge(buffers,kind):
    b=deepcopy(buffers)
    if kind=='depth': b['depth'][204,424]+=.01
    if kind=='ids': b['ids'][204,424]=3
    if kind=='mapping': b['mapping'][2]='/World/Leaf'
    if kind=='rgb': b['rgb'][:]=110
    if kind=='target_roi': b['rgb'][185:225,410:440]=150
    assert not static_render.convergence(buffers,b,{},roi_pixel=[424,204])['passed']


def test_reference_uses_final_fresh_callback(buffers,monkeypatch):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes):
        calls.append(subframes)
        return {**buffers,'sequence':len(calls)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    payload,evidence=static_render.reference_payload(None,None,{})
    assert calls==[8]*7 and payload['sequence']==7
    assert evidence['short_profile_qualified'] is False


def test_adaptive_requires_multiple_frames_and_returns_final_reference(buffers,monkeypatch):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes):
        calls.append(subframes)
        return {**buffers,'sequence':len(calls)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    payload,evidence=static_render.settled_payload(None,None,{},verify_long_reference=True)
    assert calls==[4]*4+[8]*7 and payload['sequence']==11
    assert evidence['long_reference_comparison']['passed']
    assert np.array_equal(evidence['_short_rgb'],buffers['rgb'])


def test_adaptive_rejects_long_reference_drift(buffers,monkeypatch):
    from sim_data import capture_pilot
    def step(rep,writer,subframes):
        return {**buffers,'rgb':buffers['rgb']+(20 if subframes==8 else 0)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    with pytest.raises(ValueError,match='differs from long reference'):
        static_render.settled_payload(None,None,{},verify_long_reference=True)


def test_adaptive_bounded_failure(buffers,monkeypatch):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes):
        calls.append(subframes)
        return {**buffers,'rgb':buffers['rgb']+(20 if len(calls)%2 else 0)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    with pytest.raises(ValueError,match='bounded render budget'):
        static_render.settled_payload(None,None,{},max_steps=5)
    assert len(calls)==5


@pytest.mark.parametrize('compare',[False,True])
def test_consolidation_retains_budget_and_fresh_reference(buffers,monkeypatch,compare):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes):
        calls.append(subframes)
        return {**buffers,'sequence':len(calls)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    payload,evidence=static_render.consolidated_payload(None,None,{},compare=compare)
    assert calls==([56]+[8]*7 if compare else [56])
    assert payload['sequence']==len(calls) and evidence['rt_subframes_per_step']==56


def test_noise_probe_keeps_diagnostics_and_returns_last_reference_only(buffers,monkeypatch):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes):
        calls.append(subframes)
        return {**buffers,'rgb':buffers['rgb']+(10 if len(calls)==1 else 0),'sequence':len(calls)}
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    result,evidence=static_render.noise_probe_payload(None,None,{})
    assert calls==[56]+[8]*14 and result['sequence']==15
    assert evidence['experimental_candidate_approved'] is False
    assert not evidence['comparisons']['candidate_reference']['passed']
    assert evidence['comparisons']['reference_repeat']['passed']
    assert np.array_equal(result['rgb'],buffers['rgb'])


def test_eight_subframe_snapshot_reports_actual_budget(buffers,monkeypatch):
    from sim_data import capture_pilot
    calls=[]
    def step(rep,writer,subframes): calls.append(subframes); return buffers
    monkeypatch.setattr(capture_pilot,'step_payload',step)
    _,settings=static_render.consolidated_payload(None,None,{},subframes=8)
    assert calls==[8] and settings['rt_subframes_per_step']==8
