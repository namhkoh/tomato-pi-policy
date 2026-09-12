import numpy as np
import pytest
from .knife import ShearGate,ShearParameters,BRITTLE_CUT_MODEL
from .knife_test import sample


@pytest.mark.parametrize('extra,reason',[
    ({'held':False},'stable_grasp'),({'slip':.004},'grasp_slip'),
    ({'impulses':[[.0001,0,0]]},'signed_load'),
    ({'tool_contact_upper_bound_n':.6},'full_load_cap'),
    ({'points':[[0,0,.004]]},'axial_contact'),
    ({'edge_contact_verified':False},'edge_provenance'),
    ({'axis':np.array([1.,0,0])},'stroke_alignment')])
def test_rejection_explained_without_changing_no_cut(extra,reason):
    g=ShearGate('test',ShearParameters(model=BRITTLE_CUT_MODEL))
    for i in range(20):
        assert sample(g,i*.0001,**extra) is None
        assert reason in g.diagnostic['failed_conditions']
        assert not g.diagnostic['cut_authorized']
    assert not g.completed and g.dwell==0


def test_reverse_step_report_uses_actual_relative_motion():
    g=ShearGate('test');sample(g,0.)
    assert sample(g,-.00001) is None
    assert g.diagnostic['relative_step_m']==pytest.approx(-.00001)
    assert 'nonreversing_relative_step' in g.diagnostic['failed_conditions']


def test_dwell_and_emission_remain_separate_from_physical_claim():
    g=ShearGate('test',ShearParameters(model=BRITTLE_CUT_MODEL));event=None
    for _ in range(5):event=sample(g,0.)
    assert event is not None and g.completed
    assert g.diagnostic['state']=='evidence_emitted_not_physical_fracture_verification'
    assert g.diagnostic['dwell_met'] and not g.diagnostic['cut_authorized']
    assert not event['tissue_fracture_calibrated']


def test_input_fault_cannot_leave_previous_qualifying_receipt():
    g=ShearGate('test');sample(g,0.)
    with pytest.raises(ValueError):sample(g,0.,impulse_contract='wrong')
    assert g.diagnostic==dict(state='input_validation_pending',cut_authorized=False)
