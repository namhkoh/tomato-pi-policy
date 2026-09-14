from copy import deepcopy
import pytest
from .retention_policy import evaluate
from .retention_preflight_test import case
from .retention_preflight import assess


def over_budget():
    record,kw=case();kw['local_coms'][1][0]=.8
    result=assess(record,**kw)
    assert not result['prerequisite_passed']
    return result


def test_default_preserves_static_refusal_and_all_input_evidence():
    value=over_budget();before=deepcopy(value)
    result=evaluate(value)
    assert not result['continue_to_independent_cut_planning']
    assert not result['static_failure_is_diagnostic_only'] and value==before


def test_explicit_native_experiment_grants_no_cut_or_retention_credit():
    value=over_budget();before=deepcopy(value)
    result=evaluate(value,native_trial=True)
    assert result['continue_to_independent_cut_planning']
    assert result['static_failure_is_diagnostic_only'] and not result['static_prerequisite_passed']
    for key in ('retention_verified','cut_authorized','contact_limits_changed','slip_limit_changed','production_qualified'):
        assert result[key] is False
    assert result['per_step_native_guards_required'] and value==before


@pytest.mark.parametrize('mode',[False,True])
def test_static_pass_keeps_same_prerequisite(mode):
    record,kw=case();out=evaluate(assess(record,**kw),native_trial=mode)
    assert out['continue_to_independent_cut_planning'] and out['static_prerequisite_passed']
    assert not out['static_failure_is_diagnostic_only'] and not out['retention_verified']


@pytest.mark.parametrize('bad',['binding','model','prerequisite','capacity','balanced','status','caps','within','nan','util','bool','rejection'])
def test_native_trial_cannot_bypass_invalid_or_unbalanced_evidence(bad):
    value=over_budget()
    if bad=='binding':value['native_sample_binding_checked']=False
    elif bad=='model':value['model']='unknown'
    elif bad=='prerequisite':value['prerequisite_passed']=0
    elif bad=='capacity':value['capacity']=None
    elif bad=='balanced':value['capacity']['balance_found']=False
    elif bad=='status':value['capacity']['solver_status']=1
    elif bad=='caps':value['capacity']['contact_budgets_n']=[1.,1.]
    elif bad=='within':value['capacity']['within_contact_budgets']=True
    elif bad in ('nan','util','bool'):value['capacity']['minimum_worst_finger_utilization']={'nan':float('nan'),'util':.5,'bool':True}[bad]
    else:value['rejection']='no_compressive_patch_on_both_fingers'
    with pytest.raises(ValueError):evaluate(value,native_trial=True)


def test_profile_is_opt_in_and_rejects_incomplete_execution_before_output(tmp_path):
    from .benchmark import main,parser
    from .ground_truth_trial import main as public
    assert not parser().parse_args(['--output','unused']).native_retention_trial
    out=tmp_path/'none'
    with pytest.raises(ValueError,match='Native retention experiment'):
        main(['--output',str(out),'--native-retention-trial'])
    for flags in (['--mode','right_only'],['--watch'],['--screen-settled-waiting'],['--screen-approach-start']):
        with pytest.raises(SystemExit):public(['--output',str(out),'--native-retention-trial',
            '--process-zone-trial','--through-stroke-trial','--milestone','cut_action',*flags])
    assert not out.exists()
