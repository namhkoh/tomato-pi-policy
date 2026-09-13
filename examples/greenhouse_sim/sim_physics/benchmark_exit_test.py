from copy import deepcopy
from types import SimpleNamespace as S
import pytest
from .benchmark import report_exit_code
from .cut_action import assess


def result(right_only):
    mode='right_only' if right_only else 'bimanual'
    action=assess(cut_only=right_only,grasp_verified=not right_only,planned=True,
        cut_time=1.,fault=None,records=[dict(t=3.,cut=True,native_guards_passed=True)],
        cut_event=dict(event='blade_contact_joint_release',evidence=dict(cut_strategy=mode,
            stable_left_grasp=not right_only,grasp_slip_m=None if right_only else .001,
            cut_only_ready=right_only)))
    return dict(state='passed_cut_action_not_complete_robot_task',error=None,
        source_assets_unchanged=True,cut_strategy=mode,left_grasp_verified=not right_only,
        cut_action=action,gates=dict(bounded=True,collision_clear_right_plan=True,
            blade_contact_release=True,right_withdrawal_completed=False))


@pytest.mark.parametrize('right_only',[False,True])
def test_limited_action_exit_matches_requested_mode_not_full_withdrawal(right_only):
    args=S(cut_action_trial=True,right_only_cut_trial=right_only)
    r=result(right_only)
    before=deepcopy(r)
    assert report_exit_code(r,args)==0
    assert r==before and not r['gates']['right_withdrawal_completed']
    args.cut_action_trial=False
    assert report_exit_code(r,args)==2


@pytest.mark.parametrize('failure',['fault','source','mode','grasp','state','empty','missing',
                                   'extra','guard','truthy','schema','claim','plan'])
def test_limited_exit_rejects_unverified_or_mismatched_evidence(failure):
    r=result(False);args=S(cut_action_trial=True,right_only_cut_trial=False)
    if failure=='fault':r['error']='guard fault'
    elif failure=='source':r['source_assets_unchanged']=False
    elif failure=='mode':r['cut_action']['strategy']='right_only'
    elif failure=='grasp':r['left_grasp_verified']=False
    elif failure=='state':r['state']='passed_bimanual_mechanism_not_robot_task'
    elif failure=='empty':r['cut_action']['gates']={}
    elif failure=='missing':r['cut_action']['gates'].pop('native_guards')
    elif failure=='extra':r['cut_action']['gates']['unrecognized']=True
    elif failure=='guard':r['cut_action']['gates']['native_guards']=False
    elif failure=='truthy':r['cut_action']['gates']['native_guards']=1
    elif failure=='schema':r['cut_action']['model']='unknown'
    elif failure=='claim':r['cut_action']['passed']=False
    else:r['gates']['collision_clear_right_plan']=False
    assert report_exit_code(r,args)==2


def test_cut_only_full_sequence_exit_requires_all_its_distinct_gates():
    r=result(True);r['state']='passed_cut_only_mechanism_not_robot_task'
    r['gates']=dict.fromkeys(['bounded','completed','collision_clear_right_plan','blade_contact_release',
        'released_material_separates','right_withdrawal_completed','left_parked_open_unloaded'],True)
    args=S(cut_action_trial=False,right_only_cut_trial=True)
    assert report_exit_code(r,args)==0
    for key in r['gates']:
        bad=deepcopy(r);bad['gates'][key]=False
        assert report_exit_code(bad,args)==2
    args.right_only_cut_trial=False
    assert report_exit_code(r,args)==2


@pytest.mark.parametrize('state',['passed_mechanism_qualification_not_robot_task',
    'interactive_demo_not_qualification','passed_gripper_mechanism_not_robot_task',
    'passed_bimanual_mechanism_not_robot_task','interactive_full_robot_diagnostic',
    'scene_ablation_diagnostic_not_qualification'])
def test_legacy_exit_states_unchanged(state):
    assert report_exit_code(dict(state=state),S())==0
    assert report_exit_code(dict(state=state),S(right_only_cut_trial=True))==2


@pytest.mark.parametrize('state',['error','failed_bimanual_qualification','failed_cut_only_qualification',
                                  'failed_cut_action','initializing',None])
def test_failed_or_incomplete_run_is_not_success(state):
    assert report_exit_code(dict(state=state),S())==2
