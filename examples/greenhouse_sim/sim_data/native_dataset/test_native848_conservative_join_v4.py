"""CPU tests for explicit short-route dispatch, provenance and source pools."""
from copy import deepcopy
from pathlib import Path
import pytest
from . import native848_conservative_join_v4 as consumer
from . import native848_conservative_join_v3 as prior
from .. import native848_original_short_admission_v1 as admission
from ..test_native848_original_short_admission_v1 import plan_fixture
from .test_native848_conservative_join_v3 import identity_fixture as prior_identity_fixture


def route_fixture():
    trial=(Path.cwd()/'unit_short_trial').resolve()
    plan=plan_fixture()
    intent=dict(plan_path=str(trial.parent/'unit_short_plan.json'),plan_sha256='a'*64)
    receipt=dict(state=consumer.REFERENCE_RECEIPT,training_approved=False,plan_sha256='a'*64,
        profile=admission.PROFILE,render_budget_subframes=8)
    owned=dict(returncode=0,method='subprocess_wait_on_owned_process',pid=123)
    launch=dict(pid=123,plan_sha256='a'*64,command=['python','-m',consumer.REFERENCE_WORKER,
        '--plan',intent['plan_path'],'--plan-sha256','a'*64,'--output',str(trial/'capture')])
    return trial,receipt,owned,launch,intent,plan


def test_only_explicit_short_producer_route():
    consumer.check_owned_route(*route_fixture())


@pytest.mark.parametrize('mutation',['fixed56_worker','plan_path','plan_hash','output','duplicate_flag',
    'owned_state','failed_exit','receipt_budget','receipt_profile','plan_schema'])
def test_route_mismatch_rejected(mutation):
    args=route_fixture();trial,receipt,owned,launch,intent,plan=args
    if mutation=='fixed56_worker':launch['command'][2]=prior.REFERENCE_WORKER
    elif mutation=='plan_path':launch['command'][4]=str(trial/'other.json')
    elif mutation=='plan_hash':launch['command'][6]='b'*64
    elif mutation=='output':launch['command'][-1]=str(trial/'other_capture')
    elif mutation=='duplicate_flag':launch['command'][3:3]=['--plan',intent['plan_path']]
    elif mutation=='owned_state':receipt['state']=prior.REFERENCE_RECEIPT
    elif mutation=='failed_exit':owned['returncode']=1
    elif mutation=='receipt_budget':receipt['render_budget_subframes']=56
    elif mutation=='receipt_profile':receipt['profile']='other'
    else:plan['schema']=prior.REFERENCE_PLAN
    with pytest.raises(ValueError):consumer.check_owned_route(*args)


def identity_fixture():
    anchor,source,rec,meta,audit,cache=prior_identity_fixture()
    meta['schema_version']=consumer.REFERENCE_SAMPLE
    meta['capture_role']=rec['capture_role']=audit['capture_role']='production'
    meta['direct_camera']=dict(cached_sample_id='pose_original046')
    rec['source_pose_id']=audit['source_pose_id']='pose_original046'
    return anchor,source,rec,meta,audit,cache


def test_short_view_uses_captured_original_stem_pool():
    args=identity_fixture();before=deepcopy(args)
    context=consumer.original_reference_identity_context(*args)
    assert context['conservative_view_cap_group']=='seed17_full/SubStem_46'
    assert context['source_row']==args[1] and args==before


@pytest.mark.parametrize('mutation',['old_sample','control_role','different_source_pose','generated_variant',
    'generated_pixels','anchor_pool','heldout','audit_ancestry'])
def test_short_source_provenance_cannot_be_relabelled(mutation):
    args=identity_fixture();anchor,source,rec,meta,audit,cache=args
    if mutation=='old_sample':meta['schema_version']=prior.REFERENCE_SAMPLE
    elif mutation=='control_role':meta['capture_role']='adaptation_control'
    elif mutation=='different_source_pose':rec['source_pose_id']='other_pose'
    elif mutation=='generated_variant':source['variant_id']='seed17_generated'
    elif mutation=='generated_pixels':meta['generated_plant_native_pixels']=1
    elif mutation=='anchor_pool':rec['source_target']=anchor['conservative_view_cap_group']
    elif mutation=='heldout':rec['split']='test'
    else:audit['anchor_reference_sha256']='e'*64
    with pytest.raises(ValueError):consumer.original_reference_identity_context(*args)


def test_old_routes_delegate_without_interpreting_short_fields(monkeypatch,tmp_path):
    result=tmp_path/'result.json';consumer.save(result,dict(state=prior.REFERENCE_STATE))
    spec=dict(result=dict(path=str(result),sha256=consumer.digest(result)))
    sentinel=(['prior_candidate'],['prior_hold'],['prior_exclusion'])
    monkeypatch.setattr(consumer.previous,'replay_admission',lambda s,p:sentinel if s is spec else None)
    assert consumer.replay_admission(spec,consumer.Pins()) is sentinel


def test_original_cap_hold_and_workspace_engines_unchanged():
    assert consumer.CAP==12
    assert consumer.shared_cap_selection is prior.shared_cap_selection
    assert consumer.filter_identities is prior.filter_identities
    assert consumer.held_aliases is prior.held_aliases
    assert consumer.replay_saved_workspace is prior.replay_saved_workspace
