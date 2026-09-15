"""Pure reference-selection tests; no native render or geometry qualification."""
from copy import deepcopy
import pytest
from . import prepare as module


@pytest.fixture
def inputs(monkeypatch):
    entries=[]
    for index in range(4):
        entries.append(dict(id=f'ref{index}',source_family='seed7_full',
            source_collection_plan='original.json',source_collection_plan_sha256='a'*64,
            compatibility_group='group_a',split='train',target_id=f'seed7_full/SubStem_{index}',
            source_row={'component_id':f'SubStem_{index}'},source_sample=f'sample_{index}',
            source_capture='original_capture',evidence={'rank':index}))
    bank={'entries':entries}
    job=dict(source_family='seed7_full',source_collection_plan='original.json',
        source_collection_plan_sha256='a'*64,compatibility_group='group_a',split='train',
        reviewed_target_ids=[e['target_id'] for e in entries])
    attempt=dict(entries[2],qualification_output='fresh_exact_anchor')
    monkeypatch.setattr(module.rb,'select_views',lambda b,g,t,limit:deepcopy(
        [e for e in b['entries'] if e['compatibility_group']==g and e['target_id']==t]))
    monkeypatch.setattr(module.rb,'rank_key',lambda e:(e['evidence']['rank'],))
    return bank,job,attempt


def test_exact_anchor_first_then_ranked_targets(inputs):
    bank,job,attempt=inputs
    before=deepcopy(inputs)
    result=module.choose_references(bank,job,attempt,810000,3)
    assert [e['id'] for e in result]==['ref2','ref0','ref1']
    assert inputs==before


def test_can_limit_to_one_anchor(inputs):
    assert [e['id'] for e in module.choose_references(*inputs,123,1)]==['ref2']


def test_returned_references_are_detached(inputs):
    before=deepcopy(inputs)
    result=module.choose_references(*inputs,123,2)
    result[0]['source_row']['component_id']='changed'
    result[1]['evidence']['rank']=999
    assert inputs==before


@pytest.mark.parametrize('key,value',[
    ('source_family','seed41_full'),('source_collection_plan','renamed.json'),
    ('source_collection_plan_sha256','b'*64),('compatibility_group','other'),('split','test'),
    ('source_sample','changed'),('id','missing')])
def test_reject_changed_anchor(inputs,key,value):
    bank,job,attempt=inputs;attempt[key]=value
    with pytest.raises(ValueError):module.choose_references(bank,job,attempt,123)


def test_no_unreviewed_anchor(inputs):
    bank,job,attempt=inputs;job['reviewed_target_ids'].remove(attempt['target_id'])
    with pytest.raises(ValueError):module.choose_references(bank,job,attempt,123)


def test_missing_group_target_does_not_use_raw_fallback(inputs):
    bank,job,attempt=inputs;bank['entries'][0]['compatibility_group']='other'
    with pytest.raises(ValueError):module.choose_references(bank,job,attempt,123)


def test_wrong_plan_target_not_transplanted(inputs):
    bank,job,attempt=inputs;bank['entries'][0]['source_collection_plan']='another.json'
    with pytest.raises(ValueError):module.choose_references(bank,job,attempt,123)


@pytest.mark.parametrize('seed',[-1,2**32,True,1.5])
def test_invalid_seed(inputs,seed):
    with pytest.raises(ValueError):module.choose_references(*inputs,seed)


@pytest.mark.parametrize('maximum',[0,13,True,2.5])
def test_invalid_bound(inputs,maximum):
    with pytest.raises(ValueError):module.choose_references(*inputs,123,maximum)


def test_diversified_pose_is_same_target_and_group(inputs,monkeypatch):
    bank,job,attempt=inputs
    other=dict(bank['entries'][0],id='ref0_other_pose',source_sample='another_sample')
    bank['entries'].append(other)
    monkeypatch.setattr(module.rb,'select_views',lambda b,g,t,limit:deepcopy(
        [e for e in b['entries'] if e['compatibility_group']==g and e['target_id']==t]))
    result=module.choose_references(bank,job,attempt,123,2)
    assert [e['id'] for e in result]==['ref2','ref0_other_pose']
