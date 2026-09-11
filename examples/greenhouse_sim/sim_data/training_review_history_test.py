import json
import pytest
from sim_data import training_review_history as history
from sim_data.depth_preview import sha256


@pytest.fixture
def bundle(tmp_path):
    (tmp_path/'decisions').mkdir();(tmp_path/'held.png').write_bytes(b'inspected card fixture')
    entry=dict(id='held',rgb_sha256='same_rgb',card='held.png',card_sha256=sha256(tmp_path/'held.png'))
    path=tmp_path/'bundle.json';path.write_text(json.dumps(dict(schema_version=history.SCHEMAS[0],entries=[entry])))
    record=dict(schema_version=history.SCHEMAS[0],bundle_sha256=sha256(path),entry=entry,
        reviewer_role='assistant',decision='hold',human_confirmation=False,physical_execution_approved=False)
    (tmp_path/'decisions/held.json').write_text(json.dumps(record))
    return path


def test_legacy_hold_excludes_same_rgb_despite_changed_query_or_id(bundle):
    evidence=history.negative_history([bundle])
    rows=[dict(id='new_task_id',source_plant_family='seed17',rgb_sha256='same_rgb'),
          dict(id='different',source_plant_family='seed17',rgb_sha256='other_rgb')]
    kept,excluded=history.exclude_negatives(rows,evidence)
    assert kept==rows[1:] and excluded[0]['id']=='new_task_id'
    assert excluded[0]['reason']=='historical_visual_hold_or_reject_rgb'
    assert evidence['grants_approval'] is False


def test_old_accept_never_grants_current_qa(bundle):
    path=bundle.parent/'decisions/held.json';record=json.loads(path.read_text())
    record['decision']='accept';path.write_text(json.dumps(record))
    assert history.negative_history([bundle])['records']==[]


@pytest.mark.parametrize('changed',['bundle','decision','card','scope'])
def test_changed_historical_evidence_fails(bundle,changed):
    if changed=='bundle': bundle.write_text(bundle.read_text()+' ')
    elif changed=='card': (bundle.parent/'held.png').write_bytes(b'changed')
    else:
        path=bundle.parent/'decisions/held.json';record=json.loads(path.read_text())
        if changed=='decision': record['entry']['rgb_sha256']='changed'
        else: record['human_confirmation']=True
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError): history.negative_history([bundle])


def test_pinned_inventory_rejects_omitted_bundle_and_deleted_decision(bundle):
    inventory=bundle.parent/'pinned.json'
    inventory.write_text(json.dumps(history.negative_history([bundle])))
    assert history.pinned_history(inventory,[bundle])['records']
    with pytest.raises(ValueError,match='omitted'): history.pinned_history(inventory,[])
    (bundle.parent/'decisions/held.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)): history.pinned_history(inventory,[bundle])


def test_baseline_cannot_omit_inventory_or_supply_empty_one(tmp_path):
    with pytest.raises(ValueError,match='requires a pinned'): history.pinned_history(None,[])
    path=tmp_path/'empty.json';path.write_text(json.dumps(history.negative_history([])))
    with pytest.raises(ValueError,match='Empty historical'): history.pinned_history(path,[])
