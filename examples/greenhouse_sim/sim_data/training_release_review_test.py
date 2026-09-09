from copy import deepcopy
import json

import pytest

from sim_data import training_release_review as review
from sim_data.depth_preview import sha256


def row(i=1,level='easy',family='one',profile='warm56_then8'):
    return dict(id=f'{family}_{i}',split='train',source_plant_family=family,target_id=f'{family}/petiole{i}',
                difficulty=level,rgb_sha256=str(i),source_audit_sha256='audit',source_sample_sha256='sample'+str(i),
                capture_profile=profile,query_pixel_uv=[500.,200.],answer={'status':'localized'},view_signature=str(i),task_contract_sha256='fixture-contract')


def record(r,decision='accept'):
    return dict(entry=review.identity(r),reviewer_role='assistant',human_confirmation=False,
                decision=decision,physical_execution_approved=False,inspected=review.CHECKLIST,
                notes='Actually inspected RGB and aligned native evidence in the saved card.',reviewer='Codex')


def evidence(records):
    return dict(schema_version=review.SCHEMA,policy=review.POLICY,records=records)


def test_strata_and_profiles_require_explicit_records():
    rows=[row(1),row(2),row(3,'hard'),row(4,'easy','two','established')]
    assert len(review.selected_rows(rows))==4
    with pytest.raises(ValueError,match='incomplete'): review.check_evidence(evidence([]),rows)
    result=review.check_evidence(evidence([record(r) for r in rows]),rows)
    assert result['passed'] and result['inspected_unique_samples']==4


def test_same_sample_twice_cannot_fill_inspection_quota():
    rows=[row(1),row(2)]
    with pytest.raises(ValueError,match='one/easy:1<2'):
        review.check_evidence(evidence([record(rows[0]),record(rows[0])]),rows)


@pytest.mark.parametrize('kind',['answer','image','human','hold','checklist'])
def test_wrong_or_unapproved_evidence_is_fatal(kind):
    rows=[row()]; r=record(rows[0])
    if kind=='answer': r['entry']['answer']={'status':'abstain'}
    if kind=='image': r['entry']['rgb_sha256']='changed'
    if kind=='human': r['human_confirmation']=True
    if kind=='hold': r['decision']='hold'
    if kind=='checklist': r['inspected']=[]
    with pytest.raises(ValueError): review.check_evidence(evidence([r]),rows)


def test_selection_stable_under_input_order_and_prefers_target_diversity():
    rows=[row(i) for i in range(8)]
    assert review.selected_rows(rows)==review.selected_rows(list(reversed(rows)))
    assert len(review.selected_rows(rows))==2
    assert len({r['target_id'] for r in review.selected_rows(rows)})==2


def test_record_requires_actual_inspection_and_no_overwrite(tmp_path):
    card=tmp_path/'card.png'; card.write_bytes(b'fixture-card')
    entry={**review.identity(row()),'card':'card.png','card_sha256':sha256(card)}
    bundle=tmp_path/'bundle.json'
    bundle.write_text(json.dumps(dict(schema_version=review.SCHEMA,policy=review.POLICY,entries=[entry])))
    notes='Inspected the full scene, target, query and native depth evidence.'
    with pytest.raises(ValueError,match='Actual inspection'): review.record(bundle,[row()['id']],notes)
    paths=review.record(bundle,[row()['id']],notes,inspected=True)
    assert len(paths)==1
    result=review.verify_reviews([bundle],[row()]); assert result['passed']
    with pytest.raises(ValueError,match='Never overwrite'): review.record(bundle,[row()['id']],notes,inspected=True)
    card.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed inspected card'): review.verify_reviews([bundle],[row()])
