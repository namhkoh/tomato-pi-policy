"""Explicit old-code requalification cannot silently weaken ordinary review."""
import pytest
from sim_data.automated_native_review import annotation_code_proof,run
from sim_data.depth_preview import sha256


def fixture(tmp_path):
    root=tmp_path/'code';root.mkdir()
    labels=root/'native_clear_labels.py';labels.write_text('old test code')
    query=root/'native_query_visibility.py';query.write_text('query test code')
    prior={str(labels):sha256(labels),str(query):sha256(query)}
    return root,labels,prior


def test_ordinary_review_still_rejects_changed_implementation(tmp_path):
    root,labels,prior=fixture(tmp_path);labels.write_text('changed test code')
    with pytest.raises(ValueError):annotation_code_proof(prior,False,root)


def test_explicit_mode_records_both_code_versions_without_rebinding(tmp_path):
    root,labels,prior=fixture(tmp_path);before=dict(prior);labels.write_text('changed test code')
    proof=annotation_code_proof(prior,True,root)
    assert prior==before and proof['prior_implementation_sha256']==before
    assert proof['current_implementation_sha256'][str(labels)]==sha256(labels)
    assert proof['changed_prior_files']==[str(labels)]
    assert proof['explicit_requalification'] and not proof['old_bindings_or_decisions_rewritten']


@pytest.mark.parametrize('fault',['missing_deriver','outside','wrong_name','bad_hash'])
def test_explicit_mode_rejects_unbounded_or_missing_code_evidence(tmp_path,fault):
    root,labels,prior=fixture(tmp_path)
    if fault=='missing_deriver':prior.pop(str(labels))
    elif fault=='outside':prior[str(tmp_path/'native_clear_other.py')]='0'*64
    elif fault=='wrong_name':prior[str(root/'arbitrary.py')]='0'*64
    else:prior[str(labels)]='bad'
    with pytest.raises(ValueError):annotation_code_proof(prior,True,root)


def test_requalification_does_not_make_missing_native_evidence_acceptable(tmp_path):
    result=run([tmp_path/'missing'],tmp_path/'new.json',requalify=True)
    assert result['explicit_annotation_requalification']
    assert result['integrity_held_pairs']==1 and result['production_training_approved_count']==0
