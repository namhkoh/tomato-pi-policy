import json

import pytest

from sim_data.dataset_review import write_json
from sim_data.depth_preview import sha256
from sim_data.training_review_gui import TrainingReviewApp
from sim_data.training_review_gui_test import v3_bundle, app, payload
from sim_data.training_release_review import verify_reviews


@pytest.fixture
def advice(app, tmp_path):
    folder = tmp_path/'advice'; folder.mkdir(); (folder/'findings').mkdir()
    entries = []
    for sid in ('one_1', 'one_2'):
        e = dict(app.entries[sid]); e.pop('card')
        e['binding_sha256'] = {str(p): h for p, h in app.sources[sid]['files'].items()}
        entries.append(e)
    write_json(folder/'scope.json', dict(schema='independent_advisory_visual_inspection.v1',
        bundle_sha256=app.bundle_hash, entries=entries))
    hashes = {}
    for e in entries:
        sid = e['id']; path = folder/'findings'/(sid+'.json')
        write_json(path, dict(schema='independent_advisory_visual_finding.v1', id=sid, entry=e,
            scope_sha256=sha256(folder/'scope.json'), reviewer_role='assistant', actual_visual_inspection=True,
            human_confirmation=False, training_release_approved=False, physical_execution_approved=False,
            gui_decision_written=False, source_hold_written=False, created_utc='2026-09-10T00:00:00Z',
            assessment='support' if sid == 'one_1' else 'hold_recommended',
            issue='fixture', notes='Fixture visual inspection only; never an actual human approval.'))
        hashes[sid] = sha256(path)
    write_json(folder/'assessment.json', dict(schema='independent_advisory_assessment.v1',
        bundle_sha256=app.bundle_hash, scope_sha256=sha256(folder/'scope.json'), inspected_count=2, finding_files=hashes))
    return folder


@pytest.fixture
def advised(app, advice):
    return TrainingReviewApp(app.bundle_path, advice)


def test_suggestions_do_not_count_as_approvals_or_source_blocks(advised):
    state = advised.state()
    assert state['suggestions_count'] == 2 and state['pending'] == 3
    assert state['advisory_holds'] == 1 and state['human_recorded'] == 0
    assert all(not s['source_blocks'] for s in state['samples'])
    assert len(list(advised.records.glob('*.json'))) == 1


def test_explicit_human_agreement_binds_suggestion_and_restart(advised, advice):
    suggestion = advised.suggestions.rows['one_1']
    context = dict(sha256=suggestion['sha256'], response='agree')
    result = advised.save(payload(advised, suggestion_context=context))
    assert result['saved']['suggestion_context'] == context
    assert result['saved']['human_confirmation']
    assert TrainingReviewApp(advised.bundle_path, advice).state()['human_recorded'] == 1


@pytest.mark.parametrize('context', [None, {}, {'sha256': 'stale', 'response': 'agree'},
    {'sha256': 'current', 'response': 'bad'}, {'sha256': 'current', 'response': 'disagree'}])
def test_missing_stale_inconsistent_context_refused(advised, context):
    if context and context.get('sha256') == 'current':
        context = {**context, 'sha256': advised.suggestions.rows['one_1']['sha256']}
    with pytest.raises(ValueError): advised.save(payload(advised, suggestion_context=context))
    assert advised.state()['human_recorded'] == 0


def test_human_can_disagree_with_advisory_hold_without_auto_clearing_real_hold(advised):
    p = payload(advised, suggestion_context=dict(sha256=advised.suggestions.rows['one_2']['sha256'], response='disagree'))
    p['sample_id'] = 'one_2'
    assert advised.save(p)['saved']['decision'] == 'accept'


def test_changed_suggestion_blocks_stale_browser_save(advised, advice):
    path = advice/'findings/one_1.json'
    path.write_text(path.read_text()+' ', encoding='utf-8')
    with pytest.raises(ValueError, match='Suggestion evidence changed'): advised.state()
    with pytest.raises(ValueError): TrainingReviewApp(advised.bundle_path, advice)


def test_wrong_bundle_suggestions_refused(app, advice):
    p = advice/'scope.json'; row = json.loads(p.read_text()); row['bundle_sha256'] = 'stale'
    p.write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises(ValueError, match='different bundle'): TrainingReviewApp(app.bundle_path, advice)


def test_human_followup_is_exported_and_prior_unchanged(app):
    p = payload(app); p['sample_id'] = 'one_3'
    prior = sha256(app.records/'one_3.json')
    app.save(p)
    evidence = verify_reviews([app.bundle_path], list(app.entries.values()), require_complete=False)
    assert len(evidence['records']) == 2
    assert evidence['inspected_unique_samples'] == 1
    assert sha256(app.records/'one_3.json') == prior


def test_negative_followup_blocks_export(app):
    p = payload(app); p.update(sample_id='one_3', decision='hold')
    app.save(p)
    with pytest.raises(ValueError, match='hold/reject'):
        verify_reviews([app.bundle_path], list(app.entries.values()), require_complete=False)


def test_modified_assistant_history_invalidates_followup(app):
    p = payload(app); p['sample_id'] = 'one_3'; app.save(p)
    path = app.records/'one_3.json'; path.write_text(path.read_text()+' ', encoding='utf-8')
    with pytest.raises(ValueError, match='follow-up'): app.state()


def test_legacy_task_hold_cannot_be_cleared_by_human_accept(app):
    path = app.records/'one_3.json'
    row = json.loads(path.read_text()); row['decision'] = 'hold'
    path.write_text(json.dumps(row), encoding='utf-8')
    p = payload(app); p['sample_id'] = 'one_3'
    with pytest.raises(ValueError, match='earlier task hold'): app.save(p)
    assert not (app.bundle_path.parent/'human_decisions/one_3.json').exists()


def test_client_suggestion_is_opt_in_not_auto_approval():
    from sim_data.training_review_gui import UI
    text = (UI/'app.js').read_text(encoding='utf-8')
    assert 'suggestion_context' in text and 'window.confirm' in text
    assert 'Nothing has been saved' in text and '.innerHTML' not in text
