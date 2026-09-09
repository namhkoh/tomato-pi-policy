import json
from pathlib import Path
import threading

from PIL import Image
import pytest

from sim_data.dataset_review import HEAD_CAMERA, SCHEMA as AUDIT_SCHEMA, write_json
from sim_data.depth_preview import sha256
from sim_data.training_contract import contract_hash
from sim_data.training_release_review import SCHEMA, POLICY, record
from sim_data.training_release_review_test import row
from sim_data.training_review_gui import TrainingReviewApp, SOURCE_FILES, UI, main
from sim_data.review_gui import make_server
from sim_data.review_gui_test import request, write_headers


@pytest.fixture
def v3_bundle(tmp_path):
    capture = tmp_path/'capture'; audit_dir = tmp_path/'audit'; review = tmp_path/'review'
    audit_dir.mkdir(); review.mkdir()
    bindings = {}; checked = []; entries = []
    for i in (1, 2, 3):
        d = capture/f'sample_{i:04d}'; d.mkdir(parents=True)
        files = {}
        for name in SOURCE_FILES:
            p = d/name; p.parent.mkdir(exist_ok=True)
            if p.suffix == '.png': Image.new('RGB', (848, 408), (40, 95, 50)).save(p)
            else: p.write_bytes(b'fixture-native-depth-not-a-real-training-array')
            files[name] = {'sha256': sha256(p)}; bindings[str(p)] = sha256(p)
        write_json(d/'sample.json', {'files': files}); bindings[str(d/'sample.json')] = sha256(d/'sample.json')
        e = row(i); e.update(source_sample_sha256=sha256(d/'sample.json'), rgb_sha256=files['inputs/rgb.png']['sha256'], task_contract_sha256=contract_hash(),
            answer=dict(status='localized', cut_point_uv=[100., 200.], visibility='clear', next_action='inspect_cut_region'))
        # The metadata needs a unique binding per sample even with equal test images.
        meta = json.loads((d/'sample.json').read_text()); meta['sample_id'] = d.name
        (d/'sample.json').write_text(json.dumps(meta)); bindings[str(d/'sample.json')] = sha256(d/'sample.json'); e['source_sample_sha256'] = sha256(d/'sample.json')
        if i == 2: e.update(difficulty='hard', answer=dict(status='abstain', cut_point_uv=None, visibility='occluded', next_action='change_viewpoint'))
        Image.new('RGB', (1152,1230), (30,35,40)).save(review/(e['id']+'.png'))
        e.update(card=e['id']+'.png', card_sha256=sha256(review/(e['id']+'.png'))); entries.append(e)
        checked.append(dict(sample_id=d.name, target_review_id=e['target_id'], integrity_and_recomputed_annotations_passed=True,
            quality=dict(clear_view_gate_passed=i != 2), camera=dict(mounted_robot_pov_verified=True, camera_path=HEAD_CAMERA)))
    audit = audit_dir/'audit.json'
    write_json(audit, dict(schema_version=AUDIT_SCHEMA, state='complete_engineering_audit_not_approval', training_dataset_approved=False,
        source_run=str(capture), samples=checked, bindings_sha256=bindings, cards_sha256={}))
    for e in entries: e['source_audit_sha256'] = sha256(audit)
    bundle = review/'bundle.json'
    write_json(bundle, dict(schema_version=SCHEMA, policy=POLICY, entries=entries, source_audits_sha256={str(audit):sha256(audit)}))
    record(bundle, ['one_3'], 'Fixture assistant inspection, not real human confirmation.', inspected=True)
    return bundle


@pytest.fixture
def app(v3_bundle): return TrainingReviewApp(v3_bundle)


def payload(app, **kwargs):
    return dict(sample_id='one_1', reviewer='Fixture human', decision='accept', notes='Fixture explicitly inspected query, cut and native evidence.',
                inspected=True, bundle_sha256=app.bundle_hash, **kwargs)


def test_load_readonly_pending_and_original_bytes(app, v3_bundle):
    before = sorted(app.records.glob('*.json'))
    s = app.state(); assert s['pending'] == 2 and s['human_recorded'] == 0
    assert s['samples'][2]['decision']['reviewer_role'] == 'assistant'
    assert app.images['one_1','rgb'] == app.images.paths['one_1','rgb'][0].read_bytes()
    assert sorted(app.records.glob('*.json')) == before
    assert not (v3_bundle.parent.parent/'audit/records').exists()


def test_save_persists_human_role_and_restart_resumes(app, v3_bundle):
    r = app.save(payload(app)); assert r['saved']['reviewer_role'] == 'human'
    assert r['saved']['human_confirmation'] and not r['saved']['physical_execution_approved']
    assert not r['state']['training_release_approved'] and r['source_block_path'] is None
    assert TrainingReviewApp(v3_bundle).state()['pending'] == 1
    with pytest.raises(ValueError, match='Already recorded'): app.save(payload(app))


@pytest.mark.parametrize('decision', ['hold', 'reject'])
def test_negative_review_is_export_source_block_and_does_not_modify_native_files(app, decision):
    before = {p:sha256(p) for p in app.sources['one_1']['files']}
    p = payload(app); p['decision'] = decision
    result = app.save(p); active, _ = app._source_history('one_1')
    assert active['sample_0001','human']['decision'] == decision
    assert Path(result['source_block_path']).is_file()
    assert not active['sample_0001','human']['human_prototype_label_confirmation']
    assert all(sha256(p) == h for p,h in before.items())


def test_interrupted_task_save_keeps_source_excluded_and_retry_does_not_fork_history(app, monkeypatch):
    import sim_data.training_review_gui as module
    original = module.record
    def fail(*args, **kwargs): raise OSError('fixture interruption')
    monkeypatch.setattr(module, 'record', fail)
    p = payload(app); p['decision'] = 'hold'
    with pytest.raises(OSError): app.save(p)
    assert not (app.records/'one_1.json').exists()
    with pytest.raises(ValueError, match='Source is held'): app.save(payload(app))
    monkeypatch.setattr(module, 'record', original)
    app.save(p)
    assert len(list((app.sources['one_1']['audit_path'].parent/'records').glob('*.json'))) == 1


def test_hidden_correct_abstention_can_be_accepted_despite_pilot_clear_gate(app):
    p = payload(app); p['sample_id'] = 'one_2'
    assert app.save(p)['saved']['decision'] == 'accept'


@pytest.mark.parametrize('change', [{'reviewer':''}, {'inspected':False}, {'notes':'too short'}, {'decision':[]}, {'reviewer_role':'assistant'},
                                   {'bundle_sha256':'stale'}, {'sample_id':'../escape'}])
def test_invalid_or_stale_writes_refused(app, change):
    p = payload(app); p.update(change)
    with pytest.raises(ValueError): app.save(p)
    assert app.state()['pending'] == 2


def test_existing_assistant_record_cannot_be_replaced(app):
    p = payload(app); p['sample_id'] = 'one_3'
    with pytest.raises(ValueError, match='Already recorded'): app.save(p)


@pytest.mark.parametrize('file', ['inputs/rgb.png', 'inputs/depth_m.npy', 'supervision/target_visible.png'])
def test_changed_source_blocks_save(app, file):
    p = app.sources['one_1']['audit_path'].parent.parent/'capture/sample_0001'/file
    p.write_bytes(b'changed-fixture')
    with pytest.raises(ValueError, match='Source evidence changed'): app.save(payload(app))


def test_changed_card_cannot_be_served(app):
    app.images.paths['one_1','card'][0].write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'): app.images['one_1','card']


@pytest.fixture
def server(app):
    s = make_server(app, 0, ui=UI); t = threading.Thread(target=s.serve_forever, kwargs={'poll_interval':.01}, daemon=True); t.start()
    yield s
    s.shutdown(); s.server_close(); t.join(timeout=2)


def test_http_ui_and_explicit_save_security(server, app):
    code, headers, page = request(server, 'GET', '/')
    assert code == 200 and b'Cut-point review' in page and b'__CSRF__' not in page
    assert headers['X-Frame-Options'] == 'DENY'
    for path in ('/api/state','/ui/app.js','/ui/style.css','/image/one_1/rgb','/image/one_1/card'): assert request(server,'GET',path)[0] == 200
    for path in ('/image/../../dev.md','/image/one_1/sample.json','/ui/../training_review_gui.py'): assert request(server,'GET',path)[0] == 404
    assert request(server, 'POST', '/api/review', payload(app))[0] == 403
    assert request(server, 'POST', '/api/review', payload(app), write_headers(server, app))[0] == 200
    assert request(server, 'POST', '/api/review', payload(app), write_headers(server, app))[0] == 409


def test_reopening_existing_gui_does_not_make_reviews(server, app, monkeypatch):
    opened = []; monkeypatch.setattr('sim_data.training_review_gui.webbrowser.open', opened.append)
    main(['--bundle', str(app.bundle_path), '--port', str(server.server_port), '--open'])
    assert opened == [f'http://127.0.0.1:{server.server_port}'] and app.state()['pending'] == 2


def test_client_requires_both_views_and_no_bulk_save():
    script = (UI/'app.js').read_text(encoding='utf-8'); html = (UI/'index.html').read_text(encoding='utf-8')
    assert 'viewed.has("rgb") && viewed.has("card")' in script
    assert 'setView("rgb")' in script and 'read-only' in html and 'at least 20 characters' in html
