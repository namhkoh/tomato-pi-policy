from copy import deepcopy
import io
import json

import numpy as np
from PIL import Image
import pytest

from sim_data import active_perception_pilot as pilot
from sim_data import active_perception_review as review
from sim_data.active_perception_pilot_test import sources  # Shared immutable native-source test fixture.
from sim_data.active_perception_export import export_subset
from sim_data.dataset_review import HEAD_CAMERA, read_json
from sim_data.depth_preview import sha256


@pytest.fixture
def app(sources, tmp_path, monkeypatch):
    source, states = sources
    for sid, frame in source.sources.items():
        folder = source.bundle_path.parent/sid; path = folder/'sample.json'
        m = read_json(path); m['calibration'].update(camera_path=HEAD_CAMERA,
            depth_convention='optical_axis_z_metres_not_ray_range',clipping_range_m=[.01,100.])
        (folder/'supervision').mkdir()
        Image.new('L',(848,408),255).save(folder/'supervision/target_visible.png')
        path.write_text(json.dumps(m),encoding='utf-8')
        frame['files'] = {p:sha256(p) for p in (*frame['files'],folder/'supervision/target_visible.png')}
    def check(sid):
        for p,h in source.sources[sid]['files'].items():
            if sha256(p)!=h: raise ValueError('Changed native source')
    source._check_source=check
    pilot.prepare(source.bundle_path,tmp_path/'pilot',max_negatives=0)
    monkeypatch.setattr(review,'TrainingReviewApp',lambda *a:source)
    result=review.ActiveReview(tmp_path/'pilot/pilot.json')
    result.test_source_states=states
    return result


def record(app, sid='visible__visible', role='human', decision='accept'):
    return app.record(sid,decision,'Fixture-only inspection of RGB, scope, mask and native depth.',
        role=role, reviewer='test fixture', inspected=True, expected_hash=app.pilot_hash)


def test_original_image_is_byte_identical_and_depth_is_only_coloured(app):
    assert app.state()['total']==2 and app.state()['human_reviewed']==0
    sid='visible__visible'; rgb_path=app.entries[sid]['model_input']['rgb_path']
    from pathlib import Path
    assert app.images[sid,'rgb']==Path(rgb_path).read_bytes()
    z_path=Path(app.entries[sid]['native_sidecars']['depth_path']); before=sha256(z_path)
    assert Image.open(io.BytesIO(app.images[sid,'depth'])).width==848
    assert Image.open(io.BytesIO(app.images[sid,'mask'])).size==(848,408)
    assert sha256(z_path)==before


def test_assistant_advice_never_creates_a_human_review(app):
    record(app,role='assistant')
    assert app.state()['assistant_reviewed']==1 and app.state()['human_reviewed']==0
    record(app)
    assert app.state()['human_reviewed']==1
    assert not app.state()['training_release_approved']
    assert not list(app.source_app.records.glob('*.json'))
    with pytest.raises(ValueError,match='overwrite'): record(app)


@pytest.mark.parametrize('change', [dict(source_blocks=[dict(decision='hold',notes='new source hold')]),
    dict(decision=dict(decision='reject',notes='new human rejection')),
    dict(previous_decision=dict(decision='hold',notes='legacy hold')),
    dict(suggestion=dict(suggested_decision='hold',sha256='new'))])
def test_new_source_holds_cannot_be_bypassed(app,change):
    app.test_source_states[0].update(change)
    with pytest.raises(ValueError,match='Source is held'): record(app)
    record(app,decision='hold')


def test_http_save_cannot_select_assistant_role_or_skip_inspection(app):
    payload=dict(sample_id='visible__visible',decision='accept',notes='Fixture-only review note of sufficient length.',
        reviewer='fixture',inspected=True,pilot_sha256=app.pilot_hash)
    with pytest.raises(ValueError,match='fields'): app.save(dict(payload,reviewer_role='assistant'))
    with pytest.raises(ValueError,match='inspection'): app.save(dict(payload,inspected=False))
    with pytest.raises(ValueError,match='Stale'): app.save(dict(payload,pilot_sha256='old'))
    assert app.state()['human_reviewed']==0
    assert app.save(payload)['saved']['human_confirmation'] is True


def test_changed_source_or_pilot_cannot_be_reviewed(app):
    source=next(iter(app.source_app.sources['visible']['files']))
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed|Changed'): record(app)


@pytest.mark.parametrize('corrupt', ['scope','head','split','cut','absence'])
def test_invalid_manifest_is_rejected(app,corrupt):
    p=deepcopy(app.pilot); e=p['examples'][0]
    if corrupt=='scope': e['model_input']['query_pixel_uv']=[1,2]
    if corrupt=='head': e['native_sidecars']['calibration']['camera_path']='/cinematic'
    if corrupt=='split': e['split']='test'
    if corrupt=='cut': e['proposed_answer']['cut_point_uv']=[1,2]
    if corrupt=='absence': e['kind']='no_target'
    app.path.write_text(json.dumps(p),encoding='utf-8')
    with pytest.raises(ValueError): review.ActiveReview(app.path)


def test_export_excludes_assistant_only_unreviewed_and_held(app,tmp_path):
    record(app,role='assistant'); output=tmp_path/'export'
    with pytest.raises(ValueError,match='No explicitly human'): export_subset(app.path,output)
    assert not output.exists()
    record(app)
    report=export_subset(app.path,output)
    assert report['count']==1 and report['splits']==dict(train=1,validation=0,test=0)
    assert report['training_release_approved'] is False
    row=json.loads((output/'train.jsonl').read_text(encoding='utf-8'))
    model_input=json.dumps(row['messages'][:2])
    for forbidden in ('private_evaluator','nominal_world','calibration','observed_organ','source_id'):
        assert forbidden not in model_input
    original=app.entries['visible__visible']['native_sidecars']['depth_path']
    assert sha256(output/'sidecars/visible__visible/depth_m.npy')==sha256(original)
    assert (output/row['messages'][1]['content'][0]['image']).is_file()
    with pytest.raises(ValueError,match='new export'): export_subset(app.path,output)
    app.test_source_states[0]['source_blocks']=[dict(decision='hold',notes='later source hold')]
    with pytest.raises(ValueError,match='No explicitly human'): export_subset(app.path,tmp_path/'blocked')


def test_null_cut_is_preserved_in_export(app,tmp_path):
    record(app,sid='hidden__occluded')
    report=export_subset(app.path,tmp_path/'export')
    assert report['kinds']=={'occluded':1}
    row=json.loads((tmp_path/'export/train.jsonl').read_text(encoding='utf-8'))
    answer=json.loads(row['messages'][2]['content'])
    assert answer['cut_point_uv'] is None and answer['target_state']=='unknown'


def test_svg_visibility_uses_attribute_not_an_ineffective_svg_property():
    script=(review.UI/'app.js').read_text(encoding='utf-8')
    assert "$('overlay').toggleAttribute('hidden'" in script
    assert "$('overlay').hidden=" not in script


def test_http_surface_is_local_and_requires_explicit_human_confirmation(app):
    import threading
    from sim_data.review_gui import make_server
    from sim_data.review_gui_test import request, write_headers
    with make_server(app,0,ui=review.UI) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            status,headers,body=request(server,'GET','/')
            assert status==200 and b'Task v4' in body
            assert b'__CSRF__' not in body
            assert request(server,'GET','/image/visible__visible/rgb')[0]==200
            assert request(server,'GET','/image/visible__visible/../../pilot.json')[0]==404
            payload=dict(sample_id='visible__visible',decision='accept',notes='Fixture-only observed native image evidence.',
                reviewer='fixture',inspected=True,pilot_sha256=app.pilot_hash)
            assert request(server,'POST','/api/review',payload)[0]==403
            assert request(server,'POST','/api/review',payload,write_headers(server,app))[0]==200
            assert request(server,'POST','/api/review',payload,write_headers(server,app))[0]==409
        finally:
            server.shutdown();thread.join(timeout=5)
