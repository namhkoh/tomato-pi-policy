from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from sim_data.dataset_review import write_json
from sim_data.depth_preview import sha256
from sim_data import active_perception_pilot as pilot


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path/'sources'; root.mkdir(); bundle = root/'bundle.json'; write_json(bundle, {})
    entries, frames, samples = {}, {}, []
    for sid, status in [('visible', 'localized'), ('hidden', 'abstain')]:
        folder = root/sid; folder.mkdir(); (folder/'inputs').mkdir()
        Image.new('RGB', (848,408), (50,100,50)).save(folder/'inputs/rgb.png')
        np.save(folder/'inputs/depth_m.npy', np.ones((408,848), np.float32), allow_pickle=False)
        Image.new('L',(848,408),255).save(folder/'inputs/depth_valid.png')
        meta = dict(supervision=dict(cut_region_proposal={'nominal':1}, plant_to_world_usd_row_vectors=np.eye(4).tolist(),
            nominal_world_m=[1,2,3]), calibration={'resolution':[848,408]}, robot_snapshot={}, synchronization={})
        write_json(folder/'sample.json', meta)
        answer = dict(status=status, cut_point_uv=[100,200] if status=='localized' else None,
                      visibility='clear' if status=='localized' else 'occluded')
        entries[sid] = dict(source_plant_family='family', target_id='family/petiole', split='train',
            rgb_sha256=sha256(folder/'inputs/rgb.png'), answer=answer, query_pixel_uv=[300,200], source_audit_sha256=sha256(bundle))
        frames[sid] = dict(files={p:sha256(p) for p in (folder/'sample.json', folder/'inputs/rgb.png', folder/'inputs/depth_m.npy', folder/'inputs/depth_valid.png')},
                           audit_path=bundle, audit={})
        samples.append(dict(id=sid, source_blocks=[], decision=None, previous_decision=None, suggestion=None))
    app = SimpleNamespace(bundle_path=bundle, bundle_hash=sha256(bundle), entries=entries, sources=frames,
        records=root/'decisions', suggestions=SimpleNamespace(bindings={}), _check_source=lambda sid: None,
        state=lambda: dict(samples=deepcopy(samples)))
    monkeypatch.setattr(pilot, 'TrainingReviewApp', lambda *args: app)
    return app, samples


def test_static_matched_pilot_has_private_xyz_but_null_hidden_output(sources, tmp_path):
    result = pilot.prepare(sources[0].bundle_path, tmp_path/'pilot', max_negatives=0)
    assert result['counts'] == {'visible':1,'occluded':1}
    assert result['dynamic_episodes_collected'] == 0 and not result['training_eligible']
    hidden = result['examples'][1]
    assert hidden['proposed_answer']['cut_point_uv'] is None
    assert hidden['proposed_answer']['target_state'] == 'unknown'
    assert hidden['private_evaluator']['nominal_world_m'] == [1,2,3]
    assert 'nominal_world_m' not in str(hidden['model_input'])
    assert all(not x['human_confirmation'] for x in result['examples'])
    assert (tmp_path/'pilot/review.html').is_file()


@pytest.mark.parametrize('change', [dict(source_blocks=[{'decision':'hold'}]),
    dict(decision={'decision':'reject'}), dict(previous_decision={'decision':'hold'}),
    dict(suggestion={'suggested_decision':'hold', 'sha256':'advice'})])
def test_any_hold_excludes_source_and_no_fabricated_pair(sources, tmp_path, change):
    app, samples = sources; samples[1].update(change)
    with pytest.raises(ValueError, match='No unheld'): pilot.prepare(app.bundle_path,tmp_path/'pilot',max_negatives=0)
    assert not (tmp_path/'pilot').exists()


def test_family_split_leak_is_rejected(sources,tmp_path):
    app,_=sources; app.entries['hidden']['split']='test'
    with pytest.raises(ValueError,match='split leakage'): pilot.prepare(app.bundle_path,tmp_path/'pilot',max_negatives=0)


def test_explicit_human_final_pass_can_resolve_current_advice(sources,tmp_path):
    app, samples = sources
    samples[1]['suggestion'] = dict(suggested_decision='hold', sha256='current')
    samples[1]['decision'] = dict(decision='accept', suggestion_context=dict(sha256='current', response='disagree'))
    assert pilot.prepare(app.bundle_path,tmp_path/'pilot',max_negatives=0)['counts']['occluded'] == 1


def test_old_human_accept_cannot_implicitly_resolve_new_advice(sources,tmp_path):
    app, samples = sources
    samples[1]['suggestion'] = dict(suggested_decision='hold', sha256='new')
    samples[1]['decision'] = dict(decision='accept', suggestion_context=dict(sha256='old', response='disagree'))
    with pytest.raises(ValueError,match='No unheld'): pilot.prepare(app.bundle_path,tmp_path/'pilot',max_negatives=0)


def test_changed_anatomy_is_not_a_matched_pair(sources,tmp_path):
    app,_=sources; p=app.bundle_path.parent/'hidden/sample.json'
    import json
    m=json.loads(p.read_text()); m['supervision']['cut_region_proposal']={'nominal':2}; p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='No unheld'): pilot.prepare(app.bundle_path,tmp_path/'pilot',max_negatives=0)


def test_new_pilot_never_overwrites_old_output(sources,tmp_path):
    output=tmp_path/'pilot'; output.mkdir()
    with pytest.raises(ValueError,match='new pilot'): pilot.prepare(sources[0].bundle_path,output,max_negatives=0)


def test_native_negative_requires_identity_and_valid_depth(tmp_path):
    d=tmp_path; (d/'inputs').mkdir(); (d/'supervision').mkdir()
    rgb=np.full((408,848,3),180,np.uint8); rgb[80:330,400:430]=[40,110,40]
    ids=np.zeros((408,848),np.uint32); ids[80:330,400:430]=7
    Image.fromarray(rgb).save(d/'inputs/rgb.png')
    Image.new('L',(848,408),255).save(d/'inputs/depth_valid.png')
    np.save(d/'inputs/depth_m.npy',np.ones((408,848),np.float32),allow_pickle=False)
    np.save(d/'supervision/component_id.npy',ids,allow_pickle=False)
    write_json(d/'supervision/identities.json',dict(component_catalogue=[dict(component_index=7,organ_type='main_stem')]))
    paths=[d/'supervision/component_id.npy',d/'supervision/identities.json']
    meta=dict(files={p.relative_to(d).as_posix():dict(sha256=sha256(p)) for p in paths})
    audit=dict(bindings_sha256={str(p):sha256(p) for p in paths})
    negative=pilot.negative_query(d,meta,audit,'main_stem')
    assert negative['usability']['passed'] and negative['component']['component_index']==7
    assert pilot.negative_query(d,meta,audit,'fruit') is None
    Image.new('L',(848,408),0).save(d/'inputs/depth_valid.png')
    assert pilot.negative_query(d,meta,audit,'main_stem') is None
    (d/'supervision/identities.json').write_text('{}')
    with pytest.raises(ValueError,match='bound native identity'): pilot.negative_query(d,meta,audit,'main_stem')
