import json

import numpy as np
from PIL import Image
import pytest

from sim_data import training_rescreen as rescreen
from sim_data.training_release_review_test import row
from sim_data.training_release_review import identity
from sim_data.dataset_review import read_json, write_json
from sim_data.depth_preview import sha256
from sim_data.query_visibility import QueryVisibility
from sim_data.training_contract import contract_hash


@pytest.fixture
def source(tmp_path,monkeypatch):
    d=tmp_path/'source'; (d/'inputs').mkdir(parents=True); (d/'supervision').mkdir()
    rgb=np.full((408,848,3),120,np.uint8); target=np.zeros((408,848),bool)
    target[196:209,480:550]=True; rgb[target]=(50,100,30)
    Image.fromarray(rgb).save(d/'inputs/rgb.png')
    np.save(d/'inputs/depth_m.npy',np.ones((408,848),np.float32))
    Image.fromarray(np.full((408,848),255,np.uint8)).save(d/'inputs/depth_valid.png')
    Image.fromarray(target.astype(np.uint8)*255).save(d/'supervision/target_visible.png')
    meta=dict(sample_id='sample_0001',files={'inputs/rgb.png':{'sha256':sha256(d/'inputs/rgb.png')}},
        supervision=dict(nominal_projected={'pixel_xy':[490.,200.]},
            visibility_evidence=dict(nominal={'visible_target_evidence':True},interval_probes=[])))
    write_json(d/'sample.json',meta)
    r=row(); r.update(id='one_'+meta['files']['inputs/rgb.png']['sha256'][:20],directory=d,metadata=meta,
        rgb_sha256=sha256(d/'inputs/rgb.png'),source_sample_sha256=sha256(d/'sample.json'),
        query_pixel_uv=[510.,200.],task_contract_sha256=contract_hash(),
        answer=dict(status='localized',cut_point_uv=[490.,200.],visibility='clear',next_action='inspect_cut_region'),
        label={'query_usability':QueryVisibility(rgb,target).inspect([510.,200.])})
    old=tmp_path/'previous'; old.mkdir(); Image.new('RGB',(2,2)).save(old/'old.png')
    e={**identity(r),'query_pixel_uv':[500.,200.],'task_contract_sha256':'previous-v2','card':'old.png','card_sha256':sha256(old/'old.png')}
    bundle=old/'bundle.json'
    write_json(bundle,dict(schema_version='greenhouse.grounding_stratified_visual_QA.v1',entries=[e],source_audits_sha256={}))
    gathered=dict(candidates=[r],exclusions=[],bindings={},source_plans={})
    monkeypatch.setattr(rescreen,'gather',lambda *a,**k:gathered)
    return bundle,tmp_path/'new',gathered


def test_rescreen_never_migrates_approval_or_modifies_sources(source):
    bundle,out,g=source; paths=[*bundle.parent.glob('*'),*g['candidates'][0]['directory'].rglob('*')]
    before={p:p.read_bytes() for p in paths if p.is_file()}
    result=rescreen.rescreen(bundle,out)
    assert result['surviving_prior_rows']==1 and result['changed_queries']==1
    assert not result['old_approvals_migrated'] and not (out/'decisions').exists()
    assert all(p.read_bytes()==b for p,b in before.items())
    assert read_json(out/'bundle.json')['state']=='pending_actual_visual_inspection'
    html=(out/'index.html').read_text(encoding='utf-8')
    assert '1/1 eligible' in html and html.count('<h1>')==1
    with pytest.raises(ValueError,match='new rescreen'): rescreen.rescreen(bundle,out)


def test_excluded_source_gets_diagnostic_but_no_eligible_entry(source):
    bundle,out,g=source; r=g['candidates'].pop()
    g['exclusions']=[dict(source_sample=str(r['directory']),family='one',reason='no_usable_visible_query')]
    result=rescreen.rescreen(bundle,out)
    assert result['excluded_prior_rows']==1 and not read_json(out/'bundle.json')['entries']
    c=read_json(out/'comparison.json')[0]
    assert c['current_answer'] is None and (out/c['card']).is_file()


def test_changed_original_card_fails_before_output(source):
    bundle,out,g=source; (bundle.parent/'old.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed original review'): rescreen.rescreen(bundle,out)
    assert not out.exists()
