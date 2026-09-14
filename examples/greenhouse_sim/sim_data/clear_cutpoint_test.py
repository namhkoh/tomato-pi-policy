import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from sim_data.clear_cutpoint_contract import crop_box,crop_image,screen,select_views,POLICY
from sim_data.clear_cutpoint_release import build,validate,review_ids,check_reviews
from sim_data.dataset_review import write_json
from sim_data.depth_preview import sha256
from sim_data.training_export import read_jsonl


def anatomy():
    rgb=np.full((408,848,3),100,np.uint8);mask=np.zeros((408,848),bool);mask[190:211,200:331]=True
    label=dict(accepted_interval_uv=[[220.,200.],[236.,200.]],query_pixel_uv=[300.,200.],
               proximal_visible_pixel_fraction=1.,query_cut_visible_connection_verified=True,
               answer=dict(status='localized',visibility='clear',cut_point_uv=[220.,200.],next_action='inspect_cut_region'),
               calibration=dict(camera_to_world_usd_row_vectors=np.eye(4).tolist()))
    return label,rgb,mask


def test_strict_legibility_rejects_dark_narrow_short_and_occluded():
    label,rgb,mask=anatomy();assert screen(label,rgb,mask)['passed']
    assert not screen(label,np.zeros_like(rgb),mask)['passed']
    thin=np.zeros_like(mask);thin[198:203,200:331]=True
    assert 'width_proxy_px' in screen(label,rgb,thin)['reasons']
    short=copy.deepcopy(label);short['accepted_interval_uv'][1][0]=225
    assert 'interval_length_px' in screen(short,rgb,mask)['reasons']
    mask[200,220]=False;assert not screen(label,rgb,mask)['passed']


def test_crop_is_only_query_derived_with_exact_boundary_transform():
    assert crop_box([0,0])==[0,0,384,384]
    assert crop_box([847.9,407.9])==[464,24,848,408]
    label,rgb,_=anatomy();assert crop_image(Image.fromarray(rgb),label['query_pixel_uv']).size==(768,768)
    for q in ([848,0],[-1,2],[float('nan'),0]):
        with pytest.raises(ValueError):crop_box(q)


def test_view_cap_deterministic_input_order_independent():
    rows=[dict(id=str(i),source_plant_family='a',target_id='a/t',selection_features=[i,0,1]) for i in range(30)]
    result=select_views(rows);assert len(result)==12
    assert result==select_views(list(reversed(rows)))
    assert POLICY['views_per_target']==12


@pytest.fixture
def source(tmp_path,monkeypatch):
    from sim_data import training_export
    monkeypatch.setattr(training_export,'validate',lambda *a,**k:dict(state='unit_fixture_only',release_profile='visible_occluded_v1'))
    root=tmp_path/'source';root.mkdir();files={};rows=[]
    for folder in ('images','labels','depth'): (root/folder).mkdir()
    for n,split in enumerate(('train','validation','test')):
        sid=f'id{n}';label,rgb,mask=anatomy();rgb[0,0,0]=n
        label.update(target_id=f'plant{n}/target',source_plant_family=f'plant{n}')
        paths=dict(rgb=f'images/{sid}.png',label=f'labels/{sid}.json',target_mask=f'labels/{sid}_target.png',
                   depth=f'depth/{sid}.npy',validity=f'depth/{sid}_valid.png')
        Image.fromarray(rgb).save(root/paths['rgb']);Image.fromarray(mask.astype(np.uint8)*255).save(root/paths['target_mask'])
        np.save(root/paths['depth'],np.full((408,848),.6,np.float32));Image.new('L',(848,408),255).save(root/paths['validity'])
        write_json(root/paths['label'],label)
        for p in paths.values():files[p]=sha256(root/p)
        rows.append(dict(id=sid,split=split,source_plant_family=f'plant{n}',target_id=label['target_id'],
                          difficulty='easy',answer=label['answer'],query_pixel_uv=label['query_pixel_uv'],
                          rgb_sha256=files[paths['rgb']],view_signature=sid,files=paths))
    with (root/'index.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    write_json(root/'manifest.json',dict(state='complete_visible_occluded_baseline_release',
        files_sha256=files,acceptance=dict(target_source_family_assignments={f'plant{n}':s for n,s in enumerate(('train','validation','test'))})))
    return root


def test_build_is_new_draft_copies_native_depth_and_preserves_source(source,tmp_path):
    before={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    output=tmp_path/'draft';m=build(source,output)
    assert m['state']=='draft_clear_cutpoint_not_for_training'
    assert validate(output,allow_draft=True)['rows']==3
    with pytest.raises(ValueError,match='incomplete'):validate(output)
    assert before=={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    assert (source/'depth/id0.npy').read_bytes()==(output/'depth/id0.npy').read_bytes()
    with pytest.raises(ValueError):build(source,output)
    with pytest.raises(ValueError):build(source,source/'nested')


@pytest.mark.parametrize('mutation',['crop','depth','chat','split'])
def test_tampering_rejected_even_if_own_manifest_rehashed(source,tmp_path,mutation):
    out=tmp_path/'draft';build(source,out);m=json.loads((out/'manifest.json').read_text())
    p={'crop':'crops/id0.png','depth':'depth/id0.npy','chat':'splits/train.jsonl','split':'index.jsonl'}[mutation]
    if mutation=='crop':Image.new('RGB',(768,768)).save(out/p)
    elif mutation=='depth':np.save(out/p,np.ones((408,848),np.float32))
    elif mutation=='chat':(out/p).write_text((out/p).read_text().replace('220.0','221.0'))
    else:(out/p).write_text((out/p).read_text().replace('"split": "train"','"split": "test"'))
    m['files_sha256'][p]=sha256(out/p);(out/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):validate(out,allow_draft=True)


def test_assistant_cannot_approve_heldout_or_override_hold(source,tmp_path):
    out=tmp_path/'draft';build(source,out);rows=list(read_jsonl(out/'index.jsonl'))
    records=[dict(id=r['id'],rgb_sha256=r['rgb_sha256'],reviewer='test',reviewer_type='assistant',decision='accept',reason='inspected') for r in rows]
    result=check_reviews(rows,records);assert not result['passed'] and len(result['missing'])==2
    for r in records:r['reviewer_type']='human'
    assert check_reviews(rows,records)['passed']
    records[0]['decision']='hold';assert not check_reviews(rows,records)['passed']
    with pytest.raises(ValueError):check_reviews(rows,records+[records[0]])


def test_visible_training_and_crop_inference_do_not_leak_answer(source,tmp_path):
    from sim_data.h200_train import choose_rows
    from sim_data.qwen_adapter import model_messages
    out=tmp_path/'draft';build(source,out);row=list(read_jsonl(out/'splits/train.jsonl'))[0]
    with pytest.raises(ValueError):choose_rows([row,row],'smoke')
    assert len(choose_rows([row,row],'smoke',visible_only=True))==2
    a=model_messages(row,out,coordinates='normalized_1000',decimals=2,query_crop=True)
    altered=copy.deepcopy(row);answer=json.loads(altered['messages'][-1]['content']);answer['cut_point_uv']=[240.,210.]
    altered['messages'][-1]['content']=json.dumps(answer)
    b=model_messages(altered,out,coordinates='normalized_1000',decimals=2,query_crop=True)
    assert a[1]['content'][-1]==b[1]['content'][-1]
    assert np.array_equal(np.asarray(a[1]['content'][1]['image']),np.asarray(b[1]['content'][1]['image']))
    assert len(a)==2 and len(a[1]['content'])==3
    with pytest.raises(ValueError):model_messages(row,out,query_crop=True,depth_input=True)


def test_metrics_penalize_missing_invalid_abstain_and_preserve_macros(source,tmp_path):
    from sim_data.clear_cutpoint_evaluate import metrics
    out=tmp_path/'draft';build(source,out);rows=list(read_jsonl(out/'index.jsonl'))
    predictions={rows[0]['id']:json.dumps(rows[0]['answer']),rows[1]['id']:'not json'}
    r=metrics(rows,predictions,out,'pixels')
    assert r['overall']['interval_and_target_hit']==pytest.approx(1/3)
    assert r['target_macro_interval_hit_2px']==pytest.approx(1/3)
    assert r['overall']['median_error_answered_px']==0
    with pytest.raises(ValueError):metrics(rows,{'extra':'{}'},out)


def test_review_page_no_default_accept_and_no_source_changes(source,tmp_path):
    from sim_data.clear_cutpoint_review import build as review
    out=tmp_path/'draft';build(source,out);digest=sha256(out/'manifest.json')
    result=review(out,tmp_path/'review');page=Path(result['path']).read_text()
    assert result['images']==3 and 'decisions={}' in page and 'Download decisions JSON' in page
    assert sha256(out/'manifest.json')==digest


def test_finalize_and_archive_refuse_unreviewed_draft(source,tmp_path):
    from sim_data.clear_cutpoint_release import finalize
    from sim_data.release_archive import archive
    out=tmp_path/'draft';build(source,out);reviews=tmp_path/'reviews.json';write_json(reviews,[])
    with pytest.raises(ValueError,match='incomplete'):finalize(out,tmp_path/'final',reviews)
    assert not (tmp_path/'final').exists()
    with pytest.raises(ValueError):archive(out,tmp_path/'data.zip')
    assert not (tmp_path/'data.zip').exists()


def test_macro_is_not_row_weighted(source,tmp_path):
    from sim_data.clear_cutpoint_evaluate import metrics
    out=tmp_path/'draft';build(source,out);rows=list(read_jsonl(out/'index.jsonl'))
    rows[1]['target_id']=rows[0]['target_id'];rows[1]['source_plant_family']=rows[0]['source_plant_family']
    predictions={r['id']:json.dumps(r['answer']) for r in rows[:2]}
    result=metrics(rows,predictions,out,'pixels')
    assert result['overall']['interval_hit_2px']==pytest.approx(2/3)
    assert result['target_macro_interval_hit_2px']==pytest.approx(.5)


def test_server_flags_keep_test_sealed_and_disallow_depth_crop():
    from sim_data.h200_evaluate import arguments
    from sim_data.h200_train import arguments as train_args
    base=['generate','--dataset','data','--model','model','--output','out','--split','test']
    with pytest.raises(SystemExit):arguments(base)
    assert arguments(base+['--allow-test']).allow_test
    with pytest.raises(SystemExit):train_args(['--dataset','d','--model','m','--output','o','--mode','smoke','--query-crop','--depth-input'])


def test_scoring_contract_rejects_mixed_modes(source,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sim_data.clear_cutpoint_evaluate as evaluator
    out=tmp_path/'draft';build(source,out);monkeypatch.setattr(evaluator,'validate',lambda *a:None)
    run=tmp_path/'predictions';run.mkdir();rows=list(read_jsonl(out/'index.jsonl'))
    for i in range(2):
        header=dict(kind='header',rows=1,split='train',coordinates='pixels',query_crop=bool(i),shard_index=i,shard_count=2)
        (run/f'predictions-shard-{i}.jsonl').write_text(json.dumps(header)+'\n'+json.dumps(dict(kind='prediction',id=rows[i]['id'],raw='{}'))+'\n')
    with pytest.raises(ValueError,match='Mixed'):evaluator.score_run(SimpleNamespace(dataset=out,run=run,output=tmp_path/'report.json'))


def test_transfer_preserves_draft_and_verifies_contents(source,tmp_path):
    import zipfile
    from sim_data.clear_cutpoint_transfer import package
    out=tmp_path/'draft';build(source,out)
    with pytest.raises(ValueError):package(out,tmp_path/'train.zip')
    with pytest.raises(ValueError):package(out,tmp_path/'misleading.zip',inspection_only=True)
    destination=tmp_path/'clear.inspection.zip';result=package(out,destination,inspection_only=True)
    assert not result['training_ready'] and result['sha256']==sha256(destination)
    with zipfile.ZipFile(destination) as z:
        receipt=json.loads(z.read('TRANSFER_STATUS.json'))
        assert not receipt['training_ready'] and receipt['inspection_only']
        assert 'README_TRAINING.md' in z.namelist()
        assert z.read('grounding_release/depth/id0.npy')==(source/'depth/id0.npy').read_bytes()
    with pytest.raises(ValueError):package(out,destination,inspection_only=True)


def test_old_approved_release_cannot_be_called_new_native_pool(source,tmp_path):
    with pytest.raises(ValueError,match='engineering pool'):build(source,tmp_path/'fresh',audited_source=True)
    assert not (tmp_path/'fresh').exists()


def test_fresh_pool_requires_bound_clear_capture_plan(source,tmp_path):
    m=json.loads((source/'manifest.json').read_text());m['state']='incomplete_engineering_export_do_not_claim_release'
    plan=tmp_path/'capture_plan.json';write_json(plan,dict(configuration={}))
    m['source_plans_sha256']={str(plan):sha256(plan)};(source/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Fresh clearer'):build(source,tmp_path/'fresh',audited_source=True)
