from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from sim_data import training_export as export
from sim_data.dataset_review import SCHEMA
from sim_data.depth_preview import sha256
from sim_data.training_contract import CONTRACT, contract_hash
from sim_data.query_visibility import QueryVisibility


def row(family='one',split='train',key='a',status='localized'):
    return dict(id=key,split=split,source_plant_family=family,target_id=family+'/Petiole',rgb_sha256=key,
                difficulty='easy' if status=='localized' else 'hard',query_pixel_uv=[510.,204.],
                answer=dict(status=status,cut_point_uv=[434.,204.] if status=='localized' else None,
                    visibility='clear' if status=='localized' else 'occluded',
                    next_action='inspect_cut_region' if status=='localized' else 'change_viewpoint'))


def test_small_dataset_cannot_pass_release():
    result=export.check_release_rows([row()])
    assert not result['passed'] and 'train_rows:1<10000' in result['failures']
    assert result['query_copy_baseline_median_error_px']==76.


@pytest.mark.parametrize('kind',['family','image'])
def test_split_leakage_is_fatal(kind):
    rows=[row(),row('two','test','b')]
    if kind=='family': rows[1]['source_plant_family']='one'
    else: rows[1]['rgb_sha256']='a'
    with pytest.raises(ValueError,match='leaks'): export.check_release_rows(rows)


def test_duplicate_images_within_one_split_do_not_inflate_counts():
    rows=[row(),row()]
    with pytest.raises(ValueError,match='Duplicate RGB'): export.check_release_rows(rows)


@pytest.fixture
def source_audit(tmp_path,monkeypatch):
    root=tmp_path/'source'; capture=root/'capture'; sample=capture/'sample_0001'; audit_dir=root/'audit'
    (sample/'inputs').mkdir(parents=True); (sample/'supervision').mkdir(); audit_dir.mkdir()
    rgb=np.full((408,848,3),120,np.uint8)
    mask=np.zeros((408,848),bool); mask[201:209,424:625]=True
    rgb[mask]=(50,100,30)
    Image.fromarray(rgb).save(sample/'inputs/rgb.png')
    np.save(sample/'inputs/depth_m.npy',np.full((408,848),1.,np.float32))
    Image.fromarray(np.full((408,848),255,np.uint8)).save(sample/'inputs/depth_valid.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(sample/'supervision/target_visible.png')
    files={p.relative_to(sample).as_posix():{'sha256':sha256(p)} for p in sample.rglob('*') if p.is_file()}
    metadata=dict(files=files,calibration=dict(clipping_range_m=[.04,10],intrinsics=np.eye(3).tolist(),camera_to_world_usd_row_vectors=np.eye(4).tolist()),robot_snapshot={},
                  supervision=dict(projected_interval=[{'pixel_xy':[434,204]},{'pixel_xy':[444,204]}],interval_world_m=[[0,0,0],[.01,0,0]]))
    (sample/'sample.json').write_text(json.dumps(metadata))
    plan=dict(schema_version='greenhouse.grounding_collection_plan.v1',family_assignments={'one':'train'},
              jobs=[dict(job_id='job_001',plant_family='one',split='train',source_manifest_path='unused_fixture')])
    plan_path=root/'plan.json'; plan_path.write_text(json.dumps(plan))
    manifest=dict(state='pilot_ready_for_review',source_assets_unchanged=True,package=str(tmp_path/'package'),
                  source_collection_plan_path=str(plan_path),source_collection_plan_sha256=sha256(plan_path),collection_job_id='job_001')
    (capture/'manifest.json').write_text(json.dumps(manifest))
    bindings={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
    audit=dict(schema_version=SCHEMA,state='complete_engineering_audit_not_approval',training_dataset_approved=False,
               source_run=str(capture),bindings_sha256=bindings,cards_sha256={},samples=[
                   dict(sample_id='sample_0001',integrity_and_recomputed_annotations_passed=True,camera={'mounted_robot_pov_verified':True})])
    audit_path=audit_dir/'audit.json'; audit_path.write_text(json.dumps(audit))
    result=dict(returncode=0,timed_out=False,audit_sha256=sha256(audit_path),state='audited_prototype_pending_visual_review')
    (root/'result.json').write_text(json.dumps(result))
    label=dict(eligible=True,task_id=CONTRACT['task_id'],contract_sha256=contract_hash(),
               target_id='one/Petiole',difficulty='easy',query_pixel_uv=[510.,204.],answer=row()['answer'],
               query_usability=QueryVisibility(rgb,mask).inspect([510.,204.]))
    monkeypatch.setattr(export,'audit_manifest',lambda p:{'fixture':True})
    monkeypatch.setattr(export,'derive_label',lambda *a:deepcopy(label))
    return audit_path,tmp_path/'release'


def test_incomplete_default_export_refuses_without_writing(source_audit):
    path,output=source_audit
    with pytest.raises(ValueError,match='Release coverage'): export.build([path],output)
    assert not output.exists()


def test_duplicate_audit_cannot_resurrect_a_held_rgb(source_audit,monkeypatch):
    path,_=source_audit
    other=path.parent.parent/'duplicate_audit'/'audit.json'
    other.parent.mkdir();shutil.copyfile(path,other)
    calls=[]
    def reviews(*args):
        calls.append(1)
        return {('sample_0001','assistant'):{'decision':'hold'}} if len(calls)==1 else {}
    monkeypatch.setattr(export,'active_reviews',reviews)
    def should_not_derive(*args):
        raise AssertionError('Held RGB must be excluded before deriving a label')
    monkeypatch.setattr(export,'derive_label',should_not_derive)
    with pytest.raises(ValueError,match='No eligible synthetic labels'):
        export.gather([path,other])


def test_counts_alone_cannot_complete_a_release(source_audit,monkeypatch):
    path,output=source_audit
    monkeypatch.setattr(export,'check_release_rows',lambda *a,**k:dict(passed=True,failures=[]))
    with pytest.raises(ValueError,match='Stratified visual QA incomplete'): export.build([path],output)
    assert not output.exists()


def test_portable_incomplete_export_is_lossless_and_not_misrepresented(source_audit):
    path,output=source_audit
    result=export.build([path],output,allow_incomplete=True)
    assert result['state']=='incomplete_engineering_export_do_not_claim_release'
    assert not result['acceptance']['passed']
    assert export.validate(output,allow_incomplete=True)['rows']==1
    with pytest.raises(ValueError,match='Incomplete release'): export.validate(output)
    # The loader must not depend on original source paths or the original release root.
    moved=output.parent/'portable_copy'; shutil.copytree(output,moved)
    assert export.validate(moved,allow_incomplete=True)['portable_loader_verified']
    with pytest.raises(ValueError,match='never overwrite'): export.build([path],output,allow_incomplete=True)


@pytest.mark.parametrize('kind',['nonzero','timeout','audit_hash','source_changed'])
def test_incomplete_or_changed_capture_never_exports(source_audit,kind):
    path,output=source_audit
    result_path=path.parent.parent/'result.json'; r=json.loads(result_path.read_text())
    if kind=='nonzero': r['returncode']=1
    if kind=='timeout': r['timed_out']=True
    if kind=='audit_hash': r['audit_sha256']='changed'
    if kind=='source_changed': (path.parent.parent/'capture/manifest.json').write_text('{}')
    result_path.write_text(json.dumps(r))
    with pytest.raises((ValueError,KeyError)): export.build([path],output,allow_incomplete=True)
    assert not output.exists()


@pytest.mark.parametrize('file',['splits/train.jsonl','index.jsonl','contract.json'])
def test_portable_loader_rejects_modified_training_artifacts(source_audit,file):
    path,output=source_audit
    export.build([path],output,allow_incomplete=True)
    with (output/file).open('a') as f: f.write(' ')
    with pytest.raises(ValueError,match='Changed release artifact'): export.validate(output,allow_incomplete=True)


def test_identical_scene_camera_has_same_signature_despite_numeric_roundoff():
    a={'calibration':{'intrinsics':np.eye(3).tolist(),'camera_to_world_usd_row_vectors':np.eye(4).tolist()}}
    b=deepcopy(a); b['calibration']['camera_to_world_usd_row_vectors'][3][0]+=1e-12
    assert export.view_signature(a,'one')==export.view_signature(b,'one')
    assert export.view_signature(a,'one')!=export.view_signature(a,'two')
    b['calibration']['camera_to_world_usd_row_vectors'][3][0]+=.01
    assert export.view_signature(a,'one')!=export.view_signature(b,'one')


def test_new_review_during_copy_prevents_any_release_manifest(source_audit,monkeypatch):
    audit,output=source_audit;original=export.shutil.copyfile;injected=False
    def copy_and_append_review(source,destination):
        nonlocal injected
        result=original(source,destination)
        if not injected:
            folder=audit.parent/'records';folder.mkdir(exist_ok=True)
            (folder/'new_hold.json').write_text('{}')
            injected=True
        return result
    monkeypatch.setattr(export.shutil,'copyfile',copy_and_append_review)
    with pytest.raises(ValueError,match='Review history changed'):
        export.build([audit],output,allow_incomplete=True)
    assert not (output/'manifest.json').exists()


def test_review_snapshot_detects_new_file_and_preserves_existing_record(tmp_path):
    folder=tmp_path/'reviews';snapshots={str(folder):export.review_directory_snapshot(folder)}
    export.verify_review_directories(snapshots)
    folder.mkdir();p=folder/'hold.json';p.write_text('{}')
    with pytest.raises(ValueError,match='Review history changed'): export.verify_review_directories(snapshots)
    snapshots={str(folder):export.review_directory_snapshot(folder)}
    p.write_text('{"decision":"hold"}')
    with pytest.raises(ValueError,match='Review history changed'): export.verify_review_directories(snapshots)


def test_baseline_selection_does_not_relabel_or_relax_balanced_gates():
    rows=[row(key='visible'),row(key='hidden',status='abstain'),row(key='partial')]
    rows[-1]['difficulty']='medium';rows[-1]['answer']['visibility']='partial'
    before=deepcopy(rows)
    selected,excluded=export.select_profile(rows,'visible_occluded_v1')
    assert [r['id'] for r in selected]==['visible','hidden']
    assert excluded[0]['original_difficulty']=='medium'
    assert rows==before
    assert export.RELEASE_GATES['minimum_difficulty_rows_per_split']=={'easy':20,'medium':20,'hard':20}
    assert export.BASELINE_GATES=={**export.RELEASE_GATES,
        'minimum_difficulty_rows_per_split':{'easy':20,'hard':20}}
    assert export.select_profile(rows,'balanced_v1')==(rows,[])


@pytest.mark.parametrize('kind',['difficulty','status','visibility'])
def test_profile_rejects_misclassified_answers(kind):
    r=row()
    if kind=='difficulty': r['difficulty']='hard'
    elif kind=='status': r['answer']['status']='abstain'
    else: r['answer']['visibility']='partial'
    with pytest.raises(ValueError): export.select_profile([r],'visible_occluded_v1')


def test_baseline_keeps_small_export_incomplete(source_audit):
    audit,output=source_audit
    result=export.build([audit],output,profile='visible_occluded_v1',allow_incomplete=True)
    assert result['release_profile']=='visible_occluded_v1'
    assert result['state']=='incomplete_engineering_export_do_not_claim_release'
    assert result['acceptance']['thresholds']==export.BASELINE_GATES
    assert export.validate(output,allow_incomplete=True)['rows']==1
    with pytest.raises(ValueError,match='Incomplete release'): export.validate(output)
    assert 'DATASET_CARD.md' in result['files_sha256']
    assert 'H200_HANDOFF.md' in result['files_sha256']


def test_profile_name_and_complete_state_cannot_be_swapped(source_audit):
    audit,output=source_audit
    export.build([audit],output,profile='visible_occluded_v1',allow_incomplete=True)
    path=output/'manifest.json';manifest=json.loads(path.read_text())
    manifest['state']='complete_synthetic_grounding_release'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Incomplete release'): export.validate(output)
    manifest['release_profile']='custom_relaxed';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Unsupported release profile'): export.validate(output)


def test_medium_cannot_enter_baseline_even_in_engineering_mode(source_audit,monkeypatch):
    audit,output=source_audit
    result=export.build([audit],output,allow_incomplete=True)
    # A valid balanced medium row, rehashed, is still forbidden in this profile.
    index=list(export.read_jsonl(output/'index.jsonl'))
    index[0]['difficulty']='medium';index[0]['answer']['visibility']='partial'
    (output/'index.jsonl').write_text(json.dumps(index[0])+'\n')
    result['release_profile']='visible_occluded_v1'
    result['files_sha256']['index.jsonl']=sha256(output/'index.jsonl')
    (output/'manifest.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Rows outside declared release profile'):
        export.validate(output,allow_incomplete=True)


def test_arbitrary_reduced_gate_set_is_rejected():
    gates=deepcopy(export.BASELINE_GATES);gates['minimum_rows']['train']=1
    with pytest.raises(ValueError,match='Unsupported release gate'):
        export.check_release_rows([row()],gates=gates)


def test_bounded_parallel_hash_checks_every_file_and_detects_changed_source(tmp_path):
    bindings={}
    for index in range(270):
        path=tmp_path/f'{index}.bin';path.write_bytes(str(index).encode());bindings[str(path)]=sha256(path)
    events=[];export.verify_source_bindings(bindings,progress=events.append)
    assert events[-1]['checked']==events[-1]['total']==270
    (tmp_path/'269.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Stale audit'): export.verify_source_bindings(bindings)


def test_review_list_is_explicit_nonempty_unique_and_hash_bound(tmp_path):
    path=tmp_path/'reviews.json';bundle=tmp_path/'bundle.json'
    path.write_text(json.dumps([str(bundle)]))
    values,bindings=export.read_review_list(path)
    assert values==[str(bundle.resolve())] and bindings=={str(path.resolve()):sha256(path)}
    for bad in ([],{},[None],[str(bundle),str(bundle)]):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): export.read_review_list(path)
