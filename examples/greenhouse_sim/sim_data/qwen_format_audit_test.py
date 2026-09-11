import json
import pytest
from sim_data.qwen_format_audit import audit
from sim_data.training_export import chat_row
from sim_data.depth_preview import sha256


@pytest.fixture
def release(tmp_path):
    (tmp_path/'splits').mkdir();files={}
    for split in ('train','validation','test'):
        row=chat_row(split,'images/'+split+'.png',dict(query_pixel_uv=[424.,204.],answer=dict(
            status='localized',cut_point_uv=[212.,102.],visibility='clear',next_action='inspect_cut_region')))
        path=tmp_path/'splits'/f'{split}.jsonl';path.write_text(json.dumps(row)+'\n')
        files[f'splits/{split}.jsonl']=sha256(path)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete_visible_occluded_baseline_release',
        release_profile='visible_occluded_v1',files_sha256=files,acceptance=dict(rows=dict(train=1,validation=1,test=1)))))
    return tmp_path


def test_format_audit_is_not_an_image_or_model_validation_claim(release):
    result=audit(release)
    assert result['counts']['train']==dict(rows=1,localized=1,abstain=0)
    assert result['maximum_pixel_roundtrip_error']<1e-9
    assert result['normal_release_validator_still_required']
    assert not result['image_depth_bytes_rechecked'] and not result['actual_Qwen_processor_tested']
    assert not result['training_started']


def test_modified_chat_or_incomplete_state_fail_closed(release):
    path=release/'splits/train.jsonl';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Split bytes'):audit(release)
    manifest=release/'manifest.json';value=json.loads(manifest.read_text());value['state']='incomplete'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completed narrowed'):audit(release)
