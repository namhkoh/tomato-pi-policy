import json
import pytest
from sim_data.clear_cutpoint_test import source
from sim_data.clear_cutpoint_release import build,validate,exclusion_records
from sim_data.dataset_review import read_json,write_json
from sim_data.depth_preview import sha256
from sim_data.training_export import read_jsonl


def bind_index(source):
    m=read_json(source/'manifest.json');m['files_sha256']['index.jsonl']=sha256(source/'index.jsonl')
    (source/'manifest.json').write_text(json.dumps(m))


def negative(source):
    r=list(read_jsonl(source/'index.jsonl'))[0]
    return dict(id=r['id'],rgb_sha256=r['rgb_sha256'],reviewer='unit fixture',reviewer_type='assistant',decision='hold',reason='attachment uncertain')


def test_visual_hold_removed_only_in_new_derivative_with_provenance(source,tmp_path):
    bind_index(source);before=sha256(source/'manifest.json');record=negative(source)
    reviews=tmp_path/'holds.json';write_json(reviews,[record])
    out=tmp_path/'new';build(source,out,visual_exclusions=reviews)
    assert validate(out,allow_draft=True)['rows']==2
    assert record==read_json(out/'visual_exclusions.json')[0]
    assert record['id'] not in {r['id'] for r in read_jsonl(out/'index.jsonl')}
    assert sha256(source/'manifest.json')==before
    assert read_json(out/'exclusions.json')[0]['review']['decision']=='hold'


@pytest.mark.parametrize('change',[{'decision':'accept'},{'rgb_sha256':'wrong'},{'id':'missing'},{'reviewer':''}])
def test_invalid_negative_cannot_filter(source,change):
    rows=list(read_jsonl(source/'index.jsonl'))
    with pytest.raises(ValueError):exclusion_records(rows,[dict(negative(source),**change)])


def test_negative_provenance_tampering_rejected(source,tmp_path):
    bind_index(source);p=tmp_path/'holds.json';write_json(p,[negative(source)])
    out=tmp_path/'new';build(source,out,visual_exclusions=p)
    negative_rows=read_json(out/'visual_exclusions.json');negative_rows[0]['decision']='accept'
    (out/'visual_exclusions.json').write_text(json.dumps(negative_rows))
    m=read_json(out/'manifest.json');m['files_sha256']['visual_exclusions.json']=sha256(out/'visual_exclusions.json')
    (out/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):validate(out,allow_draft=True)
