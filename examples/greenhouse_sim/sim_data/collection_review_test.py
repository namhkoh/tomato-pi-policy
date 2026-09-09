import json
import pytest

from sim_data.collection_review import BatchReview
from sim_data.depth_preview import sha256


@pytest.fixture
def reviewer(tmp_path):
    jobs=[]
    for jid in ['job_001','job_002']:
        folder=tmp_path/jid/'audit'
        folder.mkdir(parents=True)
        path=folder/'audit.json'
        path.write_text('{}')
        jobs.append({'job_id':jid,'state':'audited_prototype_pending_visual_review','returncode':0,
                     'training_eligible':False,'audit_sha256':sha256(path)})
    (tmp_path/'manifest.json').write_text(json.dumps({'state':'complete_bounded_batch_pending_visual_review',
        'training_dataset_approved':False,'jobs':jobs}))
    calls=[]
    class Child:
        def __init__(self,audit,records):
            self.jid=audit.parent.parent.name
            self.images={('sample_0001','rgb'):b'fixture'}
        def state(self):
            return {'samples':[{'sample_id':'sample_0001','recommended':True,'human_review':None,
                                 'images':{'rgb':'/image/sample_0001/rgb'}}]}
        def save(self,payload):
            calls.append((self.jid,payload))
            return {'saved':{'fixture':True}}
    return BatchReview(tmp_path,Child),calls,tmp_path


def test_repeated_sample_numbers_have_distinct_routes_and_image_paths(reviewer):
    app,calls,_=reviewer
    state=app.state()
    assert state['total']==2 and state['training_eligible']==0
    assert {s['sample_id'] for s in state['samples']}=={'job_001_sample_0001','job_002_sample_0001'}
    assert ('job_001_sample_0001','rgb') in app.images and not calls


def test_explicit_save_routes_only_to_correct_job_without_rewriting_original_id(reviewer):
    app,calls,_=reviewer
    app.save({'sample_id':'job_002_sample_0001','decision':'hold','inspected':True})
    assert calls==[('job_002',{'sample_id':'sample_0001','decision':'hold','inspected':True})]


def test_unknown_job_cannot_receive_a_review(reviewer):
    app,calls,_=reviewer
    with pytest.raises(ValueError): app.save({'sample_id':'job_003_sample_0001'})
    assert not calls


def test_changed_batch_prevents_review_writes(reviewer):
    app,calls,root=reviewer
    (root/'manifest.json').write_text('{}')
    with pytest.raises(ValueError): app.save({'sample_id':'job_001_sample_0001'})
    assert not calls
