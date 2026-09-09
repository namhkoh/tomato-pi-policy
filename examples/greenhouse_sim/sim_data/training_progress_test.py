import json

from sim_data import training_progress as progress


def test_running_batch_is_not_release_or_audited(tmp_path, monkeypatch):
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(schema_version='greenhouse.grounding_collection_plan.v1',jobs=[
        dict(job_id='job_001',plant_family='one',split='train',source_manifest_path='fixture') ])))
    batch=tmp_path/'batch'
    (batch/'job_001/capture/sample_0001').mkdir(parents=True)
    (batch/'job_001/capture/sample_0001/sample.json').write_text('{')
    monkeypatch.setattr(progress,'audit_manifest',lambda path:{})
    report=progress.summarize(plan,batch)
    assert report['release_approved'] is False and report['audited_sample_files']==0
    assert report['total_complete_sample_files']==0
    assert report['jobs'][0]['worker_state']=='running_or_unfinalized'
    assert len(report['jobs'][0]['incomplete_or_changed_samples'])==1
