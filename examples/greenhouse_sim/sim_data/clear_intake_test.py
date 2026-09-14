import pytest
from sim_data.dataset_review import write_json
from sim_data.depth_preview import sha256
from sim_data.clear_collection_intake import checked_audit,watch


def record(tmp_path):
    batch=tmp_path/'batch';folder=batch/'job_001/audit';folder.mkdir(parents=True)
    audit=folder/'audit.json';write_json(audit,dict(schema='test_only'))
    job=dict(job_id='job_001',plant_family='seed101_full',state='audited_prototype_pending_visual_review',
             returncode=0,timed_out=False,audit_path=str(audit),audit_sha256=sha256(audit))
    write_json(batch/'manifest.json',dict(state='complete_bounded_batch_pending_visual_review',jobs=[job]))
    return dict(state='audited',job_id='job_001',family='seed101_full',batch=str(batch),audit=str(audit),
                manifest_sha256=sha256(batch/'manifest.json'))


def test_intake_requires_bound_successful_audit(tmp_path):
    r=record(tmp_path);assert checked_audit(r).name=='audit.json'
    with pytest.raises(ValueError):checked_audit(dict(r,state='failure'))
    with pytest.raises(ValueError):checked_audit(dict(r,family='other'))
    with pytest.raises(ValueError):checked_audit(dict(r,manifest_sha256='changed'))


def test_intake_refuses_source_overlap(tmp_path):
    campaign=tmp_path/'campaign';campaign.mkdir();write_json(campaign/'request.json',{})
    with pytest.raises(ValueError,match='disjoint'):watch(campaign,campaign/'nested')


def test_partially_written_outcome_is_not_consumed(tmp_path):
    from sim_data.clear_collection_intake import completed_json
    path=tmp_path/'outcome.json'
    assert completed_json(path) is None
    path.write_text('{"state":')
    assert completed_json(path) is None
    path.write_text('{"state":"audited"}')
    assert completed_json(path)=={'state':'audited'}


def test_stopped_campaign_cannot_materialize_final_release(tmp_path,monkeypatch):
    import sim_data.clear_collection_intake as intake
    campaign=tmp_path/'campaign';campaign.mkdir();write_json(campaign/'request.json',{})
    write_json(campaign/'result.json',dict(state='stopped_collection'))
    def forbidden(*a,**k):raise AssertionError('Must not build a release')
    monkeypatch.setattr(intake,'materialize',forbidden)
    with pytest.raises(ValueError,match='Collection stopped'):watch(campaign,tmp_path/'out',timeout=60)
    assert (tmp_path/'out/failure.json').exists() and not (tmp_path/'out/aggregate').exists()
