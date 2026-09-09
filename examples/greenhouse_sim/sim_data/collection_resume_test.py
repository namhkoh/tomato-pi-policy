from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sim_data import collection_resume as resume, collection_run as run
from sim_data.collection_process import EXIT_SCHEMA
from sim_data.dataset_review import read_json, write_json
from sim_data.depth_preview import sha256


def item(i,action='new_directory_capture',split='train'):
    return dict(job_id=f'job_{i:03d}',action=action,split=split,maximum_frames=i*100)


def test_recovery_allocation_keeps_live_workers_first_and_all_jobs_once():
    items=[item(1,'observe_then_audit')]+[item(i,split='validation' if i%2 else 'train') for i in range(2,20)]
    lanes=resume.allocate(items)
    assert len(lanes)==3 and lanes[0]['items'][0]==items[0]
    assert sorted(i['job_id'] for l in lanes for i in l['items'])==sorted(i['job_id'] for i in items)
    assert lanes==resume.allocate([i for l in lanes for i in l['items']])
    with pytest.raises(ValueError,match='Duplicate'): resume.allocate(items+[items[0]])
    with pytest.raises(ValueError,match='Unknown'): resume.allocate([item(1,'assume_exit_zero')])


@pytest.fixture
def exited(tmp_path):
    folder=tmp_path/'job_001'; (folder/'capture').mkdir(parents=True)
    command=['python','-u','-m','sim_data.collection_worker','--job','job_001','--output',str(folder/'capture')]
    launch=dict(pid=123,command=command,started_utc=datetime.now(timezone.utc).isoformat(),timeout_s=60)
    write_json(folder/'launch.json',launch); (folder/'worker.log').write_text('fixture log')
    write_json(folder/'capture/manifest.json',dict(state='pilot_ready_for_review',samples=[{}],source_assets_unchanged=True,training_dataset_approved=False))
    receipt=dict(schema_version=EXIT_SCHEMA,launch_path=str(folder/'launch.json'),launch_sha256=sha256(folder/'launch.json'),
        pid=123,command=command,method='subprocess_wait',returncode=0,timed_out=False,observed_exit_utc=datetime.now(timezone.utc).isoformat())
    path=folder/'worker_exit.json'; write_json(path,receipt)
    return folder,path


def test_real_exit_receipt_required_even_when_capture_manifest_says_complete(exited):
    folder,path=exited
    changed=read_json(path); changed['launch_sha256']='wrong'
    other=folder/'bad_exit.json'; write_json(other,changed)
    with pytest.raises(ValueError,match='mismatch'): run.finalize_job(folder,dict(job_id='job_001',plant_family='one'),other)
    assert not (folder/'result.json').exists()


def test_exit_receipt_survives_audit_failure_without_promoting_data(exited,monkeypatch):
    folder,path=exited
    def audit(*a,**k):
        assert read_json(path)['returncode']==0
        raise RuntimeError('deliberate audit failure')
    monkeypatch.setattr('sim_data.dataset_review.audit',audit)
    result=run.finalize_job(folder,dict(job_id='job_001',plant_family='one'),path)
    assert result['returncode']==0 and result['state']=='stopped_independent_audit_failed'
    assert result['exit_receipt_sha256']==sha256(path)
    with pytest.raises(ValueError,match='overwrite'): run.finalize_job(folder,dict(job_id='job_001'),path)


def test_recovery_does_not_overwrite_prior_partial_audit(exited,monkeypatch):
    folder,path=exited; (folder/'audit').mkdir(); old=folder/'audit/partial.txt'; old.write_text('keep')
    def audit(capture,destination,**kwargs):
        assert destination==folder/'audit_recovered' and not destination.exists()
        destination.mkdir(); result={'samples':[{'quality':{'clear_view_gate_passed':True}}]}
        write_json(destination/'audit.json',result); return result
    monkeypatch.setattr('sim_data.dataset_review.audit',audit)
    result=run.finalize_job(folder,dict(job_id='job_001',plant_family='one'),path,audit_name='audit_recovered')
    assert result['state']==resume.AUDITED and old.read_text()=='keep'


def test_new_resume_skips_bound_completions_and_preserves_unproven_source(tmp_path):
    old=tmp_path/'old'; old.mkdir(); package=tmp_path/'assets'; package.mkdir()
    jobs=[dict(job_id=f'job_{i:03d}',plant_family=f'seed{i}',split='train',targets=[{}],max_rendered_views_per_target=10) for i in range(1,4)]
    p=dict(package=str(package),family_assignments={f'seed{i}':'train' for i in range(1,4)},jobs=jobs)
    plan=tmp_path/'plan.json'; write_json(plan,p)
    campaign=dict(schema_version='greenhouse.bounded_collection_campaign.v1',plans={k:dict(path=str(plan),sha256=sha256(plan)) for k in ('train','heldout')},
        minimum_free_disk_bytes=1024,lanes=[dict(lane_id='lane_01',predecessor=None,items=[dict(job_id=j['job_id']) for j in jobs])])
    c=old/'campaign.json'; write_json(c,campaign)
    first=old/'lane_01/capture_job_001/job_001'; (first/'audit').mkdir(parents=True)
    write_json(first/'audit/audit.json',{})
    write_json(first.parent/'request.json',dict(plan_sha256=sha256(plan),selected_job_ids=['job_001']))
    write_json(first/'result.json',dict(state=resume.AUDITED,returncode=0,timed_out=False,job_id='job_001',plant_family='seed1',audit_sha256=sha256(first/'audit/audit.json')))
    second=old/'lane_01/capture_job_002/job_002/capture'; second.mkdir(parents=True)
    write_json(second/'manifest.json',dict(state='pilot_ready_for_review'))
    before={f:f.read_bytes() for f in old.rglob('*') if f.is_file()}
    output=tmp_path/'resume'; r=resume.create(c,[],output)
    assert len(r['skipped_audited_jobs'])==1
    pending=[i for l in r['lanes'] for i in l['items']]
    assert {i['job_id'] for i in pending}=={'job_002','job_003'}
    assert all(i['action']=='new_directory_capture' for i in pending)
    assert all(f.read_bytes()==b for f,b in before.items())
    with pytest.raises(ValueError,match='new separate'): resume.create(c,[],output)


def test_direct_worker_receipt_is_durable_before_audit_begins(tmp_path,monkeypatch):
    plan_file=tmp_path/'plan.json'; write_json(plan_file,{})
    plan=dict(package=str(tmp_path/'assets'),jobs=[dict(job_id='job_001',plant_family='one')])
    monkeypatch.setattr(run,'load_plan',lambda p:(plan,[]))
    class Process:
        pid=123; returncode=0
        def __init__(self,command,**kwargs):
            capture=Path(command[command.index('--output')+1]); capture.mkdir()
            write_json(capture/'manifest.json',dict(state='pilot_ready_for_review',samples=[{}],source_assets_unchanged=True,training_dataset_approved=False))
        def poll(self): return 0
    monkeypatch.setattr(run.subprocess,'Popen',Process)
    def audit(capture,output,**kwargs):
        r=read_json(capture.parent/'worker_exit.json')
        assert r['returncode']==0 and r['method']=='subprocess_wait'
        assert not (capture.parent/'result.json').exists()
        raise RuntimeError('fixture interruption after a durably recorded exit')
    monkeypatch.setattr('sim_data.dataset_review.audit',audit)
    result=run.run_jobs(plan_file,tmp_path/'batch')
    assert result['state']=='stopped_independent_audit_failed'


def test_unrelated_capture_blocks_new_slot(monkeypatch,tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(resume.psutil,'process_iter',lambda fields:[SimpleNamespace(info={
        'cmdline':['python','-m','sim_data.collection_worker','--output',str(tmp_path/'elsewhere')]})])
    with pytest.raises(ValueError,match='Unrelated'):
        resume.check_inventory(dict(lanes=[],maximum_capture_workers=3),tmp_path/'ours')


def test_progress_recognizes_recovered_audit_path(exited,monkeypatch):
    from sim_data import training_progress as progress
    folder,path=exited; audit=folder/'audit_recovered'; audit.mkdir(); write_json(audit/'audit.json',{})
    write_json(folder/'result.json',dict(returncode=0,timed_out=False,state=resume.AUDITED,
        audit_path=str(audit/'audit.json'),audit_sha256=sha256(audit/'audit.json')))
    plan=folder.parent/'plan.json'
    write_json(plan,dict(schema_version='greenhouse.grounding_collection_plan.v1',jobs=[dict(
        job_id='job_001',plant_family='one',split='train',source_manifest_path='fixture')]))
    monkeypatch.setattr(progress,'audit_manifest',lambda p:{})
    assert progress.summarize(plan,folder.parent)['jobs'][0]['independently_audited']
