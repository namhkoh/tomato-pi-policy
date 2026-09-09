from copy import deepcopy
from types import SimpleNamespace

import pytest

from . import collection_campaign as campaign
from .dataset_review import read_json, write_json
from .depth_preview import sha256


def plans():
    train = dict(schema_version='greenhouse.grounding_collection_plan.v1', package='/fixture',
                 family_assignments={str(i):'train' if i<16 else 'test' if i<20 else 'validation' for i in range(24)},
                 cut_rule_sha256='rule', source_bindings_sha256={},
                 configuration=dict(view_offset=8,render_views_per_target=160))
    train['jobs'] = [dict(job_id=f'job_{i+1:03d}',plant_family=str(i),split=train['family_assignments'][str(i)],
                          targets=[{'target':str(i)}]*12,max_rendered_views_per_target=160) for i in range(24)]
    heldout = deepcopy(train); heldout['configuration']['render_views_per_target'] = 64
    for job in heldout['jobs']: job['max_rendered_views_per_target'] = 64
    return train, heldout


def predecessors():
    return [dict(job_id=f'job_{i+1:03d}',remaining_view_estimate=300,batch=str(i),request_sha256='hash') for i in range(2)]


def test_allocation_has_no_missing_duplicate_or_rescheduled_family():
    a,b = plans(); before = deepcopy((a,b)); lanes = campaign.allocate(a,b,predecessors())
    assert (a,b) == before
    assert lanes == campaign.allocate(a,b,predecessors())
    assert len(lanes) == 4 and sum(v['predecessor'] is not None for v in lanes) == 2
    rows = [r for lane in lanes for r in lane['items']]
    assert sorted(r['job_id'] for r in rows) == [f'job_{i:03d}' for i in range(3,25)]
    assert all(r['maximum_frames'] == 12*(160 if r['split']=='train' else 64) for r in rows)
    for lane in lanes:
        flags = [r['split']=='train' for r in lane['items']]
        assert flags == sorted(flags)


@pytest.mark.parametrize('change',['split','target','offset','source','duplicate_predecessor','unknown_predecessor'])
def test_allocation_rejects_changed_sources_or_reservations(change):
    a,b = plans(); pred = predecessors()
    if change=='split': b['jobs'][5]['split']='validation'
    if change=='target': b['jobs'][5]['targets']=[{'target':'different'}]
    if change=='offset': b['configuration']['view_offset']=0
    if change=='source': b['source_bindings_sha256']={'changed':'hash'}
    if change=='duplicate_predecessor': pred[1]=pred[0]
    if change=='unknown_predecessor': pred[1]['job_id']='job_999'
    with pytest.raises(ValueError): campaign.allocate(a,b,pred)


def dependency(tmp_path):
    write_json(tmp_path/'request.json',{'fixture':True})
    return dict(batch=str(tmp_path),job_id='job_001',request_sha256=sha256(tmp_path/'request.json'))


def complete_dependency(tmp_path):
    pred = dependency(tmp_path); job = tmp_path/'job_001'; (job/'audit').mkdir(parents=True)
    write_json(job/'audit/audit.json',{'fixture':True})
    result = dict(job_id='job_001',state=campaign.AUDITED,returncode=0,timed_out=False,
                  audit_sha256=sha256(job/'audit/audit.json'))
    write_json(job/'result.json',result)
    write_json(tmp_path/'manifest.json',dict(state=campaign.COMPLETE,jobs=[result]))
    return pred


def test_slot_waits_for_complete_predecessor(tmp_path):
    pred = dependency(tmp_path)
    assert campaign.predecessor_ready(pred) is False


def test_slot_is_released_only_after_matching_audited_zero_exit(tmp_path):
    pred = complete_dependency(tmp_path)
    assert campaign.predecessor_ready(pred) is True
    pred['request_sha256'] = 'stale'
    with pytest.raises(ValueError,match='Changed predecessor request'): campaign.predecessor_ready(pred)


def test_failed_predecessor_never_starts_successor(tmp_path):
    pred = dependency(tmp_path); job = tmp_path/'job_001'; job.mkdir()
    write_json(job/'result.json',dict(state='stopped_worker_nonzero_exit',returncode=1))
    with pytest.raises(ValueError,match='failed'): campaign.predecessor_ready(pred)


def test_predecessor_wait_is_bounded_and_uses_short_polls(monkeypatch):
    ticks = iter([0,0,6]); sleeps=[]
    monkeypatch.setattr(campaign,'predecessor_ready',lambda p:False)
    monkeypatch.setattr(campaign.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(campaign.time,'sleep',sleeps.append)
    with pytest.raises(ValueError,match='timed out'): campaign.wait_predecessor({},timeout_s=5)
    assert sleeps == [5]


@pytest.mark.parametrize('mode',['success','disk_low','worker_failure'])
def test_lane_runs_serially_and_stops_safely(tmp_path,monkeypatch,mode):
    a,b = plans(); pred = predecessors(); lanes = campaign.allocate(a,b,pred)
    paths={}
    for name,plan in [('train',a),('heldout',b)]:
        path=tmp_path/(name+'.json'); write_json(path,plan); paths[name]=dict(path=str(path),sha256=sha256(path))
    path=tmp_path/'campaign.json'
    write_json(path,dict(schema_version=campaign.SCHEMA,maximum_capture_workers=4,
                        plans=paths,lanes=lanes,minimum_free_disk_bytes=100,
                        implementation_sha256=sha256(campaign.__file__)))
    monkeypatch.setattr(campaign,'load_plan',lambda p:(read_json(p),[]))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=0 if mode=='disk_low' else 200))
    called=[]
    def run(plan,output,**kwargs):
        called.append(kwargs); output.mkdir()
        result=dict(state='stopped_worker_nonzero_exit' if mode=='worker_failure' else campaign.COMPLETE)
        write_json(output/'manifest.json',result); return result
    monkeypatch.setattr(campaign,'run_jobs',run)
    if mode != 'success':
        with pytest.raises(ValueError): campaign.run_lane(path,'lane_03')
        assert len(called)==(0 if mode=='disk_low' else 1)
        assert read_json(tmp_path/'lane_03/result.json')['state']=='stopped_collection_lane'
        return
    result=campaign.run_lane(path,'lane_03')
    assert result['state']=='complete_collection_lane_pending_visual_review'
    assert all(r['max_jobs']==1 and len(r['job_ids'])==1 and r['timeout_s']==14400 for r in called)
    count=len(called)
    with pytest.raises(FileExistsError): campaign.run_lane(path,'lane_03')
    assert len(called)==count
