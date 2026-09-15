"""Synthetic receipts and mocked children; never launch applications."""
import json
from pathlib import Path
import pytest
from . import campaign as c


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def controls(tmp_path):
    root=tmp_path/'controls';records=[];plans=[];bindings={}
    for i in range(3):
        plan=tmp_path/f'plan_{i}.json';write(plan,{'id':i});plans.append(str(plan));bindings[str(plan)]=c.sha256(plan)
        case=root/f'case_{i+1:03d}'
        write(case/'result.json',dict(state='generated_native_pair_captured_pending_visual_review',
            samples=[dict(sample_id='original_control'),dict(sample_id='generated_variant')]))
        records.append(dict(plan=str(plan),output=str(case),exit_code=0,
            state='native_pair_complete_pending_annotation_and_visual_control_review',result_sha256=c.sha256(case/'result.json')))
    request=dict(plans=plans,max_pairs=3,max_images=6,bindings=bindings)
    result=dict(records=records,bindings=bindings)
    write(root/'request.json',request);write(root/'result.json',result)
    return root,request,result


def test_complete_controls(tmp_path):
    root,_,_=controls(tmp_path);bindings=c.completed_controls(root)
    assert str(root/'request.json') in bindings and str(root/'case_003/result.json') in bindings


@pytest.mark.parametrize('damage',['partial','duplicate','failure','wrong_output','changed_case','changed_plan','wrong_samples'])
def test_bad_controls_block(tmp_path,damage):
    root,request,result=controls(tmp_path)
    if damage=='partial':result['records'].pop()
    elif damage=='duplicate':result['records'][1]=result['records'][0]
    elif damage=='failure':write(root/'case_002/failure.json',{})
    elif damage=='wrong_output':result['records'][0]['output']=str(tmp_path/'elsewhere')
    elif damage=='changed_case':write(root/'case_001/result.json',{})
    elif damage=='changed_plan':write(Path(request['plans'][0]),{'changed':True})
    else:
        path=root/'case_001/result.json';write(path,dict(state='generated_native_pair_captured_pending_visual_review',
            samples=[dict(sample_id='original_control')]*2));result['records'][0]['result_sha256']=c.sha256(path)
    write(root/'result.json',result)
    with pytest.raises((ValueError,KeyError)):c.completed_controls(root)


def test_queue_output_disjoint(tmp_path):
    prior,controls,proof=tmp_path/'prior',tmp_path/'controls',tmp_path/'proof/result.json'
    c.check_queue_destination(tmp_path/'new',prior,controls,proof)
    for bad in (prior,prior/'child',controls/'child',proof.parent/'child',tmp_path):
        with pytest.raises(ValueError):c.check_queue_destination(bad,prior,controls,proof)


def test_proof_hash_reaches_worker(tmp_path):
    command=c.compact_command(tmp_path/'python.bat',tmp_path/'plan.json',tmp_path/'capture',tmp_path/'proof.json','a'*64)
    assert command[command.index('--storage-qualification-sha256')+1]=='a'*64
    assert command[command.index('--storage-qualification')+1]==str(tmp_path/'proof.json')
    with pytest.raises(ValueError):c.compact_command('x','p','o','q','bad')


class FakeProcess:
    pid=999999999
    def __init__(self,fail_wait=False):self.alive=True;self.fail_wait=fail_wait;self.waits=[]
    def poll(self):return None if self.alive else 0
    def wait(self,timeout=None):
        self.waits.append(timeout)
        if self.fail_wait and timeout is None:raise KeyboardInterrupt()
        self.alive=False;return 0


def invoke(tmp_path,monkeypatch,*,announce=lambda pid:None,reserve=lambda:None,launch_check=None,process=None,bindings=None):
    process=process or FakeProcess();started=[]
    if bindings is None:
        source=tmp_path/'pinned_source';source.write_text('unchanged');bindings={str(source):c.sha256(source)}
    def popen(*args,**kwargs):started.append(True);return process
    monkeypatch.setattr(c.subprocess,'Popen',popen)
    return c.run_checked(['not_launched'],tmp_path/'worker.log',bindings=bindings,environment={},
        native=True,reserve=reserve,launch_check=launch_check,announce=announce),process,started


@pytest.mark.parametrize('failure',['status','wait'])
def test_exceptions_reap_owned_child(tmp_path,monkeypatch,failure):
    process=FakeProcess(fail_wait=failure=='wait');cleaned=[]
    def cleanup(p):cleaned.append(p.pid);p.alive=False
    monkeypatch.setattr(c,'_terminate_owned',cleanup)
    def announce(pid):
        if failure=='status':raise OSError('log full')
    with pytest.raises((OSError,KeyboardInterrupt)):
        invoke(tmp_path,monkeypatch,announce=announce,process=process)
    assert cleaned==[process.pid] and not process.alive and process.waits[-1]==30


def test_failed_cleanup_is_fatal_not_caught_as_job_failure(tmp_path,monkeypatch):
    def cleanup(p):raise RuntimeError('cannot reap')
    monkeypatch.setattr(c,'_terminate_owned',cleanup)
    with pytest.raises(c.OwnedChildCleanupError):
        invoke(tmp_path,monkeypatch,process=FakeProcess(fail_wait=True))
    assert not issubclass(c.OwnedChildCleanupError,Exception)


def test_binding_change_while_waiting_prevents_spawn(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_text('before');binding={str(source):c.sha256(source)}
    def unexpected(*a,**k):pytest.fail('Popen must not happen')
    monkeypatch.setattr(c.subprocess,'Popen',unexpected)
    with pytest.raises(ValueError):c.run_checked(['none'],tmp_path/'log',bindings=binding,environment={},native=True,
        reserve=lambda:source.write_text('after'),launch_check=None,announce=lambda pid:None)
    assert not (tmp_path/'log').exists()


def test_plan_check_failure_prevents_spawn(tmp_path,monkeypatch):
    def reject():raise ValueError('plan changed')
    def unexpected(*a,**k):pytest.fail('Popen must not happen')
    monkeypatch.setattr(c.subprocess,'Popen',unexpected)
    source=tmp_path/'pinned_source';source.write_text('unchanged')
    with pytest.raises(ValueError,match='plan changed'):
        c.run_checked(['none'],tmp_path/'log',bindings={str(source):c.sha256(source)},environment={},native=True,
            reserve=lambda:None,launch_check=reject,announce=lambda pid:None)
