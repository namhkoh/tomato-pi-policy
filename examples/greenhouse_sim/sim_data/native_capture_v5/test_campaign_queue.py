from copy import deepcopy
import json
from pathlib import Path
import pytest
from . import campaign_queue as queue


@pytest.fixture
def config(tmp_path):
    return dict(schema=queue.SCHEMA, prior_campaign=str(tmp_path/'prior'),
        controls=str(tmp_path/'controls'), storage_qualification=str(tmp_path/'proof/proof.json'),
        view_plan=str(tmp_path/'plan.json'), output=str(tmp_path/'queue'),
        scale_output=str(tmp_path/'scale'), isaac_python=str(tmp_path/'isaac/python.bat'),
        native_deps=str(tmp_path/'deps'), schedule=str(tmp_path/'schedule.json'),
        view_plan_sha256='a'*64, schedule_sha256='b'*64, rounds=4, seed_base=4000000, max_targets=12)


def test_valid_is_readonly(config):
    original=deepcopy(config)
    paths=queue.validate_request(config)
    assert paths['output']==Path(config['output'])
    assert not paths['output'].exists()
    assert config==original


@pytest.mark.parametrize('key,value', [
    ('schema','wrong'),('rounds',0),('rounds',101),('rounds',True),
    ('max_targets',0),('max_targets',13),('seed_base',-1),('seed_base',2**32),
    ('seed_base',False),('view_plan_sha256','a'*63),('schedule_sha256','G'*64),
    ('view_plan','relative.json'),('controls',None)])
def test_invalid(config,key,value):
    config[key]=value
    with pytest.raises(ValueError):queue.validate_request(config)


@pytest.mark.parametrize('source', ['prior_campaign','controls','scale_output'])
def test_output_overlap(config,source):
    config['output']=str(Path(config[source])/'child')
    with pytest.raises(ValueError):queue.validate_request(config)


def test_output_in_proof_directory(config):
    config['output']=str(Path(config['storage_qualification']).parent/'child')
    with pytest.raises(ValueError):queue.validate_request(config)


def test_output_contains_plan(config):
    config['view_plan']=str(Path(config['output'])/'plan.json')
    with pytest.raises(ValueError):queue.validate_request(config)


def test_existing_output(config):
    Path(config['output']).mkdir()
    with pytest.raises(ValueError):queue.validate_request(config)


def test_scale_parameters_preserved(config):
    command=queue.scale_command(config,'cpu.exe')
    assert command[:3]==['cpu.exe','-m','sim_data.native_capture_v4.campaign']
    assert command[command.index('--output')+1]==config['scale_output']
    assert command[command.index('--rounds')+1]=='4'
    assert command[command.index('--seed-base')+1]=='4000000'
    assert command[command.index('--max-targets')+1]=='12'
    assert '--view-plan' not in command


def test_same_cpu_scale_handoff_preserves_arguments_and_log(config, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(queue, 'verify_bindings', lambda b: seen.append(('bindings', b)))
    def run(arguments):
        seen.append(('argv', arguments))
        print('synthetic V4 status')
    monkeypatch.setattr(queue.campaign, 'main', run)
    log = tmp_path/'scale.log'
    assert queue.run_scale_in_process(config, log, {'test':'pin'}, lambda pid: seen.append(('pid', pid))) == 0
    assert seen[0] == seen[-1] == ('bindings', {'test':'pin'})
    assert ('argv', queue.scale_command(config, queue.sys.executable)[3:]) in seen
    assert ('pid', queue.os.getpid()) in seen
    assert log.read_text() == 'synthetic V4 status\n'


@pytest.mark.parametrize('exit_code', [2, 'synthetic failure'])
def test_same_cpu_scale_failure_preserved(config, tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(queue, 'verify_bindings', lambda b: None)
    def fail(arguments):
        raise SystemExit(exit_code)
    monkeypatch.setattr(queue.campaign, 'main', fail)
    assert queue.run_scale_in_process(config, tmp_path/'scale.log', {}, lambda pid: None) != 0


def test_same_cpu_scale_unexpected_error_propagates(config, tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'verify_bindings', lambda b: None)
    def fail(arguments):
        raise RuntimeError('synthetic worker failure')
    monkeypatch.setattr(queue.campaign, 'main', fail)
    with pytest.raises(RuntimeError, match='synthetic worker failure'):
        queue.run_scale_in_process(config, tmp_path/'scale.log', {}, lambda pid: None)

def test_optional_original_plan(config,tmp_path):
    config.update(original_plan=str(tmp_path/'original/plan.json'),original_plan_sha256='c'*64)
    assert queue.validate_request(config)['original_plan']==tmp_path/'original/plan.json'


@pytest.mark.parametrize('value',[None,'relative.json'])
def test_invalid_original_path(config,value):
    config.update(original_plan=value,original_plan_sha256='c'*64)
    with pytest.raises(ValueError):queue.validate_request(config)


def test_original_pin_without_plan(config):
    config['original_plan_sha256']='c'*64
    with pytest.raises(ValueError):queue.validate_request(config)


def test_original_missing_pin(config,tmp_path):
    config['original_plan']=str(tmp_path/'original.json')
    with pytest.raises(ValueError):queue.validate_request(config)


def test_original_inside_output(config):
    config.update(original_plan=str(Path(config['output'])/'original.json'),original_plan_sha256='c'*64)
    with pytest.raises(ValueError):queue.validate_request(config)

def test_explicit_worker_environments(tmp_path):
    inherited={'PYTHONPATH':'unrelated-cpu-path','KEEP':'yes'}
    cpu,native=queue.worker_environments(tmp_path/'native-deps',inherited)
    assert inherited=={'PYTHONPATH':'unrelated-cpu-path','KEEP':'yes'}
    assert cpu['KEEP']==native['KEEP']=='yes'
    assert str(tmp_path/'native-deps') not in cpu['PYTHONPATH']
    assert native['PYTHONPATH'].split(queue.os.pathsep)[0]==str(tmp_path/'native-deps')
    assert len(cpu['PYTHONPATH'].split(queue.os.pathsep))==2
    assert cpu['OPENBLAS_NUM_THREADS']==native['OPENBLAS_NUM_THREADS']=='1'


@pytest.fixture
def completed_original(tmp_path):
    capture=tmp_path/'capture'
    capture.mkdir()
    plan=tmp_path/'plan.json'
    queue.write_json(capture/'request.json',dict(plan_path=str(plan),plan_sha256='d'*64))
    queue.write_json(capture/'result.json',dict(plan_sha256='d'*64,
        request_sha256=queue.sha256(capture/'request.json')))
    return capture,plan


def test_original_completion(completed_original):
    capture,plan=completed_original
    result=queue.original_completion(capture,plan,'d'*64)
    assert result['request_sha256']==queue.sha256(capture/'request.json')
    assert result['result_sha256']==queue.sha256(capture/'result.json')


@pytest.mark.parametrize('kind',['different_plan','different_pin','changed_request','failure'])
def test_original_completion_rejects_mismatch(completed_original,kind):
    capture,plan=completed_original
    if kind=='different_plan':plan=plan.with_name('other.json')
    if kind=='different_pin':
        with pytest.raises(ValueError):queue.original_completion(capture,plan,'e'*64)
        return
    if kind=='changed_request':
        (capture/'request.json').write_text(json.dumps(dict(plan_path=str(plan),plan_sha256='e'*64)))
    if kind=='failure':queue.write_json(capture/'failure.json',dict(failed=True))
    with pytest.raises(ValueError):queue.original_completion(capture,plan,'d'*64)
