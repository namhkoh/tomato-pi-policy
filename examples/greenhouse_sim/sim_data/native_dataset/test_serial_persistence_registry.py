'''Serial registry boundaries only; Jason owns actual persistence/parity replay tests.'''
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

from . import serial_phases as q
from .test_serial_phases import fixture, harness, put, replace
from .test_serial_query_qualification import registered, mixed

p = q.persistence_kind


@pytest.fixture
def routed(registered, monkeypatch):
    calls = []
    def validate(path, pin, qualification):
        q.legacy._keys(qualification, p.JOB_FIELDS, 'test persistence fields')
        q.oc.require(qualification['persistence_policy_sha256'] == p.PERSISTENCE_POLICY_SHA256, 'Policy changed')
        calls.append(deepcopy(qualification))
        result = q.query_kind.validate_plan(path, pin, {k: qualification[k] for k in q.query_kind.JOB_FIELDS})
        result.update(qualification=deepcopy(qualification), source_plan_path=str(path))
        return result
    def recheck(checked, pin):
        q.oc.require(checked['qualification']['persistence_policy_sha256'] == p.PERSISTENCE_POLICY_SHA256,
                     'Policy changed before command')
        q.oc.pin(checked['source_plan_path'], pin)
        q.oc.bind_all(checked['bindings'])
        return p.implementation_bindings()
    monkeypatch.setattr(p, 'validate_plan', validate)
    monkeypatch.setattr(p, '_recheck', recheck)
    job = dict(deepcopy(registered.jobs[0]), id='V3', worker_kind=p.KIND,
        persistence_policy_sha256=p.PERSISTENCE_POLICY_SHA256)
    return SimpleNamespace(job=job, calls=calls)


def request_with_four_kinds(fixture, registered, routed):
    request = deepcopy(fixture.request)
    request['phases'] = [dict(id='v2', jobs=deepcopy(registered.jobs)),
        dict(id='v3', jobs=[deepcopy(routed.job)]), *request['phases']]
    request['planned_frames'] += 48
    return request


def test_exact_two_v2_then_one_v3_plan_reuse_and_fixed_argv(fixture, registered, routed):
    request = request_with_four_kinds(fixture, registered, routed)
    _, jobs = q.validate_request(request)
    assert [j['checked']['count'] for j in jobs] == [12,24,12,3,2]
    assert jobs[0]['plan_path'] == jobs[2]['plan_path'] and len(routed.calls) == 1
    job = jobs[2]
    folder = fixture.output/'v3'/'V3'
    command = q.worker_command(request['isaac_python'], p.KIND, folder/'submitted/plan.json',
        job['plan_sha256'], folder/'capture', checked=job['checked'])
    assert command[command.index('-m')+1] == p.audit.WORKER_MODULE
    for flag, value in (('--instance-backend','fast'), ('--render-budget','reference56'),
                        ('--persistence','strict_candidates_only'), ('--persistence-policy-sha256',p.PERSISTENCE_POLICY_SHA256)):
        assert command[command.index(flag)+1] == value
    assert '--qualification-witness-save-all' in command and '--profile-render' not in command
    assert command[command.index('--storage-qualification-sha256')+1] == routed.job['storage_qualification_sha256']


@pytest.mark.parametrize('fault', ['v3_twice','v2_twice','reverse','other_donor','path_alias','different_pin',
    'duplicate_id','original_reuse','policy','missing_policy','extra_mode','unregistered','original_copy'])
def test_new_registration_does_not_relax_other_duplicates_or_fields(fixture, registered, routed, fault):
    request = request_with_four_kinds(fixture, registered, routed)
    v2, v3 = request['phases'][0]['jobs'], request['phases'][1]['jobs'][0]
    if fault == 'v3_twice': request['phases'][1]['jobs'].append(dict(v3,id='V3again'))
    elif fault == 'v2_twice': v2.append(dict(v2[0],id='V2again'))
    elif fault == 'reverse': request['phases'][0], request['phases'][1] = request['phases'][1], request['phases'][0]
    elif fault == 'other_donor': v3.update(plan_path=v2[1]['plan_path'], plan_sha256=v2[1]['plan_sha256'])
    elif fault == 'path_alias': v3['plan_path'] = str(Path(v3['plan_path']).with_name('copy.json'))
    elif fault == 'different_pin': v3['plan_sha256'] = '0'*64
    elif fault == 'duplicate_id': v3['id'] = v2[0]['id']
    elif fault == 'original_reuse':
        request['phases'][1]['jobs'][0] = {k:v3[k] for k in ('id','worker_kind','plan_path','plan_sha256')}
        request['phases'][1]['jobs'][0]['worker_kind'] = q.WORKERS[0]
    elif fault == 'policy': v3['persistence_policy_sha256'] = '0'*64
    elif fault == 'missing_policy': v3.pop('persistence_policy_sha256')
    elif fault == 'extra_mode': v3['persistence'] = 'all_frames'
    elif fault == 'unregistered': v3['worker_kind'] = 'compact_query_v3'
    else:
        original = request['phases'][2]['jobs'][0]
        copied = fixture.tmp/'copied_plan/plan.json'
        copied.parent.mkdir(); copied.write_bytes(Path(original['plan_path']).read_bytes())
        request['phases'][2]['jobs'].append(dict(original,id='OCopy',plan_path=str(copied)))
    with pytest.raises((ValueError, FileNotFoundError)):
        q.validate_request(request)


@pytest.fixture
def four_run(mixed, routed, monkeypatch):
    m, h = mixed, mixed.h
    request = deepcopy(h.f.request)
    request['phases'].insert(1,dict(id='persistence',jobs=[deepcopy(routed.job)]))
    request['planned_frames'] += 12
    h.f.request = request
    h.request_pin = replace(h.request_path, request)
    def runner(command, log_path, **kw):
        if p.audit.WORKER_MODULE not in command:
            return m.runner(command, log_path, **kw)
        kw['reserve'](); kw['launch_check'](); q.oc.bind_all(kw['bindings'])
        h.launches.append(command); h.pipeline.append('v3_launch')
        kw['announce'](5000+len(h.launches))
        submitted, capture = Path(command[command.index('--batch-plan')+1]), Path(command[-1])
        assert submitted == capture.parent/'submitted/plan.json'
        capture.mkdir()
        put(capture/'request.json', dict(cpu_fixture=True, plan_path=str(submitted), plan_sha256=q.oc.sha256(submitted)))
        put(capture/'result.json', dict(cpu_fixture=True))
        h.pipeline.append('v3_exit0')
        return 0
    def audit(submitted, pin, capture, folder, checked):
        assert h.pipeline[-1] == 'v3_exit0'
        h.pipeline.append('v3_audit')
        assert checked['qualification']['persistence_policy_sha256'] == p.PERSISTENCE_POLICY_SHA256
        audit_path = folder/p.AUDIT_FILE
        audit_pin = put(audit_path, dict(cpu_fixture=True))
        parity_path = folder/p.PARITY_DIRECTORY/'CPU_FIXTURE.json'
        parity_pin = put(parity_path, dict(cpu_fixture=True,new_native_observations=0))
        proof_path = folder/p.QUALIFICATION_FILE
        proof_pin = put(proof_path, dict(cpu_fixture=True,training_diversity_increment=0))
        return dict(request_sha256=q.oc.sha256(capture/'request.json'),result_sha256=q.oc.sha256(capture/'result.json'),
            audit_path=str(audit_path),audit_sha256=audit_pin,qualification_path=str(proof_path),qualification_sha256=proof_pin,
            counts=dict(validated_callbacks=2,persisted_candidates=1,unpersisted_attempts=1,witness_samples=2),
            bindings={str(audit_path):audit_pin,str(parity_path):parity_pin,str(proof_path):proof_pin},**p.ANNOTATION)
    monkeypatch.setattr(q.legacy.campaign,'run_checked',runner)
    monkeypatch.setattr(p,'postexit_audit',audit)
    return SimpleNamespace(h=h,run=m.run,runner=runner,audit=audit)


def test_four_kind_runner_handoff_and_next_originals(four_run):
    f = four_run
    result = f.run()
    assert result['state'] == 'complete' and len(result['records']) == 5
    rows = [q.oc.read_json(r['receipt_path']) for r in result['records']]
    assert rows[0]['source_plan_path'] == rows[2]['source_plan_path']
    assert rows[0]['capture'] != rows[2]['capture'] and rows[0]['submitted_plan_path'] != rows[2]['submitted_plan_path']
    assert rows[0]['training_diversity_increment'] == rows[2]['training_diversity_increment'] == 0
    v3 = rows[2]
    assert v3['worker_kind'] == p.KIND and v3['persistence_policy_sha256'] == p.PERSISTENCE_POLICY_SHA256
    assert v3['qualification_witness_save_all'] is True and v3['native_persistence_qualified'] is False
    assert v3['persistence_mode'] == 'strict_candidates_only'
    assert rows[3]['worker_kind'] == 'original_batch.v1'
    q.oc.bind_all(v3['audit']['bindings']); q.oc.bind_all(result['bindings'])


@pytest.mark.parametrize('fault', ['exit_nonzero','copy_after_wait','audit_failure','new_capture_file','stop_during_audit'])
def test_v3_first_failure_or_clean_cooperative_stop(four_run, monkeypatch, fault):
    f, h = four_run, four_run.h
    if fault in ('exit_nonzero','copy_after_wait'):
        def runner(command, log_path, **kw):
            is_v3 = p.audit.WORKER_MODULE in command
            if is_v3 and fault == 'copy_after_wait':
                reserve = kw['reserve']
                def change():
                    reserve(); replace(Path(command[command.index('--batch-plan')+1]),{})
                kw['reserve'] = change
            code = f.runner(command,log_path,**kw)
            return 3 if is_v3 and fault == 'exit_nonzero' else code
        monkeypatch.setattr(q.legacy.campaign,'run_checked',runner)
    else:
        def audit(*a, **kw):
            if fault == 'audit_failure': raise ValueError('Insufficient qualified branch coverage')
            result = f.audit(*a,**kw)
            if fault == 'new_capture_file': put(a[2]/'unexpected.json',{})
            if fault == 'stop_during_audit':
                put(h.f.output/'stop.json',dict(schema=q.STOP_SCHEMA,request_sha256=h.request_pin,action='stop'))
                assert not (h.f.output/'result.json').exists()
            return result
        monkeypatch.setattr(p,'postexit_audit',audit)
    if fault == 'stop_during_audit':
        result = f.run()
        assert result['state'] == 'stopped_clean' and len(result['records']) == 3
    else:
        with pytest.raises(ValueError): f.run()
        assert q.oc.read_json(h.f.output/'failure.json')['state'] == 'failed_stop_first'
    assert len(h.launches) <= 3


def test_actual_v3_plan_through_registry_readonly():
    case = q.query_kind.CASES[0]
    proof = q.query_kind._ROOT/'data/sim_data/diagnostics/native_same_callback_20260916_v2/same_callback_qualification.json'
    if not p.SOURCE_PLAN.is_file() or not proof.is_file():
        pytest.skip('Host-specific exact source/proof absent')
    folder = p.SOURCE_PLAN.parent
    fields = dict(**p.ANNOTATION,persistence_policy_sha256=p.PERSISTENCE_POLICY_SHA256,
        storage_qualification=str(proof),storage_qualification_sha256='e82fab0e5e569c399aefd29f380622e002058cc58499e3d1a1c316d471ed79da',
        matched_short=dict(capture_path=str(folder/'capture'),request_sha256=case[6],result_sha256=case[7],
            audit_path=str(folder/'automatic_audit.json'),audit_sha256=case[8],
            completion_path=str(folder/'campaign_result.json'),completion_sha256=case[9]))
    checked = q.validate_plan(p.KIND,p.SOURCE_PLAN,p.PLAN_SHA256,query=fields)
    assert checked['count'] == 12 and len(checked['plan']['target_cases']) == 2
    assert checked['qualification'] == fields
