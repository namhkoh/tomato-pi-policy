'''CPU-only registration tests; synthetic captures are not native qualification.'''
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

from . import serial_phases as q
from . import serial_query_qualification as k
from .test_serial_phases import fixture, harness, put, replace
from .test_query_audit_v2 import annotated_cases, audit_case


def fields(folder, case, proof, proof_pin):
    return dict(**k.ANNOTATION, storage_qualification=str(proof), storage_qualification_sha256=proof_pin,
        matched_short=dict(capture_path=str(folder/'capture'), request_sha256=case[6], result_sha256=case[7],
            audit_path=str(folder/'automatic_audit.json'), audit_sha256=case[8],
            completion_path=str(folder/'campaign_result.json'), completion_sha256=case[9]))


@pytest.fixture
def registered(tmp_path, monkeypatch):
    root = tmp_path/'prior'
    source_plan = tmp_path/'original'/'plan.json'
    source_pin = put(source_plan, dict(package=str(tmp_path/'package')))
    source = {str(source_plan): source_pin}
    base = dict(source_collection_plan=str(source_plan), source_capture=str(tmp_path/'source_capture'),
        variant_directory=str(tmp_path/'variant'), prerequisite_directory=str(tmp_path/'sensor'))
    base_path = tmp_path/'base'/'plan.json'
    source[str(base_path)] = put(base_path, base)
    proof = tmp_path/'proof'/'qualification.json'
    proof_pin = put(proof, dict(synthetic=True))
    storage = dict(schema=k.storage.SCHEMA, storage_backend=k.storage.STORAGE_BACKEND,
        path=str(proof), sha256=proof_pin, bindings={str(proof): proof_pin})
    def qualify(path, *, expected_sha256):
        k.oc.pin(path, expected_sha256)
        return deepcopy(storage)
    monkeypatch.setattr(k.storage, 'check_qualification', qualify)
    monkeypatch.setattr(k.storage, 'verify_checked_qualification', lambda p: k.oc.bind_all(p['bindings']))
    monkeypatch.setattr(k.batch, 'check', lambda p, **kw: deepcopy(base))
    monkeypatch.setattr(k.batch, 'capture_jobs', lambda p, a: [None]*p['maximum_native_frames'])
    cases, jobs = [], []
    for index, (count, targets) in enumerate(((12, 2), (24, 4))):
        folder = root/f'case{index}'
        plan = dict(split='train', source_family='seed101_full', maximum_native_frames=count,
            resolution=[1696, 816], training_approved=False, source_cap_reset=False,
            physical_motion_commanded=False, hidden_cut_coordinates_executable=False, anchor_pair_plan=str(base_path),
            source_bindings=source, implementation_bindings=source, prerequisite_bindings=source,
            target_cases=[dict(base_pair_plan=str(base_path), target_id=f'T{i}', conservative_view_cap_group=f'T{i}',
                views=[dict(candidate_id=f'T{i}_V{j}') for j in range(count//targets)]) for i in range(targets)])
        pin = put(folder/'plan.json', plan)
        common = dict(native_instance_backend='fast', render_budget_profile='warm56_then8_trial', render_profile_experiment=False)
        rp = put(folder/'capture/request.json', dict(common, plan_path=str(folder/'plan.json'), plan_sha256=pin, training_started=False))
        result = dict(common, state='native_generated_multiview_pilot_complete_pending_review',
            training_approved=False, source_assets_unchanged=True, source_cap_reset=False,
            records=[{}]*count, target_count=targets, captured_frames=2, automatically_clear_annotation_candidates=0)
        sp = put(folder/'capture/result.json', result)
        counts = {'hold_visual_clarity': 2}
        ap = put(folder/'automatic_audit.json', dict(state='completed_automatic_annotation_replay',
            capture=str(folder/'capture'), plan_sha256=pin, request_sha256=rp, result_sha256=sp,
            counts=counts, training_approved=False, original_reviews_modified=False))
        cp = put(folder/'campaign_result.json', dict(state='native_complete_pending_review', native_exit_code=0,
            job=str(folder), source_family='seed101_full', training_approved=False, captured=2, automatic=0, audit_counts=counts))
        case = (folder.name, count, targets, 2, 0, pin, rp, sp, ap, cp)
        cases.append(case)
        jobs.append(dict(id=f'Q{index}', worker_kind=k.KIND, plan_path=str(folder/'plan.json'), plan_sha256=pin,
            **fields(folder, case, proof, proof_pin)))
    monkeypatch.setattr(k, '_PRIOR', root)
    monkeypatch.setattr(k, 'CASES', tuple(cases))
    return SimpleNamespace(jobs=jobs, cases=cases, root=root, proof=proof, source=source, storage=storage)


def check_job(job):
    return q.validate_plan(k.KIND, job['plan_path'], job['plan_sha256'], query={f: job[f] for f in k.JOB_FIELDS})


def test_checked_copies_command_and_protected_inputs(registered, tmp_path):
    job = registered.jobs[0]
    before = deepcopy(job)
    checked = check_job(job)
    assert job == before and checked['count'] == 12
    cmd = q.worker_command(str(tmp_path/'runtime/python.bat'), k.KIND, tmp_path/'submitted/plan.json',
                           job['plan_sha256'], tmp_path/'capture', checked=checked)
    assert cmd[1:3] == ['-m', k.audit.WORKER_MODULE]
    assert cmd[cmd.index('--render-budget')+1] == 'reference56'
    assert cmd[cmd.index('--instance-backend')+1] == 'fast'
    assert '--profile-render' not in cmd and '--plan' not in cmd
    assert cmd[cmd.index('--storage-qualification-sha256')+1] == job['storage_qualification_sha256']
    for root in [registered.root, registered.root/'case0', registered.proof.parent, tmp_path/'package', tmp_path/'original']:
        with pytest.raises(ValueError): q.oc.new_destination(root/'new', checked['roots'])
    checked['query']['matched_short']['capture_path'] = 'mutation'
    checked['storage']['bindings'].clear()
    assert job == before and registered.storage['bindings']


@pytest.mark.parametrize('fault', ['unknown_plan', 'old_pin', 'old_tamper', 'old_failure', 'completion_failure',
    'policy', 'epoch', 'proof', 'extra', 'targets5', 'count25', 'count_bool', 'heldout', 'worker_code', 'audit_code'])
def test_query_validation_rejects_faults(registered, monkeypatch, fault):
    job = deepcopy(registered.jobs[0])
    if fault == 'unknown_plan': job['plan_path'] = registered.jobs[1]['plan_path']
    elif fault == 'old_pin': job['matched_short']['result_sha256'] = '0'*64
    elif fault == 'old_tamper': replace(Path(job['matched_short']['capture_path'])/'result.json', {})
    elif fault == 'old_failure': put(Path(job['matched_short']['capture_path'])/'failure.json', {})
    elif fault == 'completion_failure': put(Path(job['plan_path']).parent/'failure.json', {})
    elif fault == 'policy': job['annotation_policy_sha256'] = '0'*64
    elif fault == 'epoch': job['annotation_epoch'] = 'legacy'
    elif fault == 'proof': job['storage_qualification_sha256'] = '0'*64
    elif fault == 'extra':
        query = {f: job[f] for f in k.JOB_FIELDS}
        query['render_budget'] = 'warm56_then8_trial'
        with pytest.raises(ValueError): k.validate_plan(job['plan_path'], job['plan_sha256'], query)
        return
    elif fault in ('targets5', 'count25', 'count_bool', 'heldout'):
        plan = q.oc.read_json(job['plan_path'])
        if fault == 'targets5': plan['target_cases'] *= 3
        elif fault == 'heldout': plan['source_family'] = 'seed999_full'
        else: plan['maximum_native_frames'] = True if fault == 'count_bool' else 25
        job['plan_sha256'] = replace(Path(job['plan_path']), plan)
    else:
        name = 'compact_query_v2.py' if fault == 'worker_code' else 'query_audit_v2.py'
        monkeypatch.setitem(k.FROZEN_CODE, name, '0'*64)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)): check_job(job)


@pytest.mark.parametrize('field,value', [('native_exit_code', False), ('state', 'pending'), ('job', 'foreign'),
    ('captured', 1), ('source_family', 'seed19_full')])
def test_old_completion_joins_even_with_test_allowlist_repin(registered, monkeypatch, field, value):
    job = deepcopy(registered.jobs[0])
    path = Path(job['matched_short']['completion_path'])
    done = q.oc.read_json(path)
    done[field] = value
    newpin = replace(path, done)
    case = list(registered.cases[0]); case[9] = newpin
    monkeypatch.setattr(k, 'CASES', (tuple(case), registered.cases[1]))
    job['matched_short']['completion_sha256'] = newpin
    with pytest.raises(ValueError): check_job(job)


def test_manifest_accepts_two_and_rejects_third_alias_and_overrides(fixture, registered):
    request = deepcopy(fixture.request)
    request['phases'].append(dict(id='query', jobs=deepcopy(registered.jobs)))
    request['planned_frames'] += 36
    _, jobs = q.validate_request(request)
    assert [j['checked']['count'] for j in jobs] == [3, 2, 12, 24]
    for fault in ('third', 'alias', 'override'):
        changed = deepcopy(request)
        items = changed['phases'][-1]['jobs']
        if fault == 'third': items.append(dict(items[0], id='Q3'))
        elif fault == 'alias': items[1]['plan_sha256'] = items[0]['plan_sha256']
        else: items[0]['instance_backend'] = 'legacy'
        with pytest.raises(ValueError): q.validate_request(changed)


@pytest.fixture
def synthetic(audit_case, annotated_cases, monkeypatch, tmp_path):
    def make(disposition='pass'):
        def edit(callback):
            meta = callback['metadata']
            meta.update(native_instance_backend='fast', render_budget=dict(k.budget_evidence('reference56', 0),
                actual_orchestrator_requests=7, render_seconds=1.25))
            meta['synchronization']['render_budget_subframes'] = 56
            callback['label'], callback['trace'] = k.audit.annotate_for_storage(meta, annotated_cases[disposition][0][1],
                callback['rgb'], callback['depth'], callback['valid'], callback['components'], callback['catalogue'],
                target_mask=callback['target_mask'].astype('uint8')*255)
        case = audit_case(disposition, callback_edit=edit)
        folder = tmp_path/('job_'+disposition)
        plan_path, capture = folder/'submitted/plan.json', case['capture']
        plan_path.parent.mkdir(parents=True)
        plan_path.write_bytes(case['plan_path'].read_bytes())
        plan = q.oc.read_json(plan_path)
        proof = dict(fixture_only=True)
        common = dict(native_instance_backend='fast', render_budget_profile='reference56', render_profile_experiment=False,
            compact_native_storage=True, storage_backend=k.storage.STORAGE_BACKEND)
        request = dict(q.oc.read_json(capture/'request.json'), **common, experimental_budget_override=False)
        request['plan_path'] = str(plan_path)
        result = dict(case['result'], **common, experimental_short_profile=False, render_budget_subframes_per_view=56,
            total_requested_subframes_per_view=56, target_count=1, source_cap_reset=False)
        result['records'][0]['render_budget'] = case['callback']['metadata']['render_budget']
        result['request_sha256'] = replace(capture/'request.json', request)
        replace(capture/'result.json', result)
        checked = dict(plan=plan, count=1, query=dict(matched_short={}), storage=proof,
            matched_short=dict(plan_path=str(plan_path), plan_sha256=q.oc.sha256(plan_path)),
            bindings={str(plan_path): q.oc.sha256(plan_path)}, roots=[])
        monkeypatch.setattr(k, '_matched', lambda *a: deepcopy(checked['matched_short']))
        return SimpleNamespace(case=case, capture=capture, folder=folder, plan_path=plan_path, checked=checked,
            pin=q.oc.sha256(plan_path), result=result, request=request)
    return make


def run_post(f):
    return q.postexit_audit(k.KIND, f.plan_path, f.pin, f.capture, f.folder, checked=f.checked)


@pytest.mark.parametrize('disposition,decision', [('pass', 'accept_strict_automatic_annotation_candidate'),
    ('hold', 'hold_visual_clarity'), ('exclude', 'exclude_geometry_or_visibility')])
def test_real_query_postexit_audit_not_qualification(synthetic, disposition, decision):
    f = synthetic(disposition)
    output = run_post(f)
    assert output['counts'] == {decision: 1}
    handoff = output['matched_comparison']
    assert handoff['captured_frames'] == 1 and handoff['training_diversity_increment'] == 0
    assert not handoff['comparisons_performed'] and not handoff['native_query_qualification_granted']
    assert not handoff['short_budget_qualified'] and handoff['equivalence_cutoff'] is None
    assert output['annotation_epoch'] == k.ANNOTATION['annotation_epoch']
    assert handoff['new']['audit_sha256'] == q.oc.sha256(output['audit_path'])
    q.oc.bind_all(output['bindings'])
    with pytest.raises(ValueError): run_post(f)  # create-only receipt


@pytest.mark.parametrize('fault', ['backend', 'budget', 'experiment', 'proof', 'plan', 'epoch', 'policy',
    'result_state', 'sample_budget', 'sample_steps', 'copy', 'missing_candidate', 'failure', 'source_cap_reset'])
def test_postexit_fails_closed(synthetic, fault):
    f = synthetic()
    result = f.result
    if fault == 'backend': result['native_instance_backend'] = 'legacy'
    elif fault == 'budget': result['render_budget_profile'] = 'warm56_then8_trial'
    elif fault == 'experiment': result['experimental_short_profile'] = True
    elif fault == 'proof': f.checked['storage'] = dict(other=True)
    elif fault == 'plan': result['plan_sha256'] = '0'*64
    elif fault == 'epoch': result['annotation_epoch'] = 'legacy'
    elif fault == 'policy': result['annotation_policy_sha256'] = '0'*64
    elif fault == 'result_state': result['state'] = 'pending'
    elif fault == 'source_cap_reset': result['source_cap_reset'] = True
    elif fault in ('sample_budget', 'sample_steps'):
        field = 'requested_subframes' if fault == 'sample_budget' else 'actual_orchestrator_requests'
        result['records'][0]['render_budget'][field] = 8
    elif fault == 'copy': replace(f.plan_path, {})
    elif fault == 'missing_candidate': result['records'].clear()
    else: put(f.capture/'failure.json', {})
    replace(f.capture/'result.json', result)
    with pytest.raises((ValueError, KeyError)): run_post(f)
    assert not (f.folder/'query_v2_postexit_audit.json').exists()


def test_zero_capture_reports_zero_not_qualification(synthetic):
    f = synthetic()
    result = f.result
    row = result['records'][0]
    row['state'] = 'rejected_pose'
    result.update(captured_frames=0, automatically_clear_annotation_candidates=0)
    replace(f.capture/'result.json', result)
    output = run_post(f)
    assert output['counts'] == {} and output['matched_comparison']['captured_frames'] == 0
    assert set(output['matched_comparison']['captured_by_target'].values()) == {0}
    assert output['matched_comparison']['native_query_qualification_granted'] is False


@pytest.fixture
def mixed(harness, registered, monkeypatch):
    h = harness
    request = deepcopy(h.f.request)
    request['phases'].insert(0, dict(id='query', jobs=deepcopy(registered.jobs)))
    request['planned_frames'] += 36
    h.f.request = request
    h.request_pin = replace(h.request_path, request)
    monkeypatch.setattr(k.audit, 'load_for_inspection', lambda *a: dict(report={}))
    original_audit = k.audit.audit_capture
    def audit(*a, **kw):
        h.pipeline.append('query_audit')
        h.audits.append(str(a[0]))
        return original_audit(*a, **kw)
    monkeypatch.setattr(k.audit, 'audit_capture', audit)
    def runner(command, log_path, **kw):
        if k.audit.WORKER_MODULE not in command:
            return h.runner(command, log_path, **kw)
        kw['reserve'](); kw['launch_check']()
        q.oc.bind_all(kw['bindings'])
        h.launches.append(command); h.pipeline.append('launch')
        kw['announce'](3000+len(h.launches))
        submitted = Path(command[command.index('--batch-plan')+1])
        capture = Path(command[-1])
        assert submitted == capture.parent/'submitted'/'plan.json'
        capture.mkdir()
        code = k.audit.implementation_bindings()
        common = dict(**k.ANNOTATION, annotation_policy=k.audit.annotation_policy(),
            worker_module=k.audit.WORKER_MODULE, worker_implementation_bindings=code,
            compact_implementation_bindings=code, storage_qualification=registered.storage,
            native_instance_backend='fast', render_budget_profile='reference56', render_profile_experiment=False,
            compact_native_storage=True, storage_backend=k.storage.STORAGE_BACKEND)
        rp = put(capture/'request.json', dict(common, plan_path=str(submitted), plan_sha256=q.oc.sha256(submitted),
            training_started=False, experimental_budget_override=False))
        plan = q.oc.read_json(submitted)
        rows = [dict(candidate_id=s['candidate_id'], target_id=c['target_id'], requested_spec=s, state='rejected_pose')
            for c in plan['target_cases'] for s in c['views']]
        put(capture/'result.json', dict(common, request_sha256=rp, plan_sha256=q.oc.sha256(submitted),
            state=k.audit.CAPTURE_STATE, source_assets_unchanged=True, training_approved=False, source_cap_reset=False,
            experimental_short_profile=False, render_budget_subframes_per_view=56, total_requested_subframes_per_view=56,
            target_count=len(plan['target_cases']), records=rows, captured_frames=0, automatically_clear_annotation_candidates=0))
        h.pipeline.append('exit0')
        return 0
    monkeypatch.setattr(q.legacy.campaign, 'run_checked', runner)
    return SimpleNamespace(h=h, registered=registered, runner=runner, audit=audit,
        run=lambda: q.run(h.request_path, h.request_pin))


def test_mixed_runner_e2e_real_zero_frame_query_audit_then_other_kinds(mixed):
    m, h = mixed, mixed.h
    before = [Path(j['plan_path']).read_bytes() for j in m.registered.jobs]
    result = m.run()
    assert result['state'] == 'complete' and len(result['records']) == 4
    assert h.pipeline[:6] == ['launch', 'exit0', 'query_audit']*2
    for index in (0, 1):
        row = result['records'][index]
        receipt = q.oc.read_json(row['receipt_path'])
        assert receipt['training_diversity_increment'] == 0 and receipt['exit_code'] == 0
        assert receipt['annotation_epoch'] == k.ANNOTATION['annotation_epoch']
        assert receipt['audit']['matched_comparison']['captured_frames'] == 0
        assert Path(receipt['submitted_plan_path']).read_bytes() == before[index]
        assert Path(receipt['source_plan_path']).read_bytes() == before[index]
    q.oc.bind_all(result['bindings'])


@pytest.mark.parametrize('fault', ['stop_during_audit', 'audit_error', 'exit_nonzero', 'proof_after_wait',
    'copy_after_wait', 'old_failure_after_wait', 'old_failure_after_audit', 'new_file_during_audit'])
def test_query_runner_clean_stop_and_fail_first(mixed, monkeypatch, fault):
    m, h = mixed, mixed.h
    if fault in ('stop_during_audit', 'audit_error', 'old_failure_after_audit', 'new_file_during_audit'):
        def audit(*a, **kw):
            if fault == 'audit_error': raise ValueError('Synthetic replay failure')
            result = m.audit(*a, **kw)
            if fault == 'stop_during_audit':
                put(h.f.output/'stop.json', dict(schema=q.STOP_SCHEMA, request_sha256=h.request_pin, action='stop'))
            elif fault == 'old_failure_after_audit':
                put(Path(m.registered.jobs[0]['matched_short']['capture_path'])/'failure.json', {})
            elif fault == 'new_file_during_audit': put(Path(a[0])/'unexpected.json', {})
            assert not (h.f.output/'result.json').exists()
            return result
        monkeypatch.setattr(k.audit, 'audit_capture', audit)
    else:
        def runner(command, log_path, **kw):
            if fault != 'exit_nonzero':
                reserve = kw['reserve']
                def changed():
                    reserve()
                    if fault == 'proof_after_wait': replace(m.registered.proof, {})
                    elif fault == 'copy_after_wait': replace(Path(command[command.index('--batch-plan')+1]), {})
                    else: put(Path(m.registered.jobs[0]['matched_short']['capture_path'])/'failure.json', {})
                kw['reserve'] = changed
            code = m.runner(command, log_path, **kw)
            return 3 if fault == 'exit_nonzero' else code
        monkeypatch.setattr(q.legacy.campaign, 'run_checked', runner)
    if fault == 'stop_during_audit':
        result = m.run()
        assert result['state'] == 'stopped_clean' and len(result['records']) == 1
        assert h.pipeline == ['launch', 'exit0', 'query_audit']
    else:
        with pytest.raises(ValueError): m.run()
        assert len(h.launches) <= 1 and not (h.f.output/'result.json').exists()
        assert q.oc.read_json(h.f.output/'failure.json')['state'] == 'failed_stop_first'


@pytest.mark.parametrize('index', [0, 1])
def test_actual_registered_plans_and_old_headers_readonly(index):
    case = k.CASES[index]
    folder = k._PRIOR/case[0]
    proof = k._ROOT/'data/sim_data/diagnostics/native_same_callback_20260916_v2/same_callback_qualification.json'
    if not (folder/'plan.json').is_file() or not proof.is_file():
        pytest.skip('Host-specific actual captures/proof absent')
    query = fields(folder, case, proof, 'e82fab0e5e569c399aefd29f380622e002058cc58499e3d1a1c316d471ed79da')
    checked = k.validate_plan(folder/'plan.json', case[5], query)
    assert checked['count'] == case[1] and len(checked['plan']['target_cases']) == case[2]
    assert checked['matched_short']['old_buffer_replay_performed'] is False
    q.oc.bind_all(checked['bindings'])
