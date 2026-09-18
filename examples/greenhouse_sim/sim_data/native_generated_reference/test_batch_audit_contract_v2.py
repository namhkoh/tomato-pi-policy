"""Synthetic CPU whole-audit envelope tests, never native capture evidence.

Use a declared pre-render rejection: no synthetic image can be admitted. Only
full scene-generation and storage-qualification replay are test stubs; request
hashes, source/code pins, owned command/resource declarations, decision files,
ordered coverage and final source rehashing execute through production code.
Actual sample-buffer replay is tested separately against saved native bytes.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from .test_batch_execution_v2 import request_case, put
from . import batch_execution_v2 as ex, batch_audit_v2 as audit


@pytest.mark.parametrize('mutation', ['none', 'decision', 'extra_folder', 'missing_case'])
def test_full_audit_binds_new_decisions_and_complete_coverage(request_case, monkeypatch, mutation):
    from . import batch_prepare_v2
    from .. import collection_plan
    from ..native_dataset.reference_bridge_owner_v1 import verify_owner_row
    req = request_case['request']; root = Path(req['output']); capture = root/'capture'
    capture.mkdir(parents=True)
    row = dict(ProcessId=200, ParentProcessId=1, CreationDate='UNIT_ONLY', Name='python.exe',
        ExecutablePath=sys.executable, CommandLine=subprocess.list2cmdline([sys.executable, '-B', 'unit_only.py']))
    raw = verify_owner_row(row, pid=200, executable=sys.executable)
    owner = dict(execution_request_path=request_case['path'], execution_request_sha256=request_case['sha'],
        owner_pid=200, resources=dict(memory=dict(checked=True, allowed=True, commit_headroom_bytes=20*2**30),
        disk_free_bytes=60*2**30, process_inventory=dict(classifications=['UNIT_ONLY'], blockers=[], no_blockers_observed=True),
        exclusive_launch_guaranteed=False, raw_owner_classification=raw))
    _, owner_sha = put(root/'owner_started.json', owner)
    _, worker_sha = put(root/'owned_worker.json', dict(owner_pid=200, launcher_pid=201,
        command=ex.worker_command(request_case['path'], request_case['sha'], req)))
    cr = dict(schema=ex.RESULT, execution_request_path=request_case['path'], execution_request_sha256=request_case['sha'],
        owner_started_sha256=owner_sha, owned_worker_sha256=worker_sha, plan_sha256=req['plan_sha256'],
        training_started=False, process_admission=dict(no_unrelated_kit_process_at_admission=True),
        host_memory_preflight=dict(checked=True, allowed=True, commit_headroom_bytes=20*2**30))
    _, request_sha = put(capture/'request.json', cr)
    case, spec = ex.jobs(request_case['plan'])[0]
    record = dict(candidate_id=spec['candidate_id'], target_id=case['target_id'], requested_spec=spec,
        state='rejected_pose', reason='UNIT_ONLY_PRE_RENDER_REJECTION', **ex.FLAGS)
    saved = deepcopy(record)
    if mutation == 'decision': saved['reason'] = 'DIFFERENT_UNIT_REASON'
    decision_path, decision_sha = put(capture/(spec['candidate_id']+'_decision.json'), saved)
    result = dict(schema=ex.RESULT, state='bounded_batch_capture_complete_pending_owned_exit',
        plan_sha256=req['plan_sha256'], request_sha256=request_sha, implementation_bindings=req['implementation_bindings'],
        native_exit_status_must_be_checked_by_owner=True, source_assets_unchanged=True, stage_count=1,
        render_product_count=1, storage_qualification=req['storage_qualification'], storage_backend=req['storage_backend'],
        records=[] if mutation == 'missing_case' else [record], captured_frames=0, **ex.FLAGS)
    _, result_sha = put(capture/'result.json', result)
    if mutation == 'extra_folder': (capture/'unplanned_sample').mkdir()
    monkeypatch.setattr(batch_prepare_v2, 'check_plan', lambda plan: {'UNIT_ONLY': True})
    monkeypatch.setattr(ex, 'checked_storage', lambda request: request['storage_qualification'])
    monkeypatch.setattr(collection_plan, 'load_plan', lambda path: (None, []))
    # The fixture clear-plan path is not used by either synthetic scene stub.
    plan = request_case['plan']; plan['scene_authority']['clear_plan'] = dict(path='UNIT_ONLY_CLEAR')
    pp, ph = put(Path(req['plan_path']), plan)
    req['plan_sha256'] = ph; req['source_bindings'][pp] = ph
    rp, rh = put(Path(request_case['path']), req)
    owner.update(execution_request_sha256=rh); _, owner_sha = put(root/'owner_started.json', owner)
    _, worker_sha = put(root/'owned_worker.json', dict(owner_pid=200, launcher_pid=201,
        command=ex.worker_command(rp, rh, req)))
    cr.update(execution_request_sha256=rh, owner_started_sha256=owner_sha, owned_worker_sha256=worker_sha, plan_sha256=ph)
    _, request_sha = put(capture/'request.json', cr)
    result.update(plan_sha256=ph, request_sha256=request_sha)
    _, result_sha = put(capture/'result.json', result)
    if mutation != 'none':
        with pytest.raises(ValueError): audit.audit_capture(capture, result_sha256=result_sha)
    else:
        replay = audit.audit_capture(capture, result_sha256=result_sha)
        assert replay['counts'] == {'held_pre_render': 1} and replay['captured_count'] == 0
        assert replay['source_bindings'][decision_path] == decision_sha
        assert not replay['training_approved'] and not replay['independent_geometry_qualification']
