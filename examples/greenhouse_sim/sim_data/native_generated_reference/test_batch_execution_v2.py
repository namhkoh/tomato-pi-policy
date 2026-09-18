"""CPU boundary tests only; synthetic requests never authorize native capture."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from . import batch_execution_v2 as ex, batch_worker_v2 as worker
from . import batch_owner_v2 as owner, batch_audit_v2 as audit, batch_inventory_v2 as inventory


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return str(path.resolve()), ex.oc.sha256(path)


@pytest.fixture
def request_case(tmp_path):
    root = tmp_path/'immutable'
    data, data_sha = put(root/'source.json', {'UNIT_ONLY': True})
    source = dict(source_plant_id='seed73_full', target_id='seed73_full/SubStem_41')
    generated = dict(source_plant_id='seed73_full', split_group='seed73_full',
        target_id='seed73_full_cr_730201/SubStem_41', conservative_view_cap_group=source['target_id'])
    case = dict(source_row=source, generated_row=generated, target_id=generated['target_id'],
        conservative_view_cap_group=source['target_id'], views=[dict(candidate_id='SubStem_41_view_001')])
    plan = dict(schema='greenhouse.generated_original_native_batch_plan.v2',
        state='cpu_prepared_batch_pending_bound_native_producer',
        profile='reviewed_original_current_scene_mounted_views_reference56.v2',
        resolution=ex.oc.RESOLUTION, camera_path=ex.oc.HEAD_CAMERA, render_subframes_per_view=56,
        native_instance_backend='legacy', split='train', source_family='seed73_full', split_group='seed73_full',
        source_row=source, generated_row=generated, target_cases=[case], views_per_target=1, maximum_native_frames=1,
        scene_authority=dict(source_row=source, policy=ex.oc.SCENE_POLICY, package=str(root/'package')),
        variant_directory=str(root/'variant'), source_bindings={data: data_sha}, implementation_bindings=ex.implementation_bindings())
    plan.update({k: False for k in ('training_approved', 'source_cap_reset', 'native_launched', 'independent_geometry_qualification',
        'generated_native_qualification', 'physical_motion_commanded', 'paired_848_1696_proof',
        'hidden_cut_coordinates_executable', 'native_launch_supported')})
    pp, ph = put(root/'plan.json', plan)
    runtime = tmp_path/'runtime'
    for name in ('python.bat', 'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py',
        'exts/isaacsim.simulation_app/config/python_api.md'):
        put(runtime/name, {'UNIT_ONLY': 'never executable'})
    rb = ex.runtime_bindings(runtime/'python.bat')
    sp, sh = put(root/'storage/same_callback_qualification.json', {'UNIT_ONLY': 'not native qualification'})
    proof = dict(schema='greenhouse.compact_storage_qualification_binding.v1',
        path=sp, sha256=sh, bindings={sp: sh}, storage_backend='greenhouse.compact_native_sample.v1',
        training_approved=False, training_diversity_increment=0)
    req = dict(schema=ex.SCHEMA, worker_module=ex.WORKER, storage_backend=proof['storage_backend'],
        storage_qualification=proof, maximum_frames=1, render_subframes_per_view=56, automatic_retries=False,
        dependency_environment_fully_attested=False, implementation_bindings=ex.implementation_bindings(),
        plan_path=pp, plan_sha256=ph, output=str(tmp_path/'future_run'), isaac_python=str(runtime/'python.bat'),
        native_deps=str(tmp_path/'native_deps'), runtime_bindings=rb, **ex.FLAGS)
    req['source_bindings'] = ex.oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'],
        rb, proof['bindings'], {pp: ph})
    rp, rh = put(root/'execution.json', req)
    return dict(plan=plan, request=req, path=rp, sha=rh, root=root)


def test_real_preapp_worker_prefix_imports_no_usd_kit(request_case):
    repo = Path(__file__).resolve().parents[4]
    script = """import sys
sys.path.insert(0, sys.argv[1])
from sim_data.native_generated_reference import batch_worker_v2 as worker
from sim_physics import host_memory
host_memory.preflight=lambda:dict(checked=True,allowed=True,commit_headroom_bytes=20*2**30)
try:
 worker.main(['--request',sys.argv[2],'--request-sha256',sys.argv[3]])
except ValueError as error:
 assert 'Windows Isaac Python bootstrap' in str(error)
else:
 raise AssertionError('CPU must not be admitted to native capture')
assert not any(n.split('.')[0] in ('pxr','omni','carb','isaacsim') for n in sys.modules)
print('PREAPP_NO_USD_NO_KIT')
"""
    result = subprocess.run([sys.executable, '-B', '-c', script, str(repo/'examples/greenhouse_sim'),
        request_case['path'], request_case['sha']], capture_output=True, text=True, check=True)
    assert 'PREAPP_NO_USD_NO_KIT' in result.stdout


@pytest.mark.parametrize('key,value', [('maximum_frames', 73), ('maximum_frames', 2),
    ('render_subframes_per_view', 8), ('storage_backend', 'raw'), ('source_cap_reset', True),
    ('worker_module', 'other.worker'), ('automatic_retries', True)])
def test_changed_request_rejected_before_native_work(request_case, key, value):
    req = deepcopy(request_case['request']); req[key] = value
    path, sha = put(request_case['root']/'changed.json', req)
    with pytest.raises(ValueError): ex.preflight_request(path, sha)


def test_missing_new_worker_binding_rejected(request_case):
    req = deepcopy(request_case['request'])
    path = str(Path(worker.__file__).resolve())
    assert path in req['source_bindings']
    req['source_bindings'].pop(path)
    rp, rh = put(request_case['root']/'missing.json', req)
    with pytest.raises(ValueError, match='source closure'): ex.preflight_request(rp, rh)


def test_pose_membership_and_original_cap_cannot_expand(request_case):
    plan = deepcopy(request_case['plan'])
    plan['target_cases'][0]['conservative_view_cap_group'] = 'new_seed/renewed_budget'
    with pytest.raises(ValueError, match='lineage'): ex.structural_plan(plan)
    plan = deepcopy(request_case['plan']); plan['target_cases'][0]['views'][0]['candidate_id'] = '../outside'
    with pytest.raises(ValueError, match='unsafe'): ex.structural_plan(plan)


def test_source_mutation_is_not_hidden_by_saved_pins(request_case):
    put(request_case['root']/'source.json', {'UNIT_CHANGED': True})
    with pytest.raises(ValueError): ex.preflight_request(request_case['path'], request_case['sha'])


def test_no_app_no_scene_import():
    with pytest.raises(ValueError, match='SimulationApp'): worker.collect(None, Path('unused'), {}, {})


def test_declared_hash_proof_does_not_substitute_for_storage_qualification(request_case):
    with pytest.raises((ValueError, KeyError)): ex.checked_storage(request_case['request'])


@pytest.mark.parametrize('field', ['callback_sequence', 'camera_sha256', 'rgb_sha256', 'depth_sha256'])
def test_repeated_callback_rejected(field):
    a = dict(synchronization=dict(freshness=dict(callback_sequence=7, camera_sha256='c1', rgb_sha256='r1', depth_sha256='z1')))
    b = dict(synchronization=dict(freshness=dict(callback_sequence=14, camera_sha256='c2', rgb_sha256='r2', depth_sha256='z2')))
    assert audit.verify_sequence([a, b])
    b['synchronization']['freshness'][field] = a['synchronization']['freshness'][field]
    with pytest.raises(ValueError, match='Stale'): audit.verify_sequence([a, b])


def test_nonzero_owned_exit_cannot_publish(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(audit, 'audit_capture', lambda *a, **kw: called.append(True))
    with pytest.raises(ValueError, match='cleanly'): owner.complete(tmp_path, 'not_read', '0'*64, 1)
    assert called == [] and list(tmp_path.iterdir()) == []


def test_neutral_owner_passes_unchanged_raw_classifier():
    from ..native_dataset.reference_bridge_owner_v1 import verify_owner_row
    row = dict(ProcessId=123, ParentProcessId=1, CreationDate='UNIT_ONLY', Name='python.exe', ExecutablePath=sys.executable,
        CommandLine=subprocess.list2cmdline([sys.executable, '-B', '-m',
        'sim_data.native_dataset.original_reference_batch_owner_v2', '--request',
        r'D:\research\tomato-pi-policy\data\sim_data\diagnostics\original_reference_batch_request_v2.json',
        '--request-sha256', 'a'*64]))
    result = verify_owner_row(row, pid=123, executable=sys.executable)
    assert result['classification']['classification'] == 'unrelated_python'
    assert not result['current_pid_exception_used'] and not result['supervisor_exception_used']


@pytest.mark.parametrize('mutation', ['exit', 'schema', 'approval'])
def test_invalid_receipt_rejected_before_replay(tmp_path, monkeypatch, mutation):
    receipt = dict(schema=ex.RECEIPT, state='owned_exit0_and_independent_saved_native_replay',
        exit_code=0, process_history_independently_attested=False, **ex.FLAGS)
    if mutation == 'exit': receipt['exit_code'] = 1
    if mutation == 'schema': receipt['schema'] = 'greenhouse.generated_original_reference_owned_exit.v1'
    if mutation == 'approval': receipt['training_approved'] = True
    path, sha = put(tmp_path/'launcher_receipt.json', receipt)
    called = []
    monkeypatch.setattr(audit, 'audit_capture', lambda *a, **kw: called.append(True))
    with pytest.raises(ValueError): inventory.replay_receipt(path, sha)
    assert called == []
