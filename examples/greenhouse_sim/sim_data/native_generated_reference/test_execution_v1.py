"""CPU-only synthetic boundary tests; never launch or substitute for native proof."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from . import execution_v1 as ex, worker_v1 as worker, owner_v1 as owner, audit_v1 as audit


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return str(path),ex.oc.sha256(path)


@pytest.fixture
def request_case(tmp_path):
    root=tmp_path/'immutable';data,data_sha=put(root/'source.json',{'UNIT_ONLY':True})
    source=dict(source_plant_id='seed73_full',target_id='seed73_full/SubStem_41')
    plan=dict(schema='greenhouse.generated_from_original_reference_plan.v1',
        state='cpu_prepared_scene_only_no_native_producer_bound',profile='one_target_exact_pose_original_generated_56.v1',
        execution=dict(native_launch_supported=False,producer_module=None,producer_bindings=None,
            owned_launcher_receipt=None,postexit_audit_adapter=None),modes=ex.MODES,sample_count_limit=2,
        resolution=ex.oc.RESOLUTION,camera_path=ex.oc.HEAD_CAMERA,render_subframes_per_view=56,
        native_instance_backend='legacy',pose_changes_between_modes=False,split='train',source_family='seed73_full',
        source_row=source,split_group='seed73_full',conservative_view_cap_group=source['target_id'],
        generated_row=dict(conservative_view_cap_group=source['target_id']),
        scene_authority=dict(source_row=source,policy=ex.oc.SCENE_POLICY,package=str(root/'package')),
        variant_directory=str(root/'variant'),source_bindings={data:data_sha},implementation_bindings=ex.implementation_bindings())
    plan.update({k:False for k in ('training_approved','source_cap_reset','native_launched','generated_native_qualification',
        'independent_target_novelty_approved','physical_motion_commanded','paired_848_1696_proof')})
    pp,ph=put(root/'plan.json',plan)
    runtime=tmp_path/'runtime'
    for name in ('python.bat','exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py',
                 'exts/isaacsim.simulation_app/config/python_api.md'):
        put(runtime/name,{'UNIT_ONLY':'never executable'})
    rb=ex.runtime_bindings(runtime/'python.bat')
    req=dict(schema=ex.SCHEMA,worker_module=ex.WORKER,storage='raw_native_v1',modes=list(ex.MODES),maximum_frames=2,
        render_subframes_per_view=56,automatic_retries=False,implementation_bindings=ex.implementation_bindings(),
        plan_path=pp,plan_sha256=ph,output=str(tmp_path/'future_run'),isaac_python=str(runtime/'python.bat'),
        native_deps=str(tmp_path/'native_deps'),runtime_bindings=rb,**ex.FLAGS)
    req['source_bindings']=ex.oc.merge_bindings(plan['source_bindings'],plan['implementation_bindings'],rb,{pp:ph})
    rp,rh=put(root/'execution.json',req)
    return dict(plan=plan,request=req,path=rp,sha=rh,root=root)


def test_worker_import_and_preflight_are_usd_kit_free(request_case):
    repo=Path(__file__).resolve().parents[4]
    script="""import sys
sys.path.insert(0,sys.argv[1])
from sim_data.native_generated_reference import worker_v1 as w
from sim_physics import host_memory
# Only memory availability is synthetic. Execute the real worker.main prefix,
# including actual heavy imports and the unchanged native admission helper.
host_memory.preflight=lambda:dict(checked=True,allowed=True,commit_headroom_bytes=20*2**30)
try:
 w.main(['--request',sys.argv[2],'--request-sha256',sys.argv[3]])
except ValueError as error:
 assert 'Windows Isaac Python bootstrap' in str(error)
else:
 raise AssertionError('CPU interpreter must not be admitted as native Kit')
assert not any(n.split('.')[0] in ('pxr','omni','carb','isaacsim') for n in sys.modules)
print('PREAPP_NO_USD_NO_KIT')
"""
    result=subprocess.run([sys.executable,'-B','-c',script,str(repo/'examples/greenhouse_sim'),
        request_case['path'],request_case['sha']],capture_output=True,text=True,check=True)
    assert 'PREAPP_NO_USD_NO_KIT' in result.stdout


@pytest.mark.parametrize('key,value',[('maximum_frames',3),('render_subframes_per_view',8),('storage','compact'),
    ('modes',['generated_variant','original_control']),('source_cap_reset',True),('worker_module','other.worker')])
def test_request_cannot_change_bounded_execution(request_case,key,value):
    req=deepcopy(request_case['request']);req[key]=value
    path,sha=put(request_case['root']/'mutated.json',req)
    with pytest.raises(ValueError):ex.preflight_request(path,sha)


def test_incomplete_context_bindings_rejected(request_case):
    req=deepcopy(request_case['request'])
    context=str(Path(worker.__file__).with_name('scene.py'))
    assert context in req['implementation_bindings'] and context in req['source_bindings']
    req['source_bindings'].pop(context)
    path,sha=put(request_case['root']/'missing.json',req)
    with pytest.raises(ValueError,match='source closure'):ex.preflight_request(path,sha)


def test_source_mutation_rejected(request_case):
    put(request_case['root']/'source.json',{'UNIT_MUTATION':True})
    with pytest.raises(ValueError):ex.preflight_request(request_case['path'],request_case['sha'])


def test_running_guard_before_any_scene_import(monkeypatch):
    with pytest.raises(ValueError,match='SimulationApp'):worker.collect(None,Path('unused'),{})
    class App:
        def is_running(self):return False
    monkeypatch.setitem(sys.modules,'isaacsim',types.SimpleNamespace(SimulationApp=App))
    with pytest.raises(ValueError,match='running'):ex.require_running_app(App())
    App.is_running=lambda self:True
    ex.require_running_app(App())


def test_all_new_context_owner_review_code_pinned():
    pins=ex.implementation_bindings();root=Path(worker.__file__).parent
    for name in ('__init__.py','prepare.py','scene.py','execution_v1.py','worker_v1.py','audit_v1.py','owner_v1.py','inventory_v1.py'):
        assert pins[str(root/name)]==ex.oc.sha256(root/name)
    assert not any(p.endswith('serial_phases.py') or Path(p).name.startswith('test_') for p in pins)


def test_budget_app_order_cleanup_and_no_companion_ast():
    tree=ast.parse(inspect.getsource(worker));collect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect')
    assert ast.unparse(collect.body[0])=='ex.require_running_app(app)'
    calls=[n for n in ast.walk(collect) if isinstance(n,ast.Call)]
    steps=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='step_payload']
    assert len(steps)==2 and all(ast.literal_eval(n.keywords[0].value)==8 for n in steps)
    warm=next(n for n in ast.walk(collect) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(6)')
    assert any(n in steps for n in ast.walk(warm))
    assert sum(isinstance(n.func,ast.Attribute) and n.func.attr=='render_product' for n in calls)==1
    assert not any(isinstance(n,ast.Name) and n.id=='old_manifest' for n in ast.walk(tree))
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('Popen','run') for n in calls)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    app_line=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='SimulationApp')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='check_plan' and n.lineno<app_line for n in ast.walk(main))


def test_pair_keeps_unqualified_visibility_without_approval():
    a=dict(calibration={'UNIT':1},robot_snapshot={},scene_counts={},lighting={},scene_evidence={},
        native_target_pixels=2,old_plant_native_pixels=2,
        synchronization=dict(static_guard='a'*64,freshness=dict(callback_sequence=7,rgb_sha256='r1',depth_sha256='z1')))
    b=deepcopy(a);b.update(native_target_pixels=0,old_plant_native_pixels=0)
    b['synchronization']=dict(static_guard='b'*64,freshness=dict(callback_sequence=14,rgb_sha256='r2',depth_sha256='z2'))
    # Use real frozen camera equality with matching minimal required calibration.
    from .. import native_greenhouse_pair as pair
    cal={k:1 for k in ('camera_path','resolution','intrinsics','focal_length_mm','apertures_mm',
        'aperture_offsets_mm','clipping_range_m','depth_convention','crop_resize')}
    cal['camera_to_world_usd_row_vectors']=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    a['calibration']=b['calibration']=cal
    value=audit.pair_evidence([a,b])
    assert value['complete_pair'] and not value['native_change_observed'] and not value['training_approved']
    b['synchronization']['freshness']['callback_sequence']=7
    with pytest.raises(ValueError):audit.pair_evidence([a,b])


def test_nonzero_exit_never_replays_or_publishes(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setattr(audit,'audit_capture',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='cleanly'):owner.complete(tmp_path,'not_read','0'*64,1)
    assert not called and not list(tmp_path.iterdir())


def test_interrupted_owner_waits_for_owned_child(tmp_path,monkeypatch):
    events=[]
    class Child:
        pid=123
        done=False
        def wait(self):
            events.append('wait')
            if len(events)==1:raise KeyboardInterrupt()
            self.done=True;return 1
        def poll(self):return 1 if self.done else None
    child=Child()
    monkeypatch.setattr(owner.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(owner.subprocess,'CREATE_NO_WINDOW',0,raising=False)
    with pytest.raises(KeyboardInterrupt):owner._run_child(['UNIT_NOT_LAUNCHED'],tmp_path,{})
    assert child.done and events==['wait','wait']


def test_runtime_override_environment_removed(request_case,monkeypatch):
    monkeypatch.setenv('PYTHONEXE','UNAUTHORIZED');monkeypatch.setenv('PYTHONHOME','UNAUTHORIZED')
    env=owner._environment(request_case['request'])
    assert 'PYTHONEXE' not in env and 'PYTHONHOME' not in env and env['OPENBLAS_NUM_THREADS']=='1'


def test_raw_guard_blocks_old_owner_and_accepts_neutral_command_without_exemptions():
    from ..native_dataset import reference_bridge_owner_v1 as neutral
    row=dict(ProcessId=123,ParentProcessId=1,CreationDate='2026-09-16T00:00:00+00:00',Name='python.exe',ExecutablePath=sys.executable)
    row['CommandLine']=subprocess.list2cmdline([sys.executable,'-B','-m','sim_data.native_generated_reference.owner_v1',
        'run','--request',r'D:\research\tomato-pi-policy\data\sim_data\diagnostics\generated_original_seed73_bridge_request_20260916_v2.json',
        '--request-sha256','a'*64])
    with pytest.raises(ValueError,match='blocks unchanged raw guard'):
        neutral.verify_owner_row(row,pid=123,executable=sys.executable)
    row['CommandLine']=subprocess.list2cmdline([sys.executable,'-B','-m','sim_data.native_dataset.reference_bridge_owner_v1',
        '--request',r'D:\research\tomato-pi-policy\data\sim_data\diagnostics\generated_original_seed73_bridge_request_20260916_v2.json',
        '--request-sha256','a'*64])
    result=neutral.verify_owner_row(row,pid=123,executable=sys.executable)
    assert result['classification']['classification']=='unrelated_python' and not result['current_pid_exception_used']
    assert not result['supervisor_exception_used']


@pytest.mark.parametrize('mutation',['none','capture','command','exit','code','owner','request'])
def test_inventory_requires_exact_owned_completion_before_replay(request_case,monkeypatch,mutation):
    from . import inventory_v1 as inventory
    req=request_case['request'];root=Path(req['output']);root.mkdir()
    worker_doc=dict(command=ex.worker_command(request_case['path'],request_case['sha'],req),owner_pid=100,launcher_pid=101)
    if mutation=='command':worker_doc['command'][-1]='0'*64
    _,worker_sha=put(root/'owned_worker.json',worker_doc)
    started=dict(owner_pid=200 if mutation=='owner' else 100,execution_request_path=request_case['path'],
        execution_request_sha256=request_case['sha'])
    _,started_sha=put(root/'owner_started.json',started)
    _,exit_sha=put(root/'exit.json',dict(exit_code=1 if mutation=='exit' else 0,owned_worker_sha256=worker_sha))
    _,audit_sha=put(root/'postexit_audit.json',{'UNIT_ONLY':True})
    receipt=dict(schema=ex.RECEIPT,state='owned_exit0_and_independent_saved_native_replay',exit_code=0,
        execution_request_path=request_case['path'],execution_request_sha256=request_case['sha'],
        capture=str(root/('other_capture' if mutation=='capture' else 'capture')),
        source_bindings=req['source_bindings'],implementation_bindings=deepcopy(req['implementation_bindings']),
        owned_worker_sha256=worker_sha,owner_started_sha256=started_sha,exit_sha256=exit_sha,
        postexit_audit_sha256=audit_sha,result_sha256='1'*64,request_sha256='2'*64,**ex.FLAGS)
    if mutation=='code':receipt['implementation_bindings'].pop(next(iter(receipt['implementation_bindings'])))
    if mutation=='request':receipt['execution_request_sha256']='0'*64
    path,sha=put(root/'launcher_receipt.json',receipt)
    called=[]
    class ReplayReached(Exception):pass
    def replay(*a,**k):called.append(True);raise ReplayReached('UNIT boundary only')
    monkeypatch.setattr(audit,'audit_capture',replay)
    with pytest.raises(ReplayReached if mutation=='none' else ValueError):
        inventory.build_inventory(path,receipt_sha256=sha)
    assert bool(called)==(mutation=='none')
