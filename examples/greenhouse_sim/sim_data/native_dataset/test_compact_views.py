"""AST parity and CPU-only admission/storage tests. No native processes/frames."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import ModuleType

import pytest

from .. import native_generated_views as original
from ..depth_preview import sha256
from . import compact_views as compact
from . import compact_qualification as q
from .bundle import SampleReader, digest
from .test_capture_storage import synthetic_callback


def tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def dump(node):
    return ast.dump(node, include_attributes=False)


def test_frozen_collector_ast_only_storage_and_relative_import_changes():
    assert sha256(Path(original.__file__)) == q.ORIGINAL_COLLECTOR_SHA256
    expected = tree(original.collect)
    expected.name = '_collect'
    edits = []

    class StorageOnly(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level:
                node.level += 1
            if node.module == 'capture_visibility':
                node.names = [n for n in node.names if n.name != 'write_visibility']
                edits.append('visibility import')
            return node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == 'write_sample':
                    edits.append('write_sample')
                    return ast.parse('stored=write_compact_native_sample(folder,rgb,depth,valid,metadata,instances,'
                                     'mapping,catalogue,components,organs,mask,label,trace)').body[0]
                if node.value.func.id == 'write_visibility':
                    edits.append('write_visibility')
                    return None
                if node.value.func.id == 'write_json' and dump(node.value.args[0]) == dump(
                        ast.parse("folder/'supervision/label.json'", mode='eval').body):
                    edits.append('label write')
                    return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if dump(node) == dump(ast.parse("if trace is not None:write_json(folder/'supervision/query_trace.json',trace)").body[0]):
                edits.append('trace write')
                return None
            return self.generic_visit(node)

        def visit_keyword(self, node):
            if node.arg in ('sample_sha256', 'label_sha256', 'query_trace'):
                edits.append(node.arg)
                node.value = ast.parse('stored['+repr(node.arg)+']', mode='eval').body
            return self.generic_visit(node)

    expected = StorageOnly().visit(expected)
    expected.body.insert(0, ast.parse(
        "require(not profile_render,'Compact storage excludes profile_render review PNGs')").body[0])
    assert sorted(edits) == sorted(['visibility import', 'write_sample', 'write_visibility', 'label write',
                                   'trace write', 'sample_sha256', 'label_sha256', 'query_trace'])
    assert dump(tree(compact._collect)) == dump(expected)
    imports = [n.module for n in ast.walk(expected) if isinstance(n, ast.ImportFrom)]
    assert 'native_view_pose' in imports
    assert not any(n and 'native_capture_v2' in n for n in imports)


def test_main_ast_preserves_original_native_gates_with_explicit_guard_additions():
    # Exact insertion whitelist: an unlisted physics, camera, plan, admission,
    # renderer, cleanup or output-state change fails this whole-function test.
    expected = textwrap.dedent(inspect.getsource(original.main)).replace('from .', 'from ..')
    changes = [
        ("    p.add_argument('--output',type=Path,required=True)", """    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--storage-qualification',type=Path,required=True)
    p.add_argument('--storage-qualification-sha256',required=True,
                   help='Caller-pinned original same_callback_qualification.json SHA256')"""),
        ('    a=p.parse_args()', """    a=p.parse_args()
    require(not a.profile_render,'Compact storage excludes profile_render review PNGs')
    qualification=compact_qualification.check_qualification(
        a.storage_qualification,expected_sha256=a.storage_qualification_sha256)
    compact_bindings=implementation_bindings()"""),
        ("    source_roots.append(Path(read_json(base['source_collection_plan'])['package']).resolve())",
         """    source_roots.append(Path(read_json(base['source_collection_plan'])['package']).resolve())
    source_roots.append(Path(qualification['path']).parent)"""),
        ('    output.mkdir(parents=True)', """    compact_qualification.verify_checked_qualification(qualification)
    verify_bindings(compact_bindings)
    output.mkdir(parents=True)"""),
        ('        process_admission=admission,training_started=False,automatic_retries=False,',
         """        process_admission=admission,training_started=False,automatic_retries=False,
        compact_native_storage=True,storage_backend=STORAGE_BACKEND,
        compact_implementation_bindings=compact_bindings,storage_qualification=qualification,"""),
        ('        result=collect(app,output,plan,base,profile_render=a.profile_render,',
         """        result=collect(app,output,plan,base,profile_render=a.profile_render,
                       storage_qualification=a.storage_qualification,
                       storage_qualification_sha256=a.storage_qualification_sha256,"""),
        ("        write_json(output/'result.json',result)", """        require(result['compact_implementation_bindings']==compact_bindings
                and result['storage_qualification']==qualification,'Changed compact handoff binding')
        compact_qualification.verify_checked_qualification(qualification)
        verify_bindings(compact_bindings)
        write_json(output/'result.json',result)"""),
    ]
    for old, new in changes:
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    assert dump(tree(compact.main)) == dump(ast.parse(expected).body[0])


def test_actual_storage_expressions_use_returned_logical_hashes(tmp_path):
    # Execute only the collector's two storage/bookkeeping expressions against
    # synthetic CPU arrays; never execute scene, rendering, label or pose code.
    callback = synthetic_callback()
    folder = tmp_path/'synthetic'
    storage = next(n for n in ast.walk(tree(compact._collect)) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'stored' for t in n.targets))
    decision = next(n for n in ast.walk(tree(compact._collect)) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and dump(n.value.func) == dump(ast.parse('decision.update', mode='eval').body)
        and any(k.arg == 'sample_sha256' for k in n.value.keywords))
    namespace = dict(callback, folder=folder, mask=callback['target_mask'], decision={},
        write_compact_native_sample=compact.write_compact_native_sample, render_settings={},
        view_started=time.perf_counter(), time=time)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[storage, decision], type_ignores=[])),
                 '<synthetic-storage-only>', 'exec'), namespace)
    row = namespace['decision']
    reader = SampleReader(folder)
    assert row['sample_sha256'] == digest(reader.read('sample.json'))
    assert row['label_sha256'] == digest(reader.read('supervision/label.json'))
    assert row['query_trace'] == callback['trace']
    assert not (folder/'sample.json').exists()
    assert not (folder/'supervision/label.json').exists()
    assert not list(folder.rglob('review'))


@pytest.mark.parametrize('mode', ['missing', 'bad_pin', 'profile'])
def test_invalid_storage_gate_before_native_import_or_output(tmp_path, monkeypatch, mode):
    output = tmp_path/'not-created'
    args = ['compact_views', '--batch-plan', str(tmp_path/'no-plan'), '--output', str(output),
            '--storage-qualification', str(tmp_path/'same_callback_qualification.json'),
            '--storage-qualification-sha256', 'a'*64]
    if mode == 'profile':
        args.append('--profile-render')
    elif mode == 'bad_pin':
        args[-1] = 'not-a-hash'
    monkeypatch.setattr(sys, 'argv', args)
    before = set(sys.modules)
    with pytest.raises((ValueError, FileNotFoundError)):
        compact.main()
    assert not output.exists()
    assert not any(n == 'isaacsim' or n == 'pxr' or n.startswith('omni.') for n in set(sys.modules)-before)


def test_missing_explicit_opt_in_cli_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['compact_views', '--plan', str(tmp_path/'p'), '--output', str(tmp_path/'out')])
    with pytest.raises(SystemExit) as error:
        compact.main()
    assert error.value.code == 2 and not (tmp_path/'out').exists()


def test_public_collect_cannot_bypass_admission(tmp_path):
    with pytest.raises(FileNotFoundError):
        compact.collect(None, tmp_path/'out', {}, {},
            storage_qualification=tmp_path/'same_callback_qualification.json', storage_qualification_sha256='a'*64)
    with pytest.raises(ValueError, match='profile_render'):
        compact.collect(None, tmp_path/'out', {}, {}, profile_render=True,
            storage_qualification=tmp_path/'same_callback_qualification.json', storage_qualification_sha256='a'*64)
    with pytest.raises(ValueError, match='profile_render'):
        compact._collect(None, tmp_path/'out', {}, {}, profile_render=True)
    assert not (tmp_path/'out').exists()


@pytest.fixture
def main_fixture(tmp_path, monkeypatch):
    """Explicit simulator/plan stubs; production gates are AST-compared above."""
    events = []
    proof_path = tmp_path/'proof/same_callback_qualification.json'
    proof_path.parent.mkdir()
    proof_path.write_text('{}', encoding='utf-8')
    qualification = dict(path=str(proof_path), sha256=sha256(proof_path), fixture_only=True)
    def qualify(path, *, expected_sha256):
        assert Path(path) == proof_path and expected_sha256 == qualification['sha256']
        events.append('qualify')
        return deepcopy(qualification)
    def recheck(value):
        assert value == qualification
        events.append('recheck')
        return True
    monkeypatch.setattr(q, 'check_qualification', qualify)
    monkeypatch.setattr(q, 'verify_checked_qualification', recheck)
    collection = tmp_path/'collection.json'
    collection.write_text(json.dumps(dict(package=str(tmp_path/'source-package'))), encoding='utf-8')
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(dict(implementation_bindings={str(collection): sha256(collection)})), encoding='utf-8')
    base = dict(source_capture=str(tmp_path/'source-capture'), variant_directory=str(tmp_path/'variant'),
                prerequisite_directory=str(tmp_path/'prerequisite'), source_collection_plan=str(collection))
    def check(plan, *, replay_geometry):
        events.append('full-replay' if replay_geometry else 'hash-check')
        return base
    modules = {}
    for name in ('sim_data.native_multitarget_plan', 'sim_physics.host_memory', 'sim_data.native_generated_pair', 'isaacsim'):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules['sim_data.native_multitarget_plan'].check = check
    modules['sim_physics.host_memory'].preflight = lambda: dict(allowed=True)
    modules['sim_data.native_generated_pair'].windows_worker_admission = lambda value: dict(fixture_only=True)
    class App:
        def __init__(self, options):
            events.append('app')
            assert options == dict(headless=True, width=1696, height=816, multi_gpu=False,
                renderer='RaytracedLighting', sync_loads=False, disable_viewport_updates=True,
                extra_args=['--/app/settings/persistent=false'])
        def close(self):
            events.append('close')
    modules['isaacsim'].SimulationApp = App
    def collect(app, output, plan, base, **kwargs):
        events.append('collect')
        assert kwargs == dict(profile_render=False, instance_backend='fast', render_budget=compact.TRIAL)
        return dict(state='synthetic_only', training_approved=False)
    monkeypatch.setattr(compact, '_collect', collect)
    output = tmp_path/'new-output'
    args = ['compact_views', '--batch-plan', str(plan_path), '--output', str(output),
        '--storage-qualification', str(proof_path), '--storage-qualification-sha256', sha256(proof_path),
        '--instance-backend', 'fast', '--render-budget', compact.TRIAL]
    monkeypatch.setattr(sys, 'argv', args)
    return output, events, qualification, modules


def test_main_request_result_bind_same_proof_and_own_code(main_fixture):
    output, events, qualification, _ = main_fixture
    compact.main()
    assert events == ['qualify', 'hash-check', 'recheck', 'app', 'full-replay', 'qualify',
                      'collect', 'recheck', 'recheck', 'close']
    request = json.loads((output/'request.json').read_text())
    result = json.loads((output/'result.json').read_text())
    for doc in (request, result):
        assert doc['storage_backend'] == compact.STORAGE_BACKEND and doc['compact_native_storage'] is True
        assert doc['compact_implementation_bindings'] == compact.implementation_bindings()
        assert doc['storage_qualification'] == qualification
    assert request['native_instance_backend'] == 'fast'
    assert request['render_budget_profile'] == compact.TRIAL
    assert request['training_started'] is False and result['training_approved'] is False


@pytest.mark.parametrize('mode', ['existing', 'proof_root', 'source_root', 'memory'])
def test_output_and_memory_gates_remain_before_app(main_fixture, monkeypatch, mode):
    output, events, qualification, modules = main_fixture
    if mode == 'existing':
        output.mkdir()
    elif mode == 'proof_root':
        sys.argv[sys.argv.index('--output')+1] = str(Path(qualification['path']).parent/'bad')
    elif mode == 'source_root':
        sys.argv[sys.argv.index('--output')+1] = str(output.parent/'variant/bad')
    else:
        modules['sim_physics.host_memory'].preflight = lambda: dict(allowed=False)
    with pytest.raises(ValueError):
        compact.main()
    assert 'app' not in events and not (output/'request.json').exists()


def test_failure_and_close_preserved_without_completed_result(main_fixture, monkeypatch):
    output, events, _, _ = main_fixture
    def failed(*args, **kwargs):
        raise ValueError('synthetic capture failure')
    monkeypatch.setattr(compact, '_collect', failed)
    with pytest.raises(ValueError, match='synthetic capture failure'):
        compact.main()
    assert (output/'request.json').exists() and (output/'failure.json').exists()
    assert not (output/'result.json').exists() and events[-1] == 'close'


def test_no_runtime_rebinding_and_original_code_unchanged():
    for path in (Path(compact.__file__), Path(q.__file__)):
        source = path.read_text(encoding='utf-8')
        assert 'monkeypatch' not in source and 'sys.modules' not in source
        assert 'exec(' not in source and 'setattr(' not in source
    assert sha256(Path(original.__file__)) == q.ORIGINAL_COLLECTOR_SHA256
    bindings = compact.implementation_bindings()
    assert bindings[str(Path(compact.__file__).resolve())] == sha256(Path(compact.__file__))
    assert bindings[str(Path(q.__file__).resolve())] == sha256(Path(q.__file__))


def test_cli_help_is_cpu_only():
    code = """import runpy,sys
sys.argv=['compact_views','--help']
try: runpy.run_module('sim_data.native_dataset.compact_views',run_name='__main__')
except SystemExit as exc: assert exc.code == 0
assert not any(n=='pxr' or n.startswith('pxr.') or n=='isaacsim' or n.startswith('omni.') for n in sys.modules)
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--storage-qualification-sha256' in result.stdout
