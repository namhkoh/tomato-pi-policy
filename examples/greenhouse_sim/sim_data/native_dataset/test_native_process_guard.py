"""CPU-only launch-shape tests. No real process enumeration or process control."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import native_process_guard as guard


@pytest.fixture
def bridge(tmp_path):
    project = tmp_path / 'bridge with spaces'
    venv = project / '.venv'
    scripts = venv / 'Scripts'
    scripts.mkdir(parents=True)
    (project / 'pyproject.toml').write_text('[project]\nname="isaac-sim-mcp"\n'
        '[project.scripts]\nisaac-sim-mcp="isaac_mcp.server:main"\n', encoding='utf-8')
    base = tmp_path/'base_python'
    base.mkdir()
    (base/'python.exe').write_bytes(b'fixture base interpreter')
    uv = tmp_path/'uv.exe'
    uv.write_bytes(b'fixture uv executable')
    (venv / 'pyvenv.cfg').write_text('home = '+str(base)+'\n', encoding='utf-8')
    (scripts / 'python.exe').write_bytes(b'fixture venv redirector, not executable')
    (scripts / 'isaac-sim-mcp.exe').write_bytes(b'fixture console launcher, not executable')
    return guard.bridge_proof(venv, uv_executable=uv)


def row(exe, command=None, *, name=None, pid=42):
    return dict(ProcessId=pid, Name=name or Path(exe).name, ExecutablePath=str(exe),
                CommandLine=command if command is not None else '"'+str(exe)+'"')


def launch(proof):
    return row(proof['python'], '"'+proof['python']+'" "'+proof['entrypoint']+'"')


def classify(value, proof=None, **kwargs):
    return guard.classify_process(value, bridge_proofs=[proof] if proof else [], **kwargs)


def test_exact_bridge_python_and_console_shapes_only(bridge):
    for process in (launch(bridge), row(bridge['entrypoint'])):
        result = classify(process, bridge)
        assert not result['blocking'] and result['classification'] == 'standalone_mcp_bridge'
        assert result['reason'] and result['command_line_sha256']
    process = launch(bridge)
    process.update({k:v.upper() for k,v in process.items() if isinstance(v, str)})
    assert not classify(process, bridge)['blocking']


@pytest.mark.parametrize('extra', [' --help', ' --port 9000', ' -- -c x', ' -m isaacsim',
    ' ""', ' & kit.exe', ' ; extra', '\nignored', ' "another.py"'])
def test_extra_arguments_always_block(bridge, extra):
    process = launch(bridge); process['CommandLine'] += extra
    assert classify(process, bridge)['blocking']


@pytest.mark.parametrize('change', ['missing_exe','missing_command','different_venv','base_interpreter',
    'false_argv0','relative_entry','unquoted_spaces','unclosed_quote','dotdot','name_mismatch',
    'python_module','python_code','python_switch','console_extra','only_python','unc_exe'])
def test_bridge_impostor_or_ambiguous_launch_blocks(bridge, change):
    process = launch(bridge)
    if change == 'missing_exe': process['ExecutablePath'] = None
    elif change == 'missing_command': process['CommandLine'] = None
    elif change == 'different_venv': process['CommandLine'] = process['CommandLine'].replace('bridge with spaces','other bridge')
    elif change == 'base_interpreter': process['ExecutablePath'] = r'C:\Users\USER\miniconda3\python.exe'
    elif change == 'false_argv0': process['CommandLine'] = process['CommandLine'].replace(bridge['python'],r'C:\Python\python.exe')
    elif change == 'relative_entry': process['CommandLine'] = '"'+bridge['python']+'" isaac-sim-mcp.exe'
    elif change == 'unquoted_spaces': process['CommandLine'] = bridge['python']+' '+bridge['entrypoint']
    elif change == 'unclosed_quote': process['CommandLine'] = process['CommandLine'][:-1]
    elif change == 'dotdot': process['CommandLine'] = process['CommandLine'].replace('Scripts','Scripts\\..\\Scripts')
    elif change == 'name_mismatch': process['Name'] = 'pythonw.exe'
    elif change == 'python_module': process['CommandLine'] = '"'+bridge['python']+'" -m isaac_mcp.server'
    elif change == 'python_code': process['CommandLine'] = '"'+bridge['python']+'" -c "import isaacsim"'
    elif change == 'python_switch': process['CommandLine'] = '"'+bridge['python']+'" -I "'+bridge['entrypoint']+'"'
    elif change == 'console_extra': process = row(bridge['entrypoint'], '"'+bridge['entrypoint']+'" --help')
    elif change == 'only_python': process['CommandLine'] = '"'+bridge['python']+'"'
    elif change == 'unc_exe': process['ExecutablePath'] = '\\\\?\\'+bridge['python']
    result = classify(process, bridge)
    assert result['blocking'] and result['classification'] != 'standalone_mcp_bridge'


def test_bridge_requires_explicit_proof(bridge):
    assert classify(launch(bridge))['blocking']
    with pytest.raises(ValueError):
        guard.classify_processes([launch(bridge)], bridge_proofs=[bridge, bridge])


@pytest.mark.parametrize('name', ['kit.exe', 'KIT.EXE', 'isaac-sim.exe', 'isaacsim.exe'])
def test_kit_name_never_exempt_even_if_bridge_arguments(bridge, name):
    process = launch(bridge); process['Name'] = name
    result = classify(process, bridge)
    assert result['blocking'] and result['classification'] == 'native_kit'


@pytest.mark.parametrize('exe', [r'D:\isaac-sim-6.0.1\kit\python.exe',
    r'D:\isaac-sim-6.0.1\python.exe', r'C:\Tools\isaacsim\python.exe'])
def test_runtime_path_always_blocks(exe):
    result = classify(row(exe, name='python.exe'))
    assert result['blocking'] and result['classification'] == 'native_runtime'


def test_explicit_runtime_root_precedes_bridge_exception(bridge):
    assert classify(launch(bridge), bridge, native_roots=[bridge['venv']])['blocking']


@pytest.mark.parametrize('command', [r'python.exe -m isaacsim', r'python.exe -c "import omni.kit"',
    r'python.exe -m sim_data.native_generated_views',
    r'python.exe -m sim_data.native_dataset.compact_views',
    r'python.exe -m sim_data.native_capture_v5.bounded_views capture',
    r'python.exe -m sim_data.native_original_capture.collector',
    r'python.exe D:\data\native_same_callback_worker_20260915_v1.py'])
def test_unknown_relevant_python_blocks(command):
    result = classify(row(r'C:\Python\python.exe',command,name='python.exe'))
    assert result['blocking'] and result['classification'] == 'unknown_relevant_python'


def test_unrelated_python_and_system_process_are_nonblocking():
    assert not classify(row(r'C:\Python\python.exe','"C:\\Python\\python.exe" report.py',name='python.exe'))['blocking']
    assert not classify(dict(ProcessId=0,Name='System Idle Process',ExecutablePath=None,CommandLine=None))['blocking']


@pytest.mark.parametrize('field', ['Name','ExecutablePath','CommandLine'])
def test_inaccessible_python_metadata_blocks(field):
    process = row(r'C:\Python\python.exe',name='python.exe'); process[field] = None
    assert classify(process)['blocking']


@pytest.mark.parametrize('value', [None, {}, '[]', [dict(Name='kit.exe')],
    [dict(ProcessId=True,Name='kit.exe')], [dict(ProcessId=1,Name='kit.exe')]*2])
def test_malformed_inventory_raises_not_clear(value):
    with pytest.raises(ValueError): guard.classify_processes(value)


@pytest.mark.parametrize('field', ['entrypoint','python','bindings','base_interpreter_exempted'])
def test_mutated_bridge_proof_rejected(bridge, field):
    altered = deepcopy(bridge); altered[field] = 'tampered'
    with pytest.raises(ValueError): classify(launch(bridge), altered)


def test_bridge_file_mutation_rejected(bridge):
    Path(bridge['entrypoint']).write_bytes(b'changed fixture')
    with pytest.raises(ValueError): classify(launch(bridge), bridge)


def test_mutated_console_entrypoint_declaration_rejected(bridge):
    project = Path(bridge['venv']).parent/'pyproject.toml'
    project.write_text(project.read_text().replace('isaac_mcp.server:main','isaacsim:main'))
    with pytest.raises(ValueError): guard.bridge_proof(bridge['venv'])


def test_base_child_stays_blocking_even_with_matching_parent(bridge):
    parent = launch(bridge)
    child = deepcopy(parent); child.update(ProcessId=43,ParentProcessId=42,
        ExecutablePath=r'C:\Users\USER\miniconda3\python.exe')
    result = guard.classify_processes([parent,child],bridge_proofs=[bridge])
    assert [p['ProcessId'] for p in result['blockers']] == [43]
    assert not result['exclusive_launch_guaranteed'] and not result['snapshot_completeness_verified']


def test_native_processes_api_and_hidden_readonly_inventory(bridge, monkeypatch):
    processes = [launch(bridge),dict(ProcessId=9,Name='kit.exe',ExecutablePath=None,CommandLine=None),
        row(sys.executable,'"'+sys.executable+'" queue_native_same_callback.py',pid=os.getpid())]
    calls=[]
    def read_only(command, **kwargs):
        calls.append(command)
        assert command[:3] == ['powershell.exe','-NoProfile','-NonInteractive']
        assert 'Get-CimInstance' in command[-1] and 'ExecutablePath,CommandLine' in command[-1]
        assert kwargs['timeout'] == 20 and kwargs['creationflags'] == subprocess.CREATE_NO_WINDOW
        return json.dumps(processes)
    monkeypatch.setattr(guard.subprocess,'check_output',read_only)
    result=guard.native_processes(bridge_venvs=[bridge['venv']],uv_executable=bridge['uv_executable'])
    assert len(calls)==1 and [p['ProcessId'] for p in result]==[9] and result[0]['reason']


@pytest.mark.parametrize('payload',['null','{}','[]','not-json'])
def test_query_bad_output_never_clears(bridge,monkeypatch,payload):
    monkeypatch.setattr(guard.subprocess,'check_output',lambda *a,**k:payload)
    with pytest.raises(ValueError): guard.native_processes(bridge_venvs=[bridge['venv']],uv_executable=bridge['uv_executable'])


def test_query_failure_propagates(bridge,monkeypatch):
    def fail(*args,**kwargs): raise subprocess.TimeoutExpired('read-only-cim',20)
    monkeypatch.setattr(guard.subprocess,'check_output',fail)
    with pytest.raises(subprocess.TimeoutExpired): guard.native_processes(bridge_venvs=[bridge['venv']],uv_executable=bridge['uv_executable'])


def test_only_guard_module_is_implementation_pinned():
    bindings=guard.implementation_bindings()
    assert list(bindings)==[str(Path(guard.__file__).resolve())]
    assert all(len(h)==64 for h in bindings.values())


def bridge_chain(bridge):
    uv = row(bridge['uv_executable'], '"'+bridge['uv_executable']+'" run --project "'+
        str(Path(bridge['venv']).parent)+'" --frozen isaac-sim-mcp',pid=10)
    console = row(bridge['entrypoint'],pid=20)
    redirector = launch(bridge); redirector['ProcessId']=30
    base = launch(bridge); base.update(ProcessId=40,ExecutablePath=bridge['base_python'])
    result=[uv,console,redirector,base]
    for i, item in enumerate(result):
        item.update(ParentProcessId=0 if i==0 else result[i-1]['ProcessId'],CreationDate=f'/Date({1000+i})/')
    return result


def test_actual_host_bridge_shape_needs_entire_ordered_chain(bridge):
    result=guard.classify_processes(bridge_chain(bridge),bridge_proofs=[bridge])
    assert result['blockers']==[]
    assert result['counts']==dict(standalone_mcp_bridge_uv=1,standalone_mcp_bridge=2,standalone_mcp_bridge_base_child=1)


@pytest.mark.parametrize('change',['missing_parent','missing_console','missing_date','reused_parent_pid',
    'base_extra_arg','wrong_base','uv_extra_arg','wrong_uv','wrong_project','uv_newer_than_child'])
def test_chain_breaks_never_exempt_uv_or_base_broadly(bridge,change):
    rows=bridge_chain(bridge)
    if change=='missing_parent': rows.pop(2)
    elif change=='missing_console': rows.pop(1)
    elif change=='missing_date': rows[-1].pop('CreationDate')
    elif change=='reused_parent_pid': rows[2]['CreationDate']='/Date(2000)/'
    elif change=='base_extra_arg': rows[-1]['CommandLine']+=' --extra'
    elif change=='wrong_base': rows[-1]['ExecutablePath']=r'C:\OtherPython\python.exe'
    elif change=='uv_extra_arg': rows[0]['CommandLine']+=' --extra'
    elif change=='wrong_uv': rows[0]['ExecutablePath']=r'C:\Other\uv.exe'
    elif change=='wrong_project': rows[0]['CommandLine']=rows[0]['CommandLine'].replace('bridge with spaces','other project')
    elif change=='uv_newer_than_child': rows[0]['CreationDate']='/Date(2000)/'
    blockers=guard.classify_processes(rows,bridge_proofs=[bridge])['blockers']
    assert 10 in [p['ProcessId'] for p in blockers]
    if change not in ('uv_extra_arg','wrong_uv','wrong_project','uv_newer_than_child'):
        assert 40 in [p['ProcessId'] for p in blockers]


def test_native_runtime_base_child_still_blocks(bridge):
    result=guard.classify_processes(bridge_chain(bridge),bridge_proofs=[bridge],
        native_roots=[str(Path(bridge['base_python']).parent)])
    assert {10,40}<={r['ProcessId'] for r in result['blockers']}


def test_read_only_powershell_source_monitor_is_not_a_native_launcher():
    process=row(r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
        'powershell.exe -NoProfile -Command "Get-Content D:\\data\\queue_native_same_callback.py"',name='powershell.exe')
    assert not classify(process)['blocking']
    process['CommandLine']='powershell.exe -Command "D:\\isaac-sim-6.0.1\\python.bat worker.py"'
    assert classify(process)['blocking']


@pytest.mark.parametrize('fault',[None,'loaded_isaac','loaded_kit','exe_mismatch','runtime_exe','other_pid'])
def test_only_real_prestartup_cpu_caller_is_excluded(monkeypatch,fault):
    executable=r'C:\Python\python.exe'
    if fault=='runtime_exe': executable=r'D:\isaac-sim-6.0.1\python.exe'
    monkeypatch.setattr(guard.os,'getpid',lambda:9001)
    monkeypatch.setattr(guard.sys,'executable',executable)
    if fault=='loaded_isaac': monkeypatch.setitem(guard.sys.modules,'isaacsim',object())
    if fault=='loaded_kit': monkeypatch.setitem(guard.sys.modules,'omni.kit',object())
    process=row(executable,'python.exe queue_native_same_callback.py',name='python.exe',pid=9001)
    if fault=='exe_mismatch': process['ExecutablePath']=r'C:\Other\python.exe'
    if fault=='other_pid': process['ProcessId']=9002
    report=guard.classify_processes([process])
    if fault=='other_pid':
        with pytest.raises(ValueError): guard._exclude_current_cpu([process],report,guard.DEFAULT_NATIVE_ROOTS)
    else:
        guard._exclude_current_cpu([process],report,guard.DEFAULT_NATIVE_ROOTS)
        assert bool(report['blockers']) == (fault is not None)


def test_loaded_native_caller_blocks_even_without_command_marker(monkeypatch):
    monkeypatch.setitem(guard.sys.modules,'isaacsim',object())
    process=row(sys.executable,'python.exe generic.py',pid=os.getpid())
    report=guard.classify_processes([process])
    assert not report['blockers']
    guard._exclude_current_cpu([process],report,guard.DEFAULT_NATIVE_ROOTS)
    assert report['blockers'][0]['reason']=='current_process_not_proven_prestartup_cpu'


def test_other_coordinator_requires_exact_known_command_and_code_pin(bridge,tmp_path,monkeypatch):
    import hashlib
    script=tmp_path/'queue_native_same_callback.py'
    script.write_text('# CPU coordinator fixture only\n',encoding='utf-8')
    monkeypatch.setattr(guard,'_SUPERVISOR_SCRIPTS',(str(script),))
    argv=[bridge['base_python'],str(script),'--request','pinned.json']
    pins={str(script):hashlib.sha256(script.read_bytes()).hexdigest()}
    proof=guard.supervisor_proof(argv[0],argv,pins)
    process=row(argv[0],subprocess.list2cmdline(argv))
    assert classify(process)['blocking']
    assert not guard.classify_process(process,supervisor_proofs=[proof])['blocking']
    altered=deepcopy(process);altered['CommandLine']+=' --extra'
    assert guard.classify_process(altered,supervisor_proofs=[proof])['blocking']
    altered=deepcopy(process);altered['Name']='pythonw.exe'
    assert guard.classify_process(altered,supervisor_proofs=[proof])['blocking']
    altered=deepcopy(process);altered['Name']='kit.exe'
    assert guard.classify_process(altered,supervisor_proofs=[proof])['blocking']
    script.write_text('# changed coordinator\n',encoding='utf-8')
    with pytest.raises(ValueError): guard.classify_process(process,supervisor_proofs=[proof])
