"""Opt-in Windows launch-metadata classifier; no native launches or kills.

Pass a COMPLETE CIM snapshot with ProcessId, Name, ExecutablePath, CommandLine.
Only explicit, rechecked bridge proofs permit a narrow non-renderer exemption.
Base-interpreter children require pinned pyvenv home and a live, creation-ordered
console/redirector parent chain. UV requires its exact pinned executable/command
and that entire verified child chain. No generic Python/UV/ancestor exemption.

This is not a process lock or proof of imported modules. Resumers must serialize
ownership, keep reserves, recheck immediately before launch, and retain the
native worker's own single-Kit admission. No frozen queue is patched here.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import ntpath
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tomllib

SCHEMA = "greenhouse.native_process_guard.v2"
DEFAULT_BRIDGE_VENVS = (r'D:\AgroSim2Real\external\isaac-sim-mcp\.venv',)
DEFAULT_NATIVE_ROOTS = (r'D:\isaac-sim-6.0.1',)
DEFAULT_UV_EXECUTABLE = r'C:\Users\USER\.local\bin\uv.exe'
_REPO = Path(__file__).resolve().parents[4]
_SUPERVISOR_SCRIPTS = tuple(str(_REPO/'data/sim_data/diagnostics'/name) for name in (
    'queue_same_callback_native_20260916_v2.py', 'queue_v2_visual_controls_20260916_v2.py'))
_SUPERVISOR_MODULES = {
    'sim_data.native_capture_v4.campaign': _REPO/'examples/greenhouse_sim/sim_data/native_capture_v4/campaign.py',
    'sim_data.native_capture_v5.campaign_queue': _REPO/'examples/greenhouse_sim/sim_data/native_capture_v5/campaign_queue.py'}
_LOADED = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
_PYTHON = re.compile(r"^(?:python[0-9.w-]*|py|pypy[0-9.]*)\.exe$", re.I)
_RUNTIME_PART = re.compile(r"^(?:isaac-sim(?:-\d[\w.-]*)?|isaacsim|kit)$", re.I)
_RELEVANT = re.compile(r"isaac|omni[.\\/]|simulationapp|kit\.exe|"
    r"native_(?:same_callback|generated|greenhouse)|"
    r"sim_data[.\\/](?:native_dataset[.\\/](?:compact_views|dual_storage)|"
    r"native_original_capture[.\\/]collector)|"
    r"native_capture_v5[.\\/]bounded_views", re.I)


def _require(value, message):
    if not value:
        raise ValueError(message)


def implementation_bindings():
    """Explicit loaded guard code only; never glob-pin queues or other modules."""
    for path, expected in _LOADED.items():
        _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, 'Guard code changed')
    return dict(_LOADED)


def _path(value):
    """Strict local drive path; ambiguous aliases/escapes never earn exemption."""
    if not isinstance(value, str) or not value or any(c in value for c in '\x00\r\n"%'):
        return None
    value = value.replace('/', '\\')
    p = PureWindowsPath(value)
    if not re.fullmatch(r'[A-Za-z]:', p.drive) or not p.root:
        return None
    parts = value[3:].split('\\')
    if any(not x or x in ('.', '..') or x.endswith((' ', '.')) or ':' in x for x in parts):
        return None
    return ntpath.normcase(value)


def _argv(command):
    """Only simple quoted/unquoted Windows paths; not a general shell parser.

    Reject escapes, adjacent quoted fragments and unterminated quotes. A valid
    bridge command needs no switches or shell syntax, so ambiguity fails closed.
    """
    if not isinstance(command, str) or not command or any(c in command for c in '\x00\r\n'):
        return None
    tokens, position = [], 0
    pattern = re.compile(r'[ \t]*(?:"([^"\r\n]+)"|([^ \t"]+))(?=[ \t]|$)')
    while position < len(command):
        if not command[position:].strip(' \t'):
            break
        match = pattern.match(command, position)
        if not match:
            return None
        tokens.append(match.group(1) or match.group(2))
        position = match.end()
    return tokens


def bridge_proof(venv, *, uv_executable=None):
    """Read/hash one explicitly selected standalone MCP bridge installation.

    Local path/entrypoint proof, not a signature or running-code attestation.
    Never discovers installations. Base Python is ONLY a possible chain member.
    """
    venv = Path(venv).resolve(strict=True)
    _require(venv.name == '.venv', 'Explicit bridge .venv required')
    scripts = venv / 'Scripts'
    paths = [venv.parent / 'pyproject.toml', venv / 'pyvenv.cfg',
             scripts / 'python.exe', scripts / 'isaac-sim-mcp.exe']
    blobs, bindings = {}, {}
    for path in paths:
        _require(path.resolve(strict=True) == path and path.is_file(), 'No bridge file aliases')
        data = path.read_bytes()
        _require(bool(data), 'Empty bridge proof file')
        blobs[path.name] = data
        bindings[str(path)] = hashlib.sha256(data).hexdigest()
    project = tomllib.loads(blobs['pyproject.toml'].decode('utf-8'))['project']
    _require(project.get('name') == 'isaac-sim-mcp' and
        project.get('scripts', {}).get('isaac-sim-mcp') == 'isaac_mcp.server:main',
        'Different bridge project/console entrypoint')
    _require(re.search(r'^\s*home\s*=\s*\S.+$', blobs['pyvenv.cfg'].decode('utf-8'), re.M),
             'Missing venv configuration')
    homes = re.findall(r'^\s*home\s*=\s*(.+?)\s*$', blobs['pyvenv.cfg'].decode('utf-8'), re.M)
    _require(len(homes) == 1 and _path(homes[0]) is not None, 'One absolute pyvenv home required')
    base = Path(homes[0])/'python.exe'
    extra = [base] + ([Path(uv_executable)] if uv_executable is not None else [])
    for path in extra:
        _require(_path(str(path)) is not None and path.resolve(strict=True) == path and path.is_file(),
                 'Unambiguous existing bridge base/UV executable required')
        if path != base:
            _require(path.name.lower() == 'uv.exe', 'Exact UV executable required')
        bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema=SCHEMA+'.bridge', venv=str(venv), python=str(paths[2]),
        entrypoint=str(paths[3]), bindings=bindings,
        base_python=str(base), uv_executable=str(extra[1]) if len(extra)>1 else None,
        base_interpreter_exempted=False, running_code_attested=False)


def _proofs(proofs):
    _require(isinstance(proofs, (list, tuple)), 'Explicit bridge proof list required')
    seen = set()
    for proof in proofs:
        _require(isinstance(proof, dict) and proof == bridge_proof(proof.get('venv', ''),
            uv_executable=proof.get('uv_executable')),
                 'Bridge proof missing, changed or mutated')
        key = _path(proof['venv'])
        _require(key is not None and key not in seen, 'Duplicate/invalid bridge venv')
        seen.add(key)
    return proofs


def _native_path(path, roots):
    normalized = _path(path)
    if normalized is None:
        return False
    if any(normalized == root or normalized.startswith(root+'\\') for root in roots):
        return True
    return any(_RUNTIME_PART.fullmatch(part) for part in PureWindowsPath(normalized).parts[1:-1])


def _classify(row, proofs, roots):
    _require(isinstance(row, dict) and type(row.get('ProcessId')) is int and row['ProcessId'] >= 0,
             'Missing/invalid process identity')
    name, exe, command = row.get('Name'), row.get('ExecutablePath'), row.get('CommandLine')
    result = dict(ProcessId=row['ProcessId'], Name=name, ExecutablePath=exe,
        classification='unknown', blocking=True, reasons=[])
    if isinstance(command, str):
        result['command_line_sha256'] = hashlib.sha256(command.encode('utf-8')).hexdigest()
    def finish(kind, blocking, *reasons):
        return dict(result, classification=kind, blocking=blocking, reasons=list(reasons),
                    reason='; '.join(reasons))
    if not isinstance(name, str) or not name:
        return finish('unknown_metadata', True, 'missing_process_name')
    exe_name = ntpath.basename(exe).casefold() if isinstance(exe, str) else ''
    names = {name.casefold(), exe_name}
    if names & {'kit.exe', 'isaac-sim.exe', 'isaacsim.exe'}:
        return finish('native_kit', True, 'exact_native_executable_name')
    if _native_path(exe, roots):
        return finish('native_runtime', True, 'executable_inside_native_runtime')
    python = any(_PYTHON.fullmatch(n) for n in names)
    bridge = 'isaac-sim-mcp.exe' in names
    text = '\n'.join(x for x in (name, exe, command) if isinstance(x, str))
    relevant = bool(_RELEVANT.search(text)) or any(
        _path(exe) in (_path(proof['python']), _path(proof['entrypoint'])) for proof in proofs)
    if python or bridge:
        if not isinstance(exe, str) or _path(exe) is None or not isinstance(command, str) or not command.strip():
            return finish('unknown_relevant_python' if python else 'unknown_bridge', True,
                          'missing_or_ambiguous_executable_or_command_line')
        if name.casefold() != exe_name:
            return finish('unknown_relevant_python' if python else 'unknown_bridge', True, 'name_executable_mismatch')
        argv = _argv(command)
        for proof in proofs:
            entry, interpreter = _path(proof['entrypoint']), _path(proof['python'])
            actual = _path(exe)
            if actual == entry and argv is not None and len(argv) == 1 and _path(argv[0]) == entry:
                return finish('standalone_mcp_bridge', False, 'pinned_bridge_console_only_no_arguments')
            if actual == interpreter and argv is not None and len(argv) == 2 and \
                    [_path(a) for a in argv] == [interpreter, entry]:
                return finish('standalone_mcp_bridge', False, 'pinned_bridge_venv_python_sole_console_entrypoint')
        if relevant or bridge:
            return finish('unknown_relevant_python' if python else 'unknown_bridge', True,
                          'relevant_launch_not_exact_verified_standalone_bridge')
        if argv is None:
            return finish('unknown_relevant_python', True, 'unparseable_python_command_line')
        return finish('unrelated_python', False, 'no_native_launch_marker_in_available_metadata')
    # Reading a .py/log whose path contains 'isaac' is not launch evidence.
    # UV remains fail-closed until its exact command+child chain is verified.
    # Shell wrappers only block on executable/native-module launch evidence,
    # not a substring in arbitrary monitor text. No shell safety claim is made.
    shell_launch = bool(re.search(r'kit\.exe|isaac-sim[^\s"\']*[\\/](?:python|isaac-sim)\.(?:exe|bat)|'
        r'-m\s+(?:isaacsim|omni[.]|sim_data[.]native_)', command or '', re.I))
    if relevant and (names & {'uv.exe','uvx.exe'} or shell_launch):
        return finish('unknown_relevant_launcher', True, 'native_related_launch_metadata')
    return finish('unrelated_process', False, 'no_native_launch_marker_in_available_metadata')


def _born(row):
    value = row.get('CreationDate')
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'/Date\((\d+)(?:[+-]\d{4})?\)/', value)
    if match:
        return int(match[1])/1000
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return date.timestamp() if date.tzinfo is not None else None
    except ValueError:
        return None


def _child_of(child, parent):
    a, b = _born(parent), _born(child)
    return (type(child.get('ParentProcessId')) is int and child['ParentProcessId'] == parent['ProcessId']
            and child['ProcessId'] != parent['ProcessId'] and a is not None and b is not None and a <= b)


def _mark(row, classification, reason):
    row.update(classification=classification, blocking=False, reason=reason, reasons=[reason])


def _bridge_chains(rows, classified, proofs):
    by_pid = {r['ProcessId']:r for r in rows}
    decisions = {r['ProcessId']:r for r in classified}
    for proof in proofs:
        python, entry = _path(proof['python']), _path(proof['entrypoint'])
        chains = []
        for child in rows:
            decision = decisions[child['ProcessId']]
            if decision['classification'] != 'unknown_relevant_python' or \
                    _path(child.get('ExecutablePath')) != _path(proof['base_python']) or \
                    child.get('Name','').lower() != 'python.exe':
                continue
            argv = _argv(child.get('CommandLine'))
            if argv is None or [_path(a) for a in argv] != [python, entry]:
                continue
            parent = by_pid.get(child.get('ParentProcessId'), {})
            console = by_pid.get(parent.get('ParentProcessId'), {})
            if not parent or not console or not _child_of(child, parent) or not _child_of(parent, console):
                continue
            if (_path(parent.get('ExecutablePath')) != python or _path(console.get('ExecutablePath')) != entry
                or decisions[parent['ProcessId']]['classification'] != 'standalone_mcp_bridge'
                or decisions[console['ProcessId']]['classification'] != 'standalone_mcp_bridge'
                or _argv(parent.get('CommandLine')) != argv):
                continue
            _mark(decision, 'standalone_mcp_bridge_base_child', 'pinned_pyvenv_home_and_creation_ordered_console_redirector_chain')
            chains.append((console, parent, child))
        if proof['uv_executable'] is None:
            continue
        uv = _path(proof['uv_executable'])
        project = _path(str(Path(proof['venv']).parent))
        for launcher in rows:
            decision = decisions[launcher['ProcessId']]
            argv = _argv(launcher.get('CommandLine'))
            if (decision['classification'] != 'unknown_relevant_launcher' or launcher.get('Name','').lower() != 'uv.exe'
                or _path(launcher.get('ExecutablePath')) != uv or argv is None or len(argv) != 6
                or _path(argv[0]) != uv or argv[1:3] != ['run','--project'] or _path(argv[3]) != project
                or argv[4:] != ['--frozen','isaac-sim-mcp']):
                continue
            if any(_child_of(console, launcher) for console, _, _ in chains):
                _mark(decision, 'standalone_mcp_bridge_uv', 'pinned_uv_exact_frozen_console_command_and_verified_child_chain')


def supervisor_proof(executable, argv, code_bindings):
    """Explicit exact-command/code-pin opt-in for listed CPU coordinators only.

    Never called/discovered automatically. Other processes cannot claim to be
    this process; supervisors still cannot exempt a Kit/runtime executable.
    """
    _require(_path(str(executable)) is not None and isinstance(argv, list) and argv
        and all(isinstance(a,str) for a in argv) and _path(argv[0]) == _path(str(executable))
        and _PYTHON.fullmatch(ntpath.basename(str(executable))), 'Exact CPU executable/argv required')
    args = list(argv[1:])
    while args and args[0] in ('-B','-u'):
        args.pop(0)
    code = None
    if len(args)>=2 and args[0]=='-m' and args[1] in _SUPERVISOR_MODULES:
        code = _SUPERVISOR_MODULES[args[1]]
    elif args and _path(args[0]) in {_path(p) for p in _SUPERVISOR_SCRIPTS}:
        code = Path(args[0])
    _require(code is not None and isinstance(code_bindings,dict) and str(code) in code_bindings,
             'Known coordinator and explicit source pin required')
    for path, pin in code_bindings.items():
        _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == pin, 'Supervisor source changed')
    exe = Path(executable).resolve(strict=True)
    _require(str(exe).casefold() == str(executable).casefold(), 'No supervisor executable aliases')
    return dict(schema=SCHEMA+'.supervisor', executable=str(exe), argv=list(argv),
        executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), code_bindings=dict(code_bindings))


def _supervisors(rows, classified, proofs):
    _require(isinstance(proofs,(list,tuple)), 'Explicit supervisor proofs required')
    for proof in proofs:
        _require(proof == supervisor_proof(proof['executable'],proof['argv'],proof['code_bindings']),
                 'Supervisor proof changed')
        for row, decision in zip(rows,classified):
            if (decision['classification'] in ('unknown_relevant_python','unrelated_python')
                and _path(row.get('ExecutablePath')) == _path(proof['executable'])
                and row.get('Name','').casefold() == ntpath.basename(proof['executable']).casefold()
                and _argv(row.get('CommandLine')) == proof['argv']):
                _mark(decision,'pinned_cpu_supervisor','exact_known_coordinator_command_and_code_pins')


def classify_processes(rows, *, bridge_proofs=(), native_roots=(), supervisor_proofs=()):
    """Classify supplied complete CIM rows; unknown relevant processes block.

    Do not turn exceptions, missing CIM fields, or query failures into [].
    Recheck immediately before launch in the single owned serial coordinator.
    This returns no generic 'safe to launch' or training-approval claim.
    """
    _require(isinstance(rows, list), 'Complete process snapshot must be a list')
    proofs = _proofs(bridge_proofs)
    _require(isinstance(native_roots, (tuple, list)), 'Explicit native root list required')
    roots = [_path(str(root)) for root in native_roots]
    _require(all(root is not None for root in roots), 'Absolute unambiguous native root required')
    classified = [_classify(row, proofs, roots) for row in rows]
    _require(len({row['ProcessId'] for row in classified}) == len(classified), 'Duplicate process ID')
    _bridge_chains(rows, classified, proofs)
    _supervisors(rows, classified, supervisor_proofs)
    # A proof file changing during the snapshot must not publish an exemption.
    _proofs(proofs)
    blockers = [row for row in classified if row['blocking']]
    return dict(schema=SCHEMA, blockers=blockers, classifications=classified,
        counts=dict(Counter(row['classification'] for row in classified)),
        no_blockers_observed=not blockers, snapshot_completeness_verified=False,
        exclusive_launch_guaranteed=False, process_control_performed=False,
        bridge_proofs=list(proofs))


def classify_process(row, *, bridge_proofs=(), native_roots=(), supervisor_proofs=()):
    return classify_processes([row], bridge_proofs=bridge_proofs,
                              native_roots=native_roots, supervisor_proofs=supervisor_proofs)['classifications'][0]


def _exclude_current_cpu(rows, report, native_roots):
    current = [r for r in rows if r['ProcessId'] == os.getpid()]
    _require(len(current)==1, 'Current process absent/ambiguous in complete CIM snapshot')
    own = current[0]
    decision = next(r for r in report['classifications'] if r['ProcessId']==os.getpid())
    loaded = any(name.split('.')[0] in ('isaacsim','omni','carb') for name in sys.modules)
    if (not loaded and _path(own.get('ExecutablePath')) == _path(sys.executable)
        and _PYTHON.fullmatch(own.get('Name','')) and not _native_path(sys.executable, [_path(str(p)) for p in native_roots])
        and decision['classification'] not in ('native_kit','native_runtime','unknown_metadata')):
        _mark(decision, 'current_cpu_supervisor', 'actual_self_pid_executable_and_no_loaded_kit_isaac_before_startup')
    elif not decision['blocking']:
        reason = 'current_process_not_proven_prestartup_cpu'
        decision.update(classification='unknown_current_process',blocking=True,reason=reason,reasons=[reason])
    report['blockers'] = [r for r in report['classifications'] if r['blocking']]
    report['no_blockers_observed'] = not report['blockers']
    report['counts'] = dict(Counter(r['classification'] for r in report['classifications']))


def process_inventory(*, bridge_venvs=DEFAULT_BRIDGE_VENVS, native_roots=DEFAULT_NATIVE_ROOTS,
                      uv_executable=DEFAULT_UV_EXECUTABLE, supervisor_proofs=()):
    """Read-only CIM inventory; query/proof failures raise, NEVER become clear.

    The sole helper process is hidden PowerShell for Get-CimInstance. No bridge,
    renderer or worker is launched. Explicit defaults describe this inspected
    host; callers on another host must supply their own verified bridge roots.
    """
    _require(os.name == 'nt', 'Windows CIM inventory required')
    implementation_bindings()
    proofs = [bridge_proof(path, uv_executable=uv_executable) for path in bridge_venvs]
    command = ('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        'ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process '
        '-ErrorAction Stop | Select-Object ProcessId,ParentProcessId,CreationDate,Name,ExecutablePath,CommandLine)')
    raw = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
        text=True, encoding='utf-8', errors='strict', timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(raw.lstrip('\ufeff'))
    report = classify_processes(rows, bridge_proofs=proofs, native_roots=native_roots, supervisor_proofs=supervisor_proofs)
    _require(bool(rows), 'Empty all-process CIM snapshot is not credible')
    _exclude_current_cpu(rows, report, native_roots)
    implementation_bindings()
    return report


def native_processes(*, bridge_venvs=DEFAULT_BRIDGE_VENVS, native_roots=DEFAULT_NATIVE_ROOTS,
                     uv_executable=DEFAULT_UV_EXECUTABLE, supervisor_proofs=()):
    """Drop-in read-only blockers list. Empty means no blockers OBSERVED only."""
    return process_inventory(bridge_venvs=bridge_venvs, native_roots=native_roots,
        uv_executable=uv_executable, supervisor_proofs=supervisor_proofs)['blockers']
