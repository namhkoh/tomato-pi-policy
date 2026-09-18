"""Exact finite-two admission and monitoring; no launch or process control."""
from pathlib import Path
import copy,json,shutil,subprocess
from . import native848_bulk_sibling_gate_v5 as gate
from . import native848_owned_retirement_v3 as retirement
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256

SCHEMA='greenhouse.native848_fully_labeled_two_admission.v1'
RAM_MARGIN=4*2**30
VRAM_MARGIN=2048
REFERENCE=Path('D:/research/tomato-pi-policy/data/sim_data/diagnostics/native848_full144_two_worker_resource_audit_20260917_v1/123937_942749.json')
REFERENCE_SHA='a2ef03c01b4220e8239047d6e5da9676c5be45df707208b1ae0ccb58dacf46cf'


def baseline():
    require(sha256(REFERENCE)==REFERENCE_SHA,'Actual full-population resource reference changed')
    value=read_json(REFERENCE);verify_bindings(value['source_bindings'])
    require(value['schema']=='greenhouse.full144_two_worker_resource_sample.v1'
        and value['OS_peak_counters_are_since_exact_process_birth'] is True
        and value['second_worker_admitted'] is False,'Actual single-worker resource sample required')
    return value


def process_memory(identity):
    """Read OS counters, binding the query to exact PID, executable and birth."""
    pid=gate.identity(identity)['ProcessId']
    query="$p=Get-Process -Id "+str(pid)+" -ErrorAction Stop; [pscustomobject]@{pid=$p.Id;exe=$p.Path;birth_ms=([DateTimeOffset]$p.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds();working_set_bytes=$p.WorkingSet64;peak_working_set_bytes=$p.PeakWorkingSet64;private_bytes=$p.PrivateMemorySize64;paged_bytes=$p.PagedMemorySize64;peak_paged_bytes=$p.PeakPagedMemorySize64} | ConvertTo-Json -Compress"
    value=json.loads(subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',query],
        text=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW))
    require(value['pid']==pid and Path(value['exe']).resolve()==Path(identity['ExecutablePath']).resolve()
        and identity['CreationDate']=='/Date('+str(value['birth_ms'])+')/','OS memory counter identity differs')
    require(all(type(value[k]) is int and value[k]>=0 for k in
        ('working_set_bytes','peak_working_set_bytes','private_bytes','paged_bytes','peak_paged_bytes')),'Invalid OS process memory counters')
    return value


def projections(reference,current=None,gpu=None):
    samples=[reference['native_OS_memory_counters']]+([current] if current is not None else [])
    memory=max(max(v[k] for k in ('peak_working_set_bytes','peak_paged_bytes','private_bytes')) for v in samples)+RAM_MARGIN
    vram=max(10240,reference['gpu']['used_mib']+VRAM_MARGIN,
        gpu['used_mib']+VRAM_MARGIN if gpu is not None else 0)
    return dict(additional_ram_and_commit_bytes=memory,additional_vram_mib=vram,
        ram_margin_bytes=RAM_MARGIN,vram_margin_mib=VRAM_MARGIN,
        projection_is_not_measured_peak=True,uses_whole_gpu_usage=True,
        private_bytes_projection_is_conservative_not_measured_resident_requirement=True)


def assess(memory,disk,gpu,projection=None):
    actual=bool(memory['allowed'] and memory['snapshot']['physical_available_bytes']>=4*2**30
        and memory['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30 and gpu['free_mib']>=4096)
    value=dict(actual_reserves_safe=actual,memory=memory,disk_free_bytes=disk,gpu=gpu)
    if projection is None:
        value.update(allowed=actual,new_worker_admission=False);return value
    reserve=gate.assess_resources(memory,disk,gpu,projection['additional_vram_mib'])
    physical=memory['snapshot']['physical_available_bytes']-projection['additional_ram_and_commit_bytes']
    commit=memory['commit_headroom_bytes']-projection['additional_ram_and_commit_bytes']
    value.update(projection=projection,projected_physical_available_bytes=physical,projected_commit_headroom_bytes=commit,
        unchanged_gpu_gate=reserve,new_worker_admission=True,
        allowed=bool(actual and reserve['allowed'] and physical>=4*2**30 and commit>=20*2**30))
    return value


def record_inventory(rows,coordinator,slots,*,admitting=False,register=True):
    """Retain departed identities while peers live. Missing identities never admit."""
    require(len(slots)<=2 and all(s['slot'] in (0,1) for s in slots),'Exactly bounded two slots required')
    require(gate.same_identity(gate.one(rows,coordinator['ProcessId']),coordinator),'Coordinator identity differs')
    by={r['ProcessId']:r for r in rows};require(len(by)==len(rows),'Duplicate OS process rows')
    result=copy.deepcopy(slots)
    for slot in result:
        identities=[slot['command_identity'],slot['native_identity'],*slot.get('auxiliary_identities',[])]
        for expected in identities:
            observed=by.get(expected['ProcessId'])
            require(observed is None or gate.same_identity(observed,expected),'Owned identity changed or PID reuse unproven')
        live=all(v['ProcessId'] in by for v in identities[:2])
        if register and live:
            fresh=gate.register_telemetry(rows,coordinator,[slot])[0].get('auxiliary_identities',[])
            old=slot.get('auxiliary_identities',[])
            require(not old or not fresh or fresh==old,'Recorded auxiliary identity replaced')
            slot['auxiliary_identities']=old or fresh
    allowed=gate.owned_allowlist(coordinator,result)
    if admitting:
        process=gate.inventory(allowed,rows=rows)
    else:
        base=gate.inventory([coordinator],rows=rows)
        process=gate.classify_owned(rows,allowed,base,require_all=False)
        process.update(admission_deferred=True,missing_owned_identities_retained=True,new_worker_admission=False)
    require(process['no_blockers_observed'] is True and not process['blockers'],'Unknown process blocks exact finite-two operation')
    return result,process


def measure(output,coordinator,slots,*,admitting=False,project_from=None):
    rows=None;process=None
    try:
        rows=gate.snapshot();updated,process=record_inventory(rows,coordinator,slots,admitting=admitting)
        current=None
        if project_from is not None:
            require(admitting and any(s['native_identity']==project_from for s in slots),'Projection must name registered live native')
            current=process_memory(project_from)
        memory=gate.preflight();gpu=gate.gpu_snapshot();disk=shutil.disk_usage(output).free
        projection=projections(baseline(),current,gpu if current is not None else None) if admitting else None
        value=assess(memory,disk,gpu,projection)
        value.update(schema=SCHEMA,process_snapshot=rows,process_inventory=process,recorded_slots=updated,
            native_OS_memory_counters=current,same_snapshot_classification=True,process_control_performed=False)
        return value,updated
    except BaseException as exc:
        retirement.persist_failure(output,coordinator,slots,rows,process,exc,'full_population_two_admission',admitting=admitting)
        raise
