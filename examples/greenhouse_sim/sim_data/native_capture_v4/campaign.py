"""Serial, create-only continuation wave; candidates are never release counts.

Waits for explicitly named prior collection and controls. Uses only already
qualified scheduled TRAIN anchors, V4 preparation and a qualified opt-in compact
writer. No physical robot API, source edits, split changes, or concurrent Kit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from ..dataset_review import read_json, write_json, require, verify_bindings
from ..depth_preview import sha256
from . import prepare
from ..native_dataset import native_process_guard


def planned_jobs(schedule, qualified, *, rounds, seed_base):
    require(type(rounds) is int and 1 <= rounds <= 100, 'One to100 explicit rounds per wave')
    require(type(seed_base) is int and 0 <= seed_base < 2**32, 'uint32 seed base required')
    require(isinstance(qualified, dict) and qualified, 'Completed qualified anchors required')
    jobs = {j['source_family']:j for j in schedule['jobs']}
    require(len(jobs) == len(schedule['jobs']) and set(qualified) <= set(jobs), 'Ambiguous/unknown donor')
    selected=[]
    for family in sorted(qualified):
        job=jobs[family]
        require(job['split']=='train', 'Frozen TRAIN sources only')
        attempts=[a for a in [job['anchor'],*job['fallback_anchors']] if a['attempt_id']==qualified[family]]
        require(len(attempts)==1 and attempts[0]['source_family']==family
                and attempts[0]['split']=='train', 'Wrong qualified anchor identity')
        selected.append((job,attempts[0]))
    output=[]
    for round_index in range(rounds):
        for donor_index,(job,attempt) in enumerate(selected):
            index=len(output)+1;seed=seed_base+(index-1)*1000
            require(seed+2*37 < 2**32, 'Seed range overflow')
            output.append(dict(index=index,round_index=round_index+1,seed=seed,
                source_family=job['source_family'],job_id=job['job_id'],attempt_id=attempt['attempt_id']))
    return output


def native_processes():
    return native_process_guard.native_processes()


def check_queue_destination(output, prior, controls, proof):
    output=Path(output).resolve()
    for source in (Path(prior).resolve(),Path(controls).resolve(),Path(proof).resolve().parent):
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                'Campaign output must be disjoint from prior/control/proof roots')


def completed_controls(directory):
    directory=Path(directory).resolve();request_path=directory/'request.json';result_path=directory/'result.json'
    request,result=read_json(request_path),read_json(result_path)
    plans=[str(Path(p).resolve()) for p in request['plans']]
    require(plans and len(plans)==len(set(plans))==request['max_pairs']
            and request['max_images']==2*len(plans), 'Incomplete/duplicate requested controls')
    records=result['records']
    require(len(records)==len(plans) and result['bindings']==request['bindings'], 'Incomplete control wave')
    require(set(plans)<=set(result['bindings']), 'Requested control plan lacks immutable binding')
    bindings={str(request_path):sha256(request_path),str(result_path):sha256(result_path),**result['bindings']}
    seen=set()
    for record in records:
        plan=str(Path(record['plan']).resolve())
        require(plan in plans and plan not in seen,'Unknown/duplicate completed control');seen.add(plan)
        case=directory/f'case_{plans.index(plan)+1:03d}'
        require(Path(record['output']).resolve()==case and record['exit_code']==0
            and record['state']=='native_pair_complete_pending_annotation_and_visual_control_review'
            and not (case/'failure.json').exists(),'Failed/different native control')
        path=case/'result.json'
        require(sha256(path)==record['result_sha256'],'Changed native control result')
        document=read_json(path)
        require(document['state']=='generated_native_pair_captured_pending_visual_review'
            and [s['sample_id'] for s in document['samples']]==['original_control','generated_variant'],
            'Incomplete native control pair')
        bindings[str(path)]=record['result_sha256']
    verify_bindings(bindings)
    return bindings


class OwnedChildCleanupError(BaseException):
    """Fatal: never continue a campaign after losing worker ownership."""


def _terminate_owned(process):
    require(process.poll() is None, 'Owned parent already exited; descendant cleanup cannot be certified')
    # Popen retains the live process handle. Only this owned worker tree is targeted.
    subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],check=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW)
    require(not native_processes(), 'Native processes remain after owned-tree cleanup; halt continuation')


def run_checked(command,log_path,*,bindings,environment,native,reserve,launch_check,announce):
    verify_bindings(bindings)
    if native:reserve()
    verify_bindings(bindings)
    if launch_check:launch_check()
    with Path(log_path).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            announce(process.pid)
            return process.wait()
        except BaseException:
            try:
                _terminate_owned(process)
                process.wait(timeout=30)
            except BaseException as cleanup:
                raise OwnedChildCleanupError(f'Cannot confirm owned worker tree exited; PID={process.pid}') from cleanup
            raise


def compact_command(isaac,plan,output,proof,proof_sha256):
    require(isinstance(proof_sha256,str) and len(proof_sha256)==64
            and all(c in '0123456789abcdef' for c in proof_sha256),'Explicit storage proof SHA256 required')
    return [str(Path(isaac).resolve()),'-m','sim_data.native_dataset.compact_views',
        '--batch-plan',str(plan),'--output',str(output),'--storage-qualification',str(Path(proof).resolve()),
        '--storage-qualification-sha256',proof_sha256,
        '--instance-backend','fast','--render-budget','warm56_then8_trial']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('schedule','prior-campaign','after-controls','storage-qualification','output','isaac-python'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--schedule-sha256',required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--seed-base',type=int,default=4000000)
    parser.add_argument('--max-targets',type=int,default=12)
    parser.add_argument('--native-deps',type=Path,required=True)
    args=parser.parse_args(argv)
    require(os.name=='nt','This launch coordinator uses explicit Windows process/reserve checks')
    require(1<=args.max_targets<=12 and 1<=args.rounds<=100,'Invalid explicit wave bounds')
    schedule_path=args.schedule.resolve();schedule=read_json(schedule_path)
    require(sha256(schedule_path)==args.schedule_sha256,'Schedule differs from pinned input')
    prepare.verifier.check_schedule(schedule)
    bank=read_json(schedule['source_reference_bank'])
    output=prepare.validate_destination(args.output,bank,schedule)
    check_queue_destination(output,args.prior_campaign,args.after_controls,args.storage_qualification)
    require(args.isaac_python.is_file() and args.native_deps.is_dir(),'Missing native runtime/dependencies')
    # Import only when invocation starts; these are never patched into the legacy collector.
    from ..native_dataset import compact_views
    from ..native_dataset.compact_qualification import qualify_storage
    from ..native_dataset.audit import audit_capture
    from sim_physics.host_memory import preflight
    worker_path=Path(compact_views.__file__).resolve()
    bindings={str(Path(__file__).resolve()):sha256(__file__),str(schedule_path):args.schedule_sha256,
        **prepare.implementation_bindings(),**compact_views.implementation_bindings(),
        **native_process_guard.implementation_bindings()}
    output.mkdir(parents=True)
    prior=args.prior_campaign.resolve();controls=args.after_controls.resolve()
    records=[];counter=0;failures=0
    def status(state,**extra):
        nonlocal counter
        counter+=1
        write_json(output/f'progress_{counter:06d}.json',dict(state=state,
            updated_utc=datetime.now(timezone.utc).isoformat(),records=records,
            final_training_approved=0,implementation_bindings=bindings,**extra))
    write_json(output/'request.json',dict(schedule=str(schedule_path),schedule_sha256=args.schedule_sha256,
        prior_campaign=str(prior),controls=str(controls),storage_qualification=str(args.storage_qualification.resolve()),
        rounds=args.rounds,seed_base=args.seed_base,max_targets=args.max_targets,
        native_resolution=[1696,816],render_profile='warm56_then8_trial',storage_backend='compact_native_lossless',
        implementation_bindings=bindings,training_approved=False,source_cap_reset=False))
    for directory,phase in ((prior,'prior_collection'),(controls,'matched_controls')):
        while not (directory/'result.json').exists():
            states=sorted(directory.glob('progress_*.json' if phase=='prior_collection' else 'queue_*.json'))
            if states:
                state=read_json(states[-1])['state']
                require(not ('stopped_after_three' in state or state.startswith('held_after')),
                        'Previous native failure requires diagnosis before continuation')
            verify_bindings(bindings);status('waiting_for_'+phase)
            time.sleep(30)
    prior_result=read_json(prior/'result.json')
    bindings.update(completed_controls(controls))
    # A diagnostic success does not change defaults; this explicit worker opts in.
    storage_bindings=qualify_storage(args.storage_qualification.resolve())
    storage_pin=storage_bindings[str(args.storage_qualification.resolve())]
    bindings.update(storage_bindings)
    bindings.update({str(prior/'result.json'):sha256(prior/'result.json'),
                     str(controls/'result.json'):sha256(controls/'result.json')})
    jobs=planned_jobs(schedule,prior_result['verified_anchors'],rounds=args.rounds,seed_base=args.seed_base)
    write_json(output/'planned_jobs.json',dict(jobs=jobs,training_approved=False,source_cap_reset=False))
    def reserve():
        while True:
            memory=preflight();free=shutil.disk_usage(output).free;active=native_processes()
            if memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30 and free>=60*2**30 and not active:return
            status('waiting_for_safe_native_resources',memory=memory,disk_free_bytes=free,active_native=active)
            time.sleep(30)
    def run(command,log_path,*,native,launch_check=None):
        environment=os.environ.copy();environment['OPENBLAS_NUM_THREADS']='1'
        if native:
            environment['PYTHONPATH']=os.pathsep.join([str(args.native_deps.resolve()),
                str(Path('examples').resolve()),str(Path('examples/greenhouse_sim').resolve())])
        return run_checked(command,log_path,bindings=bindings,environment=environment,native=native,
            reserve=reserve,launch_check=launch_check,
            announce=lambda pid:status('worker_running',worker_pid=pid,native=native,command=command))
    for item in jobs:
        folder=output/f"job_{item['index']:04d}_{item['source_family']}_round{item['round_index']}";folder.mkdir()
        job,attempt=prepare._scheduled(schedule,item['job_id'],item['attempt_id'])
        row=dict(**item,job=str(folder),training_approved=False,preparation_failures=[])
        prepared=None
        for retry in range(3):
            seed=item['seed']+retry*37;prepared_root=folder/f'prepare_seed{seed}'
            code=run([sys.executable,'-m','sim_data.native_capture_v4.prepare','--schedule',str(schedule_path),
                '--schedule-sha256',args.schedule_sha256,'--job-id',job['job_id'],'--attempt-id',attempt['attempt_id'],
                '--seed',str(seed),'--max-targets',str(args.max_targets),'--output',str(prepared_root)],
                folder/f'prepare_seed{seed}.log',native=False)
            try:
                require(code==0,'CPU preparation failed')
                prepared=prepare.verify_prepared(prepared_root,schedule_path,job,attempt,seed,
                    max_targets=args.max_targets,schedule_sha256=args.schedule_sha256)
                break
            except Exception:row['preparation_failures'].append(dict(seed=seed,error=traceback.format_exc()))
        if prepared is None:row.update(state='cpu_rejected_no_capture')
        else:
            write_json(folder/'plan.json',read_json(prepared['plan_path']))
            write_json(folder/'reference_provenance.json',prepared)
            try:
                def launch_check():
                    prepare.verify_prepared(prepared_root,schedule_path,job,attempt,seed,
                        max_targets=args.max_targets,schedule_sha256=args.schedule_sha256)
                    require(sha256(folder/'plan.json')==prepared['plan_sha256'],'Submitted native plan changed')
                    qualify_storage(args.storage_qualification.resolve(),expected_sha256=storage_pin)
                code=run(compact_command(args.isaac_python,folder/'plan.json',folder/'capture',
                    args.storage_qualification,storage_pin),folder/'native.log',
                    native=True,launch_check=launch_check)
                require(code==0 and (folder/'capture/result.json').exists() and not (folder/'capture/failure.json').exists(),
                        'Native capture incomplete/failed')
                audit=audit_capture(folder/'capture',folder/'plan.json');write_json(folder/'automatic_audit.json',audit)
                result=read_json(folder/'capture/result.json')
                row.update(state='native_complete_pending_review',captured=result['captured_frames'],
                    automatic=result['automatically_clear_annotation_candidates'],audit_counts=audit['counts'],
                    native_seconds=result['elapsed_seconds'],native_exit_code=code)
                failures=0
            except Exception:
                row.update(state='failed_preserved',error=traceback.format_exc());failures+=1
        records.append(row);write_json(folder/'campaign_result.json',row)
        print('SCALE_NATIVE_JOB',json.dumps(row),flush=True);status('between_jobs')
        if failures>=3:
            status('stopped_after_three_native_failures_no_approvals');raise SystemExit(2)
    status('bounded_scale_wave_complete_pending_global_admission')
    write_json(output/'result.json',dict(records=records,training_approved=False,source_cap_reset=False,
        new_biological_families=0,verified_anchors=prior_result['verified_anchors']))


if __name__=='__main__':main()
