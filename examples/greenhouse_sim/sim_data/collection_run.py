"""Serial isolated workers with durable exit-status evidence and independent audit.

One Kit process at a time, raw logs through subprocess (not shell redirection).
Never retry over an old capture, turn a failed exit into success, or infer reviews.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from .collection_plan import load_plan
from .dataset_review import read_json, require, write_json
from .depth_preview import sha256
from .collection_process import EXIT_SCHEMA


def checked_exit(folder, exit_path):
    folder,exit_path=Path(folder).resolve(),Path(exit_path).resolve()
    receipt=read_json(exit_path); launch=read_json(folder/'launch.json')
    require(receipt.get('schema_version')==EXIT_SCHEMA and
            receipt.get('method') in ('subprocess_wait','retained_verified_windows_process_handle'), 'Unverified exit method')
    require(Path(receipt['launch_path']).resolve()==folder/'launch.json' and
            receipt['launch_sha256']==sha256(folder/'launch.json') and
            receipt['pid']==launch['pid'] and receipt['command']==launch['command'], 'Exit receipt/launch mismatch')
    require(type(receipt['returncode']) is int and type(receipt['timed_out']) is bool,'Invalid exit status')
    command=launch['command']
    require(command[1:4]==['-u','-m','sim_data.collection_worker'] and
            Path(command[command.index('--output')+1]).resolve()==folder/'capture', 'Wrong native capture output')
    return receipt,launch


def finalize_job(folder, job, exit_path, *, grounding=True, audit_name='audit'):
    """Resume audit only from a real bound exit receipt, never from shutdown text."""
    folder=Path(folder).resolve(); capture=folder/'capture'
    require(not (folder/'result.json').exists(),'Never overwrite a worker result')
    require(audit_name in ('audit','audit_recovered'),'Unexpected independent audit destination')
    receipt,launch=checked_exit(folder,exit_path)
    require(launch['command'][launch['command'].index('--job')+1]==job['job_id'],'Wrong recovered job')
    manifest=read_json(capture/'manifest.json') if (capture/'manifest.json').is_file() else None
    state=worker_state(receipt['returncode'],receipt['timed_out'],manifest)
    elapsed=(datetime.fromisoformat(receipt['observed_exit_utc'])-datetime.fromisoformat(launch['started_utc'])).total_seconds()
    record=dict(job_id=job['job_id'],plant_family=job['plant_family'],state=state,
        returncode=receipt['returncode'],timed_out=receipt['timed_out'],elapsed_s=elapsed,
        capture_state=manifest.get('state') if manifest else None,log_sha256=sha256(folder/'worker.log'),
        sample_count=len(manifest.get('samples',[])) if manifest else 0,training_eligible=False,
        exit_receipt_path=str(Path(exit_path).resolve()),exit_receipt_sha256=sha256(exit_path))
    if state=='ready_for_independent_audit':
        try:
            from .dataset_review import audit
            destination=folder/audit_name
            result=audit(capture,destination,stream_cards=True) if grounding else audit(capture,destination)
            record.update(state='audited_prototype_pending_visual_review',audit_path=str(destination/'audit.json'),
                audit_sha256=sha256(destination/'audit.json'),
                numerical_clear_views=sum(s['quality']['clear_view_gate_passed'] for s in result['samples']))
        except Exception as exc:
            record.update(state='stopped_independent_audit_failed',error=str(exc))
    write_json(folder/'result.json',record)
    return record


def worker_state(returncode, timed_out, manifest):
    if timed_out:
        return 'stopped_worker_timeout'
    if returncode != 0:
        return 'stopped_worker_nonzero_exit'
    if not manifest or manifest.get('state') != 'pilot_ready_for_review' or not manifest.get('samples'):
        return 'stopped_missing_complete_capture'
    if manifest.get('source_assets_unchanged') is not True or manifest.get('training_dataset_approved') is not False:
        return 'stopped_invalid_capture_contract'
    return 'ready_for_independent_audit'


def run_jobs(plan_path, output, *, max_jobs=1, timeout_s=1200, job_ids=None, render_reference_check=False, profile_first_render=False, instance_backend='legacy', render_budget='established'):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan, _ = load_plan(plan_path)
    grounding=plan.get('schema_version')=='greenhouse.grounding_collection_plan.v1'
    require(type(max_jobs) is int and 1 <= max_jobs <= (24 if grounding else 4), 'Invalid job cap')
    require(type(timeout_s) is int and 60 <= timeout_s <= (14400 if grounding else 1800), 'Invalid worker timeout')
    selected=plan['jobs']
    if job_ids is not None:
        require(isinstance(job_ids,list) and bool(job_ids) and len(set(job_ids))==len(job_ids)
                and set(job_ids)<={j['job_id'] for j in selected}, 'Unknown or duplicate selected jobs')
        selected=[j for j in selected if j['job_id'] in job_ids]
    selected=selected[:max_jobs]
    require(bool(selected),'Empty collection schedule')
    require(type(render_reference_check) is bool and (grounding or not render_reference_check),'Invalid reference render mode')
    require(instance_backend in ('legacy','fast','compare') and (grounding or instance_backend=='legacy'),'Invalid instance backend')
    require(render_budget in ('established','single56','single56_compare','noise_probe','warm56_then8') and (grounding or render_budget=='established')
            and not(render_reference_check and render_budget!='established'),'Invalid or conflicting render budget')
    require(not output.exists() and not output.is_relative_to(Path(plan['package'])), 'Choose a new batch output outside sources')
    output.mkdir(parents=True)
    ledger = {'schema_version': 'greenhouse.native_collection_batch.v1', 'state': 'running',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'plan_path': str(plan_path),
        'plan_sha256': sha256(plan_path), 'jobs': [], 'training_dataset_approved': False,
        'maximum_concurrent_kit_workers': 1, 'automatic_retries': False}
    write_json(output/'request.json', {'plan_sha256': ledger['plan_sha256'], 'max_jobs': max_jobs, 'timeout_s': timeout_s,
                                     'selected_job_ids':[j['job_id'] for j in selected], 'render_reference_check':render_reference_check,
                                     'profile_first_render':profile_first_render,'instance_backend':instance_backend,'render_budget':render_budget})
    try:
        for job in selected:
            require(sha256(plan_path) == ledger['plan_sha256'], 'Schedule changed during batch')
            folder = output/job['job_id']
            folder.mkdir()
            capture = folder/'capture'
            command = [sys.executable, '-u', '-m', 'sim_data.collection_worker', '--plan', str(plan_path),
                       '--job', job['job_id'], '--output', str(capture)]
            if render_reference_check: command.append('--render-reference-check')
            if profile_first_render: command.append('--profile-first-render')
            command.extend(['--instance-backend',instance_backend])
            command.extend(['--render-budget',render_budget])
            started = time.monotonic()
            print('COLLECTION_JOB_START', job['job_id'], job['plant_family'], flush=True)
            timed_out = False
            with (folder/'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                write_json(folder/'launch.json', {'pid': process.pid, 'command': command,
                    'started_utc': datetime.now(timezone.utc).isoformat(), 'timeout_s': timeout_s})
                try:
                    # Short waits allow regular progress and timeout enforcement.
                    while process.poll() is None:
                        try:
                            process.wait(timeout=min(10, max(.1, timeout_s-(time.monotonic()-started))))
                        except subprocess.TimeoutExpired:
                            if time.monotonic()-started >= timeout_s:
                                timed_out = True
                                process.terminate()  # Only this exact isolated child, never the user's simulator.
                                process.wait(timeout=20)
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=20)
                    raise
                finally:
                    if process.poll() is not None:
                        write_json(folder/'worker_exit.json',dict(schema_version=EXIT_SCHEMA,
                            launch_path=str(folder/'launch.json'),launch_sha256=sha256(folder/'launch.json'),
                            pid=process.pid,command=command,method='subprocess_wait',returncode=process.returncode,
                            timed_out=timed_out,observed_exit_utc=datetime.now(timezone.utc).isoformat(),training_approved=False))
            record=finalize_job(folder,job,folder/'worker_exit.json',grounding=grounding)
            ledger['jobs'].append(record)
            print('COLLECTION_JOB_RESULT', json.dumps(record), flush=True)
            if record['state'] != 'audited_prototype_pending_visual_review':
                ledger['state'] = record['state']
                break
        else:
            ledger['state'] = 'complete_bounded_batch_pending_visual_review'
    except BaseException as exc:
        ledger.update(state='stopped_scheduler_error', error=str(exc))
        raise
    finally:
        write_json(output/'manifest.json', ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--job', action='append', dest='job_ids')
    parser.add_argument('--render-reference-check', action='store_true')
    parser.add_argument('--profile-first-render', action='store_true')
    parser.add_argument('--instance-backend',choices=['legacy','fast','compare'],default='legacy')
    parser.add_argument('--render-budget',choices=['established','single56','single56_compare','noise_probe','warm56_then8'],default='established')
    args = parser.parse_args(argv)
    result = run_jobs(args.plan, args.output, max_jobs=args.max_jobs, timeout_s=args.timeout, job_ids=args.job_ids,
                      render_reference_check=args.render_reference_check, profile_first_render=args.profile_first_render,
                      instance_backend=args.instance_backend,render_budget=args.render_budget)
    print('COLLECTION_RESULT', result['state'], str(args.output), flush=True)
    return 0 if result['state']=='complete_bounded_batch_pending_visual_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
