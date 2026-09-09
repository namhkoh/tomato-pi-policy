"""Explicit three-lane recovery; run each lane as a detached hidden process.

Skip hash-bound completed jobs, adopt only independently observed live exits,
and rerun unproven jobs in NEW directories. Never infer successful execution.
Not a reboot-persistent service; no automatic retries or training approval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

import psutil

from .collection_run import run_jobs, finalize_job, checked_exit
from .dataset_review import read_json, require, write_json
from .depth_preview import sha256

SCHEMA='greenhouse.explicit_collection_resume.v1'
AUDITED='audited_prototype_pending_visual_review'
COMPLETE='complete_bounded_batch_pending_visual_review'


def allocate(items):
    require(len({i['job_id'] for i in items})==len(items),'Duplicate recovery job')
    require(all(i['action'] in ('observe_then_audit','new_directory_capture') for i in items),'Unknown recovery action')
    adopted=[i for i in items if i['action']=='observe_then_audit']
    require(len(adopted)<=3,'Too many observed workers')
    lanes=[dict(lane_id=f'lane_{i+1:02d}',items=[]) for i in range(3)]
    loads=[0,0,0]
    for n,item in enumerate(adopted):
        lanes[n]['items'].append(item); loads[n]=item['maximum_frames']
    for item in sorted((i for i in items if i['action']=='new_directory_capture'),
                       key=lambda i:(i['split']=='train',-i['maximum_frames'],i['job_id'])):
        n=min(range(3),key=lambda n:(loads[n],n))
        lanes[n]['items'].append(item); loads[n]+=item['maximum_frames']
    return lanes


def create(campaign_path, observers, output):
    campaign_path=Path(campaign_path).resolve(); output=Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(campaign_path.parent),'Use a new separate resume directory')
    campaign=read_json(campaign_path)
    require(campaign['schema_version']=='greenhouse.bounded_collection_campaign.v1','Unknown source campaign')
    plans={k:read_json(v['path']) for k,v in campaign['plans'].items()}
    for k,v in campaign['plans'].items():
        require(sha256(v['path'])==v['sha256'],'Changed source plan')
        require(not output.is_relative_to(Path(plans[k]['package']).resolve()),'Output overlaps source assets')
    require(plans['train']['family_assignments']==plans['heldout']['family_assignments'],'Changed family split')
    folders={}
    for lane in campaign['lanes']:
        if lane['predecessor']:
            p=lane['predecessor']; folders[p['job_id']]=Path(p['batch'])/p['job_id']
        for item in lane['items']:
            folders[item['job_id']]=campaign_path.parent/lane['lane_id']/('capture_'+item['job_id'])/item['job_id']
    require(set(folders)=={j['job_id'] for j in plans['train']['jobs']},'Source campaign family coverage changed')
    observed={}
    for name in observers:
        p=Path(name).resolve(); a=read_json(p/'attached.json'); launch=Path(a['launch_path'])
        require(sha256(launch)==a['launch_sha256'] and a['method']=='retained_verified_windows_process_handle','Invalid observer binding')
        require(launch.parent not in observed,'Duplicate observer')
        observed[launch.parent]=(p,sha256(p/'attached.json'))
    items=[]; skipped=[]
    for original in plans['train']['jobs']:
        role='train' if original['split']=='train' else 'heldout'
        job=next(j for j in plans[role]['jobs'] if j['job_id']==original['job_id'])
        folder=folders[job['job_id']].resolve(); result_path=folder/'result.json'
        if result_path.is_file():
            r=read_json(result_path)
            if r.get('state')==AUDITED and r.get('returncode')==0 and not r.get('timed_out'):
                a=Path(r.get('audit_path',str(folder/'audit/audit.json'))).resolve()
                require(a.is_relative_to(folder) and sha256(a)==r['audit_sha256'],'Changed completed audit')
                require(r['job_id']==job['job_id'] and r['plant_family']==job['plant_family'],'Wrong completed job')
                request=read_json(folder.parent/'request.json')
                require(request['plan_sha256']==campaign['plans'][role]['sha256'] and job['job_id'] in request['selected_job_ids'],'Wrong completed plan')
                skipped.append(dict(job_id=job['job_id'],result_path=str(result_path),result_sha256=sha256(result_path),audit_path=str(a),audit_sha256=sha256(a)))
                continue
        item=dict(job_id=job['job_id'],family=job['plant_family'],split=job['split'],plan_role=role,
            maximum_frames=len(job['targets'])*job['max_rendered_views_per_target'],previous_job_folder=str(folder),
            action='new_directory_capture',reason='unproven_or_unstarted_source_job')
        if folder in observed:
            require(not result_path.exists(),'Do not overwrite an existing worker result')
            p,h=observed[folder]
            item.update(action='observe_then_audit',observer=str(p),observer_binding_sha256=h)
        items.append(item)
    require(set(observed)<={Path(i['previous_job_folder']) for i in items if i['action']=='observe_then_audit'},'Observer outside pending campaign')
    result=dict(schema_version=SCHEMA,source_campaign=str(campaign_path),source_campaign_sha256=sha256(campaign_path),
        created_utc=datetime.now(timezone.utc).isoformat(),plans=campaign['plans'],skipped_audited_jobs=skipped,
        lanes=allocate(items),maximum_capture_workers=3,minimum_free_disk_bytes=campaign['minimum_free_disk_bytes'],
        minimum_available_ram_bytes=8*2**30,worker_timeout_s=14400,automatic_retries=False,training_approved=False,
        implementation_sha256={n:sha256(Path(__file__).with_name(n)) for n in
            ('collection_resume.py','collection_run.py','collection_process.py')})
    output.mkdir(parents=True); write_json(output/'resume.json',result)
    return result


def wait_observed(item):
    root=Path(item['observer']); attached=read_json(root/'attached.json')
    require(sha256(root/'attached.json')==item['observer_binding_sha256'],'Changed observer binding')
    launch=read_json(attached['launch_path'])
    deadline=datetime.fromisoformat(launch['started_utc']).timestamp()+launch['timeout_s']+40
    while not (root/'exit.json').is_file():
        require(time.time()<deadline,'Observer missed the original worker deadline')
        try:
            command=psutil.Process(attached['observer_pid']).cmdline()
            require('sim_data.collection_process' in command and str(root) in command,'Observer process disappeared or changed')
        except psutil.Error:
            require((root/'exit.json').is_file(),'Observer exited without an exit receipt')
        time.sleep(2)
    receipt,_=checked_exit(item['previous_job_folder'],root/'exit.json')
    for k in ('pid','command','created_epoch','observer_pid','launch_sha256','method'):
        require(receipt[k]==attached[k],'Exit is not from the bound observer')
    return root/'exit.json'


def check_inventory(plan, root):
    allowed={Path(i['previous_job_folder']).resolve()/'capture' for lane in plan['lanes'] for i in lane['items'] if i['action']=='observe_then_audit'}
    count=0
    for p in psutil.process_iter(['cmdline']):
        try:
            args=p.info['cmdline'] or []
            if 'sim_data.collection_worker' not in args: continue
            require('--output' in args,'Unknown native worker output')
            destination=Path(args[args.index('--output')+1]).resolve()
            require(destination in allowed or destination.is_relative_to(root),'Unrelated capture active; do not oversubscribe')
            count+=1
        except (psutil.NoSuchProcess,psutil.AccessDenied): continue
    require(count<plan['maximum_capture_workers'],'Capture slot bound reached; lane stopped')


def run_lane(resume_path, lane_id):
    resume_path=Path(resume_path).resolve(); root=resume_path.parent; plan=read_json(resume_path); plan_hash=sha256(resume_path)
    require(plan['schema_version']==SCHEMA and plan['maximum_capture_workers']==3,'Invalid resume contract')
    require(plan['lanes']==allocate([i for l in plan['lanes'] for i in l['items']]),'Changed resume allocation')
    for n,h in plan['implementation_sha256'].items(): require(sha256(Path(__file__).with_name(n))==h,'Runner changed; create a new resume plan')
    require(sha256(plan['source_campaign'])==plan['source_campaign_sha256'],'Changed source campaign')
    for v in plan['plans'].values(): require(sha256(v['path'])==v['sha256'],'Changed collection plan')
    for r in plan['skipped_audited_jobs']:
        require(sha256(r['result_path'])==r['result_sha256'] and sha256(r['audit_path'])==r['audit_sha256'],'Changed skipped completion')
    lane=next((l for l in plan['lanes'] if l['lane_id']==lane_id),None); require(lane is not None,'Unknown resume lane')
    output=root/lane_id; output.mkdir()
    status=dict(schema_version=SCHEMA,lane_id=lane_id,resume_sha256=plan_hash,pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),state='running',jobs=[],training_approved=False)
    write_json(output/'request.json',status)
    try:
        for item in lane['items']:
            require(sha256(resume_path)==plan_hash,'Resume plan changed while running')
            source=plan['plans'][item['plan_role']]
            require(sha256(source['path'])==source['sha256'],'Collection plan changed while running')
            require(shutil.disk_usage(root).free>=plan['minimum_free_disk_bytes'],'Insufficient disk reserve; no deletion')
            print('RESUME_JOB_START',item['job_id'],item['family'],item['action'],flush=True)
            if item['action']=='observe_then_audit':
                receipt=wait_observed(item)
                record=finalize_job(item['previous_job_folder'],dict(job_id=item['job_id'],plant_family=item['family']),receipt,audit_name='audit_recovered')
                require(record['state']==AUDITED,'Recovered worker/audit failed; lane stopped')
                result_path=Path(item['previous_job_folder'])/'result.json'
            else:
                check_inventory(plan,root)
                require(psutil.virtual_memory().available>=plan['minimum_available_ram_bytes'],'Insufficient RAM reserve')
                batch=output/('capture_'+item['job_id'])
                result=run_jobs(source['path'],batch,max_jobs=1,timeout_s=14400,job_ids=[item['job_id']],
                    instance_backend='fast',render_budget='warm56_then8')
                require(result['state']==COMPLETE,'New worker/audit failed; lane stopped without retry')
                result_path=batch/item['job_id']/'result.json'
            event=dict(job_id=item['job_id'],action=item['action'],result_path=str(result_path),result_sha256=sha256(result_path))
            status['jobs'].append(event); write_json(output/('completed_'+item['job_id']+'.json'),event)
        status['state']='complete_resume_lane_pending_visual_review'
    except BaseException as exc:
        status.update(state='stopped_resume_lane',error=str(exc)); raise
    finally: write_json(output/'result.json',status)
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('create'); c.add_argument('--campaign',type=Path,required=True)
    c.add_argument('--observer',type=Path,action='append',default=[]); c.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run'); r.add_argument('--resume',type=Path,required=True); r.add_argument('--lane',required=True)
    a=p.parse_args(argv)
    if a.command=='create':
        result=create(a.campaign,a.observer,a.output); print('RESUME_CREATED',len(result['skipped_audited_jobs']),[(l['lane_id'],len(l['items'])) for l in result['lanes']],flush=True)
    else: print('RESUME_RESULT',run_lane(a.resume,a.lane)['state'],flush=True)


if __name__=='__main__': main()
