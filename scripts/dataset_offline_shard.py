"""Host-local capture ledger; no HTTP, tokens, shared DB, renderer or acceptance.

Use init, reserve-generated/reserve-raw, complete, then export. The frozen
adapters authenticate identities and actual capture actions. Their coordinator
contact is an in-process local Store call, explicitly not global coordination.
Linux captures still require a supported typed owner attestation; this wrapper
does not rewrite Windows evidence or supply an unverified completion fallback.
Every result remains pending cross-host duplicate and quality review.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'examples/greenhouse_sim')]
import dataset_capture_coordinator as coordinator
from dataset_multihost_pipeline import host_config

require=coordinator.require
SCOPE=dict(coordination_scope='host_local',global_review_pending=True,
    final_cross_host_duplicate_review_required=True,network_contacted=False,
    training_approved=False,accepted_training_increment=0)


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def bound(spec):
    require(pin(spec['path'])==spec,'Changed pinned offline artifact')
    return read(spec['path'])


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def local_path(path):
    require(not str(path).startswith(('\\\\','//')),'Use a host-local disk, not a network share')
    return Path(path).resolve()


def initialize(state,config_path,inventory_path,inventory_sha256):
    state=local_path(state);require(not state.exists(),'Create-only offline state directory')
    config=host_config(config_path);inventory=Path(inventory_path).read_bytes()
    require(hashlib.sha256(inventory).hexdigest()==inventory_sha256,'Baseline inventory hash differs')
    rows=[json.loads(line) for line in inventory.decode('utf8').splitlines() if line.strip()]
    require(rows and all(coordinator.HEX.fullmatch(r.get('scene_camera_key','')) for r in rows),
        'Nonempty baseline with every full-scene camera alias required')
    state.mkdir(parents=True)
    write(state/'host_config.json',config)
    (state/'baseline_inventory.jsonl').write_bytes(inventory)
    store=coordinator.Store(state/'ledger.sqlite',hosts=[config['host_id']]);store.seed(rows)
    with store.connect() as db:
        db.execute('CREATE TABLE offline_task_audit(task_id TEXT PRIMARY KEY, audit_json TEXT NOT NULL)')
    manifest=dict(schema='greenhouse.offline_shard_state.v1',host=config['host_id'],
        config=pin(state/'host_config.json'),baseline_inventory=pin(state/'baseline_inventory.jsonl'),
        baseline_rows=len(rows),database='ledger.sqlite',allocation_sha256=coordinator.digest(coordinator.ALLOCATION),
        coordinator_implementation=pin(coordinator.__file__),offline_implementation=pin(__file__),
        shared_database_permitted=False,automatic_claim_expiry=False,**SCOPE)
    write(state/'state.json',manifest)
    return manifest


def open_state(state):
    state=local_path(state);manifest=read(state/'state.json')
    require(manifest['schema']=='greenhouse.offline_shard_state.v1'
        and all(manifest.get(k)==v for k,v in SCOPE.items()),'Explicit host-local pending-review state required')
    require(manifest['database']=='ledger.sqlite' and manifest['shared_database_permitted'] is False,
        'Host-local ledger required')
    config=bound(manifest['config']);host_config(manifest['config']['path'])
    require(manifest['host']==config['host_id']
        and manifest['allocation_sha256']==coordinator.digest(coordinator.ALLOCATION)
        and manifest['coordinator_implementation']==pin(coordinator.__file__)
        and manifest['offline_implementation']==pin(__file__),'Offline authority or implementation changed')
    require(pin(manifest['baseline_inventory']['path'])==manifest['baseline_inventory'],'Baseline changed')
    store=coordinator.Store(state/'ledger.sqlite',hosts=[config['host_id']])
    require(store.stats()['seeded'],'Uninitialized baseline ledger')
    return manifest,config,store


def worker_owner(values,config,kind):
    keys=[coordinator.task_id(v) for v in values]
    require(keys and len(set(keys))==len(keys),'Finite unique identities required')
    if kind=='generated':return int(coordinator.digest(sorted(keys)),16)%config['worker_count']
    require(kind=='raw','Unsupported capture kind')
    owners={int(key,16)%config['worker_count'] for key in keys}
    require(len(owners)==1,'Raw claims must be partitioned to one configured worker')
    return next(iter(owners))


class LocalClient:
    """Reservation-only adapter. No public unverified completion method."""
    def __init__(self,store,config,worker_id,kind,audit):
        require(store.hosts==(config['host_id'],),'Offline Store must have exactly this host')
        require(type(worker_id) is int and 0<=worker_id<config['worker_count'],'Wrong worker')
        require(kind in ('generated','raw'),'Unsupported capture kind')
        self.store,self.config,self.worker_id,self.kind=store,config,worker_id,kind
        self.host=config['host_id'];self.url='offline://host-local/'+self.host
        self.audit=dict(audit,host=self.host,worker_id=worker_id,capture_kind=kind,**SCOPE)
        self.calls=0

    def claim_many(self,values):
        require(worker_owner(values,self.config,self.kind)==self.worker_id,'Another worker owns these identities')
        response=self.store.claim_many(self.host,values);self.calls+=1
        # Store owns the atomic grant. Audit inserts cannot overwrite another
        # grant; a crash leaves a durable reserved hold, never a reissued view.
        with self.store.connect() as db:
            for result in response['reservations']:
                if result['granted']:
                    db.execute('INSERT INTO offline_task_audit VALUES(?,?)',
                        (result['task_id'],coordinator.canonical(self.audit)))
        return dict(response,**SCOPE,coordinator_transport='in_process_local_store')


def reserve(state,kind,request_path,request_sha256,worker_id,output,reference_context=None,seed=None):
    manifest,config,store=open_state(state);output=Path(output).resolve()
    require(not output.exists(),'Create-only preparation output')
    request_pin=pin(request_path);require(request_pin['sha256']==request_sha256,'Changed source request')
    require(type(worker_id) is int and 0<=worker_id<config['worker_count'],'Wrong worker')
    if kind=='generated':
        require(type(seed) is int and config['new_morphology_seed_range'][0]<=seed<=config['new_morphology_seed_range'][1],
            'Declared batch seed must be inside this host namespace')
        require(reference_context is not None,'Authenticated native reference context required')
    else:require(kind=='raw' and seed is None,'Raw original views have no generated seed')
    output.mkdir(parents=True);prepared=output/'authenticated_preparation'
    client=LocalClient(store,config,worker_id,kind,dict(source_request=request_pin,
        seed=seed,seed_role='declared_batch_namespace_only_not_geometry_identity',
        config=manifest['config'],reservation_output=str(output)))
    if kind=='generated':
        adapter=importlib.import_module('sim_data.native848_multihost_generated_v1')
        value=adapter.prepare(request_pin,reference_context,prepared,manifest['config']['path'],
            worker_id=worker_id,reserve=True,client=client)
        gate=prepared/'reservation_gate.json';g=read(gate)
        eligible=g['capture_request_eligible'];reservation=pin(gate)
        capture_request=request_pin if eligible else None
    else:
        adapter=importlib.import_module('sim_data.native848_multihost_sweep_v1')
        value=adapter.prepare(request_path,request_sha256,prepared,manifest['config']['path'],worker_id,client=client)
        reservation=value['coordinator_claims'];capture_request=value['request'];eligible=bool(capture_request)
    receipt=dict(schema='greenhouse.offline_capture_reservation.v1',state=pin(Path(state)/'state.json'),
        config=manifest['config'],host=config['host_id'],worker_id=worker_id,capture_kind=kind,
        source_request=request_pin,capture_request=capture_request,reservation=reservation,
        authenticated_adapter=pin(adapter.__file__),seed=seed,
        seed_role='declared_batch_namespace_only_not_geometry_identity',
        local_store_contacted=client.calls>0,coordinator_contacted=client.calls>0,
        coordinator_transport='in_process_local_store',capture_request_eligible=eligible,
        no_network_authority_claim=True,native_launched=False,**SCOPE)
    write(output/'result.json',receipt);return receipt


def complete(state,reservation_path,reservation_sha256,capture,output):
    manifest,config,store=open_state(state);rp=pin(reservation_path)
    require(rp['sha256']==reservation_sha256,'Changed offline reservation receipt')
    reserved=bound(rp);output=Path(output).resolve();require(not output.exists(),'Create-only completion output')
    require(reserved['schema']=='greenhouse.offline_capture_reservation.v1'
        and reserved['state']==pin(Path(state)/'state.json') and reserved['config']==manifest['config']
        and reserved['host']==config['host_id'] and reserved['capture_request_eligible'] is True
        and all(reserved.get(k)==v for k,v in SCOPE.items()),'Wrong local reservation authority')
    kind=reserved['capture_kind']
    require(kind in ('generated','raw'),'Unsupported typed completion')
    module='native848_multihost_generated_complete_v1' if kind=='generated' else 'native848_multihost_complete_v1'
    adapter=importlib.import_module('sim_data.'+module)
    # No caller-supplied metrics/actions: the frozen adapter reads actual RGB,
    # owner exit, request, identities, scene and observations before any finish.
    checked=adapter.prepare(manifest['config']['path'],capture,reserved['reservation']['path'])
    require(checked['config']==config and checked['reservation']==reserved['reservation'],
        'Completion differs from authenticated host/reservation')
    checked['inputs'].close();output.mkdir(parents=True);results=[]
    for i,action in enumerate(checked['actions']):
        grant=action['reservation']
        with store.connect() as db:
            entry=db.execute('SELECT audit_json FROM offline_task_audit WHERE task_id=?',(grant['task_id'],)).fetchone()
        require(entry is not None,'Task was not granted by this offline preparation')
        audit=json.loads(entry[0])
        require(audit['worker_id']==reserved['worker_id'] and audit['source_request']==reserved['source_request']
            and audit['capture_kind']==kind,'Offline worker/request ownership differs')
        response=store.finish(config['host_id'],grant['task_id'],grant['claim_id'],action['metrics'],action['outcome'])
        item={**response,**SCOPE,'sample_id':action['sample_id'],'segment_id':action.get('segment_id')}
        write(output/f'completion_{i:04d}.json',item);results.append(item)
    checked['inputs'].close()
    result=dict(schema='greenhouse.offline_capture_completion.v1',host=config['host_id'],capture_kind=kind,
        reservation=rp,owner_complete=checked['owner_complete'],capture_result=checked['capture_result'],
        authenticated_completion_adapter=pin(adapter.__file__),source_bindings=checked['inputs'].bindings,
        completions=results,local_store_contacted=True,coordinator_contacted=True,
        coordinator_transport='in_process_local_store',native_launched=False,**SCOPE)
    write(output/'result.json',result);return result


def portable_export(state,output):
    manifest,config,store=open_state(state);output=Path(output)
    # Read tasks, aliases and audit metadata in one consistent SQLite snapshot.
    with store.connect() as db:
        db.execute('BEGIN')
        aliases={}
        for row in db.execute('SELECT alias,task_id FROM scene_aliases ORDER BY alias'):
            aliases.setdefault(row['task_id'],[]).append(row['alias'])
        audit={r['task_id']:json.loads(r['audit_json']) for r in db.execute('SELECT * FROM offline_task_audit')}
        rows=[]
        for r in db.execute('SELECT * FROM tasks ORDER BY id'):
            rows.append(dict(task_id=r['id'],identity=json.loads(r['identity_json']),host=r['host'],state=r['state'],
                scene_camera_keys=aliases.get(r['id'],[]),decoded_rgb_sha256=r['rgb'],dhash64_hex=r['dhash'],
                result=json.loads(r['result_json']) if r['result_json'] else None,
                offline_audit=audit.get(r['id']),**SCOPE))
    value=dict(schema='greenhouse.offline_shard_portable_ledger.v1',host=config['host_id'],
        state_manifest=manifest,rows=rows,claim_handles_included=False,**SCOPE)
    write(output,value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    init=sub.add_parser('init');init.add_argument('--config',required=True);init.add_argument('--inventory',required=True)
    init.add_argument('--inventory-sha256',required=True)
    for kind in ('generated','raw'):
        r=sub.add_parser('reserve-'+kind);r.add_argument('--request',required=True);r.add_argument('--request-sha256',required=True)
        r.add_argument('--worker-id',type=int,required=True);r.add_argument('--output',required=True)
        if kind=='generated':r.add_argument('--reference-context',required=True);r.add_argument('--seed',type=int,required=True)
    c=sub.add_parser('complete');c.add_argument('--reservation',required=True);c.add_argument('--reservation-sha256',required=True)
    c.add_argument('--capture',required=True);c.add_argument('--output',required=True)
    e=sub.add_parser('export');e.add_argument('--output',required=True)
    for command in sub.choices.values():command.add_argument('--state',required=True)
    a=p.parse_args()
    if a.action=='init':result=initialize(a.state,a.config,a.inventory,a.inventory_sha256)
    elif a.action.startswith('reserve-'):
        result=reserve(a.state,a.action.split('-')[1],a.request,a.request_sha256,a.worker_id,a.output,
            getattr(a,'reference_context',None),getattr(a,'seed',None))
    elif a.action=='complete':result=complete(a.state,a.reservation,a.reservation_sha256,a.capture,a.output)
    else:result=portable_export(a.state,a.output)
    print(json.dumps(dict(schema=result['schema'],host=result['host'],**SCOPE),sort_keys=True))


if __name__=='__main__':main()
