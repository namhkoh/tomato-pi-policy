"""Count only immutable exported records, with a stable mutable-status snapshot."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import time
from . import native848_bulk_delivery_v1 as frozen
from .dataset_review import require,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json

ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'data/sim_data/diagnostics'


def checked_rows(status,rows):
    require(status['schema']=='greenhouse.original848_bulk_checkpoint.v1'
        and status['native_resolution']==[848,408] and status['training_approved'] is False,
        'Actual exported848 checkpoint status required')
    counts=dict(Counter(row['split'] for row in rows))
    require(counts==status['counts'] and len(rows)==status['total_images'],'Checkpoint status/index export still changing')
    require(len({row['id'] for row in rows})==len(rows),'Repeated export record IDs')
    seen=set()
    for row in rows:
        aliases=frozen.aliases(row)
        require(not seen&aliases,'Repeated encoded/decoded RGB or conservative camera')
        seen.update(aliases)
    require(type(counts.get('train')) is int and 0<=counts['train']<=20000
        and status['goal_complete']==(counts['train']==20000),'Authoritative accepted target count differs')
    return counts['train']


def snapshot(package,output,*,attempts=5):
    """Archive a stable count; raw source status is allowed to change afterward."""
    package=Path(package).resolve();output=Path(output).resolve()
    require(package.is_relative_to(ROOT/'data/sim_data/dataset_checkpoints')
        and output.is_relative_to(D) and not output.exists(),'Workspace checkpoint and create-only count receipt required')
    require(type(attempts) is int and 1<=attempts<=5,'Bounded export-race rereads only')
    error=None
    for attempt in range(attempts):
        try:
            paths=[package/'baseline.json',*sorted((package/'chunks').glob('*.json'))]
            raw={path:path.read_bytes() for path in paths};status_raw=(package/'status.json').read_bytes()
            status=json.loads(status_raw);rows=frozen.checkpoint_rows(package)
            require(paths==[package/'baseline.json',*sorted((package/'chunks').glob('*.json'))]
                and status_raw==(package/'status.json').read_bytes()
                and all(data==path.read_bytes() for path,data in raw.items()),'Export changed during count snapshot')
            expected=[row for path in paths for row in json.loads(raw[path])['records']]
            require(rows==expected,'Frozen checkpoint reader returned another snapshot')
            count=checked_rows(status,rows)
            break
        except (ValueError,KeyError,json.JSONDecodeError) as exc:
            error=exc
            if attempt+1==attempts:raise
            time.sleep(.5)
    else:raise RuntimeError('No stable exported checkpoint count') from error
    bindings={str(path):hashlib.sha256(data).hexdigest() for path,data in raw.items()}
    baseline=json.loads(raw[paths[0]]);prior=Path(baseline['prior_checkpoint']).resolve()/'index.jsonl'
    require(prior.is_relative_to(ROOT) and sha256(prior)==baseline['prior_index_sha256'],'Exported baseline source seal changed')
    bindings[str(prior)]=baseline['prior_index_sha256']
    for path in paths[1:]:
        chunk=json.loads(raw[path]);objects={}
        if 'inventory_path' not in chunk:
            require(path.name=='recovered_4bce04fb061730a0.json'
                and bindings[str(path)]=='a49ae9ba85f6a13346d9db615f724351ed09dd90f7cae793f9658bf2fe050087'
                and len(chunk['records'])==51 and chunk['new_capture_claimed'] is False
                and chunk['training_approved'] is False,'Unknown recovery chunk cannot increase accepted count')
            verify_bindings(chunk['source_bindings'])
            for source,pin in chunk['source_bindings'].items():
                require(source not in bindings or bindings[source]==pin,'Recovery source binding conflict')
                bindings[source]=pin
            continue
        for key in ('inventory','review'):
            src=Path(chunk[key+'_path']).resolve();pin=chunk[key+'_sha256']
            require(src.is_relative_to(ROOT) and sha256(src)==pin,'Exported chunk approval seal changed')
            bindings[str(src)]=pin;objects[key]=json.loads(src.read_bytes())
        inventory,review=objects['inventory'],objects['review']
        require(review['chunk_accept_or_hold']=='accept' and review['chunk_inventory_sha256']==chunk['inventory_sha256']
            and review['run_plan_sha256']==inventory['run_plan_sha256']
            and review['protocol_sha256']==inventory['protocol_sha256'],'Held or unbound chunk cannot contribute accepted count')
    output.mkdir()
    snapshot_path=output/'status_snapshot.json'
    with snapshot_path.open('xb') as stream:stream.write(status_raw)
    bindings[str(snapshot_path)]=sha256(snapshot_path)
    bindings[str(Path(frozen.__file__).resolve())]=sha256(frozen.__file__)
    result=dict(schema='greenhouse.native848_authoritative_export_count.v1',checkpoint_path=str(package),
        observed_source_status_path=str(package/'status.json'),observed_source_status_sha256=hashlib.sha256(status_raw).hexdigest(),
        mutable_source_status_not_in_immutable_bindings=True,accepted_train_count=count,total_exported_images=len(rows),
        stop_future_native_launches=count>=20000,export_snapshot_attempt=attempt+1,source_bindings=bindings,
        count_source='frozen_checkpoint_rows_baseline_plus_immutable_exported_chunks',
        CPU_candidates_counted=False,raw_captures_counted=False,training_approved=False)
    save_json(output/'result.json',result)
    return result
