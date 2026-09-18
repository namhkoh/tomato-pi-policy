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


def check_individual_chunk(chunk,inventory,review,bindings):
    """Only the exact reviewed63 adjudication may contribute its accepted subset.

    Its full-population receipt deliberately remains HOLD. Never interpret that
    as sampled-batch acceptance, and never count the20 actual individual holds.
    """
    from . import native848_individual_adjudication_export_v1 as adjudication
    module=Path(adjudication.__file__).resolve()
    require(sha256(module)=='b199446746f9e8ad4f5a209d542d94a0546c82e11bbed25dfe1070f4cff14e1e',
        'Changed exact individual adjudication helper')
    require(chunk['schema']=='greenhouse.original848_complete_population_individual_export.v1'
        and chunk['inventory_sha256']==adjudication.INVENTORY_SHA
        and chunk['original_sampled_review_sha256']==adjudication.SAMPLED_SHA
        and chunk['review_sha256']==adjudication.FULL_SHA
        and chunk['exporter_sha256']==sha256(module)
        and Path(chunk['exporter_path']).resolve()==module
        and chunk['original_population']==63 and chunk['individual_holds_preserved']==20
        and chunk['original_sampled_prefix_still_held'] is True
        and chunk['sampling_policy_rerun'] is False and chunk['training_approved'] is False,
        'Only the exact sealed individual adjudication is accepted')
    source=Path(chunk['original_sampled_review_path']).resolve()
    require(source.is_relative_to(ROOT) and sha256(source)==adjudication.SAMPLED_SHA,
        'Original sampled HOLD receipt changed')
    sampled=json.loads(source.read_bytes())
    by,actual=adjudication.check_population(inventory,sampled,review)
    require(review['chunk_inventory_sha256']==sampled['chunk_inventory_sha256']==chunk['inventory_sha256']
        and review['original_sampled_review_sha256']==chunk['original_sampled_review_sha256']
        and review['run_plan_sha256']==sampled['run_plan_sha256']==inventory['run_plan_sha256']
        and review['protocol_sha256']==sampled['protocol_sha256']==inventory['protocol_sha256']
        and review['context_sha256']==sampled['context_sha256']==inventory['context_sha256'],
        'Individual approval lineage differs')
    require(len(chunk['records'])<=43,'Individual accepted population enlarged')
    expected={inventory['context_sha256'][:12]+'_'+name:row for name,row in by.items()}
    names=set()
    for exported in chunk['records']:
        require(exported['id'] in expected and exported['id'] not in names,'Unknown or duplicated adjudicated image')
        names.add(exported['id']);row=expected[exported['id']];name=row['observation_id']
        require(actual[name]['decision']=='accept' and exported['split']=='train'
            and exported['individual_visual_review'] is True
            and exported['review_state']=='automated_pass_and_complete_population_individual_visual_accept'
            and exported['original_sampled_prefix_still_held'] is True
            and exported['sampled_protocol_acceptance_claimed'] is False
            and exported['adjudication_inventory_sha256']==adjudication.INVENTORY_SHA
            and exported['original_sampled_hold_receipt_sha256']==adjudication.SAMPLED_SHA
            and exported['review_receipt_sha256']==adjudication.FULL_SHA
            and Path(exported['review_receipt_path']).resolve()==Path(chunk['review_path']).resolve()
            and exported['training_approved'] is False,'Held, unreviewed or relabeled adjudication record')
        for key in ('target_id','source_target','rgb_sha256','decoded_rgb_sha256','conservative_camera_signature','label_sha256'):
            require(exported[key]==row[key],'Individual exported identity differs: '+key)
        require(exported['source_plant_family']==row['source_family'],'Individual source family differs')
        label_path=Path(row['label_path']).resolve()
        require(label_path.is_relative_to(ROOT) and sha256(label_path)==row['label_sha256'],'Actual individually reviewed label changed')
        label=json.loads(label_path.read_bytes())
        for key in ('answer','query_pixel_uv','accepted_interval_uv'):
            require(exported[key]==label[key],'Individual export changed its reviewed annotation')
        bindings[str(label_path)]=row['label_sha256']
    bindings[str(source)]=adjudication.SAMPLED_SHA
    bindings[str(module)]=sha256(module)
    return len(names)


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
        if chunk.get('schema')=='greenhouse.original848_complete_population_individual_export.v1':
            check_individual_chunk(chunk,inventory,review,bindings)
            continue
        require(review['chunk_accept_or_hold']=='accept' and review['chunk_inventory_sha256']==chunk['inventory_sha256']
            and review['run_plan_sha256']==inventory['run_plan_sha256']
            and review['protocol_sha256']==inventory['protocol_sha256'],'Held or unbound chunk cannot contribute accepted count')
    output.mkdir()
    snapshot_path=output/'status_snapshot.json'
    with snapshot_path.open('xb') as stream:stream.write(status_raw)
    bindings[str(snapshot_path)]=sha256(snapshot_path)
    bindings[str(Path(frozen.__file__).resolve())]=sha256(frozen.__file__)
    bindings[str(Path(__file__).resolve())]=sha256(__file__)
    result=dict(schema='greenhouse.native848_authoritative_export_count.v2',checkpoint_path=str(package),
        observed_source_status_path=str(package/'status.json'),observed_source_status_sha256=hashlib.sha256(status_raw).hexdigest(),
        mutable_source_status_not_in_immutable_bindings=True,accepted_train_count=count,total_exported_images=len(rows),
        stop_future_native_launches=count>=20000,export_snapshot_attempt=attempt+1,source_bindings=bindings,
        count_source='frozen_checkpoint_rows_baseline_plus_immutable_exported_chunks',
        CPU_candidates_counted=False,raw_captures_counted=False,training_approved=False)
    save_json(output/'result.json',result)
    return result
