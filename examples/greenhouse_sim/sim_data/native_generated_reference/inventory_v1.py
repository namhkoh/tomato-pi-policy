"""Explicit pinned owner receipt -> all observed rows; no scans or approvals."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json

from . import execution_v1 as ex
from ..native_original_capture import contracts as oc


def build_inventory(receipt_path, *, receipt_sha256):
    from .audit_v1 import audit_capture
    from ..native_dataset import inventory as inv, original_inventory as oi
    from ..native_original_capture.prepare import case_plan
    from ..native_dataset.bundle import SampleReader
    receipt_path=oc.pin(receipt_path,receipt_sha256);root=receipt_path.parent
    oc.require(receipt_path.name=='launcher_receipt.json' and not (root/'owner_failure.json').exists(), 'Completed known owner receipt required')
    receipt=oc.read_json(receipt_path)
    oc.require(receipt['schema']==ex.RECEIPT and receipt['state']=='owned_exit0_and_independent_saved_native_replay'
        and type(receipt['exit_code']) is int and receipt['exit_code']==0
        and all(receipt[k] is v for k,v in ex.FLAGS.items()), 'Incomplete/unapproved producer contract')
    request,plan=ex.preflight_request(receipt['execution_request_path'],receipt['execution_request_sha256'])
    oc.require(Path(request['output'])==root and Path(receipt['capture'])==root/'capture'
        and receipt['source_bindings']==request['source_bindings']
        and receipt['implementation_bindings']==request['implementation_bindings'], 'Receipt targets another executor/output')
    worker=oc.read_json(oc.pin(root/'owned_worker.json',receipt['owned_worker_sha256']))
    started=oc.read_json(oc.pin(root/'owner_started.json',receipt['owner_started_sha256']))
    exited=oc.read_json(oc.pin(root/'exit.json',receipt['exit_sha256']))
    oc.require(worker['command']==ex.worker_command(receipt['execution_request_path'],receipt['execution_request_sha256'],request)
        and type(worker['launcher_pid']) is int and worker['launcher_pid']>0
        and type(worker['owner_pid']) is int and worker['owner_pid']>0
        and started['owner_pid']==worker['owner_pid']
        and started['execution_request_path']==receipt['execution_request_path']
        and started['execution_request_sha256']==receipt['execution_request_sha256']
        and type(exited['exit_code']) is int and exited['exit_code']==0
        and exited['owned_worker_sha256']==receipt['owned_worker_sha256'], 'Owned command/exit chain changed')
    saved=oc.read_json(oc.pin(root/'postexit_audit.json',receipt['postexit_audit_sha256']))
    replay=audit_capture(root/'capture',result_sha256=receipt['result_sha256'])
    oc.require(replay==saved and replay['request_sha256']==receipt['request_sha256'], 'Independent native replay differs from owner receipt')
    bindings=inv._Bindings();bindings.mapping(replay['source_bindings'])
    bindings.mapping({str(receipt_path):receipt_sha256,str(root/'postexit_audit.json'):receipt['postexit_audit_sha256'],
        str(root/'owned_worker.json'):receipt['owned_worker_sha256'],str(root/'exit.json'):receipt['exit_sha256']})
    bindings.mapping({str(root/'owner_started.json'):receipt['owner_started_sha256']})
    # Reuse original accounting on the ACTUAL verified native-original plan,
    # never on a generated document relabelled as an old manifest or V5 receipt.
    proof=bindings.document(plan['anchor_evidence']['path'],plan['anchor_evidence']['sha256'])
    native=proof['native_observation']['native_plan'];original=bindings.document(native['path'],native['sha256'])
    cases=[c for c in original['cases'] if c['source_row']==plan['source_row']]
    oc.require(len(cases)==1,'Exact authenticated original source case required')
    basis,lineage=oi._original_basis(bindings,case_plan(original,cases[0]))
    result=bindings.document(root/'capture/result.json',receipt['result_sha256'])
    captured={r['candidate_id']:r for r in result['records']}
    qualification=bindings.document(Path(plan['variant_directory'])/'qualification.json')
    generated_assets=sorted(h for name,h in qualification['output_hashes'].items() if Path(name).suffix in ('.usd','.usda','.usdc'))
    oc.require(generated_assets,'Actual generated geometry assets required')
    rows,held=[],[]
    for review in replay['records']:
        mode=review['candidate_id']
        if review['decision']=='held_pre_render':
            held.append(deepcopy(review));continue
        record=captured[mode];folder=root/'capture'/mode
        reader=SampleReader(folder,expected_bindings={'sample.json':record['sample_sha256'],
            'supervision/label.json':record['label_sha256']},expected_json=(
            {'supervision/query_trace.json':record['query_trace']} if record['query_trace'] is not None else {}))
        meta=reader.metadata;actual=deepcopy(basis)
        actual.update(lighting=deepcopy(meta['lighting']),scene_counts=deepcopy(meta['scene_counts']),renderer=meta['renderer'])
        target=meta['supervision']['target_id'];is_generated=mode=='generated_variant'
        if is_generated:
            actual['target_geometry_sha256']=inv._hash(generated_assets)
            old_root=actual['target_plant_root'];actual['target_plant_root']='/World/GeneratedNativePilot/'+plan['generated_row']['variant_id']
            for placement in actual['population_placements']:
                if placement['plant_root']==old_root:
                    placement.update(plant_root=actual['target_plant_root'],geometry_content_sha256=inv._hash(generated_assets),asset_count=len(generated_assets))
        scene,camera=oi._features(meta,actual)
        logical=review['logical_files'];fresh=meta['synchronization']['freshness']
        row=dict(sample_id=inv._hash(dict(capture=receipt['capture'],candidate_id=mode)),candidate_id=mode,
            sample_path=str(folder),capture_path=receipt['capture'],source_kind='generated_native' if is_generated else 'original_native',
            target_id=target,variant_target=target,source_family=plan['source_family'],original_donor_family=plan['source_family'],
            source_target=plan['source_row']['target_id'],conservative_view_cap_group=plan['conservative_view_cap_group'],
            split='train',resolution=list(oc.RESOLUTION),decision=review['decision'],
            context_id=('generated:' if is_generated else 'original:')+target,geometry_sha256=actual['target_geometry_sha256'],
            image_bytes=logical['inputs/rgb.png']['bytes'],encoded_rgb_sha256=review['rgb_sha256'],
            decoded_rgb_sha256=review['decoded_rgb_sha256'],decoded_rgb_bytes=1696*816*3,callback_rgb_sha256=fresh['rgb_sha256'],
            static_scene_basis=actual,annotation_review=dict(passed=review['decision']==inv.STRICT,method='automatic',
                evidence_id=receipt_sha256,evidence_basis='independently_replayed_new_producer_native_buffers',
                audit_execution_independently_verified=True,prior_audit_execution_verified_from_pin_alone=False),
            provenance=dict(receipt_path=str(receipt_path),receipt_sha256=receipt_sha256,
                plan_path=request['plan_path'],plan_sha256=request['plan_sha256'],request_sha256=receipt['request_sha256'],
                result_sha256=receipt['result_sha256'],sample_sha256=record['sample_sha256'],label_sha256=record['label_sha256'],
                logical_files=logical,lineage=lineage,original_source_row=plan['source_row'],
                generated_source_row=plan['generated_row'] if is_generated else None,anchor_evidence=plan['anchor_evidence'],
                scene_evidence=meta['scene_evidence'],callback=meta['synchronization'],
                process_history_independently_attested=False),**ex.FLAGS)
        oi._identities(row,scene,camera);rows.append(row)
    bindings.finish();oc.pin(receipt_path,receipt_sha256)
    return dict(schema='greenhouse.generated_original_reference_observed_inventory.v1',records=rows,
        held_pre_render=held,counts=dict(Counter(r['decision'] for r in rows)),
        planned_case_count=2,captured_row_count=len(rows),held_pre_render_count=len(held),
        exact_duplicates=inv._duplicates(rows),source_bindings=dict(sorted(bindings.hashes.items())),
        implementation_bindings=ex.implementation_bindings(),scope='one_explicit_completed_pair_not_global_inventory',
        geometry_hash_is_independence_proof=False,**ex.FLAGS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True);p.add_argument('--receipt-sha256',required=True)
    a=p.parse_args(argv)
    print(json.dumps(build_inventory(a.receipt,receipt_sha256=a.receipt_sha256),indent=2))


if __name__=='__main__':
    main()
