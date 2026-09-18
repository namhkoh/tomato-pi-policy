"""Export only individually adjudicated members of one fully reviewed held prefix.

The sampled HOLD stays unchanged. This never rerolls its sample, annotates a
frame, or invokes sampled-chunk acceptance. Root must serialize package writes.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
from PIL import Image
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint
from .clear_cutpoint_contract import crop_image, crop_box
from .training_export import view_signature
from . import native848_bulk_delivery_v1 as frozen

INVENTORY_SHA='340171733ca7e77ec4e7fee39340369c9dfb5b2e31a2cf92e6af82afd406cb98'
SAMPLED_SHA='4161824e39c2ea0f013aae71318f35678975a2db9ee2c73c531b814558be830c'
FULL_SHA='519334c3ca6f7f2f44f317cdce4de9d489b75c78ae8033b5d4e02be0bfcdff34'
FROZEN_DELIVERY_SHA='fdb64cb37a6a2f1f3d1953b9147eb1ed934c262ab3e426cb97c239e7e9f8eb26'


def pin_file(path, expected, pins):
    path=Path(path).resolve()
    require(sha256(path)==expected,'Changed adjudication artifact: '+str(path))
    require(str(path) not in pins or pins[str(path)]==expected,'Conflicting artifact binding')
    pins[str(path)]=expected
    return path


def check_population(inventory, sampled, full):
    """Pure guards; synthetic negative tests do not rewrite actual receipts."""
    require(inventory['schema']=='greenhouse.original848_bulk_visual_chunk.v1'
        and sampled['schema']=='greenhouse.original848_bulk_actual_sampled_visual_review.v1'
        and full['schema']=='greenhouse.original848_bulk_complete_population_visual_review.v1',
        'Complete-population adjudication required; sampled-only receipt is insufficient')
    require(sampled['chunk_accept_or_hold']=='hold' and sampled['whole_prefix_held'] is True
        and full['chunk_accept_or_hold']=='hold' and full['original_whole_prefix_hold_preserved'] is True
        and full['full_population_review_complete'] is True and full['selection_policy_rerun'] is False
        and full['sample_reroll_performed'] is False and full['chunk_export_performed'] is False,
        'Original sampled HOLD or complete-population scope changed')
    require(full['eligible_population']==full['population_individually_reviewed']==63
        and full['previous_actual_reviews_carried_unchanged']==10
        and full['new_actual_full_native_and_unscaled_crop_reviews']==53,
        'Exactly the approved complete63 population is required')
    by={r['observation_id']:r for r in inventory['records']}
    actual={r['observation_id']:r for r in full['reviews']}
    prior={r['observation_id']:r for r in sampled['reviews']}
    require(len(by)==len(inventory['records'])==len(actual)==len(full['reviews'])==63
        and set(by)==set(actual) and len(prior)==len(sampled['reviews'])==10,
        'Missing, duplicate or additional population member')
    require(set(prior)==set(inventory['selection']['sample_ids']) and set(prior)<=set(actual),
        'Original predetermined sample changed')
    require(all(actual[name]==r for name,r in prior.items()),
        'Original sampled decisions, including holds, must remain unchanged')
    counts=Counter(r['decision'] for r in actual.values())
    require(counts==dict(accept=43,hold=20) and full['individual_accepts']==43
        and full['individual_holds']==20,'Approved individual decisions changed')
    require(set(full['held_observation_ids'])=={n for n,r in actual.items() if r['decision']=='hold'}
        and set(full['accepted_observation_ids'])=={n for n,r in actual.items() if r['decision']=='accept'},
        'Decision inventories differ')
    for name,r in actual.items():
        source=by[name]
        require(source['automated_pass'] is True and source['split']=='train'
            and r['full_native_image_inspected'] is True
            and r['unscaled_lossless_association_crop_inspected'] is True
            and r['individual_visual_review'] is True and r['obvious_temporal_ghosting_observed'] is False
            and bool(r['reviewer']) and bool(r['reason']) and r['training_approved'] is False,
            'Missing actual individual inspection or unchanged automated eligibility')
        require(r['target_query_association_clear'] is (r['decision']=='accept'),
            'Accepted association is unclear, or held association was silently accepted')
        for kind in ('rgb','label','sample','observation'):
            require(r[kind+'_sha256']==source[kind+'_sha256'],'Reviewed source identity differs')
        require(r['source_sample_sha256']==source['sample_sha256'],'Source sample differs')
    return by,actual


def validate(inventory_path,inventory_sha256,sampled_path,sampled_sha256,full_path,full_sha256):
    """Read-only flat source/artifact check. No annotation or native replay."""
    require((inventory_sha256,sampled_sha256,full_sha256)==(INVENTORY_SHA,SAMPLED_SHA,FULL_SHA),
        'This helper is scoped to the exact approved held inventory and review receipts')
    require(sha256(frozen.__file__)==FROZEN_DELIVERY_SHA,'Frozen linking/index implementation changed')
    pins={str(Path(__file__).resolve()):sha256(__file__),str(Path(frozen.__file__).resolve()):FROZEN_DELIVERY_SHA}
    paths=[pin_file(p,h,pins) for p,h in ((inventory_path,inventory_sha256),(sampled_path,sampled_sha256),(full_path,full_sha256))]
    inventory,sampled,full=map(read_json,paths)
    require(full['chunk_inventory_sha256']==sampled['chunk_inventory_sha256']==inventory_sha256
        and Path(full['chunk_inventory_path']).resolve()==Path(sampled['chunk_inventory_path']).resolve()==paths[0]
        and full['original_sampled_review_sha256']==sampled_sha256
        and Path(full['original_sampled_review_path']).resolve()==paths[1]
        and full['protocol_sha256']==sampled['protocol_sha256']==inventory['protocol_sha256']
        and full['run_plan_sha256']==sampled['run_plan_sha256']==inventory['run_plan_sha256']
        and full['context_sha256']==sampled['context_sha256']==inventory['context_sha256'],
        'Actual receipt lineage differs')
    by,actual=check_population(inventory,sampled,full)
    closure_path=pin_file(inventory['prefix_closure_path'],inventory['prefix_closure_sha256'],pins)
    require(full['prefix_closure_sha256']==inventory['prefix_closure_sha256']
        and Path(full['prefix_closure_path']).resolve()==closure_path,'Closure binding differs')
    closure=read_json(closure_path)
    require(closure['records']==inventory['all_capture_rows']
        and [r for r in closure['records'] if r['automated_pass']]==inventory['records']
        and closure['flat_source_bindings_verified_at_closure'] is True
        and closure['context_sha256']==inventory['context_sha256'],'Sealed actual native/CPU population differs')
    verify_bindings(closure['source_bindings']);verify_bindings(full['source_bindings'])
    pins.update(closure['source_bindings']);pins.update(full['source_bindings'])
    for name,row in by.items():
        saved={}
        for kind in ('label','sample','workspace','query_trace','query_selection','observation'):
            saved[kind]=read_json(pin_file(row[kind+'_path'],row[kind+'_sha256'],pins))
        pin_file(row['target_mask_path'],row['target_mask_sha256'],pins)
        pin_file(row['rgb_path'],row['rgb_sha256'],pins)
        label,obs,meta=saved['label'],saved['observation'],saved['sample']
        require(label['eligible'] is True and saved['workspace']['result']['workspace_passed'] is True
            and saved['query_trace']['passed'] is True
            and label['target_id']==row['target_id']==saved['workspace']['target_id']==obs['target_id']
            and obs['context_sha256']==inventory['context_sha256'] and obs['split']=='train'
            and obs['source_family']==row['source_family'] and meta['observation_sha256']==row['observation_sha256'],
            'Original clarity, query, workspace or native identity gate differs')
        require(obs['files']['rgb']['sha256']==row['rgb_sha256']
            and Path(obs['files']['rgb']['path']).resolve()==Path(row['rgb_path']).resolve(),
            'Native RGB identity differs')
        for item in (obs['files']['buffers'],obs['mapping'],obs['geometry_proof']):pin_file(item['path'],item['sha256'],pins)
        require(read_json(obs['geometry_proof']['path'])['screen']['passed'] is True,'Native geometry was not admitted')
        with Image.open(row['rgb_path']) as rgb:
            require(rgb.size==(848,408) and rgb.mode=='RGB'
                and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_rgb_sha256'],
                'Native image or decoded alias differs')
            v=actual[name];cp=pin_file(v['association_crop_path'],v['association_crop_sha256'],pins)
            with Image.open(cp) as crop:
                expected=rgb.crop(v['association_crop_box'])
                require(crop.mode==expected.mode and crop.size==expected.size and crop.tobytes()==expected.tobytes(),
                    'Reviewed crop is not the exact unscaled lossless native crop')
        require(view_signature(meta,row['source_family'],None)==row['conservative_camera_signature'],
            'Conservative camera alias differs')
    return inventory,full,pins


def checkpoint_snapshot(output):
    output=Path(output).resolve()
    paths=[output/'baseline.json',output/'status.json',*sorted((output/'chunks').glob('*.json'))]
    if (output/'visual_holds.jsonl').exists():paths.append(output/'visual_holds.jsonl')
    return {str(p):sha256(p) for p in paths}


def inspect(inventory_path,inventory_sha256,sampled_path,sampled_sha256,full_path,full_sha256,output):
    inventory,full,pins=validate(inventory_path,inventory_sha256,sampled_path,sampled_sha256,full_path,full_sha256)
    output=Path(output).resolve();snapshot=checkpoint_snapshot(output)
    existing=frozen.checkpoint_rows(output);old_holds=frozen.checkpoint_holds(output)
    current=sum(r['split']=='train' for r in existing)
    require(read_json(output/'status.json')['counts']['train']==current and current<=20000,'Checkpoint count differs')
    actual={r['observation_id']:r for r in full['reviews']}
    held=[r for r in inventory['records'] if actual[r['observation_id']]['decision']=='hold']
    seen=set()
    for row in existing+old_holds+held:seen.update(frozen.aliases(row))
    records=[];excluded=[];remaining=20000-current
    for row in inventory['records']:
        name=row['observation_id']
        if actual[name]['decision']=='hold':
            excluded.append(dict(observation_id=name,reason='actual_individual_ambiguity_hold'));continue
        if frozen.aliases(row)&seen:
            excluded.append(dict(observation_id=name,reason='existing_dataset_or_visual_hold_alias'));continue
        if len(records)>=remaining:
            excluded.append(dict(observation_id=name,reason='global_20000_TRAIN_goal'));continue
        seen.update(frozen.aliases(row));records.append(row)
    key='individual_adjudication_'+inventory['context_sha256'][:12]+'_'+full_sha256[:16]
    require(not (output/'chunks'/(key+'.json')).exists(),'This individual adjudication was already exported')
    require(checkpoint_snapshot(output)==snapshot,'Checkpoint changed during read-only inspection')
    return dict(schema='greenhouse.original848_individual_adjudication_proposal.v1',chunk_key=key,
        inventory_path=str(Path(inventory_path).resolve()),inventory_sha256=inventory_sha256,
        original_sampled_review_path=str(Path(sampled_path).resolve()),original_sampled_review_sha256=sampled_sha256,
        review_path=str(Path(full_path).resolve()),review_sha256=full_sha256,records=records,excluded=excluded,
        current_train=current,proposed_individual_increment=len(records),projected_train=current+len(records),
        original_population=63,individual_accepts=43,individual_holds=20,source_bindings=pins,
        checkpoint_snapshot=snapshot,checkpoint_snapshot_sha256=fingerprint(snapshot),
        sampled_prefix_still_held=True,sampling_policy_rerun=False,training_approved=False)


def export_individually_reviewed(inventory_path,inventory_sha256,sampled_path,sampled_sha256,full_path,full_sha256,output):
    """Root-only serialized mutation after independent source review."""
    proposal=inspect(inventory_path,inventory_sha256,sampled_path,sampled_sha256,full_path,full_sha256,output)
    output=Path(output).resolve();inventory=read_json(inventory_path);full=read_json(full_path)
    require(checkpoint_snapshot(output)==proposal['checkpoint_snapshot'],'Checkpoint changed before export')
    # Persist all new hold aliases before any accepted image can enter the bank.
    known=set()
    for row in frozen.checkpoint_holds(output):known.update(frozen.aliases(row))
    by={r['observation_id']:r for r in inventory['records']};new_holds=[]
    for actual in full['reviews']:
        row=by[actual['observation_id']]
        if actual['decision']=='hold' and not frozen.aliases(row)<=known:
            new_holds.append(actual);known.update(frozen.aliases(row))
    if new_holds:frozen.preserve_holds(inventory,dict(reviews=new_holds),full_path,full_sha256,output)
    exports=[]
    for row in proposal['records']:
        name=row['observation_id'];sid=inventory['context_sha256'][:12]+'_'+name
        require(Path(sid).name==sid,'Unsafe export ID')
        label=read_json(row['label_path']);obs=read_json(row['observation_path'])
        evidence_root=output/'evidence/contexts'/inventory['context_sha256']
        for key,item in [('context',dict(path=obs['context_path'],sha256=obs['context_sha256'])),
                ('geometry_'+obs['geometry_proof']['pose_key'],obs['geometry_proof']),('mapping_'+obs['mapping']['sha256'],obs['mapping'])]:
            dest=evidence_root/(key+'.json')
            if dest.exists():require(sha256(dest)==item['sha256'],'Existing portable evidence changed')
            else:frozen.link_checked(item['path'],dest,item['sha256'])
        assets=dict(rgb=(row['rgb_path'],row['rgb_sha256'],f'train/images/{sid}.png'),
            label=(row['label_path'],row['label_sha256'],f'train/annotations/{sid}.json'),
            target_mask=(row['target_mask_path'],row['target_mask_sha256'],f'train/annotations/{sid}_target.png'),
            buffers=(obs['files']['buffers']['path'],obs['files']['buffers']['sha256'],f'train/depth/{sid}.npz'),
            sample=(row['sample_path'],row['sample_sha256'],f'evidence/samples/{sid}.json'),
            workspace=(row['workspace_path'],row['workspace_sha256'],f'evidence/workspace/{sid}.json'),
            observation=(row['observation_path'],row['observation_sha256'],f'evidence/observations/{sid}.json'))
        for kind in ('query_trace','query_selection'):assets[kind]=(row[kind+'_path'],row[kind+'_sha256'],f'evidence/{kind}/{sid}.json')
        files={}
        for kind,(src,pin,relative) in assets.items():frozen.link_checked(src,output/relative,pin);files[kind]=relative
        with Image.open(row['rgb_path']) as rgb:
            relative=f'train/crops/{sid}.png';dest=output/relative
            require(not dest.exists(),'Query crop collision');dest.parent.mkdir(parents=True,exist_ok=True)
            crop_image(rgb,label['query_pixel_uv']).save(dest,compress_level=1);files['crop']=relative
        exports.append(dict(id=sid,split='train',target_id=row['target_id'],source_target=row['source_target'],
            source_plant_family=row['source_family'],kind='original_bulk_native848',files=files,
            rgb_sha256=row['rgb_sha256'],label_sha256=row['label_sha256'],decoded_rgb_sha256=row['decoded_rgb_sha256'],
            conservative_camera_signature=row['conservative_camera_signature'],answer=label['answer'],
            query_pixel_uv=label['query_pixel_uv'],accepted_interval_uv=label['accepted_interval_uv'],
            crop_box_xyxy=crop_box(label['query_pixel_uv']),individual_visual_review=True,
            review_state='automated_pass_and_complete_population_individual_visual_accept',
            review_receipt_path=str(Path(full_path).resolve()),review_receipt_sha256=full_sha256,
            adjudication_inventory_sha256=inventory_sha256,original_sampled_hold_receipt_sha256=sampled_sha256,
            original_sampled_prefix_still_held=True,sampled_protocol_acceptance_claimed=False,
            per_target_view_cap=None,geometry_novelty_qualified=False,training_approved=False))
    write_json(output/'chunks'/(proposal['chunk_key']+'.json'),dict(
        schema='greenhouse.original848_complete_population_individual_export.v1',
        inventory_path=proposal['inventory_path'],inventory_sha256=inventory_sha256,
        original_sampled_review_path=proposal['original_sampled_review_path'],original_sampled_review_sha256=sampled_sha256,
        review_path=proposal['review_path'],review_sha256=full_sha256,records=exports,excluded=proposal['excluded'],
        original_population=63,individual_holds_preserved=20,original_sampled_prefix_still_held=True,
        sampling_policy_rerun=False,checkpoint_before_sha256=proposal['checkpoint_snapshot_sha256'],
        exporter_path=str(Path(__file__).resolve()),exporter_sha256=sha256(__file__),training_approved=False))
    return frozen.refresh_index(output)
