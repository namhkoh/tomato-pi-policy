"""Combine CPU shards, seal deterministic visual packets, and export passed chunks.

No rendering, label rewriting, view cap, or automatic visual acceptance.
"""
from collections import Counter
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import os
import shutil
import numpy as np
from PIL import Image
from .dataset_review import read_json,write_json,require
from .depth_preview import sha256
from .native848_bulk_admission_v1 import binding
from .clear_cutpoint_contract import crop_image,crop_box


def prepare_review(trial, admissions, prior, prior_sha256, output, protocol_path, protocol_sha256, checkpoint=None):
    from .native848_bulk_visual_sampling_v1 import sample_bulk_chunk,PROTOCOL_SHA256
    trial, output = Path(trial).resolve(),Path(output).resolve()
    receipt=read_json(trial/'result.json')
    require(not (trial/'failure.json').exists() and receipt['training_approved'] is False,
            'Successful owned bulk completion required')
    owned=read_json(binding(dict(path=receipt['owned_exit_path'],sha256=receipt['owned_exit_sha256'])))
    require(owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process',
            'Actual successful owned child exit required')
    manifest=read_json(binding(dict(path=receipt['manifest_path'],sha256=receipt['manifest_sha256'])))
    context=read_json(binding(dict(path=receipt['context_path'],sha256=receipt['context_sha256'])))
    require(manifest['context_sha256']==receipt['context_sha256'] and manifest['automatic_retries'] is False
            and manifest['source_assets_unchanged'] is True,'Capture manifest/context differs')
    old=read_json(binding(dict(path=prior,sha256=prior_sha256)))
    protocol=read_json(binding(dict(path=protocol_path,sha256=protocol_sha256)))
    require(protocol_sha256==PROTOCOL_SHA256,'Changed presealed sampling protocol')
    rows=[];pins={str((trial/'result.json').resolve()):sha256(trial/'result.json')}
    for folder in admissions:
        folder=Path(folder).resolve();request=read_json(folder/'request.json');result=read_json(folder/'result.json')
        require(not (folder/'failure.json').exists() and request['context_sha256']==receipt['context_sha256']
                and request['prior_sha256']==prior_sha256 and result['training_approved'] is False,
                'Successful matching CPU shard required')
        rows.extend(result['records']);pins[str(folder/'result.json')]=sha256(folder/'result.json')
    committed={r['observation_id']:r for r in manifest['observations']}
    require(len(committed)==manifest['committed_frames']==len(rows)
            and len({r['observation_id'] for r in rows})==len(rows)
            and {r['observation_id'] for r in rows}==set(committed),'CPU/native frame inventory differs')
    seen=set()
    for row in old['records']+old['preserved_hold_identity_rows']:
        seen.update((k,row[k]) for k in ('rgb_sha256','decoded_rgb_sha256','conservative_camera_signature'))
    if checkpoint is not None:
        for row in checkpoint_rows(checkpoint)+checkpoint_holds(checkpoint):seen.update(aliases(row))
    rows.sort(key=lambda r:r['capture_index'])
    require(len({r['capture_index'] for r in rows})==len(rows),'Repeated capture index')
    previous=None
    for row_index,row in enumerate(rows):
        native=committed[row['observation_id']]
        require(row['observation_sha256']==native['sha256']
                and Path(row['observation_path']).resolve()==Path(native['path']).resolve(), 'Frame binding differs')
        obs=read_json(binding(dict(path=row['observation_path'],sha256=row['observation_sha256'])))
        sync=obs['synchronization'];token=sync['freshness'];reference=Fraction(*sync['reference_time'])
        if previous:
            require(sync['request_index']>previous['request_index']
                    and token['callback_sequence']>previous['token']['callback_sequence']
                    and reference>=previous['reference'],'Global native continuity failed')
            require(all(token[k]!=previous['token'][k] for k in ('camera_sha256','rgb_sha256','depth_sha256')),
                    'Global adjacent-frame freshness failed')
        previous=dict(request_index=sync['request_index'],token=token,reference=reference)
        require(old['frozen_family_splits'].get(row['source_family'])=='train','Held-out family in TRAIN')
        row_aliases=aliases(row)
        if row_aliases & seen:
            row=dict(row,automated_pass=False,decision='exclude_global_duplicate_or_preserved_hold')
            rows[row_index]=row
        seen.update(row_aliases)
        for kind in ('label','sample','workspace','query_trace','query_selection'):
            binding(dict(path=row[kind+'_path'],sha256=row[kind+'_sha256']))
        if row['automated_pass']:
            label=read_json(row['label_path']);proof=read_json(row['workspace_path'])
            require(label['eligible'] is True and proof['result']['workspace_passed'] is True
                    and label['target_id']==row['target_id']==proof['target_id'], 'Candidate gate changed')
    output.mkdir(parents=True,exist_ok=False)
    eligible=[r for r in rows if r['automated_pass']]
    chunks=[]
    for start in range(0,len(eligible),256):
        chunk_id=f'chunk_{start//256:04d}'
        group=eligible[start:start+256]
        lower=0 if start==0 else eligible[start-1]['capture_index']+1
        upper=group[-1]['capture_index']
        capture_group=[r for r in rows if lower<=r['capture_index']<=upper]
        chosen=sample_bulk_chunk(group,receipt['plan_sha256'],chunk_id,startup=start==0,all_capture_rows=capture_group)
        inventory=dict(schema='greenhouse.original848_bulk_visual_chunk.v1',chunk_id=chunk_id,
            protocol_path=str(Path(protocol_path).resolve()),protocol_sha256=protocol_sha256,
            run_plan_sha256=receipt['plan_sha256'],context_sha256=receipt['context_sha256'],
            trial=str(trial),owned_result_sha256=sha256(trial/'result.json'),records=group,all_capture_rows=capture_group,
            selection=chosen,training_approved=False,accepted_training_increment=0)
        folder=output/chunk_id;folder.mkdir();write_json(folder/'inventory.json',inventory)
        chunks.append(dict(chunk_id=chunk_id,path=str(folder/'inventory.json'),sha256=sha256(folder/'inventory.json'),
                           automated_candidates=len(group),selection=chosen))
    result=dict(schema='greenhouse.original848_bulk_review_inventory.v1',native_frames=len(rows),
        automated_candidates=len(eligible),chunks=chunks,records=rows,source_bindings=pins,
        prior_path=str(Path(prior).resolve()),prior_sha256=prior_sha256,training_approved=False)
    write_json(output/'result.json',result)
    return result


def link_checked(source,destination,expected):
    source,destination=Path(source),Path(destination)
    require(sha256(source)==expected,'Changed export source')
    destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.exists(),'Export collision')
    try:os.link(source,destination)
    except OSError:shutil.copy2(source,destination)
    require(sha256(destination)==expected,'Exported artifact hash differs')


def start_checkpoint(prior_checkpoint,output):
    """Link immutable assets only; index files are newly written, never linked."""
    prior,output=Path(prior_checkpoint).resolve(),Path(output).resolve()
    require(not output.exists(),'New bulk checkpoint required');output.mkdir(parents=True)
    rows=[json.loads(line) for line in (prior/'index.jsonl').read_text(encoding='utf-8').splitlines() if line]
    selection=read_json(prior/'evidence/selection.json')
    identities={r['id']:r for r in selection['records']}
    for row in rows:
        for key in ('decoded_rgb_sha256','conservative_camera_signature'):
            row[key]=identities[row['id']][key]
    linked=set()
    for row in rows:
        for relative in row['files'].values():
            if relative in linked:continue
            src=prior/relative;link_checked(src,output/relative,sha256(src));linked.add(relative)
    write_json(output/'baseline.json',dict(prior_checkpoint=str(prior),
        prior_index_sha256=sha256(prior/'index.jsonl'),records=rows,training_approved=False))
    (output/'chunks').mkdir()
    refresh_index(output)


def export_chunk(inventory_path,inventory_sha256,review_path,review_sha256,output):
    inventory=read_json(binding(dict(path=inventory_path,sha256=inventory_sha256)))
    review=read_json(binding(dict(path=review_path,sha256=review_sha256)))
    require(review['chunk_inventory_sha256']==inventory_sha256
            and review['protocol_sha256']==inventory['protocol_sha256']
            and review['run_plan_sha256']==inventory['run_plan_sha256'], 'Review inventory/profile binding differs')
    if review['chunk_accept_or_hold']!='accept':
        preserve_holds(inventory,review,review_path,review_sha256,output)
        raise ValueError('Actual visual QA held this chunk; aliases preserved and no rows exported')
    # The inventory's selection is sealed before actual image inspection.
    from .native848_bulk_visual_sampling_v1 import PROTOCOL_SHA256
    require(inventory['protocol_sha256']==inventory['selection']['protocol_sha256']==PROTOCOL_SHA256, 'Visual protocol identity differs')
    expected=set(inventory['selection']['sample_ids'])
    reviewed={r['observation_id']:r for r in review['reviews']}
    eligibility={r['observation_id']:r['eligible_automated_candidate'] for r in inventory['selection']['samples']}
    require(len(reviewed)==len(review['reviews']) and set(reviewed)==expected,'Duplicate or differing actual review sample IDs')
    for name,r in reviewed.items():
        require(r['full_native_image_inspected'] is True
            and r['unscaled_lossless_association_crop_inspected'] is True
            and r['obvious_temporal_ghosting_observed'] is False,'Missing actual visual inspection or temporal artifact')
        if eligibility[name]:
            require(r['decision']=='accept' and r['target_query_association_clear'] is True,
                    'An eligible sampled frame did not receive actual clear acceptance')
        else:
            require(r['automated_exclusion_remains_excluded'] is True,'Excluded transition control cannot enter dataset')
    output=Path(output).resolve();exports=[]
    chunk_key=inventory['context_sha256'][:12]+'_'+inventory['chunk_id']
    require(not (output/'chunks'/(chunk_key+'.json')).exists(),'Chunk already exported')
    seen=set()
    for old_row in checkpoint_rows(output)+checkpoint_holds(output):seen.update(aliases(old_row))
    remaining=20000-read_json(output/'status.json')['counts']['train']
    require(remaining>0,'Global TRAIN goal already complete')
    exclusions=[]
    for row in inventory['records']:
        if len(exports)>=remaining:break
        row_aliases=aliases(row)
        if row_aliases & seen:
            exclusions.append(dict(observation_id=row['observation_id'],reason='existing_dataset_or_visual_hold_alias'))
            continue
        seen.update(row_aliases)
        name=row['observation_id'];sid=inventory['context_sha256'][:12]+'_'+name
        require(row['automated_pass'] and Path(sid).name==sid,'Unsafe or ineligible export row')
        label=read_json(binding(dict(path=row['label_path'],sha256=row['label_sha256'])))
        obs=read_json(binding(dict(path=row['observation_path'],sha256=row['observation_sha256'])))
        if name in reviewed:
            actual=reviewed[name]
            require(actual['rgb_sha256']==row['rgb_sha256'] and actual['label_sha256']==row['label_sha256']
                    and actual['source_sample_sha256']==row['sample_sha256'],'Reviewed artifact identity differs')
        files={}
        evidence_root=output/'evidence/contexts'/inventory['context_sha256']
        for key,pin in [('context',dict(path=obs['context_path'],sha256=obs['context_sha256'])),
                        ('geometry_'+obs['geometry_proof']['pose_key'],obs['geometry_proof']),
                        ('mapping_'+obs['mapping']['sha256'],obs['mapping'])]:
            destination=evidence_root/(key+'.json')
            if not destination.exists():link_checked(pin['path'],destination,pin['sha256'])
        assets=dict(rgb=(row['rgb_path'],row['rgb_sha256'],f'train/images/{sid}.png'),
            label=(row['label_path'],row['label_sha256'],f'train/annotations/{sid}.json'),
            target_mask=(row['target_mask_path'],row['target_mask_sha256'],f'train/annotations/{sid}_target.png'),
            buffers=(obs['files']['buffers']['path'],obs['files']['buffers']['sha256'],f'train/depth/{sid}.npz'),
            sample=(row['sample_path'],row['sample_sha256'],f'evidence/samples/{sid}.json'),
            workspace=(row['workspace_path'],row['workspace_sha256'],f'evidence/workspace/{sid}.json'),
            observation=(row['observation_path'],row['observation_sha256'],f'evidence/observations/{sid}.json'))
        for kind in ('query_trace','query_selection'):
            assets[kind]=(row[kind+'_path'],row[kind+'_sha256'],f'evidence/{kind}/{sid}.json')
        for kind,(source,pin,relative) in assets.items():
            link_checked(source,output/relative,pin);files[kind]=relative
        with Image.open(row['rgb_path']) as rgb:
            require(rgb.size==(848,408) and rgb.mode=='RGB','Native RGB required')
            crop=crop_image(rgb,label['query_pixel_uv']);relative=f'train/crops/{sid}.png'
            (output/relative).parent.mkdir(parents=True,exist_ok=True);crop.save(output/relative,compress_level=1)
            files['crop']=relative
        exports.append(dict(id=sid,split='train',target_id=row['target_id'],source_target=row['source_target'],
            source_plant_family=row['source_family'],kind='original_bulk_native848',files=files,
            rgb_sha256=row['rgb_sha256'],label_sha256=row['label_sha256'],decoded_rgb_sha256=row['decoded_rgb_sha256'],
            conservative_camera_signature=row['conservative_camera_signature'],answer=label['answer'],
            query_pixel_uv=label['query_pixel_uv'],accepted_interval_uv=label['accepted_interval_uv'],
            crop_box_xyxy=crop_box(label['query_pixel_uv']),individual_visual_review=name in reviewed,
            review_state='automated_pass_and_individual_visual_accept' if name in reviewed else 'automated_pass_with_sampled_batch_QA',
            review_receipt_path=str(Path(review_path).resolve()),review_receipt_sha256=review_sha256,
            per_target_view_cap=None,geometry_novelty_qualified=False,training_approved=False))
    write_json(output/'chunks'/(chunk_key+'.json'),dict(inventory_path=str(Path(inventory_path).resolve()),
        inventory_sha256=inventory_sha256,review_path=str(Path(review_path).resolve()),review_sha256=review_sha256,
        records=exports,excluded=exclusions,training_approved=False))
    return refresh_index(output)


def aliases(row):
    return {(key,row[key]) for key in ('rgb_sha256','decoded_rgb_sha256','conservative_camera_signature')}


def checkpoint_rows(output):
    output=Path(output);rows=read_json(output/'baseline.json')['records']
    for path in sorted((output/'chunks').glob('*.json')):rows.extend(read_json(path)['records'])
    return rows


def checkpoint_holds(output):
    path=Path(output)/'visual_holds.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.is_file() else []


def preserve_holds(inventory,review,review_path,review_sha256,output):
    by={r['observation_id']:r for r in inventory['all_capture_rows']}
    path=Path(output)/'visual_holds.jsonl'
    with path.open('a',encoding='utf-8') as stream:
        for actual in review['reviews']:
            if actual['decision'] not in ('hold','reject'):continue
            row=by[actual['observation_id']]
            require(actual['rgb_sha256']==row['rgb_sha256'] and actual['label_sha256']==row['label_sha256'],
                    'Held image identity differs')
            saved=dict(observation_id=row['observation_id'],**{k:row[k] for k in
                ('rgb_sha256','decoded_rgb_sha256','conservative_camera_signature')},
                review_path=str(Path(review_path).resolve()),review_sha256=review_sha256,reason=actual['reason'])
            stream.write(json.dumps(saved,allow_nan=False)+'\n')


def refresh_index(output):
    output=Path(output);rows=checkpoint_rows(output)
    seen=set()
    for row in rows:
        require(not (seen & aliases(row)),'Duplicate encoded/decoded RGB or conservative camera')
        seen.update(aliases(row))
    require(len({r['id'] for r in rows})==len(rows),'Duplicate export ID')
    require(len({r['rgb_sha256'] for r in rows})==len(rows),'Duplicate export RGB')
    counts=dict(Counter(r['split'] for r in rows))
    require(counts['train']<=20000,'Global TRAIN goal exceeded')
    for split in ('train','validation','test',None):
        selected=rows if split is None else [r for r in rows if r['split']==split]
        path=output/'index.jsonl' if split is None else output/split/'index.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in selected),encoding='utf-8');os.replace(temp,path)
    result=dict(schema='greenhouse.original848_bulk_checkpoint.v1',counts=counts,total_images=len(rows),
        native_resolution=[848,408],per_target_view_cap=None,train_original_targets=len({r['source_target'] for r in rows if r['split']=='train'}),
        individually_reviewed_new_bulk=sum(r.get('individual_visual_review',False) for r in rows if r.get('kind')=='original_bulk_native848'),
        goal_complete=counts['train']==20000,training_approved=False)
    temp=output/'status.tmp.json';temp.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');os.replace(temp,output/'status.json')
    return result
