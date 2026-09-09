"""Portable audited RGB grounding release, with native depth/geometry sidecars.

Never mutate pilot approvals. A new task-specific synthetic label is derived
under the versioned contract; source review history remains separate evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .audit import audit_manifest
from .dataset_review import SCHEMA, read_json, require, safe_file, verify_bindings, write_json
from .depth_preview import sha256
from .training_contract import CONTRACT, SYSTEM_PROMPT, contract_hash, derive_label, validate_answer, user_prompt
from .capture_contract import fingerprint
from .dataset_package import active_reviews
from .query_visibility import QueryVisibility

SCHEMA_RELEASE='greenhouse.grounding_training_release.v1'
# A complete first substantive release, not a ten-image plumbing demonstration.
RELEASE_GATES={'minimum_rows':{'train':10000,'validation':500,'test':500},
               'minimum_families':{'train':16,'validation':4,'test':4},
               'minimum_targets':{'train':100,'validation':20,'test':20},
               'minimum_difficulty_rows_per_split':{'easy':20,'medium':20,'hard':20},
               'minimum_unique_cut_pixel_bins_train':100,
               'minimum_query_only_error_px':10.0}
LEGACY_ENGINEERING_GATES=dict(RELEASE_GATES)
RELEASE_GATES={**RELEASE_GATES,'minimum_localized_rows':{'train':5000,'validation':250,'test':250},
               'minimum_occluded_rows':{'train':1000,'validation':50,'test':50}}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def chat_row(sid,image,label):
    answer=validate_answer(label['answer'])
    return dict(id=sid,images=[image],messages=[
        dict(role='system',content=SYSTEM_PROMPT),
        dict(role='user',content='<image>\n'+user_prompt(label['query_pixel_uv'])),
        dict(role='assistant',content=json.dumps(answer,separators=(',',':'),allow_nan=False))])


def view_signature(metadata,family,lighting=None):
    # Renderer noise makes two renders of the SAME pose have different RGB hashes.
    # Deduplicate their camera/geometry condition as well, not only their pixels.
    cal=metadata['calibration']
    return fingerprint(dict(family=family,lighting=lighting,
        camera_to_world=np.round(cal['camera_to_world_usd_row_vectors'],8).tolist(),
        intrinsics=np.round(cal['intrinsics'],8).tolist()))


def check_release_rows(rows,*,gates=None):
    gates=RELEASE_GATES if gates is None else gates
    require(gates in (RELEASE_GATES,LEGACY_ENGINEERING_GATES),'Unsupported release gate version')
    counts=Counter(r['split'] for r in rows)
    families=defaultdict(set); targets=defaultdict(set); difficulty=defaultdict(Counter)
    ancestry={}; images={}; views={}; bins=set(); errors=[]; unique_images=set()
    for r in rows:
        split=r['split']; family=r['source_plant_family']
        require(split in ('train','validation','test'),'Unknown split')
        require(ancestry.setdefault(family,split)==split,'Family leaks across splits')
        require(images.setdefault(r['rgb_sha256'],split)==split,'Image leaks across splits')
        require(r['rgb_sha256'] not in unique_images,'Duplicate RGB image in release')
        unique_images.add(r['rgb_sha256'])
        signature=r.get('view_signature')
        if signature is not None:
            require(signature not in views,'Duplicate camera/scene condition in release')
            views[signature]=split
        families[split].add(family); targets[split].add(r['target_id']); difficulty[split][r['difficulty']]+=1
        if split=='train' and r['answer']['status']=='localized':
            point=r['answer']['cut_point_uv']; query=r['query_pixel_uv']
            bins.add((int(point[0]//32),int(point[1]//32)))
            errors.append(float(np.linalg.norm(np.asarray(point)-query)))
    failures=[]
    for split in ('train','validation','test'):
        for name,values in [('rows',counts),('families',{s:len(f) for s,f in families.items()}),('targets',{s:len(t) for s,t in targets.items()})]:
            needed=gates['minimum_'+name][split]
            if values.get(split,0)<needed: failures.append(f'{split}_{name}:{values.get(split,0)}<{needed}')
        for level,needed in gates['minimum_difficulty_rows_per_split'].items():
            if difficulty[split][level]<needed: failures.append(f'{split}_{level}:{difficulty[split][level]}<{needed}')
        for kind,status in [('localized','localized'),('occluded','abstain')]:
            if 'minimum_'+kind+'_rows' in gates:
                found=sum(r['split']==split and r['answer']['status']==status for r in rows)
                needed=gates['minimum_'+kind+'_rows'][split]
                if found<needed: failures.append(f'{split}_{kind}:{found}<{needed}')
    if len(bins)<gates['minimum_unique_cut_pixel_bins_train']: failures.append('insufficient_cut_location_spread')
    median=float(np.median(errors)) if errors else None
    if median is None or median<gates['minimum_query_only_error_px']: failures.append('query_copy_shortcut')
    return dict(passed=not failures,failures=failures,rows=dict(counts),families={s:len(v) for s,v in families.items()},
        targets={s:len(v) for s,v in targets.items()},difficulty={s:dict(v) for s,v in difficulty.items()},
        unique_cut_pixel_bins_train=len(bins),query_copy_baseline_median_error_px=median,
        target_source_family_assignments=ancestry,thresholds=gates)


def gather(audit_paths, *, output=None):
    """Validate immutable sources and derive deduplicated labels without writing."""
    output=Path(output).resolve() if output is not None else None
    require(bool(audit_paths),'No audited captures')
    bindings={}; audits=[]; families={}; reports={}; source_plans={}
    for name in audit_paths:
        path=Path(name).resolve(); audit_hash=sha256(path); audit=read_json(path)
        require(audit.get('schema_version')==SCHEMA and audit.get('state')=='complete_engineering_audit_not_approval', 'Incomplete audit')
        require(audit.get('training_dataset_approved') is False,'Source pilot approval changed')
        capture=Path(audit['source_run']); manifest=read_json(capture/'manifest.json')
        if output is not None:
            require(not output.is_relative_to(capture) and not output.is_relative_to(Path(manifest['package'])), 'Output overlaps sources')
        require(manifest.get('state')=='pilot_ready_for_review' and manifest.get('source_assets_unchanged') is True, 'Incomplete capture')
        plan_path=Path(manifest['source_collection_plan_path']); plan=read_json(plan_path)
        require(plan.get('schema_version')=='greenhouse.grounding_collection_plan.v1','Pilot fixed-position frames are not release training data')
        require(sha256(plan_path)==manifest['source_collection_plan_sha256'], 'Changed source schedule')
        source_plans[str(plan_path)]=sha256(plan_path)
        jobs=[j for j in plan['jobs'] if j['job_id']==manifest['collection_job_id']]
        require(len(jobs)==1,'Unknown collection job')
        job=jobs[0]
        # A durable zero-exit ledger is required in addition to successful arrays.
        result=read_json(capture.parent/'result.json')
        require(result.get('returncode')==0 and not result.get('timed_out')
                and result.get('audit_sha256')==audit_hash and result.get('state')=='audited_prototype_pending_visual_review',
                'Capture/audit worker did not complete cleanly')
        if 'exit_receipt_path' in result:
            from .collection_run import checked_exit
            receipt_path=Path(result['exit_receipt_path'])
            require(sha256(receipt_path)==result['exit_receipt_sha256'],'Changed worker exit receipt')
            receipt,_=checked_exit(capture.parent,receipt_path)
            require(receipt['returncode']==result['returncode'] and receipt['timed_out']==result['timed_out'], 'Worker exit/result mismatch')
            bindings[str(receipt_path)]=sha256(receipt_path)
            bindings[str(capture.parent/'launch.json')]=sha256(capture.parent/'launch.json')
        family=job['plant_family']; split=job['split']
        require(plan['family_assignments'][family]==split and families.setdefault(family,split)==split,'Changed family split')
        for p,h in {**audit['bindings_sha256'],str(path):audit_hash,str(capture.parent/'result.json'):sha256(capture.parent/'result.json')}.items():
            require(bindings.setdefault(p,h)==h,'Conflicting source hashes')
        for card,h in audit['cards_sha256'].items():
            require(sha256(safe_file(path.parent,card))==h,'Changed review evidence')
        if family not in reports: reports[family]=audit_manifest(job['source_manifest_path'])
        records=[]
        for record_path in sorted((path.parent/'records').glob('*.json')):
            bindings[str(record_path)]=sha256(record_path)
            records.append(read_json(record_path))
        reviews=active_reviews(records,audit,audit_hash)
        audits.append((path,audit,capture,family,split,manifest.get('lighting'),reviews,audit_hash,manifest.get('render_budget_profile','established')))
    verify_bindings(bindings)
    candidates=[]; exclusions=[]; duplicate=set(); duplicate_views=set()
    for path,audit,capture,family,split,lighting,reviews,audit_hash,profile in audits:
        for checked in audit['samples']:
            require(checked.get('integrity_and_recomputed_annotations_passed') is True
                    and checked['camera'].get('mounted_robot_pov_verified') is True,'Unaudited observation')
            sid=checked['sample_id']; directory=capture/sid; meta=read_json(directory/'sample.json')
            if any(reviews.get((sid,role),{}).get('decision') in ('hold','reject') for role in ('assistant','human')):
                exclusions.append(dict(source_sample=str(directory),reason='explicit_reviewer_hold_or_reject',family=family)); continue
            label=derive_label(directory,meta,reports[family])
            key=meta['files']['inputs/rgb.png']['sha256']
            if not label['eligible']:
                exclusions.append(dict(source_sample=str(directory),reason=label['reason'],family=family)); continue
            # Count unique rendered images, not repeated prompts or repeated exports.
            signature=view_signature(meta,family,lighting)
            if key in duplicate or signature in duplicate_views:
                exclusions.append(dict(source_sample=str(directory),reason='duplicate_rgb_or_camera_scene_condition',family=family)); continue
            duplicate.add(key)
            duplicate_views.add(signature)
            unique_id=family+'_'+key[:20]
            candidates.append(dict(id=unique_id,split=split,source_plant_family=family,target_id=label['target_id'],
                difficulty=label['difficulty'],answer=label['answer'],query_pixel_uv=label['query_pixel_uv'],rgb_sha256=key,
                directory=directory,metadata=meta,label=label,source_audit_sha256=audit_hash,task_contract_sha256=contract_hash(),
                source_sample_sha256=sha256(directory/'sample.json'),capture_profile=profile,view_signature=signature))
    require(bool(candidates),'No eligible synthetic labels; no release written')
    return dict(candidates=candidates,exclusions=exclusions,bindings=bindings,source_plans=source_plans)


def build(audit_paths,output,*,allow_incomplete=False,visual_reviews=None):
    output=Path(output).resolve()
    require(not output.exists(),'Choose a new release output; never overwrite')
    gathered=gather(audit_paths,output=output)
    candidates,exclusions,bindings,source_plans=(gathered[k] for k in ('candidates','exclusions','bindings','source_plans'))
    gates=check_release_rows(candidates)
    if not allow_incomplete:
        require(gates['passed'],'Release coverage gates failed: '+', '.join(gates['failures']))
    from .training_release_review import verify_reviews
    review_evidence=verify_reviews(visual_reviews or [],candidates,require_complete=not allow_incomplete)
    complete=bool(gates['passed'] and review_evidence['passed'])
    output.mkdir(parents=True)
    for folder in ('images','depth','labels','splits'): (output/folder).mkdir()
    files={}; index=[]; chats=defaultdict(list)
    for r in candidates:
        sid=r['id']; directory=r.pop('directory'); meta=r.pop('metadata'); label=r.pop('label')
        paths={'rgb':f'images/{sid}.png','depth':f'depth/{sid}.npy','validity':f'depth/{sid}_valid.png',
               'label':f'labels/{sid}.json','target_mask':f'labels/{sid}_target.png'}
        for key,source in [('rgb','inputs/rgb.png'),('depth','inputs/depth_m.npy'),('validity','inputs/depth_valid.png'),
                           ('target_mask','supervision/target_visible.png')]:
            shutil.copyfile(safe_file(directory,source),output/paths[key])
            require(sha256(output/paths[key])==meta['files'][source]['sha256'],'Copied artifact hash mismatch')
        write_json(output/paths['label'],dict(**label,calibration=meta['calibration'],robot_snapshot=meta['robot_snapshot'],
            accepted_interval_uv=[p['pixel_xy'] for p in meta['supervision']['projected_interval']],
            accepted_interval_world_m=meta['supervision']['interval_world_m'],source_sample_sha256=sha256(directory/'sample.json')))
        r['files']=paths
        index.append(r)
        chats[r['split']].append(chat_row(sid,paths['rgb'],label))
        for p in paths.values(): files[p]=sha256(output/p)
    for split in ('train','validation','test'):
        path=f'splits/{split}.jsonl'
        with (output/path).open('x',encoding='utf-8') as stream:
            for row in chats[split]: stream.write(json.dumps(row,allow_nan=False)+'\n')
        files[path]=sha256(output/path)
    with (output/'index.jsonl').open('x',encoding='utf-8') as stream:
        for r in index: stream.write(json.dumps(r,allow_nan=False)+'\n')
    write_json(output/'contract.json',dict(contract=CONTRACT,system_prompt=SYSTEM_PROMPT,contract_sha256=contract_hash()))
    write_json(output/'exclusions.json',exclusions)
    write_json(output/'visual_review.json',review_evidence)
    files.update({p:sha256(output/p) for p in ('index.jsonl','contract.json','exclusions.json','visual_review.json')})
    result=dict(schema_version=SCHEMA_RELEASE,state='complete_synthetic_grounding_release' if complete else 'incomplete_engineering_export_do_not_claim_release',
        created_utc=datetime.now(timezone.utc).isoformat(),packaging='portable_self_contained_images_chats_native_depth_labels',
        contract_sha256=contract_hash(),acceptance=gates,files_sha256=files,source_plans_sha256=source_plans,
        source_bindings_sha256=bindings,excluded_counts=dict(Counter(x['reason'] for x in exclusions)),
        synthetic_training_task_only=True,physical_execution_approved=False,human_approved_count=0,
        split_scope='target_source_families_only_shared_greenhouse_backdrop_context',
        implementation_sha256={p:sha256(Path(__file__).with_name(p)) for p in
            ('training_export.py','training_contract.py','query_visibility.py','training_release_review.py')})
    verify_bindings(bindings)
    write_json(output/'manifest.json',result)
    validate(output,allow_incomplete=allow_incomplete)
    return result


def validate(root,*,allow_incomplete=False):
    """Offline loader check uses only portable artifacts, never original source paths."""
    root=Path(root); manifest=read_json(root/'manifest.json')
    require(manifest.get('schema_version')==SCHEMA_RELEASE,'Wrong release schema')
    require(manifest.get('state')=='complete_synthetic_grounding_release' or
            (allow_incomplete and manifest.get('state')=='incomplete_engineering_export_do_not_claim_release'), 'Incomplete release')
    require(manifest['contract_sha256']==contract_hash(),'Unsupported contract')
    for path,h in manifest['files_sha256'].items(): require(sha256(safe_file(root,path))==h,'Changed release artifact: '+path)
    contract=read_json(safe_file(root,'contract.json'))
    require(contract['contract']==CONTRACT and contract['system_prompt']==SYSTEM_PROMPT,'Changed task/prompt')
    rows=list(read_jsonl(safe_file(root,'index.jsonl')))
    require(len({r['id'] for r in rows})==len(rows),'Duplicate sample ids')
    indexed={r['id']:r for r in rows}; seen=set()
    for split in ('train','validation','test'):
        for chat in read_jsonl(safe_file(root,f'splits/{split}.jsonl')):
            require(chat['id'] in indexed and chat['id'] not in seen,'Unknown or duplicated chat row')
            seen.add(chat['id']); r=indexed[chat['id']]
            require(r['split']==split,'Chat split mismatch')
            label=read_json(safe_file(root,r['files']['label']))
            require(chat==chat_row(r['id'],r['files']['rgb'],label),'Prompt, answer or image mismatch')
            require(r['answer']==label['answer'] and r['query_pixel_uv']==label['query_pixel_uv'],'Index-label mismatch')
            rgb=Image.open(safe_file(root,r['files']['rgb']))
            require(rgb.mode=='RGB' and rgb.size==(848,408),'Bad model image')
            depth=np.load(safe_file(root,r['files']['depth']),allow_pickle=False)
            valid=np.asarray(Image.open(safe_file(root,r['files']['validity'])))
            require(depth.shape==valid.shape==(408,848) and depth.dtype==np.float32,'Bad native depth')
            near,far=label['calibration']['clipping_range_m']
            require(np.array_equal(valid,(np.isfinite(depth)&(depth>0)&(depth>=near)&(depth<=far)).astype(np.uint8)*255),'Depth validity changed')
            mask=np.asarray(Image.open(safe_file(root,r['files']['target_mask'])))
            x,y=np.floor(label['query_pixel_uv']).astype(int)
            require(mask.shape==(408,848) and mask[y,x]==255,'Query is not on visible target')
            usability=QueryVisibility(np.asarray(rgb),mask==255).inspect(label['query_pixel_uv'])
            require(usability['passed'] and label.get('query_usability')==usability,'Query usability failed or evidence changed')
            require(sha256(safe_file(root,r['files']['rgb']))==r['rgb_sha256'],'RGB identity mismatch')
    require(seen==set(indexed),'Index rows missing from training chats')
    gates=check_release_rows(rows,gates=manifest['acceptance']['thresholds'])
    require(gates==manifest['acceptance'],'Release acceptance summary mismatch')
    if not allow_incomplete:
        require(gates['passed'] and gates['thresholds']==RELEASE_GATES,'Coverage/balance gates failed')
        from .training_release_review import check_evidence
        require('visual_review.json' in manifest['files_sha256'],'Missing visual QA evidence')
        evidence=read_json(safe_file(root,'visual_review.json'))
        recomputed=check_evidence(evidence,rows,require_complete=True)
        require(all(evidence.get(k)==v for k,v in recomputed.items()),'Visual QA summary mismatch')
    return dict(state=manifest['state'],rows=len(rows),portable_loader_verified=True,acceptance=gates)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','validate'])
    p.add_argument('--audit',type=Path,action='append')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--visual-review',type=Path,action='append')
    a=p.parse_args(argv)
    r=build(a.audit,a.output,allow_incomplete=a.allow_incomplete,visual_reviews=a.visual_review) if a.command=='build' else validate(a.output,allow_incomplete=a.allow_incomplete)
    print(json.dumps({k:v for k,v in r.items() if k in ('state','acceptance','rows','portable_loader_verified')},indent=2),flush=True)


if __name__=='__main__': main()
