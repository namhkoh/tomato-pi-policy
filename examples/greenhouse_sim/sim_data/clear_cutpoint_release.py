"""Build a NEW visible-only derivative of a validated immutable task-v3 release.

Drafts are inspectable, never accepted by the training loader. Existing source
holds/splits are inherited. New review decisions live in a separate file.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .clear_cutpoint_contract import PROFILE, SCHEMA, POLICY, GATES, crop_box, crop_image, screen, select_views
from .dataset_review import require, read_json, write_json, safe_file
from .depth_preview import sha256
from .training_export import read_jsonl, chat_row, evidence_matches


def coverage(rows):
    result={}; failures=[]
    for split in ('train','validation','test'):
        part=[r for r in rows if r['split']==split]
        result[split]=dict(rows=len(part),targets=len({r['target_id'] for r in part}),
                           families=len({r['source_plant_family'] for r in part}))
        for kind in ('rows','targets','families'):
            if result[split][kind] < GATES['minimum_'+kind][split]: failures.append(split+'_'+kind)
    return dict(passed=not failures,failures=failures,counts=result)


def review_ids(rows):
    # Every held-out image; two deterministic training views per target.
    groups=defaultdict(list); ids=[]
    for r in sorted(rows,key=lambda r:r['id']):
        if r['split'] != 'train': ids.append(r['id'])
        else: groups[r['target_id']].append(r['id'])
    for group in groups.values(): ids.extend(group[:2])
    return sorted(ids)


def check_reviews(rows, reviews):
    by_id={r['id']:r for r in rows}; seen=set(); accepted=set(); holds=[]
    for rec in reviews:
        require(rec['id'] in by_id and rec['id'] not in seen,'Unknown/duplicate review')
        seen.add(rec['id']); row=by_id[rec['id']]
        require(rec['rgb_sha256']==row['rgb_sha256'],'Review image changed')
        require(rec['reviewer_type'] in ('human','assistant') and bool(rec['reviewer'].strip()),'Review attribution required')
        require(rec['decision'] in ('accept','hold','reject') and bool(rec['reason'].strip()),'Explicit reason/decision required')
        if rec['decision'] in ('hold','reject'): holds.append(rec['id'])
        elif rec['reviewer_type']=='human' or row['split']=='train': accepted.add(rec['id'])
    missing=sorted(set(review_ids(rows))-accepted)
    return dict(passed=not missing and not holds,missing=missing,holds=holds,
                heldout_human_review_required=True,reviewed=len(seen))


def scan(source, progress=None):
    candidates=[]; exclusions=[]
    for i,r in enumerate(read_jsonl(source/'index.jsonl')):
        if r['difficulty'] != 'easy':
            exclusions.append(dict(id=r['id'],reason='not_easy')); continue
        label=read_json(safe_file(source,r['files']['label']))
        with Image.open(safe_file(source,r['files']['rgb'])) as rgb, Image.open(safe_file(source,r['files']['target_mask'])) as mask:
            evidence=screen(label,np.asarray(rgb),np.asarray(mask)==255)
        if not evidence['passed']:
            exclusions.append(dict(id=r['id'],reason='legibility',evidence=evidence)); continue
        camera=np.asarray(label['calibration']['camera_to_world_usd_row_vectors'],float)
        feature=[*camera[3,:3],*camera[2,:3],*(np.asarray(r['query_pixel_uv'])/[848,408])]
        candidates.append(dict(r,legibility=evidence,selection_features=feature))
        if progress and i%200==0: progress(dict(stage='clear_screen',examined=i,eligible=len(candidates)))
    chosen=select_views(candidates); ids={r['id'] for r in chosen}
    exclusions += [dict(id=r['id'],reason='view_cap') for r in candidates if r['id'] not in ids]
    return chosen,exclusions


def chat(r,label):
    row=chat_row(r['id'],r['files']['rgb'],label)
    row['clear_cutpoint']=dict(profile=PROFILE,query_pixel_uv=r['query_pixel_uv'],
                               crop_box_xyxy=crop_box(r['query_pixel_uv']),crop=r['files']['crop'])
    return row


def build(source, output, *, reviews=None, progress=None, audited_source=False):
    from .training_export import validate as validate_source
    source,output=Path(source).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(source) and not source.is_relative_to(output),'Choose a new disjoint output')
    digest=sha256(source/'manifest.json')
    require(type(audited_source) is bool,'Explicit audited-source flag required')
    receipt=validate_source(source,progress=progress,**({'allow_incomplete':True} if audited_source else {}))
    source_manifest=read_json(source/'manifest.json')
    if audited_source:
        require(source_manifest['state']=='incomplete_engineering_export_do_not_claim_release','Audited source is an engineering pool, never masquerades as an approved release')
        plans=source_manifest['source_plans_sha256'];require(plans,'Native capture plans required')
        for p,h in plans.items():
            require(sha256(p)==h and read_json(p)['configuration'].get('clear_capture')=='robot_head_close_diffuse_v1',
                    'Fresh clearer native captures required for this source mode')
    rows,exclusions=scan(source,progress)
    require(rows,'No sufficiently clear candidates; collect better observations, do not relax silently')
    original=read_json(source/'manifest.json')
    records=read_json(reviews) if reviews else []
    qa=check_reviews(rows,records)
    output.mkdir(parents=True)
    for folder in ('images','depth','labels','crops','splits'): (output/folder).mkdir()
    files={}; chats=defaultdict(list)
    for r in rows:
        for path in r['files'].values():
            shutil.copyfile(safe_file(source,path),output/path)
            require(sha256(output/path)==original['files_sha256'][path],'Source artifact changed during copy')
            files[path]=sha256(output/path)
        crop=f"crops/{r['id']}.png"
        with Image.open(output/r['files']['rgb']) as rgb: crop_image(rgb,r['query_pixel_uv']).save(output/crop)
        r['files']=dict(r['files'],crop=crop);files[crop]=sha256(output/crop)
        label=read_json(output/r['files']['label']);chats[r['split']].append(chat(r,label))
    for split in ('train','validation','test'):
        with (output/f'splits/{split}.jsonl').open('x',encoding='utf-8') as f:
            for row in chats[split]: f.write(json.dumps(row,allow_nan=False)+'\n')
        files[f'splits/{split}.jsonl']=sha256(output/f'splits/{split}.jsonl')
    with (output/'index.jsonl').open('x',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,allow_nan=False)+'\n')
    shutil.copyfile(source/'manifest.json',output/'source_manifest.json')
    files['source_manifest.json']=sha256(output/'source_manifest.json')
    for name,value in [('source_validation.json',receipt),
                       ('exclusions.json',exclusions),('reviews.json',records),
                       ('contract.json',dict(profile=PROFILE,policy=POLICY,gates=GATES,
                            crop_coordinate_frame='original_full_image',native_depth='copied_byte_for_byte',
                            human_review_required_on_heldout=True,physical_execution_approved=False))]:
        write_json(output/name,value);files[name]=sha256(output/name)
    files['index.jsonl']=sha256(output/'index.jsonl')
    acceptance=coverage(rows)
    manifest=dict(schema_version=SCHEMA,release_profile=PROFILE,policy=POLICY,gates=GATES,
                  state='complete_clear_cutpoint_release' if acceptance['passed'] and qa['passed'] else 'draft_clear_cutpoint_not_for_training',
                  acceptance=acceptance,review=qa,source_manifest_sha256=digest,
                  source_manifest_copy_sha256=files['source_manifest.json'],files_sha256=files,
                  source_release=str(source),audited_source=audited_source,synthetic_perception_only=True,physical_execution_approved=False)
    require(sha256(source/'manifest.json')==digest,'Source manifest changed')
    write_json(output/'manifest.json',manifest)
    validate(output,allow_draft=True)
    return manifest


def validate(root, *, allow_draft=False, progress=None):
    root=Path(root); m=read_json(root/'manifest.json')
    require(m.get('schema_version')==SCHEMA and m.get('release_profile')==PROFILE,'Wrong clear-cutpoint schema')
    require(m['policy']==POLICY and m['gates']==GATES,'Changed experiment policy')
    require(m['state']=='complete_clear_cutpoint_release' or
            allow_draft and m['state']=='draft_clear_cutpoint_not_for_training','Clear-cutpoint visual review/coverage incomplete')
    for p,h in m['files_sha256'].items(): require(sha256(safe_file(root,p))==h,'Changed clear artifact: '+p)
    source=read_json(safe_file(root,'source_manifest.json'))
    require(sha256(root/'source_manifest.json')==m['source_manifest_copy_sha256']==m['source_manifest_sha256'],'Source receipt changed')
    if m.get('audited_source'):
        receipt=read_json(safe_file(root,'source_validation.json'))
        require(source['state']=='incomplete_engineering_export_do_not_claim_release' and
                receipt.get('portable_loader_verified') is True and receipt['state']==source['state'] and
                bool(source.get('source_plans_sha256')),'Unvalidated native source pool')
    else:
        require(source['state'] in ('complete_synthetic_grounding_release','complete_visible_occluded_baseline_release'),'Unapproved parent release')
    rows=list(read_jsonl(safe_file(root,'index.jsonl'))); ids=set();images=set();views=set();families={};counts=Counter()
    expected=defaultdict(list)
    for r in rows:
        require(r['id'] not in ids and r['rgb_sha256'] not in images and r['view_signature'] not in views,'Duplicate row/image/view')
        ids.add(r['id']);images.add(r['rgb_sha256']);views.add(r['view_signature'])
        require(r['split'] in ('train','validation','test'),'Unknown split')
        require(source['acceptance']['target_source_family_assignments'][r['source_plant_family']]==r['split'],'Changed frozen family split')
        families[r['source_plant_family']]=r['split'];counts[r['target_id']]+=1
        require(r['difficulty']=='easy','Non-easy record')
        for key,p in r['files'].items():
            require(p in m['files_sha256'],'Unbound row file')
            if key!='crop': require(m['files_sha256'][p]==source['files_sha256'].get(p),'Changed original RGB/depth/label/mask')
        require(m['files_sha256'][r['files']['rgb']]==r['rgb_sha256'],'RGB binding mismatch')
        label=read_json(safe_file(root,r['files']['label']))
        require(label['answer']==r['answer'] and label['query_pixel_uv']==r['query_pixel_uv'] and
                label['target_id']==r['target_id'] and label['source_plant_family']==r['source_plant_family'],'Changed source identity/answer/query')
        with Image.open(safe_file(root,r['files']['rgb'])) as rgb, Image.open(safe_file(root,r['files']['target_mask'])) as mask:
            ev=screen(label,np.asarray(rgb),np.asarray(mask)==255)
            require(ev['passed'] and evidence_matches(ev,r['legibility']),'Legibility changed')
            with Image.open(safe_file(root,r['files']['crop'])) as crop:
                require(np.array_equal(np.asarray(crop),np.asarray(crop_image(rgb,r['query_pixel_uv']))),'Crop not query-derived')
        expected[r['split']].append(chat(r,label))
    require(rows and max(counts.values())<=12,'Empty release or view cap exceeded')
    for split in ('train','validation','test'):
        require(list(read_jsonl(safe_file(root,f'splits/{split}.jsonl')))==expected[split],'Changed training chat')
    acceptance=coverage(rows);qa=check_reviews(rows,read_json(safe_file(root,'reviews.json')))
    require(acceptance==m['acceptance'] and qa==m['review'],'Changed coverage/review summary')
    complete=acceptance['passed'] and qa['passed']
    require((m['state']=='complete_clear_cutpoint_release')==complete,'Incorrect release state')
    require(allow_draft or complete,'Not training-ready')
    return dict(state=m['state'],release_profile=PROFILE,rows=len(rows),acceptance=acceptance,review=qa)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','validate','finalize'))
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reviews',type=Path);p.add_argument('--allow-draft',action='store_true')
    p.add_argument('--audited-source',action='store_true',help='Fresh clear-capture engineering pool, validated separately; final clear gates still mandatory')
    a=p.parse_args(argv)
    if a.command=='finalize':
        require(a.source is not None and a.reviews is not None,'Draft and explicit reviews required')
        result=finalize(a.source,a.output,a.reviews)
    elif a.command=='build':
        require(a.source is not None,'Source release required')
        result=build(a.source,a.output,reviews=a.reviews,audited_source=a.audited_source,progress=lambda x:print(json.dumps(x),flush=True))
    else: result=validate(a.output,allow_draft=a.allow_draft)
    print(json.dumps({k:result[k] for k in ('state','acceptance','review')},indent=2))


def finalize(draft,output,reviews):
    """Review-only new edition; prior holds cannot be overwritten by new accepts."""
    draft,output=Path(draft).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(draft) and not draft.is_relative_to(output),'New disjoint output required')
    validate(draft,allow_draft=True)
    old=read_json(draft/'reviews.json');incoming=read_json(reviews)
    records={r['id']:r for r in old}
    require(len({r['id'] for r in incoming})==len(incoming),'Duplicate incoming reviews')
    for rec in incoming:
        previous=records.get(rec['id'])
        require(not previous or previous['decision']=='accept' or rec==previous,'Existing hold/reject cannot be overridden')
        records[rec['id']]=rec
    rows=list(read_jsonl(draft/'index.jsonl'));qa=check_reviews(rows,list(records.values()))
    require(coverage(rows)['passed'] and qa['passed'],'Coverage or required reviews incomplete; no final release written')
    shutil.copytree(draft,output)
    # Own new copy only; never update the draft or original review file.
    (output/'reviews.json').write_text(json.dumps(list(records.values()),indent=2),encoding='utf-8')
    m=read_json(output/'manifest.json');m['files_sha256']['reviews.json']=sha256(output/'reviews.json')
    m.update(state='complete_clear_cutpoint_release',review=qa)
    m['finalized_from_manifest_sha256']=sha256(draft/'manifest.json')
    (output/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    validate(output);return m


if __name__=='__main__': main()
