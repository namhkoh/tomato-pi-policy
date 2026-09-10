"""Explicit, hash-bound stratified visual QA for a synthetic grounding release.

Preparing cards does not review them. A reviewer must actually inspect the saved
evidence and record a decision. Assistant inspection is never human confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import textwrap
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .capture_contract import fingerprint
from .dataset_review import read_json, require, review_canvas, safe_file, write_json
from .depth_preview import sha256, colour_depth

SCHEMA='greenhouse.grounding_stratified_visual_QA.v2'
CHECKLIST=['full_scene_rgb','visibility_gated_cut_overlay','native_identity_mask','native_depth',
           'target_query_and_task_answer','query_rgb_neighborhood_and_usability']
POLICY={'minimum_inspected_per_nonempty_family_difficulty':2,
        'minimum_inspected_per_capture_profile':1,
        'card_version':'visibility_gated_cut_and_separate_query_crops.v2',
        'claim':'representative_visual_QA_not_statistical_accuracy_or_human_confirmation'}


def identity(row):
    keys=('id','split','source_plant_family','target_id','difficulty','rgb_sha256',
          'source_audit_sha256','source_sample_sha256','capture_profile','query_pixel_uv','answer','view_signature','task_contract_sha256')
    return {k:row[k] for k in keys}


def selected_rows(rows):
    groups=defaultdict(list)
    for row in rows: groups[(row['source_plant_family'],row['difficulty'])].append(row)
    selected={}
    for key,group in sorted(groups.items()):
        ordered=sorted(group,key=lambda r:fingerprint(r['id']))
        # Prefer a second anatomical target when one exists, not adjacent frames.
        chosen=ordered[:1]
        if len(ordered)>1:
            chosen.append(next((r for r in ordered[1:] if r['target_id']!=chosen[0]['target_id']),ordered[1]))
        selected.update({r['id']:r for r in chosen})
    for profile in sorted({r['capture_profile'] for r in rows}):
        if not any(r['capture_profile']==profile for r in selected.values()):
            row=min((r for r in rows if r['capture_profile']==profile),key=lambda r:fingerprint(r['id']))
            selected[row['id']]=row
    return sorted(selected.values(),key=lambda r:(r['split'],r['source_plant_family'],r['difficulty'],r['id']))


def task_review_canvas(row, buffers):
    """Diagnostic card only: no overlays/crops enter the model observation."""
    rgb, depth, valid, target = buffers
    canvas=Image.new('RGB',(1152,1230),(22,26,33))
    canvas.paste(review_canvas(row['metadata'],buffers),(0,0))
    draw=ImageDraw.Draw(canvas); query=row['query_pixel_uv']
    draw.ellipse((query[0]-5,query[1]+23,query[0]+5,query[1]+33),outline='cyan',width=2)
    nominal=row['metadata']['supervision']['visibility_evidence']['nominal']
    observed=nominal.get('observed_component') or {}
    occluder=' / '.join(str(observed.get(k,'unknown')) for k in ('organ_type','component_id'))
    status=('VISIBLE CUT: localization label' if row['answer']['status']=='localized' else
            'ABSTAIN: hidden cut; no visible cut point. Foreground: '+occluder)
    excluded=row['answer']['status']=='excluded'
    if excluded: status='EXCLUDED: not a training example. '+row['answer']['reason']
    draw.text((8,850),status,fill='yellow')
    lines=[row['id']+' | '+row['split']+' / '+row['difficulty']+' / '+row['target_id'],
           f"{'OLD rejected' if excluded else 'INPUT'} query {query}; cyan circle is review-only. Training RGB is untouched.",
           'ANSWER: '+str(row['answer'])]
    draw.multiline_text((8,870),'\n'.join('\n'.join(textwrap.wrap(s,165)) for s in lines),fill='white',spacing=3)
    x,y=np.floor(query).astype(int)
    left,top=max(0,min(x-28,848-56)),max(0,min(y-28,408-56))
    box=(left,top,left+56,top+56)
    masked=rgb.copy(); masked[target]=(0,255,80)
    tiles=(Image.fromarray(rgb),Image.fromarray(masked),Image.fromarray(colour_depth(depth,valid,.04,2.)))
    titles=('QUERY: original RGB 4x','QUERY: exact native target 4x','QUERY: native camera-Z 4x')
    for i,(tile,title) in enumerate(zip(tiles,titles)):
        draw.text((i*384+8,948),title,fill='cyan')
        canvas.paste(tile.crop(box).resize((224,224),Image.Resampling.NEAREST),(i*384+8,968))
        cx=i*384+8+(query[0]-left)*4; cy=968+(query[1]-top)*4
        draw.ellipse((cx-6,cy-6,cx+6,cy+6),outline='cyan')
    draw.text((8,1206),'Engineering/visual QA only; not human confirmation, agronomic approval or physical cut safety.',fill='white')
    return canvas


def prepare(audits,output):
    from .training_export import gather
    output=Path(output).resolve()
    require(not output.exists(),'Choose a new visual QA directory')
    gathered=gather(audits,output=output)
    rows=selected_rows(gathered['candidates'])
    output.mkdir(parents=True)
    entries=[]
    for row in rows:
        directory,meta=row['directory'],row['metadata']
        rgb=np.asarray(Image.open(directory/'inputs/rgb.png'))
        depth=np.load(directory/'inputs/depth_m.npy',allow_pickle=False)
        valid=np.asarray(Image.open(directory/'inputs/depth_valid.png'))==255
        target=np.asarray(Image.open(directory/'supervision/target_visible.png'))==255
        card=row['id']+'.png'
        task_review_canvas(row,(rgb,depth,valid,target)).save(output/card)
        entries.append(dict(**identity(row),card=card,card_sha256=sha256(output/card)))
    result=dict(schema_version=SCHEMA,state='pending_actual_visual_inspection',policy=POLICY,
                created_utc=datetime.now(timezone.utc).isoformat(),entries=entries,
                source_audits_sha256={str(Path(p).resolve()):sha256(p) for p in audits},
                training_release_approved=False,human_review_performed=False)
    write_json(output/'bundle.json',result)
    return result


def record(bundle_path,ids,notes,*,decision='accept',role='assistant',reviewer='Codex',inspected=False,
           human_followup=False, suggestion_context=None):
    bundle_path=Path(bundle_path).resolve(); bundle=read_json(bundle_path)
    require(bundle.get('schema_version')==SCHEMA and bundle['policy']==POLICY,'Invalid visual bundle')
    require(inspected and isinstance(notes,str) and len(notes.strip())>=20,'Actual inspection and substantive notes required')
    require(role in ('assistant','human') and decision in ('accept','hold','reject'),'Invalid visual decision')
    require(bool(reviewer.strip()) and ids and len(ids)==len(set(ids)),'Reviewer and unique IDs required')
    entries={e['id']:e for e in bundle['entries']}
    require(set(ids)<=set(entries),'Unknown reviewed ID')
    require(not human_followup or role == 'human', 'Only a human may add a final follow-up')
    require(suggestion_context is None or (role == 'human' and len(ids) == 1), 'Suggestion context requires a single human review')
    folder=bundle_path.parent/('human_decisions' if human_followup else 'decisions'); folder.mkdir(exist_ok=True)
    paths=[]
    for sid in ids:
        e=entries[sid]
        require(sha256(safe_file(bundle_path.parent,e['card']))==e['card_sha256'],'Changed visual card')
        path=folder/(sid+'.json')
        require(not path.exists(),'Never overwrite or silently supersede a visual decision')
        prior = None
        if human_followup:
            prior_path = bundle_path.parent/'decisions'/(sid+'.json')
            prior = read_json(prior_path)
            require(prior['reviewer_role'] == 'assistant' and prior['entry'] == e
                    and prior['bundle_sha256'] == sha256(bundle_path), 'Follow-up needs a bound assistant review')
            require(decision != 'accept' or prior['decision'] == 'accept', 'Acceptance cannot clear an earlier task hold')
        row=dict(schema_version=SCHEMA,review_id=uuid.uuid4().hex,bundle_sha256=sha256(bundle_path),
                 created_utc=datetime.now(timezone.utc).isoformat(),entry=e,reviewer_role=role,reviewer=reviewer,
                 decision=decision,notes=notes,inspected=CHECKLIST,
                 human_confirmation=role=='human',physical_execution_approved=False,
                 identity_authentication='local_self_declared_role_not_authenticated')
        if prior is not None:
            row['prior_assistant_review_sha256'] = sha256(prior_path)
        if suggestion_context is not None:
            row['suggestion_context'] = suggestion_context
        write_json(path,row); paths.append(path)
    return paths


def check_evidence(evidence,rows,*,require_complete=True):
    require(evidence.get('schema_version')==SCHEMA and evidence.get('policy')==POLICY,'Invalid visual QA policy')
    indexed={r['id']:r for r in rows}; counts=Counter(); profiles=Counter(); seen=set()
    for record in evidence['records']:
        e=record['entry']; sid=e['id']
        require(sid in indexed,'Visual review is not part of this release')
        require(identity(e)==identity(indexed[sid]),'Visual review does not bind this exact task example')
        require(record['reviewer_role'] in ('assistant','human') and record['decision'] in ('accept','hold','reject'), 'Invalid reviewer/decision')
        require(record['human_confirmation'] is (record['reviewer_role']=='human'),'Assistant cannot claim human confirmation')
        require(record['physical_execution_approved'] is False and record['inspected']==CHECKLIST,'Invalid inspection scope')
        require(len(record['notes'].strip())>=20 and bool(record['reviewer'].strip()),'Missing inspection notes')
        require(record['decision']=='accept','Explicit visual hold/reject cannot enter release')
        if sid not in seen:
            counts[(e['source_plant_family'],e['difficulty'])]+=1
            profiles[e['capture_profile']]+=1
        seen.add(sid)
    available=Counter((r['source_plant_family'],r['difficulty']) for r in rows)
    failures=[]
    for key,n in sorted(available.items()):
        needed=min(n,POLICY['minimum_inspected_per_nonempty_family_difficulty'])
        if counts[key]<needed: failures.append(f'{key[0]}/{key[1]}:{counts[key]}<{needed}')
    for profile in sorted({r['capture_profile'] for r in rows}):
        if not profiles[profile]: failures.append('unreviewed_capture_profile:'+profile)
    result=dict(passed=not failures,failures=failures,inspected_unique_samples=len(seen),
                family_difficulty_counts={f'{k[0]}/{k[1]}':v for k,v in sorted(counts.items())},
                capture_profile_counts=dict(sorted(profiles.items())))
    if require_complete: require(result['passed'],'Stratified visual QA incomplete: '+', '.join(failures))
    return result


def verify_reviews(bundles,rows,*,require_complete=True):
    records=[]; sources={}; seen_reviews=set()
    for path in bundles:
        path=Path(path).resolve(); bundle=read_json(path); h=sha256(path)
        require(bundle['schema_version']==SCHEMA and bundle['policy']==POLICY,'Invalid visual review bundle')
        entries={e['id']:e for e in bundle['entries']}
        sources[str(path)]=h
        decision_paths = sorted((path.parent/'decisions').glob('*.json'))
        decision_paths += sorted((path.parent/'human_decisions').glob('*.json'))
        for p in decision_paths:
            row=read_json(p); sid=row['entry']['id']
            require(row['schema_version']==SCHEMA and row['bundle_sha256']==h and row['entry']==entries.get(sid),'Changed or stale visual decision')
            require(p.name == sid+'.json', 'Misnamed visual review')
            if p.parent.name == 'human_decisions':
                prior_path = path.parent/'decisions'/p.name
                prior = read_json(prior_path)
                require(row['reviewer_role'] == 'human' and prior['reviewer_role'] == 'assistant'
                        and row.get('prior_assistant_review_sha256') == sha256(prior_path), 'Stale human follow-up')
            require(row['review_id'] not in seen_reviews,'Duplicate visual review record')
            seen_reviews.add(row['review_id'])
            e=row['entry']; card=safe_file(path.parent,e['card'])
            require(sha256(card)==e['card_sha256'],'Changed inspected card')
            sources[str(p)]=sha256(p); sources[str(card)]=e['card_sha256']; records.append(row)
    evidence=dict(schema_version=SCHEMA,policy=POLICY,records=records,source_evidence_sha256=sources)
    evidence.update(check_evidence(evidence,rows,require_complete=require_complete))
    return evidence


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--audit',type=Path,action='append',required=True); prep.add_argument('--output',type=Path,required=True)
    rec=sub.add_parser('record'); rec.add_argument('--bundle',type=Path,required=True); rec.add_argument('--id',action='append',required=True)
    rec.add_argument('--notes',required=True); rec.add_argument('--decision',choices=['accept','hold','reject'],default='accept')
    rec.add_argument('--role',choices=['assistant','human'],default='assistant'); rec.add_argument('--reviewer',default='Codex'); rec.add_argument('--inspected',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':
        result=prepare(a.audit,a.output); print('VISUAL_QA_PENDING',len(result['entries']),flush=True)
    else:
        paths=record(a.bundle,a.id,a.notes,decision=a.decision,role=a.role,reviewer=a.reviewer,inspected=a.inspected)
        print('VISUAL_DECISIONS_RECORDED',len(paths),flush=True)


if __name__=='__main__': main()
