"""Create-only standalone materialization of the exact diverse base plus accepted additions.
No annotation/crop regeneration, new acceptance, native action, or training launch.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,re,shutil
import numpy as np
from PIL import Image
from . import native848_diverse_extension_v1 as api
SCHEMA='greenhouse.native848_diverse_standalone.v1'
EXTENSION_SHA='2db369699dab174f4cc3deea99ab2f80f6158e94bd88ed9deefbd16a47171a87'


def local_path(root,relative):
    relative=Path(relative)
    api.require(not relative.is_absolute() and '..' not in relative.parts and ':' not in str(relative), 'Relative safe asset path required')
    path=(Path(root)/relative).resolve()
    api.require(path.is_relative_to(Path(root).resolve()) and path.is_file(),'Asset escapes root or is missing')
    return path


def destination(row,key,suffix):
    sid=row['id'];split=row['split']
    api.require(isinstance(sid,str) and re.fullmatch(r'[A-Za-z0-9_-]+',sid) is not None,'Unsafe image ID')
    api.require(split in ('train','validation','test') and re.fullmatch(r'[A-Za-z0-9_]+',key) is not None,'Unsafe split/asset key')
    api.require(re.fullmatch(r'\.[A-Za-z0-9]+',suffix) is not None,'Unsafe extension')
    if key=='rgb':return f'{split}/images/{sid}{suffix}'
    if key=='label':return f'{split}/annotations/{sid}{suffix}'
    if key=='target_mask':return f'{split}/annotations/{sid}_target{suffix}'
    if key=='crop':return f'{split}/crops/{sid}{suffix}'
    if key in ('depth','validity','buffers'):return f'{split}/depth/{sid}_{key}{suffix}'
    return f'evidence/{key}/{sid}{suffix}'


def immediate_pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            yield value
        else:
            for v in value.values():yield from immediate_pins(v)
    elif isinstance(value,list):
        for v in value:yield from immediate_pins(v)


def inspect(result_path,result_sha256):
    pins=api.Pins();pins.add(dict(path=str(Path(__file__).resolve()),sha256=api.digest(__file__)))
    pins.add(dict(path=str(Path(api.__file__).resolve()),sha256=EXTENSION_SHA))
    result_path=pins.add(dict(path=str(Path(result_path).resolve()),sha256=result_sha256));root=result_path.parent
    api.require(root.parent==api.OUTPUT_ROOT,'Checkpoint source must be a direct package child')
    result=api.read(result_path)
    api.require(result['schema']==api.SCHEMA and result['state']=='diverse_base_plus_individually_reviewed_extension_complete'
        and result['base_result']['sha256']==api.BASE_RESULT_SHA and result['base_rows_removed']==0
        and result['dominant_target_images']==0,'Only a complete accepted diverse extension may materialize')
    for name,pin in result['output_bindings'].items():
        api.require(pins.add(pin)==root/name,'Extension output binding escapes source package')
    rows=api.rows_at(root/'index.jsonl');request=api.read(root/'request.json');base,frozen,_,holds=api.load_base(request,pins)
    base_map={r['id']:r for r in base};base_rows=[r for r in rows if r['registry_origin']=='user_selected_diverse_base']
    api.require({r['id']:r for r in base_rows}==base_map,'Exact original base records changed')
    api.require(dict(Counter(r['split'] for r in rows))==result['counts'] and len({r['id'] for r in rows})==len(rows), 'Source counts/IDs differ')
    global_aliases=api.read(root/'global_exclusion_aliases.json')
    holds.update((r['kind'],r['value']) for r in global_aliases['visual_hold_aliases'])
    base_manifest=api.read(api.BASE/'manifest.json');operations=[];seen=set();destinations=set()
    for row in rows:
        api.require(row['source_target']!=api.EXCLUDED_TARGET and row['target_id']!=api.EXCLUDED_TARGET
            and frozen.get(row['source_plant_family'])==row['split'],'Excluded target or changed family split')
        aliases=api.aliases(row)
        api.require(len(aliases)==3 and not aliases&seen and not aliases&holds,'Duplicate or held image/camera')
        seen.update(aliases)
        expected_root=api.BASE if row['registry_origin']=='user_selected_diverse_base' else root
        api.require(Path(row['asset_root']).resolve()==expected_root,'Unexpected source asset root')
        required={'rgb','label','target_mask','sample','workspace','crop'}
        api.require(required<=set(row['files']) and ('buffers' in row['files'] or {'depth','validity'}<=set(row['files'])), 'Essential training asset missing')
        for key,relative in row['files'].items():
            source=local_path(expected_root,relative)
            expected=(base_manifest['files'][relative]['sha256'] if expected_root==api.BASE else
                row['crop_sha256'] if key=='crop' else row['artifacts'][key]['sha256'])
            if key=='rgb':api.require(expected==row['rgb_sha256'],'Source RGB hash differs')
            if key=='label':api.require(expected==row['label_sha256'],'Source label hash differs')
            if key=='sample' and row.get('source_sample_sha256'):
                api.require(expected==row['source_sample_sha256'],'Source calibration hash differs')
            dest=destination(row,key,source.suffix)
            api.require(dest not in destinations,'Repeated destination');destinations.add(dest)
            operations.append(dict(id=row['id'],key=key,source=str(source),relative=dest,sha256=expected,bytes=source.stat().st_size))
    primary={}
    for spec in [dict(path=str(result_path),sha256=result_sha256),request['base_result'],*immediate_pins([r.get('lineage',{}) for r in rows])]:
        path=pins.add(spec)
        api.require(path.is_relative_to(api.WORKSPACE) and path.suffix.lower() in ('.json','.jsonl'),'Unexpected provenance file')
        primary[spec['sha256']]=dict(path=str(path),sha256=spec['sha256'])
    pins.verify()
    return dict(result=result,rows=rows,operations=operations,pins=pins,primary=primary,source_root=root)


def copied(source,destination,expected):
    api.require(api.digest(source)==expected,'Logical source bytes changed before copy')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    api.require(not destination.exists(),'Destination overwrite forbidden')
    shutil.copy2(source,destination)
    api.require(api.digest(destination)==expected==api.digest(source),'Source/destination logical bytes changed')


def build(result_path,*,result_sha256,output):
    output=Path(output).resolve()
    api.require(output.parent==api.OUTPUT_ROOT and not output.exists(),'New standalone sibling package required')
    checked=inspect(result_path,result_sha256);output.mkdir()
    try:
        by_id={r['id']:dict(files={},sha256={}) for r in checked['rows']};manifest={}
        for index,op in enumerate(checked['operations']):
            copied(op['source'],output/op['relative'],op['sha256'])
            by_id[op['id']]['files'][op['key']]=op['relative'];by_id[op['id']]['sha256'][op['key']]=op['sha256']
            manifest[op['relative']]=dict(sha256=op['sha256'],bytes=(output/op['relative']).stat().st_size)
            if (index+1)%500==0:print('COPIED',index+1,'/',len(checked['operations']),flush=True)
        receipts={}
        for sha,pin in checked['primary'].items():
            relative='evidence/receipts/'+sha+Path(pin['path']).suffix
            copied(pin['path'],output/relative,sha);receipts[sha]=relative
            manifest[relative]=dict(sha256=sha,bytes=(output/relative).stat().st_size)
        # Original rows retain external historical replay references in provenance only.
        api.jsonl(output/'provenance_index.jsonl',checked['rows'])
        training_rows=[]
        for row in checked['rows']:
            files=by_id[row['id']]['files'];label=api.read(output/files['label'])
            api.require(label['query_pixel_uv']==row['query_pixel_uv'] and label['answer']==row['answer']
                and label['target_id']==row['target_id'],'Copied label differs from sealed registry')
            with Image.open(output/files['rgb']) as image:
                api.require(image.size==(848,408) and image.mode=='RGB' and image.format=='PNG','Actual native lossless848 required')
                api.require(hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==row['decoded_rgb_sha256'],'Decoded native bytes differ')
            record={k:deepcopy(row[k]) for k in ('id','split','target_id','source_target','source_plant_family','kind','rgb_sha256',
                'decoded_rgb_sha256','conservative_camera_signature','label_sha256','source_sample_sha256','query_pixel_uv',
                'accepted_interval_uv','answer','annotation_schema','crop_box_xyxy','crop_source_dimensions','crop_output_dimensions',
                'crop_depends_only_on_query','crop_full_interval_contained') if k in row}
            record.update(files=files,file_sha256=by_id[row['id']]['sha256'],asset_root='.',native_resolution=[848,408],
                native_calibration_file=files['sample'],annotation_epoch=row.get('annotation_epoch'),
                individual_visual_review=api.individually_reviewed(row),registry_origin=row['registry_origin'],
                provenance_record=dict(file='provenance_index.jsonl',id=row['id']),training_started=False)
            training_rows.append(record)
        api.jsonl(output/'index.jsonl',training_rows)
        for split in ('train','validation','test'):
            selected=[r for r in training_rows if r['split']==split];api.jsonl(output/(split+'_index.jsonl'),selected)
            api.jsonl(output/split/'annotations.jsonl',selected)
            actual=len(list((output/split/'images').glob('*.png')))
            api.require(actual==len(selected),'Actual split/images count differs')
        status=dict(schema=SCHEMA,**api.distribution(training_rows),total_images=len(training_rows),
            preserved_user_base_train=477,dominant_target_images=0,labels_recomputed=False,crops_regenerated=False,
            source_counts=checked['result']['counts'],standalone_training_assets=True,
            actual_train_images=len(list((output/'train/images').glob('*.png'))),native_resolution=[848,408],
            new_individually_reviewed_images=checked['result']['accepted_new_images'],
            review_scope='Existing base retains its original individual/sampled review scope; all new additions individually reviewed',
            annotation_epochs=dict(Counter(r['annotation_epoch'] or 'preserved_original' for r in training_rows)),training_started=False)
        api.write(output/'status.json',status)
        api.write(output/'manifest.json',dict(schema=SCHEMA+'.assets',files=manifest,receipt_files_by_sha256=receipts,
            logical_bytes=sum(r['bytes'] for r in manifest.values()),all_copy_hashes_before_after_equal=True))
        api.write(output/'source_snapshot.json',dict(extension_result=dict(path=str(Path(result_path).resolve()),sha256=result_sha256),
            source_bindings=checked['pins'].values,source_packages_modified=False))
        text=('# Standalone diverse native848 dataset\n\n'
            f"TRAIN: {status['counts']['train']}; validation:48; test:90. All477 user-base TRAIN images are preserved. "
            'seed41_full/SubStem_38 is excluded. This is the actual saved count, not20,000.\n\n'
            'Each split has images/, annotations/, depth/, crops/ and annotations.jsonl. Root index.jsonl and split indices use only relative local asset paths. '
            'Load RGB using root / row["files"]["rgb"] and unchanged supervision using root / row["files"]["label"]. '
            'Use row["files"]["buffers"] when present: NPZ keys depth_m, depth_valid, renderer_instance_id, rgba_alpha. Otherwise use the depth NPY and validity PNG. '
            'All images are native848x408 RGB PNG; crops are existing query-centered training crops, copied unchanged.\n\n'
            'Original labels/calibration and provenance may contain historical replay paths; training does not need those external paths. '
            'Use the local files map, not source paths embedded in provenance. Immediate source/review receipts are copied into evidence/receipts; '
            'provenance_index.jsonl preserves original rows. Q4 additions retain false Q3 artifacts and separate approved conditional-background lineage. '
            'Base review scopes remain mixed historical individual/sampled; each new addition was individually reviewed. No training process was launched.\n')
        with (output/'README.md').open('x',encoding='utf-8') as stream:stream.write(text)
        checked['pins'].verify()
        result=dict(schema=SCHEMA,state='complete_standalone_training_assets_materialized',created_utc=datetime.now(timezone.utc).isoformat(),
            counts=status['counts'],train_physical_targets=status['train_physical_targets'],train_families=status['train_families'],
            actual_train_images=status['actual_train_images'],preserved_base_train=477,dominant_target_images=0,
            source_extension_sha256=result_sha256,all_asset_copy_hashes_equal=True,all_native_RGB_decoded_hashes_equal=True,
            annotations_changed=False,crops_changed=False,training_external_assets_required=False,training_started=False,
            output_bindings={name:dict(path=str(output/name),sha256=api.digest(output/name)) for name in
                ('index.jsonl','train_index.jsonl','validation_index.jsonl','test_index.jsonl','provenance_index.jsonl','status.json','manifest.json','source_snapshot.json','README.md')})
        api.write(output/'result.json',result);return result
    except BaseException as error:
        api.write(output/'failure.json',dict(error=repr(error),partial_output_not_admissible=True));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--result-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();r=build(a.result,result_sha256=a.result_sha256,output=a.output)
    print(json.dumps(dict(counts=r['counts'],actual_train_images=r['actual_train_images'])))


if __name__=='__main__':main()
