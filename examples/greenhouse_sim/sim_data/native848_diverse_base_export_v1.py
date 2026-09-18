"""Export the user's exact 477-image diverse base; never append or reannotate."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_tonight_bulk_v1'
OUTPUT = ROOT/'data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_diverse_base_20260917_v1'
SOURCE_INDEX_SHA = 'fd2732e3c36b1503a2821f46a65c76c9aac309a4e3e29a93ac08734dd0eb631d'
SOURCE_STATUS_SHA = 'ac0e604f64edc9da51b1f611cdff67e01c5e84bf7e625ce64efc123f8833615d'
EXCLUDED_TARGET = 'seed41_full/SubStem_38'
EXPECTED = dict(train=477, validation=48, test=90)
ALIAS_KEYS = ('rgb_sha256','decoded_rgb_sha256','conservative_camera_signature')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')


def rejected(row):
    return row.get('source_target') == EXCLUDED_TARGET or row.get('target_id') == EXCLUDED_TARGET


def inspected(row):
    return (row.get('individual_visual_review') is True or
        (row.get('review', {}).get('decision') == 'accept' and
         row.get('review', {}).get('full_native_image_inspected') is True))


def prepare():
    require(sha(SOURCE/'index.jsonl') == SOURCE_INDEX_SHA and sha(SOURCE/'status.json') == SOURCE_STATUS_SHA,
            'The explicitly authorized frozen old package changed')
    status = read(SOURCE/'status.json')
    with (SOURCE/'index.jsonl').open(encoding='utf-8') as stream:
        all_rows = [json.loads(line) for line in stream if line.strip()]
    require(dict(Counter(row['split'] for row in all_rows)) == status['counts'] ==
        dict(train=19913, validation=48, test=90), 'Old source counts differ')
    keep = [row for row in all_rows if not rejected(row)]
    require(dict(Counter(row['split'] for row in keep)) == EXPECTED, 'Exact user-selected base differs')
    train = [row for row in keep if row['split'] == 'train']
    require(len({row['source_target'] for row in train}) == 58, 'Expected 58 training targets')
    require(len({row['id'] for row in keep}) == len(keep), 'Repeated base image ID')
    families, seen, files = {}, set(), {}
    for row in keep:
        require(not rejected(row), 'Rejected target cannot enter any split')
        family = row['source_plant_family']
        require(row['source_target'].startswith(family+'/') and
            families.setdefault(family,row['split']) == row['split'], 'Source family split leakage')
        keys = {(key,row[key]) for key in ALIAS_KEYS}
        require(not keys & seen, 'Repeated RGB/decoded-RGB/camera alias in diverse base')
        seen.update(keys)
        for relative in row['files'].values():
            path = Path(relative)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe relative asset path')
            source = (SOURCE/path).resolve()
            require(source.is_relative_to(SOURCE) and source.is_file(), 'Missing or escaped source asset')
            files[path.as_posix()] = source
    pins = {str(SOURCE/'index.jsonl'):SOURCE_INDEX_SHA, str(SOURCE/'status.json'):SOURCE_STATUS_SHA,
            str(Path(__file__).resolve()):sha(__file__)}
    baseline = read(SOURCE/'baseline.json')
    selection_path = Path(baseline['prior_checkpoint'])/'evidence/selection.json'
    selection = read(selection_path)
    require(all(selection['frozen_family_splits'].get(family) == split for family,split in families.items()),
            'Frozen original family assignment changed')
    pins[str(SOURCE/'baseline.json')] = sha(SOURCE/'baseline.json')
    pins[str(selection_path)] = sha(selection_path)
    hold_path = SOURCE/'visual_holds.jsonl'
    pins[str(hold_path)] = sha(hold_path)
    with hold_path.open(encoding='utf-8') as stream:
        held = [json.loads(line) for line in stream if line.strip()]
    held += selection['preserved_hold_identity_rows']
    hold_aliases = {(key,row[key]) for row in held for key in ALIAS_KEYS if row.get(key)}
    require(not seen & hold_aliases, 'A retained base row now has a preserved visual hold')
    # Anonymous exclusion keys preserve cross-corpus dedup/holds without shipping
    # rejected target records or any of their image assets.
    old_aliases = {(key,row[key]) for row in all_rows for key in ALIAS_KEYS if row.get(key)}
    return keep, files, families, pins, old_aliases, hold_aliases


def export():
    require(not OUTPUT.exists() and OUTPUT.parent == SOURCE.parent, 'New sibling base output required')
    rows, files, families, pins, old_aliases, hold_aliases = prepare()
    OUTPUT.mkdir()
    try:
        copied = {}; logical_bytes = 0
        for index,(relative,source) in enumerate(sorted(files.items()),1):
            before = sha(source)
            destination = OUTPUT/relative; destination.parent.mkdir(parents=True,exist_ok=True)
            require(not destination.exists(), 'Asset overwrite forbidden')
            shutil.copy2(source,destination)
            require(sha(destination) == before == sha(source), 'Logical file bytes changed during copy')
            copied[relative] = dict(sha256=before, bytes=source.stat().st_size)
            logical_bytes += source.stat().st_size
            if index % 250 == 0:
                print('DIVERSE_BASE_FILES',index,len(files),flush=True)
        for row in rows:
            require(copied[Path(row['files']['rgb']).as_posix()]['sha256'] == row['rgb_sha256']
                and copied[Path(row['files']['label']).as_posix()]['sha256'] == row['label_sha256'],
                'Copied accepted RGB or label differs from its original record')
            with Image.open(OUTPUT/row['files']['rgb']) as image:
                require(image.size == (848,408) and image.mode == 'RGB' and image.format == 'PNG',
                        'Retained RGB is not native848 lossless RGB')
                require(hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == row['decoded_rgb_sha256'],
                        'Actual retained decoded RGB identity differs')
        for path, expected in pins.items():
            require(sha(path) == expected, 'Snapshot source changed during base export')
        jsonl(OUTPUT/'index.jsonl',rows)
        for split in EXPECTED:
            jsonl(OUTPUT/(split+'_index.jsonl'),[row for row in rows if row['split']==split])
        write(OUTPUT/'global_exclusion_aliases.json',dict(
            schema='greenhouse.native848_global_exclusion_aliases.v1',
            accepted_identity_aliases=[dict(kind=k,value=v) for k,v in sorted(old_aliases)],
            visual_hold_aliases=[dict(kind=k,value=v) for k,v in sorted(hold_aliases)],
            source_bindings=pins,contains_images=False,contains_training_records=False))
        train=[row for row in rows if row['split']=='train']
        target_counts=dict(sorted(Counter(row['source_target'] for row in train).items()))
        family_counts=dict(sorted(Counter(row['source_plant_family'] for row in train).items()))
        status=dict(schema='greenhouse.native848_diverse_base_checkpoint.v1',counts=EXPECTED,
            total_images=len(rows),native_resolution=[848,408],train_original_targets=58,
            train_families=len(family_counts),train_by_target=target_counts,train_by_family=family_counts,
            maximum_existing_train_views_per_target=max(target_counts.values()),
            inherited_user_selected_base_not_reduced_by_cap=True,dominant_target_images=0,
            individually_reviewed_train=sum(inspected(row) for row in train),
            remaining_train_review_scope='Original record-level evidence retained; sampled batch QA is not individual image inspection',
            frozen_family_splits=families,annotations_changed=False,new_capture_credit=0,
            training_started=False,training_approved=False)
        write(OUTPUT/'status.json',status)
        write(OUTPUT/'manifest.json',dict(schema='greenhouse.native848_diverse_base_assets.v1',
            files=copied,logical_bytes=logical_bytes,source_bindings=pins,
            all_copied_files_before_after_sha256_equal=True,all615_RGB_encoded_decoded_native_verified=True,
            source_package_modified=False,annotations_modified=False))
        readme=(
            '# Diverse native848 base\n\n477 TRAIN images across 58 physical source targets and '
            +str(len(family_counts))+' training families; 48 validation and 90 test images remain separate. '
            'Every seed41_full/SubStem_38 record is excluded from every dataset index.\n\n'
            'This is the exact existing base selected by the user. Existing per-target counts are preserved '
            '(maximum '+str(max(target_counts.values()))+'); no cap was used to discard base images. '
            'Caps on newly generated proposals are a separate policy.\n\n'
            'All selected asset files were copied byte-for-byte. RGB is native 848x408 PNG; existing query crops '
            'may be 768x768 as recorded. Labels, source IDs, splits and original individual/sampled review scope '
            'are retained without reannotation. This export adds zero newly captured images.\n\n'
            'index.jsonl and the three split indices use paths relative to this directory. status.json reports '
            'target/family counts. manifest.json pins every copied asset. The anonymous global exclusion aliases '
            'preserve duplicate and hold protections across the old corpus without shipping its rejected images. '
            'Upstream evidence paths/hashes remain in each original record and source_bindings.\n')
        with (OUTPUT/'README.md').open('x',encoding='utf-8') as stream:stream.write(readme)
        result=dict(schema='greenhouse.native848_diverse_base_export.v1',state='exact_user_selected_diverse_base_exported',
            created_utc=datetime.now(timezone.utc).isoformat(),output=str(OUTPUT),counts=EXPECTED,
            train_original_targets=58,train_families=len(family_counts),excluded_source_target=EXCLUDED_TARGET,
            excluded_train_images=19436,dominant_target_images=0,new_capture_credit=0,
            source_package_modified=False,source_bindings=pins,copied_files=len(copied),logical_bytes=logical_bytes,
            output_bindings={name:dict(path=str(OUTPUT/name),sha256=sha(OUTPUT/name)) for name in
                ('index.jsonl','train_index.jsonl','validation_index.jsonl','test_index.jsonl','status.json',
                 'manifest.json','global_exclusion_aliases.json','README.md')},training_approved=False)
        write(OUTPUT/'result.json',result)
        print(json.dumps(dict(output=str(OUTPUT),counts=EXPECTED,result_sha256=sha(OUTPUT/'result.json'))),flush=True)
        return result
    except BaseException as error:
        write(OUTPUT/'failure.json',dict(error=repr(error),partial_output_not_accepted=True,source_package_modified=False))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:export()
    else:
        rows,files,families,pins,_,_=prepare()
        print(json.dumps(dict(counts=dict(Counter(row['split'] for row in rows)),files=len(files),
                              source_bindings=pins,output=str(OUTPUT),export_performed=False)))


if __name__=='__main__':main()
