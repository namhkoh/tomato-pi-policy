"""Package an explicitly human-reviewed v4 SUBSET, never a full/action release.

Model messages contain original RGB, task instruction and the reviewed answer.
Depth/calibration and private evaluator truth stay in separate sidecar files.
No inference, simulator or physical robot is started by this command.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .active_perception_contract import SYSTEM_PROMPT, TASK_ID, contract_hash
from .active_perception_review import ActiveReview, DEFAULT_PILOT
from .capture_contract import fingerprint
from .dataset_review import require, write_json
from .depth_preview import sha256


def export_subset(pilot, output):
    output = Path(output).resolve()
    require(not output.exists(), 'Choose a new export directory')
    app = ActiveReview(pilot); app._verify_bindings()
    state = app.state(); snapshot = fingerprint(state)
    selected, excluded, families, images = [], [], {}, {}
    for row in state['samples']:
        entry = app.entries[row['id']]
        split, family = entry['split'], entry['source_plant_family']
        require(split in ('train', 'validation', 'test'), 'Invalid source split')
        image_hash = sha256(entry['model_input']['rgb_path'])
        require(families.setdefault(family, split) == split and images.setdefault(image_hash, split) == split,
                'Family/image split leakage')
        review = row['human_review']
        reason = 'source_hold' if row['source_blocks'] else 'no_human_review' if not review else review['decision'] if review['decision'] != 'accept' else None
        if reason: excluded.append(dict(id=row['id'], reason=reason))
        else: selected.append((entry, review, image_hash))
    require(selected, 'No explicitly human-accepted, unheld v4 examples; nothing exported')
    # All preflight checks precede creation. A failed copy leaves an incomplete
    # directory WITHOUT manifest.json; never replace a prior usable package.
    output.mkdir(parents=True)
    (output/'images').mkdir(); (output/'sidecars').mkdir()
    rows = {s: [] for s in ('train', 'validation', 'test')}
    bindings, provenance = {}, []
    for entry, review, image_hash in selected:
        sid = entry['id']; app.check(sid); app.buffers(sid)
        image_rel = 'images/'+image_hash+'.png'
        image_path = output/image_rel
        if not image_path.exists(): shutil.copyfile(entry['model_input']['rgb_path'], image_path)
        require(sha256(image_path) == image_hash, 'Copied RGB differs from native capture')
        bindings[image_rel] = image_hash
        rows[entry['split']].append(dict(id=sid, messages=[
            dict(role='system', content=SYSTEM_PROMPT),
            dict(role='user', content=[dict(type='image', image=image_rel),
                 dict(type='text', text=entry['model_input']['instruction'])]),
            dict(role='assistant', content=json.dumps(entry['proposed_answer'], ensure_ascii=False, allow_nan=False))]))
        side = dict(entry['native_sidecars']); folder = output/'sidecars'/sid; folder.mkdir()
        for key, name in (('depth_path', 'depth_m.npy'), ('depth_valid_path', 'depth_valid.png')):
            source = Path(side[key]); relative = f'sidecars/{sid}/{name}'
            shutil.copyfile(source, output/relative)
            require(sha256(output/relative) == sha256(source), 'Copied native sidecar changed')
            bindings[relative] = sha256(output/relative); side[key] = relative
        metadata_rel = f'sidecars/{sid}/metadata.json'
        write_json(output/metadata_rel, dict(native_sidecars=side, private_evaluator=entry['private_evaluator'],
            source_plant_family=entry['source_plant_family'], source_id=entry['source_id'], kind=entry['kind'],
            contrast_id=entry['contrast_id'], split=entry['split'], human_review=review,
            temporal_episode=False, physical_execution_approved=False))
        bindings[metadata_rel] = sha256(output/metadata_rel)
        review_path = app.records/'human'/(sid+'.json')
        provenance.append(dict(id=sid, example_sha256=fingerprint(entry), review_path=str(review_path),
                               review_sha256=sha256(review_path), rgb_sha256=image_hash))
    for split, messages in rows.items():
        name = split+'.jsonl'
        with (output/name).open('x', encoding='utf-8', newline='\n') as stream:
            for row in messages: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        bindings[name] = sha256(output/name)
    app._verify_bindings()
    require(fingerprint(app.state()) == snapshot, 'Review/source state changed during export; package incomplete')
    for rel, expected in bindings.items(): require(sha256(output/rel) == expected, 'Output changed during export')
    result = dict(schema='greenhouse.active_perception_reviewed_subset.v1', task_id=TASK_ID,
        task_contract_sha256=contract_hash(), created_utc=datetime.now(timezone.utc).isoformat(),
        state='reviewed_synthetic_perception_subset_not_full_training_release',
        training_release_approved=False, action_training_eligible=False, physical_execution_approved=False,
        dynamic_episodes=0, count=len(selected), splits={s:len(v) for s,v in rows.items()},
        kinds=dict(Counter(e['kind'] for e,_,_ in selected)), excluded=excluded,
        pilot_path=str(app.path), pilot_sha256=app.pilot_hash,
        source_bindings_sha256={str(p):h for p,h in app.bindings.items()},
        provenance=provenance, files_sha256=bindings,
        message_format='local image paths relative to package root; trainer-specific adapter required',
        model_input_policy='Only system/user messages; never load sidecars as model input',
        depth_policy='Native Isaac optical-Z copied byte-for-byte; RGB-only messages in this prototype')
    write_json(output/'manifest.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, default=DEFAULT_PILOT)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv); result = export_subset(a.pilot, a.output)
    print(json.dumps({k:result[k] for k in ('state','count','splits','training_release_approved')}, indent=2))


if __name__ == '__main__': main()
