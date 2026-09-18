"""Batch authenticated original robot poses in one unchanged plant scene.

Each component remains a separately checked original plan. Only robot snapshots
change between components; geometry, optics, lighting and target anatomy do not.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json
import numpy as np
from . import native848_bulk_plan_v1 as original
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json

SCHEMA = 'greenhouse.original848_multianchor_bulk_plan.v1'
CACHE_SCHEMA = 'greenhouse.original848_multianchor_cached_poses.v1'
VIEW_POLICY = original.VIEW_POLICY
RESULT_STATE = original.RESULT_STATE
check_profile = original.check_profile
ORIGINAL_PROFILE_SCHEMA = original.ORIGINAL_PROFILE_SCHEMA
ROOT = Path(__file__).resolve().parents[3]


def binding(spec):
    p = Path(spec['path']).resolve()
    require(p.is_relative_to(ROOT) and sha256(p) == spec['sha256'], 'Changed workspace component')
    return p


def merge(destination, values):
    for p, h in values.items():
        require(p not in destination or destination[p] == h, 'Conflicting source binding')
        destination[p] = h


def implementation_bindings():
    values = original.implementation_bindings()
    for p in (Path(__file__), Path(__file__).with_name('native848_multianchor_worker_v1.py')):
        values[str(p.resolve())] = sha256(p)
    return dict(sorted(values.items()))


def scene_identity(anchor):
    from .native_greenhouse_pair import load_source
    manifest, sample, pins = load_source(anchor['source_capture'], anchor['source_sample'])
    require(anchor['expected_robot_snapshot'] == sample['robot_snapshot'], 'Authenticated robot reference differs')
    require(anchor['split'] == 'train', 'Original TRAIN anchors only')
    optical = {k: v for k, v in anchor['expected_calibration'].items()
               if k != 'camera_to_world_usd_row_vectors'}
    identity = dict(family=anchor['source_family'], split=anchor['split'],
        collection_plan=str(Path(anchor['source_collection_plan']).resolve()),
        collection_plan_sha256=sha256(anchor['source_collection_plan']),
        original_variant=anchor['original_variant'], scene_variants=anchor['scene_variants'],
        scene_counts=anchor['expected_scene_counts'], scene_code_bindings=anchor['scene_code_bindings'],
        optical=optical, camera_mount=anchor['expected_robot_snapshot']['camera_to_head_column_vectors'])
    identity['manifest'] = {k: manifest[k] for k in
        ('lighting', 'renderer', 'scene_counts', 'variants', 'unbundled_external_prop_roots_excluded')}
    return identity, pins


def components(entries):
    require(2 <= len(entries) <= 16, 'Two to sixteen source plans required')
    first = None
    identity = None
    records = []
    pins = {}
    origins = {}
    anchor_paths = set()
    plan_paths = set()
    camera_keys = set()
    for entry in entries:
        path = binding(entry['plan'])
        require(str(path) not in plan_paths, 'Repeated component plan')
        plan_paths.add(str(path))
        plan = read_json(path)
        checked = original.check(plan)
        current_identity, source_pins = scene_identity(checked['anchor'])
        if first is None:
            first, identity = checked, current_identity
            profile = plan['profile_evidence']
        require({k:v for k,v in identity.items() if k!='camera_mount'} ==
                {k:v for k,v in current_identity.items() if k!='camera_mount'}
                and np.allclose(identity['camera_mount'],current_identity['camera_mount'],atol=1e-9,rtol=0),
                'Components do not share the exact static scene, optics and source mount')
        require(plan['profile_evidence'] == profile, 'Renderer profile differs')
        anchor_paths.add(checked['anchor']['reference_entry_id'])
        by = {r['sample_id']: r for r in checked['records']}
        selected = entry['selected_sample_ids']
        require(selected and len(selected) == len(set(selected)) and set(selected) <= set(by),
                'Unique selected component poses required')
        merge(pins, checked['source_bindings'])
        merge(pins, source_pins)
        merge(pins, {str(path): entry['plan']['sha256']})
        for sid in selected:
            require(sid not in origins, 'Repeated source pose between components')
            rec = deepcopy(by[sid])
            key = original.geometry_pose_key('multianchor-camera',
                dict(robot_root_to_world_usd_row_vectors=rec['robot_root_to_world_usd_row_vectors'],
                     joint_degrees=rec['joint_degrees']), '0' * 64)
            camera = tuple(np.asarray(rec['camera_to_world_usd_row_vectors']).round(10).reshape(-1))
            require(camera not in camera_keys, 'Repeated complete camera between components')
            camera_keys.add(camera)
            records.append(rec)
            origins[sid] = dict(plan=entry['plan'], anchor_path=rec['anchor_reference_path'],
                anchor_sha256=rec['anchor_reference_sha256'], authenticated_pose_key=key)
    counts = Counter(r['target_id'] for r in records)
    require(2 <= len(counts) and 2 <= len(anchor_paths) and len(records) <= 512,
            'Bounded multiple-target, multiple-anchor batch required')
    require('seed41_full/SubStem_38' not in counts and max(counts.values()) <= 128
            and max(counts.values()) * 2 <= len(records), 'Unbalanced or excluded target batch')
    # Round robin across targets keeps interrupted prefixes diverse.
    grouped = {t: [r for r in records if r['target_id'] == t] for t in sorted(counts)}
    records = [group[i] for i in range(max(counts.values())) for group in grouped.values() if i < len(group)]
    return dict(first=first, identity=identity, profile=profile, records=records,
                source_bindings=pins, origins=origins, counts=dict(counts))


def prepare(entries, output):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),
            'Create-only diagnostic batch required')
    data = components(entries)
    output.mkdir(parents=True)
    cache = dict(schema=CACHE_SCHEMA, source_family=data['identity']['family'],
        anchor_reference_path=data['first']['cache']['anchor_reference_path'],
        anchor_reference_sha256=data['first']['cache']['anchor_reference_sha256'],
        source_bindings=data['source_bindings'], records=data['records'],
        source_pose_components=data['origins'], static_scene_identity=data['identity'],
        resolution=[848,408], generated_geometry_used=False, training_approved=False)
    cp = output/'pose_cache.json'
    save_json(cp, cache)
    ids = [r['sample_id'] for r in data['records']]
    plan = dict(schema=SCHEMA, component_plans=entries, cache_path=str(cp), cache_sha256=sha256(cp),
        profile_evidence=data['profile'], selected_sample_ids=ids,
        schedule=[dict(observation_id='bulk_'+sid,source_pose_id=sid,capture_role='production') for sid in ids],
        max_frames=len(ids), resolution=[848,408], instance_backend='fast',
        view_policy=VIEW_POLICY, per_target_view_cap=None, total_train_goal=20000,
        training_approved=False, source_cap_reset=False, generated_geometry_used=False,
        geometry_novelty_qualified=False, automatic_retries=False, queue_frames=8,
        queue_bytes=256*2**20, writer_threads=2,
        geometry_policy='native_full_screen_each_new_complete_joint_pose',
        implementation_bindings=implementation_bindings())
    check(plan)
    pp = output/'bulk_plan.json'
    save_json(pp, plan)
    result = dict(schema='greenhouse.original848_multianchor_prepared.v1',
        plan=dict(path=str(pp),sha256=sha256(pp)), pose_cache=dict(path=str(cp),sha256=sha256(cp)),
        frames=len(ids), target_counts=data['counts'], component_count=len(entries),
        native_launched=False, accepted_training_increment=0, training_approved=False)
    save_json(output/'result.json', result)
    return result


def check(plan):
    require(plan['schema'] == SCHEMA and plan['implementation_bindings'] == implementation_bindings(),
            'Unknown or changed multi-anchor implementation')
    require(plan['resolution'] == [848,408] and plan['instance_backend'] == 'fast'
        and plan['view_policy'] == VIEW_POLICY and plan['per_target_view_cap'] is None
        and plan['total_train_goal'] == 20000
        and plan['geometry_policy'] == 'native_full_screen_each_new_complete_joint_pose'
        and all(plan[k] is False for k in ('training_approved','source_cap_reset','generated_geometry_used',
                                         'geometry_novelty_qualified','automatic_retries')), 'Changed capture scope')
    require(plan['queue_frames'] == 8 and plan['queue_bytes'] == 256*2**20 and plan['writer_threads'] == 2,
            'Changed bounded writer policy')
    cp = binding(dict(path=plan['cache_path'],sha256=plan['cache_sha256']))
    cache = read_json(cp)
    data = components(plan['component_plans'])
    require(cache['schema'] == CACHE_SCHEMA and cache['records'] == data['records']
        and cache['source_bindings'] == data['source_bindings']
        and cache['source_pose_components'] == data['origins']
        and cache['static_scene_identity'] == data['identity'], 'Aggregate source poses or static scene changed')
    require(cache['anchor_reference_path'] == data['first']['cache']['anchor_reference_path']
        and cache['anchor_reference_sha256'] == data['first']['cache']['anchor_reference_sha256'],
        'Initial stage anchor changed')
    ids = [r['sample_id'] for r in data['records']]
    require(plan['selected_sample_ids'] == ids and plan['max_frames'] == len(ids)
        and plan['schedule'] == [dict(observation_id='bulk_'+sid,source_pose_id=sid,capture_role='production') for sid in ids]
        and plan['profile_evidence'] == data['profile'], 'Aggregate native schedule/profile differs')
    pins = dict(data['source_bindings'])
    merge(pins, plan['implementation_bindings'])
    merge(pins, {str(cp):plan['cache_sha256']})
    verify_bindings(pins)
    return dict(cache=cache, anchor=data['first']['anchor'], source_plan=data['first']['source_plan'],
        reports=data['first']['reports'], profile=data['first']['profile'], records=data['records'],
        source_bindings=dict(sorted(pins.items())))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--request',required=True)
    p.add_argument('--request-sha256',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    request = read_json(binding(dict(path=a.request,sha256=a.request_sha256)))
    print(json.dumps(prepare(request['component_plans'],a.output)))
