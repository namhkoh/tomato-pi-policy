"""Streaming original848 annotation. No view cap; actual batch QA is separate.

Native callbacks are losslessly committed by the producer. This CPU consumer
checks each committed frame once, preserving the existing pixel/query gates.
Old diagnostic captures are never recursively replayed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback
from fractions import Fraction
import numpy as np
from PIL import Image
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint, project, depth_evidence
from .capture_visibility import component_masks, interval_visibility, view_quality
from .native848_pair_audit_v2 import verify_camera, world_from_row
from .native848_bulk_workspace_v1 import BulkWorkspace
from . import native848_clear_labels_v1 as labels
from . import native848_query_selection_v1 as queries
from .training_export import view_signature
from .native_sensor_payload import validate_native_static, decode_native_instances
from .native_greenhouse_pair import assert_same_camera


def binding(pin):
    path = Path(pin['path']).resolve()
    require(sha256(path) == pin['sha256'], 'Changed bound file: '+str(path))
    return path


class Admission:
    def __init__(self, context_path, context_sha256, prior_path, prior_sha256, output):
        from .native848_bulk_plan_v1 import check_profile
        self.context_path = binding(dict(path=context_path, sha256=context_sha256))
        self.context_hash = context_sha256
        self.context = ctx = read_json(self.context_path)
        require(ctx['schema'] == 'greenhouse.original848_bulk_context.v1'
                and ctx['split'] == 'train' and ctx['training_approved'] is False,
                'Original TRAIN bulk context required')
        self.plan = read_json(binding(dict(path=ctx['plan_path'], sha256=ctx['plan_sha256'])))
        require(self.plan['per_target_view_cap'] is None and self.plan['total_train_goal'] == 20000
                and self.plan['generated_geometry_used'] is False, 'Explicit uncapped original policy required')
        require(ctx['profile_evidence'] == self.plan['profile_evidence'], 'Context and plan profile evidence differ')
        self.profile = read_json(binding(ctx['profile_evidence']))
        check_profile(self.profile)
        require(self.profile['native_reference_time_policy'] == 'preserved_callback_clock_nondecreasing_not_frame_identity',
                'Native callback clock must remain explicitly separate from frame identity')
        expected_clock = ('paused_timeline_zero_delta' if self.profile['schema'] ==
            'greenhouse.native848_bulk_original_reset8_profile.v1' else 'observed_timeline_delta_after_initialization')
        require(self.profile['scheduler_clock_policy'] == expected_clock,
                'Explicit qualified clock policy required')
        self.pins = dict(ctx['source_bindings'])
        for path, pin in self.profile['source_bindings'].items():
            require(path not in self.pins or self.pins[path] == pin, 'Conflicting source binding')
            self.pins[path] = pin
        verify_bindings(self.pins)  # One flat closure, not the old recursive audits.
        self.binding_hash = fingerprint(ctx['source_bindings'])
        self.catalogue = read_json(binding(dict(path=ctx['catalogue_path'], sha256=ctx['catalogue_sha256'])))
        self.prior = read_json(binding(dict(path=prior_path, sha256=prior_sha256)))
        require(self.prior['frozen_family_splits'].get(ctx['source_family']) == 'train',
                'Held-out or unknown source family cannot enter TRAIN')
        self.seen = set()
        for row in self.prior['records'] + self.prior['preserved_hold_identity_rows']:
            for field in ('rgb_sha256', 'decoded_rgb_sha256', 'conservative_camera_signature'):
                self.seen.add((field, row[field]))
        self.targets = ctx['targets']
        for target_id, target in self.targets.items():
            row = target['source_row']
            require(row['target_id'] == target_id and target['source_family'] == ctx['source_family']
                    and target['split'] == 'train', 'Original target identity or split differs')
            expected = world_from_row(ctx['anchor'], row)
            require(set(expected) == set(target['world_geometry']) and all(
                np.allclose(expected[k], target['world_geometry'][k], atol=1e-9, rtol=0)
                for k in expected), 'Actual target geometry differs from original anatomy')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.checker = BulkWorkspace()
        self.proofs, self.maps, self.records = {}, {}, []
        self.schedule = {r['observation_id']: r for r in self.plan['schedule']}
        cache = read_json(binding(dict(path=self.plan['cache_path'],sha256=self.plan['cache_sha256'])))
        self.poses = {r['sample_id']:r for r in cache['records']}
        self.requests, self.callbacks, self.references = set(), set(), set()
        self.temporal = {}
        self.started = time.perf_counter()
        write_json(self.output/'request.json', dict(context_path=str(self.context_path),
            context_sha256=context_sha256, prior_path=str(Path(prior_path).resolve()),
            prior_sha256=prior_sha256, per_target_view_cap=None,
            source_bindings=self.pins, workspace_bindings=self.checker.bindings,
            consumer_sha256=sha256(__file__), training_approved=False))

    def geometry(self, observation):
        from .native848_bulk_plan_v1 import geometry_pose_key
        item = observation['geometry_proof']
        pin = item['sha256']
        if pin not in self.proofs:
            self.proofs[pin] = read_json(binding(item))
        proof = self.proofs[pin]
        pose = observation['robot_snapshot']
        key = geometry_pose_key(self.context['scene_revision'], pose, self.binding_hash)
        require(item['pose_key'] == proof['pose_key'] == key
                and proof['scene_revision'] == self.context['scene_revision']
                and proof['source_bindings_sha256'] == self.binding_hash
                and proof['robot_root_to_world_usd_row_vectors'] == pose['robot_root_to_world_usd_row_vectors']
                and proof['joint_degrees'] == pose['joint_degrees']
                and proof['screen'] == pose['visual_bound_screen']
                and proof['screen']['passed'] is True, 'Full current robot/scene geometry proof differs')
        return proof

    def frame(self, observation_path):
        started = time.perf_counter()
        path = Path(observation_path).resolve()
        obs = read_json(path)
        name = obs['observation_id']
        require(obs['schema'] == 'greenhouse.original848_bulk_observation.v1'
                and name in self.schedule and obs['source_pose_id'] == self.schedule[name]['source_pose_id']
                and obs['capture_role'] == 'production' and obs['split'] == 'train'
                and obs['source_family'] == self.context['source_family']
                and obs['profile'] == self.profile['profile'] and obs['training_approved'] is False
                and obs['context_sha256'] == self.context_hash
                and Path(obs['context_path']).resolve() == self.context_path,
                'Frame is not a scheduled original qualified production observation')
        require(name not in {r['observation_id'] for r in self.records}, 'Repeated observation')
        sync, request = obs['synchronization'], obs['request_evidence']
        require(sync['static_before'] == sync['static_after'] and bool(sync['static_before'])
                and request['native_requests'] == 1 and request['callback_count'] == 1
                and request['requested_subframes'] == self.profile['request_subframes'],
                'Changed scene, unmatched callback or hidden render retries')
        # Callback, request and reference tokens must be unique. Files may finish
        # asynchronously, so no file-order temporal monotonicity is assumed.
        for key, seen in [('request_index', self.requests), ('callback_sequence', self.callbacks)]:
            value = fingerprint(sync[key])
            require(value not in seen, 'Repeated native '+key)
            seen.add(value)
        require(request['render_settings'] == self.profile['render_settings']
                and request['reset_api'] == self.profile['reset_api']
                and request['reset_returned_without_exception'] is True
                and request['delta_time_seconds'] == self.profile['delta_time_seconds']
                and request['wait_for_render'] is self.profile['wait_for_render'] is True,
                'Actual capture settings/reset/clock policy differs from qualified profile')
        require(abs((request['timeline_after_seconds']-request['timeline_before_seconds'])
                    - self.profile['delta_time_seconds']) < 1e-7,
                'Observed scheduler timeline did not advance by qualified delta')
        proof = self.geometry(obs)
        cached = self.poses[obs['source_pose_id']]
        assert_same_camera(obs['calibration'],cached['calibration'])
        require(obs['target_id'] == cached['target_id']
                and obs['robot_snapshot']['joint_degrees'] == cached['joint_degrees']
                and np.allclose(obs['robot_snapshot']['robot_root_to_world_usd_row_vectors'],
                    cached['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0),
                'Captured pose differs from scheduled embodied pose')
        rgb_path = binding(obs['files']['rgb'])
        with Image.open(rgb_path) as image:
            require(image.mode == 'RGB' and image.size == (848,408), 'Native RGB848 required')
            rgb = np.asarray(image).copy()
        with np.load(binding(obs['files']['buffers']), allow_pickle=False) as data:
            require(set(data.files) == {'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},
                    'Exact lossless callback buffers required')
            depth, ids, valid, alpha = [data[k] for k in ('depth_m','renderer_instance_id','depth_valid','rgba_alpha')]
        require(depth.shape == ids.shape == valid.shape == alpha.shape == (408,848)
                and depth.dtype == np.float32 and ids.dtype == np.uint32
                and valid.dtype == bool and alpha.dtype == np.uint8, 'Native buffer type/shape differs')
        mapping_pin = obs['mapping']
        if mapping_pin['sha256'] not in self.maps:
            value = read_json(binding(mapping_pin))
            require(value['scope'] == 'all_observed_renderer_IDs_only', 'Wrong renderer mapping scope')
            self.maps[mapping_pin['sha256']] = {int(k):v for k,v in value['renderer_id_to_prim'].items()}
        mapping = self.maps[mapping_pin['sha256']]
        observed = set(map(int,np.unique(ids)))
        require(not (observed-set(mapping)-{0,1}), 'Unmapped observed renderer ID')
        require(all(not mapping.get(i,'').startswith('/World/GeneratedNativePilot/') for i in observed),
                'Generated geometry entered original frame')
        payload = dict(obs['native_payload_header'])
        payload['rgb'] = np.dstack((rgb,alpha))
        payload['distance_to_image_plane'] = depth
        payload['instance_id_segmentation'] = dict(data=ids,info=dict(idToLabels=mapping))
        require(payload['camera_params'] == obs['rendered_camera_params'], 'Native camera header differs')
        _, _, checked_valid, reference, token = validate_native_static(payload,obs['calibration'],
            sync['static_before'],sync['static_after'],sync['callback_sequence'])
        token.update(instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),mapping_sha256=fingerprint(mapping),reference_time=reference)
        require(reference == sync['reference_time'] and token == sync['freshness']
                and np.array_equal(checked_valid,valid), 'Native callback evidence differs')
        decode_native_instances(payload,[848,408])
        self.temporal[sync['request_index']] = dict(reference=reference,token=token,observation_id=name)
        target = self.targets[obs['target_id']]
        row, world = target['source_row'], target['world_geometry']
        cal = obs['calibration']
        components, organs, owners = component_masks(ids,mapping,self.catalogue)
        identity = next(c for c in self.catalogue if c['variant_id'] == row['variant_id']
                        and c['component_id'] == row['component_id'])
        radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
        nominal, interval = project([world['nominal_world_m']],cal)[0], project(world['interval_world_m'],cal)
        visibility, mask = interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,identity,radius)
        meta = dict(schema_version='greenhouse.original848_bulk_sample.v1',sample_id=name,
            calibration=cal, robot_snapshot=obs['robot_snapshot'], rendered_camera_params=obs['rendered_camera_params'],
            geometry_screen=proof['screen'], profile=obs['profile'],
            synchronization=dict(**sync,method='frozen_scene_single_native_writer_payload',scene_unchanged_during_capture=True,dynamic_recording_supported=False),
            quality=view_quality(cal,nominal,interval,radius,visibility,rgb,mask),
            supervision=dict(**world,target_id=row['target_id'],source_target_id=row['target_id'],
                split_group=obs['source_family'],cut_region_proposal=row['cut_region_proposal'],
                nominal_projected=nominal,projected_interval=interval,
                depth_evidence=depth_evidence(nominal,depth,valid,radius),visibility_evidence=visibility,
                cut_safety_validated=False),
            observation_path=str(path),observation_sha256=sha256(path),training_approved=False)
        verify_camera(meta)
        baseline = labels.derive(meta,target['anatomy_report'],rgb,depth,valid,components,self.catalogue)
        label, trace, selection = queries.annotate_v1(meta,target['anatomy_report'],rgb,depth,valid,
            components,self.catalogue,target_mask=mask.astype(np.uint8)*255,expected_label=baseline)
        workspace = self.checker.check(meta)
        hashes = dict(rgb_sha256=obs['files']['rgb']['sha256'],
            decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            conservative_camera_signature=view_signature(meta,obs['source_family'],None))
        duplicate = any((key,value) in self.seen for key,value in hashes.items())
        self.seen.update(hashes.items())
        local, traced, reachable = label['eligible'] is True, bool(trace and trace['passed']), workspace['result']['workspace_passed'] is True
        decision = ('exclude_duplicate_or_prior_hold' if duplicate else 'exclude_local_clarity' if not local
            else 'hold_full_trace' if not traced else 'hold_workspace' if not reachable
            else 'automated_candidate_pending_batch_QA')
        folder = self.output/name
        folder.mkdir()
        for filename, value in [('sample.json',meta),('label.json',label),('query_selection.json',selection),
                                ('workspace.json',workspace),('query_trace.json',trace)]:
            write_json(folder/filename,value)
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'target.png')
        record = dict(observation_id=name,capture_index=obs.get('capture_index',self.schedule[name].get('capture_index',len(self.records))),
            profile=obs['profile'],reset_segment_id=self.context_hash,head_joint_degrees={k:v for k,v in obs['robot_snapshot']['joint_degrees'].items() if k.startswith('head_')},
            source_pose_id=obs['source_pose_id'],target_id=row['target_id'],
            source_target=row['target_id'],source_family=obs['source_family'],split='train',**hashes,
            observation_path=str(path),observation_sha256=meta['observation_sha256'],
            rgb_path=str(rgb_path),label_path=str(folder/'label.json'),label_sha256=sha256(folder/'label.json'),
            sample_path=str(folder/'sample.json'),sample_sha256=sha256(folder/'sample.json'),
            workspace_path=str(folder/'workspace.json'),workspace_sha256=sha256(folder/'workspace.json'),
            target_mask_path=str(folder/'target.png'),target_mask_sha256=sha256(folder/'target.png'),
            query_trace_path=str(folder/'query_trace.json'),query_trace_sha256=sha256(folder/'query_trace.json'),
            query_selection_path=str(folder/'query_selection.json'),query_selection_sha256=sha256(folder/'query_selection.json'),
            decision=decision,automated_pass=decision=='automated_candidate_pending_batch_QA',
            individual_visual_review=False,training_approved=False,accepted_training_increment=0,
            label_metrics=label.get('clarity',{}),query_metrics=label.get('query_usability',{}),
            elapsed_seconds=time.perf_counter()-started)
        write_json(folder/'decision.json',record)
        self.records.append(record)
        with (self.output/'decisions.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        print('BULK848_CPU',name,decision,round(record['elapsed_seconds'],3),flush=True)
        return record

    def finish(self):
        previous = None
        for request_index, entry in sorted(self.temporal.items()):
            reference, token = Fraction(*entry['reference']), entry['token']
            if previous is not None:
                require(reference >= previous['reference']
                        and token['callback_sequence'] > previous['token']['callback_sequence'],
                        'Native reference/callback failed ordered continuity')
                require(all(token[k] != previous['token'][k] for k in
                        ('camera_sha256','rgb_sha256','depth_sha256')),
                        'Adjacent native camera/buffers failed freshness')
            previous = dict(reference=reference,token=token)
        stats = self.checker.finish()
        require(sha256(self.context_path) == self.context_hash, 'Batch context mutated')
        result = dict(schema='greenhouse.original848_bulk_admission.v1',
            state='automated_frame_checks_complete_pending_batch_QA_and_owned_capture_completion',
            records=self.records,native_frames_checked=len(self.records),
            automated_candidates=sum(r['automated_pass'] for r in self.records),
            elapsed_seconds=time.perf_counter()-self.started,workspace_stats=stats,
            individual_visual_review=False,training_approved=False,accepted_training_increment=0)
        write_json(self.output/'result.json',result)
        return result


def run(args):
    worker = Admission(args.context,args.context_sha256,args.prior,args.prior_sha256,args.output)
    frames = Path(args.frames)
    seen = set()
    try:
        while True:
            pending = [p for p in frames.glob('*/observation.json') if str(p) not in seen]
            if args.shard_count > 1:
                owned_pending = []
                for path in pending:
                    if read_json(path)['capture_index'] % args.shard_count == args.shard_index:
                        owned_pending.append(path)
                    else:
                        seen.add(str(path))
                pending = owned_pending
            for path in pending:
                worker.frame(path)
                seen.add(str(path))
            if not args.follow:
                break
            if (frames.parent/'result.json').is_file():
                # Producer publishes completion after all atomic frame commits.
                # Drain again after observing that terminal event to avoid a race.
                for path in frames.glob('*/observation.json'):
                    if str(path) not in seen and (args.shard_count == 1 or
                            read_json(path)['capture_index'] % args.shard_count == args.shard_index):
                        worker.frame(path)
                        seen.add(str(path))
                break
            require(not (frames.parent/'failure.json').exists(), 'Native batch failed; candidates remain unaccepted')
            time.sleep(.25)
        return worker.finish()
    except BaseException:
        write_json(worker.output/'failure.json',dict(error=traceback.format_exc(),completed=len(worker.records),training_approved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('context','context-sha256','prior','prior-sha256','frames','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    require(1 <= args.shard_count <= 8 and 0 <= args.shard_index < args.shard_count,'Bounded CPU shard required')
    run(args)
