"""Complete one genuinely closed, reserved persistent generated capture batch.

Ordinary persistent v1 only. Experimental requests and replay controls fail
closed. This reports capture/duplicate accounting, never annotation acceptance.
Token and coordinator URL are read only from the configured environment.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np

from .dataset_review import require
from .native848_bulk_io_v1 import save_json
from .native_greenhouse_pair import assert_same_camera
from .native848_multihost_complete_v1 import Inputs, host_config
from . import native848_multihost_generated_v1 as reservation_adapter
from . import native848_persistent_generated_9mm_v1 as authority
from . import native848_persistent_capture_v1 as producer
from . import native848_geometry_identity_v1 as geometry

coordinator = reservation_adapter.coordinator

def validate_gate(gate, portable, response, config, request_pin):
    require(gate['schema'] == 'greenhouse.generated_persistent_reservation_gate.v1'
        and portable['schema'] == 'greenhouse.generated_persistent_reservation_preparation.v1', 'Wrong generated reservation type')
    require(gate['source_request'] == portable['source_request'] == request_pin, 'Reserved batch request differs from actual owner')
    require(gate['coordinator_contacted'] is True and gate['all_candidate_views_reserved'] is True
        and gate['capture_request_eligible'] is True and gate['automatic_reclaim'] is False
        and gate['training_approved'] is False, 'Whole batch was not durably reserved')
    require(portable['host'] == config['host_id'] and portable['automatic_reclaim'] is False
        and portable['automatic_split'] is False and portable['mandatory_fullscene_camera_alias'] is True,
        'Host or generated identity contract differs')
    require(not portable['controls'] and gate['excluded_replay_controls'] == 0,
        'Experimental/replay-control completion is not supported by this adapter')
    rows = portable['rows']
    assignment = reservation_adapter.batch_assignment(rows, config, portable['worker_assignment']['worker_id'])
    require(assignment == gate['worker_assignment'] == portable['worker_assignment']
        and assignment['worker_owns_request'] is True, 'Wrong host worker owns this batch')
    require(reservation_adapter.validate_grants([r['identity'] for r in rows], response, config['host_id']), 'Partial or foreign grants')
    expected = {}
    aliases = set()
    for row, grant in zip(rows, response['reservations']):
        key = (row['segment_id'], row['sample_id'])
        require(key not in expected and row['task_id'] == grant['task_id'] == coordinator.task_id(row['identity']), 'Repeated or mismatched reserved identity')
        alias = row['identity']['scene_camera_key']
        require(isinstance(alias,str) and geometry.HEX.fullmatch(alias) and alias not in aliases, 'Repeated or absent scene-camera alias')
        aliases.add(alias)
        expected[key] = (row, grant)
    require(expected, 'No generated candidates reserved')
    return expected

def partition_segment(segment_id, records, frames, holds, expected):
    scheduled = [r['sample_id'] for r in records]
    require(len(set(scheduled)) == len(scheduled), 'Repeated source candidate IDs')
    frame_ids = [r['sample_id'] for r in frames]
    held_ids = [r['sample_id'] for r in holds]
    require(len(set(frame_ids)) == len(frame_ids) and len(set(held_ids)) == len(held_ids)
        and not set(frame_ids) & set(held_ids) and set(frame_ids) | set(held_ids) == set(scheduled), 'Incomplete, repeated, or foreign segment frames/holds')
    require({name for sid,name in expected if sid == segment_id} == set(scheduled), 'Segment schedule differs from reserved identities')
    require(frame_ids == [name for name in scheduled if name not in held_ids], 'Saved frames reordered from source schedule')
    for hold in holds:
        require((hold['reason'] == 'whole_robot_scene_collision' and hold['screen']['passed'] is False)
            or (hold['reason'] == '9mm_workspace' and hold['workspace']['result']['workspace_passed'] is False), 'Unsupported native hold')
    return {frame['sample_id']:frame for frame in frames}, {hold['sample_id']:hold for hold in holds}

def actual_identity(scene, foreground, geometry_sha, calibration):
    require(foreground['source_split'] == 'train' and calibration['resolution'] == [848,408]
        and calibration.get('crop_resize') is None, 'TRAIN native full-frame identity required')
    camera = geometry.rigid_rows(calibration['camera_to_world_usd_row_vectors'])
    plant = geometry.rigid_rows(foreground['plant_to_world_usd_row_vectors'])
    value = coordinator.identity(dict(geometry_sha256=geometry_sha, donor_source_family=foreground['source_family'], split='train',
        camera_to_plant_usd_row_vectors=(camera @ np.linalg.inv(plant)).tolist(),
        intrinsics=calibration['intrinsics'], resolution=[848,408]))
    value['scene_camera_key'] = geometry.scene_camera_key(scene,calibration)
    return value

def prepare(config_path, capture, reservation_path):
    inputs = Inputs()
    config_pin = inputs.local(config_path)
    config = host_config(config_pin['path'])
    capture = Path(capture).resolve()
    if capture.name != 'capture': capture = capture/'capture'
    gate_pin = inputs.local(reservation_path)
    gate = inputs.read(gate_pin)
    portable = inputs.read(gate['identities'])
    require(portable['config'] == config_pin, 'Reserved host configuration changed')
    request = inputs.read(gate['source_request'])
    require(request['schema'] == producer.REQUEST_SCHEMA, 'Only qualified ordinary persistent captures are supported')
    require(request['segments'] and all(set(s)=={'segment_id','source_request'} for s in request['segments']), 'Experimental/replay controls are unsupported')
    response = inputs.read(gate['coordinator_response'])
    expected = validate_gate(gate,portable,response,config,gate['source_request'])
    for module, key in ((reservation_adapter,'helper'),(geometry,'geometry_implementation'),(coordinator,'coordinator_implementation')):
        require(inputs.local(module.__file__) == portable[key], 'Reservation implementation drift')
    for module in (producer,authority): inputs.local(module.__file__)
    inputs.local(__file__)
    first = capture/'segments'/request['segments'][0]['segment_id']/'capture'
    # Authenticates every segment, global clocks, one real owner, child exit0,
    # scene swap chain, native population, and background closing evidence.
    auth = authority.authenticate_batch(first,inputs.read,inputs.local,producer)
    complete = auth['complete']
    require(complete['request'] == gate['source_request'], 'Native owner executed another reservation request')
    batch = inputs.read(complete['result'],within=capture)
    producer.validate_batch_registry(batch,request,capture,request_pin=complete['request'],native_identity=complete['native_identity'])
    inputs.bind(auth['source_bindings'])
    inputs.bind(portable['geometry_source_bindings'])
    inputs.read(portable['reference_context'])
    inputs.read(portable['reference_owner_complete'])
    source_segments = producer.check_request(request)
    planned_scenes = {r['segment_id']:r['scene_identity'] for r in portable['scene_identities']}
    require(len(planned_scenes) == len(request['segments']) and set(planned_scenes) == {s['segment_id'] for s in request['segments']}, 'Reserved scene identities incomplete')
    geometry_cache = {}
    actions = []
    for spec, entry, (source,candidates) in zip(request['segments'],batch['segments'],source_segments):
        sid = spec['segment_id']; directory = capture/'segments'/sid/'capture'
        segment = inputs.read(entry['result'],within=directory)
        context = inputs.read(inputs.local(directory/'context.json'))
        inputs.bind(context['source_bindings'])
        inputs.bind(candidates['source_bindings'])
        observations = [inputs.read(r,within=directory/'frames') for r in segment['frames']]
        actual, held = partition_segment(sid,candidates['records'],observations,segment['holds'],expected)
        census = context['full_scene_census']
        hashes = {}
        current_bindings = dict(portable['geometry_source_bindings'])
        for path,digest in context['source_bindings'].items():
            require(path not in current_bindings or current_bindings[path] == digest, 'Native/reserved geometry source conflict')
            current_bindings[path] = digest
        for slot in census['all_plant_roots']:
            manifest = inputs.local(slot['manifest_path'])
            require(manifest['sha256'] == slot['manifest_sha256'], 'Current scene manifest differs')
            key = (manifest['path'],manifest['sha256'])
            if key not in geometry_cache:
                current_bindings[manifest['path']] = manifest['sha256']
                geometry_cache[key] = geometry.geometry_identity(manifest['path'],current_bindings)
                inputs.bind(geometry_cache[key]['source_bindings'])
            hashes[slot['plant_root']] = geometry_cache[key]['geometry_sha256']
        scene = geometry.scene_identity(census,hashes)
        require(scene == planned_scenes[sid], 'Actual full144 geometry/placement differs from reserved scene')
        foregrounds = [s for s in census['all_plant_roots'] if s['source_geometry_modified']]
        require(len(foregrounds) == 1, 'Exactly one generated foreground required')
        foreground = foregrounds[0]
        require(inputs.local(candidates['manifest']['path']) == candidates['manifest']
            and foreground['manifest_sha256'] == candidates['manifest']['sha256'], 'Actual generated foreground source differs')
        for record in candidates['records']:
            name=record['sample_id']; row,grant=expected[(sid,name)]
            require(row['source_request'] == spec['source_request'], 'Reserved per-segment source request differs')
            calibration = actual[name]['calibration'] if name in actual else record['calibration']
            assert_same_camera(calibration,record['calibration'])
            identity = actual_identity(scene,foreground,hashes[foreground['plant_root']],calibration)
            require(identity == row['identity'] and coordinator.task_id(identity) == row['task_id'], 'Actual native geometry/camera differs from reserved identity')
            if name in actual:
                observation = actual[name]
                pose = observation['robot_snapshot']
                require(pose['joint_degrees'] == record['joint_degrees']
                    and all(np.allclose(pose[k],record[k],atol=1e-9,rtol=0) for k in ('robot_root_to_world_usd_row_vectors','camera_to_head_column_vectors')), 'Actual embodied pose differs from reserved candidate')
                rgb = observation['files']['rgb']; path = Path(rgb['path']).resolve()
                require(path == directory/'frames'/name/'rgb.png' and inputs.local(path) == rgb, 'Changed or foreign segment RGB')
                metrics = coordinator.rgb_metrics(path)
                require(metrics['encoded_rgb_sha256'] == rgb['sha256'], 'Native RGB changed while decoding')
                actions.append(dict(segment_id=sid,sample_id=name,reservation=grant,outcome='captured',metrics=metrics,rgb=rgb))
            else:
                actions.append(dict(segment_id=sid,sample_id=name,reservation=grant,outcome='no_frame',metrics=None,native_hold_reason=held[name]['reason']))
    require(len(actions) == len(expected), 'Not all reserved generated views accounted for')
    inputs.close()
    return dict(inputs=inputs,config=config,actions=actions,owner_complete=auth['public']['batch_owner_complete'],
        capture_result=complete['result'],reservation=gate_pin,geometry_count=len(geometry_cache))

def run(config,capture,reservation,output,client=None,verify_only=False):
    output=Path(output).resolve();require(not output.exists(),'Fresh completion output required')
    checked=prepare(config,capture,reservation);cfg=checked['config']
    results=[]
    if not verify_only:
        if client is None:
            client=coordinator.Client(os.environ.get(cfg['coordinator_url_env'],''),os.environ.get(cfg['coordinator_token_env'],''),cfg['host_id'])
        require(client.host==cfg['host_id'] and client.url==os.environ.get(cfg['coordinator_url_env'],client.url).rstrip('/'),'Wrong completion host/coordinator')
    output.mkdir(parents=True)
    if not verify_only:
        for index,action in enumerate(checked['actions']):
            response=client.finish(action['reservation'],action['metrics'],action['outcome'])
            expected_metrics=coordinator.metrics(action['metrics']) if action['outcome']=='captured' else None
            require(response['task_id']==action['reservation']['task_id'] and response['host']==cfg['host_id']
                and response['outcome']==action['outcome'] and response['metrics']==expected_metrics
                and response['training_approved'] is False and response['quality_and_visual_review_still_required'] is True,'Wrong completion response')
            require(response['state'] in ('complete','duplicate_hold') if action['outcome']=='captured' else response['state']=='no_frame','Unexpected completion state')
            item=dict(segment_id=action['segment_id'],sample_id=action['sample_id'],**response)
            save_json(output/f'completion_{index:04d}.json',item);results.append(item)
    checked['inputs'].close()
    result=dict(schema='greenhouse.multihost_persistent_generated_completion.v1',host=cfg['host_id'],
        owner_complete=checked['owner_complete'],capture_result=checked['capture_result'],reservation=checked['reservation'],
        coordinator_contacted=not verify_only,frames_captured=sum(a['outcome']=='captured' for a in checked['actions']),
        no_frame=sum(a['outcome']=='no_frame' for a in checked['actions']),duplicate_holds=sum(r['state']=='duplicate_hold' for r in results),
        completions=results,actual_whole_owner_exit_zero_verified=True,all_segments_and_source_geometry_authenticated=True,
        verified_geometry_manifests=checked['geometry_count'],replay_controls_completed=0,experimental_supported=False,
        source_bindings=checked['inputs'].bindings,training_approved=False,accepted_training_increment=0,
        quality_annotation_and_visual_review_still_required=True,native_launched=False)
    save_json(output/'result.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','capture','reservation','output'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    result=run(a.config,a.capture,a.reservation,a.output,verify_only=a.verify_only)
    print(json.dumps({k:result[k] for k in ('coordinator_contacted','frames_captured','no_frame','duplicate_holds','training_approved')}))

if __name__=='__main__':main()
