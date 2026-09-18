"""Prepare target-independent identities; optionally require an all-new coordinator grant.

This helper never launches a renderer. Partial/unknown reservations are held and
are never automatically released, reclaimed or converted into accepted samples.

Run as ``python -m sim_data.native848_multihost_generated_v1``. Reference context
paths and their immutable closure must be valid on the executing host. A copied
Windows receipt is not a Linux native-owner attestation; migrate that authority
through a separately qualified typed adapter, never by rewriting stored paths.
"""
from pathlib import Path
import argparse,json,os,sys
import numpy as np
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from .dataset_review import require
from .native848_bulk_io_v1 import save_json
from .native848_controlled_capture_v2 import pin,bound
from . import native848_geometry_identity_v1 as geometry
from . import native848_persistent_capture_v1 as baseline
from . import native848_persistent_experimental_capture_v1 as experimental
import dataset_capture_coordinator as coordinator

WORKER_COUNTS={'local5090':1,'thor1':4,'thor3':2}


def validate_config(config,worker_id):
    host=config['host_id']
    require(host in coordinator.ALLOCATION and host in WORKER_COUNTS,'Unknown allocated host')
    require(config['allocation_sha256']==coordinator.digest(coordinator.ALLOCATION)
        and config['allowed_foreground_families']==coordinator.ALLOCATION[host]
        and type(config['worker_count']) is int and config['worker_count']==WORKER_COUNTS[host],
        'Exact host allocation and fixed worker count required')
    require(type(worker_id) is int and 0<=worker_id<config['worker_count'],'Worker outside configured host range')
    return host


def batch_assignment(rows,config,worker_id):
    validate_config(config,worker_id)
    keys=[row['task_id'] for row in rows]
    require(keys and len(set(keys))==len(keys),'Finite unique candidate task IDs required')
    batch_id=coordinator.digest(sorted(keys))
    owner=int(batch_id,16)%config['worker_count']
    return dict(batch_id=batch_id,worker_count=config['worker_count'],worker_id=worker_id,
        owner_worker_id=owner,worker_owns_request=owner==worker_id,whole_request_single_worker=True)


def validate_grants(identities,response,host):
    require(response.get('schema')=='greenhouse.capture_reservations.v1','Unexpected coordinator response schema')
    rows=response['reservations'];require(len(rows)==len(identities),'Incomplete coordinator reply')
    seen=set()
    for identity,row in zip(identities,rows):
        key=coordinator.task_id(identity);require(row['task_id']==key and type(row['granted']) is bool,'Coordinator identity/order mismatch')
        if row['granted']:
            require(row['host']==host and type(row['claim_id']) is str and len(row['claim_id'])>=32,'Foreign or absent reservation authority')
            require(key not in seen,'Repeated granted task');seen.add(key)
    return all(row['granted'] for row in rows)

def authenticate_reference(path):
    path=Path(path).resolve();contextpin=pin(path);context=bound(contextpin);segment_result_pin=pin(path.parent/'result.json');result=bound(segment_result_pin)
    if 'context' in result:
        require(result['context']==contextpin,'Reference result/context mismatch')
    else:
        require(result['schema'] in ('greenhouse.controlled_capture_result.v2',baseline.SEGMENT_SCHEMA,experimental.SEGMENT_SCHEMA) and result['frames'],'Typed completed reference frames required')
        for receipt in result['frames']:
            observation=bound(dict(path=receipt['path'],sha256=receipt['sha256']))
            require(observation['context']==contextpin and observation['observation_id']==receipt['observation_id'],'Reference saved observation/context mismatch')
    trial=path.parent.parent
    if not (trial/'owner_complete.json').is_file():
        require(path.parent.parent.parent.name=='segments','Unknown persistent reference namespace');trial=path.parents[4]
    completepin=pin(trial/'owner_complete.json');complete=bound(completepin);closed=bound(complete['owned_exit']);require(closed['returncode']==0,'Reference capture did not close0')
    whole=bound(complete['result']);bound(complete['native_identity']);bound(complete['request'])
    require(whole['native_identity']==complete['native_identity'] and whole['request']==complete['request'],'Reference owner/result authority mismatch')
    if complete['result']!=segment_result_pin:
        require(any(s.get('result')==segment_result_pin for s in whole['segments']),'Reference segment not in completed batch')
    census=context['full_scene_census'];require(len(census['all_plant_roots'])==144 and census['unchanged_background_count']==143,'Complete143backgrounds plus1foreground required')
    modified=[r for r in census['all_plant_roots'] if r['source_geometry_modified']]
    require(len(modified)==1 and census['native_population_census_verified'] is True,'Reference native population not verified')
    return contextpin,context,completepin,modified[0]

def prepare(request_pin,reference_context,output,config_path,*,worker_id=0,reserve=False,client=None):
    output=Path(output).resolve();require(not output.exists(),'Create-only reservation output');output.mkdir()
    request=bound(request_pin);producer=experimental if request['schema']==experimental.REQUEST_SCHEMA else baseline
    require(request['schema']==producer.REQUEST_SCHEMA,'Typed persistent request required');sources=producer.check_request(request)
    configpin=pin(config_path);config=bound(configpin);host=validate_config(config,worker_id)
    contextpin,context,completepin,oldforeground=authenticate_reference(reference_context)
    census=context['full_scene_census'];base_hashes={};geometry_cache={};identity_bindings={};new_texture_bindings={}
    def geometry_for(manifest_path,bindings):
        spec=pin(manifest_path);key=(spec['path'],spec['sha256'])
        if key not in geometry_cache:
            own=dict(bindings);own[spec['path']]=spec['sha256'];geometry_cache[key]=geometry.geometry_identity(manifest_path,own,allow_new_texture_bindings=True)
        value=geometry_cache[key];identity_bindings.update(value['source_bindings']);new_texture_bindings.update(value['newly_measured_texture_bindings']);return value['geometry_sha256']
    for slot in census['all_plant_roots']:
        if slot['plant_root']==oldforeground['plant_root']:continue
        require(not slot['source_geometry_modified'],'More than one generated foreground')
        require(pin(slot['manifest_path'])['sha256']==slot['manifest_sha256'],'Reference background manifest changed')
        base_hashes[slot['plant_root']]=geometry_for(slot['manifest_path'],context['source_bindings'])
    rows=[];controls=[];seen={};scene_records=[]
    for spec,(source,candidates) in zip(request['segments'],sources):
        family=candidates['source_family'];require(family==oldforeground['source_family'] and family in config['allowed_foreground_families'],'Wrong reference donor or allocated host')
        require(oldforeground['source_split']=='train','Held-out donor prohibited')
        require(pin(candidates['manifest']['path'])==candidates['manifest'],'Generated manifest differs from authenticated candidate')
        hashes=dict(base_hashes);hashes[oldforeground['plant_root']]=geometry_for(candidates['manifest']['path'],candidates['source_bindings'])
        scene=geometry.scene_identity(census,hashes);scene_records.append(dict(segment_id=spec['segment_id'],scene_identity=scene))
        selected=spec.get('selected_sample_ids',[r['sample_id'] for r in candidates['records']]);lookup={r['sample_id']:r for r in candidates['records']}
        for name in selected:
            record=lookup[name];anchor=bound(record['anchor']);sceneplan=Path(anchor['source_collection_plan']).resolve()
            require(context['source_collection_plan']==dict(path=str(sceneplan),sha256=anchor['source_bindings'][str(sceneplan)]),'Reference scene plan differs')
            P=np.asarray(anchor['expected_original_world']['plant_to_world_usd_row_vectors']);require(np.allclose(P,oldforeground['plant_to_world_usd_row_vectors'],atol=1e-9,rtol=0),'Reference foreground placement differs')
            cal=record['calibration'];require(cal['resolution']==[848,408] and cal.get('crop_resize') is None,'Native fullframe required')
            identity=coordinator.identity(dict(geometry_sha256=hashes[oldforeground['plant_root']],donor_source_family=family,split='train',
                camera_to_plant_usd_row_vectors=(np.asarray(cal['camera_to_world_usd_row_vectors'])@np.linalg.inv(P)).tolist(),intrinsics=cal['intrinsics'],resolution=[848,408]))
            identity['scene_camera_key']=geometry.scene_camera_key(scene,cal)
            key=coordinator.task_id(identity);item=dict(segment_id=spec['segment_id'],sample_id=name,source_request=spec['source_request'],identity=identity,task_id=key)
            control=spec.get('purpose')=='paired_control_export_prohibited'
            if control:
                require(key in seen and spec['paired_control_reference']==dict(segment_id=seen[key]['segment_id'],sample_id=seen[key]['sample_id']),'Control does not replay a preceding exact identity')
                controls.append(dict(**item,reservation_excluded=True,accepted_increment=0,replay_of=seen[key]));continue
            require(key not in seen,'Repeated candidate geometry/camera');seen[key]=dict(segment_id=spec['segment_id'],sample_id=name);rows.append(item)
    require(len({r['identity']['scene_camera_key'] for r in rows})==len(rows),'Repeated scene/world-camera candidate')
    assignment=batch_assignment(rows,config,worker_id)
    value=dict(schema='greenhouse.generated_persistent_reservation_preparation.v1',source_request=request_pin,reference_context=contextpin,reference_owner_complete=completepin,config=configpin,
        rows=rows,controls=controls,scene_identities=scene_records,geometry_source_bindings=identity_bindings,newly_measured_texture_bindings=new_texture_bindings,
        texture_identity_provenance='Newly measured texture pins are current identity inputs, not historical capture evidence.',
        geometry_implementation=pin(geometry.__file__),coordinator_implementation=pin(coordinator.__file__),helper=pin(__file__),host=host,worker_assignment=assignment,
        mandatory_fullscene_camera_alias=True,all_novel_grants_required=True,automatic_reclaim=False,automatic_split=False,
        geometry_and_camera_identity_uses_no_target_hint=True,planned_scene_substitutes_only_authenticated_foreground=True,
        source_request_validated=True,coordinator_reserved=False,native_launch_authorized=False,native_launched=False,accepted_training_increment=0)
    for path,sha in identity_bindings.items():
        require(pin(path)['sha256']==sha,'Identity source changed before closing receipt')
    require(pin(request_pin['path'])==request_pin,'Source request changed during identity preparation')
    save_json(output/'identities.json',value)
    finish_reservation(value,config,output,reserve=reserve,client=client)
    return value

def finish_reservation(value,config,output,*,reserve=False,client=None):
    """Apply the whole-batch worker gate before any coordinator contact."""
    output=Path(output);assignment=value['worker_assignment'];rows=value['rows'];controls=value['controls']
    request_pin=value['source_request'];host=value['host']
    if not assignment['worker_owns_request']:
        save_json(output/'reservation_gate.json',dict(schema='greenhouse.generated_persistent_reservation_gate.v1',
            identities=pin(output/'identities.json'),source_request=request_pin,worker_assignment=assignment,
            all_candidate_views_reserved=False,capture_request_eligible=False,coordinator_contacted=False,
            reason='whole_request_owned_by_another_worker',excluded_replay_controls=len(controls),
            native_launched=False,automatic_reclaim=False,training_approved=False))
        return
    if reserve:
        if client is None:client=coordinator.Client(os.environ.get(config['coordinator_url_env'],''),os.environ.get(config['coordinator_token_env'],''),host)
        response=client.claim_many([r['identity'] for r in rows]);save_json(output/'coordinator_response_private.json',response)
        all_granted=validate_grants([r['identity'] for r in rows],response,host)
        require(pin(request_pin['path'])==request_pin,'Source request changed after reservation')
        receipt=dict(schema='greenhouse.generated_persistent_reservation_gate.v1',identities=pin(output/'identities.json'),source_request=request_pin,
            coordinator_response=pin(output/'coordinator_response_private.json'),worker_assignment=assignment,coordinator_contacted=True,all_candidate_views_reserved=all_granted,excluded_replay_controls=len(controls),
            capture_request_eligible=all_granted,native_launched=False,automatic_reclaim=False,training_approved=False)
        save_json(output/'reservation_gate.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--reference-context',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default=str(R/'configs/dataset_capture/local5090.json'));p.add_argument('--reserve',action='store_true');p.add_argument('--worker-id',type=int,default=0);a=p.parse_args()
    v=prepare(dict(path=str(Path(a.request).resolve()),sha256=a.request_sha256),a.reference_context,a.output,a.config,worker_id=a.worker_id,reserve=a.reserve)
    print(json.dumps(dict(identities=pin(Path(a.output)/'identities.json'),candidates=len(v['rows']),excluded_controls=len(v['controls']),reservation_attempted=a.reserve and v['worker_assignment']['worker_owns_request'],worker_assignment=v['worker_assignment'],native_launched=False)))
