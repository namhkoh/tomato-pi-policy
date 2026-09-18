"""Offline persistent segment consumer: one real batch owner, frozen generated numerics."""
from . import native848_generated_9mm_v3 as frozen

EVALUATION_SCHEMA = "greenhouse.native848_persistent_generated_evaluation.v1"
FROZEN_SHA256 = "2896c7f5a2adcf0ff62c738fbf385885cbb11c17532ddca5dd9d0ca1f00495bc"
from . import native848_generated_9mm_v2 as base
from .native848_generated_9mm_v2 import (
    Path, np, deepcopy, CatalogueIndex, read_json, require, verify_bindings, sha256,
    labels, original, joint, ambiguity, answer_contract, geometry_9mm, component_masks,
    build_inventory, _pin, _unique, validate_planned_geometry, evaluate_frame, assess_frame,
    CONTEXT_SCHEMA, CENSUS_SCHEMA, fingerprint,
)

def validate_collection_plan_provenance(context,candidates):
    """Keep original camera/scene authority separate from generation authority."""
    scene_pin=context['source_collection_plan']
    generator_pin=context['generator_source_collection_plan']
    require(scene_pin==context['full_scene_census']['source_collection_plan']
        and generator_pin==candidates['source_plan'],'Scene/generator collection plan roles differ')
    scene_path=_pin(scene_pin);generator_path=_pin(generator_pin)
    scene_plan=read_json(scene_path);generator_plan=read_json(generator_path)
    require(scene_plan['family_assignments']==generator_plan['family_assignments']
        and scene_plan['source_bindings_sha256']==generator_plan['source_bindings_sha256'],
        'Scene/generator source families or original assets differ')
    family=candidates['source_family']
    require(scene_plan['family_assignments'][family]==context['dataset_split']=='train',
        'Actual source donor split differs')
    pins={str(scene_path):scene_pin['sha256'],str(generator_path):generator_pin['sha256']}
    require(1<=len(candidates['records'])<=128,'Bounded original camera anchors required')
    for record in candidates['records']:
        anchor_path=_pin(record['anchor']);anchor=read_json(anchor_path)
        require(Path(anchor['source_collection_plan']).resolve()==scene_path
            and anchor['source_bindings'].get(str(scene_path))==scene_pin['sha256'],
            'Camera anchor must authenticate the actual scene collection plan')
        require(anchor['source_family']==record['source_family']==family
            and anchor['split']==context['dataset_split'],'Camera donor or split differs')
        require(candidates['source_bindings'].get(str(anchor_path))==record['anchor']['sha256'],
            'Camera anchor is not bound by the candidate preparation')
        pins[str(anchor_path)]=record['anchor']['sha256']
    return pins

def evaluate_segment(capturepath,output):
    """Evaluate one segment after its entire real persistent batch genuinely closes.

    The actual batch owner and ordered segment registry are authenticated directly;
    no segment owner is invented. Saved inputs remain unchanged and all results
    remain candidate evidence for subsequent actual visual review.
    """
    from collections import Counter
    import sys
    from fractions import Fraction
    from PIL import Image
    from .native848_bulk_io_v1 import save_json
    from .collection_plan import load_plan
    from .audit import audit_manifest
    from .native848_bulk_workspace_v1 import BulkWorkspace
    from .native848_pair_audit_v2 import verify_camera
    from .native_greenhouse_pair import assert_same_camera
    from . import native848_persistent_capture_v1 as producer
    from .native848_bulk_plan_v1 import check_profile
    capture=Path(capturepath).resolve();output=Path(output).resolve();trial=capture.parents[3]
    require(sha256(frozen.__file__)==FROZEN_SHA256,'Frozen numerical evaluator changed')
    require(capture.name=='capture' and capture.is_dir() and not output.exists()
        and not output.is_relative_to(trial) and not (capture/'failure.json').exists(),
        'Fresh separate output and successful controlled capture required')
    pins={}
    def bound(spec,*,within=None):
        reduced={k:spec[k] for k in ('path','sha256')};path=_pin(reduced)
        if within is not None:require(path.is_relative_to(within),'Artifact escaped its actual capture')
        pins[str(path)]=reduced['sha256'];return path
    def read(spec,*,within=None):return read_json(bound(spec,within=within))
    def local(path):
        path=Path(path).resolve();return dict(path=str(path),sha256=sha256(path))
    authority=authenticate_batch(capture,read,local,producer)
    complete=authority['complete'];result=authority['segment_result']
    request=authority['source_request'];candidates=authority['candidates']
    for path,h in authority['source_bindings'].items():
        require(path not in pins or pins[path]==h,'Conflicting batch implementation/source authority');pins[path]=h
    context=read(local(capture/'context.json'))
    require(context['schema']==CONTEXT_SCHEMA and context['training_approved'] is False
        and context['accepted_training_increment']==0,'Typed generated-only context required')
    require(context['profile']==candidates['profile'],'Source optical profile changed')
    for path,h in validate_collection_plan_provenance(context,candidates).items():
        require(path not in pins or pins[path]==h,'Conflicting camera/generator source authority');pins[path]=h
    verify_bindings(context['source_bindings']);pins.update(context['source_bindings'])
    census=read(local(capture/'census.json'))
    require(census==context['full_scene_census'],'Saved census differs from context')
    catalogue=read(context['catalogue'],within=capture);index=CatalogueIndex(catalogue)
    _,reports=load_plan(bound(context['source_collection_plan']))
    generated_report=read(local(capture/'generated_report.json'))
    fresh=audit_manifest(bound(candidates['manifest']))
    require(all(generated_report[k]==fresh[k] for k in ('plant_id','manifest_sha256','components','targets','status')),
        'Saved generated anatomy differs from manifest')
    reports.append(generated_report)
    inventory=build_inventory(context,reports,catalogue)
    for entry in inventory:
        for path,h in entry['geometry_evidence_source_bindings'].items():
            require(path not in pins or pins[path]==h,'Conflicting geometry evidence source');pins[path]=h
        if entry['physical_geometry_evidence'] is not None:bound(entry['physical_geometry_evidence'])
    for module in (sys.modules[__name__],frozen,base,labels,original,joint,ambiguity,answer_contract):
        path=str(Path(module.__file__).resolve());pins[path]=sha256(path)
    profile=read(candidates['profile']);check_profile(profile)
    require(context['renderer_settings']==profile['render_settings'],'Captured renderer differs from source optics profile')
    records=_unique(candidates['records'],'sample_id')
    frameids=[r['observation_id'] for r in result['frames']]
    require(len(frameids)==len(set(frameids)) and set(frameids)<=set(records),'Foreign/repeated frame IDs')
    expected_requests=(len(profile['warmup_steps']) if frameids else 0)+len(frameids)
    require(result['request_count']==len(result['requests'])==expected_requests,'Native warmup/production accounting differs')
    for n,e in enumerate(result['requests'],result['request_index_start']+1):
        require(e['request_index']==n and e['native_requests']==e['callback_count']==1
            and e['callback_sequence_after']==n and e['callback_sequence_before']==n-1
            and e['requested_subframes']==profile['request_subframes']
            and e['delta_time_seconds']==profile['delta_time_seconds'] and e['wait_for_render'] is True
            and abs(e['timeline_after_seconds']-e['timeline_before_seconds']-profile['delta_time_seconds'])<1e-7,
            'Actual same-callback/reset optical profile differs')
    output.mkdir();checker=BulkWorkspace();summaries=[];previous=None
    try:
        for receipt in result['frames']:
            observation=read(receipt,within=capture);name=observation['observation_id'];record=records[name]
            require(observation['schema']==producer.OBSERVATION_SCHEMA
                and observation['sample_id']==name and observation['training_approved'] is False
                and observation['accepted_training_increment']==0
                and observation['source_family']==candidates['source_family']
                and observation['morphology_id']==generated_report['plant_id']==result['morphology_id'],
                'Observation generated lineage differs')
            require(read(observation['context'],within=capture)==context,'Observation belongs to another context')
            assert_same_camera(observation['calibration'],record['calibration'])
            pose=observation['robot_snapshot']
            require(pose['joint_degrees']==record['joint_degrees']
                and np.allclose(pose['robot_root_to_world_usd_row_vectors'],record['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0)
                and pose['visual_bound_screen']['passed'] is True,'Actual body/collision screen differs')
            rgbpath=bound(observation['files']['rgb'],within=capture)
            with Image.open(rgbpath) as image:
                require(image.mode=='RGB' and image.size==(848,408),'Native unscaled RGB required');rgb=np.asarray(image).copy()
            with np.load(bound(observation['files']['buffers'],within=capture),allow_pickle=False) as data:
                require(set(data.files)=={'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},'Exact native buffers required')
                depth,ids,valid,alpha=[data[k].copy() for k in ('depth_m','renderer_instance_id','depth_valid','rgba_alpha')]
            mapping={int(k):v for k,v in read(observation['mapping'],within=capture)['renderer_id_to_prim'].items()}
            payload=deepcopy(observation['native_payload_header'])
            payload.update(rgb=np.dstack((rgb,alpha)),distance_to_image_plane=depth,
                           instance_id_segmentation=dict(data=ids,info=dict(idToLabels=mapping)))
            sync=observation['synchronization']
            _,_,native_valid,reference,token,native_ids,native_mapping=producer.primitives.native_freshness(
                payload,observation['calibration'],sync['static_before'],sync['static_after'],sync['callback_sequence'])
            require(token==sync['freshness'] and reference==sync['reference_time']
                and np.array_equal(native_valid,valid) and np.array_equal(native_ids,ids) and native_mapping==mapping,
                'Saved native callback buffers/calibration/freshness differ')
            require(observation['request_evidence']==result['requests'][sync['request_index']-result['request_index_start']-1]
                and sync['request_index']==receipt['request_index']
                and observation['request_evidence']['settings']==context['renderer_settings'],'Actual frame request differs')
            if previous is not None:
                require(Fraction(*reference)>=Fraction(*previous['reference_time'])
                    and sync['callback_sequence']>previous['callback_sequence']
                    and all(token[k]!=previous[k] for k in ('camera_sha256','rgb_sha256','depth_sha256')),
                    'Repeated camera or stale native RGB-D')
            previous=dict(token,reference_time=reference)
            matrix=next(r['plant_to_world_usd_row_vectors'] for r in census['all_plant_roots']
                        if r['plant_root']==census['foreground_root'])
            expected=geometry_9mm(generated_report,candidates['target_component_id'],matrix,observation['calibration'])
            require(observation['geometry']==expected,'Actual saved generated9mm geometry changed')
            validate_planned_geometry(expected,record['pilot_9mm_geometry'])
            metadata=dict(schema='greenhouse.generated_native9mm_sample.v1',sample_id=name,
                calibration=observation['calibration'],robot_snapshot=pose,rendered_camera_params=observation['rendered_camera_params'],
                synchronization=sync,geometry_screen=pose['visual_bound_screen'],source_family=candidates['source_family'],
                geometry_source_id=generated_report['plant_id'],split='train',training_approved=False,accepted_training_increment=0)
            verify_camera(metadata)
            components,_,owners=component_masks(ids,mapping,index)
            unknown=[];pixel_classes=Counter()
            for rid in np.unique(ids):
                rid=int(rid);path=mapping.get(rid,'');mask=ids==rid;count=int(mask.sum());owner=owners[rid]
                if owner:category='authenticated_component_'+owner['organ_type']
                elif rid==0 and not path and np.all(~valid[mask]) and np.all(np.isposinf(depth[mask])):category='empty_background'
                elif path.startswith(('/World/Environment/GreenHouse/','/World/Gutters/','/World/RBY1/','/World/Floor','/World/Ground')):category='nonplant_greenhouse_or_robot'
                else:category='unknown_identity';unknown.append(dict(renderer_id=rid,prim_path=path,pixels=count))
                pixel_classes[category]+=count
            annotation=evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
            for target in annotation['targets']:
                proof=target.get('attachment_joint_ownership')
                if proof is not None:require(all(pins.get(p)==h for p,h in proof['source_bindings'].items()),'Unbound current geometry joint proof')
            folder=output/name;folder.mkdir()
            coverage=dict(schema='greenhouse.native848_generated_coverage_audit.v1',frame_id=name,
                observation=dict(path=str(Path(receipt['path']).resolve()),sha256=receipt['sha256']),
                context=observation['context'],capture_split='train',plant_instance_count=144,
                original_background_instances=143,generated_foreground_instances=1,
                complete_all_eligible_ground_truth=not unknown and not annotation['target_census']['unknown_target_ids'],
                blocking_unknowns=unknown+annotation['target_census']['unknown_target_ids'],visible_cross_split_petiole_targets=[],
                pixel_categories=dict(pixel_classes),catalogue_petiole_count=len(inventory),source_bindings=dict(pins),
                new_independent_source_family=False,training_approved=False,accepted_training_increment=0)
            save_json(folder/'coverage.json',coverage)
            annotation['target_census']['full_scene_coverage']=local(folder/'coverage.json')
            annotation['target_census']['complete']=coverage['complete_all_eligible_ground_truth'] and annotation['target_census']['catalogue_complete']
            annotation['frame_blocked_by_unverified_full_scene_coverage']=not coverage['complete_all_eligible_ground_truth']
            annotation['observation']=coverage['observation'];annotation['source_bindings']=dict(pins)
            assessment=assess_frame(metadata,annotation,checker)
            save_json(folder/'sample.json',metadata);save_json(folder/'annotation.json',annotation);save_json(folder/'ambiguity.json',assessment)
            summaries.append(dict(sample_id=name,metadata=local(folder/'sample.json'),annotation=local(folder/'annotation.json'),
                coverage=local(folder/'coverage.json'),ambiguity=local(folder/'ambiguity.json'),
                strict_candidate_ids=annotation['target_census']['candidate_target_ids'],
                candidate_source_target_ids=[r['source_target_id'] for r in annotation['targets'] if r['status']=='candidate_pending_visual_review'],
                candidate_geometry_target_ids=[r['geometry_target_id'] for r in annotation['targets'] if r['status']=='candidate_pending_visual_review'],
                generated_foreground_is_sole_strict_candidate=(len(annotation['target_census']['candidate_target_ids'])==1
                    and all(r['geometry_source_id']==generated_report['plant_id'] for r in annotation['targets'] if r['status']=='candidate_pending_visual_review')),
                unknown_target_ids=annotation['target_census']['unknown_target_ids'],
                single_answer_candidate=assessment['single_answer_unambiguous'] and assessment['assessment_complete'],
                actual_visual_review=False,training_approved=False,accepted_training_increment=0))
        stats=checker.finish();verify_bindings(pins)
        value=dict(schema=EVALUATION_SCHEMA,source_capture=str(capture),
            batch_owner_complete=local(trial/'owner_complete.json'),batch_capture_result=complete['result'],
            segment_result=local(capture/'result.json'),persistent_batch_authority=authority['public'],
            frames_evaluated=len(summaries),
            records=summaries,workspace_stats=stats,source_bindings=pins,native_control_performed=False,
            actual_visual_review=False,training_approved=False,accepted_training_increment=0)
        save_json(output/'result.json',value);return value
    finally:checker.finish()


def validate_registry(request_pin, identity_pin, batch, scheduled, registered, values, contexts, observations, sources, profiles, producer):
    """Pure ordered-registry/clock/swap validation; caller authenticated every pin."""
    from fractions import Fraction
    require(batch['schema']==producer.RESULT_SCHEMA and batch['request']==request_pin
        and batch['native_identity']==identity_pin and batch['training_approved'] is False
        and batch['accepted_training_increment']==0
        and batch['state']=='captured_pending_complete_annotation_and_visual_review'
        and batch['same_stage_product_writer_for_all_segments'] is True,
        'Typed unapproved persistent batch required')
    require(batch['segments']==registered and len(scheduled)==len(registered)==len(values)==len(contexts)==len(observations)==len(sources)==len(profiles),
        'Complete ordered batch registry required')
    cursor=0;callback=0;frames=0;holds=0;previous_ref=None;previous_timeline=None;previous_frame=None
    prior_geometry=None;compatibility=None;all_ids=set();background_hash=None
    for i,(spec,entry,value,context,obslist,source,profile) in enumerate(zip(scheduled,registered,values,contexts,observations,sources,profiles)):
        sid=f's{i:03d}'
        require(spec['segment_id']==entry['segment_id']==value['segment_id']==sid
            and value['schema']==producer.SEGMENT_SCHEMA
            and value['source_request']==spec['source_request']
            and value['batch_request']==request_pin and value['native_identity']==identity_pin,
            'Segment/source/native authority differs')
        require(value['state']=='captured_pending_complete_annotation_and_visual_review'
            and value['training_approved'] is False and value['accepted_training_increment']==0
            and value['original_143_backgrounds_preserved'] is True,
            'Segment capture-only population contract differs')
        require(len(value['frames'])==entry['frames']==len(obslist)<=source['max_frames']
            and len(value['holds'])==entry['holds'], 'Segment frame/hold registry differs')
        require(value['request_index_start']==cursor and value['callback_sequence_start']==callback,
            'Segment global clock offset differs')
        steps=(list(profile['warmup_steps']) if obslist else [])+[profile['request_subframes']]*len(obslist)
        require(value['request_count']==len(value['requests'])==len(steps), 'Segment request count differs')
        for e,subframes in zip(value['requests'],steps):
            cursor+=1;callback+=1
            require(e['request_index']==cursor and e['callback_sequence_before']==callback-1
                and e['callback_sequence_after']==callback and e['native_requests']==e['callback_count']==1
                and e['requested_subframes']==subframes and e['delta_time_seconds']==profile['delta_time_seconds']
                and e['wait_for_render'] is True
                and abs(e['timeline_after_seconds']-e['timeline_before_seconds']-profile['delta_time_seconds'])<1e-7,
                'Persistent request/callback/profile accounting differs')
            ref=Fraction(*e['reference_time'])
            require(previous_ref is None or (ref>=previous_ref and e['previous_reference_time'] is not None
                and Fraction(*e['previous_reference_time'])==previous_ref), 'Global native callback clock regressed/reset')
            require(previous_timeline is None or e['timeline_before_seconds']>=previous_timeline-1e-7,
                'Global scheduler clock regressed')
            previous_ref=ref;previous_timeline=e['timeline_after_seconds']
        require(value['request_index_end']==cursor and value['callback_sequence_end']==callback,
            'Segment ending clock differs')
        expected_authority=dict(batch_request=request_pin,source_request=spec['source_request'],segment_id=sid,native_identity=identity_pin)
        require(context['schema']==CONTEXT_SCHEMA and context['persistent_batch_authority']==expected_authority,
            'Context belongs to another persistent segment')
        swap=context['persistent_foreground_swap'];census=context['full_scene_census']
        require(context['native_population_census_verified'] is True
            and context['original_143_backgrounds_preserved'] is True
            and census['unchanged_background_count']==143 and census['active_counts']['component_plants']==144,
            'Segment full background population differs')
        if background_hash is None:background_hash=census['background_invariance_sha256']
        require(census['background_invariance_sha256']==background_hash,'Background identity changed between segments')
        require(swap['schema']=='greenhouse.persistent_foreground_transaction.v1' and swap['sequence']==i+1
            and swap['source_profile']==context['profile'] and swap['unchanged_background_count']==143
            and swap['original_foreground_restored_before_adapter'] is True
            and swap['outside_foreground_opinions_preserved'] is True
            and swap['fresh_collision_catalogue_static_and_native_evidence_required'] is True
            and swap['native_capture_validated'] is False and swap['training_approved'] is False
            and swap['accepted_training_increment']==0 and swap['current_geometry']==census['foreground_identity']
            and swap['current_geometry']['morphology_id']==value['morphology_id'],
            'Fresh foreground transaction evidence differs')
        if i==0:
            require(swap['previous_geometry']==dict(geometry_source_id=context['geometry_sources'][value['morphology_id']]['donor_source_family']),
                    'Initial foreground donor differs')
            compatibility=swap['compatibility_sha256']
        else:
            require(swap['previous_geometry']==prior_geometry and swap['compatibility_sha256']==compatibility,
                    'Foreground swap chain differs')
        prior_geometry=swap['current_geometry']
        for j,(receipt,obs) in enumerate(zip(value['frames'],obslist)):
            sync=obs['synchronization'];expected_index=value['request_index_start']+len(profile['warmup_steps'])+j+1
            require(obs['schema']==producer.OBSERVATION_SCHEMA
                and obs['segment_id']==sid and obs['batch_request']==request_pin
                and obs['source_request']==spec['source_request'] and obs['morphology_id']==value['morphology_id']
                and obs['sample_id']==obs['observation_id']==receipt['observation_id']
                and sync['request_index']==receipt['request_index']==expected_index
                and sync['callback_sequence']==expected_index
                and obs['request_evidence']==value['requests'][expected_index-value['request_index_start']-1],
                'Observation segment/ID/global request authority differs')
            require((sid,obs['observation_id']) not in all_ids,'Duplicate observation ID inside segment')
            all_ids.add((sid,obs['observation_id']));token=sync['freshness']
            require(token['callback_sequence']==sync['callback_sequence']
                and token['reference_time']==sync['reference_time']==obs['request_evidence']['reference_time'],
                'Observation callback token authority differs')
            if previous_frame is not None:
                require(Fraction(*sync['reference_time'])>=Fraction(*previous_frame['reference_time'])
                    and token['callback_sequence']>previous_frame['callback_sequence']
                    and all(token[k]!=previous_frame[k] for k in ('rgb_sha256','depth_sha256','instance_sha256')),
                    'Stale buffers or regressed global frame clock')
            previous_frame=token
        frames+=len(obslist);holds+=len(value['holds'])
    require(batch['frames']==frames and batch['holds']==holds and batch['request_count']==cursor
        and batch['callback_count']==callback,'Batch aggregate differs from actual segments')
    closure=batch['background_closure']
    require(closure['schema']=='greenhouse.persistent_foreground_transaction.v1'
        and closure['sequence']==len(scheduled) and closure['compatibility_sha256']==compatibility
        and closure['unchanged_background_count']==143 and closure['protected_source_union_rehashed'] is True
        and type(closure['source_file_count']) is int and closure['source_file_count']>0,
        'Final persistent background/source union closure differs')


def authenticate_batch(capture,read,local,producer):
    """Bind the one real owner to a segment; never synthesize child completion."""
    batch_capture=capture.parents[2];trial=batch_capture.parent;sid=capture.parent.name
    require(capture.name=='capture' and capture.parent.parent.name=='segments'
        and batch_capture.name=='capture' and not (capture.parent/'owner_complete.json').exists()
        and not (batch_capture/'failure.json').exists(),'Canonical actual segment path required')
    complete=read(local(trial/'owner_complete.json'))
    require(complete['training_approved'] is False and complete['accepted_training_increment']==0
        and Path(complete['owned_exit']['path']).resolve()==trial/'owned_exit.json'
        and Path(complete['owner_started']['path']).resolve()==trial/'owner_started.json'
        and Path(complete['result']['path']).resolve()==batch_capture/'result.json'
        and Path(complete['native_identity']['path']).resolve()==batch_capture/'native_identity.json',
        'Completion belongs to another actual batch lifecycle')
    exited=read(complete['owned_exit'],within=trial);started=read(complete['owner_started'],within=trial)
    worker=read(exited['owned_worker'],within=trial);batch=read(complete['result'],within=batch_capture)
    require(exited['returncode']==0 and Path(exited['owned_worker']['path']).resolve()==trial/'owned_worker.json',
        'Genuine actual batch native exit0 required')
    require(complete['request']==started['request']==batch['request']
        and complete['native_identity']==batch['native_identity']
        and complete['frames']==batch['frames'] and complete['segments']==batch['segments'],
        'Batch owner completion registry differs')
    request=read(complete['request']);sources=producer.check_request(request)
    identity=read(complete['native_identity'],within=batch_capture);command=started['command'];gate=producer.production.identity_gate
    require(identity['expected_command']==worker['command']==command
        and identity['command']['ProcessId']==worker['launcher_pid']
        and identity['owner']['ProcessId']==worker['owner_pid']
        and gate.same_identity(identity['owner'],started['resources']['raw_owner_classification']['metadata']),
        'Actual batch native owner identity differs')
    gate.command_child(identity['command'],identity['owner'],command);gate.native_child(identity['native'],identity['command'],command[1:])
    require(command[1:6]==['-B','-u','-m',producer.MODULE,'--capture'],'Wrong persistent native module')
    def argument(flag):
        require(command.count(flag)==1 and command.index(flag)+1<len(command),'Unique batch worker argument required')
        return command[command.index(flag)+1]
    require(Path(argument('--output')).resolve()==batch_capture
        and Path(argument('--request')).resolve()==Path(started['request']['path']).resolve()
        and argument('--request-sha256')==started['request']['sha256'],'Actual batch command request/output differs')
    values=[];contexts=[];observations=[];profiles=[];bindings=dict(request['implementation_bindings']);chosen=None
    for i,(spec,entry,(source,candidates)) in enumerate(zip(request['segments'],batch['segments'],sources)):
        target=batch_capture/'segments'/spec['segment_id']/'capture'
        require(Path(entry['result']['path']).resolve()==target/'result.json'
            and not (target/'failure.json').exists(),'Segment result path escaped the batch registry')
        value=read(entry['result'],within=target);context=read(local(target/'context.json'));obslist=[]
        candidate_ids={r['sample_id'] for r in candidates['records']}
        require(len(candidate_ids)==len(candidates['records']), 'Repeated source candidate ID')
        for receipt in value['frames']:
            obs=read(receipt,within=target)
            require(obs['sample_id'] in candidate_ids and obs['source_family']==candidates['source_family'],
                'Observation not in its exact source candidate schedule')
            require(Path(obs['context']['path']).resolve()==target/'context.json'
                and read(obs['context'],within=target)==context,'Observation context differs from registered segment')
            obslist.append(obs)
        values.append(value);contexts.append(context);observations.append(obslist);profiles.append(read(candidates['profile']))
        require(context['profile']==candidates['profile'] and context['persistent_foreground_swap']['qualification']==candidates['qualification'],
            'Segment source geometry/profile differs')
        # Initial donor is authenticated by the exact source request, not a string heuristic.
        if i==0:require(context['persistent_foreground_swap']['previous_geometry']==dict(geometry_source_id=candidates['source_family']),
                        'Initial original donor differs')
        for p,h in {**source['implementation_bindings'],**candidates['source_bindings']}.items():
            require(p not in bindings or bindings[p]==h,'Conflicting segment source pin');bindings[p]=h
        if spec['segment_id']==sid:
            require(target==capture,'Selected segment path differs');chosen=(i,source,candidates,value)
    validate_registry(complete['request'],complete['native_identity'],batch,request['segments'],batch['segments'],values,contexts,observations,[x[0] for x in sources],profiles,producer)
    require(chosen is not None,'Requested segment is absent from completed batch')
    i,source,candidates,value=chosen
    return dict(complete=complete,segment_result=value,source_request=source,candidates=candidates,source_bindings=bindings,
        public=dict(batch_owner_complete=local(trial/'owner_complete.json'),batch_result=complete['result'],batch_request=complete['request'],
            native_identity=complete['native_identity'],segment_id=sid,segment_index=i,segment_result=local(capture/'result.json'),
            source_request=request['segments'][i]['source_request'],segment_is_not_separate_native_owner=True,
            all_batch_segments_and_global_clocks_authenticated=True))
