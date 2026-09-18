"""Experimental B7/C1/D1/B1 in one native process: candidate views plus one export-prohibited control.

Every morphology has its own current census, collision/static baseline and frame
receipts. The one real native owner closes the whole batch; no segment pretends
to be a separate Isaac process. Original numerical and optical rules are reused.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import argparse,os,sys,time,traceback
import numpy as np
from .capture_contract import fingerprint,jsonable
from .dataset_review import read_json,require,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import FrameSink,save_json
from . import native848_controlled_capture_v2 as baseline
from . import native848_bulk_worker_v1 as primitives
from . import native848_fully_labeled_worker_v3 as production
from . import native848_persistent_experimental_plan_v1 as diagnostic
MODULE='sim_data.native848_persistent_experimental_capture_v1'
REQUEST_SCHEMA='greenhouse.native848_persistent_experimental_capture_request.v1'
RESULT_SCHEMA='greenhouse.native848_persistent_experimental_batch_capture.v1'
SEGMENT_SCHEMA='greenhouse.native848_persistent_experimental_segment_capture.v1'
OBSERVATION_SCHEMA='greenhouse.native848_persistent_experimental_native_observation.v1'
FLAGS=dict(baseline.FLAGS,experimental_profile=True)
pin=baseline.pin
bound=baseline.bound
validate_pose=baseline.validate_pose


def implementation_bindings():
    names=('native848_persistent_foreground_v1.py','native848_controlled_capture_v2.py','native848_persistent_capture_v1.py','native848_persistent_experimental_plan_v1.py')
    require(sha256(Path(__file__).with_name('native848_persistent_capture_v1.py'))=='785ef36c61f58a5c3634072ad13181144dd6ed32db2cd1e08dc53c1b26ba0ee7','Frozen persistent source changed')
    return {**baseline.implementation_bindings(),str(Path(__file__).resolve()):sha256(__file__),
        **{str(Path(__file__).with_name(n).resolve()):sha256(Path(__file__).with_name(n)) for n in names}}


def compatible_key(candidates):
    anchors=[bound(r['anchor']) for r in candidates['records']]
    require(all(a['original_variant']==anchors[0]['original_variant'] for a in anchors),'Mixed foreground placements')
    a=anchors[0];scene=Path(a['source_collection_plan']).resolve()
    return dict(source_family=candidates['source_family'],source_plan=candidates['source_plan'],
        scene_plan=dict(path=str(scene),sha256=a['source_bindings'][str(scene)]),
        original_variant=a['original_variant'],profile=candidates['profile'])


def check_request(request):
    require(request['schema']==REQUEST_SCHEMA and request['implementation_bindings']==implementation_bindings(),
        'Exact diagnostic persistent implementation required')
    verify_bindings(request['implementation_bindings'])
    require(len(request['segments'])==4,'Exactly three candidates and one paired control required')
    cache={};segments=[]
    for spec in request['segments']:
        key=(spec['source_request']['path'],spec['source_request']['sha256'])
        if key not in cache:
            source=bound(spec['source_request']);cache[key]=(source,baseline.check_request(source))
        segments.append(cache[key])
    diagnostic.validate_schedule(request,segments)
    keys=[compatible_key(c) for _,c in segments]
    require(all(k==keys[0] for k in keys) and request['compatibility']==keys[0],
        'Same donor/slot/scene/source optical profile required')
    for spec,(_,c) in zip(request['segments'],segments):diagnostic.actual_profile(spec,c['profile'],bound(c['profile']))
    return segments


def make_request(source_b,source_c,source_d):
    pins=[source_b,source_c,source_d];sources=[bound(x) for x in pins]
    candidates=[baseline.check_request(x) for x in sources]
    specs=[]
    for i in range(4):
        k=i if i<3 else 0;ids=[r['sample_id'] for r in candidates[k]['records']]
        if i==3:ids=ids[:1]
        specs.append(dict(segment_id=f's{i:03d}',source_request=pins[k],selected_sample_ids=ids,
            warmup_request_count=diagnostic.COUNTS[i],purpose=diagnostic.CANDIDATE if i<3 else diagnostic.CONTROL,
            paired_control_reference=None if i<3 else dict(segment_id='s000',sample_id=ids[0])))
    total=sum(len(s['selected_sample_ids']) for s in specs)
    value=dict(schema=REQUEST_SCHEMA,segments=specs,max_frames=total,potential_candidate_frames=total-1,
        experimental_profile_policy=diagnostic.make_profile(candidates[0]['profile']),
        compatibility=compatible_key(candidates[0]),implementation_bindings=implementation_bindings(),**FLAGS)
    check_request(value);return value


def validate_batch_registry(result, request, capture_root, *, request_pin, native_identity):
    """Check exact ordered segment files and continuous writer counters at closure."""
    capture_root=Path(capture_root).resolve()
    require(result['schema']==RESULT_SCHEMA and result['request']==request_pin
        and result['native_identity']==native_identity and result['training_approved'] is False
        and result['same_stage_product_writer_for_all_segments'] is True
        and result['experimental_profile'] is True and result['accepted_training_increment']==0
        and result['experimental_profile_policy']==request['experimental_profile_policy'],
        'Invalid persistent batch authority')
    require(len(result['segments'])==len(request['segments']), 'Incomplete segment registry')
    frames=holds=request_end=callback_end=0
    for scheduled,registered in zip(request['segments'],result['segments']):
        sid=scheduled['segment_id']
        expected=capture_root/'segments'/sid/'capture'/'result.json'
        require(registered['segment_id']==sid and registered['result']==pin(expected),
            'Segment registry path or order changed')
        value=bound(registered['result']);source=bound(scheduled['source_request'])
        require(value['schema']==SEGMENT_SCHEMA and value['segment_id']==sid
            and value['source_request']==scheduled['source_request']
            and value['batch_request']==request_pin and value['native_identity']==native_identity
            and value['training_approved'] is False and value['original_143_backgrounds_preserved'] is True,
            'Segment authority differs')
        require(value['request_index_start']==request_end and value['callback_sequence_start']==callback_end,
            'Discontinuous segment writer counters')
        context=bound(pin(expected.parent/'context.json'))
        candidate=bound(source['candidates'])
        actual=diagnostic.actual_profile(scheduled,candidate['profile'],bound(candidate['profile']))
        diagnostic.validate_evidence(scheduled,actual,context,value,[bound(f) for f in value['frames']])
        events=value['requests'];count=len(events)
        steps=(actual['warmup_steps'] if value['frames'] else [])+[8]*len(value['frames'])
        require(count==len(steps),'Diagnostic actual warmup/request count differs')
        require(value['request_count']==count and value['request_index_end']==request_end+count
            and value['callback_sequence_end']==callback_end+count,'Segment counter total differs')
        for i,event in enumerate(events):
            require(event['request_index']==request_end+i+1
                and event['callback_sequence_before']==callback_end+i
                and event['callback_sequence_after']==callback_end+i+1 and event['callback_count']==1
                and event['native_requests']==1 and event['requested_subframes']==steps[i]
                and event['delta_time_seconds']==actual['delta_time_seconds'] and event['wait_for_render'] is True,
                'Segment request/callback sequence differs')
        require(0<=len(value['frames'])<=len(scheduled['selected_sample_ids'])
            and registered['frames']==len(value['frames']) and registered['holds']==len(value['holds']),
            'Segment frame totals differ')
        frame_ids=set();frame_paths=set();frame_requests=set()
        for frame in value['frames']:
            path=Path(frame['path']).resolve()
            require(path.is_relative_to(expected.parent/'frames') and path not in frame_paths,
                'Duplicate frame or frame outside its capture')
            observation=bound(frame);name=observation['observation_id']
            require(name in scheduled['selected_sample_ids'],'Wrong experimental camera')
            require(name not in frame_ids and frame['observation_id']==name==observation['sample_id']
                and observation['schema']==OBSERVATION_SCHEMA and observation['segment_id']==sid
                and observation['batch_request']==request_pin
                and observation['source_request']==scheduled['source_request']
                and observation['morphology_id']==value['morphology_id']
                and observation['training_approved'] is False,
                'Frame identity or segment authority differs')
            require(observation['context']==pin(expected.parent/'context.json'),
                'Frame belongs to another segment context')
            context=bound(observation['context']);authority=context['persistent_batch_authority']
            require(context['profile_role']=='source_optical_settings_and_baseline_warmup_reference_only'
                and context['actual_experimental_profile']==value['actual_experimental_profile']
                and len(value['actual_experimental_profile']['warmup_steps'])==scheduled['warmup_request_count'],
                'Explicit diagnostic warmup evidence differs')
            require(authority==dict(batch_request=request_pin,source_request=scheduled['source_request'],
                segment_id=sid,native_identity=native_identity),'Frame native context authority differs')
            sync=observation['synchronization'];index=sync['request_index']
            require(type(index) is int and request_end<index<=value['request_index_end']
                and index not in frame_requests,'Duplicate or foreign frame request')
            event=events[index-request_end-1]
            require(observation['request_evidence']==event
                and sync['callback_sequence']==event['callback_sequence_after'],
                'Frame does not correspond to its actual callback')
            frame_ids.add(name);frame_paths.add(path);frame_requests.add(index)
        frames+=len(value['frames']);holds+=len(value['holds'])
        request_end=value['request_index_end'];callback_end=value['callback_sequence_end']
    require(result['frames']==frames<=request['max_frames'] and result['holds']==holds
        and result['request_count']==request_end and result['callback_count']==callback_end,
        'Batch aggregate differs from actual segments')
    closure=result['background_closure']
    require(closure['sequence']==len(request['segments']) and closure['unchanged_background_count']==143
        and closure['protected_source_union_rehashed'] is True,
        'Full background closing verification missing')
    return dict(frames=frames,holds=holds,requests=request_end,callbacks=callback_end)


def capture_segment(app, request, candidates, output, *, scheduled, source_request_pin, batch_request_pin, segment_id, session, writer, batch_native_identity, previous, anchor_cache):
    import omni.usd
    import omni.replicator.core as rep
    from .native848_pilot_plan_v1 import raw_anchor
    from .native848_original_direct_worker_v2 import apply_cached_pose
    from .native848_morphology_diagnostic_v1 import diagnostic_catalogue
    from .native848_bulk_plan_v1 import check_profile
    from .native848_bulk_workspace_v1 import BulkWorkspace
    from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
    from .static_guard import StaticSceneMonitor
    from .capture_pilot import source_hashes
    from .capture_scene import calibration
    from .capture_sensor import calibration_for_native_resolution
    from .native_greenhouse_pair import assert_same_camera
    records=[r for r in candidates['records'] if r['sample_id'] in scheduled['selected_sample_ids']]
    require([r['sample_id'] for r in records]==scheduled['selected_sample_ids'],'Exact selected experimental cameras required')
    anchors=[bound(r['anchor']) for r in records]
    for record,anchor in zip(records,anchors):
        stamp=(anchor['source_capture'],anchor['source_sample'],fingerprint(anchor))
        if stamp not in anchor_cache:
            require(raw_anchor(anchor['source_capture'],anchor['source_sample'])==anchor,'Original robot anchor changed')
            anchor_cache.add(stamp)
        validate_pose(record,anchor)
        require(anchor['original_variant']==anchors[0]['original_variant'], 'All views must share foreground placement')
    profile=bound(candidates['profile']);check_profile(profile)
    actual=diagnostic.actual_profile(scheduled,candidates['profile'],profile)
    started=time.perf_counter();timings={}
    scene=session.scene
    stage,robot,settings=scene['stage'],scene['robot'],scene['settings']
    population=session.replace(candidates,anchor=anchors[0],profile_pin=candidates['profile'])
    timings['foreground_swap_seconds']=time.perf_counter()-started
    tick=time.perf_counter()
    catalogue=diagnostic_catalogue(stage,population,scene['reports'])
    timings['catalogue_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    census=deepcopy(population['scene_policy_evidence'])
    require(len(census['all_plant_roots'])==144 and census['unchanged_background_count']==143
            and len(catalogue)==population['counts']['components'], 'Native full population differs')
    census['native_population_census_verified']=True
    census['deterministic_census_sha256']=fingerprint({k:v for k,v in census.items() if k!='deterministic_census_sha256'})
    save_json(output/'census.json',census);save_json(output/'catalogue.json',catalogue)
    save_json(output/'generated_report.json',population['generated_report'])
    timings['census_report_serialization_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    render_settings=primitives.apply_profile(rep,settings,profile)
    timings['profile_application_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    bindings={**candidates['source_bindings'],**population['source_bindings'],**source_hashes(stage),**request['implementation_bindings'],**implementation_bindings()}
    timings['initial_source_hash_collection_seconds']=time.perf_counter()-tick
    morphology=population['foreground_identity']['morphology_id']
    geometry_source=dict(manifest=candidates['manifest'],qualification=candidates['qualification'],
        donor_source_family=candidates['source_family'],source_split='train',
        physical_geometry_evidence=request['physical_geometry_evidence'])
    variants=deepcopy(population['variants'])
    for v in variants:v.setdefault('geometry_source_id',v['source_plant_id'])
    context=dict(schema='greenhouse.native848_generated_population_context.v1',dataset_split='train',
        native_population_census_verified=True,original_143_backgrounds_preserved=True,
        full_scene_census=census,scene_variants=variants,geometry_sources={morphology:geometry_source},
        source_bindings=bindings,source_collection_plan=census['source_collection_plan'],
        generator_source_collection_plan=candidates['source_plan'],
        native_resolution=[848,408],catalogue=pin(output/'catalogue.json'),profile=candidates['profile'],
        profile_role='source_optical_settings_and_baseline_warmup_reference_only',
        actual_experimental_profile=actual,selected_experimental_sample_ids=scheduled['selected_sample_ids'],capture_purpose=scheduled['purpose'],paired_control_reference=scheduled['paired_control_reference'],export_prohibited=scheduled['purpose']==diagnostic.CONTROL,
        renderer_settings=render_settings,persistent_batch_authority=dict(batch_request=batch_request_pin,source_request=source_request_pin,segment_id=segment_id,native_identity=batch_native_identity),persistent_foreground_swap=population['persistent_swap_evidence'],**FLAGS)
    holds=[];requests=[];committed=[]
    request_start=writer.request_index;callback_start=writer.sequence
    with ExitStack() as cleanup:
        tick=time.perf_counter()
        geometry=StaticGeometryPartsCache(stage,robot['root'],include_generated_plants=True);cleanup.callback(geometry.close)
        timings['geometry_cache_initialization_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        workspace=BulkWorkspace();cleanup.callback(workspace.finish)
        timings['workspace_initialization_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        sink=FrameSink(output/'frames',max_frames=8,max_bytes=256*2**20,workers=2);cleanup.callback(sink.close)
        timings['frame_sink_initialization_seconds']=time.perf_counter()-tick
        warmed=False;monitor=None;frozen_sources=None
        for record in records:
            if len(committed)>=len(records):break
            tick=time.perf_counter()
            pose,cal=apply_cached_pose(stage,robot,record)
            timings['pose_application_seconds']=timings.get('pose_application_seconds',0)+time.perf_counter()-tick
            tick=time.perf_counter()
            screen=geometry()
            timings['collision_screen_seconds']=timings.get('collision_screen_seconds',0)+time.perf_counter()-tick
            if not screen['passed']:
                holds.append(dict(sample_id=record['sample_id'],reason='whole_robot_scene_collision',screen=screen));continue
            pose['visual_bound_screen']=screen
            # The point is reconstructed from the current generated manifest, never inherited.
            tick=time.perf_counter()
            from .native848_all_petiole_9mm_v2 import geometry_9mm
            from pxr import UsdGeom
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(census['foreground_root'])),float)
            geo=geometry_9mm(population['generated_report'],candidates['target_component_id'],matrix,cal)
            planned=record['pilot_9mm_geometry']
            require(np.allclose(geo['oriented_centerline_world_m'],planned['oriented_centerline_world_m'],rtol=0,atol=1e-9)
                    and np.allclose(geo['nominal']['world_m'],planned['nominal']['world_m'],rtol=0,atol=1e-9)
                    and np.allclose(geo['nominal']['projected']['pixel_xy'],planned['nominal']['projected']['pixel_xy'],rtol=0,atol=1e-6),
                    'Generated camera proposal has stale geometry')
            timings['current_9mm_geometry_seconds']=timings.get('current_9mm_geometry_seconds',0)+time.perf_counter()-tick
            tick=time.perf_counter()
            reach=workspace.check(dict(calibration=cal,robot_snapshot=pose,
                supervision=dict(target_id=record['target_id'],nominal_world_m=geo['nominal']['world_m'])))
            timings['workspace_check_seconds']=timings.get('workspace_check_seconds',0)+time.perf_counter()-tick
            if not reach['result']['workspace_passed']:
                holds.append(dict(sample_id=record['sample_id'],reason='9mm_workspace',workspace=reach));continue
            if not warmed:
                tick=time.perf_counter()
                omni.usd.get_context().reset_renderer_accumulation()
                for steps in actual['warmup_steps']:
                    _,e=primitives.request_once(rep,writer,steps,profile['delta_time_seconds']);requests.append(e)
                timings['warmup_requests_seconds']=time.perf_counter()-tick
                tick=time.perf_counter()
                frozen_sources=source_hashes(stage)
                timings['post_warmup_source_hash_seconds']=time.perf_counter()-tick
                tick=time.perf_counter()
                for path,h in frozen_sources.items():
                    require(path not in bindings or bindings[path]==h,'Conflicting post-warmup source')
                    bindings[path]=h
                verify_bindings(bindings)
                save_json(output/'context.json',context)
                timings['post_warmup_binding_verify_context_save_seconds']=time.perf_counter()-tick
                tick=time.perf_counter()
                monitor=StaticSceneMonitor(stage,robot['root']);cleanup.callback(monitor.close)
                timings['static_monitor_initialization_seconds']=time.perf_counter()-tick
                warmed=True
            tick=time.perf_counter()
            before=monitor.begin()
            reset=omni.usd.get_context().reset_renderer_accumulation()
            timings['production_static_begin_reset_seconds']=timings.get('production_static_begin_reset_seconds',0)+time.perf_counter()-tick
            tick=time.perf_counter()
            payload,e=primitives.request_once(rep,writer,profile['request_subframes'],profile['delta_time_seconds'])
            timings['production_request_seconds']=timings.get('production_request_seconds',0)+time.perf_counter()-tick
            tick=time.perf_counter()
            after=monitor.token()
            rgb,depth,valid,reference,token,ids,mapping=primitives.native_freshness(payload,cal,before,after,writer.sequence,previous=previous,expect_change=previous is not None)
            previous=token
            assert_same_camera(calibration_for_native_resolution(calibration(stage),(848,408)),cal)
            require({k:jsonable(settings.get(k)) for k in render_settings}==render_settings,'Renderer settings changed')
            e.update(settings=render_settings,reset_returned_value=jsonable(reset));requests.append(e)
            sample_id=record['sample_id']
            mp=output/(sample_id+'_mapping.json');save_json(mp,dict(renderer_id_to_prim={str(k):v for k,v in mapping.items()}))
            observation=dict(schema=OBSERVATION_SCHEMA,observation_id=sample_id,sample_id=sample_id,
                segment_id=segment_id,batch_request=batch_request_pin,source_request=source_request_pin,
                morphology_id=morphology,source_family=candidates['source_family'],context=pin(output/'context.json'),
                capture_purpose=scheduled['purpose'],paired_control_reference=scheduled['paired_control_reference'],export_prohibited=scheduled['purpose']==diagnostic.CONTROL,
                calibration=cal,robot_snapshot=pose,mapping=pin(mp),geometry=geo,workspace=reach,
                rendered_camera_params=jsonable(payload['camera_params']),request_evidence=e,
                native_payload_header=jsonable({k:v for k,v in payload.items() if k not in ('rgb','distance_to_image_plane','instance_id_segmentation')}),
                synchronization=dict(scene_unchanged_during_capture=True,dynamic_recording_supported=False,
                    reference_time=reference,freshness=token,static_before=before,static_after=after,
                    callback_sequence=writer.sequence,request_index=writer.request_index,
                    render_frame=jsonable(payload['pilot_render_frame'])),timing=dict(request_seconds=e['elapsed_seconds']),**FLAGS)
            sink.submit(observation,rgb,depth,ids,valid,np.asarray(payload['rgb'])[:,:,3].copy())
            committed.append(sample_id)
            timings['production_validation_serialization_seconds']=timings.get('production_validation_serialization_seconds',0)+time.perf_counter()-tick
            print('CONTROLLED_FRAME_SAVED',sample_id,flush=True)
        tick=time.perf_counter()
        receipts=sink.close()
        timings['sink_drain_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        require(len(receipts)==len(committed)<=len(records) and writer.capture_error is None, 'Capture callback/write mismatch')
        if not warmed:
            frozen_sources=source_hashes(stage)
            for path,h in frozen_sources.items():
                require(path not in bindings or bindings[path]==h,'Conflicting final source')
                bindings[path]=h
            save_json(output/'context.json',context)
        require(source_hashes(stage)==frozen_sources,'Generated scene sources changed between captures')
        verify_bindings(bindings)
        require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Source layer dirtied')
        timings['closing_source_checks_seconds']=time.perf_counter()-tick
        timings['segment_total_seconds']=time.perf_counter()-started
        save_json(output/'result.json',dict(schema=SEGMENT_SCHEMA,state='captured_pending_complete_annotation_and_visual_review',
            source_request=source_request_pin,batch_request=batch_request_pin,segment_id=segment_id,native_identity=batch_native_identity,
            frames=receipts,holds=holds,requests=requests,request_count=writer.request_index-request_start,
            request_index_start=request_start,request_index_end=writer.request_index,
            callback_sequence_start=callback_start,callback_sequence_end=writer.sequence,timings=timings,
            morphology_id=morphology,actual_experimental_profile=actual,selected_experimental_sample_ids=scheduled['selected_sample_ids'],capture_purpose=scheduled['purpose'],paired_control_reference=scheduled['paired_control_reference'],export_prohibited=scheduled['purpose']==diagnostic.CONTROL,original_143_backgrounds_preserved=True,**FLAGS))
    return dict(segment_id=segment_id,result=pin(output/"result.json"),frames=len(receipts),holds=len(holds)),previous


def capture(app,request,segments,output,*,request_pin):
    import omni.replicator.core as rep
    from .native848_fully_labeled_scene_v1 import prepare_native_scene,policy
    from .native848_persistent_foreground_v1 import snapshot_original
    from .review_camera import HEAD_CAMERA
    from .capture_pilot import source_hashes
    started=time.perf_counter();first=segments[0][1];anchor=bound(first['records'][0]['anchor'])
    scene=prepare_native_scene(app,anchor,policy('train'))
    preparation_seconds=time.perf_counter()-started
    batches=output/'segments';batches.mkdir();results=[];previous=None;anchor_cache=set()
    with ExitStack() as cleanup:
        product=rep.create.render_product(HEAD_CAMERA,(848,408));cleanup.callback(product.destroy)
        writer=primitives.make_bulk_writer(rep,None);writer.attach([product]);cleanup.callback(writer.detach)
        # Record protected scene opinions after the render pipeline authors its setup.
        session=snapshot_original(scene,anchor,profile_pin=first['profile'],diagnostic_dir=output/'scene_protection')
        native_identity=pin(output/'native_identity.json')
        for scheduled,(source,candidates) in zip(request['segments'],segments):
            segment_root=batches/scheduled['segment_id'];segment_root.mkdir();target=segment_root/'capture';target.mkdir()
            require(writer.active is False and writer.capture_error is None,'Inactive healthy writer required before morphology swap')
            result,previous=capture_segment(app,source,candidates,target,scheduled=scheduled,
                source_request_pin=scheduled['source_request'],batch_request_pin=request_pin,
                segment_id=scheduled['segment_id'],session=session,writer=writer,
                batch_native_identity=native_identity,previous=previous,anchor_cache=anchor_cache)
            results.append(result)
            print('PERSISTENT_SEGMENT_SAVED',result,flush=True)
        require(not writer.active and writer.capture_error is None,'Final writer failure')
        require(sum(r['frames'] for r in results)<=request['max_frames'],'Exceeded batch schedule')
        verify_bindings(request['implementation_bindings'])
        background_closure=session.verify_backgrounds()
        require(not any(not l.anonymous and l.dirty for l in scene['stage'].GetUsedLayers()),'Dirty original source layer')
        result=dict(schema=RESULT_SCHEMA,state='captured_pending_complete_annotation_and_visual_review',
            request=request_pin,native_identity=native_identity,segments=results,
            frames=sum(r['frames'] for r in results),holds=sum(r['holds'] for r in results),
            request_count=writer.request_index,callback_count=writer.sequence,
            same_stage_product_writer_for_all_segments=True,scene_preparation_seconds=preparation_seconds,
            total_capture_seconds=time.perf_counter()-started,background_closure=background_closure,experimental_profile_policy=request['experimental_profile_policy'],**FLAGS)
        validate_batch_registry(result,request,output,request_pin=request_pin,native_identity=native_identity)
        save_json(output/'result.json',result)


def native_command(args):
    return [str(Path('D:/isaac-sim-6.0.1/python.bat')),'-B','-u','-m',MODULE,'--capture',
            '--request',str(args.request.resolve()),'--request-sha256',args.request_sha256,'--output',str(args.output.resolve())]

def native_main(args):
    require(sha256(args.request)==args.request_sha256,'Request changed')
    request=read_json(args.request);candidates=check_request(request)
    require(not args.output.exists(),'Create-only capture output required');args.output.mkdir()
    gate=production.identity_gate;rows=gate.snapshot();native=gate.one(rows,os.getpid());command=gate.one(rows,native['ParentProcessId']);owner=gate.one(rows,command['ParentProcessId'])
    expected=native_command(args);gate.command_child(command,owner,expected);gate.native_child(native,command,expected[1:])
    save_json(args.output/'native_identity.json',dict(native=native,command=command,owner=owner,expected_command=expected))
    app=None;success=False
    try:
        from .native_generated_pair import windows_worker_admission
        save_json(args.output/'native_process_admission.json',windows_worker_admission(None))
        from isaacsim import SimulationApp
        config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',
                    sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false'])
        config=production.telemetry_optout.prepare_config(config,argv=sys.argv[1:],environment=os.environ)
        app=SimulationApp(config)
        import carb.settings
        save_json(args.output/'telemetry.json',production.telemetry_optout.validate_runtime(carb.settings.get_settings().get,environment=os.environ))
        capture(app,request,candidates,args.output,request_pin=pin(args.request))
        require(sha256(args.request)==args.request_sha256,'Request changed during capture');success=True
    except BaseException:
        save_json(args.output/'failure.json',dict(error=traceback.format_exc(),**FLAGS));raise
    finally:
        if app is not None:app.close(exit_code=0 if success else 1)

def owned_run(args):
    from .native_generated_reference.owner_v1 import owner_lock,resources,_run_child
    from .native_original_capture import serial_queue
    from .native848_task_telemetry_optout_v1 import child_environment
    from .native_dataset.native_process_guard import native_processes
    require(sha256(args.request)==args.request_sha256,'Request changed')
    request=read_json(args.request);check_request(request)
    require(not args.output.exists() and args.output.parent.is_dir(),'Fresh output in existing directory required')
    command_args=argparse.Namespace(request=args.request,request_sha256=args.request_sha256,output=args.output/'capture')
    command=native_command(command_args)
    with owner_lock():
        evidence=resources(args.output.parent);args.output.mkdir()
        save_json(args.output/'owner_started.json',dict(request=pin(args.request),command=command,resources=evidence,**FLAGS))
        root=Path(__file__).resolve().parents[3]
        _,environment=serial_queue.v5.worker_environments(root/'data/sim_data/native_capture_deps_py312_v1')
        code=_run_child(command,args.output,child_environment(environment))
        save_json(args.output/'owned_exit.json',dict(returncode=code,owned_worker=pin(args.output/'owned_worker.json')))
        require(code==0 and not native_processes(),'Native capture did not close cleanly')
        identity=read_json(args.output/'capture/native_identity.json')
        require(identity['owner']['ProcessId']==os.getpid() and identity['command']['ProcessId']==read_json(args.output/'owned_worker.json')['launcher_pid'], 'Native owner mismatch')
        gate=production.identity_gate
        require(gate.same_identity(identity['owner'],evidence['raw_owner_classification']['metadata']),'Owner identity changed')
        gate.command_child(identity['command'],identity['owner'],command);gate.native_child(identity['native'],identity['command'],command[1:])
        result=read_json(args.output/'capture/result.json')
        require(result['schema']==RESULT_SCHEMA and 0<=result['frames']<=request['max_frames'] and len(result['segments'])==len(request['segments'])
                and result['training_approved'] is False and not (args.output/'capture/failure.json').exists(),'Invalid native result')
        require(result['request']==pin(args.request) and result['native_identity']==pin(args.output/'capture/native_identity.json'),
                'Native result belongs to another request or process')
        validate_batch_registry(result,request,args.output/'capture',request_pin=pin(args.request),
            native_identity=pin(args.output/'capture/native_identity.json'))
        check_request(request)
        save_json(args.output/'owner_complete.json',dict(result=pin(args.output/'capture/result.json'),
            request=pin(args.request),native_identity=pin(args.output/'capture/native_identity.json'),
            owned_exit=pin(args.output/'owned_exit.json'),owner_started=pin(args.output/'owner_started.json'),
            frames=result['frames'],segments=result['segments'],**FLAGS))
        print(dict(frames=result['frames'],holds=result['holds'],segments=len(result['segments']),output=str(args.output)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',action='store_true')
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args();(native_main if arguments.capture else owned_run)(arguments)
