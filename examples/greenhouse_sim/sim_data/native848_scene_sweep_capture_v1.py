"""One static, fully labeled greenhouse; many embodied views; annotation offline."""
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
import argparse, os, sys, time, traceback
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint, jsonable
from .native848_bulk_io_v1 import FrameSink, save_json
from . import native848_bulk_worker_v1 as sensor
from . import native848_fully_labeled_worker_v3 as production
from . import native848_fully_labeled_scene_v1 as population
MODULE='sim_data.native848_scene_sweep_capture_v1'
REQUEST_SCHEMA='greenhouse.native848_scene_sweep_request.v1'
RESULT_SCHEMA='greenhouse.native848_scene_sweep_capture.v1'
OBSERVATION_SCHEMA='greenhouse.native848_scene_sweep_observation.v1'
CONTEXT_SCHEMA='greenhouse.native848_scene_sweep_context.v1'
FLAGS=dict(training_approved=False,accepted_training_increment=0,single_answer_verified=False)
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha256(p))
def bound(p):
    require(pin(p['path'])==p,'Changed pinned input');return read_json(p['path'])
def implementation_bindings():
    names=('native848_original_direct_worker_v2.py','static_geometry_parts_cache_v1.py','native848_all_petiole_9mm_v2.py','static_guard.py','native848_bulk_io_v1.py')
    paths=[Path(__file__),Path(sensor.__file__),Path(production.__file__),Path(population.__file__)]+[Path(__file__).with_name(n) for n in names]
    return {str(p.resolve()):sha256(p) for p in paths}
def check_request(req):
    require(req['schema']==REQUEST_SCHEMA and req['implementation_bindings']==implementation_bindings(),'Exact sweep implementation required')
    verify_bindings(req['implementation_bindings'])
    anchor=bound(req['scene_anchor']);census=bound(req['expected_census']);profile=bound(req['profile'])
    from .native848_bulk_plan_v1 import check_profile
    check_profile(profile)
    records=bound(req['records'])['records'];proof=bound(req['cpu_preflight'])
    require(proof['passed'] is True and proof['records']==req['records'] and proof['expected_census']==req['expected_census'],'Fresh CPU sweep proof does not bind this exact schedule/scene')
    verify_bindings(proof['source_bindings'])
    cpu_rows=proof['per_record'];require(len(cpu_rows)==len(records) and len({r['sample_id'] for r in cpu_rows})==len(records),'Complete unique CPU pose evidence required')
    require({r['sample_id'] for r in cpu_rows}=={r['sample_id'] for r in records} and all(all(r.get(k) is True for k in ('fk_calibration_passed','whole_robot_collision_passed','workspace_passed')) for r in cpu_rows),'Every scheduled camera needs current FK/collision/workspace checks')
    require(req['scene_policy']==population.policy(anchor['split']) and census['policy']==req['scene_policy'],'One exact fully labeled scene required')
    require(census['active_counts']['component_plants']==144 and census['complete_active_plant_anatomy'],'All144 labeled plants required')
    require(req['training_approved'] is False and type(req['accepted_training_increment']) is int and req['accepted_training_increment']==0 and req['single_answer_verified'] is False,'Raw captures are not accepted images')
    require(1<=len(records)<=4096 and req['max_frames']==len(records),'Finite exact camera schedule required')
    slots={r['plant_root']:r for r in census['all_plant_roots']};ids=set();cameras=set()
    require(len(census['all_plant_roots'])==len(slots)==144 and all(r['source_split']==anchor['split'] for r in slots.values()),'Exactly144 unique same-split plant roots required')
    for r in records:
        name=r['sample_id'];require(name not in ids and Path(name).name==name,'Unique safe sample identity required');ids.add(name)
        root=r['plant_root'];v=r['target_variant'];require(root in slots and v['plant_root']==root and v['variant_id']==slots[root]['variant_id'] and r['source_family']==v['source_plant_id']==v['split_group']==slots[root]['source_family'],'Actual instance/source identity differs')
        require(r['source_family']==r['source_row']['source_plant_id'] and not(r['source_family']=='seed41_full' and r['source_row']['component_id']=='SubStem_38'),'Foreign or excluded source target')
        key=tuple(np.asarray(r['calibration']['camera_to_world_usd_row_vectors']).round(9).ravel());require(key not in cameras,'Duplicate camera within unchanged scene');cameras.add(key)
        require(r['target_id']==v['variant_id']+'/'+r['source_row']['component_id'],'Wrong compound instance/component target identity')
        for value,column in ((r['robot_root_to_world_usd_row_vectors'],False),(r['calibration']['camera_to_world_usd_row_vectors'],False),(r['camera_to_head_column_vectors'],True)):
            m=np.asarray(value,float);require(m.shape==(4,4) and np.isfinite(m).all(),'Finite4x4 transform required')
            if column:m=m.T
            require(np.allclose(m[:,3],[0,0,0,1],atol=1e-9,rtol=0) and np.allclose(m[:3,:3]@m[:3,:3].T,np.eye(3),atol=1e-7,rtol=0) and abs(np.linalg.det(m[:3,:3])-1)<1e-7,'Proper rigid pose transform required')
    return dict(anchor=anchor,census=census,profile=profile,records=records,cpu_preflight=proof)
def validate_result(result,request,output):
    from .native_greenhouse_pair import assert_same_camera
    request_pin=pin(request);req=bound(request_pin);output=Path(output).resolve()
    require(result['schema']==RESULT_SCHEMA and result['request']==request_pin
            and all(result.get(k)==v and type(result.get(k)) is type(v) for k,v in FLAGS.items()),'Wrong or accepted sweep result')
    require(result['stage_count']==result['render_product_count']==1 and result['scene_assets_unchanged'] is True,'One unchanged scene required')
    require(result['context']==pin(output/'context.json') and result['native_identity']==pin(output/'native_identity.json'),'Canonical context/native authority required')
    context=bound(result['context']);bound(result['native_identity'])
    require(context['schema']==CONTEXT_SCHEMA and context['request']==request_pin
            and context['native_identity']==result['native_identity']
            and context['scene_anchor']==req['scene_anchor'] and context['profile']==req['profile']
            and all(context.get(k)==v and type(context.get(k)) is type(v) for k,v in FLAGS.items()),'Foreign or accepted context authority')
    require(context['census']==pin(output/'census.json') and bound(context['census'])==bound(req['expected_census'])==context['full_scene_census'],'Fixed full144 census differs')
    for key,filename in (('catalogue','catalogue.json'),('reports','reports.json')):
        require(context[key]==pin(output/filename),'Canonical fixed scene anatomy artifact required');bound(context[key])
    for key in ('scene_anchor','expected_census','profile','records','cpu_preflight'):
        require(context['source_bindings'].get(req[key]['path'])==req[key]['sha256'],'Unbound sweep source')
    require(all(context['source_bindings'].get(path)==h for path,h in req['implementation_bindings'].items()),'Unbound sweep implementation')
    verify_bindings(context['source_bindings'])
    records=bound(req['records'])['records'];scheduled={r['sample_id']:r for r in records}
    require(len(scheduled)==len(records)==req['max_frames'],'Unique complete scheduled poses required')
    held=[]
    for hold in result['holds']:
        name=hold['sample_id'];require(name in scheduled and name not in held
            and hold['reason']=='whole_robot_scene_collision' and hold['screen']['passed'] is False,'Foreign, repeated or unsupported hold')
        held.append(name)
    expected_names=[r['sample_id'] for r in records if r['sample_id'] not in held]
    require(expected_names and len(result['frames'])==len(expected_names),'Camera schedule incomplete')
    profile=bound(req['profile']);steps=list(profile['warmup_steps']);warmup_count=len(steps)
    events=result['requests'];require(result['warmup_requests']==events[:warmup_count]
        and len(result['warmup_requests'])==warmup_count,'Warmup ledger differs from profile')
    require(type(result['callback_count']) is int and type(result['request_count']) is int
        and result['callback_count']==result['request_count']==len(events)==warmup_count+len(expected_names),'Request callback accounting differs')
    for i,event in enumerate(events,1):
        expected_steps=steps[i-1] if i<=warmup_count else profile['request_subframes']
        require(all(type(event[k]) is int for k in ('request_index','callback_sequence_before','callback_sequence_after','callback_count','native_requests'))
            and event['request_index']==i and event['callback_sequence_before']==i-1 and event['callback_sequence_after']==i
            and event['callback_count']==event['native_requests']==1,'Noncontinuous writer ledger')
        require('error' not in event and event['requested_subframes']==expected_steps
            and event['delta_time_seconds']==profile['delta_time_seconds'] and event['wait_for_render'] is True,'Request profile or success differs')
        callbacks=event['callbacks'];require(len(callbacks)==1 and callbacks[0]['callback_sequence']==i
            and callbacks[0]['request_index']==i,'Callback event authority differs')
    for ordinal,(receipt,name) in enumerate(zip(result['frames'],expected_names),warmup_count+1):
        path=Path(receipt['path']).resolve();require(path==output/'frames'/name/'observation.json','Foreign or noncanonical frame location')
        observation=bound(dict(path=receipt['path'],sha256=receipt['sha256']));record=scheduled[name];sync=observation['synchronization']
        require(observation['schema']==OBSERVATION_SCHEMA and observation['observation_id']==observation['sample_id']==name
            and observation['context']==result['context'] and receipt['observation_id']==name
            and receipt['request_index']==sync['request_index']==sync['callback_sequence']==ordinal
            and observation['request_evidence']==events[ordinal-1],'Frame does not match its production callback')
        require(all(observation.get(k)==v and type(observation.get(k)) is type(v) for k,v in FLAGS.items()),'Raw observation cannot be accepted')
        require(observation['source_family']==record['source_family'] and observation['target_variant']==record['target_variant']
            and observation['plant_root']==record['plant_root']
            and observation['target_id']==record['target_variant']['variant_id']+'/'+record['source_row']['component_id'],'Frame instance target differs')
        assert_same_camera(observation['calibration'],record['calibration'])
        pose=observation['robot_snapshot'];require(pose['joint_degrees']==record['joint_degrees']
            and all(np.allclose(pose[k],record[k],atol=1e-9,rtol=0) for k in ('robot_root_to_world_usd_row_vectors','camera_to_head_column_vectors'))
            and pose['visual_bound_screen']['passed'] is True and pose['whole_robot_collision_checked'] is True,'Frame embodied pose/collision differs')
    return context


def capture(app,req,checked,output,request_pin):
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom
    from .native848_original_direct_worker_v2 import apply_cached_pose
    from .native848_fully_labeled_coverage_v3 import component_catalogue
    from .native848_all_petiole_9mm_v2 import geometry_9mm
    from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
    from .static_guard import StaticSceneMonitor
    from .capture_pilot import source_hashes
    from .capture_scene import calibration
    from .capture_sensor import calibration_for_native_resolution
    from .native_greenhouse_pair import assert_same_camera
    from .review_camera import HEAD_CAMERA
    started=time.perf_counter();scene=population.prepare_native_scene(app,checked['anchor'],req['scene_policy'])
    census=scene['scene_policy_evidence'];require(census==checked['census'],'Actual greenhouse differs from prepared scene')
    stage,robot=scene['stage'],scene['robot'];profile=checked['profile'];records=checked['records'];render_settings=sensor.apply_profile(rep,scene['settings'],profile)
    catalogue=component_catalogue(stage,scene['records'],scene['reports'],scene['variants'])
    save_json(output/'census.json',census);save_json(output/'catalogue.json',catalogue);save_json(output/'reports.json',scene['reports'])
    report_by={r['plant_id']:r for r in scene['reports']}
    bindings={**scene['original_population_bindings'],**source_hashes(stage),**req['implementation_bindings'],**profile['source_bindings']}
    for key in ('scene_anchor','expected_census','profile','records','cpu_preflight'):bindings[req[key]['path']]=req[key]['sha256']
    bindings.update(checked['cpu_preflight'].get('source_bindings',{}))
    product=rep.create.render_product(HEAD_CAMERA,[848,408]);writer=sensor.make_bulk_writer(rep,None);writer.attach([product])
    holds=[];requests=[];warmups=[];previous=None;committed=[];timings=dict(scene_setup_seconds=time.perf_counter()-started)
    try:
        with ExitStack() as cleanup:
            geometry=StaticGeometryPartsCache(stage,robot['root'],include_generated_plants=True);cleanup.callback(geometry.close)
            sink=FrameSink(output/'frames',max_frames=8,max_bytes=256*2**20,workers=2);cleanup.callback(sink.close)
            monitor=None;context_pin=None
            for record in records:
                tick=time.perf_counter();pose,cal=apply_cached_pose(stage,robot,record);screen=geometry()
                if not screen['passed']:
                    holds.append(dict(sample_id=record['sample_id'],reason='whole_robot_scene_collision',screen=screen));continue
                pose['visual_bound_screen']=screen;pose['whole_robot_collision_checked']=True
                matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(record['plant_root'])),float)
                geo=geometry_9mm(report_by[record['source_family']],record['source_row']['component_id'],matrix,cal)
                planned=record['pilot_9mm_geometry'];require(np.allclose(geo['nominal']['world_m'],planned['nominal']['world_m'],atol=1e-9,rtol=0) and np.allclose(geo['nominal']['projected']['pixel_xy'],planned['nominal']['projected']['pixel_xy'],atol=1e-6,rtol=0),'Stale instance cut geometry')
                if monitor is None:
                    warm_started=time.perf_counter();omni.usd.get_context().reset_renderer_accumulation()
                    for steps in profile['warmup_steps']:
                        _,e=sensor.request_once(rep,writer,steps,profile['delta_time_seconds']);requests.append(e);warmups.append(e)
                    timings['one_time_warmup_seconds']=time.perf_counter()-warm_started
                    scene_sources=source_hashes(stage)
                    for path,h in scene_sources.items():
                        require(path not in bindings or bindings[path]==h,'Conflicting scene source');bindings[path]=h
                    verify_bindings(bindings)
                    monitor=StaticSceneMonitor(stage,robot['root']);cleanup.callback(monitor.close)
                    context=dict(schema=CONTEXT_SCHEMA,request=request_pin,native_identity=pin(output/'native_identity.json'),source_bindings=bindings,scene_anchor=req['scene_anchor'],profile=req['profile'],renderer_settings=render_settings,
                        census=pin(output/'census.json'),full_scene_census=census,catalogue=pin(output/'catalogue.json'),reports=pin(output/'reports.json'),scene_variants=scene['variants'],dataset_split=checked['anchor']['split'],source_collection_plan=census['source_collection_plan'],
                        task_input_policy=dict(query_input=False,clean_full_rgb=True,aligned_depth=True),camera_transition_motion_validated=False,static_snapshots_only=True,**FLAGS)
                    save_json(output/'context.json',context);context_pin=pin(output/'context.json')
                before=monitor.begin();reset=omni.usd.get_context().reset_renderer_accumulation()
                payload,e=sensor.request_once(rep,writer,profile['request_subframes'],profile['delta_time_seconds'])
                e.update(render_settings=render_settings,settings=render_settings,reset_api=profile['reset_api'],reset_returned_without_exception=True,reset=dict(api=profile['reset_api'],returned_without_exception=True,returned_value=jsonable(reset)));requests.append(e)
                after=monitor.token();rgb,depth,valid,reference,token,ids,mapping=sensor.native_freshness(payload,cal,before,after,writer.sequence,previous,expect_change=previous is not None);previous=token
                assert_same_camera(calibration_for_native_resolution(calibration(stage),[848,408]),cal)
                require({k:jsonable(scene['settings'].get(k)) for k in render_settings}==render_settings,'Render settings changed')
                name=record['sample_id'];mp=output/(name+'_mapping.json');save_json(mp,dict(renderer_id_to_prim={str(k):v for k,v in mapping.items()},scope='all_observed_renderer_IDs_only'))
                observation=dict(schema=OBSERVATION_SCHEMA,observation_id=name,sample_id=name,context=context_pin,source_family=record['source_family'],target_variant=record['target_variant'],plant_root=record['plant_root'],
                    target_id=record['target_variant']['variant_id']+'/'+record['source_row']['component_id'],target_id_scope='private_pose_proposal_not_model_input_or_unique_label',source_pose_provenance=record['source_pose_provenance'],
                    calibration=cal,robot_snapshot=pose,geometry=geo,mapping=pin(mp),rendered_camera_params=jsonable(payload['camera_params']),request_evidence=e,
                    native_payload_header=jsonable({k:v for k,v in payload.items() if k not in ('rgb','distance_to_image_plane','instance_id_segmentation')}),
                    synchronization=dict(reference_time=reference,freshness=token,static_before=before,static_after=after,callback_sequence=writer.sequence,request_index=writer.request_index,render_frame=jsonable(payload['pilot_render_frame'])),
                    timing=dict(request_seconds=e['elapsed_seconds'],before_enqueue_seconds=time.perf_counter()-tick),**FLAGS)
                sink.submit(observation,rgb,depth,ids,valid,np.asarray(payload['rgb'])[:,:,3].copy());committed.append(name)
                print('SCENE_SWEEP_FRAME',len(committed),name,round(e['elapsed_seconds'],3),flush=True)
            receipts=sink.close();require(context_pin is not None,'No collision-free camera was captured')
            require(source_hashes(stage)==scene_sources,'Scene asset set changed');verify_bindings(bindings);require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Source layer dirtied')
            require(len(receipts)==len(committed) and writer.capture_error is None,'Frame sink mismatch')
            timings['total_capture_seconds']=time.perf_counter()-started
            result=dict(schema=RESULT_SCHEMA,request=request_pin,context=context_pin,native_identity=pin(output/'native_identity.json'),frames=receipts,holds=holds,requests=requests,warmup_requests=warmups,
                request_count=writer.request_index,callback_count=writer.sequence,stage_count=1,render_product_count=1,scene_assets_unchanged=True,timings=timings,**FLAGS)
            validate_result(result,req['request_path'],output);save_json(output/'result.json',result)
    finally:
        writer.detach();product.destroy()


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
        request["request_path"]=str(args.request.resolve())
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
        require(not (args.output/'capture/failure.json').exists(),'Failed native capture')
        validate_result(result,args.request,args.output/'capture')
        check_request(request)
        save_json(args.output/'owner_complete.json',dict(result=pin(args.output/'capture/result.json'),
            request=pin(args.request),native_identity=pin(args.output/'capture/native_identity.json'),
            owned_exit=pin(args.output/'owned_exit.json'),owner_started=pin(args.output/'owner_started.json'),
            frames=len(result['frames']),**FLAGS))
        print(dict(frames=len(result['frames']),holds=len(result['holds']),output=str(args.output)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',action='store_true')
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args();(native_main if arguments.capture else owned_run)(arguments)
