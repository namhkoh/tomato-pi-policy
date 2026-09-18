"""Uniformly filtered pilot scene with the frozen native848 capture primitives.

One stage/product and flat source validation per batch. Every observation still
uses an actual embodied pose, geometry screen, native same-callback buffers and
static/FK checks. CPU annotation and sampled batch QA are separate consumers.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
from fractions import Fraction
import argparse
import hashlib
import shutil
import time
import traceback
import numpy as np
from . import native848_pilot_plan_v1 as api
from . import native848_bulk_worker_v1 as frozen
from .native848_data_roots_v1 import diagnostic
make_bulk_writer=frozen.make_bulk_writer
request_once=frozen.request_once
native_freshness=frozen.native_freshness
apply_profile=frozen.apply_profile
FROZEN_CAPTURE_SHA256="5306fe59b579aa8e7bfc021ac39639ef7e0e62b06a851b12a6ab79a96daa204a"
from .native848_bulk_io_v1 import FrameSink, save_json
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint, jsonable


def capture(app, output, plan_path, plan, checked):
    import omni.usd
    import omni.replicator.core as rep
    from .native848_pilot_scene_v1 import prepare_native_scene
    from .native848_original_direct_worker_v2 import apply_cached_pose
    from .native848_fast_profile_benchmark_v1 import freeze_and_prove_clock_safety
    from .capture_pilot import source_hashes
    from .capture_scene import calibration, target_world_geometry
    from .capture_viewpoints import component_catalogue
    from .capture_sensor import LEGACY_RESOLUTION, calibration_for_native_resolution
    from .native_greenhouse_pair import assert_same_camera
    from .native848_pair_audit_v2 import world_from_row
    from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
    from .static_guard import StaticSceneMonitor
    from .review_camera import HEAD_CAMERA

    started = time.perf_counter()
    anchor, profile = checked['anchor'], checked['profile']
    records = checked['records']
    scene = prepare_native_scene(app, anchor, checked['pilot_scene_policy'])
    cpu=checked['cpu_scene_preflight']
    require(scene['scene_policy_evidence']['deterministic_census_sha256']==cpu['deterministic_census_sha256'],
            'Native filtered plant census differs from exact CPU preflight')
    save_json(output/'full_scene_census.json',scene['scene_policy_evidence'])
    stage, robot, settings = scene['stage'], scene['robot'], scene['settings']
    render_settings = apply_profile(rep, settings, profile)
    if profile['schema']==api.ORIGINAL_PROFILE_SCHEMA:
        clock_proof=dict(schema='greenhouse.original848_bulk_paused_clock.v1',delta_time_seconds=0,
            scheduler_clock_policy=api.PAUSED_CLOCK_POLICY,original_scene_setup_unchanged=False,uniform_pilot_scene_policy_applied=True,
            static_scene_guard_still_required=True,native_engine_frame_identity_claimed=False)
    else:
        clock_proof=freeze_and_prove_clock_safety(stage)
        require(clock_proof['delta_time_seconds']==profile['delta_time_seconds'],'Clock safety profile differs')
    save_json(output/'clock_safety.json', clock_proof)
    catalogue = component_catalogue(stage, scene['records'], scene['reports'], scene['variants'])
    require(len(catalogue) == scene['counts']['components'], 'Original component population changed')
    save_json(output/'catalogue.json', catalogue)
    save_json(output/'anatomy_reports.json', scene['reports'])
    report = next(r for r in scene['reports'] if r['plant_id'] == anchor['source_family'])
    targets = {}
    for rec in records:
        row = rec['source_row']
        if row['target_id'] in targets:
            continue
        world = target_world_geometry(stage, scene['original_variant'], row)
        expected = world_from_row(anchor, row)
        require(set(world) == set(expected) and all(np.allclose(world[k],expected[k],atol=1e-9,rtol=0)
            for k in world), 'Actual original target geometry changed')
        targets[row['target_id']] = dict(source_row=row, anatomy_report=report, world_geometry=world,
            source_family=anchor['source_family'], split=checked['capture_split'], original_variant=scene['original_variant'])
    original_pins = {**scene['original_population_bindings'],**source_hashes(stage)}
    product = rep.create.render_product(HEAD_CAMERA, LEGACY_RESOLUTION)
    warmup_folder = output/'warmup'
    warmup_folder.mkdir()
    writer = make_bulk_writer(rep, warmup_folder/'all_callbacks')
    writer.attach([product])
    geometry = StaticGeometryPartsCache(stage, robot['root'], include_generated_plants=True)
    monitor = sink = None
    decisions, warmup_requests = [], []
    previous = None
    proofs, mapping_files = {}, {}
    try:
        # Warm the exact product at the first scheduled embodied pose. Control
        # frames remain diagnostics and are never submitted to the frame sink.
        apply_cached_pose(stage, robot, records[0])
        reset = omni.usd.get_context().reset_renderer_accumulation()
        for n in profile['warmup_steps']:
            _, evidence = request_once(rep, writer, n, profile['delta_time_seconds'])
            warmup_requests.append(evidence)
        save_json(warmup_folder/'request_evidence.json', dict(requests=warmup_requests,
            reset_api=profile['reset_api'], reset_returned_value=jsonable(reset),
            settings=render_settings, control_frames_excluded=True))
        writer.debug_folder = None
        verify_bindings(original_pins)
        full_pins = source_hashes(stage)
        bindings = dict(checked['source_bindings'])
        for path, pin in {**original_pins, **full_pins}.items():
            require(path not in bindings or bindings[path] == pin, 'Conflicting actual scene source')
            bindings[path] = pin
        monitor = StaticSceneMonitor(stage, robot['root'])
        binding_hash = fingerprint(bindings)
        scene_revision = fingerprint(dict(static_baseline=monitor.baseline, source_bindings_sha256=binding_hash,
            scene_counts=scene['counts'], lighting=scene['old_manifest']['lighting']))
        proof_directory, mapping_directory = output/'geometry_proofs', output/'renderer_mappings'
        proof_directory.mkdir(); mapping_directory.mkdir()
        context = dict(schema=api.CONTEXT_SCHEMA, plan_path=str(plan_path.resolve()), plan_sha256=sha256(plan_path),
            profile_evidence=plan['profile_evidence'], source_bindings=bindings, source_bindings_sha256=binding_hash,
            scene_revision=scene_revision, static_baseline=monitor.baseline, anchor=anchor,
            cache_path=plan['cache_path'], cache_sha256=plan['cache_sha256'],
            source_reports_path=str((output/'anatomy_reports.json').resolve()),
            source_reports_sha256=sha256(output/'anatomy_reports.json'), source_family=anchor['source_family'],
            split=checked['capture_split'], scene_counts=scene['counts'], lighting=scene['old_manifest']['lighting'],
            renderer=settings.get('/rtx/rendermode'), render_settings=render_settings,
            catalogue_path=str((output/'catalogue.json').resolve()), catalogue_sha256=sha256(output/'catalogue.json'),
            targets=targets, geometry_proofs_directory=str(proof_directory.resolve()),
            warmup_evidence_path=str((warmup_folder/'request_evidence.json').resolve()),
            warmup_evidence_sha256=sha256(warmup_folder/'request_evidence.json'),
            geometry_proof_policy='every_actual_robot_world_transform_checked_including_head',
            scene_policy=checked['pilot_scene_policy'],
            full_scene_census=scene['scene_policy_evidence'],
            full_scene_census_path=str((output/'full_scene_census.json').resolve()),
            full_scene_census_sha256=sha256(output/'full_scene_census.json'),
            cpu_scene_preflight=checked['cpu_scene_preflight_pin'],
            task_input_policy=dict(query_input=False,clean_full_rgb=True,aligned_depth=True),
            training_approved=False)
        context_path = output/'batch_context.json'
        save_json(context_path, context)
        context_hash = sha256(context_path)
        sink = FrameSink(output/'frames', max_frames=plan['queue_frames'], max_bytes=plan['queue_bytes'],
            workers=plan['writer_threads'])
        for capture_index, (item, rec) in enumerate(zip(plan['schedule'],records)):
            tick = time.perf_counter()
            name = item['observation_id']
            decision = dict(observation_id=name, source_pose_id=item['source_pose_id'], capture_index=capture_index,
                target_id=rec['target_id'], state='pending', native_requests=0, training_approved=False)
            try:
                if (output/'STOP_AFTER_CURRENT_FRAME').exists():
                    decision['state'] = 'not_requested_user_stop'
                    break
                pose, cal = apply_cached_pose(stage, robot, rec)
                applied = time.perf_counter()
                screen = geometry()
                pose['visual_bound_screen'] = screen
                key = api.geometry_pose_key(scene_revision, pose, binding_hash)
                proof = dict(schema='greenhouse.original848_bulk_geometry_proof.v1', pose_key=key,
                    scene_revision=scene_revision, source_bindings_sha256=binding_hash,
                    robot_root_to_world_usd_row_vectors=pose['robot_root_to_world_usd_row_vectors'],
                    joint_degrees=pose['joint_degrees'], screen=screen)
                proof_path = proof_directory/(key+'.json')
                if key not in proofs:
                    save_json(proof_path,proof)
                    proofs[key] = dict(pose_key=key,path=str(proof_path.resolve()),sha256=sha256(proof_path))
                else:
                    require(read_json(proof_path)==proof,'Same complete pose produced different geometry')
                if not screen['passed']:
                    decision['state'] = 'rejected_native_geometry'
                    continue
                screened = time.perf_counter()
                require({k:jsonable(settings.get(k)) for k in render_settings} == render_settings,
                    'Renderer settings changed before production request')
                before = monitor.begin()
                reset = omni.usd.get_context().reset_renderer_accumulation()
                payload, request = request_once(rep,writer,profile['request_subframes'],profile['delta_time_seconds'])
                decision['native_requests'] = 1
                request.update(settings=render_settings,render_settings=render_settings,reset_api=profile['reset_api'],
                    reset_returned_without_exception=True, reset=dict(api=profile['reset_api'],
                    returned_without_exception=True,returned_value=jsonable(reset)))
                after = monitor.token()
                changed = previous is not None and fingerprint(cal)!=previous['camera_sha256']
                rgb,depth,valid,reference,token,ids,mapping = native_freshness(payload,cal,before,after,
                    writer.sequence,previous,changed)
                previous = token
                require({k:jsonable(settings.get(k)) for k in render_settings} == render_settings,
                    'Renderer settings changed during request')
                assert_same_camera(calibration_for_native_resolution(calibration(stage),LEGACY_RESOLUTION),cal)
                require(not any(p.startswith('/World/GeneratedNativePilot/') for p in mapping.values()),
                    'Generated geometry entered original scene')
                value = dict(renderer_id_to_prim={str(k):v for k,v in mapping.items()},
                    scope='all_observed_renderer_IDs_only')
                mapping_key = fingerprint(value)
                if mapping_key not in mapping_files:
                    path = mapping_directory/(mapping_key+'.json');save_json(path,value)
                    mapping_files[mapping_key] = dict(path=str(path.resolve()),sha256=sha256(path),scope=value['scope'])
                header = jsonable({k:v for k,v in payload.items()
                    if k not in ('rgb','distance_to_image_plane','instance_id_segmentation')})
                observation = dict(schema=api.OBSERVATION_SCHEMA,observation_id=name,capture_index=capture_index,
                    source_pose_id=item['source_pose_id'],target_id=rec['target_id'],source_family=anchor['source_family'],
                    target_id_scope='private_pose_provenance_not_model_input_or_unique_label',
                    split=checked['capture_split'],capture_role='qualification_candidate',profile=profile['profile'],
                    context_path=str(context_path.resolve()),context_sha256=context_hash,calibration=cal,
                    rendered_camera_params=jsonable(payload['camera_params']),robot_snapshot=pose,
                    geometry_proof=proofs[key],native_payload_header=header,
                    synchronization=dict(reference_time=reference,freshness=token,static_before=before,static_after=after,
                        render_frame=jsonable(payload['pilot_render_frame']),callback_sequence=writer.sequence,
                        request_index=writer.request_index),request_evidence=request,mapping=mapping_files[mapping_key],
                    timing=dict(pose_apply_seconds=applied-tick,geometry_screen_seconds=screened-applied,
                        request_seconds=request['elapsed_seconds'],before_enqueue_seconds=time.perf_counter()-tick),
                    training_approved=False)
                # Freshness already returns owned RGB/depth/ID/validity arrays.
                # Copy alpha once; never hand a live annotator buffer to threads.
                wait = sink.submit(observation,rgb,depth,ids,valid,
                    np.asarray(payload['rgb'])[:,:,3].copy())
                decision.update(state='queued_lossless_callback_pending_cpu_admission',queue_submit_seconds=wait,
                    total_producer_seconds=time.perf_counter()-tick)
                print('BULK848_QUEUED',capture_index,name,round(request['elapsed_seconds'],4),flush=True)
            except BaseException:
                decision.update(state='failed_preserved_no_retry',error=traceback.format_exc(),
                    last_native_request=getattr(writer,'last_request_evidence',None))
                raise
            finally:
                decisions.append(decision)
                save_json(output/('decision_'+name+'.json'),decision)
        committed = sink.close()
        require(source_hashes(stage)==full_pins,'Actual scene asset population changed')
        verify_bindings(bindings)
        require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()),'Source layer dirtied')
        require(writer.capture_error is None and not writer.active,'Unaccounted native callback')
        require(sha256(context_path)==context_hash,'Immutable context changed')
        manifest = dict(schema='greenhouse.original848_bulk_manifest.v1',context_path=str(context_path.resolve()),
            context_sha256=context_hash,observations=committed,decisions=decisions,geometry_proofs=list(proofs.values()),
            renderer_mappings=list(mapping_files.values()),native_request_count=writer.request_index,
            native_callback_count=writer.sequence,warmup_request_count=len(warmup_requests),
            production_native_requests=sum(r['native_requests'] for r in decisions),committed_frames=len(committed),
            source_assets_unchanged=True,queue_peak_uncompressed_bytes=sink.peak_bytes,
            geometry_cache=geometry.diagnostics(),elapsed_seconds=time.perf_counter()-started,
            stage_count=1,render_product_count=1,automatic_retries=False,training_approved=False,
            accepted_training_increment=0)
        save_json(output/'batch_manifest.json',manifest)
        return dict(state=api.RESULT_STATE,manifest_path=str((output/'batch_manifest.json').resolve()),
            manifest_sha256=sha256(output/'batch_manifest.json'),committed_frames=len(committed),
            training_approved=False,accepted_training_increment=0)
    finally:
        if sink is not None:
            sink.close()
        if monitor is not None:
            monitor.close()
        geometry.close()
        writer.detach()
        product.destroy()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene-preflight',type=Path,required=True)
    parser.add_argument('--scene-preflight-sha256',required=True)
    args=parser.parse_args(argv)
    require(sha256(frozen.__file__)==FROZEN_CAPTURE_SHA256,'Frozen native capture primitives changed')
    require(sha256(args.plan)==args.plan_sha256,'Changed pilot plan')
    require(sha256(args.scene_preflight)==args.scene_preflight_sha256,'Changed CPU scene preflight')
    plan=read_json(args.plan);output=diagnostic(args.output)
    require(plan['schema']==api.SCHEMA and not output.exists(),'New bulk output required')
    from sim_physics.host_memory import preflight
    from .native_generated_pair import windows_worker_admission
    memory=preflight();disk=shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30,
        'Native memory/disk reserves unavailable')
    process=windows_worker_admission(None)
    output.mkdir(parents=True)
    save_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(args.plan.resolve()),plan_sha256=args.plan_sha256,
        scene_preflight=dict(path=str(args.scene_preflight.resolve()),sha256=args.scene_preflight_sha256),host_memory_preflight=memory,
        available_disk_bytes=disk,process_admission=process,automatic_retries=False,training_approved=False))
    app=None;succeeded=False
    try:
        # No plan evaluation or native USD imports before SimulationApp owns ABI.
        from isaacsim import SimulationApp
        require(sha256(plan['profile_evidence']['path'])==plan['profile_evidence']['sha256'],'Changed actual profile')
        profile=read_json(plan['profile_evidence']['path'])
        if profile['schema']==api.ORIGINAL_PROFILE_SCHEMA:
            config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',
                sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false'])
        else:
            config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RealTimePathTracing',
                anti_aliasing=2,sync_loads=False,disable_viewport_updates=True,
                extra_args=['--/app/settings/persistent=false','--/rtx/rtpt/enabled=true'])
        app=SimulationApp(config)
        checked=api.check(plan)
        cpu=read_json(args.scene_preflight)
        require(cpu['schema']=='greenhouse.native848_pilot_scene_cpu_preflight.v1'
            and cpu['plan_sha256']==args.plan_sha256 and cpu['scene_policy']==checked['pilot_scene_policy']
            and cpu['all_selected_pose_screens_passed'] is True
            and cpu['selected_sample_ids']==plan['selected_sample_ids'], 'Exact selected CPU scene preflight required')
        checked['cpu_scene_preflight']=cpu
        checked['cpu_scene_preflight_pin']=dict(path=str(args.scene_preflight.resolve()),sha256=args.scene_preflight_sha256)
        for path,pin in cpu['source_bindings'].items():
            require(path not in checked['source_bindings'] or checked['source_bindings'][path]==pin,'CPU source binding conflict')
            checked['source_bindings'][path]=pin
        checked['source_bindings'][str(args.scene_preflight.resolve())]=args.scene_preflight_sha256
        verify_bindings(checked['source_bindings'])
        result=capture(app,output,args.plan,plan,checked)
        require(sha256(args.plan)==args.plan_sha256,'Bulk plan changed during capture')
        result['plan_sha256']=args.plan_sha256
        save_json(output/'result.json',result);succeeded=True
    except BaseException:
        save_json(output/'failure.json',dict(state='native848_filtered_pilot_failed',error=traceback.format_exc(),
            automatic_retries=False,training_approved=False,accepted_training_increment=0))
        raise
    finally:
        if app is not None:
            app.close(exit_code=0 if succeeded else 1)


if __name__=='__main__':
    main()
