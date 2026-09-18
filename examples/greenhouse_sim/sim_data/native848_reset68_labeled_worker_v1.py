"""Capture all144 original slots populated by complete labeled component plants.

Foreground geometry is preserved. Background shape substitution is explicit,
source families remain distinct from instances, and no acceptance is inherited.
Native census must match the new full-population CPU proof before coverage is
marked verified. Capture/FK/geometry/resource primitives remain unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
from fractions import Fraction
import argparse
import os
import sys
from . import native848_task_telemetry_optout_v1 as telemetry_optout
from . import native848_bulk_sibling_gate_v5 as identity_gate
import hashlib
import shutil
import time
import traceback
import numpy as np
from . import native848_reset68_labeled_plan_v1 as api
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


PENDING_COVERAGE='complete_manifest_backed_plant_anatomy_pending_native_census'
VERIFIED_COVERAGE='complete_manifest_backed_plant_anatomy_native_census_verified'


def population_metadata(split,*,native):
    require(split in ('train','validation','test'),'Explicit family-grouped split required')
    return dict(dataset_split=split,annotation_coverage=VERIFIED_COVERAGE if native else PENDING_COVERAGE,
        native_population_census_verified=native,no_query9mm_acceptance_pending=True,
        filtered_scene_annotation_export_eligible=False,render_budget_diagnostic_replay=True,training_export_eligible=False)


def require_scope(value,*,native):
    require(all(value.get(k)==v for k,v in population_metadata(value.get('dataset_split'),native=native).items()),
        'Explicit full144 labeled-population scope required')
    require(value.get('training_approved') is False and type(value.get('accepted_training_increment')) is int
        and value['accepted_training_increment']==0,'Capture must carry zero acceptance and no training approval')
    require('split' not in value,'Use explicit full-population dataset_split rather than filtered split')


def require_native_census(census,cpu):
    identity=deepcopy(census);saved=identity.pop('deterministic_census_sha256')
    require(saved==fingerprint(identity),'Native population census fingerprint differs')
    require(census['schema']=='greenhouse.native848_fully_labeled_census.v1'
        and census['dataset_split']==cpu['dataset_split'],'Typed same-split native population census required')
    require(census['deterministic_census_sha256']==cpu['deterministic_census_sha256']
        and census['active_counts']['component_plants']==144 and census['active_counts']['backdrop_instances']==0
        and len(census['all_plant_roots'])==144 and census['removed_roots']==[]
        and census['complete_active_plant_anatomy'] is True,
        'All144 retained slots and complete manifest-backed anatomy must match new native census')


def capture(app, output, plan_path, plan, checked):
    import omni.usd
    import omni.replicator.core as rep
    from .native848_fully_labeled_scene_v1 import prepare_native_scene
    from .native848_original_direct_worker_v2 import apply_cached_pose
    from .native848_fast_profile_benchmark_v1 import freeze_and_prove_clock_safety
    from .capture_pilot import source_hashes
    from .capture_scene import calibration, target_world_geometry
    from .native848_fully_labeled_coverage_v1 import component_catalogue
    from .capture_sensor import LEGACY_RESOLUTION, calibration_for_native_resolution
    from .native_greenhouse_pair import assert_same_camera
    from .native848_pair_audit_v2 import world_from_row
    from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
    from .capture_scene import StaticSceneMonitor
    from .review_camera import HEAD_CAMERA

    started = time.perf_counter()
    anchor, profile = checked['anchor'], checked['profile']
    records = checked['records']
    scene = prepare_native_scene(app, anchor, checked['scene_policy'])
    scene_ready=time.perf_counter()
    require_scope(plan,native=False)
    cpu=checked['cpu_scene_preflight'];require_scope(cpu,native=False)
    require_native_census(scene['scene_policy_evidence'],cpu)
    save_json(output/'full_scene_census.json',scene['scene_policy_evidence'])
    stage, robot, settings = scene['stage'], scene['robot'], scene['settings']
    render_settings = apply_profile(rep, settings, profile)
    if profile['schema']==api.ORIGINAL_PROFILE_SCHEMA:
        clock_proof=dict(schema='greenhouse.original848_bulk_paused_clock.v1',delta_time_seconds=0,
            scheduler_clock_policy=api.PAUSED_CLOCK_POLICY,original_scene_setup_unchanged=False,full_original_slot_population_retained=True,
            background_geometry_replaced_with_manifest_backed_labeled_plants=True,
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
            source_family=anchor['source_family'], source_pose_split=checked['capture_split'], original_variant=scene['original_variant'])
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
        warmup_started=time.perf_counter()
        apply_cached_pose(stage, robot, records[0])
        reset = omni.usd.get_context().reset_renderer_accumulation()
        for n in profile['warmup_steps']:
            _, evidence = request_once(rep, writer, n, profile['delta_time_seconds'])
            warmup_requests.append(evidence)
        warmup_finished=time.perf_counter()
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
        context = dict(**population_metadata(checked['capture_split'],native=True), accepted_training_increment=0, schema=api.CONTEXT_SCHEMA, plan_path=str(plan_path.resolve()), plan_sha256=sha256(plan_path),
            profile_evidence=plan['profile_evidence'], diagnostic_profile=api.PROFILE,
            per_request_subframe_schedule=[item['requested_subframes'] for item in plan['schedule']], source_bindings=bindings, source_bindings_sha256=binding_hash,
            scene_revision=scene_revision, static_baseline=monitor.baseline, anchor=anchor,
            process_startup_timing=checked['process_startup_timing'],pose_anchor_sources=checked['pose_anchor_sources'],
            scene_variants=scene['variants'],source_family_identity_policy='source_plant_id_and_split_group_not_instance_variant_id',
            cache_path=plan['cache_path'], cache_sha256=plan['cache_sha256'],
            source_reports_path=str((output/'anatomy_reports.json').resolve()),
            source_reports_sha256=sha256(output/'anatomy_reports.json'), source_family=anchor['source_family'],
            source_pose_split=checked['capture_split'], scene_counts=scene['counts'], lighting=scene['old_manifest']['lighting'],
            renderer=settings.get('/rtx/rendermode'), render_settings=render_settings,
            catalogue_path=str((output/'catalogue.json').resolve()), catalogue_sha256=sha256(output/'catalogue.json'),
            targets=targets, geometry_proofs_directory=str(proof_directory.resolve()),
            warmup_evidence_path=str((warmup_folder/'request_evidence.json').resolve()),
            warmup_evidence_sha256=sha256(warmup_folder/'request_evidence.json'),
            geometry_proof_policy='every_actual_robot_world_transform_checked_including_head',
            scene_policy=checked['scene_policy'],
            full_scene_census=scene['scene_policy_evidence'],
            full_scene_census_path=str((output/'full_scene_census.json').resolve()),
            full_scene_census_sha256=sha256(output/'full_scene_census.json'),
            cpu_scene_preflight=checked['cpu_scene_preflight_pin'],
            task_input_policy=dict(query_input=False,clean_full_rgb=True,aligned_depth=True),
            training_approved=False)
        require_scope(context,native=True)
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
                decision.update(pose_apply_seconds=applied-tick,geometry_screen_seconds=time.perf_counter()-applied)
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
                budget=item['requested_subframes']
                require(budget in (6,8) and profile['request_subframes']==8,'Exact diagnostic6/8 override of reference8 required')
                payload, request = request_once(rep,writer,budget,profile['delta_time_seconds'])
                decision['native_requests'] = 1
                decision['native_request_seconds']=request['elapsed_seconds']
                request.update(diagnostic_profile=api.PROFILE,reference_profile_request_subframes=8,diagnostic_requested_subframes=budget,
                    settings=render_settings,render_settings=render_settings,reset_api=profile['reset_api'],
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
                observation = dict(**population_metadata(checked['capture_split'],native=True), accepted_training_increment=0, schema=api.OBSERVATION_SCHEMA,observation_id=name,capture_index=capture_index,
                    source_pose_id=item['source_pose_id'],target_id=rec['target_id'],source_family=anchor['source_family'],
                    target_id_scope='private_pose_provenance_not_model_input_or_unique_label',
                    source_pose_split=checked['capture_split'],capture_role='reset6_8_ABA_diagnostic',profile=api.PROFILE,reference_profile=profile['profile'],diagnostic_phase=item['phase'],
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
                require_scope(observation,native=True)
                wait = sink.submit(observation,rgb,depth,ids,valid,
                    np.asarray(payload['rgb'])[:,:,3].copy())
                decision.update(state='queued_lossless_callback_pending_cpu_admission',queue_submit_seconds=wait,
                    total_producer_seconds=time.perf_counter()-tick)
                print('FULLPOP848_QUEUED',capture_index,name,round(request['elapsed_seconds'],4),flush=True)
            except BaseException:
                decision.update(state='failed_preserved_no_retry',error=traceback.format_exc(),
                    last_native_request=getattr(writer,'last_request_evidence',None))
                raise
            finally:
                decisions.append(decision)
                save_json(output/('decision_'+name+'.json'),decision)
        writer_drain_started=time.perf_counter()
        committed = sink.close()
        writer_drained=time.perf_counter()
        require(source_hashes(stage)==full_pins,'Actual scene asset population changed')
        verify_bindings(bindings)
        require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()),'Source layer dirtied')
        require(writer.capture_error is None and not writer.active,'Unaccounted native callback')
        require(sha256(context_path)==context_hash,'Immutable context changed')
        manifest = dict(**population_metadata(checked['capture_split'],native=True),schema='greenhouse.native848_reset68_labeled_manifest.v1',context_path=str(context_path.resolve()),
            context_sha256=context_hash,observations=committed,decisions=decisions,geometry_proofs=list(proofs.values()),
            renderer_mappings=list(mapping_files.values()),native_request_count=writer.request_index,
            native_callback_count=writer.sequence,warmup_request_count=len(warmup_requests),
            production_native_requests=sum(r['native_requests'] for r in decisions),committed_frames=len(committed),
            source_assets_unchanged=True,queue_peak_uncompressed_bytes=sink.peak_bytes,
            geometry_cache=geometry.diagnostics(),elapsed_seconds=time.perf_counter()-started,
            timing_breakdown=dict(scene_setup_seconds=scene_ready-started,post_scene_setup_seconds=warmup_started-scene_ready,
                warmup_seconds=warmup_finished-warmup_started,pose_apply_seconds=sum(r.get('pose_apply_seconds',0) for r in decisions),
                geometry_screen_seconds=sum(r.get('geometry_screen_seconds',0) for r in decisions),
                native_request_seconds=sum(r.get('native_request_seconds',0) for r in decisions),
                writer_drain_seconds=writer_drained-writer_drain_started,source_validation_and_close_seconds=time.perf_counter()-writer_drained),
            stage_count=1,render_product_count=1,automatic_retries=False,training_approved=False,
            accepted_training_increment=0)
        require_scope(manifest,native=True)
        save_json(output/'batch_manifest.json',manifest)
        return dict(**population_metadata(checked['capture_split'],native=True),schema='greenhouse.native848_reset68_labeled_capture.v1',state=api.RESULT_STATE,manifest_path=str((output/'batch_manifest.json').resolve()),
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


OPT_OUT_CONTRACT=dict(schema='greenhouse.native848_serial_telemetry_optout_contract.v1',
    environment_override={telemetry_optout.ENV_KEY:'1'},extra_args=[telemetry_optout.DISABLE_ARG],
    keep_telemetry_extension_graph=True,process_resource_checks_unchanged=True,
    settings_evidence_is_not_native_qualification=True)
HELPER_SHA256='e329884fc54ff5daf4b2ee0cae36842ef1a19b455c923cf005f31295bb1e64e6'
BASE_WORKER_SHA256='54d53f64d69642b128b1201c715de25e38105eb04d123fca84c8a3828ac45fc9'
INSTALLED_STARTUP_BINDINGS={'D:\\isaac-sim-6.0.1\\exts\\isaacsim.simulation_app\\isaacsim\\simulation_app\\simulation_app.py': '36cd69cf4874eaab93962d7d5f8ac88c6450eb8eae8aa4ba29fc390fcdbea849', 'D:\\isaac-sim-6.0.1\\apps\\isaacsim.exp.base.kit': '78b5c4e6c177a96289ded1644ca74db8385c11672c7ed774c7b7b0bc3195f400', 'D:\\isaac-sim-6.0.1\\apps\\isaacsim.exp.base.python.kit': 'd280f12eb97d68a59bd0145185e6b8098e6901f50e823cd4cb4281aefb781a0a', 'D:\\isaac-sim-6.0.1\\exts\\isaacsim.core.telemetry\\config\\extension.toml': 'ddf17f55091ec1c771052256e0dbc9a69d9decb32aed8ee4d071ddb0a3ee96e4', 'D:\\isaac-sim-6.0.1\\exts\\isaacsim.core.telemetry\\isaacsim\\core\\telemetry\\extension.py': '009c55e4154b2b079521906dc298d6e8c6e65197574568fbc2a9d92f5b0d4d05', 'D:\\isaac-sim-6.0.1\\kit\\kernel\\py\\omni\\kit\\app\\_impl\\telemetry_helpers.py': 'cdfdea83b2937e8b28806cc8d4996fa067dcb72977d09f7133b7ddeaa82c55bb', 'D:\\isaac-sim-6.0.1\\kit\\kernel\\py\\omni\\structuredlog\\_structuredlog.pyi': 'ef6a4d640528ff108e915e9612f4a26aa046075cb5004e8ebb2ab7faadff70a5', 'D:\\isaac-sim-6.0.1\\extscache\\omni.kit.telemetry-0.5.2+f9bf0dda.wx64.r.cp312\\config\\extension.toml': '2f9103e22f80cfb48dc21a18483d745cb4b99226f32957852b4089af41b6f9de', 'D:\\isaac-sim-6.0.1\\extscache\\omni.kit.telemetry-0.5.2+f9bf0dda.wx64.r.cp312\\docs\\KitTelemetry.md': 'b6dc021fbba86f992ca88e4b661e5171a747f683b884198efb9678a31f049e62'}


def optout_bindings():
    bindings={**INSTALLED_STARTUP_BINDINGS,
        str(Path(telemetry_optout.__file__).resolve()):HELPER_SHA256,
        str(Path(__file__).with_name('native848_pilot_worker_v1.py').resolve()):BASE_WORKER_SHA256,
        str(Path(__file__).with_name('native848_pilot_worker_v2.py').resolve()):'61e09bb752841491220c15fd755fc572195636049a3a2bd129e6c2afaa770891',
        str(Path(__file__).with_name('native848_fully_labeled_coverage_v1.py').resolve()):sha256(Path(__file__).with_name('native848_fully_labeled_coverage_v1.py')),
        str(Path(__file__).resolve()):sha256(__file__),
        str(Path(identity_gate.__file__).resolve()):'3766afb02212dfa0bb432bb61c6dc9b3e45d7fbd259d778be6e7bc2b9a7150d4'}
    verify_bindings(bindings);return bindings


def native_command(args):
    return ['D:/isaac-sim-6.0.1/python.bat','-B','-u','-m','sim_data.native848_reset68_labeled_worker_v1',
        '--plan',str(args.plan.resolve()),'--plan-sha256',args.plan_sha256,
        '--scene-preflight',str(args.scene_preflight.resolve()),'--scene-preflight-sha256',args.scene_preflight_sha256,
        '--output',str(args.output.resolve())]


def record_native_identity(args):
    rows=identity_gate.snapshot();native=identity_gate.one(rows,os.getpid())
    command=identity_gate.one(rows,native['ParentProcessId']);owner=identity_gate.one(rows,command['ParentProcessId'])
    expected=native_command(args);expected[0]=str(Path(expected[0]))
    identity_gate.command_child(command,owner,expected)
    identity_gate.native_child(native,command,expected[1:])
    return dict(schema='greenhouse.native848_reset68_labeled_worker_identity.v1',owner_identity=owner,
        command_identity=command,native_identity=native,command=expected,
        same_snapshot_identity_checks=True,identity_snapshot=[owner,command,native],source_bindings=optout_bindings())


def validate_native_identity(value,command,*,owner_identity=None):
    require(value['schema']=='greenhouse.native848_reset68_labeled_worker_identity.v1'
        and value['same_snapshot_identity_checks'] is True and value['command']==command
        and command[1:5]==['-B','-u','-m','sim_data.native848_reset68_labeled_worker_v1'],
        'Exact fully labeled native module identity required')
    snapshot=value['identity_snapshot']
    require(len(snapshot)==3 and len({r['ProcessId'] for r in snapshot})==3,'Exact three-process identity snapshot required')
    for key in ('owner_identity','command_identity','native_identity'):
        require(identity_gate.same_identity(identity_gate.one(snapshot,value[key]['ProcessId']),value[key]),
            'Recorded actual process identity changed from startup snapshot')
    if owner_identity is not None:
        require(identity_gate.same_identity(value['owner_identity'],owner_identity),'Worker belongs to another actual serial owner')
    identity_gate.command_child(value['command_identity'],value['owner_identity'],command)
    identity_gate.native_child(value['native_identity'],value['command_identity'],command[1:])
    return value


def runtime_evidence(phase,config,native_pin):
    import carb.settings
    return dict(schema='greenhouse.native848_reset68_labeled_runtime_evidence.v1',phase=phase,
        contract=OPT_OUT_CONTRACT,config=config,native_identity=native_pin,
        settings=telemetry_optout.validate_runtime(carb.settings.get_settings().get,environment=os.environ),
        source_bindings=optout_bindings())


def validate_optout_capture(capture,value,context):
    capture=Path(capture).resolve();bindings={}
    require_scope(value,native=True);require_scope(context,native=True)
    require(value['telemetry_optout_contract']==OPT_OUT_CONTRACT,'Opt-out capture contract differs')
    def read(spec,name):
        path=Path(spec['path']).resolve()
        require(path==capture/name and sha256(path)==spec['sha256'],'Changed exact opt-out evidence')
        bindings[str(path)]=spec['sha256'];return read_json(path)
    identity=read(value['native_identity'],'native_identity.json')
    validate_native_identity(identity,identity['command'])
    expected=optout_bindings()
    for phase in ('start','end'):
        evidence=read(value['telemetry_optout_'+phase],'telemetry_optout_'+phase+'.json')
        settings=evidence['settings']
        require(evidence['schema']=='greenhouse.native848_reset68_labeled_runtime_evidence.v1'
            and evidence['phase']==phase and evidence['contract']==OPT_OUT_CONTRACT
            and evidence['native_identity']==value['native_identity'] and evidence['source_bindings']==expected,
            'Exact runtime opt-out provenance required')
        require(settings['schema']==telemetry_optout.SCHEMA and settings['structured_log_enabled'] is False
            and settings['environment_override']==OPT_OUT_CONTRACT['environment_override']
            and settings['telemetry_extension_exclusion_requested'] is False
            and settings['only_settings_verified'] is True and settings['transmitter_absence_verified'] is False
            and settings['capture_qualified'] is False and settings['process_or_resource_exemption'] is False,
            'Opt-out runtime settings changed or native qualification fabricated')
        if phase=='start':start=evidence
        else:require(evidence['config']==start['config'],'Runtime launch configuration changed')
    require(identity['source_bindings']==expected,'Actual identity startup source binding differs')
    require(start['config']['extra_args'][-1:]==[telemetry_optout.DISABLE_ARG]
        and telemetry_optout.EXCLUDE_ARG not in start['config']['extra_args'],'Default extension graph must be retained')
    for path,digest in {**expected,**{str((capture/name).resolve()):sha256(capture/name) for name in ('native_identity.json','telemetry_optout_start.json')}}.items():
        require(context['source_bindings'].get(path)==digest,'Capture context omitted exact opt-out startup provenance')
    bindings.update(expected);return dict(source_bindings=bindings,native_identity=identity,start=start)


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
    require(plan['schema']==api.SCHEMA and not output.exists(),'New fully labeled population output required')
    require_scope(plan,native=False)
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
        startup_bindings=optout_bindings()
        require(os.environ.get(telemetry_optout.ENV_KEY)=='1','Required child-only early telemetry override missing')
        identity=record_native_identity(args)
        save_json(output/'native_identity.json',identity)
        native_pin=dict(path=str((output/'native_identity.json').resolve()),sha256=sha256(output/'native_identity.json'))
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
        config=telemetry_optout.prepare_config(config,argv=list(argv) if argv is not None else sys.argv[1:],environment=os.environ)
        app_constructor_started=time.perf_counter()
        app=SimulationApp(config)
        app_constructor_seconds=time.perf_counter()-app_constructor_started
        save_json(output/'telemetry_optout_start.json',runtime_evidence('start',config,native_pin))
        plan_check_started=time.perf_counter()
        checked=api.check(plan)
        checked['process_startup_timing']=dict(app_constructor_seconds=app_constructor_seconds,plan_authentication_seconds=time.perf_counter()-plan_check_started)
        cpu=read_json(args.scene_preflight);require_scope(cpu,native=False)
        require(cpu['schema']=='greenhouse.native848_reset68_labeled_cpu_preflight.v1'
            and cpu['plan_sha256']==args.plan_sha256 and cpu['scene_policy']==checked['scene_policy']
            and cpu['all_selected_pose_screens_passed'] is True and cpu['source_full_scene_pose_checks_reused_exactly'] is True
            and cpu['selected_sample_ids']==plan['selected_sample_ids']
            and cpu['pose_anchor_sources']==checked['pose_anchor_sources'], 'Exact selected CPU scene preflight required')
        checked['cpu_scene_preflight']=cpu
        checked['cpu_scene_preflight_pin']=dict(path=str(args.scene_preflight.resolve()),sha256=args.scene_preflight_sha256)
        for path,pin in cpu['source_bindings'].items():
            require(path not in checked['source_bindings'] or checked['source_bindings'][path]==pin,'CPU source binding conflict')
            checked['source_bindings'][path]=pin
        checked['source_bindings'][str(args.scene_preflight.resolve())]=args.scene_preflight_sha256
        for path,digest in {**startup_bindings,**{str((output/name).resolve()):sha256(output/name) for name in ('native_identity.json','telemetry_optout_start.json')}}.items():
            require(path not in checked['source_bindings'] or checked['source_bindings'][path]==digest,'Opt-out startup source conflict')
            checked['source_bindings'][path]=digest
        verify_bindings(checked['source_bindings'])
        result=capture(app,output,args.plan,plan,checked)
        require(sha256(args.plan)==args.plan_sha256,'Bulk plan changed during capture')
        save_json(output/'telemetry_optout_end.json',runtime_evidence('end',config,native_pin))
        result.update(telemetry_optout_contract=OPT_OUT_CONTRACT,native_identity=native_pin,
            **{'telemetry_optout_'+phase:dict(path=str((output/('telemetry_optout_'+phase+'.json')).resolve()),sha256=sha256(output/('telemetry_optout_'+phase+'.json'))) for phase in ('start','end')})
        result['plan_sha256']=args.plan_sha256
        save_json(output/'result.json',result);succeeded=True
    except BaseException:
        save_json(output/'failure.json',dict(state='native848_fully_labeled_failed',error=traceback.format_exc(),
            automatic_retries=False,training_approved=False,accepted_training_increment=0))
        raise
    finally:
        if app is not None:
            app.close(exit_code=0 if succeeded else 1)


if __name__=='__main__':
    main()
