"""Capture generated full144 plants with authenticated far packing and reset6.

Original source camera profile remains separately pinned. All143 background
shapes/materials remain; the complete original/generated collision stage is
independent of the packed render stage.

Replicator RGB, optical-Z and instance IDs share one callback. Geometry and
visibility are evaluated afresh; this capture stage never approves training.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import argparse
import os
import sys
import time
import traceback
import numpy as np

from .capture_contract import fingerprint, jsonable
from .dataset_review import read_json, require, verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import FrameSink, save_json
from . import native848_bulk_worker_v1 as primitives
from . import native848_fully_labeled_worker_v3 as production
from . import native848_reset6_production_plan_v1 as render_api

MODULE = 'sim_data.native848_generated_farmerge_capture_v1'
REQUEST_SCHEMA = 'greenhouse.generated_farmerge_capture_request.v1'
RESULT_SCHEMA = 'greenhouse.generated_farmerge_capture_result.v1'
FINITE_RECIPE_SCHEMA = 'greenhouse.native848_finite_farmerge_population.v1'
FLAGS = dict(training_approved=False, accepted_training_increment=0,
             single_answer_verified=False, morphology_diversity_approved=False)


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path),sha256=sha256(path))


def bound(spec):
    path = Path(spec['path']).resolve()
    require(path.is_file() and sha256(path)==spec['sha256'], 'Changed input: '+str(path))
    return read_json(path)


def implementation_bindings():
    names = ['native848_generated_morphology_scene_v2.py', 'native848_morphology_diagnostic_v1.py',
             'native848_fully_labeled_scene_v1.py', 'native848_original_direct_worker_v2.py',
             'native848_bulk_io_v1.py', 'static_geometry_parts_cache_v1.py', 'static_guard.py',
             'prepare_controlled_views_v2.py', 'native848_bulk_workspace_v1.py',
             'native848_farmerge_overlay_v1.py', 'native848_farmerge_finite_population_v1.py',
             'native848_farmerge_annotation_v1.py', 'native848_farmerge_pair_scene_v1.py',
             'native848_reset6_production_plan_v1.py', 'native848_farmerge_pair_worker_v1.py']
    return {**production.optout_bindings(), str(Path(__file__).resolve()):sha256(__file__),
            **{str(Path(__file__).with_name(n).resolve()):sha256(Path(__file__).with_name(n)) for n in names}}


def check_request(request):
    require(request['schema']==REQUEST_SCHEMA and type(request['max_frames']) is int
            and 1<=request['max_frames']<=128 and request['training_approved'] is False,
            'Exact bounded unapproved capture request required')
    require(request['implementation_bindings']==implementation_bindings(), 'Capture implementation changed')
    verify_bindings(request['implementation_bindings'])
    render_api.check_profile(bound(request['production_profile']))
    recipe=bound(request['farmerge_recipe'])
    require(recipe['schema']==FINITE_RECIPE_SCHEMA and recipe['foreground_runtime_geometry_must_be_kept'] is True
            and recipe['unchanged_packed_background_count']==137 and len(recipe['whole_original_background_roots'])==6,
            'Reviewed finite six-whole-plant packing required')
    candidates=bound(request['candidates']); physical=bound(request['physical_geometry_evidence'])
    require(physical['status']=='qualified_proximal_mesh_identity'
            and physical['proximal_mesh_exact_30mm'] is True
            and physical['physical_9mm_source_correspondence'] is True
            and physical['shared_junction_preserved'] is True,
            'Fresh proximal mesh and junction correspondence required before capture')
    require(physical['qualification']==candidates['qualification']
            and candidates['target_component_id'] in physical['modified_radius_qualified_component_ids'],
            'Physical evidence does not qualify this generated target')
    verify_bindings(physical['source_bindings'])
    require(candidates['schema']=='greenhouse.controlled_mounted_camera_candidates.v1'
            and 1<=len(candidates['records'])<=128 and request['max_frames']==len(candidates['records']), 'Finite exact camera schedule required')
    require(physical['all_controlled_targets_qualified'] is True
            and physical['primary_component_id']==candidates['target_component_id'],
            'All modified petioles and the nominated target must have physical evidence')
    verify_bindings(candidates['source_bindings'])
    require(len({r['sample_id'] for r in candidates['records']})==len(candidates['records'])
            and len({fingerprint(r['calibration']) for r in candidates['records']})==len(candidates['records']), 'Repeated camera identity or physical calibration')
    anchors=[bound(record['anchor']) for record in candidates['records']]
    scene_paths={str(Path(anchor['source_collection_plan']).resolve()) for anchor in anchors}
    require(len(scene_paths)==1,'Selected cameras must share their authenticated scene collection plan')
    scene_path=next(iter(scene_paths))
    scene_pins=[dict(path=scene_path,sha256=anchor['source_bindings'][scene_path]) for anchor in anchors]
    require(all(spec==scene_pins[0] for spec in scene_pins),'Selected scene plan pins differ')
    scene_plan=bound(scene_pins[0]);generator_plan=bound(candidates['source_plan'])
    require(scene_plan['family_assignments']==generator_plan['family_assignments']
            and scene_plan['source_bindings_sha256']==generator_plan['source_bindings_sha256'],
            'Scene and generator plans differ in plant assets or donor splits')
    return candidates


def validate_pose(record, anchor):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from .native_view_pose import bounded_reference_root
    reference=anchor['expected_robot_snapshot']
    require(record['source_family']==anchor['source_family'] and anchor['split']=='train', 'Pose donor mismatch')
    require(record['camera_to_head_column_vectors']==reference['camera_to_head_column_vectors'], 'Camera mount changed')
    require(record['joint_degrees'].keys()==reference['joint_degrees'].keys(), 'Robot joints omitted')
    for name,value in reference['joint_degrees'].items():
        if not name.startswith('head_'):
            require(record['joint_degrees'][name]==value, 'Arm/torso joint changed')
    root=np.asarray(record['robot_root_to_world_usd_row_vectors']).T
    nominal=record['pilot_9mm_geometry']['nominal']['world_m']
    require(np.allclose(bounded_reference_root(reference,record['requested_spec'],nominal),root,atol=1e-9,rtol=0),
            'Robot body exceeds authenticated bound')
    camera=root@Rby1Kinematics().all_link_transforms(record['joint_degrees'])['link_head_2']@np.asarray(record['camera_to_head_column_vectors'])
    require(np.allclose(camera.T,record['calibration']['camera_to_world_usd_row_vectors'],atol=1e-9,rtol=0), 'Camera FK differs')


def authenticate_anchors(records):
    """Authenticate each exact source anchor once; validate every actual pose."""
    from .native848_pilot_plan_v1 import raw_anchor
    verified={};anchors=[]
    for record in records:
        anchor=bound(record['anchor'])
        key=(str(Path(anchor['source_capture']).resolve()),anchor['source_sample'],record['anchor']['sha256'])
        if key not in verified:
            require(raw_anchor(anchor['source_capture'],anchor['source_sample'])==anchor,'Original robot anchor changed')
            verified[key]=anchor
        else:require(verified[key]==anchor,'Same anchor pin resolved to different contents')
        validate_pose(record,anchor);anchors.append(anchor)
        require(anchor['original_variant']==anchors[0]['original_variant'],'All views must share foreground placement')
    return anchors


def make_request(candidates_pin,physical_pin,recipe_pin,production_profile_pin):
    request=dict(schema=REQUEST_SCHEMA,candidates=candidates_pin,physical_geometry_evidence=physical_pin,
        farmerge_recipe=recipe_pin,production_profile=production_profile_pin,
        max_frames=len(bound(candidates_pin)['records']),implementation_bindings=implementation_bindings(),**FLAGS)
    check_request(request);return request


def capture(app, request, candidates, output, *, request_pin):
    # USD must bind to Kit's ABI after SimulationApp startup, never pip's pxr.
    from . import native848_farmerge_overlay_v1 as overlay_api
    import omni.usd
    import omni.replicator.core as rep
    from . import native848_generated_morphology_scene_v2 as generated
    from .native848_fully_labeled_scene_v1 import prepare_native_scene, policy
    from .native848_pilot_plan_v1 import raw_anchor
    from .native848_original_direct_worker_v2 import apply_cached_pose
    from .native848_morphology_diagnostic_v1 import diagnostic_catalogue
    from .native848_bulk_plan_v1 import check_profile
    from .native848_bulk_workspace_v1 import BulkWorkspace
    from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
    from .static_guard import StaticSceneMonitor
    from .capture_pilot import source_hashes
    from .review_camera import HEAD_CAMERA
    from .capture_scene import calibration
    from .capture_sensor import calibration_for_native_resolution
    from .native_greenhouse_pair import assert_same_camera
    records=candidates['records']
    anchors=authenticate_anchors(records)
    source_profile=bound(candidates['profile']);check_profile(source_profile)
    profile=bound(request['production_profile']);render_api.check_profile(profile)
    require(profile['render_settings']==source_profile['render_settings'] and profile['warmup_steps']==[8]*7
            and profile['request_subframes']==6 and profile['delta_time_seconds']==source_profile['delta_time_seconds']==0,
            'Only the proven request-subframe change is permitted')
    scene=prepare_native_scene(app,anchors[0],policy('train'))
    stage,robot,settings=scene['stage'],scene['robot'],scene['settings']
    population=generated.replace_controlled_foreground(stage,scene,
        qualification_pin=candidates['qualification'], donor_plan_pin=candidates['source_plan'], anchor=anchors[0])
    catalogue=diagnostic_catalogue(stage,population,scene['reports'])
    census=deepcopy(population['scene_policy_evidence'])
    require(len(census['all_plant_roots'])==144 and census['unchanged_background_count']==143
            and len(catalogue)==population['counts']['components'], 'Native full population differs')
    census['native_population_census_verified']=True
    census['deterministic_census_sha256']=fingerprint({k:v for k,v in census.items() if k!='deterministic_census_sha256'})
    population['scene_policy_evidence']=census
    overlay=overlay_api.install_overlay(scene,population,catalogue,request['farmerge_recipe'])
    save_json(output/'render_census.json',overlay['render_census'])
    save_json(output/'census.json',census);save_json(output/'catalogue.json',catalogue)
    save_json(output/'generated_report.json',population['generated_report'])
    from .native848_farmerge_pair_worker_v1 import apply_profile as apply_proven_profile
    render_settings=apply_proven_profile(rep,settings,profile)
    bindings={**candidates['source_bindings'],**population['source_bindings'],**overlay['source_bindings'],**source_hashes(stage),**request['implementation_bindings']}
    morphology=population['foreground_identity']['morphology_id']
    geometry_source=dict(manifest=candidates['manifest'],qualification=candidates['qualification'],
        donor_source_family=candidates['source_family'],source_split='train',
        physical_geometry_evidence=request['physical_geometry_evidence'])
    variants=deepcopy(population['variants'])
    for v in variants:v.setdefault('geometry_source_id',v['source_plant_id'])
    context=dict(schema='greenhouse.native848_generated_farmerge_population_context.v1',dataset_split='train',
        native_population_census_verified=True,original_143_backgrounds_preserved=True,
        full_scene_census=census,scene_variants=variants,geometry_sources={morphology:geometry_source},
        source_bindings=bindings,source_collection_plan=census['source_collection_plan'],
        generator_source_collection_plan=candidates['source_plan'],
        native_resolution=[848,408],catalogue=pin(output/'catalogue.json'),profile=request['production_profile'],
        renderer_settings=render_settings,production_profile=request['production_profile'],
        source_pose_profile=candidates['profile'],render_census=pin(output/'render_census.json'),
        farmerge_recipe=request['farmerge_recipe'],coarse_pixels_never_claim_individual_component_visibility=True,**FLAGS)
    holds=[];requests=[];committed=[]
    with ExitStack() as cleanup:
        geometry=overlay_api.OriginalCollisionBridge(overlay);cleanup.callback(geometry.close)
        workspace=BulkWorkspace();cleanup.callback(workspace.finish)
        product=rep.create.render_product(HEAD_CAMERA,(848,408));cleanup.callback(product.destroy)
        writer=primitives.make_bulk_writer(rep,None);writer.attach([product]);cleanup.callback(writer.detach)
        sink=FrameSink(output/'frames',max_frames=8,max_bytes=256*2**20,workers=2);cleanup.callback(sink.close)
        warmed=False;previous=None;monitor=None;frozen_sources=None
        for record in records:
            if len(committed)>=request['max_frames']:break
            pose,cal=apply_cached_pose(stage,robot,record)
            screen=geometry.check(record,pose,cal)
            if not screen['passed']:
                holds.append(dict(sample_id=record['sample_id'],reason='whole_robot_scene_collision',screen=screen));continue
            pose['visual_bound_screen']=screen
            bounds=overlay_api.coverage.robot_bounds(dict(calibration=cal,robot_snapshot=pose),workspace)
            coarse_bounds=overlay_api.check_coarse_bounds(overlay,bounds)
            # The point is reconstructed from the current generated manifest, never inherited.
            from .native848_all_petiole_9mm_v2 import geometry_9mm
            from pxr import UsdGeom
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(census['foreground_root'])),float)
            geo=geometry_9mm(population['generated_report'],candidates['target_component_id'],matrix,cal)
            planned=record['pilot_9mm_geometry']
            require(np.allclose(geo['oriented_centerline_world_m'],planned['oriented_centerline_world_m'],rtol=0,atol=1e-9)
                    and np.allclose(geo['nominal']['world_m'],planned['nominal']['world_m'],rtol=0,atol=1e-9)
                    and np.allclose(geo['nominal']['projected']['pixel_xy'],planned['nominal']['projected']['pixel_xy'],rtol=0,atol=1e-6),
                    'Generated camera proposal has stale geometry')
            reach=workspace.check(dict(calibration=cal,robot_snapshot=pose,
                supervision=dict(target_id=record['target_id'],nominal_world_m=geo['nominal']['world_m'])))
            if not reach['result']['workspace_passed']:
                holds.append(dict(sample_id=record['sample_id'],reason='9mm_workspace',workspace=reach));continue
            if not warmed:
                omni.usd.get_context().reset_renderer_accumulation()
                for steps in profile['warmup_steps']:
                    _,e=primitives.request_once(rep,writer,steps,profile['delta_time_seconds']);requests.append(e)
                require(overlay_api.verify_overlay(overlay)==overlay['render_census'],'Packed scene changed during warmup')
                frozen_sources=source_hashes(stage)
                for path,h in frozen_sources.items():
                    require(path not in bindings or bindings[path]==h,'Conflicting post-warmup source')
                    bindings[path]=h
                verify_bindings(bindings)
                save_json(output/'context.json',context)
                monitor=StaticSceneMonitor(stage,robot['root']);cleanup.callback(monitor.close)
                warmed=True
            before=monitor.begin()
            reset=omni.usd.get_context().reset_renderer_accumulation()
            payload,e=primitives.request_once(rep,writer,profile['request_subframes'],profile['delta_time_seconds'])
            after=monitor.token()
            rgb,depth,valid,reference,token,ids,mapping=primitives.native_freshness(payload,cal,before,after,writer.sequence,previous=previous,expect_change=previous is not None)
            previous=token
            assert_same_camera(calibration_for_native_resolution(calibration(stage),(848,408)),cal)
            require({k:jsonable(settings.get(k)) for k in render_settings}==render_settings,'Renderer settings changed')
            e.update(settings=render_settings,reset_returned_value=jsonable(reset));requests.append(e)
            sample_id=record['sample_id']
            mp=output/(sample_id+'_mapping.json');save_json(mp,dict(renderer_id_to_prim={str(k):v for k,v in mapping.items()},
                plant_ID_ownership=overlay_api.classify_mapping(overlay,mapping)))
            observation=dict(schema='greenhouse.generated_farmerge_native_observation.v1',observation_id=sample_id,sample_id=sample_id,
                morphology_id=morphology,source_family=candidates['source_family'],context=pin(output/'context.json'),
                calibration=cal,robot_snapshot=pose,mapping=pin(mp),geometry=geo,workspace=reach,
                render_census=pin(output/'render_census.json'),coarse_surface_exclusion=coarse_bounds,
                rendered_camera_params=jsonable(payload['camera_params']),request_evidence=e,
                native_payload_header=jsonable({k:v for k,v in payload.items() if k not in ('rgb','distance_to_image_plane','instance_id_segmentation')}),
                synchronization=dict(scene_unchanged_during_capture=True,dynamic_recording_supported=False,
                    reference_time=reference,freshness=token,static_before=before,static_after=after,
                    callback_sequence=writer.sequence,request_index=writer.request_index,
                    render_frame=jsonable(payload['pilot_render_frame'])),timing=dict(request_seconds=e['elapsed_seconds']),**FLAGS)
            sink.submit(observation,rgb,depth,ids,valid,np.asarray(payload['rgb'])[:,:,3].copy())
            committed.append(sample_id)
            print('CONTROLLED_FRAME_SAVED',sample_id,flush=True)
        receipts=sink.close()
        require(len(receipts)==len(committed)<=request['max_frames'] and writer.capture_error is None, 'Capture callback/write mismatch')
        if not warmed:
            frozen_sources=source_hashes(stage)
            for path,h in frozen_sources.items():
                require(path not in bindings or bindings[path]==h,'Conflicting final source')
                bindings[path]=h
            save_json(output/'context.json',context)
        require(source_hashes(stage)==frozen_sources,'Generated scene sources changed between captures')
        require(overlay_api.verify_overlay(overlay)==overlay['render_census'],'Packed scene changed after production')
        verify_bindings(bindings)
        require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Source layer dirtied')
        save_json(output/'result.json',dict(schema=RESULT_SCHEMA,state='captured_pending_complete_annotation_and_visual_review',
            request=request_pin,native_identity=pin(output/'native_identity.json'),
            frames=receipts,holds=holds,requests=requests,request_count=writer.request_index,
            morphology_id=morphology,original_143_backgrounds_preserved=True,
            render_census=pin(output/'render_census.json'),production_profile=request['production_profile'],**FLAGS))


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
        require(result['schema']==RESULT_SCHEMA and 0<=len(result['frames'])<=request['max_frames']
                and result['training_approved'] is False and not (args.output/'capture/failure.json').exists(),'Invalid native result')
        require(result['request']==pin(args.request) and result['native_identity']==pin(args.output/'capture/native_identity.json'),
                'Native result belongs to another request or process')
        for row in result['frames']:bound(row)
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
