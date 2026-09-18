"""Exactly two generated-foreground native RGB-D diagnostics; no admission/export.

An old plan authenticates only camera/body/profile provenance. Each generated
variant gets a fresh full144 scene, exact143-background invariance, collision
screen and native callback. Capsule radii remain uncertified proxy bounds.
"""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import os
import shutil
import sys
import time
import traceback
import numpy as np
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint,jsonable
from .native848_bulk_io_v1 import FrameSink,save_json
from . import native848_fully_labeled_plan_v3 as api
from . import native848_fully_labeled_worker_v3 as production
from . import native848_bulk_worker_v1 as primitives
from . import native848_generated_morphology_scene_v1 as generated_scene

REQUEST_SCHEMA='greenhouse.native848_morphology_diagnostic_request.v1'
CONTEXT_SCHEMA='greenhouse.native848_morphology_diagnostic_context.v1'
OBSERVATION_SCHEMA='greenhouse.native848_morphology_diagnostic_observation.v1'
RESULT_SCHEMA='greenhouse.native848_morphology_diagnostic_result.v1'
IDENTITY_SCHEMA='greenhouse.native848_morphology_diagnostic_worker_identity.v1'
MODULE='sim_data.native848_morphology_diagnostic_v1'
SOURCE_PLAN_SHA='a578f838cb22190e1a084b1a335e34db8355a4b7c8c0163e049e36974eb20039'
SOURCE_POSE_ID='full144_train53_reviewed28_reviewed_expand1_seed53_full_SubStem_41_2366a626c0c33aa80892_025'
SCOPE=dict(source_family='seed53_full',source_split='train',diagnostic_only=True,
    new_independent_source_family=False,morphology_diversity_approved=False,
    training_approved=False,accepted_training_increment=0,source_acceptance_inherited=False,
    capsule_radius_used_for_physical_eligibility=False,single_answer_verified=False,
    dataset_export_eligible=False)


def require_pin(pin):
    require(isinstance(pin,dict) and set(pin)=={'path','sha256'}
            and isinstance(pin['path'],str) and Path(pin['path']).is_absolute()
            and isinstance(pin['sha256'],str) and len(pin['sha256'])==64
            and all(c in '0123456789abcdef' for c in pin['sha256']),'Exact absolute path/SHA256 pin required')


def require_request_shape(request):
    require(isinstance(request,dict) and set(request)=={'schema','source_pose_plan','source_pose_id',
        'target_component_id','variants','max_frames','frames_per_variant','source_family','source_split',
        'training_approved','accepted_training_increment','implementation_bindings'},'Exact diagnostic request fields required')
    require(request['schema']==REQUEST_SCHEMA and request['max_frames']==2
            and type(request['max_frames']) is int and request['frames_per_variant']==1
            and type(request['frames_per_variant']) is int,'Exactly one frame per morphology, two total')
    require(request['source_family']=='seed53_full' and request['source_split']=='train'
            and request['target_component_id']=='SubStem_41' and request['source_pose_id']==SOURCE_POSE_ID,
            'Bounded exact donor/target/pose required')
    require(request['training_approved'] is False and type(request['accepted_training_increment']) is int
            and request['accepted_training_increment']==0,'Diagnostic cannot grant acceptance')
    require_pin(request['source_pose_plan'])
    require(request['source_pose_plan']['sha256']==SOURCE_PLAN_SHA,'Exact source pose plan required')
    variants=request['variants'];require(isinstance(variants,list) and len(variants)==2,'Exactly two variants required')
    for entry in variants:
        require(isinstance(entry,dict) and set(entry)=={'qualification','manifest'},'Exact morphology asset pins required')
        for pin in entry.values():require_pin(pin)
        require(Path(entry['qualification']['path']).parent==Path(entry['manifest']['path']).parent
                and Path(entry['qualification']['path']).name=='qualification.json'
                and Path(entry['manifest']['path']).name=='manifest.json','One exact variant directory required')
    require(len({e['qualification']['path'] for e in variants})==2
            and len({e['manifest']['sha256'] for e in variants})==2,'Repeated morphology rejected')
    require(isinstance(request['implementation_bindings'],dict) and request['implementation_bindings'],
            'Reviewed implementation bindings required')


def implementation_bindings():
    # Hash source files without importing USD-dependent modules before SimulationApp.
    names=('native848_fully_labeled_scene_v1.py','native848_original_direct_worker_v2.py',
        'static_guard.py','static_geometry_parts_cache_v1.py','native848_all_petiole_9mm_v1.py')
    paths=[Path(__file__),Path(api.__file__),Path(production.__file__),Path(primitives.__file__),
           Path(generated_scene.__file__)]+[Path(__file__).with_name(name) for name in names]
    result=production.optout_bindings()
    for path in paths:result[str(path.resolve())]=sha256(path)
    return result


def check_request(request):
    """Call only after SimulationApp owns native USD ABI; no inherited new-scene pass."""
    require_request_shape(request)
    require(request['implementation_bindings']==implementation_bindings(),'Diagnostic implementation changed')
    verify_bindings(request['implementation_bindings'])
    pp=Path(request['source_pose_plan']['path']).resolve()
    require(sha256(pp)==request['source_pose_plan']['sha256'],'Pose plan changed')
    plan=read_json(pp);checked=api.check(plan)
    matches=[r for r in checked['records'] if r['sample_id']==request['source_pose_id']]
    require(len(matches)==1,'Exact source pose record missing')
    record=matches[0];anchor=checked['anchor']
    require(record['source_family']==anchor['source_family']=='seed53_full'
            and record['source_row']['component_id']=='SubStem_41'
            and checked['capture_split']==anchor['split']=='train','Authenticated pose lineage differs')
    require(checked['profile']['schema']==api.ORIGINAL_PROFILE_SCHEMA
            and checked['profile']['warmup_steps']==[8]*7
            and checked['profile']['request_subframes']==8
            and checked['profile']['delta_time_seconds']==0,'Exact old reset8 optical profile required')
    donor_plan=Path(anchor['source_collection_plan']).resolve()
    donor_pin=dict(path=str(donor_plan),sha256=sha256(donor_plan))
    auth=[]
    for entry in request['variants']:
        require(sha256(entry['manifest']['path'])==entry['manifest']['sha256'],'Generated manifest changed')
        value=generated_scene.authenticate_morphology(entry['qualification'],donor_pin,anchor)
        require(Path(value['report']['manifest_path']).resolve()==Path(entry['manifest']['path']).resolve()
                and value['report']['manifest_sha256']==entry['manifest']['sha256'],'Generated report binding differs')
        auth.append(value)
    require(len({v['qualification']['variant_id'] for v in auth})==2
            and len({v['qualification']['generated_geometry_hash'] for v in auth})==2,'Repeated generated geometry rejected')
    return dict(plan=plan,checked=checked,record=record,donor_plan_pin=donor_pin,morphologies=auth)


def diagnostic_catalogue(stage,population,donor_reports):
    """All active components, separately typed generated geometry identity."""
    donors={r['plant_id']:r for r in donor_reports};by_root={v['plant_root']:v for v in population['variants']}
    primary=population['scene_policy_evidence']['foreground_root'];items=[]
    require(len(population['records'])==len(by_root)==144,'Complete144 catalogue required')
    for record in population['records']:
        variant=by_root[record['plant_root']]
        report=population['generated_report'] if record['plant_root']==primary else donors[variant['source_plant_id']]
        require(set(record['component_paths'])==set(report['components']),'Catalogue omitted component')
        for cid,path in record['component_paths'].items():
            prim=stage.GetPrimAtPath(path);require(prim and prim.IsActive(),'Inactive diagnostic component')
            items.append(dict(component_id=cid,prim_path=path,organ_type=report['components'][cid]['type'],
                variant_id=variant['variant_id'],source_plant_id=variant['source_plant_id'],
                split_group=variant['split_group'],geometry_source_id=report['plant_id'],
                source_geometry_modified=record['plant_root']==primary))
    require(len(items)==population['counts']['components'],'Complete census component count differs')
    return [dict(row,component_index=i) for i,row in enumerate(sorted(items,key=lambda r:r['prim_path']),1)]


def generated_locator(report,component,matrix,calibration):
    from .native848_all_petiole_9mm_v1 import geometry_9mm
    value=geometry_9mm(report,component,matrix,calibration)
    def rename(item):
        if isinstance(item,dict):
            return {('proxy_radius_upper_bound_m' if k=='radius_m' else k):rename(v) for k,v in item.items()}
        if isinstance(item,list):return [rename(v) for v in item]
        return item
    return dict(schema='greenhouse.generated_morphology_geometry_locator.v1',**SCOPE,
        nominal_arc_m=.009,geometry=rename(value),capsule_radius_policy=generated_scene.RADIUS_POLICY,
        annotation_status='geometry_locator_not_training_label',
        physical_surface_radius_qualified=False,eligible_candidate_count='not_evaluated')


def pixel_diagnostic(locator,depth,valid,ids,mapping,catalogue):
    nominal=locator['geometry']['nominal'];projected=nominal['projected'];uv=projected['pixel_xy']
    result=dict(projected=projected,native_visible_eligible_claimed=False,depth_consistency_accepted=False)
    if projected['projection_status']!='in_frame':return result
    x,y=np.floor(uv).astype(int);renderer_id=int(ids[y,x]);path=mapping.get(renderer_id)
    owners=[r for r in catalogue if path and (path==r['prim_path'] or path.startswith(r['prim_path']+'/'))]
    owner=max(owners,key=lambda r:len(r['prim_path'])) if owners else None
    measured=float(depth[y,x]) if valid[y,x] else None
    result.update(pixel_xy=[int(x),int(y)],renderer_id=renderer_id,renderer_prim_path=path,
        component_owner=owner,depth_valid=bool(valid[y,x]),native_optical_depth_m=measured,
        centerline_optical_depth_residual_m=None if measured is None else measured-projected['camera_optical_xyz_m'][2])
    return result


def capture_variant(app,output,request,prepared,index):
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom
    from .native848_fully_labeled_scene_v1 import prepare_native_scene
    from .native848_original_direct_worker_v2 import apply_cached_pose
    from .capture_pilot import source_hashes
    from .capture_scene import calibration
    from .capture_sensor import LEGACY_RESOLUTION,calibration_for_native_resolution
    from .native_greenhouse_pair import assert_same_camera
    from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
    from .static_guard import StaticSceneMonitor
    from .review_camera import HEAD_CAMERA
    started=time.perf_counter();output.mkdir()
    checked=prepared['checked'];anchor=checked['anchor'];record=prepared['record'];profile=checked['profile']
    scene=prepare_native_scene(app,anchor,checked['scene_policy'])
    population=generated_scene.replace_foreground(scene['stage'],scene,
        qualification_pin=request['variants'][index]['qualification'],donor_plan_pin=prepared['donor_plan_pin'],
        anchor=anchor,generated_report=prepared['morphologies'][index]['report'])
    stage,robot,settings=scene['stage'],scene['robot'],scene['settings']
    census=population['scene_policy_evidence']
    require(census['schema']==generated_scene.CENSUS_SCHEMA and census['unchanged_background_count']==143
            and len(census['all_plant_roots'])==144 and census['removed_roots']==[],
            'Exactly143 unchanged backgrounds plus one generated full foreground required')
    save_json(output/'generated_census.json',census)
    catalogue=diagnostic_catalogue(stage,population,scene['reports']);save_json(output/'catalogue.json',catalogue)
    save_json(output/'generated_anatomy_report.json',population['generated_report'])
    pose,cal=apply_cached_pose(stage,robot,record)
    render_settings=primitives.apply_profile(rep,settings,profile)
    primary=census['foreground_root']
    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(primary)),float)
    locator=generated_locator(population['generated_report'],request['target_component_id'],matrix,cal)
    save_json(output/'geometry_locator.json',locator)
    bindings=dict(checked['source_bindings'])
    for path,h in {**population['source_bindings'],**source_hashes(stage),**request['implementation_bindings']}.items():
        require(path not in bindings or bindings[path]==h,'Conflicting actual generated source')
        bindings[path]=h
    full_scene_sources=source_hashes(stage);verify_bindings(bindings)
    with ExitStack() as cleanup:
        geometry=StaticGeometryPartsCache(stage,robot['root'],include_generated_plants=True);cleanup.callback(geometry.close)
        screen=geometry();save_json(output/'geometry_screen.json',screen)
        require(screen['passed'],'Generated whole-robot scene collision screen failed; no production frame requested')
        pose['visual_bound_screen']=screen
        product=rep.create.render_product(HEAD_CAMERA,LEGACY_RESOLUTION);cleanup.callback(product.destroy)
        warmup=output/'warmup';warmup.mkdir()
        writer=primitives.make_bulk_writer(rep,warmup/'all_callbacks')
        writer.attach([product]);cleanup.callback(writer.detach)
        reset=omni.usd.get_context().reset_renderer_accumulation();requests=[]
        for steps in profile['warmup_steps']:
            _,evidence=primitives.request_once(rep,writer,steps,profile['delta_time_seconds']);requests.append(evidence)
        save_json(warmup/'request_evidence.json',dict(requests=requests,reset_returned_value=jsonable(reset),
            seven_control_callbacks_excluded=True,render_settings=render_settings))
        writer.debug_folder=None
        full_scene_sources=source_hashes(stage)
        for path,h in full_scene_sources.items():
            require(path not in bindings or bindings[path]==h,'Warmup source binding changed')
            bindings[path]=h
        verify_bindings(bindings)
        monitor=StaticSceneMonitor(stage,robot['root']);cleanup.callback(monitor.close)
        context=dict(schema=CONTEXT_SCHEMA,**SCOPE,generated_foreground_identity=population['foreground_identity'],
            source_pose_id=record['sample_id'],source_pose_plan=request['source_pose_plan'],
            original_pose_used_only_as_camera_body_profile_provenance=True,old_scene_acceptance_inherited=False,
            exact_same_pose_for_both_variants=True,fresh_scene_per_variant=True,
            source_bindings=bindings,source_bindings_sha256=fingerprint(bindings),anchor=anchor,
            profile_evidence=prepared['plan']['profile_evidence'],render_settings=render_settings,
            scene_variants=population['variants'],generated_census=dict(path=str((output/'generated_census.json').resolve()),
                sha256=sha256(output/'generated_census.json')),
            morphology_qualification=request['variants'][index]['qualification'],
            generated_manifest=request['variants'][index]['manifest'],catalogue=dict(path=str((output/'catalogue.json').resolve()),
                sha256=sha256(output/'catalogue.json')),static_baseline=monitor.baseline,
            geometry_screen=dict(path=str((output/'geometry_screen.json').resolve()),sha256=sha256(output/'geometry_screen.json')),
            geometry_locator=dict(path=str((output/'geometry_locator.json').resolve()),sha256=sha256(output/'geometry_locator.json')),
            native_resolution=[848,408],native_population_matches_new_census=True,
            original_143_backgrounds_preserved=True,new_capture_visibility_uniqueness_not_admitted=True)
        save_json(output/'diagnostic_context.json',context)
        before=monitor.begin()
        require({k:jsonable(settings.get(k)) for k in render_settings}==render_settings,'Renderer changed before request')
        reset=omni.usd.get_context().reset_renderer_accumulation()
        payload,request_evidence=primitives.request_once(rep,writer,profile['request_subframes'],profile['delta_time_seconds'])
        after=monitor.token()
        rgb,depth,valid,reference,token,ids,mapping=primitives.native_freshness(payload,cal,before,after,writer.sequence)
        require({k:jsonable(settings.get(k)) for k in render_settings}==render_settings,'Renderer changed during request')
        assert_same_camera(calibration_for_native_resolution(calibration(stage),LEGACY_RESOLUTION),cal)
        request_evidence.update(settings=render_settings,reset_api=profile['reset_api'],
            reset_returned_without_exception=True,reset_returned_value=jsonable(reset))
        save_json(output/'renderer_mapping.json',dict(renderer_id_to_prim={str(k):v for k,v in mapping.items()}))
        annotation=dict(**locator,native_nominal_diagnostic=pixel_diagnostic(locator,depth,valid,ids,mapping,catalogue))
        save_json(output/'diagnostic_annotation.json',annotation)
        observation=dict(schema=OBSERVATION_SCHEMA,**SCOPE,observation_id='morphology_'+str(index+1),
            capture_role='generated_geometry_native_integration_diagnostic',morphology_id=population['foreground_identity']['morphology_id'],
            context=dict(path=str((output/'diagnostic_context.json').resolve()),sha256=sha256(output/'diagnostic_context.json')),
            calibration=cal,robot_snapshot=pose,source_pose_id=record['sample_id'],
            rendered_camera_params=jsonable(payload['camera_params']),request_evidence=request_evidence,
            mapping=dict(path=str((output/'renderer_mapping.json').resolve()),sha256=sha256(output/'renderer_mapping.json')),
            diagnostic_annotation=dict(path=str((output/'diagnostic_annotation.json').resolve()),sha256=sha256(output/'diagnostic_annotation.json')),
            native_payload_header=jsonable({k:v for k,v in payload.items() if k not in ('rgb','distance_to_image_plane','instance_id_segmentation')}),
            synchronization=dict(reference_time=reference,freshness=token,static_before=before,static_after=after,
                render_frame=jsonable(payload['pilot_render_frame']),callback_sequence=writer.sequence,request_index=writer.request_index),
            timing=dict(request_seconds=request_evidence['elapsed_seconds']))
        sink=FrameSink(output/'frames',max_frames=1,max_bytes=64*2**20,workers=1);cleanup.callback(sink.close)
        sink.submit(observation,rgb,depth,ids,valid,np.asarray(payload['rgb'])[:,:,3].copy())
        committed=sink.close()
        require(len(committed)==1 and writer.request_index==8 and writer.sequence==8
                and writer.capture_error is None and not writer.active,'Exactly seven warmups plus one production callback required')
        require(source_hashes(stage)==full_scene_sources,'Generated scene assets changed during capture')
        verify_bindings(bindings)
        require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()),'Source layer dirtied')
        result=dict(schema=RESULT_SCHEMA,**SCOPE,state='one_native_morphology_diagnostic_saved_pending_review',
            committed_frames=1,observations=committed,morphology_id=population['foreground_identity']['morphology_id'],
            production_requests=1,warmup_requests=7,freshness=token,elapsed_seconds=time.perf_counter()-started,
            original_143_backgrounds_preserved=True,source_assets_unchanged=True)
        save_json(output/'result.json',result)
    # Product/writer/monitor/sink have all closed before another fresh scene.
    return dict(path=str((output/'result.json').resolve()),sha256=sha256(output/'result.json'),**result)


def native_command(args):
    return [str(Path('D:/isaac-sim-6.0.1/python.bat')),'-B','-u','-m',MODULE,'--request',str(args.request.resolve()),
            '--request-sha256',args.request_sha256,'--output',str(args.output.resolve())]


def record_native_identity(args):
    gate=production.identity_gate;rows=gate.snapshot();native=gate.one(rows,os.getpid())
    command=gate.one(rows,native['ParentProcessId']);owner=gate.one(rows,command['ParentProcessId'])
    expected=native_command(args)
    gate.command_child(command,owner,expected);gate.native_child(native,command,expected[1:])
    return dict(schema=IDENTITY_SCHEMA,owner_identity=owner,command_identity=command,native_identity=native,
                command=expected,identity_snapshot=[owner,command,native],same_snapshot_identity_checks=True)


def validate_native_identity(value,command,*,owner_identity=None):
    gate=production.identity_gate
    require(value['schema']==IDENTITY_SCHEMA and value['command']==command
            and command[1:5]==['-B','-u','-m',MODULE] and value['same_snapshot_identity_checks'] is True,
            'Exact diagnostic native command required')
    snapshot=value['identity_snapshot'];require(len(snapshot)==3 and len({r['ProcessId'] for r in snapshot})==3,'Three actual process identities required')
    for key in ('owner_identity','command_identity','native_identity'):
        require(gate.same_identity(gate.one(snapshot,value[key]['ProcessId']),value[key]),'Recorded process identity differs')
    if owner_identity is not None:require(gate.same_identity(value['owner_identity'],owner_identity),'Wrong diagnostic owner')
    gate.command_child(value['command_identity'],value['owner_identity'],command)
    gate.native_child(value['native_identity'],value['command_identity'],command[1:])
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    from .native848_data_roots_v1 import diagnostic
    from sim_physics.host_memory import preflight
    from .native_generated_pair import windows_worker_admission
    require(sha256(args.request)==args.request_sha256,'Diagnostic request changed')
    request=read_json(args.request);require_request_shape(request)
    output=diagnostic(args.output);require(not output.exists(),'Create-only diagnostic output required')
    memory=preflight();disk=shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30,'Native reserves unavailable')
    process=windows_worker_admission(None);output.mkdir(parents=True)
    save_json(output/'request.json',dict(schema=REQUEST_SCHEMA,request_path=str(args.request.resolve()),
        request_sha256=args.request_sha256,created_utc=datetime.now(timezone.utc).isoformat(),
        memory=memory,disk_free_bytes=disk,process_admission=process,**SCOPE))
    app=None;succeeded=False
    try:
        require(request['implementation_bindings']==implementation_bindings(),'Reviewed code changed')
        verify_bindings(request['implementation_bindings'])
        require(os.environ.get(production.telemetry_optout.ENV_KEY)=='1','Child-only telemetry override required')
        identity=record_native_identity(args);save_json(output/'native_identity.json',identity)
        from isaacsim import SimulationApp
        config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',
            sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false'])
        config=production.telemetry_optout.prepare_config(config,argv=list(argv) if argv is not None else sys.argv[1:],environment=os.environ)
        app=SimulationApp(config)
        import carb.settings
        def telemetry(phase):
            return dict(phase=phase,config=config,settings=production.telemetry_optout.validate_runtime(
                carb.settings.get_settings().get,environment=os.environ),**SCOPE)
        save_json(output/'telemetry_start.json',telemetry('start'))
        prepared=check_request(request)
        results=[capture_variant(app,output/('variant_'+str(i+1)),request,prepared,i) for i in range(2)]
        require(all(r['committed_frames']==1 for r in results),'Exactly two production frames required')
        require(all(results[0]['freshness'][k]!=results[1]['freshness'][k] for k in ('rgb_sha256','depth_sha256')),
                'Both generated scenes returned identical native RGB/depth')
        require(sha256(args.request)==args.request_sha256,'Request changed during native capture')
        verify_bindings(request['implementation_bindings']);save_json(output/'telemetry_end.json',telemetry('end'))
        save_json(output/'result.json',dict(schema=RESULT_SCHEMA,**SCOPE,
            state='two_native_morphology_diagnostics_saved_pending_review',committed_frames=2,
            max_frames=2,frames_per_variant=1,
            frames=[dict(morphology_id=r['morphology_id'],observation=r['observations'][0]) for r in results],
            native_request_count=16,production_requests=2,warmup_requests=14,fresh_scene_count=2,variants=results,
            request_path=str(args.request.resolve()),request_sha256=args.request_sha256,
            native_identity=dict(path=str((output/'native_identity.json').resolve()),sha256=sha256(output/'native_identity.json')),
            automatic_retries=False));succeeded=True
    except BaseException:
        save_json(output/'failure.json',dict(schema=RESULT_SCHEMA,**SCOPE,state='failed_diagnostic_preserved_no_retry',
                                            error=traceback.format_exc(),automatic_retries=False))
        raise
    finally:
        if app is not None:app.close(exit_code=0 if succeeded else 1)


if __name__=='__main__':main()

