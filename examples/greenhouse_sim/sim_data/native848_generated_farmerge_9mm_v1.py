"""Generated full144 annotation with positive exclusions for every packed petiole.

Only renderer ownership and profile dispatch differ from the generated batch
consumer. Numerical target and ambiguity predicates remain frozen.
"""
from copy import deepcopy
import ast
import inspect
from . import native848_generated_9mm_v2 as base
from . import native848_generated_9mm_v3 as source_consumer
from . import native848_farmerge_annotation_v1 as far
from . import native848_farmerge_overlay_v1 as overlay_api
from .native848_generated_9mm_v2 import (
    Path,np,CatalogueIndex,read_json,require,verify_bindings,sha256,labels,original,joint,
    ambiguity,answer_contract,geometry_9mm,component_masks,_pin,_unique,
    validate_planned_geometry,evaluate_frame,CENSUS_SCHEMA,fingerprint,
)

CONTEXT_SCHEMA='greenhouse.native848_generated_farmerge_population_context.v1'
COVERAGE_SCHEMA='greenhouse.native848_generated_farmerge_coverage.v1'
validate_collection_plan_provenance=source_consumer.validate_collection_plan_provenance


def build_inventory(context,reports,catalogue):
    # Explicit typed dispatch. The authentic saved context is never rewritten.
    tree=ast.parse(inspect.getsource(base.build_inventory))
    namespace=dict(vars(base));namespace['CONTEXT_SCHEMA']=CONTEXT_SCHEMA
    exec(compile(tree,__file__+'::typed_inventory','exec'),namespace)
    return namespace['build_inventory'](context,reports,catalogue)


def compile_runtime_groups(catalogue,logical,render,recipe,read):
    """Remap only scene-local indices, using exact source paths and face lineage."""
    require(render['schema']==overlay_api.SCHEMA and render['logical_plant_slots']==144
            and render['actual_native_coarse_groups']==137 and len(render['fully_retained_background_roots'])==6
            and render['logical_census_sha256']==fingerprint(logical)
            and render['foreground_kept_from_current_generated_population'] is True
            and render['individual_far_component_pixel_IDs_claimed'] is False,
            'Complete typed finite render census required')
    identity=deepcopy(render);saved=identity.pop('deterministic_census_sha256')
    require(saved==fingerprint(identity),'Render census fingerprint differs')
    reference=read(recipe['base_logical_census']);old_catalogue=read(recipe['base_catalogue']);selection=read(recipe['selection'])
    overlay_api.finite.background_match(logical,reference,recipe['primary_root'])
    entries={r['plant_root']:r for r in recipe['plants']};by={r['component_index']:r for r in catalogue}
    selected=[];provenance={};seen=set()
    require(render['fully_retained_background_roots']==recipe['whole_original_background_roots'], 'Whole plant partition changed')
    for group in render['native_coarse_group_catalogue']:
        root=group['plant_root'];entry=entries[root]
        require(entry['mode']=='unchanged_packed_background' and group['artifact']==entry['artifact']
                and group['face_provenance']==entry['provenance'] and root!=recipe['primary_root'], 'Packed source differs')
        _pin(entry['artifact'])
        mapping=overlay_api.remap_members(catalogue,old_catalogue,selection['selected_components'],root)
        require(mapping==group['original_to_current_component_indices']
                and sorted(mapping.values())==group['member_component_indices'], 'Scene-local component remap differs')
        require(not seen.intersection(mapping.values()),'Repeated runtime coarse component');seen.update(mapping.values())
        for row in selection['selected_components']:
            if str(row['component_index']) in mapping:
                current=mapping[str(row['component_index'])]
                selected.append(dict(row,**by[current]))
        p=deepcopy(read(entry['provenance']))
        require({str(f['original_component_index']) for f in p['faces']}==set(mapping),'Original face membership differs')
        for face in p['faces']:face['original_component_index']=mapping[str(face['original_component_index'])]
        provenance[group['group_id']]=p
    require(len(selected)==len(selection['selected_components']),'Packed selection omitted original components')
    normalized=dict(selected_components=selected,protected_component_indices=sorted(set(by)-seen))
    return far.compile_groups(catalogue,render,normalized,provenance)


def replace_coarse_rows(annotation,inventory,catalogue,groups,bounds):
    lineage={r['target_id']:{k:r[k] for k in ('geometry_source_id','geometry_target_id','morphology_id','generated_geometry','physical_geometry_evidence')} for r in annotation['targets']}
    replaced=far.merged_rows(annotation,inventory,catalogue,groups,bounds)
    for row in annotation['targets']:row.update(lineage[row['target_id']])
    return replaced


def assess_frame(metadata,annotation,checker):
    coverage=read_json(_pin(annotation['target_census']['full_scene_coverage']))
    require(coverage['schema']==COVERAGE_SCHEMA
            and coverage['individual_coarse_component_pixel_visibility_claimed'] is False
            and coverage['all_near_components_evaluated_with_unchanged_numeric_predicates'] is True
            and coverage['coarse_actual_pose_bounds']['all_components_outside_both_actual_arm_bounds'] is True
            and coverage['coarse_actual_pose_bounds']['minimum_clearance_m']>0
            and coverage['coarse_actual_pose_bounds']['packed_world_coordinate_expansion_m']==far.EPS,
            'Actual positive coarse exclusion coverage required')
    return base.assess_frame(metadata,annotation,checker)


def evaluate_capture(capturepath,output):
    """Evaluate one genuinely closed controlled capture, at most128 actual frames.

    The new owner/context is authenticated directly; no original-only owner or
    metadata schema is emitted. Saved inputs remain unchanged and all results
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
    from . import native848_generated_farmerge_capture_v1 as producer
    from . import native848_reset6_production_plan_v1 as render_api
    from .native848_bulk_plan_v1 import check_profile
    capture=Path(capturepath).resolve();output=Path(output).resolve();trial=capture.parent
    require(capture.name=='capture' and capture.is_dir() and not output.exists()
        and not output.is_relative_to(capture) and not (capture/'failure.json').exists(),
        'Fresh separate output and successful controlled capture required')
    pins={}
    def bound(spec,*,within=None):
        reduced={k:spec[k] for k in ('path','sha256')};path=_pin(reduced)
        if within is not None:require(path.is_relative_to(within),'Artifact escaped its actual capture')
        pins[str(path)]=reduced['sha256'];return path
    def read(spec,*,within=None):return read_json(bound(spec,within=within))
    def local(path):
        path=Path(path).resolve();return dict(path=str(path),sha256=sha256(path))
    complete=read(local(trial/'owner_complete.json'))
    require(complete['training_approved'] is False and complete['accepted_training_increment']==0,
        'Capture-only owner completion required')
    require(Path(complete['owned_exit']['path']).resolve()==trial/'owned_exit.json'
        and Path(complete['owner_started']['path']).resolve()==trial/'owner_started.json',
        'Completion belongs to another owner lifecycle')
    exited=read(complete['owned_exit'],within=trial);started=read(complete['owner_started'],within=trial)
    worker=read(exited['owned_worker'],within=trial)
    require(exited['returncode']==0 and Path(exited['owned_worker']['path']).resolve()==trial/'owned_worker.json',
        'Genuine exact owned native exit0 required')
    require(Path(complete['result']['path']).resolve()==capture/'result.json','Owner completed another capture')
    result=read(complete['result'],within=capture)
    require(result['schema']==producer.RESULT_SCHEMA and result['state']=='captured_pending_complete_annotation_and_visual_review'
        and result['training_approved'] is False and result['accepted_training_increment']==0
        and 0<=len(result['frames'])<=128 and complete['frames']==len(result['frames']),
        'Bounded typed capture-only completion required')
    require(complete['request']==started['request']==result['request']
        and complete['native_identity']==result['native_identity']
        and Path(complete['native_identity']['path']).resolve()==capture/'native_identity.json',
        'Completion/request/native identity pins differ')
    request=read(complete['request']);candidates=producer.check_request(request)
    require(len(result['frames'])<=request['max_frames'],'Capture exceeded finite schedule')
    pins.update(request['implementation_bindings']);pins.update(candidates['source_bindings'])
    require(request['implementation_bindings'].get(str(Path(producer.__file__).resolve()))==sha256(producer.__file__),
        'Actual controlled owner implementation unbound')
    identity=read(complete['native_identity'],within=capture);command=started['command']
    require(identity['expected_command']==worker['command']==command
        and identity['command']['ProcessId']==worker['launcher_pid']
        and identity['owner']['ProcessId']==worker['owner_pid'],'Native owner/command identity differs')
    gate=producer.production.identity_gate
    require(gate.same_identity(identity['owner'],started['resources']['raw_owner_classification']['metadata']),
        'Historical owner identity differs')
    gate.command_child(identity['command'],identity['owner'],command)
    gate.native_child(identity['native'],identity['command'],command[1:])
    require(command[1:6]==['-B','-u','-m',producer.MODULE,'--capture'],'Wrong actual capture module')
    def argument(flag):
        require(command.count(flag)==1 and command.index(flag)+1<len(command),'Unique worker argument required')
        return command[command.index(flag)+1]
    require(Path(argument('--output')).resolve()==capture
        and Path(argument('--request')).resolve()==Path(started['request']['path']).resolve()
        and argument('--request-sha256')==started['request']['sha256'],'Owner request/output binding differs')
    context=read(local(capture/'context.json'))
    require(context['schema']==CONTEXT_SCHEMA and context['training_approved'] is False
        and context['accepted_training_increment']==0,'Typed generated-only context required')
    require(context['source_pose_profile']==candidates['profile']
        and context['profile']==context['production_profile']==request['production_profile']
        and context['farmerge_recipe']==request['farmerge_recipe'],'Source/production profile provenance differs')
    for path,h in validate_collection_plan_provenance(context,candidates).items():
        require(path not in pins or pins[path]==h,'Conflicting camera/generator source authority');pins[path]=h
    verify_bindings(context['source_bindings']);pins.update(context['source_bindings'])
    census=read(local(capture/'census.json'))
    require(census==context['full_scene_census'],'Saved census differs from context')
    catalogue=read(context['catalogue'],within=capture);index=CatalogueIndex(catalogue)
    render=read(context['render_census'],within=capture);recipe=read(context['farmerge_recipe'])
    require(result['render_census']==context['render_census'] and result['production_profile']==request['production_profile'],
        'Result/context render provenance differs')
    verify_bindings(recipe['source_bindings']);pins.update(recipe['source_bindings'])
    groups=compile_runtime_groups(catalogue,census,render,recipe,read)
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
    for module in (sys.modules[__name__],source_consumer,base,far,overlay_api,render_api,labels,original,joint,ambiguity,answer_contract):
        path=str(Path(module.__file__).resolve());pins[path]=sha256(path)
    source_profile=read(candidates['profile']);check_profile(source_profile)
    profile=read(request['production_profile']);render_api.check_profile(profile)
    require(source_profile['render_settings']==profile['render_settings'],'Production changed renderer settings')
    require(context['renderer_settings']==profile['render_settings'],'Captured renderer differs from source optics profile')
    records=_unique(candidates['records'],'sample_id')
    frameids=[r['observation_id'] for r in result['frames']]
    require(len(frameids)==len(set(frameids)) and set(frameids)<=set(records),'Foreign/repeated frame IDs')
    expected_requests=(len(profile['warmup_steps']) if frameids else 0)+len(frameids)
    require(result['request_count']==len(result['requests'])==expected_requests,'Native warmup/production accounting differs')
    for n,e in enumerate(result['requests'],1):
        require(e['request_index']==n and e['native_requests']==e['callback_count']==1
            and e['callback_sequence_after']==n and e['callback_sequence_before']==n-1
            and e['requested_subframes']==(profile['warmup_steps'][n-1] if n<=len(profile['warmup_steps']) else profile['request_subframes'])
            and e['delta_time_seconds']==profile['delta_time_seconds'] and e['wait_for_render'] is True
            and abs(e['timeline_after_seconds']-e['timeline_before_seconds']-profile['delta_time_seconds'])<1e-7,
            'Actual same-callback/reset optical profile differs')
    output.mkdir();checker=BulkWorkspace();summaries=[];previous=None
    try:
        for receipt in result['frames']:
            observation=read(receipt,within=capture);name=observation['observation_id'];record=records[name]
            require(observation['schema']=='greenhouse.generated_farmerge_native_observation.v1'
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
            mapping_value=read(observation['mapping'],within=capture)
            mapping={int(k):v for k,v in mapping_value['renderer_id_to_prim'].items()}
            payload=deepcopy(observation['native_payload_header'])
            payload.update(rgb=np.dstack((rgb,alpha)),distance_to_image_plane=depth,
                           instance_id_segmentation=dict(data=ids,info=dict(idToLabels=mapping)))
            sync=observation['synchronization']
            _,_,native_valid,reference,token,native_ids,native_mapping=producer.primitives.native_freshness(
                payload,observation['calibration'],sync['static_before'],sync['static_after'],sync['callback_sequence'])
            require(token==sync['freshness'] and reference==sync['reference_time']
                and np.array_equal(native_valid,valid) and np.array_equal(native_ids,ids) and native_mapping==mapping,
                'Saved native callback buffers/calibration/freshness differ')
            require(observation['request_evidence']==result['requests'][sync['request_index']-1]
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
            metadata=dict(schema='greenhouse.generated_farmerge_native9mm_sample.v1',sample_id=name,
                calibration=observation['calibration'],robot_snapshot=pose,rendered_camera_params=observation['rendered_camera_params'],
                synchronization=sync,geometry_screen=pose['visual_bound_screen'],source_family=candidates['source_family'],
                geometry_source_id=generated_report['plant_id'],split='train',training_approved=False,accepted_training_increment=0)
            verify_camera(metadata)
            components,pixel_proof=far.classify_pixels(ids,mapping_value,catalogue,groups,depth,valid)
            unknown=pixel_proof['blocking_unknowns'];pixel_classes=pixel_proof['pixel_categories']
            bounds=original.robot_bounds(metadata,checker);coarse=far.actual_coarse_bounds(groups,bounds)
            require(coarse['all_components_outside_both_actual_arm_bounds']
                and coarse==observation['coarse_surface_exclusion']
                and observation['render_census']==context['render_census'],'Actual coarse bounds or render census differ')
            annotation=evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
            replaced=replace_coarse_rows(annotation,inventory,catalogue,groups,bounds)
            for target in annotation['targets']:
                proof=target.get('attachment_joint_ownership')
                if proof is not None:require(all(pins.get(p)==h for p,h in proof['source_bindings'].items()),'Unbound current geometry joint proof')
            folder=output/name;folder.mkdir()
            coverage=dict(schema=COVERAGE_SCHEMA,frame_id=name,
                observation=dict(path=str(Path(receipt['path']).resolve()),sha256=receipt['sha256']),
                context=observation['context'],capture_split='train',plant_instance_count=144,
                original_background_instances=143,generated_foreground_instances=1,
                complete_all_eligible_ground_truth=not unknown and not annotation['target_census']['unknown_target_ids'],
                blocking_unknowns=unknown+annotation['target_census']['unknown_target_ids'],visible_cross_split_petiole_targets=[],
                pixel_categories=dict(pixel_classes),catalogue_petiole_count=len(inventory),source_bindings=dict(pins),
                coarse_actual_pose_bounds=coarse,coarse_petiole_positive_exclusions=replaced,
                individual_coarse_component_pixel_visibility_claimed=False,
                all_near_components_evaluated_with_unchanged_numeric_predicates=True,
                render_census=context['render_census'],farmerge_recipe=context['farmerge_recipe'],
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
        value=dict(schema='greenhouse.native848_generated_farmerge_capture_evaluation.v1',source_capture=str(capture),
            owner_complete=local(trial/'owner_complete.json'),capture_result=complete['result'],frames_evaluated=len(summaries),
            records=summaries,workspace_stats=stats,source_bindings=pins,native_control_performed=False,
            actual_visual_review=False,training_approved=False,accepted_training_increment=0)
        save_json(output/'result.json',value);return value
    finally:checker.finish()
