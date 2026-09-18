"""Export reviewed generated RGB-D with explicit farmerge provenance.

The frozen generated exporter supplies all numerical, visual-review, and
sensor checks. This adapter adds typed render provenance and verifies that
every coarse petiole has a positive reach or anatomy exclusion. No reannotation.
"""
from . import native848_generated_dataset_v1 as prior
from . import native848_generated_farmerge_9mm_v1 as fast
from . import native848_generated_farmerge_capture_v1 as producer
from .native848_generated_dataset_v1 import (
    Counter,deepcopy,Path,hashlib,io,json,np,Image,base,evaluated,cache,roots,fingerprint,
    require,write,pin,_unique,validate_partition,validate_review,validate_qualification,
    validate_authorization,REVIEW_SCHEMA,AUTH_SCHEMA,
)

SCHEMA='greenhouse.native848_reviewed_generated_farmerge_supplement.v1'
FRAME_SCHEMA='greenhouse.native848_reviewed_generated_farmerge_frame.v1'
EVALUATION_SCHEMA='greenhouse.native848_generated_farmerge_capture_evaluation.v1'
PINS={**prior.PINS,
    prior.__file__:'1b7e397469cdd3872c60350f91526f4175cca6c306dd631d08600f93ed9122ef',
    fast.__file__:'8db35f3ace1b450dc8f75ede117bb58c2c3d97133fb15b521a02916bb568a72f',
    producer.__file__:'8feae7f535aa67a16aa435c1a74ce406311a1f18786c89c8ebc47bfe067022af',
    fast.overlay_api.__file__:'a095f37d0dafa62a36f7b937c9ff2e7fa26ac54f7866430ce23b920b3045491e'}


def authenticate_render(context,census,catalogue,captured,read,bindings):
    require(captured['schema']==producer.RESULT_SCHEMA
            and captured['render_census']==context['render_census']
            and captured['production_profile']==context['production_profile']==context['profile'],
            'Typed fast capture/render/profile pins differ')
    request=read(captured['request']);candidates=read(request['candidates'])
    require(request['schema']==producer.REQUEST_SCHEMA
            and request['farmerge_recipe']==context['farmerge_recipe']
            and request['production_profile']==context['production_profile']
            and candidates['profile']==context['source_pose_profile'], 'Source and production authority differ')
    render=read(context['render_census']);recipe=read(context['farmerge_recipe'])
    require(render['finite_population_recipe']==context['farmerge_recipe']
            and render['original_collision_representation_retained'] is True
            and render['background_shapes_materials_and_placements_preserved'] is True,
            'Full scene representation and collision provenance required')
    bindings(recipe['source_bindings'])
    return fast.compile_runtime_groups(catalogue,census,render,recipe,read)


def validate_coarse_coverage(annotation,coverage,observation,context,catalogue,groups):
    require(coverage['schema']==fast.COVERAGE_SCHEMA
            and coverage['render_census']==observation['render_census']==context['render_census']
            and coverage['farmerge_recipe']==context['farmerge_recipe']
            and coverage['individual_coarse_component_pixel_visibility_claimed'] is False
            and coverage['all_near_components_evaluated_with_unchanged_numeric_predicates'] is True,
            'Typed coarse ownership/exclusion coverage required')
    bounds=fast.far.actual_coarse_bounds(groups,annotation['outer_workspace_bounds'])
    require(bounds==coverage['coarse_actual_pose_bounds']==observation['coarse_surface_exclusion']
            and bounds['all_components_outside_both_actual_arm_bounds'] is True
            and bounds['minimum_clearance_m']>0 and bounds['packed_world_coordinate_expansion_m']==fast.far.EPS,
            'Coarse geometry is not excluded at this actual robot pose')
    rows=_unique(annotation['targets'],'target_id')
    coarse=[r for r in catalogue if r['component_index'] in groups['selected'] and r['organ_type']=='sub_stem']
    require(len(coarse)==coverage['coarse_petiole_positive_exclusions'], 'Missing coarse petiole exclusion')
    for component in coarse:
        row=rows[component['variant_id']+'/'+component['component_id']]
        require(row['generated_geometry'] is False and row['status']=='excluded'
                and row['reason'] in (evaluated.original.OUTSIDE_REASON,'not_anatomically_eligible_leaf_petiole'),
                'Coarse pixels cannot prove invisibility or generated target eligibility')
        require(evaluated.ambiguity.continuously_visible(row) is not True,
                'Coarse annotation cannot fabricate individual pixel visibility')
        if row['reason']==evaluated.original.OUTSIDE_REASON:
            evaluated.original.validate_exclusion(row,annotation['outer_workspace_bounds'])
    return bounds


def load_model_inputs(record,root):
    require(record['schema']==FRAME_SCHEMA and record['task_id']==base.TASK
        and set(record['inputs'])==base.INPUT_KEYS,'Generated model input contract differs')
    root=Path(root).resolve()
    def file(spec):
        require(set(spec)=={'path','sha256'},'Exact local asset pin required')
        p=(root/spec['path']).resolve();require(p.is_relative_to(root) and base.digest(p)==spec['sha256'],'Changed/escaping sensor asset');return p
    rgb=np.asarray(Image.open(file(record['inputs']['rgb'])).convert('RGB'))
    depth=np.load(file(record['inputs']['depth_m']),allow_pickle=False);valid=np.load(file(record['inputs']['depth_valid']),allow_pickle=False)
    cal=json.loads(file(record['inputs']['calibration']).read_text());require(set(cal)==set(base.CAL_KEYS),'Hidden calibration/query field')
    base.validate_arrays(rgb,depth,valid,dict(cal,crop_resize=None))
    robot=base.validate_workspace_context(json.loads(file(record['workspace_filter_context']).read_text()))
    return dict(rgb=rgb,depth_m=depth,depth_valid=valid,calibration=cal,workspace_filter_context=robot,instruction=base.INSTRUCTION)


def materialize(evaluation_pin,selected_sample_ids,review_pin,output,*,authorization_pin):
    """One immutable source session, at most16 individually reviewed images."""
    selected=list(selected_sample_ids);require(0<len(selected)<=16 and len(set(selected))==len(selected),'Finite distinct selection required')
    output=roots.checkpoint_output(output);session=cache.Session()
    for path,h in PINS.items():session.authenticate(path,h)
    implementation_sha=session.digest(__file__)
    def read(spec):return json.loads(session.read_pin(spec).read_text())
    def local(path):return dict(path=str(Path(path).resolve()),sha256=session.digest(path))
    def bindings(value):session.verify_bindings(value)
    auth=read(authorization_pin);validate_authorization(auth,evaluation_pin,selected,review_pin,implementation_sha)
    value=read(evaluation_pin);review=read(review_pin)
    for key,role in [('native_outer_wait','capture'),('annotation_outer_wait','annotation')]:
        observed=review['actual_outer_process_completion'][role]
        require(observed['tool_session']==auth[key]['session_id'] and observed['exit_code']==auth[key]['returncode']==0
            and observed['wait_observed_by']==review['reviewer'],'Authorization differs from actual review closure')
    require(value['schema']==EVALUATION_SCHEMA and value['frames_evaluated']==len(value['records'])
        and value['native_control_performed'] is False and value['accepted_training_increment']==0,'Completed typed evaluation required')
    records=_unique(value['records'],'sample_id');require(set(selected)<=set(records),'Selection outside completed evaluation')
    bindings(value['source_bindings']);bindings(value['workspace_stats']['source_bindings'])
    require(all(value['source_bindings'].get(str(Path(module.__file__).resolve()))==PINS[module.__file__]
                for module in (evaluated,fast,producer,fast.overlay_api)), 'Fast evaluation/capture implementation unbound')
    capture=roots.resolve_evidence(value['source_capture']);trial=capture.parent
    complete=read(value['owner_complete']);captured=read(value['capture_result'])
    require(Path(value['owner_complete']['path']).resolve()==trial/'owner_complete.json'
        and Path(value['capture_result']['path']).resolve()==capture/'result.json'
        and complete['result']==value['capture_result'],'Capture/evaluation closure differs')
    exit_record=read(complete['owned_exit']);worker=read(exit_record['owned_worker']);started=read(complete['owner_started'])
    require(exit_record['returncode']==0 and complete['request']==started['request']==captured['request']
        and complete['native_identity']==captured['native_identity'] and worker['command']==started['command'],'Native owned closure differs')
    identity=read(complete['native_identity'])
    require(identity['expected_command']==worker['command'] and identity['command']['ProcessId']==worker['launcher_pid']
        and identity['owner']['ProcessId']==worker['owner_pid'],'Native identity differs')
    require(complete['frames']==value['frames_evaluated']==len(captured['frames']) and captured['training_approved'] is False
        and captured['accepted_training_increment']==0,'Incomplete capture evaluation')
    capture_frames=_unique(captured['frames'],'observation_id');require(set(capture_frames)==set(records),'Evaluation omitted actual frame')
    context=read(local(capture/'context.json'));census=read(local(capture/'census.json'));catalogue=read(context['catalogue'])
    bindings(context['source_bindings'])
    require(context['schema']==fast.CONTEXT_SCHEMA and context['dataset_split']=='train'
        and context['native_population_census_verified'] is True and context['original_143_backgrounds_preserved'] is True
        and context['full_scene_census']==census,'Typed full generated native population required')
    check=deepcopy(census);h=check.pop('deterministic_census_sha256');require(h==fingerprint(check),'Census fingerprint differs')
    require(census['schema']==evaluated.CENSUS_SCHEMA and census['unchanged_background_count']==143
        and census['active_counts']['component_plants']==144 and census['active_counts']['backdrop_instances']==0
        and census['removed_roots']==[] and census['source_assets_unchanged'] is True,'Full population/source preservation differs')
    groups=authenticate_render(context,census,catalogue,captured,read,bindings)
    variants=_unique(context['scene_variants'],'variant_id');plants=_unique(census['all_plant_roots'],'variant_id')
    require(set(variants)==set(plants) and len(plants)==144,'All144 instance roots required')
    plan=read(context['source_collection_plan']);catalogue_counts=Counter(c['variant_id'] for c in catalogue)
    require(len({c['component_index'] for c in catalogue})==len(catalogue)
        and len({c['prim_path'] for c in catalogue})==len(catalogue),'Nonunique native component ownership')
    for vid,v in variants.items():
        p=plants[vid];require(v['source_plant_id']==v['split_group']==p['source_family'] and p['source_split']==plan['family_assignments'][v['source_plant_id']]=='train'
            and v['plant_root']==p['plant_root'] and p['component_count']==catalogue_counts[vid],'Instance anatomy/split census differs')
    modified=[v for v in variants.values() if v['source_geometry_modified']];require(len(modified)==1,'Exactly one generated foreground required')
    generated=modified[0];gid=generated['geometry_source_id'];donor=generated['source_plant_id'];source=context['geometry_sources'][gid]
    require(gid!=donor and source['donor_source_family']==donor and source['source_split']=='train','Generated donor lineage differs')
    qualification=read(source['qualification']);physical=read(source['physical_geometry_evidence']);bindings(physical['source_bindings'])
    require(physical['status']=='qualified_proximal_mesh_identity' and physical['all_controlled_targets_qualified'] is True
        and physical['proximal_mesh_exact_30mm'] is True and physical['physical_9mm_source_correspondence'] is True
        and physical['shared_junction_preserved'] is True,'Generated physical9mm proof missing')
    validate_qualification(qualification,donor,gid,plan['family_assignments'])
    report=read(local(capture/'generated_report.json'));require(report['plant_id']==gid and report['manifest_sha256']==source['manifest']['sha256'],'Generated source report differs')
    checked=[]
    for name in selected:
        r=records[name];meta=read(r['metadata']);ann=read(r['annotation']);cov=read(r['coverage']);amb=read(r['ambiguity'])
        bindings(ann['source_bindings']);bindings(cov['source_bindings']);bindings(ann['outer_workspace_bounds']['source_bindings'])
        obs=read(ann['observation']);require(ann['observation']==cov['observation']
            and {k:capture_frames[name][k] for k in ('path','sha256')}==ann['observation'],'Frame receipt differs')
        require(obs['sample_id']==obs['observation_id']==name==ann['frame_id']==meta['sample_id']==cov['frame_id']
            and obs['calibration']==meta['calibration'] and obs['robot_snapshot']==meta['robot_snapshot']
            and ann['private_robot_context']==dict(calibration=meta['calibration'],robot_snapshot=meta['robot_snapshot'])
            and cov['context']==obs['context'] and read(obs['context'])==context,'Actual robot/camera/context differs')
        require(cov['complete_all_eligible_ground_truth'] is True and cov['blocking_unknowns']==[]
            and cov['visible_cross_split_petiole_targets']==[] and cov['plant_instance_count']==144
            and cov['original_background_instances']==143 and cov['generated_foreground_instances']==1
            and ann['target_census']['full_scene_coverage']==r['coverage'],'Unverified/partial coverage')
        validate_coarse_coverage(ann,cov,obs,context,catalogue,groups)
        target=validate_partition(ann,catalogue,variants,amb)
        require(r['strict_candidate_ids']==[target['target_id']] and r['candidate_source_target_ids']==[target['source_target_id']]
            and r['candidate_geometry_target_ids']==[target['geometry_target_id']] and r['single_answer_candidate'] is True
            and r['unknown_target_ids']==[] and r['generated_foreground_is_sole_strict_candidate'] is True,'Summary candidate differs')
        require(target['source_component_id'] in physical['radius_qualified_component_ids']
            and target['physical_geometry_evidence']==source['physical_geometry_evidence'],'Candidate physical radius not qualified')
        expected=evaluated.geometry_9mm(report,target['source_component_id'],plants[target['plant_instance_id']]['plant_to_world_usd_row_vectors'],meta['calibration'])
        require(expected==target['geometry'],'Candidate source9mm geometry differs')
        answer=base.target_answer(target,meta['calibration'])
        require(ann['outer_workspace_bounds']['robot_state_sha256']==fingerprint(dict(calibration=meta['calibration'],robot_snapshot=meta['robot_snapshot'])),'Workspace bound belongs to another pose')
        rgb_pin=obs['files']['rgb'];rgb_path=session.read_pin(rgb_pin);rgb=np.asarray(session.image(rgb_path)).copy()
        with session.arrays(session.read_pin(obs['files']['buffers'])) as arrays:depth=arrays['depth_m'];valid=arrays['depth_valid']
        base.validate_arrays(rgb,depth,valid,meta['calibration']);reviewed=validate_review(review,evaluation_pin,r,rgb_pin)
        robot=base.workspace_context(meta);base.validate_workspace_context(robot)
        checked.append(dict(record=r,metadata=meta,annotation=ann,ambiguity=amb,rgb_pin=rgb_pin,rgb_bytes=session.bytes(rgb_path),rgb=rgb,
            depth=depth,valid=valid,target=target,answer=answer,robot=robot,review=reviewed,observation=ann['observation']))
    require(len({hashlib.sha256(x['rgb'].tobytes()).hexdigest() for x in checked})==len(checked),'Repeated decoded RGB')
    require(len({base.camera_signature(x['metadata'],gid) for x in checked})==len(checked),'Repeated camera within morphology')
    output.mkdir();rows=[]
    try:
        for x in checked:
            name=x['record']['sample_id'];target=x['target'];folder=output/'frames'/hashlib.sha256(name.encode()).hexdigest()[:24];folder.mkdir(parents=True)
            with (folder/'rgb.png').open('xb') as f:f.write(x['rgb_bytes'])
            require(base.digest(folder/'rgb.png')==x['rgb_pin']['sha256'],'RGB bytes changed')
            for key in ('depth','valid'):
                with (folder/('depth_m.npy' if key=='depth' else 'depth_valid.npy')).open('xb') as f:np.save(f,x[key],allow_pickle=False)
            write(folder/'calibration.json',base.sensor_calibration(x['metadata']['calibration']));write(folder/'workspace_context.json',x['robot'])
            write(folder/'annotation_private.json',x['annotation']);write(folder/'ambiguity_private.json',x['ambiguity']);write(folder/'review.json',review)
            def asset(name):return dict(path=(folder/name).relative_to(output).as_posix(),sha256=base.digest(folder/name))
            row=dict(schema=FRAME_SCHEMA,task_id=base.TASK,annotation_epoch=base.EPOCH,frame_id=name,split='train',source_family=donor,
                donor_source_family=donor,source_target_id=target['source_target_id'],geometry_source_id=gid,morphology_id=gid,
                geometry_target_id=target['geometry_target_id'],plant_instance_id=target['plant_instance_id'],source_geometry_modified=True,
                new_independent_source_family=False,morphology_diversity_approved=False,decoded_rgb_sha256=hashlib.sha256(x['rgb'].tobytes()).hexdigest(),
                conservative_camera_signature=base.camera_signature(x['metadata'],gid),donor_camera_signature=base.camera_signature(x['metadata'],donor),
                answer={'cut_point_uv':x['answer']['cut_point_uv']},answer_audit={k:v for k,v in x['answer'].items() if k!='cut_point_uv'},
                inputs={k:asset(n) for k,n in [('rgb','rgb.png'),('depth_m','depth_m.npy'),('depth_valid','depth_valid.npy'),('calibration','calibration.json')]},
                workspace_filter_context=asset('workspace_context.json'),private_annotation=asset('annotation_private.json'),
                independent_ambiguity_assessment=asset('ambiguity_private.json'),actual_review=asset('review.json'),
                individual_full_frame_review=True,human_review_claimed=False,model_instruction=base.INSTRUCTION,
                source_provenance=dict(evaluation=evaluation_pin,**x['record'],rgb=x['rgb_pin'],observation=x['observation'],review=review_pin,
                    qualification=source['qualification'],physical_geometry_evidence=source['physical_geometry_evidence'],
                    render_census=context['render_census'],farmerge_recipe=context['farmerge_recipe'],
                    production_profile=context['production_profile'],source_pose_profile=context['source_pose_profile']))
            load_model_inputs(row,output);rows.append(row)
        with (output/'index.jsonl').open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        session.close();write(output/'validation_session.json',session.report())
        manifest=dict(schema=SCHEMA,state='complete_reviewed_generated_farmerge_supplement_pending_union',task_id=base.TASK,annotation_epoch=base.EPOCH,
            unique_images=len(rows),counts={'train':len(rows)},source_family_counts=dict(Counter(r['source_family'] for r in rows)),
            source_target_counts=dict(Counter(r['source_target_id'] for r in rows)),morphology_counts=dict(Counter(r['morphology_id'] for r in rows)),
            independent_donor_family_count=1,new_independent_donor_families=0,morphology_diversity_approved=False,
            farmerge_representation_preserves_all_background_shapes_materials=True,individual_coarse_component_pixel_visibility_claimed=False,
            render_census=context['render_census'],farmerge_recipe=context['farmerge_recipe'],
            fully_populated_greenhouse=True,plant_instance_count=144,original_background_instances=143,generated_foreground_instances=1,
            native_resolution=[848,408],nominal_arc_m=.009,input_query=False,input_target_id=False,input_ground_truth_crop=False,input_ground_truth_mask=False,
            raw_float32_depth_retained=True,existing_original_dataset_untouched=True,numerics_reused_from_completed_pinned_evaluation=True,
            annotation_recomputed=False,model_inputs_loaded_after_copy=True,source_evaluation=evaluation_pin,actual_review=review_pin,
            authorization=authorization_pin,validation_session=pin(output/'validation_session.json'),index=pin(output/'index.jsonl'),
            implementation_sha256=implementation_sha,training_started=False,training_approved=False)
        write(output/'manifest.json',manifest);write(output/'result.json',dict(schema=SCHEMA,state=manifest['state'],unique_images=len(rows),
            manifest=pin(output/'manifest.json'),training_started=False,training_approved=False))
        (output/'README.md').write_text('Generated-morphology supplement: native848x408 RGB-D, exactly one9mm cut point.\n'
            'Use native848_generated_farmerge_dataset_v1.load_model_inputs; only four sensor inputs and measured robot workspace context are model inputs.\n'
            'Donor TRAIN split remains unchanged. Morphology IDs do not create new donor families. No union or training is claimed.\n',encoding='utf-8')
        return manifest
    except BaseException as exc:
        write(output/'failure.json',dict(error=repr(exc),partial_output_not_admissible=True));raise
