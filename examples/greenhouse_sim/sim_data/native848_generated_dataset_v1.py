"""A typed, reviewed generated-morphology supplement; no native/IK replay.

Reuse an explicitly authorized completed generated evaluation, authenticate its
entire immutable source union, and check census/identity/numerical consistency.
Model inputs remain the same four clean sensor assets as the original dataset.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib, io, json
import numpy as np
from PIL import Image
from . import native848_unique_petiole_dataset_v8 as base
from . import native848_generated_9mm_v2 as evaluated
from . import native848_annotation_cache_v1 as cache
from . import native848_data_roots_v1 as roots
from .capture_contract import fingerprint

SCHEMA='greenhouse.native848_reviewed_generated_supplement.v1'
FRAME_SCHEMA='greenhouse.native848_reviewed_generated_frame.v1'
REVIEW_SCHEMA='greenhouse.native848_generated_actual_review.v1'
AUTH_SCHEMA='greenhouse.native848_generated_export_authorization.v1'
EVALUATION_SCHEMA='greenhouse.native848_generated_capture_evaluation.v1'
PINS={base.__file__:'f3934b4f89b717866e58415c6700184d77ec497b6886005ff0c7e642ea41414a',
 evaluated.__file__:'20dcaf90ad42a61c42805d60eef6ca345f0eea773e7ceb812acfdcfc60ddaaee',
 cache.__file__:'45d597a50775c61abd51507b5bbc6ba45eaa7064a7dfaa73ffecb39d29349174',
 cache.validation.__file__:'6e54fa3c8d3c5a38b591e8dfcd7538052e32d259c00196c8a164342f5d44bbdd'}
require=base.require
write=base.write
pin=base.pin


def _unique(rows,key):
    out={r[key]:r for r in rows};require(len(out)==len(rows),'Repeated '+key);return out


def validate_partition(annotation,catalogue,variants,ambiguity):
    require(annotation['schema']==evaluated.SCHEMA and annotation['annotation_epoch']==base.EPOCH
        and annotation['numerical_predicate_epoch']==base.EPOCH,'Typed generated current-epoch evaluation required')
    rows=_unique(annotation['targets'],'target_id')
    expected={r['variant_id']+'/'+r['component_id']:r for r in catalogue if r['organ_type']=='sub_stem'}
    census=annotation['target_census']
    require(census['complete'] is True and census['catalogue_complete'] is True
        and not annotation['frame_blocked_by_unknown_targets'] and not annotation['frame_blocked_by_unverified_full_scene_coverage']
        and not annotation['banned_candidate_instance_ids'],'Complete unblocked generated census required')
    require(set(rows)==set(expected)==set(census['catalogue_petiole_ids'])
        and len(census['catalogue_petiole_ids'])==len(rows),'Missing/repeated full-population petiole')
    for status,key in [('candidate_pending_visual_review','candidate_target_ids'),('excluded','excluded_target_ids'),('unknown','unknown_target_ids')]:
        require(sorted(t['target_id'] for t in rows.values() if t['status']==status)==sorted(census[key]),'Status partition changed')
    require(not census['unknown_target_ids'] and len(census['candidate_target_ids'])==1,'Exactly one strict answer and no unknown required')
    require(sum(len(census[k]) for k in ('candidate_target_ids','excluded_target_ids','unknown_target_ids'))==len(rows),'Foreign target status')
    for tid,t in rows.items():
        c=expected[tid];v=variants[c['variant_id']];cid=c['component_id'];donor=v['source_plant_id'];geometry=v['geometry_source_id']
        require(t['annotation_epoch']==base.EPOCH and t['source_family']==c['source_plant_id']==c['split_group']==v['split_group']==donor
            and t['source_component_id']==cid and t['source_target_id']==donor+'/'+cid
            and t['plant_instance_id']==c['variant_id'] and t['geometry_source_id']==c['geometry_source_id']==geometry
            and t['geometry_target_id']==geometry+'/'+cid
            and t['generated_geometry']==v['source_geometry_modified'],'Target instance/donor/generated identity differs')
        require(t['morphology_id']==(geometry if t['generated_geometry'] else None),'Morphology identity differs')
    primary=rows[census['candidate_target_ids'][0]]
    require(primary['generated_geometry'] is True and primary['geometry_source_id']!=primary['source_family']
        and primary['source_target_id']!=evaluated.BANNED_SOURCE_TARGET,'Generated foreground answer with original donor lineage required')
    require(ambiguity['schema']==evaluated.AMBIGUITY_SCHEMA and ambiguity['frame_id']==annotation['frame_id']
        and ambiguity['strict_label_statuses_preserved'] is True and ambiguity['assessment_complete'] is True
        and ambiguity['single_answer_unambiguous'] is True and ambiguity['decision']=='candidate_pending_actual_visual_review'
        and ambiguity['strict_candidate_ids']==census['candidate_target_ids'] and ambiguity['primary_target_id']==primary['target_id']
        and not ambiguity['reachable_alternative_ids'] and not ambiguity['unknown_alternative_ids'],'Independent single-answer assessment failed')
    alternatives=_unique(ambiguity['alternative_assessments'],'target_id')
    require(set(alternatives)==set(rows)-{primary['target_id']},'Missing alternative assessment')
    continuous=[]
    for tid,assessment in alternatives.items():
        t=rows[tid];visible=evaluated.ambiguity.continuously_visible(t)
        require(assessment['source_family']==t['source_family'] and assessment['original_strict_status']==t['status']
            and assessment['original_strict_reason']==t['reason'] and assessment['strict_eligibility_promoted'] is False
            and assessment['continuous_nominal_junction_visible']==visible,'Alternative evidence differs')
        if t['reason']==evaluated.original.OUTSIDE_REASON:
            evaluated.original.validate_exclusion(t,annotation['outer_workspace_bounds'])
            require(assessment['assessment']=='outside_both_authenticated_arm_outer_bounds'
                and assessment['outer_workspace_exclusion']==t['outer_workspace_exclusion'],'Outside proof differs')
        elif visible is False:
            require(assessment['assessment']=='objective_anatomy_or_native_visibility_exclusion','Native exclusion differs')
        else:
            require(visible is True and assessment['assessment']=='outside_conservative_reach_bound','Unresolved/reachable alternative')
            continuous.append(tid);w=assessment['workspace'];point=base.point_at_9mm(t['geometry']['oriented_centerline_world_m'])
            require(w['target_id']==tid and w['result']['workspace_passed'] is False and w['result']['status']=='outside_outer_reach_bound'
                and w['per_frame_camera_FK_verified'] is True and np.allclose(w['nominal_world_m'],point,atol=1e-9,rtol=0),'Alternative reach proof differs')
    require(sorted(continuous)==sorted(ambiguity['continuously_visible_alternative_ids']),'Continuous alternative population differs')
    return primary


def validate_review(review,evaluation_pin,record,rgb_pin):
    require(review['schema']==REVIEW_SCHEMA and review['evaluation']==evaluation_pin
        and review['human_review_claimed'] is False and isinstance(review['reviewer'],str) and review['reviewer'],'Actual attributed generated review required')
    frames=_unique(review['frames'],'sample_id');require(record['sample_id'] in frames,'Selected frame not reviewed')
    r=frames[record['sample_id']]
    require(r['annotation']==record['annotation'] and r['rgb']==rgb_pin and r['decision']=='accept'
        and all(r.get(k) is True for k in ('actual_full_native_rgb_viewed','actual_native_junction_crop_viewed',
            'actual_first_leaf_context_viewed','actual_flagged_contexts_viewed'))
        and isinstance(r['reason'],str) and r['reason'].strip(),'Incomplete or held generated visual review')
    return r


def validate_qualification(q,donor,gid,family_assignments):
    require(q['schema']=='greenhouse.multisubtree_controlled_qualification.v4' and q['source_family']==q['split_group']==donor
        and q['variant_id']==gid and q['split']==family_assignments[donor]=='train'
        and q['frozen_family_assignments']==family_assignments and q['source_assets_unchanged'] is True
        and q['new_donor_family_created'] is False and q['source_cap_reset'] is False,'Qualification donor split/lineage differs')


def validate_authorization(auth,evaluation_pin,selected,review_pin,implementation_sha):
    require(auth['schema']==AUTH_SCHEMA and auth['evaluation']==evaluation_pin and auth['review']==review_pin
        and auth['selected_sample_ids']==selected and auth['exporter_sha256']==implementation_sha
        and auth['dataset_only'] is True and auth['root_authorized'] is True and auth['training_authorized'] is False
        and auth['blocking_findings']==[],'Exact generated supplement authorization required')
    for key in ('native_outer_wait','annotation_outer_wait'):
        require(auth[key]['returncode']==0 and type(auth[key]['session_id']) is int and auth[key]['session_id']>0,'Genuine root-confirmed outer wait0 required')


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
    require(value['source_bindings'].get(str(Path(evaluated.__file__).resolve()))==PINS[evaluated.__file__],'Evaluation implementation unbound')
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
    require(context['schema']==evaluated.CONTEXT_SCHEMA and context['dataset_split']=='train'
        and context['native_population_census_verified'] is True and context['original_143_backgrounds_preserved'] is True
        and context['full_scene_census']==census,'Typed full generated native population required')
    check=deepcopy(census);h=check.pop('deterministic_census_sha256');require(h==fingerprint(check),'Census fingerprint differs')
    require(census['schema']==evaluated.CENSUS_SCHEMA and census['unchanged_background_count']==143
        and census['active_counts']['component_plants']==144 and census['active_counts']['backdrop_instances']==0
        and census['removed_roots']==[] and census['source_assets_unchanged'] is True,'Full population/source preservation differs')
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
                    qualification=source['qualification'],physical_geometry_evidence=source['physical_geometry_evidence']))
            load_model_inputs(row,output);rows.append(row)
        with (output/'index.jsonl').open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        session.close();write(output/'validation_session.json',session.report())
        manifest=dict(schema=SCHEMA,state='complete_reviewed_generated_supplement_pending_union',task_id=base.TASK,annotation_epoch=base.EPOCH,
            unique_images=len(rows),counts={'train':len(rows)},source_family_counts=dict(Counter(r['source_family'] for r in rows)),
            source_target_counts=dict(Counter(r['source_target_id'] for r in rows)),morphology_counts=dict(Counter(r['morphology_id'] for r in rows)),
            independent_donor_family_count=1,new_independent_donor_families=0,morphology_diversity_approved=False,
            fully_populated_greenhouse=True,plant_instance_count=144,original_background_instances=143,generated_foreground_instances=1,
            native_resolution=[848,408],nominal_arc_m=.009,input_query=False,input_target_id=False,input_ground_truth_crop=False,input_ground_truth_mask=False,
            raw_float32_depth_retained=True,existing_original_dataset_untouched=True,numerics_reused_from_completed_pinned_evaluation=True,
            annotation_recomputed=False,model_inputs_loaded_after_copy=True,source_evaluation=evaluation_pin,actual_review=review_pin,
            authorization=authorization_pin,validation_session=pin(output/'validation_session.json'),index=pin(output/'index.jsonl'),
            implementation_sha256=implementation_sha,training_started=False,training_approved=False)
        write(output/'manifest.json',manifest);write(output/'result.json',dict(schema=SCHEMA,state=manifest['state'],unique_images=len(rows),
            manifest=pin(output/'manifest.json'),training_started=False,training_approved=False))
        (output/'README.md').write_text('Generated-morphology supplement: native848x408 RGB-D, exactly one9mm cut point.\n'
            'Use native848_generated_dataset_v1.load_model_inputs; only four sensor inputs and measured robot workspace context are model inputs.\n'
            'Donor TRAIN split remains unchanged. Morphology IDs do not create new donor families. No union or training is claimed.\n',encoding='utf-8')
        return manifest
    except BaseException as exc:
        write(output/'failure.json',dict(error=repr(exc),partial_output_not_admissible=True));raise
