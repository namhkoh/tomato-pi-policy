"""Dataset-only no-query SINGLE uniquely eligible petiole9mm contract. No model or training runner."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from PIL import Image
from . import native848_data_roots_v1 as roots
from .capture_contract import project, fingerprint
from . import native848_unique9mm_ambiguity_v2 as ambiguity
AMBIGUITY_SHA256='8152cd4194b2544badf3886f35be7e9bff2a68b4df65aa9fe620108408a99ec5'

SCHEMA='greenhouse.native848_fully_labeled_frame_validation.v1'
TASK='greenhouse.native848_unique_petiole_9mm.v1'
EPOCH='greenhouse.native848_all_petiole_9mm.v1'
FRAME_SCHEMA='greenhouse.native848_unique_petiole_frame.v1'
REVIEW_SCHEMA='greenhouse.native848_unique_petiole_frame_review.v1'
BANNED='seed41_full/SubStem_38'
INSTRUCTION=('Locate the cut point on the single eligible intact petiole attached to a main stem. '
 'The point is 9 mm along the petiole centerline from the attachment toward the leaves. '
 'Return exactly one cut_point_uv in the original 848 x 408 full image. '
 'Eligibility includes visibility and a separate robot-kinematics workspace filter.')
CAL_KEYS=('resolution','intrinsics','depth_convention','optical_frame','pixel_convention','clipping_range_m')
INPUT_KEYS={'rgb','depth_m','depth_valid','calibration'}
JOINT_NAMES=frozenset([f'torso_{i}' for i in range(6)]+[f'head_{i}' for i in range(2)]+[f'{side}_arm_{i}' for side in ('left','right') for i in range(7)])
CONTEXT_KEYS={'schema','joint_degrees','optical_to_robot_base_usd_row_vectors','camera_to_head_column_vectors','reachability_semantics'}

from . import native848_pilot_annotation_v12 as full_annotation
from . import native848_fully_labeled_worker_v1 as full_worker
from . import native848_fully_labeled_scene_v1 as full_scene
ANNOTATION_SHA='4e3f107c3c21b22c7e8304aae2cc710fa811f0b8c024b5c919f8695ed3166203'
WORKER_SHA='749c9239729c82254f076ddbc2cfccd0d14e3cc09c0b3f576fb00d2a3ca50f9d'
SCENE_SHA='f00841a7b7e246796bcd34e2c31ffd9d1ac4358a5294340d33c355c9239ddd17'



def fully_labeled_identity(annotation):
    """Bind every instance target to the complete native catalogue and source family."""
    for module,expected in ((full_annotation,ANNOTATION_SHA),(full_worker,WORKER_SHA),(full_scene,SCENE_SHA)):
        require(digest(module.__file__)==expected,'Frozen full-population validator changed')
    require(annotation['source_bindings'].get(str(Path(full_annotation.__file__).resolve()))==ANNOTATION_SHA,
        'Annotation lacks actual fully labeled evaluator provenance')
    full_annotation.require_source_target_lineage(annotation)
    def read(spec):return json.loads(read_pin(spec).read_text())
    coverage=read(annotation['target_census']['full_scene_coverage'])
    require(coverage['coverage_scope']=='complete144_manifest_backed_instances'
        and coverage['plant_instance_count']==144 and coverage['clones_count_as_independent_families'] is False
        and coverage['source_target_identity_policy']==full_annotation.LINEAGE_POLICY,
        'Complete labeled-population coverage required')
    context=read(coverage['context']);full_worker.require_scope(context,native=True)
    require(context['schema']=='greenhouse.native848_fully_labeled_context.v1'
        and context['scene_policy']==full_scene.policy(context['dataset_split']),'Typed full-population context required')
    cp=dict(path=context['full_scene_census_path'],sha256=context['full_scene_census_sha256'])
    require(coverage['actual_fully_labeled_scene_census']==cp,'Coverage census substitution')
    census=read(cp);cpu=read(context['cpu_scene_preflight'])
    full_worker.require_scope(cpu,native=False)
    require(cpu['schema'] in {v[5] for v in full_annotation.OWNERS.values()},'Typed full-population CPU scene proof required')
    if cpu['schema']!='greenhouse.native848_fully_labeled_cpu_preflight.v1':
        require(context['pose_anchor_sources']==cpu['pose_anchor_sources'],'CPU pose-anchor source population differs')
    require(census==context['full_scene_census'],'Native census differs from context')
    full_worker.require_native_census(census,cpu)
    require(cpu['plan_sha256']==context['plan_sha256'],'CPU scene proof belongs to another plan')
    catalogue=read(dict(path=context['catalogue_path'],sha256=context['catalogue_sha256']))
    reports_list=read(dict(path=context['source_reports_path'],sha256=context['source_reports_sha256']))
    reports={r['plant_id']:r for r in reports_list}
    require(len(reports)==len(reports_list) and catalogue==full_annotation.expected_catalogue(context,reports),'Catalogue differs from complete original anatomical hierarchy')
    roots={r['plant_root']:r for r in census['all_plant_roots']}
    variants={v['variant_id']:v for v in context['scene_variants']}
    require(len(roots)==len(census['all_plant_roots'])==len(variants)==len(context['scene_variants'])==144,
        'All144 distinct plant instances required')
    per_instance=Counter(r['variant_id'] for r in catalogue)
    require(set(per_instance)==set(variants) and all(per_instance[k]==roots[v['plant_root']]['component_count'] for k,v in variants.items()),'Catalogue omits or substitutes an occupied plant instance')
    by={}
    for row in catalogue:
        key=row['variant_id']+'/'+row['component_id']
        require(key not in by,'Duplicate component identity')
        v=variants[row['variant_id']];r=roots[v['plant_root']]
        require(r['active_after_policy'] is True and r['authenticated_component_plant'] is True
            and row['source_plant_id']==row['split_group']==v['source_plant_id']==v['split_group']==r['source_family']
            and r['source_split']==context['dataset_split'],'Unauthenticated or cross-split source lineage')
        by[key]=row
    petioles={key:row for key,row in by.items() if row['organ_type']=='sub_stem'}
    require(len(catalogue)==coverage['active_catalogue_components']==context['scene_counts']['components']
        and len(petioles)==coverage['catalogue_petiole_count']
        and set(petioles)==set(annotation['target_census']['catalogue_petiole_ids'])
        and set(petioles)=={t['target_id'] for t in annotation['targets']},'Full native petiole census differs')
    for t in annotation['targets']:
        row=petioles[t['target_id']]
        require(t['source_family']==row['source_plant_id'] and t['source_component_id']==row['component_id']
            and t['plant_instance_id']==row['variant_id'],'Annotation relabeled a source target or clone')
        if t.get('reason')==full_annotation.full_coverage.OUTSIDE_REASON or 'outer_workspace_exclusion' in t:
            v=variants[t['plant_instance_id']];r=roots[v['plant_root']]
            entry=dict(target_id=t['target_id'],source_family=t['source_family'],report=reports[t['source_family']],
                plant_to_world_usd_row_vectors=r['plant_to_world_usd_row_vectors'])
            geometry=full_annotation.full_coverage.source_geometry(entry)
            require(geometry==t['geometry'],'Outside-reach geometry differs from pinned native instance anatomy')
            full_annotation.full_coverage.validate_exclusion(t,annotation['outer_workspace_bounds'])

    return coverage,context,petioles


def require(value,message):
    if not value:raise ValueError(message)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def canonical_sha(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def pin(p):return dict(path=str(Path(p).resolve()),sha256=digest(p))


def read_pin(spec):
    require(set(spec)=={'path','sha256'},'Exact file pin required')
    p=roots.resolve_evidence(spec['path']);require(p.is_file() and digest(p)==spec['sha256'],'Changed pinned evidence')
    return p


def sensor_calibration(cal):
    require(cal['resolution']==[848,408] and cal['crop_resize'] is None,'Unscaled native calibration required')
    require(cal['depth_convention']=='optical_axis_z_metres_not_ray_range','Optical-axis metric depth required')
    k=np.asarray(cal['intrinsics'],float)
    require(k.shape==(3,3) and np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0 and np.allclose(k[2],[0,0,1]),'Invalid K')
    return {k:cal[k] for k in CAL_KEYS}


def workspace_context(metadata):
    """Whitelist measured state; never expose desired_cut_pixel/framing hints."""
    robot=metadata['robot_snapshot'];cal=metadata['calibration']
    cw=np.asarray(cal['camera_to_world_usd_row_vectors'],float)
    bw=np.asarray(robot['robot_root_to_world_usd_row_vectors'],float)
    mount=np.asarray(robot['camera_to_head_column_vectors'],float)
    for matrix in (cw,bw,mount):require(matrix.shape==(4,4) and np.isfinite(matrix).all(),'Finite measured transforms required')
    require(np.allclose(cw[:,3],[0,0,0,1]) and np.allclose(bw[:,3],[0,0,0,1]),'USD row-vector transform required')
    joints=robot['joint_degrees'];require(isinstance(joints,dict) and set(joints)==JOINT_NAMES and all(type(v) in (int,float) and np.isfinite(v) for v in joints.values()),'Measured joint state required')
    require(set(joints)==JOINT_NAMES,'Unexpected joint-state key')
    opt_to_base=np.diag([1.,-1.,-1.,1.])@cw@np.linalg.inv(bw)
    return dict(schema='greenhouse.native848_workspace_filter_context.v1',joint_degrees=joints,
        optical_to_robot_base_usd_row_vectors=opt_to_base.tolist(),camera_to_head_column_vectors=mount.tolist(),
        reachability_semantics='deterministic_robot_kinematics_filter_not_visual_or_agronomic_inference')


def validate_workspace_context(context):
    require(set(context)==CONTEXT_KEYS and context['schema']=='greenhouse.native848_workspace_filter_context.v1',
            'Robot context contains hidden framing/query hint')
    require(context['reachability_semantics']=='deterministic_robot_kinematics_filter_not_visual_or_agronomic_inference',
            'Reachability semantics differ')
    joints=context['joint_degrees']
    require(isinstance(joints,dict) and set(joints)==JOINT_NAMES
        and all(type(v) in (int,float) and np.isfinite(v) for v in joints.values()),'Hidden/nonfinite joint-state field')
    for key in ('optical_to_robot_base_usd_row_vectors','camera_to_head_column_vectors'):
        matrix=np.asarray(context[key],float);require(matrix.shape==(4,4) and np.isfinite(matrix).all(),'Invalid robot context transform')
    require(np.allclose(np.asarray(context['optical_to_robot_base_usd_row_vectors'])[:,3],[0,0,0,1]),'Invalid optical-to-base row convention')
    return context


def validate_arrays(rgb,depth,valid,cal):
    require(rgb.shape==(408,848,3) and rgb.dtype==np.uint8,'Native848x408 uint8 RGB required')
    require(depth.shape==valid.shape==(408,848) and depth.dtype==np.float32 and valid.dtype==bool,'Aligned float32 metric depth and bool validity required')
    near,far=cal['clipping_range_m']
    require(np.array_equal(valid,np.isfinite(depth)&(depth>0)&(depth>=near)&(depth<=far)),'Depth validity differs from actual clip/finite mask')
    sensor_calibration(cal)


def point_at_9mm(chain):
    chain=np.asarray(chain,float)
    require(chain.ndim==2 and chain.shape[1]==3 and len(chain)>=2 and np.isfinite(chain).all(),'Oriented finite world centerline required')
    lengths=np.linalg.norm(np.diff(chain,axis=0),axis=1)
    require((lengths>1e-12).all() and lengths.sum()>=.009,'Degenerate or short centerline')
    cumulative=np.r_[0.,np.cumsum(lengths)];i=min(int(np.searchsorted(cumulative,.009,side='right')-1),len(lengths)-1)
    return chain[i]+(.009-cumulative[i])/lengths[i]*(chain[i+1]-chain[i])


def target_answer(target,cal):
    require(target['status']=='candidate_pending_visual_review' and target['automated_pass'] is True,'Only evaluated candidates can be accepted')
    require(target['annotation_epoch']==EPOCH and target['nominal_arc_m']==.009,'Exact new9mm epoch required')
    require(target['support_is_permissible_cut_interval'] is False,'Visibility support is not cut tolerance')
    geo=target['geometry'];expected=point_at_9mm(geo['oriented_centerline_world_m'])
    attachment_expected=project([geo['oriented_centerline_world_m'][0]],cal)[0]
    require(np.allclose(attachment_expected['camera_optical_xyz_m'],geo['attachment_projected']['camera_optical_xyz_m'],rtol=0,atol=1e-8),'Centerline does not start at actual attachment')
    require(geo['nominal']['arc_m']==.009 and np.allclose(expected,target['cut_world_m'],rtol=0,atol=1e-9)
            and np.allclose(expected,geo['nominal']['world_m'],rtol=0,atol=1e-9),'Cut does not follow9mm arc length')
    projected=project([expected],cal)[0]
    require(projected['projection_status']=='in_frame'
            and np.allclose(projected['pixel_xy'],target['cut_point_uv'],rtol=0,atol=1e-6)
            and np.allclose(projected['camera_optical_xyz_m'],target['cut_optical_xyz_m'],rtol=0,atol=1e-9),'Cut projection/calibration differs')
    workspace=target['workspace']
    require(workspace['target_id']==target['target_id'] and workspace['result']['workspace_passed'] is True
        and np.allclose(workspace['nominal_world_m'],expected,rtol=0,atol=1e-9)
        and workspace['per_frame_camera_FK_verified'] is True
        and workspace['per_frame_cached_solution_FK_verified'] is True
        and isinstance(workspace['solve_input_sha256'],str) and len(workspace['solve_input_sha256'])==64,
        'Workspace proof is not the verified 9 mm problem')
    require(target['visibility']['nominal']['visible'] is True and all(p['visible'] is True for p in target['visibility']['support'])
        and target['local_clarity']['passed'] is True,'Missing new nominal/support/clarity checks')
    junction=target['junction_continuity'];probes=junction['probes']
    arcs=np.asarray([q['arc_m'] for q in probes],float)
    require(junction['attachment_visible'] is True and junction['all_attachment_to_9mm_probes_visible'] is True
        and junction['maximum_arc_step_m']<=.0005 and len(arcs)>=19 and np.isfinite(arcs).all()
        and abs(arcs[0])<1e-12 and abs(arcs[-1]-.009)<1e-12
        and (np.diff(arcs)>0).all() and (np.diff(arcs)<=.0005+1e-12).all()
        and all(q['visible'] is True and q['projected']['projection_status']=='in_frame'
                and q['depth']['status']=='depth_consistent_not_visibility_verified' for q in probes),
        'Entire actual attachment-to-9mm junction must be visible')
    parent=target['parent_context'];z=projected['camera_optical_xyz_m'][2]
    diameter=2*geo['nominal']['radius_m']*min(cal['intrinsics'][0][0],cal['intrinsics'][1][1])/z
    required=max(3.,diameter*.65)
    require(parent['visible_parent_pixels']>=6 and np.isfinite(parent['cut_to_parent_distance_px'])
        and abs(parent['required_distance_px']-required)<1e-8
        and parent['cut_to_parent_distance_px']>=required,'Actual parent visibility/clearance insufficient')
    box=target['target_bbox_xyxy'];require(len(box)==4 and all(type(x) is int for x in box)
        and 0<=box[0]<box[2]<=848 and 0<=box[1]<box[3]<=408,'Visible bbox must be half-open native pixels')
    attachment=geo['attachment_projected'];require(attachment['projection_status']=='in_frame','Visible attachment required')
    return dict(bbox_xyxy=box,attachment_pixel_uv=attachment['pixel_xy'],cut_point_uv=target['cut_point_uv'],
                cut_optical_xyz_m=target['cut_optical_xyz_m'])


def validate_census(annotation):
    require(annotation['schema']==annotation['annotation_epoch']==EPOCH and annotation['training_approved'] is False,'New candidate-only annotation required')
    census=annotation['target_census'];rows=annotation['targets'];ids=[t['target_id'] for t in rows]
    require(len(ids)==len(set(ids)) and sorted(ids)==sorted(census['catalogue_petiole_ids']) and census['complete'] is True and census['catalogue_complete'] is True,'Incomplete/duplicate target census')
    for status,key in [('candidate_pending_visual_review','candidate_target_ids'),('excluded','excluded_target_ids'),('unknown','unknown_target_ids')]:
        require(sorted(t['target_id'] for t in rows if t['status']==status)==sorted(census[key]),'Census status population differs')
    require(all(t['status'] in ('candidate_pending_visual_review','excluded','unknown') for t in rows),'Unknown census status')
    require(not census['unknown_target_ids'] and annotation['frame_blocked_by_unknown_targets'] is False,'Unresolved possible petiole blocks frame')
    require(not any(t['target_id']==BANNED and t['status']=='candidate_pending_visual_review' for t in rows),'Eligible permanently excluded target holds whole frame')
    require(annotation['frame_blocked_by_unverified_full_scene_coverage'] is False
        and census['full_scene_coverage_validation_required'] is True,'Unverified full-scene coverage blocks frame')
    spec=census['full_scene_coverage'];coverage=json.loads(read_pin({k:spec[k] for k in ('path','sha256')}).read_text())
    require(coverage['schema']=='greenhouse.native848_all_petiole_coverage_audit.v1'
        and coverage.get('CPU_fixture_only') is not True and coverage['complete_all_eligible_ground_truth'] is True
        and not coverage['blocking_unknowns'] and not coverage['visible_cross_split_petiole_targets'],
        'Authoritative coverage has unknown/cross-split possible positives')
    require(annotation['observation']==dict(path=coverage['observation_path'],sha256=coverage['observation_sha256']),
        'Coverage belongs to a different actual frame')
    for path,h in coverage['source_bindings'].items():require(digest(path)==h,'Coverage source changed')
    fully_labeled_identity(annotation)
    return exactly_one_candidate(rows)


def camera_signature(metadata,family):
    cal=metadata['calibration']
    return fingerprint(dict(family=family,lighting=None,
        camera_to_world=np.round(cal['camera_to_world_usd_row_vectors'],8).tolist(),
        intrinsics=np.round(cal['intrinsics'],8).tolist()))


def validate_source_identity(spec,metadata,annotation,target):
    coverage,verified_context,petioles=fully_labeled_identity(annotation)
    families=coverage['scene_source_families'];split=coverage['capture_split']
    require(isinstance(families,dict) and families and split in ('train','validation','test')
        and all(isinstance(k,str) and k and v==split for k,v in families.items()),'Scene contains unauthenticated or cross-split family')
    family=target['source_family']
    require(family in families and spec['source_family']==family and spec['split']==split==families[family],
        'Export family/split must match the actual uniquely eligible target')
    require(spec['conservative_camera_signature']==camera_signature(metadata,family),'Caller-supplied camera dedup identity differs')
    context_path=read_pin(coverage['context']);context=json.loads(context_path.read_text())
    require(context==verified_context and context['schema']=='greenhouse.native848_fully_labeled_context.v1'
        and context['dataset_split']==split and context['scene_policy']['dataset_split']==split,
        'Actual native pilot context split differs')
    observation_path=read_pin(annotation['observation']);observation=json.loads(observation_path.read_text())
    full_worker.require_scope(observation,native=True)
    require(observation['schema']=='greenhouse.native848_fully_labeled_observation.v1'
        and observation['observation_id']==metadata['sample_id']==annotation['frame_id']
        and observation['dataset_split']==split and observation['calibration']==metadata['calibration']
        and observation['robot_snapshot']==metadata['robot_snapshot'],'Actual native frame state differs')
    require(Path(observation['context_path']).resolve()==context_path and observation['context_sha256']==coverage['context']['sha256'],
        'Actual native observation context differs')
    for name in ('rgb','buffers'):
        require(Path(spec[name]['path']).resolve()==Path(observation['files'][name]['path']).resolve()
            and spec[name]['sha256']==observation['files'][name]['sha256'],'Frame sensor asset substitution')

    for path,h in annotation['source_bindings'].items():require(digest(path)==h,'Annotation source changed')
    return dict(source_family=family,source_target_id=target['source_target_id'],source_component_id=target['source_component_id'],plant_instance_id=target['plant_instance_id'],split=split,scene_source_families=families,context=coverage['context'])


def exactly_one_candidate(rows):
    require(not any(t['status']=='unknown' for t in rows),'Unknown target prevents uniqueness')
    candidates=[t for t in rows if t['status']=='candidate_pending_visual_review']
    require(len(candidates)==1,'Exactly one genuinely eligible petiole required; multi/zero targets held')
    return candidates


def validate_actual_review(review,annotation,annotation_pin):
    require(review['schema']==REVIEW_SCHEMA and review['annotation']==annotation_pin
        and review['frame_id']==annotation['frame_id'] and review.get('CPU_fixture_only') is not True,'Exact actual frame review required')
    require(review['decision']=='accept' and review['actual_full_native_rgb_viewed'] is True
        and review['all_candidate_targets_inspected'] is True and review['no_query_task_reviewed'] is True
        and review['no_obvious_ghosting'] is True and review['single_answer_uniqueness_reviewed'] is True,'Actual complete unique-target no-query frame review required')
    expected=annotation['target_census']['candidate_target_ids'];decisions=review['target_decisions']
    require(len(decisions)==len(expected) and {d['target_id'] for d in decisions}==set(expected)
        and all(d['decision']=='accept' and d['visible_9mm_attachment_association_clear'] is True for d in decisions),'Every candidate needs actual9mm clarity review')
    if not expected:require(review.get('complete_true_negative_explicitly_reviewed') is True,'Empty targets require reviewed complete negative')


def load_model_inputs(record,root):
    """Only clean full sensors and separately whitelisted robot context escape."""
    require(record['schema']==FRAME_SCHEMA and record['task_id']==TASK and set(record['inputs'])==INPUT_KEYS,'Hidden or missing model input cue')
    root=Path(root).resolve()
    def file(spec):
        require(set(spec)=={'path','sha256'},'Exact local asset pin required')
        p=(root/spec['path']).resolve();require(p.is_relative_to(root) and p.is_file() and digest(p)==spec['sha256'],'Changed/escaping model asset')
        return p
    rgb=np.asarray(Image.open(file(record['inputs']['rgb'])).convert('RGB'))
    depth=np.load(file(record['inputs']['depth_m']),allow_pickle=False)
    valid=np.load(file(record['inputs']['depth_valid']),allow_pickle=False)
    cal=json.loads(file(record['inputs']['calibration']).read_text());require(set(cal)==set(CAL_KEYS),'Calibration contains target/query hints')
    full=dict(cal,crop_resize=None);validate_arrays(rgb,depth,valid,full)
    context=validate_workspace_context(json.loads(file(record['workspace_filter_context']).read_text()))
    return dict(rgb=rgb,depth_m=depth,depth_valid=valid,calibration=cal,
        workspace_filter_context=context,instruction=INSTRUCTION)


def unique_counts(records):
    for key in ('frame_id','decoded_rgb_sha256','conservative_camera_signature'):
        values=[r[key] for r in records];require(len(values)==len(set(values)),'Repeated frame/RGB/camera identity')
    families={}
    for row in records:
        answer=row['answer'];uv=np.asarray(answer.get('cut_point_uv'),float)
        require(set(answer)=={'cut_point_uv'} and uv.shape==(2,) and np.isfinite(uv).all() and ((uv>=0)&(uv<[848,408])).all(),'Exactly one finite pixel answer required')
        require(row['split'] in ('train','validation','test'),'Invalid split')
        require(families.setdefault(row['source_family'],row['split'])==row['split'],'Family crossed splits')
        for family,split in row.get('scene_source_families',{}).items():
            require(split==row['split'] and families.setdefault(family,split)==split,'Visible scene family crossed splits')
    targets=Counter(r['source_target_id'] for r in records)
    require(all(k.startswith(r['source_family']+'/') and k!=BANNED for r in records for k in [r['source_target_id']]),'Source-target lineage or ban differs')
    require(max(targets.values(),default=0)<=200,'Source target exceeds200 views across all instances')
    return dict(source_target_count=len(targets),source_target_counts=dict(targets),source_family_count=len({r['source_family'] for r in records}),unique_images=len(records),counts=dict(Counter(r['split'] for r in records)),
        eligible_petiole_instances=len(records),legacy10mm_images_in_new_count=0)
