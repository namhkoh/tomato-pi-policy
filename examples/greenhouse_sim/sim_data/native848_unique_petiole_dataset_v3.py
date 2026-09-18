"""Dataset-only no-query SINGLE uniquely eligible petiole9mm contract. No model or training runner."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from PIL import Image
from . import native848_data_roots_v1 as roots
from . import native848_fully_labeled_frame_validation_v1 as frame_validator
FRAME_VALIDATOR_SHA='fdc0d1af840835f2ca07e413f69dcf1938f933211546f1ca6f8d43775a4fedb2'
from .capture_contract import project, fingerprint
from . import native848_unique9mm_ambiguity_v2 as ambiguity
from . import native848_unique9mm_group_review_v2 as group_review
AMBIGUITY_SHA256='8152cd4194b2544badf3886f35be7e9bff2a68b4df65aa9fe620108408a99ec5'
GROUP_SHA256='f25c7c8f6e94192b5b4c39bbe3c7acd6fc27fd1d0bc46b159d94d1e724ecb0d4'

SCHEMA='greenhouse.native848_fully_labeled_frame_dataset.v3'
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



def camera_signature(metadata,family):
    cal=metadata['calibration']
    return fingerprint(dict(family=family,lighting=None,
        camera_to_world=np.round(cal['camera_to_world_usd_row_vectors'],8).tolist(),
        intrinsics=np.round(cal['intrinsics'],8).tolist()))





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

def reviewed_scope(spec,review,annotation,group_cache=None):
    require(digest(ambiguity.__file__)==AMBIGUITY_SHA256,'Changed independent ambiguity guard')
    ambiguity.validate_assessment(spec['ambiguity'],spec['metadata'],spec['annotation'],annotation)
    if review['schema']==group_review.REVIEW_SCHEMA:
        require(digest(group_review.__file__)==GROUP_SHA256,'Changed predeclared group protocol')
        inventory=review['inventory']
        key=(spec['review']['path'],spec['review']['sha256'],inventory['path'],inventory['sha256'])
        if group_cache is None:group_cache={}
        if key not in group_cache:
            group_cache[key]=group_review.validate_group_review(inventory['path'],inventory['sha256'],spec['review']['path'],spec['review']['sha256'])
        checked=group_cache[key]
        row=checked['records'].get(annotation['frame_id'])
        require(row is not None and row['decision']=='accept' and row['annotation']==spec['annotation'],
            'Frame lacks an accepted exact group review')
        return dict(review_scope=row['review_scope'],individual_full_frame_review=row['individual_full_frame_review'],group_id=row['group_id'])
    validate_actual_review(review,annotation,spec['annotation'])
    require(review['independent_ambiguity_assessment']==spec['ambiguity']
        and review['near_eligible_alternatives_inspected'] is True,'Actual competing-answer visual assessment required')
    return dict(review_scope='individual',individual_full_frame_review=True,group_id=None)


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




def materialize_frames(frames,output,*,root_authorization):
    """One create-only pilot package; root pins exact source annotations/reviews.

    Each frame spec: metadata, rgb, buffers, annotation and review exact pins,
    source_family, split, conservative_camera_signature. The root authorization
    binds every frame spec and the current implementation after native/census
    authentication. No synthetic fixture authorization and no implicit reuse.
    """
    require(digest(frame_validator.__file__)==FRAME_VALIDATOR_SHA,'Fully labeled frame validator changed')
    output=roots.checkpoint_output(output);auth_path=read_pin(root_authorization)
    auth=json.loads(auth_path.read_text())
    require(auth['schema']=='greenhouse.native848_unique_petiole_dataset_authorization.v1'
        and auth['task_id']==TASK and auth['dataset_only'] is True
        and auth['native_and_inventory_sources_authenticated'] is True
        and auth['blocking_findings']==[] and auth.get('CPU_fixture_only') is not True
        and auth['exporter_sha256']==digest(__file__) and auth['frames_sha256']==canonical_sha(frames),
        'Exact reviewed finite dataset authorization required')
    require(frames,'No empty pilot export');checked=[];group_cache={};pins={str(auth_path):root_authorization['sha256']}
    for spec in frames:
        paths={k:read_pin(spec[k]) for k in ('metadata','rgb','buffers','annotation','review','ambiguity')}
        pins.update({str(paths[k]):spec[k]['sha256'] for k in paths})
        meta=json.loads(paths['metadata'].read_text());annotation=json.loads(paths['annotation'].read_text());review=json.loads(paths['review'].read_text())
        require(annotation['frame_id']==meta['sample_id'],'Annotation/source frame mismatch')
        require(annotation['private_robot_context']['calibration']==meta['calibration']
            and annotation['private_robot_context']['robot_snapshot']==meta['robot_snapshot'],'Annotation used different sensor/robot state')
        with Image.open(paths['rgb']) as im:
            require(im.mode=='RGB' and im.size==(848,408),'Native source RGB required');rgb=np.asarray(im).copy()
        with np.load(paths['buffers'],allow_pickle=False) as b:depth=b['depth_m'].copy();valid=b['depth_valid'].copy()
        validate_arrays(rgb,depth,valid,meta['calibration']);targets=validate_census(annotation)
        scope=reviewed_scope(spec,review,annotation,group_cache)
        identity=validate_source_identity(spec,meta,annotation,targets[0])
        target=target_answer(targets[0],meta['calibration'])
        answer={'cut_point_uv':target['cut_point_uv']}
        answer_audit={k:v for k,v in target.items() if k!='cut_point_uv'}
        context=workspace_context(meta)
        checked.append((spec,paths,meta,annotation,rgb,depth,valid,answer,context,answer_audit,identity,scope))
    # Refuse duplicate views before creating any output.
    preview=[dict(frame_id=x[3]['frame_id'],decoded_rgb_sha256=hashlib.sha256(x[4].tobytes()).hexdigest(),
        conservative_camera_signature=x[0]['conservative_camera_signature'],source_family=x[10]['source_family'],source_target_id=x[10]['source_target_id'],plant_instance_id=x[10]['plant_instance_id'],split=x[10]['split'],scene_source_families=x[10]['scene_source_families'],answer=x[7]) for x in checked]
    counts=unique_counts(preview);output.mkdir();records=[]
    try:
        for base,item in zip(preview,checked):
            spec,paths,meta,annotation,rgb,depth,valid,answer,context,answer_audit,identity,scope=item
            folder=output/'frames'/hashlib.sha256(base['frame_id'].encode()).hexdigest()[:24];folder.mkdir(parents=True)
            # New copies preserve full RGB; clean depth/validity files contain no renderer IDs.
            import shutil
            with paths['rgb'].open('rb') as src,(folder/'rgb.png').open('xb') as dst:shutil.copyfileobj(src,dst)
            require(digest(folder/'rgb.png')==spec['rgb']['sha256'],'RGB copy changed')
            for name,array in [('depth_m',depth),('depth_valid',valid)]:
                with (folder/(name+'.npy')).open('xb') as f:np.save(f,array,allow_pickle=False)
            write(folder/'calibration.json',sensor_calibration(meta['calibration']));write(folder/'workspace_context.json',context)
            write(folder/'annotation_private.json',annotation);write(folder/'review.json',json.loads(paths['review'].read_text()))
            write(folder/'ambiguity_private.json',json.loads(paths['ambiguity'].read_text()))
            def local(name):return dict(path=(folder/name).relative_to(output).as_posix(),sha256=digest(folder/name))
            record=dict(schema=FRAME_SCHEMA,task_id=TASK,annotation_epoch=EPOCH,answer_audit=answer_audit,**base,
                inputs={k:local(v) for k,v in [('rgb','rgb.png'),('depth_m','depth_m.npy'),('depth_valid','depth_valid.npy'),('calibration','calibration.json')]},
                workspace_filter_context=local('workspace_context.json'),private_annotation=local('annotation_private.json'),actual_review=local('review.json'),
                independent_ambiguity_assessment=local('ambiguity_private.json'),
                source_provenance=spec,model_instruction=INSTRUCTION,**scope)
            load_model_inputs(record,output);records.append(record)
        with (output/'index.jsonl').open('x',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
        for path,h in pins.items():require(digest(path)==h,'Source changed during export')
        manifest=dict(fully_populated_greenhouse=True,plant_instance_count=144,clones_count_as_independent_families=False,schema=SCHEMA,task_id=TASK,annotation_epoch=EPOCH,**counts,nominal_arc_m=.009,input_query=False,input_target_id=False,
            input_ground_truth_crop=False,input_ground_truth_mask=False,raw_float32_depth_retained=True,
            reachability='separate measured robot-context kinematics filter',source_bindings=pins,
            ambiguity_policy=ambiguity.POLICY,individual_visual_reviewed_images=sum(r['individual_full_frame_review'] for r in records),
            group_only_reviewed_images=sum(not r['individual_full_frame_review'] for r in records),
            source_family_counts=dict(Counter(r['source_family'] for r in records)),
            root_authorization=root_authorization,index=dict(path='index.jsonl',sha256=digest(output/'index.jsonl')),
            legacy_dataset_untouched=True,training_started=False,training_approved=False)
        write(output/'manifest.json',manifest)
        (output/'README.md').write_text('# One uniquely eligible petiole,9mm cutpoint\n\nOne row is one native848x408 image. Exactly one eligible petiole must be proved by a complete census and independent competing-answer assessment; multi-target and unknown frames are held. The model answer is one cut_point_uv. Load inputs through load_model_inputs. Never pass index/private annotation, masks, target identities or ground-truth crops as model inputs. Float32 optical-Z metres and validity are retained without colormap quantization. Reachability uses separate measured robot context, not RGB inference. Existing711 query-conditioned10mm records are legacy and are not included automatically.\n',encoding='utf-8')
        write(output/'result.json',dict(schema=SCHEMA,state='complete_reviewed_unique_petiole_no_query_9mm_dataset',manifest=pin(output/'manifest.json'),**counts,training_approved=False))
        return manifest
    except BaseException as exc:
        write(output/'failure.json',dict(error=repr(exc),partial_output_not_admissible=True));raise

from .native848_fully_labeled_frame_validation_v1 import validate_census,validate_source_identity,unique_counts
