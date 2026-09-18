"""Authenticate full144 labeled captures and evaluate every instance petiole at9mm.

Candidate-only CPU evidence. Complete scene coverage and unique eligibility are
separate from actual visual review; this module never exports/trains/accepts.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse,hashlib,json,importlib,sys
import numpy as np
from PIL import Image
from . import native848_fully_labeled_plan_v1 as plan1
from . import native848_fully_labeled_plan_v2 as plan2
from . import native848_fully_labeled_plan_v3 as plan3
from . import native848_fully_labeled_worker_v1 as worker1
from . import native848_fully_labeled_worker_v2 as worker2
from . import native848_fully_labeled_worker_v3 as worker3
from . import native848_fully_labeled_sibling_worker_v1 as two_worker
from . import native848_fully_labeled_two_closure_v1 as two_closure
from . import native848_reset6_production_worker_v1 as production6_worker
from . import native848_reset6_production_plan_v1 as production6_plan
from . import native848_fully_labeled_scene_v1 as scene_api
from . import native848_all_petiole_9mm_v1 as labels
from . import native848_unique9mm_ambiguity_v2 as ambiguity
from .native848_bulk_workspace_v1 import BulkWorkspace
from .native848_pair_audit_v2 import verify_camera
from .native_greenhouse_pair import assert_same_camera
from .native_sensor_payload import validate_native_static,decode_native_instances
from .capture_contract import fingerprint
from . import native848_fully_labeled_coverage_v2 as full_coverage
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
from .native848_data_roots_v1 import diagnostic

SCHEMA='greenhouse.native848_fully_labeled_9mm_evaluation.v15'
OWNERS={
 'greenhouse.native848_fully_labeled_owner.v1':('run_native848_fully_labeled_serial_v1.py','2b583788f34e101e856bec8baa3bbb1dd26f74968522f2a831fe637dd3af7f2e',worker1,'749c9239729c82254f076ddbc2cfccd0d14e3cc09c0b3f576fb00d2a3ca50f9d',plan1,'greenhouse.native848_fully_labeled_cpu_preflight.v1'),
 'greenhouse.native848_fully_labeled_owner.v2':('run_native848_fully_labeled_serial_v2.py','7d32da56bd1f3bc8e8b00266cc9b6535bd1bf06b66811f636ee6fcfc7987c965',worker2,'a8f9838ec29d5c7c575642353bc0af6a384a39488506685b75fa7922b87d705e',plan2,'greenhouse.native848_fully_labeled_cpu_preflight.v2'),
 'greenhouse.native848_fully_labeled_owner.v3':('run_native848_fully_labeled_serial_v3.py','a538cd4df56c4be0dc4a9fb77804744addc283c3b1fb7acfcf75f3f33fe97656',worker3,'8882e3d0c73d26ce2ffc2fa49d801fb8c2e684718b4f2581a31180a4f5099abe',plan3,'greenhouse.native848_fully_labeled_cpu_preflight.v3'),
 'greenhouse.native848_fully_labeled_owner.v4':('run_native848_fully_labeled_serial_v4.py','2ba2a07a86dccbf9f02da55668f609f132bfc13527175c262447e0aa51e13a37',worker3,'8882e3d0c73d26ce2ffc2fa49d801fb8c2e684718b4f2581a31180a4f5099abe',plan3,'greenhouse.native848_fully_labeled_cpu_preflight.v3'),
 'greenhouse.native848_fully_labeled_owner.v5':('run_native848_fully_labeled_serial_v5.py','5ea36f87e8ee4a0a1a5a914b9743230d23c95f8943a7010fa13f018d48668bc5',worker3,'8882e3d0c73d26ce2ffc2fa49d801fb8c2e684718b4f2581a31180a4f5099abe',plan3,'greenhouse.native848_fully_labeled_cpu_preflight.v3'),
 'greenhouse.native848_fully_labeled_two_slot.v1':('run_native848_fully_labeled_two_v1.py','181622b9c174940ef33b278daf3552f1626bc581e58dbca6a753df31844524d5',two_worker,'1605add9268f769eebe852946b57b083a3056b5a53fc29927e433dd377bc2f91',plan3,'greenhouse.native848_fully_labeled_cpu_preflight.v3'),
 'greenhouse.native848_reset6_production_owner.v1':('run_native848_reset6_production_serial_v1.py','128adbf868745a3f800dc6ff868704fa362761af14e98b425abf3c4b0151fd74',production6_worker,'5e773e7c296903083f34b7357ca015e47dc92101fcad59a0409b1f27826a2dcf',production6_plan,production6_plan.CPU_SCHEMA),
 'greenhouse.native848_fully_labeled_owner.v6':('run_native848_fully_labeled_serial_v6.py','5c604d001dd4e2657f2c6e813a2b8ef8252f1cc8ce185be6cb868173cb91f615',worker3,'8882e3d0c73d26ce2ffc2fa49d801fb8c2e684718b4f2581a31180a4f5099abe',plan3,'greenhouse.native848_fully_labeled_cpu_preflight.v3')}
SLOT_SCHEMA='greenhouse.native848_fully_labeled_two_slot.v1'
PARENT_WRAPPER_SCHEMA='greenhouse.native848_fully_labeled_slot_annotation_parent.v1'
TWO_CLOSURE_SHA='381153e90f3bb952ff67fc709752786a7800f6800445e9ce4a79954e77906021'
COVERAGE_SHA256='cda0e9e39acefc6bd029bf5e1e7e014c211382527bf8e81356d9c482516bef97'
BANNED_SOURCE_TARGET='seed41_full/SubStem_38'
LINEAGE_POLICY='original_source_family_and_component_id_across_all_cloned_instances.v1'
AMBIGUITY_SHA256='8152cd4194b2544badf3886f35be7e9bff2a68b4df65aa9fe620108408a99ec5'
ROOT=Path(__file__).resolve().parents[3]


def typed_owner(owner):
    require(owner.get('schema') in OWNERS,'Unknown full-population owner schema')
    return OWNERS[owner['schema']]


def authenticate_owner(trial,owner,result_sha256,read,bound,parent_metadata,parent_recovery=None):
    if owner.get('schema')==SLOT_SCHEMA:
        require(parent_recovery is None and isinstance(parent_metadata,dict)
            and set(parent_metadata)=={'path','sha256'},'Exact typed parent wrapper pin required for successful finite-two slot')
        wrapper=read(parent_metadata['path'],parent_metadata['sha256'])
        require(set(wrapper)=={'schema','parent_closure'} and wrapper['schema']==PARENT_WRAPPER_SCHEMA,
            'Typed five-pin parent-closure wrapper required')
        parent=wrapper['parent_closure']
        require(isinstance(parent,dict) and set(parent)=={'metadata','result','terminal','outer_launch','outer_exit'},
            'Exact whole-parent closure pin population required')
        require(all(isinstance(p,dict) and set(p)=={'path','sha256'} and isinstance(p['path'],str) and p['path']
            and isinstance(p['sha256'],str) and len(p['sha256'])==64 and all(c in '0123456789abcdef' for c in p['sha256'])
            for p in parent.values()),'Exact parent path/SHA256 pins required')
        owner_name,owner_sha,optout_worker,worker_sha,plan_api,cpu_schema=typed_owner(owner)
        path=ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'/owner_name
        bound(path,owner_sha);bound(optout_worker.__file__,worker_sha);bound(two_closure.__file__,TWO_CLOSURE_SHA)
        # The frozen closure imports its coordinator by script name. Resolve that
        # import from the authenticated directory, including non-CLI callers.
        directory=str(path.parent.resolve());inserted=directory not in sys.path
        if inserted:sys.path.insert(0,directory)
        try:
            coordinator=importlib.import_module(owner_name[:-3])
            require(Path(coordinator.__file__).resolve()==path.resolve(),'Unexpected coordinator import path')
            proof=two_closure.authenticate_slot(trial,result_sha256,parent)
        finally:
            if inserted:sys.path.remove(directory)
        require(proof['schema']==two_closure.SCHEMA and proof['parent_closure']==parent
            and proof['both_actual_waits_verified'] is True and proof['actual_outer_wait0_verified'] is True
            and proof['coordinator_lock_released'] is True and proof['training_approved'] is False
            and proof['accepted_training_increment']==0,'Complete actual finite-two closure required')
        for p,h in proof['source_bindings'].items():bound(p,h)
        return dict(proof,parent_wrapper=dict(parent_metadata))
    require(parent_metadata is None and parent_recovery is None,'Typed full-population serial owner only')
    owner_name,owner_sha,optout_worker,worker_sha,plan_api,cpu_schema=typed_owner(owner)
    optout_worker.require_scope(owner,native=True)
    require(owner['state']=='owned_fully_labeled_population_capture_exited_pending_annotation_and_review'
        and owner['complete_anatomy_annotation'] is False and owner['individual_visual_review'] is False,
        'Actual complete full-population capture-only owner required')
    complete=read(trial/'owner_complete.json',sha256(trial/'owner_complete.json'))
    optout_worker.require_scope(complete,native=True)
    require(complete['result_sha256']==result_sha256 and complete['capture_only'] is True
        and complete['serial_native_lock_released_after_this_receipt'] is True,'Actual serial completion required')
    path=ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'/owner_name
    require(owner['source_bindings'].get(str(path.resolve()))==owner_sha,'Unknown full-population owner implementation')
    bound(path,owner_sha)
    require(owner['source_bindings'].get(str(Path(optout_worker.__file__).resolve()))==worker_sha==sha256(optout_worker.__file__),
        'Owner did not bind actual typed worker')
    captured=read(owner['capture_result_path'],owner['capture_result_sha256'])
    context=read(owner['context_path'],owner['context_sha256'])
    proof=optout_worker.validate_optout_capture(trial/'capture',captured,context)
    launch=read(trial/'launch.json',sha256(trial/'launch.json'))
    actual=optout_worker.validate_native_identity(proof['native_identity'],launch['command'])
    require(actual['command_identity']['ProcessId']==launch['pid'],'Actual full-population command PID differs')
    command=launch['command']
    def argument(flag):
        require(command.count(flag)==1 and command.index(flag)+1<len(command),'Unique actual worker argument required')
        return command[command.index(flag)+1]
    require(Path(argument('--output')).resolve()==trial/'capture'
        and Path(argument('--plan')).resolve()==Path(context['plan_path']).resolve()
        and argument('--plan-sha256')==context['plan_sha256']==owner['plan_sha256']
        and Path(argument('--scene-preflight')).resolve()==Path(context['cpu_scene_preflight']['path']).resolve()
        and argument('--scene-preflight-sha256')==context['cpu_scene_preflight']['sha256'],
        'Actual worker command differs from complete capture')
    exit_receipt=read(owner['owned_exit_path'],owner['owned_exit_sha256'])
    require(Path(owner['owned_exit_path']).resolve()==trial/'owned_exit.json'
        and exit_receipt['returncode']==0 and exit_receipt['method']=='subprocess_wait_on_owned_process'
        and exit_receipt['pid']==launch['pid'] and exit_receipt['launch_sha256']==sha256(trial/'launch.json'),
        'Actual exact native child wait required')
    for field,relative in (('capture_result_path','result.json'),('manifest_path','batch_manifest.json'),('context_path','batch_context.json')):
        require(Path(owner[field]).resolve()==trial/'capture'/relative,'Exact owned capture path required')
    for p,h in proof['source_bindings'].items():bound(p,h)
    return complete


def source_lineage_policy(annotation):
    """Source-target bans and diversity keys never depend on clone instance IDs."""
    blocked=[]
    for target in annotation['targets']:
        instance,component=target['target_id'].split('/')
        family=target['source_family']
        require(isinstance(family,str) and family and '/' not in family
            and target['plant_instance_id']==instance and target['source_component_id']==component
            and target['source_target_id']==family+'/'+component,'Original source-target lineage differs')
        if target['source_target_id']==BANNED_SOURCE_TARGET and target['status']=='candidate_pending_visual_review':
            blocked.append(target['target_id'])
    return dict(policy=LINEAGE_POLICY,source_target_key_fields=['source_family','source_component_id'],
        clones_are_independent_families=False,clones_are_novel_morphologies=False,
        permanently_banned_source_target=BANNED_SOURCE_TARGET,banned_candidate_instance_ids=blocked,
        accepted_training_increment=0)


def require_source_target_lineage(annotation):
    policy=source_lineage_policy(annotation)
    require(annotation['source_target_identity_policy']==policy,'Saved source-target policy differs')
    require(not policy['banned_candidate_instance_ids'],'Permanently excluded source target holds every clone frame')
    return policy


def expected_catalogue(context,reports):
    census=context['full_scene_census'];roots={r['plant_root']:r for r in census['all_plant_roots'] if r['active_after_policy']}
    variants={v['plant_root']:v for v in context['scene_variants']}
    require(len(roots)==144 and set(roots)==set(census['retained_roots'])==set(variants),'Active source roots differ')
    entries=[]
    for root,state in sorted(roots.items()):
        v=variants[root];report=reports[v['source_plant_id']]
        require(state['authenticated_component_plant'] is True and state['source_family']==v['source_plant_id']
                and state['source_split']==context['dataset_split'] and not v['added_components']
                and not v['added_component_paths'] and v['source_geometry_modified'] is False and v['split_group']==v['source_plant_id'],'Unauthenticated/cross-split active plant')
        components=report['components']
        for key,c in components.items():
            chain=[key];parent=c['parent']
            while parent is not None:
                require(parent in components and parent not in chain,'Invalid component ancestry')
                chain.append(parent);parent=components[parent]['parent']
            entries.append(dict(component_id=key,prim_path=root+'/'+'/'.join(reversed(chain)),organ_type=c['type'],
                                variant_id=v['variant_id'],source_plant_id=v['source_plant_id'],split_group=v['split_group']))
    return [dict(r,component_index=i) for i,r in enumerate(sorted(entries,key=lambda r:r['prim_path']),1)]


def completed_schedule(plan,manifest):
    expected=[r['observation_id'] for r in plan['schedule']]
    decisions=manifest['decisions'];observations=manifest['observations']
    require(len(expected)==len(set(expected))==plan['max_frames']
        and [r['observation_id'] for r in decisions]==expected,'Complete ordered finite native schedule required')
    require(all(r['state'] in ('queued_lossless_callback_pending_cpu_admission','rejected_native_geometry') for r in decisions),
        'Incomplete or failed native decisions cannot be chunked')
    committed=[r['observation_id'] for r in decisions if r['state']=='queued_lossless_callback_pending_cpu_admission']
    require([r['observation_id'] for r in observations]==committed and len(committed)==manifest['committed_frames'],
        'Committed population differs from complete finite decisions')
    require(manifest['production_native_requests']==sum(r['native_requests'] for r in decisions)==len(committed)
        and manifest['native_request_count']==manifest['native_callback_count']==manifest['warmup_request_count']+len(committed),
        'Complete native request/callback accounting differs')
    return observations


def finite_indices(total,start_index,frame_count):
    require(type(total) is int and total>=0 and type(start_index) is int and type(frame_count) is int
        and 1<=frame_count<=16 and (0<=start_index<total or total==start_index==0),
        'Explicit finite committed-frame chunk of at most16 required')
    return list(range(start_index,min(total,start_index+frame_count)))


def run(trial,result_sha256,output,*,parent_metadata=None,parent_recovery=None,start_index=0,frame_count=16):
    trial=diagnostic(trial);output=diagnostic(output)
    require(not output.exists() and not (trial/'failure.json').exists(),'New output and successful owned pilot required')
    pins={}
    def bound(path,pin):
        path=Path(path).resolve();require(sha256(path)==pin,'Changed artifact '+str(path));pins[str(path)]=pin;return path
    def read(path,pin):return read_json(bound(path,pin))
    owner=read(trial/'result.json',result_sha256)
    owner_name,owner_sha,optout_worker,worker_sha,plan_api,cpu_schema=typed_owner(owner)
    closure=authenticate_owner(trial,owner,result_sha256,read,bound,parent_metadata,parent_recovery)
    exit_receipt=read(owner['owned_exit_path'],owner['owned_exit_sha256'])
    require(exit_receipt['returncode']==0 and exit_receipt['method']=='subprocess_wait_on_owned_process','Actual owned exit0 required')
    verify_bindings(owner['source_bindings']);pins.update(owner['source_bindings'])
    native=read(owner['capture_result_path'],owner['capture_result_sha256'])
    manifest=read(owner['manifest_path'],owner['manifest_sha256'])
    context=read(owner['context_path'],owner['context_sha256'])
    for item in (native,manifest,context):optout_worker.require_scope(item,native=True)
    require(native['schema']=='greenhouse.native848_fully_labeled_capture.v1' and manifest['schema']=='greenhouse.native848_fully_labeled_manifest.v1','Typed full-population artifacts required')
    require(native['state']==plan_api.RESULT_STATE and context['schema']==plan_api.CONTEXT_SCHEMA
            and native['manifest_sha256']==owner['manifest_sha256'] and manifest['context_sha256']==owner['context_sha256']
            and native['committed_frames']==manifest['committed_frames']==owner['committed_frames'], 'New native pilot closure differs')
    require(not (trial/'capture/failure.json').exists() and manifest['source_assets_unchanged'] is True,'Native capture failure/source mutation')
    plan=read(context['plan_path'],context['plan_sha256'])
    optout_worker.require_scope(plan,native=False)
    require(plan['schema']==plan_api.SCHEMA and plan['capture_split']==context['dataset_split']
            and context['scene_policy']==scene_api.policy(context['dataset_split'])==plan['scene_policy'], 'Actual split/scene policy differs')
    census=read(context['full_scene_census_path'],context['full_scene_census_sha256'])
    require(census==context['full_scene_census'] and census['complete_active_plant_anatomy'] is True
            and census['active_counts']==context['scene_counts'] and census['active_counts']['backdrop_instances']==0,
            'Exact full filtered-scene census required')
    identity=dict(census);saved_identity=identity.pop('deterministic_census_sha256')
    require(fingerprint(identity)==saved_identity,'Scene census identity differs')
    cpu=read(context['cpu_scene_preflight']['path'],context['cpu_scene_preflight']['sha256'])
    require(cpu['schema']==cpu_schema,'Typed owner CPU proof differs')
    if plan_api is not plan1:
        require(context['pose_anchor_sources']==cpu['pose_anchor_sources'],'Bulk source-pose anchor population differs')
    optout_worker.require_scope(cpu,native=False)
    optout_worker.require_native_census(census,cpu)
    require(cpu['deterministic_census_sha256']==saved_identity and cpu['plan_sha256']==context['plan_sha256'],
            'Native scene did not match exact CPU preflight')
    reports_list=read(context['source_reports_path'],context['source_reports_sha256'])
    reports={r['plant_id']:r for r in reports_list};require(len(reports)==len(reports_list),'Duplicate anatomical reports')
    catalogue=read(context['catalogue_path'],context['catalogue_sha256'])
    require(catalogue==expected_catalogue(context,reports) and len(catalogue)==context['scene_counts']['components'],
            'Incomplete actual active component catalogue')
    source_plan=read(context['anchor']['source_collection_plan'],census['source_collection_plan']['sha256'])
    roots={r['plant_root']:r for r in census['all_plant_roots'] if r['active_after_policy']}
    require(all(source_plan['family_assignments'][r['source_family']]==context['dataset_split'] for r in roots.values()),'Active family belongs to another split')
    inventory=full_coverage.target_inventory(context,reports,catalogue)
    catalogue_index=full_coverage.CatalogueIndex(catalogue)
    pose_cache=read(plan['cache_path'],plan['cache_sha256'])
    pose_by_id={r['sample_id']:r for r in pose_cache['records']}
    require(len(pose_by_id)==len(pose_cache['records']), 'Duplicate planned pose')
    schedule={r['observation_id']:r for r in plan['schedule']}
    require(len(schedule)==plan['max_frames'] and len(manifest['decisions'])==plan['max_frames'],'Complete finite native decision population required')
    require({r['observation_id'] for r in manifest['decisions']}==set(schedule),'Native decision schedule differs')
    observations=completed_schedule(plan,manifest)
    indices=finite_indices(len(observations),start_index,frame_count)
    profile=read(plan['profile_evidence']['path'],plan['profile_evidence']['sha256']);plan_api.check_profile(profile)
    require(sha256(ambiguity.__file__)==AMBIGUITY_SHA256,'Frozen no-query ambiguity policy changed')
    require(sha256(full_coverage.__file__)==COVERAGE_SHA256,'Frozen complete anatomy coverage helper changed')
    checker=BulkWorkspace();records=[]
    for path,h in {**checker.bindings,str(Path(__file__).resolve()):sha256(__file__),str(Path(labels.__file__).resolve()):sha256(labels.__file__),str(Path(ambiguity.__file__).resolve()):AMBIGUITY_SHA256,str(Path(full_coverage.__file__).resolve()):COVERAGE_SHA256,str(Path(full_coverage.robot_preview.__file__).resolve()):sha256(full_coverage.robot_preview.__file__)}.items():
        require(path not in pins or pins[path]==h,'Annotation source pin conflicts');pins[path]=h
    output.mkdir(parents=True)
    requests=set();callbacks=set();seen_names=set();previous=None
    for committed_index in indices:
        committed=observations[committed_index]
        # FrameSink manifest rows contain immutable observation file/path pins.
        observation_path=committed.get('observation_path',committed.get('path'))
        observation_hash=committed.get('observation_sha256',committed.get('sha256'))
        obs=read(observation_path,observation_hash)
        optout_worker.require_scope(obs,native=True)
        name=obs['observation_id']
        require(name not in seen_names,'Repeated observation identity');seen_names.add(name)
        require(name in schedule and obs['schema']==plan_api.OBSERVATION_SCHEMA
                and obs['capture_role']=='fully_labeled_qualification_candidate' and obs['source_pose_id']==schedule[name]['source_pose_id']
                and obs['context_sha256']==owner['context_sha256'] and obs['dataset_split']==context['dataset_split'], 'Unscheduled or wrong-epoch observation')
        cached=pose_by_id[obs['source_pose_id']]
        assert_same_camera(obs['calibration'],cached['calibration'])
        require(obs['target_id']==cached['target_id'] and obs['robot_snapshot']['joint_degrees']==cached['joint_degrees']
            and np.allclose(obs['robot_snapshot']['robot_root_to_world_usd_row_vectors'],cached['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0), 'Actual planned embodied pose differs')
        rgb_path=bound(obs['files']['rgb']['path'],obs['files']['rgb']['sha256'])
        with Image.open(rgb_path) as im:
            require(im.mode=='RGB' and im.size==(848,408),'Exact native RGB required');rgb=np.asarray(im).copy()
        with np.load(bound(obs['files']['buffers']['path'],obs['files']['buffers']['sha256']),allow_pickle=False) as a:
            require(set(a.files)=={'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},'Exact callback buffers required')
            depth,ids,valid,alpha=[a[k].copy() for k in ('depth_m','renderer_instance_id','depth_valid','rgba_alpha')]
        mapping_value=read(obs['mapping']['path'],obs['mapping']['sha256'])
        mapping={int(k):v for k,v in mapping_value['renderer_id_to_prim'].items()}
        sync=obs['synchronization'];request=obs['request_evidence']
        require(sync['request_index'] not in requests and sync['callback_sequence'] not in callbacks,'Repeated native request/callback')
        requests.add(sync['request_index']);callbacks.add(sync['callback_sequence'])
        require(request['native_requests']==request['callback_count']==1 and request['requested_subframes']==profile['request_subframes']
                and request['render_settings']==profile['render_settings'] and request['reset_returned_without_exception'] is True
                and request['wait_for_render'] is True and request['delta_time_seconds']==profile['delta_time_seconds'], 'Native profile/reset/request differs')
        payload=dict(obs['native_payload_header']);payload.update(rgb=np.dstack((rgb,alpha)),distance_to_image_plane=depth,
            instance_id_segmentation=dict(data=ids,info=dict(idToLabels=mapping)))
        _,_,checked_valid,reference,token=validate_native_static(payload,obs['calibration'],sync['static_before'],sync['static_after'],sync['callback_sequence'])
        token.update(instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),mapping_sha256=fingerprint(mapping),reference_time=reference)
        require(token==sync['freshness'] and reference==sync['reference_time'] and np.array_equal(valid,checked_valid), 'Actual native callback replay differs')
        require(abs((request['timeline_after_seconds']-request['timeline_before_seconds'])-profile['delta_time_seconds'])<1e-7,'Native scheduler delta differs')
        current_reference=Fraction(*reference)
        if previous is not None:
            require(current_reference>=previous['reference'] and sync['callback_sequence']>previous['sequence']
                and all(token[k]!=previous['token'][k] for k in ('camera_sha256','rgb_sha256','depth_sha256')), 'Ordered freshness/continuity failed')
        previous=dict(reference=current_reference,sequence=sync['callback_sequence'],token=token)
        decode_native_instances(payload,[848,408])
        geometry=read(obs['geometry_proof']['path'],obs['geometry_proof']['sha256'])
        pose=obs['robot_snapshot'];key=plan_api.geometry_pose_key(context['scene_revision'],pose,context['source_bindings_sha256'])
        require(geometry['pose_key']==obs['geometry_proof']['pose_key']==key and geometry['screen']==pose['visual_bound_screen']
                and geometry['scene_revision']==context['scene_revision'] and geometry['source_bindings_sha256']==context['source_bindings_sha256']
                and geometry['screen']['passed'] is True and geometry['joint_degrees']==pose['joint_degrees']
                and geometry['robot_root_to_world_usd_row_vectors']==pose['robot_root_to_world_usd_row_vectors'], 'Per-frame robot geometry differs')
        components,_,renderer_owners=full_coverage.component_masks(ids,mapping,catalogue_index)
        unknown=[];pixel_classes=Counter()
        for rid in np.unique(ids):
            rid=int(rid);path=mapping.get(rid,'');count=int(np.sum(ids==rid));owner_component=renderer_owners[rid]
            require(not any(path==r or path.startswith(r+'/') for r in census['removed_roots']), 'Removed plant appeared in actual native pixels')
            if owner_component:category='authenticated_component_'+owner_component['organ_type']
            elif rid==0 and not path and np.all(~valid[ids==rid]) and np.all(np.isposinf(depth[ids==rid])):category='empty_background'
            elif path.startswith(('/World/Environment/GreenHouse/','/World/Gutters/','/World/RBY1/','/World/Floor','/World/Ground')):category='nonplant_greenhouse_or_robot'
            else:category='unknown_identity';unknown.append(dict(renderer_id=rid,prim_path=path,pixels=count))
            pixel_classes[category]+=count
        metadata=dict(schema_version='greenhouse.native848_no_query_9mm_sample.v1',sample_id=name,
            calibration=obs['calibration'],robot_snapshot=pose,rendered_camera_params=obs['rendered_camera_params'],
            synchronization=dict(**sync,scene_unchanged_during_capture=True,dynamic_recording_supported=False),
            geometry_screen=geometry['screen'],source_family=obs['source_family'],split=obs['dataset_split'],training_approved=False)
        verify_camera(metadata)
        annotation=full_coverage.evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
        folder=output/name;folder.mkdir()
        coverage=dict(schema='greenhouse.native848_all_petiole_coverage_audit.v1',
            observation_path=str(Path(observation_path).resolve()),observation_sha256=observation_hash,
            context_path=owner['context_path'],context_sha256=owner['context_sha256'],
            context=dict(path=owner['context_path'],sha256=owner['context_sha256']),
            capture_split=context['dataset_split'],scene_source_families={r['source_family']:r['source_split'] for r in roots.values()},
            complete_all_eligible_ground_truth=not unknown and not annotation['target_census']['unknown_target_ids'],
            blocking_unknowns=unknown+annotation['target_census']['unknown_target_ids'],visible_cross_split_petiole_targets=[],
            active_catalogue_components=len(catalogue),catalogue_petiole_count=len(inventory),
            actual_fully_labeled_scene_census=dict(path=context['full_scene_census_path'],sha256=context['full_scene_census_sha256']),
            coverage_scope='complete144_manifest_backed_instances',plant_instance_count=144,
            objective_both_arm_outer_exclusion_count=sum(t.get('reason')==full_coverage.OUTSIDE_REASON for t in annotation['targets']),
            clones_count_as_independent_families=False,source_target_identity_policy=LINEAGE_POLICY,
            pixel_categories=dict(pixel_classes),source_bindings=dict(pins),training_approved=False)
        save_json(folder/'coverage.json',coverage)
        annotation['target_census']['full_scene_coverage']=dict(path=str(folder/'coverage.json'),sha256=sha256(folder/'coverage.json'))
        annotation['target_census']['complete']=coverage['complete_all_eligible_ground_truth'] and annotation['target_census']['catalogue_complete']
        annotation['frame_blocked_by_unverified_full_scene_coverage']=not coverage['complete_all_eligible_ground_truth']
        annotation['observation']=dict(path=str(Path(observation_path).resolve()),sha256=observation_hash)
        annotation['source_bindings']=dict(pins)
        annotation['implementation']=dict(path=str(Path(labels.__file__).resolve()),sha256=sha256(labels.__file__))
        annotation['coverage_implementation']=dict(path=str(Path(full_coverage.__file__).resolve()),sha256=COVERAGE_SHA256)
        annotation['source_target_identity_policy']=source_lineage_policy(annotation)
        banned=annotation['source_target_identity_policy']['banned_candidate_instance_ids']
        count=len(annotation['target_census']['candidate_target_ids'])
        strict_unique=annotation['target_census']['complete'] and count==1
        unique=strict_unique and not banned
        annotation['single_target_pilot_eligible']=False
        annotation['single_target_automated_candidate']=bool(unique)
        annotation['state']='unique_candidate_pending_actual_full_frame_visual_review' if unique else 'held_not_proven_single_eligible_target'
        save_json(folder/'sample.json',metadata);save_json(folder/'annotation.json',annotation)
        def spec(path):return dict(path=str(path),sha256=sha256(path))
        # Strict target geometry/status remains unchanged. The separate no-query
        # assessment can only hold a frame; it never promotes a failed target.
        assessment=ambiguity.assess_frame(read_json(folder/'sample.json'),read_json(folder/'annotation.json'),checker,
            metadata_pin=spec(folder/'sample.json'),annotation_pin=spec(folder/'annotation.json'))
        save_json(folder/'ambiguity.json',assessment)
        unique=bool(strict_unique and not banned and assessment['single_answer_unambiguous'] and assessment['assessment_complete'])
        candidate_families={t['source_family'] for t in annotation['targets'] if t['automated_pass']}
        candidate_family=next(iter(candidate_families)) if unique else None
        candidate_sources=[t['source_target_id'] for t in annotation['targets'] if t['automated_pass']]
        records.append(dict(committed_index=committed_index,frame_id=name,source_target_ids=candidate_sources,banned_source_target_instance_ids=banned,source_family=candidate_family,scene_initializer_family=obs['source_family'],split=obs['dataset_split'],
            metadata=spec(folder/'sample.json'),annotation=spec(folder/'annotation.json'),coverage=spec(folder/'coverage.json'),
            rgb=obs['files']['rgb'],buffers=obs['files']['buffers'],candidate_count=count,
            complete_census=annotation['target_census']['complete'],unique_automated_candidate=bool(unique),
            strict_unique_automated_candidate=strict_unique,ambiguity=spec(folder/'ambiguity.json'),
            reachable_alternative_ids=assessment['reachable_alternative_ids'],unknown_alternative_ids=assessment['unknown_alternative_ids'],
            candidate_target_ids=annotation['target_census']['candidate_target_ids'],actual_visual_review=False,
            training_approved=False,accepted_training_increment=0))
    stats=checker.finish();verify_bindings(pins)
    result=dict(schema=SCHEMA,coverage_scope='complete144_manifest_backed_instances',source_target_identity_policy=LINEAGE_POLICY,owner_closure=closure,owner_result=dict(path=str(trial/'result.json'),sha256=result_sha256),
        total_capture_frames=len(observations),evaluated_indices=indices,requested_start_index=start_index,requested_frame_count=frame_count,
        completed_whole_capture_authenticated=True,all_committed_frames_evaluated=indices==list(range(len(observations))),
        records=records,frames_evaluated=len(records),unique_automated_candidates=sum(r['unique_automated_candidate'] for r in records),
        strict_unique_candidates=sum(r['strict_unique_automated_candidate'] for r in records),
        ambiguity_policy=ambiguity.POLICY,workspace_stats=stats,implementation=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)),
        source_bindings=pins,actual_visual_review=False,training_approved=False,accepted_training_increment=0)
    save_json(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True);parser.add_argument('--result-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--start-index',type=int,default=0);parser.add_argument('--frame-count',type=int,default=16)
    parser.add_argument('--parent-metadata');parser.add_argument('--parent-metadata-sha256')
    parser.add_argument('--parent-recovery');parser.add_argument('--parent-recovery-sha256')
    args=parser.parse_args();require(bool(args.parent_metadata)==bool(args.parent_metadata_sha256),'Exact parent metadata pin pair required')
    require(bool(args.parent_recovery)==bool(args.parent_recovery_sha256),'Exact recovery audit pin pair required')
    require(not (args.parent_recovery and args.parent_metadata),'Explicit recovery and successful-parent modes are exclusive')
    parent_metadata=dict(path=str(Path(args.parent_metadata).resolve()),sha256=args.parent_metadata_sha256) if args.parent_metadata else None
    parent_recovery=dict(path=str(Path(args.parent_recovery).resolve()),sha256=args.parent_recovery_sha256) if args.parent_recovery else None
    value=run(args.trial,args.result_sha256,args.output,parent_metadata=parent_metadata,parent_recovery=parent_recovery,start_index=args.start_index,frame_count=args.frame_count)
    print(json.dumps(dict(frames=value['frames_evaluated'],unique_candidates=value['unique_automated_candidates'],training_approved=False)))
