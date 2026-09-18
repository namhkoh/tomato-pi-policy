"""CPU-only annotation of the frozen two-frame far-merge diagnostic.

Coarse IDs never imply invisible components. Every merged semantic petiole
gets a fresh, positive both-arm exclusion at the recorded robot pose. All
retained component pixels use the frozen numeric and ambiguity evaluators.
This is a diagnostic consumer, not a dataset/export schema adapter.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse
import ast
import inspect
import hashlib
import importlib
import sys
import time
import numpy as np
from PIL import Image
from . import native848_farmerge_pair_plan_v1 as api
from . import native848_farmerge_pair_worker_v1 as worker
from . import native848_fully_labeled_coverage_v3 as coverage
from . import native848_pilot_annotation_v16 as original
from . import native848_all_petiole_9mm_v2 as labels
from . import native848_unique9mm_ambiguity_v3 as ambiguity
from .native848_bulk_workspace_v1 import BulkWorkspace
from .native848_bulk_io_v1 import save_json
from .native848_pair_audit_v2 import verify_camera
from .native_greenhouse_pair import assert_same_camera
from .native_sensor_payload import validate_native_static, decode_native_instances
from .capture_contract import fingerprint
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256

SCHEMA = 'greenhouse.native848_farmerge_9mm_diagnostic.v1'
COVERAGE_SCHEMA = 'greenhouse.native848_farmerge_complete_reach_coverage.v1'
ROOT = Path(__file__).resolve().parents[3]
OWNER_DIR = ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'
OWNER_SHA = '9ac80834a958f12a8ae1a0e948e78ce07cf966cfe701c6c53450fd8a00e83c8c'
EPS = 1e-6


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


class Inputs:
    def __init__(self):
        self.bindings = {}

    def bound(self, spec):
        path = Path(spec['path']).resolve()
        require(path.is_file() and sha256(path) == spec['sha256'], 'Changed input: '+str(path))
        old = self.bindings.setdefault(str(path), spec['sha256'])
        require(old == spec['sha256'], 'Conflicting source pin')
        return path

    def read(self, spec):
        return read_json(self.bound(spec))


def compile_groups(catalogue, census, selection, provenance):
    """Exact logical membership and material-subset paths, independent of ID claims."""
    by = {r['component_index']: r for r in catalogue}
    require(len(by) == len(catalogue), 'Repeated component index')
    selected = {r['component_index']: r for r in selection['selected_components']}
    require(len(selected) == len(selection['selected_components']), 'Repeated selected component')
    protected = selection['protected_component_indices']
    require(len(protected) == len(set(protected)) and not set(selected) & set(protected)
            and set(selected) | set(protected) == set(by), 'Incomplete coarse/retained partition')
    for i, r in selected.items():
        require({k:r[k] for k in by[i]} == by[i], 'Selected logical identity differs')
    groups = {}; members = set(); paths = {}
    for g in census['native_coarse_group_catalogue']:
        ids = g['member_component_indices']
        require(g['group_id'] not in groups and len(ids) == len(set(ids)) and ids
                and not members.intersection(ids) and set(ids) <= set(selected), 'Repeated/foreign coarse members')
        require(g['prim_path'] == g['plant_root']+'/__FarCoarseGroup_0'
                and all(by[i]['source_plant_id'] == g['source_family']
                        and by[i]['prim_path'].startswith(g['plant_root']+'/') for i in ids),
                'Coarse group crosses authenticated source/root')
        p = provenance[g['group_id']]
        require(p['merged_mesh_path'] == '/Plant/__FarCoarseGroup_0'
                and p['plant']['plant_root'] == g['plant_root']
                and p['plant']['source_family'] == g['source_family'], 'Foreign face provenance')
        faces = p['faces']
        require({r['original_component_index'] for r in faces} == set(ids), 'Face/member partition differs')
        expected_paths = {g['prim_path']}
        for r in faces:
            row = by[r['original_component_index']]
            require(r['original_component_id'] == row['component_id']
                    and (r['original_mesh_path'] == row['prim_path'] or r['original_mesh_path'].startswith(row['prim_path']+'/'))
                    and r['world_coordinate_max_euclidean_error_m'] <= EPS,
                    'Face provenance identity/rounding differs')
            expected_paths.add(g['prim_path']+'/M_'+r['material_graph_sha256'])
        for path in expected_paths:
            require(path not in paths, 'Repeated coarse renderer path')
            paths[path] = g
        members.update(ids); groups[g['group_id']] = g
    require(members == set(selected), 'Omitted coarse components')
    return dict(selected=selected, groups=groups, paths=paths)


def classify_pixels(ids, mapping_value, catalogue, groups, depth, valid):
    index = coverage.CatalogueIndex(catalogue)
    mapping = {int(k):v for k,v in mapping_value['renderer_id_to_prim'].items()}
    expected = {}; unknown = []; counts = Counter()
    for rid in np.unique(ids):
        rid = int(rid); path = mapping.get(rid, ''); count = int(np.sum(ids == rid))
        if path in groups['paths']:
            g = groups['paths'][path]
            owner = dict(kind='coarse_far_group', group_id=g['group_id'], prim_path=g['prim_path'],
                         member_component_indices=g['member_component_indices'], face_provenance=g['face_provenance'],
                         individual_component_pixel_ID_claimed=False)
            category = 'authenticated_coarse_far_group'; expected[str(rid)] = owner
        elif path.startswith('/World/PackPlants/'):
            row = index.owner(path)
            if row is None or row['component_index'] in groups['selected']:
                category = 'unknown_identity'; unknown.append(dict(renderer_id=rid, prim_path=path, pixels=count))
            else:
                expected[str(rid)] = dict(kind='exact_retained_component', component_index=row['component_index'],
                    variant_id=row['variant_id'], component_id=row['component_id'], source_family=row['source_plant_id'], prim_path=row['prim_path'])
                category = 'authenticated_component_'+row['organ_type']
        elif rid == 0 and not path and np.all(~valid[ids == rid]) and np.all(np.isposinf(depth[ids == rid])):
            category = 'empty_background'
        elif path.startswith(('/World/Environment/GreenHouse/','/World/Gutters/','/World/RBY1/','/World/Floor','/World/Ground')):
            category = 'nonplant_greenhouse_or_robot'
        else:
            category = 'unknown_identity'; unknown.append(dict(renderer_id=rid, prim_path=path, pixels=count))
        counts[category] += count
    saved = mapping_value['plant_ID_ownership']
    require(all(saved.get(k) == v for k,v in expected.items()), 'Saved ownership differs from independently reconstructed identity')
    require(set(saved) <= {str(k) for k in mapping}, 'Ownership without renderer ID')
    components, _, _ = coverage.component_masks(ids, mapping, index)
    # Never attribute packed faces to their original individual component IDs.
    for rid in np.unique(ids):
        if expected.get(str(int(rid)), {}).get('kind') == 'coarse_far_group':
            require(np.all(components[ids == rid] == 0), 'Coarse pixel falsely has individual ownership')
    return components, dict(pixel_categories=dict(counts), blocking_unknowns=unknown,
                            independently_classified_plant_IDs=len(expected))


def merged_rows(annotation, inventory, catalogue, groups, bounds):
    """Replace every coarse zero-pixel shortcut, including off-screen coarse plants."""
    entries = {e['target_id']:e for e in inventory}
    by_id = {r['variant_id']+'/'+r['component_id']:r for r in catalogue}
    replacements = 0
    for n, old in enumerate(annotation['targets']):
        tid = old['target_id']; component = by_id[tid]
        if component['component_index'] not in groups['selected']:
            continue
        entry = entries[tid]; replacements += 1
        base = dict(target_id=tid, source_family=entry['source_family'], source_component_id=component['component_id'],
                    source_target_id=entry['source_target_id'], plant_instance_id=component['variant_id'],
                    annotation_epoch=labels.ANNOTATION_EPOCH, nominal_arc_m=.009,
                    visibility_support_arc_m=list(labels.SUPPORT_ARC_M), support_is_permissible_cut_interval=False,
                    actual_visual_review=False, automated_pass=False, training_approved=False)
        if entry.get('semantic_leaf_petiole_candidate') is False:
            row = dict(base, status='excluded', reason='not_anatomically_eligible_leaf_petiole',
                       anatomy_reason_codes=entry['anatomy_reason_codes'])
        else:
            try:
                require(entry.get('semantic_leaf_petiole_candidate') is True, 'Missing anatomical classification')
                geo = coverage.source_geometry(entry); proof = coverage.outside_proof(tid, geo, bounds)
                require(proof is not None, 'Coarse petiole not excluded at actual robot pose')
                require(all(proof['both_shoulder_distances_m'][a] > b['conservative_probe_reach_m']+bounds['margin_m']+EPS
                            for a,b in bounds['arms'].items()), 'Coarse exclusion lacks packed-coordinate margin')
                row = dict(base, status='excluded', reason=coverage.OUTSIDE_REASON, geometry=geo,
                           cut_world_m=geo['nominal']['world_m'], outer_workspace_exclusion=proof)
                coverage.validate_exclusion(row, bounds)
            except (ValueError, KeyError, AssertionError) as exc:
                row = dict(base, status='unknown', reason='unresolved_coarse_petiole: '+str(exc))
        annotation['targets'][n] = row
    census = annotation['target_census']
    for key,status in [('candidate_target_ids','candidate_pending_visual_review'),('excluded_target_ids','excluded'),('unknown_target_ids','unknown')]:
        census[key] = [r['target_id'] for r in annotation['targets'] if r['status'] == status]
    annotation['frame_blocked_by_unknown_targets'] = bool(census['unknown_target_ids'])
    return replacements


def actual_coarse_bounds(groups, bounds):
    """Check every selected component AABB again, with 1um packing expansion."""
    rows = list(groups['selected'].values())
    low = np.asarray([r['original_world_min'] for r in rows], float)-EPS
    high = np.asarray([r['original_world_max'] for r in rows], float)+EPS
    require(low.shape == high.shape == (len(rows),3) and np.isfinite(low).all()
            and np.isfinite(high).all() and (high >= low).all(), 'Invalid original component bounds')
    clearance = np.full(len(rows), np.inf)
    for arm in bounds['arms'].values():
        center = np.asarray(arm['shoulder_world_m'])
        dist = np.linalg.norm(np.maximum(np.maximum(low-center, center-high), 0), axis=1)
        clearance = np.minimum(clearance, dist-arm['conservative_probe_reach_m']-bounds['margin_m'])
    failed = [rows[i]['component_index'] for i in np.flatnonzero(clearance <= 0)]
    return dict(component_count=len(rows), all_components_outside_both_actual_arm_bounds=not failed,
                nonexcluded_component_indices=failed, minimum_clearance_m=float(clearance.min()),
                packed_world_coordinate_expansion_m=EPS, bounds_sha256=fingerprint(bounds), visibility_assumed=False)


def adapted_numeric_trees():
    """Exactly two type-dispatch substitutions; no numeric/policy edits."""
    census=ast.parse(inspect.getsource(ambiguity.dataset.validate_census))
    changed=0
    for node in ast.walk(census):
        if isinstance(node,ast.Constant) and node.value=='greenhouse.native848_all_petiole_coverage_audit.v1':
            node.value=COVERAGE_SCHEMA;changed+=1
    require(changed==1,'Frozen census schema dispatch changed')
    assessment=ast.parse(inspect.getsource(ambiguity.assess_frame));changed=0
    for node in ast.walk(assessment):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='dataset' and node.func.attr=='validate_census':
            node.func=ast.Name(id='validate_farmerge_census',ctx=ast.Load());changed+=1
    require(changed==1,'Frozen ambiguity census dispatch changed')
    return ast.fix_missing_locations(census),ast.fix_missing_locations(assessment)


def validate_farmerge_census(annotation):
    audit=read_json(ambiguity.dataset.read_pin(annotation['target_census']['full_scene_coverage']))
    require(audit['schema']==COVERAGE_SCHEMA and audit['individual_coarse_component_pixel_visibility_claimed'] is False
            and audit['all_near_components_evaluated_with_unchanged_numeric_predicates'] is True
            and audit['coarse_actual_pose_bounds']['all_components_outside_both_actual_arm_bounds'] is True
            and audit['coarse_actual_pose_bounds']['packed_world_coordinate_expansion_m']==EPS
            and audit['coarse_actual_pose_bounds']['minimum_clearance_m']>0,
            'Complete typed coarse reach coverage required')
    census,_=adapted_numeric_trees();namespace=dict(vars(ambiguity.dataset))
    exec(compile(census,__file__+'::typed_census','exec'),namespace)
    return namespace['validate_census'](annotation)


def assess_farmerge(metadata,annotation,checker,**pins):
    _,assessment=adapted_numeric_trees();namespace=dict(vars(ambiguity))
    namespace['validate_farmerge_census']=validate_farmerge_census
    exec(compile(assessment,__file__+'::unchanged_numeric_ambiguity','exec'),namespace)
    return namespace['assess_frame'](metadata,annotation,checker,**pins)


def authenticate(trial, owner_sha, outer, inputs):
    owner_path = OWNER_DIR/'run_native848_farmerge_pair_serial_v1.py'
    require(sha256(owner_path) == OWNER_SHA, 'Frozen farmerge owner changed')
    sys.path.insert(0, str(OWNER_DIR))
    owner_api = importlib.import_module('run_native848_farmerge_pair_serial_v1')
    owner = inputs.read(dict(path=str(trial/'result.json'), sha256=owner_sha))
    closure = owner_api.check_predecessor(dict(metadata=pin(outer/'metadata.json'),
        result=pin(trial/'result.json'), terminal=pin(trial/'owner_complete.json')))
    outer_pins = owner_api.check_outer_predecessor(dict(predecessor_outer_launch=pin(outer/'launch.json'),
        predecessor_outer_exit=pin(outer/'owned_exit.json')), closure)
    inputs.bindings.update(closure['source_bindings']); inputs.bindings.update(outer_pins)
    context = inputs.read(dict(path=owner['context_path'], sha256=owner['context_sha256']))
    plan = inputs.read(dict(path=context['plan_path'], sha256=context['plan_sha256']))
    manifest = inputs.read(dict(path=owner['manifest_path'], sha256=owner['manifest_sha256']))
    require(context['source_bindings_sha256'] == fingerprint(context['source_bindings']), 'Context source fingerprint differs')
    verify_bindings(context['source_bindings']); inputs.bindings.update(context['source_bindings'])
    for module,h in [(coverage,original.COVERAGE_SHA256),(ambiguity,original.AMBIGUITY_SHA256),
                     (labels,'c5704371017b9f5943664e8a4d156d5e7f2e6d71f8b48ef8e5f4efee01ae13c1'),
                     (labels.joint_ownership,'37f7b121202a3e04c45166ab47c8e976bc7d4c74cfa546d1ac1ae65bf4d2f6aa')]:
        inputs.bound(dict(path=module.__file__,sha256=h))
    inputs.bound(pin(original.__file__))
    return owner, context, plan, manifest, closure


def run(trial, owner_sha, outer, output):
    started = time.perf_counter(); trial=Path(trial).resolve(); outer=Path(outer).resolve(); output=Path(output).resolve()
    require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(), 'Create-only local diagnostic output')
    inputs=Inputs(); owner,context,plan,manifest,closure=authenticate(trial,owner_sha,outer,inputs)
    print('Authenticated completed native owner and original collision evidence', flush=True)
    census=inputs.read(dict(path=context['full_scene_census_path'],sha256=context['full_scene_census_sha256']))
    logical=inputs.read(dict(path=context['logical_original_census_path'],sha256=context['logical_original_census_sha256']))
    logical_context=dict(context,full_scene_census=logical)
    reports_list=inputs.read(dict(path=context['source_reports_path'],sha256=context['source_reports_sha256']))
    reports={r['plant_id']:r for r in reports_list};require(len(reports)==len(reports_list),'Duplicate source reports')
    catalogue=inputs.read(dict(path=context['catalogue_path'],sha256=context['catalogue_sha256']))
    require(catalogue==original.expected_catalogue(logical_context,reports),'Logical complete anatomy catalogue differs')
    inventory=coverage.target_inventory(logical_context,reports,catalogue)
    source_plan=inputs.read(logical['source_collection_plan'])
    require(all(source_plan['family_assignments'][r['source_family']]==context['dataset_split'] for r in logical['all_plant_roots']), 'Cross-split source geometry')
    prototype=inputs.read(api.PROTOTYPE);selection=inputs.read(prototype['selection'])
    require(all(g['selection']==prototype['selection'] for g in census['native_coarse_group_catalogue']), 'Coarse selection pin differs')
    provenance={g['group_id']:inputs.read(g['face_provenance']) for g in census['native_coarse_group_catalogue']}
    groups=compile_groups(catalogue,census,selection,provenance)
    cache=inputs.read(dict(path=plan['cache_path'],sha256=plan['cache_sha256']));poses={r['sample_id']:r for r in cache['records']}
    observations=original.completed_schedule(plan,manifest);require(len(observations)==2,'Exactly the completed pair required')
    profile=inputs.read(plan['profile_evidence']);api.check_profile(profile)
    output.mkdir(parents=True);checker=BulkWorkspace();records=[];previous=None
    inputs.bindings.update(checker.bindings);inputs.bindings[str(Path(__file__).resolve())]=sha256(__file__)
    try:
        for committed in observations:
            obs=inputs.read(dict(path=committed['path'],sha256=committed['sha256']));worker.require_scope(obs,native=True)
            cached=poses[obs['source_pose_id']];assert_same_camera(obs['calibration'],cached['calibration'])
            require(obs['robot_snapshot']['joint_degrees']==cached['joint_degrees'] and np.allclose(obs['robot_snapshot']['robot_root_to_world_usd_row_vectors'],cached['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0),'Recorded embodied pose differs')
            with Image.open(inputs.bound(obs['files']['rgb'])) as im:
                require(im.mode=='RGB' and im.size==(848,408),'Native RGB848 required');rgb=np.asarray(im).copy()
            with np.load(inputs.bound(obs['files']['buffers']),allow_pickle=False) as data:
                require(set(data.files)=={'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},'Exact callback buffers required')
                depth,ids,valid,alpha=[data[k].copy() for k in ('depth_m','renderer_instance_id','depth_valid','rgba_alpha')]
            mv=inputs.read(obs['mapping']);mapping={int(k):v for k,v in mv['renderer_id_to_prim'].items()}
            sync=obs['synchronization'];request=obs['request_evidence']
            payload=dict(obs['native_payload_header']);payload.update(rgb=np.dstack((rgb,alpha)),distance_to_image_plane=depth,instance_id_segmentation=dict(data=ids,info=dict(idToLabels=mapping)))
            _,_,checked_valid,reference,token=validate_native_static(payload,obs['calibration'],sync['static_before'],sync['static_after'],sync['callback_sequence'])
            token.update(instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),mapping_sha256=fingerprint(mapping),reference_time=reference)
            require(token==sync['freshness'] and reference==sync['reference_time'] and np.array_equal(valid,checked_valid),'Native callback freshness replay differs')
            require(request['requested_subframes']==6 and request['delta_time_seconds']==0 and request['render_settings']==profile['render_settings'],'Frozen renderer request differs')
            decode_native_instances(payload,[848,408])
            if previous:
                require(Fraction(*reference)>=previous[0] and sync['callback_sequence']>previous[1] and all(token[k]!=previous[2][k] for k in ('camera_sha256','rgb_sha256','depth_sha256')),'Repeated/stale native callback')
            previous=(Fraction(*reference),sync['callback_sequence'],token)
            geometry=inputs.read(obs['geometry_proof']);pose=obs['robot_snapshot'];key=api.geometry_pose_key(context['scene_revision'],pose,context['source_bindings_sha256'])
            require(geometry['pose_key']==obs['geometry_proof']['pose_key']==key and geometry['screen']==pose['visual_bound_screen'] and geometry['screen']['passed'] is True and geometry['source_bindings_sha256']==context['source_bindings_sha256'] and geometry['scene_revision']==context['scene_revision'],'Actual original collision proof differs')
            components,pixel_audit=classify_pixels(ids,mv,catalogue,groups,depth,valid)
            metadata=dict(schema_version='greenhouse.native848_no_query_9mm_sample.v1',sample_id=obs['observation_id'],calibration=obs['calibration'],robot_snapshot=pose,rendered_camera_params=obs['rendered_camera_params'],synchronization=dict(**sync,scene_unchanged_during_capture=True,dynamic_recording_supported=False),geometry_screen=geometry['screen'],source_family=obs['source_family'],split=obs['dataset_split'],training_approved=False)
            verify_camera(metadata)
            annotation=coverage.evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
            bounds=annotation['outer_workspace_bounds'];aabb=actual_coarse_bounds(groups,bounds)
            replaced=merged_rows(annotation,inventory,catalogue,groups,bounds)
            for target in annotation['targets']:
                joint=target.get('attachment_joint_ownership')
                if joint:require(all(inputs.bindings.get(p)==h for p,h in joint['source_bindings'].items()),'Joint evidence not bound to captured source geometry')
            folder=output/obs['observation_id'];folder.mkdir()
            unknown=pixel_audit['blocking_unknowns']+annotation['target_census']['unknown_target_ids']
            if not aabb['all_components_outside_both_actual_arm_bounds']:unknown.append('coarse_component_not_outside_actual_bounds')
            audit=dict(schema=COVERAGE_SCHEMA,observation_path=committed['path'],observation_sha256=committed['sha256'],visible_cross_split_petiole_targets=[],source_bindings=dict(inputs.bindings),observation=dict(path=committed['path'],sha256=committed['sha256']),logical_census=dict(path=context['logical_original_census_path'],sha256=context['logical_original_census_sha256']),native_render_census=dict(path=context['full_scene_census_path'],sha256=context['full_scene_census_sha256']),selection=prototype['selection'],**pixel_audit,coarse_actual_pose_bounds=aabb,coarse_petiole_rows_replaced=replaced,complete_all_eligible_ground_truth=not unknown,blocking_target_or_geometry_unknowns=unknown,individual_coarse_component_pixel_visibility_claimed=False,all_near_components_evaluated_with_unchanged_numeric_predicates=True,training_approved=False)
            save_json(folder/'coverage.json',audit)
            annotation['target_census'].update(full_scene_coverage=pin(folder/'coverage.json'),complete=not unknown and annotation['target_census']['catalogue_complete'])
            annotation.update(frame_blocked_by_unverified_full_scene_coverage=bool(unknown),observation=audit['observation'],source_bindings=dict(inputs.bindings),implementation=pin(labels.__file__),coverage_implementation=pin(__file__),source_target_identity_policy=original.source_lineage_policy(annotation),single_target_pilot_eligible=False)
            annotation['single_target_automated_candidate']=annotation['target_census']['complete'] and len(annotation['target_census']['candidate_target_ids'])==1
            save_json(folder/'sample.json',metadata);save_json(folder/'annotation.json',annotation)
            assessment=assess_farmerge(read_json(folder/'sample.json'),read_json(folder/'annotation.json'),checker,metadata_pin=pin(folder/'sample.json'),annotation_pin=pin(folder/'annotation.json'))
            save_json(folder/'ambiguity.json',assessment)
            strong=annotation['single_target_automated_candidate'] and not annotation['source_target_identity_policy']['banned_candidate_instance_ids'] and assessment['assessment_complete'] and assessment['single_answer_unambiguous']
            records.append(dict(frame_id=obs['observation_id'],source_pose_id=obs['source_pose_id'],metadata=pin(folder/'sample.json'),annotation=pin(folder/'annotation.json'),coverage=pin(folder/'coverage.json'),ambiguity=pin(folder/'ambiguity.json'),candidate_target_ids=annotation['target_census']['candidate_target_ids'],strict_unique=annotation['single_target_automated_candidate'],strong_candidate=bool(strong),reachable_alternative_ids=assessment['reachable_alternative_ids'],unknown_alternative_ids=assessment['unknown_alternative_ids'],coarse_petiole_count=replaced,unknown_count=len(unknown),actual_visual_review=False,training_approved=False))
            print(dict(frame=len(records),strict=records[-1]['strict_unique'],strong=bool(strong),unknowns=len(unknown)),flush=True)
    finally:
        stats=checker.finish()
    verify_bindings(inputs.bindings)
    result=dict(schema=SCHEMA,owner_result=pin(trial/'result.json'),owner_closure=closure,frames=records,frames_evaluated=len(records),strong_candidates=sum(r['strong_candidate'] for r in records),source_bindings=inputs.bindings,workspace_stats=stats,numerical_thresholds_unchanged=True,ordinary_annotation_dispatch_compatible=False,export_adapter_implemented=False,actual_visual_review=False,training_approved=False,accepted_training_increment=0,elapsed_seconds=time.perf_counter()-started)
    save_json(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True);parser.add_argument('--result-sha256',required=True);parser.add_argument('--outer',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.trial,args.result_sha256,args.outer,args.output)
    print(dict(result=pin(Path(args.output)/'result.json'),frames=result['frames_evaluated'],strong=result['strong_candidates']),flush=True)
