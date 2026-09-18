"""Typed generated foreground9mm candidates with original donor lineage.

One complete generated plant and143 complete original instances. Numerical
predicates are frozen epoch2; a generated geometry ID never becomes a new donor.
This CPU module does not capture, review, export, or grant training acceptance.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint,depth_evidence
from . import native848_all_petiole_9mm_v2 as labels
from . import native848_fully_labeled_coverage_v3 as original
from . import native848_joint_ownership_v1 as joint
from . import native848_unique9mm_ambiguity_v3 as ambiguity
from . import native848_unique9mm_joint_contract_v1 as answer_contract

SCHEMA='greenhouse.native848_generated_9mm_annotation.v1'
CONTEXT_SCHEMA='greenhouse.native848_generated_population_context.v1'
CENSUS_SCHEMA='greenhouse.native848_controlled_morphology_census.v1'
INVENTORY_SCHEMA='greenhouse.native848_generated_target_inventory.v1'
JOINT_SCHEMA='greenhouse.native848_generated_attachment_joint_owner.v1'
AMBIGUITY_SCHEMA='greenhouse.native848_generated_9mm_ambiguity.v1'
ANNOTATION_EPOCH=labels.ANNOTATION_EPOCH
NOMINAL_ARC_M=labels.NOMINAL_ARC_M
SUPPORT_ARC_M=labels.SUPPORT_ARC_M
geometry_9mm=labels.geometry_9mm
local_screen=labels.local_screen
CatalogueIndex=original.CatalogueIndex
component_masks=original.component_masks
BANNED_SOURCE_TARGET='seed41_full/SubStem_38'


def _pin(spec):
    require(isinstance(spec,dict) and set(spec)=={'path','sha256'},'Exact evidence pin required')
    require(isinstance(spec['path'],str) and Path(spec['path']).is_absolute(),'Absolute evidence path required')
    p=Path(spec['path']).resolve()
    require(p.is_file() and sha256(p)==spec['sha256'],'Evidence changed: '+str(p))
    return p


def _unique(rows,key):
    result={r[key]:r for r in rows}
    require(len(result)==len(rows),'Duplicate '+key)
    return result


def _identity(entry,catalogue):
    instance,cid=entry['target_id'].split('/')
    report=entry['report'];geometry=entry['geometry_source_id'];donor=entry['source_family']
    require(report['plant_id']==geometry and donor==entry['split_group']
        and entry['source_target_id']==donor+'/'+cid and entry['geometry_target_id']==geometry+'/'+cid,
        'Generated geometry and original donor identities must remain separate')
    matches=[r for r in catalogue if r['variant_id']==instance and r['component_id']==cid]
    require(len(matches)==1 and matches[0]['geometry_source_id']==geometry
        and matches[0]['source_plant_id']==matches[0]['split_group']==donor,'Exact instance/donor/geometry required')
    return matches[0]


def generated_joint_owner(entry,catalogue):
    """Same authored candidate and actual shared mesh seam; only identity differs."""
    _identity(entry,catalogue)
    variant,cid=entry['target_id'].split('/');report=entry['report']
    candidate=joint.candidate_stub(report,cid)
    if candidate is None:return set(),None
    ancestor=candidate['ancestor_component_id']
    matches=[r for r in catalogue if r['variant_id']==variant and r['component_id']==ancestor]
    require(len(matches)==1 and matches[0]['organ_type']=='main_stem'
        and matches[0]['source_plant_id']==entry['source_family']
        and matches[0]['geometry_source_id']==entry['geometry_source_id'],'Same generated instance joint owner required')
    bindings={}
    manifest_path=joint._bound(report['manifest_path'],report['manifest_sha256'],bindings)
    raw={c['id']:c for c in read_json(manifest_path)['components']};paths={}
    for key in (cid,candidate['declared_parent_component_id'],ancestor):
        c,source=report['components'][key],raw[key]
        fields=(('parent','parent'),('type','type'),('file','file'),
            ('attachment_plant_m','attach_point'),('capsules_local_m','capsules'),('radius_m','radius'))
        require(all(c[a]==source[b] for a,b in fields)
            and c['translation_plant_m']==source['transform']['translate'],'Generated report topology differs from manifest')
        path=(manifest_path.parent/source['file']).resolve()
        require(path.parent==manifest_path.parent,'Bound generated component required')
        paths[key]=joint._bound(path,c['asset_sha256'],bindings)
    stem=joint._mesh_points(paths[ancestor],report['components'][ancestor]['translation_plant_m'],bindings)
    petiole=joint._mesh_points(paths[cid],report['components'][cid]['translation_plant_m'],bindings)
    seam=joint.mesh_seam(stem,petiole,candidate)
    verify_bindings(bindings)
    if seam is None:return set(),None
    return {matches[0]['component_index']},dict(schema=JOINT_SCHEMA,target_id=entry['target_id'],
        source_family=entry['source_family'],source_target_id=entry['source_target_id'],
        geometry_source_id=entry['geometry_source_id'],geometry_target_id=entry['geometry_target_id'],
        plant_instance_id=variant,additional_main_stem_component_id=ancestor,
        additional_component_index=matches[0]['component_index'],
        eligibility='exact_immediate_main_stem_ancestor_with_authored_matching_attachment_stub',
        candidate=candidate,mesh_seam=seam,source_bindings=bindings,allowed_only_below_arc_m=.008,
        depth_predicate_unchanged=True,nominal_and_support_ownership_unchanged=True)


def evaluate_generated_target(metadata,rgb,depth,valid,components,catalogue,entry,workspace_checker):
    _identity(entry,catalogue)
    require(entry.get('generated_geometry') is True,'Generated entry required')
    target=_identity(entry,catalogue)
    # Zero pixels and semantic ineligibility do not need an invented thickness.
    if np.any(components==target['component_index']) and entry.get('semantic_leaf_petiole_candidate') is True:
        require(entry.get('physical_geometry_verified') is True,'Generated current9mm physical geometry correspondence unavailable')
    return _evaluate_geometry_target(metadata,rgb,depth,valid,components,catalogue,entry,workspace_checker)



def _evaluate_geometry_target(metadata, rgb, depth, valid, components, catalogue, entry, workspace_checker):
    target_id = entry['target_id']
    variant, component_id = target_id.split('/')
    report = entry['report']
    require(report['plant_id'] == entry['geometry_source_id'], 'Report/geometry identity differs')
    target_matches = [c for c in catalogue if c['variant_id'] == variant and c['component_id'] == component_id]
    require(len(target_matches) == 1 and target_matches[0]['organ_type'] == 'sub_stem', 'Exact petiole catalogue identity required')
    target = target_matches[0]
    base = dict(target_id=target_id, source_family=entry['source_family'],
                status='unknown', reason=None, annotation_epoch=ANNOTATION_EPOCH,
                nominal_arc_m=NOMINAL_ARC_M, visibility_support_arc_m=list(SUPPORT_ARC_M),
                support_is_permissible_cut_interval=False, actual_visual_review=False,
                automated_pass=False, training_approved=False)
    if not np.any(components == target['component_index']):
        return dict(base, status='excluded', reason='not_visible_zero_authenticated_component_pixels')
    if entry.get('semantic_leaf_petiole_candidate') is False:
        return dict(base, status='excluded', reason='not_anatomically_eligible_leaf_petiole',
                    anatomy_reason_codes=entry.get('anatomy_reason_codes', []))
    anatomy_targets = [t for t in report.get('targets', []) if t['component_id'] == component_id]
    if anatomy_targets:
        require(len(anatomy_targets) == 1, 'Ambiguous anatomical target')
        anatomy = anatomy_targets[0]
        leaves = [key for key in anatomy['expected_detached_component_ids']
                  if report['components'][key]['type'] == 'leaf']
        if anatomy['protected_descendant_ids'] or not leaves:
            return dict(base, status='excluded', reason='protected_descendant_or_no_leaf_petiole')
    elif entry.get('semantic_leaf_petiole_candidate') is not True:
        return dict(base, reason='missing_anatomical_leaf_petiole_classification')
    component = report['components'][component_id]
    if component.get('type') != 'sub_stem' or component.get('deleafed') is not False:
        return dict(base, status='excluded', reason='not_intact_petiole')
    if component.get('parent') not in report['components']:
        return dict(base, reason='missing_parent_anatomy')
    if report['components'][component['parent']]['type'] != 'main_stem':
        return dict(base, status='excluded', reason='not_direct_main_stem_petiole')
    parents = [c for c in catalogue if c['variant_id'] == variant and c['component_id'] == component['parent']]
    require(len(parents) == 1, 'Exact main stem catalogue identity required')
    parent = parents[0]
    geo = geometry_9mm(report, component_id, entry['plant_to_world_usd_row_vectors'], metadata['calibration'])
    base.update(geometry=geo, cut_world_m=geo['nominal']['world_m'],
                cut_point_uv=geo['nominal']['projected']['pixel_xy'],
                cut_optical_xyz_m=geo['nominal']['projected']['camera_optical_xyz_m'])
    target_y, target_x = np.nonzero(components == target['component_index'])
    base['target_bbox_convention'] = 'xmin_ymin_inclusive_xmax_ymax_exclusive_native_pixels'
    base['target_bbox_xyxy'] = ([int(target_x.min()), int(target_y.min()), int(target_x.max()+1), int(target_y.max()+1)]
                                if len(target_x) else None)

    def probe(point, allowed):
        p = point['projected']
        evidence = depth_evidence(p, depth, valid, point['radius_m'])
        visible, observed = False, None
        if p['projection_status'] == 'in_frame':
            x, y = np.floor(p['pixel_xy']).astype(int)
            observed = int(components[y, x])
            visible = observed in allowed and evidence['status'] == 'depth_consistent_not_visibility_verified'
        return dict(arc_m=point['arc_m'], projected=p, depth=evidence,
                    observed_component_index=observed, visible=bool(visible))

    # Only an authenticated mesh seam can add a same-instance joint owner.
    additional_parent_indices, joint_proof = generated_joint_owner(entry, catalogue)
    parent_indices = {parent['component_index']} | additional_parent_indices
    base['attachment_joint_ownership'] = joint_proof
    own = {target['component_index']}
    nominal = probe(geo['nominal'], own)
    support = [probe(p, own) for p in geo['visibility_support']]
    proximal = [probe(p, own if p['arc_m'] >= .008 else own | parent_indices) for p in geo['proximal']]
    junction = [probe(p, own if p['arc_m'] >= .008 else own | parent_indices) for p in geo['junction_to_cut']]
    base['junction_continuity'] = dict(attachment_visible=junction[0]['visible'],
        all_attachment_to_9mm_probes_visible=all(p['visible'] for p in junction),
        maximum_arc_step_m=.0005, probes=junction)
    base['visibility'] = dict(nominal=nominal, support=support, proximal=proximal)
    if not nominal['visible'] or not all(p['visible'] for p in support):
        uncertain = any(p['projected']['projection_status'] == 'in_frame'
                        and p['depth']['status'].startswith('unknown') for p in [nominal, *support])
        return dict(base, status='unknown' if uncertain else 'excluded', reason='cut_or_support_not_native_visible')
    if not all(p['visible'] for p in junction):
        uncertain = any(p['projected']['projection_status'] == 'in_frame'
                        and p['depth']['status'].startswith('unknown') for p in junction)
        return dict(base, status='unknown' if uncertain else 'excluded', reason='attachment_to_9mm_continuity_not_verified')
    unique = {}
    for p in proximal:
        if p['projected']['projection_status'] == 'in_frame':
            unique.setdefault(tuple(np.floor(p['projected']['pixel_xy']).astype(int)), []).append(p['visible'])
    fraction = sum(all(values) for values in unique.values())/len(unique) if unique else 0.
    mask = components == target['component_index']
    screen = local_screen(rgb, mask, [p['projected']['pixel_xy'] for p in support], fraction)
    base['local_clarity'] = screen
    if not screen['passed']:
        return dict(base, status='excluded', reason='strict_native_local_clarity_failed')
    attach = geo['attachment_projected']
    if attach['projection_status'] != 'in_frame':
        return dict(base, status='excluded', reason='attachment_out_of_frame')
    x, y = np.floor(attach['pixel_xy']).astype(int)
    # Apply the same clearance formula to every proven joint-forming stem.
    parent_mask = np.isin(components, sorted(parent_indices))
    count = int(parent_mask[max(0, y-12):min(408, y+13), max(0, x-12):min(848, x+13)].sum())
    yy, xx = np.nonzero(parent_mask)
    uv = np.asarray(base['cut_point_uv'])
    distance = float(np.min(np.hypot(xx+.5-uv[0], yy+.5-uv[1]))) if len(xx) else None
    z = geo['nominal']['projected']['camera_optical_xyz_m'][2]
    cal = metadata['calibration']
    diameter = 2*geo['nominal']['radius_m']*min(cal['intrinsics'][0][0], cal['intrinsics'][1][1])/z
    base['parent_context'] = dict(parent_component_indices=sorted(parent_indices),
                                  parent_scope='declared_parent_and_authenticated_joint_forming_segment',
                                  visible_parent_pixels=count, cut_to_parent_distance_px=distance,
                                  required_distance_px=max(3., diameter*.65), estimated_diameter_px=float(diameter))
    if count < 6 or distance is None or distance < max(3., diameter*.65):
        return dict(base, status='excluded', reason='insufficient_visible_parent_clearance')
    # The solver's exact cache key includes this NEW target. No 10 mm IK reuse.
    meta = deepcopy(metadata)
    meta['supervision'] = dict(target_id=target_id, nominal_world_m=base['cut_world_m'])
    workspace = workspace_checker.check(meta)
    require(workspace['target_id'] == target_id
            and np.allclose(workspace['nominal_world_m'], base['cut_world_m'], atol=1e-12, rtol=0),
            'Workspace proof must bind the new 9 mm point')
    base['workspace'] = workspace
    if workspace['result']['workspace_passed'] is not True:
        return dict(base, status='excluded', reason='workspace_9mm_failed')
    return dict(base, status='candidate_pending_visual_review', reason='9mm_local_visibility_and_workspace_passed',
                automated_pass=True)



def build_inventory(context,reports,catalogue):
    """Authenticate all144 anatomical reports once, then enumerate every petiole."""
    from .audit import audit_manifest
    from .native848_controlled_9mm_evidence_v4 import authenticate_9mm_evidence
    require(context.get('schema')==CONTEXT_SCHEMA and context.get('dataset_split')=='train',
        'Typed TRAIN generated population required')
    require(context.get('native_population_census_verified') is True
        and context.get('original_143_backgrounds_preserved') is True,'Actual generated population verification required')
    bindings=dict(context['source_bindings']);verify_bindings(bindings)
    census=context['full_scene_census']
    require(census.get('schema')==CENSUS_SCHEMA and census.get('native_population_census_verified') is True,
        'Actual typed controlled population census required')
    identity=deepcopy(census);saved=identity.pop('deterministic_census_sha256')
    require(saved==fingerprint(identity),'Generated census fingerprint differs')
    require(census['unchanged_background_count']==143 and census['removed_roots']==[]
        and census['active_counts']['component_plants']==144 and census['active_counts']['backdrop_instances']==0,
        'Exactly143 originals plus one generated full plant required')
    plan_pin=census['source_collection_plan']
    require(context['source_collection_plan']==plan_pin,'Inventory source plan differs from actual scene census')
    plan=read_json(_pin(plan_pin))
    require(bindings.get(str(Path(plan_pin['path']).resolve()))==plan_pin['sha256'],'Unbound donor split plan')
    roots=_unique(census['all_plant_roots'],'plant_root')
    variants=_unique(context['scene_variants'],'variant_id')
    require(len(roots)==len(variants)==144 and {v['plant_root'] for v in variants.values()}==set(roots),
        'All144 original planting slots required')
    reports=_unique(reports,'plant_id') if isinstance(reports,list) else reports
    index=CatalogueIndex(catalogue)
    require(set(index.by_variant)==set(variants),'Catalogue omitted plant instance')
    generated=[v for v in variants.values() if v.get('source_geometry_modified') is True]
    require(len(generated)==1,'Exactly one generated foreground required')
    modified=generated[0];geometry=modified['geometry_source_id'];family=modified['source_plant_id']
    require(geometry!=family and set(context['geometry_sources'])=={geometry},'Distinct single generated geometry source required')
    spec=context['geometry_sources'][geometry]
    require(spec['donor_source_family']==family and spec['source_split']=='train','Generated donor split differs')
    require(census['source_qualification']==spec['qualification'] and census['foreground_root']==modified['plant_root'],
        'Generated foreground qualification/root differs from actual census')
    generated_report=reports[geometry];donor_report=reports[family]
    require(Path(generated_report['manifest_path']).resolve()==_pin(spec['manifest'])
        and generated_report['manifest_sha256']==spec['manifest']['sha256'],'Generated manifest differs')
    _pin(spec['qualification']);_pin(spec['physical_geometry_evidence'])
    physical=authenticate_9mm_evidence(spec['physical_geometry_evidence'],generated_report=generated_report,
        donor_report=donor_report,qualification_pin=spec['qualification'])
    verify_bindings(physical['source_bindings'])
    qualified=set(physical['radius_qualified_component_ids'])
    require(qualified<=set(generated_report['components']),'Foreign physical radius component')
    used_geometry={v.get('geometry_source_id',v['source_plant_id']) for v in variants.values()}|{family}
    for gid in used_geometry:
        report=reports[gid];manifest=Path(report['manifest_path']).resolve()
        require(report['plant_id']==gid and bindings.get(str(manifest))==report['manifest_sha256'],
            'Anatomical report must bind its geometry manifest')
        fresh=audit_manifest(manifest)
        require(all(report[k]==fresh[k] for k in ('plant_id','manifest_sha256','components','targets','status')),
            'Anatomical report differs from reconstructed source')
        for component in report['components'].values():
            path=str((manifest.parent/component['file']).resolve())
            require(bindings.get(path)==component['asset_sha256'],'Unbound anatomical component asset')
    entries=[];expected=[]
    for instance,variant in sorted(variants.items()):
        root=roots[variant['plant_root']];donor=variant['source_plant_id'];gid=variant.get('geometry_source_id',donor)
        is_generated=instance==modified['variant_id']
        require(variant['split_group']==root['source_family']==donor
            and root['source_split']==plan['family_assignments'][donor]==context['dataset_split']
            and root['active_after_policy'] is True and root['authenticated_component_plant'] is True,
            'Inactive, cross-split, or relabeled donor')
        require((is_generated and gid==geometry and variant['source_geometry_modified'] is True)
            or (not is_generated and gid==donor and variant.get('source_geometry_modified') is False),
            'Only one explicitly generated plant allowed')
        require(not variant.get('added_components') and not variant.get('added_component_paths'),'Changed anatomical graph unsupported')
        report=reports[gid]
        require(Path(root['manifest_path']).resolve()==Path(report['manifest_path']).resolve()
            and root['manifest_sha256']==report['manifest_sha256'],'Census geometry manifest differs')
        matrix=np.asarray(root['plant_to_world_usd_row_vectors'],float)
        require(matrix.shape==(4,4) and np.isfinite(matrix).all()
            and np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),rtol=0,atol=1e-7)
            and abs(np.linalg.det(matrix[:3,:3])-1)<1e-7
            and np.allclose(matrix[:,3],[0,0,0,1],rtol=0,atol=1e-9),'Rigid metre-scale source placement required')
        require(root['component_count']==len(report['components'])==len(index.by_variant[instance]),'Incomplete component population')
        targets=_unique(report['targets'],'component_id')
        for cid,c in report['components'].items():
            chain=[cid];parent=c['parent']
            while parent is not None:
                require(parent in report['components'] and parent not in chain,'Invalid anatomy graph')
                chain.append(parent);parent=report['components'][parent]['parent']
            path=variant['plant_root']+'/'+'/'.join(reversed(chain))
            expected.append(dict(component_id=cid,prim_path=path,organ_type=c['type'],variant_id=instance,
                source_plant_id=donor,split_group=donor,geometry_source_id=gid))
            if c['type']!='sub_stem':continue
            anatomy=targets[cid]
            leaves=[k for k in anatomy['expected_detached_component_ids'] if report['components'][k]['type']=='leaf']
            semantic=(c.get('deleafed') is False and report['components'].get(c['parent'],{}).get('type')=='main_stem'
                      and bool(leaves) and not anatomy['protected_descendant_ids'])
            entry=dict(schema=INVENTORY_SCHEMA,target_id=instance+'/'+cid,source_family=donor,split_group=donor,
                source_target_id=donor+'/'+cid,source_component_id=cid,geometry_source_id=gid,
                geometry_target_id=gid+'/'+cid,morphology_id=geometry if is_generated else None,
                generated_geometry=is_generated,report=report,plant_to_world_usd_row_vectors=matrix.tolist(),
                semantic_leaf_petiole_candidate=bool(semantic),anatomy_reason_codes=anatomy['reason_codes'],
                physical_geometry_verified=not is_generated or cid in qualified,
                physical_geometry_evidence=spec['physical_geometry_evidence'] if is_generated else None,
                geometry_evidence_source_bindings=physical['source_bindings'] if is_generated else {})
            entries.append(entry)
    expected=[dict(row,component_index=i) for i,row in enumerate(sorted(expected,key=lambda r:r['prim_path']),1)]
    keys=set(expected[0])
    require([{k:r[k] for k in keys} for r in catalogue]==expected,'Actual catalogue differs from complete generated anatomy')
    verify_bindings(bindings);verify_bindings(physical['source_bindings'])
    return entries


def _source_geometry(entry):
    if not entry['generated_geometry']:return original.source_geometry(entry)
    # No radius is used by the objective outer-sphere exclusion. Recompute the
    # same exact9mm centreline and rigid world transform without donor aliasing.
    cal=dict(camera_to_world_usd_row_vectors=np.eye(4).tolist(),intrinsics=np.eye(3).tolist(),
             resolution=[848,408],clipping_range_m=[.001,1000.])
    geo=geometry_9mm(entry['report'],entry['source_component_id'],entry['plant_to_world_usd_row_vectors'],cal)
    nominal=geo['nominal']
    return dict(nominal={k:nominal[k] for k in ('arc_m','world_m','point_plant_m','radius_m')},
        oriented_centerline_world_m=geo['oriented_centerline_world_m'],centerline_arc_distances_m=geo['centerline_arc_distances_m'],
        centerline_total_length_m=geo['centerline_total_length_m'],source_chain_reversed=geo['source_chain_reversed'],
        attachment_endpoint_error_m=geo['attachment_endpoint_error_m'])


def evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,workspace_checker,*,scene_coverage=None):
    """Every native-visible alternative is evaluated; missing evidence stays unknown."""
    index=CatalogueIndex(catalogue)
    result=labels.evaluate_frame(metadata,rgb,depth,valid,components,catalogue,[],workspace_checker,
                                 scene_coverage=scene_coverage)
    entries=_unique(inventory,'target_id');expected=set(result['target_census']['catalogue_petiole_ids'])
    require(set(entries)<=expected,'Foreign generated inventory target')
    bounds=original.robot_bounds(metadata,workspace_checker)
    visible=set(map(int,np.unique(components)));rows=[]
    for old in result['targets']:
        tid=old['target_id'];entry=entries.get(tid)
        if entry is None:rows.append(old);continue
        instance,cid=tid.split('/')
        try:
            target=_identity(entry,index.by_variant[instance])
            require(target['organ_type']=='sub_stem','Exact petiole required')
            if target['component_index'] not in visible:
                row=original.prior._zero_pixel_row(entry)
            else:
                proof=None
                if entry['semantic_leaf_petiole_candidate']:
                    geo=_source_geometry(entry);proof=original.outside_proof(tid,geo,bounds)
                if proof is not None:
                    row=dict(target_id=tid,source_family=entry['source_family'],status='excluded',reason=original.OUTSIDE_REASON,
                        annotation_epoch=ANNOTATION_EPOCH,nominal_arc_m=.009,visibility_support_arc_m=list(SUPPORT_ARC_M),
                        support_is_permissible_cut_interval=False,actual_visual_review=False,automated_pass=False,training_approved=False,
                        geometry=geo,cut_world_m=geo['nominal']['world_m'],outer_workspace_exclusion=proof)
                elif entry['generated_geometry']:
                    row=evaluate_generated_target(metadata,rgb,depth,valid,components,index.by_variant[instance],entry,workspace_checker)
                else:
                    row=labels.evaluate_target(metadata,rgb,depth,valid,components,index.by_variant[instance],entry,workspace_checker)
        except (ValueError,KeyError,AssertionError) as exc:
            row=dict(target_id=tid,status='unknown',reason='invalid_target_evidence: '+str(exc),
                     automated_pass=False,actual_visual_review=False,training_approved=False)
        row.update(annotation_epoch=ANNOTATION_EPOCH,source_family=entry['source_family'],
            source_component_id=cid,source_target_id=entry['source_target_id'],plant_instance_id=instance,
            geometry_source_id=entry['geometry_source_id'],geometry_target_id=entry['geometry_target_id'],
            morphology_id=entry['morphology_id'],generated_geometry=entry['generated_geometry'],
            physical_geometry_evidence=entry['physical_geometry_evidence'])
        rows.append(row)
    result.update(schema=SCHEMA,numerical_predicate_epoch=ANNOTATION_EPOCH,targets=rows,outer_workspace_bounds=bounds,
        donor_lineage_policy='original_donor_family_and_component_across_morphologies_and_clones',
        new_independent_source_family=False,training_approved=False,accepted_training_increment=0)
    census=result['target_census']
    census.update(catalogue_complete=set(entries)==expected,
        candidate_target_ids=[r['target_id'] for r in rows if r['status']=='candidate_pending_visual_review'],
        excluded_target_ids=[r['target_id'] for r in rows if r['status']=='excluded'],
        unknown_target_ids=[r['target_id'] for r in rows if r['status']=='unknown'])
    result['frame_blocked_by_unknown_targets']=bool(census['unknown_target_ids'])
    result['banned_candidate_instance_ids']=[r['target_id'] for r in rows
        if r.get('source_target_id')==BANNED_SOURCE_TARGET and r['status']=='candidate_pending_visual_review']
    return result


def assess_frame(metadata,annotation,workspace_checker):
    """Same continuous-junction and fixed-left workspace rules, no strict promotion."""
    require(annotation['schema']==SCHEMA and annotation['numerical_predicate_epoch']==ANNOTATION_EPOCH,
        'Typed generated annotation required')
    require(annotation['frame_id']==metadata['sample_id']
        and annotation['private_robot_context']==dict(robot_snapshot=metadata['robot_snapshot'],calibration=metadata['calibration']),
        'Exact native robot and camera required')
    rows=annotation['targets'];census=annotation['target_census'];ids=[t['target_id'] for t in rows]
    require(len(ids)==len(set(ids)) and sorted(ids)==sorted(census['catalogue_petiole_ids']),
        'Incomplete or duplicate target population')
    for status,key in [('candidate_pending_visual_review','candidate_target_ids'),('excluded','excluded_target_ids'),('unknown','unknown_target_ids')]:
        require(sorted(t['target_id'] for t in rows if t['status']==status)==sorted(census[key]),'Census status population differs')
    base=dict(schema=AMBIGUITY_SCHEMA,frame_id=metadata['sample_id'],training_approved=False,accepted_training_increment=0,
        strict_label_statuses_preserved=True,alternative_assessments=[],reachable_alternative_ids=[],unknown_alternative_ids=[],
        continuously_visible_alternative_ids=[],strict_candidate_ids=census['candidate_target_ids'],
        single_answer_unambiguous=False,decision='hold')
    if (not census['complete'] or annotation['frame_blocked_by_unverified_full_scene_coverage']
        or not census['catalogue_complete'] or census['unknown_target_ids'] or annotation['frame_blocked_by_unknown_targets']
        or annotation['banned_candidate_instance_ids'] or len(census['candidate_target_ids'])!=1):
        return dict(base,assessment_complete=False,reason='strict_or_complete_census_failed')
    candidate=next(t for t in rows if t['target_id']==census['candidate_target_ids'][0])
    answer_contract.target_answer(candidate,metadata['calibration'])
    bounds=original.robot_bounds(metadata,workspace_checker)
    require(bounds==annotation['outer_workspace_bounds'],'Workspace bounds changed')
    base['primary_target_id']=candidate['target_id']
    for target in rows:
        if target['target_id']==candidate['target_id']:continue
        visible=ambiguity.continuously_visible(target)
        item=dict(target_id=target['target_id'],source_family=target.get('source_family'),
            original_strict_status=target['status'],original_strict_reason=target.get('reason'),
            continuous_nominal_junction_visible=visible,strict_eligibility_promoted=False)
        if target.get('reason')==original.OUTSIDE_REASON:
            original.validate_exclusion(target,bounds)
            require(visible is None,'Objective exclusion must not fabricate visibility')
            item.update(assessment='outside_both_authenticated_arm_outer_bounds',workspace_recheck_required=False,
                outer_workspace_exclusion=target['outer_workspace_exclusion'],visibility_not_required_for_objective_reach_exclusion=True)
        elif visible is False:item.update(assessment='objective_anatomy_or_native_visibility_exclusion',workspace_recheck_required=False)
        elif visible is None:
            item.update(assessment='unresolved_alternative_evidence',workspace_recheck_required=True)
            base['unknown_alternative_ids'].append(target['target_id'])
        else:
            base['continuously_visible_alternative_ids'].append(target['target_id'])
            try:
                cut=answer_contract.point_at_9mm(target['geometry']['oriented_centerline_world_m'])
                require(np.allclose(cut,target['cut_world_m'],rtol=0,atol=1e-9),'Alternative is not exact9mm')
                meta=deepcopy(metadata);meta['supervision']=dict(target_id=target['target_id'],nominal_world_m=cut.tolist())
                proof=workspace_checker.check(meta)
                require(proof['target_id']==target['target_id'] and np.allclose(proof['nominal_world_m'],cut,rtol=0,atol=1e-9)
                    and proof['per_frame_camera_FK_verified'] is True,'Alternative workspace state differs')
                passed=proof['result']['workspace_passed'];require(type(passed) is bool,'Unknown workspace result')
                outside=not passed and proof['result']['status']=='outside_outer_reach_bound'
                item.update(workspace_recheck_required=True,workspace=proof,
                    assessment='reachable_competing_answer' if passed else ('outside_conservative_reach_bound' if outside else 'unresolved_ik_search'))
                if passed:base['reachable_alternative_ids'].append(target['target_id'])
                elif not outside:base['unknown_alternative_ids'].append(target['target_id'])
            except (ValueError,KeyError,TypeError,AssertionError) as exc:
                item.update(assessment='unresolved_alternative_workspace',error=str(exc),workspace_recheck_required=True)
                base['unknown_alternative_ids'].append(target['target_id'])
        base['alternative_assessments'].append(item)
    require(len(base['alternative_assessments'])==len(rows)-1,'Missing alternative')
    clear=not base['reachable_alternative_ids'] and not base['unknown_alternative_ids']
    return dict(base,assessment_complete=not base['unknown_alternative_ids'],single_answer_unambiguous=clear,
        decision='candidate_pending_actual_visual_review' if clear else 'hold',
        reason='no_reachable_or_unresolved_continuously_visible_alternative' if clear else 'reachable_or_unresolved_competing_answer')


def validate_planned_geometry(actual,planned):
    require(np.allclose(actual['oriented_centerline_world_m'],planned['oriented_centerline_world_m'],rtol=0,atol=1e-9)
        and np.allclose(actual['nominal']['world_m'],planned['nominal']['world_m'],rtol=0,atol=1e-9)
        and np.allclose(actual['nominal']['projected']['pixel_xy'],planned['nominal']['projected']['pixel_xy'],rtol=0,atol=1e-6),
        'Planned generated9mm geometry differs from current native geometry')


def validate_collection_plan_provenance(context,candidates):
    """Keep original camera/scene authority separate from generation authority."""
    scene_pin=context['source_collection_plan']
    generator_pin=context['generator_source_collection_plan']
    require(scene_pin==context['full_scene_census']['source_collection_plan']
        and generator_pin==candidates['source_plan'],'Scene/generator collection plan roles differ')
    scene_path=_pin(scene_pin);generator_path=_pin(generator_pin)
    scene_plan=read_json(scene_path);generator_plan=read_json(generator_path)
    require(scene_plan['family_assignments']==generator_plan['family_assignments']
        and scene_plan['source_bindings_sha256']==generator_plan['source_bindings_sha256'],
        'Scene/generator source families or original assets differ')
    family=candidates['source_family']
    require(scene_plan['family_assignments'][family]==context['dataset_split']=='train',
        'Actual source donor split differs')
    pins={str(scene_path):scene_pin['sha256'],str(generator_path):generator_pin['sha256']}
    require(1<=len(candidates['records'])<=3,'Bounded original camera anchors required')
    for record in candidates['records']:
        anchor_path=_pin(record['anchor']);anchor=read_json(anchor_path)
        require(Path(anchor['source_collection_plan']).resolve()==scene_path
            and anchor['source_bindings'].get(str(scene_path))==scene_pin['sha256'],
            'Camera anchor must authenticate the actual scene collection plan')
        require(anchor['source_family']==record['source_family']==family
            and anchor['split']==context['dataset_split'],'Camera donor or split differs')
        require(candidates['source_bindings'].get(str(anchor_path))==record['anchor']['sha256'],
            'Camera anchor is not bound by the candidate preparation')
        pins[str(anchor_path)]=record['anchor']['sha256']
    return pins


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
    from . import native848_controlled_capture_v2 as producer
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
    require(context['profile']==candidates['profile'],'Source optical profile changed')
    for path,h in validate_collection_plan_provenance(context,candidates).items():
        require(path not in pins or pins[path]==h,'Conflicting camera/generator source authority');pins[path]=h
    verify_bindings(context['source_bindings']);pins.update(context['source_bindings'])
    census=read(local(capture/'census.json'))
    require(census==context['full_scene_census'],'Saved census differs from context')
    catalogue=read(context['catalogue'],within=capture);index=CatalogueIndex(catalogue)
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
    for module in (sys.modules[__name__],labels,original,joint,ambiguity,answer_contract):
        path=str(Path(module.__file__).resolve());pins[path]=sha256(path)
    profile=read(candidates['profile']);check_profile(profile)
    require(context['renderer_settings']==profile['render_settings'],'Captured renderer differs from source optics profile')
    records=_unique(candidates['records'],'sample_id')
    frameids=[r['observation_id'] for r in result['frames']]
    require(len(frameids)==len(set(frameids)) and set(frameids)<=set(records),'Foreign/repeated frame IDs')
    expected_requests=(len(profile['warmup_steps']) if frameids else 0)+len(frameids)
    require(result['request_count']==len(result['requests'])==expected_requests,'Native warmup/production accounting differs')
    for n,e in enumerate(result['requests'],1):
        require(e['request_index']==n and e['native_requests']==e['callback_count']==1
            and e['callback_sequence_after']==n and e['callback_sequence_before']==n-1
            and e['requested_subframes']==profile['request_subframes']
            and e['delta_time_seconds']==profile['delta_time_seconds'] and e['wait_for_render'] is True
            and abs(e['timeline_after_seconds']-e['timeline_before_seconds']-profile['delta_time_seconds'])<1e-7,
            'Actual same-callback/reset optical profile differs')
    output.mkdir();checker=BulkWorkspace();summaries=[];previous=None
    try:
        for receipt in result['frames']:
            observation=read(receipt,within=capture);name=observation['observation_id'];record=records[name]
            require(observation['schema']=='greenhouse.controlled_native_observation.v1'
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
            mapping={int(k):v for k,v in read(observation['mapping'],within=capture)['renderer_id_to_prim'].items()}
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
            previous=token
            matrix=next(r['plant_to_world_usd_row_vectors'] for r in census['all_plant_roots']
                        if r['plant_root']==census['foreground_root'])
            expected=geometry_9mm(generated_report,candidates['target_component_id'],matrix,observation['calibration'])
            require(observation['geometry']==expected,'Actual saved generated9mm geometry changed')
            validate_planned_geometry(expected,record['pilot_9mm_geometry'])
            metadata=dict(schema='greenhouse.generated_native9mm_sample.v1',sample_id=name,
                calibration=observation['calibration'],robot_snapshot=pose,rendered_camera_params=observation['rendered_camera_params'],
                synchronization=sync,geometry_screen=pose['visual_bound_screen'],source_family=candidates['source_family'],
                geometry_source_id=generated_report['plant_id'],split='train',training_approved=False,accepted_training_increment=0)
            verify_camera(metadata)
            components,_,owners=component_masks(ids,mapping,index)
            unknown=[];pixel_classes=Counter()
            for rid in np.unique(ids):
                rid=int(rid);path=mapping.get(rid,'');mask=ids==rid;count=int(mask.sum());owner=owners[rid]
                if owner:category='authenticated_component_'+owner['organ_type']
                elif rid==0 and not path and np.all(~valid[mask]) and np.all(np.isposinf(depth[mask])):category='empty_background'
                elif path.startswith(('/World/Environment/GreenHouse/','/World/Gutters/','/World/RBY1/','/World/Floor','/World/Ground')):category='nonplant_greenhouse_or_robot'
                else:category='unknown_identity';unknown.append(dict(renderer_id=rid,prim_path=path,pixels=count))
                pixel_classes[category]+=count
            annotation=evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
            for target in annotation['targets']:
                proof=target.get('attachment_joint_ownership')
                if proof is not None:require(all(pins.get(p)==h for p,h in proof['source_bindings'].items()),'Unbound current geometry joint proof')
            folder=output/name;folder.mkdir()
            coverage=dict(schema='greenhouse.native848_generated_coverage_audit.v1',frame_id=name,
                observation=dict(path=str(Path(receipt['path']).resolve()),sha256=receipt['sha256']),
                context=observation['context'],capture_split='train',plant_instance_count=144,
                original_background_instances=143,generated_foreground_instances=1,
                complete_all_eligible_ground_truth=not unknown and not annotation['target_census']['unknown_target_ids'],
                blocking_unknowns=unknown+annotation['target_census']['unknown_target_ids'],visible_cross_split_petiole_targets=[],
                pixel_categories=dict(pixel_classes),catalogue_petiole_count=len(inventory),source_bindings=dict(pins),
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
        value=dict(schema='greenhouse.native848_generated_capture_evaluation.v1',source_capture=str(capture),
            owner_complete=local(trial/'owner_complete.json'),capture_result=complete['result'],frames_evaluated=len(summaries),
            records=summaries,workspace_stats=stats,source_bindings=pins,native_control_performed=False,
            actual_visual_review=False,training_approved=False,accepted_training_increment=0)
        save_json(output/'result.json',value);return value
    finally:checker.finish()
