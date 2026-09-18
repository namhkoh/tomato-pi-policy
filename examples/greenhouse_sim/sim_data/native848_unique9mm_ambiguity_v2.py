"""Independent one-answer ambiguity guard; never promotes a failed strict label.

Intact alternatives with native-visible nominal/attachment-to-cut tissue
need their own 9 mm workspace check, even when strict detail or luma failed.
"""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from . import native848_unique_petiole_dataset_v1 as dataset
from .dataset_review import verify_bindings
from .native848_bulk_workspace_v1 import BulkWorkspace
from . import native848_fully_labeled_coverage_v2 as outer

SCHEMA='greenhouse.native848_unique9mm_ambiguity_assessment.v2'
POLICY='all_intact_nominal_and_junction_visible_9mm_alternatives_need_workspace_with_authenticated_both_arm_outer_exclusions.v2'
DATASET_SHA256='35575a0edeea122ee93b1650da43cf2e258d6c71fa01c3960c04f6d3db6755f2'
OBJECTIVE_EARLY_EXCLUSIONS=frozenset(('not_visible_zero_authenticated_component_pixels',
    'not_anatomically_eligible_leaf_petiole','protected_descendant_or_no_leaf_petiole',
    'not_intact_petiole','not_direct_main_stem_petiole'))
require=dataset.require


def continuously_visible(target):
    if target.get('status')=='unknown':return None
    if target.get('reason') in OBJECTIVE_EARLY_EXCLUSIONS:return False
    try:
        v=target['visibility'];nominal=v['nominal']
        require(type(nominal['visible']) is bool,'Missing nominal visibility decision')
        if nominal['projected']['projection_status']=='in_frame' and nominal['depth']['status'].startswith('unknown'):return None
        if nominal['visible'] is False:return False
        j=target['junction_continuity'];probes=j['probes']
        arcs=np.asarray([p['arc_m'] for p in probes],float)
        require(len(arcs)>=19 and np.isfinite(arcs).all() and abs(arcs[0])<1e-12
            and abs(arcs[-1]-.009)<1e-12 and (np.diff(arcs)>0).all()
            and (np.diff(arcs)<=.0005+1e-12).all(),'Incomplete junction evidence')
        required=[nominal,*probes]
        require(all(type(p['visible']) is bool for p in required),'Missing visibility decision')
        # Unknown depth cannot be converted to an objective absent alternative.
        if any(p['projected']['projection_status']=='in_frame' and p['depth']['status'].startswith('unknown') for p in required):return None
        return all(p['visible'] is True for p in required)
    except (KeyError,ValueError,TypeError):return None


def assess_frame(metadata,annotation,workspace_checker,*,metadata_pin,annotation_pin):
    require(dataset.digest(dataset.__file__)==DATASET_SHA256,'Frozen strict label validator changed')
    require(json.loads(dataset.read_pin(metadata_pin).read_text())==metadata
        and json.loads(dataset.read_pin(annotation_pin).read_text())==annotation,'Exact saved source objects required')
    require(metadata['sample_id']==annotation['frame_id']
        and annotation['private_robot_context']==dict(robot_snapshot=metadata['robot_snapshot'],calibration=metadata['calibration']),
        'Alternative evaluation used different camera/robot state')
    source_bindings={str(dataset.read_pin(metadata_pin)):metadata_pin['sha256'],
        str(dataset.read_pin(annotation_pin)):annotation_pin['sha256'],str(Path(__file__).resolve()):dataset.digest(__file__),
        str(Path(dataset.__file__).resolve()):DATASET_SHA256,**workspace_checker.bindings}
    bounds=outer.verify_annotation_outer_exclusions(annotation,metadata,workspace_checker)
    source_bindings.update(bounds['source_bindings'])
    base=dict(schema=SCHEMA,policy=POLICY,frame_id=annotation['frame_id'],metadata=metadata_pin,annotation=annotation_pin,
        source_bindings=source_bindings,strict_label_statuses_preserved=True,detail_failure_not_used_to_exclude_alternative=True,beyond_cut_support_failure_not_used_to_exclude_alternative=True,
        all_catalogue_target_ids=[t['target_id'] for t in annotation['targets']],
        strict_candidate_ids=annotation['target_census']['candidate_target_ids'],alternative_assessments=[],
        reachable_alternative_ids=[],unknown_alternative_ids=[],continuously_visible_alternative_ids=[],
        training_approved=False,accepted_training_increment=0)
    try:primary=dataset.validate_census(annotation)[0];dataset.target_answer(primary,metadata['calibration'])
    except (ValueError,KeyError,TypeError) as exc:
        return dict(base,assessment_complete=False,single_answer_unambiguous=False,decision='hold',reason='strict_or_complete_census_failed: '+str(exc))
    base['primary_target_id']=primary['target_id']
    for target in annotation['targets']:
        if target['target_id']==primary['target_id']:continue
        visible=continuously_visible(target)
        item=dict(target_id=target['target_id'],source_family=target.get('source_family'),
            original_strict_status=target['status'],original_strict_reason=target.get('reason'),
            continuous_nominal_junction_visible=visible,strict_eligibility_promoted=False)
        if target.get('reason')==outer.OUTSIDE_REASON:
            outer.validate_exclusion(target,bounds)
            require(visible is None,'Objective reach exclusion must not fabricate visibility')
            item.update(assessment='outside_both_authenticated_arm_outer_bounds',workspace_recheck_required=False,
                visibility_not_required_for_objective_reach_exclusion=True,outer_workspace_exclusion=target['outer_workspace_exclusion'])
        elif visible is False:
            item.update(assessment='objective_anatomy_or_native_visibility_exclusion',workspace_recheck_required=False)
        elif visible is None:
            item.update(assessment='unresolved_alternative_evidence',workspace_recheck_required=True)
            base['unknown_alternative_ids'].append(target['target_id'])
        else:
            base['continuously_visible_alternative_ids'].append(target['target_id'])
            try:
                cut=dataset.point_at_9mm(target['geometry']['oriented_centerline_world_m'])
                require(np.allclose(cut,target['cut_world_m'],rtol=0,atol=1e-9),'Alternative point is not9mm')
                meta=deepcopy(metadata);meta['supervision']=dict(target_id=target['target_id'],nominal_world_m=cut.tolist())
                proof=workspace_checker.check(meta)
                require(proof['target_id']==target['target_id'] and np.allclose(proof['nominal_world_m'],cut,rtol=0,atol=1e-9)
                    and proof['per_frame_camera_FK_verified'] is True,'Alternative workspace point/state differs')
                passed=proof['result']['workspace_passed'];require(type(passed) is bool,'Unknown workspace result')
                objective_outside=(not passed and proof['result']['status']=='outside_outer_reach_bound')
                item.update(workspace_recheck_required=True,workspace=proof,
                    assessment='reachable_competing_answer' if passed else ('outside_conservative_reach_bound' if objective_outside else 'unresolved_ik_search'))
                if passed:base['reachable_alternative_ids'].append(target['target_id'])
                elif not objective_outside:base['unknown_alternative_ids'].append(target['target_id'])
            except (ValueError,KeyError,TypeError,AssertionError) as exc:
                item.update(assessment='unresolved_alternative_workspace',error=str(exc),workspace_recheck_required=True)
                base['unknown_alternative_ids'].append(target['target_id'])
        base['alternative_assessments'].append(item)
    require(len(base['alternative_assessments'])==len(annotation['targets'])-1,'Incomplete alternative population')
    clear=not base['reachable_alternative_ids'] and not base['unknown_alternative_ids']
    verify_bindings(source_bindings)
    return dict(base,assessment_complete=not base['unknown_alternative_ids'],single_answer_unambiguous=clear,
        decision='candidate_pending_actual_visual_review' if clear else 'hold',
        reason='no_reachable_or_unresolved_continuously_visible_alternative' if clear else 'reachable_or_unresolved_competing_answer')


def run(metadata_spec,annotation_spec,output):
    output=dataset.roots.diagnostic(output);require(not output.exists(),'Create-only ambiguity diagnostic required')
    metadata=json.loads(dataset.read_pin(metadata_spec).read_text());annotation=json.loads(dataset.read_pin(annotation_spec).read_text())
    checker=BulkWorkspace()
    value=assess_frame(metadata,annotation,checker,metadata_pin=metadata_spec,annotation_pin=annotation_spec)
    value['workspace_stats']=checker.finish();output.mkdir(parents=True);dataset.write(output/'result.json',value)
    return dataset.pin(output/'result.json')


def validate_assessment(spec,metadata_spec,annotation_spec,annotation):
    result=json.loads(dataset.read_pin(spec).read_text())
    require(result['schema']==SCHEMA and result['policy']==POLICY and result['metadata']==metadata_spec
        and result['annotation']==annotation_spec and result['frame_id']==annotation['frame_id']
        and result['source_bindings'].get(str(Path(__file__).resolve()))==dataset.digest(__file__),
        'Exact independent ambiguity assessment required')
    require(result['assessment_complete'] is True and result['single_answer_unambiguous'] is True
        and result['decision']=='candidate_pending_actual_visual_review'
        and not result['reachable_alternative_ids'] and not result['unknown_alternative_ids']
        and result['strict_label_statuses_preserved'] is True,'Competing or unresolved answer holds entire frame')
    metadata=json.loads(dataset.read_pin(metadata_spec).read_text())
    bounds=outer.verify_annotation_outer_exclusions(annotation,metadata)
    ids=[t['target_id'] for t in annotation['targets']];primary=annotation['target_census']['candidate_target_ids']
    require(len(primary)==1 and result['primary_target_id']==primary[0]
        and result['all_catalogue_target_ids']==ids and result['strict_candidate_ids']==primary
        and sorted(x['target_id'] for x in result['alternative_assessments'])==sorted(set(ids)-set(primary)),
        'Ambiguity assessment omitted an alternative')
    by={x['target_id']:x for x in result['alternative_assessments']}
    require(len(by)==len(result['alternative_assessments']),'Duplicate ambiguity alternative')
    for target in annotation['targets']:
        if target['target_id']==primary[0]:continue
        item=by[target['target_id']];visible=continuously_visible(target)
        require(item['continuous_nominal_junction_visible'] is visible
            and item['original_strict_status']==target['status']
            and item['original_strict_reason']==target.get('reason') and item['strict_eligibility_promoted'] is False,
            'Alternative evidence or strict status changed')
        if target.get('reason')==outer.OUTSIDE_REASON:
            outer.validate_exclusion(target,bounds)
            require(visible is None and item['assessment']=='outside_both_authenticated_arm_outer_bounds'
                and item['visibility_not_required_for_objective_reach_exclusion'] is True
                and item['workspace_recheck_required'] is False and item['outer_workspace_exclusion']==target['outer_workspace_exclusion'],
                'Unproved outside-both-arm exclusion cannot pass')
            continue
        require(visible is not None,'Unresolved alternative cannot pass')
        if visible:
            proof=item['workspace'];cut=dataset.point_at_9mm(target['geometry']['oriented_centerline_world_m'])
            require(item['assessment']=='outside_conservative_reach_bound'
                and proof['result']['workspace_passed'] is False and proof['result']['status']=='outside_outer_reach_bound' and proof['target_id']==target['target_id']
                and np.allclose(proof['nominal_world_m'],cut,rtol=0,atol=1e-9)
                and proof['per_frame_camera_FK_verified'] is True,'Reachable or mismatched alternative proof')
        else:require(item['assessment']=='objective_anatomy_or_native_visibility_exclusion','Alternative exclusion differs')
    verify_bindings(result['source_bindings']);return result
