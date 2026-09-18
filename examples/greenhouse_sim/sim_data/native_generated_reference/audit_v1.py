"""Fresh raw-buffer replay for the distinct original-reference producer v1.

CPU or already-running Kit only: full bank/catalogue checks can import USD.
Pins authenticate declared provenance, not historical OS process execution.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json

import numpy as np

from . import execution_v1 as ex
from ..native_original_capture import contracts as oc


def world_from_row(plan, row):
    from ..capture_contract import transform_points
    matrix = plan['expected_original_world']['plant_to_world_usd_row_vectors']
    proposal = row['cut_region_proposal']
    samples = proposal['accepted_centerline_interval']['samples']; points = []
    for a,b in zip(samples,samples[1:]):
        steps = max(1,int(np.ceil((b['arc_distance_m']-a['arc_distance_m'])/.001)))
        for t in np.linspace(0,1,steps,endpoint=False):
            points.append(np.asarray(a['point_plant_m'])*(1-t)+np.asarray(b['point_plant_m'])*t)
    points.append(samples[-1]['point_plant_m'])
    return dict(plant_to_world_usd_row_vectors=deepcopy(matrix),
        nominal_world_m=transform_points([proposal['nominal']['point_plant_m']],matrix)[0].tolist(),
        interval_world_m=transform_points(points,matrix).tolist())


def expected_catalogue(plan, mode, reports, generated):
    items = []
    for variant in plan['scene_authority']['scene_variants']:
        oc.require(not variant['added_components'] and not variant['added_component_paths'], 'Unexpected added background anatomy')
        family, root = variant['variant_id'], variant['plant_root']
        source = variant['source_plant_id']
        report = reports[family]
        if mode == 'generated_variant' and family == plan['source_family']:
            family = generated['variant_id']; root = '/World/GeneratedNativePilot/'+family
            report = generated['report']
        components = report['components']
        for key, component in components.items():
            chain, parent = [key], component['parent']
            while parent is not None:
                oc.require(parent in components and parent not in chain, 'Invalid component ancestry')
                chain.append(parent); parent = components[parent]['parent']
            items.append(dict(component_id=key,prim_path=root+'/'+'/'.join(reversed(chain)),
                organ_type=component['type'],variant_id=family,source_plant_id=source,split_group=variant['split_group']))
    return [dict(item,component_index=i) for i,item in enumerate(sorted(items,key=lambda x:x['prim_path']),1)]


def pair_evidence(samples):
    from ..native_greenhouse_pair import assert_same_camera
    if len(samples) != 2:
        return dict(complete_pair=False, native_change_observed=False, reason='pre_render_hold', **ex.FLAGS)
    a,b = samples
    assert_same_camera(a['calibration'],b['calibration'])
    oc.require(all(a[k] == b[k] for k in ('robot_snapshot','scene_counts','lighting','scene_evidence')),
               'Matched pair changed current scene/pose')
    x,y = a['synchronization'],b['synchronization']
    oc.require(y['freshness']['callback_sequence'] > x['freshness']['callback_sequence']
        and x['static_guard'] != y['static_guard'], 'Substitution/static callback evidence missing')
    changed = all(x['freshness'][k] != y['freshness'][k] for k in ('rgb_sha256','depth_sha256'))
    observed = changed and a['native_target_pixels'] > 0 and b['native_target_pixels'] > 0 and b['old_plant_native_pixels'] == 0
    return dict(complete_pair=True, native_change_observed=bool(observed), buffers_changed=changed,
        old_foreground_native_pixels=b['old_plant_native_pixels'],
        reason='observed_not_morphology_or_training_approval' if observed else 'native_change_not_qualified', **ex.FLAGS)


def verify_actual_camera(metadata, expected_calibration):
    """Same tolerance as worker; preserve actual values and independent native FK."""
    from ..native_greenhouse_pair import assert_same_camera
    from ..native_original_capture.audit import verify_camera
    assert_same_camera(metadata['calibration'],expected_calibration)
    verify_camera(metadata)


def review_sample(capture, plan, mode, record, reports, generated, bindings):
    from ..native_dataset import inventory as inv
    from ..native_dataset.bundle import SampleReader
    from ..capture_contract import fingerprint, project, depth_evidence
    from ..capture_visibility import interval_visibility, view_quality, component_masks, ORGAN_IDS
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    from .scene import current_scene_evidence

    folder = oc.safe_file(capture,mode)
    oc.require(not (folder/'bundle.json').exists(), 'This executor has no compact storage proof')
    reader = SampleReader(folder,expected_bindings={'sample.json':record['sample_sha256'],
        'supervision/label.json':record['label_sha256']},expected_json=(
        {'supervision/query_trace.json':record['query_trace']} if record['query_trace'] is not None else {}))
    preview = reader.metadata
    audited = dict(sample_sha256=record['sample_sha256'],label_sha256=record['label_sha256'],
                   rgb_sha256=preview['files']['inputs/rgb.png']['sha256'])
    meta,label,trace,logical,decoded = inv._read_sample(bindings,folder,record,audited)
    oc.require(trace is not None or not (folder/'supervision/query_trace.json').exists(), 'Unexpected excluded trace')
    row = plan['source_row' if mode == 'original_control' else 'generated_row']
    oc.require(meta['schema_version'] == ex.SAMPLE and meta['state'] == 'fresh_native_reference_pair_pending_replay'
        and meta['sample_id'] == mode and meta['training_sample_approved'] is False
        and meta['historical_labels_inherited'] is False and meta['native_instance_backend'] == 'legacy'
        and meta['robot_snapshot'] == plan['expected_robot_snapshot']
        and meta['scene_counts'] == plan['scene_authority']['expected_scene_counts']
        and meta['synchronization']['render_budget_subframes'] == 56
        and meta['input_policy'] == plan['input_policy'], 'Sample changed bounded current scene/camera policy')
    expected_scene = current_scene_evidence(plan,lighting=meta['lighting'],counts=meta['scene_counts'],renderer=meta['renderer'])
    oc.require(meta['scene_evidence'] == expected_scene and meta['geometry_screen'] == record['screen']
        and meta['geometry_screen']['passed'] is True, 'Actual scene/geometry record mismatch')
    sup = meta['supervision']; world = world_from_row(plan,row)
    oc.require(sup['target_id'] == row['target_id'] == record['target_id']
        and sup['source_target_id'] == sup['conservative_view_cap_group'] == plan['conservative_view_cap_group']
        and sup['split_group'] == plan['source_family'] and sup['cut_region_proposal'] == row['cut_region_proposal']
        and sup['cut_safety_validated'] is False, 'Different generated/original target ancestry')
    for key,value in world.items():
        oc.require(np.allclose(sup[key],value,atol=1e-9,rtol=0), 'Metric anatomy/placement changed: '+key)
    verify_actual_camera(meta,plan['expected_calibration'])
    ids = reader.array('supervision/renderer_instance_id.npy')
    identities = reader.json('supervision/identities.json')
    catalogue = identities['component_catalogue']
    oc.require(catalogue == expected_catalogue(plan,mode,reports,generated)
        and len(catalogue) == meta['scene_counts']['components'] and identities['organ_ids'] == ORGAN_IDS,
        'Full original/generated/background hierarchy differs')
    mapping = {int(k):v for k,v in identities['renderer_id_to_prim'].items()}
    oc.require(meta['native_instance_sha256'] == oc.digest(ids.tobytes())
        and meta['native_mapping_sha256'] == fingerprint(mapping), 'Same-callback identity fingerprints differ')
    components,organs,owners = component_masks(ids,mapping,catalogue)
    rgb,depth,valid = reader.image('inputs/rgb.png'),reader.array('inputs/depth_m.npy'),reader.image('inputs/depth_valid.png') != 0
    target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
    nominal,interval = project([world['nominal_world_m']],meta['calibration'])[0],project(world['interval_world_m'],meta['calibration'])
    radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
    visibility,mask = interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,radius)
    quality = view_quality(meta['calibration'],nominal,interval,radius,visibility,rgb,mask)
    old_root = plan['scene_authority']['original_variant']['plant_root']
    old_ids = [i for i,path in mapping.items() if path == old_root or path.startswith(old_root+'/')]
    oc.require(sup['nominal_projected'] == nominal and sup['projected_interval'] == interval
        and sup['depth_evidence'] == depth_evidence(nominal,depth,valid,radius)
        and sup['visibility_evidence'] == visibility and meta['quality'] == quality
        and meta['native_target_pixels'] == int(mask.sum())
        and meta['old_plant_native_pixels'] == int(np.isin(ids,old_ids).sum()), 'Stored native visibility/quality evidence differs')
    report = reports[plan['source_family']] if mode == 'original_control' else generated['report']
    expected_label = derive(meta,report,rgb,depth,valid,components,catalogue)
    expected_trace = trace_review(meta,report,expected_label,rgb,depth,valid,components,catalogue) if expected_label['eligible'] else None
    oc.require(label == expected_label and trace == expected_trace, 'Saved label/trace differs from fresh replay')
    inv._trace_consistency(label,trace)
    decision = inv.EXCLUDE if not label['eligible'] else inv.STRICT if trace['passed'] else inv.HOLD
    oc.require(record['decision'] == decision, 'Stored annotation decision differs')
    return dict(candidate_id=mode,target_id=row['target_id'],decision=decision,reason=label['reason'],
        clarity_reasons=label.get('clarity',{}).get('reasons'),trace_reasons=trace['reasons'] if trace else None,
        sample_sha256=record['sample_sha256'],label_sha256=record['label_sha256'],
        rgb_sha256=logical['inputs/rgb.png']['sha256'],decoded_rgb_sha256=decoded,
        logical_files=logical,label_replayed_exact=True,trace_replayed_exact=trace is not None,
        native_callback_hashes_verified=True,native_ID_masks_replayed=True, **ex.FLAGS), meta


def audit_capture(capture, *, result_sha256):
    from .prepare import check_plan
    from ..collection_plan import load_plan
    from ..native_dataset import inventory as inv
    from ..capture_contract import project
    capture = Path(capture).resolve()
    oc.require(not (capture/'failure.json').exists(), 'Failed worker cannot become a completed capture')
    result = oc.read_json(oc.pin(capture/'result.json',result_sha256))
    request_path = oc.pin(capture/'request.json',result['request_sha256'])
    request = oc.read_json(request_path)
    execution,plan = ex.preflight_request(request['execution_request_path'],request['execution_request_sha256'])
    owner=oc.read_json(oc.pin(capture.parent/'owner_started.json',request['owner_started_sha256']))
    owned=oc.read_json(oc.pin(capture.parent/'owned_worker.json',request['owned_worker_sha256']))
    oc.require(owner['execution_request_path']==request['execution_request_path']
        and owner['execution_request_sha256']==request['execution_request_sha256']
        and owner['owner_pid']==owned['owner_pid']
        and owned['command']==ex.worker_command(request['execution_request_path'],request['execution_request_sha256'],execution),
        'Worker request differs from owned launch declaration')
    oc.require(capture == Path(execution['output'])/'capture'
        and result['schema'] == ex.RESULT and result['state'] == 'bounded_pair_capture_complete_pending_owned_exit'
        and request['schema'] == ex.RESULT and result['plan_sha256'] == request['plan_sha256'] == execution['plan_sha256']
        and result['implementation_bindings'] == execution['implementation_bindings']
        and request['process_admission']['no_unrelated_kit_process_at_admission'] is True
        and request['host_memory_preflight']['checked'] is True and request['host_memory_preflight']['allowed'] is True
        and result['source_assets_unchanged'] is True and result['stage_count'] == result['render_product_count'] == 1
        and all(result[k] is v for k,v in ex.FLAGS.items()), 'Wrong producer/result/admission scope')
    generated = check_plan(plan)
    _,all_reports = load_plan(plan['scene_authority']['clear_plan']['path'])
    reports = {r['plant_id']:r for r in all_reports}
    records = result['records']; oc.require([r['candidate_id'] for r in records] == ex.MODES, 'Incomplete/reordered pair')
    bindings = inv._Bindings(); bindings.mapping(execution['source_bindings'])
    bindings.mapping({str(capture/'result.json'):result_sha256,str(request_path):result['request_sha256'],
        request['execution_request_path']:request['execution_request_sha256'],
        str(capture.parent/'owner_started.json'):request['owner_started_sha256'],
        str(capture.parent/'owned_worker.json'):request['owned_worker_sha256']})
    reviewed,samples = [],[]
    for mode,record in zip(ex.MODES,records):
        row = plan['source_row' if mode == 'original_control' else 'generated_row']
        oc.require(record['target_id'] == row['target_id'], 'Different capture target')
        if record['state'] == 'held_pre_render':
            world = world_from_row(plan,row)
            projected = project([world['nominal_world_m'],*world['interval_world_m']],plan['expected_calibration'])
            reasons = (['static_geometry_screen_failed'] if not record['screen']['passed'] else [])
            if any(p['projection_status'] != 'in_frame' for p in projected): reasons.append('interval_clipped_or_out_of_frame')
            oc.require(reasons and record['reasons'] == reasons and not (capture/mode).exists(), 'Unexplained pre-render hold')
            reviewed.append(dict(candidate_id=mode,target_id=row['target_id'],decision='held_pre_render',
                reasons=reasons,geometry_screen_evidence='producer_declaration_not_offline_recomputed', **ex.FLAGS))
        else:
            oc.require(record['state'] == 'native_captured', 'Unknown capture record')
            item,meta = review_sample(capture,plan,mode,record,reports,generated,bindings)
            reviewed.append(item); samples.append(meta)
    pair = pair_evidence(samples)
    oc.require(pair == result['pair_evidence'], 'Pair evidence changed')
    bindings.finish(); ex.implementation_bindings(); oc.pin(capture/'result.json',result_sha256)
    return dict(schema=ex.AUDIT,capture=str(capture),result_sha256=result_sha256,
        request_sha256=result['request_sha256'],plan_sha256=execution['plan_sha256'],
        execution_request_path=request['execution_request_path'],execution_request_sha256=request['execution_request_sha256'],
        records=reviewed,counts=dict(Counter(r['decision'] for r in reviewed)),pair_evidence=pair,
        source_bindings=dict(sorted(bindings.hashes.items())),implementation_bindings=ex.implementation_bindings(),
        review_method='independent_full_native_ID_Z_RGB_anatomy_camera_label_trace_replay', **ex.FLAGS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',required=True);p.add_argument('--result-sha256',required=True)
    args=p.parse_args(argv)
    print(json.dumps(audit_capture(args.capture,result_sha256=args.result_sha256),indent=2))


if __name__ == '__main__':
    main()
