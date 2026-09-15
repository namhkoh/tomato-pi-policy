"""Replay static native annotations; never grants a training-release approval."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..dataset_review import read_json, require, verify_bindings
from ..depth_preview import sha256
from ..plant_variant_catalogue import load_for_inspection
from ..native_clear_labels import derive
from ..automated_native_review import trace_review, check_native_evidence
from .bundle import SampleReader, digest
from . import bundle

LOADED_REVIEW_CODE = {str(p.resolve()):sha256(p) for p in
                      (Path(__file__),Path(bundle.__file__))}

def audit_capture(capture, plan_path, *, storage_root=None):
    verify_bindings(LOADED_REVIEW_CODE)
    capture, plan_path = Path(capture).resolve(), Path(plan_path).resolve()
    storage_root = Path(storage_root).resolve() if storage_root is not None else capture
    require(not (capture/'failure.json').exists(), 'Failed native capture cannot be audited')
    result = read_json(capture/'result.json'); plan = read_json(plan_path)
    request = read_json(capture/'request.json')
    require(request['plan_sha256']==sha256(plan_path) and
            Path(request['plan_path']).resolve()==plan_path, 'Capture request/plan mismatch')
    require(plan.get('split')=='train' and plan.get('resolution')==[1696,816] and
            all(plan.get(k) is False for k in ('training_approved','source_cap_reset',
                'physical_motion_commanded','hidden_cut_coordinates_executable')), 'Unexpected plan contract')
    require(request.get('training_started') is False and result.get('training_approved') is False,
            'Unexpected capture training claim')
    require(result.get('state')=='native_generated_multiview_pilot_complete_pending_review',
            'Incomplete native result state')
    planned={}
    planned_cases={}
    for case in plan['target_cases']:
        for spec in case['views']:
            require(spec['candidate_id'] not in planned, 'Duplicate planned candidate')
            planned[spec['candidate_id']]=(case['target_id'],spec)
            planned_cases[spec['candidate_id']]=case
    seen=set()
    for row in result['records']:
        name=row['candidate_id']
        require(row['state'] in ('rejected_pose','rejected_possible_geometry_overlap',
                'native_captured_pending_review'), 'Unknown capture decision state')
        require(name in planned and name not in seen, 'Unplanned or duplicate capture record')
        seen.add(name)
        require((row['target_id'],row['requested_spec'])==planned[name], 'Record target/view differs from plan')
    require(seen==planned.keys(), 'Incomplete planned capture decisions')
    require(result['source_assets_unchanged'] is True, 'Changed source assets')
    verify_bindings(plan['source_bindings'])
    verify_bindings(plan['implementation_bindings'])
    verify_bindings(plan['prerequisite_bindings'])
    anchor = read_json(plan['anchor_pair_plan'])
    generated = load_for_inspection(anchor['variant_directory'], anchor['source_collection_plan'])
    records = []
    for row in result['records']:
        if row['state'] != 'native_captured_pending_review': continue
        name = row['candidate_id']
        require(isinstance(name,str) and name not in ('','.', '..') and '/' not in name and '\\' not in name,
                'Invalid capture identifier')
        folder = (storage_root/name).resolve()
        require(folder.parent == storage_root, 'Capture path escapes root')
        reader = SampleReader(folder,expected_bindings={'sample.json':row['sample_sha256'],
            'supervision/label.json':row['label_sha256']},expected_json=(
                {'supervision/query_trace.json':row['query_trace']} if row.get('query_trace') is not None else {}))
        meta = reader.metadata
        require(meta['sample_id']==row['candidate_id'] and
                meta['supervision']['target_id']==row['target_id'] and
                meta['robot_snapshot']==row['robot_snapshot'] and meta['geometry_screen']==row['screen'],
                'Sample target/robot/screen differs from planned capture record')
        cap_group=planned_cases[row['candidate_id']]['conservative_view_cap_group']
        require(meta['supervision']['conservative_view_cap_group']==cap_group,
                'Sample source-target ancestry differs from plan')
        require(digest(reader.read('sample.json')) == row['sample_sha256'], 'Changed sample metadata')
        require(digest(reader.read('supervision/label.json')) == row['label_sha256'], 'Changed label')
        reader.verify_all()
        rgb = reader.image('inputs/rgb.png')
        depth = reader.array('inputs/depth_m.npy')
        valid = reader.image('inputs/depth_valid.png')!=0
        components = reader.array('supervision/component_id.npy')
        target = reader.image('supervision/target_visible.png')
        catalogue = reader.json('supervision/identities.json')['component_catalogue']
        check_native_evidence(meta,rgb,depth,components,target,catalogue)
        label = derive(meta,generated['report'],rgb,depth,valid,components,catalogue)
        require(label['target_id']==row['target_id'] and label['source_plant_family']==plan['source_family']
                and label['conservative_view_cap_group']==cap_group, 'Label ancestry differs from plan')
        require(label == reader.json('supervision/label.json'), 'Replayed label differs')
        trace = None
        if label['eligible']:
            trace = trace_review(meta,generated['report'],label,rgb,depth,valid,components,catalogue)
            require(trace == reader.json('supervision/query_trace.json'), 'Replayed trace differs')
        require(trace == row.get('query_trace'), 'Replayed trace differs from capture result')
        auto = bool(label['eligible'] and trace['passed'])
        require(auto == row['automatic_annotation_eligible'], 'Changed automatic decision')
        decision = ('accept_strict_automatic_annotation_candidate' if auto else
                    'hold_visual_clarity' if label['eligible'] else 'exclude_geometry_or_visibility')
        records.append(dict(sample=str(folder),target_id=label['target_id'],decision=decision,
            review_method='replayed_anatomy_exact_native_buffers_and_trace',visual_review_performed=False,
            label_replayed_exact=True,trace_replayed_exact=trace is not None,
            native_callback_hashes_verified=True,source_and_file_hashes_verified=True,
            reason=label['reason'],trace_reasons=trace['reasons'] if trace else None,
            source_target_group=label['conservative_view_cap_group'],source_family=label['source_plant_family'],
            render_profile=meta['render_budget']['profile'],
            training_approved=False,source_cap_reset=False,physical_execution_approved=False,
            sample_sha256=digest(reader.read('sample.json')),rgb_sha256=digest(reader.read('inputs/rgb.png')),
            label_sha256=digest(reader.read('supervision/label.json'))))
    require(len(records)==result['captured_frames'], 'Capture count mismatch')
    require(sum(r['decision']=='accept_strict_automatic_annotation_candidate' for r in records)
            ==result['automatically_clear_annotation_candidates'], 'Automatic count mismatch')
    verify_bindings(LOADED_REVIEW_CODE)
    return dict(state='completed_automatic_annotation_replay',capture=str(capture),records=records,
        counts=dict(Counter(r['decision'] for r in records)),training_approved=False,
        source_assets_unchanged=True,original_reviews_modified=False,
        result_sha256=sha256(capture/'result.json'),plan_sha256=sha256(plan_path),
        request_sha256=sha256(capture/'request.json'),review_code_bindings=LOADED_REVIEW_CODE)
