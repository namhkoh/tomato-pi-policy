"""Explicit query-V2 replay and shared annotation serialization contract.

Full selection evidence lives in label.json, including upstream exclusions;
query_trace.json is the exact composite trace (or absent for exclusions).
No third sidecar, evidence packing, old-epoch relabel, or training approval.

canonical_annotation_metadata removes ONLY storage-generated "files", after
the caller verifies its byte bindings. Epoch/camera/freshness fields remain.
The input-metadata hash is stored in the label, never in its own input object.

CLI: --capture DIR --plan PLAN --result-sha256 SHA --output NEW_RECEIPT
Initial integration accepts the existing multi-target/batch plan format only.
The worker and this audit are source-pinned together without circular imports.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..dataset_review import read_json, require, verify_bindings, write_json
from ..depth_preview import sha256
from ..plant_variant_catalogue import load_for_inspection
from ..automated_native_review import check_native_evidence
from ..capture_contract import fingerprint
from ..native_clear_contract import contract_hash
from . import query_selection_v2 as selector
from .query_selection_v2 import annotate_v2, SCHEMA as ANNOTATION_EPOCH
from . import compact_qualification
from .bundle import SampleReader, digest
from . import bundle

SAMPLE_SCHEMA = 'greenhouse.generated_native_multiview_sample.v2'
CAPTURE_STATE = 'native_query_v2_multiview_complete_pending_review'
AUDIT_STATE = 'completed_query_v2_automatic_annotation_replay'
WORKER_MODULE = 'sim_data.native_dataset.compact_query_v2'
FROZEN_SELECTOR_SHA256 = 'b2cfba71878d556ccf6a63f0129e022c73601d82b6ca6e6e82e34e18ab258d84'
_HERE = Path(__file__).resolve().parent
_POLICY_PATHS = [_HERE/'query_selection_v2.py', *[_HERE.parent/name for name in (
    'native_clear_labels.py', 'automated_native_review.py', 'native_clear_contract.py',
    'native_query_visibility.py', 'query_visibility.py', 'cut_regions.py', 'capture_contract.py')]]
require(sha256(selector.__file__) == FROZEN_SELECTOR_SHA256, 'Reviewed query selector changed')
_POLICY = dict(annotation_epoch=ANNOTATION_EPOCH,
    selector_source_sha256=FROZEN_SELECTOR_SHA256,
    native_task_contract_sha256=contract_hash(),
    policy_source_sha256={p.name:sha256(p) for p in _POLICY_PATHS},
    legacy_trace_policy=deepcopy(selector.legacy_review.POLICY),
    local_query_policy=deepcopy(selector.QUERY_POLICY),
    candidate_count=selector.CANDIDATES, minimum_query_arc_m=selector.MIN_QUERY_M,
    maximum_query_arc_m=selector.MAX_QUERY_M, minimum_query_cut_distance_px=selector.MIN_QUERY_CUT_PX,
    grid_schema=selector.GRID_SCHEMA, anchor_mm=selector.START_MM, step_mm=selector.STEP_MM,
    anatomical_knots_included=True, query_endpoint_included=True, legacy_trace_required=True,
    evidence_storage='complete_selection_evidence_nested_in_label.v1',
    metadata_fingerprint_omits_only=['files'],
    training_approved=False, source_cap_reset=False, native_depth_reconstructed=False)
POLICY_SHA256 = fingerprint(_POLICY)
LOADED_REVIEW_CODE = dict(compact_qualification.implementation_bindings(), **{
    str(p.resolve()):sha256(p) for p in (
        Path(__file__), _HERE/'compact_query_v2.py', _HERE/'compact_views.py',
        Path(bundle.__file__), *_POLICY_PATHS)})


def implementation_bindings():
    """Exact loaded worker/auditor/selector/storage and policy-helper map."""
    verify_bindings(LOADED_REVIEW_CODE)
    return dict(LOADED_REVIEW_CODE)


def annotation_policy():
    return deepcopy(_POLICY)


def annotation_fields():
    return dict(annotation_epoch=ANNOTATION_EPOCH, annotation_policy_sha256=POLICY_SHA256)


def canonical_annotation_metadata(metadata):
    """Private copy; remove ONLY the compact/raw writer's generated file map."""
    require(isinstance(metadata, dict), 'Annotation metadata must be an object')
    result = deepcopy(metadata)
    result.pop('files', None)
    return result


def annotate_for_storage(metadata, report, rgb, depth, valid, components, catalogue, *, target_mask):
    """One frozen selector invocation; deterministic complete JSON evidence.

    Capture calls before serialization. Audit verifies stored byte bindings first.
    target_mask must already be the unchanged native target PNG values, uint8
    0/255; no ID conversion, image/depth resampling or query-mask reconstruction.
    """
    require(all(metadata.get(k) == v for k,v in annotation_fields().items()), 'Wrong annotation epoch/policy')
    canonical = canonical_annotation_metadata(metadata)
    label, trace, evidence = annotate_v2(canonical, report, rgb, depth, valid, components, catalogue,
                                         target_mask=target_mask)
    input_hash = fingerprint(canonical)
    require(label['annotation_epoch'] == evidence['annotation_epoch'] == ANNOTATION_EPOCH
            and evidence['schema'] == ANNOTATION_EPOCH
            and evidence['input_fingerprints']['metadata'] == input_hash, 'Annotation input fingerprint changed')
    require((trace is not None) == label['eligible'], 'Annotation/trace disposition mismatch')
    if trace is not None:
        require(trace['annotation_epoch'] == ANNOTATION_EPOCH
                and trace['passed'] == (evidence['selected_query'] is not None)
                and trace['passed'] == (trace['legacy_trace']['passed'] is True
                                       and trace['fixed_grid']['passed'] is True), 'Trace conjunction changed')
    require('query_selection_evidence' not in label, 'Reserved evidence field already present')
    label = dict(label, annotation_policy_sha256=POLICY_SHA256,
        annotation_input_metadata_sha256=input_hash,
        query_selection_evidence_sha256=fingerprint(evidence),
        query_selection_evidence=evidence)
    return label, trace


def verify_epoch_documents(request, result):
    expected = implementation_bindings()
    for document in (request, result):
        require(all(document.get(k) == v for k,v in annotation_fields().items())
                and document.get('annotation_policy') == _POLICY, 'Mixed/unknown annotation epoch or policy')
        require(document.get('worker_module') == WORKER_MODULE
                and document.get('worker_implementation_bindings') == expected
                and document.get('compact_implementation_bindings') == expected,
                'Wrong query worker/annotation implementation')
    require(request['storage_qualification'] == result['storage_qualification'],
            'Different request/result storage proof')
    compact_qualification.verify_checked_qualification(request['storage_qualification'])

def audit_capture(capture, plan_path, *, result_sha256, storage_root=None):
    verify_bindings(LOADED_REVIEW_CODE)
    capture, plan_path = Path(capture).resolve(), Path(plan_path).resolve()
    storage_root = Path(storage_root).resolve() if storage_root is not None else capture
    require(not (capture/'failure.json').exists(), 'Failed native capture cannot be audited')
    source_pins = {str(capture/'result.json'):result_sha256,
                   str(capture/'request.json'):sha256(capture/'request.json'), str(plan_path):sha256(plan_path)}
    require(isinstance(result_sha256, str) and len(result_sha256) == 64
            and all(c in '0123456789abcdef' for c in result_sha256), 'Explicit result SHA256 required')
    verify_bindings(source_pins)
    result = read_json(capture/'result.json'); plan = read_json(plan_path)
    request = read_json(capture/'request.json')
    verify_epoch_documents(request, result)
    require(result['request_sha256'] == source_pins[str(capture/'request.json')]
            and result['plan_sha256'] == source_pins[str(plan_path)], 'Unbound query capture completion')
    require('target_cases' in plan, 'Initial query-V2 audit requires a batch plan')
    require(request['plan_sha256']==sha256(plan_path) and
            Path(request['plan_path']).resolve()==plan_path, 'Capture request/plan mismatch')
    require(plan.get('split')=='train' and plan.get('resolution')==[1696,816] and
            all(plan.get(k) is False for k in ('training_approved','source_cap_reset',
                'physical_motion_commanded','hidden_cut_coordinates_executable')), 'Unexpected plan contract')
    require(request.get('training_started') is False and result.get('training_approved') is False,
            'Unexpected capture training claim')
    require(result.get('state')==CAPTURE_STATE,
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
        require(meta.get('schema_version') == SAMPLE_SCHEMA
                and all(meta.get(k) == v for k,v in annotation_fields().items()), 'Mixed sample annotation epoch')
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
        label, trace = annotate_for_storage(meta,generated['report'],rgb,depth,valid,components,catalogue,
                                             target_mask=target)
        require(label['target_id']==row['target_id'] and label['source_plant_family']==plan['source_family']
                and label['conservative_view_cap_group']==cap_group, 'Label ancestry differs from plan')
        require(label == reader.json('supervision/label.json'), 'Replayed label differs')
        if label['eligible']:
            require(trace == reader.json('supervision/query_trace.json'), 'Replayed trace differs')
        else:
            require(('supervision/query_trace.json' not in reader.manifest['files'] if reader.manifest
                     else not (folder/'supervision/query_trace.json').exists()),
                    'Excluded sample contains a trace')
        require(trace == row.get('query_trace'), 'Replayed trace differs from capture result')
        auto = bool(label['eligible'] and trace['passed'])
        require(auto == row['automatic_annotation_eligible'], 'Changed automatic decision')
        decision = ('accept_strict_automatic_annotation_candidate' if auto else
                    'hold_visual_clarity' if label['eligible'] else 'exclude_geometry_or_visibility')
        records.append(dict(sample=str(folder),target_id=label['target_id'],decision=decision,
            review_method='replayed_native_query_v2_full_selection_and_composite_trace',visual_review_performed=False,
            **annotation_fields(),
            annotation_input_metadata_sha256=label['annotation_input_metadata_sha256'],
            query_selection_evidence_sha256=label['query_selection_evidence_sha256'],
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
    verify_bindings(source_pins)
    verify_bindings(plan['source_bindings'])
    verify_epoch_documents(request, result)
    return dict(state=AUDIT_STATE,capture=str(capture),records=records,
        **annotation_fields(), annotation_policy=annotation_policy(),
        worker_module=WORKER_MODULE, worker_implementation_bindings=implementation_bindings(),
        counts=dict(Counter(r['decision'] for r in records)),training_approved=False,
        source_assets_unchanged=True,original_reviews_modified=False,
        result_sha256=sha256(capture/'result.json'),plan_sha256=sha256(plan_path),
        request_sha256=sha256(capture/'request.json'),review_code_bindings=LOADED_REVIEW_CODE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--storage-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    plan = read_json(args.plan)
    protected = [args.capture.resolve(), args.plan.resolve(),
                 *(Path(p).resolve() for key in ('source_bindings','implementation_bindings','prerequisite_bindings')
                   for p in plan[key])]
    if args.storage_root is not None:
        protected.append(args.storage_root.resolve())
    require(not output.exists() and output != Path(output.anchor)
            and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
            'New audit receipt outside capture and immutable inputs required')
    review = audit_capture(args.capture, args.plan, result_sha256=args.result_sha256,
                           storage_root=args.storage_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, review)


if __name__ == '__main__':
    main()
