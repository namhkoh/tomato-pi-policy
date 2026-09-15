"""Synthetic CPU evidence only. No native worker/model/network is launched."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from . import short_reference_compare as compare
from . import query_audit_v2 as v2
from .capture_storage import write_compact_native_sample
from .bundle import SampleReader, digest
from .test_query_selection_v2 import native_case, seal
from ..capture_contract import fingerprint, project
from ..native_clear_labels import derive
from ..automated_native_review import trace_review


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    return compare.sha256(path)


@pytest.fixture(scope='module')
def native_samples(tmp_path_factory):
    root = tmp_path_factory.mktemp('short_reference_native_synthetic')
    result = {}
    for reference in (False, True):
        for index, name in enumerate(('warm', 'short')):
            args = native_case()
            meta, report, rgb, depth, valid, components, catalogue = args
            components = args[5] = components.astype(np.uint32)
            # Distinct camera and saved observations between views; same across runs.
            meta['sample_id'] = name
            meta['calibration']['intrinsics'][0][2] += index*.01
            meta['supervision']['nominal_projected'] = project(
                [meta['supervision']['nominal_world_m']], meta['calibration'])[0]
            meta['supervision'].update(source_target_id='fixture/Petiole',
                conservative_view_cap_group='fixture/Petiole', cut_safety_validated=False)
            rgb[0,0] += index
            depth[0,0] += np.float32(index*.001)
            budget = compare.budget_evidence(compare.REFERENCE if reference else compare.TRIAL, index)
            budget.update(actual_orchestrator_requests=budget['native_step_calls'], render_seconds=1.)
            meta.update(schema_version=v2.SAMPLE_SCHEMA if reference else compare.LEGACY_SCHEMA,
                state='native_static_multiview_candidate_pending_review', training_sample_approved=False,
                robot_snapshot={'view':index}, geometry_screen={'passed':True},
                native_instance_backend='fast', render_budget=budget, lighting={'fixture':True},
                renderer='synthetic_not_rendered', scene_counts={'fixture':True}, input_policy={'fixture':True})
            meta['synchronization'].update(engine_frame_id_verified=False, render_budget_subframes=budget['requested_subframes'],
                static_guard='0'*64, reference_time=[index,1], native_render_frame={'frameNumber':-1})
            seal(args)
            meta['synchronization']['freshness']['callback_sequence'] = index+1
            if reference:
                meta.update(v2.annotation_fields())
                label, trace = v2.annotate_for_storage(*args, target_mask=(components == 1).astype(np.uint8)*255)
            else:
                label = derive(*args)
                trace = trace_review(meta, report, label, *args[2:]) if label['eligible'] else None
            folder = root/('new' if reference else 'old')/name
            # Deliberately different renderer integers for the SAME prim identities.
            instances = components.copy()
            instances[components == 1] = 72 if reference else 12
            instances[components == 2] = 73 if reference else 13
            mapping = {'72' if reference else '12':'/fixture/Petiole',
                       '73' if reference else '13':'/fixture/Main'}
            stored = write_compact_native_sample(folder, rgb, depth, valid, meta, instances,
                mapping, catalogue, components, np.zeros_like(components, dtype=np.uint8), components == 1, label, trace)
            row = dict(candidate_id=name, target_id='fixture/Petiole', requested_spec={'candidate_id':name},
                state=compare.CAPTURED, screen=meta['geometry_screen'], robot_snapshot=meta['robot_snapshot'],
                render_budget=budget, eligible_annotation=label['eligible'], label_reason=label['reason'],
                automatic_annotation_eligible=bool(trace and trace['passed']), elapsed_seconds=2., **stored)
            result[reference,name] = dict(folder=folder, row=row, report=report, metadata=meta)
    return result


@pytest.fixture
def case(tmp_path, monkeypatch, native_samples):
    """Real reader/storage/selector; only external source/qualification proof stubs."""
    monkeypatch.setattr(compare, 'load_for_inspection', lambda *a: {'report':native_samples[False,'warm']['report']})
    monkeypatch.setattr(v2.compact_qualification, 'verify_checked_qualification', lambda _: True)
    anchor = tmp_path/'source'/'anchor.json'
    anchor_pin = write(anchor, dict(variant_directory=str(tmp_path/'source'/'variant'),
                                   source_collection_plan=str(tmp_path/'source'/'collection.json')))
    binding = {str(anchor):anchor_pin}
    names = ['reject','warm','short']
    plan = dict(split='train', source_family='fixture', resolution=[1696,816],
        training_approved=False, source_cap_reset=False, physical_motion_commanded=False,
        hidden_cut_coordinates_executable=False, maximum_native_frames=3, anchor_pair_plan=str(anchor),
        source_bindings=binding, prerequisite_bindings=binding,
        implementation_bindings={str(Path(compare.__file__).resolve()):compare.sha256(compare.__file__)},
        target_cases=[dict(target_id='fixture/Petiole', conservative_view_cap_group='fixture/Petiole',
                           views=[{'candidate_id':n} for n in names])])
    plan_path = tmp_path/'source'/'plan.json'
    plan_pin = write(plan_path, plan)
    sides, data = [], []
    for reference in (False,True):
        folder = tmp_path/('new_job' if reference else 'old_job')
        capture = folder/'capture'
        rows = [dict(candidate_id='reject',target_id='fixture/Petiole',requested_spec={'candidate_id':'reject'},
                     state='rejected_possible_geometry_overlap',screen={'passed':False})]
        audits = []
        for name in names[1:]:
            saved = native_samples[reference,name]
            shutil.copytree(saved['folder'], capture/name)
            rows.append(deepcopy(saved['row']))
            audits.append(dict(sample=str(capture/name),target_id='fixture/Petiole',
                decision='accept_strict_automatic_annotation_candidate',
                sample_sha256=saved['row']['sample_sha256'], label_sha256=saved['row']['label_sha256'],
                rgb_sha256=digest(SampleReader(capture/name).read('inputs/rgb.png'))))
        request = dict(plan_path=str(plan_path),plan_sha256=plan_pin,training_started=False,
            native_instance_backend='fast', render_budget_profile=compare.REFERENCE if reference else compare.TRIAL,
            render_profile_experiment=False)
        if reference:
            request.update(**v2.annotation_fields(), annotation_policy=v2.annotation_policy(),
                worker_module=v2.WORKER_MODULE, worker_implementation_bindings=v2.implementation_bindings(),
                compact_implementation_bindings=v2.implementation_bindings(),storage_qualification={'synthetic':True})
        request_pin = write(capture/'request.json', request)
        result = dict(request, state=v2.CAPTURE_STATE if reference else
            'native_generated_multiview_pilot_complete_pending_review', request_sha256=request_pin,
            training_approved=False,source_cap_reset=False,source_assets_unchanged=True,
            records=rows,captured_frames=2,automatically_clear_annotation_candidates=2)
        result_pin = write(capture/'result.json',result)
        audit = dict(state=v2.AUDIT_STATE if reference else 'completed_automatic_annotation_replay',
            capture=str(capture),plan_sha256=plan_pin,request_sha256=request_pin,result_sha256=result_pin,
            training_approved=False,original_reviews_modified=False,records=audits,
            counts={'accept_strict_automatic_annotation_candidate':2})
        audit_path = folder/'audit.json'
        audit_pin = write(audit_path,audit)
        side = dict(capture_path=str(capture),plan_path=str(plan_path),plan_sha256=plan_pin,
            request_sha256=request_pin,result_sha256=result_pin,audit_path=str(audit_path),audit_sha256=audit_pin)
        if not reference:
            done = dict(state='native_complete_pending_review',native_exit_code=0,job=str(folder),
                source_family='fixture',training_approved=False,captured=2,automatic=2,audit_counts=audit['counts'])
            done_path = folder/'campaign_result.json'
            side.update(completion_path=str(done_path),completion_sha256=write(done_path,done))
        sides.append(side)
        data.append(dict(request=request,result=result,audit=audit))
    handoff = dict(schema=compare.HANDOFF,old=sides[0],new=sides[1],backend='fast',new_render_budget=compare.REFERENCE,
        old_annotation_epoch_unchanged=True,comparisons_performed=False,equivalence_cutoff=None,
        native_query_qualification_granted=False,short_budget_qualified=False,source_cap_reset=False,
        training_approved=False,training_diversity_increment=0,planned_proposals=3,captured_frames=2)
    receipt = dict(schema='greenhouse.native_serial_phases.job.v1',state='owned_exit0_postexit_audited_pending_admission',
        worker_kind='query_v2_qualification.v1',exit_code=0,training_approved=False,source_cap_reset=False,
        plan_sha256=plan_pin,submitted_plan_path=str(plan_path),capture=sides[1]['capture_path'],
        audit={k:sides[1][k] for k in ('request_sha256','result_sha256','audit_path','audit_sha256')})
    receipt['audit']['matched_comparison'] = handoff
    path = tmp_path/'new_job'/'receipt.json'
    return dict(path=path, pin=write(path,receipt), receipt=receipt, data=data, plan=plan, sides=sides)


def run(case):
    return compare.compare_receipt(case['path'], expected_sha256=case['pin'])


def repin(case):
    case['pin'] = write(case['path'],case['receipt'])


def reader_for(sample):
    row = sample['row']
    return SampleReader(sample['folder'],expected_bindings={'sample.json':row['sample_sha256'],
        'supervision/label.json':row['label_sha256']},
        expected_json={'supervision/query_trace.json':row['query_trace']} if row['query_trace'] else {})


@pytest.mark.parametrize('reference',[False,True])
def test_actual_storage_native_v2_replay_and_overlay_isolation(native_samples,reference):
    sample = native_samples[reference,'short']
    reader = reader_for(sample)
    before = deepcopy(reader.metadata)
    result = compare.replay_frame(reader,sample['row'],sample['report'],reference=reference)
    repeated = compare.replay_frame(reader,sample['row'],sample['report'],reference=reference)
    proof = result['proof']
    assert proof == repeated['proof']
    assert reader.metadata == before
    assert proof['native_v2_capture'] is reference and proof['comparison_only'] is not reference
    assert proof['metadata_overlay'] == ({} if reference else v2.annotation_fields())
    assert proof['original_label_replayed_exact'] and proof['original_trace_replayed_exact']
    private = deepcopy(before)
    private.update(proof['metadata_overlay'])
    assert proof['comparison_input_metadata_sha256'] == fingerprint(v2.canonical_annotation_metadata(private))
    assert private['schema_version'] == before['schema_version']
    assert private['synchronization'] == before['synchronization']
    assert private['render_budget'] == before['render_budget']
    assert not proof['v2_label']['training_approved']
    assert min(proof['v2_trace']['fixed_grid']['probes'],key=lambda p:p['arc_m'])['arc_m'] == .008


@pytest.mark.parametrize('storage',['compact','raw'])
def test_full_handoff_join_budget_semantics_review_and_no_source_writes(case,storage):
    if storage == 'raw':
        # Test fixtures ONLY: create byte-exact raw siblings via the verified reader.
        for side in case['sides']:
            for name in ('warm','short'):
                folder=Path(side['capture_path'])/name
                moved=folder.with_name(name+'_synthetic_compact_fixture')
                folder.rename(moved)
                reader=SampleReader(moved)
                for logical in reader.manifest['files']:
                    dest=folder/logical
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(reader.read(logical))
    before = {str(p):compare.sha256(p) for p in case['path'].parent.parent.rglob('*') if p.is_file()}
    report = run(case)
    after = {str(p):compare.sha256(p) for p in case['path'].parent.parent.rglob('*') if p.is_file()}
    assert before == after
    assert report['budget_counts'] == {'56_vs_56_warmup':1,'8_vs_56':1}
    assert report['state_counts'] == {'matched_rejection':1,'compared_unqualified':2}
    assert [r['candidate_id'] for r in report['records']] == ['reject','warm','short']
    assert all(report[k] == v for k,v in compare.FLAGS.items())
    assert report['equivalence_cutoff'] is None
    for row in report['records'][1:]:
        assert row['core_differences'] == row['selection_differences'] == []
        assert row['metrics']['full']['semantic_prim_changed_pixels'] == 0
        assert row['metrics']['full']['semantic_component_changed_pixels'] == 0
        assert row['metrics']['full']['native_z_absolute_delta_m']['maximum'] == 0
        assert row['metrics']['full']['rgb_absolute_delta_8bit']['maximum'] == 0
        assert row['camera_freshness']['engine_frame_id_verified'] is False
    assert all(v['coordinate_system']=='native_1696x816_half_open_xyxy_no_resize' for v in report['review_index'])
    for item in report['review_index']:
        for key,box in item['native_regions_xyxy'].items():
            if 'query_crop' in key:
                assert box[2]-box[0] == box[3]-box[1] == 768
    destination = case['path'].parent.parent/'comparison.json'
    saved = compare.write_report(report,destination)
    assert saved['sha256'] == compare.sha256(destination)
    assert json.loads(destination.read_bytes()) == report
    with pytest.raises(ValueError,match='New file'):
        compare.write_report(report,destination)
    with pytest.raises(ValueError,match='protected'):
        compare.write_report(report,case['path'].parent/'must_not_write.json')


@pytest.mark.parametrize('target',['receipt','request','result','audit','completion','plan','payload','bundle'])
def test_tamper_fails_closed(case,target):
    old = case['sides'][0]
    path = {'receipt':case['path'], 'request':Path(old['capture_path'])/'request.json',
        'result':Path(old['capture_path'])/'result.json','audit':Path(old['audit_path']),
        'completion':Path(old['completion_path']),'plan':Path(old['plan_path']),
        'payload':Path(old['capture_path'])/'warm'/'payload'/'inputs'/'rgb.png',
        'bundle':Path(old['capture_path'])/'warm'/'bundle.json'}[target]
    if target == 'bundle':
        body = json.loads(path.read_bytes())
        body['files']['inputs/rgb.png']['logical_sha256'] = '0'*64
        write(path,body)
    else:
        path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises((ValueError,KeyError)):
        run(case)


@pytest.mark.parametrize('field,value',[
    ('state','owned_child_running'),('exit_code',True),('exit_code',1),
    ('worker_kind','original_batch.v1'),('training_approved',True),('plan_sha256','0'*64)])
def test_completed_receipt_boundaries(case,field,value):
    case['receipt'][field]=value
    repin(case)
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize('field,value',[
    ('backend','legacy'),('new_render_budget',compare.TRIAL),('short_budget_qualified',True),
    ('equivalence_cutoff',.0002),('comparisons_performed',True),('training_diversity_increment',1),
    ('old_annotation_epoch_unchanged',False),('native_query_qualification_granted',True)])
def test_handoff_no_threshold_or_approval_or_diversity_claims(case,field,value):
    case['receipt']['audit']['matched_comparison'][field]=value
    repin(case)
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize('field',['target_id','requested_spec','candidate_id','state'])
def test_rows_reject_wrong_identity_and_state(case,field):
    result = deepcopy(case['data'][0]['result'])
    result['records'][1][field]='wrong'
    with pytest.raises(ValueError):
        compare._rows(result,case['plan'])


def test_duplicate_row_rejected_and_missing_decision_retained(case):
    result = deepcopy(case['data'][0]['result'])
    result['records'].append(result['records'][0])
    with pytest.raises(ValueError,match='duplicate'):
        compare._rows(result,case['plan'])
    result['records'] = result['records'][1:-1]
    rows, planned = compare._rows(result,case['plan'])
    assert 'reject' in planned and 'reject' not in rows


@pytest.mark.parametrize('profile,index,expected',[
    (compare.TRIAL,0,56),(compare.TRIAL,1,8),(compare.TRIAL,10,8),(compare.REFERENCE,9,56)])
def test_per_image_budget_uses_captured_index_not_proposal(profile,index,expected):
    budget = compare.budget_evidence(profile,index)
    budget['actual_orchestrator_requests'] = budget['native_step_calls']
    meta = {'render_budget':budget,'synchronization':{'render_budget_subframes':expected}}
    result = compare._budget(meta,{'render_budget':budget},profile,index)
    assert result['actual_requested_subframes'] == expected and result['exact_nominal_request_count']


def test_retries_are_measured_not_disguised_as_eight():
    budget = compare.budget_evidence(compare.TRIAL,1)
    budget['actual_orchestrator_requests'] = 2
    result = compare._budget({'render_budget':budget,'synchronization':{'render_budget_subframes':8}},
                             {'render_budget':budget},compare.TRIAL,1)
    assert result['actual_requested_subframes'] == 16 and result['retry_requests'] == 1
    assert not result['exact_nominal_request_count']


@pytest.mark.parametrize('fault',['row','sync','requests','bool','warmup'])
def test_inconsistent_budget_rejected(fault):
    budget = compare.budget_evidence(compare.TRIAL,1)
    budget['actual_orchestrator_requests'] = 1
    row = {'render_budget':deepcopy(budget)}
    meta = {'render_budget':budget,'synchronization':{'render_budget_subframes':8}}
    if fault == 'row': row['render_budget']['requested_subframes']=56
    if fault == 'sync': meta['synchronization']['render_budget_subframes']=56
    if fault == 'requests': budget['actual_orchestrator_requests']=0; row['render_budget']=budget
    if fault == 'bool': budget['actual_orchestrator_requests']=True; row['render_budget']=budget
    with pytest.raises(ValueError):
        compare._budget(meta,row,compare.TRIAL,0 if fault == 'warmup' else 1)


def test_semantic_renumbering_not_raw_integer_comparison():
    a=np.array([[0,1,2,3]],np.uint32)
    b=np.array([[0,1,7,8]],np.uint32)
    assert not compare.semantic_changes(a,{'2':'/A','3':'/B'},b,{'7':'/A','8':'/B'}).any()
    assert compare.semantic_changes(a,{'2':'/A','3':'/B'},b,{'7':'/B','8':'/A'}).tolist()==[[False,False,True,True]]
    # Same raw integer with different prim identity IS a difference.
    assert compare.semantic_changes(a,{'2':'/A','3':'/B'},a,{'2':'/B','3':'/B'}).sum()==1


@pytest.mark.parametrize('mapping',[{}, {'02':'/A'},{'2':'not/a/prim'},{'2':None},{'-2':'/A'}])
def test_unknown_or_malformed_semantic_identity_is_not_background(mapping):
    a=np.array([[2]],np.uint32)
    with pytest.raises(ValueError):
        compare.semantic_changes(a,mapping,a,{'2':'/A'})


@pytest.mark.parametrize('field', ['calibration','robot_snapshot','scene_counts','renderer','lighting',
    'geometry_screen','rendered_camera_params','input_policy','unexpected_scene_field'])
def test_core_scene_camera_changes_hold(native_samples,field):
    a=deepcopy(native_samples[False,'warm']['metadata'])
    b=deepcopy(a)
    b[field]={'changed':True}
    assert field in compare._core_changes(a,b)


@pytest.mark.parametrize('field',['nominal_world_m','interval_world_m','source_target_id','split_group',
    'target_id','conservative_view_cap_group','plant_to_world_usd_row_vectors','cut_safety_validated'])
def test_core_supervision_mismatch_not_hidden(native_samples,field):
    a=deepcopy(native_samples[False,'warm']['metadata'])
    b=deepcopy(a)
    b['supervision'][field]='changed'
    assert 'supervision.'+field in compare._core_changes(a,b)


def test_cross_run_freshness_tokens_and_observation_variation_not_core(native_samples):
    a=deepcopy(native_samples[False,'warm']['metadata'])
    b=deepcopy(a)
    b['synchronization']['freshness']['callback_sequence']+=12
    b['synchronization']['reference_time']=[100,1]
    b['supervision']['depth_evidence']={'changed_observation':True}
    b['quality']['target_mask_dark_fraction']=.1
    assert compare._core_changes(a,b)==[]
    b['synchronization']['static_guard']='changed'
    assert compare._core_changes(a,b)==['synchronization.static_guard']


def test_numeric_metrics_have_no_point_two_mm_cutoff_and_empty_means_unknown():
    assert compare._stats([])=={'count':0,'mean':None,'p95':None,'maximum':None}
    assert compare._stats([.000199,.000201])['maximum']==.000201
    assert set(compare._stats([1,2]))=={'count','mean','p95','maximum'}


@pytest.mark.parametrize('padding',[-1,817,True,1.5])
def test_bad_display_padding_rejected_before_io(padding):
    with pytest.raises(ValueError,match='padding'):
        compare.compare_receipt('absent',expected_sha256='0'*64,roi_padding_px=padding)


def test_roi_clipping_not_coordinate_resizing():
    assert compare._box([[1.2,3.4]],64)==[0,0,67,69]
    assert compare._box([[-1000,-1000]],0) is None
    assert compare._box([[1695,815]],0)==[1695,815,1696,816]


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_pinned_json_rejects_ambiguous_or_nonfinite_payload(tmp_path,raw):
    path=tmp_path/'bad.json'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        compare._json(path,digest(raw),{})


def test_conflicting_pin_rejected(tmp_path):
    path=tmp_path/'one.json'
    pin=write(path,{'fixture':True})
    with pytest.raises(ValueError,match='Conflicting'):
        compare._pin(path,pin,{str(path):'0'*64})


def test_no_launch_write_or_legacy_probe_threshold_imports():
    source=Path(compare.__file__).read_text(encoding='utf-8')
    assert 'native_render_probe' not in source
    assert '.0002' not in source and '0.2mm' not in source
    assert 'subprocess' not in source and 'SimulationApp' not in source
    assert 'junction_before_8mm_verified=False' in source
    assert source.count(".open('xb')")==1
    assert 'boxes need explicit visual review' in source and 'need human review' not in source


@pytest.mark.parametrize('location',['top','geometry_screen'])
def test_unrecognized_metadata_timer_is_an_honest_core_hold(native_samples,location):
    a=deepcopy(native_samples[False,'warm']['metadata'])
    b=deepcopy(a)
    if location == 'top':
        b['elapsed_seconds']=1.0
        assert compare._core_changes(a,b)==['elapsed_seconds']
    else:
        b[location]['elapsed_seconds']=1.0
        assert compare._core_changes(a,b)==['geometry_screen']


@pytest.mark.parametrize('field',['callback_sequence','camera_sha256','rgb_sha256','depth_sha256'])
def test_stale_within_run_fails_independently_of_cross_run(field):
    prior=dict(callback_sequence=5,camera_sha256='a',rgb_sha256='b',depth_sha256='c')
    current=dict(callback_sequence=6,camera_sha256='d',rgb_sha256='e',depth_sha256='f')
    assert compare._freshness({'synchronization':{'freshness':current}},prior)==current
    current[field]=prior[field]
    with pytest.raises(ValueError,match='stale'):
        compare._freshness({'synchronization':{'freshness':current}},prior)
    # No cross-run predecessor supplied: equality with another run is legitimate.
    assert compare._freshness({'synchronization':{'freshness':current}},None)==current


@pytest.mark.parametrize('sequence',[0,-1,True,1.2])
def test_invalid_initial_callback_sequence(sequence):
    with pytest.raises(ValueError,match='callback sequence'):
        compare._freshness({'synchronization':{'freshness':{'callback_sequence':sequence}}},None)


def test_metric_native_z_and_validity_and_components_are_separate():
    frame=dict(instances=np.array([[0,12],[12,0]],np.uint32),
        identities={'renderer_id_to_prim':{'12':'/fixture/Petiole'},'component_catalogue':[
            {'component_index':1,'variant_id':'fixture','component_id':'Petiole'}]},
        components=np.array([[0,1],[1,0]],np.uint32), target=np.array([[False,True],[True,False]]),
        rgb=np.full((2,2,3),20,np.uint8),depth=np.ones((2,2),np.float32),valid=np.ones((2,2),bool))
    other=deepcopy(frame)
    other['depth'][0,0]+=np.float32(.003)
    other['valid'][0,1]=False
    other['depth'][1,1]=np.nan
    other['instances'][1,0]=0
    other['components'][1,0]=0
    other['target'][1,0]=False
    other['rgb'][1,0]=40
    with np.errstate(invalid='ignore'):
        metrics=compare._measure(frame,other,{'tiny':[0,0,2,2],'absent':None})
    m=metrics['tiny']
    assert m['native_z_absolute_delta_m']['count']==2
    assert m['native_z_absolute_delta_m']['maximum']>.0029
    assert m['validity_changed_pixels']==1 and m['valid_nonfinite_new']==1
    assert m['semantic_prim_changed_pixels']==m['semantic_component_changed_pixels']==m['target_mask_changed_pixels']==1
    assert m['rgb_absolute_delta_8bit']['maximum']==20
    assert metrics['absent']=={'available':False}
    assert not any('pass' in key or 'threshold' in key for key in m)
    # Component integers are also frame-local, not anatomical identity.
    reindexed=deepcopy(frame)
    reindexed['components'][frame['components']==1]=44
    reindexed['identities']['component_catalogue'][0]['component_index']=44
    assert compare._measure(frame,reindexed,{'tiny':[0,0,2,2]})['tiny']['semantic_component_changed_pixels']==0


@pytest.mark.parametrize('reference',[False,True])
def test_wrong_source_epoch_rejected_before_replay(native_samples,reference):
    sample=native_samples[not reference,'warm']
    with pytest.raises(ValueError,match='source epoch'):
        compare.replay_frame(reader_for(sample),sample['row'],sample['report'],reference=reference)


@pytest.mark.parametrize('field',['sample_sha256','label_sha256','target_id','robot_snapshot','screen','query_trace'])
def test_replay_row_binding_tamper(native_samples,field):
    sample=native_samples[False,'warm']
    row=deepcopy(sample['row'])
    row[field]='changed'
    with pytest.raises(ValueError):
        compare.replay_frame(reader_for(sample),row,sample['report'],reference=False)


def test_optional_actual_old_compact_legacy_replay_only():
    """Set SHORT_REFERENCE_OLD_CAPTURE to job1/capture; source writes are forbidden.

    Uses fixed independently supplied old result/plan pins from the approved
    two-job protocol, NOT a self-pinned user-selected capture. No reference claim.
    """
    location=os.environ.get('SHORT_REFERENCE_OLD_CAPTURE')
    if not location:
        pytest.skip('Optional existing job1 old compact replay; no reference capture implied')
    root=Path(location).resolve()
    pins={}
    result=compare._json(root/'result.json','4bfd0dc54442187faea5db1bd7de3d6fc4e74236f55bd6f6b7a6946426556a34',pins)
    plan=compare._json(root.parent/'plan.json','b1f448a6afd5d4af5ed6f1e203f7842c29a0ab962587fa4da2503a5e480ee7dd',pins)
    compare.verify_bindings(plan['source_bindings'])
    compare.verify_bindings(plan['implementation_bindings'])
    compare.verify_bindings(plan['prerequisite_bindings'])
    anchor_path=str(Path(plan['anchor_pair_plan']).resolve())
    anchor_pin=next(m[anchor_path] for m in (plan['source_bindings'],plan['prerequisite_bindings']) if anchor_path in m)
    anchor=compare._json(anchor_path,anchor_pin,pins)
    generated=compare.load_for_inspection(anchor['variant_directory'],anchor['source_collection_plan'])
    row=next(r for r in result['records'] if r['state']==compare.CAPTURED)
    reader=SampleReader(root/row['candidate_id'],expected_bindings={
        'sample.json':row['sample_sha256'],'supervision/label.json':row['label_sha256']},
        expected_json={'supervision/query_trace.json':row['query_trace']} if row['query_trace'] else {})
    before={str(p):compare.sha256(p) for p in reader.root.rglob('*') if p.is_file()}
    frame=compare.replay_frame(reader,row,generated['report'],reference=False)
    budget=compare._budget(frame['meta'],row,compare.TRIAL,0)
    regions=compare._regions(frame,frame,generated['report'],64)
    metrics=compare._measure(frame,frame,regions)
    assert budget['actual_requested_subframes']==56  # NOT evidence for 8-vs-56.
    assert metrics['full']['semantic_prim_changed_pixels']==0
    assert metrics['full']['native_z_absolute_delta_m']['maximum']==0
    assert frame['proof']['original_label_replayed_exact'] and frame['proof']['comparison_only']
    assert not frame['proof']['native_v2_capture']
    assert before=={str(p):compare.sha256(p) for p in reader.root.rglob('*') if p.is_file()}
