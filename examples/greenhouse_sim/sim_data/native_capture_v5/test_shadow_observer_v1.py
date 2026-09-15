"""Synthetic observer/owned-exit fixtures only; no native launch or capture claim."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from . import shadow_observer_v1 as p
from . import shadow_observer_cpu_v1 as cpu


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    p.write_json_new(path, value)
    return p.file_pin(path)


def matrix():
    return [[float(i == j) for j in range(4)] for i in range(4)]


def calibration():
    return dict(resolution=[1696, 816], crop_resize=None, clipping_range_m=[.01, 10.],
        intrinsics=[[941., 0., 848.], [0., 941., 408.], [0., 0., 1.]],
        camera_to_world_usd_row_vectors=[[1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,1.]])


@pytest.fixture
def fixture(tmp_path):
    producer = tmp_path/'native_generated_reference/worker_v1.py'
    producer.parent.mkdir(); producer.write_text('# synthetic receipt producer fixture; never executed\n')
    bank = write(tmp_path/'bank.json', {'synthetic': True})
    clear = write(tmp_path/'clear.json', {'synthetic': True, 'split': 'train'})
    proof = write(tmp_path/'proof.json', dict(bank=bank, entry_id='anchor'))
    geometry = write(tmp_path/'variant/manifest.json', {'synthetic_geometry_fixture': True})
    plan = write(tmp_path/'plan.json', dict(schema='greenhouse.generated_from_original_reference_plan.v1',
        split='train', scene_authority=dict(clear_plan=clear), anchor_evidence=proof,
        variant_directory=str(tmp_path/'variant'), source_bindings={geometry['path']: geometry['sha256']},
        expected_original_world=dict(plant_to_world_usd_row_vectors=matrix()),
        modes=['original_control','generated_variant'], sample_count_limit=2,
        source_row=dict(target_id='original/target',component_id='target'),
        generated_row=dict(target_id='variant/target',component_id='target')))
    spec = dict(schema=p.PREFIX+'spec', plan=plan, bank=bank, clear_plan=clear, anchor_entry_id='anchor',
        producer_module=p.WORKER_MODULE, producer_bindings={str(producer):p.sha(producer)},
        implementation_bindings=p.implementation_bindings(), candidates=[
            dict(candidate_id='original',mode='original_control',target_id='original/target',component_id='target'),
            dict(candidate_id='generated',mode='generated_variant',target_id='variant/target',component_id='target')], flags=p.FLAGS)
    context = dict(kind='generated_context',context_id='plant',variant_directory=str(tmp_path/'variant'),
        geometry_bindings={geometry['path']:geometry['sha256']},plant_to_world_usd_row_vectors=matrix(),
        scene_evidence_sha256='1'*64)
    def event(candidate_id, counter=0):
        return dict(kind='pre_render',candidate_id=candidate_id,context_id='plant' if candidate_id=='generated' else None,
            calibration=calibration(),robot_snapshot_sha256='2'*64,writer_request_index_before=counter)
    return dict(root=tmp_path,spec=spec,context=context,event=event)


def journal(f, *, first=None, second=None, missing=False):
    writer = p.EventWriter(f['root']/'journal', p.canonical(f['spec']))
    assert writer.bind_context(p.canonical(f['context'])) is None
    assert writer.record_candidate(p.canonical(first or f['event']('original'))) is None
    if not missing:
        assert writer.record_candidate(p.canonical(second or f['event']('generated', 7))) is None
    assert writer.seal() is None
    return p.file_pin(f['root']/'journal/manifest.json')


def exit_fixture(f, manifest, *, exit_code=0, alter_outcomes=None):
    ledger = {}; spec, contexts, events, m = p.read_journal(manifest, ledger)
    spec_pin = m['spec']
    request = write(f['root']/'native/request.json', dict(plan_path=spec['plan']['path'],
        plan_sha256=spec['plan']['sha256'], shadow_observer_spec=spec_pin))
    worker = dict(pid=12345, command=[str(Path(sys.executable).resolve()), '-m', p.WORKER_MODULE])
    launch = write(f['root']/'owned_launch.json', dict(schema=p.PREFIX+'owned_launch',state='owned_worker_running',
        worker=worker,spec=spec_pin,native_request=request,producer_bindings=spec['producer_bindings'],flags=p.FLAGS))
    rows=[]
    for candidate in spec['candidates']:
        e=events[candidate['candidate_id']]; payload=e['payload']; captured=payload['kind']!='geometry_hold'
        sample=write(f['root']/('native/'+candidate['candidate_id']+'.json'),{'synthetic':True}) if captured else None
        rows.append(dict(candidate_id=candidate['candidate_id'],state='captured' if captured else 'geometry_hold',
            event_seal=e['seal'],first_render_request_index=payload.get('writer_request_index_before',0)+1 if captured else None,
            sample=sample,calibration_sha256=p.digest(p.canonical(payload.get('calibration',calibration()))) if captured else None,
            robot_snapshot_sha256='2'*64 if captured else None,
            old_decision='accept_strict_automatic_annotation_candidate' if captured else None,native_foreground_components=[]))
    if alter_outcomes is not None: alter_outcomes(rows)
    outcomes=write(f['root']/'outcomes.json',dict(schema=p.PREFIX+'native_outcomes',manifest=manifest,records=rows,flags=p.FLAGS))
    result=write(f['root']/'native/result.json',dict(shadow_observer_manifest=manifest,request_sha256=request['sha256']))
    audit=write(f['root']/'native_audit.json',dict(result_sha256=result['sha256'],shadow_observer_outcomes=outcomes))
    failure=write(f['root']/'native/failure.json',{'synthetic_failure':True}) if exit_code else None
    pins={**spec['producer_bindings'],**spec['implementation_bindings']}
    receipt=dict(schema=p.PREFIX+'owned_exit',state='owned_worker_exited',exit_code=exit_code,worker=worker,
        launch_event=launch,manifest=manifest,native_request=request,native_result=result if not exit_code else None,
        native_failure=failure,native_audit=audit if not exit_code else None,outcomes=outcomes if not exit_code else None,
        source_bindings=pins,flags=p.FLAGS)
    return write(f['root']/'owned_exit.json',receipt)


def synthetic_loader(monkeypatch):
    # Patch ONLY the new processor's private test boundary, never frozen globals.
    # The actual frozen inspect_interval still runs on immutable synthetic arrays.
    from .test_visibility_shadow import snapshot, mesh
    from dataclasses import replace
    context=snapshot([mesh('target',2.)])
    def load(value):
        return replace(context, source_bindings=tuple(sorted(value['geometry_bindings'].items()))), dict(synthetic=True)
    monkeypatch.setattr(cpu,'_load',load)


def test_public_event_methods_return_none_and_seal_before_return(fixture):
    pin=journal(fixture)
    ledger={};spec,contexts,events,manifest=p.read_journal(pin,ledger)
    assert list(events)==['original','generated'] and len(contexts)==1
    assert events['original']['payload']['context_id'] is None
    assert events['generated']['seal']['sha256'] in ledger.values()
    assert all(Path(path).is_file() for path in ledger)
    p.verify(ledger)


def test_end_to_end_after_exit_uses_frozen_predictor_and_retains_controls(fixture,monkeypatch):
    manifest=journal(fixture); receipt=exit_fixture(fixture,manifest); synthetic_loader(monkeypatch)
    result=cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu')
    report=p.read_pin(result)
    assert report['prediction_counts']=={'not_run_original_control':1,'unknown':1}
    assert report['candidate_count']==2 and not report['failures']
    assert report['skip_adopted'] is report['ranking_adopted'] is report['native_launched'] is False
    sealed=p.read_pin(report['predictions'])
    assert sealed['native_outcomes_joined'] is False
    p.verify(report['source_and_evidence_bindings'])


@pytest.mark.parametrize('kind,code,status', [('geometry_hold','existing_geometry_hold','not_run_geometry_hold'),
    ('observer_error','pose_unavailable','observer_error')])
def test_all_native_holds_and_errors_preserved(fixture,monkeypatch,kind,code,status):
    manifest=journal(fixture,second=dict(kind=kind,candidate_id='generated',code=code))
    receipt=exit_fixture(fixture,manifest)
    monkeypatch.setattr(cpu,'_load',lambda _:pytest.fail('No prediction for held/error candidate'))
    report=p.read_pin(cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu'))
    assert report['candidate_count']==2 and report['prediction_counts'][status]==1


def test_missing_candidate_is_not_omitted(fixture,monkeypatch):
    manifest=journal(fixture,missing=True);receipt=exit_fixture(fixture,manifest)
    report=p.read_pin(cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu'))
    sealed=p.read_pin(report['predictions'])
    assert sealed['predictions'][1]['code']=='missing_event'


@pytest.mark.parametrize('field',['rgb','depth','validity','components','old_decision','visibility_evidence'])
def test_sensor_or_outcome_payload_rejected_and_sanitized(fixture,field):
    event=fixture['event']('generated',7);event[field]=[[1,2,3]]
    pin=journal(fixture,second=event)
    _,_,events,_=p.read_journal(pin,{})
    assert events['generated']['payload']==dict(kind='observer_error',candidate_id='generated',code='invalid_event')
    assert field.encode() not in (fixture['root']/'journal/event_000002.json').read_bytes()


@pytest.mark.parametrize('where',['calibration','context'])
def test_nested_sensor_fields_rejected(fixture,where):
    if where=='context':
        value=deepcopy(fixture['context']);value['rgb']=[1]
        with pytest.raises(ValueError):p.validate_context(value)
    else:
        value=fixture['event']('generated');value['calibration']['labels']=[1]
        with pytest.raises(ValueError):p.validate_candidate(value,fixture['spec']['candidates'][1],{'plant':fixture['context']})


def test_duplicate_json_nonfinite_and_mutable_input_rejected():
    for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',bytearray(b'{}')]:
        with pytest.raises(ValueError):p.decode(raw)


def test_tampered_event_and_unknown_producer_rejected(fixture):
    pin=journal(fixture)
    path=fixture['root']/'journal/event_000002.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):p.read_journal(pin,{})
    bad=deepcopy(fixture['spec']);bad['producer_module']='sim_data.other.collector'
    with pytest.raises(ValueError):p.validate_spec(bad)


@pytest.mark.parametrize('change',['state','pid','module','exit_bool','source_pin'])
def test_running_or_foreign_exit_cannot_reach_predictor(fixture,monkeypatch,change):
    manifest=journal(fixture);pin=exit_fixture(fixture,manifest);r=p.read_pin(pin)
    if change=='state':r['state']='owned_worker_running'
    elif change=='pid':r['worker']['pid']+=1
    elif change=='module':r['worker']['command'][2]='other'
    elif change=='exit_bool':r['exit_code']=False
    else:r['source_bindings'].pop(next(iter(r['source_bindings'])))
    wrong=write(fixture['root']/'bad_exit.json',r)
    monkeypatch.setattr(cpu,'_load',lambda _:pytest.fail('Must not reach geometry before valid exit'))
    with pytest.raises(ValueError):cpu.process(wrong['path'],exit_sha256=wrong['sha256'],output=fixture['root']/'cpu')
    assert not (fixture['root']/'cpu').exists()


def test_outcomes_are_read_only_after_all_predictions_are_sealed(fixture,monkeypatch):
    manifest=journal(fixture);pin=exit_fixture(fixture,manifest);synthetic_loader(monkeypatch)
    original=p.read_pin
    def guarded(value,ledger=None):
        if value['path']==str(fixture['root']/'outcomes.json'):
            assert (fixture['root']/'cpu/predictions.json').is_file()
            sealed=p.decode((fixture['root']/'cpu/predictions.json').read_bytes())
            assert len(sealed['predictions'])==2 and not sealed['native_outcomes_joined']
        return original(value,ledger)
    monkeypatch.setattr(p,'read_pin',guarded)
    cpu.process(pin['path'],exit_sha256=pin['sha256'],output=fixture['root']/'cpu')


def test_prediction_exception_retained_no_tuning_or_retry(fixture,monkeypatch):
    manifest=journal(fixture);pin=exit_fixture(fixture,manifest);synthetic_loader(monkeypatch)
    def fail(*args):raise ValueError('synthetic unsupported input')
    monkeypatch.setattr(cpu,'_predict',fail)
    report=p.read_pin(cpu.process(pin['path'],exit_sha256=pin['sha256'],output=fixture['root']/'cpu'))
    assert report['prediction_counts']['prediction_error_unknown']==1 and len(report['failures'])==1


def test_nonzero_owned_exit_preserves_candidates_without_outcome_join(fixture,monkeypatch):
    manifest=journal(fixture);pin=exit_fixture(fixture,manifest,exit_code=1);synthetic_loader(monkeypatch)
    report=p.read_pin(cpu.process(pin['path'],exit_sha256=pin['sha256'],output=fixture['root']/'cpu'))
    assert report['native_exit_code']==1 and report['candidate_count']==2
    assert all(r['comparison']=='unavailable' for r in report['comparison'])


def test_original_control_cannot_be_tested_against_generated_geometry(fixture):
    value=fixture['event']('original');value['context_id']='plant'
    with pytest.raises(ValueError):p.validate_candidate(value,fixture['spec']['candidates'][0],{'plant':fixture['context']})


def test_geometry_source_final_reread_is_mandatory(fixture,monkeypatch):
    manifest=journal(fixture);pin=exit_fixture(fixture,manifest);synthetic_loader(monkeypatch)
    original=cpu._predict
    def changed(*args):
        result=original(*args)
        (fixture['root']/'variant/manifest.json').write_bytes(b'changed')
        return result
    monkeypatch.setattr(cpu,'_predict',changed)
    with pytest.raises(ValueError):cpu.process(pin['path'],exit_sha256=pin['sha256'],output=fixture['root']/'cpu')
    assert (fixture['root']/'cpu/failure.json').exists() and not (fixture['root']/'cpu/report.json').exists()


def test_imports_do_not_load_usd_numpy_predictor_or_native():
    code="import sys; sys.path.insert(0,sys.argv[1]); from sim_data.native_capture_v5 import shadow_observer_v1,shadow_observer_cpu_v1; assert not any(k.split('.')[0] in ('pxr','omni','isaacsim','numpy') for k in sys.modules); assert 'sim_data.native_capture_v5.visibility_shadow' not in sys.modules"
    result=subprocess.run([sys.executable,'-B','-c',code,str(Path(__file__).resolve().parents[2])],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
