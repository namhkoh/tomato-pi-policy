"""Preparation preflight and launcher handoff tests; no native success claims."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from . import prepare as module
from .test_prepare import inputs


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return path


@pytest.fixture
def source(tmp_path):
    package=tmp_path/'package';package.mkdir()
    capture=tmp_path/'old_capture';capture.mkdir()
    plan=write(tmp_path/'plans/original/plan.json',{'package':str(package)})
    asset=write(package/'house/base.json',{})
    bank=dict(checkpoint=str(tmp_path/'checkpoint'),source_bindings={str(asset):module.sha256(asset)},
        entries=[dict(source_capture=str(capture),source_collection_plan=str(plan),
                      source_collection_plan_sha256=module.sha256(plan))])
    schedule={'qualification_output_root':str(tmp_path/'sensor')}
    return bank,schedule,plan,package,capture


@pytest.mark.parametrize('target',['package','plan','capture','checkpoint','sensor','asset_dir','ancestor','root'])
def test_reject_source_root_overlap(source,tmp_path,target):
    bank,schedule,plan,package,capture=source
    output={'package':package/'new_job','plan':plan.parent/'new_job','capture':capture/'new_job',
        'checkpoint':Path(bank['checkpoint'])/'new_job','sensor':Path(schedule['qualification_output_root'])/'new_job',
        'asset_dir':package/'house/new_job','ancestor':tmp_path,'root':Path(tmp_path.anchor)}[target]
    with pytest.raises(ValueError):module.validate_destination(output,bank,schedule)
    assert not (package/'new_job').exists()


def test_allow_fresh_disjoint_destination(source,tmp_path):
    bank,schedule,*_=source;out=tmp_path/'collection/new_job'
    assert module.validate_destination(out,bank,schedule)==out.resolve()
    assert not out.exists()


def test_reject_existing_destination(source,tmp_path):
    bank,schedule,*_=source;out=tmp_path/'collection/new_job';out.mkdir(parents=True)
    with pytest.raises(ValueError):module.validate_destination(out,bank,schedule)


def test_reject_changed_source_plan(source,tmp_path):
    bank,schedule,plan,*_=source;write(plan,{'package':str(tmp_path/'elsewhere')})
    with pytest.raises(ValueError):module.validate_destination(tmp_path/'new_job',bank,schedule)


def test_sensor_preflight_forwards_exact_anchor(tmp_path,monkeypatch):
    from .. import generated_capture, capture_sensor
    proof=tmp_path/'native';write(proof/'result.json',{})
    attempt=dict(qualification_output=str(proof),source_capture='the_capture',source_sample='sample_7',
                 source_row={'target_id':'seed7/SubStem_2'},calibration={'source':'exact'})
    seen=[]
    monkeypatch.setattr(capture_sensor,'calibration_for_native_resolution',lambda cal,res:dict(cal,native=list(res)))
    monkeypatch.setattr(generated_capture,'verify_sensor_prerequisite',lambda request:seen.append(request) or {'bindings':{'native':'pin'}})
    assert module.verify_fresh_anchor(attempt)=={'bindings':{'native':'pin'}}
    assert seen==[dict(prerequisite_directory=str(proof),source_capture='the_capture',source_sample='sample_7',
        source_row=attempt['source_row'],expected_calibration={'source':'exact','native':[1696,816]})]


@pytest.mark.parametrize('failure',[False,True])
def test_sensor_preflight_requires_complete_unfailed_pair(tmp_path,failure):
    proof=tmp_path/'native'
    if failure:
        write(proof/'result.json',{});write(proof/'failure.json',{})
    with pytest.raises(ValueError):module.verify_fresh_anchor({'qualification_output':str(proof)})


def test_prepare_rejects_bad_proof_before_creating_output(inputs,source,tmp_path,monkeypatch):
    bank,job,attempt=inputs
    source_bank,schedule,plan,*_=source
    for entry in bank['entries']:
        entry.update(source_capture=source_bank['entries'][0]['source_capture'],source_collection_plan=str(plan),
                     source_collection_plan_sha256=module.sha256(plan))
    job.update(job_id='donor_001',source_collection_plan=str(plan),source_collection_plan_sha256=module.sha256(plan))
    attempt=dict(bank['entries'][2],attempt_id='a0',qualification_output=str(tmp_path/'sensor/a0'))
    job.update(anchor=attempt,fallback_anchors=[])
    bank.update(checkpoint=source_bank['checkpoint'],source_bindings=source_bank['source_bindings'])
    bank_path=write(tmp_path/'bank.json',bank)
    schedule.update(jobs=[job],source_reference_bank=str(bank_path),source_reference_bank_sha256=module.sha256(bank_path))
    schedule_path=write(tmp_path/'schedule.json',schedule)
    monkeypatch.setattr(module.schedule_api,'check',lambda s:True)
    def refuse(a):
        raise ValueError('native payload missing')
    monkeypatch.setattr(module,'verify_fresh_anchor',refuse)
    out=tmp_path/'collection/new_job'
    with pytest.raises(ValueError,match='native payload missing'):
        module.prepare(schedule_path,'donor_001','a0',123,out)
    assert not out.exists()


@pytest.fixture
def handoff(tmp_path,monkeypatch):
    from .. import native_multitarget_plan
    out=tmp_path/'job'
    proof=write(tmp_path/'proof.json',{})
    source_plan=write(tmp_path/'source.json',{})
    job=dict(job_id='donor_001',source_collection_plan=str(source_plan),source_collection_plan_sha256=module.sha256(source_plan),
        compatibility_group='group',source_family='seed7_full')
    reference=dict(id='ref0',source_capture=str(tmp_path/'capture'),source_sample='sample_1',
        source_family='seed7_full',source_collection_plan=str(source_plan),compatibility_group='group',
        target_id='seed7_full/SubStem_1',source_row={'component_id':'SubStem_1'})
    attempt=dict(reference,attempt_id='anchor0',qualification_output=str(tmp_path/'proof'))
    bank=write(tmp_path/'bank.json',{'entries':[reference]})
    schedule=write(tmp_path/'schedule.json',dict(source_reference_bank=str(bank),source_reference_bank_sha256=module.sha256(bank)))
    bindings={str(proof):module.sha256(proof)}
    base=write(out/'SubStem_1_base.json',dict(reference,prerequisite_directory=attempt['qualification_output'],
        generated_row={'target_id':'variant/SubStem_1','component_id':'SubStem_1'}))
    plan=write(out/'plan.json',dict(source_family='seed7_full',split='train',resolution=[1696,816],
        prerequisite_bindings=bindings,views_per_target=6,maximum_native_frames=6,anchor_pair_plan=str(base),
        target_cases=[dict(base_pair_plan=str(base),base_pair_plan_sha256=module.sha256(base),
            target_id='variant/SubStem_1',conservative_view_cap_group=reference['target_id'],views=[{}]*6)]))
    monkeypatch.setattr(module,'verify_fresh_anchor',lambda a:{'bindings':bindings})
    # Native plan validation has its own tests; this fixture tests added lineage.
    monkeypatch.setattr(native_multitarget_plan,'check',lambda p,replay_geometry:True)
    request=dict(schedule=str(schedule),schedule_sha256=module.sha256(schedule),**job,
        attempt_id='anchor0',seed=123,max_targets=3,reference_ids=['ref0'],training_approved=False,
        implementation_bindings={str(Path(module.__file__).resolve()):module.sha256(module.__file__)},
        sensor_prerequisite_bindings=bindings)
    result=dict(state='generated_reference_bank_job_prepared_pending_native_capture',**request,
        plan_path=str(plan),plan_sha256=module.sha256(plan),source_cap_reset=False,native_capture_started=False,
        physical_execution_approved=False,targets=[reference['target_id']],maximum_native_frames=6)
    write(out/'prepare_request.json',request);write(out/'prepared.json',result)
    return out,schedule,job,attempt,result


def test_valid_handoff(handoff):
    out,schedule,job,attempt,result=handoff
    assert module.verify_prepared(out,schedule,job,attempt,123)==result


@pytest.mark.parametrize('key,value',[
    ('state','preparing'),('job_id','wrong'),('attempt_id','wrong'),('seed',124),('max_targets',12),
    ('source_collection_plan','wrong.json'),('source_collection_plan_sha256','b'*64),
    ('compatibility_group','other'),('schedule_sha256','b'*64),('reference_ids',['wrong']),
    ('training_approved',True),('source_cap_reset',True),('native_capture_started',True),
    ('physical_execution_approved',True),('plan_sha256','b'*64),('implementation_bindings',{})])
def test_handoff_rejects_changed_receipt(handoff,key,value):
    out,schedule,job,attempt,result=handoff;result[key]=value
    write(out/'prepared.json',result)
    with pytest.raises(ValueError):module.verify_prepared(out,schedule,job,attempt,123)


def test_handoff_rejects_partial_output(handoff):
    out,schedule,job,attempt,result=handoff
    (out/'prepared.json').unlink()
    with pytest.raises(FileNotFoundError):module.verify_prepared(out,schedule,job,attempt,123)


def test_handoff_rejects_changed_plan(handoff):
    out,schedule,job,attempt,result=handoff;write(out/'plan.json',{'changed':True})
    with pytest.raises(ValueError):module.verify_prepared(out,schedule,job,attempt,123)


@pytest.mark.parametrize('key,value',[
    ('source_capture','another_capture'),('source_sample','other_sample'),
    ('source_collection_plan','another.json'),('source_family','seed41_full'),
    ('source_row',{'component_id':'wrong'}),('prerequisite_directory','another_proof')])
def test_rehashed_foreign_base_does_not_match_schedule(handoff,key,value):
    out,schedule,job,attempt,result=handoff
    plan=module.read_json(out/'plan.json');case=plan['target_cases'][0]
    base=module.read_json(case['base_pair_plan']);base[key]=value
    write(case['base_pair_plan'],base);case['base_pair_plan_sha256']=module.sha256(case['base_pair_plan'])
    write(out/'plan.json',plan);result['plan_sha256']=module.sha256(out/'plan.json')
    write(out/'prepared.json',result)
    with pytest.raises(ValueError):module.verify_prepared(out,schedule,job,attempt,123)


def test_foreign_proof_rejected_even_if_hashes_valid(handoff,tmp_path):
    out,schedule,job,attempt,result=handoff
    proof=write(tmp_path/'other_proof.json',{})
    result['sensor_prerequisite_bindings']={str(proof):module.sha256(proof)}
    request=module.read_json(out/'prepare_request.json')
    request['sensor_prerequisite_bindings']=result['sensor_prerequisite_bindings']
    write(out/'prepare_request.json',request);write(out/'prepared.json',result)
    with pytest.raises(ValueError):module.verify_prepared(out,schedule,job,attempt,123)
