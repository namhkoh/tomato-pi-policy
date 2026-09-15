"""Additional synthetic ordering/I/O/integrity tests for the new observer only."""
from copy import deepcopy

import pytest

from . import shadow_observer_v1 as p
from . import shadow_observer_cpu_v1 as cpu
from .test_shadow_observer_v1 import fixture, journal, exit_fixture, synthetic_loader, write


def test_io_gap_preserves_later_candidate_and_manifest_error(fixture, monkeypatch):
    writer=p.EventWriter(fixture['root']/'journal',p.canonical(fixture['spec']))
    writer.bind_context(p.canonical(fixture['context']))
    original=p.write_json_new
    def fail_first(path,value):
        if path.name=='event_000001.json':raise OSError('synthetic write failure')
        return original(path,value)
    monkeypatch.setattr(p,'write_json_new',fail_first)
    assert writer.record_candidate(p.canonical(fixture['event']('original'))) is None
    assert writer.record_candidate(p.canonical(fixture['event']('generated',7))) is None
    writer.seal()
    _,_,events,manifest=p.read_journal(p.file_pin(fixture['root']/'journal/manifest.json'),{})
    assert len(events)==2 and events['original']['payload']['code']=='event_io_error'
    assert events['generated']['payload']['kind']=='pre_render'


@pytest.mark.parametrize('where',['spec','context','calibration'])
def test_unknown_nested_fields_fail_validation(fixture,where):
    if where=='spec':
        value=deepcopy(fixture['spec']);value['candidate_override']=[]
        with pytest.raises(ValueError):p.validate_spec(value)
    elif where=='context':
        value=deepcopy(fixture['context']);value['geometry_bindings']={'relative.usd':'a'*64}
        with pytest.raises(ValueError):p.validate_context(value)
    else:
        value=fixture['event']('generated');value['calibration']['intrinsics'][0][0]=True
        with pytest.raises(ValueError):p.validate_candidate(value,fixture['spec']['candidates'][1],{'plant':fixture['context']})


def test_integer_zero_cannot_replace_false_flag(fixture):
    value=deepcopy(fixture['spec']);value['flags']['training_approved']=0
    with pytest.raises(ValueError):p.validate_spec(value)


def test_no_source_mutation_or_prediction_during_recording(fixture,monkeypatch):
    before={path:p.sha(path) for path in fixture['spec']['implementation_bindings']}
    monkeypatch.setattr(cpu,'_load',lambda _:pytest.fail('Recording must not load geometry'))
    pin=journal(fixture)
    assert p.read_pin(pin)['flags']==p.FLAGS
    assert before=={path:p.sha(path) for path in before}


@pytest.mark.parametrize('field',['counter','calibration','seal','omission'])
def test_outcome_chronology_pose_or_inventory_tamper_rejected_after_prediction_seal(fixture,monkeypatch,field):
    manifest=journal(fixture)
    def change(rows):
        if field=='counter':rows[1]['first_render_request_index']+=1
        elif field=='calibration':rows[1]['calibration_sha256']='a'*64
        elif field=='seal':rows[1]['event_seal']=rows[0]['event_seal']
        else:rows.pop()
    receipt=exit_fixture(fixture,manifest,alter_outcomes=change);synthetic_loader(monkeypatch)
    with pytest.raises(ValueError):cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu')
    assert (fixture['root']/'cpu/predictions.json').exists()
    assert (fixture['root']/'cpu/failure.json').exists()
    assert not (fixture['root']/'cpu/report.json').exists()


def test_source_tamper_cannot_reach_geometry(fixture,monkeypatch):
    manifest=journal(fixture);receipt=exit_fixture(fixture,manifest)
    (fixture['root']/'variant/manifest.json').write_bytes(b'changed before processor')
    monkeypatch.setattr(cpu,'_load',lambda _:pytest.fail('Source integrity check must run first'))
    with pytest.raises(ValueError):cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu')


def test_manifest_candidate_omission_rejected(fixture):
    pin=journal(fixture); value=p.read_pin(pin);value['candidates'].pop()
    changed=write(fixture['root']/'journal/bad_manifest.json',value)
    with pytest.raises(ValueError):p.read_journal(changed,{})


def test_extra_attempts_never_expand_registered_inventory(fixture):
    writer=p.EventWriter(fixture['root']/'journal',p.canonical(fixture['spec']))
    writer.bind_context(p.canonical(fixture['context']))
    for name in ('original','generated','extra'):
        writer.record_candidate(p.canonical(fixture['event'](name)))
    writer.seal()
    _,_,events,m=p.read_journal(p.file_pin(fixture['root']/'journal/manifest.json'),{})
    assert len(events)==2 and m['faults']==['extra_or_late_candidate']


def test_create_only_journal_and_cpu_destination(fixture,monkeypatch):
    manifest=journal(fixture)
    with pytest.raises(ValueError):p.EventWriter(fixture['root']/'journal',p.canonical(fixture['spec']))
    receipt=exit_fixture(fixture,manifest);synthetic_loader(monkeypatch)
    cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu')
    with pytest.raises(ValueError):cpu.process(receipt['path'],exit_sha256=receipt['sha256'],output=fixture['root']/'cpu')
