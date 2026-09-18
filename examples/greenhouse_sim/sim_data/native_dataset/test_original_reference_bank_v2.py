"""CPU producer/chain/review checks; synthetic fixtures are never new captures."""
from copy import deepcopy
from pathlib import Path

import pytest

from . import original_reference_bank_v2 as rb
from .test_original_inventory import original as make_original
from .test_original_reference_bank import reference as add_reference
from .test_original_inventory_v2 import serial
from .test_original_phase_inventory import phase
from .test_inventory import write, pin, json_at


@pytest.fixture
def original(tmp_path, monkeypatch):
    return add_reference.__wrapped__(make_original.__wrapped__(tmp_path, monkeypatch))


def source_pins(f):
    return [f['serial']['original']['pin'], f['serial']['captures'][0]['pin'], f['batches'][0]['pin']]


def review(f, bank, decision=rb.SUPPORTED_REVIEW, name='positive_review.json'):
    entry = next(e for e in bank['entries'] if e['native_observation'] is not None)
    proof = entry['native_observation']
    paths = [Path(proof[k]['path']) for k in ('sample', 'rgb', 'label', 'result', 'postexit_audit', 'launcher', 'trace')]
    doc = dict(schema=rb.REVIEW_SCHEMA, reviewer='UNIT ONLY', visual_review_performed=True,
        human_review_performed=False, training_approved=False, training_count_increment=0,
        target_id=entry['target_id'], decision=decision, observations='Synthetic review fixture, not native visual evidence',
        source_bindings={str(p): pin(p) for p in paths})
    path = f['root'] / name
    return rb.ReviewPin(str(path), write(path, doc))


@pytest.mark.parametrize('index,producer,planned', [(0, 'v5_original_pilot.v1', 1),
    (1, 'serial39_original.v2', 16), (2, 'serial_phases_original.v1', 16)])
def test_each_named_producer_retains_both_original_chains_and_holds(phase, index, producer, planned):
    chosen = source_pins(phase)[index]
    root = phase['serial']['original']['root']
    before = {str(p): pin(p) for p in root.rglob('*') if p.is_file()}
    out = rb.build_bank([chosen])
    assert out['schema'] == rb.SCHEMA and out['inputs']['captures'][0]['producer'] == producer
    assert out['counts']['planned_cases'] == planned and out['counts']['captured_cases'] == 1
    assert out['counts']['reference_ready_cases'] == 0
    captured = next(e for e in out['entries'] if e['native_observation'])
    assert captured['launcher_producer'] == captured['native_observation']['launcher_producer'] == producer
    assert captured['pose_prior']['calibration']['resolution'] == [848, 408]
    assert captured['native_observation']['calibration']['resolution'] == [1696, 816]
    assert captured['scene_authority']['actual_lighting']['dome_intensity'] == 6000
    assert captured['pose_prior']['declaration']['prior_lighting']['dome_intensity'] == 1200
    assert all(out[k] == v for k, v in rb._FLAGS.items())
    assert not captured['reference_status']['usable_as_original_reference']
    assert before == {str(p): pin(p) for p in root.rglob('*') if p.is_file()}


def test_phase_reference_requires_exact_review_and_replays_when_verified(phase):
    chosen = phase['batches'][0]['pin']
    unreviewed = rb.build_bank([chosen])
    positive = review(phase, unreviewed)
    path = phase['root'] / 'published/bank.json'
    receipt = rb.write_bank(path, [chosen], reviews=[positive])
    bank = json_at(path)
    assert bank['counts']['reference_ready_cases'] == 1
    entry = next(e for e in bank['entries'] if e['native_observation'])
    proof = rb.verify_anchor(path, bank_sha256=receipt['sha256'], entry_id=entry['id'])
    assert proof['schema'] == rb.PROOF_SCHEMA and not proof['training_approved']
    assert proof['native_observation']['launcher_producer'] == 'serial_phases_original.v1'
    assert not proof['paired_848_1696_proof'] and not proof['legacy_sensor_prerequisite_satisfied']
    with pytest.raises(ValueError):
        rb.v1.check_bank(bank)
    with pytest.raises(ValueError):
        rb.verify_anchor(path, bank_sha256='0'*64, entry_id=entry['id'])
    with pytest.raises(ValueError):
        rb.write_bank(path, [chosen], reviews=[positive])


def test_conflicting_visual_history_withholds_phase_reference(phase):
    chosen = phase['batches'][0]['pin']
    bank = rb.build_bank([chosen])
    positive = review(phase, bank)
    negative = review(phase, bank, 'hold_for_review', 'negative_review.json')
    out = rb.build_bank([chosen], reviews=[positive, negative])
    assert out['counts']['reference_ready_cases'] == 0
    entry = next(e for e in out['entries'] if e['native_observation'])
    assert entry['native_decision'] == rb.inv.STRICT and len(entry['visual_reviews']) == 2


def test_mixed_producers_use_distinct_schema_and_exact_replay(phase):
    out = rb.build_bank(source_pins(phase))
    assert out['counts']['captured_cases'] == 3 and out['counts']['planned_cases'] == 33
    assert len({p['producer'] for p in out['inputs']['captures']}) == 3
    assert rb.check_bank(out) == out
    altered = deepcopy(out)
    altered['counts']['reference_ready_cases'] = 3
    altered['sha256'] = rb.inv._hash({k: v for k, v in altered.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='differs from independently replayed'):
        rb.check_bank(altered)


def test_phase_receipt_cannot_be_relabelled_serial39(phase):
    bank = rb.build_bank([phase['batches'][0]['pin']])
    bank['inputs']['captures'][0]['producer'] = 'serial39_original.v2'
    with pytest.raises(ValueError):
        rb.check_bank(bank)
