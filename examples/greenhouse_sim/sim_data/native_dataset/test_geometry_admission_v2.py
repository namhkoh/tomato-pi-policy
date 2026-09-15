# Synthetic pinned metadata only; no file/native I/O or real qualifications.
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from . import geometry_admission_v2 as a
from .test_morphology_frame_v2 import geometry
from .test_provisional import coverage, inventory, row, sealed


def pin(value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return a.PinnedJSON(raw, sha256(raw).hexdigest())


def fixture(variants=1, *, leaf_counts=None, deltas=None):
    splits = {f'donor{i}': 'train' if i < 16 else 'validation' if i < 20 else 'test' for i in range(24)}
    family, target = 'donor0', 'donor0/stem'
    views, records, packets = [], [], {}
    for index in range(variants):
        if leaf_counts is not None:
            family, target = f'donor{index}', f'donor{index}/stem'
        directory, plan = f'output/{index}', 'original_plan.json'
        qsha = a.digest('qualification' + str(index)); source_manifest = a.digest('original_manifest')
        extracted = []
        for kind in ('original', 'generated'):
            q = geometry()
            if leaf_counts is not None:
                q['leaves'] = q['leaves'][:leaf_counts[index]]
            if kind == 'generated':
                q['leaves'][0]['centroid'][1] += deltas[index] if deltas is not None else .03 * (index+1)
            identity = dict(kind=kind, source_family=family, source_target=target,
                            manifest_sha256=source_manifest if kind == 'original' else a.digest(directory),
                            mesh_sha256={'stem': a.digest(kind+str(index) if kind == 'generated' else kind)})
            extracted.append(dict(context_id='actual-output:' + a.digest(identity), kind=kind,
                source_family=family, source_target=target, identity=identity, input_geometry=q,
                input_geometry_sha256=a.digest(q), target_id=target if kind == 'original' else directory+'/stem',
                split='train'))
        for item in extracted:
            item['source_context_id'] = extracted[0]['context_id']
        packet = pin(sealed(dict(schema='greenhouse.actual_output_morphology.v1', catalogue_replayed=True,
            directory=directory, frozen_splits=splits, frozen_splits_sha256=a.digest(splits), records=extracted,
            bindings={directory+'/qualification.json': qsha, plan: a.digest(plan)},
            code_bindings={'extractor.py': a.digest('code')})))
        path = directory+'/extracted.json'; packets[path] = packet
        sample = f'sample-{index}'
        for item in extracted:
            records.append(dict(context_id=item['context_id'], source_target=target, kind=item['kind'],
                extracted_packet=path, extracted_packet_sha256=packet.sha256,
                input_geometry_sha256=item['input_geometry_sha256'],
                associated_sample_ids=[sample] if item['kind'] == 'generated' else [],
                frame_v2=a.frame.describe(item['input_geometry'])))
        views.append(row(sample, family=family, target=target, target_id=directory+'/stem', provenance=dict(
            sample_sha256=a.digest(sample), lineage=dict(source_collection_plan=plan, source_plan_sha256=a.digest(plan),
            frozen_family_assignments_sha256=a.digest(splits), generated_qualification_sha256=qsha,
            source_manifest_sha256=source_manifest,
            generated_component_asset_sha256=extracted[1]['identity']['mesh_sha256']['stem']))))
    inv = inventory(views)
    report = dict(schema='greenhouse.observed_actual_morphology_snapshot.v1', inventory_sha256=inv['sha256'],
        threshold_selected=None, qualified_geometry=False, training_approved=False, counts=dict(records=len(records)),
        records=records, bindings={'inventory.json': pin(inv).sha256})
    return dict(inventory=pin(inv), descriptor_report=pin(report), packets=packets,
                image_graph=pin(coverage(inv)), frozen_splits=splits)


def qualified(data):
    inv, report = data['inventory'].read(), data['descriptor_report'].read()
    reviews, bindings, keys = [], {}, []
    for item in report['records']:
        if item['kind'] != 'generated':
            continue
        view = next(r for r in inv['records'] if r['sample_id'] == item['associated_sample_ids'][0])
        original = next(r for r in report['records'] if r['kind'] == 'original' and r['source_target'] == item['source_target'])
        artifacts = {role: dict(path=view['sample_id']+'/'+role, sha256=a.digest(view['sample_id']+role)) for role in a.ARTIFACTS}
        artifacts['generated_rgb']['sha256'] = view['encoded_rgb_sha256']
        bindings.update({v['path']: v['sha256'] for v in artifacts.values()})
        keys.append(item['context_id'])
        reviews.append(dict(context_id=item['context_id'], source_context_id=original['context_id'],
            descriptor_sha256=item['frame_v2']['descriptor_sha256'],
            source_descriptor_sha256=original['frame_v2']['descriptor_sha256'], sample_id=view['sample_id'],
            sample_sha256=view['provenance']['sample_sha256'], annotation_evidence_id=view['annotation_review']['evidence_id'],
            generated_qualification_sha256=view['provenance']['lineage']['generated_qualification_sha256'],
            artifacts=artifacts, decision='joint_visual_geometry_distinct'))
    data['reviews'] = pin(dict(schema=a.REVIEW_SCHEMA, bindings=bindings, contexts=reviews, overlays=[]))
    data['nuisance'] = pin(dict(schema='greenhouse.actual_morphology_frame_v2_evaluation.v1',
        descriptor_schema=a.frame.SCHEMA, frame_version=a.frame.FRAME_VERSION,
        source_bindings={'source.json': a.digest('source')}, output_bindings={'controls.json': a.digest('controls')},
        control_definition=dict(numerical_tolerance=1e-9, uniform_scales=[.4, 1], translation=[1, 2, 3],
            leaf_order='reversed', irrelevant_names='renamed', rotations=[{'synthetic': True}]),
        summary=dict(controls=dict(synthetic=dict(comparisons=2, passes_at_1e_9=2, holds=0,
                                                 selection_changes=0, max_distance=1e-12)))))
    data['controls'] = pin(dict(schema=a.CONTROL_SCHEMA, frame_version=a.frame.FRAME_VERSION, metric=a.METRIC,
        threshold=.01, qualification_status='externally_qualified_exact_contexts', context_ids=sorted(keys),
        positive_context_ids=sorted(keys), descriptor_report_sha256=data['descriptor_report'].sha256,
        inventory_sha256=inv['sha256'], image_graph_sha256=data['image_graph'].read()['sha256'],
        reviews_sha256=data['reviews'].sha256, nuisance_sha256=data['nuisance'].sha256,
        frozen_splits_sha256=a.digest(data['frozen_splits'])))
    return data


def change(data, field, mutate, *, bind=False):
    value = data[field].read(); mutate(value); data[field] = pin(value)
    if bind:
        control = data['controls'].read(); control[field+'_sha256'] = data[field].sha256
        data['controls'] = pin(control)


def test_default_merges_all_variants_no_qualification_or_donor_inflation():
    result = a.adapt_contexts(**fixture(2))
    assert len(set(result['groups'].values())) == 1
    assert not result['additional_budget_context_ids'] and result['geometry']['threshold'] is None
    assert a.Counter(result['frozen_splits'].values()) == dict(train=16, validation=4, test=4)
    assert result['max_views_per_group'] == 12 and result['train_goal'] == 20000
    assert not result['training_approved'] and not result['calibration_validated']
    assert not result['source_cap_reset'] and not result['admission_performed']


def test_explicit_synthetic_controls_unlock_only_reviewed_contexts_not_release():
    result = a.adapt_contexts(**qualified(fixture(2)))
    assert len(result['additional_budget_context_ids']) == 2
    assert len(set(result['groups'].values())) == 3
    assert result['eligible_sample_ids'] == ['sample-0', 'sample-1']
    assert result['geometry']['comparisons_complete'] and result['geometry']['compared_pairs'] == 3
    assert all(value is False for key, value in result['scope'].items() if key != 'new_biological_families')


def test_input_and_return_mutation_do_not_change_pins_or_later_results():
    data = qualified(fixture()); before = deepcopy(data)
    result = a.adapt_contexts(**data); expected = deepcopy(result)
    result['contexts'].clear(); result['frozen_splits'].clear(); result['baseline_selection']['selected'].clear()
    assert data == before and a.adapt_contexts(**data) == expected
    data['reviews'] = a.PinnedJSON(data['reviews'].data+b' ', data['reviews'].sha256)
    with pytest.raises(ValueError, match='Changed pinned'):a.adapt_contexts(**data)


@pytest.mark.parametrize('field', ['controls', 'nuisance', 'reviews'])
def test_missing_qualification_evidence_never_unlocks(field):
    data = qualified(fixture()); data[field] = None
    if field == 'controls':
        assert not a.adapt_contexts(**data)['additional_budget_context_ids']
    else:
        with pytest.raises(ValueError):a.adapt_contexts(**data)


@pytest.mark.parametrize('field', ['descriptor_report_sha256', 'inventory_sha256', 'image_graph_sha256',
                                   'reviews_sha256', 'nuisance_sha256', 'frozen_splits_sha256'])
def test_stale_control_bindings_fail(field):
    data = qualified(fixture()); change(data, 'controls', lambda v:v.update({field: '0'*64}))
    with pytest.raises(ValueError, match='Stale'):a.adapt_contexts(**data)


@pytest.mark.parametrize('value', [None, True, -1, '0.01'])
def test_no_invented_threshold(value):
    data = qualified(fixture()); change(data, 'controls', lambda v:v.update(threshold=value))
    with pytest.raises(ValueError, match='threshold'):a.adapt_contexts(**data)


@pytest.mark.parametrize('field', ['source_plan_sha256', 'generated_qualification_sha256',
    'source_manifest_sha256', 'generated_component_asset_sha256', 'frozen_family_assignments_sha256'])
def test_native_output_lineage_is_not_just_matching_target_names(field):
    data = fixture(); inv = data['inventory'].read()
    inv['records'][0]['provenance']['lineage'][field] = '0'*64
    inv = sealed(inv); data['inventory'] = pin(inv); data['image_graph'] = pin(coverage(inv))
    change(data, 'descriptor_report', lambda v:v.update(inventory_sha256=inv['sha256'],
                                                       bindings={'inventory.json': data['inventory'].sha256}))
    with pytest.raises(ValueError, match='lineage'):a.adapt_contexts(**data)


def test_changed_descriptor_even_with_resealed_report_rejected():
    data = fixture()
    change(data, 'descriptor_report', lambda v:v['records'][1]['frame_v2']['descriptor']['fixed'].__setitem__(0, 100))
    with pytest.raises(ValueError, match='V2 result'):a.adapt_contexts(**data)


def test_wrong_extracted_packet_pin_rejected():
    data = fixture(); change(data, 'descriptor_report', lambda v:v['records'][0].update(extracted_packet_sha256='0'*64))
    with pytest.raises(ValueError, match='packet pin'):a.adapt_contexts(**data)


@pytest.mark.parametrize('field', ['descriptor_sha256', 'source_descriptor_sha256', 'sample_sha256',
                                  'annotation_evidence_id', 'generated_qualification_sha256'])
def test_stale_review_rejected(field):
    data = qualified(fixture()); change(data, 'reviews', lambda v:v['contexts'][0].update({field:'0'*64}), bind=True)
    with pytest.raises(ValueError, match='bindings mismatch'):a.adapt_contexts(**data)


def test_missing_native_artifact_or_positive_review_rejected():
    data = qualified(fixture()); change(data, 'reviews', lambda v:v['contexts'][0]['artifacts'].pop('native_pair_audit'), bind=True)
    with pytest.raises(ValueError, match='artifacts'):a.adapt_contexts(**data)
    data = qualified(fixture()); change(data, 'controls', lambda v:v.update(positive_context_ids=[]))
    with pytest.raises(ValueError, match='[Pp]ositive'):a.adapt_contexts(**data)


@pytest.mark.parametrize('field,value', [('passes_at_1e_9', 1), ('holds', 1), ('selection_changes', 1),
                                       ('max_distance', .1), ('comparisons', True)])
def test_failed_or_incomplete_nuisance_is_not_qualification(field, value):
    data = qualified(fixture())
    change(data, 'nuisance', lambda v:v['summary']['controls']['synthetic'].update({field:value}), bind=True)
    with pytest.raises(ValueError, match='nuisance'):a.adapt_contexts(**data)


def test_visual_hold_overrides_qualification_and_baseline():
    data = qualified(fixture())
    change(data, 'reviews', lambda v:v['contexts'][0].update(decision='hold'), bind=True)
    with pytest.raises(ValueError, match='[Pp]ositive'):a.adapt_contexts(**data)
    data['controls'] = None
    result = a.adapt_contexts(**data)
    assert not result['eligible_sample_ids'] and not result['baseline_selection']['selected']


def test_control_role_can_supply_evidence_but_cannot_be_selected():
    data = qualified(fixture())
    change(data, 'reviews', lambda v:v['overlays'].append(dict(sample_id='sample-0', evidence_id='role', control_role='control')), bind=True)
    result = a.adapt_contexts(**data)
    assert result['additional_budget_context_ids'] and not result['eligible_sample_ids']


def test_missing_geometry_preserves_original_cap_no_extra_budget():
    data = fixture()
    change(data, 'descriptor_report', lambda v:v.update(records=[], counts=dict(records=0)))
    result = a.adapt_contexts(**data)
    assert len(set(result['groups'].values())) == 1 and result['geometry']['missing_context_ids']
    assert not result['additional_budget_context_ids']


def test_original_split_reservations_cannot_be_replaced_by_observed_subset():
    data = fixture(); data['frozen_splits'] = {'donor0': 'train'}
    with pytest.raises(ValueError, match='reservations'):a.adapt_contexts(**data)


def test_incomplete_image_graph_fails_closed():
    data = fixture(); graph = data['image_graph'].read(); graph['complete'] = False
    data['image_graph'] = pin(sealed(graph))
    with pytest.raises(ValueError, match='comparisons'):a.adapt_contexts(**data)


def test_cross_cardinality_is_merged_not_novelty_or_global_qualification_erasure():
    result = a.adapt_contexts(**qualified(fixture(2, leaf_counts=[1, 2])))
    assert not result['additional_budget_context_ids']
    assert result['geometry']['unresolved_pairs'] == 4
    assert result['geometry']['compared_pairs'] == 2
    assert result['geometry']['unexamined_pairs'] == 0
    assert not result['geometry']['comparisons_complete']
    assert result['geometry']['pair_accounting_complete']
    assert len(result['geometry']['conservative_merge_edges']) == 4
    assert sum(r['qualified_geometry'] for r in result['contexts'].values()) == 2
    assert len(set(result['groups'].values())) == 1


def own_view_review(data, index=1):
    inv = data['inventory'].read(); sample = inv['records'][index]
    sample['provenance']['plan_sha256'] = a.digest('plan-own')
    sample['provenance']['result_sha256'] = a.digest('result-own')
    sample['annotation_review']['evidence_id'] = a.digest('audit-own')
    inv = sealed(inv); data['inventory'] = pin(inv); data['image_graph'] = pin(coverage(inv))
    change(data, 'descriptor_report', lambda v:v.update(inventory_sha256=inv['sha256'],
        bindings={'inventory.json': data['inventory'].sha256}))
    review = data['reviews'].read(); entry = review['contexts'][index]
    entry['decision'] = 'geometry_distinct_plausible'
    entry['annotation_evidence_id'] = sample['annotation_review']['evidence_id']
    entry['artifacts'] = {role: dict(path=sample['sample_id']+'/'+role, sha256=a.digest(role))
                          for role in a.VIEW_ARTIFACTS}
    for role, value in dict(native_plan=sample['provenance']['plan_sha256'],
        native_result=sample['provenance']['result_sha256'], native_audit=sample['annotation_review']['evidence_id'],
        generated_rgb=sample['encoded_rgb_sha256']).items():
        entry['artifacts'][role]['sha256'] = value
    review['bindings'].update({v['path']: v['sha256'] for v in entry['artifacts'].values()})
    data['reviews'] = pin(review)
    control = data['controls'].read()
    control.update(positive_context_ids=[review['contexts'][0]['context_id']],
        inventory_sha256=inv['sha256'], image_graph_sha256=data['image_graph'].read()['sha256'],
        descriptor_report_sha256=data['descriptor_report'].sha256, reviews_sha256=data['reviews'].sha256)
    data['controls'] = pin(control)
    return entry['context_id']


def test_later_variant_uses_own_clear_view_without_an_original_rerender():
    data = qualified(fixture(2)); key = own_view_review(data)
    result = a.adapt_contexts(**data)
    assert key in result['additional_budget_context_ids']
    assert 'original_rgb' not in data['reviews'].read()['contexts'][1]['artifacts']


@pytest.mark.parametrize('role', ['native_plan', 'native_result', 'native_audit'])
def test_own_view_receipts_must_match_actual_inventory_not_just_be_named(role):
    data = qualified(fixture(2)); own_view_review(data)
    def mutate(value):
        item = value['contexts'][1]['artifacts'][role]
        item['sha256'] = '0'*64; value['bindings'][item['path']] = '0'*64
    change(data, 'reviews', mutate, bind=True)
    with pytest.raises(ValueError, match='Own native'):a.adapt_contexts(**data)


def test_ordinary_view_review_cannot_replace_a_paired_positive_control():
    data = qualified(fixture(2)); key = own_view_review(data)
    change(data, 'controls', lambda v:v.update(positive_context_ids=[key]))
    with pytest.raises(ValueError, match='artifacts'):a.adapt_contexts(**data)


def test_unknown_cross_split_context_merge_quarantines_candidate_views():
    data = qualified(fixture()); inv = data['inventory'].read()
    inv['records'].append(row('heldout', family='donor16', split='validation'))
    inv['counts']['captured_rows'] += 1
    inv = sealed(inv); data['inventory'] = pin(inv); data['image_graph'] = pin(coverage(inv))
    change(data, 'descriptor_report', lambda v:v.update(inventory_sha256=inv['sha256'],
        bindings={'inventory.json': data['inventory'].sha256}))
    change(data, 'controls', lambda v:v.update(inventory_sha256=inv['sha256'],
        image_graph_sha256=data['image_graph'].read()['sha256'],
        descriptor_report_sha256=data['descriptor_report'].sha256))
    result = a.adapt_contexts(**data)
    assert not result['eligible_sample_ids'] and not result['additional_budget_context_ids']
    assert result['geometry']['pair_accounting_complete']


def test_diagnostic_set_metric_does_not_silently_inherit_bottleneck_calibration():
    from .morphology_set_v1 import METRIC
    data = qualified(fixture()); change(data, 'controls', lambda v:v.update(metric=METRIC))
    with pytest.raises(ValueError, match='nuisance'):a.adapt_contexts(**data)


def set_mode(data):
    control = data['controls'].read(); definition = data['nuisance'].read()['control_definition']
    definition['rotations'] = [dict(group='synthetic', rotation_index=0)]
    proof = dict(metric=a.leaf_set.METRIC, frame_version=a.frame.FRAME_VERSION, descriptor_schema=a.frame.SCHEMA,
        metric_implementation=dict(path='set.py', sha256=a.digest('set-code')),
        frame_implementation=dict(path='frame.py', sha256=a.digest('frame-code')))
    bound = {item['path']: item['sha256'] for item in (proof['metric_implementation'], proof['frame_implementation'])}
    bound['packet.json'] = a.digest('synthetic-packet')
    base = dict(source_packet='packet.json', source_packet_sha256=bound['packet.json'], source_packet_row=0,
                descriptor_sha256=a.digest('baseline'), input_geometry_sha256=a.digest('geometry'))
    measurements = []
    for scale in definition['uniform_scales']:
        measurements.append(dict(row_id='row', group='synthetic', rotation_index=0, uniform_scale=scale,
            baseline_descriptor_sha256=base['descriptor_sha256'], input_geometry_sha256=a.digest(scale),
            descriptor_sha256=a.digest(('descriptor', scale)), state='descriptor_available_not_calibrated',
            holds=[], selection_unchanged=True, passed=True,
            comparison=dict(metric=a.leaf_set.METRIC, distance=0., fixed_linf=0., leaf_forward_linf=0.,
                            leaf_reverse_linf=0., leaf_hausdorff_linf=0.)))
    nuisance = dict(schema=a.SET_NUISANCE_SCHEMA, metric=a.leaf_set.METRIC,
        frame_version=a.frame.FRAME_VERSION, descriptor_schema=a.frame.SCHEMA, metric_proof=proof,
        source_bindings=bound, frozen_splits_sha256=control['frozen_splits_sha256'],
        threshold_selected=None, calibration_validated=False, qualified_geometry=False, training_approved=False,
        control_definition=definition, control_definition_sha256=a.digest(definition),
        baseline_rows={'row': base}, controls=measurements, measurement_records_sha256=a.digest(measurements),
        summary=dict(actual_rows=1, controls=dict(synthetic=dict(comparisons=2, passes_at_1e_9=2,
            holds=0, selection_changes=0, max_distance=0.))))
    data['nuisance'] = pin(sealed(nuisance))
    described = {r['context_id']: r['frame_v2']['descriptor'] for r in data['descriptor_report'].read()['records']}
    review = data['reviews'].read()
    for entry in review['contexts']:
        entry.update(geometry_metric=a.leaf_set.METRIC, metric_proof_sha256=a.digest(proof),
            distance_to_original=a.leaf_set.compare(described[entry['context_id']],
                                                     described[entry['source_context_id']])['distance'])
    data['reviews'] = pin(review)
    control.update(metric=a.leaf_set.METRIC, metric_proof_sha256=a.digest(proof), threshold=.02,
                   nuisance_sha256=data['nuisance'].sha256, reviews_sha256=data['reviews'].sha256)
    data['controls'] = pin(control)
    return data


def mutate_set_nuisance(data, mutate):
    nuisance = data['nuisance'].read(); mutate(nuisance)
    nuisance['measurement_records_sha256'] = a.digest(nuisance['controls'])
    nuisance['control_definition_sha256'] = a.digest(nuisance['control_definition'])
    data['nuisance'] = pin(sealed(nuisance))
    change(data, 'controls', lambda v:v.update(nuisance_sha256=data['nuisance'].sha256))


def test_set_mode_resolves_mixed_nonempty_cardinalities_without_merge_collapse(monkeypatch):
    data = set_mode(qualified(fixture(2, leaf_counts=[1, 2])))
    original = a.leaf_set.compare; calls = []
    def observed(left, right):
        calls.append((left['leaf_count'], right['leaf_count'])); return original(left, right)
    monkeypatch.setattr(a.leaf_set, 'compare', observed)
    result = a.adapt_contexts(**data)
    assert result['geometry']['metric'] == a.leaf_set.METRIC
    assert result['geometry']['compared_pairs'] == 6 and not result['geometry']['unresolved_pairs']
    assert result['geometry']['comparisons_complete'] and not result['geometry']['conservative_merge_edges']
    assert len(result['additional_budget_context_ids']) == 2 and (1, 2) in calls + [(b,a) for a,b in calls]


@pytest.mark.parametrize('field', ['geometry_metric', 'metric_proof_sha256', 'distance_to_original'])
def test_set_mode_review_must_bind_the_actual_metric_proof_and_recomputed_distance(field):
    data = set_mode(qualified(fixture()))
    change(data, 'reviews', lambda v:v['contexts'][0].update({field: 123 if field == 'distance_to_original' else 'wrong'}), bind=True)
    with pytest.raises(ValueError, match='metric/distance'):a.adapt_contexts(**data)


@pytest.mark.parametrize('damage', ['mode', 'impl', 'missing', 'duplicate', 'score', 'record_mode', 'summary', 'pass'])
def test_set_nuisance_mode_implementation_coverage_and_scores_are_checked_even_when_repinned(damage):
    data = set_mode(qualified(fixture()))
    def mutate(value):
        if damage == 'mode':value['metric'] = a.METRIC
        elif damage == 'impl':value['metric_proof']['metric_implementation']['sha256'] = '0'*64
        elif damage == 'missing':value['controls'].pop()
        elif damage == 'duplicate':value['controls'][1] = deepcopy(value['controls'][0])
        elif damage == 'score':value['controls'][0]['comparison']['distance'] = 1e-12
        elif damage == 'record_mode':value['controls'][0]['comparison']['metric'] = a.METRIC
        elif damage == 'summary':value['summary']['controls']['synthetic']['max_distance'] = 1e-12
        else:value['controls'][0]['passed'] = False
    mutate_set_nuisance(data, mutate)
    with pytest.raises(ValueError):a.adapt_contexts(**data)


def test_implementation_must_be_bound_even_if_new_proof_digest_is_explicit():
    data = set_mode(qualified(fixture()))
    mutate_set_nuisance(data, lambda v:v['metric_proof']['metric_implementation'].update(sha256='0'*64))
    proof_sha = a.digest(data['nuisance'].read()['metric_proof'])
    change(data, 'controls', lambda v:v.update(metric_proof_sha256=proof_sha))
    with pytest.raises(ValueError, match='Unbound metric'):a.adapt_contexts(**data)


def test_set_receipt_does_not_silently_run_bottleneck_mode():
    data = set_mode(qualified(fixture())); change(data, 'controls', lambda v:v.update(metric=a.METRIC))
    with pytest.raises(ValueError, match='nuisance'):a.adapt_contexts(**data)


def test_set_nuisance_canonical_seal_is_checked_beyond_the_outer_byte_pin():
    data = set_mode(qualified(fixture()))
    change(data, 'nuisance', lambda v:v.update(sha256='0'*64), bind=True)
    with pytest.raises(ValueError, match='Stale sealed'):a.adapt_contexts(**data)


@pytest.mark.parametrize('new_mode', [False, True])
def test_novelty_withheld_is_not_a_label_hold_or_measured_equivalence(new_mode):
    data = qualified(fixture(2))
    if new_mode:data = set_mode(data)
    withheld = data['reviews'].read()['contexts'][1]['context_id']
    positive = data['reviews'].read()['contexts'][0]['context_id']
    change(data, 'reviews', lambda v:v['contexts'][1].update(decision='novelty_withheld'), bind=True)
    change(data, 'controls', lambda v:v.update(positive_context_ids=[positive]))
    result = a.adapt_contexts(**data)
    assert result['eligible_sample_ids'] == ['sample-0', 'sample-1']
    assert result['baseline_selection']['counts']['train'] == 2
    assert not result['contexts'][withheld]['qualified_geometry'] and not result['geometry']['equivalence_edges']
    assert result['groups'][withheld] == result['groups'][result['contexts'][withheld]['source_context_id']]
    assert 'explicit_visual_hold' not in result['baseline_selection']['excluded'].get('sample-1', [])


def test_novelty_withhold_cannot_be_relabelled_as_a_positive_control():
    data = set_mode(qualified(fixture()))
    change(data, 'reviews', lambda v:v['contexts'][0].update(decision='novelty_withheld'), bind=True)
    with pytest.raises(ValueError, match='Positive requires'):a.adapt_contexts(**data)


def test_set_mode_does_not_treat_an_unsupported_empty_set_as_infinite_novelty():
    desc = a.frame.describe(geometry())['descriptor']
    empty = deepcopy(desc); empty.update(leaves=[], leaf_count=0)
    assert a._distance(a.leaf_set.METRIC, desc, empty) is None


def test_equivalence_bridge_shares_one_budget_and_unqualified_bridge_cannot_reset_it():
    data = qualified(fixture(3, deltas=[.12, .13, .14]))
    records = data['descriptor_report'].read()['records']
    generated = [r for r in records if r['kind'] == 'generated']
    distance = a.frame.distance(generated[0]['frame_v2']['descriptor'], generated[1]['frame_v2']['descriptor'])
    change(data, 'controls', lambda v:v.update(threshold=distance*1.1))
    result = a.adapt_contexts(**data)
    assert len(result['geometry']['equivalence_edges']) == 2
    assert len(set(result['groups'].values())) == 2  # original plus one transitive generated pool
    middle = generated[1]['context_id']
    change(data, 'reviews', lambda v:next(r for r in v['contexts'] if r['context_id'] == middle).update(decision='hold'), bind=True)
    change(data, 'controls', lambda v:v.update(positive_context_ids=[k for k in v['positive_context_ids'] if k != middle]))
    result = a.adapt_contexts(**data)
    assert not result['additional_budget_context_ids']
    assert len(set(result['groups'].values())) == 1
    assert 'sample-1' not in result['eligible_sample_ids']


def test_visual_hold_propagates_to_exact_rgb_alias_through_underlying_selector():
    data = fixture(2); inv = data['inventory'].read()
    inv['records'][1]['decoded_rgb_sha256'] = inv['records'][0]['decoded_rgb_sha256']
    inv = sealed(inv); data['inventory'] = pin(inv); data['image_graph'] = pin(coverage(inv))
    change(data, 'descriptor_report', lambda v:v.update(inventory_sha256=inv['sha256'],
                                                       bindings={'inventory.json': data['inventory'].sha256}))
    data['reviews'] = pin(dict(schema=a.REVIEW_SCHEMA, contexts=[], overlays=[
        dict(sample_id='sample-0', evidence_id='visual-hold', visual_hold=True)]))
    result = a.adapt_contexts(**data)
    assert not result['eligible_sample_ids']
    assert len(set(result['view_duplicate_groups'].values())) == 1


def test_json_duplicate_keys_and_nonfinite_values_are_not_resealed_away():
    for document in ('{QxQ:1,QxQ:2}', '{QxQ:NaN}', '[]'):
        raw = document.replace('Q', chr(34)).encode()
        with pytest.raises(ValueError):a.PinnedJSON(raw, sha256(raw).hexdigest()).read()


def test_review_artifact_hash_mismatch_and_alias_conflict_fail():
    data = qualified(fixture())
    change(data, 'reviews', lambda v:v['contexts'][0]['artifacts']['visual_review'].update(sha256='0'*64), bind=True)
    with pytest.raises(ValueError, match='artifact'):a.adapt_contexts(**data)
    with pytest.raises(ValueError, match='Conflicting'):
        a._bindings({'a/b': '0'*64, 'a\\b': '1'*64})
