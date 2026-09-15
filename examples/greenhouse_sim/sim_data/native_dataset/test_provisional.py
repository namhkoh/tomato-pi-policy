# Synthetic accounting/coverage tests; no capture, image I/O or audit proof.
from copy import deepcopy
import pytest

from . import provisional as p


def sealed(value):
    value = deepcopy(value)
    value.pop('sha256', None)
    value['sha256'] = p.digest(value)
    return value


def row(key, family='A', target=None, split='train', **changes):
    target = target or family + '/stem'
    value = dict(sample_id=key, target_id='generated/' + key, context_id='generated-' + key,
        source_family=family, source_target=target, conservative_view_cap_group=target,
        split=split, resolution=[1696, 816], training_approved=False, source_cap_reset=False,
        encoded_rgb_sha256=p.digest('png-' + key), decoded_rgb_sha256=p.digest('rgb-' + key),
        scene_sha256=p.digest('scene-' + key), camera_sha256=p.digest('camera'), decision=p.STRICT,
        annotation_review=dict(passed=True, method='automatic', evidence_id='synthetic-audit'))
    value.update(changes)
    return value


def inventory(rows):
    return sealed(dict(schema='greenhouse.pinned_native_observed_inventory.v1',
        state='observed_inventory_only_not_global_admission', training_approved=False,
        source_cap_reset=False, counts=dict(captured_rows=len(rows)), records=rows))


def coverage(packet, edges=(), pruned=0):
    rows = sorted(packet['records'], key=lambda r: r['sample_id'])
    total = len(rows) * (len(rows) - 1) // 2
    pins = [dict(sample_id=r['sample_id'], path=r['sample_id']+'/inputs/rgb.png',
                 sha256=r['encoded_rgb_sha256'], decoded_rgb_sha256=r['decoded_rgb_sha256'],
                 nominal_uv=[800.5, 400.5]) for r in rows]
    return sealed(dict(schema=p.PAIR_SCHEMA, inventory_sha256=packet['sha256'], metric=p.METRIC,
        threshold=p.CUTOFF, dimensions=[1696, 816], patch_radius=64,
        patch_anchor='nominal_10mm_cut_point_not_anatomical_attachment',
        sample_ids=[r['sample_id'] for r in rows], image_pins=pins, near_image_edges=list(edges),
        edge_scores=[p.CUTOFF]*len(edges), complete=True, global_collection_complete=False,
        threshold_calibrated=False, training_approved=False,
        counts=dict(total_pairs=total, resolved_pairs=total, exact_compared_pairs=total-pruned,
                    bound_pruned_pairs=pruned),
        input_bindings={r['path']: r['sha256'] for r in pins} or {'empty-inventory': packet['sha256']},
        implementation_bindings={'synthetic-index.py': p.digest('test-code')}))


def select(rows, *, edges=(), overlays=(), splits=None, pruned=0):
    packet = inventory(rows)
    return p.select_observed_train(packet, splits or {'A': 'train'}, coverage(packet, edges, pruned), overlays=overlays)


def test_shared_original_cap_no_invented_quotas_and_honest_scope():
    result = select([row(f'view-{i:02}') for i in range(30)])
    assert result['counts'] == {'train': 12} and result['source_counts'] == {'A/stem': 12}
    assert result['train_goal'] == 20000 and result['train_shortfall'] == 19988
    assert all(v == ['original_source_cap'] for v in result['excluded'].values())
    assert not result['scope']['heldout_counted'] and not result['scope']['global_complete']
    assert not result['scope']['pruning_bounds_independently_verified']
    assert not result['training_approved'] and not result['calibration_validated']
    assert result['policy']['threshold'] == p.CUTOFF


def test_strict_only_and_holds_stay_in_duplicate_graph():
    rows = [row('a'), row('b', decision='hold_visual_clarity'), row('c')]
    result = select(rows, edges=[['a', 'b'], ['b', 'c']])
    assert len(result['selected']) == 1 and len(set(result['view_duplicate_groups'].values())) == 1
    assert 'strict_annotation_required' in result['excluded']['b']


@pytest.mark.parametrize('identity', ['decoded_rgb_sha256', 'scene_sha256'])
@pytest.mark.parametrize('overlay', [dict(visual_hold=True), dict(control_role='control'),
                                     dict(control_role='diagnostic'), dict(control_role='replay')])
def test_negative_overlays_propagate_exact_aliases_not_near_neighbours(identity, overlay):
    rows = [row('a'), row('b'), row('c')]; rows[1][identity] = rows[0][identity]
    result = select(rows, edges=[['a', 'c']],
                    overlays=[dict(sample_id='a', evidence_id='visual-or-role-receipt', **overlay)])
    assert [r['sample_id'] for r in result['selected']] == ['c']
    reason = 'explicit_visual_hold' if 'visual_hold' in overlay else 'control_role'
    assert reason in result['excluded']['a'] and reason in result['excluded']['b']


def test_positive_overlay_cannot_rescue_non_strict_or_clear_an_alias_hold():
    rows = [row('a', decision='exclude_geometry_or_visibility'), row('b')]
    rows[1]['decoded_rgb_sha256'] = rows[0]['decoded_rgb_sha256']
    result = select(rows, overlays=[dict(sample_id='a', evidence_id='hold', visual_hold=True),
        dict(sample_id='b', evidence_id='not-an-override', visual_hold=False, control_role='observation')])
    assert not result['selected']
    assert 'strict_annotation_required' in result['excluded']['a']
    assert 'explicit_visual_hold' in result['excluded']['b']


def test_known_heldout_alias_quarantines_train_without_counting_heldout():
    a, b = row('a'), row('b', family='V', split='validation')
    b['decoded_rgb_sha256'] = a['decoded_rgb_sha256']
    result = select([a, b], splits={'A': 'train', 'V': 'validation'})
    assert result['counts'] == {'train': 0}
    assert all('cross_split_view_duplicate' in r for r in result['excluded'].values())


def test_cross_donor_near_duplicates_share_one_representative():
    result = select([row('a'), row('b', family='B')], splits={'A': 'train', 'B': 'train'}, edges=[['a', 'b']])
    assert result['counts'] == {'train': 1}


@pytest.mark.parametrize('field,value', [('complete', False), ('sample_ids', ['a']),
    ('sample_ids', ['b', 'a']), ('input_bindings', {}), ('implementation_bindings', {'bad.py': 'wrong'}),
    ('threshold', .04), ('metric', 'invented'), ('patch_radius', 32), ('patch_anchor', 'anatomical_junction'),
    ('dimensions', [848, 408]), ('inventory_sha256', '0'*64), ('image_pins', []),
    ('threshold_calibrated', True), ('global_collection_complete', True), ('training_approved', True)])
def test_graph_contract_must_match_exact_inventory_and_metric(field, value):
    packet = inventory([row('a'), row('b')]); graph = coverage(packet); graph[field] = value
    with pytest.raises(ValueError): p.select_observed_train(packet, {'A': 'train'}, sealed(graph))


@pytest.mark.parametrize('field,value', [('total_pairs', 0), ('resolved_pairs', 0),
    ('exact_compared_pairs', 0), ('bound_pruned_pairs', 1), ('resolved_pairs', True), ('total_pairs', -1)])
def test_missing_or_inconsistent_comparison_counts_fail(field, value):
    packet = inventory([row('a'), row('b')]); graph = coverage(packet); graph['counts'][field] = value
    with pytest.raises(ValueError): p.select_observed_train(packet, {'A': 'train'}, sealed(graph))


@pytest.mark.parametrize('field,value', [('sha256', '0'*64), ('decoded_rgb_sha256', '0'*64),
                                       ('nominal_uv', [0, 0]), ('path', 'another-image')])
def test_image_pins_cannot_substitute_inventory_buffers(field, value):
    packet = inventory([row('a')]); graph = coverage(packet); graph['image_pins'][0][field] = value
    with pytest.raises(ValueError): p.select_observed_train(packet, {'A': 'train'}, sealed(graph))


@pytest.mark.parametrize('edges', [[['a', 'missing']], [['a', 'a']], [['a', 'b'], ['b', 'a']], [['a']], ['ab']])
def test_unknown_repeated_or_invalid_edges_fail(edges):
    with pytest.raises(ValueError): select([row('a'), row('b')], edges=edges)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1., .04, True, None])
def test_bad_edge_scores_fail(score):
    packet = inventory([row('a'), row('b')]); graph = coverage(packet, [['a', 'b']]); graph['edge_scores'] = [score]
    with pytest.raises(ValueError): p.select_observed_train(packet, {'A': 'train'}, sealed(graph))


def test_index_bound_accounting_does_not_claim_recomputation():
    assert select([row('a'), row('b')], pruned=1)['counts']['train'] == 2
    with pytest.raises(ValueError): select([row('a'), row('b')], edges=[['a', 'b']], pruned=1)
    assert select([])['counts']['train'] == 0
    assert select([row('a')])['counts']['train'] == 1


def test_stale_seals_and_changed_caller_data_are_rejected():
    packet = inventory([row('a')]); graph = coverage(packet)
    packet['records'][0]['decision'] = 'hold_visual_clarity'
    with pytest.raises(ValueError): p.select_observed_train(packet, {'A': 'train'}, graph)
    packet = inventory([row('a')]); graph = coverage(packet); graph['complete'] = False
    with pytest.raises(ValueError): p.select_observed_train(packet, {'A': 'train'}, graph)


@pytest.mark.parametrize('changes', [dict(split='test'), dict(source_family='B'),
    dict(conservative_view_cap_group='A/reset'), dict(training_approved=True), dict(source_cap_reset=True)])
def test_ancestry_split_and_approval_mismatches_fail(changes):
    with pytest.raises(ValueError): select([row('a', **changes)])


@pytest.mark.parametrize('overlay', [dict(sample_id='missing', visual_hold=True),
    dict(sample_id='a', visual_hold=1), dict(sample_id='a', control_role='approved'), dict(sample_id='a')])
def test_bad_overlays_fail(overlay):
    with pytest.raises(ValueError): select([row('a')], overlays=[dict(evidence_id='receipt', **overlay)])


def test_deterministic_balance_no_mutation_and_detached_results():
    rows = [row('a'), row('b', target='A/other'), row('c', family='B')]
    packet = inventory(rows); graph = coverage(packet); splits = {'A': 'train', 'B': 'train'}
    overlays = [dict(sample_id='b', evidence_id='review', visual_hold=False)]
    before = deepcopy((packet, graph, splits, overlays))
    result = p.select_observed_train(packet, splits, graph, overlays=overlays)
    assert [r['sample_id'] for r in result['selected']] == ['b', 'c', 'a']
    assert result['selected'] == select(rows[::-1], splits=splits, overlays=overlays)['selected']
    assert (packet, graph, splits, overlays) == before
    result['overlays'][0]['visual_hold'] = True; result['selected'][0]['target_id'] = 'changed'
    assert (packet, graph, splits, overlays) == before
