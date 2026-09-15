"""Synthetic thresholds/features ONLY; no actual qualification or adoption."""
import ast
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import pytest

from . import representative_packing as p
from . import morphology_frame_v2 as frame


def h(value):
    return sha256(str(value).encode()).hexdigest()


def pin(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return p.PinnedJSON(raw, sha256(raw).hexdigest())


def descriptor(x):
    return dict(schema=frame.SCHEMA, frame_version=frame.FRAME_VERSION,
                fixed=[0.]*104, leaves=[[x]*15], leaf_count=1)


def context(key, x, *, family='f0', original='O'):
    value = descriptor(x)
    return dict(kind='original' if key == original else 'generated', source_family=family,
        source_target=family+'/SubStem_41', source_context_id=original, old_pool_id='pool:'+family,
        geometry_identity_sha256=h('mesh:'+key), frame_reference_sha256=h('frame:'+family),
        descriptor=value, descriptor_sha256=p._digest(value))


def sample(key, ctx, *, family='f0', split='train'):
    return dict(context_id=ctx, source_family=family, source_target=family+'/SubStem_41', split=split,
        decoded_rgb_sha256=h('decoded:'+key), scene_sha256=h('scene:'+key), camera_sha256=h('camera:'+key),
        sample_sha256=h('sample:'+key), encoded_rgb_sha256=h('rgb:'+key), audit_sha256=h('audit:'+key),
        annotation_review=dict(passed=True, method='automatic', evidence_id='audit:'+key), decision=p.STRICT)


def proofs(roles, prefix):
    return {r: dict(evidence_id=prefix+':'+r, sha256=h(prefix+':'+r)) for r in roles}


def sync_geometry(d):
    """Build truthful synthetic evidence after intentionally changing the fixture."""
    snap = d['snapshot']; contexts = snap['contexts']; samples = snap['samples']
    for c in contexts.values(): c['descriptor_sha256'] = p._digest(c['descriptor'])
    records = []
    for a, b in combinations(sorted(contexts), 2):
        x, y = contexts[a]['descriptor'], contexts[b]['descriptor']
        if x is None or y is None or not x['leaf_count'] or not y['leaf_count']:
            records.append(dict(left=a, right=b, state='unknown', reason='missing_or_empty_descriptor'))
        else:
            records.append(dict(left=a, right=b, state='resolved', distance=p.metric.distance(x, y)))
    d['pairs']['records'] = records
    graph = d['pairs']['view_graph']; graph['sample_ids'] = sorted(samples)
    graph['total_pairs'] = graph['resolved_pairs'] = len(samples)*(len(samples)-1)//2
    for r in d['reviews']['contexts']:
        c = contexts[r['context_id']]; s = samples[r['sample_id']]
        r.update(source_context_id=c['source_context_id'], descriptor_sha256=c['descriptor_sha256'],
                 source_descriptor_sha256=contexts[c['source_context_id']]['descriptor_sha256'],
                 sample_sha256=s['sample_sha256'])
        r['artifacts']['native_audit']['sha256'] = s['audit_sha256']
        r['artifacts']['generated_rgb']['sha256'] = s['encoded_rgb_sha256']


def fixture():
    d = {k: dict(schema=v) for k, v in p.SCHEMAS.items() if k != 'result'}
    splits = {f'f{i}': 'train' if i < 16 else 'validation' if i < 20 else 'test' for i in range(24)}
    d['snapshot'].update(inventory_sha256=h('inventory'),
        contexts={k: context(k, x) for k, x in [('O', -4), ('A', 0), ('B', .75), ('C', 1.5)]},
        samples={k.lower(): sample(k.lower(), k) for k in ('O', 'A', 'B', 'C')},
        frozen_splits=splits, overlays=[], training_approved=False, source_cap_reset=False)
    d['pairs'].update(metric=p.metric.METRIC, metric_implementation_sha256=p.METRIC_SHA256,
        frame_implementation_sha256=p.FRAME_SHA256, records=[],
        view_graph=dict(inventory_sha256=h('inventory'), evidence_id='view_graph', sha256=h('view_graph'),
            external_authority='synthetic_test_authority', complete=True, sample_ids=[],
            total_pairs=0, resolved_pairs=0, near_edges=[]))
    d['reviews'].update(metric=p.metric.METRIC, contexts=[dict(context_id=k, source_context_id='O',
        sample_id=k.lower(), descriptor_sha256='', source_descriptor_sha256='', sample_sha256='',
        decision='geometry_distinct_plausible', artifacts=proofs(p.REVIEW_ROLES, k)) for k in ('A', 'B', 'C')])
    d['controls'].update(metric=p.metric.METRIC, threshold=1., status='externally_qualified_controls',
        external_authority='synthetic_test_authority', qualified_context_ids=['A', 'B', 'C'],
        evidence=proofs(p.PROOF_ROLES, 'controls'))
    d['policy'].update(enabled=True, version=p.VERSION, ordering=p.ORDER, metric=p.metric.METRIC,
        metric_implementation_sha256=p.METRIC_SHA256, frame_implementation_sha256=p.FRAME_SHA256,
        threshold=1., frozen_splits_sha256=p._digest(splits))
    d['contribution_plan'].update(baseline_sample_ids=['o'], candidate_samples_by_context={'A': ['a'], 'B': ['b'], 'C': ['c']})
    sync_geometry(d)
    return d


def bind(d):
    d = deepcopy(d); result = {}
    for name, dependencies in [('snapshot', ()), ('pairs', ('snapshot',)), ('reviews', ('snapshot', 'pairs')),
                               ('controls', ('snapshot', 'pairs', 'reviews')), ('policy', ('controls',)),
                               ('contribution_plan', ('snapshot', 'pairs', 'reviews', 'policy'))]:
        for dep in dependencies: d[name][dep+'_sha256'] = result[dep].sha256
        result[name] = pin(d[name])
    return result


def run(d):
    return p.pack_representatives(**bind(d))


def test_chain_is_packing_not_equivalence_or_capacity():
    d = fixture(); r = run(d)
    assert r['proposed_representative_ids'] == ['A', 'C']
    assert r['rejected']['B'] == ['near_pair']
    assert r['contributing_sample_ids'] == ['a', 'c', 'o']
    assert r['old_pool_by_context'] == {k: 'pool:f0' for k in ('O', 'A', 'B', 'C')}
    assert all(w['state'] == 'separated' for w in r['final_pair_checks'])
    assert r['maximal_relative_to_declared_constraints'] is True and r['evidence_relative'] is True
    for flag in ('maximum_claimed', 'biological_family_novelty_claimed', 'context_entitlements_granted',
                 'source_cap_reset', 'caps_checked', 'training_approved', 'policy_adopted',
                 'native_execution_verified', 'visual_review_truth_verified', 'calibration_reexecuted',
                 'source_files_reverified', 'implementation_files_reverified', 'global_collection_complete'):
        assert r[flag] is False
    assert r['sha256'] == p._digest({k: v for k, v in r.items() if k != 'sha256'})


def test_nonrepresentative_B_in_original_pool_blocks_both_endpoints():
    d = fixture(); d['contribution_plan'].update(baseline_sample_ids=['o', 'b'], candidate_samples_by_context={'A': ['a'], 'C': ['c']})
    r = run(d)
    assert not r['proposed_representative_ids'] and r['contributing_sample_ids'] == ['b', 'o']
    assert all(any(w['right'] == 'B' and w['state'] == 'blocked' and w['role'] == 'actual_contributor'
                   for w in r['candidate_pair_checks'][k]) for k in ('A', 'C'))


def test_unused_unqualified_bridge_does_not_transitively_collapse_endpoints():
    d = fixture(); d['controls']['qualified_context_ids'].remove('B')
    d['reviews']['contexts'][1]['decision'] = 'novelty_withheld'
    r = run(d)
    assert r['proposed_representative_ids'] == ['A', 'C']
    assert 'native_visual_qualification_required' in r['rejected']['B'] and not r['sample_exclusions']['b']


def test_old_pool_bridge_is_checked_even_if_novelty_withheld():
    d = fixture(); d['controls']['qualified_context_ids'].remove('B')
    d['reviews']['contexts'][1]['decision'] = 'novelty_withheld'
    d['contribution_plan'].update(baseline_sample_ids=['b'], candidate_samples_by_context={'A': ['a'], 'C': ['c']})
    assert not run(d)['proposed_representative_ids']


def test_every_proposed_sample_checked_not_only_native_review_witness():
    d = fixture(); d['snapshot']['samples']['a2'] = sample('a2', 'A')
    d['snapshot']['samples']['a2']['decision'] = 'hold_visual_clarity'
    d['contribution_plan']['candidate_samples_by_context']['A'].append('a2'); sync_geometry(d)
    assert 'strict_annotation_required' in run(d)['rejected']['A']


def test_new_original_is_checked_against_earlier_representative():
    d = fixture(); c = d['snapshot']['contexts']; s = d['snapshot']['samples']
    c['Q'] = context('Q', .5, family='f1', original='Q')
    c['C'] = context('C', 4., family='f1', original='Q'); s['c'] = sample('c', 'C', family='f1')
    d['contribution_plan']['candidate_samples_by_context'].pop('B'); sync_geometry(d)
    r = run(d)
    assert r['proposed_representative_ids'] == ['A']
    assert any(w['left'] == 'A' and w['right'] == 'Q' and w['role'] == 'new_source_original'
               and w['state'] == 'blocked' for w in r['candidate_pair_checks']['C'])


@pytest.mark.parametrize('where', ['source', 'contributor', 'representative'])
def test_unknown_never_proves_separation(where):
    d = fixture(); endpoints = {'source': {'A', 'O'}, 'contributor': {'A', 'B'}, 'representative': {'A', 'C'}}[where]
    if where == 'contributor':
        d['contribution_plan'].update(baseline_sample_ids=['b'], candidate_samples_by_context={'A': ['a'], 'C': ['c']})
    for row in d['pairs']['records']:
        if {row['left'], row['right']} == endpoints:
            row.pop('distance'); row.update(state='unknown', reason='external_evidence_missing')
    r = run(d)
    assert 'unknown_pair' in r['rejected']['C' if where == 'representative' else 'A']
    assert r['coverage']['unknown_pairs'] == 1 and not r['coverage']['distances_complete']


@pytest.mark.parametrize('missing', [None, dict(schema=frame.SCHEMA, frame_version=frame.FRAME_VERSION,
                                             fixed=[0.]*104, leaves=[], leaf_count=0)])
def test_missing_empty_descriptor_stays_unknown_old_pool(missing):
    d = fixture(); d['snapshot']['contexts']['A']['descriptor'] = missing; sync_geometry(d)
    r = run(d)
    assert 'A' not in r['proposed_representative_ids'] and r['old_pool_by_context']['A'] == 'pool:f0'
    assert r['coverage']['unknown_pairs'] == 3


@pytest.mark.parametrize('identity', ['rgb', 'pose'])
def test_negative_hold_propagates_exact_sample_aliases(identity):
    d = fixture(); samples = d['snapshot']['samples']
    if identity == 'rgb': samples['a']['decoded_rgb_sha256'] = samples['b']['decoded_rgb_sha256']
    else:
        for k in ('scene_sha256', 'camera_sha256'): samples['a'][k] = samples['b'][k]
    d['snapshot']['overlays'] = [dict(sample_id='b', evidence_id='negative', visual_hold=True)]
    r = run(d)
    assert 'explicit_visual_hold' in r['rejected']['A'] and r['overlays'] == d['snapshot']['overlays']


@pytest.mark.parametrize('role', ['control', 'diagnostic', 'replay'])
def test_control_rows_never_contribute(role):
    d = fixture(); d['snapshot']['overlays'] = [dict(sample_id='a', evidence_id='role', control_role=role)]
    assert 'control_role' in run(d)['rejected']['A']


def test_context_hold_is_preserved_as_negative_overlay():
    d = fixture(); d['controls']['qualified_context_ids'].remove('A')
    d['reviews']['contexts'][0]['decision'] = 'hold'
    r = run(d)
    assert r['review_hold_overlays'][0]['sample_id'] == 'a'
    assert 'explicit_visual_hold' in r['rejected']['A']


def test_invalid_baseline_never_silently_removed():
    d = fixture(); d['snapshot']['overlays'] = [dict(sample_id='o', evidence_id='hold', visual_hold=True)]
    with pytest.raises(ValueError, match='Invalid baseline'): run(d)


def test_exact_geometry_alias_cannot_receive_second_representative_role():
    d = fixture(); c = d['snapshot']['contexts']
    c['C']['descriptor'] = deepcopy(c['A']['descriptor']); c['C']['geometry_identity_sha256'] = c['A']['geometry_identity_sha256']
    sync_geometry(d); r = run(d)
    assert r['proposed_representative_ids'] == ['A'] and 'exact_geometry_alias' in r['rejected']['C']


def test_exact_geometry_alias_contradictory_metric_rejected():
    d = fixture(); d['snapshot']['contexts']['C']['geometry_identity_sha256'] = d['snapshot']['contexts']['A']['geometry_identity_sha256']
    with pytest.raises(ValueError, match='alias contradicts'): run(d)


def test_zero_feature_distance_is_not_declared_exact_mesh_equivalence():
    d = fixture(); d['snapshot']['contexts']['C']['descriptor'] = deepcopy(d['snapshot']['contexts']['A']['descriptor'])
    sync_geometry(d); r = run(d)
    assert r['geometry_alias_groups']['A'] != r['geometry_alias_groups']['C']
    assert 'near_pair' in r['rejected']['C']


def test_cross_split_near_view_bridge_is_quarantined_even_if_not_contributing():
    d = fixture(); d['snapshot']['contexts']['H'] = context('H', 9., family='f16', original='H')
    d['snapshot']['samples']['h'] = sample('h', 'H', family='f16', split='validation')
    sync_geometry(d); d['pairs']['view_graph']['near_edges'] = [['a', 'h']]
    r = run(d)
    assert 'cross_split_view_duplicate' in r['rejected']['A']
    assert r['frozen_splits'] == d['snapshot']['frozen_splits']


def test_near_view_duplicates_cannot_both_contribute():
    d = fixture(); d['pairs']['view_graph']['near_edges'] = [['a', 'c']]
    assert 'duplicate_view' in run(d)['rejected']['C']


def test_cross_split_exact_geometry_alias_quarantined_without_any_heldout_image():
    d = fixture(); contexts = d['snapshot']['contexts']
    contexts['H'] = context('H', 9., family='f16', original='H')
    contexts['G'] = context('G', 0., family='f16', original='H')
    contexts['G']['geometry_identity_sha256'] = contexts['A']['geometry_identity_sha256']
    sync_geometry(d); r = run(d)
    assert 'cross_split_exact_geometry_alias' in r['rejected']['A']
    assert 'A' not in r['proposed_representative_ids']


def test_generated_donor_frame_must_match_original():
    d = fixture(); d['snapshot']['contexts']['A']['frame_reference_sha256'] = h('wrong_frame')
    with pytest.raises(ValueError, match='ancestry/pool'): run(d)


def test_maximal_not_maximum_due_to_frozen_order():
    d = fixture(); c = d['snapshot']['contexts']
    c['A']['descriptor'] = descriptor(.75); c['B']['descriptor'] = descriptor(0.)
    sync_geometry(d); r = run(d)
    assert r['proposed_representative_ids'] == ['A'] and r['maximum_claimed'] is False
    assert p.metric.distance(c['B']['descriptor'], c['C']['descriptor']) > d['policy']['threshold']


def test_input_permutations_do_not_change_packing_or_checks():
    d = fixture(); r = run(d)
    d['snapshot']['contexts'] = dict(reversed(list(d['snapshot']['contexts'].items())))
    d['snapshot']['samples'] = dict(reversed(list(d['snapshot']['samples'].items())))
    d['pairs']['records'].reverse()
    for row in d['pairs']['records']: row['left'], row['right'] = row['right'], row['left']
    d['reviews']['contexts'].reverse(); d['controls']['qualified_context_ids'].reverse()
    d['contribution_plan']['candidate_samples_by_context'] = {'C': ['c'], 'B': ['b'], 'A': ['a']}
    s = run(d)
    for key in ('candidate_order', 'proposed_representative_ids', 'contributing_sample_ids', 'rejected',
                'candidate_pair_checks', 'final_pair_checks', 'old_pool_by_context'):
        assert r[key] == s[key]


def test_detached_inputs_results_and_pure_execution():
    d = fixture(); before = deepcopy(d); args = bind(d); input_bytes = {k: v.data for k, v in args.items()}
    r = p.pack_representatives(**args); r['old_pool_by_context']['A'] = 'bad'; r['overlays'].append('bad')
    assert d == before and {k: v.data for k, v in args.items()} == input_bytes
    assert run(d)['old_pool_by_context']['A'] == 'pool:f0'
    tree = ast.parse(Path(p.__file__).read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ('open', 'eval', 'exec', 'compile') for n in ast.walk(tree))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(name and any(word in name for word in ('capture', 'audit', 'subprocess', 'pathlib')) for name in imports)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'reversed_duplicate', 'self', 'outside', 'score', 'boolean', 'negative', 'unknown_without_reason'])
def test_global_pair_damage_rejected(damage):
    d = fixture(); records = d['pairs']['records']; row = records[0]
    if damage == 'missing': records.pop()
    elif damage in ('duplicate', 'reversed_duplicate'):
        extra = deepcopy(row)
        if damage == 'reversed_duplicate': extra['left'], extra['right'] = extra['right'], extra['left']
        records.append(extra)
    elif damage == 'self': row['right'] = row['left']
    elif damage == 'outside': row['right'] = 'not_in_snapshot'
    elif damage == 'score': row['distance'] += .001
    elif damage == 'boolean': row['distance'] = True
    elif damage == 'negative': row['distance'] = -1.
    else: row.pop('distance'); row['state'] = 'unknown'
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('field', ['complete', 'sample_ids', 'resolved_pairs', 'inventory_sha256'])
def test_view_global_coverage_damage_rejected(field):
    d = fixture(); g = d['pairs']['view_graph']
    g[field] = {'complete': False, 'sample_ids': ['a'], 'resolved_pairs': 0, 'inventory_sha256': h('wrong')}[field]
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('doc,field', [('policy', 'threshold'), ('controls', 'threshold'), ('controls', 'evidence'),
                                    ('snapshot', 'overlays'), ('contribution_plan', 'baseline_sample_ids')])
def test_no_default_required_evidence(doc, field):
    d = fixture(); del d[doc][field]
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('value', [True, -1., '1'])
def test_invalid_threshold_rejected(value):
    d = fixture(); d['policy']['threshold'] = d['controls']['threshold'] = value
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('role', sorted(p.PROOF_ROLES))
def test_every_controls_evidence_role_required(role):
    d = fixture(); del d['controls']['evidence'][role]
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('role', sorted(p.REVIEW_ROLES))
def test_every_native_visual_review_role_required(role):
    d = fixture(); del d['reviews']['contexts'][0]['artifacts'][role]
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('damage', ['audit', 'rgb', 'sample', 'descriptor', 'source', 'no_review', 'false_qualification'])
def test_native_visual_exact_joins_required(damage):
    d = fixture(); r = d['reviews']['contexts'][0]
    if damage == 'audit': r['artifacts']['native_audit']['sha256'] = h('bad')
    elif damage == 'rgb': r['artifacts']['generated_rgb']['sha256'] = h('bad')
    elif damage == 'sample': r['sample_sha256'] = h('bad')
    elif damage == 'descriptor': r['descriptor_sha256'] = h('bad')
    elif damage == 'source': r['source_context_id'] = 'C'
    elif damage == 'no_review': d['reviews']['contexts'].pop(0)
    else: d['controls']['status'] = 'unqualified'
    with pytest.raises(ValueError): run(d)


def test_conflicting_normalized_evidence_pins_rejected():
    d = fixture(); refs = d['controls']['evidence']
    refs['native_controls']['evidence_id'] = 'proof\\same'
    refs['visual_controls']['evidence_id'] = 'proof/same'
    with pytest.raises(ValueError, match='Conflicting evidence'): run(d)


@pytest.mark.parametrize('damage', ['source_pool', 'source_target', 'sample_split', 'original_count', 'policy_split_pin', 'descriptor_seal', 'quota', 'order', 'metric_pin'])
def test_frozen_identity_and_policy_damage_rejected(damage):
    d = fixture()
    if damage == 'source_pool': d['snapshot']['contexts']['A']['old_pool_id'] = 'new_budget'
    elif damage == 'source_target': d['snapshot']['contexts']['A']['source_target'] = 'f1/SubStem_41'
    elif damage == 'sample_split': d['snapshot']['samples']['a']['split'] = 'validation'
    elif damage == 'original_count': d['snapshot']['frozen_splits']['f16'] = 'train'
    elif damage == 'policy_split_pin': d['policy']['frozen_splits_sha256'] = h('bad')
    elif damage == 'descriptor_seal': d['snapshot']['contexts']['A']['descriptor_sha256'] = h('bad')
    elif damage == 'quota': d['policy']['train_goal'] = 20000
    elif damage == 'order': d['policy']['ordering'] = 'maximize_yield'
    else: d['policy']['metric_implementation_sha256'] = h('bad')
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('damage', ['wrong_context', 'duplicate', 'overlap', 'empty', 'original_candidate'])
def test_contribution_membership_must_be_exact(damage):
    d = fixture(); plan = d['contribution_plan']; candidates = plan['candidate_samples_by_context']
    if damage == 'wrong_context': candidates['A'] = ['c']
    elif damage == 'duplicate': candidates['A'] = ['a', 'a']
    elif damage == 'overlap': plan['baseline_sample_ids'].append('a')
    elif damage == 'empty': candidates['A'] = []
    else: candidates['O'] = ['o']; plan['baseline_sample_ids'] = []
    with pytest.raises(ValueError): run(d)


@pytest.mark.parametrize('doc', ['snapshot', 'pairs', 'controls', 'reviews', 'contribution_plan', 'policy'])
def test_pinned_bytes_and_cross_bindings_cannot_be_spoofed(doc):
    args = bind(fixture()); original = args[doc]
    args[doc] = p.PinnedJSON(original.data+b' ', original.sha256)
    with pytest.raises(ValueError, match='pinned JSON'): p.pack_representatives(**args)
    args[doc] = p.PinnedJSON(original.data+b' ', sha256(original.data+b' ').hexdigest())
    if doc != 'contribution_plan':
        with pytest.raises(ValueError): p.pack_representatives(**args)


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]'])
def test_malformed_pinned_json_rejected(raw):
    with pytest.raises(ValueError): p.PinnedJSON(raw, sha256(raw).hexdigest()).read()


def test_unpinned_or_missing_controls_rejected():
    args = bind(fixture()); args['controls'] = None
    with pytest.raises(ValueError): p.pack_representatives(**args)
