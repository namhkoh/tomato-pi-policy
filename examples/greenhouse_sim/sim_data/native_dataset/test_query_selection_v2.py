"""Synthetic CPU tests only; no real dataset writes or native launch."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest

from . import query_selection_v2 as selector
from ..capture_contract import fingerprint
from ..native_clear_contract_test import fixture
from ..native_query_visibility import NativeQueryVisibility
from ..native_clear_labels import derive
from ..automated_native_review import trace_review


def native_case():
    args = list(fixture())
    seal(args)
    return args


def seal(args):
    meta, _, rgb, depth, _, _, _ = args
    meta['synchronization'].update(method='frozen_scene_single_native_writer_payload',
        freshness=dict(camera_sha256=fingerprint(meta['calibration']),
                       rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                       depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest()))


def run(args, **kwargs):
    target = (args[5] == 1).astype(np.uint8)*255
    return selector.select_query(*args, target_mask=target, **kwargs)


def test_grid_has_stable_anchored_prefix_knots_and_exact_endpoint():
    lengths = [0., .0091, .0142, .0341, .0573, .2]
    short = selector.fixed_photometric_arcs(lengths, .045)
    long = selector.fixed_photometric_arcs(lengths, .07575)
    assert {float(mm)/1000 for mm in range(8, 46, 5)} <= set(short)
    assert {.0091, .0142, .0341, .045} <= set(short)
    assert short[0] == .008 and short[-1] == .045
    assert long[-1] == .07575 and .0573 in long
    assert set(short)-{.045} <= set(long)
    assert max(np.diff(short)) <= .005+1e-15
    assert len(short) == len(set(short))
    assert .07575 not in short


@pytest.mark.parametrize('end', [.044999, .251, float('nan'), float('inf'), True, -.01])
def test_invalid_grid_end_fails_closed(end):
    with pytest.raises(ValueError):
        selector.fixed_photometric_arcs([0., .3], end)


@pytest.mark.parametrize('lengths', [[.001, .2], [0., .02, .01, .2], [0., 0., .2],
                                    [0., float('nan')], [0.], [[0., .2]]])
def test_invalid_grid_anatomy_rejected(lengths):
    with pytest.raises(ValueError):
        selector.fixed_photometric_arcs(lengths, .045)


def test_grid_probe_budget_is_bounded():
    with pytest.raises(ValueError):
        selector.fixed_photometric_arcs(np.linspace(0, .3, 12001), .2)


@pytest.mark.parametrize('legacy,fixed,expected', [
    (True, True, []), (False, True, ['legacy_full_trace_failed']),
    (True, False, ['anchored_photometric_grid_failed']),
    (False, False, ['legacy_full_trace_failed', 'anchored_photometric_grid_failed'])])
def test_mandatory_conjunction_cannot_promote_by_resampling(legacy, fixed, expected):
    assert selector._joint_rejections({'passed': legacy}, {'passed': fixed}) == expected


def test_success_exact_candidates_deterministic_selection_and_no_mutations():
    args = native_case()
    baseline = derive(*args)
    before_json = [deepcopy(args[i]) for i in (0, 1, 6)]
    before_arrays = [a.tobytes() for a in args[2:6]]
    result = run(args, expected_label=baseline)
    repeated = run(args, expected_label=baseline)
    assert result == repeated
    assert result['candidate_count'] == 41 and result['locally_usable_count'] == 41
    assert result['both_pass_count'] == 41
    assert [r['arc_m'] for r in result['candidates']] == np.linspace(.045, .2*.85, 41).tolist()
    selected = result['selected_query']
    assert selected['arc_m'] >= .045 and selected['legacy_trace']['passed'] and selected['fixed_grid']['passed']
    assert np.linalg.norm(np.asarray(selected['query_pixel_uv'])-baseline['nominal_pixel_uv']) >= 18.
    assert not result['training_approved'] and not result['continuous_visibility_verified']
    assert result['context']['shared_visibility_precomputations'] == 1
    assert result['context']['fixed_cache_hits'] > 0
    for i, prior in zip((0, 1, 6), before_json):
        assert args[i] == prior
    assert [a.tobytes() for a in args[2:6]] == before_arrays
    assert derive(*args) == baseline
    row = next(r for r in result['candidates'] if r['candidate_index'] == selected['candidate_index'])
    selected['query_pixel_uv'][0] = -99
    assert row['query_pixel_uv'][0] >= 0  # Returned selected evidence is a fresh copy.


def test_original_selected_candidate_legacy_trace_replays_exactly():
    args = native_case()
    baseline = derive(*args)
    result = run(args, expected_label=baseline)
    old_row = next(r for r in result['candidates'] if r['query_evidence'] == baseline['query_evidence'])
    expected = trace_review(args[0], args[1], baseline, *args[2:])
    assert old_row['legacy_trace'] == expected


def test_context_owns_immutable_exact_native_arrays_not_caller_aliases():
    args = native_case()
    target = (args[5] == 1).astype(np.uint8)*255
    ctx = selector._Invocation(*args, target)
    ctx.visibility = NativeQueryVisibility(ctx.rgb, ctx.mask, (1696, 816))
    short = ctx.fixed_review(.045)
    args[2][408, 947] = 0
    args[3][408, 947] = .5
    args[5][408, 947] = 0
    args[0]['sample_id'] = 'changed_outside_context'
    target[408, 947] = 0
    assert ctx.fixed_review(.045) == short
    assert ctx.meta['sample_id'] != args[0]['sample_id']
    for array in (ctx.rgb, ctx.depth, ctx.valid, ctx.components, ctx.target_mask):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_fixed_grid_observes_knots_and_shared_prefix_without_recalculation():
    args = native_case()
    # Add an actual straight-chain anatomical knot, not a new candidate/gate.
    args[1]['components']['Petiole']['capsules_local_m'][0].insert(1, [.0341, 0, 0, .002])
    ctx = selector._Invocation(*args, (args[5] == 1).astype(np.uint8)*255)
    ctx.visibility = NativeQueryVisibility(ctx.rgb, ctx.mask, (1696, 816))
    a, b = ctx.fixed_review(.045), ctx.fixed_review(.07575)
    a_probes = {p['arc_m']: p for p in a['probes'] if p['arc_m'] != .045}
    b_probes = {p['arc_m']: p for p in b['probes']}
    assert .0341 in a_probes
    assert all(b_probes[k] == v for k, v in a_probes.items())
    assert ctx.photo_hits == len(a_probes)


@pytest.mark.parametrize('fault', ['cut_id', 'cut_depth', 'dark_cut', 'missing_parent'])
def test_upstream_frozen_rejection_cannot_be_rescued(fault):
    args = native_case()
    if fault == 'cut_id':
        args[5][:, 878:909] = 0
    elif fault == 'cut_depth':
        args[3][408, 878] = .5
    elif fault == 'dark_cut':
        args[2][args[5] == 1] = 0
    else:
        args[5][args[5] == 2] = 0
    seal(args)
    result = run(args)
    assert result['state'] == 'legacy_upstream_rejection_preserved'
    assert result['candidate_count'] == 0 and result['selected_query'] is None


def test_disconnected_query_mask_is_not_joined():
    args = native_case()
    args[5][:, 950:955] = 0
    seal(args)
    result = run(args)
    assert result['baseline_reason'] == 'no_usable_visible_query_connected_to_cut'
    assert result['candidate_count'] == 41 and result['locally_usable_count'] == 0
    assert result['selected_query'] is None
    assert result['rejection_counts']['different_exact_target_island'] == 41


@pytest.mark.parametrize('fault', ['trace_identity', 'trace_depth', 'trace_contrast'])
def test_full_trace_or_fixed_failure_preserved_with_locally_usable_endpoints(fault):
    args = native_case()
    if fault == 'trace_identity':
        args[5][408, 950] = 2  # Connected silhouette around a one-pixel interruption.
    elif fault == 'trace_depth':
        args[3][408, 950] = .5
    else:
        args[2][390:426, 925:980] = [50, 100, 30]
    seal(args)
    result = run(args)
    assert result['locally_usable_count'] > 0
    assert result['both_pass_count'] == 0 and result['selected_query'] is None
    assert all(r['rejections'] for r in result['candidates'])
    if fault == 'trace_contrast':
        assert result['rejection_counts']['anchored_photometric_grid_failed'] > 0
    else:
        assert result['rejection_counts']['legacy_full_trace_failed'] == result['locally_usable_count']


def test_exact_legacy_descending_short_chain_skips_below_45mm():
    args = native_case()
    args[1]['components']['Petiole']['capsules_local_m'][0][-1][0] = .05
    result = run(args)
    assert result['candidate_count'] == 41
    assert result['locally_usable_count'] == 1 and result['selected_query']['arc_m'] == .045
    assert result['rejection_counts']['below_minimum_query_arc'] == 40


@pytest.mark.parametrize('fault', ['rgb_pin', 'depth_pin', 'target_mask', 'float_ids', 'validity'])
def test_native_evidence_and_buffer_contract_not_bypassed(fault):
    args = native_case()
    target = (args[5] == 1).astype(np.uint8)*255
    if fault == 'rgb_pin':
        args[2][0, 0] = 0
    elif fault == 'depth_pin':
        args[3][0, 0] = .5
    elif fault == 'target_mask':
        target[0, 0] = 255
    elif fault == 'float_ids':
        args[5] = args[5].astype(float)
    else:
        args[4][0, 0] = False
    with pytest.raises(ValueError):
        selector.select_query(*args, target_mask=target)


def test_stale_label_rejected_and_no_caller_injected_context():
    args = native_case()
    baseline = derive(*args)
    stale = deepcopy(baseline)
    stale['query_pixel_uv'][0] += .1
    with pytest.raises(ValueError, match='Saved label'):
        run(args, expected_label=stale)
    with pytest.raises(TypeError):
        run(args, context=object())


@pytest.mark.parametrize('disposition', ['pass', 'hold', 'exclude'])
def test_combined_epoch_and_faithful_disposition_mapping(disposition):
    args = native_case()
    if disposition == 'hold':
        args[5][408, 950] = 2
    elif disposition == 'exclude':
        args[5][:, 878:909] = 0
    seal(args)
    baseline = derive(*args)
    before = deepcopy(baseline)
    label, trace, evidence = selector.annotate_v2(*args,
        target_mask=(args[5] == 1).astype(np.uint8)*255, expected_label=baseline)
    assert evidence['annotation_epoch'] == selector.ANNOTATION_EPOCH == label['annotation_epoch']
    assert baseline == before and 'annotation_epoch' not in baseline
    assert not label['training_approved']
    if disposition == 'pass':
        assert label['eligible'] and trace['passed']
        assert trace['legacy_trace']['passed'] and trace['fixed_grid']['passed']
        assert label['query_evidence'] == evidence['selected_query']['query_evidence']
        assert label['answer'] == baseline['answer']
    elif disposition == 'hold':
        assert label['eligible'] and not trace['passed']
        assert trace['original_query_retained_as_failed_fallback']
        assert label['query_evidence'] == baseline['query_evidence']
        assert label['answer'] == baseline['answer']
        assert evidence['both_pass_count'] == 0
    else:
        assert not label['eligible'] and trace is None
        assert label['reason'] == baseline['reason']
