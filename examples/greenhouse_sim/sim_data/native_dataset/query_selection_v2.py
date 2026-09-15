"""Standalone, CPU-only query selection; no worker integration or label writes.

A legacy-local candidate must pass BOTH the unmodified legacy full trace and
an additional endpoint-independent anchored photometric grid. This is sampled
native evidence, never continuous-visibility, visual, training or motion approval.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import numpy as np

from .. import automated_native_review as legacy_review
from .. import native_clear_labels as legacy_labels
from ..capture_contract import fingerprint, project, transform_points, depth_evidence
from ..cut_regions import _oriented_chain, _sample
from ..dataset_review import require
from ..native_clear_contract import RESOLUTION, contract_hash
from ..native_query_visibility import NativeQueryVisibility, POLICY as QUERY_POLICY

SCHEMA = 'greenhouse.native_query_selection.v2'
ANNOTATION_EPOCH = SCHEMA
GRID_SCHEMA = 'greenhouse.anchored_query_photometry.v2'
CANDIDATES = 41
START_MM = 8
STEP_MM = 5
MIN_QUERY_M = .045
MAX_QUERY_M = .25
MIN_QUERY_CUT_PX = 18.
MAX_PROBES = 12000


def fixed_photometric_arcs(cumulative, end):
    """8,13,18,...mm + every in-range anatomical knot + exact query endpoint.

    Integer-mm construction gives the same shared-prefix floats for every end.
    Endpoint probes themselves need not form a nested set across different ends.
    """
    lengths = np.asarray(cumulative, dtype=float)
    require(lengths.ndim == 1 and 2 <= len(lengths) <= MAX_PROBES
            and np.isfinite(lengths).all() and lengths[0] == 0
            and np.all(np.diff(lengths) > 0), 'Invalid cumulative anatomy')
    require(type(end) in (float, int) and np.isfinite(end)
            and MIN_QUERY_M <= end <= min(float(lengths[-1]), MAX_QUERY_M),
            'Invalid fixed-grid query arc')
    regular = [float(mm)/1000 for mm in range(START_MM, 251, STEP_MM)
               if float(mm)/1000 <= end]
    knots = [float(d) for d in lengths if START_MM/1000 <= d <= end]
    arcs = sorted(set([*regular, *knots, float(end)]))
    require(len(arcs) <= MAX_PROBES, 'Fixed photometric probe budget exceeded')
    return arcs


def _snapshot(array, shape, dtype=None, integer=False):
    require(isinstance(array, np.ndarray) and array.shape == shape,
            'Exact native buffer shape required')
    require((dtype is None or array.dtype == dtype)
            and (not integer or np.issubdtype(array.dtype, np.integer)),
            'Exact native buffer dtype required')
    # Bytes-backed arrays cannot have writeability re-enabled. No aliasing caller
    # arrays, dtype conversion, resampling, or depth reconstruction.
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(shape)


class _Invocation:
    """Private, non-injectable, one-call context; never a cross-frame cache."""
    def __init__(self, metadata, report, rgb, depth, valid, components, catalogue, target_mask):
        self.meta, self.report, self.catalogue = map(deepcopy, (metadata, report, catalogue))
        shape = (RESOLUTION[1], RESOLUTION[0])
        self.rgb = _snapshot(rgb, (*shape, 3), np.uint8)
        self.depth = _snapshot(depth, shape, np.float32)
        self.valid = _snapshot(valid, shape, np.bool_)
        self.components = _snapshot(components, shape, integer=True)
        self.target_mask = _snapshot(target_mask, shape, np.uint8)
        self.args = (self.meta, self.report, self.rgb, self.depth, self.valid,
                     self.components, self.catalogue)
        legacy_review.check_native_evidence(self.meta, self.rgb, self.depth,
                                             self.components, self.target_mask, self.catalogue)
        self.baseline = legacy_labels.derive(*self.args)
        sup = self.meta['supervision']
        self.variant, self.key = sup['target_id'].split('/')
        self.component = self.report['components'][self.key]
        targets = [c for c in self.catalogue if c['variant_id'] == self.variant and c['component_id'] == self.key]
        require(len(targets) == 1, 'Unique exact native target required')
        self.index = targets[0]['component_index']
        self.mask = self.components == self.index
        self.chain, self.lengths, _, error = _oriented_chain(self.component, 1e-6)
        require(error <= 1e-6, 'Invalid anatomical attachment')
        self.at_cache, self.photo_cache = {}, {}
        self.photo_hits = 0
        self.visibility = None

    def at(self, arc):
        arc = float(arc)
        if arc not in self.at_cache:
            geometry = _sample(self.chain, self.lengths, arc, self.component['translation_plant_m'])
            world = transform_points([geometry['point_plant_m']],
                                     self.meta['supervision']['plant_to_world_usd_row_vectors'])[0]
            projected = project([world], self.meta['calibration'])[0]
            evidence = depth_evidence(projected, self.depth, self.valid, geometry['petiole_radius_m'])
            visible = False
            if projected['projection_status'] == 'in_frame':
                x, y = np.floor(projected['pixel_xy']).astype(int)
                visible = bool(self.mask[y, x] and evidence['status'] == 'depth_consistent_not_visibility_verified')
            self.at_cache[arc] = dict(arc_m=arc, world_m=list(world), projected=projected,
                                     visible=visible, depth_status=evidence['status'])
        return self.at_cache[arc]

    def photo(self, arc):
        if arc in self.photo_cache:
            self.photo_hits += 1
        else:
            probe = self.at(arc)
            projected = probe['projected']
            if projected['projection_status'] != 'in_frame':
                checked = dict(passed=False, reasons=['fixed_grid_projection_not_in_frame'])
            else:
                checked = self.visibility.inspect([float(v) for v in projected['pixel_xy']])
            self.photo_cache[arc] = dict(arc_m=float(arc), pixel_uv=projected['pixel_xy'],
                                        usability=checked, passed=checked['passed'])
        return self.photo_cache[arc]

    def fixed_review(self, end):
        arcs = fixed_photometric_arcs(self.lengths, float(end))
        probes = [deepcopy(self.photo(arc)) for arc in arcs]
        failures = [p for p in probes if not p['passed']]
        return dict(schema=GRID_SCHEMA, passed=not failures, arc_interval_m=[.008, float(end)],
                    regular_anchor_mm=START_MM, regular_step_mm=STEP_MM,
                    anatomical_knots_included=True, exact_query_endpoint_included=True,
                    probe_count=len(probes), probes=probes, failures=failures,
                    reasons=sorted({reason for p in failures for reason in p['usability']['reasons']}),
                    continuous_visibility_verified=False)


def _joint_rejections(legacy, fixed):
    """Neither resampling nor fixed-grid success can bypass legacy failures."""
    reasons = []
    if legacy['passed'] is not True:
        reasons.append('legacy_full_trace_failed')
    if fixed['passed'] is not True:
        reasons.append('anchored_photometric_grid_failed')
    return reasons


def select_query(metadata, report, rgb, depth, valid, components, catalogue, *,
                 target_mask, expected_label=None):
    """Return new selection evidence, never a modified/native-saved label.

    report must come from the existing validated source catalogue/audit API.
    File/source/worker ownership remains the caller's responsibility; this API
    checks explicit native arrays and frozen derivation, not capture receipts.
    expected_label, if supplied, must equal the complete fresh legacy derivation.

    All 41 original candidate arcs are enumerated after upstream legacy gates.
    A legacy upstream rejection is never rescued. The no-query rejection is
    enumerated diagnostically and must still contain zero usable candidates.
    """
    require(legacy_review.POLICY['minimum_trace_arc_m'] == .008
            and legacy_review.POLICY['local_photometric_probe_step_m'] == .005,
            'Unexpected legacy trace policy; new selector version required')
    ctx = _Invocation(metadata, report, rgb, depth, valid, components, catalogue, target_mask)
    return _select(ctx, expected_label=expected_label)


def _select(ctx, *, expected_label=None):
    require(legacy_review.POLICY['minimum_trace_arc_m'] == .008
            and legacy_review.POLICY['local_photometric_probe_step_m'] == .005,
            'Unexpected legacy trace policy; new selector version required')
    baseline = ctx.baseline
    if expected_label is not None:
        require(expected_label == baseline, 'Saved label differs from fresh legacy derivation')
    output = dict(schema=SCHEMA, annotation_epoch=ANNOTATION_EPOCH,
        task=baseline['task'], contract_sha256=contract_hash(),
        target_id=ctx.meta['supervision']['target_id'], sample_id=ctx.meta['sample_id'],
        scope='new_query_selection_evidence_not_relabel_or_admission',
        policy=dict(candidate_rule='exact_legacy_linspace_41_45mm_to_min_85pct_250mm',
                    minimum_query_arc_m=MIN_QUERY_M, minimum_query_cut_distance_px=MIN_QUERY_CUT_PX,
                    legacy_full_trace_required=True, anchored_grid_required=True,
                    fixed_grid_schema=GRID_SCHEMA, anchor_mm=START_MM, step_mm=STEP_MM,
                    anatomical_knots=True, query_endpoint=True,
                    legacy_trace_policy=deepcopy(legacy_review.POLICY), local_query_policy=deepcopy(QUERY_POLICY)),
        baseline_label_eligible=baseline['eligible'], baseline_reason=baseline['reason'],
        expected_label_verified=expected_label is not None, baseline_label_sha256=fingerprint(baseline),
        input_fingerprints=dict(metadata=fingerprint(ctx.meta), report=fingerprint(ctx.report),
            catalogue=fingerprint(ctx.catalogue), **{
                name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in
                (('rgb', ctx.rgb), ('depth', ctx.depth), ('valid', ctx.valid),
                 ('components', ctx.components), ('target_mask', ctx.target_mask))}),
        candidates=[], candidate_count=0, locally_usable_count=0, legacy_trace_pass_count=0,
        fixed_grid_pass_count=0, both_pass_count=0, selected_query=None,
        candidate_scan_performed=False, native_callback_and_target_mask_verified=True,
        training_approved=False, physical_execution_approved=False, source_cap_reset=False,
        visual_review_performed=False, continuous_visibility_verified=False,
        existing_label_modified=False, native_depth_reconstructed=False,
        source_catalogue_and_owned_capture_verification_required_by_caller=True)
    if not baseline['eligible'] and baseline['reason'] != 'no_usable_visible_query_connected_to_cut':
        output.update(state='legacy_upstream_rejection_preserved', rejection_counts={baseline['reason']: 1})
        return output
    ctx.visibility = NativeQueryVisibility(ctx.rgb, ctx.mask, RESOLUTION)
    nominal = np.asarray(baseline['nominal_pixel_uv'])
    nx, ny = np.floor(nominal).astype(int)
    connected = int(ctx.visibility.components[ny, nx])
    records = []
    for index, distance in enumerate(np.linspace(.045, min(float(ctx.lengths[-1])*.85, .25), 41)):
        p = deepcopy(ctx.at(float(distance)))
        row = dict(candidate_index=index, arc_m=float(distance), query_evidence=p,
                   query_pixel_uv=None, query_usability=None, local_passed=False,
                   legacy_trace=None, fixed_grid=None, both_passed=False, rejections=[])
        reasons = row['rejections']
        if distance < .045:
            reasons.append('below_minimum_query_arc')
        elif not p['visible']:
            reasons.append('projection_not_in_frame' if p['projected']['projection_status'] != 'in_frame'
                           else 'native_identity_or_depth_failed')
        else:
            q = [round(float(v), 1) for v in p['projected']['pixel_xy']]
            row['query_pixel_uv'] = q
            if not all(0 <= v < bound for v, bound in zip(q, RESOLUTION)):
                reasons.append('rounded_query_outside_frame')
            else:
                x, y = np.floor(q).astype(int)
                if not connected or int(ctx.visibility.components[y, x]) != connected:
                    reasons.append('different_exact_target_island')
                elif np.linalg.norm(np.asarray(q)-nominal) < 18:
                    reasons.append('query_cut_distance_below_18px')
                else:
                    checked = ctx.visibility.inspect(q)
                    row['query_usability'] = checked
                    if not checked['passed']:
                        reasons.append('local_query_usability_failed')
                    else:
                        row['local_passed'] = True
                        # No injected checker, patched global, or reusable label.
                        trial = deepcopy(baseline)
                        trial.update(query_pixel_uv=deepcopy(q), query_evidence=deepcopy(p),
                                     query_usability=deepcopy(checked))
                        row['legacy_trace'] = legacy_review.trace_review(
                            ctx.meta, ctx.report, trial, ctx.rgb, ctx.depth,
                            ctx.valid, ctx.components, ctx.catalogue)
                        row['fixed_grid'] = ctx.fixed_review(float(distance))
                        reasons.extend(_joint_rejections(row['legacy_trace'], row['fixed_grid']))
                        row['both_passed'] = not reasons
        records.append(row)
    local = [r for r in records if r['local_passed']]
    if not baseline['eligible']:
        require(not local, 'Frozen no-query rejection and enumeration disagree')
    else:
        selected_legacy = [r for r in local if r['query_evidence'] == baseline['query_evidence']
                           and r['query_pixel_uv'] == baseline['query_pixel_uv']
                           and r['query_usability'] == baseline['query_usability']]
        require(len(selected_legacy) == 1, 'Original selected candidate missing or changed')
    passing = [r for r in records if r['both_passed']]
    seed = hashlib.sha256((SCHEMA+'\0'+output['target_id']+'\0'+output['sample_id']).encode()).digest()
    selected = passing[int.from_bytes(seed[:8], 'little') % len(passing)] if passing else None
    output.update(state='query_selected_pending_future_integration' if selected else 'no_query_passes_both_checks',
        candidates=records, candidate_count=len(records), candidate_scan_performed=True,
        locally_usable_count=len(local), legacy_trace_pass_count=sum(r['legacy_trace']['passed'] for r in local),
        fixed_grid_pass_count=sum(r['fixed_grid']['passed'] for r in local), both_pass_count=len(passing),
        rejection_counts=dict(Counter(reason for r in records for reason in r['rejections'])),
        selection_rule='sha256(schema_NUL_target_NUL_sample)_first8_little_endian_mod_passing_in_candidate_order',
        selected_query=deepcopy(selected), context=dict(scope='immutable_invocation_only',
            shared_visibility_precomputations=1, legacy_trace_checker_not_replaced=True,
            fixed_unique_arcs=len(ctx.photo_cache), fixed_cache_hits=ctx.photo_hits))
    return output


def annotate_v2(metadata, report, rgb, depth, valid, components, catalogue, *,
                target_mask, expected_label=None):
    '''Return (new_label, composite_trace_or_None, selection_evidence) once.

    Nothing is persisted. A successful query gets a fresh label and composite
    trace requiring BOTH checks. If no query passes but the frozen label is
    locally eligible, retain its original query with a FAILED composite trace
    (hold, not exclude). Frozen exclusions keep eligible=False and trace=None.
    Never route v2 labels through frozen equality or disguise the new epoch.
    '''
    ctx = _Invocation(metadata, report, rgb, depth, valid, components, catalogue, target_mask)
    selection = _select(ctx, expected_label=expected_label)
    label = deepcopy(ctx.baseline)
    label['annotation_epoch'] = ANNOTATION_EPOCH
    trace = None
    if label['eligible']:
        chosen = selection['selected_query']
        if chosen is not None:
            label.update(query_pixel_uv=deepcopy(chosen['query_pixel_uv']),
                         query_evidence=deepcopy(chosen['query_evidence']),
                         query_usability=deepcopy(chosen['query_usability']))
        else:
            chosen = next(r for r in selection['candidates']
                          if r['query_evidence'] == ctx.baseline['query_evidence']
                          and r['query_pixel_uv'] == ctx.baseline['query_pixel_uv'])
            require(not chosen['both_passed'], 'No-pass fallback cannot pass')
        passed = chosen['legacy_trace']['passed'] is True and chosen['fixed_grid']['passed'] is True
        require(passed == (selection['selected_query'] is not None), 'Selection/trace disagreement')
        trace = dict(schema='greenhouse.native_query_trace.v2', annotation_epoch=ANNOTATION_EPOCH,
            passed=passed, reasons=[] if passed else
                ['no_query_passes_legacy_and_anchored_photometry', *chosen['rejections']],
            query_pixel_uv=deepcopy(label['query_pixel_uv']), query_arc_m=label['query_evidence']['arc_m'],
            legacy_trace=deepcopy(chosen['legacy_trace']), fixed_grid=deepcopy(chosen['fixed_grid']),
            candidate_count=selection['candidate_count'], both_pass_count=selection['both_pass_count'],
            original_query_retained_as_failed_fallback=selection['selected_query'] is None,
            training_approved=False, continuous_visibility_verified=False, native_depth_reconstructed=False)
        require(label['answer'] == ctx.baseline['answer'], 'Cut answer changed')
    return label, trace, selection
