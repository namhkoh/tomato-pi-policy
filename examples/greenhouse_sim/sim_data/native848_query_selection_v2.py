"""Future-only native query selection with foreground separation.

Every candidate first passes the frozen v1 local, full-trace and anchored-grid
checks. This additional engineering screen is not human readability approval.
No saved label, held image, cut answer or renderer buffer is modified here.
"""
from copy import deepcopy
import numpy as np
from . import native848_query_selection_v1 as previous
from .capture_contract import fingerprint
from .dataset_review import require

SCHEMA = 'greenhouse.native848_query_selection.v2'
ANNOTATION_EPOCH = SCHEMA
POLICY = dict(schema='greenhouse.native848_query_foreground_clearance.v1',
    minimum_clearance_px=8.0, search_radius_px=32.0, comparable_depth_margin_m=0.005,
    coordinate_convention='integer_native_pixel_sample_locations_no_resampling',
    query_depth_source='actual_native_depth_at_floor_query_pixel',
    competitor='all_valid_positive_finite_depth_non_target_pixels_at_query_depth_plus_margin_or_nearer',
    invalid_non_target_depth_within_minimum_clearance='hold_candidate',
    rank='descending_clipped_clearance_then_interior_then_contrast_then_ascending_candidate_index',
    scope='engineering_screen_not_human_or_vlm_readability_validation')


def foreground_clearance(query, depth, valid, target_mask):
    """Use actual buffers, including merged foreign mesh IDs via non-target mask."""
    require(depth.shape == valid.shape == target_mask.shape == (408,848)
            and depth.dtype == np.float32 and valid.dtype == np.bool_
            and target_mask.dtype == np.bool_, 'Exact native clearance buffers required')
    uv = np.asarray(query, dtype=float)
    require(uv.shape == (2,) and np.isfinite(uv).all()
            and 0 <= uv[0] < 848 and 0 <= uv[1] < 408, 'Native query required')
    x,y = np.floor(uv).astype(int)
    require(target_mask[y,x] and valid[y,x] and np.isfinite(depth[y,x])
            and depth[y,x] > 0, 'Query lacks actual target depth')
    radius = POLICY['search_radius_px']
    x0,x1=max(0,int(np.floor(uv[0]-radius))),min(848,int(np.ceil(uv[0]+radius))+1)
    y0,y1=max(0,int(np.floor(uv[1]-radius))),min(408,int(np.ceil(uv[1]+radius))+1)
    yy,xx=np.mgrid[y0:y1,x0:x1]
    distance=np.hypot(xx-uv[0],yy-uv[1])
    local_depth=depth[y0:y1,x0:x1]
    foreign=~target_mask[y0:y1,x0:x1]
    known=valid[y0:y1,x0:x1] & np.isfinite(local_depth) & (local_depth>0)
    within=distance<=radius
    competitor=foreign & known & within & (local_depth<=float(depth[y,x])+POLICY['comparable_depth_margin_m'])
    unknown=foreign & ~known & (distance<POLICY['minimum_clearance_px'])
    nearest=None
    if competitor.any():
        indexed=np.where(competitor,distance,np.inf)
        iy,ix=np.unravel_index(indexed.argmin(),indexed.shape)
        clearance=float(indexed[iy,ix])
        nearest=dict(pixel_uv=[int(xx[iy,ix]),int(yy[iy,ix])],depth_m=float(local_depth[iy,ix]),
                     distance_px=clearance)
    else:
        clearance=radius  # Censored lower bound; never claim infinite clearance.
    reasons=[]
    if clearance<POLICY['minimum_clearance_px']: reasons.append('nearby_foreign_foreground_geometry')
    if unknown.any(): reasons.append('unknown_non_target_depth_near_query')
    return dict(schema=POLICY['schema'],passed=not reasons,reasons=reasons,
        query_pixel_uv=list(map(float,uv)),query_depth_m=float(depth[y,x]),
        clearance_px=clearance,clearance_censored_at_search_radius=nearest is None,
        nearest_competitor=nearest,competitor_pixels_within_8px=int((competitor & (distance<8.)).sum()),
        unknown_depth_pixels_within_8px=int(unknown.sum()),policy=deepcopy(POLICY))


def _select(ctx, expected_label=None):
    old=previous._select(ctx,expected_label=expected_label)
    output=deepcopy(old)
    output.update(schema=SCHEMA,annotation_epoch=ANNOTATION_EPOCH,
        previous_selection_sha256=fingerprint(old),previous_selection_schema=old['schema'],
        previous_selected_candidate_index=(old['selected_query'] or {}).get('candidate_index'),
        foreground_policy=deepcopy(POLICY),foreground_pass_count=0,
        selection_rule=POLICY['rank'],selected_query=None)
    passing=[]
    for row in output['candidates']:
        row['foreground_clearance']=None
        row['all_checks_passed']=False
        if row['both_passed']:
            check=foreground_clearance(row['query_pixel_uv'],ctx.depth,ctx.valid,ctx.mask)
            row['foreground_clearance']=check
            row['all_checks_passed']=check['passed']
            if check['passed']: passing.append(row)
    if passing:
        chosen=min(passing,key=lambda r:(-r['foreground_clearance']['clearance_px'],
            -r['query_usability']['interior_radius_px'],-r['query_usability']['local_contrast_8bit'],
            r['candidate_index']))
        output.update(selected_query=deepcopy(chosen),state='query_selected_pending_future_visual_QA')
    elif old['candidate_scan_performed']:
        output['state']='no_query_passes_all_existing_checks_and_foreground_clearance'
    output['foreground_pass_count']=len(passing)
    return output


def select_query(metadata,report,rgb,depth,valid,components,catalogue,*,target_mask,expected_label=None):
    ctx=previous._Invocation(metadata,report,rgb,depth,valid,components,catalogue,target_mask)
    return _select(ctx,expected_label)


def annotate_v2(metadata,report,rgb,depth,valid,components,catalogue,*,target_mask,expected_label=None):
    ctx=previous._Invocation(metadata,report,rgb,depth,valid,components,catalogue,target_mask)
    selection=_select(ctx,expected_label)
    label=deepcopy(ctx.baseline)
    label['annotation_epoch']=ANNOTATION_EPOCH
    trace=None
    if label['eligible']:
        chosen=selection['selected_query']
        selected=chosen is not None
        if selected:
            label.update(query_pixel_uv=deepcopy(chosen['query_pixel_uv']),
                query_evidence=deepcopy(chosen['query_evidence']),query_usability=deepcopy(chosen['query_usability']))
        else:
            chosen=next(r for r in selection['candidates'] if r['query_evidence']==ctx.baseline['query_evidence']
                        and r['query_pixel_uv']==ctx.baseline['query_pixel_uv'])
        passed=bool(selected and chosen['both_passed'] and chosen['all_checks_passed'])
        require(passed==selected,'Selected query failed preserved or new checks')
        trace=dict(schema='greenhouse.native848_query_trace.v2',annotation_epoch=ANNOTATION_EPOCH,
            passed=passed,reasons=[] if passed else ['no_query_passes_preserved_checks_and_foreground_clearance'],
            query_pixel_uv=deepcopy(label['query_pixel_uv']),query_arc_m=label['query_evidence']['arc_m'],
            legacy_trace=deepcopy(chosen['legacy_trace']),fixed_grid=deepcopy(chosen['fixed_grid']),
            foreground_clearance=deepcopy(chosen['foreground_clearance']),
            candidate_count=selection['candidate_count'],both_pass_count=selection['both_pass_count'],
            foreground_pass_count=selection['foreground_pass_count'],
            original_query_retained_as_failed_fallback=not selected,
            training_approved=False,continuous_visibility_verified=False,native_depth_reconstructed=False)
    require(label.get('answer')==ctx.baseline.get('answer'),'Cut answer changed')
    return label,trace,selection
