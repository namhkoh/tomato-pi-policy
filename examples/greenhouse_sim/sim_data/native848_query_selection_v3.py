"""Future-only native query selection with measured route separation.

Engineering screen, not globally calibrated human readability. Frozen v1
local/trace/grid gates remain mandatory. Only native renderer0 + positive
infinite depth + invalid depth + no mapped prim is authenticated empty background.
"""
from copy import deepcopy
import hashlib
import numpy as np
from . import native848_query_selection_v1 as previous
from .capture_contract import fingerprint
from .capture_visibility import component_masks
from .dataset_review import require

SCHEMA='greenhouse.native848_query_selection.v3'
ANNOTATION_EPOCH=SCHEMA
POLICY=dict(schema='greenhouse.native848_query_route_separation.v1',
    endpoint_foreground_clearance_px=8.,route_foreground_clearance_px=6.,
    route_visible_foreign_plant_clearance_px=6.,search_radius_px=32.,
    comparable_depth_margin_m=.005,route_start_m=.008,
    empty_background='renderer_ID0_and_positive_infinite_depth_and_invalid_and_ID0_has_no_mapped_prim',
    unknown_surface='all_other_invalid_non_target_depth',
    visible_plant_identity='actual_renderer_prim_under_/World/PackPlants/_regardless_of_depth',
    parent_treatment='exact_same_variant_direct_parent_component_only_until_existing_cut_interval_end_20mm; never_descendants; only_new_separation_exemption',
    rank='descending_minimum_route_plant_and_foreground_clearance_then_endpoint_clearance_then_interior_then_contrast_then_ascending_index',
    coordinate_convention='integer_native_pixel_sample_locations_no_resampling',
    threshold_status='engineering_screen_selected_after_pinned_four_held_five_clear_measurements_not_blind_or_global_calibration',
    scope='future_candidates_only_actual_sampled_visual_QA_still_required')


def trace_samples(ctx,end):
    """Exactly the frozen full-trace sampling, including perspective refinement."""
    p=previous.legacy_review.POLICY;start=p['minimum_trace_arc_m']
    require(start==POLICY['route_start_m'],'Changed frozen route start')
    breaks=sorted(set([start,end]+[float(x) for x in ctx.lengths if start<x<end]));arcs=[]
    for left,right in zip(breaks[:-1],breaks[1:]):
        a,b=ctx.at(left)['projected'],ctx.at(right)['projected']
        require(a['projection_status']==b['projection_status']=='in_frame','Route outside native image')
        length=np.linalg.norm(np.asarray(a['pixel_xy'])-b['pixel_xy'])
        count=int(np.ceil(max((right-left)/p['maximum_step_m'],length/p['maximum_projected_step_px'])))
        require(len(arcs)+count+1<=p['maximum_probes'],'Trace budget exceeded')
        arcs.extend(np.linspace(left,right,max(1,count)+1)[:-1].tolist())
    arcs.append(end)
    for _ in range(12):
        uv=np.asarray([ctx.at(x)['projected']['pixel_xy'] for x in arcs]);gaps=np.linalg.norm(np.diff(uv,axis=0),axis=1)
        bad=np.flatnonzero(gaps>p['maximum_projected_step_px']*(1+1e-9))
        if not len(bad):break
        require(len(arcs)+len(bad)<=p['maximum_probes'],'Trace budget exceeded')
        arcs=sorted(arcs+[(arcs[i]+arcs[i+1])/2 for i in bad])
    else:raise ValueError('Trace perspective refinement failed')
    return arcs,uv,float(gaps.max())


class Separation:
    def __init__(self,ctx,renderer_ids,renderer_mapping):
        require(isinstance(renderer_ids,np.ndarray) and renderer_ids.shape==(408,848)
                and renderer_ids.dtype==np.uint32,'Actual native uint32 renderer IDs required')
        self.ids=previous._snapshot(renderer_ids,(408,848),np.uint32)
        require(isinstance(renderer_mapping,dict) and all(isinstance(v,str) for v in renderer_mapping.values()),
                'Actual renderer-to-prim mapping required')
        self.mapping={int(k):v for k,v in renderer_mapping.items()}
        require(len(self.mapping)==len(renderer_mapping) and not(set(map(int,np.unique(self.ids)))-set(self.mapping)-{0,1}),
                'Ambiguous or missing renderer mapping')
        decoded,_,_=component_masks(self.ids,self.mapping,ctx.catalogue)
        require(np.array_equal(decoded,ctx.components),'Renderer IDs/mapping do not reconstruct exact native component mask')
        self.ctx=ctx
        self.known=ctx.valid & np.isfinite(ctx.depth) & (ctx.depth>0)
        self.empty=(self.ids==0)&np.isposinf(ctx.depth)&~ctx.valid if not self.mapping.get(0,'').startswith('/') else np.zeros((408,848),bool)
        self.plants=np.isin(self.ids,[k for k,v in self.mapping.items() if v.startswith('/World/PackPlants/')])
        parent=[c for c in ctx.catalogue if c['variant_id']==ctx.variant and c['component_id']==ctx.component['parent']]
        require(len(parent)==1,'Authenticated exact parent required')
        self.parent=ctx.components==parent[0]['component_index']
        self.parent_id=parent[0]['component_id']
        interval=ctx.meta['supervision']['cut_region_proposal']['accepted_centerline_interval']['arc_range_m']
        require(interval==[.01,.02],'Explicit original10–20mm cut interval required')
        self.parent_end=float(interval[1])

    def probe(self,uv,arc,*,endpoint=False):
        uv=np.asarray(uv,float);x,y=np.floor(uv).astype(int);ctx=self.ctx
        require(0<=x<848 and 0<=y<408 and ctx.mask[y,x] and self.known[y,x],'Actual target depth required')
        x0,x1=max(0,x-33),min(848,x+34);y0,y1=max(0,y-33),min(408,y+34)
        yy,xx=np.mgrid[y0:y1,x0:x1];distance=np.hypot(xx-uv[0],yy-uv[1]);region=np.s_[y0:y1,x0:x1]
        depth=ctx.depth[region];foreign=~ctx.mask[region];known=self.known[region]
        exempt=self.parent[region] if not endpoint and arc<=self.parent_end else np.zeros(foreign.shape,bool)
        valid_foreign=foreign & known & ~exempt & (distance<=32.)
        fg=valid_foreign & (depth<=float(ctx.depth[y,x])+.005)
        plants=valid_foreign & self.plants[region]
        unknown=foreign & ~known & ~self.empty[region] & (distance<(8. if endpoint else 6.))
        def nearest(mask):
            if not mask.any():return dict(distance_px=32.,censored=True)
            ds=np.where(mask,distance,np.inf);iy,ix=np.unravel_index(ds.argmin(),ds.shape)
            rid=int(self.ids[y0+iy,x0+ix])
            return dict(distance_px=float(ds[iy,ix]),censored=False,pixel_uv=[int(xx[iy,ix]),int(yy[iy,ix])],
                renderer_id=rid,prim_path=self.mapping.get(rid),depth_m=float(depth[iy,ix]),target_depth_m=float(ctx.depth[y,x]))
        foreground,visible=nearest(fg),nearest(plants);reasons=[]
        if foreground['distance_px']<(8. if endpoint else 6.):reasons.append('foreign_foreground_too_close')
        if not endpoint and visible['distance_px']<6.:reasons.append('visible_foreign_plant_too_close_to_route')
        if unknown.any():reasons.append('unknown_nonbackground_surface_depth')
        return dict(arc_m=float(arc),route_pixel_uv=list(map(float,uv)),passed=not reasons,reasons=reasons,
            foreground=foreground,visible_foreign_plant=visible,unknown_surface_pixels=int(unknown.sum()),
            authenticated_empty_background_pixels=int((self.empty[region]&(distance<(8. if endpoint else 6.))).sum()),
            parent_exemption_active=bool(not endpoint and arc<=self.parent_end))

    def candidate(self,row):
        endpoint=self.probe(row['query_pixel_uv'],row['arc_m'],endpoint=True)
        arcs,uv,maximum=trace_samples(self.ctx,row['arc_m'])
        require(len(arcs)==row['legacy_trace']['probe_count']
                and abs(maximum-row['legacy_trace']['maximum_projected_step_px'])<1e-9,'Frozen trace sampling differs')
        probes=[self.probe(p,a) for a,p in zip(arcs,uv)]
        failures=[p for p in probes if not p['passed']]
        def minimum(key):
            p=min(probes,key=lambda p:p[key]['distance_px'])
            return dict(arc_m=p['arc_m'],route_pixel_uv=p['route_pixel_uv'],**p[key])
        return dict(schema=POLICY['schema'],passed=endpoint['passed'] and not failures,
            endpoint=endpoint,minimum_route_foreground=minimum('foreground'),
            minimum_route_visible_foreign_plant=minimum('visible_foreign_plant'),
            route_start_m=.008,route_end_m=row['arc_m'],probe_count=len(arcs),
            route_arcs_sha256=fingerprint(arcs),route_projected_pixels_sha256=fingerprint(uv.tolist()),
            failure_probe_count=len(failures),first_failure_probes=failures[:8],
            reasons=sorted(set(endpoint['reasons']+[r for p in failures for r in p['reasons']])),
            authenticated_empty_background_probe_count=sum(p['authenticated_empty_background_pixels']>0 for p in probes),
            unknown_surface_probe_count=sum(p['unknown_surface_pixels']>0 for p in probes),
            exact_parent_component=self.parent_id,parent_exemption_arc_range_m=[.008,self.parent_end],
            original_trace_and_photometry_unchanged=True)


def _select(ctx,renderer_ids,renderer_mapping,expected_label=None):
    old=previous._select(ctx,expected_label=expected_label)
    separation=Separation(ctx,renderer_ids,renderer_mapping)
    out=deepcopy(old);out.update(schema=SCHEMA,annotation_epoch=SCHEMA,selected_query=None,
        previous_selection_schema=old['schema'],previous_selection_sha256=fingerprint(old),
        previous_selected_candidate_index=(old['selected_query'] or {}).get('candidate_index'),
        separation_policy=deepcopy(POLICY),separation_pass_count=0,selection_rule=POLICY['rank'],
        native_renderer_ids_sha256=hashlib.sha256(separation.ids.tobytes()).hexdigest(),
        native_renderer_mapping_sha256=fingerprint(separation.mapping))
    passing=[]
    for row in out['candidates']:
        row['route_separation']=None;row['all_checks_passed']=False
        if row['both_passed']:
            row['route_separation']=separation.candidate(row)
            row['all_checks_passed']=row['route_separation']['passed']
            if row['all_checks_passed']:passing.append(row)
    def rank(r):
        s=r['route_separation']
        return (-min(s['minimum_route_foreground']['distance_px'],s['minimum_route_visible_foreign_plant']['distance_px']),
            -s['endpoint']['foreground']['distance_px'],-r['query_usability']['interior_radius_px'],
            -r['query_usability']['local_contrast_8bit'],r['candidate_index'])
    if passing:out.update(selected_query=deepcopy(min(passing,key=rank)),state='query_selected_pending_actual_visual_QA')
    elif old['candidate_scan_performed']:out['state']='no_query_passes_preserved_checks_and_route_separation'
    out['separation_pass_count']=len(passing)
    return out


def select_query(metadata,report,rgb,depth,valid,components,catalogue,*,target_mask,renderer_ids,renderer_mapping,expected_label=None):
    ctx=previous._Invocation(metadata,report,rgb,depth,valid,components,catalogue,target_mask)
    return _select(ctx,renderer_ids,renderer_mapping,expected_label)


def annotate_v3(metadata,report,rgb,depth,valid,components,catalogue,*,target_mask,renderer_ids,renderer_mapping,expected_label=None):
    ctx=previous._Invocation(metadata,report,rgb,depth,valid,components,catalogue,target_mask)
    selection=_select(ctx,renderer_ids,renderer_mapping,expected_label)
    label=deepcopy(ctx.baseline);label['annotation_epoch']=SCHEMA;trace=None
    if label['eligible']:
        chosen=selection['selected_query'];selected=chosen is not None
        if selected:label.update(query_pixel_uv=deepcopy(chosen['query_pixel_uv']),query_evidence=deepcopy(chosen['query_evidence']),query_usability=deepcopy(chosen['query_usability']))
        else:chosen=next(r for r in selection['candidates'] if r['query_evidence']==ctx.baseline['query_evidence'] and r['query_pixel_uv']==ctx.baseline['query_pixel_uv'])
        passed=bool(selected and chosen['both_passed'] and chosen['all_checks_passed'])
        require(passed==selected,'Selected query failed mandatory gates')
        trace=dict(schema='greenhouse.native848_query_trace.v3',annotation_epoch=SCHEMA,passed=passed,
            reasons=[] if passed else ['no_query_passes_preserved_checks_and_route_separation'],
            query_pixel_uv=deepcopy(label['query_pixel_uv']),query_arc_m=label['query_evidence']['arc_m'],
            legacy_trace=deepcopy(chosen['legacy_trace']),fixed_grid=deepcopy(chosen['fixed_grid']),
            route_separation=deepcopy(chosen['route_separation']),candidate_count=selection['candidate_count'],
            both_pass_count=selection['both_pass_count'],separation_pass_count=selection['separation_pass_count'],
            original_query_retained_as_failed_fallback=not selected,training_approved=False,
            continuous_visibility_verified=False,native_depth_reconstructed=False)
    require(label.get('answer')==ctx.baseline.get('answer'),'Cut answer changed')
    return label,trace,selection
