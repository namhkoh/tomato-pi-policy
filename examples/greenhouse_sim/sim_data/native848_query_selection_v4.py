"""Fresh-frame, individually reviewed background exception; Q3 stays immutable."""
from copy import deepcopy
import hashlib
import numpy as np
from . import native848_query_selection_v3 as q3
from .capture_contract import fingerprint
from .dataset_review import require
SCHEMA='greenhouse.native848_query_selection.v4'
ANNOTATION_EPOCH=SCHEMA
POLICY=dict(schema='greenhouse.native848_conditional_background_separation.v1',
    preserved_q3_policy=deepcopy(q3.POLICY),minimum_background_depth_gap_m=.20,
    minimum_foreign_plant_centerline_distance_px=4.,
    all_violating_pixels_at_all_trace_probes_checked=True,
    rank='shortest_preserved_valid_query_then_ascending_candidate_index',
    scope='fresh_original_bulk_only_actual_individual_full_native_and_shorter_query_crop_required',
    threshold_status='uncalibrated_engineering_bounds_explicitly_authorized_20260917_no_global_readability_claim',
    q3_results_preserved=True,automatic_training_acceptance=False)


def background_pixels(distances,gaps):
    distances=np.asarray(distances,float);gaps=np.asarray(gaps,float)
    require(distances.shape==gaps.shape and distances.ndim==1 and len(distances)>0
        and np.isfinite(distances).all() and np.isfinite(gaps).all(),'Finite complete violating pixels required')
    return dict(violating_pixel_count=len(distances),
        minimum_centerline_distance_px=float(distances.min()),minimum_depth_gap_m=float(gaps.min()),
        passed=bool((distances>=4.).all() and (gaps>=.20).all()),
        all_violating_pixels_checked=True,pixel_distance_and_gap_sha256=fingerprint(
            dict(distance_px=distances.tolist(),depth_gap_m=gaps.tolist())))


class Separation(q3.Separation):
    def conditional_probe(self,uv,arc,*,endpoint=False):
        original=self.probe(uv,arc,endpoint=endpoint)
        out=dict(arc_m=float(arc),route_pixel_uv=list(map(float,uv)),q3_probe=original,
            protected_foreground_passed=not any(r!='visible_foreign_plant_too_close_to_route' for r in original['reasons']),
            background_exception_applied=False,conditional_background_passed=original['passed'],background_pixels=None)
        if original['passed']:return out
        if endpoint or original['reasons']!=['visible_foreign_plant_too_close_to_route']:return out
        uv=np.asarray(uv,float);x,y=np.floor(uv).astype(int);ctx=self.ctx
        x0,x1=max(0,x-33),min(848,x+34);y0,y1=max(0,y-33),min(408,y+34)
        yy,xx=np.mgrid[y0:y1,x0:x1];distance=np.hypot(xx-uv[0],yy-uv[1]);region=np.s_[y0:y1,x0:x1]
        exempt=self.parent[region] if arc<=self.parent_end else np.zeros(distance.shape,bool)
        offending=(~ctx.mask[region]) & self.known[region] & ~exempt & self.plants[region] & (distance<6.)
        require(offending.any(),'Q3 background violation has no corresponding actual pixels')
        proof=background_pixels(distance[offending],ctx.depth[region][offending]-float(ctx.depth[y,x]))
        out.update(background_exception_applied=proof['passed'],conditional_background_passed=proof['passed'],background_pixels=proof)
        return out

    def conditional_candidate(self,row):
        arcs,uv,maximum=q3.trace_samples(self.ctx,row['arc_m'])
        require(len(arcs)==row['legacy_trace']['probe_count']
            and abs(maximum-row['legacy_trace']['maximum_projected_step_px'])<1e-9,'Frozen trace sampling changed')
        endpoint=self.conditional_probe(row['query_pixel_uv'],row['arc_m'],endpoint=True)
        probes=[self.conditional_probe(p,a) for a,p in zip(arcs,uv)]
        require(row['route_separation']==super().candidate(row),'Saved Q3 candidate route no longer matches actual pixels')
        protected=endpoint['protected_foreground_passed'] and all(p['protected_foreground_passed'] for p in probes)
        passed=endpoint['conditional_background_passed'] and all(p['conditional_background_passed'] for p in probes)
        exceptions=[p for p in probes if p['background_exception_applied']]
        return dict(schema=POLICY['schema'],passed=bool(passed),protected_foreground_passed=bool(protected),
            conditional_background_passed=bool(passed),Q1passed=row['both_passed'],
            q3_passed=row['route_separation']['passed'],original_q3_route=deepcopy(row['route_separation']),
            endpoint=endpoint,probes=probes,probe_count=len(probes),
            route_arcs_sha256=fingerprint(arcs),route_projected_pixels_sha256=fingerprint(uv.tolist()),
            exception_probe_count=len(exceptions),background_exception_applied=bool(exceptions),
            all_route_probes_recorded=True,all_violating_pixels_at_each_probe_checked=True,
            actual_individual_visual_review_required=True,training_approved=False)


def annotate_v4(metadata,report,rgb,depth,valid,components,catalogue,*,target_mask,renderer_ids,renderer_mapping,
                expected_q3_label,expected_q3_trace,expected_q3_selection):
    ctx=q3.previous._Invocation(metadata,report,rgb,depth,valid,components,catalogue,target_mask)
    old_label,old_trace,old_selection=q3.annotate_v3(metadata,report,rgb,depth,valid,components,catalogue,
        target_mask=target_mask,renderer_ids=renderer_ids,renderer_mapping=renderer_mapping,expected_label=ctx.baseline)
    require(old_label==expected_q3_label and old_trace==expected_q3_trace and old_selection==expected_q3_selection,
        'Exact original Q3 replay differs; no exception allowed')
    selection=deepcopy(old_selection)
    selection.update(schema=SCHEMA,annotation_epoch=SCHEMA,original_q3_selection_sha256=fingerprint(old_selection),
        conditional_background_policy=deepcopy(POLICY),selected_query=None,conditional_pass_count=0,
        state='no_query_passes_conditional_background_policy',individual_visual_review_required=True)
    screen=Separation(ctx,renderer_ids,renderer_mapping);passing=[]
    for row in selection['candidates']:
        row['conditional_background']=None;row['conditional_checks_passed']=False
        if row['both_passed']:
            proof=screen.conditional_candidate(row);row['conditional_background']=proof
            row['conditional_checks_passed']=proof['passed']
            if proof['passed']:passing.append(row)
    if passing:
        chosen=deepcopy(min(passing,key=lambda r:(r['arc_m'],r['candidate_index'])))
        selection.update(selected_query=chosen,conditional_pass_count=len(passing),
            state='conditional_query_selected_pending_actual_individual_visual_review')
    label=deepcopy(old_label);label['annotation_epoch']=SCHEMA;trace=None
    if label['eligible']:
        chosen=selection['selected_query'];passed=chosen is not None
        if passed:
            label.update(query_pixel_uv=deepcopy(chosen['query_pixel_uv']),query_evidence=deepcopy(chosen['query_evidence']),
                query_usability=deepcopy(chosen['query_usability']))
            legacy,grid=chosen['legacy_trace'],chosen['fixed_grid'];proof=chosen['conditional_background']
        else:legacy,grid,proof=(old_trace or {}).get('legacy_trace'),(old_trace or {}).get('fixed_grid'),None
        trace=dict(schema='greenhouse.native848_query_trace.v4',annotation_epoch=SCHEMA,passed=passed,
            query_pixel_uv=deepcopy(label['query_pixel_uv']),query_arc_m=label['query_evidence']['arc_m'],
            legacy_trace=deepcopy(legacy),fixed_grid=deepcopy(grid),conditional_background=proof,
            original_q3_passed=bool(old_trace and old_trace['passed']),
            original_q3_trace_sha256=fingerprint(old_trace),original_query_retained_as_failed_fallback=not passed,
            individual_visual_review_required=True,training_approved=False)
    require(label.get('answer')==old_label.get('answer') and label['target_id']==old_label['target_id'],
        'Original cut answer/target changed')
    return label,trace,selection

