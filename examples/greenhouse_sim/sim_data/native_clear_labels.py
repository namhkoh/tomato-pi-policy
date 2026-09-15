"""Recompute native clear-only labels from anatomy and exact saved Isaac buffers."""
import hashlib
import numpy as np
from .capture_contract import project, transform_points, depth_evidence
from .cut_regions import _oriented_chain, _sample
from .dataset_review import require
from .native_clear_contract import RESOLUTION,TASK,CONTRACT,contract_hash,clear_screen,validate_answer
from .native_query_visibility import NativeQueryVisibility


def interval_arc_samples(cumulative, low=.01, high=.02, maximum_step=.001):
    """Retain anatomical bends and match native per-segment subdivision.

    The straight legacy interval has11 probes; curved intervals can have more.
    Saved world labels are never resampled to conceal an anatomical mismatch.
    """
    cumulative=np.asarray(cumulative,float)
    require(cumulative.ndim==1 and len(cumulative)>=2 and np.isfinite(cumulative).all()
            and cumulative[0]==0 and np.all(np.diff(cumulative)>0),"Invalid cumulative anatomy")
    require(0<=low<high<=cumulative[-1] and 0<maximum_step<=.001,"Invalid cut sampling bounds")
    knots=[low]+[float(s) for s in cumulative if low<s<high]+[high]
    if len(knots)==2:
        steps=max(1,int(np.ceil((high-low)/maximum_step)))
        require(steps<4096,"Excessively dense cut interval")
        # Preserve the exact legacy floating-point samples for straight intervals.
        return [float(d) for d in np.linspace(low,high,steps+1)]
    result=[]
    for a,b in zip(knots,knots[1:]):
        steps=max(1,int(np.ceil((b-a)/maximum_step)))
        require(len(result)+steps<4096,"Excessively dense cut interval")
        result.extend(float(a+(b-a)*t) for t in np.linspace(0,1,steps,endpoint=False))
    return [*result,high]


def derive(metadata, report, rgb, depth, valid, components, catalogue):
    sup,cal=metadata["supervision"],metadata["calibration"]
    require(cal["resolution"]==list(RESOLUTION) and cal["crop_resize"] is None
            and cal["depth_convention"]=="optical_axis_z_metres_not_ray_range","Exact native calibration required")
    require(metadata["synchronization"]["scene_unchanged_during_capture"] is True
            and metadata["synchronization"]["dynamic_recording_supported"] is False,"Static native capture required")
    require(rgb.shape==(816,1696,3) and rgb.dtype==np.uint8 and depth.dtype==np.float32
            and depth.shape==valid.shape==components.shape==(816,1696)
            and valid.dtype==bool and np.issubdtype(components.dtype,np.integer),"Invalid native input buffers")
    near,far=cal["clipping_range_m"]
    require(np.array_equal(valid,np.isfinite(depth)&(depth>0)&(depth>=near)&(depth<=far)),
            "Native validity disagrees with clipping/depth")
    variant,component_id=sup["target_id"].split("/")
    component=report["components"][component_id]
    require(component["type"]=="sub_stem","Cut target must be petiole, not main stem/fruit")
    targets=[c for c in catalogue if c["variant_id"]==variant and c["component_id"]==component_id]
    parents=[c for c in catalogue if c["variant_id"]==variant and c["component_id"]==component["parent"]]
    require(len(targets)==len(parents)==1,"Exact native target/parent identity required")
    target,parent=targets[0],parents[0]
    require(target["component_index"]!=parent["component_index"]
            and report["components"][component["parent"]]["type"]=="main_stem","Invalid parent context")
    mask=components==target["component_index"]
    chain,lengths,_,_=_oriented_chain(component,1e-6)
    origin,matrix=component["translation_plant_m"],sup["plant_to_world_usd_row_vectors"]

    def probe(distance,allowed):
        g=_sample(chain,lengths,float(distance),origin)
        world=transform_points([g["point_plant_m"]],matrix)[0]
        projected=project([world],cal)[0]
        evidence=depth_evidence(projected,depth,valid,g["petiole_radius_m"])
        visible=False
        if projected["projection_status"]=="in_frame":
            x,y=np.floor(projected["pixel_xy"]).astype(int)
            visible=int(components[y,x]) in allowed and evidence["status"]=="depth_consistent_not_visibility_verified"
        return dict(arc_m=float(distance),world_m=list(world),projected=projected,
                    visible=bool(visible),depth_status=evidence["status"])

    require(lengths[-1]>=.030,"Petiole too short for proximal evidence")
    nominal=probe(.01,{target["component_index"]})
    interval=[probe(d,{target["component_index"]}) for d in interval_arc_samples(lengths)]
    expected_interval=np.asarray([p["world_m"] for p in interval],float)
    captured_interval=np.asarray(sup["interval_world_m"],float)
    require(np.allclose(nominal["world_m"],sup["nominal_world_m"],atol=1e-8,rtol=0)
            and captured_interval.shape==expected_interval.shape
            and np.allclose(expected_interval,captured_interval,atol=1e-8,rtol=0),
            "Captured cut geometry differs from recomputed10-20mm anatomy")
    require(nominal["projected"]["projection_status"]==sup["nominal_projected"]["projection_status"],
            "Nominal projection status changed")
    if nominal["projected"]["projection_status"]=="in_frame":
        require(np.allclose(nominal["projected"]["pixel_xy"],sup["nominal_projected"]["pixel_xy"],atol=1e-5,rtol=0),
                "Captured pixel annotation is stale")
    base=dict(task=TASK,contract_sha256=contract_hash(),eligible=False,training_approved=False,
        visual_review_performed=False,target_id=sup["target_id"],source_plant_family=sup["split_group"],
        conservative_view_cap_group=sup.get("source_target_id",sup["target_id"]),
        resolution=list(RESOLUTION),nominal_world_m=nominal["world_m"],
        nominal_pixel_uv=nominal["projected"].get("pixel_xy"),
        accepted_interval_uv=[p["projected"].get("pixel_xy") for p in interval],
        native_depth_reconstructed=False,hidden_cut_coordinates_executable=False)
    if not nominal["visible"] or not all(p["visible"] for p in interval):
        return dict(base,reason="cut_interval_not_fully_native_visible")
    proximal=[probe(d,{target["component_index"]} if d>=.008 else
                    {target["component_index"],parent["component_index"]}) for d in np.linspace(.004,.030,27)]
    unique={}
    for p in proximal:
        if p["projected"]["projection_status"]=="in_frame":
            unique.setdefault(tuple(np.floor(p["projected"]["pixel_xy"]).astype(int)),[]).append(p["visible"])
    fraction=sum(all(v) for v in unique.values())/len(unique) if unique else 0.
    screen=clear_screen(rgb,mask,base["accepted_interval_uv"],fraction)
    base.update(clarity=screen,proximal_evidence=proximal)
    if not screen["passed"]:
        return dict(base,reason="strict_native_clarity_failed")
    attach=project(transform_points([component["attachment_plant_m"]],matrix),cal)[0]
    if attach["projection_status"]!="in_frame":
        return dict(base,reason="attachment_out_of_frame")
    x,y=np.floor(attach["pixel_xy"]).astype(int)
    parent_mask=components==parent["component_index"]
    parent_pixels=int(parent_mask[max(0,y-12):min(816,y+13),max(0,x-12):min(1696,x+13)].sum())
    yy,xx=np.nonzero(parent_mask)
    uv=np.asarray(base["nominal_pixel_uv"])
    distance=float(np.min(np.hypot(xx+.5-uv[0],yy+.5-uv[1]))) if len(xx) else None
    base.update(attachment_pixel_uv=attach["pixel_xy"],attachment_parent_visible_pixels=parent_pixels,
                cut_to_visible_parent_px=distance)
    diameter=metadata["quality"]["estimated_petiole_diameter_px"]
    if parent_pixels<6 or distance is None or distance<max(3.,diameter*.65):
        return dict(base,reason="insufficient_visible_parent_clearance")
    query_screen=NativeQueryVisibility(rgb,mask,RESOLUTION)
    nx,ny=np.floor(uv).astype(int)
    connected=int(query_screen.components[ny,nx])
    candidates=[]
    for d in np.linspace(.045,min(float(lengths[-1])*.85,.25),41):
        if d<.045:continue
        p=probe(d,{target["component_index"]})
        if not p["visible"]:continue
        q=[round(float(v),1) for v in p["projected"]["pixel_xy"]]
        if not all(0<=v<bound for v,bound in zip(q,RESOLUTION)):continue
        x,y=np.floor(q).astype(int)
        if not connected or int(query_screen.components[y,x])!=connected or np.linalg.norm(np.asarray(q)-uv)<18:continue
        usability=query_screen.inspect(q)
        if usability["passed"]:candidates.append((p,q,usability))
    if not candidates:
        return dict(base,reason="no_usable_visible_query_connected_to_cut")
    index=int.from_bytes(hashlib.sha256((sup["target_id"]+metadata["sample_id"]).encode()).digest()[:4],"little")%len(candidates)
    evidence,query,usability=candidates[index]
    answer=dict(status="localized",cut_point_uv=[round(float(v),2) for v in uv],
                visibility="clear",next_action="inspect_cut_region")
    validate_answer(answer)
    x,y=np.floor(answer["cut_point_uv"]).astype(int)
    if not (0<=x<1696 and 0<=y<816 and mask[y,x] and valid[y,x]):
        return dict(base,reason="rounded_nominal_outside_visible_target")
    return dict(base,eligible=True,reason="native_clear_candidate_pending_individual_review",
                query_pixel_uv=query,query_evidence=evidence,query_usability=usability,
                query_cut_visible_connection_verified=True,
                query_association_scope="8_connected_exact_native_target_mask_no_gap_filling",
                answer=answer)
