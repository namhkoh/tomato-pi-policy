"""Fresh native848 curved-target label derivation with unchanged pixel gates.

Separate from frozen1696 and legacy release consumers. Recomputes nominal10mm,
curved10-20mm interval, proximal visibility, parent context and distal query.
A candidate also needs full centerline trace, independent native audit, actual
workspace proof, individual visual review and global duplicate/source grouping.
No target novelty, release, model export or physical cutting approval.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from scipy.ndimage import distance_transform_edt
from .capture_contract import project,transform_points,depth_evidence
from .cut_regions import _oriented_chain,_sample
from .dataset_review import require
from .capture_sensor import LEGACY_RESOLUTION,normalized_coordinates
from .clear_cutpoint_contract import POLICY as LEGACY_POLICY
from .native_query_visibility import NativeQueryVisibility,POLICY as QUERY_POLICY
from .native_clear_labels import interval_arc_samples
from .automated_native_review import POLICY as TRACE_POLICY,trace_review

RESOLUTION=LEGACY_RESOLUTION
TASK='greenhouse.native848_curved_clear_cutpoint_rgb.v1'
POLICY=deepcopy(LEGACY_POLICY)
CONTRACT=dict(task=TASK,resolution=list(RESOLUTION),nominal_arc_m=.01,accepted_arc_m=[.01,.02],
    clear_policy=POLICY,query_policy=QUERY_POLICY,trace_policy=TRACE_POLICY,
    minimum_distal_query_arc_m=.045,minimum_query_to_cut_distance_px=18.,
    minimum_visible_parent_pixels=6,minimum_cut_to_parent_distance_px=3.,
    pixel_threshold_basis='unchanged_absolute_native_pixel_legibility_limits',
    training_approved=False,source_cap_reset=False,model_export_qualified=False)

def contract_hash():
    return hashlib.sha256(json.dumps(CONTRACT,sort_keys=True,allow_nan=False).encode()).hexdigest()

def validate_answer(value):
    require(isinstance(value,dict) and set(value)=={'status','cut_point_uv','visibility','next_action'},'Exact answer fields required')
    require(value['status']=='localized' and value['visibility']=='clear' and value['next_action']=='inspect_cut_region','Clear candidate answer required')
    point=value['cut_point_uv']
    require(isinstance(point,list) and len(point)==2 and all(type(v) in (int,float) for v in point),'Numeric native848 answer required')
    normalized_coordinates(point,RESOLUTION)
    return value

def clear_screen(rgb,mask,interval,proximal_fraction):
    require(rgb.shape==(408,848,3) and rgb.dtype==np.uint8 and mask.shape==(408,848)
            and mask.dtype==bool,"Native848 RGB and boolean target mask required")
    line=np.asarray(interval,float)
    require(line.ndim==2 and line.shape[1]==2 and len(line)>=2 and np.isfinite(line).all()
            and ((line>=0)&(line<RESOLUTION)).all(),"Invalid projected interval")
    xy=np.floor(line).astype(int)
    x,y=xy.T
    x0,y0=np.maximum(xy.min(axis=0)-8,0)
    x1,y1=np.minimum(xy.max(axis=0)+9,RESOLUTION)
    pixels=rgb[y0:y1,x0:x1][mask[y0:y1,x0:x1]].astype(float)
    luma=pixels @ np.asarray([.2126,.7152,.0722]) if len(pixels) else np.asarray([0.])
    result=dict(interval_length_px=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum()),
        width_proxy_px=float(np.median(2*distance_transform_edt(mask)[y,x])),
        local_median_luma=float(np.median(luma)),local_dark_fraction=float(np.mean(luma<30)),
        all_interval_samples_on_target=bool(mask[y,x].all()),proximal_visible_fraction=float(proximal_fraction))
    reasons=[]
    for metric,bound in (("interval_length_px","minimum_interval_px"),("width_proxy_px","minimum_width_proxy_px"),
                         ("local_median_luma","minimum_local_median_luma"),
                         ("proximal_visible_fraction","minimum_proximal_visible_fraction")):
        if not np.isfinite(result[metric]) or result[metric]<POLICY[bound]: reasons.append(metric)
    if result["local_dark_fraction"]>POLICY["maximum_local_dark_fraction"]: reasons.append("local_dark_fraction")
    if not result["all_interval_samples_on_target"]: reasons.append("interval_not_fully_visible")
    return dict(result,passed=not reasons,reasons=reasons)


def derive(metadata, report, rgb, depth, valid, components, catalogue):
    sup,cal=metadata["supervision"],metadata["calibration"]
    require(cal["resolution"]==list(RESOLUTION) and cal["crop_resize"] is None
            and cal["depth_convention"]=="optical_axis_z_metres_not_ray_range","Exact native calibration required")
    require(metadata["synchronization"]["scene_unchanged_during_capture"] is True
            and metadata["synchronization"]["dynamic_recording_supported"] is False,"Static native capture required")
    require(rgb.shape==(408,848,3) and rgb.dtype==np.uint8 and depth.dtype==np.float32
            and depth.shape==valid.shape==components.shape==(408,848)
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
    parent_pixels=int(parent_mask[max(0,y-12):min(408,y+13),max(0,x-12):min(848,x+13)].sum())
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
    if not (0<=x<848 and 0<=y<408 and mask[y,x] and valid[y,x]):
        return dict(base,reason="rounded_nominal_outside_visible_target")
    return dict(base,eligible=True,reason="native848_curved_clear_candidate_pending_individual_review",
                query_pixel_uv=query,query_evidence=evidence,query_usability=usability,
                query_cut_visible_connection_verified=True,
                query_association_scope="8_connected_exact_native_target_mask_no_gap_filling",
                answer=answer)
