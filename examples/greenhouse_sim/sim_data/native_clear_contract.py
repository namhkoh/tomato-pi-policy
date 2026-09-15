"""Separate native1696 clear-task contract. No legacy export or training gate changes."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from .capture_sensor import HIRES_RESOLUTION, checked_resolution, normalized_coordinates
from .clear_cutpoint_contract import POLICY as LEGACY_CLEAR_POLICY
from .dataset_review import require
from .native_query_visibility import POLICY as QUERY_POLICY
from .qwen_coordinates import format_point

RESOLUTION = HIRES_RESOLUTION
TASK = "greenhouse.native_clear_cutpoint_rgb.v1"
POLICY = dict(deepcopy(LEGACY_CLEAR_POLICY), crop_source_size=768, crop_output_size=768)
CONTRACT = dict(task=TASK, resolution=list(RESOLUTION), nominal_arc_m=.01, accepted_arc_m=[.01,.02],
    clear_policy=POLICY, query_policy=QUERY_POLICY, minimum_distal_query_arc_m=.045,
    minimum_query_to_cut_distance_px=18., minimum_visible_parent_pixels=6,
    minimum_cut_to_parent_distance_px=3.,
    pixel_threshold_basis="same_absolute_native_pixel_legibility_limits_not_legacy_FOV",
    crop="native768_square_query_only_shifted_inside_full_frame_no_resize",
    coordinates="canonical_full_native_pixels_model_normalized_1000_2dp",
    model_input="clean_RGB_and_distal_query_only_optional_query_crop",
    unsupported=["training_release","metric_action","occluded_cut_execution","new_donor_family_approval"])

SYSTEM = (
    "You localize a synthetic tomato petiole cut point in a mounted robot-head RGB image. "
    "The query identifies the target petiole, NOT its cut location. Trace the visible petiole "
    "to its attachment to the main stem. The nominal cut is 10 mm along the petiole toward "
    "the leaves; 10-20 mm is the evaluation arc interval, not a spherical tolerance. "
    "Use visible evidence; abstain when the junction or cut region is not distinguishable. "
    "All query and output coordinates refer to the ORIGINAL 1696x816 full frame and use "
    "[0,1000): x_norm=1000*x/1696 and y_norm=1000*y/816; top-left edge origin, "
    "x right, y down, independent of model resizing. "
    "Return only JSON with exactly status (localized or abstain), cut_point_uv "
    "(a two-number normalized array or null), visibility (clear, partial or occluded), "
    "and next_action (inspect_cut_region or change_viewpoint). "
    "This is perception only, not approval of any blade motion."
)


def contract_hash():
    return hashlib.sha256(json.dumps(CONTRACT,sort_keys=True,allow_nan=False).encode()).hexdigest()


def point(value, normalized=False):
    require(isinstance(value,list) and len(value)==2
            and all(type(v) in (int,float) for v in value), "Two numeric coordinates required")
    normalized_coordinates(value,RESOLUTION,inverse=normalized)  # Explicit finite bounds check.
    return value


def validate_answer(value, *, normalized=False):
    require(isinstance(value,dict) and set(value)=={"status","cut_point_uv","visibility","next_action"},
            "Exact four answer fields required")
    if value["status"] == "localized":
        point(value["cut_point_uv"],normalized)
        require(value["visibility"] in ("clear","partial") and value["next_action"]=="inspect_cut_region",
                "Invalid localization semantics")
    else:
        require(value == dict(status="abstain",cut_point_uv=None,visibility="occluded",next_action="change_viewpoint"),
                "Invalid abstention semantics")
    return value


def model_answer(answer):
    result=deepcopy(validate_answer(answer))
    if result["cut_point_uv"] is not None:
        result["cut_point_uv"]=format_point(normalized_coordinates(result["cut_point_uv"],RESOLUTION).tolist(),2)
    return validate_answer(result,normalized=True)


def canonical_answer(answer):
    result=deepcopy(validate_answer(answer,normalized=True))
    if result["cut_point_uv"] is not None:
        result["cut_point_uv"]=normalized_coordinates(result["cut_point_uv"],RESOLUTION,inverse=True).tolist()
    return validate_answer(result)


def crop_box(query):
    q=np.asarray(point(query),float)
    xy=np.clip(np.floor(q-384),0,np.asarray(RESOLUTION)-768).astype(int)
    return [int(xy[0]),int(xy[1]),int(xy[0]+768),int(xy[1]+768)]


def user_prompt(query, *, crop=False):
    normalized=format_point(normalized_coordinates(point(query),RESOLUTION).tolist(),2)
    text=(f"The target petiole passes through normalized coordinates ({normalized[0]}, {normalized[1]}). "
          "Locate its nominal cut point if the junction and cut region are visible; otherwise abstain.")
    if crop:
        text+=(f" Image 1 is the full frame. Image 2 is its native query-centred crop at full-frame "
               f"pixel bounds {crop_box(query)}. Output normalized coordinates in image 1, never crop coordinates.")
    return text


def model_messages(rgb,query,*,answer=None,crop=False):
    require(isinstance(rgb,Image.Image) and rgb.mode=="RGB" and rgb.size==RESOLUTION,"Native1696 RGB required")
    point(query)
    content=[dict(type="image",image=rgb.copy())]
    if crop:
        content.append(dict(type="image",image=rgb.crop(crop_box(query))))  # No resizing.
    content.append(dict(type="text",text=user_prompt(query,crop=crop)))
    result=[dict(role="system",content=[dict(type="text",text=SYSTEM)]),dict(role="user",content=content)]
    if answer is not None:
        result.append(dict(role="assistant",content=[dict(type="text",text=json.dumps(model_answer(answer),separators=(",",":")))]))
    return result


def clear_screen(rgb,mask,interval,proximal_fraction):
    require(rgb.shape==(816,1696,3) and rgb.dtype==np.uint8 and mask.shape==(816,1696)
            and mask.dtype==bool,"Native1696 RGB and boolean target mask required")
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
