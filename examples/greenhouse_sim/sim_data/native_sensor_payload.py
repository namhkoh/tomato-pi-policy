"""Resolution-aware native payload checks, isolated from the active legacy path.

No image rescaling, reconstructed depth, fabricated engine frame IDs or training
approval. Integration must still freeze/check the scene and verify the mounted
camera independently. Legacy production functions are deliberately unchanged.
"""
from fractions import Fraction
import hashlib
import numpy as np
from .capture_contract import project, transform_points, fingerprint
from .capture_sensor import checked_resolution


def native_dimensions(value):
    """Accept native integer arrays without weakening configuration validation."""
    dimensions = np.asarray(value)
    if dimensions.shape != (2,) or not np.issubdtype(dimensions.dtype, np.integer):
        raise ValueError("Native resolution must contain two integer dimensions")
    return checked_resolution(dimensions.tolist())


def validate_native_payload(payload, calibration, previous_reference=None):
    resolution=checked_resolution(calibration["resolution"])
    width,height=resolution
    required={"rgb","distance_to_image_plane","camera_params","ReferenceTime","reference_time"}
    if not required<=payload.keys(): raise ValueError("Incomplete native writer payload")
    ref=payload["reference_time"]
    if (not isinstance(ref,(tuple,list)) or len(ref)!=2
            or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in ref)
            or ref[0]<0 or ref[1]<=0):
        raise ValueError("Invalid native reference time")
    ref=list(map(int,ref)); rt=payload["ReferenceTime"]
    if Fraction(*ref)!=Fraction(int(rt["referenceTimeNumerator"]),int(rt["referenceTimeDenominator"])):
        raise ValueError("Mismatched native reference time")
    if previous_reference is not None and Fraction(*ref)<=Fraction(*previous_reference):
        raise ValueError("Stale native reference time")
    products=[v for k,v in payload.items() if k.startswith("rp_")]
    if (len(products)!=1 or products[0]["camera"]!=calibration["camera_path"]
            or native_dimensions(products[0]["resolution"])!=resolution):
        raise ValueError("Native render product differs from declared camera/resolution")
    rgb=np.asarray(payload["rgb"]); depth=np.asarray(payload["distance_to_image_plane"])
    if rgb.shape!=(height,width,4) or rgb.dtype!=np.uint8: raise ValueError("Invalid native RGBA")
    if depth.shape!=(height,width) or depth.dtype!=np.float32: raise ValueError("Invalid native optical Z")
    if float(rgb[:,:,:3].std())<1: raise ValueError("Blank native RGB")
    params=payload["camera_params"]
    if native_dimensions(params["renderProductResolution"])!=resolution or not np.isclose(params["metersPerSceneUnit"],1):
        raise ValueError("Native camera units/resolution mismatch")
    view=np.asarray(params["cameraViewTransform"],float).reshape(4,4)
    if not np.allclose(view,np.linalg.inv(calibration["camera_to_world_usd_row_vectors"]),atol=5e-5,rtol=0):
        raise ValueError("Native camera pose differs from authored pose")
    projection=np.asarray(params["cameraProjection"],float).reshape(4,4)
    probes=np.asarray([[0,0,-1],[.1,.15,-1.2],[-.2,-.1,-2]])
    clip=np.column_stack((probes,np.ones(3)))@projection
    if not np.isfinite(clip).all() or np.any(np.abs(clip[:,3])<1e-8): raise ValueError("Invalid native projection")
    pixels=(clip[:,:2]/clip[:,3,None]*[1,-1]+1)*(np.asarray(resolution)/2)
    world=transform_points(probes,calibration["camera_to_world_usd_row_vectors"])
    expected=np.asarray([p["pixel_xy"] for p in project(world,calibration)])
    if not np.allclose(pixels,expected,atol=.01,rtol=0): raise ValueError("Native projection differs from pixel intrinsics")
    near,far=calibration["clipping_range_m"]
    if not np.isfinite([near,far]).all() or not 0<near<far: raise ValueError("Invalid native clipping range")
    valid=np.isfinite(depth)&(depth>0)&(depth>=near)&(depth<=far)
    if valid.mean()<.05: raise ValueError("Insufficient valid native depth")
    return rgb[:,:,:3].copy(),depth.copy(),valid,ref


def validate_native_static(payload, calibration, before, after, sequence, previous=None):
    if not isinstance(before,str) or len(before)!=64 or before!=after:
        raise ValueError("Unchanged static scene guard required")
    if type(sequence) is not int or sequence<=0: raise ValueError("Fresh writer callback required")
    rgb,depth,valid,ref=validate_native_payload(payload,calibration)
    token=dict(callback_sequence=sequence,camera_sha256=fingerprint(calibration),
               rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
               depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())
    if previous is not None:
        if sequence<=previous["callback_sequence"]: raise ValueError("Stale writer callback")
        if token["camera_sha256"]==previous["camera_sha256"]: raise ValueError("Distinct camera required")
        if any(token[k]==previous[k] for k in ("rgb_sha256","depth_sha256")):
            raise ValueError("Unchanged native buffers after camera motion")
    return rgb,depth,valid,ref,token


def decode_native_instances(payload, resolution):
    width,height=checked_resolution(resolution)
    annotation=payload.get("instance_id_segmentation")
    if not isinstance(annotation,dict) or not {"data","info"}<=annotation.keys():
        raise ValueError("Native instance annotation required")
    data=np.asarray(annotation["data"])
    if data.shape!=(height,width) or data.dtype!=np.uint32:
        raise ValueError("Native instance shape/type mismatch")
    labels=annotation["info"].get("idToLabels")
    if not isinstance(labels,dict): raise ValueError("Native identity mapping required")
    mapping={}
    for key,value in labels.items():
        if not isinstance(value,str): raise ValueError("Expected native prim identity")
        identifier=int(key)
        if not 0<=identifier<=np.iinfo(np.uint32).max: raise ValueError("Invalid renderer ID")
        if identifier in mapping and mapping[identifier]!=value: raise ValueError("Conflicting native IDs")
        mapping[identifier]=value
    if set(map(int,np.unique(data)))-mapping.keys()-{0,1}: raise ValueError("Unmapped native IDs")
    if not any(v.startswith("/") for v in mapping.values()): raise ValueError("No native prim identities")
    return data.copy(),mapping
