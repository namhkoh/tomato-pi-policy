"""Colour display of saved native Isaac depth; no depth estimation or rendering.

Run: python -m sim_data.depth_preview --run path/to/completed_rgbd_pilot
Writes a new companion review directory. Original observations/labels are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Explicit display palette, near -> far. This is not a learned depth model.
COLOURS = np.array([[253,231,37], [122,209,81], [34,168,132],
                   [42,120,142], [65,68,135], [68,1,84]], float)


def colour_depth(depth, valid, near_m, far_m):
    depth, valid = np.asarray(depth), np.asarray(valid)
    if depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != np.bool_:
        raise ValueError("Expected 2D depth and a matching boolean validity mask")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("Expected native floating-point metric depth")
    if not np.isfinite([near_m,far_m]).all() or not 0 <= near_m < far_m:
        raise ValueError("Display range must be finite and increasing")
    usable = valid & np.isfinite(depth) & (depth > 0)
    height, width = depth.shape
    y, x = np.indices((height,width))
    # Checkerboard means invalid/no measurement, never zero metres or distant surface.
    gray = np.where((x//8 + y//8) % 2 == 0, 75, 105).astype(np.uint8)
    rgb = np.repeat(gray[:,:,None],3,axis=2)
    t = np.clip((depth[usable].astype(float)-near_m)/(far_m-near_m),0,1)
    stops = np.linspace(0,1,len(COLOURS))
    rgb[usable] = np.rint(np.column_stack([np.interp(t,stops,COLOURS[:,c]) for c in range(3)])).astype(np.uint8)
    return rgb


def labelled_heatmap(depth, valid, near_m, far_m, title):
    """Keep the original pixel plane intact; add the legend below, not over it."""
    pixels = colour_depth(depth,valid,near_m,far_m)
    height, width = depth.shape
    canvas = Image.new("RGB",(width,height+118),(22,26,33))
    canvas.paste(Image.fromarray(pixels),(0,0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",15)
        small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",13)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((14,height+7),title,fill="white",font=font)
    draw.text((14,height+29),"Native Replicator camera-Z depth in metres. Yellow = nearer; purple = farther.",fill=(215,220,230),font=small)
    left,right = 18,width-210
    bar = np.linspace(near_m,far_m,right-left+1,dtype=np.float32)[None,:]
    canvas.paste(Image.fromarray(colour_depth(bar,np.ones(bar.shape,bool),near_m,far_m)).resize((right-left+1,15)),(left,height+54))
    for fraction in np.linspace(0,1,5):
        x = left + fraction*(right-left)
        value = near_m + fraction*(far_m-near_m)
        label = ("<= " if fraction==0 else ">= " if fraction==1 else "") + f"{value:.2f} m"
        if fraction==0: anchor="la"
        elif fraction==1: anchor="ra"
        else: anchor="ma"
        draw.text((x,height+72),label,fill="white",font=small,anchor=anchor)
    invalid = colour_depth(np.zeros((16,24),np.float32),np.zeros((16,24),bool),near_m,far_m)
    canvas.paste(Image.fromarray(invalid),(width-190,height+54))
    draw.text((width-158,height+53),"Invalid / no depth",fill="white",font=small)
    draw.text((14,height+97),"Display colours saturate outside this range. Raw depth is unchanged; no RGB-based depth estimation.",fill=(190,195,205),font=small)
    return canvas


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def generate(run, output=None, near_m=.04, far_m=2.):
    run = Path(run).resolve()
    output = Path(output).resolve() if output is not None else run/"depth_heatmaps"
    manifest_path = run/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "pilot_ready_for_review" or not manifest.get("samples"):
        raise ValueError("Expected a completed capture pilot with native depth")
    if output.exists():
        raise FileExistsError("Choose a new review output directory; existing outputs are not overwritten")
    if output.is_relative_to(run) and output.parent != run:
        raise ValueError("Companion output must not be placed inside an existing sample")
    if not np.isfinite([near_m,far_m]).all() or not 0 <= near_m < far_m:
        raise ValueError("Invalid display range")
    # Validate every native observation before publishing any new preview.
    samples, original_hashes = [], {manifest_path:sha256(manifest_path)}
    for entry in manifest["samples"]:
        sample_id = entry["sample_id"]
        if Path(sample_id).name != sample_id or not sample_id.startswith("sample_") or "\\" in sample_id:
            raise ValueError("Invalid sample identity")
        directory = run/sample_id
        metadata_path = directory/"sample.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        original_hashes[metadata_path] = sha256(metadata_path)
        for name in ("inputs/rgb.png","inputs/depth_m.npy","inputs/depth_valid.png"):
            path = directory/name
            actual = sha256(path)
            if actual != metadata["files"][name]["sha256"]:
                raise ValueError(f"Native observation hash mismatch: {path}")
            original_hashes[path] = actual
        cal = metadata["calibration"]
        if cal["depth_convention"] != "optical_axis_z_metres_not_ray_range":
            raise ValueError("Unsupported native depth convention")
        depth = np.load(directory/"inputs/depth_m.npy",allow_pickle=False)
        valid = np.asarray(Image.open(directory/"inputs/depth_valid.png"))
        shape = tuple(reversed(cal["resolution"]))
        if depth.shape != shape or depth.dtype != np.float32 or valid.shape != shape:
            raise ValueError("Depth shape/dtype/validity does not match capture calibration")
        clip_near,clip_far = cal["clipping_range_m"]
        expected = np.isfinite(depth)&(depth>0)&(depth>=clip_near)&(depth<=clip_far)
        if not np.array_equal(valid,expected.astype(np.uint8)*255):
            raise ValueError("Saved validity mask does not match native depth and clipping")
        if len({s["sample_id"] for s in samples}) != len(samples) or any(s["sample_id"]==sample_id for s in samples):
            raise ValueError("Duplicate sample identity")
        samples.append({"sample_id":sample_id,"target":entry["target_review_id"],
                        "depth":depth,"valid":expected,"clipping":(clip_near,clip_far)})
    output.mkdir(parents=True)
    result = {"schema_version":"greenhouse.native_depth_display.v1","state":"building",
              "source_run":str(run),"source_annotator":"distance_to_image_plane",
              "depth_estimated_from_rgb":False,"depth_reconstructed_from_geometry":False,
              "display_only":True,"shared_near_view_range_m":[near_m,far_m],
              "invalid_display":"gray_checkerboard","samples":[]}
    import os
    def relative(path):
        return Path(os.path.relpath(path,output)).as_posix()
    lines = ["# Native Isaac Sim depth: colour review", "",
             "These heatmaps colour the saved Replicator `distance_to_image_plane` values directly. "
             "No depth is inferred from RGB or reconstructed from plant geometry. Original depth, RGB and labels are unchanged.", "",
             f"The near-view display uses the SAME linear {near_m:.2f}-{far_m:.2f} metre scale for every sample. "
             "Yellow is near; green/blue is intermediate; purple is far. Colours saturate at the scale endpoints; "
             "grey checkerboard means invalid/no depth, not a surface. Each image includes its distance legend.", "",
             "This is ideal simulator camera-Z depth, not calibrated physical D405 noise. "
             "Heatmaps are review-only; the float32 NPY remains the metric observation.", "",
             f"[Original RGB/label review]({relative(run/'review.md')})", ""]
    try:
        for s in samples:
            name,depth,valid = s["sample_id"],s["depth"],s["valid"]
            near_path,full_path = output/f"{name}_depth_near.png", output/f"{name}_depth_full.png"
            labelled_heatmap(depth,valid,near_m,far_m,f"{name} / {s['target']} - near-scene depth").save(near_path)
            labelled_heatmap(depth,valid,*s["clipping"],f"{name} / {s['target']} - full camera clipping range").save(full_path)
            lines += [f"## {name} / {s['target']}", "", "Original camera RGB:", "",
                      f"![Native camera RGB]({relative(run/name/'inputs/rgb.png')})", "", "Native simulator depth, colour display:", "",
                      f"![Native camera depth heatmap]({near_path.name})", "",
                      f"[Full camera-range heatmap]({full_path.name}) / [Raw native depth]({relative(run/name/'inputs/depth_m.npy')})", ""]
            result["samples"].append({"sample_id":name,"source_depth_sha256":original_hashes[run/name/"inputs/depth_m.npy"],
                "full_view_range_m":list(s["clipping"]),"valid_depth_fraction":float(valid.mean()),
                "valid_pixels_above_near_display_range":int(np.count_nonzero(valid&(depth>far_m))),
                "files":{p.name:sha256(p) for p in (near_path,full_path)}})
        for path,expected in original_hashes.items():
            if sha256(path)!=expected:
                raise ValueError(f"Source changed while displaying depth: {path}")
        result.update(state="complete_review_only",source_observations_and_metadata_unchanged=True)
        (output/"review.md").write_text("\n".join(lines),encoding="utf-8")
    except Exception:
        result["state"]="failed_do_not_use"
        raise
    finally:
        (output/"manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--near",type=float,default=.04)
    parser.add_argument("--far",type=float,default=2.)
    args=parser.parse_args(argv)
    print(generate(args.run,args.output,args.near,args.far)/"review.md")


if __name__=="__main__":
    main()
