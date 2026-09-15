"""Create-only compact CAPTURE storage for already validated native callbacks.

No renderer, review PNGs, label derivation, depth reconstruction or approval.
Logical NPY files are made in RAM and preserved byte-for-byte by the native
codec. This is not a migration path for existing captures; use pack_sample for
those. Callback fingerprints check consistency, not independent qualification.
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..automated_native_review import check_native_evidence
from ..capture_contract import jsonable
from ..capture_sensor import checked_resolution
from ..capture_visibility import ORGAN_IDS
from ..dataset_review import require
from ..native_lossless_codec import decode, encode
from .bundle import ARRAYS, MAX_FILE_BYTES, REQUIRED, SCHEMA, SampleReader, digest, read_bounded, safe_path


def _json_bytes(value):
    return json.dumps(jsonable(value), indent=2, allow_nan=False).encode("utf-8")


def _png_bytes(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def write_compact_native_sample(directory, rgb, depth, valid, metadata, instances,
                                mapping, catalogue, components, organs, target_mask,
                                label, trace):
    """Persist one validated callback without ever staging raw NPYs on disk.

    RGB is HxWx3 uint8; depth is float32; renderer/component IDs are uint32;
    organ IDs are uint8; valid/target masks are bool. All arrays must be C-order
    at the explicit supported calibration resolution. Caller data is not edited.
    The caller still owns native sensor/scene/identity validation and label/trace
    derivation. An eligible label requires a supplied trace (including failures).

    Return sample_sha256, label_sha256 and query_trace for a future capture
    result row. These hash the logical JSON bytes read through SampleReader,
    also bound as source_sample_sha256/source_label_sha256 in bundle.json.
    The manifest is written last; an interrupted payload is never published.
    """
    directory = Path(directory).resolve()
    require(not directory.exists(), "Create-only destination required")
    metadata, label, trace = jsonable(metadata), jsonable(label), jsonable(trace)
    require(isinstance(metadata, dict) and "files" not in metadata,
            "Fresh capture metadata required; do not rewrite existing file bindings")
    require(metadata.get("training_sample_approved") is False
            and metadata.get("training_approved", False) is False,
            "Capture storage cannot grant training approval")
    require(isinstance(label, dict) and type(label.get("eligible")) is bool
            and label.get("training_approved") is False, "Unapproved capture label required")
    require(trace is None or isinstance(trace, dict), "Query trace must be a JSON object")
    require(not label["eligible"] or trace is not None, "Eligible label requires query trace")
    calibration = metadata["calibration"]
    width, height = checked_resolution(calibration["resolution"])
    require(calibration.get("crop_resize") is None
            and calibration.get("depth_convention") == "optical_axis_z_metres_not_ray_range",
            "Uncropped native optical-Z calibration required")
    shape = (height, width)
    buffers = {}
    for name, array, dtype, expected_shape in (
        ("rgb", rgb, np.uint8, (*shape, 3)),
        ("depth", depth, np.float32, shape),
        ("valid", valid, np.bool_, shape),
        ("instances", instances, np.uint32, shape),
        ("components", components, np.uint32, shape),
        ("organs", organs, np.uint8, shape),
        ("target_mask", target_mask, np.bool_, shape),
    ):
        require(isinstance(array, np.ndarray) and array.shape == expected_shape
                and array.dtype == dtype and array.flags.c_contiguous,
                "Invalid native shape/dtype/C-order: " + name)
        # Validate and serialize the same private snapshot, not live caller arrays.
        buffers[name] = array.copy()
    target_png = buffers["target_mask"].astype(np.uint8) * 255
    catalogue, mapping = jsonable(catalogue), jsonable(mapping)
    check_native_evidence(metadata, buffers["rgb"], buffers["depth"],
                          buffers["components"], target_png, catalogue)

    logical = {
        "inputs/rgb.png": _png_bytes(buffers["rgb"]),
        "inputs/depth_m.npy": _npy_bytes(buffers["depth"]),
        "inputs/depth_valid.png": _png_bytes(buffers["valid"].astype(np.uint8) * 255),
        "supervision/renderer_instance_id.npy": _npy_bytes(buffers["instances"]),
        "supervision/component_id.npy": _npy_bytes(buffers["components"]),
        "supervision/organ_type.png": _png_bytes(buffers["organs"]),
        "supervision/target_visible.png": _png_bytes(target_png),
        "supervision/identities.json": _json_bytes({
            "renderer_id_to_prim": mapping, "component_catalogue": catalogue,
            "organ_ids": ORGAN_IDS,
            "unmapped_scope": "background greenhouse and backdrop plants lack per-organ manifests",
            "ids_stable_across_frames": False,
        }),
    }
    require(logical.keys() == REQUIRED, "Complete native observations required")
    metadata["files"] = {
        name: dict(sha256=digest(raw), role="observation" if name.startswith("inputs/")
                   else "ground_truth_supervision") for name, raw in logical.items()
    }
    logical["sample.json"] = _json_bytes(metadata)
    logical["supervision/label.json"] = _json_bytes(label)
    if trace is not None:
        logical["supervision/query_trace.json"] = _json_bytes(trace)

    entries, payloads = {}, {}
    for name, raw in logical.items():
        require(0 < len(raw) <= MAX_FILE_BYTES, "Logical file exceeds reader bound")
        packed = encode(raw) if name in ARRAYS else raw
        if name in ARRAYS:
            require(decode(packed) == raw, "Non-exact native roundtrip")
        require(0 < len(packed) <= MAX_FILE_BYTES, "Stored file exceeds reader bound")
        stored_name = "payload/" + name + (".ghn" if name in ARRAYS else "")
        payloads[stored_name] = packed
        entries[name] = dict(stored_path=stored_name,
            encoding="native_npy_byteplanes_zstd" if name in ARRAYS else "identity",
            logical_bytes=len(raw), stored_bytes=len(packed),
            logical_sha256=digest(raw), stored_sha256=digest(packed))
    sample_hash = digest(logical["sample.json"])
    label_hash = digest(logical["supervision/label.json"])
    manifest = dict(schema=SCHEMA, files=entries, omitted_review_files={},
        source_sample_sha256=sample_hash, source_label_sha256=label_hash,
        source_query_trace=trace, lossy=False, depth_recomputed=False,
        source_removed=False, training_approved=False,
        rgb_policy="unchanged_original_png", depth_policy="exact_native_optical_z_npy_bytes",
        review_policy="not_generated_review_only_pngs_not_model_input",
        source_logical_bytes=sum(e["logical_bytes"] for e in entries.values()),
        stored_payload_bytes=sum(e["stored_bytes"] for e in entries.values()))
    manifest_bytes = _json_bytes(manifest)
    require(len(manifest_bytes) <= MAX_FILE_BYTES, "Manifest exceeds reader bound")

    directory.mkdir(parents=True, exist_ok=False)
    for stored_name, packed in payloads.items():
        path = safe_path(directory, stored_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(packed)
    # Check every on-disk payload before publishing the completion marker.
    for stored_name, packed in payloads.items():
        require(read_bounded(safe_path(directory, stored_name)) == packed,
                "Stored capture changed before publish")
    with (directory / "bundle.json").open("xb") as stream:
        stream.write(manifest_bytes)
    reader = SampleReader(directory)
    reader.verify_all()
    require(digest(reader.read("sample.json")) == sample_hash
            and digest(reader.read("supervision/label.json")) == label_hash,
            "Capture bindings changed at publish")
    return dict(sample_sha256=sample_hash, label_sha256=label_hash, query_trace=trace)
