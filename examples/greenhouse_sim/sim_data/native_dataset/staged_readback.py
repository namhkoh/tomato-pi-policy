"""Opt-in synchronous CUDA readback; no renderer, storage, or audit changes.

Verified against the installed Isaac 6.0.1 sources pinned below:
* annotators.py:425/842/1895: public_name, hostPtr/buffPtr, RGBA8/Z32.
* writers.py:295/1187: alias wiring and per-product clone preservation.
* OgnWriter.py:129-164: all annotators assembled into one writer callback.
* annotator_utils.py:208/346/609: format, Warp shape, node device routing.
* SyntheticData.py:455/619/658/699: SAME render vars, distinct copy paths.
* Warp types.py:4083: numpy() performs a synchronous default-stream D2H copy.

No guessed pointer casts, GPU-ID decoding, asynchronous queues, or depth math.
CPU mode delegates unchanged. Compare mode requires CPU and GPU mirrors in
one callback. Staged mode is an UNQUALIFIED timing trial requiring a pinned
compare receipt. Neither mode grants training/storage/render qualification.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import time

import numpy as np
from ..native_instance_adapter import make_native_writer as _cpu_writer

SCHEMA = "native_staged_readback.v1"
SCOPE = "same_writer_callback_same_product_all_RGBA_and_native_Z_bits"
MODES = ("cpu", "compare", "staged")
SENSORS = {
    "rgb": ("LdrColorSD", "staged_rgb", np.dtype("uint8")),
    "distance_to_image_plane": ("DistanceToImagePlaneSD", "staged_z", np.dtype("float32")),
}
_CORE = "extscache/omni.replicator.core-1.13.27+110.1.1.wx64.r.cp312/omni/replicator/core/"
SDK_PINS = {
    _CORE + "scripts/annotators.py": "8d5f92d0af16bb1565a7f327767be51a58af9002583f5dccfdaa64b655e32568",
    _CORE + "scripts/writers.py": "e3b7e8e5270ca757ef7d5056fd871468559411df0e93afff77fd0fc4af648b2b",
    _CORE + "scripts/utils/annotator_utils.py": "bb420a06e7b54994aa539091f7b626a19949f03b9a9d2be3a03f9d6da224cb81",
    _CORE + "ogn/python/impl/nodes/OgnWriter.py": "6e22d7e2624db965d75ebd69f87f083ff482fec99f749bf76931bf6cda8bef46",
    "extscache/omni.syntheticdata-0.6.15+f9bf0dda.wx64.r.cp312/omni/syntheticdata/scripts/SyntheticData.py":
        "1437976157db8045f872a92c60ad615022061d823fdbfdc619e7b81478896e6f",
    "extscache/omni.warp.core-1.13.0+wx64/warp/_src/types.py":
        "38e8b08858dae9f21f20e7992c4ed3aa0c31b107236f1e1787bb988d055888a0",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_local_api(isaac_root):
    """Read only; does not import Isaac, Warp, or initialize CUDA."""
    bindings = []
    for relative, expected in SDK_PINS.items():
        binding = file_binding(Path(isaac_root) / relative)
        _require(binding["sha256"] == expected, "Unverified local SDK source: " + relative)
        bindings.append(binding)
    return bindings


def _runtime_bindings(rep, wp):
    # Pin the actual loaded implementations, not merely another installation.
    objects = [
        rep.annotators, rep.writers, rep.annotators.annotator_utils,
        importlib.import_module("omni.replicator.core.ogn.python.impl.nodes.OgnWriter"),
        importlib.import_module("omni.syntheticdata.scripts.SyntheticData"), wp.array,
    ]
    bindings = []
    for obj, (relative, expected) in zip(objects, SDK_PINS.items()):
        binding = file_binding(inspect.getsourcefile(obj))
        _require(Path(binding["path"]).as_posix().endswith(relative)
                 and binding["sha256"] == expected, "Unverified loaded SDK: " + relative)
        bindings.append(binding)
    return bindings


def implementation_bindings():
    root = Path(__file__).resolve().parent.parent
    return [file_binding(p) for p in (
        Path(__file__), root / "native_instance_adapter.py", root / "capture_pilot.py",
        root / "native_sensor_payload.py", root / "native_generated_views.py",
    )]


def _bits(array):
    return np.ascontiguousarray(array).view(np.uint8).reshape(-1)


def _digest(array):
    return hashlib.sha256(_bits(array)).hexdigest()


def _array(value, name, resolution):
    width, height = resolution
    shape = (height, width, 4) if name == "rgb" else (height, width)
    _require(isinstance(value, np.ndarray) and value.shape == shape
             and value.dtype == SENSORS[name][2], "Unexpected native array: " + name)
    return value


def _stage(value, name, resolution, wp):
    # Reject NumPy, dict, empty, CPU Warp, or an unverified tensor protocol.
    _require(isinstance(value, wp.array)
             and re.fullmatch(r"cuda(?::[0-9]+)?", str(value.device)) is not None,
             "Expected CUDA Warp array: " + name)
    # SDK numpy() synchronizes its D2H copy; copy again to own the CPU buffer
    # independently of Warp, the next callback, and any cached annotator data.
    return _array(value.numpy(), name, resolution).copy(order="C")


def _context(data, resolution):
    _require(isinstance(data, dict), "Writer callback dictionary required")
    products = [(k, v) for k, v in data.items() if k.startswith("rp_")]
    _require(len(products) == 1, "Exactly one render product required")
    product_key, product = products[0]
    _require(isinstance(product, dict) and isinstance(product.get("camera"), str)
             and product["camera"].startswith("/")
             and np.array_equal(product.get("resolution"), resolution),
             "Render product camera/resolution mismatch")
    _require(all(k in data for k in ("camera_params", "pilot_render_frame", "ReferenceTime", "reference_time")),
             "Incomplete synchronized callback")
    ref, rt = data["reference_time"], data["ReferenceTime"]
    _require(isinstance(ref, (list, tuple)) and len(ref) == 2
             and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in ref)
             and ref[0] >= 0 and ref[1] > 0, "Invalid callback reference time")
    _require(Fraction(*map(int, ref)) == Fraction(int(rt["referenceTimeNumerator"]),
             int(rt["referenceTimeDenominator"])), "Callback reference time mismatch")
    _require(np.array_equal(data["camera_params"].get("renderProductResolution"), resolution),
             "Callback camera resolution mismatch")
    return dict(render_product=product_key, camera=product["camera"],
                reference_time=list(map(int, ref)))


def _process(data, resolution, backend, mode, wp):
    from ..native_instance_adapter import normalize
    started = time.perf_counter()
    context = _context(data, resolution)
    staged, comparisons, transfer_seconds = {}, {}, {}
    for name, (_, alias, _) in SENSORS.items():
        _require(alias in data, "Missing same-callback GPU mirror: " + alias)
        transfer_started = time.perf_counter()
        staged[name] = _stage(data[alias], name, resolution, wp)
        transfer_seconds[name] = time.perf_counter() - transfer_started
        if mode == "compare":
            _require(name in data, "Missing same-callback CPU reference: " + name)
            cpu = _array(data[name], name, resolution)
            a, b = _bits(cpu), _bits(staged[name])
            differing_bytes = int(np.count_nonzero(a != b))
            if differing_bytes:
                raise ValueError(f"Same-callback {name} differs: {differing_bytes} bytes")
            comparisons[name] = dict(shape=list(cpu.shape), dtype=cpu.dtype.str,
                compared_bytes=int(cpu.nbytes), differing_bytes=0,
                cpu_sha256=_digest(cpu), staged_sha256=_digest(staged[name]))
        else:
            _require(name not in data, "CPU mirror unexpectedly attached in staged trial")
    # Existing full-ID normalization/compare remains in force. All remaining
    # fields (including mappings, camera and frame evidence) are owned CPU copies.
    excluded = set(SENSORS) | {v[1] for v in SENSORS.values()}
    owned = deepcopy(normalize({k: v for k, v in data.items() if k not in excluded}, backend, resolution))
    owned.update(staged)
    evidence = dict(schema=SCHEMA, mode=mode, scope=SCOPE if mode == "compare" else None,
        same_callback_exact=mode == "compare", native_qualified=False, training_approved=False,
        resolution=list(resolution), instance_backend=backend, **context,
        transfer_seconds=transfer_seconds, comparisons=comparisons)
    if mode == "compare":
        from ..capture_contract import jsonable
        evidence["rgb_sha256"] = _digest(staged["rgb"][..., :3])
        evidence["depth_sha256"] = comparisons["distance_to_image_plane"]["staged_sha256"]
        evidence["instances_sha256"] = _digest(owned["instance_id_segmentation"]["data"])
        for key in ("camera_params", "pilot_render_frame"):
            evidence[key + "_sha256"] = hashlib.sha256(json.dumps(jsonable(owned[key]),
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    evidence["callback_processing_seconds"] = time.perf_counter() - started
    return owned, evidence


def require_compare_receipt(path, sha256, bindings, resolution, backend):
    """Pinned diagnostic evidence only, never a claim an audit/approval ran."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    binding = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    _require(binding["sha256"] == sha256, "Comparison receipt hash mismatch")
    proof = json.loads(raw)
    _require(proof.get("schema") == SCHEMA and proof.get("mode") == "compare"
             and proof.get("state") == "diagnostic_complete"
             and proof.get("native_qualified") is False
             and proof.get("bindings") == bindings
             and proof.get("resolution") == list(resolution)
             and proof.get("instance_backend") == backend
             and proof.get("render_budget") == "reference56", "Incompatible comparison receipt")
    captures = proof.get("captured_comparisons", [])
    _require(len(captures) >= 3, "At least three captured same-callback comparisons required")
    _require(len({c["camera_sha256"] for c in captures}) >= 3
             and len({c["callback_sequence"] for c in captures}) == len(captures),
             "Distinct captured callbacks/cameras required")
    callbacks = proof.get("callbacks", [])
    _require(callbacks and all(c.get("same_callback_exact") is True and c.get("scope") == SCOPE
                              and c.get("mode") == "compare" for c in callbacks),
             "Incomplete same-callback evidence")
    by_sequence = {c["callback_sequence"]: c for c in callbacks}
    _require(len(by_sequence) == len(callbacks), "Duplicate callback evidence")
    _require(list(by_sequence) == list(range(1, len(callbacks) + 1)), "Nonsequential callback evidence")
    width, height = resolution
    for callback in callbacks:
        _require(callback.get("resolution") == list(resolution)
                 and callback.get("instance_backend") == backend
                 and callback.get("native_qualified") is False, "Incompatible callback evidence")
        for name in SENSORS:
            item = callback.get("comparisons", {}).get(name, {})
            shape = [height, width, 4] if name == "rgb" else [height, width]
            _require(item.get("shape") == shape and item.get("dtype") == SENSORS[name][2].str
                     and item.get("compared_bytes") == width * height * 4
                     and item.get("differing_bytes") == 0
                     and item.get("cpu_sha256") == item.get("staged_sha256")
                     and re.fullmatch("[0-9a-f]{64}", str(item.get("cpu_sha256"))) is not None,
                     "Incomplete full-buffer bitwise comparison")
        _require(callback.get("depth_sha256") == callback["comparisons"]["distance_to_image_plane"]["staged_sha256"],
                 "Inconsistent native-Z comparison digest")
        for key in ("rgb_sha256", "instances_sha256", "camera_params_sha256", "pilot_render_frame_sha256"):
            _require(re.fullmatch("[0-9a-f]{64}", str(callback.get(key))) is not None,
                     "Missing callback provenance: " + key)
    for captured in captures:
        callback = by_sequence.get(captured["callback_sequence"], {})
        _require(all(captured.get(k) and captured[k] == callback.get(k)
                     for k in ("rgb_sha256", "depth_sha256")), "Capture/callback digest mismatch")
    pins = proof.get("artifact_bindings", [])
    _require(len(pins) >= len(captures) + 2 and len({p["path"] for p in pins}) == len(pins),
             "Missing/duplicate request/result/sample source pins")
    for pin in pins:
        _require(file_binding(pin["path"]) == pin, "Mutated comparison artifact")
    return binding


def _new_writer(rep, wp, resolution, backend, mode, base, bindings):
    class StagedReadbackWriter(rep.Writer):
        def __init__(self):
            self.version = SCHEMA
            self.annotators = []
            for annotation in base.annotators:
                if annotation.name not in SENSORS or mode == "compare":
                    self.annotators.append(annotation)
            for name, (template, alias, _) in SENSORS.items():
                annotation = rep.annotators.Annotator(name=name, template_name=template,
                    device="cuda", do_array_copy=False, public_name=alias)
                _require(annotation.template_name == template + "buffPtr",
                         "GPU buffer export unavailable: " + name)
                self.annotators.append(annotation)
            _require(len({a.name for a in self.annotators}) == len(self.annotators),
                     "Duplicate writer payload labels")
            self.sequence, self.latest, self.request_index = 0, None, 0
            self.capture_error = None
            self.readback_records = []
            self.readback_bindings = bindings

        def attach(self, render_products, **kwargs):
            _require(isinstance(render_products, (list, tuple)) and len(render_products) == 1,
                     "Exactly one explicit render product required")
            return super().attach(render_products, **kwargs)

        def write(self, data):
            if self.capture_error is not None:
                raise RuntimeError("Readback writer is poisoned by an earlier callback") from self.capture_error
            try:
                owned, record = _process(data, resolution, backend, mode, wp)
                record.update(callback_sequence=self.sequence + 1, request_index=self.request_index)
                self.readback_records.append(record)
                self.latest = owned
                self.sequence += 1
            except Exception as exc:
                self.capture_error = exc
                self.latest = None
                raise

        def write_metadata(self):
            self._is_metadata_written = True

    return StagedReadbackWriter()


def make_staged_writer(rep, resolution, backend, *, mode="cpu", compare_receipt=None):
    """Factory for an explicitly isolated runtime wrapper; does not attach/render.

    compare_receipt is (absolute_path, sha256) for mode='staged'. Compare mode
    itself needs no receipt. Do not compose with a different readback patch.
    """
    _require(mode in MODES, "Unknown readback mode")
    if mode == "cpu":
        _require(compare_receipt is None, "CPU default does not consume comparison evidence")
        return _cpu_writer(rep, resolution, backend)
    _require(tuple(resolution) == (1696, 816) and backend in ("fast", "compare"),
             "Explicit full native resolution and fast/compare IDs required")
    import warp as wp
    bindings = _runtime_bindings(rep, wp) + implementation_bindings()
    if mode == "staged":
        _require(isinstance(compare_receipt, tuple) and len(compare_receipt) == 2,
                 "Pinned same-callback receipt required before staged timing trial")
        require_compare_receipt(*compare_receipt, bindings, resolution, backend)
    else:
        _require(compare_receipt is None, "Compare mode must produce its own evidence")
    base = _cpu_writer(rep, resolution, backend)
    return _new_writer(rep, wp, resolution, backend, mode, base, bindings)
