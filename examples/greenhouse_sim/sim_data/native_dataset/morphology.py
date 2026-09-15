"""Pinned, read-only descriptors of ACTUAL saved rigid-leaf V2 outputs.

``extract_output(OutputPin(...))`` replays the existing catalogue, then reads
authored meshes and saved manifest chains (not predicted recipe coordinates).
It returns original + generated contexts, exact provenance IDs, and explicit
domain holds. It neither discovers captures nor verifies native pixels. Maxwell's
inventory adapter can join source_family/source_target/target_id; this module
does not own view accounting, equivalence edges, thresholds or release approval.

Only single-chain intact petioles on unchanged main stems with direct leaves are
supported. Meshes must be static, uninstanced, self-contained and have identity
local transforms. Leaf covariance is a vertex-weighted summary, NOT full shape.
Integrity failures raise ValueError; unsupported domains return no descriptor.
No writes, renderer imports, generator calls, or calibration declarations.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from .. import (audit, clear_scale_capacity, cut_regions, dataset_review,
               depth_preview, geometry, morphology_context, plant_variant_catalogue,
               plant_variant_usd, plant_variants, procedural_leaf_transport,
               procedural_petiole_catalogue, procedural_petiole_catalogue_v2,
               procedural_petiole_geometry, procedural_petiole_usd,
               procedural_petiole_v2, procedural_petiole_warp)

SCHEMA = "greenhouse.actual_output_morphology.v1"
SUPPORTED_VERSION = "curved_relocated_rigid_leaf_static.v2"
_MODULES = (audit, clear_scale_capacity, cut_regions, dataset_review, depth_preview,
            geometry, morphology_context, plant_variant_catalogue, plant_variant_usd,
            plant_variants, procedural_leaf_transport, procedural_petiole_catalogue,
            procedural_petiole_catalogue_v2, procedural_petiole_geometry,
            procedural_petiole_usd, procedural_petiole_v2, procedural_petiole_warp)


def _hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Unreadable bound file: {path}") from exc


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


_LOADED_CODE = {str(Path(p).resolve()): _hash(p) for p in
                [__file__, *[m.__file__ for m in _MODULES], cut_regions.DEFAULT_RULE]}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify(bindings):
    for path, expected in bindings.items():
        _require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected),
                 "Explicit SHA256 pin required")
        _require(_hash(path) == expected, "Changed bound file: " + str(path))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class OutputPin:
    """Caller-pinned qualification + frozen source plan; paths may be relative.

    The qualification must bind the saved manifest and all output assets.
    No implicit newest-file selection or self-issued provenance approval.
    """
    directory: Path
    qualification_sha256: str
    source_plan: Path
    source_plan_sha256: str


def _chain(component):
    if len(component["capsules_local_m"]) != 1:
        raise _Unsupported("petiole_requires_one_saved_chain")
    try:
        curve = procedural_petiole_usd.donor_curve(component)
    except ValueError as exc:
        raise _Unsupported("unsupported_petiole_chain: " + str(exc)) from exc
    result = np.column_stack((curve["points"] + component["translation_plant_m"], curve["radius"]))
    _valid_chain(result)
    return result


def _valid_chain(chain):
    if (chain.ndim != 2 or chain.shape[1] != 4 or len(chain) < 2
            or not np.isfinite(chain).all() or np.any(chain[:, 3] <= 0)
            or np.any(np.linalg.norm(np.diff(chain[:, :3], axis=0), axis=1) <= 1e-12)):
        raise _Unsupported("nonfinite_or_degenerate_chain")


def _nearest_parent(parent, origin):
    candidates, skipped = [], []
    for index, raw in enumerate(parent["capsules_local_m"]):
        chain = np.asarray(raw, dtype=float).copy()
        try:
            _valid_chain(chain)
        except _Unsupported as exc:
            skipped.append(dict(index=index, reason=str(exc)))
            continue
        chain[:, :3] += parent["translation_plant_m"]
        delta = np.diff(chain[:, :3], axis=0)
        fraction = np.clip(np.einsum("ij,ij->i", origin - chain[:-1, :3], delta)
                           / np.einsum("ij,ij->i", delta, delta), 0, 1)
        distance = float(np.min(np.linalg.norm(origin - chain[:-1, :3] - fraction[:, None] * delta, axis=1)))
        candidates.append((distance, index, chain))
    if not candidates:
        raise _Unsupported("no_valid_parent_chain")
    candidates.sort(key=lambda item: (item[0], item[1]))
    # Do not invent an axis for equally near, distinct chains. No novelty epsilon.
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        if not np.array_equal(candidates[0][2], candidates[1][2]):
            raise _Unsupported("ambiguous_equidistant_parent_chains")
    distance, index, chain = candidates[0]
    return chain, dict(index=index, distance_m=distance, skipped=skipped,
                       selection="nearest_valid_saved_chain_to_original_proximal_point")


def _mesh(component, root, bindings):
    """Fresh anonymous file layer avoids accidentally reading cached USD points."""
    from pxr import Sdf, Usd, UsdGeom
    path = audit.safe_asset(root, component["file"])
    _require(bindings.get(str(path)) == component["asset_sha256"], "Unbound descriptor mesh")
    _verify({str(path): component["asset_sha256"]})
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    _require(bool(layer), "Unreadable authored USD mesh")
    if layer.subLayerPaths:
        raise _Unsupported("composed_or_external_mesh_layers")
    stage = Usd.Stage.Open(layer)
    for prim in stage.TraverseAll():
        if (prim.HasAuthoredReferences() or prim.HasAuthoredPayloads() or prim.IsInstance()
                or prim.IsInstanceable() or prim.GetVariantSets().GetNames()
                or prim.HasAuthoredInherits() or prim.HasAuthoredSpecializes()):
            raise _Unsupported("composed_instanced_or_variant_mesh")
        if not prim.IsActive():
            raise _Unsupported("inactive_mesh_domain")
        if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
            raise _Unsupported("time_sampled_mesh_domain")
        if prim.IsA(UsdGeom.Xformable):
            matrix = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float)
            if not np.array_equal(matrix, np.eye(4)):
                raise _Unsupported("nonidentity_local_mesh_transform")
    if len(stage.GetUsedLayers()) != 2:  # root + anonymous session only
        raise _Unsupported("composed_or_external_mesh_layers")
    points, _ = plant_variant_usd.mesh_arrays(stage)
    if len(points) < 3:
        raise _Unsupported("insufficient_leaf_mesh_vertices")
    _verify({str(path): component["asset_sha256"]})
    return points + component["translation_plant_m"]


def _inputs(report, reference, key, bindings):
    components = report["components"]
    target = components[key]
    if target["type"] != "sub_stem" or target["deleafed"] is not False:
        raise _Unsupported("requires_intact_petiole")
    parent = components.get(target["parent"], {})
    if parent.get("type") != "main_stem":
        raise _Unsupported("requires_main_stem_parent")
    members = audit.descendants(components, key)
    leaves = [components[name] for name in members if name != key]
    if not leaves or any(c["type"] != "leaf" or c["parent"] != key for c in leaves):
        raise _Unsupported("requires_nonempty_direct_leaves_only")
    current = _chain(target)
    parent_chain, selection = _nearest_parent(parent, reference[0, :3])
    root = Path(report["manifest_path"]).parent
    # Inspect the actual target and parent assets too; do not certify capsules alone.
    _mesh(target, root, bindings)
    _mesh(parent, root, bindings)
    summaries = []
    for leaf in leaves:
        vertices = _mesh(leaf, root, bindings)
        summaries.append(dict(attachment=leaf["attachment_plant_m"],
            centroid=vertices.mean(axis=0).tolist(), axis=leaf["axis_plant"],
            covariance=np.cov(vertices, rowvar=False).tolist()))
    inputs = dict(reference_petiole=reference.tolist(), parent=parent_chain.tolist(),
                  current_petiole=current.tolist(), leaves=summaries)
    meshes = {name: components[name]["asset_sha256"] for name in sorted([parent["id"], *members])}
    return inputs, selection, meshes


def extract_output(pin, *, component_ids=None):
    """Verify a pinned V2 output; return deterministic original/generated records.

    An explicit component subset must consist of qualification targets. Unsupported
    targets are held individually. Unsupported generator versions are held as a
    whole output, without treating their unverified ancestry as an original.
    File/code tampering, stale catalogue proofs or changed frozen splits RAISE.
    Calls the existing CPU catalogue without changing any bound implementation.
    """
    _verify(_LOADED_CODE)
    directory, plan_path = Path(pin.directory).resolve(), Path(pin.source_plan).resolve()
    qualification = directory / "qualification.json"
    bindings = {str(qualification): pin.qualification_sha256, str(plan_path): pin.source_plan_sha256}
    _verify(bindings)
    _require(not (directory / "FAILED.json").exists(), "Failed generation is unusable")
    receipt, plan = _json(qualification), _json(plan_path)
    capacity = clear_scale_capacity.inventory(plan)
    assignments = plan["family_assignments"]
    _require(capacity["family_count"] == 24 and Counter(assignments.values()) ==
             Counter(train=16, validation=4, test=4), "Original 24-donor reservations required")
    result = dict(schema=SCHEMA, state="actual_output_descriptors_not_novelty_admission",
        directory=str(directory), records=[], holds=[], frozen_splits=assignments,
        frozen_splits_sha256=_digest(assignments), bindings=bindings,
        code_bindings=dict(_LOADED_CODE), training_approved=False, qualified_geometry=False,
        calibration_validated=False, source_cap_reset=False, new_biological_family=False,
        native_capture_verified=False, global_inventory_complete=False, catalogue_replayed=False,
        scope="saved_manifest_chains_and_authored_mesh_vertex_summaries",
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__))

    def finish():
        _verify(bindings)
        _verify(_LOADED_CODE)
        _require(not (directory / "FAILED.json").exists(), "Generation failed during inspection")
        result["sha256"] = _digest(result)
        return result

    if receipt.get("version") != SUPPORTED_VERSION:
        result["holds"].append(dict(reason="unsupported_generator_version", version=receipt.get("version")))
        return finish()
    family = receipt["source_family"]
    _require(assignments.get(family) == receipt.get("split") == "train"
             and receipt.get("split_group") == family, "Frozen TRAIN ancestry required")
    _require(receipt["source_plan_sha256"] == pin.source_plan_sha256
             and receipt["frozen_family_assignments"] == assignments, "Frozen source plan changed")
    source_path = Path(receipt["source_manifest_path"]).resolve()
    jobs = [job for job in plan["jobs"] if job["plant_family"] == family]
    _require(len(jobs) == 1 and Path(jobs[0]["source_manifest_path"]).resolve() == source_path,
             "Original donor manifest differs from frozen job")
    source_pins = plan["source_bindings_sha256"]
    _require(str(source_path) in source_pins, "Unpinned original manifest")
    bindings[str(source_path)] = source_pins[str(source_path)]
    for path, expected in receipt["source_bindings"].items():
        path = Path(path).resolve()
        _require(path.is_relative_to(source_path.parent), "Source binding escaped donor")
        _require(str(path) not in bindings or bindings[str(path)] == expected, "Conflicting source binding")
        bindings[str(path)] = expected
    for name, expected in receipt["output_hashes"].items():
        bindings[str(audit.safe_asset(directory, name))] = expected
    _require(str(directory / "manifest.json") in bindings, "Generated manifest not bound")
    _verify(bindings)
    # Full existing verification, including saved output geometry and recipe/code pins.
    catalogue = plant_variant_catalogue.load_for_inspection(directory, plan_path)
    _require(catalogue["qualification_sha256"] == pin.qualification_sha256
             and catalogue["source_plan_sha256"] == pin.source_plan_sha256
             and catalogue["source_family"] == catalogue["split_group"] == family,
             "Catalogue provenance mismatch")
    result["catalogue_replayed"] = True
    source = audit.audit_manifest(source_path)
    _require(source["status"] != "blocked", "Original manifest audit blocked")
    for component in source["components"].values():
        path = str(audit.safe_asset(source_path.parent, component["file"]))
        _require(source_pins.get(path) == component["asset_sha256"], "Original mesh changed from frozen plan")
        _require(path not in bindings or bindings[path] == component["asset_sha256"], "Conflicting original mesh binding")
        bindings[path] = component["asset_sha256"]
    report = catalogue["report"]
    _require(Path(report["manifest_path"]).resolve() == directory / "manifest.json"
             and report["manifest_sha256"] == bindings[str(directory / "manifest.json")],
             "Generated catalogue manifest mismatch")
    targets = {t["component_id"]: t for t in receipt["targets"]}
    _require(len(targets) == len(receipt["targets"]), "Repeated qualification target")
    keys = sorted(targets) if component_ids is None else list(component_ids)
    _require(keys and all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys)
             and set(keys) <= targets.keys(), "Unique qualification target subset required")
    allowed = {r["component_id"] for r in catalogue["rows"]}
    rejection = {r["component_id"]: r["reasons"] for r in catalogue["rejected"]}
    from pxr import Usd
    result["runtime"]["usd"] = list(Usd.GetVersion())
    result["catalogue_sha256"] = _digest(catalogue)
    result["source_family"] = family
    for key in sorted(keys):
        ancestry = family + "/" + key
        _require(targets[key]["source_target_id"] == targets[key]["conservative_view_cap_group"] == ancestry,
                 "Qualification target ancestry mismatch")
        if key not in allowed:
            result["holds"].append(dict(source_target=ancestry, reason="catalogue_target_rejected",
                                        details=rejection.get(key, [])))
            continue
        try:
            reference = _chain(source["components"][key])
            _require(source["components"][key]["parent"] == report["components"][key]["parent"],
                     "Target parent ancestry changed")
            parent_id = source["components"][key]["parent"]
            _require({k: v for k, v in source["components"][parent_id].items() if k != "asset_sha256"}
                     == {k: v for k, v in report["components"][parent_id].items() if k != "asset_sha256"},
                     "V2 requires unchanged parent context")
            pair = []
            for kind, current in (("original", source), ("generated", report)):
                inputs, parent, meshes = _inputs(current, reference, key, bindings)
                try:
                    value = morphology_context.descriptor(**inputs)
                except ValueError as exc:
                    raise _Unsupported("unsupported_descriptor_frame: " + str(exc)) from exc
                identity = dict(kind=kind, source_family=family, source_target=ancestry,
                                manifest_sha256=current["manifest_sha256"], mesh_sha256=meshes)
                pair.append(dict(context_id="actual-output:" + _digest(identity), kind=kind,
                    source_family=family, source_target=ancestry,
                    target_id=ancestry if kind == "original" else catalogue["variant_id"] + "/" + key,
                    split="train", descriptor=value, descriptor_sha256=_digest(value),
                    normalized_fingerprint=morphology_context.fingerprint(value),
                    input_geometry=inputs, input_geometry_sha256=_digest(inputs), parent_chain=parent,
                    identity=identity, qualified_geometry=False, training_approved=False,
                    source_cap_reset=False, new_biological_family=False))
            for row in pair:
                row["source_context_id"] = pair[0]["context_id"]
            result["records"].extend(pair)
        except _Unsupported as exc:
            result["holds"].append(dict(source_target=ancestry, reason=str(exc)))
    return finish()
