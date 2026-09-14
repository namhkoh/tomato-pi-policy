"""Deterministic TRAIN-only petiole variants for a static-perception pilot.

This is not the missing growth generator, a new biological family, or a physics
model. Each intact leaf subtree receives one similarity transform about its
attachment. Mesh baking is handled by plant_variant_usd; no pixel labels survive.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .audit import audit_manifest, descendants, safe_asset
from .clear_scale_capacity import inventory
from .cut_regions import load_rule, propose_cut_region

VERSION = "petiole_similarity_pilot.v1"
# Engineering perturbations, NOT learned biology. Final dimensions/angles must
# also be within the observed TRAIN envelope. Identity is deliberately absent.
SCALES = (.9, 1.1)
AZIMUTHS = (-15., 15.)
TILTS = (-12., 12.)
PHYSICS_FIELDS = {
    "mass", "density", "density_kind", "collider", "joint", "diagonal_inertia",
    "center_of_mass", "stiffness", "stiffness_scale", "petiole_stiffness_scale",
    "petiole_zone_top", "damping_ratio", "damping_frac", "petiole_damping_frac",
    "joint_armature",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(value):
    v = np.asarray(value, dtype=float)
    require(v.shape == (3,) and np.isfinite(v).all() and np.linalg.norm(v) > 1e-10,
            "Finite nonzero 3-vector required")
    return v / np.linalg.norm(v)


def rotation(axis, degrees):
    a = unit(axis)
    require(math.isfinite(degrees), "Finite rotation required")
    x, y, z = a
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rad = math.radians(degrees)
    return np.eye(3) + math.sin(rad)*skew + (1-math.cos(rad))*(skew @ skew)


def validate_similarity(matrix, scale):
    r = np.asarray(matrix, dtype=float)
    require(r.shape == (3, 3) and np.isfinite(r).all()
            and np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0)
            and abs(np.linalg.det(r)-1) < 1e-10, "Proper orthonormal rotation required")
    require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
            "Positive uniform scale required")
    return r


def world_point(point, anchor, matrix, scale):
    r = validate_similarity(matrix, scale)
    return (np.asarray(anchor) + scale*(r @ (np.asarray(point)-anchor))).tolist()


def transformed_component(raw, anchor, matrix, scale):
    """Local points rotate/scale; plant-frame origins move about the same anchor."""
    r = validate_similarity(matrix, scale)
    c = deepcopy(raw)
    c["transform"] = {"translate": world_point(raw["transform"]["translate"], anchor, r, scale)}
    c["attach_point"] = world_point(raw["attach_point"], anchor, r, scale)
    c["axis"] = (r @ unit(raw["axis"])).tolist()
    if "capsules" in raw:
        c["capsules"] = [[[*((scale*r @ np.asarray(p[:3])).tolist()), p[3]*scale]
                          for p in chain] for chain in raw["capsules"]]
    for name in ("length", "radius", "stub_length"):
        if name in raw:
            c[name] = raw[name]*scale
    # Do not silently reuse donor inertia, joint constants or mass on new shapes.
    for name in PHYSICS_FIELDS:
        c.pop(name, None)
    return c


def normalized(raw):
    return dict(id=raw["id"], type=raw["type"], parent=raw["parent"],
                translation_plant_m=raw["transform"]["translate"],
                attachment_plant_m=raw["attach_point"], axis_plant=raw["axis"],
                capsules_local_m=raw.get("capsules", []), deleafed=raw.get("deleafed"))


def parent_tangent(component, parent):
    """Nearest capsule segment in the unchanged parent, not global vertical."""
    p = np.array(component["attachment_plant_m"]) - parent["translation_plant_m"]
    choices = []
    for chain in parent.get("capsules_local_m", []):
        for a, b in zip(chain, chain[1:]):
            a, b = np.array(a[:3]), np.array(b[:3])
            d = b-a
            if np.dot(d, d) <= 1e-16:
                continue
            t = np.clip(np.dot(p-a, d)/np.dot(d, d), 0, 1)
            choices.append((float(np.linalg.norm(p-a-t*d)), unit(d)))
    require(bool(choices), "Parent must have a nondegenerate centerline")
    return min(choices, key=lambda x: x[0])[1]


def dimensions(component, parent, rule):
    proposal = propose_cut_region(component, parent, rule)
    require(proposal["status"] == "proposed_geometry_only", "Unusable petiole centerline")
    # Tangent at the actual anchor (not at the 10mm cut when the curve bends).
    chain = component["capsules_local_m"][0]
    if proposal["source_chain_reversed"]:
        chain = chain[::-1]
    tangent = unit(np.array(chain[1][:3])-chain[0][:3])
    parent_axis = parent_tangent(component, parent)
    return dict(length_m=proposal["centerline_total_length_m"], radius_m=chain[0][3],
                attachment_angle_deg=math.degrees(math.acos(float(np.clip(
                    np.dot(tangent, parent_axis), -1, 1))))), tangent, parent_axis


def load_training_sources(plan_path):
    """Use frozen reservations verbatim; never fit on or load held-out meshes."""
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    inventory(plan)
    sources = {}
    pins = plan["source_bindings_sha256"]
    for job in plan["jobs"]:
        if job["split"] != "train":
            continue
        manifest_path = Path(job["source_manifest_path"]).resolve()
        require(manifest_path.parent.name == job["plant_family"], "Source family/path mismatch")
        require(pins.get(str(manifest_path)) == file_hash(manifest_path), "Source manifest binding changed")
        report = audit_manifest(manifest_path)
        require(report["status"] != "blocked", "Source structural audit failed")
        for c in report["components"].values():
            asset = safe_asset(manifest_path.parent, c["file"])
            require(pins.get(str(asset)) == c["asset_sha256"], "Source component binding changed")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources[job["plant_family"]] = dict(report=report, raw=raw, job=job)
    return plan, sources


def training_envelope(plan, sources):
    require(set(sources) == {k for k, s in plan["family_assignments"].items() if s == "train"},
            "Envelope must use exactly the frozen TRAIN families")
    rows, rejected = [], []
    rule = load_rule()
    for family, source in sorted(sources.items()):
        report = source["report"]
        for t in report["targets"]:
            if t["status"] == "excluded":
                continue
            c = report["components"][t["component_id"]]
            parent = report["components"][c["parent"]]
            try:
                metrics, _, _ = dimensions(c, parent, rule)
                rows.append(dict(source_family=family, component_id=c["id"], **metrics))
            except ValueError as exc:
                rejected.append(dict(source_family=family, component_id=c["id"], reason=str(exc)))
    require(bool(rows), "No valid training morphology")
    bounds = {field: [min(r[field] for r in rows), max(r[field] for r in rows)]
              for field in ("length_m", "radius_m", "attachment_angle_deg")}
    return dict(schema_version="greenhouse.training_morphology_envelope.v1",
                source_families=sorted(sources), source_manifest_hashes={
                    f: s["report"]["manifest_sha256"] for f, s in sorted(sources.items())},
                bounds=bounds, measured_candidates=len(rows), rejected=rejected,
                parameter_observations=rows, heldout_used=False,
                biological_distribution_validated=False,
                perturbations=dict(scale=list(SCALES), azimuth_deg=list(AZIMUTHS), tilt_deg=list(TILTS)),
                bounds_semantics="observed_train_min_max_not_botanical_safety",
                distribution_fitting="none_engineering_grid_filtered_by_train_envelope")


def plan_variant(source, envelope, seed, max_targets=8):
    require(type(seed) is int and 0 <= seed < 2**32, "Seed must be uint32")
    require(type(max_targets) is int and 1 <= max_targets <= 36, "Pilot requires 1..36 targets")
    report, raw, job = (source[k] for k in ("report", "raw", "job"))
    family = job["plant_family"]
    require(job["split"] == "train" and family in envelope["source_families"], "TRAIN-only generator pilot")
    require(envelope["source_manifest_hashes"][family] == report["manifest_sha256"],
            "Envelope/source mismatch")
    raw_by_id = {c["id"]: c for c in raw["components"]}
    generated = deepcopy(raw)
    generated["generator"], generated["version"], generated["seed"] = VERSION, "1.0.0", seed
    for key in PHYSICS_FIELDS:
        generated.pop(key, None)
    generated["physics_supported"] = False
    generated["intended_use"] = "static_perception_geometry_pilot_not_collection_approved"
    # Remove physics metadata from *all* output components; no stale donor physics.
    for c in generated["components"]:
        for key in PHYSICS_FIELDS:
            c.pop(key, None)
    out = {c["id"]: c for c in generated["components"]}
    changes, exclusions = [], []
    rule = load_rule()
    targets = sorted(job["targets"], key=lambda t: digest([VERSION, seed, t["component_id"]]))
    for t in targets:
        if len(changes) >= max_targets:
            break
        key = t["component_id"]
        c = report["components"][key]
        parent = report["components"][c["parent"]]
        members = descendants(report["components"], key)
        try:
            require(c["type"] == "sub_stem" and c["deleafed"] is False and parent["type"] == "main_stem",
                    "Only intact direct-main-stem petioles")
            require(all(report["components"][m]["type"] in ("sub_stem", "leaf") for m in members)
                    and any(report["components"][m]["type"] == "leaf" for m in members),
                    "Protected or leafless subtree")
            require(math.dist(c["translation_plant_m"], c["attachment_plant_m"]) <= 1e-8,
                    "Pilot requires a component origin at its attachment")
            _, tangent, paxis = dimensions(c, parent, rule)
            tilt_axis = unit(np.cross(tangent, paxis))
            combos = sorted(itertools.product(SCALES, AZIMUTHS, TILTS),
                            key=lambda x: digest([VERSION, seed, key, list(x)]))
            selected = None
            for scale, azimuth, tilt in combos:
                r = rotation(paxis, azimuth) @ rotation(tilt_axis, tilt)
                transformed = transformed_component(raw_by_id[key], c["attachment_plant_m"], r, scale)
                metrics, _, _ = dimensions(normalized(transformed), parent, rule)
                if all(envelope["bounds"][field][0] <= value <= envelope["bounds"][field][1]
                       for field, value in metrics.items()):
                    selected = (scale, azimuth, tilt, r, metrics)
                    break
            require(selected is not None, "No perturbation within measured training envelope")
            scale, azimuth, tilt, r, metrics = selected
            for member in members:
                out[member] = transformed_component(raw_by_id[member], c["attachment_plant_m"], r, scale)
            proposal = propose_cut_region(normalized(out[key]), parent, rule)
            shape = dict(parent_axis=paxis.tolist(), parent_capsules=parent["capsules_local_m"],
                         petiole_capsules=out[key]["capsules"], subtree_assets={
                             m: report["components"][m]["asset_sha256"] for m in members})
            changes.append(dict(component_id=key, source_target_id=t["target_id"],
                members=members, anchor_plant_m=c["attachment_plant_m"], rotation=r.tolist(), scale=scale,
                azimuth_deg=azimuth, tilt_deg=tilt, generated_dimensions=metrics,
                exact_morphology_hash=digest(shape),
                # Deliberately do not reset the view cap by inventing new target IDs.
                conservative_view_cap_group=t["target_id"], novelty_approved=False,
                cut_region_proposal=proposal, local_mesh_validation="pending",
                visual_review="pending", target_id=None))
        except ValueError as exc:
            exclusions.append(dict(component_id=key, reason=str(exc)))
    require(bool(changes), "No structurally usable variant targets")
    generated["components"] = [out[c["id"]] for c in raw["components"]]
    config = dict(version=VERSION, seed=seed, max_targets=max_targets,
                  source_family=family, source_manifest_sha256=report["manifest_sha256"],
                  envelope_sha256=digest(envelope), changes=[
                      {k: c[k] for k in ("component_id", "rotation", "scale")} for c in changes])
    variant_id = family + "_pv_" + digest(config)[:16]
    for change in changes:
        change["target_id"] = variant_id + "/" + change["component_id"]
    provenance = dict(schema_version="greenhouse.petiole_variant_plan.v1",
        version=VERSION, variant_id=variant_id, configuration=config,
        source_family=family, split="train", split_group=family,
        source_manifest_path=report["manifest_path"], source_manifest_sha256=report["manifest_sha256"],
        source_component_hashes={k: c["asset_sha256"] for k, c in report["components"].items()},
        changes=changes, exclusions=exclusions, generated_target_count=len(changes),
        new_independent_source_families=0, full_growth_generator=False,
        approved_novel_target_count=0, training_eligible=False, physics_validated=False,
        native_capture_validated=False, review_decisions_inherited=False,
        collision_validation="pending", state="planned_geometry_not_capture_approved")
    generated["variant_provenance"] = {k: provenance[k] for k in (
        "version", "variant_id", "source_family", "split", "split_group", "training_eligible")}
    return dict(manifest=generated, provenance=provenance)

