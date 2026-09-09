"""Versioned, arc-length cut proposals. No execution or horticultural approval."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .audit import finite_number, vector

DEFAULT_RULE = Path(__file__).with_name("configs") / "cut_rule_prototype_v1.json"


def validate_rule(rule):
    if rule.get("schema_version") != "greenhouse.cut_rule.v1" or not isinstance(rule.get("rule_id"), str) or not rule["rule_id"]:
        raise ValueError("Invalid cut-rule identity/schema")
    if (rule.get("units") != "meters" or rule.get("distance_reference") !=
            "manifest_attachment_along_petiole_centerline_toward_leaves"):
        raise ValueError("Unsupported cut-rule coordinate convention")
    interval, nominal = rule.get("accepted_interval_m"), rule.get("nominal_distance_m")
    if (not vector(interval, 2) or not finite_number(nominal)
            or not 0 < interval[0] <= nominal <= interval[1] or interval[0] >= interval[1]):
        raise ValueError("Invalid nominal/accepted arc-length interval")
    tolerance = rule.get("anchor_match_tolerance_m")
    if not finite_number(tolerance) or not 0 < tolerance <= 1e-5:
        raise ValueError("Anchor tolerance must be a small numerical tolerance, not a geometric offset")
    if (rule.get("status") != "prototype_user_agreed" or rule.get("horticultural_validation") != "pending"
            or rule.get("physical_execution_validated") is not False):
        raise ValueError("This exporter only supports unvalidated prototype cut rules")


def load_rule(path=DEFAULT_RULE):
    raw = Path(path).read_bytes()
    rule = json.loads(raw)
    validate_rule(rule)
    return rule


def rule_fingerprint(rule):
    validate_rule(rule)
    return hashlib.sha256(json.dumps(rule, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _oriented_chain(component, tolerance):
    if component.get("type") != "sub_stem" or component.get("deleafed") is not False:
        raise ValueError("not_an_intact_petiole")
    origin, attachment, axis = (component.get(k) for k in ("translation_plant_m", "attachment_plant_m", "axis_plant"))
    if not all(vector(v) for v in (origin, attachment, axis)) or math.dist(axis, [0, 0, 0]) < 1e-8:
        raise ValueError("invalid_component_frame")
    chains = component.get("capsules_local_m")
    # Multiple chains may represent a branching network. Do not guess a trunk.
    if not isinstance(chains, list) or len(chains) != 1:
        raise ValueError("missing_or_ambiguous_centerline")
    chain = deepcopy(chains[0])
    if not isinstance(chain, list) or len(chain) < 2 or not all(vector(p, 4) and p[3] > 0 for p in chain):
        raise ValueError("invalid_centerline_points")
    local_attachment = [attachment[i] - origin[i] for i in range(3)]
    endpoint_errors = [math.dist(chain[j][:3], local_attachment) for j in (0, -1)]
    matches = [i for i, e in enumerate(endpoint_errors) if e <= tolerance]
    if len(matches) != 1:
        raise ValueError("attachment_not_unique_centerline_endpoint")
    reversed_chain = matches[0] == 1
    if reversed_chain:
        chain.reverse()
    cumulative = [0.0]
    for a, b in zip(chain, chain[1:]):
        distance = math.dist(a[:3], b[:3])
        if distance <= 1e-12:
            raise ValueError("degenerate_centerline_segment")
        cumulative.append(cumulative[-1] + distance)
    direction = [chain[1][i] - chain[0][i] for i in range(3)]
    if sum(direction[i] * axis[i] for i in range(3)) <= 0:
        raise ValueError("centerline_direction_disagrees_with_petiole_axis")
    return chain, cumulative, reversed_chain, min(endpoint_errors)


def _sample(chain, cumulative, distance, origin):
    if distance < 0 or distance > cumulative[-1] + 1e-12:
        raise ValueError("centerline_too_short_for_rule")
    # At a vertex the outward segment defines the one-sided tangent.
    index = next((i for i in range(len(chain) - 1) if distance < cumulative[i + 1] - 1e-12), len(chain) - 2)
    a, b = chain[index], chain[index + 1]
    length = cumulative[index + 1] - cumulative[index]
    t = (distance - cumulative[index]) / length
    return {"arc_distance_m": distance,
            "point_plant_m": [origin[i] + a[i] + t * (b[i] - a[i]) for i in range(3)],
            "tangent_plant": [(b[i] - a[i]) / length for i in range(3)],
            "petiole_radius_m": a[3] + t * (b[3] - a[3])}


def propose_cut_region(component, parent, rule):
    """Invalid geometry returns an explicit rejection and no point/region."""
    validate_rule(rule)
    result = {"schema_version": "greenhouse.cut_region_proposal.v1", "rule_id": rule["rule_id"],
              "rule_sha256": rule_fingerprint(rule), "status": "geometry_flagged", "reason_codes": [],
              "coordinate_frame": "plant", "distance_metric": "polyline_arc_length_m",
              "nominal": None, "accepted_centerline_interval": None,
              "horticultural_validation": "pending", "human_cut_review": "pending",
              "training_label_approved": False, "physical_execution_validated": False,
              "blade_clearance": "not_tested", "protected_organ_clearance": "not_tested",
              "proposed_cut_plane_normal_plant": None,
              "cut_plane_is_tool_pose": False, "geometry_warnings": []}
    try:
        if parent.get("type") != "main_stem" or component.get("parent") != parent.get("id"):
            raise ValueError("not_direct_main_stem_petiole")
        chain, cumulative, reversed_chain, error = _oriented_chain(component, rule["anchor_match_tolerance_m"])
        low, high = rule["accepted_interval_m"]
        if cumulative[-1] < high - 1e-12:
            raise ValueError("centerline_too_short_for_rule")
        origin = component["translation_plant_m"]
        distances = [low] + [s for s in cumulative if low < s < high] + [high]
        samples = [_sample(chain, cumulative, s, origin) for s in distances]
        nominal = _sample(chain, cumulative, rule["nominal_distance_m"], origin)
        result.update(status="proposed_geometry_only", nominal=nominal,
                      accepted_centerline_interval={"arc_range_m": [low, high], "samples": samples,
                                                    "shape": "centerline_segment_not_sphere",
                                                    "radial_tolerance_m": None},
                      proposed_cut_plane_normal_plant=nominal["tangent_plant"],
                      centerline_total_length_m=cumulative[-1], source_chain_reversed=reversed_chain,
                      attachment_to_centerline_endpoint_error_m=error)
        # Only a local proxy diagnostic, not a mesh or blade-stroke check.
        gaps = []
        parent_origin = parent.get("translation_plant_m")
        if vector(parent_origin):
            point = [nominal["point_plant_m"][i] - parent_origin[i] for i in range(3)]
            for parent_chain in parent.get("capsules_local_m", []):
                for a, b in zip(parent_chain, parent_chain[1:]):
                    if not (vector(a, 4) and vector(b, 4) and a[3] > 0 and b[3] > 0):
                        continue
                    direction = [b[i] - a[i] for i in range(3)]
                    length2 = sum(x * x for x in direction)
                    if length2 <= 1e-16:
                        continue
                    t = max(0.0, min(1.0, sum((point[i] - a[i]) * direction[i] for i in range(3)) / length2))
                    gaps.append(math.dist(point, [a[i] + t * direction[i] for i in range(3)])
                                - (a[3] + t * (b[3] - a[3])))
        result["parent_proxy_diagnostic"] = {"scope": "nominal_point_only_approximate_capsule_distance",
            "nominal_point_to_parent_surface_m": min(gaps) if gaps else None,
            "full_interval_or_blade_clearance_validated": False}
        if gaps and min(gaps) < nominal["petiole_radius_m"]:
            result["geometry_warnings"].append("nominal_petiole_envelope_may_overlap_parent_capsule")
        elif not gaps:
            result["geometry_warnings"].append("parent_capsule_diagnostic_unavailable")
    except ValueError as exc:
        result["reason_codes"] = [str(exc)]
    return result


def add_cut_region_overlay(stage, row, root="/World/DraftLabels"):
    """Display-only widths/markers; geometry remains under the active edit layer."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    proposal = row.get("cut_region_proposal", {})
    if proposal.get("status") != "proposed_geometry_only":
        return False
    materials = {}
    for name, colour in {"CutIntervalMaterial": (1, .015, .75), "NominalCutMaterial": (1, 1, 1)}.items():
        material = UsdShade.Material.Define(stage, root + "/" + name)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour) * .1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material
    samples = proposal["accepted_centerline_interval"]["samples"]
    curve = UsdGeom.BasisCurves.Define(stage, root + "/ProposedCutInterval_NOT_Approved")
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(samples)])
    curve.CreatePointsAttr([Gf.Vec3f(*s["point_plant_m"]) for s in samples])
    curve.CreateWidthsAttr([s["petiole_radius_m"] * 2.4 for s in samples])
    curve.SetWidthsInterpolation("vertex")
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(materials["CutIntervalMaterial"])
    marker = UsdGeom.Sphere.Define(stage, root + "/NominalCut_NOT_Approved")
    marker.CreateRadiusAttr(.003)
    marker.AddTranslateOp().Set(Gf.Vec3d(*proposal["nominal"]["point_plant_m"]))
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(materials["NominalCutMaterial"])
    return True
