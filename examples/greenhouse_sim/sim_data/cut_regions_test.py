"""Arc-length labels preserve the prototype/approval boundary."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from sim_data.audit import DEFAULT_PACK, audit_manifest
from sim_data.candidate_branches import plan_branches
from sim_data.cut_regions import add_cut_region_overlay, load_rule, propose_cut_region, rule_fingerprint
from sim_data.label_drafts import prepare_drafts, markdown_packet


class CutRegionTests(unittest.TestCase):
    def setUp(self):
        self.rule = load_rule()
        self.parent = {"id": "Main", "type": "main_stem"}
        self.component = {"id": "Petiole", "type": "sub_stem", "parent": "Main", "deleafed": False,
                          "translation_plant_m": [1, 2, 3], "attachment_plant_m": [1, 2, 3],
                          "axis_plant": [1, 0, 0],
                          "capsules_local_m": [[[0, 0, 0, .003], [.03, 0, 0, .0015]]]}

    def proposal(self):
        return propose_cut_region(self.component, self.parent, self.rule)

    def test_straight_distances_radius_tangent_and_no_approval(self):
        before = deepcopy(self.component)
        result = self.proposal()
        self.assertEqual(result["status"], "proposed_geometry_only")
        self.assertEqual(result["nominal"]["point_plant_m"], [1.01, 2, 3])
        self.assertAlmostEqual(result["nominal"]["petiole_radius_m"], .0025)
        self.assertEqual(result["nominal"]["tangent_plant"], [1, 0, 0])
        region = result["accepted_centerline_interval"]
        self.assertEqual(region["samples"][-1]["point_plant_m"], [1.02, 2, 3])
        self.assertEqual(region["arc_range_m"], [.01, .02])
        self.assertIsNone(region["radial_tolerance_m"])
        self.assertFalse(result["training_label_approved"])
        self.assertFalse(result["physical_execution_validated"])
        self.assertEqual(result["blade_clearance"], "not_tested")
        self.assertEqual(result["human_cut_review"], "pending")
        self.assertEqual(before, self.component)
        json.dumps(result, allow_nan=False)

    def test_curved_branch_uses_arc_length_not_straight_axis_or_euclidean_distance(self):
        self.component["capsules_local_m"] = [[[0, 0, 0, .002], [.006, 0, 0, .002],
                                                [.006, .009, 0, .002], [.026, .009, 0, .001]]]
        result = self.proposal()
        self.assertLess(math.dist(result["nominal"]["point_plant_m"], [1.006, 2.004, 3]), 1e-12)
        self.assertEqual(result["nominal"]["tangent_plant"], [0, 1, 0])
        region = result["accepted_centerline_interval"]["samples"]
        self.assertEqual([p["arc_distance_m"] for p in region], [.01, .015, .02])
        self.assertAlmostEqual(sum(math.dist(a["point_plant_m"], b["point_plant_m"])
                                   for a, b in zip(region, region[1:])), .01)
        self.assertLess(math.dist(result["nominal"]["point_plant_m"], [1, 2, 3]), .01)

    def test_reverse_chain_orientation_does_not_change_cut_points(self):
        original = self.proposal()
        self.component["capsules_local_m"][0].reverse()
        result = self.proposal()
        self.assertTrue(result["source_chain_reversed"])
        self.assertEqual(result["nominal"], original["nominal"])
        self.assertEqual(result["accepted_centerline_interval"], original["accepted_centerline_interval"])

    def test_component_origin_is_not_assumed_to_equal_attachment(self):
        self.component["attachment_plant_m"] = [1.005, 2, 3]
        self.component["capsules_local_m"][0][0][0] = .005
        result = self.proposal()
        self.assertEqual(result["status"], "proposed_geometry_only")
        self.assertAlmostEqual(result["nominal"]["point_plant_m"][0], 1.015)

    def test_short_ambiguous_invalid_and_misaligned_branches_get_no_coordinates(self):
        cases = {
            "short": lambda c: c["capsules_local_m"][0][-1].__setitem__(0, .015),
            "multi_chain": lambda c: c["capsules_local_m"].append(deepcopy(c["capsules_local_m"][0])),
            "no_chain": lambda c: c.__setitem__("capsules_local_m", []),
            "unattached": lambda c: c.__setitem__("attachment_plant_m", [10, 10, 10]),
            "degenerate": lambda c: c["capsules_local_m"][0].insert(1, c["capsules_local_m"][0][0][:]),
            "nonfinite": lambda c: c["capsules_local_m"][0][1].__setitem__(0, float("nan")),
            "invalid_radius": lambda c: c["capsules_local_m"][0][1].__setitem__(3, -1),
            "wrong_direction": lambda c: c.__setitem__("axis_plant", [-1, 0, 0]),
            "deleafed": lambda c: c.__setitem__("deleafed", True),
            "wrong_parent": lambda c: c.__setitem__("parent", "Other"),
        }
        for case, mutate in cases.items():
            component = deepcopy(self.component)
            mutate(component)
            result = propose_cut_region(component, self.parent, self.rule)
            with self.subTest(case=case):
                self.assertEqual(result["status"], "geometry_flagged")
                self.assertTrue(result["reason_codes"])
                self.assertIsNone(result["nominal"])
                self.assertIsNone(result["accepted_centerline_interval"])
                json.dumps(result, allow_nan=False)

    def test_rule_validation_and_fingerprint(self):
        before = rule_fingerprint(self.rule)
        modified = deepcopy(self.rule)
        modified["nominal_distance_m"] = .015
        self.assertNotEqual(before, rule_fingerprint(modified))
        for key, value in (("nominal_distance_m", .009), ("accepted_interval_m", [.02, .01]),
                           ("units", "mm"), ("physical_execution_validated", True),
                           ("anchor_match_tolerance_m", .01), ("horticultural_validation", "approved")):
            modified = deepcopy(self.rule)
            modified[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                rule_fingerprint(modified)

    def test_parent_proxy_warning_is_not_silently_treated_as_safe_cut(self):
        self.parent.update(translation_plant_m=[1, 2, 3],
                           capsules_local_m=[[[0, 0, -.1, .009], [0, 0, .1, .009]]])
        result = self.proposal()
        self.assertAlmostEqual(result["parent_proxy_diagnostic"]["nominal_point_to_parent_surface_m"], .001)
        self.assertIn("nominal_petiole_envelope_may_overlap_parent_capsule", result["geometry_warnings"])
        self.assertFalse(result["physical_execution_validated"])
        self.assertEqual(result["blade_clearance"], "not_tested")

    def test_overlay_points_match_json_and_invalid_geometry_has_no_overlay(self):
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("USD runtime required")
        stage = Usd.Stage.CreateInMemory()
        stage.SetEditTarget(stage.GetSessionLayer())
        root_before = stage.GetRootLayer().ExportToString()
        proposal = self.proposal()
        self.assertTrue(add_cut_region_overlay(stage, {"cut_region_proposal": proposal}))
        path = "/World/DraftLabels/ProposedCutInterval_NOT_Approved"
        points = UsdGeom.BasisCurves(stage.GetPrimAtPath(path)).GetPointsAttr().Get()
        for actual, expected in zip(points, proposal["accepted_centerline_interval"]["samples"]):
            self.assertLess(math.dist(actual, expected["point_plant_m"]), 1e-6)
        self.assertEqual(stage.GetRootLayer().ExportToString(), root_before)
        empty = Usd.Stage.CreateInMemory()
        self.assertFalse(add_cut_region_overlay(empty, {"cut_region_proposal": {"status": "geometry_flagged"}}))
        self.assertFalse(list(empty.Traverse()))

    @unittest.skipUnless(DEFAULT_PACK.is_dir(), "Extracted source package required")
    def test_six_real_variants_opt_in_proposals_leave_original_reviews_and_labels_unset(self):
        variants = [plan_branches(audit_manifest(DEFAULT_PACK / "plants/components" / name / "manifest.json"))
                    for name in ("seed101_full", "seed103_full")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "variants.json"
            path.write_text(json.dumps(variants))
            before = path.read_bytes()
            plain, _ = prepare_drafts(path)
            packet, reports = prepare_drafts(path, self.rule)
            self.assertNotIn("cut_rule", plain)
            self.assertEqual(packet["cut_rule_sha256"], rule_fingerprint(self.rule))
            self.assertEqual(len(packet["labels"]), 6)
            for row in packet["labels"]:
                result = row["cut_region_proposal"]
                self.assertEqual(result["status"], "proposed_geometry_only")
                self.assertEqual(result["nominal"]["arc_distance_m"], .01)
                self.assertAlmostEqual(math.dist(result["nominal"]["point_plant_m"], row["attachment_plant_m"]), .01)
                self.assertEqual(row["human_review_status"], "pending")
                self.assertFalse(row["cut_approval"])
                self.assertIsNone(row["canonical_cut_point_m"])
                self.assertIsNone(row["admissible_cut_region"])
                self.assertIsNone(row["grasp_region"])
            self.assertIn("magenta", markdown_packet(packet))
            self.assertIn("10-20 mm", markdown_packet(packet))
            self.assertTrue(all(t["canonical_cut_point_m"] is None for r in reports for t in r["targets"]))
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
