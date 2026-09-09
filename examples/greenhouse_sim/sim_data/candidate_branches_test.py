"""Candidate variant tests against the actual supplied package, with optional USD."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import unittest

from sim_data.audit import DEFAULT_PACK, audit_manifest
from sim_data.candidate_branches import RECIPES, add_candidate_branches, plan_branches

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Usd = None


@unittest.skipUnless(DEFAULT_PACK.is_dir(), "Supplied package not extracted")
class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {name: audit_manifest(DEFAULT_PACK / "plants/components" / name / "manifest.json")
                       for name in RECIPES}

    def test_counts_source_identity_pending_labels_and_original_unchanged(self):
        for name, report in self.reports.items():
            original = deepcopy(report)
            for count in (1, 2, 3):
                with self.subTest(plant=name, count=count):
                    variant = plan_branches(report, count)
                    self.assertEqual(variant, plan_branches(report, count))
                    self.assertEqual(len(variant["candidates"]), count)
                    self.assertEqual(variant["split_group"], name)
                    self.assertNotEqual(variant["variant_id"], name)
                    self.assertFalse(variant["physics_enabled"])
                    self.assertFalse(variant["annotation_workflow_supported"])
                    self.assertEqual(report, original)
                    json.dumps(variant, allow_nan=False)
                    for target in variant["candidates"]:
                        self.assertEqual(target["state"], "attached")
                        self.assertEqual(target["anatomy_review"], "pending")
                        self.assertGreater(target["descendant_type_counts"]["leaf"], 0)
                        self.assertEqual(target["protected_descendant_ids"], [])
                        for field in ("canonical_cut_point_m", "admissible_cut_region", "grasp_region"):
                            self.assertIsNone(target[field])
                        self.assertLessEqual(target["parent_capsule_gap_m"], .002)

    def test_rigid_subtree_translation_keeps_every_leaf_parent_capsule_and_axis(self):
        for report in self.reports.values():
            variant = plan_branches(report)
            for target in variant["candidates"]:
                for key in target["expected_detached_component_ids"]:
                    c = variant["added_components"][key]
                    original = report["components"][c["source_component_id"]]
                    self.assertEqual(c["axis_plant"], original["axis_plant"])
                    self.assertEqual(c["capsules_local_m"], original["capsules_local_m"])
                    self.assertEqual(c["asset_sha256"], original["asset_sha256"])
                    for field in ("attachment_plant_m", "translation_plant_m"):
                        expected = [original[field][i] + target["translation_from_donor_m"][i] for i in range(3)]
                        self.assertLess(math.dist(c[field], expected), 1e-12)
                    if key != target["component_id"]:
                        parent = variant["added_components"][c["parent"]]
                        self.assertEqual(parent["source_component_id"], original["parent"])

    def test_signature_changes_with_recipe_or_source_geometry(self):
        report = deepcopy(self.reports["seed101_full"])
        initial = plan_branches(report)["variant_id"]
        self.assertNotEqual(initial, plan_branches(report, 2)["variant_id"])
        report["components"]["SubStem_56"]["asset_sha256"] = "changed"
        self.assertNotEqual(initial, plan_branches(report)["variant_id"])

    def test_reject_bad_counts_unknown_plants_and_blocked_sources(self):
        report = deepcopy(self.reports["seed101_full"])
        for count in (0, -1, 4, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                plan_branches(report, count)
        report["plant_id"] = "unknown"
        with self.assertRaises(ValueError):
            plan_branches(report)
        report["status"] = "blocked"
        with self.assertRaises(ValueError):
            plan_branches(report)

    def test_reject_nonempty_stub_protected_donor_and_detached_attachment(self):
        for mutation in ("nonempty", "intact", "protected", "gap", "origin", "direction"):
            report = deepcopy(self.reports["seed101_full"])
            components = report["components"]
            if mutation == "nonempty":
                components["SubStem_56"]["parent"] = "SubStem_35"
            elif mutation == "intact":
                components["SubStem_35"]["deleafed"] = False
            elif mutation == "protected":
                leaf = next(c for c in components.values() if c["parent"] == "SubStem_56")
                leaf["type"] = "fruit"
            elif mutation == "gap":
                components["SubStem_35"]["attachment_plant_m"][0] += 10
            elif mutation == "origin":
                components["SubStem_56"]["translation_plant_m"][0] += .01
            else:
                components["SubStem_56"]["axis_plant"][0] = -1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan_branches(report)

    @unittest.skipIf(Usd is None, "Run USD checks with Isaac Python")
    def test_real_usd_session_only_geometry_world_attachments_and_idempotency_guard(self):
        from sim_data.geometry import assemble_plant, bounds_by_component, point_bounds_distance
        for name, report in self.reports.items():
            directory = Path(report["manifest_path"]).parent
            fingerprints = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
            stage = Usd.Stage.CreateInMemory()
            paths = assemble_plant(stage, "/World/Plant", report)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                root = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Plant"))
                root.AddTranslateOp().Set(Gf.Vec3d(-.005, .5, .9))
                root.AddRotateZOp().Set(25)
            original_session = stage.GetSessionLayer().ExportToString()
            original_root = stage.GetRootLayer().ExportToString()
            record = dict(plant_root="/World/Plant", manifest_path=report["manifest_path"], component_paths=paths)
            variant = add_candidate_branches(stage, record)
            self.assertEqual(stage.GetRootLayer().ExportToString(), original_root)
            combined_paths = {**paths, **variant["added_component_paths"]}
            bounds, mesh_counts = bounds_by_component(stage, combined_paths)
            cache = UsdGeom.XformCache()
            matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Plant"))
            for key, c in variant["added_components"].items():
                actual = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(combined_paths[key])).ExtractTranslation()
                expected = matrix.Transform(Gf.Vec3d(*c["translation_plant_m"]))
                self.assertLess(math.dist(actual, expected), 1e-9)
                self.assertGreater(mesh_counts[key], 0)
            for target in variant["candidates"]:
                self.assertFalse(stage.GetPrimAtPath(paths[target["replaced_stub_id"]]).IsActive())
                self.assertTrue(stage.GetPrimAtPath(paths[target["source_donor_id"]]).IsActive())
                point = target["attachment_world_m"]
                for key in (target["component_id"], target["parent_component_id"]):
                    self.assertLessEqual(point_bounds_distance(point, bounds[key]), .002)
            after = stage.GetSessionLayer().ExportToString()
            with self.assertRaisesRegex(ValueError, "already exists"):
                add_candidate_branches(stage, record)
            self.assertEqual(stage.GetSessionLayer().ExportToString(), after)
            stage.GetSessionLayer().ImportFromString(original_session)
            self.assertTrue(all(stage.GetPrimAtPath(paths[k]).IsActive() for k in variant["replaced_stub_ids"]))
            self.assertFalse(any(stage.GetPrimAtPath(p) for p in variant["added_component_paths"].values()))
            self.assertEqual(fingerprints, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in fingerprints})


if __name__ == "__main__":
    unittest.main()
