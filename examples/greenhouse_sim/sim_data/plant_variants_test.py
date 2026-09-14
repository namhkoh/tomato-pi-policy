"""Pure transform, lineage, reproducibility and real-package generator checks."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from sim_data.plant_variants import (
    PHYSICS_FIELDS, digest, dimensions, load_training_sources, normalized, plan_variant,
    rotation, training_envelope, transformed_component, unit, validate_similarity, world_point)
from sim_data.cut_regions import load_rule, propose_cut_region

PLAN = Path("data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json")


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(id="Petiole", type="sub_stem", parent="Stem",
            transform={"translate": [1., 2., 3.]}, attach_point=[1., 2., 3.],
            axis=[1., 0., 0.], capsules=[[[0., 0., 0., .003], [.1, 0., 0., .002]]],
            radius=.003, length=.1, stub_length=.004, deleafed=False,
            mass=.5, diagonal_inertia=[1, 2, 3], center_of_mass=[.05, 0, 0],
            joint={"type": "fixed_breakable"}, stiffness=12)
        self.parent = dict(id="Stem", type="main_stem", translation_plant_m=[1., 2., 3.],
                           capsules_local_m=[[[0, 0, -.1, .005], [0, 0, .1, .005]]])

    def test_rotation_right_handed_no_reflection(self):
        r = rotation([0, 0, 1], 90)
        np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(r), 1)
        for bad in (np.diag([-1, 1, 1]), np.diag([2, 1, 1]), np.full((3, 3), np.nan)):
            with self.subTest(matrix=str(bad)), self.assertRaises(ValueError):
                validate_similarity(bad, 1)

    def test_bad_axis_scale_rejected(self):
        for a in ([0, 0, 0], [0, 1], [math.nan, 0, 1]):
            with self.subTest(axis=a), self.assertRaises(ValueError):
                unit(a)
        for s in (0, -1, math.inf, True):
            with self.subTest(scale=s), self.assertRaises(ValueError):
                validate_similarity(np.eye(3), s)

    def test_anchor_fixed_radius_length_and_centerline_coherent(self):
        original = deepcopy(self.raw)
        r, s, a = rotation([0, 0, 1], 15), 1.1, self.raw["attach_point"]
        c = transformed_component(self.raw, a, r, s)
        self.assertEqual(self.raw, original)
        self.assertEqual(c["attach_point"], a)
        self.assertEqual(c["transform"]["translate"], a)
        self.assertAlmostEqual(c["capsules"][0][0][3], .0033)
        self.assertAlmostEqual(c["length"], .11)
        expected = s*r @ np.array([.1, 0, 0])
        np.testing.assert_allclose(c["capsules"][0][-1][:3], expected)
        self.assertFalse(PHYSICS_FIELDS.intersection(c))

    def test_descendant_world_mesh_and_attachment_use_one_similarity(self):
        leaf = deepcopy(self.raw)
        leaf["transform"]["translate"] = [1.1, 2., 3.]
        leaf["attach_point"] = [1.08, 2., 3.]
        r, s, a = rotation([0, 1, 0], -12), .9, self.raw["attach_point"]
        c = transformed_component(leaf, a, r, s)
        local_point = np.array([.03, .02, -.01])
        actual = np.array(c["transform"]["translate"]) + s*r @ local_point
        expected = world_point(np.array(leaf["transform"]["translate"])+local_point, a, r, s)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(c["attach_point"], world_point(leaf["attach_point"], a, r, s))

    def test_new_label_resamples_10mm_not_scaled_donor_label(self):
        r = rotation([0, 0, 1], 15)
        c = transformed_component(self.raw, self.raw["attach_point"], r, 1.1)
        proposal = propose_cut_region(normalized(c), self.parent, load_rule())
        self.assertEqual(proposal["status"], "proposed_geometry_only")
        distance = math.dist(proposal["nominal"]["point_plant_m"], self.raw["attach_point"])
        self.assertAlmostEqual(distance, .01)
        self.assertNotAlmostEqual(distance, .011)
        self.assertFalse(proposal["training_label_approved"])


@unittest.skipUnless(PLAN.is_file(), "Original frozen collection plan not available")
class RealSourcePlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.sources = load_training_sources(PLAN)
        cls.envelope = training_envelope(cls.plan, cls.sources)

    def test_envelope_only_train_and_measured_counts(self):
        self.assertEqual(len(self.sources), 16)
        self.assertEqual(self.envelope["measured_candidates"], 584)
        self.assertFalse(self.envelope["heldout_used"])
        self.assertEqual(set(self.sources), {k for k, v in self.plan["family_assignments"].items() if v == "train"})
        self.assertFalse(set(self.sources) & {k for k, v in self.plan["family_assignments"].items() if v != "train"})

    def test_missing_or_extra_family_rejected(self):
        for name in ("missing", "heldout"):
            sources = dict(self.sources)
            if name == "missing":
                sources.pop(next(iter(sources)))
            else:
                sources["seed13_full"] = next(iter(sources.values()))
            with self.subTest(name=name), self.assertRaises(ValueError):
                training_envelope(self.plan, sources)

    def test_deterministic_and_no_approval_or_new_family_claim(self):
        source = self.sources["seed101_full"]
        original = deepcopy(source)
        v = plan_variant(source, self.envelope, 20260915)
        self.assertEqual(v, plan_variant(source, self.envelope, 20260915))
        self.assertEqual(source, original)
        self.assertNotEqual(v["provenance"]["variant_id"],
                            plan_variant(source, self.envelope, 20260916)["provenance"]["variant_id"])
        p = v["provenance"]
        self.assertEqual(p["split_group"], "seed101_full")
        self.assertEqual(p["new_independent_source_families"], 0)
        self.assertEqual(p["approved_novel_target_count"], 0)
        for key in ("training_eligible", "physics_validated", "native_capture_validated",
                    "review_decisions_inherited"):
            self.assertFalse(p[key])
        json.dumps(v, allow_nan=False)

    def test_unchanged_protected_geometry_no_stale_physics_and_valid_labels(self):
        source = self.sources["seed101_full"]
        v = plan_variant(source, self.envelope, 20260915)
        old = {c["id"]: c for c in source["raw"]["components"]}
        new = {c["id"]: c for c in v["manifest"]["components"]}
        members = {k for change in v["provenance"]["changes"] for k in change["members"]}
        for key, c in new.items():
            self.assertFalse(PHYSICS_FIELDS.intersection(c))
            self.assertEqual(c["parent"], old[key]["parent"])
            if key not in members:
                self.assertEqual(c, {k: x for k, x in old[key].items() if k not in PHYSICS_FIELDS})
        for change in v["provenance"]["changes"]:
            self.assertFalse(change["novelty_approved"])
            self.assertEqual(change["conservative_view_cap_group"], change["source_target_id"])
            self.assertEqual(change["anchor_plant_m"], new[change["component_id"]]["attach_point"])
            self.assertEqual(change["cut_region_proposal"]["status"], "proposed_geometry_only")
            for key in change["members"]:
                self.assertIn(new[key]["type"], ("sub_stem", "leaf"))
            for field, value in change["generated_dimensions"].items():
                lo, hi = self.envelope["bounds"][field]
                self.assertLessEqual(lo, value)
                self.assertLessEqual(value, hi)

    def test_reject_bad_parameters_and_heldout_spoof(self):
        for seed, count in ((True, 8), (-1, 8), (2**32, 8), (1, 0), (1, 37), (1, True)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                plan_variant(self.sources["seed101_full"], self.envelope, seed, count)
        source = deepcopy(self.sources["seed101_full"])
        source["job"]["split"] = "test"
        with self.assertRaises(ValueError):
            plan_variant(source, self.envelope, 1)
        source = deepcopy(self.sources["seed101_full"])
        source["report"]["manifest_sha256"] = "changed"
        with self.assertRaises(ValueError):
            plan_variant(source, self.envelope, 1)

    def test_protected_descendant_or_detached_anchor_cannot_be_generated(self):
        for mode in ("fruit", "anchor"):
            source = deepcopy(self.sources["seed101_full"])
            source["job"]["targets"] = source["job"]["targets"][:1]
            key = source["job"]["targets"][0]["component_id"]
            if mode == "fruit":
                child = next(c for c in source["report"]["components"].values() if c["parent"] == key)
                child["type"] = "fruit"
            else:
                source["report"]["components"][key]["translation_plant_m"][0] += .01
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plan_variant(source, self.envelope, 1)


if __name__ == "__main__":
    unittest.main()

