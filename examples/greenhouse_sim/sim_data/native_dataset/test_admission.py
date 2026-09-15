"""Pure-stdlib synthetic accounting tests, NOT measured native qualification."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import random
import unittest

from .admission import Policy, decoded_rgb_digest, finalize_context_groups, select_candidates


def policy(**changes):
    # These toy exact-match thresholds are not calibrated native thresholds.
    gates = {kind: dict(metric="synthetic-" + kind, threshold=0.0,
                        evidence_id="test-only-calibration", validated=True)
             for kind in ("geometry", "near_image")}
    return replace(Policy("toy.v1", dict(train=20000, validation=2, test=3), gates), **changes)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def context(family="A", target="A/p", source="root", qualified=False):
    return dict(source_family=family, source_target=target, source_context_id=source,
                qualified_geometry=qualified, geometry_evidence_id="toy-geometry" if qualified else None)


def view(key, cid="root", family="A", target="A/p", **changes):
    result = dict(sample_id=key, target_id="generated/" + key, context_id=cid,
                  source_family=family, source_target=target,
                  decoded_rgb_sha256=decoded_rgb_digest(bytes.fromhex(sha(key))[:6], width=2, height=1),
                  scene_sha256=sha("scene:" + key), camera_sha256=sha("pose-and-optics"),
                  annotation_review=dict(passed=True, method="automatic", evidence_id="toy-replay"))
    result.update(changes)
    return result


def select(rows, contexts=None, splits=None, edges=(), near=(), declaration=None):
    declaration = declaration or policy()
    groups = finalize_context_groups(contexts if contexts is not None else {"root": context()}, edges,
                                     policy=declaration, evidence_id="toy-global-geometry-comparison")
    return select_candidates(rows, splits if splits is not None else {"A": "train"}, groups,
                             policy=declaration, near_image_edges=near,
                             image_inventory_evidence_id="toy-global-image-comparison")


class AdmissionTests(unittest.TestCase):
    def test_later_bridge_merges_groups_before_cap_accounting(self):
        contexts = {"root": context(), **{k: context(qualified=True) for k in "abcd"}}
        rows = [view(f"{k}-{i}", k) for k in "abcd" for i in range(8)]
        edges = [("a", "b"), ("c", "d")]
        self.assertEqual(select(rows, contexts, edges=edges)["counts"]["train"], 24)
        edges.append(("b", "c"))
        result = select(rows, contexts, edges=edges)
        self.assertEqual(result["counts"]["train"], 12)
        self.assertEqual(len(result["context_counts"]), 1)
        self.assertEqual(Counter(r for rs in result["excluded"].values() for r in rs), {"context_cap": 20})
        self.assertEqual(result, select(rows[::-1], dict(reversed(list(contexts.items()))), edges=edges[::-1]))

    def test_rigid_scale_rename_and_generated_ids_cannot_reset_cap(self):
        contexts = {"root": context(), **{f"rigid-scale-rename-{i}": context() for i in range(30)}}
        rows = [view(f"new-generated-id-{i}", f"rigid-scale-rename-{i}") for i in range(30)]
        for cap in (12, 3):
            with self.subTest(cap=cap):
                result = select(rows, contexts, declaration=policy(max_views_per_context=cap))
                self.assertEqual(result["counts"]["train"], cap)
                self.assertEqual(len({r["augmentation_context_id"] for r in result["selected"]}), 1)
                self.assertEqual({r["source_target"] for r in result["selected"]}, {"A/p"})
                self.assertFalse(result["new_biological_families"])

    def test_qualified_context_budgets_preserve_original_source(self):
        contexts = {"root": context(), "bend": context(qualified=True), "leaf": context(qualified=True)}
        rows = [view(f"{k}-{i}", k) for k in ("bend", "leaf") for i in range(13)]
        result = select(rows, contexts)
        self.assertEqual(result["counts"]["train"], 24)
        self.assertEqual(sorted(result["context_counts"].values()), [12, 12])
        self.assertEqual({r["source_target"] for r in result["selected"]}, {"A/p"})
        self.assertEqual(select(rows, contexts, edges=[("root", "bend"), ("bend", "leaf")])["counts"]["train"], 12)

    def test_context_cap_is_global_not_per_donor_or_target(self):
        contexts = {"root": context(), "b": context("B", "B/p", "b")}
        rows = [view(f"a-{i}") for i in range(10)] + [view(f"b-{i}", "b", "B", "B/p") for i in range(10)]
        result = select(rows, contexts, {"A": "train", "B": "train"}, edges=[("root", "b")])
        self.assertEqual(result["family_counts"], {"A": 6, "B": 6})
        self.assertEqual(list(result["context_counts"].values()), [12])

    def test_cross_split_context_rejects_even_without_heldout_view(self):
        contexts = {"root": context(), "held": context("V", "V/p", "held")}
        result = select([view("train")], contexts, {"A": "train", "V": "validation"}, edges=[("root", "held")])
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["excluded"]["train"], ["cross_split_context_duplicate"])

    def test_cross_split_rgb_duplicate_rejects_both_including_failed_review(self):
        contexts = {"root": context(), "held": context("V", "V/p", "held")}
        a, b = view("a"), view("b", "held", "V", "V/p")
        b["decoded_rgb_sha256"] = a["decoded_rgb_sha256"]
        b["annotation_review"]["passed"] = False
        result = select([a, b], contexts, {"A": "train", "V": "validation"})
        self.assertFalse(result["selected"])
        for key in ("a", "b"):
            self.assertIn("cross_split_view_duplicate", result["excluded"][key])

    def test_same_scene_camera_noisy_rgb_rejected_but_changed_pose_allowed(self):
        a, b = view("a"), view("b")
        self.assertNotEqual(a["decoded_rgb_sha256"], b["decoded_rgb_sha256"])
        b["scene_sha256"] = a["scene_sha256"]
        result = select([b, a])
        self.assertEqual([r["sample_id"] for r in result["selected"]], ["a"])
        self.assertEqual(result["excluded"]["b"], ["duplicate_view"])
        b["camera_sha256"] = sha("different-pose")
        self.assertEqual(select([a, b])["counts"]["train"], 2)

    def test_cross_split_same_camera_noisy_rgb_rejected(self):
        contexts = {"root": context(), "held": context("T", "T/p", "held")}
        a, b = view("a"), view("b", "held", "T", "T/p")
        b["scene_sha256"] = a["scene_sha256"]
        result = select([a, b], contexts, {"A": "train", "T": "test"})
        self.assertFalse(result["selected"])
        self.assertTrue(all("cross_split_view_duplicate" in rs for rs in result["excluded"].values()))

    def test_near_image_bridge_connects_rgb_and_pose_groups_through_failed_review(self):
        contexts = {"root": context(), "held": context("V", "V/p", "held")}
        rows = [view(k) for k in "abc"] + [view("d", "held", "V", "V/p")]
        rows[1]["annotation_review"]["passed"] = False
        rows[1]["decoded_rgb_sha256"] = rows[0]["decoded_rgb_sha256"]
        rows[3]["scene_sha256"] = rows[2]["scene_sha256"]
        result = select(rows, contexts, {"A": "train", "V": "validation"}, near=[("b", "c")])
        self.assertFalse(result["selected"])
        self.assertEqual(len(set(result["view_duplicate_groups"].values())), 1)
        self.assertTrue(all("cross_split_view_duplicate" in rs for rs in result["excluded"].values()))

    def test_decoded_rgb_duplicate_ignores_encoded_file_metadata(self):
        a, b = view("a"), view("b")
        b["decoded_rgb_sha256"] = a["decoded_rgb_sha256"]
        a["encoded_png_sha256"], b["encoded_png_sha256"] = sha("encoding1"), sha("encoding2")
        self.assertEqual(select([a, b])["counts"]["train"], 1)
        pixels = bytes(range(18))
        self.assertEqual(decoded_rgb_digest(pixels, width=3, height=2),
                         decoded_rgb_digest(memoryview(pixels), width=3, height=2))
        self.assertNotEqual(decoded_rgb_digest(pixels, width=3, height=2),
                            decoded_rgb_digest(pixels, width=2, height=3))

    def test_balanced_deterministic_family_target_context_round_robin(self):
        contexts, rows = {}, []
        for family, targets, variants, views in [("A", 3, 2, 4), ("B", 1, 4, 6)]:
            for t in range(targets):
                target = f"{family}/p{t}"
                original = target + "/original"
                contexts[original] = context(family, target, original)
                for c in range(variants):
                    key = f"{target}/context{c}"
                    contexts[key] = context(family, target, original, qualified=True)
                    rows += [view(f"{key}/view{i:02}", key, family, target) for i in range(views)]
        result = select(rows, contexts, {"A": "train", "B": "train"})
        selected = result["selected"]
        self.assertEqual([r["source_family"] for r in selected], ["A", "B"] * 24)
        a = [r for r in selected if r["source_family"] == "A"]
        b = [r for r in selected if r["source_family"] == "B"]
        self.assertEqual(Counter(r["source_target"] for r in a[:12]), {"A/p0": 4, "A/p1": 4, "A/p2": 4})
        self.assertEqual(sorted(Counter(r["augmentation_context_id"] for r in b[:12]).values()), [3] * 4)
        for target in ("A/p0", "A/p1", "A/p2"):
            first = [r for r in a if r["source_target"] == target][:2]
            self.assertEqual(len({r["augmentation_context_id"] for r in first}), 2)
        random.Random(37).shuffle(rows)
        self.assertEqual(result, select(rows, contexts, {"B": "train", "A": "train"}))

    def test_exhausted_donor_does_not_block_or_upsample(self):
        contexts = {"root": context(), "b": context("B", "B/p", "b")}
        rows = [view(f"a{i}") for i in range(5)] + [view("b", "b", "B", "B/p")]
        result = select(rows, contexts, {"A": "train", "B": "train"})
        self.assertEqual([r["source_family"] for r in result["selected"]], ["A", "B", "A", "A", "A", "A"])

    def test_auto_and_manual_review_never_grant_release_approval(self):
        a, b, c = view("auto"), view("manual"), view("failed")
        b["annotation_review"]["method"] = "manual"
        c["annotation_review"]["passed"] = False
        result = select([a, b, c])
        self.assertEqual(result["counts"]["train"], 2)
        self.assertEqual(result["excluded"]["failed"], ["annotation_review_required"])
        self.assertFalse(result["training_approved"])
        self.assertTrue(result["release_validation_required"])
        self.assertEqual(result["calibration_status"], "externally_attested_not_verified_here")
        self.assertTrue(all(r["training_approved"] is False for r in result["selected"]))

    def test_generated_source_identity_and_claimed_split_cannot_change(self):
        for changes, reason in [({"source_target": "new-target"}, "source_identity_mismatch"),
                                ({"source_family": "new-family"}, "source_identity_mismatch"),
                                ({"split": "test"}, "frozen_split_mismatch")]:
            with self.subTest(changes=changes):
                result = select([view("x", **changes)])
                self.assertFalse(result["selected"])
                self.assertIn(reason, result["excluded"]["x"])

    def test_bad_provenance_and_renamed_original_rejected(self):
        for changes in [{"source_target": "new"}, {"source_family": "new"},
                        {"source_context_id": "missing"}, {"geometry_evidence_id": ""}, {"qualified_geometry": 1}]:
            variant = context(qualified=True)
            variant.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                select([], {"root": context(), "variant": variant})
        with self.assertRaisesRegex(ValueError, "Multiple original"):
            select([], {"root": context(), "renamed": context(source="renamed")})
        with self.assertRaisesRegex(ValueError, "Original donor missing"):
            select([], splits={"different-family": "train"})

    def test_calibration_gates_require_evidence_and_no_default_thresholds(self):
        for kind in ("geometry", "near_image"):
            for field, value in [("threshold", None), ("threshold", float("nan")), ("threshold", float("inf")),
                                 ("threshold", -1), ("threshold", True), ("validated", False), ("validated", 1),
                                 ("evidence_id", ""), ("metric", "")]:
                gates = policy().calibration
                gates[kind][field] = value
                with self.subTest(kind=kind, field=field, value=value), self.assertRaises(ValueError):
                    select([], declaration=policy(calibration=gates))
        with self.assertRaises(ValueError):
            select([], declaration=policy(calibration={}))

    def test_counts_require_minimum_20000_train_and_explicit_additional_heldout(self):
        for counts in [dict(train=16000, validation=2000, test=2000), dict(train=20000),
                       dict(train=20000, validation=0, test=1), dict(train=True, validation=1, test=1),
                       dict(train=20000, validation=1.5, test=1), dict(train=20000, validation=1, test=0)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                select([], declaration=policy(counts=counts))
        for cap in (0, -1, True, 1.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                select([], declaration=policy(max_views_per_context=cap))

    def test_changed_policy_and_unfinalized_groups_rejected(self):
        groups = finalize_context_groups({"root": context()}, [], policy=policy(), evidence_id="global")
        for changed in (policy(version="toy.v2"), policy(max_views_per_context=24)):
            with self.assertRaisesRegex(ValueError, "policy binding"):
                select_candidates([], {"A": "train"}, groups, policy=changed, near_image_edges=[],
                                  image_inventory_evidence_id="global")
        groups["finalized"] = False
        with self.assertRaisesRegex(ValueError, "Finalized"):
            select_candidates([], {"A": "train"}, groups, policy=policy(), near_image_edges=[],
                              image_inventory_evidence_id="global")

    def test_unknown_edges_contexts_and_duplicate_sample_ids_rejected(self):
        for kwargs in ({"edges": [("root", "missing")]}, {"near": [("a", "missing")]}, {"near": [("a",)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select([view("a")], **kwargs)
        with self.assertRaisesRegex(ValueError, "Unique sample_id"):
            select([view("a"), view("a")])
        with self.assertRaisesRegex(ValueError, "outside global inventory"):
            select([view("a", "new-context")])

    def test_malformed_rgb_and_missing_global_evidence_rejected(self):
        for pixels, width, height in [(b"abc", True, 1), (b"abc", 0, 1), (b"abc", 2, 1), ("abc", 1, 1)]:
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                decoded_rgb_digest(pixels, width=width, height=height)
        with self.assertRaises(ValueError):
            select([view("a", decoded_rgb_sha256="encoded-file-hash")])
        with self.assertRaises(ValueError):
            finalize_context_groups({}, [], policy=policy(), evidence_id="")
        groups = finalize_context_groups({}, [], policy=policy(), evidence_id="global")
        with self.assertRaises(ValueError):
            select_candidates([], {}, groups, policy=policy(), near_image_edges=[], image_inventory_evidence_id="")

    def test_inputs_and_reviews_unchanged_output_detached(self):
        rows, splits, contexts, declaration = [view("a")], {"A": "train"}, {"root": context()}, policy()
        rows[0]["capture"] = {"path": "frozen-capture", "training_approved": False}
        groups = finalize_context_groups(contexts, [], policy=declaration, evidence_id="global")
        before = deepcopy((rows, splits, contexts, declaration, groups))
        result = select_candidates(rows, splits, groups, policy=declaration, near_image_edges=[],
                                   image_inventory_evidence_id="global")
        result["policy"]["calibration"]["geometry"]["evidence_id"] = "output-edit"
        result["selected"][0]["source_target"] = "output-edit"
        self.assertEqual((rows, splits, contexts, declaration, groups), before)
        for field in ("captures_modified", "original_reviews_modified", "frozen_splits_modified"):
            self.assertFalse(result[field])

    def test_20000_train_plus_heldout_is_still_only_candidate_accounting(self):
        contexts, rows = {}, []
        splits = {"A": "train", "V": "validation", "T": "test"}
        for family, n in [("A", 20001), ("V", 3), ("T", 4)]:
            target, original = family + "/p", family + "/original"
            contexts[original] = context(family, target, original)
            for i in range(n):
                key = f"{family}/qualified-{i // 12:05}"
                contexts[key] = context(family, target, original, qualified=True)
                rows.append(view(f"{family}/view-{i:05}", key, family, target))
        result = select(rows, contexts, splits)
        self.assertEqual(result["counts"], dict(train=20000, validation=2, test=3))
        self.assertEqual(len(result["selected"]), 20005)
        self.assertTrue(result["candidate_counts_met"])
        self.assertEqual(result["shortfall"], dict(train=0, validation=0, test=0))
        self.assertEqual(len(result["excluded"]), 3)
        self.assertTrue(all(n <= 12 for n in result["context_counts"].values()))
        self.assertFalse(result["training_approved"])
        self.assertTrue(all(r["training_approved"] is False for r in result["selected"]))
        # 20k TOTAL (19,995 train + 5 heldout) does not meet the train minimum.
        limits = {"A": 19995, "V": 2, "T": 3}
        short_rows = [r for r in rows if int(r["sample_id"].rsplit("-", 1)[1]) < limits[r["source_family"]]]
        self.assertEqual(len(short_rows), 20000)
        short = select(short_rows, contexts, splits)
        self.assertEqual(short["shortfall"], dict(train=5, validation=0, test=0))
        self.assertFalse(short["candidate_counts_met"])
        no_heldout = select([r for r in rows if r["source_family"] == "A"], contexts, splits)
        self.assertEqual(no_heldout["shortfall"], dict(train=0, validation=2, test=3))
        self.assertFalse(no_heldout["candidate_counts_met"])


if __name__ == "__main__":
    unittest.main()
