"""Draft assistance never impersonates human review or invents cut labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim_data.audit import DEFAULT_PACK, audit_manifest
from sim_data.candidate_branches import plan_branches
from sim_data.label_drafts import label_color_counts, main, markdown_packet, prepare_drafts


@unittest.skipUnless(DEFAULT_PACK.is_dir(), "Extracted source package required")
class DraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = [audit_manifest(DEFAULT_PACK / "plants/components" / name / "manifest.json")
                       for name in ("seed101_full", "seed103_full")]
        cls.variants = [plan_branches(r) for r in cls.sources]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "candidate_branches.json"
        self.write(self.variants)

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_six_drafts_have_correct_topology_without_approval_or_cut_coordinates(self):
        packet, reports = prepare_drafts(self.path)
        self.assertEqual([r["draft_id"] for r in packet["labels"]], [f"B{i:02d}" for i in range(1, 7)])
        self.assertFalse(packet["human_review_performed"])
        for row in packet["labels"]:
            self.assertFalse(row["training_label_approved"])
            self.assertFalse(row["cut_approval"])
            self.assertFalse(row["training_input_allowed"])
            self.assertEqual(row["human_review_status"], "pending")
            self.assertEqual(row["greenhouse_occlusion"], "not_measured")
            for field in ("canonical_cut_point_m", "admissible_cut_region", "grasp_region"):
                self.assertIsNone(row[field])
            report = next(r for r in reports if r["plant_id"] == row["variant_id"])
            self.assertEqual(report["components"][row["component_id"]]["parent"], row["main_stem_parent_id"])
            self.assertGreater(len(row["leaf_component_ids"]), 0)
            self.assertEqual(row["protected_descendant_ids"], [])
            self.assertTrue(all(report["components"][k]["type"] == "leaf" for k in row["leaf_component_ids"]))
        for report, variant in zip(reports, self.variants):
            self.assertFalse(set(variant["replaced_stub_ids"]) & set(report["components"]))
            self.assertTrue(all(c["parent"] is None or c["parent"] in report["components"]
                                for c in report["components"].values()))

    def test_modified_candidates_source_fingerprints_and_duplicate_variants_rejected(self):
        for field in ("added_components", "source_manifest_sha256", "variant_id", "candidates"):
            values = deepcopy(self.variants)
            if field == "added_components":
                next(iter(values[0][field].values()))["translation_plant_m"][0] += .1
            elif field == "candidates":
                values[0][field][0]["canonical_cut_point_m"] = [1, 2, 3]
            else:
                values[0][field] = "changed"
            self.write(values)
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_drafts(self.path)
        self.write([self.variants[0], self.variants[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_drafts(self.path)

    def test_launch_world_coordinates_are_not_misrepresented_as_live_state(self):
        values = deepcopy(self.variants)
        for value in values:
            for t in value["candidates"]:
                t["attachment_world_m"] = [100, 200, 300]
        self.write(values)
        packet, _ = prepare_drafts(self.path)
        self.assertIn("not_live", packet["render_scope"])
        self.assertTrue(all("attachment_world_m" not in row for row in packet["labels"]))

    def test_metadata_cli_does_not_write_human_review_or_change_sources(self):
        output = self.directory / "drafts"
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(main(["--variants", str(self.path), "--output", str(output), "--no-render"]), 0)
        self.assertEqual({p.name for p in output.iterdir()}, {"draft_labels.json", "review.md"})
        packet = json.loads((output / "draft_labels.json").read_text())
        self.assertEqual(packet["state"], "metadata_only_draft")
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        text = markdown_packet(packet)
        self.assertIn("NOT the cutting point", text)
        self.assertIn("pending", text)
        self.assertNotIn("_labelled.png", text)
        with self.assertRaises(SystemExit):
            main(["--variants", str(self.path), "--output", str(output), "--no-render"])

    def test_cli_refuses_source_package_output(self):
        with self.assertRaises(SystemExit):
            main(["--variants", str(self.path), "--output", str(DEFAULT_PACK / "draft_labels"), "--no-render"])

    def test_blank_render_is_not_accepted_as_label_evidence(self):
        from PIL import Image, ImageDraw
        path = self.directory / "qa_fixture.png"
        image = Image.new("RGB", (100, 100), "white")
        image.save(path)
        with self.assertRaisesRegex(ValueError, "blank target"):
            label_color_counts(path)
        draw = ImageDraw.Draw(image)
        for i, colour in enumerate(("red", "green", "blue")):
            draw.rectangle((i * 30, 10, i * 30 + 10, 30), fill=colour)
        image.save(path)
        self.assertTrue(all(n >= 25 for n in label_color_counts(path).values()))


if __name__ == "__main__":
    unittest.main()
