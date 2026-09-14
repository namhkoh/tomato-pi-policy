"""Generated catalogue contracts tested on new temporary USD fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sim_data.audit import audit_manifest
from sim_data.plant_variant_catalogue import load_for_inspection, assemble_for_inspection
from sim_data.plant_variant_usd import write_json_new, write_variant
from sim_data.plant_variants import file_hash, plan_variant
from sim_data import plant_variant_usd_test as usd_fixture

Usd = usd_fixture.Usd


@unittest.skipIf(Usd is None, "USD Python unavailable")
class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        donor = self.root/"seed_fixture"
        donor.mkdir()
        components = []
        for key, kind, parent, position in (
                ("Stem", "main_stem", None, [0, 0, 0]),
                ("Petiole", "sub_stem", "Stem", [0, 0, 0]),
                ("Leaf", "leaf", "Petiole", [.08, 0, 0])):
            usd_fixture.WriterTests().box(donor/(key+".usda"))
            c = dict(id=key, type=kind, file=key+".usda", parent=parent,
                transform={"translate": position}, attach_point=position, axis=[1, 0, 0])
            if key == "Stem":
                c["capsules"] = [[[0, 0, -.1, .005], [0, 0, .1, .005]]]
            if key == "Petiole":
                c.update(deleafed=False, capsules=[[[0, 0, 0, .003], [.1, 0, 0, .003]]])
            components.append(c)
        raw = dict(generator="fixture", version="1", seed=1, units="meters", up_axis="Z",
                   component_count=3, components=components)
        write_json_new(donor/"manifest.json", raw)
        report = audit_manifest(donor/"manifest.json")
        self.job = dict(plant_family="seed_fixture", split="train", source_manifest_path=str(donor/"manifest.json"),
                        targets=[dict(component_id="Petiole", target_id="seed_fixture/Petiole")])
        self.plan = dict(family_assignments={"seed_fixture": "train"}, jobs=[self.job],
                         source_bindings_sha256={str(p): file_hash(p) for p in donor.iterdir() if p.is_file()})
        self.plan_path = self.root/"source_plan.json"
        write_json_new(self.plan_path, self.plan)
        source = dict(raw=raw, report=report, job=self.job)
        envelope = dict(source_families=["seed_fixture"],
            source_manifest_hashes={"seed_fixture": report["manifest_sha256"]},
            bounds={"length_m": [.01, 1], "radius_m": [.0001, .1], "attachment_angle_deg": [0, 180]})
        self.planned = plan_variant(source, envelope, 1, 1)
        output = self.root/"output"
        result = write_variant(source, self.planned, output)
        self.directory = output/result["variant_id"]
        write_json_new(output/"training_envelope.json", envelope)
        self.frozen_path = output/"frozen_lineage.json"
        write_json_new(self.frozen_path, dict(source_plan_sha256=file_hash(self.plan_path),
                                            family_assignments=self.plan["family_assignments"]))

    def read_receipt(self):
        return json.loads((self.directory/"qualification.json").read_text())

    def change_receipt(self, receipt):
        (self.directory/"qualification.json").write_text(json.dumps(receipt), encoding="utf-8")

    def load(self):
        return load_for_inspection(self.directory, self.plan_path)

    def test_valid_fixture_assembles_catalogue_and_rejects_duplicate_root(self):
        c = self.load()
        self.assertEqual(len(c["rows"]), 1)
        self.assertEqual(c["split_group"], "seed_fixture")
        self.assertFalse(c["training_eligible"])
        stage = Usd.Stage.CreateInMemory()
        result = assemble_for_inspection(stage, "/World/Generated", c)
        self.assertEqual(len(result["record"]["component_paths"]), 3)
        self.assertEqual(result["rows"][0]["conservative_view_cap_group"], "seed_fixture/Petiole")
        with self.assertRaises(ValueError):
            assemble_for_inspection(stage, "/World/Generated", c)
        with self.assertRaises(ValueError):
            assemble_for_inspection(Usd.Stage.CreateNew(str(self.root/"saved.usda")), "/World/Generated", c)

    def test_stale_output_bytes_or_missing_mesh_binding_rejected(self):
        r = self.read_receipt()
        r["output_hashes"].pop("Petiole.usda")
        self.change_receipt(r)
        with self.assertRaisesRegex(ValueError, "Unbound generated mesh"):
            self.load()

    def test_modified_source_bytes_rejected(self):
        source = self.root/"seed_fixture/Leaf.usda"
        source.write_text(source.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Frozen donor"):
            self.load()

    def test_changed_generated_metadata_even_with_updated_hash_rejected(self):
        path = self.directory/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["components"][2]["transform"]["translate"][0] += .01
        path.write_text(json.dumps(manifest))
        r = self.read_receipt()
        r["output_hashes"]["manifest.json"] = file_hash(path)
        self.change_receipt(r)
        with self.assertRaisesRegex(ValueError, "attachment/centerline"):
            self.load()

    def test_cut_proposal_or_membership_tampering_rejected(self):
        original = self.read_receipt()
        for mode in ("label", "member", "approval"):
            r = deepcopy(original)
            if mode == "label":
                r["changes"][0]["cut_region_proposal"]["nominal"]["point_plant_m"][0] += .001
            elif mode == "member":
                r["changes"][0]["members"].append("Stem")
            else:
                r["training_eligible"] = True
            self.change_receipt(r)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.load()

    def test_family_assignment_change_rejected(self):
        self.plan["family_assignments"]["seed_fixture"] = "test"
        self.plan_path.write_text(json.dumps(self.plan))
        # Even updating the local frozen-plan receipt cannot promote a test donor.
        frozen = dict(source_plan_sha256=file_hash(self.plan_path),
                      family_assignments=self.plan["family_assignments"])
        self.frozen_path.write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, "donor/split"):
            self.load()

    def test_failed_surface_probe_stays_out_of_rows_without_approval(self):
        r = self.read_receipt()
        r["local_surface_diagnostics"][0]["surface_probe"]["passed"] = False
        self.change_receipt(r)
        c = self.load()
        self.assertEqual(c["rows"], [])
        self.assertIn("sparse_mesh_surface_probe_failed", c["rejected"][0]["reasons"])
        self.assertFalse(c["training_eligible"])

    def test_changed_catalogue_frame_rejected_at_composition(self):
        c = self.load()
        c["report"]["components"]["Leaf"]["translation_plant_m"][0] += .01
        with self.assertRaisesRegex(ValueError, "catalogue geometry"):
            assemble_for_inspection(Usd.Stage.CreateInMemory(), "/Plant", c)

    def test_inherited_attachment_warning_is_withheld_not_silently_accepted(self):
        with patch("sim_data.geometry.audit_geometry", return_value=dict(maximum_translation_error_m=0,
                warnings=[dict(component_id="Petiole", code="attachment_to_parent_bounds_m_over_2mm")])):
            c = self.load()
        self.assertEqual(c["rows"], [])
        self.assertIn("attachment_to_parent_bounds_m_over_2mm", c["rejected"][0]["reasons"])

    def test_reject_failed_partial_output_and_stale_code_receipt(self):
        r = self.read_receipt()
        r["implementation_hashes"]["plant_variants.py"] = "0"*64
        self.change_receipt(r)
        with self.assertRaisesRegex(ValueError, "qualification code"):
            self.load()
        write_json_new(self.directory/"FAILED.json", {"state": "failed"})
        with self.assertRaisesRegex(ValueError, "Failed generator"):
            self.load()


if __name__ == "__main__":
    unittest.main()
