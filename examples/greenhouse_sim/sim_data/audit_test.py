"""Standard-library tests; USD checks additionally run under Isaac's python.bat."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim_data.audit import audit_manifest, main, safe_asset
from sim_data.review import record_review

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Usd = None


def component(key, kind, parent, position, **extra):
    return {"id": key, "type": kind, "parent": parent, "file": key + ".usda",
            "transform": {"translate": position}, "attach_point": position,
            "axis": [1.0, 0.0, 0.0], **extra}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plant = self.root / "package/plants/components/test_plant"
        self.plant.mkdir(parents=True)
        self.path = self.plant / "manifest.json"
        self.manifest = {"units": "meters", "up_axis": "Z", "component_count": 3, "components": [
            component("Main", "main_stem", None, [10, 1, 0]),
            component("Petiole", "sub_stem", "Main", [11, 1, 0], deleafed=False, radius=0.003,
                      length=0.3, capsules=[[[0, 0, 0, 0.003], [0.3, 0, 0, 0.003]]]),
            component("Leaf", "leaf", "Petiole", [11.2, 1, 0.1]),
        ]}

    def write(self):
        self.manifest["component_count"] = len(self.manifest["components"])
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")
        for item in self.manifest["components"]:
            # Test assets are tiny; actual geometry is authored only in USD tests.
            if isinstance(item.get("file"), str) and ":" not in item["file"] and ".." not in item["file"]:
                (self.plant / item["file"]).write_text("#usda 1.0\n", encoding="utf-8")
        return audit_manifest(self.path)

    def codes(self, report):
        return {i["code"] for i in report["issues"]}

    def test_intact_leaf_requires_review_and_has_no_cut_label(self):
        report = self.write()
        target = report["targets"][0]
        self.assertEqual(target["status"], "needs_review")
        self.assertEqual(target["expected_detached_component_ids"], ["Leaf", "Petiole"])
        self.assertIsNone(target["canonical_cut_point_m"])
        self.assertIsNone(target["grasp_region"])
        self.assertFalse(report["geometry_verified"])

    def test_deleafed_stub_is_excluded_even_if_leaf_metadata_remains(self):
        self.manifest["components"][1]["deleafed"] = True
        target = self.write()["targets"][0]
        self.assertEqual(target["status"], "excluded")
        self.assertIn("already_deleafed", target["reason_codes"])

    def test_fruit_in_detached_subtree_excludes_target(self):
        self.manifest["components"].append(component("Fruit", "fruit", "Petiole", [11.3, 1, 0]))
        target = self.write()["targets"][0]
        self.assertEqual(target["status"], "excluded")
        self.assertEqual(target["protected_descendant_ids"], ["Fruit"])

    def test_unknown_deleafed_state_blocks_target(self):
        del self.manifest["components"][1]["deleafed"]
        self.assertEqual(self.write()["targets"][0]["status"], "blocked")

    def test_missing_parent_blocks_plant(self):
        self.manifest["components"][1]["parent"] = "Missing"
        report = self.write()
        self.assertIn("missing_parent", self.codes(report))
        self.assertEqual(report["status"], "blocked")

    def test_cycles_do_not_recurse_forever(self):
        self.manifest["components"][0]["parent"] = "Petiole"
        report = self.write()
        self.assertIn("parent_cycle", self.codes(report))
        self.assertEqual(report["targets"][0]["status"], "blocked")

    def test_duplicate_ids_block(self):
        self.manifest["components"].append(copy.deepcopy(self.manifest["components"][1]))
        self.assertIn("duplicate_component_id", self.codes(self.write()))

    def test_nonfinite_coordinates_block_and_report_is_json_serializable(self):
        self.manifest["components"][1]["transform"]["translate"] = [float("nan"), 0, 0]
        report = self.write()
        self.assertIn("invalid_translation", self.codes(report))
        json.dumps(report, allow_nan=False)

    def test_wrong_units_and_rotation_are_not_silently_ignored(self):
        self.manifest["units"] = "millimeters"
        self.manifest["components"][0]["transform"]["rotate"] = [0, 90, 0]
        codes = self.codes(self.write())
        self.assertIn("unsupported_coordinate_convention", codes)
        self.assertIn("unsupported_transform_ops", codes)

    def test_path_traversal_and_missing_assets(self):
        for name in ("../escape.usd", "C:/escape.usd", "/escape.usd", "foo:stream", "..\\escape.usd"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_asset(self.plant, name)
        self.write()
        (self.plant / "Petiole.usda").unlink()
        self.assertIn("missing_component_file", self.codes(audit_manifest(self.path)))

    def test_no_leaf_descendants_is_excluded(self):
        self.manifest["components"].pop()
        self.assertIn("no_leaf_descendants", self.write()["targets"][0]["reason_codes"])

    def test_manifest_order_does_not_change_targets(self):
        first = self.write()
        self.manifest["components"].reverse()
        second = self.write()
        self.assertEqual(first["targets"], second["targets"])

    def test_audit_preserves_source_bytes(self):
        self.write()
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.plant.iterdir()}
        audit_manifest(self.path)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.plant.iterdir()})

    def test_review_requires_human_notes_and_never_approves_cutting(self):
        report = self.write()
        target_id = report["targets"][0]["target_id"]
        with self.assertRaises(ValueError):
            record_review(report, target_id, "anatomy_confirmed", "", "", self.root / "reviews")
        result = record_review(report, target_id, "anatomy_confirmed", "test reviewer", "fixture only", self.root / "reviews")
        row = json.loads(result.read_text())
        self.assertFalse(row["cut_approval"])
        self.assertEqual(report["targets"][0]["anatomy_review"], "pending")

    def test_cannot_confirm_excluded_or_changed_manifest(self):
        self.manifest["components"][1]["deleafed"] = True
        report = self.write()
        target_id = report["targets"][0]["target_id"]
        with self.assertRaises(ValueError):
            record_review(report, target_id, "anatomy_confirmed", "test", "test", self.root / "reviews")
        self.path.write_text("{}")
        with self.assertRaises(ValueError):
            record_review(report, target_id, "excluded", "test", "test", self.root / "reviews")

    def test_cli_refuses_source_package_output_and_existing_runs(self):
        self.write()
        package = self.root / "package"
        for output in (package / "output", self.root):
            with self.subTest(output=output), self.assertRaises(SystemExit):
                main(["--package", str(package), "--output", str(output)])

    def test_changed_component_invalidates_review(self):
        report = self.write()
        (self.plant / "Leaf.usda").write_text("changed asset")
        with self.assertRaisesRegex(ValueError, "Component asset changed"):
            record_review(report, report["targets"][0]["target_id"], "anatomy_confirmed",
                          "test", "test", self.root / "reviews")

    def test_review_cannot_write_into_source_package(self):
        report = self.write()
        with self.assertRaisesRegex(ValueError, "source package"):
            record_review(report, report["targets"][0]["target_id"], "unresolved",
                          "test", "test", self.root / "package/reviews")


@unittest.skipIf(Usd is None, "pxr is available under Isaac python.bat")
class GeometryTests(unittest.TestCase):
    setUp = AuditTests.setUp
    write = AuditTests.write

    def usd_report(self):
        self.write()
        for item in self.manifest["components"]:
            stage = Usd.Stage.CreateNew(str(self.plant / (item["id"] + "_geometry.usda")))
            root = UsdGeom.Xform.Define(stage, "/Part")
            stage.SetDefaultPrim(root.GetPrim())
            mesh = UsdGeom.Mesh.Define(stage, "/Part/Mesh")
            mesh.CreatePointsAttr([(x, y, z) for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2])
            stage.GetRootLayer().Save()
            item["file"] = item["id"] + "_geometry.usda"
        self.path.write_text(json.dumps(self.manifest))
        return audit_manifest(self.path)

    def test_parent_global_positions_are_not_accumulated(self):
        from sim_data.geometry import assemble_plant, audit_geometry

        report = self.usd_report()
        stage = Usd.Stage.CreateInMemory()
        before = stage.GetRootLayer().ExportToString()
        paths = assemble_plant(stage, "/World/Plant", report)
        position = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(paths["Petiole"])).ExtractTranslation()
        self.assertLess((position - Gf.Vec3d(11, 1, 0)).GetLength(), 1e-9)
        self.assertEqual(before, stage.GetRootLayer().ExportToString())
        check = audit_geometry(report)
        self.assertLess(check["maximum_translation_error_m"], 1e-9)
        # Leaf bounds must not expand its parent's own component bounds.
        self.assertLess(check["components"]["Petiole"]["own_bounds_plant_m"][1][0], 11.1)

    def test_review_isolation_and_close_restore_visibility(self):
        from sim_data.geometry import assemble_plant
        from sim_data.review_scene import ReviewScene, OVERLAY

        report = self.usd_report()
        stage = Usd.Stage.CreateInMemory()
        paths = assemble_plant(stage, "/World/Plant", report)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            other = UsdGeom.Mesh.Define(stage, "/World/Other")
            other.CreateVisibilityAttr("inherited")
            unknown = UsdGeom.Mesh.Define(stage, "/World/Unauthored")
        before = stage.GetRootLayer().ExportToString()
        scene = ReviewScene(stage, "/World/Plant", paths, report)
        scene.select(report["targets"][0])
        scene.isolate()
        self.assertEqual(other.GetVisibilityAttr().Get(), "invisible")
        scene.close()
        self.assertEqual(other.GetVisibilityAttr().Get(), "inherited")
        self.assertFalse(stage.GetSessionLayer().GetAttributeAtPath(unknown.GetVisibilityAttr().GetPath()))
        self.assertFalse(stage.GetPrimAtPath(OVERLAY))
        self.assertEqual(before, stage.GetRootLayer().ExportToString())

    def test_occupied_overlay_path_is_preserved(self):
        from sim_data.geometry import assemble_plant
        from sim_data.review_scene import ReviewScene, OVERLAY

        report = self.usd_report()
        stage = Usd.Stage.CreateInMemory()
        paths = assemble_plant(stage, "/World/Plant", report)
        UsdGeom.Xform.Define(stage, OVERLAY)
        before = stage.GetRootLayer().ExportToString()
        with self.assertRaisesRegex(ValueError, "occupied"):
            ReviewScene(stage, "/World/Plant", paths, report)
        self.assertEqual(before, stage.GetRootLayer().ExportToString())


if __name__ == "__main__":
    unittest.main()
