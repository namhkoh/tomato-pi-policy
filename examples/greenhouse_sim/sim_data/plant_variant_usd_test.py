"""CPU USD serialization/surface diagnostics. No renderer or physics launch."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sim_data.audit import DEFAULT_PACK, audit_manifest
from sim_data.plant_variant_usd import (copy_component, cut_surface_probe, first_ray_hit,
                                      mesh_arrays, write_json_new, write_variant)
from sim_data.plant_variants import plan_variant, rotation

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
except ImportError:
    Usd = None


class RayTests(unittest.TestCase):
    def test_two_sided_hit_and_miss(self):
        triangle = np.array([[[1, -1, -1], [1, 1, -1], [1, 0, 1]]], dtype=float)
        self.assertAlmostEqual(first_ray_hit(triangle, [0, 0, 0], np.array([1, 0, 0])), 1)
        self.assertAlmostEqual(first_ray_hit(triangle[:, ::-1], [0, 0, 0], np.array([1, 0, 0])), 1)
        self.assertIsNone(first_ray_hit(triangle, [0, 3, 0], np.array([1, 0, 0])))
        self.assertIsNone(first_ray_hit(triangle, [0, 0, 0], np.array([-1, 0, 0])))
        self.assertIsNone(first_ray_hit(triangle, [0, 0, 0], np.array([0, 1, 0])))


@unittest.skipIf(Usd is None, "USD Python unavailable")
class WriterTests(unittest.TestCase):
    def box(self, path, low=(0., -.003, -.003), high=(.1, .003, .003)):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, "Z")
        root = UsdGeom.Xform.Define(stage, "/Component")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/Component/Mesh")
        mesh.CreatePointsAttr([(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1])
                               for z in (low[2], high[2])])
        mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1,
                                          2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
        mesh.CreateNormalsAttr([(0., 0., 2.)]*8)
        mesh.SetNormalsInterpolation("vertex")
        mesh.CreateExtentAttr([low, high])
        mesh.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["PhysicsCollisionAPI"]))
        mesh.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        stage.GetRootLayer().Save()
        return stage

    def test_serialized_mesh_normals_extents_topology_and_source_unchanged(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            source = self.box(d/"source.usda")
            original = (d/"source.usda").read_bytes()
            r = rotation([1, 1, 1], 12)
            receipt = copy_component(d/"source.usda", d/"output.usdc", r, .9, d, {})
            self.assertEqual(receipt["removed_physics_apis"], 1)
            target = Usd.Stage.Open(str(d/"output.usdc"))
            old, _ = mesh_arrays(source)
            new, _ = mesh_arrays(target)
            np.testing.assert_allclose(new, .9*(old @ r.T), atol=1e-8)
            mesh = UsdGeom.Mesh.Get(target, "/Component/Mesh")
            normals = np.asarray(mesh.GetNormalsAttr().Get())
            np.testing.assert_allclose(normals, np.tile(r @ [0, 0, 1], (8, 1)), atol=1e-7)
            np.testing.assert_allclose(np.asarray(mesh.GetExtentAttr().Get()), [new.min(0), new.max(0)], atol=1e-8)
            self.assertEqual(list(mesh.GetFaceVertexIndicesAttr().Get()),
                             list(UsdGeom.Mesh.Get(source, "/Component/Mesh").GetFaceVertexIndicesAttr().Get()))
            self.assertEqual((d/"source.usda").read_bytes(), original)
            self.assertFalse(any("physics" in s.lower() for s in mesh.GetPrim().GetAppliedSchemas()))
            self.assertFalse(mesh.GetPrim().GetAttribute("physics:collisionEnabled").HasAuthoredValueOpinion())

    def test_refuse_overwrite_nonidentity_animation_and_composed_asset(self):
        for mode in ("overwrite", "transform", "animated", "reference"):
            with self.subTest(mode=mode), TemporaryDirectory() as temp:
                d = Path(temp)
                stage = self.box(d/"source.usda")
                if mode == "overwrite":
                    self.box(d/"output.usda")
                elif mode == "transform":
                    UsdGeom.Xformable(stage.GetDefaultPrim()).AddTranslateOp().Set((1, 0, 0))
                elif mode == "animated":
                    UsdGeom.Mesh.Get(stage, "/Component/Mesh").GetPointsAttr().Set(
                        [(0, 0, 0)]*8, Usd.TimeCode(1))
                else:
                    self.box(d/"ref.usda")
                    stage.GetDefaultPrim().GetReferences().AddReference("ref.usda")
                stage.GetRootLayer().Save()
                with self.assertRaises(ValueError):
                    copy_component(d/"source.usda", d/"output.usda", np.eye(3), 1., d, {})

    def test_source_zero_normals_preserved_and_reported_not_invented(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            stage = self.box(d/"source.usda")
            UsdGeom.Mesh.Get(stage, "/Component/Mesh").GetNormalsAttr().Set([(0., 0., 0.)]*8)
            stage.GetRootLayer().Save()
            receipt = copy_component(d/"source.usda", d/"output.usda", rotation([0, 0, 1], 15), .9, d, {})
            self.assertEqual(receipt["preserved_source_zero_normals"], 8)
            copied = Usd.Stage.Open(str(d/"output.usda"))
            np.testing.assert_array_equal(UsdGeom.Mesh.Get(copied, "/Component/Mesh").GetNormalsAttr().Get(),
                                          np.zeros((8, 3)))

    def test_faceless_source_stub_preserved_but_cannot_supply_surface_hits(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            stage = self.box(d/"source.usda")
            mesh = UsdGeom.Mesh.Get(stage, "/Component/Mesh")
            mesh.GetFaceVertexCountsAttr().Set([])
            mesh.GetFaceVertexIndicesAttr().Set([])
            stage.GetRootLayer().Save()
            receipt = copy_component(d/"source.usda", d/"output.usda", np.eye(3), 1., d, {})
            self.assertEqual(receipt["preserved_source_faceless_meshes"], 1)
            copied = Usd.Stage.Open(str(d/"output.usda"))
            points, triangles = mesh_arrays(copied)
            self.assertEqual(len(points), 8)
            self.assertEqual(triangles.shape, (0, 3, 3))
            self.assertIsNone(first_ray_hit(triangles, [0, 0, 0], np.array([1, 0, 0])))

    def test_texture_path_escape_and_missing_rejected(self):
        for asset in ("../escape.png", "C:/outside.png", "missing.png"):
            with self.subTest(asset=asset), TemporaryDirectory() as temp:
                d = Path(temp)
                stage = self.box(d/"source.usda")
                shader = UsdShade.Shader.Define(stage, "/Component/Texture")
                shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset))
                stage.GetRootLayer().Save()
                with self.assertRaises(ValueError):
                    copy_component(d/"source.usda", d/"output.usda", np.eye(3), 1., d, {})

    def test_surface_probe_requires_actual_surface_near_all_three_cut_samples(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            component = dict(id="Petiole", parent="Stem", type="sub_stem", deleafed=False,
                translation_plant_m=[0, 0, 0], attachment_plant_m=[0, 0, 0],
                axis_plant=[1, 0, 0], capsules_local_m=[[[0, 0, 0, .003], [.1, 0, 0, .003]]])
            parent = dict(id="Stem", type="main_stem", translation_plant_m=[0, 0, 0],
                          capsules_local_m=[[[0, 0, -.1, .005], [0, 0, .1, .005]]])
            good = cut_surface_probe(self.box(d/"good.usda"), component, parent)
            bad = cut_surface_probe(self.box(d/"bad.usda", (.05, -.003, -.003)), component, parent)
            self.assertTrue(good["passed"])
            self.assertFalse(bad["passed"])
            self.assertFalse(good["protected_organ_clearance_validated"])
            self.assertEqual([r["arc_m"] for r in good["samples"]], [.01, .015, .02])

    @unittest.skipUnless(DEFAULT_PACK.is_dir(), "Supplied package unavailable")
    def test_real_leaf_texture_and_two_part_petiole_are_preserved(self):
        root = DEFAULT_PACK/"plants/components/seed101_full"
        with TemporaryDirectory() as temp:
            d = Path(temp)
            copies = {}
            for file in ("Leaf_000.usdc", "SubStem_56.usdc"):
                receipt = copy_component(root/file, d/file, rotation([0, 0, 1], 15), .9, root, copies)
                self.assertEqual(receipt["meshes"], 1 if file.startswith("Leaf") else 2)
                if file.startswith("Leaf"):
                    self.assertTrue(receipt["texture_hashes"])
                    for relative in receipt["texture_hashes"]:
                        self.assertEqual((root/relative).read_bytes(), (d/relative).read_bytes())

    def test_complete_fixture_write_labels_lineage_create_only(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            donor = d/"seed_fixture"
            donor.mkdir()
            for file in ("Stem.usda", "Petiole.usda", "Leaf.usda"):
                self.box(donor/file)
            components = []
            for key, kind, parent, origin in (
                    ("Stem", "main_stem", None, [0, 0, 0]),
                    ("Petiole", "sub_stem", "Stem", [0, 0, 0]),
                    ("Leaf", "leaf", "Petiole", [.08, 0, 0])):
                c = dict(id=key, type=kind, parent=parent, file=key+".usda",
                         transform={"translate": origin}, attach_point=origin, axis=[1, 0, 0])
                if key == "Petiole":
                    c.update(deleafed=False, capsules=[[[0, 0, 0, .003], [.1, 0, 0, .003]]])
                if key == "Stem":
                    c.update(capsules=[[[0, 0, -.1, .005], [0, 0, .1, .005]]])
                components.append(c)
            raw = dict(generator="fixture", version="1", seed=1, units="meters", up_axis="Z",
                       component_count=3, components=components)
            write_json_new(donor/"manifest.json", raw)
            report = audit_manifest(donor/"manifest.json")
            source = dict(raw=raw, report=report, job=dict(plant_family="seed_fixture", split="train",
                targets=[dict(component_id="Petiole", target_id="seed_fixture/Petiole")]))
            envelope = dict(source_families=["seed_fixture"],
                source_manifest_hashes={"seed_fixture": report["manifest_sha256"]},
                bounds={"length_m": [.01, 1], "radius_m": [.0001, .1], "attachment_angle_deg": [0, 180]})
            planned = plan_variant(source, envelope, 1, 1)
            original = deepcopy(planned)
            result = write_variant(source, planned, d/"new")
            self.assertEqual(planned, original)
            self.assertEqual(result["surface_probe_pass_count"], 1)
            self.assertFalse(result["collection_approved"])
            self.assertFalse(result["training_eligible"])
            self.assertTrue(result["source_assets_unchanged"])
            self.assertEqual(result["split_group"], "seed_fixture")
            with self.assertRaises(ValueError):
                write_variant(source, planned, d/"new")
            with self.assertRaises(ValueError):
                write_variant(source, planned, donor/"forbidden")


if __name__ == "__main__":
    unittest.main()
