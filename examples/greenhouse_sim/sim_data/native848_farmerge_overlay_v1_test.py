from copy import deepcopy
import unittest
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
from . import native848_farmerge_overlay_v1 as a


def triangle(stage,path):
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    mesh.CreateNormalsAttr([(0,0,1)]*3);mesh.SetNormalsInterpolation('vertex')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(0,1)])
    return mesh


class Guards(unittest.TestCase):
    def identity(self,index=3):
        return dict(component_index=index,prim_path='/World/PackPlants/P/Stem/Petiole',component_id='Petiole',
                    organ_type='sub_stem',source_plant_id='donor',split_group='donor')

    def test_indices_are_explicitly_remapped_after_foreground_count_changes(self):
        base=self.identity();current=dict(base,component_index=10,geometry_source_id='donor',source_geometry_modified=False)
        self.assertEqual(a.remap_members([current],[base],[base],'/World/PackPlants/P'),{'3':10})

    def test_background_cannot_silently_become_generated_or_another_donor(self):
        base=self.identity()
        for change in (dict(source_geometry_modified=True),dict(geometry_source_id='derived'),dict(source_plant_id='other'),dict(split_group='other')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                a.remap_members([dict(base,**change)],[base],[base],'/World/PackPlants/P')

    def test_selection_index_path_disagreement_rejected(self):
        base=self.identity();selected=dict(base,prim_path='/World/PackPlants/P/Other')
        with self.assertRaises(ValueError):a.remap_members([base],[base],[selected],'/World/PackPlants/P')

    def test_missing_and_duplicate_logical_paths_rejected(self):
        base=self.identity()
        for rows in ([],[base,dict(base,component_index=4)]):
            with self.assertRaises(ValueError):a.remap_members(rows,[base],[base],'/World/PackPlants/P')

    def test_collision_snapshot_survives_render_deletion_and_pose_is_independent(self):
        stage=Usd.Stage.CreateInMemory();stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.Xform.Define(stage,'/World/PackPlants/P')
        mesh=triangle(stage,'/World/PackPlants/P/Original')
        robot=UsdGeom.Xform.Define(stage,'/Robot');robot.AddTranslateOp().Set((1,2,3))
        original=a.snapshot_mesh(mesh.GetPrim(),UsdGeom.XformCache())
        collision,keep,before,root=a.clone_collision_stage(stage)
        stage.RemovePrim('/World/PackPlants/P');triangle(stage,'/World/PackPlants/P/Packed')
        self.assertTrue(collision.GetPrimAtPath('/World/PackPlants/P/Original'))
        self.assertFalse(collision.GetPrimAtPath('/World/PackPlants/P/Packed'))
        self.assertEqual(original,a.snapshot_mesh(collision.GetPrimAtPath(original['path']),UsdGeom.XformCache()))
        self.assertEqual(keep.ExportToString(),before)
        collision.GetPrimAtPath('/Robot').GetAttribute('xformOp:translate').Set((4,5,6))
        self.assertEqual(tuple(stage.GetPrimAtPath('/Robot').GetAttribute('xformOp:translate').Get()),(1,2,3))
        self.assertEqual(stage.GetRootLayer().ExportToString(),root)

    def overlay_fixture(self):
        stage=Usd.Stage.CreateInMemory();stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.Xform.Define(stage,'/World/PackPlants')
        for i in range(144):
            root=UsdGeom.Xform.Define(stage,'/World/PackPlants/P'+str(i))
            root.AddTranslateOp().Set((i,0,0));triangle(stage,str(root.GetPath())+'/Original')
        retained=[a.snapshot_mesh(p,UsdGeom.XformCache()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        return dict(stage=stage,retained_meshes=retained,render_groups=[],logical_census={'generated':'distinct'},logical_catalogue=[],
            recipe_pin={},selection_pin={},primary_root='/World/PackPlants/P0',whole_roots=[],collision_snapshot={},native_texture_bindings={})

    def test_source_mesh_with_no_uv_or_normals_preserves_absence(self):
        stage=Usd.Stage.CreateInMemory();mesh=UsdGeom.Mesh.Define(stage,'/Mesh')
        mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
        sig=a.retained_signature(mesh)
        self.assertIsNone(sig['normals']);self.assertNotIn('st',sig['primvars'])

    def test_retained_generated_shape_change_rejected(self):
        overlay=self.overlay_fixture();a.verify_overlay(overlay)
        mesh=UsdGeom.Mesh(overlay['stage'].GetPrimAtPath('/World/PackPlants/P0/Original'))
        mesh.GetPointsAttr().Set([(0,0,0),(1.1,0,0),(0,1,0)])
        with self.assertRaises(ValueError):a.verify_overlay(overlay)

    def test_retained_material_and_slot_transform_changes_rejected(self):
        for kind in ('material','transform'):
            with self.subTest(kind=kind):
                overlay=self.overlay_fixture();stage=overlay['stage']
                if kind=='material':stage.GetPrimAtPath('/World/PackPlants/P1/Original').CreateRelationship('material:binding').SetTargets(['/ChangedMaterial'])
                else:stage.GetPrimAtPath('/World/PackPlants/P1').GetAttribute('xformOp:translate').Set((999,0,0))
                with self.assertRaises(ValueError):a.verify_overlay(overlay)

    def test_missing_plant_or_extra_unaccounted_geometry_rejected(self):
        for kind in ('missing','extra'):
            with self.subTest(kind=kind):
                overlay=self.overlay_fixture();stage=overlay['stage']
                if kind=='missing':stage.RemovePrim('/World/PackPlants/P3')
                else:triangle(stage,'/World/PackPlants/P3/Extra')
                with self.assertRaises(ValueError):a.verify_overlay(overlay)


if __name__=='__main__':unittest.main()
