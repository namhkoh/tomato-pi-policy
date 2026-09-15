"""CPU tests for actual curved meshes, source preservation and attachment placement."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from sim_data.audit import DEFAULT_PACK, audit_manifest
from sim_data.plant_variant_usd import copy_component
from sim_data.procedural_petiole_geometry import CurveSpec, curved_centerline
from sim_data.procedural_petiole_warp import CurveWarp
from sim_data.procedural_petiole_usd import choose_attachment, deform_new_copy, donor_curve
from sim_data.procedural_petiole_catalogue import check_component

try:
    from pxr import Usd, UsdGeom, Sdf
except ImportError:
    Usd = None


class AttachmentTests(unittest.TestCase):
    def fixture(self):
        parent = dict(translation_plant_m=[0,0,0],capsules_local_m=[[[0,0,0,.005],[0,0,.12,.005]]])
        triangles = np.array([[[.005,-.1,0],[.005,.1,0],[.005,.1,.12]],
                              [[.005,-.1,0],[.005,.1,.12],[.005,-.1,.12]]])
        return parent, triangles

    def test_actual_surface_not_centerline(self):
        parent, tri = self.fixture()
        anchor, receipt = choose_attachment(parent,[.005,0,.05],np.array([1,0,0]),.02,tri)
        np.testing.assert_allclose(anchor,[.005,0,.07])
        self.assertAlmostEqual(receipt['parent_surface_ray_hit_m'],.005)
        self.assertAlmostEqual(receipt['anchor_displacement_m'],.02)
        self.assertFalse(receipt['protected_organ_clearance_verified'])

    def test_missing_surface_and_near_endpoint_rejected(self):
        parent, tri = self.fixture()
        for surface, offset in ((np.empty((0,3,3)),.02),(tri,.068),(tri,.001)):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                choose_attachment(parent,[.005,0,.05],np.array([1,0,0]),offset,surface)


@unittest.skipIf(Usd is None,'USD unavailable')
class SerializedTests(unittest.TestCase):
    def triangle(self,path,mode='faceVarying'):
        stage=Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
        root=UsdGeom.Xform.Define(stage,'/Test');stage.SetDefaultPrim(root.GetPrim())
        mesh=UsdGeom.Mesh.Define(stage,'/Test/Mesh')
        mesh.CreatePointsAttr([(0,0,0),(.08,0,0),(.04,.02,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
        mesh.CreateNormalsAttr([(0,0,1)]*3);mesh.SetNormalsInterpolation(mode)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set([(0,0),(1,0),(.5,1)])
        stage.GetRootLayer().Save()
        return stage

    def warp(self):
        new=curved_centerline(CurveSpec(.11,.002,bend_normal_rad=.6),[1,0,0])
        return CurveWarp([[0,0,0],[.1,0,0]],new,[.005,0,.02],1.1)

    def test_normals_positions_uv_topology_readback_and_source_unchanged(self):
        for mode in ('faceVarying','vertex','varying'):
            with self.subTest(mode=mode),TemporaryDirectory() as temp:
                root=Path(temp);self.triangle(root/'source.usda',mode)
                before=(root/'source.usda').read_bytes()
                copy_component(root/'source.usda',root/'copy.usda',np.eye(3),1,root,{})
                warp=self.warp();new_origin=warp.map(np.zeros((1,3)))[0]
                deform_new_copy(root/'copy.usda',np.zeros(3),new_origin,warp)
                qa=check_component(root/'source.usda',root/'copy.usda',np.zeros(3),new_origin,warp)
                self.assertEqual((root/'source.usda').read_bytes(),before)
                self.assertTrue(qa[0]['serialized_geometry_checked'])
                self.assertTrue(qa[0]['topology_uv_materials_preserved'])
                self.assertLess(qa[0]['maximum_point_error_m'],1e-7)

    def test_detect_stale_position_normal_uv_and_topology(self):
        for damage in ('position','normal','uv','topology'):
            with self.subTest(damage=damage),TemporaryDirectory() as temp:
                root=Path(temp);self.triangle(root/'source.usda')
                copy_component(root/'source.usda',root/'copy.usda',np.eye(3),1,root,{})
                stage=Usd.Stage.Open(str(root/'copy.usda'));mesh=UsdGeom.Mesh.Get(stage,'/Test/Mesh')
                if damage=='position':mesh.GetPointsAttr().Set([(0,0,0),(.1,0,0),(.04,.02,0)])
                elif damage=='normal':mesh.GetNormalsAttr().Set([(0,1,0)]*3)
                elif damage=='uv':UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Set([(0,0)]*3)
                else:mesh.GetFaceVertexIndicesAttr().Set([2,1,0])
                stage.GetRootLayer().Save()
                with self.assertRaises(ValueError):check_component(root/'source.usda',root/'copy.usda',np.zeros(3),np.zeros(3))

    def test_unsupported_normal_interpolation_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);self.triangle(root/'source.usda','uniform')
            copy_component(root/'source.usda',root/'copy.usda',np.eye(3),1,root,{})
            with self.assertRaises(ValueError):deform_new_copy(root/'copy.usda',np.zeros(3),np.zeros(3),self.warp())

    @unittest.skipUnless(DEFAULT_PACK.is_dir(),'Source package unavailable')
    def test_real_petiole_and_leaf_preserve_detail_and_normal_channels(self):
        donor=DEFAULT_PACK/'plants/components/seed101_full'
        report=audit_manifest(donor/'manifest.json');c=report['components']['SubStem_41']
        curve=donor_curve(c);origin=np.array(c['translation_plant_m'])
        direction=curve['points'][1]-curve['points'][0]
        new=curved_centerline(CurveSpec(float(curve['arc'][-1]),float(curve['radius'][0]),bend_normal_rad=.4),direction)
        warp=CurveWarp(curve['points']+origin,new,origin+[0,0,.02])
        with TemporaryDirectory() as temp:
            root=Path(temp);textures={}
            for key in ('SubStem_41','Leaf_000'):
                c=report['components'][key];path=donor/c['file'];before=path.read_bytes()
                copy_component(path,root/c['file'],np.eye(3),1,donor,textures)
                old=np.array(c['translation_plant_m']);new_origin=warp.map(old[None,:])[0]
                deform_new_copy(root/c['file'],old,new_origin,warp)
                qa=check_component(path,root/c['file'],old,new_origin,warp)
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(len(qa),2 if key.startswith('Sub') else 1)


if __name__=='__main__':unittest.main()
