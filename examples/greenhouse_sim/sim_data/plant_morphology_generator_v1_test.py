"""Focused geometry invariants; no native, source edits or plant generation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from . import plant_morphology_generator_v1 as g

SOURCE=Path(__file__).resolve().parents[3]/'data/sim_data/package_20260905/tomato_greenhouse_pack/plants/components/seed53_full/manifest.json'


class WholePlantMorphologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=json.loads(SOURCE.read_text())
        cls.config=g.configuration(cls.source,2026091801)
        cls.warp=g.PlantWarp(cls.config)

    def test_seeded_configuration_and_metadata_repeat_exactly(self):
        self.assertEqual(self.config,g.configuration(self.source,2026091801))
        a,ea=g.transformed_manifest(self.source,self.warp,'a','seed53_full','train')
        b,eb=g.transformed_manifest(self.source,self.warp,'a','seed53_full','train')
        self.assertEqual(a,b);self.assertEqual(ea,eb)

    def test_every_organ_graph_and_deleaf_state_are_preserved(self):
        new,error=g.transformed_manifest(self.source,self.warp,'variant','seed53_full','train')
        self.assertEqual(len(new['components']),len(self.source['components']))
        for old,row in zip(self.source['components'],new['components']):
            self.assertEqual((old['id'],old['parent'],old['type'],old.get('deleafed')),
                             (row['id'],row['parent'],row['type'],row.get('deleafed')))
            np.testing.assert_allclose(row['attach_point'],self.warp.map(np.array([old['attach_point']]))[0],atol=1e-12)
            np.testing.assert_allclose(row['transform']['translate'],
                self.warp.map(np.array([old['transform']['translate']]))[0],atol=1e-12)
            self.assertTrue(g.PHYSICS_FIELDS.isdisjoint(row))
        self.assertLessEqual(error,g.MAX_CENTERLINE_INTERPOLATION_ERROR_M)
        self.assertFalse(new['new_independent_source_family'])
        self.assertEqual(new['source_family'],'seed53_full')
        self.assertEqual(new['source_split'],'train')

    def test_root_fixed_and_vertical_map_strictly_monotone(self):
        np.testing.assert_array_equal(self.warp.map(self.warp.base[None,:])[0],self.warp.base)
        p=np.tile(self.warp.base,(10001,1));p[:,2]+=np.linspace(-10,10,len(p))
        self.assertTrue((np.diff(self.warp.map(p)[:,2])>0).all())
        self.assertTrue((np.linalg.det(self.warp.jacobian(p))>0).all())

    def test_analytic_jacobian_matches_independent_finite_difference(self):
        p=np.array([[.1,.2,-.1],[-.5,.6,1.4],[.3,-.2,3.]])
        numeric=np.stack([(self.warp.map(p+1e-6*axis)-self.warp.map(p-1e-6*axis))/2e-6
                          for axis in np.eye(3)],axis=2)
        np.testing.assert_allclose(self.warp.jacobian(p),numeric,atol=1e-9,rtol=1e-7)

    def test_transformed_normal_stays_perpendicular_to_transformed_tangents(self):
        p=np.array([[.1,.2,1.2]]);normal=np.array([[0.,1.,0.]])
        n,qa=self.warp.normals(p,normal);j=self.warp.jacobian(p)[0]
        self.assertAlmostEqual(float(n[0]@(j@np.array([1.,0.,0.]))),0.,places=12)
        self.assertAlmostEqual(float(n[0]@(j@np.array([0.,0.,1.]))),0.,places=12)
        self.assertAlmostEqual(float(np.linalg.norm(n[0])),1.,places=12)
        self.assertTrue(qa['global_injectivity_proven'])
        zero,_=self.warp.normals(p,np.zeros((1,3)));np.testing.assert_array_equal(zero,0)

    def test_global_distance_and_mesh_interpolation_bounds(self):
        rng=np.random.default_rng(77);a=rng.uniform(-2,4,(2000,3));b=rng.uniform(-2,4,(2000,3))
        old=np.linalg.norm(a-b,axis=1);new=np.linalg.norm(self.warp.map(a)-self.warp.map(b),axis=1)
        self.assertTrue((new<=old*self.warp.radius_upper_scale+1e-12).all())
        self.assertTrue((new>=old*self.warp.distance_lower_scale-1e-12).all())
        weights=rng.uniform(0,1,(len(a),1))
        true=self.warp.map(weights*a+(1-weights)*b)
        linear=weights*self.warp.map(a)+(1-weights)*self.warp.map(b)
        error=np.linalg.norm(true-linear,axis=1)
        self.assertTrue((error<=self.warp.interpolation_bound(a[:,2]-b[:,2])+1e-12).all())

    def test_capsule_polyline_tracks_dense_warp_and_bounds_original_proxy(self):
        original=np.array([[0.,0.,0.,.003],[.04,.12,.3,.002]])
        origin=np.array([.1,.2,1.]);new_origin=self.warp.map(origin[None,:])[0]
        output,bound=self.warp.chain(original,origin,new_origin)
        chain=np.asarray(output);n=len(chain)-1
        for i in range(n):
            ta,tb=i/n,(i+1)/n
            for f in (.1,.3,.5,.9):
                source=(1-(ta+(tb-ta)*f))*original[0]+(ta+(tb-ta)*f)*original[1]
                center=self.warp.map((source[:3]+origin)[None,:])[0]
                line=(1-f)*chain[i,:3]+f*chain[i+1,:3]+new_origin
                self.assertLessEqual(float(np.linalg.norm(center-line)),bound+1e-12)
                for axis in np.eye(3):
                    surface=self.warp.map((source[:3]+origin+source[3]*axis)[None,:])[0]
                    radius=(1-f)*chain[i,3]+f*chain[i+1,3]
                    self.assertLessEqual(float(np.linalg.norm(surface-line)),radius+1e-12)

    def test_same_junction_in_distinct_local_frames_maps_to_same_world_point(self):
        junction=np.array([.01,.5,.8])
        a=np.array([0.,.45,.7]);b=np.array([.02,.51,.81])
        new_a=self.warp.map(a[None,:])[0];new_b=self.warp.map(b[None,:])[0]
        local_a=(self.warp.map((a+(junction-a))[None,:])[0]-new_a).astype(np.float32)
        local_b=(self.warp.map((b+(junction-b))[None,:])[0]-new_b).astype(np.float32)
        np.testing.assert_allclose(new_a+local_a,new_b+local_b,atol=1e-8,rtol=0)

    def test_two_seed_shapes_are_nonrigid_and_not_scaled_copies(self):
        a,_=g.transformed_manifest(self.source,self.warp,'a','seed53_full','train')
        wb=g.PlantWarp(g.configuration(self.source,2026091802))
        b,_=g.transformed_manifest(self.source,wb,'b','seed53_full','train')
        ma=g.shape_metrics(self.source,a);mb=g.shape_metrics(self.source,b)
        self.assertNotEqual(ma['generated_normalized_stem_distance_hash'],mb['generated_normalized_stem_distance_hash'])
        self.assertGreater(g.shape_metrics(a,b)['maximum_normalized_stem_distance_change'],1e-5)

    def test_identity_translation_rotation_and_uniform_scale_rejected(self):
        angle=.4;c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        for scale,rot,offset in [(1,np.eye(3),np.zeros(3)),(1,np.eye(3),np.ones(3)),
                                  (1,rotation,np.ones(3)),(1.2,rotation,np.ones(3))]:
            variant=deepcopy(self.source)
            for row in variant['components']:
                row['transform']['translate']=(scale*rot@np.array(row['transform']['translate'])+offset).tolist()
            with self.assertRaisesRegex(ValueError,'uniform-scale'):
                g.shape_metrics(self.source,variant)

    def test_actual_deleafed_degenerate_stubs_stay_ineligible_without_false_zero_dimension(self):
        new,_=g.transformed_manifest(self.source,self.warp,'variant','seed53_full','train')
        stubs=[c for c in new['components'] if c.get('inherited_degenerate_capsule_preserved')]
        self.assertEqual(len(stubs),17)
        self.assertTrue(all(c['deleafed'] is True for c in stubs))
        self.assertTrue(all('length' not in c for c in stubs))
        self.assertTrue(all(c['length']>0 for c in new['components'] if 'length' in c))
        self.assertTrue(all(np.linalg.norm(np.diff(np.asarray(c['capsules'][0])[:,:3],axis=0))==0 for c in stubs))
        invalid=deepcopy(self.source)
        key=stubs[0]['id'];next(c for c in invalid['components'] if c['id']==key)['deleafed']=False
        with self.assertRaisesRegex(ValueError,'Zero-length intact petiole'):
            g.transformed_manifest(invalid,self.warp,'bad','seed53_full','train')

    def test_unsafe_warp_parameters_rejected(self):
        for key,value in [('vertical_scale',0.),('spacing_wave',.5),('horizontal_bend_m',[1.,0.])]:
            config=deepcopy(self.config);config[key]=value
            with self.assertRaises(ValueError):g.PlantWarp(config)


if __name__=='__main__':unittest.main(verbosity=2)

