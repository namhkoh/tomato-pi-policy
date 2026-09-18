import unittest
import numpy as np
from sim_data.background_scoring_v1 import score


class BackgroundScoringTests(unittest.TestCase):
    def setUp(self):
        self.cat=[dict(prim_path='/plants/fg/Leaf',component_index=1,variant_id='fg',organ_type='leaf'),
                  dict(prim_path='/plants/bg/Leaf',component_index=2,variant_id='bg',organ_type='leaf')]
        self.ids=np.array([[1,2,3,4],[5,6,7,0]],np.uint32)
        self.mapping={'renderer_id_to_prim':{'1':'/plants/fg/Leaf/mesh','2':'/plants/bg/Leaf/mesh',
            '3':'/World/RBY1/link','4':'/World/Floor/mesh','5':'/World/Gutters/a','6':'/unknown','7':'/plants/bg/LeafFake'}}
        self.depth=np.ones((2,4));self.depth[0==self.ids]=np.inf
        self.valid=np.isfinite(self.depth)
    def run_score(self, **kw):
        return score(self.ids,self.mapping,self.cat,{'fg'},depth=self.depth,valid=self.valid,tile_rows=2,tile_columns=4,**kw)
    def test_exact_ownership_exclusions_unknown(self):
        r=self.run_score();self.assertEqual(r['background_plant_pixel_fraction'],1/8)
        self.assertEqual(r['unknown_unmapped_fraction'],2/8);self.assertEqual(r['visible_background_instance_count'],1)
        self.assertEqual(r['tiles']['occupied_any_plant_pixels'],1)
    def test_unmapped_not_plant(self):
        del self.mapping['renderer_id_to_prim']['2'];r=self.run_score()
        self.assertEqual(r['background_plant_pixel_fraction'],0);self.assertEqual(r['pixel_categories']['unmapped'],1)
    def test_coarse_plant_not_leaf(self):
        self.mapping['renderer_id_to_prim']['2']='/packed/bg'
        g=dict(prim_path='/packed/bg',member_component_indices=[2])
        self.mapping['plant_ID_ownership']={'2':dict(kind='coarse_far_group',**g)}
        r=self.run_score(coarse_groups=[g]);self.assertEqual(r['background_plant_pixel_fraction'],1/8)
        self.assertEqual(r['background_leaf_pixel_fraction_lower_bound'],0)
    def test_coarse_mismatch_rejected(self):
        self.mapping['renderer_id_to_prim']['2']='/packed/bg'
        with self.assertRaises(ValueError):self.run_score(coarse_groups=[dict(prim_path='/packed/bg',member_component_indices=[2])])
    def test_coarse_foreground_rejected(self):
        with self.assertRaises(ValueError):self.run_score(coarse_groups=[dict(prim_path='/packed/fg',member_component_indices=[1])])
    def test_uint32_required(self):
        self.ids=self.ids.astype(np.int32)
        with self.assertRaises(ValueError):self.run_score()

if __name__=='__main__':unittest.main()
