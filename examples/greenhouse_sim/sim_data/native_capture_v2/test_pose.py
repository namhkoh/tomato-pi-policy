import unittest,copy
import numpy as np
from sim_data.native_capture_v2.pose import POLICY,bounded_reference_root
from sim_data.native_capture_v2.plan import specs


class WiderPoseTests(unittest.TestCase):
    def setUp(self):
        self.row=np.eye(4);self.row[3,:3]=[.4,.2,.101]
        self.ref=dict(robot_root_to_world_usd_row_vectors=self.row.tolist(),visual_bound_screen=dict(passed=True))
        self.target=[0,0,1.5]
    def test_nine_unique_reproducible_actual_base_views(self):
        s=specs(self.ref,self.target,'stable');self.assertEqual(s,specs(self.ref,self.target,'stable'))
        self.assertEqual(len(s),9);self.assertEqual(len({(r['root_x_m'],r['y_offset_m']) for r in s}),9)
        for row in s:
            m=bounded_reference_root(self.ref,row,self.target)
            np.testing.assert_array_equal(m[:3,:3],self.row[:3,:3].T)
            self.assertEqual(m[2,3],.101);self.assertGreaterEqual(m[0,3],.48)
    def test_opposite_side_moves_outward_not_inward(self):
        self.row[3,0]=-.4;self.ref['robot_root_to_world_usd_row_vectors']=self.row.tolist()
        for row in specs(self.ref,self.target,'stable'):
            self.assertLessEqual(bounded_reference_root(self.ref,row,self.target)[0,3],-.48)
    def test_limits_policy_heading_and_displacement(self):
        good=specs(self.ref,self.target,'stable')[0]
        for key,value in [('root_x_m',.7),('root_x_m',.39),('y_offset_m',.31),
                          ('root_yaw_degrees',2),('base_heading_preserved',False),
                          ('view_policy','legacy'),('base_displacement_from_reference_m',0),('root_x_m',float('nan'))]:
            with self.subTest(key=key,value=value):
                row=dict(good);row[key]=value
                with self.assertRaises(ValueError):bounded_reference_root(self.ref,row,self.target)
    def test_invalid_reference_or_ambiguous_aisle(self):
        good=specs(self.ref,self.target,'stable')[0]
        self.ref['visual_bound_screen']['passed']=False
        with self.assertRaises(ValueError):bounded_reference_root(self.ref,good,self.target)
        self.ref['visual_bound_screen']['passed']=True
        with self.assertRaises(ValueError):bounded_reference_root(self.ref,good,[.4,0,1.5])
        self.row[0,0]=-1;self.ref['robot_root_to_world_usd_row_vectors']=self.row.tolist()
        with self.assertRaises(ValueError):bounded_reference_root(self.ref,good,self.target)


if __name__=='__main__':unittest.main()
