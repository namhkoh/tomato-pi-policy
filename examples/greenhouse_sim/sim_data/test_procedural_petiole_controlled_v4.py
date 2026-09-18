"""Portable signed/multi-subtree control checks, without assets or renderer."""
from copy import deepcopy
from pathlib import Path
import math
import sys
import unittest
import numpy as np
if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_data.procedural_distal_deformation_v1 import SignedProximalShear,rotated_source_direction
from sim_data.procedural_petiole_controlled_v4 import control_for,component_owners
from sim_data.procedural_proximal_shear_v1 import controlled_curve,component_transport,transform_metadata
from sim_data.native848_controlled_9mm_evidence_v2 import centerline_identity


class SignedControlChecks(unittest.TestCase):
    def test_finite_signed_amplitudes_and_identity(self):
        for amplitude in [-.035,-.018,.008,.025,.035]:
            with self.subTest(amplitude=amplitude):
                control=control_for(amplitude,37.);self.assertEqual(control['amplitude_m'],amplitude)
                direction=rotated_source_direction([0,0,1],[1,0,0],37.)
                warp=SignedProximalShear([0,0,0],[1,0,0],direction,amplitude,.032)
                points=np.array([[.009,.004,0],[.019,0,.003],[.030,.002,0]])
                self.assertTrue(np.array_equal(warp.map(points),points))
                p=np.array([[.080,.002,.001]])
                self.assertTrue(np.allclose(warp.map(p)-p,amplitude*direction,atol=1e-15,rtol=0))
    def test_angle_and_signed_direction_equivalence(self):
        first=rotated_source_direction([0,0,1],[1,0,0],0.)
        quarter=rotated_source_direction([0,0,1],[1,0,0],90.)
        opposite=rotated_source_direction([0,0,1],[1,0,0],180.)
        self.assertAlmostEqual(float(first@quarter),0.,places=14)
        self.assertTrue(np.allclose(opposite,-first,atol=1e-15,rtol=0))
        a=SignedProximalShear([0,0,0],[1,0,0],first,-.025,.032)
        b=SignedProximalShear([0,0,0],[1,0,0],opposite,.025,.032)
        points=np.array([[.05,.005,.001],[.07,.001,.002]])
        self.assertTrue(np.allclose(a.map(points),b.map(points),atol=1e-15,rtol=0))
    def test_negative_jacobian_against_numeric(self):
        w=SignedProximalShear([0,0,0],[1,0,0],[0,1,0],-.032,.034)
        p=np.array([[x,.002,.003] for x in [.01,.034,.040,.050,.060,.08]])
        h=1e-7;n=np.stack([(w.map(p+h*np.eye(3)[i])-w.map(p-h*np.eye(3)[i]))/(2*h) for i in range(3)],axis=2)
        self.assertLess(float(np.max(abs(n-w.jacobian(p)))),1e-5)
        self.assertTrue(np.all(np.linalg.det(w.jacobian(p))>.05));self.assertTrue(np.all(np.linalg.cond(w.jacobian(p))<30))
    def test_oversized_control_rejected_without_adjustment(self):
        for value in [-1.,1.]:
            with self.assertRaisesRegex(ValueError,'unchanged Jacobian guard'):
                SignedProximalShear([0,0,0],[1,0,0],[0,1,0],value,.049)
    def test_invalid_amplitudes(self):
        for value in [0.,True,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):control_for(value,0.)
    def test_invalid_angles(self):
        for value in [181.,-181.,float('nan'),True]:
            with self.assertRaises(ValueError):control_for(.025,value)
    def test_nontransverse_direction_rejected(self):
        with self.assertRaises(ValueError):SignedProximalShear([0,0,0],[1,0,0],[1,1,0],.01,.032)
    def test_global_coordinate_invariant_for_signed_warp(self):
        w=SignedProximalShear([.1,.2,.3],[1,0,0],[0,1,0],-.025,.032)
        p=np.random.default_rng(241).normal(size=(100,3))*.05+w.anchor;q=w.map(p)
        self.assertTrue(np.array_equal(w.coordinate(q),w.coordinate(p)))
        self.assertTrue(np.allclose(q-w.amplitude*w.weights(q)[:,None]*w.direction_vector,p,rtol=0,atol=1e-15))


class MultiSubtreeChecks(unittest.TestCase):
    def change(self,key,anchor,amplitude,angle):
        anchor=np.asarray(anchor,float);old=np.array([[0,0,0],[.028,0,0],[.12,.004,0]])+anchor
        w=SignedProximalShear(anchor,[1,0,0],rotated_source_direction([0,0,1],[1,0,0],angle),amplitude,.031)
        curve,_=controlled_curve(old,[.003,.0029,.0027],w)
        return dict(component_id=key,members=[key,'Leaf_'+key],warp=w,curve=curve,new_anchor_m=anchor.tolist())
    def test_disjoint_subtree_owner_map(self):
        a=self.change('A',[0,0,0],.018,0.);b=self.change('B',[0,.2,0],-.028,70.)
        owners=component_owners([a,b]);self.assertEqual(set(owners),{'A','Leaf_A','B','Leaf_B'})
        self.assertIs(owners['Leaf_A'],a);self.assertIs(owners['Leaf_B'],b)
    def test_overlapping_subtree_rejected(self):
        a=self.change('A',[0,0,0],.018,0.);b=self.change('B',[0,.2,0],-.028,70.)
        b['members'].append('Leaf_A')
        with self.assertRaisesRegex(ValueError,'Overlapping'):component_owners([a,b])
    def test_duplicate_component_control_rejected(self):
        a=self.change('A',[0,0,0],.018,0.)
        with self.assertRaises(ValueError):component_owners([a,deepcopy(a)])
    def test_empty_controls_rejected(self):
        with self.assertRaises(ValueError):component_owners([])
    def test_both_modified_petiole_proximal_functions_preserved(self):
        for key,anchor,amplitude,angle in [('A',[0,0,0],.018,0.),('B',[0,.2,0],-.028,70.)]:
            c=self.change(key,anchor,amplitude,angle)
            raw=dict(id=key,type='sub_stem',parent='MainStem',transform={'translate':anchor},attach_point=anchor,
                axis=[1,0,0],radius=.003,capsules=[[[0,0,0,.003],[.028,0,0,.0029],[.12,.004,0,.0027]]])
            changed=transform_metadata(raw,c)
            old=dict(type='sub_stem',deleafed=False,translation_plant_m=anchor,attachment_plant_m=anchor,
                axis_plant=[1,0,0],radius_m=.003,capsules_local_m=raw['capsules'])
            new=dict(old,capsules_local_m=changed['capsules'])
            proof=centerline_identity({'components':{key:old}},{'components':{key:new}},key)
            self.assertTrue(proof['protected_source_centerline_identity'])
    def test_leaf_graph_and_local_shape_preserved(self):
        for c in [self.change('A',[0,0,0],.018,0.),self.change('B',[0,.2,0],-.028,70.)]:
            anchor=np.asarray(c['new_anchor_m']);raw=dict(id='Leaf_'+c['component_id'],type='leaf',parent=c['component_id'],
                transform={'translate':(anchor+[.09,.001,.002]).tolist()},attach_point=(anchor+[.09,0,0]).tolist(),
                axis=[0,1,0],capsules=[[[0,0,0,.001],[.01,.02,0,.001]]])
            changed=transform_metadata(raw,c);self.assertEqual(changed['id'],raw['id']);self.assertEqual(changed['parent'],raw['parent'])
            self.assertEqual(changed['capsules'],raw['capsules']);self.assertEqual(changed['axis'],raw['axis'])
            expected=c['warp'].amplitude*c['warp'].direction_vector
            self.assertTrue(np.allclose(np.asarray(changed['attach_point'])-raw['attach_point'],expected,atol=1e-15,rtol=0))
    def test_wrong_leaf_owner_rejected(self):
        c=self.change('A',[0,0,0],.018,0.)
        with self.assertRaises(ValueError):component_transport(dict(id='Leaf_B',type='leaf',parent='B',attach_point=[.09,0,0]),c)


if __name__=='__main__':unittest.main()
