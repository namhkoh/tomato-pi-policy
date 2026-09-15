import copy
import unittest
import numpy as np
from sim_data.morphology_context import descriptor,equivalent,fingerprint,ContextInventory,validate


class MorphologyContextTests(unittest.TestCase):
    def setUp(self):
        self.donor=np.array([[.01,0,0,.002],[.06,0,.01,.0015],[.13,.01,.02,.001]])
        self.parent=np.array([[0,0,-.1,.006],[0,0,.15,.005]])
        self.current=self.donor.copy();self.current[1,1]=.02
        self.leaves=[dict(attachment=[.08,.015,.01],centroid=[.09,.08,.04],axis=[0,1,0],covariance=np.diag([.001,.002,.0001]))]

    def make(self):return descriptor(self.donor,self.parent,self.current,self.leaves)

    def test_invariant_rotation_translation_uniform_scale_and_names(self):
        a=self.make();rng=np.random.default_rng(42)
        for scale in [.01,.4,1,100]:
            r,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(r)<0:r[:,0]*=-1
            t=rng.normal(size=3)
            def chain(q):return np.column_stack((q[:,:3]@r.T*scale+t,q[:,3]*scale))
            leaves=[dict(attachment=np.asarray(l['attachment'])@r.T*scale+t,
                centroid=np.asarray(l['centroid'])@r.T*scale+t,axis=np.asarray(l['axis'])@r.T,
                covariance=r@l['covariance']@r.T*scale**2,irrelevant_name='renamed') for l in self.leaves]
            b=descriptor(chain(self.donor),chain(self.parent),chain(self.current),leaves)
            self.assertTrue(equivalent(a,b,1e-9));self.assertEqual(fingerprint(a),fingerprint(b))

    def test_names_and_reordering_do_not_add_novelty(self):
        second=copy.deepcopy(self.leaves[0]);second['centroid']=[.07,-.1,.04]
        self.leaves.append(second);a=self.make()
        self.leaves.reverse();self.leaves[0]['name']='not_a_feature'
        b=self.make();self.assertTrue(equivalent(a,b,0));self.assertEqual(a,b)

    def test_context_and_curve_changes_are_measured(self):
        a=self.make();self.leaves[0]['centroid'][1]+=.03
        self.assertFalse(equivalent(a,self.make(),.05))
        self.leaves[0]['centroid'][1]-=.03;self.current[1,1]+=.03
        self.assertFalse(equivalent(a,self.make(),.05))

    def test_radius_and_parent_context_changes(self):
        a=self.make();self.current[:,3]*=3
        self.assertFalse(equivalent(a,self.make(),.01))
        self.current=self.donor.copy();self.current[1,1]=.02;self.parent[-1,2]+=.05
        self.assertFalse(equivalent(a,self.make(),.05))

    def test_leaf_orientation_covariance_not_just_eigenvalues(self):
        a=self.make();self.leaves[0]['covariance']=np.diag([.002,.001,.0001])
        self.assertFalse(equivalent(a,self.make(),.01))

    def test_bipartite_matching_not_greedy(self):
        a=self.make();a['leaves']=[np.zeros(15).tolist(),np.zeros(15).tolist()];a['leaf_count']=2
        b=copy.deepcopy(a);a['leaves'][0][0]=.1;b['leaves'][1][0]=.2
        self.assertTrue(equivalent(a,b,.11))
        b['leaves'][1][0]=.4;self.assertFalse(equivalent(a,b,.11))

    def test_inventory_cross_batch_and_transitive_neighbours(self):
        inv=ContextInventory(.05);a=self.make()
        self.assertTrue(inv.add('one',a)['novel_context_candidate'])
        b=copy.deepcopy(a);b['fixed'][0]+=.04
        self.assertFalse(inv.add('two',b)['novel_context_candidate'])
        c=copy.deepcopy(a);c['fixed'][0]+=.08
        r=inv.add('three',c);self.assertEqual(r['near_duplicate_of'],['two'])
        self.assertFalse(r['training_approved']);self.assertFalse(r['source_cap_reset'])
        with self.assertRaises(ValueError):inv.add('one',a)

    def test_empty_leaves(self):
        self.leaves=[];a=self.make();self.assertTrue(equivalent(a,a,0))

    def test_malformed_inputs(self):
        for bad in [float('nan'),-1,True]:
            with self.assertRaises(ValueError):ContextInventory(bad)
        a=self.make();b=copy.deepcopy(a);b['leaf_count']+=1
        with self.assertRaises(ValueError):validate(b)
        b=copy.deepcopy(a);b['fixed'][0]=float('inf')
        with self.assertRaises(ValueError):validate(b)
        self.current[1]=self.current[0]
        with self.assertRaises(ValueError):self.make()


if __name__=='__main__':unittest.main()
