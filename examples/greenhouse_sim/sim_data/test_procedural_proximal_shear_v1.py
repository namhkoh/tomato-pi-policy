"""Portable proximal-shear and curved-source regression tests.

Run directly with Python or collect with unittest/pytest. These tests use only
small in-code geometry fixtures; no diagnostic snapshots, plans, plant assets,
generated outputs, or native renderer are needed. NumPy and the sim_data CPU
geometry dependencies (including USD Python bindings) must be installed.
"""
from pathlib import Path
import sys
import unittest
import numpy as np

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from sim_data.procedural_proximal_shear_v1 import (
    ProximalShear, protected_face_indices, component_transport,
    controlled_curve, transform_metadata,
)
from sim_data.native848_controlled_9mm_evidence_v2 import centerline_identity


class ShearChecks(unittest.TestCase):
    def setUp(self):self.w=ProximalShear([0,0,0],[1,0,0],[0,1,0],.025,.04)
    def test_exact_proximal_identity(self):
        p=np.array([[x,y,z] for x in [-.01,0,.009,.019,.03,.04] for y in [-.004,0,.004] for z in [-.004,0,.004]])
        self.assertTrue(np.array_equal(self.w.map(p),p))
    def test_distal_rigid_shift(self):
        p=np.array([[.06,.01,0],[.1,.2,.3]])
        self.assertTrue(np.allclose(self.w.map(p)-p,[0,.025,0],rtol=0,atol=1e-15))
    def test_numeric_jacobian(self):
        p=np.array([[x,.01,.003] for x in np.linspace(.035,.065,21)])
        h=1e-7;numeric=np.stack([(self.w.map(p+h*np.eye(3)[i])-self.w.map(p-h*np.eye(3)[i]))/(2*h) for i in range(3)],axis=2)
        self.assertLess(float(np.max(abs(numeric-self.w.jacobian(p)))),1e-8)
    def test_global_injective(self):
        rng=np.random.default_rng(61);p=rng.uniform(-.1,.2,(1000,3));q=self.w.map(p)
        recovered=q-self.w.amplitude*self.w.weights(q)[:,None]*self.w.direction_vector
        self.assertTrue(np.allclose(p,recovered,rtol=0,atol=1e-15))
    def test_normals_and_guard(self):
        p=np.array([[.02,0,0],[.05,0,0],[.08,0,0]]);n=np.array([[.3,.4,.5]]*3)
        out,q=self.w.normals(p,n);self.assertTrue(np.array_equal(out[[0,2]],n[[0,2]]))
        expected=np.linalg.solve(self.w.jacobian(p[1]).T,n[1]);expected/=np.linalg.norm(expected)
        self.assertTrue(np.allclose(expected,out[1]));self.assertGreater(q['minimum_determinant'],.05);self.assertLess(q['maximum_condition_number'],30)
    def test_no_longitudinal_entry(self):
        p=np.array([[.030001,.01,.02],[.04,.1,.02],[.08,-.1,.03]])
        self.assertTrue(np.array_equal(self.w.coordinate(p),self.w.coordinate(self.w.map(p))))
    def test_protect_whole_crossing_face(self):
        p=np.array([[.029,0,0],[.045,0,0],[.045,.002,0],[.07,0,0],[.08,0,0],[.08,.002,0]])
        ids,faces,_=protected_face_indices(p,[3,3],[0,1,2,3,4,5],[0,0,0],[1,0,0])
        self.assertEqual(ids.tolist(),[0,1,2]);self.assertEqual(faces,[0])
    def test_reject_wrong_direction(self):
        with self.assertRaises(ValueError):ProximalShear([0,0,0],[1,0,0],[1,1,0],.025,.04)
    def test_reject_unsupported_blend(self):
        with self.assertRaises(ValueError):ProximalShear([0,0,0],[1,0,0],[0,1,0],.025,.059)
    def test_reject_wrong_amplitude(self):
        with self.assertRaises(ValueError):ProximalShear([0,0,0],[1,0,0],[0,1,0],.026,.04)
    def test_leaf_transport_and_rejection(self):
        change=dict(component_id='SubStem_1',warp=self.w)
        raw=dict(id='Leaf_1',type='leaf',parent='SubStem_1',attach_point=[.08,0,0])
        leaf=component_transport(raw,change);self.assertTrue(np.array_equal(leaf.direction([0,0,0],[.3,.4,.5]),[.3,.4,.5]))
        raw['attach_point']=[.05,0,0]
        with self.assertRaises(ValueError):component_transport(raw,change)
        raw['parent']='Other'
        with self.assertRaises(ValueError):component_transport(raw,change)
    def test_insert_proximal_knots(self):
        c,s=controlled_curve([[0,0,0],[.11,0,0],[.15,.01,0]],[.003,.0028,.0025],self.w)
        self.assertTrue(all(v in s for v in [0,.03,.04,.06,.11]));self.assertTrue(np.array_equal(c['points'][s<=.04],np.c_[s[s<=.04],np.zeros((sum(s<=.04),2))]))


class CurvedChecks(unittest.TestCase):
    def test_projection_knots_on_every_intersecting_segment(self):
        old=np.array([[0,0,0],[.028,0,0],[.09,.003,.001],[.12,.005,.003]])
        radii=np.array([.003,.0029,.0028,.0027]);warp=ProximalShear(old[0],[1,0,0],[0,1,0],.025,.031)
        curve,samples=controlled_curve(old,radii,warp);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(old,axis=0),axis=1))]
        self.assertTrue(all(x in samples for x in arc))
        points=np.column_stack([np.interp(samples,arc,old[:,i]) for i in range(3)])
        for level in np.linspace(warp.guard,.060,33):self.assertLess(float(np.min(abs(points[:,0]-level))),1e-14)
        self.assertTrue(np.array_equal(curve['points'][samples<=.03],points[samples<=.03]))
        for i,x in enumerate(arc):self.assertEqual(curve['radius'][np.flatnonzero(samples==x)[0]],radii[i])
    def test_nonmonotonic_segment_intersections(self):
        old=np.array([[0,0,0],[.075,0,0],[.04,.04,0],[.10,.07,0]])
        warp=ProximalShear([0,0,0],[1,0,0],[0,1,0],.025,.031)
        curve,s=controlled_curve(old,[.003]*4,warp);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(old,axis=0),axis=1))]
        x=np.interp(s,arc,old[:,0])
        for lo,hi in [(arc[0],arc[1]),(arc[1],arc[2]),(arc[2],arc[3])]:
            self.assertLess(float(np.min(abs(x[(s>lo)&(s<hi)]-.05))),.001)
    def test_out_of_protected_identity_rejected(self):
        class BadField(ProximalShear):
            def map(self,points):
                p=super().map(points);p=np.array(p);p[...,1]+=.00001;return p
        bad=BadField([0,0,0],[1,0,0],[0,1,0],.025,.031)
        with self.assertRaises(ValueError):controlled_curve([[0,0,0],[.02,0,0],[.1,.004,0]],[.003]*3,bad)
    def test_short_prefix_rejected(self):
        warp=ProximalShear([0,0,0],[1,0,0],[0,1,0],.025,.031)
        with self.assertRaises(ValueError):controlled_curve([[0,0,0],[.02,0,0]],[.003]*2,warp)


class ProtectedCenterlineChecks(unittest.TestCase):
    @staticmethod
    def reports(local_points,radii,guard=.031):
        local=np.asarray(local_points,float);radii=np.asarray(radii,float)
        anchor=np.array([.039923,.595920,.500784]);old_absolute=local+anchor
        tangent=local[1]-local[0];tangent/=np.linalg.norm(tangent)
        direction=np.cross([0,0,1],tangent);direction/=np.linalg.norm(direction)
        warp=ProximalShear(anchor,tangent,direction,.025,guard)
        curve,samples=controlled_curve(old_absolute,radii,warp)
        key='SubStem_fixture'
        raw=dict(id=key,type='sub_stem',parent='MainStem_fixture',
            transform={'translate':anchor.tolist()},attach_point=anchor.tolist(),
            axis=tangent.tolist(),radius=float(radii[0]),
            capsules=[np.c_[local,radii].tolist()])
        change=dict(component_id=key,warp=warp,curve=curve,new_anchor_m=anchor.tolist())
        metadata=transform_metadata(raw,change)
        old_component=dict(type='sub_stem',deleafed=False,translation_plant_m=anchor.tolist(),attachment_plant_m=anchor.tolist(),
            axis_plant=tangent.tolist(),radius_m=float(radii[0]),capsules_local_m=raw['capsules'])
        new_component=dict(type='sub_stem',deleafed=False,translation_plant_m=metadata['transform']['translate'],
            attachment_plant_m=metadata['attach_point'],axis_plant=metadata['axis'],
            radius_m=metadata['radius'],capsules_local_m=metadata['capsules'])
        return {'components':{key:old_component}}, {'components':{key:new_component}}, key, samples

    @staticmethod
    def bent53_fixture():
        # Three original donor53 knots reproduce its28.885mm first segment and
        # 2.53degree bend. This is numeric regression data, not an asset binding.
        return ProtectedCenterlineChecks.reports(
            [[0,0,0],[.025085,.014223,-.001678],[.130888,.070768,-.004279]],
            [.003019,.002900,.002736])

    def test_bent_prefix_point_radius_and_open_tangents(self):
        old,new,key,samples=self.bent53_fixture()
        proof=centerline_identity(old,new,key)
        self.assertTrue(proof['protected_source_centerline_identity'])
        self.assertTrue(proof['curved_source_prefix_supported'])
        self.assertTrue(any(.0288<r['source_arc_m']<.0289 for r in proof['rows']))
        self.assertEqual(proof['tangent_policy'],
            'each_open_segment_midpoint_plus_nominal9mm_and_support19mm')
        self.assertTrue(all(r['maximum_tangent_error']<=1e-13 for r in proof['tangent_rows']))
        self.assertTrue(all(r['radius_error_m']<=1e-15 for r in proof['rows']))

    def test_altered_proximal_radius_rejected(self):
        old,new,key,_=self.bent53_fixture()
        new['components'][key]['capsules_local_m'][0][1][3]+=.0001
        with self.assertRaisesRegex(ValueError,'Proximal centerline/radius function changed'):centerline_identity(old,new,key)

    def test_altered_original_bend_rejected(self):
        old,new,key,_=self.bent53_fixture()
        new['components'][key]['capsules_local_m'][0][1][1]+=.0001
        with self.assertRaisesRegex(ValueError,'Proximal centerline/radius function changed'):centerline_identity(old,new,key)

    def test_original_straight_prefix_shape_unchanged(self):
        for length,guard in [(.10478654953761957,.0451093441028005),
                             (.10894605232866403,.03080776932167505)]:
            with self.subTest(length=length):
                old,new,key,_=self.reports([[0,0,0],[length,0,0],[length+.04,.003,0]],
                    [.003,.0028,.0027],guard=guard)
                proof=centerline_identity(old,new,key)
                self.assertTrue(proof['protected_source_centerline_identity'])
                self.assertTrue(all(r['maximum_point_error_m']<=1e-14 for r in proof['rows']))


if __name__=='__main__':
    unittest.main()
