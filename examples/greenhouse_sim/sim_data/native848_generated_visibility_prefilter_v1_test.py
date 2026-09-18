import json,unittest
import numpy as np
from . import native848_generated_visibility_prefilter_v1 as f
from .native_capture_v5 import visibility_shadow as m

class RayPrefilterTests(unittest.TestCase):
 def point(self,z=.21,arc=.029,pixel=(.2,.3),radius=.003):
  return dict(arc_m=arc,radius_m=radius,projected=dict(projection_status='in_frame',pixel_xy=list(pixel),camera_optical_xyz_m=[0,0,z]))
 def hit(self,z,cid='SubStem_41'):
  return dict(component_id=cid,possible_optical_z_m=z,certain_optical_z_m=z,prim_path='/test')
 def ray(self,*hits):return dict(hits=list(hits),unknown_meshes=[])
 def test_same_component_depth_reject(self):
  r=f.probe(self.point(),self.ray(self.hit(.2)),'SubStem_41',False)
  self.assertTrue(r['geometrically_rejected']);self.assertAlmostEqual(r['tolerance_m'],.006)
 def test_native_tolerance_boundary(self):
  self.assertFalse(f.probe(self.point(z=.2059),self.ray(self.hit(.2)),'SubStem_41',False)['geometrically_rejected'])
 def test_other_component_ownership(self):
  r=f.probe(self.point(z=.205),self.ray(self.hit(.204),self.hit(.203,'Leaf_000')),'SubStem_41',False)
  self.assertEqual(r['reason'],'authored_other_component_in_front')
 def test_near_joint_owner_deferred(self):
  r=f.probe(self.point(z=.205,arc=.004),self.ray(self.hit(.204),self.hit(.203,'MainStem_27')),'SubStem_41',True)
  self.assertFalse(r['geometrically_rejected'])
 def test_missing_surface_is_unknown_not_reject(self):
  self.assertFalse(f.probe(self.point(),self.ray(),'SubStem_41',False)['geometrically_rejected'])
 def test_offaxis_distance_converts_to_optical_z(self):
  triangles=np.array([[[-3,-2,.2],[4,-2,.2],[-3,5,.2]]],float)
  mesh=m.Mesh('SubStem_41','sub_stem','/test',triangles.min((0,1)),triangles.max((0,1)),triangles,triangles,np.array([0]),1,())
  snapshot=m.Snapshot((mesh,),b'{}',np.eye(4),(),b'{}')
  cal=dict(intrinsics=[[2.,0,0],[0,2.,0],[0,0,1]],camera_to_world_usd_row_vectors=np.diag([1.,-1.,-1.,1.]).tolist(),clipping_range_m=[.01,1.])
  r=f._cast(snapshot,cal,(1,0));self.assertAlmostEqual(r['hits'][0]['certain_optical_z_m'],.2)
 def test_proximal_unique_pixels_and_native95percent(self):
  rows=[dict(arc_m=.004+i*.001,projected=dict(projection_status='in_frame'),pixel_xy=[i,0],geometrically_rejected=i<2) for i in range(27)]
  groups=dict(nominal=[],support=[],junction=[],proximal=rows+rows[:2]);r=f.aggregate(groups)
  self.assertEqual(r['proximal_unique_pixels'],27);self.assertAlmostEqual(r['proximal_possible_visible_fraction_upper_bound'],25/27);self.assertTrue(r['reject'])
  rows[1]['geometrically_rejected']=False;self.assertFalse(f.aggregate(dict(groups,proximal=rows))['reject'])
 def test_supported_probe_rejection_is_not_fraction_diluted(self):
  bad=dict(geometrically_rejected=True,projected=dict(projection_status='in_frame'),pixel_xy=[0,0]);good=dict(bad,geometrically_rejected=False)
  result=f.aggregate(dict(nominal=[good],support=[good,bad],junction=[good],proximal=[good]));self.assertTrue(result['reject']);self.assertIn('support_authored_geometry_rejection',result['reasons'])
if __name__=='__main__':unittest.main()
