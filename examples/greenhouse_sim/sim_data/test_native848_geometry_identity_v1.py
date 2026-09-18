"""Content identity regressions: relabels collide; geometry/material changes do not."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import native848_geometry_identity_v1 as g

USD = """#usda 1.0
def Xform "Part" {
 def Mesh "Mesh" {
  point3f[] points = [(0,0,0),(1,0,0),(0,1,0)]
  int[] faceVertexCounts = [3]
  int[] faceVertexIndices = [0,1,2]
 }
 def Shader "Texture" {
  asset inputs:file = @textures/color.png@
 }
}
"""

class GeometryIdentityTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  (self.root/'textures').mkdir();(self.root/'textures/color.png').write_bytes(b'bound texture content')
  (self.root/'part.usda').write_text(USD)
  self.manifest=dict(seed=1,generator='one',version='1',source_family='seed1_full',variant_id='v1',
    output_path='/one/path',units='meters',up_axis='Z',component_count=1,
    components=[dict(id='MainStem_00',parent=None,type='main_stem',file='part.usda',
      transform={'translate':[0,0,0]},attach_point=[0,0,0],capsules=[[[0,0,0,.01],[0,1,0,.01]]])])
  self.path=self.root/'manifest.json';self.write()
 def write(self):self.path.write_text(json.dumps(self.manifest))
 def bindings(self):return {str(p.resolve()):g.file_sha256(p) for p in self.root.rglob('*') if p.is_file()}
 def identity(self):return g.geometry_identity(self.path,self.bindings())['geometry_sha256']
 def test_seed_family_variant_path_and_filename_relabel_are_identical(self):
  before=self.identity();self.manifest.update(seed=99,generator='two',version='8',source_family='seed99_full',variant_id='v9',output_path='/other/host')
  (self.root/'renamed.usda').write_bytes((self.root/'part.usda').read_bytes());self.manifest['components'][0]['file']='renamed.usda';self.write()
  self.assertEqual(before,self.identity())
 def test_attachment_capsules_and_unknown_metadata_are_part_of_geometry(self):
  original=deepcopy(self.manifest);before=self.identity()
  for field,value in [('attach_point',[0,.001,0]),('capsules',[[[0,0,0,.02],[0,1,0,.01]]]),('new_geometry_field',2)]:
   self.manifest=deepcopy(original);self.manifest['components'][0][field]=value;self.write();self.assertNotEqual(before,self.identity())
 def test_mesh_and_referenced_texture_bytes_change_digest(self):
  before=self.identity();(self.root/'part.usda').write_text(USD.replace('(1,0,0)','(1.1,0,0)'));mesh=self.identity();self.assertNotEqual(before,mesh)
  (self.root/'textures/color.png').write_bytes(b'other material content');self.assertNotEqual(mesh,self.identity())
 def test_changed_or_unbound_dependency_is_rejected(self):
  pins=self.bindings();(self.root/'textures/color.png').write_bytes(b'changed')
  with self.assertRaises(ValueError):g.geometry_identity(self.path,pins)
  pins=self.bindings();del pins[str((self.root/'part.usda').resolve())]
  with self.assertRaises(ValueError):g.geometry_identity(self.path,pins)
 def test_unpinned_texture_is_explicit_new_identity_input_and_extra_pin_checked(self):
  pins=self.bindings();texture=str((self.root/'textures/color.png').resolve());sha=pins.pop(texture)
  with self.assertRaises(ValueError):g.geometry_identity(self.path,pins)
  result=g.geometry_identity(self.path,pins,allow_new_texture_bindings=True)
  self.assertEqual(result['new_identity_dependency_bindings'],{texture:sha})
  current=g.geometry_identity(self.path,pins,{texture:sha})
  self.assertEqual(current['geometry_sha256'],result['geometry_sha256'])
  self.assertEqual(current['extra_authenticated_dependency_bindings'],{texture:sha})
  with self.assertRaises(ValueError):g.geometry_identity(self.path,pins,{texture:'f'*64})
 def test_close_rehash_detects_change_during_dependency_resolution(self):
  from pxr import UsdUtils
  actual=UsdUtils.ComputeAllDependencies;pins=self.bindings()
  def mutate(path):
   result=actual(path);(self.root/'part.usda').write_text(USD+'\n# changed during operation\n');return result
  with patch.object(UsdUtils,'ComputeAllDependencies',side_effect=mutate):
   with self.assertRaises(ValueError):g.geometry_identity(self.path,pins)
 def test_parent_graph_and_path_escape_fail_closed(self):
  for key,value in [('parent','absent'),('file','../escape.usda')]:
   self.manifest['components'][0][key]=value;self.write()
   with self.assertRaises(ValueError):self.identity()
   self.manifest['components'][0].update(parent=None,file='part.usda')
 def test_shared_scene_key_ignores_kind_and_manifest_names(self):
  slots=[];geometries={}
  for i in range(144):
   matrix=g.np.eye(4);matrix[3,0]=i;root='/World/p'+str(i)
   slots.append(dict(plant_root=root,source_family='seed1_full',source_split='train',plant_to_world_usd_row_vectors=matrix.tolist(),manifest_path='/old/path',source_geometry_modified=False));geometries[root]='a'*64
  census=dict(all_plant_roots=slots,dataset_split='train',complete_active_plant_anatomy=True)
  before=g.scene_identity(census,geometries)
  for row in slots:row.update(source_geometry_modified=True,manifest_path='/new/path')
  self.assertEqual(before,g.scene_identity(census,geometries))
  geometries[slots[0]['plant_root']]='b'*64;self.assertNotEqual(before,g.scene_identity(census,geometries))

if __name__=='__main__':unittest.main()
