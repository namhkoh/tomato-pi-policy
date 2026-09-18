"""Real USD 137-packed/6-whole refresh tests; synthetic sources, no native app."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from pxr import Usd,UsdGeom,Sdf,Gf
from . import native848_persistent_far_overlay_v1 as p
from .native848_farmerge_overlay_v1_test import triangle


class RefreshTests(unittest.TestCase):
    def setUp(self):
        stage=Usd.Stage.CreateInMemory();stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.Xform.Define(stage,'/World/PackPlants')
        UsdGeom.Xform.Define(stage,'/World/Robot').AddTranslateOp().Set((1,2,3))
        UsdGeom.Camera.Define(stage,'/World/Robot/Camera')
        UsdGeom.Xform.Define(stage,'/Render/Product').GetPrim().CreateAttribute('test:value',Sdf.ValueTypeNames.Int).Set(9)
        self.rows=[];plants=[]
        for i in range(144):
            root='/World/PackPlants/P'+str(i)
            x=UsdGeom.Xform.Define(stage,root);x.AddTranslateOp().Set((i,0,0));triangle(stage,root+'/Original')
            world=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(x.GetPrim()),float).tolist()
            self.rows.append(dict(component_index=i+1,prim_path=root+'/Original',component_id='SubStem_0',organ_type='sub_stem',
                variant_id='A' if i==0 else 'P'+str(i),source_plant_id='donor',split_group='donor',
                geometry_source_id='A' if i==0 else 'donor',source_geometry_modified=i==0))
            plants.append(dict(plant_root=root,source_family='donor',manifest_sha256='source' if i else 'A',plant_to_world_usd_row_vectors=world))
        scene=dict(stage=stage,robot=dict(root='/World/Robot'))
        collision=p.clone_original_collision_scene(scene)
        self.stage=stage;self.collision=collision['stage'];self.primary='/World/PackPlants/P0'
        self.census=dict(foreground_root=self.primary,unchanged_background_count=143,all_plant_roots=plants)
        self.census['deterministic_census_sha256']=p.fingerprint(self.census)
        groups=[];selected=[]
        for i in range(7,144):
            root='/World/PackPlants/P'+str(i);stage.RemovePrim(root+'/Original');mesh=triangle(stage,root+'/__FarCoarseGroup_0')
            row=self.rows[i];selected.append(deepcopy(row))
            groups.append(dict(prim_path=str(mesh.GetPath()),plant_root=root,
                plant_to_world_usd_row_vectors=plants[i]['plant_to_world_usd_row_vectors'],
                expected_signature=p.frozen.frozen.mesh_signature(mesh),expected_material_subsets={},
                member_component_indices=[i+1],original_to_current_component_indices={str(i+1):i+1}))
        retained=[p.frozen.snapshot_mesh(x,UsdGeom.XformCache()) for x in stage.Traverse()
            if x.IsA(UsdGeom.Mesh) and not str(x.GetPath()).endswith('__FarCoarseGroup_0')]
        overlay=dict(stage=stage,retained_meshes=retained,render_groups=groups,logical_census=deepcopy(self.census),logical_catalogue=self.rows,
            recipe_pin={},selection_pin={},primary_root=self.primary,whole_roots=['/World/PackPlants/P'+str(i) for i in range(1,7)],
            collision_snapshot={},native_texture_bindings={})
        # Construct only the state after the frozen install. Source authentication
        # and actual install are covered separately by the real-scene rehearsal.
        self.session=p.PersistentFarOverlay.__new__(p.PersistentFarOverlay)
        s=self.session;s.scene=scene;s.collision_scene=collision;s.stage=stage;s.collision_stage=self.collision
        s.primary=self.primary;s.reference=deepcopy(self.census);s.background_catalogue=p._background_catalogue(self.rows,self.primary)
        s.base_catalogue=deepcopy(self.rows);s.selection=dict(selected_components=selected);s.bindings={str(p.Path(p.__file__).resolve()):p.sha256(p.__file__)};s.sequence=0;s.failed=False
        s._keepalive={};s._templates=[];s.overlay=overlay;s.initial_groups=deepcopy(groups)
        s.whole_retained=[r for r in retained if not r['path'].startswith(self.primary+'/')]
        s._protect=s._protected_layers(stage);s._collision_protect=s._protected_layers(self.collision)
        s.camera_bank=dict(records={'known':{}},proofs=[])
        s._refresh_metadata(self.population('A'),self.rows)
        self.group_specs={g['prim_path']:stage.GetSessionLayer().GetPrimAtPath(g['prim_path']).layer.ExportToString() for g in groups[:1]}

    def population(self,name):
        census=deepcopy(self.census);census['all_plant_roots'][0]['manifest_sha256']=name
        census.pop('deterministic_census_sha256');census['deterministic_census_sha256']=p.fingerprint(census)
        return dict(scene_policy_evidence=census,counts=dict(components=144,component_plants=144),
            persistent_swap_evidence=dict(outside_foreground_opinions_preserved=True,unchanged_background_count=143),
            source_bindings={},template_stages={})

    def update(self,name,offset=.2):
        mesh=UsdGeom.Mesh(self.collision.GetPrimAtPath(self.primary+'/Original'))
        mesh.GetPointsAttr().Set([(0,0,0),(1+offset,0,0),(0,1,0)])
        rows=deepcopy(self.rows);rows[0].update(geometry_source_id=name,variant_id=name)
        return self.population(name),rows

    def test_real_usd_a_b_a_preserves_packed_roots_and_product(self):
        before=[p.frozen.frozen.mesh_signature(UsdGeom.Mesh(self.stage.GetPrimAtPath(g['prim_path']))) for g in self.session.overlay['render_groups']]
        pop,rows=self.update('B');self.session.refresh(pop,rows)
        pop,rows=self.update('A',0);value=self.session.refresh(pop,rows)
        after=[p.frozen.frozen.mesh_signature(UsdGeom.Mesh(self.stage.GetPrimAtPath(g['prim_path']))) for g in value['render_groups']]
        self.assertEqual(before,after);self.assertEqual(len(value['render_groups']),137)
        self.assertEqual(len(value['whole_roots']),6);self.assertEqual(self.session.sequence,3)
        self.assertEqual(self.stage.GetPrimAtPath('/Render/Product').GetAttribute('test:value').Get(),9)
        self.assertEqual(p._meshes(self.stage,self.primary),p._meshes(self.collision,self.primary))
        self.assertEqual(self.session.verify_backgrounds()['logical_plant_slots'],144)

    def test_renderer_robot_and_camera_opinions_survive_copy(self):
        self.stage.GetPrimAtPath('/World/Robot').GetAttribute('xformOp:translate').Set((9,8,7))
        UsdGeom.Camera(self.stage.GetPrimAtPath('/World/Robot/Camera')).GetFocalLengthAttr().Set(77)
        self.stage.GetPrimAtPath('/Render/Product').GetAttribute('test:value').Set(19)
        self.session.refresh(*self.update('B'))
        self.assertEqual(tuple(self.stage.GetPrimAtPath('/World/Robot').GetAttribute('xformOp:translate').Get()),(9,8,7))
        self.assertEqual(UsdGeom.Camera(self.stage.GetPrimAtPath('/World/Robot/Camera')).GetFocalLengthAttr().Get(),77)
        self.assertEqual(self.stage.GetPrimAtPath('/Render/Product').GetAttribute('test:value').Get(),19)

    def test_current_component_indices_explicitly_remapped(self):
        pop,rows=self.update('B')
        for row in rows:row['component_index']+=1000
        value=self.session.refresh(pop,rows)
        self.assertEqual(value['render_groups'][0]['member_component_indices'],[1008])
        self.assertEqual(value['render_groups'][0]['original_to_current_component_indices'],{'8':1008})

    def test_packed_mesh_mutation_rejected_before_foreground_copy(self):
        mesh=UsdGeom.Mesh(self.stage.GetPrimAtPath('/World/PackPlants/P7/__FarCoarseGroup_0'))
        mesh.GetPointsAttr().Set([(9,0,0),(1,0,0),(0,1,0)])
        before=p._meshes(self.stage,self.primary)
        with self.assertRaises(ValueError):self.session.refresh(*self.update('B'))
        self.assertEqual(before,p._meshes(self.stage,self.primary))

    def test_full_collision_background_mutation_rejected(self):
        self.collision.GetPrimAtPath('/World/PackPlants/P8').GetAttribute('xformOp:translate').Set((999,0,0))
        with self.assertRaises(ValueError):self.session.refresh(*self.update('B'))

    def test_changed_background_identity_rejected(self):
        pop,rows=self.update('B');rows[8]['source_plant_id']='other'
        with self.assertRaises(ValueError):self.session.refresh(pop,rows)

    def test_missing_collision_component_rejected(self):
        self.collision.RemovePrim('/World/PackPlants/P10/Original')
        with self.assertRaises(ValueError):self.session.refresh(*self.update('B'))

    def test_source_geometry_never_uses_packed_collision_proxy(self):
        self.session.refresh(*self.update('B'))
        self.assertFalse(self.collision.GetPrimAtPath('/World/PackPlants/P7/__FarCoarseGroup_0'))
        self.assertTrue(self.collision.GetPrimAtPath('/World/PackPlants/P7/Original'))
        self.assertTrue(self.session.overlay['collision_snapshot']['collision_background_not_packed'])

    def test_failed_foreground_copy_restores_previous_slot(self):
        before=p._meshes(self.stage,self.primary);pop,rows=self.update('B')
        real=p._meshes
        def mismatch(stage,root):
            result=real(stage,root)
            return result+[dict(fake=True)] if stage==self.stage else result
        with patch.object(p,'_meshes',side_effect=mismatch):
            with self.assertRaises(ValueError):self.session.refresh(pop,rows)
        self.assertEqual(before,p._meshes(self.stage,self.primary));self.assertTrue(self.session.failed)

    def test_unknown_finite_camera_rejected(self):
        record=dict(calibration={},joint_degrees={},robot_root_to_world_usd_row_vectors=[],camera_to_head_column_vectors=[])
        with self.assertRaises(ValueError):self.session.check_camera(record)


if __name__=='__main__':unittest.main(verbosity=2)
