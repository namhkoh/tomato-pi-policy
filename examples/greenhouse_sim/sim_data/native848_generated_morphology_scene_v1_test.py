"""Synthetic 144-slot USD integration checks; no native or production scene load."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pxr import Gf, Usd, UsdGeom

from . import native848_generated_morphology_scene_v1 as g


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def pin(path):
    return dict(path=str(path.resolve()), sha256=g.sha256(path))


def component_asset(path, size):
    stage=Usd.Stage.CreateNew(str(path)); root=UsdGeom.Xform.Define(stage,'/Component')
    stage.SetDefaultPrim(root.GetPrim()); UsdGeom.Cube.Define(stage,'/Component/Mesh').GetSizeAttr().Set(size)
    stage.GetRootLayer().Save()


class GeneratedForegroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='generated-foreground-')
        self.base=Path(self.tmp.name); self.donor=self.base/'seed53_full';self.donor.mkdir()
        self.generated=self.base/'seed53_full_morph_fixture';self.generated.mkdir()
        rows=[]
        for cid,parent,kind,position in [('MainStem_0',None,'main_stem',[0.,0.,0.]),
            ('SubStem_0','MainStem_0','sub_stem',[0.,0.,.4]),
            ('Leaf_0','SubStem_0','leaf',[.1,0.,.4])]:
            component_asset(self.donor/(cid+'.usda'), .03)
            component_asset(self.generated/(cid+'.usda'), .04)
            rows.append(dict(id=cid,parent=parent,type=kind,file=cid+'.usda',transform=dict(translate=position),
                attach_point=position,axis=[1.,0.,0.],radius=.002,length=.1,
                capsules=[[[0.,0.,0.,.002],[.1,0.,0.,.002]]],deleafed=False))
        manifest=dict(units='meters',up_axis='Z',component_count=3,components=rows)
        write(self.donor/'manifest.json',manifest)
        new=deepcopy(manifest);new.update(variant_id=self.generated.name,source_family=self.donor.name,
            split_group=self.donor.name,source_split='train',capsule_radius_policy=g.RADIUS_POLICY)
        new['components'][1]['transform']['translate'][2]=.42
        write(self.generated/'manifest.json',new)
        source_pins={str(p.resolve()):g.sha256(p) for p in self.donor.iterdir()}
        self.plan_path=self.base/'plan.json'
        self.plan=dict(package=str(self.base),family_assignments={self.donor.name:'train'},
            jobs=[dict(plant_family=self.donor.name,split='train',source_manifest_path=str(self.donor/'manifest.json'))],
            source_bindings_sha256=source_pins)
        write(self.plan_path,self.plan);self.plan_pin=pin(self.plan_path)
        generator=Path(g.__file__).with_name('plant_morphology_generator_v1.py').resolve()
        self.q=dict(schema=g.QUALIFICATION_SCHEMA,variant_id=self.generated.name,source_family=self.donor.name,
            split_group=self.donor.name,original_source_family_cap_group=self.donor.name,source_split='train',
            source_assets_unchanged=True,full_component_graph_preserved=True,source_deleaf_state_preserved=True,
            training_approved=False,native_capture_validated=False,rendered_geometry_qualified=False,
            morphology_diversity_approved=False,new_independent_source_family=False,review_decisions_inherited=False,
            capsules_are_certified_mesh_geometry=False,accepted_training_increment=0,static_perception_only=True,
            physics_supported=False,capsule_radius_policy=g.RADIUS_POLICY,source_geometry_hash='donor',
            generated_geometry_hash='different_fixture',source_component_count=3,
            source_bindings={**source_pins,str(self.plan_path):self.plan_pin['sha256']},texture_source_bindings={},
            implementation_hashes={str(generator):g.GENERATOR_SHA256},
            output_hashes={p.name:g.sha256(p) for p in self.generated.iterdir()})
        self.qp=self.generated/'qualification.json';self.save_q()
        self.stage=Usd.Stage.CreateInMemory();self.stage.SetEditTarget(self.stage.GetSessionLayer())
        parent=UsdGeom.Xform.Define(self.stage,'/World/PackPlants')
        parent.AddTranslateOp().Set(Gf.Vec3d(1.,2.,3.))
        template,relative=g.full._template(self.donor/'manifest.json')
        records=[];variants=[];census=[]
        for index in range(144):
            path='/World/PackPlants/Slot_%03d'%index
            prim=UsdGeom.Xform.Define(self.stage,path);prim.AddTranslateOp().Set(Gf.Vec3d(index*.3,0.,0.))
            prim.AddRotateZOp().Set(float(index))
            prim.GetPrim().GetReferences().AddReference(template.GetRootLayer().identifier,'/Plant')
            paths={cid:path+p[len('/Plant'):] for cid,p in relative.items()}
            records.append(dict(plant_root=path,manifest_path=str(self.donor/'manifest.json'),component_paths=paths))
            variants.append(dict(plant_root=path,variant_id='original_%03d'%index,source_plant_id=self.donor.name,
                split_group=self.donor.name,source_geometry_modified=False,added_components={},added_component_paths={}))
            census.append(dict(plant_root=path,variant_id=variants[-1]['variant_id'],source_family=self.donor.name,
                source_plant_id=self.donor.name,source_split='train',source_geometry_modified=False,
                authenticated_component_plant=True,component_count=3,manifest_path=str(self.donor/'manifest.json'),
                manifest_sha256=source_pins[str(self.donor/'manifest.json')],plant_to_world_usd_row_vectors=g._world(self.stage,path),
                original_slot_mapping=dict(plant_root=path),active_after_policy=True))
        self.anchor=dict(source_family=self.donor.name,split='train',source_collection_plan=str(self.plan_path),
            original_variant=deepcopy(variants[5]))
        receipt=dict(schema=g.full.CENSUS_SCHEMA,policy=g.full.policy('train'),complete_active_plant_anatomy=True,
            source_assets_modified=False,all_plant_roots=census,source_collection_plan=self.plan_pin,
            active_counts=dict(component_plants=144,backdrop_instances=0,components=432))
        receipt['deterministic_census_sha256']=g.fingerprint(receipt)
        self.population=dict(records=records,variants=variants,counts=dict(component_plants=144,backdrop_instances=0,components=432),
            template_stages={'donor':(template,relative)},scene_policy_evidence=receipt,
            original_population_bindings={**source_pins,str(self.plan_path):self.plan_pin['sha256']})
        self.before=self.stage.GetSessionLayer().ExportToString()

    def tearDown(self):
        self.stage=None;self.population=None;self.tmp.cleanup()

    def save_q(self):
        write(self.qp,self.q);self.qpin=pin(self.qp)

    def run_adapter(self, **overrides):
        kwargs=dict(qualification_pin=self.qpin,donor_plan_pin=self.plan_pin,anchor=self.anchor)
        kwargs.update(overrides)
        return g.replace_foreground(self.stage,self.population,**kwargs)

    def assert_unchanged(self):
        self.assertEqual(self.before,self.stage.GetSessionLayer().ExportToString())

    def test_only_foreground_changes_with_real_144_usd_slots(self):
        frozen=deepcopy({k:v for k,v in self.population.items() if k!='template_stages'})
        result=self.run_adapter();primary=self.anchor['original_variant']['plant_root']
        for key in ('records','variants'):
            self.assertEqual([r for r in result[key] if r['plant_root']!=primary],
                             [r for r in frozen[key] if r['plant_root']!=primary])
        old=frozen['scene_policy_evidence']['all_plant_roots']
        new=result['scene_policy_evidence']['all_plant_roots']
        self.assertEqual([r for r in old if r['plant_root']!=primary],[r for r in new if r['plant_root']!=primary])
        for entry in old:
            np.testing.assert_allclose(g._world(self.stage,entry['plant_root']),entry['plant_to_world_usd_row_vectors'],rtol=0,atol=1e-12)
        self.assertEqual(len(self.stage.GetPrimAtPath('/World/PackPlants').GetChildren()),144)
        self.assertEqual(result['counts'],self.population['counts'])
        self.assertEqual(result['schema'],g.SCHEMA)
        self.assertNotEqual(result['scene_policy_evidence']['schema'],g.full.CENSUS_SCHEMA)
        self.assertEqual(result['foreground_identity']['source_family'],'seed53_full')
        self.assertEqual(result['generated_report']['plant_id'],self.generated.name)
        self.assertEqual(result['generated_component_lineage']['SubStem_0']['source_target_id'],'seed53_full/SubStem_0')
        self.assertEqual(result['generated_component_lineage']['SubStem_0']['target_id'],self.generated.name+'/SubStem_0')
        self.assertTrue(result['source_geometry_modified']);self.assertTrue(result['source_assets_unchanged'])
        self.assertFalse(result['capsules_are_certified_mesh_geometry']);self.assertFalse(result['training_approved'])
        self.assertFalse(result['native_capture_validated']);self.assertEqual(result['accepted_training_increment'],0)
        g.verify_bindings(result['source_bindings'])
        self.assertEqual(self.population['scene_policy_evidence'],frozen['scene_policy_evidence'])

    def test_split_and_family_rejection_before_mutation(self):
        for key,value in [('source_split','validation'),('source_family','seed99_full'),('split_group','seed99_full')]:
            original=self.q[key];self.q[key]=value;self.save_q()
            with self.assertRaises(ValueError):self.run_adapter()
            self.assert_unchanged();self.q[key]=original

    def test_approval_and_radius_escalation_rejected(self):
        for key in ('native_capture_validated','training_approved','capsules_are_certified_mesh_geometry','morphology_diversity_approved'):
            self.q[key]=True;self.save_q()
            with self.assertRaises(ValueError):self.run_adapter()
            self.assert_unchanged();self.q[key]=False

    def test_pins_and_supplied_report_are_authenticated(self):
        report=g.audit_manifest(self.generated/'manifest.json')
        report['components']['SubStem_0']['attachment_plant_m']=[99.,0.,0.]
        with self.assertRaisesRegex(ValueError,'report differs'):self.run_adapter(generated_report=report)
        with self.assertRaisesRegex(ValueError,'Changed source'):
            self.run_adapter(qualification_pin=dict(path=str(self.qp),sha256='0'*64))
        self.assert_unchanged()

    def test_graph_change_rejected_even_with_new_asset_pin(self):
        path=self.generated/'manifest.json';manifest=g.read_json(path)
        manifest['components'][2]['parent']='MainStem_0';write(path,manifest)
        self.q['output_hashes']['manifest.json']=g.sha256(path);self.save_q()
        with self.assertRaisesRegex(ValueError,'graph changed'):self.run_adapter()
        self.assert_unchanged()

    def test_background_transform_and_missing_slot_rejected(self):
        path=self.population['records'][7]['plant_root']
        UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetOrderedXformOps()[0].Set(Gf.Vec3d(99.,0.,0.))
        changed=self.stage.GetSessionLayer().ExportToString()
        with self.assertRaisesRegex(ValueError,'world transform differs'):self.run_adapter()
        self.assertEqual(changed,self.stage.GetSessionLayer().ExportToString())
        self.stage.RemovePrim(path)
        with self.assertRaisesRegex(ValueError,'144'):self.run_adapter()

    def test_background_lineage_rejected_with_recomputed_census(self):
        receipt=self.population['scene_policy_evidence'];receipt['all_plant_roots'][7]['source_split']='test'
        receipt['deterministic_census_sha256']=g.fingerprint({k:v for k,v in receipt.items() if k!='deterministic_census_sha256'})
        with self.assertRaisesRegex(ValueError,'background lineage'):self.run_adapter()
        self.assert_unchanged()

    def test_wrong_instance_component_path_rejected(self):
        self.population['records'][7]['component_paths']['SubStem_0']=self.population['records'][8]['component_paths']['SubStem_0']
        with self.assertRaisesRegex(ValueError,'another instance'):self.run_adapter()
        self.assert_unchanged()

    def test_second_replacement_requires_fresh_original_scene(self):
        result=self.run_adapter();after=self.stage.GetSessionLayer().ExportToString()
        with self.assertRaisesRegex(ValueError,'sequential replacement'):self.run_adapter()
        self.assertEqual(after,self.stage.GetSessionLayer().ExportToString())
        self.population=result
        with self.assertRaisesRegex(ValueError,'Original full144'):self.run_adapter()

    def test_failed_postcheck_rolls_back_anonymous_foreground(self):
        with patch.object(g,'verify_bindings',side_effect=ValueError('closing source pin changed')):
            with self.assertRaisesRegex(ValueError,'closing source pin'):self.run_adapter()
        self.assert_unchanged()
        self.assertEqual(self.run_adapter()['unchanged_background_count'],143)

    def test_unbound_generated_reference_rejected_before_replacement(self):
        external=self.base/'external.usda';component_asset(external,.5)
        path=self.generated/'Leaf_0.usda';stage=Usd.Stage.Open(str(path))
        stage.GetDefaultPrim().GetReferences().AddReference(str(external));stage.GetRootLayer().Save();stage=None
        self.q['output_hashes']['Leaf_0.usda']=g.sha256(path);self.save_q()
        with self.assertRaisesRegex(ValueError,'Unbound generated USD dependency'):self.run_adapter()
        self.assert_unchanged()


if __name__=='__main__':unittest.main(verbosity=2)
