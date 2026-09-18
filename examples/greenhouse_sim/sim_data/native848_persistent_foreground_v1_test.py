"""Real USD transaction tests; synthetic asset authentication, no renderer needed."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pxr import Gf, UsdGeom

from . import native848_persistent_foreground_v1 as p
from . import native848_generated_morphology_scene_v1 as old
from . import native848_generated_morphology_scene_v1_test as fixture
from .native848_generated_morphology_scene_v1_test import component_asset, pin, write


class PersistentForegroundTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.GeneratedForegroundTests()
        self.f.setUp()
        self.f.anchor['source_bindings'] = {str(self.f.plan_path):self.f.plan_pin['sha256']}
        self.f.q.update(version=p.generated.CONTROLLED_VERSION, split='train')
        self.f.save_q()
        self.stage = self.f.stage
        UsdGeom.Xform.Define(self.stage, '/World/Robot').AddTranslateOp().Set(Gf.Vec3d(1,2,3))
        UsdGeom.Camera.Define(self.stage, '/World/Robot/Camera').GetFocalLengthAttr().Set(30)
        UsdGeom.Xform.Define(self.stage, '/Render/Product').GetPrim().CreateAttribute(
            'test:value', __import__('pxr.Sdf', fromlist=['ValueTypeNames']).ValueTypeNames.Int).Set(7)
        self.f.population.update(stage=self.stage,robot=dict(root='/World/Robot'))
        profile=self.f.base/'profile.json';write(profile,dict(schema='test.profile',profile='unchanged'))
        self.profile=pin(profile)
        self.a=self.candidate(self.f.qpin)
        directory=self.f.base/'seed53_full_morph_fixture_B';directory.mkdir()
        for cid in ('MainStem_0','SubStem_0','Leaf_0'):
            component_asset(directory/(cid+'.usda'),.06)
        manifest=deepcopy(old.read_json(self.f.generated/'manifest.json'))
        manifest['variant_id']=directory.name;write(directory/'manifest.json',manifest)
        q=deepcopy(self.f.q);q['variant_id']=directory.name;q['generated_geometry_hash']='different_fixture_B'
        q['output_hashes']={f.name:old.sha256(f) for f in directory.iterdir()}
        write(directory/'qualification.json',q);self.b=self.candidate(pin(directory/'qualification.json'))
        # Exercise the frozen actual USD mutation. Synthetic fixtures do not claim
        # v4 physical qualification; the separate real-asset rehearsal covers it.
        self.auth=patch.object(p.generated,'authenticate_controlled',side_effect=old.authenticate_morphology)
        self.auth.start()
        self.session=p.snapshot_original(self.f.population,self.f.anchor,profile_pin=self.profile)

    def candidate(self,q):
        return dict(qualification=q,source_plan=self.f.plan_pin,source_family='seed53_full',profile=self.profile)

    def tearDown(self):
        self.auth.stop();self.session=None;self.stage=None;self.f.tearDown()

    def text(self):return self.f.stage.GetSessionLayer().ExportToString()

    def test_real_usd_a_b_a_and_backgrounds_remain_exact(self):
        first=self.session.replace(self.a);a=self.text()
        second=self.session.replace(self.b)
        self.assertNotEqual(first['foreground_identity']['morphology_id'],second['foreground_identity']['morphology_id'])
        third=self.session.replace(self.a)
        self.assertEqual(third['generated_report'],first['generated_report'])
        self.assertEqual(third['scene_policy_evidence'],first['scene_policy_evidence'])
        self.assertEqual(len(self.stage.GetPrimAtPath('/World/PackPlants').GetChildren()),144)
        self.assertEqual(self.session.sequence,3)
        for value in (first,second,third):
            self.assertEqual(value['persistent_swap_evidence']['unchanged_background_count'],143)
            self.assertFalse(value['training_approved'])
            self.assertFalse(value['native_capture_validated'])
        self.assertEqual(self.session.verify_backgrounds()["unchanged_background_count"],143)
        self.assertGreater(len(self.session._keepalive),len(self.session._baseline_layers))

    def test_robot_camera_product_edits_between_calls_survive_swap(self):
        self.session.replace(self.a)
        robot=UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Robot'))
        robot.GetOrderedXformOps()[0].Set(Gf.Vec3d(9,8,7))
        camera=UsdGeom.Camera(self.stage.GetPrimAtPath('/World/Robot/Camera'))
        camera.GetFocalLengthAttr().Set(41)
        product=self.stage.GetPrimAtPath('/Render/Product').GetAttribute('test:value');product.Set(21)
        before=old._layer_hash(self.stage.GetSessionLayer(),self.session.primary)
        self.session.replace(self.b)
        self.assertEqual(old._layer_hash(self.stage.GetSessionLayer(),self.session.primary),before)
        self.assertEqual(camera.GetFocalLengthAttr().Get(),41)
        self.assertEqual(product.Get(),21)

    def test_other_donor_split_slot_and_scene_rejected_before_mutation(self):
        self.session.replace(self.a)
        for change in ('family','split','slot','scene'):
            anchor=deepcopy(self.f.anchor)
            if change=='family':anchor['source_family']='seed7_full'
            elif change=='split':anchor['split']='test'
            elif change=='slot':anchor['original_variant']['plant_root']='/World/PackPlants/Slot_006'
            else:anchor['source_collection_plan']=str(self.f.base/'other.json')
            before=self.text()
            with self.assertRaises(ValueError):self.session.replace(self.b,anchor=anchor)
            self.assertEqual(self.text(),before)

    def test_changed_profile_and_assignment_rejected(self):
        other=self.f.base/'other_profile.json';write(other,dict(profile='other'))
        candidate=deepcopy(self.b);candidate['profile']=pin(other)
        before=self.text()
        with self.assertRaisesRegex(ValueError,'profile'):self.session.replace(candidate)
        plan=deepcopy(self.f.plan);plan['family_assignments']['seed53_full']='test'
        path=self.f.base/'other_plan.json';write(path,plan)
        candidate=deepcopy(self.b);candidate['source_plan']=pin(path)
        with self.assertRaisesRegex(ValueError,'assignments'):self.session.replace(candidate)
        self.assertEqual(self.text(),before)

    def test_background_transform_edit_cannot_be_erased_by_restore(self):
        path=self.f.population['records'][9]['plant_root']
        UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetOrderedXformOps()[0].Set(Gf.Vec3d(9,9,9))
        before=self.text()
        with self.assertRaisesRegex(ValueError,'Protected background'):self.session.replace(self.a)
        self.assertEqual(self.text(),before)

    def test_removed_background_component_is_rejected(self):
        path=self.f.population['records'][9]['component_paths']['Leaf_0']
        self.stage.GetPrimAtPath(path).SetActive(False)
        before=self.text()
        with self.assertRaises(ValueError):self.session.replace(self.a)
        self.assertEqual(self.text(),before)

    def test_current_foreground_out_of_band_edit_rejected(self):
        self.session.replace(self.a)
        self.stage.GetPrimAtPath(self.session.primary).SetCustomDataByKey('unexpected',True)
        before=self.text()
        with self.assertRaisesRegex(ValueError,'outside the transaction'):self.session.replace(self.b)
        self.assertEqual(self.text(),before)

    def test_generated_anonymous_template_edit_rejected(self):
        result=self.session.replace(self.a)
        template=result['template_stages'][result['foreground_identity']['morphology_id']][0]
        template.GetDefaultPrim().SetCustomDataByKey('unexpected',True)
        before=self.text()
        with self.assertRaisesRegex(ValueError,'template changed'):self.session.replace(self.b)
        self.assertEqual(self.text(),before)

    def test_protected_source_file_edit_rejected_before_restore(self):
        self.session.replace(self.a)
        path=self.f.donor/'Leaf_0.usda';path.write_text(path.read_text()+'\n# changed\n')
        before=self.text()
        with self.assertRaisesRegex(ValueError,'Stale audit/review'):self.session.replace(self.b)
        self.assertEqual(self.text(),before)

    def test_failed_adapter_restores_previous_foreground_only_and_blocks_reuse(self):
        self.session.replace(self.a);before=self.text()
        with patch.object(p.generated,'replace_controlled_foreground',side_effect=ValueError('qualification failed')):
            with self.assertRaisesRegex(ValueError,'qualification failed'):self.session.replace(self.b)
        self.assertEqual(self.text(),before)
        with self.assertRaisesRegex(ValueError,'Failed foreground context'):self.session.replace(self.a)


    def test_diagnostic_text_identifies_metadata_change_without_accepting_it(self):
        directory=self.f.base/'protected_debug'
        self.session=p.snapshot_original(self.f.population,self.f.anchor,profile_pin=self.profile,diagnostic_dir=directory)
        baseline={key:path.read_bytes() for key,path in self.session._diagnostic_baselines.items()}
        self.stage.GetRootLayer().comment='simulated renderer setup after snapshot'
        with self.assertRaisesRegex(ValueError,'Protected background'):
            self.session.replace(self.a)
        receipt=json.loads((directory/'protected_changes.json').read_text())
        self.assertTrue(receipt['protected_guard_unchanged'])
        self.assertTrue(any('simulated renderer setup' in Path(r['diff']).read_text() for r in receipt['changed_layers']))
        self.assertEqual(self.session.sequence,0)
        for key,path in self.session._diagnostic_baselines.items():
            self.assertEqual(path.read_bytes(),baseline[key])
        self.assertEqual(self.stage.GetRootLayer().comment,'simulated renderer setup after snapshot')

    def test_diagnostic_baseline_after_setup_keeps_guard_and_allows_authorized_edits(self):
        self.stage.GetRootLayer().comment='simulated renderer setup before snapshot'
        directory=self.f.base/'protected_debug_after_setup'
        self.session=p.snapshot_original(self.f.population,self.f.anchor,profile_pin=self.profile,diagnostic_dir=directory)
        self.stage.GetPrimAtPath('/Render/Product').GetAttribute('test:value').Set(31)
        self.session.replace(self.a)
        self.assertEqual(self.session.verify_backgrounds()['unchanged_background_count'],143)
        self.assertFalse((directory/'protected_changes.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
