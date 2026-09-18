"""Two-plan provenance boundary, using synthetic144 slots and mocked geometry replay."""
from copy import deepcopy
import unittest
from . import native848_controlled_scene_v1_test as fixtures
from . import native848_generated_morphology_scene_v1 as g

class TwoPlanTests(unittest.TestCase):
    setUp_fixture=fixtures.ControlledForegroundTests.setUp
    tearDown=fixtures.ControlledForegroundTests.tearDown
    setUp_original=fixtures.ControlledForegroundTests.setUp_fixture
    save_q=fixtures.ControlledForegroundTests.save_q
    assert_unchanged=fixtures.ControlledForegroundTests.assert_unchanged
    replay=fixtures.ControlledForegroundTests.replay
    run_adapter=fixtures.ControlledForegroundTests.run_adapter
    def setUp(self):
        # Original fixture setter invokes self.setUp_fixture; avoid recursive aliases.
        self.setUp_fixture=self.setUp_original
        fixtures.ControlledForegroundTests.setUp(self)
        self.scene_path=self.base/'actual_scene_plan.json';self.scene_plan=deepcopy(self.plan)
        self.scene_plan['configuration']={'distinct_capture_provenance':True}
        self.refresh_scene()
        self.anchor_before=deepcopy(self.anchor)
    def refresh_scene(self):
        fixtures.write(self.scene_path,self.scene_plan);self.scene_pin=fixtures.pin(self.scene_path)
        self.anchor['source_collection_plan']=str(self.scene_path)
        self.anchor['source_bindings']={str(self.scene_path):self.scene_pin['sha256']}
        c=self.population['scene_policy_evidence'];c['source_collection_plan']=self.scene_pin
        c['deterministic_census_sha256']=g.fingerprint({k:v for k,v in c.items() if k!='deterministic_census_sha256'})
        self.population['original_population_bindings'][str(self.scene_path)]=self.scene_pin['sha256']
    def test_legitimate_different_plans_preserve_both_provenances(self):
        result=self.run_adapter();c=result['scene_policy_evidence']
        self.assertEqual(c['source_collection_plan'],self.scene_pin)
        self.assertEqual(c['generator_source_collection_plan'],self.plan_pin)
        self.assertEqual(c['unchanged_background_count'],143)
        self.assertEqual(self.anchor,self.anchor_before)
    def test_other_family_split_mismatch_rejected(self):
        self.scene_plan['family_assignments']['other_family']='test';self.refresh_scene()
        with self.assertRaisesRegex(ValueError,'family assignments differ'):self.run_adapter()
        self.assert_unchanged()
    def test_donor_asset_mismatch_rejected(self):
        path=str((self.donor/'SubStem_0.usda').resolve())
        self.scene_plan['source_bindings_sha256'][path]='0'*64;self.refresh_scene()
        with self.assertRaisesRegex(ValueError,'component differs between'):self.run_adapter()
        self.assert_unchanged()
    def test_unbound_scene_plan_rejected(self):
        self.anchor['source_bindings']={}
        with self.assertRaisesRegex(ValueError,'Explicit SHA256'):self.run_adapter()
        self.assert_unchanged()

if __name__=='__main__':unittest.main(verbosity=2)
