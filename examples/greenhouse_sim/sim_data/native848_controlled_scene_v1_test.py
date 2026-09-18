"""Controlled dispatch/144-slot mutation tests; catalogue geometry replay is mocked."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from . import native848_generated_morphology_scene_v1 as g
from .native848_generated_morphology_scene_v1_test import GeneratedForegroundTests,write,pin
from . import procedural_petiole_controlled_catalogue_v3 as catalogue

class ControlledForegroundTests(unittest.TestCase):
    setUp_fixture=GeneratedForegroundTests.setUp
    tearDown=GeneratedForegroundTests.tearDown
    save_q=GeneratedForegroundTests.save_q
    assert_unchanged=GeneratedForegroundTests.assert_unchanged

    def setUp(self):
        self.setUp_fixture()
        self.anchor['source_bindings']={str(self.plan_path.resolve()):self.plan_pin['sha256']}
        self.q.update(version=g.CONTROLLED_VERSION,split='train',source_plan_path=str(self.plan_path),
            source_plan_sha256=self.plan_pin['sha256'],source_manifest_path=str(self.donor/'manifest.json'),
            recipes=[dict(component_id='SubStem_0')],
            targets=[dict(component_id='SubStem_0',source_target_id='seed53_full/SubStem_0')],
            code_sha256={'procedural_petiole_controlled_catalogue_v3.py':g.sha256(catalogue.__file__)},
            proximal_mesh_identity=dict(protected_source_arc_m=[0.,.03],complete_intersecting_faces_exact=True,
                source_mesh_surface_equality_continuous_in_protected_halfspace=True,
                distal_faces_remain_outside_protected_halfspace=True,unchanged_numeric_radius_definition=True))
        self.save_q()

    def replay(self):
        return dict(schema_version='greenhouse.proximal_preserved_control_inspection_catalogue.v3',
            qualification_sha256=self.qpin['sha256'],source_plan_sha256=self.plan_pin['sha256'],
            source_family='seed53_full',split_group='seed53_full',variant_id=self.generated.name,
            training_eligible=False,source_cap_reset=False,rows=[dict(component_id='SubStem_0')],rejected=[],
            proximal_mesh_identity=deepcopy(self.q['proximal_mesh_identity']),report=g.audit_manifest(self.generated/'manifest.json'))

    def run_adapter(self,**kwargs):
        args=dict(qualification_pin=self.qpin,donor_plan_pin=self.plan_pin,anchor=self.anchor);args.update(kwargs)
        with patch.object(catalogue,'load_for_inspection',return_value=self.replay()) as replay:
            result=g.replace_controlled_foreground(self.stage,self.population,**args)
            replay.assert_called_once_with(self.generated,self.plan_path)
            return result

    def test_controlled_preserves143_and_has_truthful_radius_contract(self):
        before=deepcopy({k:v for k,v in self.population.items() if k!='template_stages'})
        result=self.run_adapter();primary=self.anchor['original_variant']['plant_root']
        for field in ('records','variants'):
            self.assertEqual([r for r in result[field] if r['plant_root']!=primary],
                             [r for r in before[field] if r['plant_root']!=primary])
        census=result['scene_policy_evidence']
        self.assertEqual([r for r in census['all_plant_roots'] if r['plant_root']!=primary],
            [r for r in before['scene_policy_evidence']['all_plant_roots'] if r['plant_root']!=primary])
        self.assertEqual(census['schema'],g.CONTROLLED_CENSUS_SCHEMA)
        self.assertEqual(census['capsule_radius_policy'],g.CONTROLLED_RADIUS_POLICY)
        self.assertEqual(census['unchanged_background_count'],143)
        self.assertFalse(census['current_9mm_geometry_qualified'])
        self.assertFalse(census['physical_9mm_evidence_authenticated_by_scene'])
        self.assertEqual(result['foreground_identity']['geometry_source_id'],self.generated.name)
        self.assertEqual(result['foreground_identity']['source_family'],'seed53_full')
        self.assertEqual(census['deterministic_census_sha256'],g.fingerprint({k:v for k,v in census.items() if k!='deterministic_census_sha256'}))
        self.assertEqual(len(self.stage.GetPrimAtPath('/World/PackPlants').GetChildren()),144)

    def test_v2_rejected_before_catalogue_or_mutation(self):
        self.q['version']='source_frame_controlled_rigid_leaf_static.v2';self.save_q()
        with patch.object(catalogue,'load_for_inspection') as replay:
            with self.assertRaisesRegex(ValueError,'controlled v3'):
                g.replace_controlled_foreground(self.stage,self.population,qualification_pin=self.qpin,
                    donor_plan_pin=self.plan_pin,anchor=self.anchor)
            replay.assert_not_called()
        self.assert_unchanged()

    def test_insufficient_proximal_proof_rejected(self):
        self.q['proximal_mesh_identity']['complete_intersecting_faces_exact']=False;self.save_q()
        with self.assertRaisesRegex(ValueError,'proximal30mm'):self.run_adapter()
        self.assert_unchanged()

    def test_wrong_split_and_supplied_report_rejected(self):
        self.q['split']='test';self.save_q()
        with self.assertRaisesRegex(ValueError,'TRAIN lineage'):self.run_adapter()
        self.q['split']='train';self.save_q()
        report=self.replay()['report'];report['components']['SubStem_0']['attachment_plant_m']=[99.,0.,0.]
        with self.assertRaisesRegex(ValueError,'Supplied controlled report'):self.run_adapter(generated_report=report)
        self.assert_unchanged()

    def test_changed_component_pin_rejected(self):
        self.q['output_hashes']['SubStem_0.usda']='0'*64;self.save_q()
        with self.assertRaisesRegex(ValueError,'Changed source'):self.run_adapter()
        self.assert_unchanged()

if __name__=='__main__':unittest.main(verbosity=2)
