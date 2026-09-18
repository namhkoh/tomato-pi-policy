"""Focused preview contract checks; no generated assets or rendering."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sim_data.plant_morphology_preview_v1 import (
    QUALIFICATION_SCHEMA, check_receipt, geometry_9mm, load_inputs, remember)
from sim_data.depth_preview import sha256

class PreviewTests(unittest.TestCase):
    def report(self):
        return dict(components=dict(P=dict(type='main_stem'), T=dict(
            type='sub_stem', parent='P', deleafed=False,
            translation_plant_m=[1., 2., 3.], attachment_plant_m=[1., 2., 3.],
            axis_plant=[1., 0., 0.],
            capsules_local_m=[[[0., 0., 0., .003], [.006, 0., 0., .003],
                               [.006, .030, 0., .002]]])))
    def receipt(self):
        return dict(schema=QUALIFICATION_SCHEMA, source_family='seed53_full',
            split_group='seed53_full', original_source_family_cap_group='seed53_full',
            source_split='train', variant_id='new_variant',
            output_hashes={'manifest.json': 'a'*64}, full_component_graph_preserved=True,
            source_deleaf_state_preserved=True, source_assets_unchanged=True,
            training_approved=False, native_capture_validated=False,
            accepted_training_increment=0, new_independent_source_family=False)
    def test_metric_arc_not_scaled_or_euclidean_distance(self):
        result=geometry_9mm(self.report(),'T')
        np.testing.assert_allclose(result['nominal_point_plant_m'],[1.006,2.003,3.],atol=1e-14)
        self.assertEqual(result['nominal_arc_m'],.009)
        self.assertFalse(result['support_is_permissible_cut_interval'])
        self.assertFalse(result['surface_or_native_visibility_verified'])
    def test_ambiguous_chain_and_detached_leaf_rejected(self):
        for mutate in ('second_chain','deleafed'):
            report=self.report()
            if mutate=='second_chain':report['components']['T']['capsules_local_m']*=2
            else:report['components']['T']['deleafed']=True
            with self.assertRaises(ValueError):geometry_9mm(report,'T')
    def test_receipt_typed_diagnostic_lineage(self):
        receipt=self.receipt()
        check_receipt(receipt,'seed53_full',Path('new_variant/manifest.json'),'a'*64)
        for key,value in [('schema','old'),('source_family','renamed'),
                          ('source_split','unspecified'),('training_approved',True),
                          ('full_component_graph_preserved',False),
                          ('new_independent_source_family',True),('accepted_training_increment',1)]:
            changed=deepcopy(receipt);changed[key]=value
            with self.assertRaises(ValueError):
                check_receipt(changed,'seed53_full',Path('new_variant/manifest.json'),'a'*64)
    def test_exactly_two_generated_inputs(self):
        for n in (0,1,3):
            with self.assertRaisesRegex(ValueError,'Exactly two'):
                load_inputs('none','a'*64,['x']*n,['a'*64]*n,['a'*64]*n)
    def test_modified_asset_pin_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'asset';path.write_bytes(b'original')
            expected=sha256(path);pins={};remember(pins,path,expected)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):remember(pins,path,expected)
if __name__=='__main__':unittest.main()
