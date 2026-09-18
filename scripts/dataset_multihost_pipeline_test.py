"""Focused host/recipe/toolchain/dispatch tests; no asset generation or GPU."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import dataset_multihost_pipeline as pipeline

REPO = Path(__file__).resolve().parents[1]
CONTRACT = dict(resolution=[848, 408], nominal_cut_arc_m=.009,
                exactly_one_eligible_petiole=True, query_input=False,
                full_labeled_plant_count=144, minimum_background_plant_fraction=.40)
DISPATCH = {
    'sweep-raw': ('native848_scene_sweep_cached_annotation_v1', 'run', {'mode': 'raw'}),
    'sweep-ordinary': ('native848_scene_sweep_cached_annotation_v1', 'run', {'mode': 'ordinary'}),
    'controlled-generated': ('native848_generated_9mm_v3', 'evaluate_capture', {}),
    'persistent-generated': ('native848_persistent_generated_9mm_v1', 'evaluate_segment', {}),
    'experimental-generated': ('native848_persistent_experimental_9mm_v1', 'evaluate_segment', {}),
}


def save(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def census():
    return dict(dataset_split='train', complete_active_plant_anatomy=True, full_active_catalogue_required=True,
                all_plant_roots=[dict(plant_root='/World/P'+str(i), source_family='seed43_full',
                    source_split='train', component_count=400, active_after_policy=True,
                    authenticated_component_plant=True) for i in range(144)],
                active_counts=dict(component_plants=144, backdrop_instances=0),
                removed_roots=[], extra_plant_asset_roots_outside_packplants=[],
                removed_merged_or_unknown_count=0, removed_cross_split_count=0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.configs = {host: json.loads((REPO/'configs/dataset_capture'/f'{host}.json').read_text())
                        for host in ('local5090', 'thor1', 'thor3')}
        self.source = self.root/'locked.py'
        self.source.write_text('VALUE = 1\n')
        self.lock = self.root/'lock.json'
        self.lock_value = dict(schema='greenhouse.multihost_toolchain_lock.v1',
                              annotation_contract=CONTRACT,
                              files={'locked.py': pipeline.sha(self.source)})
        save(self.lock, self.lock_value)

    def config_path(self, value):
        path = self.root/'host.json'
        save(path, value)
        return path

    def test_host_configs_are_disjoint_and_owned(self):
        owners = []
        for host, config in self.configs.items():
            self.assertEqual(pipeline.host_config(self.config_path(config))['host_id'], host)
            owners += config['allowed_foreground_families']
            wrong = deepcopy(config)
            wrong['allowed_foreground_families'] = ['seed999_full']
            with self.assertRaises(ValueError):
                pipeline.host_config(self.config_path(wrong))
        self.assertEqual(len(owners), len(set(owners)))
        self.assertEqual(len(owners), 16)

    def test_host_seed_namespace_cannot_be_reassigned(self):
        config=deepcopy(self.configs['thor1'])
        config['new_morphology_seed_range']=[3000000,3999999]
        with self.assertRaises(ValueError):pipeline.host_config(self.config_path(config))

    def test_seed_bounds_and_host_ownership(self):
        for host, config in self.configs.items():
            family = config['allowed_foreground_families'][0]
            low, high = config['new_morphology_seed_range']
            for seed in (low, high):
                self.assertEqual(len(pipeline.build_controls(config, family, seed, ['SubStem_41'])), 1)
            for seed in (low-1, high+1, True, float(low)):
                with self.assertRaises(ValueError):
                    pipeline.build_controls(config, family, seed, ['SubStem_41'])
            foreign = next(x for h, c in self.configs.items() if h != host
                           for x in c['allowed_foreground_families'])
            with self.assertRaises(ValueError):
                pipeline.build_controls(config, foreign, low, ['SubStem_41'])

    def test_banned_duplicate_and_unbounded_targets_rejected(self):
        c = self.configs['local5090'];seed=c['new_morphology_seed_range'][0]
        for targets in (['SubStem_38'], ['SubStem_41','SubStem_41'], [],
                        ['SubStem_'+str(i) for i in range(9)]):
            with self.assertRaises(ValueError):
                pipeline.build_controls(c, 'seed41_full', seed, targets)

    def test_recipes_reproducible_and_distinct_controls(self):
        c=self.configs['thor1'];targets=['SubStem_42','SubStem_43']
        first=pipeline.build_controls(c,'seed43_full',2000000,targets)
        self.assertEqual(first,pipeline.build_controls(c,'seed43_full',2000000,targets))
        other=pipeline.build_controls(c,'seed43_full',2000001,targets)
        self.assertNotEqual(first,other)
        self.assertEqual([v['component_id'] for v in first],targets)
        self.assertTrue(all(v['control']['amplitude_m']==.025 for v in first+other))
        self.assertEqual(len({v['control']['direction_angle_degrees'] for v in first}),2)

    def test_changed_shared_native_contract_rejected(self):
        for key,value in [('fully_labeled_plant_count',143),('query_input',True),
                          ('split','validation'),('native_mask_width_minimum_px',7),
                          ('minimum_background_plant_fraction',.1)]:
            config=deepcopy(self.configs['thor1']);config[key]=value
            with self.assertRaises(ValueError):pipeline.host_config(self.config_path(config))

    def test_toolchain_changed_file_and_path_escape_hold(self):
        with patch.object(pipeline,'ROOT',self.root):
            self.assertEqual(pipeline.verify_lock(self.lock)['verified_files'],1)
            self.source.write_text('VALUE = 2\n')
            with self.assertRaises(ValueError):pipeline.verify_lock(self.lock)
            value=deepcopy(self.lock_value);value['files']={'../escape.py':'a'*64};save(self.lock,value)
            with self.assertRaises(ValueError):pipeline.verify_lock(self.lock)

    def test_background_actual_structure_only_not_image_admission(self):
        report=pipeline.check_background(census())
        self.assertEqual(report['plant_slots'],144)
        self.assertTrue(report['actual_image_background_coverage_still_required'])
        self.assertIs(report['training_approved'],False)

    def test_missing_duplicate_inactive_or_cross_split_plant_rejected(self):
        mutations=(lambda c:c['all_plant_roots'].pop(),
                   lambda c:c['all_plant_roots'][0].update(plant_root='/World/P1'),
                   lambda c:c['all_plant_roots'][0].update(active_after_policy=False),
                   lambda c:c['all_plant_roots'][0].update(source_split='test'))
        for mutate in mutations:
            c=census();mutate(c)
            with self.assertRaises(ValueError):pipeline.check_background(c)

    def test_unknown_untracked_backdrop_and_census_split_rejected(self):
        mutations=(lambda c:c['all_plant_roots'][0].update(authenticated_component_plant=False),
                   lambda c:c['active_counts'].update(backdrop_instances=1),
                   lambda c:c.update(removed_merged_or_unknown_count=1),
                   lambda c:c.update(dataset_split='validation'),
                   lambda c:c.update(full_active_catalogue_required=False),
                   lambda c:c.update(removed_cross_split_count=1))
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                c=census();mutate(c)
                with self.assertRaises(ValueError):pipeline.check_background(c)

    def invoke_annotation(self,kind,mutation=None):
        mod_name,func_name,kwargs=DISPATCH[kind]
        module=importlib.import_module('sim_data.'+mod_name)
        real=getattr(module,func_name)
        capture=self.root/'capture';out=self.root/('annotation_'+kind)
        # Check the real callable's signature, then intercept only its execution.
        inspect.signature(real).bind(capture,out,**kwargs)
        def evaluate(*args,**kw):
            out.mkdir();save(out/'result.json',{'frames_evaluated':1})
            if mutation:mutation()
            return {'frames_evaluated':1}
        config=self.config_path(self.configs['thor1'])
        argv=['pipeline','--lock',str(self.lock),'--config',str(config),'annotate',
              '--kind',kind,'--capture',str(capture),'--output',str(out)]
        with patch.object(pipeline,'ROOT',self.root),patch.object(sys,'argv',argv),             patch.object(module,func_name,autospec=True,side_effect=evaluate) as call,             redirect_stdout(io.StringIO()):
            pipeline.main()
        call.assert_called_once_with(capture,out,**kwargs)
        return out

    def test_all_annotation_dispatches_match_actual_frozen_signatures(self):
        for kind in DISPATCH:
            with self.subTest(kind=kind):
                out=self.invoke_annotation(kind)
                receipt=json.loads((out/'shared_toolchain_receipt.json').read_text())
                self.assertEqual(receipt['capture_kind'],kind)
                self.assertIs(receipt['training_approved'],False)
                self.assertEqual(receipt['annotation_result_sha256'],pipeline.sha(out/'result.json'))

    def test_closing_source_mutation_holds_without_receipt(self):
        with self.assertRaises(ValueError):
            self.invoke_annotation('sweep-raw',lambda:self.source.write_text('VALUE = 2\n'))
        self.assertFalse((self.root/'annotation_sweep-raw/shared_toolchain_receipt.json').exists())

    def test_coherent_source_and_lock_mutation_also_holds(self):
        def mutate():
            self.source.write_text('VALUE = 2\n')
            value=deepcopy(self.lock_value);value['files']['locked.py']=pipeline.sha(self.source)
            save(self.lock,value)
        with self.assertRaises(ValueError):self.invoke_annotation('sweep-raw',mutate)
        self.assertFalse((self.root/'annotation_sweep-raw/shared_toolchain_receipt.json').exists())


if __name__=='__main__':
    unittest.main()
