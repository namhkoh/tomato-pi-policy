"""Focused typed-dispatch checks; fixtures are not exported or accepted data."""
from copy import deepcopy
from pathlib import Path
import ast, inspect, json, tempfile, unittest
from unittest.mock import patch
from . import native848_delivery_bundle_v1 as prior
from . import native848_delivery_bundle_v2 as bundle

DIAG=Path('D:/research/tomato-pi-policy/data/sim_data/diagnostics/native848_delivery_bundle_v2_CPU_20260918_v1')


def row(i=0):
    return dict(schema=bundle.generated_farmerge.FRAME_SCHEMA,task_id=bundle.original.TASK,
        annotation_epoch=bundle.original.EPOCH,frame_id=f'fixture{i}',split='train',
        source_family='seed47_full',donor_source_family='seed47_full',source_target_id='seed47_full/SubStem_41',
        geometry_source_id='fixture_morphology',morphology_id='fixture_morphology',
        new_independent_source_family=False,source_geometry_modified=True,individual_full_frame_review=True,
        inputs={k:{} for k in bundle.original.INPUT_KEYS},model_instruction=bundle.original.INSTRUCTION,
        answer={'cut_point_uv':[100,100]},decoded_rgb_sha256=f'rgb{i}',
        conservative_camera_signature=f'camera{i}',donor_camera_signature=f'donor_camera{i}')


class AdapterChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=DIAG,prefix='fixture_')
        self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(DIAG.resolve()))

    def tearDown(self):self.temp.cleanup()

    def component(self,*,change_manifest=None,change_row=None,state=None):
        refs={}
        for name in ('render_census','farmerge_recipe','production_profile','source_pose_profile'):
            p=self.root/(name+'.json');p.write_text('{}');refs[name]=bundle.pin(p)
        r=row();r['source_provenance']=deepcopy(refs)
        if change_row:change_row(r)
        index=self.root/'index.jsonl';index.write_text(json.dumps(r)+'\n')
        m=dict(schema=bundle.generated_farmerge.SCHEMA,task_id=bundle.original.TASK,
            annotation_epoch=bundle.original.EPOCH,fully_populated_greenhouse=True,plant_instance_count=144,
            training_started=False,index=bundle.pin(index),unique_images=1,
            implementation_sha256=bundle.LOADER_PINS['generated_farmerge'],
            farmerge_representation_preserves_all_background_shapes_materials=True,
            individual_coarse_component_pixel_visibility_claimed=False,
            original_background_instances=143,generated_foreground_instances=1,
            annotation_recomputed=False,model_inputs_loaded_after_copy=True,
            render_census=refs['render_census'],farmerge_recipe=refs['farmerge_recipe'])
        if change_manifest:change_manifest(m)
        manifest=self.root/'manifest.json';manifest.write_text(json.dumps(m))
        result=self.root/'result.json';result.write_text(json.dumps(dict(schema=bundle.generated_farmerge.SCHEMA,
            state=state or bundle.COMPLETE_STATES['generated_farmerge'],manifest=bundle.pin(manifest),unique_images=1)))
        return dict(component_id='fast_fixture',kind='generated_farmerge',root=str(self.root),
            result=bundle.pin(result),manifest=bundle.pin(manifest),index=bundle.pin(index)),r

    def test_fast_typed_component_keeps_exact_source_row(self):
        spec,r=self.component();root,rows=bundle._component(spec)
        self.assertEqual(rows,[r]);self.assertEqual(root,self.root)
        entry=bundle.entry_for('fast','generated_farmerge',0,r)
        self.assertTrue(entry['generated_geometry']);self.assertFalse(entry['new_independent_source_family'])
        self.assertEqual(entry['source_row_sha256'],bundle.original.canonical_sha(r))

    def test_fast_incomplete_result_rejected(self):
        spec,_=self.component(state='pending_review')
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_fake_component_pixel_visibility_rejected(self):
        spec,_=self.component(change_manifest=lambda m:m.update(individual_coarse_component_pixel_visibility_claimed=True))
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_removed_background_rejected(self):
        spec,_=self.component(change_manifest=lambda m:m.update(original_background_instances=142))
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_wrong_exporter_pin_rejected(self):
        spec,_=self.component(change_manifest=lambda m:m.update(implementation_sha256='0'*64))
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_changed_recipe_bytes_rejected(self):
        spec,_=self.component();(self.root/'farmerge_recipe.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_fast_recipe_link_mismatch_rejected(self):
        spec,_=self.component(change_row=lambda r:r['source_provenance']['farmerge_recipe'].update(sha256='0'*64))
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_fast_validation_split_rejected(self):
        spec,_=self.component(change_row=lambda r:r.update(split='validation'))
        with self.assertRaises(ValueError):bundle._component(spec)

    def test_fast_row_cannot_masquerade_as_ordinary_generated(self):
        with self.assertRaises(ValueError):bundle.entry_for('fast','generated',0,row())

    def test_fast_donor_lineage_cannot_be_reassigned(self):
        r=row();r['donor_source_family']='seed53_full'
        with self.assertRaises(ValueError):bundle.entry_for('fast','generated_farmerge',0,r)

    def test_each_kind_uses_exact_sensor_loader_and_separate_answer(self):
        for kind,loader in bundle.LOADERS.items():
            r=row();r['schema']=loader.FRAME_SCHEMA
            dataset=object.__new__(bundle.DeliveryDataset)
            dataset.entries=[dict(component_id='c',source_row_number=0,answer=deepcopy(r['answer']))]
            dataset.components={'c':(self.root,[r],kind)}
            inputs={'sensors':'fixture'}
            with patch.object(loader,'load_model_inputs',return_value=inputs) as load:
                result=dataset.load(0);load.assert_called_once_with(r,self.root)
            self.assertIs(result['inputs'],inputs);self.assertEqual(result['answer'],r['answer'])
            self.assertNotIn('answer',result['inputs'])

    def test_duplicate_and_split_predicates_unchanged(self):
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(prior.counts))),
                         ast.dump(ast.parse(inspect.getsource(bundle.counts))))
        a=bundle.entry_for('fast','generated_farmerge',0,row())
        for key in ('source_frame_id','decoded_rgb_sha256','geometry_camera_signature'):
            b=bundle.entry_for('fast','generated_farmerge',1,row(1));b[key]=a[key]
            with self.assertRaises(ValueError):bundle.counts([a,b])
        b=bundle.entry_for('fast','generated_farmerge',1,row(1));b['split']='test'
        with self.assertRaises(ValueError):bundle.counts([a,b])

    def test_existing450_index_compatibility_and_real_loaders(self):
        request=json.loads((DIAG.parent/'native848_delivery450_assembly_20260918_v1/request.json').read_text())
        entries=[]
        for spec in request['components']:
            before=bundle.original.digest(spec['index']['path'])
            root,rows=bundle._component(spec);old_root,old_rows=prior._component(spec)
            self.assertEqual(root,old_root);self.assertEqual(rows,old_rows)
            for n,r in enumerate(rows):
                a=prior.entry_for(spec['component_id'],spec['kind'],n,r)
                b=bundle.entry_for(spec['component_id'],spec['kind'],n,r)
                a['schema']=bundle.ENTRY_SCHEMA;self.assertEqual(a,b);entries.append(b)
            self.assertEqual(before,bundle.original.digest(spec['index']['path']))
            loaded=bundle.LOADERS[spec['kind']].load_model_inputs(rows[0],root)
            self.assertNotIn('answer',loaded);self.assertNotIn('target_id',loaded)
            self.assertEqual(loaded['rgb'].shape,(408,848,3))
            if spec['kind']=='generated':
                fixture=deepcopy(rows[0]);fixture['schema']=bundle.generated_farmerge.FRAME_SCHEMA
                self.assertEqual(bundle.generated_farmerge.load_model_inputs(fixture,root)['rgb'].shape,(408,848,3))
        self.assertEqual(len(entries),450);self.assertEqual(bundle.counts(entries),prior.counts(entries))


if __name__=='__main__':unittest.main()
