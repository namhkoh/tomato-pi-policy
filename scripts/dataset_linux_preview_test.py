"""Portable localization guard tests: no Isaac/native or local dataset fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dataset_linux_preview as m


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.package=Path(self.temp.name)/'package';self.package.mkdir()
        self.old='D:\\original\\tomato_greenhouse_pack'
        self.asset=self.package/'plants'/'a.usdc';self.asset.parent.mkdir();self.asset.write_bytes(b'unchanged geometry')
        self.texture=self.package/'plants'/'texture.png';self.texture.write_bytes(b'unchanged texture')
        self.plan=dict(schema_version='greenhouse.grounding_collection_plan.v1',package=self.old,
            source_bindings_sha256={self.old+'\\plants\\a.usdc':m.sha(self.asset)},
            jobs=[dict(plant_family='seed43_full',source_manifest_path=self.old+'\\plants\\manifest.json',targets=[dict(id='SubStem_42',point=[1,2,3])])],
            family_assignments={'seed43_full':'train'},configuration={'seed':0},cut_rule={'nominal':.01},
            cut_rule_sha256='a'*64,selection_audit=[],split_scope='frozen',camera_requirement='848x408',training_dataset_approved=False)
        self.index=dict(files=[dict(source_path=self.old+'\\plants\\'+p.name,sha256=m.sha(p),bytes=p.stat().st_size) for p in (self.asset,self.texture)])

    def rebuild(self):
        new=deepcopy(self.plan);new['package']=str(self.package.resolve())
        new['source_bindings_sha256']={str(self.asset.resolve()):m.sha(self.asset)}
        new['jobs'][0]['source_manifest_path']=str((self.package/'plants/manifest.json').resolve())
        return new

    def test_foreign_windows_path_is_relocated_componentwise(self):
        self.assertEqual(m.destination(self.old+'\\plants\\a.usdc',self.old,self.package),self.asset)

    def test_prefix_lookalike_is_not_inside_source(self):
        with self.assertRaises(ValueError):m.relative_source(self.old+'_other/a.usdc',self.old)

    def test_parent_traversal_refused(self):
        with self.assertRaises(ValueError):m.relative_source(self.old+'/plants/../a.usdc',self.old)

    def test_absolute_linux_source_also_supported(self):
        self.assertEqual(m.relative_source('/assets/pack/plants/a.usdc','/assets/pack'),('plants','a.usdc'))

    def test_exact_package_and_texture_verified(self):
        bindings,archive=m.verify_package(self.plan,self.index,self.package)
        self.assertEqual(bindings,{str(self.asset):m.sha(self.asset)})
        self.assertIn(str(self.texture),archive)

    def test_changed_texture_fails_even_if_not_in_old_plan(self):
        self.texture.write_bytes(b'changed texture')
        with self.assertRaises(ValueError):m.verify_package(self.plan,self.index,self.package)

    def test_archive_disagreement_with_historical_geometry_refused(self):
        self.plan['source_bindings_sha256'][self.old+'\\plants\\a.usdc']='f'*64
        with self.assertRaises(ValueError):m.verify_package(self.plan,self.index,self.package)

    def test_missing_archive_binding_refused(self):
        self.index['files']=self.index['files'][1:]
        with self.assertRaises(ValueError):m.verify_package(self.plan,self.index,self.package)

    def test_duplicate_archive_destination_refused(self):
        self.index['files'].append(deepcopy(self.index['files'][0]))
        with self.assertRaises(ValueError):m.verify_package(self.plan,self.index,self.package)

    def test_rebuilt_schedule_accepts_only_path_changes(self):
        new=self.rebuild();m.compare_rebuilt(self.plan,new,self.package,new['source_bindings_sha256'])
        self.assertEqual(self.plan['package'],self.old)

    def test_changed_split_target_or_geometry_refused(self):
        for field in ('split','geometry','target'):
            new=self.rebuild()
            if field=='split':new['family_assignments']['seed43_full']='test'
            elif field=='geometry':new['jobs'][0]['targets'][0]['point'][0]+=1
            else:new['jobs'][0]['targets'][0]['id']='SubStem_43'
            with self.subTest(field=field),self.assertRaises(ValueError):m.compare_rebuilt(self.plan,new,self.package,new['source_bindings_sha256'])

    def test_foreign_input_hash_refused_before_parsing(self):
        p=Path(self.temp.name)/'plan.json';p.write_text('{}')
        with self.assertRaises(ValueError):m.bound(p,'0'*64)

    def test_output_cannot_touch_sources_or_exist(self):
        for output in (self.package,self.package/'new',self.package.parent):
            with self.subTest(output=output),self.assertRaises(ValueError):m.prepare(None,None,None,None,None,self.package,output,None)


if __name__=='__main__':unittest.main()
