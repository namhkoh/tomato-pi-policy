"""Current baseline owner pin and real SnapshotPath regression; no native/export."""
import ast,inspect,json,tempfile,unittest
from pathlib import Path
from . import native848_persistent_reviewed_export_v2 as export
from . import native848_persistent_reviewed_export_v1 as previous
from . import native848_persistent_reviewed_export_v1_test as oldtests


class AuthorityTests(oldtests.AuthorityTests):
    def check(self):return export.validate_persistent_authority(self.value,self.authority,self.capture,self.delegation)


class BackgroundTests(oldtests.BackgroundTests):
    def check(self):export.validate_background_review(self.review,self.record,self.obspin,self.obs,self.context,self.score,self.amb)


class DelegationTests(oldtests.DelegationTests):
    def setUp(self):
        super().setUp();self.review['reviewer']='assistant_native_recovery2';self.d['reviewer']='assistant_native_recovery2'
    def check(self):export.validate_delegation(self.d,self.ep,self.selected,self.rp,self.review,self.auth)


class FrozenParityTests(unittest.TestCase):
    def test_only_snapshot_coercion_changes_materialize(self):
        old=inspect.getsource(previous.materialize)
        expected=old.replace("path.is_relative_to(within)","Path(path).is_relative_to(Path(within))")
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(export.materialize))),ast.dump(ast.parse(expected)))

    def test_authority_background_and_scientific_functions_unchanged(self):
        for name in ('validate_persistent_authority','validate_background_review'):
            self.assertEqual(inspect.getsource(getattr(previous,name)),inspect.getsource(getattr(export,name)))
        for name in ('validate_partition','validate_review','validate_qualification','validate_authorization','load_model_inputs'):
            self.assertIs(getattr(export,name),getattr(export.frozen,name))
        self.assertIs(export.validate_delegation,export.delegated.validate_delegation)

    def test_all_implementation_pins_match_current_frozen_sources(self):
        for path,sha in export.PINS.items():self.assertEqual(export.base.digest(path),sha,path)

    def test_only_the_producer_pin_differs_from_previous(self):
        changes={path for path in export.PINS if export.PINS[path]!=previous.PINS[path]}
        self.assertEqual(changes,{export.producer.__file__})


class SnapshotPathTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'data/sim_data/diagnostics');self.root=Path(self.directory.name)
        self.file=self.root/'receipt.json';self.file.write_text('{"returncode":0}')
        self.spec=export.pin(self.file);self.session=export.cache.Session()
        tree=ast.parse(inspect.getsource(export.materialize))
        reader=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='read')
        namespace=dict(session=self.session,Path=Path,require=export.require,json=json)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[reader],type_ignores=[])),'<actual exporter reader>','exec'),namespace)
        self.read=namespace['read']
    def tearDown(self):self.session.close();self.directory.cleanup()
    def test_real_authenticated_snapshot_reads_within_scope(self):
        self.assertEqual(type(self.session.read_pin(self.spec)).__name__,'SnapshotPath')
        self.assertEqual(self.read(self.spec,within=self.root),{'returncode':0})
    def test_outside_scope_still_rejected(self):
        with self.assertRaises(ValueError):self.read(self.spec,within=self.root/'other')


if __name__=='__main__':unittest.main(verbosity=2)
