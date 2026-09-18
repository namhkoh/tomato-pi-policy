"""Candidate-only experimental export: authority mutations and frozen payload parity."""
import ast,inspect,unittest
from copy import deepcopy
from pathlib import Path
from . import native848_persistent_experimental_export_v1 as export
from . import native848_persistent_reviewed_export_v1 as baseline
from . import native848_persistent_reviewed_export_v1_test as oldtests


class AuthorityTests(oldtests.AuthorityTests):
    def setUp(self):
        super().setUp()
        actual=dict(schema=export.producer.diagnostic.ACTUAL_SCHEMA,warmup_steps=[8],request_subframes=8,
            shortened_warmup_qualified=False,experimental_profile=True,training_approved=False,accepted_training_increment=0)
        purpose=dict(capture_purpose=export.producer.diagnostic.CANDIDATE,paired_control_reference=None,export_prohibited=False)
        self.authority['public'].update(experimental_profile=True,actual_experimental_profile=actual,**purpose)
        self.authority['segment_result'].update(actual_experimental_profile=actual,**purpose)
        self.authority['scheduled']=dict(purpose=export.producer.diagnostic.CANDIDATE,paired_control_reference=None,warmup_request_count=1)
        self.value.update(schema=export.EVALUATION_SCHEMA,experimental_profile=True,production_profile_qualification=False,**purpose)
        self.value['persistent_batch_authority']=deepcopy(self.authority['public'])
        self.delegation['persistent_batch_authority']=deepcopy(self.authority['public'])
    def check(self):return export.validate_persistent_authority(self.value,self.authority,self.capture,self.delegation)
    def test_honest_control_still_rejected(self):
        for item in (self.value,self.authority['public'],self.authority['segment_result']):
            item.update(capture_purpose=export.producer.diagnostic.CONTROL,export_prohibited=True,
                paired_control_reference=dict(segment_id='s000',sample_id='a'))
        self.authority['scheduled'].update(purpose=export.producer.diagnostic.CONTROL,paired_control_reference=dict(segment_id='s000',sample_id='a'))
        self.value['persistent_batch_authority']=deepcopy(self.authority['public']);self.delegation['persistent_batch_authority']=deepcopy(self.authority['public'])
        with self.assertRaises(ValueError):self.check()
    def test_hidden_control_marker_rejected(self):
        self.authority['segment_result']['export_prohibited']=True
        with self.assertRaises(ValueError):self.check()
    def test_false_broad_profile_qualification_rejected(self):
        self.value['production_profile_qualification']=True
        with self.assertRaises(ValueError):self.check()
    def test_faked_warmup_count_rejected(self):
        self.authority['public']['actual_experimental_profile']['warmup_steps']=[8]*7
        with self.assertRaises(ValueError):self.check()
    def test_unchanged_eight_production_subframes_required(self):
        self.authority['public']['actual_experimental_profile']['request_subframes']=6
        with self.assertRaises(ValueError):self.check()


class BackgroundTests(oldtests.BackgroundTests):
    def check(self):export.validate_background_review(self.review,self.record,self.obspin,self.obs,self.context,self.score,self.amb)


class DelegationTests(oldtests.DelegationTests):
    def check(self):export.validate_delegation(self.d,self.ep,self.selected,self.rp,self.review,self.auth)


class PayloadTests(unittest.TestCase):
    def test_sensor_loader_and_numeric_validators_unchanged(self):
        for name in ('validate_partition','validate_review','validate_qualification','validate_authorization','load_model_inputs'):
            self.assertIs(getattr(export,name),getattr(export.frozen,name))
        self.assertEqual(export.FRAME_SCHEMA,export.frozen.FRAME_SCHEMA)
        self.assertEqual(export.SCHEMA,export.frozen.SCHEMA)
        self.assertEqual(inspect.getsource(export.validate_background_review),inspect.getsource(baseline.validate_background_review))
    def test_actual_numerical_and_sensor_body_unchanged(self):
        old=inspect.getsource(baseline.materialize);new=inspect.getsource(export.materialize)
        start="    context=read(local(capture/'context.json'))";end='    output.mkdir();rows=[]'
        self.assertEqual(old[old.index(start):old.index(end)],new[new.index(start):new.index(end)])
        start="            with (folder/'rgb.png').open('xb')";end='            row=dict(schema=FRAME_SCHEMA'
        self.assertEqual(old[old.index(start):old.index(end)],new[new.index(start):new.index(end)])
        self.assertIn('load_model_inputs(row,output);rows.append(row)',new)
        self.assertNotIn('evaluate_segment(',new)
        ast.parse(Path(export.__file__).read_text())
    def test_correct_actual_experimental_source_pins(self):
        for module in (export.persistent,export.producer,export.background,export.delegated):
            self.assertEqual(export.base.digest(module.__file__),export.PINS[module.__file__])
    def test_control_validation_precedes_materialization(self):
        source=inspect.getsource(export.materialize)
        self.assertLess(source.index('validate_persistent_authority('),source.index('output.mkdir()'))
        self.assertIn("paired_control_exported=False",source)



class SnapshotPathRegressionTests(unittest.TestCase):
    def setUp(self):
        # Exercise the exact nested exporter reader, with the real completed
        # owner receipt and real Session/SnapshotPath rather than a plain mock.
        import json
        self.trial=Path('C:/Users/USER/tomato-vlm-data-20260917/diagnostics/persistent41_B7C1D1B1_native_20260918_v1')
        complete=json.loads((self.trial/'owner_complete.json').read_text())
        self.spec=complete['owned_exit'];self.session=export.cache.Session()
        module=ast.parse(inspect.getsource(export.materialize));read=next(x for x in module.body[0].body if isinstance(x,ast.FunctionDef) and x.name=='read')
        ns=dict(session=self.session,Path=Path,require=export.require,json=json)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[read],type_ignores=[])),'<actual exporter read>','exec'),ns)
        self.read=ns['read']
    def tearDown(self):self.session.close()
    def test_real_snapshot_within_trial_reads_authenticated_bytes(self):
        import json
        snapshot=self.session.read_pin(self.spec)
        self.assertEqual(type(snapshot).__name__,'SnapshotPath')
        self.assertEqual(self.read(self.spec,within=self.trial),json.loads(snapshot.read_text()))
    def test_real_snapshot_outside_scope_still_rejected(self):
        with self.assertRaises(ValueError):self.read(self.spec,within=self.trial/'capture')

if __name__=='__main__':unittest.main(verbosity=2)
