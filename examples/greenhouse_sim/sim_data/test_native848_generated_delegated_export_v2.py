"""Focused observer-attribution checks; no materialization or approval is issued."""
from pathlib import Path
from copy import deepcopy
import ast,inspect,json,tempfile,unittest
from . import native848_generated_delegated_export_v1 as old
from . import native848_generated_delegated_export_v2 as new

D=Path('D:/research/tomato-pi-policy/data/sim_data/diagnostics')
CHECKS=['all source and artifact hashes','sensor validity and exact native848x408',
 'full144 census and exactly one9mm cutpoint','positive alternative exclusions',
 'visual review accept decisions','donor TRAIN lineage and morphology identity','exact duplicate rejection']


class ObserverChecks(unittest.TestCase):
    def setUp(self):
        self.review_pin=new.pin(D/'native848_multisubtree19_four_actual_review_20260918_v2/actual_review.json')
        self.review=json.loads(new.base.read_pin(self.review_pin).read_text());self.eval=self.review['evaluation']
        ev=json.loads(new.base.read_pin(self.eval).read_text());self.selected=[r['sample_id'] for r in self.review['frames']]
        self.wait=self.review['actual_outer_process_completion']['annotation']['actual_wait_receipt']
        self.delegation=dict(schema=new.DELEGATION_SCHEMA,authorized_by='assistant_root',reviewer=self.review['reviewer'],
            evaluation=self.eval,review=self.review_pin,selected_sample_ids=self.selected,training_launch_authorized=False,
            required_checks_preserved=CHECKS,authorization='CPU validator fixture only; does not authorize an export',
            source_owner_complete=ev['owner_complete'],source_capture=ev['source_capture'],annotation_wait_receipt=self.wait,
            root_observed_completion={'capture':deepcopy(self.review['actual_outer_process_completion']['capture'])})
        self.auth=dict(native_outer_wait=dict(session_id=96407,returncode=0),annotation_outer_wait=dict(session_id=68613,returncode=0),annotation_wait_receipt=self.wait)
        self.temp=tempfile.TemporaryDirectory(dir=D/'native848_generated_delegated_export_v2_CPU_20260918_v1',prefix='fixture_')
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(D.resolve()))

    def tearDown(self):self.temp.cleanup()
    def check(self):new.validate_delegation(self.delegation,self.eval,self.selected,self.review_pin,self.review,self.auth)
    def replace_wait(self,**changes):
        value=json.loads(new.base.read_pin(self.wait).read_text());value.update(changes)
        p=Path(self.temp.name)/'wait.json';p.write_text(json.dumps(value));spec=new.pin(p)
        self.delegation['annotation_wait_receipt']=spec;self.auth['annotation_wait_receipt']=spec
        self.review['actual_outer_process_completion']['annotation']['actual_wait_receipt']=spec

    def test_real_agent_wait_matches_actual19_evaluation(self):self.check()
    def test_missing_wait_rejected(self):
        del self.delegation['annotation_wait_receipt']
        with self.assertRaises(ValueError):self.check()
    def test_different_wait_pin_rejected(self):
        self.auth['annotation_wait_receipt']={'path':'other','sha256':'0'*64}
        with self.assertRaises(ValueError):self.check()
    def test_changed_wait_bytes_rejected(self):
        self.replace_wait();p=Path(self.delegation['annotation_wait_receipt']['path']);p.write_text('{}')
        with self.assertRaises(ValueError):self.check()
    def test_nonzero_wait_rejected(self):
        self.replace_wait(exit_code=1)
        with self.assertRaises(ValueError):self.check()
    def test_different_session_rejected(self):
        self.replace_wait(tool_session=68614)
        with self.assertRaises(ValueError):self.check()
    def test_other_evaluation_rejected(self):
        self.replace_wait(evaluation={'path':'different','sha256':'0'*64})
        with self.assertRaises(ValueError):self.check()
    def test_other_native_owner_rejected(self):
        self.replace_wait(source_owner_complete={'path':'different','sha256':'0'*64})
        with self.assertRaises(ValueError):self.check()
    def test_capture_must_still_be_root_observed(self):
        self.review['actual_outer_process_completion']['capture']['wait_observed_by']=self.review['reviewer']
        with self.assertRaises(ValueError):self.check()
    def test_agent_observer_cannot_be_someone_else(self):
        self.replace_wait(wait_observed_by='assistant_unrelated')
        with self.assertRaises(ValueError):self.check()
    def test_existing_root_observed_contract_supported(self):
        d=json.loads((D/'native848_multisubtree41_delegated_export_20260918_v1/root_delegation.json').read_text())
        review=json.loads(new.base.read_pin(d['review']).read_text());d['schema']=new.DELEGATION_SCHEMA
        auth=dict(native_outer_wait=dict(session_id=67115,returncode=0),annotation_outer_wait=dict(session_id=25809,returncode=0))
        new.validate_delegation(d,d['evaluation'],d['selected_sample_ids'],d['review'],review,auth)
    def test_materialize_and_numeric_functions_unchanged(self):
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(old.materialize))),ast.dump(ast.parse(inspect.getsource(new.materialize))))
        for name in ('validate_partition','validate_review','validate_qualification','validate_authorization','load_model_inputs'):
            self.assertIs(getattr(old,name),getattr(new,name))


if __name__=='__main__':unittest.main()
