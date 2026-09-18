"""Observer-boundary regression tests; no native execution or materialization."""
import ast
import inspect
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from . import native848_persistent_reviewed_export_v3 as export
from . import native848_persistent_reviewed_export_v2 as previous

ROOT=Path(__file__).resolve().parents[3]
DIAGNOSTICS=ROOT/'data/sim_data/diagnostics'
HANDOFF=DIAGNOSTICS/'native848_generated67_multihost12_actual_review_20260919_v1/ready_handoff.json'
REQUIRED=['all source and artifact hashes','sensor validity and exact native848x408',
    'full144 census and exactly one9mm cutpoint','positive alternative exclusions',
    'visual review accept decisions','donor TRAIN lineage and morphology identity','exact duplicate rejection']


class ObserverBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=DIAGNOSTICS)
        self.folder=Path(self.directory.name);self.counter=0
        self.selected=['sample'];self.capture=str(self.folder/'native/capture/segments/s001/capture')
        self.owner={'path':str(self.folder/'native/owner_complete.json'),'sha256':'owner'}
        self.batch={'path':str(self.folder/'native/capture/result.json'),'sha256':'batch'}
        self.public=dict(batch_owner_complete=self.owner,batch_result=self.batch,segment_id='s001')
        self.value=dict(schema=export.EVALUATION_SCHEMA,persistent_batch_authority=self.public,
            batch_owner_complete=self.owner,batch_capture_result=self.batch,
            source_capture=self.capture,frames_evaluated=3)
        self.ep=self.save(self.value)
        self.summary=dict(schema='greenhouse.generated67_multihost12_annotation_summary.v1',
            annotation_wait_observer='assistant_legacy_coverage',capture_wait_observer='root',
            batch_owner_complete=self.owner,batch_result=self.batch,actual_visual_review=False,
            native_launched=False,export_performed=False,accepted_training_increment=0,
            frames_evaluated=12,expected_maximum_frames=12,
            segments=[dict(segment_id=f's00{i}',actual_child_returncode=0,frames_evaluated=3,
                evaluation=self.ep if i==1 else {'path':'other','sha256':str(i)},
                source_capture=self.capture if i==1 else 'other') for i in range(4)])
        self.sp=self.save(self.summary)
        self.receipt=dict(schema='greenhouse.actual_tool_process_wait.v1',session_id=10387,
            exit_code=0,observer='assistant_legacy_coverage',tool_chunk='3bdafa',result=self.sp,
            all_four_child_returncodes_zero=True,native_launch_by_this_agent=False,
            native_capture_wait=dict(observer='root',session_id=50027,tool_chunk='848362',exit_code=0))
        self.wp=self.save(self.receipt)
        self.review=dict(reviewer='assistant_native_recovery2',actual_outer_process_completion=dict(
            capture=dict(tool_session=50027,exit_code=0,wait_observed_by='assistant_root',completion_chunk='848362'),
            annotation=dict(tool_session=10387,exit_code=0,wait_observed_by='assistant_legacy_coverage',
                completion_chunk='3bdafa',actual_wait_receipt=self.wp)))
        self.rp=self.save(self.review)
        self.construct()

    def construct(self):
        self.d=dict(schema=export.DELEGATION_SCHEMA,authorized_by='assistant_root',
            reviewer=self.review['reviewer'],evaluation=self.ep,review=self.rp,selected_sample_ids=self.selected,
            training_launch_authorized=False,required_checks_preserved=REQUIRED,
            authorization='CPU test fixture only; no real export authorization.',
            root_observed_completion={'capture':deepcopy(self.review['actual_outer_process_completion']['capture'])},
            annotation_wait_observer='assistant_legacy_coverage',annotation_wait_receipt=self.wp,
            annotation_summary=self.sp,persistent_batch_authority=self.public,
            source_owner_complete=self.owner,source_capture=self.capture)
        self.auth=dict(native_outer_wait=dict(session_id=50027,returncode=0),
            annotation_outer_wait=dict(session_id=10387,returncode=0),
            annotation_wait_observer='assistant_legacy_coverage',annotation_wait_receipt=self.wp,
            annotation_summary=self.sp)

    def save(self,value):
        self.counter+=1;p=self.folder/f'{self.counter}.json'
        p.write_text(json.dumps(value),encoding='utf8');return export.pin(p)
    def tearDown(self):self.directory.cleanup()
    def check(self):
        return export.validate_delegation(self.d,self.ep,self.selected,self.rp,self.review,self.auth)
    def reject(self):
        with self.assertRaises((ValueError,KeyError)):self.check()
    def replace_receipt(self):
        self.wp=self.save(self.receipt)
        self.d['annotation_wait_receipt']=self.auth['annotation_wait_receipt']=self.wp
        self.review['actual_outer_process_completion']['annotation']['actual_wait_receipt']=self.wp
    def replace_summary(self):
        self.sp=self.save(self.summary)
        self.d['annotation_summary']=self.auth['annotation_summary']=self.sp
        self.receipt['result']=self.sp;self.replace_receipt()

    def test_distinct_real_roles_are_allowed(self):self.check()
    def test_spoofed_observer_rejected(self):
        self.d['annotation_wait_observer']=self.auth['annotation_wait_observer']='assistant_native_recovery2'
        self.review['actual_outer_process_completion']['annotation']['wait_observed_by']='assistant_native_recovery2'
        self.reject()
    def test_false_root_annotation_rejected(self):
        self.d['root_observed_completion']['annotation']=deepcopy(self.review['actual_outer_process_completion']['annotation'])
        self.reject()
    def test_annotation_receipt_missing_rejected(self):
        del self.d['annotation_wait_receipt'];self.reject()
    def test_annotation_session_mismatch_rejected(self):
        self.auth['annotation_outer_wait']['session_id']=7;self.reject()
    def test_annotation_chunk_mismatch_rejected(self):
        self.review['actual_outer_process_completion']['annotation']['completion_chunk']='wrong';self.reject()
    def test_nonzero_receipt_rejected(self):
        self.receipt['exit_code']=1;self.replace_receipt();self.reject()
    def test_receipt_wrong_root_capture_rejected(self):
        self.receipt['native_capture_wait']['session_id']=50028;self.replace_receipt();self.reject()
    def test_summary_pin_mismatch_rejected(self):
        self.d['annotation_summary']=dict(self.sp,sha256='0'*64);self.reject()
    def test_changed_receipt_bytes_rejected(self):
        Path(self.wp['path']).write_text('{}');self.reject()
    def test_summary_observer_mismatch_rejected(self):
        self.summary['annotation_wait_observer']='assistant_native_recovery2';self.replace_summary();self.reject()
    def test_summary_wrong_owner_rejected(self):
        self.summary['batch_owner_complete']={'path':'other','sha256':'other'};self.replace_summary();self.reject()
    def test_summary_failed_child_rejected(self):
        self.summary['segments'][2]['actual_child_returncode']=1;self.replace_summary();self.reject()
    def test_summary_repeated_segment_rejected(self):
        self.summary['segments'][2]['segment_id']='s001';self.replace_summary();self.reject()
    def test_summary_wrong_evaluation_rejected(self):
        self.summary['segments'][1]['evaluation']={'path':'other','sha256':'other'};self.replace_summary();self.reject()
    def test_summary_wrong_source_capture_rejected(self):
        self.summary['segments'][1]['source_capture']=str(self.folder/'other');self.replace_summary();self.reject()
    def test_evaluation_wrong_source_capture_rejected(self):
        self.value['source_capture']=str(self.folder/'other');self.ep=self.save(self.value)
        self.d['evaluation']=self.ep;self.summary['segments'][1]['evaluation']=self.ep
        self.replace_summary();self.reject()
    def test_unauthorized_reviewer_rejected(self):
        self.d['reviewer']='assistant_legacy_coverage';self.reject()


class FrozenParityTests(unittest.TestCase):
    def test_materialize_ast_identical_to_frozen_v2(self):
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(export.materialize))),
            ast.dump(ast.parse(inspect.getsource(previous.materialize))))
    def test_authority_background_and_numerics_unchanged(self):
        for name in ('validate_persistent_authority','validate_background_review'):
            self.assertEqual(inspect.getsource(getattr(export,name)),inspect.getsource(getattr(previous,name)))
        for name in ('validate_partition','validate_review','validate_qualification','validate_authorization','load_model_inputs'):
            self.assertIs(getattr(export,name),getattr(previous,name))
        self.assertEqual(export.PINS,previous.PINS)
        for path,sha in export.PINS.items():self.assertEqual(export.base.digest(path),sha,path)
    def test_frozen_v2_bytes_preserved(self):
        self.assertEqual(export.base.digest(previous.__file__),
            'b81e3d562899d1005e09693a33291097c2c73e08bee8f616a1b6ffcb7bb09505')


@unittest.skipUnless(HANDOFF.exists(),'Local actual donor67 evidence not present')
class ActualDonor67DryValidationTests(unittest.TestCase):
    setUp=ObserverBoundaryTests.setUp
    tearDown=ObserverBoundaryTests.tearDown
    save=ObserverBoundaryTests.save
    construct=ObserverBoundaryTests.construct
    check=ObserverBoundaryTests.check
    def test_actual_donor67_handoff_without_materializing(self):
        h=json.loads(HANDOFF.read_text());self.ep=h['evaluation'];self.rp=h['review']
        self.review=json.loads(export.base.read_pin(self.rp).read_text())
        self.selected=h['selected_sample_ids'];self.capture=h['source_capture'];self.owner=h['source_owner_complete']
        self.public=h['persistent_batch_authority'];self.sp=h['summary'];self.wp=h['annotation_wait_receipt']
        self.construct();self.check()
        self.assertEqual(self.selected,['seed67_full_SubStem_42_dense_v2_view_2'])
        self.assertEqual(len(h['excluded_coordinator_holds']),11)


if __name__=='__main__':unittest.main(verbosity=2)
