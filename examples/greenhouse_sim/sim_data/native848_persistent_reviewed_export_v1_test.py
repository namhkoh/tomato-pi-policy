"""CPU authority mutations and frozen materialization parity; no native/export."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import unittest

import numpy as np

from . import native848_persistent_reviewed_export_v1 as export


def pin(name):
    return dict(path=str(Path('fixture')/name),sha256='fixture-'+name)


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.capture=Path('fixture/trial/capture/segments/s001/capture').resolve()
        public=dict(batch_owner_complete=pin('trial/owner_complete.json'),batch_result=pin('trial/capture/result.json'),
            batch_request=pin('batch_request.json'),native_identity=pin('native_identity.json'),
            segment_id='s001',segment_index=1,segment_result=dict(path=str(self.capture/'result.json'),sha256='segment'),
            source_request=pin('source_request.json'),segment_is_not_separate_native_owner=True,
            all_batch_segments_and_global_clocks_authenticated=True)
        segment=dict(segment_id='s001',source_request=public['source_request'],batch_request=public['batch_request'],
            native_identity=public['native_identity'],training_approved=False,accepted_training_increment=0,
            frames=[dict(observation_id='a'),dict(observation_id='b')])
        self.authority=dict(public=public,segment_result=segment,complete=dict(result=public['batch_result']))
        self.value=dict(schema=export.EVALUATION_SCHEMA,native_control_performed=False,training_approved=False,
            accepted_training_increment=0,persistent_batch_authority=deepcopy(public),batch_owner_complete=public['batch_owner_complete'],
            batch_capture_result=public['batch_result'],segment_result=public['segment_result'],source_capture=str(self.capture),
            frames_evaluated=2,records=[dict(sample_id='a'),dict(sample_id='b')])
        self.delegation=dict(source_owner_complete=public['batch_owner_complete'],source_capture=str(self.capture),
            persistent_batch_authority=deepcopy(public))

    def check(self):
        return export.validate_persistent_authority(self.value,self.authority,self.capture,self.delegation)

    def test_complete_segment_under_actual_batch_accepted(self):
        self.assertIs(self.check(),self.authority['segment_result'])

    def test_ordinary_evaluation_not_aliased_to_persistent(self):
        self.value['schema']='greenhouse.native848_generated_capture_evaluation.v1'
        with self.assertRaises(ValueError):self.check()

    def test_fake_segment_owner_rejected(self):
        self.value['batch_owner_complete']=pin('segments/s001/owner_complete.json')
        with self.assertRaises(ValueError):self.check()

    def test_cross_segment_evaluation_rejected(self):
        self.value['persistent_batch_authority']['segment_id']='s000'
        with self.assertRaises(ValueError):self.check()

    def test_foreign_complete_batch_result_rejected(self):
        self.authority['complete']['result']=pin('other_batch/result.json')
        with self.assertRaises(ValueError):self.check()

    def test_source_capture_escape_rejected(self):
        self.value['source_capture']=str(self.capture.parent/'other')
        with self.assertRaises(ValueError):self.check()

    def test_root_delegation_must_bind_same_batch_and_segment(self):
        for key in ('source_owner_complete','source_capture','persistent_batch_authority'):
            with self.subTest(key=key):
                saved=deepcopy(self.delegation)
                self.delegation[key]=str(self.capture.parent) if key=='source_capture' else {}
                with self.assertRaises(ValueError):self.check()
                self.delegation=saved

    def test_segment_source_request_and_native_identity_rejected(self):
        for key in ('source_request','native_identity','batch_request'):
            with self.subTest(key=key):
                saved=self.authority['segment_result'][key]
                self.authority['segment_result'][key]=pin('foreign')
                with self.assertRaises(ValueError):self.check()
                self.authority['segment_result'][key]=saved

    def test_omitted_extra_duplicate_frame_rejected(self):
        for records in ([dict(sample_id='a')],[dict(sample_id='a'),dict(sample_id='foreign')],
                        [dict(sample_id='a'),dict(sample_id='a')]):
            with self.subTest(records=records):
                self.value['records']=records;self.value['frames_evaluated']=len(records)
                with self.assertRaises(ValueError):self.check()


class DelegationTests(unittest.TestCase):
    def setUp(self):
        self.ep=pin('evaluation');self.rp=pin('review');self.selected=['a']
        self.review=dict(reviewer='assistant_legacy_coverage',actual_outer_process_completion={
            'capture':dict(tool_session=1,exit_code=0,wait_observed_by='assistant_root'),
            'annotation':dict(tool_session=2,exit_code=0,wait_observed_by='assistant_root')})
        self.auth=dict(native_outer_wait=dict(session_id=1,returncode=0),annotation_outer_wait=dict(session_id=2,returncode=0))
        self.d=dict(schema=export.delegated.DELEGATION_SCHEMA,authorized_by='assistant_root',
            reviewer=self.review['reviewer'],evaluation=self.ep,review=self.rp,selected_sample_ids=self.selected,
            training_launch_authorized=False,authorization='Materialize this actual reviewed selection only.',
            required_checks_preserved=['all source and artifact hashes','sensor validity and exact native848x408',
                'full144 census and exactly one9mm cutpoint','positive alternative exclusions',
                'visual review accept decisions','donor TRAIN lineage and morphology identity','exact duplicate rejection'],
            root_observed_completion=deepcopy(self.review['actual_outer_process_completion']))

    def check(self):
        export.validate_delegation(self.d,self.ep,self.selected,self.rp,self.review,self.auth)

    def test_honest_independent_root_wait_and_reviewer(self):self.check()

    def test_reviewer_cannot_claim_root_wait(self):
        self.review['actual_outer_process_completion']['annotation']['wait_observed_by']='assistant_legacy_coverage'
        with self.assertRaises(ValueError):self.check()

    def test_nonzero_or_different_outer_session_rejected(self):
        for field,value in [('returncode',1),('session_id',999)]:
            with self.subTest(field=field):
                saved=deepcopy(self.auth);self.auth['native_outer_wait'][field]=value
                with self.assertRaises(ValueError):self.check()
                self.auth=saved


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        ids=np.ones((408,848),np.uint32);ids[:,424:]=2
        depth=np.ones(ids.shape,np.float32);valid=np.ones(ids.shape,bool)
        catalogue=[dict(prim_path='/fg/p',component_index=1,variant_id='fg',organ_type='sub_stem'),
            dict(prim_path='/bg/p',component_index=2,variant_id='bg',organ_type='leaf')]
        self.score=export.background.score(ids,dict(renderer_id_to_prim={'1':'/fg/p','2':'/bg/p'}),
            catalogue,['fg'],depth=depth,valid=valid)
        self.record=dict(sample_id='a');self.obspin=pin('observation')
        self.obs=dict(context=pin('context'),mapping=pin('mapping'),files=dict(buffers=pin('buffers')))
        self.context=dict(catalogue=pin('catalogue'));self.amb=dict(continuously_visible_alternative_ids=['bg/target'])
        self.review=dict(background_vines_visible=True,flagged_contexts=[dict(target_id='bg/target',actual_native_context_viewed=True)],
            actual_background_measurement=dict(sample_id='a',observation=self.obspin,context=self.obs['context'],
                catalogue=self.context['catalogue'],mapping=self.obs['mapping'],buffers=self.obs['files']['buffers'],
                score=deepcopy(self.score),meets_root_additional_background_criterion=True,criterion_is_not_visual_acceptance=True))

    def check(self):
        export.validate_background_review(self.review,self.record,self.obspin,self.obs,self.context,self.score,self.amb)

    def test_real_native_shape_pixel_score_passes(self):
        self.assertEqual(self.score['background_plant_pixel_fraction'],.5);self.check()

    def test_below_forty_percent_rejected_even_when_review_claims_pass(self):
        self.score['background_plant_pixel_fraction']=.399
        self.review['actual_background_measurement']['score']=deepcopy(self.score)
        with self.assertRaises(ValueError):self.check()

    def test_changed_or_foreign_measurement_rejected(self):
        for key in ('observation','mapping','buffers','context','catalogue'):
            with self.subTest(key=key):
                saved=deepcopy(self.review);self.review['actual_background_measurement'][key]=pin('foreign')
                with self.assertRaises(ValueError):self.check()
                self.review=saved
        self.review['actual_background_measurement']['score']['background_plant_pixel_fraction']=.9
        with self.assertRaises(ValueError):self.check()

    def test_unknown_pixels_and_coarse_identity_rejected(self):
        for key,value in [('unknowns',[dict(renderer_id=5)]),('unknown_unmapped_fraction',.001),('coarse_organ_identity_unavailable',True)]:
            with self.subTest(key=key):
                saved=deepcopy(self.score);self.score[key]=value
                self.review['actual_background_measurement']['score']=deepcopy(self.score)
                with self.assertRaises(ValueError):self.check()
                self.score=saved

    def test_missing_or_duplicate_or_unviewed_alternative_rejected(self):
        for flags in ([],[dict(target_id='other',actual_native_context_viewed=True)],
                      [dict(target_id='bg/target',actual_native_context_viewed=False)],
                      [dict(target_id='bg/target',actual_native_context_viewed=True)]*2):
            with self.subTest(flags=flags):
                self.review['flagged_contexts']=flags
                with self.assertRaises(ValueError):self.check()


class FrozenParityTests(unittest.TestCase):
    def test_numerical_review_and_loader_functions_are_exact_original_objects(self):
        for name in ('validate_partition','validate_review','validate_qualification','validate_authorization','load_model_inputs'):
            self.assertIs(getattr(export,name),getattr(export.frozen,name))
        self.assertIs(export.validate_delegation,export.delegated.validate_delegation)

    def test_numeric_census_and_sensor_materialization_statements_unchanged(self):
        original=inspect.getsource(export.delegated.materialize)
        actual=inspect.getsource(export.materialize)
        # The complete census/geometry/ambiguity/answer block is copied literally.
        start="    context=read(local(capture/'context.json'))"
        end="        rgb_pin=obs['files']['rgb']"
        self.assertEqual(original[original.index(start):original.index(end)],actual[actual.index(start):actual.index(end)])
        # Copy bytes/arrays/private evidence and immediately round-trip the same loader.
        start="            with (folder/'rgb.png').open('xb')"
        end="            row=dict(schema=FRAME_SCHEMA"
        self.assertEqual(original[original.index(start):original.index(end)],actual[actual.index(start):actual.index(end)])
        self.assertIn('load_model_inputs(row,output);rows.append(row)',actual)
        ast.parse(Path(export.__file__).read_text())

    def test_actual_authority_helper_is_called_without_evaluation_replay(self):
        source=inspect.getsource(export.materialize)
        self.assertIn('persistent.authenticate_batch(capture,read,local,producer)',source)
        self.assertNotIn('evaluate_segment(',source)
        self.assertNotIn("value['owner_complete']=",source)
        self.assertEqual(export.FRAME_SCHEMA,export.frozen.FRAME_SCHEMA)


if __name__=='__main__':unittest.main(verbosity=2)
