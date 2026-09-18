"""CPU guards for sweep export; optional real evaluation validation never exports."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_data import native848_scene_sweep_dataset_v1 as p

REAL_EVALUATION = None
REAL_RESULT = None


class ExportAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / 'must_not_exist'
        self.ev = self.save('evaluation.json', {})
        self.review = self.save('review.json', {'reviewer': 'assistant_native_recovery2'})
        self.auth = dict(schema='greenhouse.native848_scene_sweep_export_authorization.v1',
            evaluation=self.ev, review=self.review, selected_frame_ids=['frame'],
            exporter=p.pin(p.__file__), dataset_only=True, actual_capture_exit_code=0,
            actual_annotation_exit_code=0, blocking_findings=[], reviewer='assistant_native_recovery2')
        self.addCleanup(self.tmp.cleanup)

    def save(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return p.pin(path)

    def reject(self, auth, selected=None, message='Exact root-reviewed'):
        spec = self.save('authorization.json', auth)
        with patch.object(p.base.roots, 'checkpoint_output', return_value=self.output), \
                patch.object(p.evaluated, 'authenticate_capture') as authenticate:
            with self.assertRaisesRegex(ValueError, message):
                p.materialize(self.ev, selected or ['frame'], self.review, self.output,
                              authorization_pin=spec)
            authenticate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_authorization_is_required_keyword(self):
        with self.assertRaises(TypeError):
            p.materialize(self.ev, ['frame'], self.review, self.output)
        self.assertFalse(self.output.exists())

    def test_changed_selection_or_source_pins_cannot_reuse_authorization(self):
        for key, value in [('selected_frame_ids', ['other']),
                           ('evaluation', {'path': 'other', 'sha256': '0' * 64}),
                           ('review', {'path': 'other', 'sha256': '0' * 64}),
                           ('exporter', {'path': p.__file__, 'sha256': '0' * 64})]:
            with self.subTest(key=key):
                auth = deepcopy(self.auth); auth[key] = value
                self.reject(auth)

    def test_failed_process_or_blocking_review_cannot_export(self):
        for key, value in [('actual_capture_exit_code', 1), ('actual_annotation_exit_code', 1),
                           ('blocking_findings', ['unresolved']), ('dataset_only', False),
                           ('reviewer', 'assistant_root')]:
            with self.subTest(key=key):
                auth = deepcopy(self.auth); auth[key] = value
                self.reject(auth)

    def test_forged_visual_reviewer_fails_full_materialize(self):
        self.review = self.save('review.json', {'reviewer': 'fabricated_fixture_identity'})
        self.auth['review'] = self.review
        self.reject(self.auth)

    def test_even_authorized_fixture_requires_complete_evaluation(self):
        self.ev = self.save('evaluation.json', dict(schema=p.evaluated.SCHEMA,
            completed_whole_capture_authenticated=False))
        self.auth['evaluation'] = self.ev
        self.reject(self.auth, message='Complete offline capture')

    def test_empty_or_repeated_selection_is_rejected_before_io(self):
        for selected in ([], ['frame', 'frame']):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, 'Nonempty unique'):
                p.materialize(self.ev, selected, self.review, self.output, authorization_pin={})
        self.assertFalse(self.output.exists())


class ActualReviewGuardTests(unittest.TestCase):
    def setUp(self):
        self.ev = {'path': 'fixture_evaluation', 'sha256': '0' * 64}
        self.record = dict(frame_id='frame', **{k: {'path': k, 'sha256': '1' * 64}
            for k in ('annotation', 'ambiguity', 'rgb', 'background')})
        self.row = dict(self.record, decision='accept', actual_full_native_rgb_viewed=True,
            junction_and_first_leaf_reviewed=True, no_obvious_ghosting=True,
            single_answer_uniqueness_reviewed=True, background_vines_visible=True,
            all_flagged_alternatives_inspected=True, inspected_alternative_ids=['other'])
        self.review = dict(schema=p.REVIEW_SCHEMA, evaluation=self.ev,
            human_review_claimed=False, records=[self.row], CPU_fixture_only=True)
        self.assessment = {'alternative_assessments': [dict(target_id='other', workspace_recheck_required=True)]}

    def test_complete_unit_fixture_passes_review_gate_only(self):
        self.assertEqual(p.checked_review(self.review, self.ev, self.record, self.assessment), self.row)

    def test_each_required_actual_review_step_is_mandatory(self):
        for key in ('actual_full_native_rgb_viewed', 'junction_and_first_leaf_reviewed',
                    'no_obvious_ghosting', 'single_answer_uniqueness_reviewed',
                    'background_vines_visible', 'all_flagged_alternatives_inspected'):
            review = deepcopy(self.review); review['records'][0][key] = False
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Actual complete'):
                p.checked_review(review, self.ev, self.record, self.assessment)

    def test_missing_or_duplicate_flag_inspection_rejected(self):
        for values in ([], ['other', 'other']):
            review = deepcopy(self.review); review['records'][0]['inspected_alternative_ids'] = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'Unreviewed flagged'):
                p.checked_review(review, self.ev, self.record, self.assessment)

    def test_changed_image_or_annotation_cannot_inherit_review(self):
        for key in ('annotation', 'ambiguity', 'rgb', 'background'):
            record = deepcopy(self.record); record[key]['sha256'] = '2' * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Review evidence differs'):
                p.checked_review(self.review, self.ev, record, self.assessment)

    def test_duplicate_review_or_hold_decision_rejected(self):
        review = deepcopy(self.review); review['records'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate review'):
            p.checked_review(review, self.ev, self.record, self.assessment)
        review = deepcopy(self.review); review['records'][0]['decision'] = 'hold'
        with self.assertRaisesRegex(ValueError, 'Actual complete'):
            p.checked_review(review, self.ev, self.record, self.assessment)


class FailedAlternativeGuardTests(unittest.TestCase):
    def test_reachable_unknown_or_incomplete_assessment_rejected_before_metadata_read(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'assessment.json'
            metadata = {'path': 'metadata', 'sha256': '0' * 64}
            annotation = {'path': 'annotation', 'sha256': '1' * 64}
            result = dict(schema=p.ambiguity.SCHEMA, policy=p.ambiguity.POLICY,
                metadata=metadata, annotation=annotation, frame_id='frame',
                source_bindings={str(Path(p.ambiguity.__file__).resolve()): p.base.digest(p.ambiguity.__file__)},
                assessment_complete=True, single_answer_unambiguous=True,
                decision='candidate_pending_actual_visual_review', reachable_alternative_ids=[],
                unknown_alternative_ids=[], strict_label_statuses_preserved=True)
            for key, value in [('reachable_alternative_ids', ['other']),
                               ('unknown_alternative_ids', ['other']),
                               ('assessment_complete', False), ('decision', 'hold')]:
                changed = deepcopy(result); changed[key] = value
                path.write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(key=key), patch.object(p.ambiguity.dataset, 'read_pin', return_value=path) as read:
                    with self.assertRaisesRegex(ValueError, 'Competing or unresolved answer'):
                        p.ambiguity.validate_assessment(p.pin(path), metadata, annotation, {'frame_id': 'frame'})
                    self.assertEqual(read.call_count, 1)


class SensorGuardTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((408, 848, 3), dtype=np.uint8)
        self.depth = np.ones((408, 848), dtype=np.float32)
        self.valid = np.ones((408, 848), dtype=bool)
        self.cal = dict(resolution=[848, 408], crop_resize=None,
            depth_convention='optical_axis_z_metres_not_ray_range',
            intrinsics=[[400, 0, 424], [0, 400, 204], [0, 0, 1]],
            clipping_range_m=[0.01, 10], optical_frame='fixture', pixel_convention='fixture')

    def test_native_arrays_and_invalid_pixel_mask(self):
        p.base.validate_arrays(self.rgb, self.depth, self.valid, self.cal)
        depth = self.depth.copy(); depth[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Depth validity'):
            p.base.validate_arrays(self.rgb, depth, self.valid, self.cal)

    def test_resize_or_ray_range_or_float64_depth_is_rejected(self):
        for change in ({'crop_resize': [0, 0, 400, 400]},
                       {'depth_convention': 'ray_range'}, {'resolution': [424, 204]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.base.validate_arrays(self.rgb, self.depth, self.valid, dict(self.cal, **change))
        with self.assertRaisesRegex(ValueError, 'Aligned float32'):
            p.base.validate_arrays(self.rgb, self.depth.astype(np.float64), self.valid, self.cal)

    def test_shared_original_loader_rejects_query_or_mask_inputs_before_read(self):
        row = dict(schema=p.FRAME_SCHEMA, task_id=p.base.TASK,
                   inputs={k: {} for k in (*p.base.INPUT_KEYS, 'target_mask')})
        with self.assertRaisesRegex(ValueError, 'Hidden or missing'):
            p.load_model_inputs(row, '.')


class RealCompletedSweepDryValidation(unittest.TestCase):
    def test_real_complete_capture_and_one_candidate_without_export_or_fake_review(self):
        global REAL_RESULT
        if REAL_EVALUATION is None:
            self.skipTest('Supply --evaluation for completed native evidence')
        evaluation_pin = p.pin(REAL_EVALUATION)
        value = json.loads(Path(REAL_EVALUATION).read_text())
        module = p.EVALUATORS[value['schema']]
        bound = {}
        def read(spec, *, within=None):
            path = Path(spec['path']).resolve()
            self.assertTrue(within is None or path.is_relative_to(Path(within).resolve()))
            self.assertEqual(p.base.digest(path), spec['sha256'])
            bound[str(path)] = spec['sha256']
            return json.loads(path.read_text())
        p.verify_bindings(p.PINS)
        p.verify_bindings(value['source_bindings'])
        capture = Path(value['source_capture']).resolve()
        complete, result, request, checked, context, observations = module.authenticate_capture(capture, read, p.pin)
        self.assertEqual(value['capture_result'], complete['result'])
        self.assertEqual({r['frame_id'] for r in value['records']}, {r['sample_id'] for r in observations})
        census, catalogue, reports = (read(context[k]) for k in ('census', 'catalogue', 'reports'))
        source_plan = read(context['source_collection_plan'])
        inventory, _ = module.prepare_inventory(context, census, reports, catalogue, source_plan)
        changed = deepcopy(census)
        changed['all_plant_roots'][0]['source_split'] = 'validation'
        del changed['deterministic_census_sha256']
        changed['deterministic_census_sha256'] = module.fingerprint(changed)
        with self.assertRaisesRegex(ValueError, 'Every scene source family must remain TRAIN'):
            module.prepare_inventory(dict(context, full_scene_census=changed), changed,
                                     reports, catalogue, source_plan)
        changed_catalogue = list(catalogue)
        changed_catalogue[0] = dict(catalogue[0], source_plant_id='forged_original_donor')
        with self.assertRaisesRegex(ValueError, 'Exact complete native catalogue'):
            module.prepare_inventory(context, census, reports, changed_catalogue, source_plan)
        by = {c['variant_id'] + '/' + c['component_id']: c for c in catalogue if c['organ_type'] == 'sub_stem'}
        self.assertEqual(len(by), len(inventory))
        record = next(r for r in value['records'] if r['unique_automated_candidate'] and r['background_criterion_passed'])
        ann, meta, bg = (read(record[k]) for k in ('annotation', 'metadata', 'background'))
        target = p.strict.validate_census(ann)[0]
        for t in ann['targets']:
            c = by[t['target_id']]
            self.assertEqual((t['source_family'], t['source_component_id'], t['plant_instance_id'], t['source_target_id']),
                             (c['source_plant_id'], c['component_id'], c['variant_id'], c['source_plant_id'] + '/' + c['component_id']))
        assessment = p.ambiguity.validate_assessment(record['ambiguity'], record['metadata'], record['annotation'], ann)
        rgb = np.asarray(Image.open(p.base.read_pin(record['rgb'])).convert('RGB'))
        with np.load(p.base.read_pin(record['buffers']), allow_pickle=False) as a:
            depth, valid, ids = (a[k].copy() for k in ('depth_m', 'depth_valid', 'renderer_instance_id'))
        p.base.validate_arrays(rgb, depth, valid, meta['calibration'])
        obs = next(o for o in observations if o['sample_id'] == record['frame_id'])
        score = p.background.score(ids, read(obs['mapping']), catalogue, [target['plant_instance_id']], depth=depth, valid=valid)
        self.assertEqual(score, bg['score'])
        self.assertGreaterEqual(score['background_plant_pixel_fraction'], 0.4)
        self.assertEqual(score['unknown_unmapped_fraction'], 0)
        answer = p.base.target_answer(target, meta['calibration'])
        with self.assertRaises((KeyError, ValueError)):
            p.checked_review({}, evaluation_pin, record, assessment)
        REAL_RESULT = dict(evaluation=evaluation_pin, actual_capture_frames=len(observations),
            census_petiole_count=len(inventory), frame_id=record['frame_id'], source_target_id=target['source_target_id'],
            answer=answer, native_arrays_validated=True, ambiguity_validated=True,
            background_fraction=score['background_plant_pixel_fraction'], actual_visual_approval_inherited=False,
            export_executed=False, annotation_recomputed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evaluation', type=Path)
    args, remaining = parser.parse_known_args()
    REAL_EVALUATION = args.evaluation
    result = unittest.main(argv=[sys.argv[0], *remaining], verbosity=2, exit=False).result
    if REAL_RESULT is not None:
        print('REAL_DRY_VALIDATION ' + json.dumps(REAL_RESULT, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
