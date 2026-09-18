"""Nonnative identity, exclusion, and unchanged numerical dispatch checks."""
import ast
import inspect
import unittest
from copy import deepcopy
from unittest.mock import patch
from . import native848_generated_farmerge_capture_v1 as capture
from . import native848_generated_farmerge_9mm_v1 as consumer
from . import native848_pilot_plan_v1 as anchors
from .native848_farmerge_annotation_v1_test import fixture,bounds,geometry


class Guards(unittest.TestCase):
    def test_every_pose_checked_but_identical_source_authenticated_once(self):
        a=dict(source_capture='D:/source',source_sample='s',original_variant={'source':'donor'})
        records=[dict(anchor={'path':'D:/anchor','sha256':'h'},sample_id=str(i)) for i in range(4)]
        with patch.object(capture,'bound',return_value=a),patch.object(anchors,'raw_anchor',return_value=a) as raw,patch.object(capture,'validate_pose') as pose:
            self.assertEqual(capture.authenticate_anchors(records),[a]*4)
            self.assertEqual(raw.call_count,1);self.assertEqual(pose.call_count,4)

    def test_changed_anchor_source_is_not_covered_by_memoization(self):
        a=dict(source_capture='D:/source',source_sample='s',original_variant={'source':'donor'})
        b=dict(a,source_sample='different')
        records=[dict(anchor={'path':'D:/a','sha256':'h'}),dict(anchor={'path':'D:/b','sha256':'h2'})]
        with patch.object(capture,'bound',side_effect=[a,b]),patch.object(anchors,'raw_anchor',return_value=a) as raw,patch.object(capture,'validate_pose'):
            with self.assertRaises(ValueError):capture.authenticate_anchors(records)
            self.assertEqual(raw.call_count,2)

    def test_all_frozen_near_numeric_functions_are_direct_aliases(self):
        self.assertIs(consumer.evaluate_frame,consumer.base.evaluate_frame)
        self.assertIs(consumer.geometry_9mm,consumer.base.geometry_9mm)
        self.assertIs(consumer.validate_planned_geometry,consumer.base.validate_planned_geometry)

    def test_inventory_dispatch_does_not_rewrite_inputs_or_numeric_source(self):
        # Namespace constant is the sole dispatch adaptation; the source AST is
        # compiled verbatim, and the context object remains the caller's object.
        source=inspect.getsource(consumer.build_inventory)
        self.assertIn('ast.parse(inspect.getsource(base.build_inventory))',source)
        self.assertNotIn("context['schema']=",source)
        self.assertNotIn('context.update',source)

    def fixture(self,x):
        cat,census,selection,provenance=fixture()
        groups=consumer.far.compile_groups(cat,census,selection,provenance)
        line=dict(geometry_source_id='seed7_full',geometry_target_id='seed7_full/SubStem_1',morphology_id=None,generated_geometry=False,physical_geometry_evidence=None)
        retained=dict(target_id='P/SubStem_2',status='candidate_pending_visual_review',reason='exact_native_candidate',**dict(line,geometry_source_id='actual_new_geometry',generated_geometry=True))
        annotation=dict(targets=[dict(target_id='P/SubStem_1',status='excluded',reason='not_visible_zero_authenticated_component_pixels',**line),retained],target_census={})
        entry=dict(target_id='P/SubStem_1',source_family='seed7_full',source_target_id='seed7_full/SubStem_1',semantic_leaf_petiole_candidate=True)
        with patch.object(consumer.far.coverage,'source_geometry',return_value=geometry(x)):
            count=consumer.replace_coarse_rows(annotation,[entry],cat,groups,bounds())
        self.assertEqual(count,1);self.assertIs(annotation['targets'][1],retained)
        self.assertEqual(annotation['targets'][0]['geometry_source_id'],'seed7_full')
        return annotation

    def test_coarse_no_pixels_is_replaced_by_positive_current_bound(self):
        result=self.fixture(2.)
        self.assertEqual(result['targets'][0]['reason'],consumer.original.OUTSIDE_REASON)
        self.assertNotIn('visibility',result['targets'][0])
        self.assertFalse(result['targets'][0]['generated_geometry'])

    def test_reachable_coarse_geometry_blocks_instead_of_assuming_invisibility(self):
        result=self.fixture(.1)
        self.assertEqual(result['targets'][0]['status'],'unknown')
        self.assertEqual(result['target_census']['unknown_target_ids'],['P/SubStem_1'])

    def test_capture_clock_has_separate_source8_and_production6_profile(self):
        source=inspect.getsource(capture.capture)
        self.assertIn("source_profile=bound(candidates['profile']);check_profile(source_profile)",source)
        self.assertIn("profile=bound(request['production_profile']);render_api.check_profile(profile)",source)
        self.assertIn("profile['warmup_steps']==[8]*7",source)
        self.assertIn("profile['request_subframes']==6",source)
        self.assertIn('geometry=overlay_api.OriginalCollisionBridge(overlay)',source)


if __name__=='__main__':unittest.main()
