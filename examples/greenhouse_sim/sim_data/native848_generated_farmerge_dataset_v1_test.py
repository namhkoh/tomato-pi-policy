"""Exporter guards only: no files copied and no acceptance records created."""
from copy import deepcopy
from unittest.mock import patch
import unittest
from . import native848_generated_farmerge_dataset_v1 as a
from .native848_farmerge_annotation_v1_test import fixture,bounds,geometry


class Guards(unittest.TestCase):
    def evidence(self):
        cat,census,selection,provenance=fixture()
        groups=a.fast.far.compile_groups(cat,census,selection,provenance)
        entry=dict(target_id='P/SubStem_1',source_family='seed7_full',source_target_id='seed7_full/SubStem_1',semantic_leaf_petiole_candidate=True)
        annotation=dict(targets=[dict(target_id='P/SubStem_1',status='excluded',reason='not_visible_zero_authenticated_component_pixels')],target_census={},outer_workspace_bounds=bounds())
        with patch.object(a.fast.far.coverage,'source_geometry',return_value=geometry(2.)):
            a.fast.far.merged_rows(annotation,[entry],cat,groups,bounds())
        annotation['targets'][0]['generated_geometry']=False
        context=dict(render_census={'path':'render','sha256':'r'},farmerge_recipe={'path':'recipe','sha256':'p'})
        actual=a.fast.far.actual_coarse_bounds(groups,bounds())
        coverage=dict(schema=a.fast.COVERAGE_SCHEMA,**context,individual_coarse_component_pixel_visibility_claimed=False,
            all_near_components_evaluated_with_unchanged_numeric_predicates=True,
            coarse_actual_pose_bounds=actual,coarse_petiole_positive_exclusions=1)
        observation=dict(render_census=context['render_census'],coarse_surface_exclusion=actual)
        return annotation,coverage,observation,context,cat,groups

    def test_exact_coarse_bound_and_lineage_pass(self):
        result=a.validate_coarse_coverage(*self.evidence())
        self.assertTrue(result['all_components_outside_both_actual_arm_bounds'])

    def test_no_pixel_exclusion_cannot_replace_coarse_reach_proof(self):
        values=self.evidence();values[0]['targets'][0]['reason']='not_visible_zero_authenticated_component_pixels'
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_missing_coarse_petiole_rejected(self):
        values=self.evidence();values[1]['coarse_petiole_positive_exclusions']=0
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_changed_body_invalidates_saved_clearance(self):
        values=self.evidence();values[0]['outer_workspace_bounds']=bounds(2.)
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_forged_visible_coarse_component_rejected(self):
        values=self.evidence();values[1]['individual_coarse_component_pixel_visibility_claimed']=True
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_generated_foreground_cannot_enter_coarse_partition(self):
        values=self.evidence();values[0]['targets'][0]['generated_geometry']=True
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_foreign_render_pin_rejected(self):
        values=self.evidence();values[2]['render_census']=dict(path='other',sha256='x')
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_old_coverage_schema_is_not_silently_relabelled(self):
        values=self.evidence();values[1]['schema']='greenhouse.native848_generated_coverage_audit.v1'
        with self.assertRaises(ValueError):a.validate_coarse_coverage(*values)

    def test_anatomical_exclusion_remains_permitted_without_visibility_claim(self):
        values=self.evidence();row=values[0]['targets'][0]
        row['reason']='not_anatomically_eligible_leaf_petiole';row['anatomy_reason_codes']=['deleafed']
        a.validate_coarse_coverage(*values)

    def test_frozen_partition_review_and_authorization_are_reused(self):
        for name in ('validate_partition','validate_review','validate_authorization','validate_qualification'):
            self.assertIs(getattr(a,name),getattr(a.prior,name))

    def test_old_frame_schema_is_not_accepted_as_fast_export(self):
        record=dict(schema=a.prior.FRAME_SCHEMA,task_id=a.base.TASK,inputs={})
        with self.assertRaises(ValueError):a.load_model_inputs(record,'.')


if __name__=='__main__':unittest.main()
