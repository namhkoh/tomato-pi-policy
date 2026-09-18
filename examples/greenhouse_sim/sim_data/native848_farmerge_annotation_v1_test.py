"""Adversarial membership, ownership and actual-pose exclusion checks; no native."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import ast
import inspect
import numpy as np
from . import native848_farmerge_annotation_v1 as a


def fixture():
    root='/World/PackPlants/P'
    cat=[dict(component_index=i,component_id='SubStem_'+str(i),prim_path=root+'/SubStem_'+str(i),
              organ_type='sub_stem',variant_id='P',source_plant_id='seed7_full',split_group='seed7_full') for i in (1,2)]
    selected=dict(cat[0],original_world_min=[2.,0,0],original_world_max=[2.1,.1,.1])
    selection=dict(selected_components=[selected],protected_component_indices=[2])
    g=dict(group_id='g',plant_root=root,prim_path=root+'/__FarCoarseGroup_0',source_family='seed7_full',
           member_component_indices=[1],face_provenance=dict(path='face',sha256='a'))
    face=dict(original_component_index=1,original_component_id='SubStem_1',original_mesh_path=cat[0]['prim_path']+'/mesh',
              world_coordinate_max_euclidean_error_m=0.,material_graph_sha256='abc')
    p=dict(merged_mesh_path='/Plant/__FarCoarseGroup_0',plant=dict(plant_root=root,source_family='seed7_full'),faces=[face])
    census=dict(native_coarse_group_catalogue=[g]);provenance={'g':p}
    return cat,census,selection,provenance


def bounds(center=0.):
    return dict(schema=a.coverage.BOUNDS_SCHEMA,margin_m=.001,arms={side:dict(shoulder_world_m=[center,0.,0.],conservative_probe_reach_m=.8) for side in ('left','right')})


def geometry(x):
    return dict(nominal=dict(arc_m=.009,world_m=[x+.009,0.,0.],point_plant_m=[x+.009,0.,0.],radius_m=.003),
                oriented_centerline_world_m=[[x,0.,0.],[x+.04,0.,0.]])


class Guards(unittest.TestCase):
    def setUp(self):
        self.cat,self.census,self.selection,self.provenance=fixture()

    def compile(self):
        return a.compile_groups(self.cat,self.census,self.selection,self.provenance)

    def test_complete_group_and_exact_material_path(self):
        result=self.compile()
        self.assertEqual(set(result['paths']),{'/World/PackPlants/P/__FarCoarseGroup_0','/World/PackPlants/P/__FarCoarseGroup_0/M_abc'})

    def test_missing_member_rejected(self):
        self.census['native_coarse_group_catalogue'][0]['member_component_indices']=[]
        with self.assertRaises(ValueError):self.compile()

    def test_duplicate_member_rejected(self):
        self.census['native_coarse_group_catalogue'][0]['member_component_indices']=[1,1]
        with self.assertRaises(ValueError):self.compile()

    def test_cross_family_rejected(self):
        self.census['native_coarse_group_catalogue'][0]['source_family']='seed9_full'
        with self.assertRaises(ValueError):self.compile()

    def test_stale_selected_identity_rejected(self):
        self.selection['selected_components'][0]['component_id']='SubStem_99'
        with self.assertRaises(ValueError):self.compile()

    def test_missing_retained_partition_rejected(self):
        self.selection['protected_component_indices']=[]
        with self.assertRaises(ValueError):self.compile()

    def test_foreign_face_mesh_rejected(self):
        self.provenance['g']['faces'][0]['original_mesh_path']='/World/PackPlants/Q/SubStem_1/mesh'
        with self.assertRaises(ValueError):self.compile()

    def test_excess_rounding_rejected(self):
        self.provenance['g']['faces'][0]['world_coordinate_max_euclidean_error_m']=2e-6
        with self.assertRaises(ValueError):self.compile()

    def mapped(self,path,claim):
        groups=self.compile();ids=np.array([[42]],np.uint32)
        return a.classify_pixels(ids,dict(renderer_id_to_prim={'42':path},plant_ID_ownership={'42':claim}),self.cat,groups,np.ones((1,1),np.float32),np.ones((1,1),bool))

    def test_coarse_pixel_has_no_fabricated_component(self):
        g=self.census['native_coarse_group_catalogue'][0]
        claim={k:g[k] for k in ('group_id','prim_path','member_component_indices','face_provenance')}
        claim.update(kind='coarse_far_group',individual_component_pixel_ID_claimed=False)
        mask,audit=self.mapped(g['prim_path'],claim)
        self.assertEqual(mask.tolist(),[[0]]);self.assertFalse(audit['blocking_unknowns'])

    def test_forged_component_claim_rejected(self):
        g=self.census['native_coarse_group_catalogue'][0]
        with self.assertRaises(ValueError):self.mapped(g['prim_path'],dict(kind='exact_retained_component',component_index=1))

    def test_unrecognized_coarse_child_is_unknown(self):
        _,audit=self.mapped('/World/PackPlants/P/__FarCoarseGroup_0/fake',dict(kind='coarse_far_group'))
        self.assertEqual(len(audit['blocking_unknowns']),1)

    def test_old_selected_component_path_is_unknown(self):
        _,audit=self.mapped(self.cat[0]['prim_path']+'/mesh',dict(kind='exact_retained_component'))
        self.assertEqual(len(audit['blocking_unknowns']),1)

    def test_retained_exact_path_mask_unchanged(self):
        row=self.cat[1]
        claim=dict(kind='exact_retained_component',component_index=2,variant_id='P',component_id='SubStem_2',source_family='seed7_full',prim_path=row['prim_path'])
        mask,audit=self.mapped(row['prim_path']+'/mesh',claim)
        self.assertEqual(mask.tolist(),[[2]]);self.assertFalse(audit['blocking_unknowns'])

    def test_changed_pose_invalidates_bounds(self):
        groups=self.compile()
        self.assertTrue(a.actual_coarse_bounds(groups,bounds())['all_components_outside_both_actual_arm_bounds'])
        self.assertFalse(a.actual_coarse_bounds(groups,bounds(2.))['all_components_outside_both_actual_arm_bounds'])

    def test_absent_zero_mapping_only_empty_invalid_infinite_background(self):
        ids=np.array([[0]],np.uint32);mapping=dict(renderer_id_to_prim={},plant_ID_ownership={})
        _,audit=a.classify_pixels(ids,mapping,self.cat,self.compile(),np.array([[np.inf]],np.float32),np.array([[False]]))
        self.assertEqual(audit['pixel_categories'],{'empty_background':1})
        _,audit=a.classify_pixels(ids,mapping,self.cat,self.compile(),np.array([[1.]],np.float32),np.array([[True]]))
        self.assertEqual(len(audit['blocking_unknowns']),1)

    def test_unmapped_nonzero_invalid_depth_stays_unknown(self):
        _,audit=a.classify_pixels(np.array([[7]],np.uint32),dict(renderer_id_to_prim={},plant_ID_ownership={}),self.cat,self.compile(),np.array([[np.inf]],np.float32),np.array([[False]]))
        self.assertEqual(len(audit['blocking_unknowns']),1)

    def test_frozen_census_diff_only_schema_literal(self):
        adapted,_=a.adapted_numeric_trees()
        restored=deepcopy(adapted);changes=0
        for node in ast.walk(restored):
            if isinstance(node,ast.Constant) and node.value==a.COVERAGE_SCHEMA:
                node.value='greenhouse.native848_all_petiole_coverage_audit.v1';changes+=1
        self.assertEqual(changes,1)
        self.assertEqual(ast.dump(restored),ast.dump(ast.parse(inspect.getsource(a.ambiguity.dataset.validate_census))))

    def test_frozen_ambiguity_diff_only_census_dispatch(self):
        _,adapted=a.adapted_numeric_trees();restored=deepcopy(adapted);changes=0
        for node in ast.walk(restored):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='validate_farmerge_census':
                node.func=ast.Attribute(value=ast.Name(id='dataset',ctx=ast.Load()),attr='validate_census',ctx=ast.Load());changes+=1
        self.assertEqual(changes,1)
        self.assertEqual(ast.dump(restored),ast.dump(ast.parse(inspect.getsource(a.ambiguity.assess_frame))))

    def replace(self,geo,semantic=True):
        retained=dict(target_id='P/SubStem_2',status='candidate_pending_visual_review',reason='retained_sentinel',automated_pass=True)
        annotation=dict(targets=[dict(target_id='P/SubStem_1',status='excluded',reason='not_visible_zero_authenticated_component_pixels'),retained],target_census={})
        entry=dict(target_id='P/SubStem_1',source_family='seed7_full',source_target_id='seed7_full/SubStem_1',semantic_leaf_petiole_candidate=semantic,anatomy_reason_codes=['deleafed'] if not semantic else [])
        with patch.object(a.coverage,'source_geometry',return_value=geo):
            count=a.merged_rows(annotation,[entry],self.cat,self.compile(),bounds())
        self.assertEqual(count,1);self.assertIs(annotation['targets'][1],retained)
        return annotation

    def test_outside_proof_never_claims_visibility(self):
        row=self.replace(geometry(2.))['targets'][0]
        self.assertEqual(row['reason'],a.coverage.OUTSIDE_REASON)
        self.assertNotIn('visibility',row);self.assertNotIn('junction_continuity',row)
        self.assertIsNone(a.ambiguity.continuously_visible(row))

    def test_reachable_coarse_target_unknown_not_invisible(self):
        result=self.replace(geometry(.1))
        self.assertEqual(result['targets'][0]['status'],'unknown')
        self.assertEqual(result['target_census']['unknown_target_ids'],['P/SubStem_1'])
        self.assertEqual(result['target_census']['candidate_target_ids'],['P/SubStem_2'])

    def test_anatomical_exclusion_never_claims_pixel_absence(self):
        row=self.replace(geometry(2.),semantic=False)['targets'][0]
        self.assertEqual(row['reason'],'not_anatomically_eligible_leaf_petiole')
        self.assertFalse(a.ambiguity.continuously_visible(row))


if __name__=='__main__':unittest.main()
