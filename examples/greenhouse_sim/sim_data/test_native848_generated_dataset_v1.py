"""Portable rejection tests; synthetic records never constitute native QA."""
import copy, unittest
from . import native848_generated_dataset_v1 as m

def fixture():
    donor='seed101_full';gid='generated_shape';vid='background_instance'
    variants={gid:dict(source_plant_id=donor,split_group=donor,geometry_source_id=gid,source_geometry_modified=True),
        vid:dict(source_plant_id=donor,split_group=donor,geometry_source_id=donor,source_geometry_modified=False)}
    catalogue=[];rows=[]
    for instance,status in [(gid,'candidate_pending_visual_review'),(vid,'excluded')]:
        v=variants[instance];geometry=v['geometry_source_id'];cid='SubStem_41'
        catalogue.append(dict(variant_id=instance,component_id=cid,organ_type='sub_stem',source_plant_id=donor,split_group=donor,geometry_source_id=geometry))
        rows.append(dict(target_id=instance+'/'+cid,source_family=donor,source_component_id=cid,source_target_id=donor+'/'+cid,
            plant_instance_id=instance,geometry_source_id=geometry,geometry_target_id=geometry+'/'+cid,
            annotation_epoch=m.base.EPOCH,generated_geometry=v['source_geometry_modified'],morphology_id=gid if instance==gid else None,
            status=status,reason='9mm_local_visibility_and_workspace_passed' if instance==gid else 'not_visible_zero_authenticated_component_pixels'))
    primary,other=rows
    census=dict(complete=True,catalogue_complete=True,catalogue_petiole_ids=[r['target_id'] for r in rows],
        candidate_target_ids=[primary['target_id']],excluded_target_ids=[other['target_id']],unknown_target_ids=[])
    ann=dict(schema=m.evaluated.SCHEMA,annotation_epoch=m.base.EPOCH,numerical_predicate_epoch=m.base.EPOCH,frame_id='f',
        targets=rows,target_census=census,frame_blocked_by_unknown_targets=False,frame_blocked_by_unverified_full_scene_coverage=False,banned_candidate_instance_ids=[])
    amb=dict(schema=m.evaluated.AMBIGUITY_SCHEMA,frame_id='f',strict_label_statuses_preserved=True,assessment_complete=True,
        single_answer_unambiguous=True,decision='candidate_pending_actual_visual_review',strict_candidate_ids=[primary['target_id']],
        primary_target_id=primary['target_id'],reachable_alternative_ids=[],unknown_alternative_ids=[],continuously_visible_alternative_ids=[],
        alternative_assessments=[dict(target_id=other['target_id'],source_family=donor,original_strict_status='excluded',original_strict_reason=other['reason'],
            strict_eligibility_promoted=False,continuous_nominal_junction_visible=False,assessment='objective_anatomy_or_native_visibility_exclusion')])
    return ann,catalogue,variants,amb

class PartitionTests(unittest.TestCase):
    def test_generated_identity_keeps_donor(self):
        a,c,v,b=fixture();r=m.validate_partition(a,c,v,b);self.assertNotEqual(r['source_family'],r['geometry_source_id'])
    def test_missing_petiole(self):
        a,c,v,b=fixture();a['targets'].pop()
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_duplicate_petiole(self):
        a,c,v,b=fixture();a['targets'].append(copy.deepcopy(a['targets'][0]))
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_second_strict_candidate(self):
        a,c,v,b=fixture();a['targets'][1]['status']='candidate_pending_visual_review';a['target_census']['candidate_target_ids'].append(a['targets'][1]['target_id']);a['target_census']['excluded_target_ids']=[]
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_unknown_not_negative(self):
        a,c,v,b=fixture();a['targets'][1]['status']='unknown';a['target_census']['excluded_target_ids']=[];a['target_census']['unknown_target_ids']=[a['targets'][1]['target_id']]
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_clone_cannot_change_donor(self):
        a,c,v,b=fixture();a['targets'][1]['source_family']='generated_shape'
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_generated_cannot_impersonate_original_geometry(self):
        a,c,v,b=fixture();a['targets'][0]['geometry_source_id']='seed101_full'
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_missing_alternative(self):
        a,c,v,b=fixture();b['alternative_assessments']=[]
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)
    def test_reachable_alternative(self):
        a,c,v,b=fixture();b['reachable_alternative_ids']=[a['targets'][1]['target_id']]
        with self.assertRaises(ValueError):m.validate_partition(a,c,v,b)

class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.alloc={'seed101_full':'train','seed31_full':'test'}
        self.q=dict(schema='greenhouse.multisubtree_controlled_qualification.v4',source_family='seed101_full',split_group='seed101_full',
            variant_id='generated_shape',split='train',frozen_family_assignments=self.alloc.copy(),source_assets_unchanged=True,
            new_donor_family_created=False,source_cap_reset=False)
    def test_actual_v4_split_field(self):
        m.validate_qualification(self.q,'seed101_full','generated_shape',self.alloc)
    def test_wrong_split(self):
        self.q['split']='test'
        with self.assertRaises(ValueError):m.validate_qualification(self.q,'seed101_full','generated_shape',self.alloc)
    def test_donor_relabel(self):
        self.q['split_group']='generated_shape'
        with self.assertRaises(ValueError):m.validate_qualification(self.q,'seed101_full','generated_shape',self.alloc)
    def test_frozen_allocation_change(self):
        self.q['frozen_family_assignments']['seed31_full']='train'
        with self.assertRaises(ValueError):m.validate_qualification(self.q,'seed101_full','generated_shape',self.alloc)


class ReviewAndAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.p={'path':'p','sha256':'0'*64};self.record={'sample_id':'f','annotation':self.p}
        self.row=dict(sample_id='f',annotation=self.p,rgb=self.p,decision='accept',actual_full_native_rgb_viewed=True,
            actual_native_junction_crop_viewed=True,actual_first_leaf_context_viewed=True,actual_flagged_contexts_viewed=True,reason='Actual reviewer explanation')
        self.review=dict(schema=m.REVIEW_SCHEMA,evaluation=self.p,reviewer='assistant_test_fixture',human_review_claimed=False,frames=[self.row])
    def test_review_requires_leaf_context(self):
        self.row['actual_first_leaf_context_viewed']=False
        with self.assertRaises(ValueError):m.validate_review(self.review,self.p,self.record,self.p)
    def test_hold_never_exported(self):
        self.row['decision']='hold'
        with self.assertRaises(ValueError):m.validate_review(self.review,self.p,self.record,self.p)
    def test_review_cannot_bind_different_rgb(self):
        with self.assertRaises(ValueError):m.validate_review(self.review,self.p,self.record,dict(path='different',sha256='1'*64))
    def test_exact_review_passes(self):
        self.assertEqual(m.validate_review(self.review,self.p,self.record,self.p),self.row)
    def test_nonzero_outer_exit_rejected(self):
        auth=dict(schema=m.AUTH_SCHEMA,evaluation=self.p,review=self.p,selected_sample_ids=['f'],exporter_sha256='1'*64,
            dataset_only=True,root_authorized=True,training_authorized=False,blocking_findings=[],native_outer_wait=dict(session_id=1,returncode=1),annotation_outer_wait=dict(session_id=2,returncode=0))
        with self.assertRaises(ValueError):m.validate_authorization(auth,self.p,['f'],self.p,'1'*64)
    def test_hidden_query_rejected_before_read(self):
        with self.assertRaises(ValueError):m.load_model_inputs(dict(schema=m.FRAME_SCHEMA,task_id=m.base.TASK,inputs={'query':{}}),'.')
    def test_original_ds8_schema_rejected(self):
        with self.assertRaises(ValueError):m.load_model_inputs(dict(schema=m.base.FRAME_SCHEMA,task_id=m.base.TASK,inputs={}),'.')

if __name__=='__main__':unittest.main()
