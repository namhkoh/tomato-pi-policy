"""CPU contract fixtures only; no native execution or acceptance claimed."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from sim_data import native848_persistent_generated_9mm_v1 as api
from sim_data import native848_generated_9mm_v3 as frozen


def fixture():
    p=types.SimpleNamespace(RESULT_SCHEMA='batch',SEGMENT_SCHEMA='segment',OBSERVATION_SCHEMA='observation')
    rp={'path':'request','sha256':'a'};ip={'path':'identity','sha256':'b'}
    scheduled=[];entries=[];values=[];contexts=[];observations=[];sources=[];profiles=[];prev=None
    for i in range(2):
        sid=f's{i:03d}';sr={'path':sid,'sha256':sid};gid=f'newshape{i}';start=i*2
        scheduled.append(dict(segment_id=sid,source_request=sr));profile=dict(warmup_steps=[8],request_subframes=8,delta_time_seconds=0)
        requests=[]
        for j in range(2):
            n=start+j+1
            requests.append(dict(request_index=n,callback_sequence_before=n-1,callback_sequence_after=n,
                native_requests=1,callback_count=1,requested_subframes=8,delta_time_seconds=0,wait_for_render=True,
                timeline_before_seconds=0,timeline_after_seconds=0,reference_time=[n,1],previous_reference_time=None if n==1 else [n-1,1]))
        token=dict(callback_sequence=start+2,reference_time=[start+2,1],camera_sha256='same_across_morphologies',
            rgb_sha256=f'rgb{i}',depth_sha256=f'depth{i}',instance_sha256=f'ids{i}')
        obs=dict(schema=p.OBSERVATION_SCHEMA,segment_id=sid,batch_request=rp,source_request=sr,morphology_id=gid,
            sample_id=f'frame{i}',observation_id=f'frame{i}',synchronization=dict(request_index=start+2,
                callback_sequence=start+2,reference_time=[start+2,1],freshness=token),request_evidence=requests[1])
        receipt=dict(observation_id=f'frame{i}',request_index=start+2)
        value=dict(schema=p.SEGMENT_SCHEMA,segment_id=sid,source_request=sr,batch_request=rp,native_identity=ip,
            state='captured_pending_complete_annotation_and_visual_review',training_approved=False,accepted_training_increment=0,
            original_143_backgrounds_preserved=True,frames=[receipt],holds=[],request_index_start=start,
            callback_sequence_start=start,request_index_end=start+2,callback_sequence_end=start+2,
            request_count=2,requests=requests,morphology_id=gid)
        current=dict(morphology_id=gid);swap=dict(schema='greenhouse.persistent_foreground_transaction.v1',sequence=i+1,
            source_profile={'path':'profile','sha256':'p'},unchanged_background_count=143,original_foreground_restored_before_adapter=True,
            outside_foreground_opinions_preserved=True,fresh_collision_catalogue_static_and_native_evidence_required=True,
            native_capture_validated=False,training_approved=False,accepted_training_increment=0,current_geometry=current,
            previous_geometry=prev or dict(geometry_source_id='donor'),compatibility_sha256='compat')
        ctx=dict(schema=api.CONTEXT_SCHEMA,persistent_batch_authority=dict(batch_request=rp,source_request=sr,segment_id=sid,native_identity=ip),
            persistent_foreground_swap=swap,full_scene_census=dict(foreground_identity=current,unchanged_background_count=143,active_counts=dict(component_plants=144),background_invariance_sha256='bg'),native_population_census_verified=True,original_143_backgrounds_preserved=True,profile=swap['source_profile'],
            geometry_sources={gid:dict(donor_source_family='donor')})
        entries.append(dict(segment_id=sid,frames=1,holds=0));values.append(value);contexts.append(ctx);observations.append([obs]);sources.append(dict(max_frames=1));profiles.append(profile);prev=current
    batch=dict(schema=p.RESULT_SCHEMA,request=rp,native_identity=ip,training_approved=False,accepted_training_increment=0,
        state='captured_pending_complete_annotation_and_visual_review',same_stage_product_writer_for_all_segments=True,
        segments=entries,frames=2,holds=0,request_count=4,callback_count=4,background_closure=dict(schema='greenhouse.persistent_foreground_transaction.v1',sequence=2,compatibility_sha256='compat',unchanged_background_count=143,protected_source_union_rehashed=True,source_file_count=12))
    return [rp,ip,batch,scheduled,entries,values,contexts,observations,sources,profiles,p]


class PersistentContracts(unittest.TestCase):
    def check(self,x):api.validate_registry(*x)
    def reject(self,fn):
        x=fixture();fn(x)
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):self.check(x)
    def test_happy_two_segments_same_camera_different_geometry(self):self.check(fixture())
    def test_wrong_segment(self):self.reject(lambda x:x[5][1].update(segment_id='s000'))
    def test_wrong_native_identity(self):self.reject(lambda x:x[5][1].update(native_identity={'path':'other','sha256':'b'}))
    def test_clock_reset_at_swap(self):self.reject(lambda x:x[5][1].update(request_index_start=0))
    def test_wrong_source_request(self):self.reject(lambda x:x[5][1].update(source_request={'path':'foreign','sha256':'x'}))
    def test_context_other_segment(self):self.reject(lambda x:x[6][1]['persistent_batch_authority'].update(segment_id='s000'))
    def test_swap_chain_wrong(self):self.reject(lambda x:x[6][1]['persistent_foreground_swap'].update(previous_geometry={'morphology_id':'other'}))
    def test_background_swap_false(self):self.reject(lambda x:x[6][1]['persistent_foreground_swap'].update(unchanged_background_count=142))
    def test_reference_regression(self):self.reject(lambda x:x[5][1]['requests'][0].update(reference_time=[0,1]))
    def test_stale_ids(self):self.reject(lambda x:x[7][1][0]['synchronization']['freshness'].update(instance_sha256='ids0'))
    def test_wrong_frame_authority(self):self.reject(lambda x:x[7][1][0].update(source_request=x[3][0]['source_request']))
    def test_missing_registered_segment(self):self.reject(lambda x:x[5].pop())
    def test_wrong_global_aggregate(self):self.reject(lambda x:x[2].update(frames=1))
    def test_closing_source_rehash_required(self):self.reject(lambda x:x[2]['background_closure'].update(protected_source_union_rehashed=False))
    def test_background_identity_chain(self):self.reject(lambda x:x[6][1]['full_scene_census'].update(background_invariance_sha256='different'))
    def test_closing_swap_sequence(self):self.reject(lambda x:x[2]['background_closure'].update(sequence=1))
    def test_numeric_functions_shared_unchanged(self):
        for name in ('evaluate_frame','assess_frame','geometry_9mm','component_masks','build_inventory','validate_planned_geometry'):
            self.assertIs(getattr(api,name),getattr(frozen,name))
    def test_frame_numerical_body_ast_unchanged(self):
        old=Path(frozen.__file__).read_text();new=Path(api.__file__).read_text()
        a=old[old.index('            assert_same_camera('):old.index('        stats=checker.finish()')]
        b=new[new.index('            assert_same_camera('):new.index('        stats=checker.finish()')]
        a=a.replace("result['requests'][sync['request_index']-1]","result['requests'][sync['request_index']-result['request_index_start']-1]")
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
