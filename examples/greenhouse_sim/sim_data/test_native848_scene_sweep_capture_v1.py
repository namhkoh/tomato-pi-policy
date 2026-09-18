"""CPU request/registry mutation tests for a single unchanged greenhouse."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from . import native848_scene_sweep_capture_v1 as p


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    return p.pin(path)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        tmp=self.stack.enter_context(tempfile.TemporaryDirectory(prefix='scene-sweep-test-'))
        self.root=Path(tmp);self.output=self.root/'capture';self.output.mkdir()
        self.stack.enter_context(patch('sim_data.native848_bulk_plan_v1.check_profile'))
        self.matrix=np.eye(4).tolist()
        self.anchor=dict(split='train',source_family='seed19_full')
        self.slots=[dict(plant_root='/World/PackPlants/P'+str(i),variant_id='slot'+str(i),
            source_family='seed19_full',source_plant_id='seed19_full',source_split='train',
            plant_to_world_usd_row_vectors=self.matrix,component_count=1,
            active_after_policy=True,authenticated_component_plant=True) for i in range(144)]
        self.census=dict(policy=p.population.policy('train'),active_counts=dict(component_plants=144,components=144),
            complete_active_plant_anatomy=True,all_plant_roots=self.slots)
        self.records=[]
        for i in range(2):
            camera=np.eye(4);camera[3,0]=i
            self.records.append(dict(sample_id='view'+str(i),plant_root=self.slots[i]['plant_root'],
                source_family='seed19_full',target_id='slot'+str(i)+'/SubStem_42',
                target_variant=dict(plant_root=self.slots[i]['plant_root'],variant_id='slot'+str(i),
                    source_plant_id='seed19_full',split_group='seed19_full',source_geometry_modified=False,added_components={},added_component_paths={}),
                source_row=dict(source_plant_id='seed19_full',component_id='SubStem_42'),
                calibration=dict(camera_to_world_usd_row_vectors=camera.tolist(),width=848,height=408),
                robot_root_to_world_usd_row_vectors=self.matrix,camera_to_head_column_vectors=self.matrix,joint_degrees={},
                source_pose_provenance={},pilot_9mm_geometry=dict(nominal=dict(world_m=[0,0,1],projected=dict(pixel_xy=[400,200])))))
        self.profile=dict(warmup_steps=[8]*7,request_subframes=8,delta_time_seconds=0)
        self.request=dict(schema=p.REQUEST_SCHEMA,implementation_bindings=p.implementation_bindings(),
            scene_anchor=write(self.root/'anchor.json',self.anchor),expected_census=write(self.root/'census.json',self.census),
            profile=write(self.root/'profile.json',self.profile),records=write(self.root/'records.json',dict(records=self.records)),
            scene_policy=p.population.policy('train'),max_frames=2,**p.FLAGS)
        self.proof=dict(schema='greenhouse.native848_fixed_scene_sweep_cpu.v1',passed=True,success=True,selected=2,scene_anchor=self.request['scene_anchor'],profile=self.request['profile'],records=self.request['records'],expected_census=self.request['expected_census'],
            all_selected_pose_screens_passed=True,all_selected_9mm_workspace_checks_passed=True,
            source_bindings={self.request['scene_anchor']['path']:self.request['scene_anchor']['sha256']},per_record=[dict(sample_id=r['sample_id'],fk_calibration_passed=True,whole_robot_collision_passed=True,workspace_passed=True) for r in self.records],poses=[dict(sample_id=r['sample_id'],passed=True) for r in self.records])
        self.request['cpu_preflight']=write(self.root/'cpu.json',self.proof)
        self.request_pin=write(self.root/'request.json',self.request)

    def save_records(self):
        self.request['records']=write(self.root/'records.json',dict(records=self.records))
        self.proof['records']=self.request['records']
        self.request['cpu_preflight']=write(self.root/'cpu.json',self.proof)

    def save_census(self):
        self.request['expected_census']=write(self.root/'census.json',self.census)
        self.proof['expected_census']=self.request['expected_census']
        self.request['cpu_preflight']=write(self.root/'cpu.json',self.proof)


class RequestTests(Fixture):
    def test_two_different_instance_targets_in_one_scene(self):
        self.assertEqual(len(p.check_request(self.request)['records']),2)

    def test_changed_pinned_record_bytes_rejected(self):
        (self.root/'records.json').write_text('{}')
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_duplicate_camera_rejected_even_for_another_target(self):
        self.records[1]['calibration']=deepcopy(self.records[0]['calibration']);self.save_records()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_foreign_instance_root_rejected(self):
        self.records[0]['plant_root']='/World/PackPlants/Other';self.save_records()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_same_variant_id_cannot_claim_another_family(self):
        self.records[0]['target_variant']['source_plant_id']='seed11_full';self.save_records()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_compound_target_id_must_match_instance_and_component(self):
        self.records[0]['target_id']='slot1/SubStem_42';self.save_records()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_complete_census_cannot_duplicate_or_drop_slots(self):
        self.census['all_plant_roots'][-1]=deepcopy(self.census['all_plant_roots'][0]);self.save_census()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_cross_split_slot_rejected(self):
        self.census['all_plant_roots'][0]['source_split']='test';self.save_census()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_nonrigid_embodied_camera_pose_rejected(self):
        self.records[0]['robot_root_to_world_usd_row_vectors']=np.zeros((4,4)).tolist();self.save_records()
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_cpu_result_from_other_schedule_rejected(self):
        self.proof['records']=write(self.root/'other_records.json',dict(records=[]))
        self.request['cpu_preflight']=write(self.root/'cpu.json',self.proof)
        with self.assertRaises(ValueError):p.check_request(self.request)

    def test_raw_request_cannot_claim_unique_answer_or_acceptance(self):
        self.request['single_answer_verified']=True
        with self.assertRaises(ValueError):p.check_request(self.request)


class ResultTests(Fixture):
    def setUp(self):
        super().setUp()
        identity=write(self.output/'native_identity.json',dict(owner='synthetic'))
        self.context=dict(schema=p.CONTEXT_SCHEMA,request=self.request_pin,native_identity=identity,
            source_bindings={**self.request['implementation_bindings'],**{self.request[k]['path']:self.request[k]['sha256'] for k in ('scene_anchor','expected_census','profile','records','cpu_preflight')}},scene_anchor=self.request['scene_anchor'],profile=self.request['profile'],full_scene_census=self.census,census=write(self.output/'census.json',self.census),catalogue=write(self.output/'catalogue.json',[]),reports=write(self.output/'reports.json',[]),**p.FLAGS)
        self.context_pin=write(self.output/'context.json',self.context)
        self.events=[dict(request_index=i,callback_sequence_before=i-1,callback_sequence_after=i,
            callback_count=1,native_requests=1,requested_subframes=8,delta_time_seconds=0,wait_for_render=True,callbacks=[dict(callback_sequence=i,request_index=i)]) for i in range(1,9)]
        self.obs=dict(schema=p.OBSERVATION_SCHEMA,observation_id='view0',sample_id='view0',context=self.context_pin,
            source_family=self.records[0]['source_family'],target_variant=self.records[0]['target_variant'],
            target_id=self.records[0]['target_id'],plant_root=self.records[0]['plant_root'],
            request_evidence=self.events[-1],synchronization=dict(request_index=8,callback_sequence=8),calibration=deepcopy(self.records[0]['calibration']),robot_snapshot=dict(joint_degrees={},robot_root_to_world_usd_row_vectors=self.matrix,camera_to_head_column_vectors=self.matrix,visual_bound_screen=dict(passed=True),whole_robot_collision_checked=True),**p.FLAGS)
        receipt=write(self.output/'frames/view0/observation.json',self.obs);receipt.update(observation_id='view0',request_index=8)
        self.result=dict(schema=p.RESULT_SCHEMA,request=self.request_pin,native_identity=identity,context=self.context_pin,
            stage_count=1,render_product_count=1,scene_assets_unchanged=True,frames=[receipt],
            holds=[dict(sample_id='view1',reason='whole_robot_scene_collision',screen=dict(passed=False))],requests=self.events,
            warmup_requests=deepcopy(self.events[:7]),request_count=8,callback_count=8,**p.FLAGS)

    def save_obs(self):
        receipt=self.result['frames'][0];receipt.update(write(Path(receipt['path']),self.obs))
        receipt.update(observation_id=self.obs['observation_id'],request_index=self.obs['synchronization']['request_index'])

    def validate(self):return p.validate_result(self.result,self.root/'request.json',self.output)

    def test_valid_captured_plus_collision_hold_accounting(self):
        self.assertEqual(self.validate(),self.context)

    def test_callback_gap_rejected(self):
        self.events[-1]['callback_sequence_after']=10
        with self.assertRaises(ValueError):self.validate()

    def test_foreign_or_duplicate_hold_cannot_complete_schedule(self):
        self.result['holds'][0]['sample_id']='view0'
        with self.assertRaises(ValueError):self.validate()

    def test_frame_cannot_borrow_warmup_callback(self):
        self.obs['synchronization'].update(request_index=1,callback_sequence=1);self.obs['request_evidence']=self.events[0];self.save_obs()
        with self.assertRaises(ValueError):self.validate()

    def test_frame_request_evidence_must_equal_ledger(self):
        self.obs['request_evidence']=dict(self.events[-1],callback_count=2);self.save_obs()
        with self.assertRaises(ValueError):self.validate()

    def test_frame_cannot_claim_acceptance(self):
        self.obs['training_approved']=True;self.save_obs()
        with self.assertRaises(ValueError):self.validate()

    def test_foreign_context_request_rejected(self):
        self.context['request']=write(self.root/'another_request.json',{})
        self.result['context']=write(self.output/'context.json',self.context)
        self.obs['context']=self.result['context'];self.save_obs()
        with self.assertRaises(ValueError):self.validate()

    def test_frame_cannot_change_its_target_instance(self):
        self.obs['target_variant']=deepcopy(self.records[1]['target_variant']);self.save_obs()
        with self.assertRaises(ValueError):self.validate()

    def test_result_cannot_claim_unique_answer(self):
        self.result['single_answer_verified']=True
        with self.assertRaises(ValueError):self.validate()


    def test_changed_native_authority_rejected(self):
        self.result['native_identity']=write(self.root/'unrelated_identity.json',{})
        with self.assertRaises(ValueError):self.validate()

    def test_changed_source_after_capture_rejected(self):
        (self.root/'anchor.json').write_text('{}')
        with self.assertRaises(ValueError):self.validate()

    def test_changed_embodied_calibration_rejected(self):
        self.obs['calibration']['camera_to_world_usd_row_vectors'][3][0]+=0.1;self.save_obs()
        with self.assertRaises(ValueError):self.validate()

    def test_shortened_warmup_or_wrong_subframes_rejected(self):
        for kind in ('warmup','production'):
            original=deepcopy(self.result)
            if kind=='warmup':self.result['warmup_requests'].pop()
            else:self.result['requests'][-1]['requested_subframes']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate()
            self.result=original


class LifecycleTests(Fixture):
    def test_all_collision_holds_cleanly_release_single_product_writer(self):
        from . import native848_fully_labeled_coverage_v3 as coverage
        from . import native848_original_direct_worker_v2 as direct
        from . import static_geometry_parts_cache_v1 as cache
        from . import capture_pilot
        events=[];omni=types.ModuleType('omni');omni.__path__=[]
        usd=types.ModuleType('omni.usd');rep_pkg=types.ModuleType('omni.replicator');rep_pkg.__path__=[]
        rep=types.ModuleType('omni.replicator.core');omni.usd=usd;omni.replicator=rep_pkg;rep_pkg.core=rep
        product=types.SimpleNamespace(destroy=lambda:events.append('destroy'))
        rep.create=types.SimpleNamespace(render_product=lambda *args:events.append('product') or product)
        writer=types.SimpleNamespace(attach=lambda value:events.append('attach'),detach=lambda:events.append('detach'))
        class Geometry:
            def __call__(self):return dict(passed=False)
            def close(self):events.append('geometry_close')
        class Sink:
            def close(self):return []
        scene=dict(scene_policy_evidence=self.census,stage=object(),robot=dict(root='/Robot'),settings={},records=[],
            reports=[],variants=[],original_population_bindings={})
        checked=dict(anchor=self.anchor,census=self.census,profile=dict(self.profile,source_bindings={}),
            records=self.records,cpu_preflight=dict(source_bindings={}))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,{'omni':omni,'omni.usd':usd,'omni.replicator':rep_pkg,'omni.replicator.core':rep}))
            stack.enter_context(patch.object(p.population,'prepare_native_scene',side_effect=lambda *a:events.append('scene') or scene))
            stack.enter_context(patch.object(p.sensor,'apply_profile',return_value={}))
            stack.enter_context(patch.object(p.sensor,'make_bulk_writer',return_value=writer))
            stack.enter_context(patch.object(p.sensor,'request_once',side_effect=AssertionError('Held poses must not render')))
            stack.enter_context(patch.object(coverage,'component_catalogue',return_value=[]))
            stack.enter_context(patch.object(capture_pilot,'source_hashes',return_value={}))
            stack.enter_context(patch.object(direct,'apply_cached_pose',side_effect=lambda *a:events.append('pose') or ({},{})))
            stack.enter_context(patch.object(cache,'StaticGeometryPartsCache',return_value=Geometry()))
            stack.enter_context(patch.object(p,'FrameSink',return_value=Sink()))
            with self.assertRaisesRegex(ValueError,'No collision-free'):
                p.capture(None,self.request,checked,self.output,request_pin=self.request_pin)
        self.assertEqual(events,['scene','product','attach','pose','pose','geometry_close','detach','destroy'])
        self.assertFalse((self.output/'result.json').exists())

    def test_failed_actual_child_cannot_create_owner_complete(self):
        from contextlib import nullcontext
        from .native_generated_reference import owner_v1 as owner
        from .native_original_capture import serial_queue
        args=types.SimpleNamespace(request=self.root/'request.json',request_sha256=self.request_pin['sha256'],output=self.root/'owned')
        events=[]
        def failed(command,folder,environment):
            events.append('returned_child_exit1');write(folder/'owned_worker.json',dict(launcher_pid=123));return 1
        with ExitStack() as stack:
            stack.enter_context(patch.object(p,'check_request',return_value={}))
            stack.enter_context(patch.object(owner,'owner_lock',return_value=nullcontext()))
            stack.enter_context(patch.object(owner,'resources',return_value={}))
            stack.enter_context(patch.object(owner,'_run_child',side_effect=failed))
            stack.enter_context(patch.object(serial_queue.v5,'worker_environments',return_value=({},{})))
            with self.assertRaisesRegex(ValueError,'did not close cleanly'):p.owned_run(args)
        self.assertEqual(events,['returned_child_exit1'])
        self.assertEqual(json.loads((args.output/'owned_exit.json').read_text())['returncode'],1)
        self.assertFalse((args.output/'owner_complete.json').exists())


class ImportTests(unittest.TestCase):
    def test_native_admission_import_does_not_load_usd_or_start_app(self):
        environment=dict(os.environ);environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        code=('import sys,json;from sim_data import native848_scene_sweep_capture_v1;'
              'from sim_data.native_generated_pair import windows_worker_admission;'
              'print(json.dumps([x for x in ("pxr.Usd","omni.usd","isaacsim") if x in sys.modules]))')
        run=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=environment,check=True)
        self.assertEqual(json.loads(run.stdout),[])


if __name__=='__main__':unittest.main(verbosity=2)
