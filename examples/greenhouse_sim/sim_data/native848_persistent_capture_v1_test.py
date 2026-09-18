"""CPU-only persistent producer bounds, registry mutation and lifecycle tests."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import native848_persistent_capture_v1 as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return p.pin(path)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        scene=str(Path('test-scene.json').resolve())
        self.anchor=dict(original_variant=dict(variant_id='original',plant_root='/World/PackPlants/P'),
            source_collection_plan=scene,source_bindings={scene:'scene-sha'})
        self.candidate=dict(source_family='seed19_full',source_plan=dict(path='plan',sha256='plan-sha'),
            profile=dict(path='profile',sha256='profile-sha'),records=[dict(anchor=dict(path='anchor',sha256='anchor-sha'))])
        self.objects={'anchor':self.anchor,'source':dict(max_frames=1,candidates=self.candidate)}
        self.stack.enter_context(patch.object(p,'implementation_bindings',return_value={'implementation':'sha'}))
        self.stack.enter_context(patch.object(p,'verify_bindings'))
        self.stack.enter_context(patch.object(p,'bound',side_effect=lambda s:self.objects[s['path']]))
        self.stack.enter_context(patch.object(p.baseline,'check_request',side_effect=lambda s:s['candidates']))

    def request(self,n=2):
        return p.make_request([dict(path='source',sha256='sha') for _ in range(n)])

    def test_valid_repeat_geometry_diagnostic_is_explicitly_segmented(self):
        request=self.request(3)
        self.assertEqual([s['segment_id'] for s in request['segments']],['s000','s001','s002'])
        self.assertEqual(len(p.check_request(request)),3)

    def test_segment_and_total_bounds(self):
        for count in (0,17):
            with self.assertRaises(ValueError):self.request(count)
        self.objects['source']['max_frames']=128
        with self.assertRaisesRegex(ValueError,'total frame'):self.request(9)
        self.assertEqual(self.request(8)['max_frames'],1024)

    def test_boolean_total_is_rejected(self):
        request=self.request(1);request['max_frames']=True
        with self.assertRaises(ValueError):p.check_request(request)

    def test_canonical_unique_segment_order(self):
        for value in ('s009','../s000','s000'):
            request=self.request();request['segments'][1]['segment_id']=value
            with self.assertRaises(ValueError):p.check_request(request)

    def test_incompatible_donor_profile_slot_and_scene(self):
        for field in ('source_family','profile','original_variant','source_collection_plan'):
            candidate=deepcopy(self.candidate);anchor=deepcopy(self.anchor)
            candidate['records'][0]['anchor']['path']='anchor2'
            if field=='source_family':candidate[field]='seed67_full'
            elif field=='profile':candidate[field]=dict(path='another',sha256='sha')
            elif field=='original_variant':anchor[field]['plant_root']='/World/PackPlants/Q'
            else:
                scene=str(Path('another-scene.json').resolve());anchor[field]=scene;anchor['source_bindings'][scene]='sha'
            self.objects['anchor2']=anchor;self.objects['source2']=dict(max_frames=1,candidates=candidate)
            with self.assertRaisesRegex(ValueError,'same donor/slot/scene/profile'):
                p.make_request([dict(path='source',sha256='sha'),dict(path='source2',sha256='sha')])


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='persistent-registry-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.capture=self.root/'capture';self.capture.mkdir()
        self.identity=write(self.capture/'native_identity.json',dict(owner='synthetic'))
        profile=write(self.root/'profile.json',dict(warmup_steps=[8]*7))
        self.sources=[];scheduled=[]
        for i in range(2):
            ids=['frame'+str(i),'held'+str(i)]
            candidates=write(self.root/('candidates%d.json'%i),dict(profile=profile,records=[dict(sample_id=s) for s in ids]))
            source=write(self.root/('source%d.json'%i),dict(max_frames=2,candidates=candidates))
            self.sources.append(source);scheduled.append(dict(segment_id='s%03d'%i,source_request=source))
        self.request=dict(schema=p.REQUEST_SCHEMA,segments=scheduled,max_frames=4,training_approved=False)
        self.request_pin=write(self.root/'request.json',self.request)
        self.values=[];self.observations=[];self.registered=[]
        for i in range(2):
            sid='s%03d'%i;folder=self.capture/'segments'/sid/'capture';folder.mkdir(parents=True)
            events=[dict(request_index=i*8+j+1,callback_sequence_before=i*8+j,
                callback_sequence_after=i*8+j+1,callback_count=1) for j in range(8)]
            context=write(folder/'context.json',dict(persistent_batch_authority=dict(batch_request=self.request_pin,
                source_request=self.sources[i],segment_id=sid,native_identity=self.identity)))
            obs=dict(schema=p.OBSERVATION_SCHEMA,segment_id=sid,morphology_id='morph'+str(i),training_approved=False,sample_id='frame'+str(i),observation_id='frame'+str(i),
                batch_request=self.request_pin,source_request=self.sources[i],context=context,
                request_evidence=events[-1],synchronization=dict(request_index=i*8+8,callback_sequence=i*8+8))
            path=folder/'frames'/obs['sample_id']/'observation.json';receipt=write(path,obs)
            receipt.update(observation_id=obs['observation_id'],request_index=i*8+8)
            value=dict(schema=p.SEGMENT_SCHEMA,segment_id=sid,morphology_id='morph'+str(i),source_request=self.sources[i],batch_request=self.request_pin,
                native_identity=self.identity,training_approved=False,original_143_backgrounds_preserved=True,
                request_index_start=i*8,request_index_end=i*8+8,callback_sequence_start=i*8,callback_sequence_end=i*8+8,
                request_count=8,requests=events,frames=[receipt],holds=[dict(sample_id='held'+str(i),reason='whole_robot_scene_collision')])
            self.values.append(value);self.observations.append(obs)
            self.registered.append(dict(segment_id=sid,result=write(folder/'result.json',value),frames=1,holds=1))
        self.result=dict(schema=p.RESULT_SCHEMA,request=self.request_pin,native_identity=self.identity,
            training_approved=False,same_stage_product_writer_for_all_segments=True,segments=self.registered,
            frames=2,holds=2,request_count=16,callback_count=16,
            background_closure=dict(schema='greenhouse.persistent_foreground_transaction.v1',sequence=2,
                unchanged_background_count=143,protected_source_union_rehashed=True,compatibility_sha256='c'*64))

    def save(self,index):
        self.registered[index]['result']=write(Path(self.registered[index]['result']['path']),self.values[index])

    def save_obs(self,index):
        receipt=self.values[index]['frames'][0]
        receipt.update(write(Path(receipt['path']),self.observations[index]));self.save(index)

    def validate(self):
        return p.validate_batch_registry(self.result,self.request,self.capture,
            request_pin=self.request_pin,native_identity=self.identity)

    def test_two_segment_global_bookkeeping_positive(self):
        self.assertEqual(self.validate(),dict(frames=2,holds=2,requests=16,callbacks=16))

    def test_swapped_registry_entries_rejected(self):
        self.result['segments']=list(reversed(self.result['segments']))
        with self.assertRaises(ValueError):self.validate()

    def test_foreign_segment_authority_rejected(self):
        for field in ('source_request','batch_request','native_identity'):
            saved=deepcopy(self.values[1]);self.values[1][field]=dict(path='other',sha256='bad');self.save(1)
            with self.assertRaises(ValueError):self.validate()
            self.values[1]=saved;self.save(1)

    def test_discontinuous_global_counters_rejected(self):
        for field in ('request_index_start','callback_sequence_start','request_index_end','callback_sequence_end'):
            saved=self.values[1][field];self.values[1][field]+=1;self.save(1)
            with self.assertRaises(ValueError):self.validate()
            self.values[1][field]=saved;self.save(1)

    def test_wrong_callback_event_rejected(self):
        self.values[1]['requests'][0]['callback_count']=2;self.save(1)
        with self.assertRaises(ValueError):self.validate()

    def test_frame_outside_segment_rejected(self):
        self.values[1]['frames'][0]=deepcopy(self.values[0]['frames'][0]);self.save(1)
        with self.assertRaises(ValueError):self.validate()

    def test_frame_file_byte_mutation_rejected(self):
        Path(self.values[0]['frames'][0]['path']).write_text('{}')
        with self.assertRaises(ValueError):self.validate()

    def test_duplicate_frame_receipt_rejected(self):
        self.values[0]['frames'].append(deepcopy(self.values[0]['frames'][0]));self.values[0]['holds']=[]
        self.registered[0].update(frames=2,holds=0);self.result.update(frames=3,holds=1);self.save(0)
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_observation_segment_or_request_rejected(self):
        for field,bad in [('segment_id','s999'),('source_request',self.sources[1]),('batch_request',dict(path='other',sha256='bad'))]:
            saved=deepcopy(self.observations[0]);self.observations[0][field]=bad;self.save_obs(0)
            with self.assertRaises(ValueError):self.validate()
            self.observations[0]=saved;self.save_obs(0)

    def test_observation_callback_must_match_its_event(self):
        self.observations[1]['synchronization']['request_index']=8;self.save_obs(1)
        with self.assertRaises(ValueError):self.validate()

    def test_background_closure_and_batch_totals_rejected(self):
        for field,bad in [('unchanged_background_count',142),('protected_source_union_rehashed',False),('sequence',1)]:
            saved=self.result['background_closure'][field];self.result['background_closure'][field]=bad
            with self.assertRaises(ValueError):self.validate()
            self.result['background_closure'][field]=saved
        self.result['frames']=3
        with self.assertRaises(ValueError):self.validate()


class LifecycleTests(unittest.TestCase):
    def execute(self,fail_second=False):
        from . import native848_fully_labeled_scene_v1 as full
        from . import native848_persistent_foreground_v1 as helper
        events=[];tmp=tempfile.TemporaryDirectory(prefix='persistent-lifecycle-');self.addCleanup(tmp.cleanup)
        output=Path(tmp.name);write(output/'native_identity.json',{})
        omni=types.ModuleType('omni');omni.__path__=[]
        pkg=types.ModuleType('omni.replicator');pkg.__path__=[]
        rep=types.ModuleType('omni.replicator.core');omni.replicator=pkg;pkg.core=rep
        product=types.SimpleNamespace(destroy=lambda:events.append('destroy_product'))
        rep.create=types.SimpleNamespace(render_product=lambda *a:events.append('create_product') or product)
        writer=types.SimpleNamespace(active=False,capture_error=None,request_index=0,sequence=0,
            attach=lambda x:events.append('attach'),detach=lambda:events.append('detach'))
        stage=types.SimpleNamespace(GetUsedLayers=lambda:[])
        scene=dict(stage=stage)
        session=types.SimpleNamespace(verify_backgrounds=lambda:events.append('close_background') or
            dict(sequence=2,unchanged_background_count=143,protected_source_union_rehashed=True))
        request=dict(segments=[dict(segment_id='s000',source_request={}),dict(segment_id='s001',source_request={})],
            max_frames=2,implementation_bindings={})
        candidates=dict(records=[dict(anchor={})],profile={})
        calls=[]
        def segment(*args,**kwargs):
            i=len(calls);calls.append(kwargs)
            self.assertIs(kwargs['writer'],writer)
            self.assertEqual(kwargs['previous'],None if i==0 else {'last':'s000'})
            self.assertIs(kwargs['anchor_cache'],calls[0]['anchor_cache'])
            if i==1 and fail_second:raise ValueError('second segment failed')
            writer.request_index+=8;writer.sequence+=8
            events.append('segment'+str(i)+'_drained')
            return dict(segment_id=kwargs['segment_id'],frames=1,holds=0),{'last':kwargs['segment_id']}
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules',{'omni':omni,'omni.replicator':pkg,'omni.replicator.core':rep}))
            stack.enter_context(patch.object(full,'prepare_native_scene',side_effect=lambda *a:events.append('prepare_scene') or scene))
            stack.enter_context(patch.object(helper,'snapshot_original',side_effect=lambda *a,**k:events.append('snapshot') or session))
            stack.enter_context(patch.object(p.primitives,'make_bulk_writer',return_value=writer))
            stack.enter_context(patch.object(p,'bound',return_value={}))
            stack.enter_context(patch.object(p,'capture_segment',side_effect=segment))
            stack.enter_context(patch.object(p,'verify_bindings'))
            stack.enter_context(patch.object(p,'validate_batch_registry',return_value={}))
            if fail_second:
                with self.assertRaisesRegex(ValueError,'second segment failed'):
                    p.capture(None,request,[({},candidates),({},candidates)],output,request_pin={})
            else:p.capture(None,request,[({},candidates),({},candidates)],output,request_pin={})
        return events,output,writer

    def test_one_scene_product_writer_with_continuous_cross_segment_token(self):
        events,output,writer=self.execute()
        self.assertEqual(events,['prepare_scene','create_product','attach','snapshot','segment0_drained',
            'segment1_drained','close_background','detach','destroy_product'])
        self.assertEqual((writer.request_index,writer.sequence),(16,16))
        self.assertTrue((output/'result.json').exists())

    def test_second_segment_failure_still_detaches_and_destroys_once(self):
        events,output,writer=self.execute(True)
        self.assertEqual(events[-2:],['detach','destroy_product'])
        self.assertNotIn('close_background',events)
        self.assertFalse((output/'result.json').exists())


class NativeBoundaryTests(unittest.TestCase):
    def test_fresh_import_through_worker_admission_loads_no_usd_abi(self):
        import os,subprocess,sys
        env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        code=("import sys,json; from sim_data import native848_persistent_capture_v1; "
              "from sim_data.native_generated_pair import windows_worker_admission; "
              "print(json.dumps([n for n in sys.modules if n in ('pxr.Usd','omni.usd','isaacsim')]))")
        run=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=env,check=True)
        self.assertEqual(json.loads(run.stdout),[])

    def test_owner_command_has_no_generic_registry_exemption(self):
        import os,subprocess,sys
        if os.name!='nt':self.skipTest('Windows command identity parser')
        from .native_dataset import native_process_guard as guard
        row=dict(ProcessId=123,ParentProcessId=122,CreationDate='2026-09-18T16:00:00Z',Name='python.exe',
            ExecutablePath=sys.executable,CommandLine=subprocess.list2cmdline([sys.executable,'-B','-u','-m',p.MODULE,
            '--request','D:/data/persistent19/request.json','--request-sha256','0'*64,'--output','C:/data/persistent19']))
        result=guard.classify_process(row,native_roots=guard.DEFAULT_NATIVE_ROOTS)
        self.assertFalse(result['blocking']);self.assertEqual(result['classification'],'unrelated_python')
        row['ExecutablePath']='D:/isaac-sim-6.0.1/kit/python/kit.exe';row['Name']='kit.exe'
        self.assertTrue(guard.classify_process(row,native_roots=guard.DEFAULT_NATIVE_ROOTS)['blocking'])


if __name__=='__main__':unittest.main(verbosity=2)
