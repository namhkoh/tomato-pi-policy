"""CPU-only bounded warmup candidate/control tests with real pinned file graphs."""
from copy import deepcopy
import hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from sim_data import native848_persistent_experimental_plan_v1 as policy
from sim_data import native848_persistent_experimental_capture_v1 as producer
from sim_data import native848_persistent_experimental_9mm_v1 as consumer
from sim_data import native848_generated_9mm_v3 as frozen
from sim_data.test_native848_persistent_shortwarm_v1 import fixture as previous_fixture


def fixture():
    x=previous_fixture();cursor=0;prior=None;counts=(4,4,3,1)
    for i,nframes in enumerate(counts):
        spec=x[3][i];value=x[5][i];ctx=x[6][i];profile=x[9][i];k=i if i<3 else 0
        ids=[f'camera{k}_{j}' for j in range(nframes)];gid=f'shape{k}'
        spec.pop('selected_sample_id');spec.update(source_request=dict(path=f'source{k}',sha256=f'sh{k}'),selected_sample_ids=ids,
            warmup_request_count=policy.COUNTS[i],purpose=policy.CANDIDATE if i<3 else policy.CONTROL,
            paired_control_reference=None if i<3 else dict(segment_id='s000',sample_id=ids[0]))
        profile.pop('diagnostic_only');profile.update(schema=policy.ACTUAL_SCHEMA,warmup_steps=[8]*policy.COUNTS[i],experimental_profile=True,shortened_warmup_qualified=False)
        events=[]
        for j in range(policy.COUNTS[i]+nframes):
            number=cursor+j+1
            events.append(dict(request_index=number,callback_sequence_before=number-1,callback_sequence_after=number,native_requests=1,callback_count=1,
                requested_subframes=8,delta_time_seconds=0,wait_for_render=True,timeline_before_seconds=0,timeline_after_seconds=0,
                reference_time=[number,1],previous_reference_time=None if number==1 else [number-1,1]))
        purpose=dict(capture_purpose=spec['purpose'],paired_control_reference=spec['paired_control_reference'],export_prohibited=i==3)
        observations=[];receipts=[]
        for j,name in enumerate(ids):
            obs=deepcopy(x[7][i][0]);number=cursor+policy.COUNTS[i]+j+1
            obs.pop('diagnostic_only');obs.update(schema=producer.OBSERVATION_SCHEMA,source_request=spec['source_request'],morphology_id=gid,
                sample_id=name,observation_id=name,request_evidence=events[policy.COUNTS[i]+j],experimental_profile=True,**purpose)
            token=dict(callback_sequence=number,reference_time=[number,1],camera_sha256=name,rgb_sha256=f'rgb{i}_{j}',depth_sha256=f'depth{i}_{j}',instance_sha256=f'ids{i}_{j}')
            obs['synchronization']=dict(request_index=number,callback_sequence=number,reference_time=[number,1],freshness=token)
            observations.append(obs);receipts.append(dict(observation_id=name,request_index=number))
        for item in (value,ctx):
            item.pop('diagnostic_only');item.pop('actual_diagnostic_profile');item.pop('selected_diagnostic_sample_id')
            item.update(actual_experimental_profile=profile,selected_experimental_sample_ids=ids,experimental_profile=True,**purpose)
        value.update(schema=producer.SEGMENT_SCHEMA,source_request=spec['source_request'],morphology_id=gid,frames=receipts,
            request_index_start=cursor,callback_sequence_start=cursor,requests=events,request_count=len(events),request_index_end=cursor+len(events),callback_sequence_end=cursor+len(events))
        current=dict(morphology_id=gid);ctx['geometry_sources']={gid:dict(donor_source_family='donor')}
        ctx['persistent_batch_authority']['source_request']=spec['source_request'];ctx['full_scene_census']['foreground_identity']=current
        ctx['persistent_foreground_swap'].update(current_geometry=current,previous_geometry=prior or dict(geometry_source_id='donor'))
        x[4][i]['frames']=nframes;x[7][i]=observations;cursor+=len(events);prior=current
    batch=x[2];batch.pop('diagnostic_only');batch.pop('diagnostic_profile')
    batch.update(schema=producer.RESULT_SCHEMA,frames=12,request_count=cursor,callback_count=cursor,experimental_profile=True,
        experimental_profile_policy=policy.make_profile(x[9][0]['source_optical_profile']))
    x[-1]=producer;return x


def schedule(x):
    sources=[]
    for i in range(4):
        k=i if i<3 else 0
        sources.append((x[8][i],dict(profile=x[9][i]['source_optical_profile'],manifest=dict(path=f'm{k}',sha256=f'm{k}'),
            records=[dict(sample_id=name) for name in x[3][k]['selected_sample_ids']])))
    request=dict(segments=x[3],experimental_profile_policy=x[2]['experimental_profile_policy'],experimental_profile=True,
        training_approved=False,accepted_training_increment=0,max_frames=12,potential_candidate_frames=11)
    return request,sources


class Contracts(unittest.TestCase):
    def reject(self,fn):
        x=fixture();fn(x)
        with self.assertRaises(ValueError):consumer.validate_registry(*x)
    def test_happy_twelve_frames_twenty_two_requests(self):
        x=fixture();policy.validate_schedule(*schedule(x));consumer.validate_registry(*x)
        self.assertEqual([v['request_index_end'] for v in x[5]],[11,16,20,22])
    def test_control_cannot_be_candidate(self):self.reject(lambda x:x[6][3].update(capture_purpose=policy.CANDIDATE))
    def test_control_cannot_be_exportable(self):self.reject(lambda x:x[7][3][0].update(export_prohibited=False))
    def test_wrong_warmup_evidence(self):self.reject(lambda x:x[9][1].update(warmup_steps=[8]*7))
    def test_wrong_production_subframes(self):self.reject(lambda x:x[5][1]['requests'][-1].update(requested_subframes=6))
    def test_global_clock_reset(self):self.reject(lambda x:x[5][1].update(request_index_start=0))
    def test_stale_buffer(self):self.reject(lambda x:x[7][1][0]['synchronization']['freshness'].update(rgb_sha256='rgb0_3'))
    def test_wrong_profile_role(self):self.reject(lambda x:x[6][1].update(profile_role='qualified_production'))
    def test_background_changed(self):self.reject(lambda x:x[6][1]['full_scene_census'].update(background_invariance_sha256='other'))
    def test_schedule_replay_different_camera_rejected(self):
        x=fixture();r,c=schedule(x);r['segments'][3]['selected_sample_ids']=['camera0_1']
        with self.assertRaises(ValueError):policy.validate_schedule(r,c)
    def test_schedule_extra_frame_rejected(self):
        r,c=schedule(fixture());r['max_frames']=13
        with self.assertRaises(ValueError):policy.validate_schedule(r,c)
    def test_schedule_source_order_changed(self):
        r,c=schedule(fixture());r['segments'][1]['selected_sample_ids'].reverse()
        with self.assertRaises(ValueError):policy.validate_schedule(r,c)
    def test_numerical_frame_body_identical(self):
        old=Path(frozen.__file__).read_text();new=Path(consumer.__file__).read_text()
        old=old[old.index('            assert_same_camera('):old.index('        stats=checker.finish()')]
        new=new[new.index('            assert_same_camera('):new.index('        stats=checker.finish()')]
        self.assertEqual(old.replace("result['requests'][sync['request_index']-1]","result['requests'][sync['request_index']-result['request_index_start']-1]"),new)
        for name in ('evaluate_frame','assess_frame','geometry_9mm','component_masks','build_inventory','validate_planned_geometry'):
            self.assertIs(getattr(consumer,name),getattr(frozen,name))


class Lifecycle(unittest.TestCase):
    def run_case(self,mutation=None):
        with tempfile.TemporaryDirectory() as tmp,patch.object(policy,'check_profile'):
            trial=Path(tmp)/'trial';capture=trial/'capture';capture.mkdir(parents=True)
            def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            def save(p,x):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return pin(p)
            def read(spec,within=None):
                p=Path(spec['path']).resolve()
                if within is not None:self.assertTrue(p.is_relative_to(within))
                self.assertEqual(pin(p),{k:spec[k] for k in ('path','sha256')});return json.loads(p.read_text())
            x=fixture();profilepin=save(trial/'profile.json',dict(warmup_steps=[8]*7,request_subframes=8,delta_time_seconds=0,render_settings={'test':True}))
            sources=[];qpin=save(trial/'q.json',{});sourcepins=[]
            for i in range(3):
                candidate=dict(source_bindings={},profile=profilepin,qualification=qpin,source_family='donor',records=[dict(sample_id=name) for name in x[3][i]['selected_sample_ids']])
                cp=save(trial/f'candidates{i}.json',candidate);source=dict(max_frames=len(candidate['records']),candidates=cp,implementation_bindings={})
                sp=save(trial/f'source{i}.json',source);sources.append((source,candidate));sourcepins.append(sp)
            for i,spec in enumerate(x[3]):spec['source_request']=sourcepins[i if i<3 else 0]
            request=dict(segments=x[3],implementation_bindings={},experimental_profile_policy=policy.make_profile(profilepin),max_frames=12)
            rp=save(trial/'request.json',request)
            command=['python.bat','-B','-u','-m',producer.MODULE,'--capture','--request',rp['path'],'--request-sha256',rp['sha256'],'--output',str(capture)]
            owner=dict(ProcessId=5);identity=dict(expected_command=command,owner=owner,command=dict(ProcessId=7),native=dict(ProcessId=9))
            if mutation=='owner':identity['owner']=dict(ProcessId=6)
            ip=save(capture/'native_identity.json',identity);entries=[]
            for i in range(4):
                spec=x[3][i];target=capture/'segments'/spec['segment_id']/'capture';context=x[6][i];value=x[5][i]
                actual=policy.actual_profile(spec,profilepin,read(profilepin));context.update(profile=profilepin,actual_experimental_profile=actual)
                context['persistent_batch_authority']=dict(batch_request=rp,source_request=spec['source_request'],segment_id=spec['segment_id'],native_identity=ip)
                context['persistent_foreground_swap'].update(source_profile=profilepin,qualification=qpin)
                if mutation=='control_export' and i==3:context['export_prohibited']=False
                cp=save(target/'context.json',context);receipts=[]
                for obs in x[7][i]:
                    obs.update(batch_request=rp,source_request=spec['source_request'],context=cp,source_family='donor')
                    if mutation=='foreign_frame' and i==2:obs['sample_id']='foreign'
                    op=save(target/'frames'/obs['observation_id']/'observation.json',obs)
                    receipts.append(dict(op,observation_id=obs['observation_id'],request_index=obs['synchronization']['request_index']))
                value.update(batch_request=rp,source_request=spec['source_request'],native_identity=ip,actual_experimental_profile=actual,frames=receipts)
                vp=save(target/'result.json',value);entries.append(dict(segment_id=spec['segment_id'],result=vp,frames=len(receipts),holds=0))
            batch=x[2];batch.update(request=rp,native_identity=ip,segments=entries,experimental_profile_policy=request['experimental_profile_policy'])
            if mutation is None:producer.validate_batch_registry(batch,request,capture,request_pin=rp,native_identity=ip)
            bp=save(capture/'result.json',batch);wp=save(trial/'owned_worker.json',dict(command=command,launcher_pid=7,owner_pid=5))
            ep=save(trial/'owned_exit.json',dict(returncode=1 if mutation=='nonzero' else 0,owned_worker=wp))
            sp=save(trial/'owner_started.json',dict(request=rp,command=command,resources=dict(raw_owner_classification=dict(metadata=owner))))
            save(trial/'owner_complete.json',dict(training_approved=False,accepted_training_increment=0,experimental_profile=True,
                owned_exit=ep,owner_started=sp,result=bp,native_identity=ip,request=rp,frames=12,segments=entries))
            gate=types.SimpleNamespace(same_identity=lambda a,b:a==b,command_child=lambda *a:None,native_child=lambda *a:None)
            mockproducer=types.SimpleNamespace(**{n:getattr(producer,n) for n in ('MODULE','RESULT_SCHEMA','SEGMENT_SCHEMA','OBSERVATION_SCHEMA','diagnostic')},
                production=types.SimpleNamespace(identity_gate=gate),check_request=lambda _:sources+[sources[0]])
            return consumer.authenticate_batch(capture/'segments/s003/capture',read,pin,mockproducer)
    def test_actual_pinned_twelve_frame_closure(self):
        r=self.run_case();self.assertTrue(r['public']['export_prohibited']);self.assertEqual(r['public']['segment_index'],3)
    def test_wrong_owner(self):
        with self.assertRaises(ValueError):self.run_case('owner')
    def test_nonzero_exit(self):
        with self.assertRaises(ValueError):self.run_case('nonzero')
    def test_foreign_frame(self):
        with self.assertRaises(ValueError):self.run_case('foreign_frame')
    def test_control_export_rejected(self):
        with self.assertRaises(ValueError):self.run_case('control_export')

if __name__=='__main__':unittest.main()
