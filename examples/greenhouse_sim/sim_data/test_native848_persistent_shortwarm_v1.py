"""Finite CPU contract tests; no simulator or native acceptance claims."""
from copy import deepcopy
import hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from sim_data import native848_persistent_shortwarm_plan_v1 as policy
from sim_data import native848_persistent_shortwarm_capture_v1 as producer
from sim_data import native848_persistent_shortwarm_9mm_v1 as consumer
from sim_data import native848_generated_9mm_v3 as frozen
from sim_data.test_native848_persistent_generated_9mm_v1 import fixture as oldfixture


def fixture():
    old=oldfixture();rp,ip=old[:2];specs=[];entries=[];values=[];contexts=[];obslist=[];sources=[];profiles=[]
    cursor=0;prior=None;pp=dict(path='profile',sha256='profilehash')
    for i,count in enumerate(policy.COUNTS):
        sid=f's{i:03d}';kind=i%2;name=f'camera{kind}';gid=f'shape{kind}';sourcepin=dict(path=f'source{kind}',sha256=f'sh{kind}')
        spec=dict(segment_id=sid,source_request=sourcepin,selected_sample_id=name,warmup_request_count=count)
        profile=dict(schema=policy.ACTUAL_SCHEMA,source_optical_profile=pp,warmup_steps=[8]*count,request_subframes=8,
            delta_time_seconds=0,render_settings={'test':True},reset_before_production=True,diagnostic_only=True,
            training_approved=False,accepted_training_increment=0)
        events=[]
        for j in range(count+1):
            n=cursor+j+1
            events.append(dict(request_index=n,callback_sequence_before=n-1,callback_sequence_after=n,
                native_requests=1,callback_count=1,requested_subframes=8,delta_time_seconds=0,wait_for_render=True,
                timeline_before_seconds=0,timeline_after_seconds=0,reference_time=[n,1],previous_reference_time=None if n==1 else [n-1,1]))
        obs=deepcopy(old[7][0][0]);obs.update(schema=producer.OBSERVATION_SCHEMA,segment_id=sid,source_request=sourcepin,
            morphology_id=gid,sample_id=name,observation_id=name,request_evidence=events[-1],diagnostic_only=True,
            training_approved=False,accepted_training_increment=0)
        end=cursor+count+1
        token=dict(callback_sequence=end,reference_time=[end,1],camera_sha256=f'camera{kind}',rgb_sha256=f'rgb{i}',depth_sha256=f'depth{i}',instance_sha256=f'ids{i}')
        obs['synchronization']=dict(request_index=end,callback_sequence=end,reference_time=[end,1],freshness=token)
        value=deepcopy(old[5][0]);value.update(schema=producer.SEGMENT_SCHEMA,segment_id=sid,source_request=sourcepin,
            frames=[dict(observation_id=name,request_index=end)],request_index_start=cursor,callback_sequence_start=cursor,
            request_index_end=end,callback_sequence_end=end,request_count=len(events),requests=events,morphology_id=gid,
            actual_diagnostic_profile=profile,selected_diagnostic_sample_id=name,diagnostic_only=True)
        ctx=deepcopy(old[6][0]);current=dict(morphology_id=gid)
        ctx.update(profile=pp,profile_role=policy.PROFILE_ROLE,actual_diagnostic_profile=profile,
            selected_diagnostic_sample_id=name,diagnostic_only=True,training_approved=False,accepted_training_increment=0,
            geometry_sources={gid:dict(donor_source_family='donor')})
        ctx['persistent_batch_authority'].update(segment_id=sid,source_request=sourcepin)
        ctx['persistent_foreground_swap'].update(sequence=i+1,source_profile=pp,current_geometry=current,
            previous_geometry=prior or dict(geometry_source_id='donor'))
        ctx['full_scene_census']['foreground_identity']=current
        specs.append(spec);entries.append(dict(segment_id=sid,frames=1,holds=0));values.append(value);contexts.append(ctx)
        obslist.append([obs]);sources.append(dict(max_frames=4));profiles.append(profile);cursor=end;prior=current
    batch=deepcopy(old[2]);batch.update(schema=producer.RESULT_SCHEMA,segments=entries,frames=4,request_count=cursor,
        callback_count=cursor,diagnostic_only=True,diagnostic_profile=policy.make_profile(pp))
    batch['background_closure']['sequence']=4
    return [rp,ip,batch,specs,entries,values,contexts,obslist,sources,profiles,producer]


class ShortWarm(unittest.TestCase):
    def reject(self,mutate):
        x=fixture();mutate(x)
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):consumer.validate_registry(*x)
    def test_happy_four_segments_20_requests_repeated_camera_ids(self):
        x=fixture();consumer.validate_registry(*x);self.assertEqual(x[2]['request_count'],20)
        self.assertEqual([v['request_index_end'] for v in x[5]],[8,16,18,20])
    def test_false_seven_warmup_claim_on_short_segment(self):self.reject(lambda x:x[5][2]['actual_diagnostic_profile'].update(warmup_steps=[8]*7))
    def test_production_subframes_changed(self):self.reject(lambda x:x[5][3]['requests'][-1].update(requested_subframes=6))
    def test_profile_role_not_baseline_claim(self):self.reject(lambda x:x[6][2].update(profile_role='qualified_production'))
    def test_wrong_diagnostic_camera(self):self.reject(lambda x:x[7][2][0].update(sample_id='other'))
    def test_accepted_increment_rejected(self):self.reject(lambda x:x[5][2].update(accepted_training_increment=1))
    def test_global_clock_reset(self):self.reject(lambda x:x[5][2].update(request_index_start=0))
    def test_stale_buffers_at_swap(self):self.reject(lambda x:x[7][2][0]['synchronization']['freshness'].update(rgb_sha256='rgb1'))
    def test_background_changed(self):self.reject(lambda x:x[6][2]['full_scene_census'].update(background_invariance_sha256='wrong'))
    def test_numerical_frame_body_identical(self):
        new=Path(consumer.__file__).read_text();old=Path(frozen.__file__).read_text()
        a=old[old.index('            assert_same_camera('):old.index('        stats=checker.finish()')]
        b=new[new.index('            assert_same_camera('):new.index('        stats=checker.finish()')]
        a=a.replace("result['requests'][sync['request_index']-1]","result['requests'][sync['request_index']-result['request_index_start']-1]")
        self.assertEqual(a,b)
        for n in ('evaluate_frame','assess_frame','geometry_9mm','component_masks','build_inventory','validate_planned_geometry'):
            self.assertIs(getattr(consumer,n),getattr(frozen,n))
    def schedule(self):
        x=fixture();candidates=[]
        for i in range(4):
            candidates.append((x[8][i],dict(profile=x[9][i]['source_optical_profile'],manifest={'path':f'm{i%2}','sha256':f'm{i%2}'},records=[dict(sample_id=f'camera{i%2}') ])))
        return dict(segments=x[3],diagnostic_profile=x[2]['diagnostic_profile'],diagnostic_only=True,training_approved=False,accepted_training_increment=0,max_frames=4),candidates
    def test_exact_bounded_schedule(self):policy.validate_schedule(*self.schedule())
    def test_schedule_pair_camera_change_rejected(self):
        r,c=self.schedule();r['segments'][2]['selected_sample_id']='another';c[2][1]['records'].append(dict(sample_id='another'))
        with self.assertRaises(ValueError):policy.validate_schedule(r,c)
    def test_schedule_more_frames_rejected(self):
        r,c=self.schedule();r['max_frames']=5
        with self.assertRaises(ValueError):policy.validate_schedule(r,c)
    def test_schedule_same_shape_rejected(self):
        r,c=self.schedule();c[1][1]['manifest']=c[0][1]['manifest']
        with self.assertRaises(ValueError):policy.validate_schedule(r,c)
    def test_profile_preserves_source_bytes(self):
        source=dict(warmup_steps=[8]*7,request_subframes=8,delta_time_seconds=0,render_settings={'setting':'same'})
        before=deepcopy(source)
        with patch.object(policy,'check_profile') as check:
            actual=policy.actual_profile(fixture()[3][2],{'path':'profile','sha256':'p'},source)
        check.assert_called_once_with(source);self.assertEqual(source,before);self.assertEqual(actual['warmup_steps'],[8])


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
            for i in range(2):
                candidate=dict(source_bindings={},profile=profilepin,qualification=qpin,source_family='donor',records=[dict(sample_id=f'camera{i}')])
                cp=save(trial/f'candidates{i}.json',candidate);source=dict(max_frames=4,candidates=cp,implementation_bindings={})
                sp=save(trial/f'source{i}.json',source);sources.append((source,candidate));sourcepins.append(sp)
            for i,spec in enumerate(x[3]):spec['source_request']=sourcepins[i%2]
            request=dict(segments=x[3],implementation_bindings={},diagnostic_profile=policy.make_profile(profilepin),max_frames=4)
            rp=save(trial/'request.json',request)
            command=['python.bat','-B','-u','-m',producer.MODULE,'--capture','--request',rp['path'],'--request-sha256',rp['sha256'],'--output',str(capture)]
            owner=dict(ProcessId=5);identity=dict(expected_command=command,owner=owner,command=dict(ProcessId=7),native=dict(ProcessId=9))
            if mutation=='owner':identity['owner']=dict(ProcessId=6)
            ip=save(capture/'native_identity.json',identity);entries=[]
            for i in range(4):
                spec=x[3][i];target=capture/'segments'/spec['segment_id']/'capture';context=x[6][i];value=x[5][i];obs=x[7][i][0]
                actual=policy.actual_profile(spec,profilepin,read(profilepin));context.update(profile=profilepin,actual_diagnostic_profile=actual)
                context['persistent_batch_authority']=dict(batch_request=rp,source_request=spec['source_request'],segment_id=spec['segment_id'],native_identity=ip)
                context['persistent_foreground_swap'].update(source_profile=profilepin,qualification=qpin)
                if mutation=='fake_profile' and i==2:context['profile_role']='qualified_production'
                cp=save(target/'context.json',context);obs.update(batch_request=rp,source_request=spec['source_request'],context=cp,source_family='donor')
                if mutation=='foreign_frame' and i==2:obs['sample_id']='foreign'
                op=save(target/'frames'/obs['observation_id']/'observation.json',obs)
                value.update(batch_request=rp,source_request=spec['source_request'],native_identity=ip,actual_diagnostic_profile=actual,
                    frames=[dict(op,observation_id=obs['observation_id'],request_index=obs['synchronization']['request_index'])])
                vp=save(target/'result.json',value);entries.append(dict(segment_id=spec['segment_id'],result=vp,frames=1,holds=0))
            batch=x[2];batch.update(request=rp,native_identity=ip,segments=entries,diagnostic_profile=request['diagnostic_profile'])
            if mutation is None:producer.validate_batch_registry(batch,request,capture,request_pin=rp,native_identity=ip)
            bp=save(capture/'result.json',batch);wp=save(trial/'owned_worker.json',dict(command=command,launcher_pid=7,owner_pid=5))
            ep=save(trial/'owned_exit.json',dict(returncode=1 if mutation=='nonzero' else 0,owned_worker=wp))
            sp=save(trial/'owner_started.json',dict(request=rp,command=command,resources=dict(raw_owner_classification=dict(metadata=owner))))
            save(trial/'owner_complete.json',dict(training_approved=False,accepted_training_increment=0,diagnostic_only=True,
                owned_exit=ep,owner_started=sp,result=bp,native_identity=ip,request=rp,frames=4,segments=entries))
            gate=types.SimpleNamespace(same_identity=lambda a,b:a==b,command_child=lambda *a:None,native_child=lambda *a:None)
            mockproducer=types.SimpleNamespace(**{n:getattr(producer,n) for n in ('MODULE','RESULT_SCHEMA','SEGMENT_SCHEMA','OBSERVATION_SCHEMA','diagnostic')},
                production=types.SimpleNamespace(identity_gate=gate),check_request=lambda _:sources*2)
            return consumer.authenticate_batch(capture/'segments/s002/capture',read,pin,mockproducer)
    def test_actual_pinned_four_segment_closure(self):
        v=self.run_case();self.assertEqual(v['public']['segment_index'],2);self.assertEqual(v['public']['actual_diagnostic_profile']['warmup_steps'],[8])
    def test_wrong_owner_rejected(self):
        with self.assertRaises(ValueError):self.run_case('owner')
    def test_nonzero_exit_rejected(self):
        with self.assertRaises(ValueError):self.run_case('nonzero')
    def test_foreign_frame_rejected(self):
        with self.assertRaises(ValueError):self.run_case('foreign_frame')
    def test_fake_profile_claim_rejected(self):
        with self.assertRaises(ValueError):self.run_case('fake_profile')

if __name__=='__main__':unittest.main()
