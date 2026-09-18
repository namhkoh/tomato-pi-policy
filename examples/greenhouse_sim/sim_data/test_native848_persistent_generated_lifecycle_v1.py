"""Pinned on-disk batch fixtures; mock only the external process-identity gate."""
import hashlib,json,tempfile,types,unittest
from pathlib import Path
from sim_data import native848_persistent_generated_9mm_v1 as api
from sim_data.test_native848_persistent_generated_9mm_v1 import fixture

class Lifecycle(unittest.TestCase):
    def run_case(self, mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            trial=Path(tmp)/'trial';batchdir=trial/'capture';batchdir.mkdir(parents=True)
            def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            def save(p,x):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return pin(p)
            def read(spec,within=None):
                p=Path(spec['path']).resolve()
                if within is not None:self.assertTrue(p.is_relative_to(within))
                self.assertEqual(pin(p),{k:spec[k] for k in ('path','sha256')})
                return json.loads(p.read_text())
            x=fixture();producer=x[-1];producer.MODULE='sim_data.native848_persistent_capture_v1'
            gate=types.SimpleNamespace(same_identity=lambda a,b:a==b,command_child=lambda *a:None,native_child=lambda *a:None)
            producer.production=types.SimpleNamespace(identity_gate=gate)
            request_path=trial/'request.json';identity_path=batchdir/'native_identity.json'
            profile_pin=save(trial/'profile.json',x[9][0]);qpin=save(trial/'qualification.json',{})
            sources=[];scheduled=[]
            for i in range(2):
                source=dict(max_frames=1,implementation_bindings={})
                sp=save(trial/f'source{i}.json',source);scheduled.append(dict(segment_id=f's{i:03d}',source_request=sp))
                candidates=dict(source_bindings={},profile=profile_pin,qualification=qpin,source_family='donor',records=[dict(sample_id=f'frame{i}')])
                sources.append((source,candidates))
            request=dict(segments=scheduled,implementation_bindings={});rp=save(request_path,request)
            command=['python.bat','-B','-u','-m',producer.MODULE,'--capture','--request',str(request_path),'--request-sha256',rp['sha256'],'--output',str(batchdir)]
            owner=dict(ProcessId=5);identity=dict(expected_command=command,owner=owner,command=dict(ProcessId=7),native=dict(ProcessId=9))
            if mutation=='owner':identity['owner']=dict(ProcessId=6)
            ip=save(identity_path,identity);entries=[]
            for i in range(2):
                target=batchdir/'segments'/f's{i:03d}'/'capture';target.mkdir(parents=True)
                context=x[6][i];value=x[5][i];obs=x[7][i][0];spec=scheduled[i]
                context['persistent_batch_authority']=dict(batch_request=rp,source_request=spec['source_request'],segment_id=spec['segment_id'],native_identity=ip)
                context['profile']=profile_pin;context['persistent_foreground_swap'].update(source_profile=profile_pin,qualification=qpin)
                cp=save(target/'context.json',context)
                obs.update(batch_request=rp,source_request=spec['source_request'],context=cp,source_family='donor')
                if mutation=='foreign_frame' and i==1:obs.update(sample_id='not-scheduled')
                op=save(target/'frames'/obs['observation_id']/'observation.json',obs)
                value.update(source_request=spec['source_request'],batch_request=rp,native_identity=ip,
                    frames=[dict(op,observation_id=obs['observation_id'],request_index=obs['synchronization']['request_index'])])
                vp=save(target/'result.json',value);entries.append(dict(segment_id=spec['segment_id'],result=vp,frames=1,holds=0))
            batch=x[2];batch.update(request=rp,native_identity=ip,segments=entries);bp=save(batchdir/'result.json',batch)
            wp=save(trial/'owned_worker.json',dict(command=command,launcher_pid=7,owner_pid=5))
            ep=save(trial/'owned_exit.json',dict(returncode=1 if mutation=='nonzero' else 0,owned_worker=wp))
            sp=save(trial/'owner_started.json',dict(request=rp,command=command,resources=dict(raw_owner_classification=dict(metadata=owner))))
            save(trial/'owner_complete.json',dict(training_approved=False,accepted_training_increment=0,
                owned_exit=ep,owner_started=sp,result=bp,native_identity=ip,request=rp,frames=2,segments=entries))
            producer.check_request=lambda request:sources
            return api.authenticate_batch(batchdir/'segments/s001/capture',read,pin,producer)
    def test_actual_pinned_batch_success(self):
        r=self.run_case();self.assertEqual(r['public']['segment_index'],1)
        self.assertTrue(r['public']['segment_is_not_separate_native_owner'])
    def test_actual_wrong_owner_rejected(self):
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):self.run_case('owner')
    def test_actual_nonzero_exit_rejected(self):
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):self.run_case('nonzero')
    def test_actual_foreign_frame_rejected(self):
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):self.run_case('foreign_frame')

if __name__=='__main__':unittest.main()
