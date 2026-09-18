"""Persistent completion guards: reservations never replace native-owner proof."""
from pathlib import Path
from copy import deepcopy
import json
import tempfile
import unittest
from . import native848_multihost_generated_complete_v1 as m

class GeneratedCompletionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic claims deliberately have no network authority. They exercise
        # the same typed checks on every server, with no local assets/captures.
        root=Path(__file__).resolve().parents[3]
        cls.config=json.loads((root/'configs/dataset_capture/local5090.json').read_text())
        request=dict(path='synthetic-request.json',sha256='a'*64)
        pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        rows=[]
        for i in range(2):
            matrix=deepcopy(pose);matrix[3][0]=i*.01
            identity=m.coordinator.identity(dict(geometry_sha256='b'*64,donor_source_family='seed19_full',split='train',
                camera_to_plant_usd_row_vectors=matrix,intrinsics=[[470,0,424],[0,470,204],[0,0,1]],resolution=[848,408]))
            identity['scene_camera_key']=str(i+1)*64
            rows.append(dict(segment_id='s000',sample_id='view_'+str(i),source_request=request,
                identity=identity,task_id=m.coordinator.task_id(identity)))
        assignment=m.reservation_adapter.batch_assignment(rows,cls.config,0)
        cls.portable=dict(schema='greenhouse.generated_persistent_reservation_preparation.v1',source_request=request,
            host='local5090',automatic_reclaim=False,automatic_split=False,mandatory_fullscene_camera_alias=True,
            controls=[],rows=rows,worker_assignment=assignment)
        cls.response=dict(schema='greenhouse.capture_reservations.v1',reservations=[dict(task_id=r['task_id'],
            host='local5090',granted=True,claim_id='synthetic-not-a-real-claim-'+'x'*32) for r in rows])
        cls.gate=dict(schema='greenhouse.generated_persistent_reservation_gate.v1',source_request=request,
            coordinator_contacted=True,all_candidate_views_reserved=True,capture_request_eligible=True,
            automatic_reclaim=False,training_approved=False,excluded_replay_controls=0,worker_assignment=assignment)

    def check_gate(self,gate=None,portable=None,response=None):
        return m.validate_gate(self.gate if gate is None else gate,self.portable if portable is None else portable,
            self.response if response is None else response,self.config,self.gate['source_request'])

    def test_exact_reserved_candidates(self):
        self.assertEqual(len(self.check_gate()),2)

    def test_wrong_owner_request_refused(self):
        gate=deepcopy(self.gate);gate['source_request']['sha256']='f'*64
        with self.assertRaises(ValueError):self.check_gate(gate=gate)

    def test_wrong_claim_task_refused(self):
        response=deepcopy(self.response);response['reservations'][0]['task_id']='f'*64
        with self.assertRaises(ValueError):self.check_gate(response=response)

    def test_wrong_claim_host_refused(self):
        response=deepcopy(self.response);response['reservations'][0]['host']='thor1'
        with self.assertRaises(ValueError):self.check_gate(response=response)

    def test_partial_reservation_refused(self):
        response=deepcopy(self.response);response['reservations'][0]['granted']=False
        with self.assertRaises(ValueError):self.check_gate(response=response)

    def test_replay_control_never_completed(self):
        portable=deepcopy(self.portable);portable['controls']=[deepcopy(portable['rows'][0])]
        with self.assertRaisesRegex(ValueError,'replay-control'):self.check_gate(portable=portable)

    def test_repeated_segment_sample_refused(self):
        portable=deepcopy(self.portable);portable['rows'][1]['segment_id']=portable['rows'][0]['segment_id'];portable['rows'][1]['sample_id']=portable['rows'][0]['sample_id']
        with self.assertRaises(ValueError):self.check_gate(portable=portable)

    def test_worker_assignment_cannot_change(self):
        gate=deepcopy(self.gate);gate['worker_assignment']['owner_worker_id']=1
        with self.assertRaises(ValueError):self.check_gate(gate=gate)

    def test_exact_frames_and_no_frame_hold_partition(self):
        expected={('s000','one'):None,('s000','two'):None}
        frames,holds=m.partition_segment('s000',[{'sample_id':'one'},{'sample_id':'two'}],
            [{'sample_id':'one'}],[{'sample_id':'two','reason':'whole_robot_scene_collision','screen':{'passed':False}}],expected)
        self.assertEqual(set(frames),{'one'});self.assertEqual(set(holds),{'two'})

    def test_missing_or_foreign_segment_frame_refused(self):
        expected={('s000','one'):None}
        for frames in ([],[{'sample_id':'foreign'}],[{'sample_id':'one'},{'sample_id':'one'}]):
            with self.assertRaises(ValueError):m.partition_segment('s000',[{'sample_id':'one'}],frames,[],expected)

    def test_false_native_hold_refused(self):
        with self.assertRaises(ValueError):m.partition_segment('s000',[{'sample_id':'one'}],[],
            [{'sample_id':'one','reason':'whole_robot_scene_collision','screen':{'passed':True}}],{('s000','one'):None})

    def test_cross_segment_sample_alias_does_not_complete_other_claim(self):
        with self.assertRaises(ValueError):m.partition_segment('s001',[{'sample_id':'one'}],[{'sample_id':'one'}],[],{('s000','one'):None})

    def test_actual_camera_or_geometry_change_alters_reserved_identity(self):
        pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        foreground={'source_split':'train','source_family':'seed19_full','plant_to_world_usd_row_vectors':pose}
        calibration={'camera_to_world_usd_row_vectors':pose,'intrinsics':[[470,0,424],[0,470,204],[0,0,1]],'resolution':[848,408]}
        scene={'schema':'unit_scene','full144_geometry_digest':'a'*64}
        original=m.actual_identity(scene,foreground,'b'*64,calibration)
        changed=deepcopy(calibration);changed['camera_to_world_usd_row_vectors'][3][0]=.01
        moved=m.actual_identity(scene,foreground,'b'*64,changed)
        self.assertNotEqual(m.coordinator.task_id(original),m.coordinator.task_id(moved))
        self.assertNotEqual(original['scene_camera_key'],moved['scene_camera_key'])
        self.assertNotEqual(m.coordinator.task_id(original),m.coordinator.task_id(m.actual_identity(scene,foreground,'c'*64,calibration)))

    def test_changed_background_alters_scene_alias(self):
        pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        foreground={'source_split':'train','source_family':'seed19_full','plant_to_world_usd_row_vectors':pose}
        cal={'camera_to_world_usd_row_vectors':pose,'intrinsics':[[470,0,424],[0,470,204],[0,0,1]],'resolution':[848,408]}
        a=m.actual_identity({'background_geometry':'a'*64},foreground,'b'*64,cal)
        b=m.actual_identity({'background_geometry':'c'*64},foreground,'b'*64,cal)
        self.assertEqual(m.coordinator.task_id(a),m.coordinator.task_id(b))
        self.assertNotEqual(a['scene_camera_key'],b['scene_camera_key'])

    def test_unclosed_owner_never_contacts_coordinator(self):
        class Client:
            host='local5090';url='http://127.0.0.1:8769';calls=0
            def finish(self,*args):self.calls+=1;raise AssertionError('Should never contact coordinator')
        client=Client()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);inputs=m.Inputs()
            def write(name,value):
                p=path/name;p.write_text(json.dumps(value));return inputs.local(p)
            config=write('config.json',self.config)
            request=write('request.json',dict(schema=m.producer.REQUEST_SCHEMA,
                segments=[dict(segment_id='s000',source_request=dict(path='unused',sha256='c'*64))]))
            portable=deepcopy(self.portable);portable.update(config=config,source_request=request)
            for module,key in ((m.reservation_adapter,'helper'),(m.geometry,'geometry_implementation'),(m.coordinator,'coordinator_implementation')):
                portable[key]=inputs.local(module.__file__)
            gate=deepcopy(self.gate);gate.update(source_request=request,identities=write('identities.json',portable),
                coordinator_response=write('response.json',self.response))
            gate_pin=write('gate.json',gate)
            # Real prepare/authenticate_batch reaches the missing owner file;
            # source hash gates run first. No native proof is mocked as passing.
            with self.assertRaises(FileNotFoundError) as error:
                m.run(config['path'],path/'unclosed',gate_pin['path'],path/'out',client=client)
            self.assertIn('owner_complete',str(error.exception))
            self.assertEqual(client.calls,0);self.assertFalse((path/'out').exists())

class OptionalLocalIntegration(unittest.TestCase):
    def test_real_sixteen_reserved_candidates(self):
        root=Path(__file__).resolve().parents[3]
        path=root/'data/sim_data/diagnostics/native848_generated19_multihost_reservation_20260918_v1/reservation_gate.json'
        if not path.exists():self.skipTest('Optional local reserved batch fixture unavailable')
        def read(p):return json.loads(Path(p).read_text())
        gate=read(path);portable=read(gate['identities']['path'])
        response=read(gate['coordinator_response']['path']);config=read(portable['config']['path'])
        self.assertEqual(len(m.validate_gate(gate,portable,response,config,gate['source_request'])),16)

if __name__=='__main__':unittest.main(verbosity=2)
