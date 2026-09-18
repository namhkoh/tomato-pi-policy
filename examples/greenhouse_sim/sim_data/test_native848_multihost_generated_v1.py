"""Pure worker/claim boundary checks; never contacts a live coordinator."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_data import native848_multihost_generated_v1 as h


def config(host):
    return dict(host_id=host,allocation_sha256=h.coordinator.digest(h.coordinator.ALLOCATION),
        allowed_foreground_families=h.coordinator.ALLOCATION[host],worker_count=h.WORKER_COUNTS[host])


class GeneratedReservationTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(geometry_sha256='a'*64,donor_source_family='seed19_full',split='train',
            camera_to_plant_usd_row_vectors=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
            intrinsics=[[470.,0.,424.],[0.,470.,204.],[0.,0.,1.]],resolution=[848,408],scene_camera_key='b'*64)
        self.key=h.coordinator.task_id(self.identity)
        self.response=dict(schema='greenhouse.capture_reservations.v1',reservations=[
            dict(task_id=self.key,granted=True,host='local5090',claim_id='c'*48)])

    def test_exact_all_grants(self):
        self.assertTrue(h.validate_grants([self.identity],self.response,'local5090'))

    def test_denied_claim_holds_batch(self):
        self.response['reservations'][0]=dict(task_id=self.key,granted=False,reason='already_reserved')
        self.assertFalse(h.validate_grants([self.identity],self.response,'local5090'))

    def test_wrong_task_host_or_handle(self):
        for key,value in [('task_id','d'*64),('host','thor1'),('claim_id','')]:
            with self.subTest(key=key):
                response=deepcopy(self.response);response['reservations'][0][key]=value
                with self.assertRaises(ValueError):h.validate_grants([self.identity],response,'local5090')

    def test_wrong_schema_or_missing_row(self):
        for response in [dict(schema='wrong',reservations=[]),dict(schema='greenhouse.capture_reservations.v1',reservations=[])]:
            with self.assertRaises(ValueError):h.validate_grants([self.identity],response,'local5090')

    def test_all_fixed_hosts_and_workers(self):
        for host,count in h.WORKER_COUNTS.items():
            for worker in range(count):self.assertEqual(h.validate_config(config(host),worker),host)

    def test_wrong_host_allocation_count_or_worker(self):
        cases=[]
        c=config('thor1');c['host_id']='unknown';cases.append((c,0))
        c=config('thor1');c['allowed_foreground_families']=['seed19_full'];cases.append((c,0))
        c=config('thor1');c['worker_count']=1;cases.append((c,0))
        cases += [(config('thor1'),-1),(config('thor1'),4),(config('local5090'),True)]
        for c,worker in cases:
            with self.subTest(config=c,worker=worker):
                with self.assertRaises(ValueError):h.validate_config(c,worker)

    def test_batch_shard_is_order_independent_and_has_one_owner(self):
        rows=[dict(task_id=x*64) for x in 'abc']
        for host,count in h.WORKER_COUNTS.items():
            assignments=[h.batch_assignment(rows,config(host),w) for w in range(count)]
            self.assertEqual(sum(a['worker_owns_request'] for a in assignments),1)
            self.assertEqual(assignments[0]['batch_id'],h.coordinator.digest(sorted(r['task_id'] for r in rows)))
            self.assertEqual(assignments[0],h.batch_assignment(rows[::-1],config(host),0))

    def test_empty_or_repeated_candidate_keys_rejected(self):
        for rows in [[],[dict(task_id=self.key),dict(task_id=self.key)]]:
            with self.assertRaises(ValueError):h.batch_assignment(rows,config('local5090'),0)

    def test_target_names_cannot_change_key(self):
        changed=deepcopy(self.identity);changed.update(target_id='other/petiole',sample_id='renamed',morphology_id='seed_only_change')
        self.assertEqual(h.coordinator.task_id(changed),self.key)

    def test_changed_geometry_or_pose_changes_key(self):
        changed=deepcopy(self.identity);changed['geometry_sha256']='f'*64
        self.assertNotEqual(h.coordinator.task_id(changed),self.key)
        changed=deepcopy(self.identity);changed['camera_to_plant_usd_row_vectors'][3][0]=.01
        self.assertNotEqual(h.coordinator.task_id(changed),self.key)

    def fixture(self,output,host='local5090',worker=0):
        request=output/'request.json';request.write_text('{}')
        rows=[dict(task_id=self.key,identity=self.identity)]
        value=dict(worker_assignment=h.batch_assignment(rows,config(host),worker),rows=rows,
            controls=[dict(reservation_excluded=True,accepted_increment=0)],source_request=h.pin(request),host=host)
        h.save_json(output/'identities.json',value)
        return value

    def test_nonowner_never_contacts_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);owner=h.batch_assignment([dict(task_id=self.key)],config('thor1'),0)['owner_worker_id']
            value=self.fixture(output,'thor1',(owner+1)%4);client=Mock()
            h.finish_reservation(value,config('thor1'),output,reserve=True,client=client)
            client.claim_many.assert_not_called()
            gate=json.loads((output/'reservation_gate.json').read_text())
            self.assertFalse(gate['capture_request_eligible']);self.assertFalse(gate['coordinator_contacted'])

    def test_owner_claims_candidates_only_and_excludes_control(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);value=self.fixture(output);client=Mock();client.claim_many.return_value=self.response
            h.finish_reservation(value,config('local5090'),output,reserve=True,client=client)
            client.claim_many.assert_called_once_with([self.identity])
            gate=json.loads((output/'reservation_gate.json').read_text())
            self.assertTrue(gate['capture_request_eligible']);self.assertEqual(gate['excluded_replay_controls'],1)
            self.assertFalse(gate['native_launched'])

    def test_partial_grant_never_makes_capture_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);value=self.fixture(output);client=Mock()
            response=deepcopy(self.response);response['reservations'][0]=dict(task_id=self.key,granted=False,reason='already_reserved')
            client.claim_many.return_value=response
            h.finish_reservation(value,config('local5090'),output,reserve=True,client=client)
            gate=json.loads((output/'reservation_gate.json').read_text())
            self.assertFalse(gate['capture_request_eligible']);self.assertFalse(gate['automatic_reclaim'])


if __name__=='__main__':unittest.main()
