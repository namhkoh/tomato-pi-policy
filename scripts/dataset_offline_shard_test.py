"""Real host-local SQLite races/duplicate checks; no HTTP or native capture."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dataset_offline_shard as offline

c=offline.coordinator


def view(x=0.,family='seed101_full',geometry='b'*64,alias=None):
    return dict(geometry_sha256=geometry,donor_source_family=family,split='train',
        camera_to_plant_usd_row_vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[x,0,0,1]],
        intrinsics=[[470,0,424],[0,470,204],[0,0,1]],resolution=[848,408],
        scene_camera_key=alias or c.digest({'scene':geometry,'world_x':x}))


def rgb(n=1,dhash='0'*16):return dict(decoded_rgb_sha256=f'{n:064x}',dhash64_hex=dhash)


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.state=self.root/'thor1'
        self.config_path=offline.ROOT/'configs/dataset_capture/thor1.json'
        self.baseline=dict(view(family='seed19_full',geometry='a'*64),**rgb(1),stage='accepted')
        self.inventory=self.root/'inventory.jsonl';self.inventory.write_text(json.dumps(self.baseline)+'\n')
        self.inventory_sha=offline.pin(self.inventory)['sha256']
        offline.initialize(self.state,self.config_path,self.inventory,self.inventory_sha)
        self.manifest,self.config,self.store=offline.open_state(self.state)

    def client(self,item=None,kind='raw',worker=None):
        item=item or view();values=[item]
        if worker is None:worker=offline.worker_owner(values,self.config,kind)
        return offline.LocalClient(c.Store(self.state/'ledger.sqlite',hosts=['thor1']),self.config,worker,kind,
            dict(source_request={'path':'fixture','sha256':'request'},seed=2000000))

    def claim(self,item=None):
        item=item or view();return self.client(item).claim_many([item])['reservations'][0]

    def test_network_is_never_needed(self):
        with patch('socket.create_connection',side_effect=AssertionError('network forbidden')),\
             patch.object(c,'Client',side_effect=AssertionError('HTTP client forbidden')):
            grant=self.claim();self.assertTrue(grant['granted'])
            value=offline.portable_export(self.state,self.root/'export.json')
            self.assertTrue(value['global_review_pending']);self.assertFalse(value['network_contacted'])

    def test_atomic_same_worker_race_one_grant(self):
        item=view();barrier=threading.Barrier(2)
        def run():
            client=self.client(item);barrier.wait(timeout=5)
            return client.claim_many([item])['reservations'][0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers=[f.result(timeout=15) for f in [pool.submit(run),pool.submit(run)]]
        self.assertEqual(sum(r['granted'] for r in answers),1)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM offline_task_audit').fetchone()[0],1)

    def test_wrong_worker_fails_before_claim(self):
        item=view();owner=offline.worker_owner([item],self.config,'raw')
        with self.assertRaisesRegex(ValueError,'Another worker'):
            self.client(item,worker=(owner+1)%4).claim_many([item])
        self.assertEqual(len(self.store.export()),1)

    def test_generated_whole_batch_ownership(self):
        values=[view(.1),view(.2)];owner=offline.worker_owner(values,self.config,'generated')
        self.assertEqual(owner,int(c.digest(sorted(c.task_id(v) for v in values)),16)%4)
        client=offline.LocalClient(self.store,self.config,owner,'generated',dict(seed=2000000))
        response=client.claim_many(values)
        self.assertTrue(all(r['granted'] for r in response['reservations']))

    def test_wrong_host_roster_rejected(self):
        other=c.Store(self.root/'other.sqlite',hosts=['thor3'])
        with self.assertRaisesRegex(ValueError,'exactly this host'):
            offline.LocalClient(other,self.config,0,'raw',{})

    def test_baseline_other_host_donor_scene_alias_blocks(self):
        candidate=view(alias=self.baseline['scene_camera_key'])
        answer=self.claim(candidate)
        self.assertFalse(answer['granted']);self.assertEqual(answer['reason'],'same_scene_camera_already_reserved')

    def test_cross_donor_geometry_rename_rejected(self):
        with self.assertRaisesRegex(ValueError,'renamed'):
            self.claim(view(geometry='a'*64))

    def test_exact_and_near_pose_duplicate_hold(self):
        self.assertTrue(self.claim()['granted'])
        self.assertEqual(self.claim()['reason'],'exact_view_already_reserved')
        self.assertEqual(self.claim(view(.003))['reason'],'near_pose_already_reserved')

    def test_exact_rgb_and_perceptual_holds_remain_pending(self):
        first=self.claim(view(.1));second=self.claim(view(.2))
        # Store tests use synthetic metrics only. Production CLI has no metrics
        # input and obtains them exclusively from authenticated capture.prepare.
        a=self.store.finish('thor1',first['task_id'],first['claim_id'],rgb(1))
        b=self.store.finish('thor1',second['task_id'],second['claim_id'],rgb(2,'0000000000000001'))
        self.assertEqual(a['state'],'duplicate_hold');self.assertEqual(b['state'],'duplicate_hold')
        self.assertEqual(a['duplicate_candidates'][0]['kind'],'exact_rgb')
        self.assertFalse(a['training_approved'])

    def test_portable_export_keeps_aliases_and_omits_private_handles(self):
        item=view();grant=self.claim(item)
        result=offline.portable_export(self.state,self.root/'export.json')
        text=json.dumps(result);self.assertNotIn(grant['claim_id'],text);self.assertNotIn('"claim_id"',text)
        row=next(r for r in result['rows'] if r['task_id']==grant['task_id'])
        self.assertEqual(row['scene_camera_keys'],[item['scene_camera_key']])
        self.assertEqual(row['offline_audit']['seed'],2000000)
        self.assertTrue(all(r['global_review_pending'] and r['accepted_training_increment']==0 for r in result['rows']))

    def test_missing_baseline_alias_rejected(self):
        del self.baseline['scene_camera_key'];self.inventory.write_text(json.dumps(self.baseline)+'\n')
        with self.assertRaisesRegex(ValueError,'every full-scene'):
            offline.initialize(self.root/'new',self.config_path,self.inventory,offline.pin(self.inventory)['sha256'])

    def test_changed_baseline_or_host_config_fails_closed(self):
        Path(self.manifest['config']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Changed pinned'):
            offline.open_state(self.state)

    def test_two_hosts_independent_and_final_global_review_mandatory(self):
        other=self.root/'thor3';cfg=offline.ROOT/'configs/dataset_capture/thor3.json'
        offline.initialize(other,cfg,self.inventory,self.inventory_sha)
        _,oc,os=offline.open_state(other)
        first=view();second=view(family='seed53_full',geometry='c'*64,alias=first['scene_camera_key'])
        self.assertTrue(self.claim(first)['granted'])
        owner=offline.worker_owner([second],oc,'raw')
        response=offline.LocalClient(os,oc,owner,'raw',{}).claim_many([second])
        self.assertTrue(response['reservations'][0]['granted'])
        # Distinct offline ledgers cannot promise cross-host deduplication.
        self.assertTrue(response['global_review_pending'])
        self.assertTrue(response['final_cross_host_duplicate_review_required'])

    def completion_fixture(self):
        item=view();client=self.client(item);grant=client.claim_many([item])['reservations'][0]
        rp=self.root/'reserved.json'
        value=dict(schema='greenhouse.offline_capture_reservation.v1',state=offline.pin(self.state/'state.json'),
            config=self.manifest['config'],host='thor1',capture_request_eligible=True,capture_kind='raw',
            worker_id=client.worker_id,source_request=client.audit['source_request'],reservation={'path':'claims','sha256':'claims'},**offline.SCOPE)
        offline.write(rp,value);return rp,value,grant

    def test_completion_authentication_failure_never_finishes(self):
        rp,value,grant=self.completion_fixture()
        adapter=SimpleNamespace(prepare=lambda *args:(_ for _ in ()).throw(ValueError('Unsupported native owner')))
        with patch.object(offline.importlib,'import_module',return_value=adapter),\
             patch.object(c.Store,'finish',side_effect=AssertionError('must not finish')):
            with self.assertRaisesRegex(ValueError,'Unsupported native owner'):
                offline.complete(self.state,rp,offline.pin(rp)['sha256'],'capture',self.root/'completion')
        self.assertEqual(next(r for r in self.store.export() if r['task_id']==grant['task_id'])['state'],'reserved')

    def test_completion_consumes_only_authenticated_adapter_actions(self):
        rp,value,grant=self.completion_fixture();close_calls=[]
        checked=dict(config=self.config,reservation=value['reservation'],
            inputs=SimpleNamespace(close=lambda:close_calls.append(True),bindings={}),
            actions=[dict(sample_id='actual',reservation=grant,outcome='captured',metrics=rgb(77,'ffffffffffffffff'))],
            owner_complete={'path':'owner','sha256':'owner'},capture_result={'path':'result','sha256':'result'})
        adapter=SimpleNamespace(prepare=lambda *args:checked,__file__=__file__)
        with patch.object(offline.importlib,'import_module',return_value=adapter):
            result=offline.complete(self.state,rp,offline.pin(rp)['sha256'],'capture',self.root/'completion')
        self.assertEqual(result['completions'][0]['state'],'complete');self.assertEqual(len(close_calls),2)
        self.assertTrue(result['global_review_pending']);self.assertFalse(result['training_approved'])
        self.assertFalse(hasattr(offline.LocalClient,'finish'))


if __name__=='__main__':unittest.main(verbosity=2)
