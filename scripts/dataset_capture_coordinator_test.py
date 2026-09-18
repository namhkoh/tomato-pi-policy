"""Focused coordinator tests: real SQLite races and real loopback HTTP.

No renderer, capture, or production ledger is touched. Optional baseline check:
python scripts/dataset_capture_coordinator_test.py --inventory path/to/inventory.jsonl
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager
import argparse
import hashlib
import json
import socket
import sqlite3
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from unittest.mock import patch

import dataset_capture_coordinator as c

INVENTORY = None
TOKEN = 'test-token-not-for-production-' + 'x' * 32

def pose(x=0.0):
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [x, 0, 0, 1]]

def view(x=0.0, family='seed101_full', geometry='a' * 64, split='train'):
    return dict(geometry_sha256=geometry, donor_source_family=family, split=split,
                camera_to_plant_usd_row_vectors=pose(x), intrinsics=[[470, 0, 424], [0, 470, 204], [0, 0, 1]],
                resolution=[848, 408], scene_camera_key=c.digest({'test_scene': 1, 'world_x': x}))

def rgb(n=1, perceptual='0' * 16):
    return dict(decoded_rgb_sha256=f'{n:064x}', dhash64_hex=perceptual)

@contextmanager
def serving(store):
    server = c.make_server(('127.0.0.1', 0), store, TOKEN)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'coord.sqlite'
        self.store = c.Store(self.path)
        self.store.seed([])

    def claim(self, item=None, host='thor1'):
        return self.store.claim_many(host, [view() if item is None else item])['reservations'][0]

    def test_simultaneous_two_host_claims_exactly_one_grant(self):
        other = c.Store(self.path)
        barrier = threading.Barrier(2)
        def race(store, host):
            barrier.wait(timeout=3)
            return store.claim_many(host, [view()])['reservations'][0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(race, self.store, 'thor1'), pool.submit(race, other, 'thor3')]
            answers = [f.result(timeout=10) for f in futures]
        self.assertEqual(sum(a['granted'] for a in answers), 1)
        self.assertEqual(len(self.store.export()), 1)
        self.assertEqual({a['task_id'] for a in answers}, {c.task_id(view())})

    def test_simultaneous_same_host_workers_atomic_exact_identity(self):
        other = c.Store(self.path)
        barrier = threading.Barrier(2)
        def race(store):
            barrier.wait(timeout=3)
            return store.claim_many('thor1', [view()])['reservations'][0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = [f.result(timeout=10) for f in [pool.submit(race, self.store), pool.submit(race, other)]]
        self.assertEqual(sum(a['granted'] for a in answers), 1)
        self.assertEqual(next(a for a in answers if not a['granted'])['reason'], 'exact_view_already_reserved')
        self.assertEqual(len(self.store.export()), 1)

    def test_cross_donor_same_scene_camera_atomic_alias(self):
        other = c.Store(self.path)
        barrier = threading.Barrier(2)
        a = view(family='seed101_full', geometry='a' * 64)
        b = view(family='seed53_full', geometry='b' * 64)
        def race(store, host, item):
            barrier.wait(timeout=3)
            return store.claim_many(host, [item])['reservations'][0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(race, self.store, 'thor1', a), pool.submit(race, other, 'thor3', b)]
            answers = [f.result(timeout=10) for f in futures]
        self.assertEqual(sum(a['granted'] for a in answers), 1)
        self.assertEqual(next(a for a in answers if not a['granted'])['reason'], 'same_scene_camera_already_reserved')
        self.assertEqual(len(self.store.export()), 1)
        restarted = c.Store(self.path)
        self.assertFalse(restarted.claim_many('thor1', [a])['reservations'][0]['granted'])
        self.assertFalse(restarted.claim_many('thor3', [b])['reservations'][0]['granted'])

    def test_donor_allocation_is_enforced_and_persistent(self):
        rejected = self.claim(host='thor3')
        self.assertEqual(rejected['reason'], 'donor_assigned_to_other_host')
        self.assertEqual(rejected['assigned_host'], 'thor1')
        self.assertEqual(self.store.export(), [])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM donors').fetchone()[0], 0)
        self.assertTrue(self.claim()['granted'])
        changed = deepcopy(c.ALLOCATION)
        changed['thor1'].remove('seed101_full')
        changed['thor3'].append('seed101_full')
        with patch.object(c, 'ALLOCATION', changed):
            with self.assertRaisesRegex(ValueError, 'allocation changed'):
                c.Store(self.path)

    def test_hint_path_host_seed_and_lighting_do_not_change_view(self):
        first = view()
        changed = dict(first, target_id='another/target', source_target_id='seed101_full/SubStem_99',
                       target_hint='different', path='/different/server.png', host='thor3', seed=999,
                       lighting='different', physical_plant_instance_id='clone17')
        self.assertEqual(c.task_id(first), c.task_id(changed))
        self.assertTrue(self.claim(first)['granted'])
        self.assertEqual(self.claim(changed)['reason'], 'exact_view_already_reserved')

    def test_clone_world_translation_does_not_create_new_local_view(self):
        a = dict(view(.2), camera_to_world_usd_row_vectors=pose(1.2), plant_to_world_usd_row_vectors=pose(1))
        b = dict(view(.2), camera_to_world_usd_row_vectors=pose(101.2), plant_to_world_usd_row_vectors=pose(101))
        b['scene_camera_key'] = c.digest({'test_scene': 1, 'world_x': 101.2})
        self.assertEqual(c.task_id(a), c.task_id(b))
        self.claim(a)
        self.assertEqual(self.claim(b)['reason'], 'exact_view_already_reserved')

    def test_near_pose_cannot_recollect_with_new_hint(self):
        self.claim(view())
        near = self.claim(view(.003))
        self.assertFalse(near['granted'])
        self.assertIn('near', near['reason'])
        self.assertTrue(self.claim(view(.02))['granted'])

    def test_restart_does_not_expire_or_steal_reservation(self):
        first = self.claim()
        with self.store.connect() as db:
            db.execute('UPDATE tasks SET created=0,updated=0')
        restarted = c.Store(self.path)
        answer = restarted.claim_many('thor1', [view()])['reservations'][0]
        self.assertFalse(answer['granted'])
        self.assertEqual(answer['existing_host'], 'thor1')
        sealed = restarted.finish('thor1', first['task_id'], first['claim_id'], outcome='no_frame')
        self.assertEqual(sealed['state'], 'no_frame')
        self.assertFalse(restarted.claim_many('thor1', [view()])['reservations'][0]['granted'])

    def test_split_leakage_and_geometry_renaming_rejected(self):
        self.claim()
        with self.assertRaises(ValueError):
            self.claim(view(.1, split='validation'))
        with self.assertRaises(ValueError):
            self.claim(view(.2, family='seed103_full'), 'local5090')
        self.assertEqual(len(self.store.export()), 1)

    def test_unseen_donor_wrong_split_rejected_before_first_claim(self):
        with self.assertRaisesRegex(ValueError, 'TRAIN only'):
            self.claim(view(family='seed7_full', split='test'))
        self.assertEqual(self.store.export(), [])
        self.assertTrue(self.claim(view(family='seed7_full'))['granted'])

    def test_baseline_scene_alias_blocks_different_donor_claim(self):
        baseline = dict(view(), **rgb(17), stage='accepted')
        self.store.seed([baseline])
        self.store.seed([baseline])
        candidate = view(family='seed53_full', geometry='b' * 64)
        answer = self.claim(candidate, 'thor3')
        self.assertFalse(answer['granted'])
        self.assertEqual(answer['reason'], 'same_scene_camera_already_reserved')
        self.assertEqual(answer['existing_host'], 'baseline')
        self.assertEqual(len(self.store.export()), 1)

    def test_conflicting_baseline_alias_rolls_back_whole_import(self):
        a = dict(view(), **rgb(17), stage='accepted')
        b = dict(view(.2, family='seed53_full', geometry='b' * 64), **rgb(18), stage='raw')
        b['scene_camera_key'] = a['scene_camera_key']
        with self.assertRaisesRegex(ValueError, 'repeated scene-camera'):
            self.store.seed([a, b])
        self.assertEqual(self.store.export(), [])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM scene_aliases').fetchone()[0], 0)

    def test_baseline_import_after_live_claim_fails_closed(self):
        self.claim()
        with self.assertRaisesRegex(ValueError, 'before live claims'):
            self.store.seed([dict(view(.2), **rgb(19), stage='raw')])
        self.assertEqual(len(self.store.export()), 1)

    def test_wrong_host_and_wrong_handle_cannot_complete(self):
        q = self.claim()
        for host, handle in [('thor3', q['claim_id']), ('thor1', 'wrong')]:
            with self.assertRaises(ValueError):
                self.store.finish(host, q['task_id'], handle, rgb())
        self.assertEqual(self.store.export()[0]['state'], 'reserved')

    def test_malformed_batch_has_no_partial_claim(self):
        invalid = view(.2)
        invalid['camera_to_plant_usd_row_vectors'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            self.store.claim_many('thor1', [view(), invalid])
        self.assertEqual(self.store.export(), [])

    def test_lineage_conflict_rolls_back_entire_batch(self):
        with self.assertRaises(ValueError):
            self.store.claim_many('thor1', [view(), view(.2, geometry='b' * 64, split='test')])
        self.assertEqual(self.store.export(), [])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM donors').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM scene_aliases').fetchone()[0], 0)

    def test_missing_alias_has_no_partial_claim(self):
        missing = view(.2)
        del missing['scene_camera_key']
        with self.assertRaisesRegex(ValueError, 'alias required'):
            self.store.claim_many('thor1', [view(), missing])
        self.assertEqual(self.store.export(), [])

    def test_exact_rgb_duplicate_is_held_across_different_views(self):
        a, b = [self.claim(view(x)) for x in (0, .1)]
        self.store.finish('thor1', a['task_id'], a['claim_id'], rgb(1))
        done = self.store.finish('thor1', b['task_id'], b['claim_id'], rgb(1, 'f' * 16))
        self.assertEqual(done['state'], 'duplicate_hold')
        self.assertEqual(done['duplicate_candidates'][0]['kind'], 'exact_rgb')
        self.assertFalse(done['training_approved'])

    def test_near_hash_is_review_hold_never_quality_acceptance(self):
        a, b = [self.claim(view(x)) for x in (0, .1)]
        self.store.finish('thor1', a['task_id'], a['claim_id'], rgb(1))
        done = self.store.finish('thor1', b['task_id'], b['claim_id'], rgb(2, '0000000000000007'))
        self.assertEqual(done['state'], 'duplicate_hold')
        self.assertEqual(done['duplicate_candidates'][0]['kind'], 'perceptual_review')
        self.assertEqual(done['duplicate_candidates'][0]['dhash_hamming'], 3)
        self.assertFalse(done['training_approved'])
        self.assertTrue(done['quality_and_visual_review_still_required'])

    def test_completion_is_idempotent_and_immutable(self):
        q = self.claim()
        first = self.store.finish('thor1', q['task_id'], q['claim_id'], rgb(1))
        self.assertEqual(c.Store(self.path).finish('thor1', q['task_id'], q['claim_id'], rgb(1)), first)
        for outcome, value in [('captured', rgb(2)), ('failed', None)]:
            with self.assertRaises(ValueError):
                self.store.finish('thor1', q['task_id'], q['claim_id'], value, outcome)
        self.assertEqual(self.store.export()[0]['result'], first)
        self.assertNotIn('claim', self.store.export()[0])

    def test_server_authentication_and_http_claim_completion(self):
        with serving(self.store) as url:
            for token in ('wrong-' + 'x' * 32, ''):
                req = Request(url + '/status', headers={'Authorization': 'Bearer ' + token})
                with self.assertRaises(HTTPError) as caught:
                    urlopen(req, timeout=2)
                self.assertEqual(caught.exception.code, 401)
            client = c.Client(url, TOKEN, 'thor1', timeout=2)
            self.assertTrue(client.call('/status')['seeded'])
            q = client.claim_many([view()])['reservations'][0]
            self.assertTrue(q['granted'])
            done = client.finish(q, rgb(3))
            self.assertFalse(done['training_approved'])
            self.assertEqual(client.finish(q, rgb(3)), done)

    def test_unavailable_server_fails_closed_without_local_fallback(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        before = self.store.export()
        with self.assertRaises((URLError, TimeoutError, ConnectionError, OSError)):
            c.Client(f'http://127.0.0.1:{port}', TOKEN, 'thor1', timeout=.2).claim_many([view()])
        self.assertEqual(self.store.export(), before)

    def test_unseeded_database_and_changed_host_roster_reject(self):
        other = c.Store(Path(self.temp.name) / 'unseeded.sqlite')
        with self.assertRaises(ValueError):
            other.claim_many('thor1', [view()])
        with self.assertRaises(ValueError):
            c.Store(self.path, ('thor1', 'thor3'))

class PortableTests(unittest.TestCase):
    def test_actual_baseline_import_is_idempotent_and_preserves_stages(self):
        if INVENTORY is None:
            self.skipTest('Pass --inventory for the real portable baseline import')
        rows = [json.loads(line) for line in INVENTORY.read_text().splitlines()]
        self.assertEqual(len(rows), 559)
        self.assertEqual(sum(r['stage'] == 'accepted' for r in rows), 491)
        with tempfile.TemporaryDirectory() as temp:
            store = c.Store(Path(temp) / 'baseline.sqlite')
            once = store.seed(rows)
            twice = store.seed(rows)
            self.assertEqual(once, twice)
            self.assertEqual(once['states'], {'baseline_accepted': 491, 'baseline_raw': 68})
            self.assertEqual(len(store.export()), 559)
            self.assertTrue(all('scene_camera_key' in r for r in rows), 'Use the shared-geometry v2 inventory')
            with store.connect() as db:
                self.assertEqual(db.execute('SELECT count(*) FROM scene_aliases').fetchone()[0], 559)
            original = next(r for r in rows if r['donor_source_family'] == 'seed101_full')
            request = dict(original, scene_camera_key='f' * 64)
            answer = store.claim_many('thor1', [request])['reservations'][0]
            self.assertFalse(answer['granted'])
            self.assertEqual(answer['reason'], 'exact_view_already_reserved')

    def test_rgb_header_dimensions_encoding_and_dhash_protocol(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            image = Image.new('RGB', (848, 408), (31, 44, 59))
            a, b = Path(temp) / 'a.png', Path(temp) / 'b.png'
            image.save(a, compress_level=0)
            image.save(b, compress_level=9)
            ma, mb = c.rgb_metrics(a), c.rgb_metrics(b)
            self.assertNotEqual(ma['encoded_rgb_sha256'], mb['encoded_rgb_sha256'])
            self.assertEqual(ma['decoded_rgb_sha256'], mb['decoded_rgb_sha256'])
            self.assertEqual(ma['decoded_rgb_sha256'], hashlib.sha256(b'RGB|uint8|408|848|3\n' + image.tobytes()).hexdigest())
            self.assertEqual(ma['dhash64_hex'], '0000000000000000')
            image.resize((424, 204)).save(b)
            with self.assertRaises(ValueError):
                c.rgb_metrics(b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--inventory', type=Path)
    args, remaining = parser.parse_known_args()
    INVENTORY = args.inventory
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
