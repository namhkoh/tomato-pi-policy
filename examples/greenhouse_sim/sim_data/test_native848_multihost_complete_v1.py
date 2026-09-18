"""Bounded completion guards; actual local six-frame fixture is opt-in."""
from pathlib import Path
from copy import deepcopy
import json
import tempfile
import unittest
from . import native848_multihost_complete_v1 as m

class CompletionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.prep = cls.root/'data/sim_data/diagnostics/native848_local_multihost_mesh6_prepare_20260918_v1'
        if not (cls.prep/'claims.json').exists():
            raise unittest.SkipTest('Actual local capture fixture unavailable')
        def read(path): return json.loads(Path(path).read_text())
        cls.claims = read(cls.prep/'claims.json')
        cls.config = read(cls.claims['config']['path'])
        cls.identities = read(cls.claims['identities']['path'])['records']
        cls.source = read(read(cls.claims['source_request']['path'])['records']['path'])['records']
        cls.actual = read(cls.prep/'records.json')['records']
        cls.coordinator = m.adapter.coordinator_module()

    def match(self, claims=None, identities=None, actual=None):
        return m.match_reservations(self.claims if claims is None else claims,
            self.identities if identities is None else identities, self.source,
            self.actual if actual is None else actual, self.config, self.coordinator)

    def test_exact_granted_subset(self):
        self.assertEqual(set(self.match()), {r['sample_id'] for r in self.actual})
        self.assertEqual(len(self.match()), 6)

    def test_wrong_claim_identity_refused(self):
        claims = deepcopy(self.claims)
        claims['response']['reservations'][0]['task_id'] = 'f'*64
        with self.assertRaises(ValueError): self.match(claims=claims)

    def test_foreign_host_refused(self):
        claims = deepcopy(self.claims);claims['host'] = 'thor1'
        with self.assertRaises(ValueError): self.match(claims=claims)

    def test_wrong_native_sample_refused(self):
        actual = deepcopy(self.actual);actual[0]['sample_id'] = 'foreign'
        with self.assertRaises(ValueError): self.match(actual=actual)

    def test_changed_camera_refused(self):
        actual = deepcopy(self.actual);actual[0]['calibration']['camera_to_world_usd_row_vectors'][3][0] += .01
        with self.assertRaises(ValueError): self.match(actual=actual)

    def test_missing_frame_record_refused(self):
        with self.assertRaises(ValueError): self.match(actual=self.actual[:-1])

    def test_identity_order_refused(self):
        identities = deepcopy(self.identities);identities[0],identities[1] = identities[1],identities[0]
        with self.assertRaises(ValueError): self.match(identities=identities)

    def test_generated_reservations_fail_closed(self):
        claims = deepcopy(self.claims);claims['schema'] = 'greenhouse.generated_reservation_gate.v1'
        with self.assertRaises(ValueError): self.match(claims=claims)

    def test_changed_pin_and_outside_capture_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'input.json';path.write_text('{}')
            inputs = m.Inputs();pin = inputs.local(path);path.write_text('{"altered":true}')
            with self.assertRaises(ValueError): inputs.read(pin)
            inputs = m.Inputs();pin = inputs.local(path)
            with self.assertRaises(ValueError): inputs.read(pin,within=Path(temp)/'other')

if __name__ == '__main__':
    unittest.main(verbosity=2)
