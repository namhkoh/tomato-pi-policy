"""Identity tests independent of native rendering and external coordinators."""
from copy import deepcopy
import unittest
import tempfile
from pathlib import Path

import numpy as np

from . import native848_multihost_sweep_v1 as adapter


def slot(family='seed1_full', x=0.0, root='/World/Plant1', manifest='a' * 64):
    matrix = np.eye(4)
    matrix[3, 0] = x
    return dict(plant_root=root, source_family=family, source_split='train',
                manifest_sha256=manifest, source_geometry_modified=False,
                plant_to_world_usd_row_vectors=matrix.tolist())


def record(plant, camera=None, component='SubStem_41', sample='s1'):
    if camera is None:
        camera = np.eye(4)
        camera[3, :3] = [.4, -.2, 1.5]
    return dict(sample_id=sample, source_family=plant['source_family'], split='train',
                plant_root=plant['plant_root'], source_geometry_modified=False,
                target_id='hint/' + component,
                target_variant=dict(source_geometry_modified=False),
                source_row=dict(source_plant_id=plant['source_family'],
                                source_manifest_sha256=plant['manifest_sha256'],
                                component_id=component),
                calibration=dict(resolution=[848, 408], crop_resize=None,
                                 intrinsics=[[470.9, 0, 424], [0, 470.9, 204], [0, 0, 1]],
                                 camera_to_world_usd_row_vectors=np.asarray(camera).tolist()))


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.plant = slot()
        self.row = record(self.plant)
        self.scene = {'scene': 'unchanged'}

    def identity(self, row=None, plant=None):
        return adapter.view_identity(row or self.row, plant or self.plant, self.scene,
                                     (plant or self.plant)['manifest_sha256'])

    def test_component_sample_and_paths_do_not_change_identity(self):
        other = deepcopy(self.row)
        other.update(sample_id='foreign_host_name', target_id='new_hint/SubStem_99',
                     source_pose_provenance={'path': '/different/host/source.json'})
        other['source_row']['component_id'] = 'SubStem_99'
        self.assertEqual(self.identity(), self.identity(other))

    def test_same_world_image_hinted_at_other_donor_has_common_alias(self):
        second = slot('seed2_full', 1.0, '/World/Plant2', 'b' * 64)
        first_id = self.identity()
        second_id = self.identity(record(second), second)
        self.assertNotEqual(first_id['local_view_key'], second_id['local_view_key'])
        self.assertEqual(first_id['scene_camera_key'], second_id['scene_camera_key'])

    def test_clone_local_repeat_has_common_local_key(self):
        clone = slot(x=2.0, root='/World/Clone')
        camera = np.asarray(self.row['calibration']['camera_to_world_usd_row_vectors'])
        camera[3, 0] += 2.0
        repeated = self.identity(record(clone, camera), clone)
        self.assertEqual(self.identity()['local_view_key'], repeated['local_view_key'])
        self.assertNotEqual(self.identity()['scene_camera_key'], repeated['scene_camera_key'])

    def test_usd_row_composition_matches_transformed_point(self):
        angle = .63
        transform = np.eye(4)
        transform[:2, :2] = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        transform[3, :3] = [2.3, -1.2, .8]
        moved = deepcopy(self.plant)
        moved['plant_to_world_usd_row_vectors'] = transform.tolist()
        camera = np.asarray(self.row['calibration']['camera_to_world_usd_row_vectors']) @ transform
        self.assertEqual(self.identity()['local_view_key'],
                         self.identity(record(moved, camera), moved)['local_view_key'])

    def test_meaningful_new_camera_remains_distinct(self):
        other = deepcopy(self.row)
        other['calibration']['camera_to_world_usd_row_vectors'][3][1] += .02
        self.assertNotEqual(self.identity()['local_view_key'], self.identity(other)['local_view_key'])
        self.assertNotEqual(self.identity()['scene_camera_key'], self.identity(other)['scene_camera_key'])

    def test_generated_or_foreign_manifest_rejected(self):
        for change in ('generated', 'manifest'):
            other = deepcopy(self.row)
            if change == 'generated':
                other['source_geometry_modified'] = True
            else:
                other['source_row']['source_manifest_sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                self.identity(other)

    def test_crop_nonfinite_and_improper_transform_rejected(self):
        for change in ('crop', 'nonfinite', 'scale'):
            other = deepcopy(self.row)
            if change == 'crop':
                other['calibration']['crop_resize'] = [1, 1]
            elif change == 'nonfinite':
                other['calibration']['intrinsics'][0][0] = float('nan')
            else:
                other['calibration']['camera_to_world_usd_row_vectors'][0][0] = 2
            with self.assertRaises(ValueError):
                self.identity(other)

    def test_scene_identity_ignores_file_and_prim_names_but_not_geometry(self):
        slots = [slot(x=float(i), root='/World/P' + str(i)) for i in range(144)]
        census = dict(source_assets_modified=False, complete_active_plant_anatomy=True,
                      dataset_split='train', all_plant_roots=slots)
        first = adapter.original_scene_identity(census, {s['plant_root']:'a'*64 for s in slots})
        for i, row in enumerate(slots):
            row['plant_root'] = '/World/Renamed' + str(i)
            row['manifest_path'] = '/server/data/' + str(i)
        self.assertEqual(first, adapter.original_scene_identity(census, {s['plant_root']:'a'*64 for s in slots}))
        slots[0]['plant_to_world_usd_row_vectors'][3][0] += .01
        self.assertNotEqual(first, adapter.original_scene_identity(census, {s['plant_root']:'a'*64 for s in slots}))



class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = adapter.coordinator_module()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = self.coordinator.Store(Path(self.temp.name) / 'isolated.sqlite')
        self.store.seed([])
        self.plant = slot('seed41_full')
        self.row = record(self.plant)

    def identities(self, records):
        return [dict(sample_id=r['sample_id'], **adapter.view_identity(r, p, {'fixed': True}, p['manifest_sha256']))
                for r, p in records]

    def test_atomic_claim_same_image_with_two_hints_is_one(self):
        second = slot('seed103_full', 2, '/World/Other', 'b' * 64)
        identities = self.identities([(self.row, self.plant), (record(second, sample='second'), second)])
        response = self.store.claim_many('local5090', [adapter.coordinator_identity(v) for v in identities])
        self.assertEqual(adapter.granted_indices(identities, response, 'local5090', self.coordinator), [0])
        self.assertEqual(response['reservations'][1]['reason'], 'same_scene_camera_already_reserved')
        self.assertEqual(self.store.stats()['states'], {'reserved': 1})

    def test_clone_and_renamed_component_cannot_claim_twice(self):
        clone = slot('seed41_full', 2, '/World/Clone')
        camera = np.asarray(self.row['calibration']['camera_to_world_usd_row_vectors'])
        camera[3, 0] += 2
        identities = self.identities([(self.row, self.plant), (record(clone, camera, 'SubStem_99', 'second'), clone)])
        response = self.store.claim_many('local5090', [adapter.coordinator_identity(v) for v in identities])
        self.assertEqual(adapter.granted_indices(identities, response, 'local5090', self.coordinator), [0])
        self.assertEqual(response['reservations'][1]['reason'], 'exact_view_already_reserved')

    def test_worker_partitions_disjoint_and_complete(self):
        config = dict(schema='greenhouse.multihost_capture_config.v1', host_id='local5090',
                      allowed_foreground_families=self.coordinator.ALLOCATION['local5090'],
                      allocation_sha256=self.coordinator.digest(self.coordinator.ALLOCATION), worker_count=4)
        values = []
        for i in range(24):
            row = deepcopy(self.row)
            row['sample_id'] = str(i)
            row['calibration']['camera_to_world_usd_row_vectors'][3][1] += .02 * i
            values += self.identities([(row, self.plant)])
        groups = [set(adapter.partition_identities(values, config, worker, self.coordinator)[0]) for worker in range(4)]
        self.assertEqual(set.union(*groups), set(range(24)))
        self.assertEqual(sum(map(len, groups)), 24)
        config['allowed_foreground_families'] = ['seed103_full']
        with self.assertRaises(ValueError):
            adapter.partition_identities(values, config, 0, self.coordinator)

    def test_wrong_host_or_forged_grant_fail_closed(self):
        identities = self.identities([(self.row, self.plant)])
        denied = self.store.claim_many('thor1', [adapter.coordinator_identity(v) for v in identities])
        self.assertEqual(adapter.granted_indices(identities, denied, 'thor1', self.coordinator), [])
        response = self.store.claim_many('local5090', [adapter.coordinator_identity(v) for v in identities])
        for field, value in [('host', 'thor3'), ('task_id', 'f' * 64), ('claim_id', '')]:
            forged = deepcopy(response)
            forged['reservations'][0][field] = value
            with self.assertRaises(ValueError):
                adapter.granted_indices(identities, forged, 'local5090', self.coordinator)


if __name__ == '__main__':
    unittest.main()

