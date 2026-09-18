"""Small lineage and duplicate-policy checks for the reference bundle."""
import tempfile
from pathlib import Path
import unittest
from . import native848_delivery_bundle_v1 as bundle


def entry(i=0,geometry='seed101_full',split='train'):
    return dict(source_frame_id=f'frame{i}',decoded_rgb_sha256=f'rgb{i}',
        geometry_source_id=geometry,geometry_camera_signature=f'camera{i}',
        donor_source_family='seed101_full',donor_camera_signature=f'donorcam{i}',
        source_target_id='seed101_full/SubStem_41',split=split,
        generated_geometry=geometry!='seed101_full')


class DeliveryChecks(unittest.TestCase):
    def test_same_donor_pose_different_geometry_keeps_lineage(self):
        a=entry();b=entry(1,'seed101_full_modified')
        b['donor_camera_signature']=a['donor_camera_signature']
        result=bundle.counts([a,b])
        self.assertEqual(result['original_donor_count'],1)
        self.assertEqual(result['generated_geometry_count'],1)
        self.assertEqual(result['new_independent_donors_from_generated'],0)
        self.assertEqual(result['donor_camera_positions_reused_across_geometries'],1)

    def test_duplicate_camera_within_geometry_rejected(self):
        a=entry();b=entry(1);b['geometry_camera_signature']=a['geometry_camera_signature']
        with self.assertRaises(ValueError):bundle.counts([a,b])

    def test_exact_rgb_duplicate_across_morphologies_rejected(self):
        a=entry();b=entry(1,'seed101_full_modified');b['decoded_rgb_sha256']=a['decoded_rgb_sha256']
        with self.assertRaises(ValueError):bundle.counts([a,b])

    def test_donor_split_leak_rejected(self):
        with self.assertRaises(ValueError):bundle.counts([entry(),entry(1,'seed101_full_modified','test')])

    def test_generated_identity_cannot_create_donor(self):
        row=dict(schema=bundle.generated.FRAME_SCHEMA,source_family='seed101_full',
            donor_source_family='seed11_full',geometry_source_id='modified101',morphology_id='modified101',
            new_independent_source_family=False,source_geometry_modified=True)
        with self.assertRaises(ValueError):bundle.entry_for('generated','generated',0,row)

    def test_failed_bundle_rejected_before_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):bundle.DeliveryDataset(directory)


if __name__=='__main__':unittest.main()
