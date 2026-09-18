"""Real recorded camera/FK regression cases for the generated capture boundary."""
from copy import deepcopy
from pathlib import Path
import unittest

from .dataset_review import read_json
from . import native848_controlled_capture_v1 as capture


class PoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[3]
        cls.candidates=read_json(root/'data/sim_data/diagnostics/native848_controlled_new_donors25mm_20260918_v1/seed11_camera_candidates/candidates.json')

    def pair(self):
        record=deepcopy(self.candidates['records'][0])
        return record,read_json(record['anchor']['path'])

    def test_three_actual_robot_camera_proposals_reproduce_fk(self):
        self.assertEqual(len(self.candidates['records']),3)
        for record in self.candidates['records']:
            capture.validate_pose(record,read_json(record['anchor']['path']))

    def test_arm_cannot_change_under_head_only_pose(self):
        record,anchor=self.pair();record['joint_degrees']['left_arm_0']+=1
        with self.assertRaisesRegex(ValueError,'Arm/torso'):capture.validate_pose(record,anchor)

    def test_detached_camera_is_rejected(self):
        record,anchor=self.pair();record['calibration']['camera_to_world_usd_row_vectors'][3][0]+=.001
        with self.assertRaisesRegex(ValueError,'Camera FK'):capture.validate_pose(record,anchor)

    def test_body_cannot_exceed_original_bound(self):
        record,anchor=self.pair();record['requested_spec']['root_x_m']+=.041
        with self.assertRaises(ValueError):capture.validate_pose(record,anchor)

    def test_camera_mount_cannot_move(self):
        record,anchor=self.pair();record['camera_to_head_column_vectors'][0][3]+=.001
        with self.assertRaisesRegex(ValueError,'mount'):capture.validate_pose(record,anchor)

    def test_camera_donor_cannot_be_relabeled(self):
        record,anchor=self.pair();record['source_family']='seed7_full'
        with self.assertRaisesRegex(ValueError,'donor'):capture.validate_pose(record,anchor)

    def test_native_command_uses_resolved_exact_input_paths(self):
        import argparse
        args=argparse.Namespace(request=Path('request.json'),request_sha256='0'*64,output=Path('capture'))
        command=capture.native_command(args)
        self.assertEqual(command[1:6],['-B','-u','-m',capture.MODULE,'--capture'])
        self.assertEqual(command[7],str(args.request.resolve()))
        self.assertEqual(command[-1],str(args.output.resolve()))


if __name__=='__main__':unittest.main()
