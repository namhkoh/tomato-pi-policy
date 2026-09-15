import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sim_data.depth_preview import sha256
from sim_data.native_capture_v2.plan import build,check


class WiderPlanTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'base.json';source=root/'source.json'
        matrix=np.eye(4);matrix[3,:3]=[.4,.2,.101]
        base.write_text(json.dumps(dict(source_row=dict(component_id='SubStem_1'),
            expected_robot_snapshot=dict(robot_root_to_world_usd_row_vectors=matrix.tolist(),visual_bound_screen=dict(passed=True)))))
        source.write_text(json.dumps(dict(anchor_pair_plan=str(base),source_family='train_family',
            source_bindings={str(base):sha256(base)},implementation_bindings={},prerequisite_bindings={str(base):sha256(base)},
            target_cases=[dict(base_pair_plan=str(base),base_pair_plan_sha256=sha256(base),
                target_id='variant/SubStem_1',expected_nominal_world_m=[0,0,1.5],views=[],
                conservative_view_cap_group='train_family/SubStem_1')])))
        return source,base

    @patch('sim_data.native_capture_v2.plan.source_check',return_value=dict(source_sample='sample_1'))
    def test_rebuild_and_source_check_mode(self,gate):
        with tempfile.TemporaryDirectory() as d:
            source,base=self.fixture(Path(d));a=build(source,replay_geometry=False)
            self.assertEqual(a['maximum_native_frames'],9)
            self.assertFalse(a['training_approved']);self.assertFalse(a['source_cap_reset'])
            self.assertFalse(a['physical_motion_commanded']);self.assertFalse(a['hidden_cut_coordinates_executable'])
            self.assertEqual(check(a,replay_geometry=False),json.loads(base.read_text()))
            self.assertTrue(all(c.kwargs=={'replay_geometry':False} for c in gate.call_args_list))
            check(a,replay_geometry=True);self.assertEqual(gate.call_args.kwargs,{'replay_geometry':True})

    @patch('sim_data.native_capture_v2.plan.source_check',return_value=dict(source_sample='sample_1'))
    def test_plan_pose_and_approval_mutations_rejected(self,_):
        with tempfile.TemporaryDirectory() as d:
            source,_base=self.fixture(Path(d));a=build(source)
            b=copy.deepcopy(a);b['target_cases'][0]['views'][0]['root_x_m']+=.01
            with self.assertRaises(ValueError):check(b)
            b=copy.deepcopy(a);b['training_approved']=True
            with self.assertRaises(ValueError):check(b)
            b=copy.deepcopy(a);b['maximum_native_frames']=100
            with self.assertRaises(ValueError):check(b)

    @patch('sim_data.native_capture_v2.plan.source_check',return_value=dict(source_sample='sample_1'))
    def test_changed_source_bytes_rejected(self,_):
        with tempfile.TemporaryDirectory() as d:
            source,base=self.fixture(Path(d));a=build(source)
            base.write_text(base.read_text()+'\n')
            with self.assertRaises(ValueError):check(a,replay_geometry=False)

    @patch('sim_data.native_capture_v2.plan.source_check',side_effect=ValueError('source qualification failed'))
    def test_source_gate_failure_propagates(self,_):
        with tempfile.TemporaryDirectory() as d:
            source,_base=self.fixture(Path(d))
            with self.assertRaisesRegex(ValueError,'source qualification failed'):build(source)


if __name__=='__main__':unittest.main()
