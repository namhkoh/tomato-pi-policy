from copy import deepcopy
from unittest.mock import patch
import unittest
from . import native848_farmerge_finite_population_v1 as a


class Guards(unittest.TestCase):
    def test_surface_entry_retains_whole_plant(self):
        row=dict(component_index=1,prim_path='/World/PackPlants/P/Main/Leaf',organ_type='leaf',original_world_min=[.2,0,0],original_world_max=[.3,.1,.1])
        bounds=dict(arms={'left':dict(shoulder_world_m=[0,0,0],conservative_probe_reach_m=.8)},margin_m=.001)
        roots,causes=a.affected_whole_plants(dict(selected_components=[row]),[bounds])
        self.assertEqual(roots,['/World/PackPlants/P']);self.assertEqual(causes[0]['component_index'],1)

    def test_far_surface_but_near9mm_still_retains_plant(self):
        row=dict(component_index=1,prim_path='/World/PackPlants/P/Petiole',organ_type='sub_stem',original_world_min=[2,0,0],original_world_max=[2.1,.1,.1],petiole_exclusion=dict(nominal_arc_m=.009,nominal_world_m=[.2,0,0]))
        bounds=dict(arms={'left':dict(shoulder_world_m=[0,0,0],conservative_probe_reach_m=.8)},margin_m=.001)
        self.assertEqual(a.affected_whole_plants(dict(selected_components=[row]),[bounds])[0],['/World/PackPlants/P'])

    def test_one_of_multiple_pose_bounds_retains_plant(self):
        row=dict(component_index=1,prim_path='/World/PackPlants/P/Main',organ_type='main_stem',original_world_min=[2,0,0],original_world_max=[2.1,.1,.1])
        first=dict(arms={'left':dict(shoulder_world_m=[0,0,0],conservative_probe_reach_m=.8)},margin_m=.001)
        second=deepcopy(first);second['arms']['left']['shoulder_world_m']=[2,0,0]
        self.assertEqual(a.affected_whole_plants(dict(selected_components=[row]),[first])[0],[])
        self.assertEqual(a.affected_whole_plants(dict(selected_components=[row]),[first,second])[0],['/World/PackPlants/P'])

    def test_background_identity_and_transform_are_fixed_foreground_may_differ(self):
        rows=[dict(plant_root='/World/PackPlants/P'+str(i),source_family='f',manifest_sha256='a',plant_to_world_usd_row_vectors=[[i]]) for i in range(144)]
        original=dict(all_plant_roots=rows);actual=deepcopy(original);primary=rows[0]['plant_root']
        actual['all_plant_roots'][0]['manifest_sha256']='new_foreground';a.background_match(actual,original,primary)
        actual['all_plant_roots'][1]['manifest_sha256']='other_asset'
        with self.assertRaises(ValueError):a.background_match(actual,original,primary)
        actual=deepcopy(original);actual['all_plant_roots'][1]['plant_to_world_usd_row_vectors']=[[999]]
        with self.assertRaises(ValueError):a.background_match(actual,original,primary)

    def test_candidate_robot_state_must_match_original_cpu(self):
        snap=dict(joint_degrees={'head':1},robot_root_to_world_usd_row_vectors=[[1]],camera_to_head_column_vectors=[[2]])
        record=dict(sample_id='generated',source_camera_sample_id='original',source_camera_plan={'path':'p','sha256':'h'},calibration={'cal':1},**deepcopy(snap))
        candidates=dict(source_camera_plan=record['source_camera_plan'],records=[record])
        cpu=dict(plan_sha256='h',all_selected_pose_screens_passed=True,poses=[dict(sample_id='original',passed=True,proof=dict(actual_robot_snapshot=snap,actual_calibration={'cal':1},screen={'passed':True}))])
        with patch.object(a.coverage,'robot_bounds',return_value={'proof':'FK'}), patch.object(a,'assert_same_camera'):
            row=a.candidate_bounds(candidates,cpu,dict(selected_sample_ids=['original']),None)[0]
            self.assertTrue(row['generated_scene_collision_pending'])
            record['joint_degrees']['head']=2
            with self.assertRaises(ValueError):a.candidate_bounds(candidates,cpu,dict(selected_sample_ids=['original']),None)


if __name__=='__main__':unittest.main()
