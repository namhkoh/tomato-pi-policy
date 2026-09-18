"""Native848 sample-specific kinematic workspace evidence, separate from visual acceptance.

Uses the recorded fixed base/torso and the actual left-gripper visual midpoint.
No reach result is a collision-free trajectory or a physical cutting approval.
"""
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from pxr import Gf, Usd, UsdGeom
from greenhouse_sim import robot_kinematics as rk, robot_model

SCHEMA='greenhouse.clear848_kinematic_workspace.v1'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def rigid(value):
    m=np.asarray(value,dtype=float)
    require(m.shape==(4,4) and np.isfinite(m).all() and np.allclose(m[3],[0,0,0,1],atol=1e-7)
        and np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-6)
        and np.isclose(np.linalg.det(m[:3,:3]),1,atol=1e-6),'Finite rigid metre-scale transform required')
    return m

class WorkspaceChecker:
    def __init__(self):
        self.model=rk.Rby1Kinematics()
        self.stage=Usd.Stage.Open(str(robot_model.DEFAULT_ASSET))
        require(bool(self.stage),'Robot asset missing')
        self.root=self.stage.GetDefaultPrim()
        self.root_path=str(self.root.GetPath())
        files=[Path(__file__).resolve(),Path(rk.__file__).resolve(),Path(robot_model.__file__).resolve(),robot_model.DEFAULT_URDF]
        files += [Path(layer.realPath).resolve() for layer in self.stage.GetUsedLayers() if not layer.anonymous]
        self.bindings={str(p):digest(p) for p in files}
        self.stage.SetEditTarget(self.stage.GetSessionLayer())

    def finish(self):
        require(all(digest(p)==h for p,h in self.bindings.items()),'Workspace source changed')
        require(all(not layer.dirty for layer in self.stage.GetUsedLayers() if not layer.anonymous),'Authored robot layer changed')

    def pose_and_probe(self,meta):
        require(meta['calibration']['resolution']==[848,408],'Native848 capture required')
        pose=meta['robot_snapshot']; angles=pose['joint_degrees']
        require(pose['joint_limits_checked'] is True and pose['visual_bound_screen']['passed'] is True,'Clear static robot pose required')
        world=rigid(np.asarray(pose['robot_root_to_world_usd_row_vectors']).T)
        links=self.model.all_link_transforms(angles)
        expected=world@links['link_head_2']@rigid(pose['camera_to_head_column_vectors'])
        observed=rigid(np.asarray(meta['calibration']['camera_to_world_usd_row_vectors']).T)
        require(np.allclose(expected,observed,atol=2e-6,rtol=0),'Recorded camera does not reproduce from robot FK')
        UsdGeom.Xformable(self.root).MakeMatrixXform().Set(Gf.Matrix4d(world.T.tolist()))
        for name,matrix in links.items():
            prim=self.stage.GetPrimAtPath(self.root_path+'/'+name)
            require(bool(prim),'Missing robot link '+name)
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(matrix.T.tolist()))
        cache=UsdGeom.XformCache(); boxes=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
        ee=self.stage.GetPrimAtPath(self.root_path+'/ee_left')
        inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(ee)).T)
        centers=[]
        for finger in ('ee_finger_l1','ee_finger_l2'):
            prim=self.stage.GetPrimAtPath(self.root_path+'/'+finger+'/visuals')
            require(bool(prim) and UsdGeom.Imageable(prim).ComputeVisibility()!='invisible','Visible left gripper required')
            bound=boxes.ComputeWorldBound(prim).ComputeAlignedRange()
            require(not bound.IsEmpty(),'Empty finger geometry')
            corners=np.asarray(list(itertools.product(*zip(bound.GetMin(),bound.GetMax()))))
            local=(inverse@np.c_[corners,np.ones(8)].T).T[:,:3]
            centers.append((local.min(axis=0)+local.max(axis=0))/2)
        return dict(base=world@links['base'],torso=[angles[f'torso_{i}'] for i in range(6)],
            left=[angles[f'left_arm_{i}'] for i in range(7)],right=[angles[f'right_arm_{i}'] for i in range(7)],
            probe=np.mean(centers,axis=0))

    def solve(self,pose,target,seconds=4.0):
        target=np.asarray(target,dtype=float)
        require(target.shape==(3,) and np.isfinite(target).all(),'Finite target required')
        start=time.perf_counter(); model=self.model
        distance=float(np.linalg.norm(target-model.arm_shoulder_position_m('left',pose['base'],pose['torso'])))
        outer=model.maximum_endpoint_reach_m('left')+float(np.linalg.norm(pose['probe']))
        result=dict(workspace_passed=False,status='no_position_solution_found',arm='left',
            shoulder_distance_m=distance,conservative_outer_bound_m=outer,attempts=0,candidate=None)
        if distance>outer+.001:
            result['status']='outside_outer_reach_bound';return result
        low,high=model.arm_limits_degrees('left');rng=np.random.default_rng(731)
        seeds=[pose['left'],(low+high)/2,*[rng.uniform(low+2,high-2) for _ in range(3)]]
        def deadline():
            if time.perf_counter()-start>seconds:raise TimeoutError('bounded_workspace_search_time_limit')
        try:
            for seed in seeds:
                deadline();result['attempts']+=1
                sol=model.solve_position('left',local_point_m=pose['probe'],target_point_m=target,seed_degrees=seed,
                    base_matrix=pose['base'],torso_degrees=pose['torso'],maximum_evaluations=160,check_cancel=deadline)
                if not sol.succeeded:continue
                margin=model.arm_joint_limit_margin_degrees('left',sol.joint_degrees)
                clearance=model.inter_arm_clearance(sol.joint_degrees,pose['right'],pose['base'],pose['torso'])
                if margin<0 or clearance.clearance_m<0:continue
                actual=(model.forward('left',sol.joint_degrees,pose['base'],pose['torso'])@np.r_[pose['probe'],1])[:3]
                require(float(np.linalg.norm(actual-target))<.001,'Workspace FK residual exceeds existing1mm solver tolerance')
                result.update(workspace_passed=True,status='position_ik_with_joint_limits_and_inter_arm_screen',candidate=dict(
                    **asdict(sol),probe_world_m=actual.tolist(),joint_limit_margin_degrees=float(margin),
                    inter_arm_clearance_m=float(clearance.clearance_m)))
                break
        except TimeoutError:result['status']='search_time_limit_not_proof_of_unreachability'
        result['elapsed_seconds']=time.perf_counter()-start
        return result

    def check_sample(self,path,expected_sha256):
        path=Path(path).resolve();require(digest(path)==expected_sha256,'Changed native sample')
        meta=json.loads(path.read_text());pose=self.pose_and_probe(meta)
        result=self.solve(pose,meta['supervision']['nominal_world_m'])
        require(digest(path)==expected_sha256,'Native sample changed during workspace check')
        return dict(schema=SCHEMA,sample_path=str(path),sample_sha256=expected_sha256,
            target_id=meta['supervision']['target_id'],nominal_world_m=meta['supervision']['nominal_world_m'],
            source_bindings=dict(self.bindings),probe_definition='midpoint_of_actual_left_finger_visual_bounds_at_recorded_pose',
            probe_ee_m=pose['probe'].tolist(),base_fixed=True,torso_fixed=True,orientation_constrained=False,
            result=result,scope='kinematic_position_workspace_and_stationary_other_arm_screen',
            full_scene_arm_collision_checked=False,approach_path_checked=False,physical_cut_approved=False,
            visual_review_performed=False,training_approved=False)
