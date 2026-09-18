"""Reuse identical position-IK problems; still verify each recorded camera by FK."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker, require, digest


class BulkWorkspace:
    def __init__(self):
        self.checker = WorkspaceChecker()
        self.bindings = {**self.checker.bindings, str(Path(__file__).resolve()): digest(__file__)}
        self.solutions = {}
        self.checked_frames = 0
        self.solve_count = 0

    def check(self, meta):
        # This reproduces the actual camera from every recorded joint/root/mount
        # and recomputes the left-finger visual midpoint. No camera check is cached.
        pose = self.checker.pose_and_probe(meta)
        target = np.asarray(meta['supervision']['nominal_world_m'], dtype=float)
        require(target.shape == (3,) and np.isfinite(target).all(), 'Finite target required')
        # These are every numerical input consumed by WorkspaceChecker.solve.
        # Head joints do not enter its arm position-IK problem; they were checked above.
        inputs = {key: np.asarray(pose[key], dtype=float).tolist()
                  for key in ('base', 'torso', 'left', 'right', 'probe')}
        inputs['target'] = target.tolist()
        encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = hashlib.sha256(encoded).hexdigest()
        hit = key in self.solutions
        if not hit:
            self.solutions[key] = dict(inputs=inputs, result=self.checker.solve(pose, target))
            self.solve_count += 1
        saved = self.solutions[key]
        require(saved['inputs'] == inputs, 'Workspace cache collision')
        result = deepcopy(saved['result'])
        # Recheck a passing cached solution with independent FK, limits and the
        # stationary opposite arm. An old elapsed time is never called a new solve.
        if result['workspace_passed']:
            joints = np.asarray(result['candidate']['joint_degrees'], dtype=float)
            model = self.checker.model
            actual = (model.forward('left', joints, pose['base'], pose['torso']) @ np.r_[pose['probe'], 1])[:3]
            require(float(np.linalg.norm(actual-target)) < .001, 'Cached workspace FK residual exceeds 1mm')
            require(model.arm_joint_limit_margin_degrees('left', joints) >= 0
                    and model.inter_arm_clearance(joints, pose['right'], pose['base'], pose['torso']).clearance_m >= 0,
                    'Cached workspace joint limits or opposite-arm clearance failed')
        self.checked_frames += 1
        return dict(schema='greenhouse.native848_bulk_workspace.v1',
            target_id=meta['supervision']['target_id'], nominal_world_m=target.tolist(),
            solve_input_sha256=key, identical_problem_reused=hit,
            per_frame_camera_FK_verified=True, per_frame_cached_solution_FK_verified=bool(result['workspace_passed']),
            result=result, probe_ee_m=pose['probe'].tolist(),
            scope='kinematic_position_workspace_and_stationary_other_arm_screen',
            full_scene_arm_collision_checked=False, approach_path_checked=False,
            physical_cut_approved=False, training_approved=False)

    def finish(self):
        self.checker.finish()
        require(all(digest(path) == pin for path, pin in self.bindings.items()), 'Workspace implementation changed')
        return dict(checked_frames=self.checked_frames, distinct_position_ik_problems=self.solve_count,
                    source_bindings=self.bindings)
