"""Regressions for deterministic target-conditioned base placement."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import base_planner
from greenhouse_sim import robot_kinematics


class _FakeModel:
    def solve_position_axes(self, _side, *, target_point_m, base_matrix, **_kwargs):
        advance = -float(base_matrix[1, 3])
        segment = round(float(target_point_m[2]))
        succeeded = segment < 2 or advance >= 0.03
        return robot_kinematics.IKResult(
            joint_degrees=(float(segment),) + (0.0,) * 6,
            position_error_m=0.0 if succeeded else 0.02,
            orientation_error_rad=0.0,
            cost=0.0,
            succeeded=succeeded,
        )

    def forward(self, _side, arm_degrees, base_matrix):
        matrix = np.asarray(base_matrix, dtype=np.float64).copy()
        matrix[:3, 3] = (1.0, 1.0, float(arm_degrees[0]))
        return matrix


def test_planner_advances_until_a_distal_segment_is_reachable() -> None:
    candidates = tuple(
        base_planner.GraspCandidate(
            collider=f"link_{segment}",
            body=f"body_{segment}",
            segment=segment,
            role="petiole_grasp" if segment == 3 else "petiole_cut_zone",
            centre_m=(0.0, -1.0, float(segment)),
            axis=(0.0, 0.0, 1.0),
        )
        for segment in range(4)
    )

    plan = base_planner.plan_target_conditioned_base(
        _FakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=candidates,
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        minimum_camera_clearance_m=0.0,
    )

    assert plan is not None
    assert plan.advance_m == 0.03
    assert plan.selected_grasp_segment == 3
    np.testing.assert_allclose(plan.position_m, (0.0, -0.03, 0.0))
