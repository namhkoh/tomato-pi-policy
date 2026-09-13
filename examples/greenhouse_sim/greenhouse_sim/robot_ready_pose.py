"""Pure Model A SDK ready constants; safe to read BEFORE SimulationApp.

Keep this module free of USD/Omniverse imports. Values are the same ready pose
used by robot_scene and the SDK multi-control/leader-arm examples, in degrees.
"""
SDK_READY_POSE_DEGREES = {
    **{name: value for name, value in zip(
        (f"torso_{index}" for index in range(6)),
        (0.0, 45.0, -90.0, 45.0, 0.0, 0.0), strict=True)},
    **{name: value for name, value in zip(
        (f"right_arm_{index}" for index in range(7)),
        (0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0), strict=True)},
    **{name: value for name, value in zip(
        (f"left_arm_{index}" for index in range(7)),
        (0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0), strict=True)},
    "head_0": 0.0,
    "head_1": 0.0,
}
