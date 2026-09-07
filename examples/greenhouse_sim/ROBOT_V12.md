# Model A v1.2 and three D405 cameras

Implemented on `koh-dev/sim-data`, 2026-09-07. Source CAD and vendor meshes are
unchanged. No physical robot connection or command was used for validation.

## Build and launch

The installed Isaac 5.1 importer builds the asset; the current package runs it
in Isaac 6.0.1. Isaac 6 removed the legacy URDF command used by this builder.

```bat
"C:\Program Files\FreeCAD 1.0\bin\python.exe" examples\greenhouse_sim\extract_robot_hardware_v12.py
D:\isaac-sim\python.bat examples\greenhouse_sim\build_robot.py
examples\greenhouse_sim\run_sim_data.cmd
```

The build output is `data/greenhouse_sim/robots/rby1a_v1.2/rby1a_v1.2.usd`.
Its configuration layers are isolated from the retained v1.0 build. The builder
uses byte-identical, USD-safe aliases for four vendor filenames containing dots;
without this, the legacy importer produces invalid SdfPaths and a null prim.
Kinematics, default assets, tool frames and head limits now use the same v1.2
URDF. The old extra force-sensor joint is absent: arm_6-to-EE distance changes
from 154.8 to 126.1 mm. The cached counterhold seed was re-solved without relaxing
the 38-degree joint-reserve / 76 N force-capacity / 50-evaluation regression gates.

The package opens on **Robot head D405**. The preview panel also provides
**Left wrist D405**, **Right wrist D405**, and the three inspection views.
Robot placement is in the aisle at `(central_gutter_x + 1.0, 0, 0)` m, yaw 180
degrees, SDK-ready torso/arms, head pitch -15 degrees. The right tool remains
knife-only. The fixed preview pose is computed from the v1.2 URDF; it does not
use servo stepping, teleop, or plant dynamics. Use `--no-robot` for anatomy-only
review. Existing review files are retained.

## Mounting geometry

| Mount | Geometry / datum | Implementation |
|---|---|---|
| Head bracket | Supplied `HeadCam_Bracket_D405.FCStd`, saved final `Body/Fillet001`; four 36 x 26 mm holes | Exported saved BRep without recomputing newer FreeCAD features; rotated 90 degrees about Z onto NECK_2's 26 x 36 mm pattern, centred at `(0, -0.34, 45)` mm in head_2 |
| Head D405 | Camera's rear 20 mm pair; bracket holes at x=+/-10, z=40 mm | Rear mating face at bracket y=-18 mm; camera looks forward along head +X. The old optical-origin/window alignment was not a valid screw alignment |
| Both wrist brackets | Exact supplied `D405_Wrist_Bracket_v2-Body.stl`, unchanged 40 x 44 x 41 mm geometry | Robot-side 18 mm pair and camera-side 20 mm pair; rigid handed rotations place them on opposite wrist faces, looking toward EE -Z |
| Wrist connection | Official LINK_13/20_v1.1 meshes have a 34 mm pair, not the old high-speed-gripper CAD's 18 mm pair | Explicit **simulator-designed adapter**, with a 5 mm plate and 10 mm standoffs, bridges the patterns; not silently represented as a direct bolt-on fit |

The wrist screw datums are `(x=+/-17, y=-37/+37, z=-98.58)` mm in right/left
arm_6, or z=27.52 mm in the v1.2 EE frame. The adapter and bracket are both
included in collision-planning bounds. USD uses convex-decomposed bracket and
adapter collisions, avoiding a solid bounding box across their open regions.
The right D405 body is rotated on its symmetric rear pair; its optical axes
remain attached to the body rather than being independently rolled for display.

**The adapter is not supplied lab hardware and is not approved for fabrication.**
Its load capacity, fastener length, threading, cable routing, contact seating and
full arm-motion mechanical clearance need lab review. Replace its procedural
geometry with the real adapter CAD when available. The source bracket files are
not modified to force an incorrect hole pattern to fit.

## Camera contract and limits

All three fitted cameras use **848 x 408**, as requested. Head/wrist viewport
selection, teleop/probe/RL recording defaults and the camera smoke renders use
that resolution. Inspection viewports remain separate. Explicit recording-size
CLI overrides remain available.

Intrinsics are a **nominal synthetic pinhole**, not a calibrated D405 mode:
horizontal FOV 84 degrees, square pixels, principal point `(424, 204)` and
vertical FOV derived from the requested aspect ratio. The authored camera stores
resolution, K, role and calibration status. No physical calibration, distortion,
stereo/noise realism or registered RGB-D export is claimed by this increment.

Phase 1 gallery images still use a diagnostic close-up camera and contain review
overlays. They are **not** clean head-camera training images or approved cutting
labels. Paired head-view annotation/export remains subsequent dataset work.

## Evidence and validation

```bat
D:\isaac-sim-6.0.1\python.bat examples\greenhouse_sim\verify_robot_v12.py
D:\isaac-sim-6.0.1\python.bat examples\greenhouse_sim\launch_sim_data.py --headless --frames 30 --output data\sim_data\robot_v12_package_final
```

- 311 focused robot/hardware/kinematics/interactive/teleop/review tests passed,
  10 subtests passed; one live-SimulationApp-only test skipped in standalone pytest.
- Actual Isaac 6 fit smoke: seven rendered views, including all three 848x408
  camera prims. [Report](../../data/sim_data/robot_v12_fit/report.json).
- [Head mount](../../data/sim_data/robot_v12_fit/head_mount.png),
  [left wrist mount](../../data/sim_data/robot_v12_fit/left_mount.png),
  [right wrist mount](../../data/sim_data/robot_v12_fit/right_mount.png),
  [whole robot](../../data/sim_data/robot_v12_fit/overall.png).
- Bounded actual greenhouse/head-view preview:
  [RGB image](../../data/sim_data/robot_v12_package_final/preview.png),
  [status](../../data/sim_data/robot_v12_package_final/status.json).
- Existing isolated dynamics fit inspector successfully initialized 22 ready
  joints, all three cameras and the knife-only tool in Isaac 6:
  [report](../../data/sim_data/robot_v12_fit/dynamic_report.json).

These are fit/import/camera checks, **not** full v1.2 bimanual grasp/cut or online
RL acceptance. The supplied package still has no plant grasp/cut dynamics.
Previously validated trajectories/checkpoints must be revalidated with the new
tool offsets and hardware collision envelopes before benchmark use.
