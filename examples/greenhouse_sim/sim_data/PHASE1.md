# Phase 1: asset audit and anatomy review

Status (2026-09-07): automatic audit, batch gallery, exception sampling and saved
review history are implemented. Initial human reviews exist on two plants;
broader sampling, horticultural cut/grasp rules and physical validation remain open.

Overall plan: [vlm_train_data.md](../../../../vlm_train_data.md).

## Run the package audit

From the repository root:

```bat
examples\greenhouse_sim\run_sim_data_audit.cmd --check-usd
```

The default source is the supplied, unpacked package under
`data/sim_data/package_20260905/tomato_greenhouse_pack`. Use `--package` to
override it. Output goes to a new timestamped, gitignored directory under
`data/sim_data/audits/`. An explicit `--output` must not already exist and must
be outside the source package. Exit code 2 indicates a blocked manifest or
failed USD audit. A zero exit code does **not** approve any cutting target.

The manifest-only audit can also run without Isaac or third-party packages:

```bat
cd examples\greenhouse_sim
python -m sim_data.audit
```

`--check-usd` additionally requires `pxr`, provided by the Isaac launcher. It
assembles each plant into an anonymous stage's session layer, using
parent-subtracted translations, and compares component placement and each
component's own mesh bounds. Descendant leaves must not enlarge a parent
petiole's geometry bounds. All original component layers remain read-only.

## Report contract

Each plant JSON contains:

- Manifest and component-USD SHA-256 fingerprints.
- Coordinate convention and structural errors/warnings.
- Parent graph, attachment positions, axes, and capsule polylines.
- One review record per substem, including excluded stubs.
- Expected detached component IDs and type counts.
- Explicitly separate anatomy review, agronomic eligibility, observability,
  and physical-execution status.
- Optional USD placement and attachment-to-own/parent-bounds diagnostics.

`summary.json` aggregates counts and warning codes. These are review artifacts,
not training labels. Textures are not included in the component-USD fingerprint;
full rendering provenance belongs in the later synchronized capture stage.

### Initial rules: `intact_leaf_petiole_review.v1`

An intact substem directly attached to a main-stem component, with leaf
descendants and no protected descendants, is `needs_review`.

Already-deleafed branches, non-primary petioles, branches without leaves, and
branches carrying protected structures are excluded from this initial task.
Unknown deleafed state or structural errors block the affected review candidate.
The protected types are main stem, truss, fruit, and flower. Unknown organ types
block the structural audit rather than being silently treated as safe.

No candidate receives a canonical cut point, admissible cut region, or grasp
region. Those fields stay null until the rules are reviewed and implemented.
The attachment marker must not be interpreted as the approved knife location.

The 2 mm attachment-to-bounds diagnostic is a numerical review threshold, not
an agronomic clearance rule. AABB agreement is not exact surface attachment,
reachability, collision freedom, or biological/physical validity.

## Open the visual review mode

### Floor-aligned robot preview (2026-09-07)

The package launcher now seats the complete RB-Y1 using actual triangle hits
on `/World/Environment/GreenHouse/floor` beneath the rendered wheels. The local
floor top is approximately Z=0.101 m; the distant raised strips at Z=0.111 m
must not be used as the robot's floor height. Both wheels and sampled chassis
footprint must have level support. Missing/hidden geometry, unsupported floor
topology, and uneven support fail explicitly rather than falling back to Z=0.

The translation is a session-only geometric placement, not gravity settling or
a full collision/physics test. All three cameras move with the robot. Launch
`status.json` records the floor path, root position, wheel clearances and limited
scope in `robot.floor_alignment`. Source layers and human decisions are unchanged.
Isolated hardware-render tools without a floor explicitly report placement as
`not_requested`; the greenhouse package launcher always requests alignment.

Validation: `D:\isaac-sim-6.0.1\python.bat -m pytest
examples/greenhouse_sim/sim_data/floor_alignment_test.py
examples/greenhouse_sim/sim_data/audit_test.py
examples/greenhouse_sim/sim_data/batch_test.py -q` from the repository root:
52 passed, plus 10 subtests. Includes the actual supplied floor/v1.2 robot,
session-only/idempotent placement, transformed parents, floor holes/raised strips,
missing/hidden/out-of-bounds geometry and non-level-support rejection.

### Start/resume review

The annotation launcher defaults to `--right-tool gripper`: the stock right
gripper body and both finger visuals are restored, the knife subtree is inactive,
and both wrist D405 assemblies plus the head camera are retained. Grippers start
closed at the URDF zero positions. These are static visuals, not working grasp
contacts. The generated source asset and separate physics workflow stay knife-only.
`--right-tool knife_only` selects the legacy preview geometry; its current knife
mount projects back toward the wrist and needs a fit correction before reuse.
It must not be interpreted as validated hardware fit or physical execution.

The restored gripper was checked in a real nine-view Isaac render at
`data/sim_data/robot_v12_gripper_restore_20260907_1441/`; `right_hand.png` includes
the complete stock gripper and fingers, and `report.json` records `gripper` mode.
55 focused preview/floor/review tests passed (plus 10 subtests). No source USD,
camera mount, joint-limit contract, or human annotation was overwritten.

From the repository root:

```bat
examples\greenhouse_sim\run_sim_data.cmd --review --review-plant seed101_full
```

`--review-plant` selects the first assembled plant and implies `--review`.
The new review catalog covers all 24 plants without relaunching for each one.
Only one extra plant is loaded at a time, at the first detailed plant's placement.
While inspecting that sample, the original detailed plants are temporarily hidden
and then restored. These are inspection views, not original-placement visibility
labels or robot camera observations.

Review launches now default to fresh `data/sim_data/phase1_review_<timestamp>_<id>`
directories. Previous `data/sim_data/*/anatomy_reviews` directories are read
automatically, including legacy individual review files. For custom locations,
add `--review-history D:\path\to\anatomy_reviews` (repeatable). `--output` still
overrides the current output directory and must be outside the source package.

The timeline is paused on entry. Keep it paused during anatomy review.

### Faster workflow: six-image gallery

1. In **Phase 1 - batch anatomy review**, keep **Mixed sample** selected.
2. Click **Build six-image gallery**. Wait for the ready message before moving
   the viewport camera; six asynchronous captures share the existing viewport.
3. Inspect each thumbnail. Choose **Anatomy matches** for correct anatomy, then
   explicitly check those cards or click **Select eligible captured cards**.
   That button excludes unconfirmable stubs/blocked targets when confirming
   anatomy. No selection is made automatically.
4. Click **Save selected + next page**. Your reviewer name is mandatory; the
   preset reason provides the recorded notes, so repetitive typing is unnecessary.
   Optional additional notes come from the main panel.
5. For uncertain cards, use **Inspect in 3D**, orbit/isolate, then choose a preset
   reason and **Save current reason + next**. Saving explicitly supersedes earlier
   decisions for that target without deleting or rewriting any original records.

Choose the reason before using the select-eligible button. If a mixed selection
contains an excluded/blocked target, the entire anatomy-confirmation batch is
rejected rather than partially saved; the error identifies the invalid target.
Use separate batches for different reasons. Unchecked and skipped cards remain
unreviewed. They are skipped only for this gallery pass and can be revisited via
**Restart sample** or another queue; this is not approval.

Queues:

- **Mixed sample**: up to two flagged, three normal and one negative example per
  six-card page where available. Each group is plant-balanced and hash-shuffled
  deterministically, not consecutive substem indices. This is a coverage sample,
  not a statistical accuracy guarantee or an instruction to review all targets.
- **Exceptions**: geometry/structural flags, unresolved feedback, conflicting
  decisions and ambiguous legacy exclusions. Human issues appear first.
- **Unreviewed**: remaining decisions, balanced across plants.
- **All / revisit**: includes already reviewed targets for corrections.

Old "out of bounds" / "too high" notes are preserved and flagged for clarification,
not automatically reinterpreted. The new presets separate **wrong anatomy**,
**outside camera view**, **occluded**, and **outside intended workspace**. A workspace
exclusion is reviewer intent, NOT measured robot reachability or anatomy failure.
Visibility reasons remain unresolved for anatomy; explicit workspace exclusions
are skipped as completed exclusions, not counted as anatomy confirmations.

History is accepted only for matching manifest/component hashes and rule version.
Stale/malformed records cannot approve a target. Conflicting decisions stay flagged
until an explicit new review supersedes them. Completed current decisions are
skipped on relaunch. Previously skipped/unresolved items can reappear.

Additional controls:

- Previous/next substem, including old stubs for negative review.
- Camera focus on the selected attachment.
- Yellow attachment marker, green detached-subtree bounds, blue manifest
  centerline, and red bounds on nearby protected main-stem/fruit structures.
- Target isolation with reversible visibility changes.
- Preset reasons, optional additional notes, and automatic advance after saving.
  Excluded/blocked targets cannot be anatomy-confirmed.
- Close-and-restore, including the previous camera and visibility opinions.

The overlays are diagnostics, not exact segmentation or exhaustive obstacle
maps. Review views must not be used as model inputs. Closing review removes
the session-only overlays; it does not save or modify source USDs.

Review decisions are new UUID-named JSON files under the launch output's
`anatomy_reviews/` directory. Individual files use `greenhouse.anatomy_review.v2`;
batches use `greenhouse.anatomy_batch.v1` with a `records` list of individual
decisions. Readers must handle both formats (and original v1 records).
A batch is one atomically published file: all selected cards and source hashes
are checked first. No failed batch produces partial approvals. Preset notes,
scope, reviewer, source fingerprints and explicit supersession IDs are retained.
Batch records additionally reference the captured image SHA-256 and camera pose.
Review captures and `gallery.json` live under `review_gallery/<id>/`; they are
not dataset inputs, exact segmentations, or validated occlusion measurements.
Even `anatomy_confirmed` records have `cut_approval: false` and
`physical_executability: not_tested`.

The gallery uses matching prior USD audits for geometry flags rather than running
24 USD assemblies on every launch. Missing/stale checks show **geometry_not_checked**,
never a pass. Run `run_sim_data_audit.cmd --check-usd` after asset/rule changes.

## Robot POV versus anatomy inspection (2026-09-07)

When the robot is loaded, review now defaults to its **actual mounted head D405**
at **848x408**, including target navigation and six-card gallery capture. The old
automatic floating close-up was useful for anatomy but did not match the robot's
view or the fixed-base reach test. **Robot head POV / refresh** restores the full
scene and selects that camera; **Diagnostic close-up** explicitly selects the
floating inspection view at 1280x720. The chosen mode persists across targets.
Without a loaded robot the panel starts in diagnostic mode, never fake robot POV.
Neither button moves/re-aims the robot or changes the camera's mounted transform.

The panel shows an attachment-point projection check at selection/refresh:
inside image frustum, outside image, behind camera, or outside clipping range.
**Inside image is not proof of visibility:** occlusion is not measured here.
Review/gallery records now include camera path, resolution, intrinsics and pose,
robot/plant poses, attachment position/projection, and explicit diagnostic-only
scope. Gallery capture rejects changed camera/pose/resolution context. Existing
annotations are preserved; no old view context is invented. Review overlays are
still present, so these images are NOT clean VLM training observations.

Reachability is a separate question. An out-of-range result now explicitly says
**from CURRENT base/torso pose** and that repositioning was not searched. Do not
exclude an anatomical candidate solely for failing at the robot's parking pose.
The reviewed seed103_full/SubStem_47 snapshot put the attachment at approximately
(0.210, 1.289, 1.754) m while the base was at (0.800, 0, 0.101) m. Moving only
along the row to Y=1.289 m reduced the conservative shoulder distance to 0.796 m,
still above the 0.766 m probe bound with the existing torso. Thus being opposite
the target helps, but does not guarantee reach without suitable standoff/torso.
This was a hypothetical numeric check, not an applied or collision-safe stance.

Next: select valid floor-supported, gutter-clear base/torso observation stances,
then render the head POV and evaluate arm reach for each stance. Keep anatomy,
view-dependent occlusion, and stance-dependent reach labels separate. Base/torso
stance search, automatic robot relocation, and full collision-safe feasibility
are not implemented by the POV change.

Validation: 344 focused tests and 10 subtests passed (one skip). Isaac smoke at
`data/sim_data/reachability_smokes/20260907T061610Z_785a6e97/` passed actual head
POV persistence, explicit close-up switching, six mounted-head 848x408 gallery
captures, unchanged robot/source/review data, and existing reachability checks.
The rendered UI was visually inspected. The fixture is deliberately relocated
for a known position-IK case, not evidence of a valid production robot stance.

## Automatic robot reachability diagnostics (2026-09-07)

In the Phase 1 panel, select an intact target, restore the non-isolated scene,
pause the timeline and press **Check both arms** under **Automatic reachability**.
The v1.2 robot must be loaded. This does not move the simulated or physical robot.
Stock grippers are supported; a missing/knife-only gripper is reported as unsupported
for that arm's gripper probe, not as an unreachable target.

The checker reads actual link transforms and reconstructs the measured joints
against the exact v1.2 URDF. Arbitrary link edits, scaling and out-of-limit poses
are rejected. It uses the selected component's current world transform, not a
hard-coded robot/plant placement or imported initial joint-state attributes.
Each arm is evaluated independently with base, torso and the other arm fixed.

| Result | Meaning |
| --- | --- |
| `position_ik_found` | A joint-limit-compliant position solution within 1 mm was found. Orientation is unconstrained. |
| `outside_outer_reach_bound` | Beyond a conservative shoulder-to-probe upper bound for this fixed base/torso. Repositioning may change the result. |
| `no_solution_found` | Bounded numerical search failed; NOT proof of unreachability. |
| `time_budget_exhausted` / `cancelled` | Incomplete search, not a negative feasibility label. |
| `unsupported_or_missing_gripper` | No two-finger probe available; knife reach has not been assessed. |

The diagnostic target is the manifest attachment, NOT an approved grasp or cut
point. The probe is the midpoint of rendered finger-bound centres, NOT a calibrated
grasp TCP. The search uses up to five deterministic seeds, 160 optimizer evaluations
per seed and four seconds per arm (including endpoint screening). The bounded
numeric work runs off the UI thread; USD snapshot collection stays on the main thread.

The first position solution gets a **partial conservative endpoint overlap screen**:
arm capsules against the stationary other arm, and arm/tool envelopes against
selected-plant component AABBs and gutter AABBs. `possible_overlap` may be a false
positive from coarse bounds or intended target contact; it does not erase the
position solution. The checker does not keep searching for collision-free alternatives.
It does NOT certify full self-collision, other plants/greenhouse structures, moving
tool vs stationary arm/tool, approach paths, grasp orientation/force, cutting or
simultaneous bimanual execution. A clear partial screen is not safety approval.

**Show left/right IK outline** draws orange arm centreline segments and the probe
point in a session-only, non-physical diagnostic overlay. It does not command joints
and is not an animated trajectory or a full collision envelope. Hide it for clean
views; changing targets, screened scene geometry or robot transforms invalidates
the result and removes it automatically. Closing review also removes it.

Completed diagnostics are atomically saved under the launch output's `reachability/`
directory as `greenhouse.reachability.v1` JSON, separate from human reviews. They
include source hashes, current transforms/joints, target/probe definitions, tool
envelopes, search budgets/errors and overlap details. Gallery-borrowed plants are
explicitly marked `relocated_review_sample`; no result is asset-global reachability.
Stored results describe their snapshot, not future scene poses. No human anatomy
decision, training label or cut/grasp approval is written or changed.

Verified: 335 regression tests and 10 subtests passed (one skip). Real Isaac UI
smoke: `data/sim_data/reachability_smokes/20260907T055757Z_e076afe6/`, containing
`result.json`, diagnostic JSON and `diagnostic_fixture_ui.png`. The fixture plant
was deliberately relocated to a known attainable point to exercise the UI; both
position solutions showed possible overlaps. This is UI/kinematics evidence, NOT
a safe grasp/cut demonstration. The screenshot was visually inspected; cancellation,
stale-result invalidation, unchanged robot transforms and preserved sources/reviews
were checked. Run it with `D:\isaac-sim-6.0.1\python.bat -m sim_data.reachability_smoke`
from `examples\greenhouse_sim`.

## Validation commands

```bat
cd examples\greenhouse_sim
D:\isaac-sim-6.0.1\python.bat -m unittest sim_data.audit_test sim_data.batch_test -v
D:\isaac-sim-6.0.1\python.bat -m sim_data.review_smoke
D:\isaac-sim-6.0.1\python.bat -m sim_data.review_smoke --capture-ui
```

The smoke test starts with one real plant, catalogs 24, captures a six-card
multi-plant gallery, and checks navigation, isolation/restoration, default-empty
selection and rejection of empty reviewer identity. It writes `review_only.png`,
gallery images and `result.json` to a fresh ignored run directory. It verifies
source preservation and closes the app. No human approval is produced.

Verified on 2026-09-07: 42 focused tests passed under Isaac Python. The final
UI-enabled smoke is `data/sim_data/review_smokes/20260907T023900Z_46a955da/`,
with six real thumbnails, `gallery_ui.png`, and a passing `result.json`.
All 24 manifest/component fingerprints were unchanged. The UI screenshot shows
the deliberately tested empty-reviewer rejection; no human decisions were saved.
Earlier headless screenshots hid the UI by default; `--capture-ui` now explicitly
sets `hide_ui=False` and the final screenshot was visually inspected.

The existing visible application keeps the old Python classes until relaunched.
Save any pending notes, close that window, and use the launch command above to
load the new panel. Saved decisions resume automatically; unsaved text does not.

## Initial audit findings (2026-09-06)

- 24 plants: 870 intact leaf-bearing petioles require review; 971 stubs excluded.
- No structural blockers in the supplied manifests.
- 492 zero-length capsule-chain warnings, all on already-deleafed components.
- Assembled translations agree to approximately 2.24e-16 m maximum error.
- 91 attachment-to-parent-AABB diagnostics exceed 2 mm (90 substems, one
  truss); maximum observed gap is about 4.23 mm. These need review, not silent
  acceptance or automatic source-asset editing.

Initial all-plant USD audit:
`data/sim_data/audits/20260906T073659Z_1f43ab9a/`.

The headless review UI smoke passed at
`data/sim_data/review_smokes/20260906T073913Z_d9c0e596/`, including a rendered
review-only capture and unchanged source fingerprints. Earlier smoke attempts
are retained as failure diagnostics, not successful evidence. Capture checks
wait for a complete PNG, not just the asynchronous viewport callback, and
errors are persisted before Isaac's fast shutdown.

## Remaining Phase 1 acceptance work

### Optional lower-branch candidate preview (2026-09-07)

From the repository root:

```bat
examples\greenhouse_sim\run_sim_data.cmd --extra-cut-candidates 3
```

Adds three attached leaf-bearing petioles to each of the two detailed foreground
plants (`seed101_full`, `seed103_full`), six total. Complete existing donor
subtrees are translated to existing lower-main-stem attachment nodes; the empty
recipient stubs are hidden only in the session layer. Donor geometry, original
source files, robot pose, backdrop instances and saved reviews are preserved.
These are attached candidates that could become orphan branches AFTER cutting,
not detached or floating branches. Use **Added lower branches** in the preview
controls for a diagnostic close-up; all three robot D405 views remain available.

`--extra-cut-candidates 0` (the default) omits augmentation; values 1 and 2 add
that many branches per foreground plant. Nonzero values create a unique default
output directory containing `candidate_branches.json`, `preview.png` (mounted
head view), `added_lower_branches.png` (diagnostic view), and `status.json`.
The sidecar retains donor hashes, derived IDs, complete component/parent mapping,
replacement sites, world attachments, variant fingerprint and original plant
split group. Do not count these copies as independent source plants in splits.

Scope: **static visual augmentation only**. Capsule/mesh-bound attachment checks
do not certify botanical eligibility, inter-branch clearance, reachability,
grasp/cut dynamics, or executable trajectories. All added targets remain pending;
cut points/regions and grasp regions are null. The current annotation panel reads
original manifests, so combining augmentation with `--review`/`--review-plant`
is rejected until a derived-plant annotation adapter is implemented. No existing
approval is inherited. Source robot camera-only USD overrides are preserved on
restart; they are distinguished from a fully defined existing robot.

### Assisted labels for added branches (draft packet)

From `examples/greenhouse_sim`, use Isaac Python to prepare version-matched
anatomy proposals and three diagnostic images per added branch:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.label_drafts --variants ..\..\data\sim_data\candidate_preview_20260907_160045_809b00\candidate_branches.json
```

The output under `data/sim_data/label_drafts/` contains `review.md`,
`draft_labels.json`, and normal-appearance / isolated colour-labelled renders
plus an alternate attachment close-up. The 9 mm yellow diagnostic marker is
exaggerated for readability, not a physical region or proposed cut interval.
`--no-render` creates metadata-only drafts without starting Isaac. Output must
be new and outside the source package; no previous reviews are overwritten.

Blue = target petiole; green = its leaf subtree; red = main-stem parent;
yellow = attachment, NOT the cut point. Review IDs B01-B06 are local to that
packet and map to full variant/target identities in JSON. These are offline
plant-frame reconstructions, not live captures, robot POV, or training inputs.
Isolation removes occlusion and must not be used to claim greenhouse visibility.
Sources/variant geometry are re-audited before and after generation; mismatches
fail rather than importing stale or modified labels. All human decisions remain
pending and cut/grasp regions stay null. Blank/material-loss labelled renders
fail a colour-presence check; this does not replace visual anatomy review.
This draft helper does not yet enable
the original-manifest review UI for added variants or certify physical execution.

### Prototype cut-region review (2026-09-07)

The user-agreed engineering rule is recorded in
`sim_data/configs/cut_rule_prototype_v1.json`: 10 mm nominal, accepted 10-20 mm
along the petiole centreline from its manifest attachment towards the leaves.
This is an asymmetric +0/+10 mm interval, NOT a spherical radius or a guarantee
of external stub length. Horticultural suitability remains pending grower review.

From `examples/greenhouse_sim`, add the rule to the draft command:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.label_drafts --variants ..\..\data\sim_data\candidate_preview_20260907_160045_809b00\candidate_branches.json --cut-rule sim_data\configs\cut_rule_prototype_v1.json
```

The new packet retains anatomy views and adds a fourth cut-region close-up for
each supported target. White = nominal point; magenta = 10-20 mm segment;
yellow = attachment. Display spheres/band thickness are exaggerated and do not
define radial acceptance. Geometry is obtained by piecewise-linear arc-length
interpolation of the supplied capsule chain, retaining bends within the interval.
Attachment-matched reversed chains are handled; ambiguous multi-chain geometry,
degenerate/non-finite points, inconsistent direction, non-petiole/deleafed targets,
missing attachment matches and insufficient length produce explicit flags with no
cut coordinates. No extrapolation or default axis-based point is substituted.

`cut_region_proposal` stores rule identity/hash, proposed nominal position,
centreline interval with radii/tangents, and a suggested local cut-plane normal
(NOT a full tool pose). Existing approved-label fields stay null. Per-target
anatomy/cut approval, horticultural validation, blade/protected-organ clearance,
robot reachability and physical execution remain separate and unapproved.
Omitting `--cut-rule` preserves anatomy-only exports. Prior packets/source assets
and reviews are not overwritten. Review these six proposals before the planned
20-sample synchronized robot-head RGB-D pilot. A smaller nine-view static
capture pilot is now implemented below; the full semantic exporter is pending.

The packet also reports the nominal point's approximate distance to the parent
capsule surface. B02 (~0.95 mm) and B04 (~1.91 mm) are closer than their petiole
radii and carry a potential envelope-overlap warning. Review their attachment
geometry before approval. This diagnostic checks neither the exact parent mesh
nor the full accepted interval or blade stroke; no warning does not mean safe.
The manifest attachment may lie inside the main-stem envelope, so 10 mm along
the petiole does not imply 10 mm outside the stem surface. Final six-target
review packet (24 images):
`data/sim_data/cut_region_drafts/20260907_235517_89790c/review.md`.

### Full-greenhouse RGB-D capture pilot (2026-09-08)

Run from `examples/greenhouse_sim` with Isaac 6.0.1 Python:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.capture_pilot --drafts ..\..\data\sim_data\cut_region_drafts\20260907_235517_89790c\draft_labels.json
```

This is a separate bounded headless process, not a command to the live simulator
or lab robot. It loads the supplied greenhouse, the same two detailed plants and
142 backdrop instances, all six added branches and the complete stock-gripper
RB-Y1 v1.2. Unbundled external props are excluded exactly as in the existing
preview. Plants are not isolated, recoloured or moved for capture. Only in the
disposable session, existing gutter rigid bodies and robot dynamics are frozen.
An anonymous root wrapper preserves source-relative asset resolution and keeps
Kit root-layer bookkeeping away from source USD files. Loaded source file
hashes and dirty-layer checks are verified before publishing a successful run.

Nine planned images use provisional targets B03/B05/B06 at three whole-robot
aisle offsets with URDF-limited head-joint framing. The camera stays mounted;
there is no floating camera or ground-truth-centred crop. Target pixel positions
vary, but this remains a geometry-guided pilot, not a deployment/evaluation
view-selection policy. Joint/floor checks do NOT validate collision-free poses,
arm reachability or paths between snapshots. B01/B02/B04 are held for junction
checks; neither the previous human reviews nor cut approvals are changed.

Each `sample_000X/` contains:

- `inputs/rgb.png`: lossless 848x408 clean full-scene RGB, no debug overlays.
- `inputs/depth_m.npy`: raw float32 optical-axis Z in metres, not ray distance.
- `inputs/depth_valid.png`: separate mask rejecting zero, non-finite and
  out-of-clipping-range depth. Raw invalid depth is preserved in the NPY file.
- `sample.json`: intrinsics/extrinsics, robot/plant transforms, source IDs,
  provisional world/camera/pixel cut coordinates, scene-state fingerprint,
  capture checks and per-file SHA-256 values. Only the selected target is labelled;
  this is NOT an exhaustive annotation of every eligible branch in the image.
- `review/overlay.png`: projected white nominal point/magenta interval; never
  a model input. A marker can indicate HIDDEN geometry, not an observed surface.

The run's `manifest.json` records configuration, loaded source USD hashes,
variant provenance, actual Isaac version, code hashes and the calibration smoke
results. `review.md` pairs clean images with overlays. Outputs must be new and
outside the source package; failures are retained as `failed_do_not_train`.
Use `--max-samples 1` for a one-view smoke or `--smoke-only` for calibration only.

Synchronization scope is deliberately **static-only**. RGB, depth, camera
parameters and native reference metadata are copied from one Replicator writer
payload. This installed renderer reports zero time/NoFrameNumber for paused
renders, so we do NOT invent an engine frame ID from the callback counter.
Instead, capture requires unchanged scene/material/pose fingerprints, matching
rendered camera matrices/projection, fresh writer callbacks and changed RGB AND
depth buffers after a distinct camera pose. Enabled rigid bodies and animated
attributes are rejected. A real-GPU preflight changes both camera position and
known plane depth (2.0/2.4 m), checks on/off-axis Z and both buffers' freshness.
Dynamic frame-ID synchronization and moving-demo recording remain unsupported.
The installed graph emits host-buffer sync-cycle warnings; native dynamic timing
needs a separate fix before using this path for robot motion data.

Depth evidence is a conservative single-pixel centreline test with radius + 3 mm
tolerance, not an organ-instance mask, visibility fraction or a safe-cut decision.
It distinguishes foreground-occlusion evidence, depth consistency, invalid depth,
no matching surface and out-of-frame. The head-camera model is ideal pinhole;
real D405 depth noise/material failure behavior is not modelled. Difficulty,
grasp regions, physical paths, instance masks and training approval remain unset.

Reference: [NVIDIA Replicator annotator and writer documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html).

Verified pilot: [nine full-scene input/overlay pairs](../../../data/sim_data/rgbd_pilots/pilot_20260908_005526/review.md).
The manifest is `pilot_ready_for_review` with unchanged source assets. All nine
records and 36 image/array hashes passed an independent post-publication audit;
the selected robot/sim-data tests passed 407 tests (one skipped, 47 subtests).
All nine overlays were visually inspected. Eight samples have depth consistency
without verified target visibility; sample_0009 flags foreground occlusion.
At native resolution these petioles are only about 2.2-2.6 pixels wide, and the
cut intervals about 2-5 pixels long, with strong backlighting. Improve physically
appropriate viewpoints and add organ-instance/visibility evidence before
scaling collection. This pilot is not a training-approved dataset.

### Viewpoint refinement and rendered organ visibility (2026-09-08)

The baseline command above is preserved. Opt into the refined pilot with:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.capture_pilot --drafts ..\..\data\sim_data\cut_region_drafts\20260907_235517_89790c\draft_labels.json --refine-views
```

`capture_viewpoints.py` generates 12 bounded base-XY/head-joint candidates per
target: original X and 0.1/0.2/0.3 m nearer, with -0.3/0/+0.3 m aisle offsets.
It preserves robot yaw, arm/torso pose, physical camera mounting, 848x408 optics,
lighting and every plant. Floor-support and head-joint checks remain enabled.
Candidate snapshots are not navigation commands or validated paths.

The geometry screen includes visible robot boundables and scene instances.
World AABBs provide the broad phase. Large merged backdrop boxes are refined
against actual plant triangles and enclosing robot-local boxes, with a 10 mm
margin. Mixed triangle/quad assets use both quad diagonal choices conservatively.
Structural geometry retains conservative box checks. Possible intersections
are rejected, not called exact collisions. Self-collision, dynamics, hidden
collision geometry, paths and containment inside closed plant volumes are not
certified. No foliage is removed to create a passing view.

`capture_visibility.py` consumes uncolorized native renderer instance IDs from
the same static writer payload as RGB-D. A GPU smoke verifies a known 2.0/2.4 m
surface and the identity/depth change when a foreground cube is introduced and
removed. Real greenhouse masks must also change after camera movement. Native
dynamic frame-ID limitations of the baseline still apply.

Exact renderer prim paths map to source/variant component IDs. The deepest
component owner wins, so child leaves are not mislabelled as their parent
petiole. The two detailed plants have organ-level provenance; backdrop plants
and greenhouse props retain renderer identity but no invented organ labels.

Additional files, all separate from `inputs/`:

- `supervision/renderer_instance_id.npy`: raw uint32 native instance IDs.
- `supervision/component_id.npy`: uint32 component IDs for detailed plants;
  zero means unmapped, not necessarily background.
- `supervision/organ_type.png`: numeric organ IDs, keyed in `identities.json`.
- `supervision/target_visible.png`: binary exact visible-petiole identity mask.
- `supervision/identities.json`: renderer paths, component catalogue and scope.
- `review/visible_target.png`: green highlight of exact target-mask pixels only.

The nominal point needs BOTH exact petiole identity and consistent optical Z.
The interval reports coverage over unique projected pixels, conservatively
requiring all centreline samples sharing a pixel to agree. This is not an
amodal surface visibility fraction or an executable cut decision. Occluding
leaf/component identities are reported when supported; unknowns remain unknown.

`capture_search.py` ranks rendered candidates and retains up to three per target,
preferring 15 cm base-XY diversity without trading a passing quality gate for a
failed view. Provisional clear-view gates require: nominal identity/depth match,
the full interval in frame, at least 80% sampled-pixel coverage, estimated
petiole width >=3 px, interval length >=4 px, and no more than 60% of the visible
petiole mask below luminance 20/255. These are engineering checks, not validated
learning sufficiency or easy/medium/hard definitions. All candidate decisions
and rejection reasons are retained in `viewpoint_search.json`; selected images
that fail a gate remain explicitly diagnostic. No human approval is inferred.

Verified run: [nine refined input/overlay/mask examples](../../../data/sim_data/rgbd_pilots/refined_20260908_135740/review.md).
36 poses were screened, ten rejected and 26 rendered. Three selected examples
pass the provisional clear-view gates: B05 samples 0004/0005 and B06 sample 0007.
Their projected widths are 3.08-3.61 px and intervals 7.38-8.97 px. B03 still
needs better useful viewing geometry; the remaining six selections are explicit
diagnostic examples, not accepted training images. All nine overlays were
visually reviewed; 90 exported-file hashes, mask ownership, calibration, 872
source USD hashes and six capture-module hashes were independently checked.
Regression suite: 435 passed, one skipped, 47 subtests passed. The headless run
completed in about 19.6 minutes alongside the live preview; native annotation
and warm-up throughput needs profiling before collection is scaled.

### Native-depth colour review

Saved Replicator `distance_to_image_plane` arrays can be displayed directly,
without estimating depth from RGB, reconstructing geometry or rerunning Isaac:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.depth_preview --run ..\..\data\sim_data\rgbd_pilots\refined_20260908_135740
```

This writes a NEW `depth_heatmaps/` companion directory and refuses overwrite.
For subsequent runs on the same inputs, supply a new `--output` directory.
All samples share a linear 0.04-2.0 m near-scene display scale by default
(`--near`/`--far`); additional images show the full camera clipping range.
Yellow is nearer, green/blue intermediate, purple farther. Grey checkerboard
means invalid/no depth. Legend endpoint colours saturate; raw float32 metre
values, RGB, validity masks, cut labels and original capture metadata remain
unchanged. The original 848x408 pixel plane is intact, with a legend added below
it. These PNGs are review-only, not replacement metric observations.

[Latest nine RGB/depth heatmap pairs](../../../data/sim_data/rgbd_pilots/refined_20260908_135740/depth_heatmaps/review.md).
Native observation hashes are checked before/after export, and the companion
manifest records direct simulator provenance, display scales and output hashes.

### Robot-head verification and explicit prototype review (2026-09-08)

Current review index: [robot_head_prototype_v1](../../../data/sim_data/datasets/robot_head_prototype_v1/review.md).
This is a versioned REFERENCE INDEX, not a self-contained image archive or
approved training dataset. Original RGB/depth/masks/metadata remain immutable.

`dataset_review.py` performs an offline CPU-only audit; it does not change the
running Isaac stage or command the physical robot. It rebuilds the mounted head
camera in an in-memory robot USD, then checks each saved camera transform against
`robot base -> URDF torso/head FK -> fixed head-camera mount`. It rejects a free
Perspective camera, changed mount/optics, inconsistent renderer view/projection,
out-of-limit joints, non-rigid base transforms, resized or cropped observations.
The expected camera is:

```text
/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera
```

All nine latest samples passed. Camera translation residuals are below 1e-12 m
(numerical agreement between simulated transforms, NOT physical camera accuracy).
Sampling moves the whole simulated robot base and its real head joints, not an
independent cinematic camera. The poses are geometry-guided static snapshots;
they do not follow the current lab robot, and navigation between poses and arm
reachability remain unvalidated. D405 rendering remains ideal pinhole rather
than a real-camera noise/calibration model.

The audit also revalidates source hashes/branch recipes/cut geometry, recomputes
plant-to-world projection, native-depth validity, deepest-component mask ownership,
target masks, identity/depth visibility and quality gates. A saved native depth
pixel is the front surface, not the anatomical centreline; the two are retained
separately. At reviewed samples 0004/0005/0007 the surface-to-centreline differences
are about 1.98/2.08/2.41 mm respectively. These are diagnostics, not accuracy
measurements against an external physical ground truth.

Explicit assistant visual review:

- B05 0004/0005: recommend human prototype-label review; the petiole is
  distinguishable against lighter background. Width is still only 3.61/3.08 px.
  Both are the SAME plant/target, not independent geometry examples.
- B06 0007: hold. Its identity/depth checks pass, but the junction has weak RGB
  separation against adjacent main-stem, fruit and leaf geometry. Request another
  actual robot-head view or explicit ambiguity review; do not label it easy merely
  because a numeric threshold passed. This is not proof of nominal-pixel occlusion.
- The six failed-gate samples remain diagnostic holds, not accepted positives,
  labelled negatives or hard-difficulty ground truth.

Review cards show the original full-scene RGB above three labelled 4x nearest-pixel
crops (cut overlay, native mask, native depth). The crops are REVIEW-ONLY; they
never replace the 848x408 model inputs. No depth is estimated from RGB.

From `examples/greenhouse_sim`, audit into a NEW directory:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.dataset_review --run ..\..\data\sim_data\rgbd_pilots\refined_20260908_135740 --output ..\..\data\sim_data\dataset_reviews\MY_NEW_AUDIT
```

`dataset_package.py record` creates a new UUID review record. Reviewer roles are
explicit and locally self-declared, NOT authenticated identities. Assistant
`recommend/hold/reject` is separate from human `confirm/hold/reject`. Human confirm
means agreement with this image's prototype point/interval, NOT horticultural
certification, blade safety or training approval. This increment recorded ONLY
assistant reviews. Existing older anatomy reviews are not modified or promoted.

The human reviewer may run the following AFTER personally inspecting the sample
(replace YOUR_NAME and notes; use hold/reject if appropriate):

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.dataset_package record --audit ..\..\data\sim_data\dataset_reviews\review_20260908_v2\audit.json --sample sample_0004 --role human --reviewer "YOUR_NAME" --decision confirm --notes "Reviewed RGB, proposed interval, native mask and depth; prototype label matches." --inspected-rgb-mask-depth --output ..\..\data\sim_data\dataset_reviews\review_20260908_v2\records
```

Use `--supersedes PATH_TO_PRIOR_REVIEW.json` to revise a decision, keeping both
records. Conflicting decisions, missing history, cross-sample/role supersession,
stale source hashes and edited review cards fail closed. Holds remain holds even
if another role confirms; resolve them through explicit revision.

Build a NEW index version after additional review:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.dataset_package build --audit ..\..\data\sim_data\dataset_reviews\review_20260908_v2\audit.json --reviews ..\..\data\sim_data\dataset_reviews\review_20260908_v2\records --version robot-head-prototype.v2 --output ..\..\data\sim_data\datasets\robot_head_prototype_v2
```

Outputs: `samples.jsonl`, full review history, linked `review.md`, and a completion
`manifest.json` written last with artifact/source hashes. Input allowlist is only
clean RGB, native metric depth and validity. Calibration, base/joint snapshots,
provisional labels/masks, camera checks and review roles are separate fields.
There are only two source plant families; train/validation/test and difficulty
remain unassigned. Related source-family camera views/branch copies/appearance
variants must stay in the same split. Reference paths require preserving the
source capture and audit directories; do not send the index alone as a dataset.

Even human-confirmed prototype labels are NOT training-eligible in this v1
workflow: horticultural rule validation, coverage and the versioned accepted
annotation specification remain outstanding. No grasp/path supervision, physical
cut validation or dynamic synchronization is introduced here.

### Lightweight browser review GUI (2026-09-08)

From the repository root, run (or double-click):

```bat
examples\greenhouse_sim\run_dataset_review.cmd
```

It opens <http://127.0.0.1:8877> and starts with the two recommended samples.
Running the launcher again reopens the matching existing review server. No Kit
application, GPU, cloud service or physical robot connection is used. The server
requires Python with NumPy/Pillow; the launcher uses the existing Isaac Python
runtime when available, but it never creates a SimulationApp. Override the
runtime with `ISAAC_SIM_PYTHON` or launch `python -m sim_data.review_gui --open`
from `examples/greenhouse_sim`. `--audit`, `--records` and `--port` are optional.

Workflow:

1. Enter your name once (remembered locally by the browser).
2. Inspect the default all-evidence card: original robot-head RGB, proposed cut,
   exact native petiole mask and native depth. Separate tabs and a full-size
   image link are available. Magnified cards remain review-only.
3. Tick the explicit inspection checkbox and click Confirm label, Hold/uncertain
   or Reject label. Notes are optional; a decision-specific default is recorded
   only after the explicit click. This does not approve training or cut safety.
4. The decision saves immediately and advances to the next pending sample in
   the selected set. Use All samples to inspect the seven holds; Confirm is
   disabled on failed numerical gates. Previous/Next or left/right arrow keys
   navigate without saving. There is no bulk approve or automatic confirmation.

Reviews are stored next to the audit in `records/*.json`, through the existing
hash-bound human review API. Opening/reloading the GUI never records a decision.
Revisiting a sample creates explicit supersession history. Source/card changes,
conflicting reviews or stale browser tabs block writes. An assistant hold is not
silently cleared by a separate human confirmation. Existing versioned indices
remain immutable; build a NEW package version after reviewing, using the command
above. The GUI resumes from saved records rather than local browser state.

The server binds only to 127.0.0.1, serves allowlisted assets (no directory/file
browser), requires exact local Host/Origin plus a per-process CSRF token for
writes, uses no remote JS/fonts/CDNs, and never logs review request bodies.
Roles remain self-declared local review attribution, not authenticated identity.
Browser end-to-end Confirm/Hold/Reject, auto-advance, disabled failed-gate confirm,
revision and reload tests used synthetic temporary fixtures, not real annotations.

### Reviewed snapshot and focused robot-base/head recapture (2026-09-08)

The original reviews are now frozen into a NEW index:
[robot_head_prototype_v2](../../../data/sim_data/datasets/robot_head_prototype_v2/review.md).
It contains human confirmations for original B05 0004/0005 and the human B03
0001 hold. Six other samples remain unreviewed by the human and diagnostically
held. No training or physical-cut approval is inferred. Original v1 and all
capture observations remain unchanged.

To propose better actual robot-head views without increasing image resolution,
changing camera mounting, moving plants or tuning lighting:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.viewpoint_plan --drafts ..\..\data\sim_data\cut_region_drafts\20260907_235517_89790c\draft_labels.json --output ..\..\data\sim_data\viewpoint_plans\MY_NEW_PLAN
```

The CPU-only planner samples the whole robot base on the same +X aisle side,
with 15-40 cm approach, lateral offsets up to 40 cm and absolute yaw 150/180/210
degrees. Every pose preserves the arm/torso joint configuration and camera mount,
aligns the wheels to the existing floor and solves only the real head joints.
The base-X/target-X gap must stay at least 35 cm. This gap is merely a sampling
bound, not a collision certificate. Existing triangle-vs-robot-bound checks
still reject possible foliage/structure overlap; self-collision, navigation
between poses and dynamic contacts remain unvalidated.

Candidates need predicted width >=3 px and projected interval >=5 px, then are
ranked for useful projected geometry and base/yaw diversity. Visibility and
visual clarity remain UNKNOWN until native rendering and image review. The
current plan screened 165 poses, retaining six B03 and three B06 candidates:
`data/sim_data/viewpoint_plans/focus_20260908_v1/plan.json`.

Render only the selected, independently re-screened candidates:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.capture_pilot --drafts ..\..\data\sim_data\cut_region_drafts\20260907_235517_89790c\draft_labels.json --refine-views --view-plan ..\..\data\sim_data\viewpoint_plans\focus_20260908_v1\plan.json --max-samples 6 --output ..\..\data\sim_data\rgbd_pilots\MY_NEW_FOCUSED_RUN
```

The capture uses the same native RGB/depth/instance buffers, static-scene guards,
quality gates and human-approval separation as the original pilot. It preserves
all rendered-candidate decisions and selects up to three images per target.
Old confirmations DO NOT transfer to a new view; the audit hash identifies the
capture version, and the browser GUI now displays its capture-run name.

Plan loading before SimulationApp must not import USD or Omni. USD imports in
the CPU-planning function are lazy, verified in a fresh Python subprocess; this
avoids standalone USD/Kit binding conflicts. The initial failed focused run is
retained as `failed_do_not_train`, never used as dataset evidence.

Focused capture results are saved at
`data/sim_data/rgbd_pilots/focused_20260908_v2/`. Nine planned candidates were
rendered; six selected views (three B03, three B06) passed the independent
source/calibration/FK/projection/depth/instance-mask audit. Four passed numerical
clear-view gates, but this is not four visually accepted examples:

- B03 new samples 0001/0002/0003: assistant inspected the full-scene RGB, cut
  overlay, native mask and native depth and recommends human prototype review.
  Petiole widths are 3.97/3.65/3.68 px, with 6.97/9.08/6.30 px cut intervals.
  These are three views of ONE target, not independent target geometries.
- B06 new sample 0004: numerical visibility passes, but RGB separation near the
  main stem/fruit/leaf remains ambiguous; assistant hold.
- B06 new samples 0005/0006: the nominal cut is obscured by native components
  `Added03_Leaf_204` / `Added03_Leaf_205`, respectively. Camera-Z is 62.34/89.76 mm
  in front of the target centreline at that pixel. Keep as occlusion diagnostics,
  not clear cut-point examples or validated difficulty labels.

The new index is
[robot_head_focused_v2](../../../data/sim_data/datasets/robot_head_focused_v2/review.md):
three assistant recommendations, three holds, zero human confirmations at export
and zero training-eligible samples. No old-run approvals transfer to new images.
Focused v2 corrects a report-generator count left over from the original
nine-image pilot; focused v1 is retained unchanged. Both reference the same six
captures and assistant decisions. Counts now come from the exported rows/families.
Review it independently of the original page, from `examples/greenhouse_sim`:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.review_gui --audit ..\..\data\sim_data\dataset_reviews\focused_20260908_v1\audit.json --port 8878 --open
```

The focused browser is <http://127.0.0.1:8878>; the original remains on port 8877.
Keep the default Recommended filter for the three new B03 views; All samples
also exposes the B06 holds. Native 848x408 RGB is the observation; enlarged cards
and colourized native-Z are review-only. All captured views use the mounted
robot head camera, with unchanged source geometry, lighting and camera mount.

Capture process caveat: the focused retry finalized its six-sample manifest as
`pilot_ready_for_review` with unchanged sources, and the subsequent independent
audit passed all six. However, the process returned exit code 1 at Kit shutdown,
without the final post-close result print. The shutdown cause remains unresolved;
this is verified static image evidence, not a clean-exit or production-recorder
claim. The WriterSyncGate warnings and static-only synchronization scope remain.

Final selected regressions: 525 passed, one skipped, 47 subtests passed. The
next coverage step is independent source plants/targets after the focused review;
do not inflate dataset diversity with more views of B03/B05 alone.

### Remaining Phase 1 acceptance checklist

The confirmed pilot is now preserved in
[original v3](../../../data/sim_data/datasets/robot_head_prototype_v3/review.md)
and [focused v3](../../../data/sim_data/datasets/robot_head_focused_v3/review.md).
Five confirmed images are two target geometries; new source targets still need
their own image/anatomy review. The prototype rule is not horticultural approval.

1. Use the saved first-plant reviews as an initial pilot; no need to repeat them.
2. Resolve or explicitly exclude attachment warnings and ambiguous anatomy.
3. Agree on horticultural target eligibility, admissible cut/stub geometry,
   and proposed grasp rules. Do not infer these from an attachment coordinate.
4. Use mixed-gallery sampling across independent plants and explicit exception
   review. The earlier roughly-50 proposal was a planning estimate, not a mandatory
   quota. Coverage and resolution of systematic issues determine readiness.
5. Version the accepted annotation specification before Phase 2 capture work.

### Bounded native multi-plant pilot (2026-09-08)

From `examples/greenhouse_sim`, create a NEW immutable plan:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.collection_plan --plants 2 --targets 2 --views 2 --output ..\..\data\sim_data\collection_plans\MY_NEW_PLAN
```

The first plan is `data/sim_data/collection_plans/native_20260908_v1/plan.json`.
It selects seed11_full and seed17_full, native SubStem_41/42, without branch
augmentation. A 1.05-1.70 m world-height band prioritizes the initial robot-view
pilot; targets outside it are NOT labelled unreachable or ineligible. All
selection/exclusion reasons and source fingerprints are recorded. The agreed
10 mm nominal / 10-20 mm centreline rule remains provisional per target.

The reserved 16/4/4 train/validation/test assignment is by target source family.
Already reviewed seed101/103 cannot enter held-out groups. Collection currently
selects only new train families. The surrounding greenhouse and backdrop assets
remain shared; this is NOT scene-disjoint evaluation. Packages stay unapproved,
and target-level split reservations do not enable training or assign difficulty.

Run jobs serially, with the existing Isaac Python environment:

```bat
D:\isaac-sim-6.0.1\python.bat -u -m sim_data.collection_run --plan ..\..\data\sim_data\collection_plans\native_20260908_v1\plan.json --output ..\..\data\sim_data\collection_batches\MY_NEW_BATCH --max-jobs 2 --timeout 1200
```

Each `job_XXX` contains `launch.json`, raw `worker.log`, `capture/`, `result.json`
and, after successful process/capture validation, an independent `audit/` with
review cards. The batch manifest is finalized last. Logs retain failures;
timeouts terminate only the exact child started by this scheduler, never a live
user simulator. There is no automatic retry, output reuse, source edit or human
review action. A zero process exit alone is insufficient: complete observations,
unchanged sources and the independent audit must pass before the next job starts.

Existing review tools accept each successful native job's `audit/audit.json`.
First inspect its full-scene RGB, nominal/interval overlay, exact native petiole
mask and camera-Z, then write explicit assistant recommendations/holds. Human
confirmations remain a separate step. No isolated/recoloured images become RGB
inputs, and no projected-but-hidden cut is presented as visible or executable.

The completed first native batch is
[native_20260908_v2](../../../data/sim_data/collection_batches/native_20260908_v2/visual_review.md).
Both workers exited 0; eight samples passed independent audit. Assistant visual
inspection recommended five views and held three (seven numerical passes are
not seven visually accepted views). The user has now confirmed those five views.

Open both plants in a single lightweight page, from examples/greenhouse_sim:

```bat
D:\isaac-sim-6.0.1\python.bat -m sim_data.collection_review --batch ..\..\data\sim_data\collection_batches\native_20260908_v2 --port 8879 --open
```

Check the header native_20260908_v2. UI sample IDs have job_001/job_002 prefixes
to prevent cross-plant mistakes; records retain original sample IDs and audits.
No simulator, GPU or physical-robot connection is needed. Confirmations remain
prototype image-label decisions, not horticultural/training/physical approval.
