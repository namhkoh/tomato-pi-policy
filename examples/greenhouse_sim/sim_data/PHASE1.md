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

1. Use the saved first-plant reviews as an initial pilot; no need to repeat them.
2. Resolve or explicitly exclude attachment warnings and ambiguous anatomy.
3. Agree on horticultural target eligibility, admissible cut/stub geometry,
   and proposed grasp rules. Do not infer these from an attachment coordinate.
4. Use mixed-gallery sampling across independent plants and explicit exception
   review. The earlier roughly-50 proposal was a planning estimate, not a mandatory
   quota. Coverage and resolution of systematic issues determine readiness.
5. Version the accepted annotation specification before Phase 2 capture work.
