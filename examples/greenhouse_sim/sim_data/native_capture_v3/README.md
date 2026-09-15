# Reviewed-reference native collection support

Opt-in static data preparation on `koh-dev/sim-data`. No training, physical
robot commands, source-asset edits, split changes or release approval.

## Components

- `reference_bank.py`: reconstruct a hash-bound bank of reviewed TRAIN original
  head-camera references from a pinned clear-data checkpoint. Keep the exact
  original collection plan, camera/robot pose, source donor and target. Unknown
  native trace evidence remains unknown. `check(bank)` rebuilds and verifies it.
- `reference_schedule.py`: choose one compatible scene/plan group per donor,
  prioritizing reviewed-target coverage and recorded quality. Schedule fresh
  exact-anchor native camera pairs, with bounded same-group fallback anchors.
  Visit every donor's primary before fallback rounds. No existing proof is reused.
- `prepare.py`: after exact native sensor verification, construct a new curved
  plant with up to three reviewed target references by default. Keep the qualified
  anchor first; choose other targets only from the same original plan/group.
  Generate six proposed actual robot-head views per selected target using the
  existing guarded multi-target planner. These are snapshots, not robot paths.
- `preselect.py`: cheap projected interval/diameter/framing check for prospective
  camera poses. It does not compute native depth or establish visibility. It is
  not yet enabled in the native collector.

Destination guards protect original packages, source-bound directories, plans,
captures, checkpoints and sensor proofs. `verify_prepared(...)` is the launcher
handoff: require a completed receipt, unchanged code and source hashes, exact
scheduled anchor/target bases, matching native proof, and target/view bounds.
A leftover `plan.json` from a failed preparation is not sufficient.

All candidates retain their original donor-target view-cap group. Generated
names, alternate source plans and repeated cameras cannot reset release budgets.
The new recipe does not waive global morphology/image duplicate qualification.

## Current evidence (2026-09-15)

`data/sim_data/diagnostics/reference_bank_20260915_v2.json` contains 312 historical
848x408 references, 52 targets, 15 TRAIN donors, four original plans and 26
compatible groups. 17,089 source bindings were checked. All strict-native-trace
values are unknown: these are historical pose priors, not 312 new native images.

`diagnostics/reference_schedule_20260915_v1.json` selects 15 groups/51 targets,
15 primary and 25 conditional fallback camera qualifications. Each pair includes
848x408 and 1696x816 sensor captures plus separate known-surface controls. These
diagnostics contribute ZERO training-image diversity. The schedule alone does
not mean native qualification has run.

Projection preselection replay on 416 previously captured TRAIN frames rejected
16 non-strict candidates and zero of 99 strict candidates. The filter took
0.244 seconds; rejected frames had cost 91.38 seconds. This is a retrospective
upper bound on savings, not a measured collector speedup or a universal guarantee.
Receipt: `diagnostics/native_projection_preselect_20260915_v1/report.json`.

The full sim-data regression checkpoint v49 passed 1,536 tests +181 subtests
(128.88 seconds), including the strengthened preparation/plan handoff. The
separately in-progress inventory tests were excluded. Tests exercise fixtures and provenance gates; they
do not establish new native image quality or manipulation performance.

## Use

Run CPU preparation in a separate Python process, with `PYTHONPATH` containing
`examples` and `examples/greenhouse_sim`. Do not import/replay USD geometry before
the native worker initializes `SimulationApp`.

```powershell
python -m sim_data.native_capture_v3.reference_bank --help
python -m sim_data.native_capture_v3.reference_schedule --help
python -m sim_data.native_capture_v3.prepare --help
```

The local serial continuation recipe is
`data/sim_data/diagnostics/run_reference_bank_campaign_20260915_v1.py` (ignored
runtime artifact, not a portable command supplied by Git). It waits for the
current 18-plant campaign, requires no other Kit process, 20 GiB Windows commit
headroom and 60 GiB disk reserve, then qualifies each selected donor and collects
two bounded generated-plant jobs. Failed attempts and labels remain on disk.
It uses the existing RAW writer and explicit `warm56_then8_trial` render profile,
not the compact writer or one/two-subframe synchronization-failing experiments.
Check live progress/result receipts to distinguish queued from executed work.

The high-resolution release, held-out capture, calibrated global duplicate
admission and Qwen export remain separate pending gates. No 20k ZIP is ready.
