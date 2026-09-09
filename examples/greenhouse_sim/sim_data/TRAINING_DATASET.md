# Synthetic cut-point grounding dataset (task v3)

This is the first perception-only training dataset, not the full bimanual
deleafing-policy dataset. The original reviewed pilot remains immutable.

Latest checkpoint (2026-09-09): **4,134 v3 candidates / 5,787 audited raw**,
with 2,992 train / 368 validation / 774 test. Fresh v3 visual inspection covers
41 unique cards: 39 accepts and two source-enforced holds. There are still 79
pending cards in the broader 108-card set; its newly held card also needs a
replacement in a fresh release-QA selection. Collection is running, but volume,
balance, visual QA and portable complete-release validation remain unfinished.
No VLM fine-tuning has started. Current receipts:
`data/sim_data/status/v3_after_nominal_rgb_hold_20260909.json` and
`data/sim_data/status/visual_qa_followup_20260909_v3.json`.

Current visual-review UI: <http://127.0.0.1:8880>, launched from
`examples/greenhouse_sim/run_training_review.cmd`. See
[TRAINING_REVIEW_GUI.md](TRAINING_REVIEW_GUI.md) for the pending-card workflow,
saved decisions and source-level Hold/Reject enforcement. This is the task-v3
reviewer, not the historical prototype UI on ports 8877-8879.

Earlier status (2026-09-09 post-anatomy-audit checkpoint): **collection in progress, no
complete release; current task v3 needs fresh visual QA**. The one-by-one audit
of all 63 wave-2 cards found 33 correct visible nominal labels and 30 hidden-cut
abstentions (16 foreground leaves, 9 fruit, 5 main stems). Two query fragments
were visually inadequate; the earlier representative QA was too permissive.
The old white/magenta overlays also misleadingly displayed hidden geometry on
foreground surfaces. These were review overlays, not marks in training RGB.

V3 adds a conservative native-query usability screen and visibility-gated
review cards. Re-screening the same 63 sources retains 54 candidates (32
localizations, 22 abstentions) and excludes 9; 45 retained queries changed.
The reported `seed41_full_7ab04f3e6924202fd315` is now excluded because its query
neighborhood is too dark, even though the native identity is the petiole.
Both earlier visual holds are excluded too. All original data/decisions remain
unchanged. Old approvals cannot authorize changed v3 queries.

Evidence: `data/sim_data/dataset_audits/wave2_anatomy_20260909/audit_report.html`;
corrected one-by-one browser:
`data/sim_data/dataset_reviews/grounding_wave2_rescreen_20260909_v3/index.html`.
Only four retained v3 examples have fresh explicit assistant review records at
this checkpoint; the remainder are pending, not implicitly accepted.

The completed 29-audit recount yields **3,246 v3 candidates / 4,623 audited raw**:
2,102 train, 368 validation, 776 test. All 4,623 native optical-Z arrays match
their capture-time byte hashes. Fresh v3 stratified QA is pending; train volume
and class counts, validation volume and held-out medium coverage still fail
release gates. Receipt: `data/sim_data/status/post_anatomy_audit_v3_20260909.json`.
The previous campaign supervisors have exited. Two renderers are still running;
seed37/seed29 jobs finished writing but lack a final worker-exit ledger. These
unfinalized outputs are excluded. Durable supervision must be restored using
verifiable recovery or new-directory reruns; never invent clean exit records.

Follow-up (same day): detached three-lane supervision is now running; see
[COLLECTION_RECOVERY.md](COLLECTION_RECOVERY.md). One seed17 exit was observed
as actual code 0 and its 1,164 frames entered independent auditing. Other
unproven/unstarted jobs use fresh directories. A new explicit visual hold reduces
the verified baseline to **3,245 candidates** (2,102/368/775 by split), still
4,623 audited raw. Twenty wave-2 v3 cards have been inspected, with 19 accepts
and one hold enforced by the exporter. The broader 108-card / 24-family v3
bundle has 14 exact-evidence reused v3 reviews and 94 pending; full QA is not
complete. No v2 approvals, live frames or uncompleted recovered audits inflate
the current candidate count.

Recovery completed subsequently: the independently audited seed17 source adds
890 eligible candidates from 1,164 raw frames, bringing the current snapshot to
**4,135 v3 candidates / 5,787 audited raw** (2,992 train / 368 validation / 775
test). All recovered native depth byte hashes match and cross-snapshot RGB/view
deduplication found no duplicates. Receipt:
`data/sim_data/status/recovered_seed17_v3_20260909.json`. Six recovered easy/
medium/hard cards were actually inspected and their accepted bindings verified;
fresh v3 QA totals 26 unique decisions (25 accepts, one enforced hold), with the
broader 94 cards still pending. Capture supervision advanced to new seed103;
seed37/seed31 are saving new frames but are not yet audited additions. No complete
release or VLM fine-tuning is claimed; volume/balance and full QA gates still apply.

Further RGB audit held `seed61_full_cd2eabc75725cdcefcdc`: the native mask
identifies the petiole, but its nominal cut region blends into a tomato in the
untouched RGB. The lossless overlay crop and original unmarked RGB were both
inspected. This is a readability hold, not a claim that the mask identifies a
fruit or that native depth proves occlusion. Source-level exclusion is verified
even without the task review bundle. This subtracts one test/easy candidate
from the 4,135 checkpoint above; all source data and old decisions are preserved.
The v3 query usability screen does not independently guarantee nominal-cut RGB
readability. Broader nominal-region inspection remains necessary; geometric
clear visibility alone must not be presented as perceptual or physical approval.

### Historical v2 evidence (not current v3 approval)

The historical v2 engineering export is
`data/sim_data/training_releases/grounding_v2_engineering_20260909_v1` (317 rows,
215 train / 102 validation / no test). Its portable loader and the selected
41-example, nine-family visual QA passed; size, split and difficulty coverage
remain incomplete. Larger workers are collecting separately. The task contract
was v2; the portable release container schema remains v1. See `dev.md` for evidence.

The initial complete 24-family coverage recount found 834 eligible examples
from 1,103 audited raw frames (544 train / 141 validation / 149 test). Family,
target-diversity and cut-spread checks pass. All-family representative visual QA
has since passed: 104 actually inspected examples across two v2 review bundles.
Adding the first two completed scale jobs gives a verified 26-audit snapshot
of 1,721 eligible examples from 2,343 raw (1,232 train / 141 validation / 348 test),
with 116 source-bound visual inspections. Volume and medium-class coverage remain
incomplete; this is not a complete release. The 317-row engineering export above
is a separately validated subset, not an extra 317 unique training examples.

## Task

The observation is the unmodified 848x408 RGB image from the RB-Y1 A v1.2
mounted head D405 plus a textual query pixel identifying the target petiole.
The query comes from an exactly visible distal petiole centreline point at
least 45 mm from the attachment and at least 18 image pixels from the nominal
cut. It is an explicit target-selection input, not a model-predicted detector.
No crop, overlay, depth heatmap, instance mask, source component identifier,
cut coordinate or hidden geometry appears in the prompt image.

The model returns JSON with `status`, `cut_point_uv`, `visibility`, and
`next_action`. A localizable target gets the projected nominal 10 mm cut
point; an independently identified foreground occlusion gets a null cut point
and `change_viewpoint`. Unknown visibility and ambiguous junction context are
excluded, not turned into false negatives. No generated prose claims blade
clearance or a safe executable action.

Distances are along the source petiole centreline from its manifest attachment:
nominal 10 mm, evaluation interval 10-20 mm. This user-agreed synthetic labeling
convention is NOT proof of agronomic suitability or external stub length.
Image coordinates use continuous original pixels, top-left edge origin,
x right/y down. Pixel centres are `(i+0.5,j+0.5)`.

Native optical-axis depth in metres, validity, camera calibration, robot pose,
anatomical camera/world XYZ, and the accepted projected interval are sidecars.
Centreline XYZ is not substituted for front-surface camera depth. RGB alone
does not guarantee metric 3D inference; RGB-D modeling remains a separate task.

Depth must come directly from Isaac Sim Replicator's `distance_to_image_plane`
annotator on the same head-camera render product as RGB. The writer copies its
native float32 array unchanged into `inputs/depth_m.npy` (408 rows x 848 columns).
No custom ray casting, RGB depth estimator, centreline projection, interpolation
or hole filling supplies or replaces this measurement. Invalid values remain in
the raw array and are described by a separate validity mask. `depth_preview`
only maps those saved values to colours for inspection; heatmaps are never the
metric depth observation or a substitute for it. This is ideal simulator depth,
not a calibrated model of physical D405 noise.

## Collection and quality

- The 24 original target families retain the frozen seed-0 split: 16 train,
  4 validation, 4 test. The greenhouse/backdrop context is shared. This is NOT
  a scene-disjoint or real-world generalization benchmark.
- Static robot base/head snapshots use unchanged camera optics/mounts and
  stock arms/grippers. Camera framing is continuous and varied, not the
  pilot's three predictable cut-pixel locations. The snapshots are geometrically
  guided, not autonomous viewpoint selection or collision-free navigation.
  Optional `--vary-torso` uses real URDF torso joints (not a raised root or moved
  camera bracket) to cover different plant heights. Exact FK/limits, floor
  alignment and scene geometry screens still apply; self collision/path motion
  are not certified.
- All masks/depth originate from the native synchronized Replicator writer.
  Static scene/content/FK guards apply. Dynamic sensor synchronization is NOT
  validated. No physics, teleop or lab-robot motion is introduced.
- Per-image integrity, projection, visible identity, native depth and camera
  FK checks precede training-label derivation. Failed worker exits are excluded.
- V3 query gates use the original RGB and exact eight-connected native target
  mask: at least 100 island pixels, 64 target pixels and 20 px extent inside
  a 33x33 neighborhood, 1.5 px interior radius, and 16 px frame margin. No more
  than half the local target pixels may have luminance below 20/255. Median
  target/background luminance contrast must be at least 6/255 with at least
  16 background-ring pixels. The three-pixel ring measures background only;
  no target-mask gap filling or depth modification is performed. These are
  conservative engineering screens, not calibrated human/VLM readability
  guarantees; they may reject valid but dark or low-contrast petioles. Threshold
  changes require a new contract and re-review, not quota-driven relaxation.
- The portable loader independently recomputes query usability from copied RGB
  and the exact native target mask. If no usable query candidate exists, the
  source is excluded rather than relabeled as a useful hard example.
- Review cards suppress cut marks for hidden/unknown nominal visibility and
  show a separate query RGB/mask/native-Z crop. Magenta interval segments require
  visible native probe evidence and are clipped to native target pixels.
- Positive labels additionally require resolved proximal context and visible
  parent/target separation. The first native regression matches all five
  human-confirmed examples and excludes all three held examples. This is a
  small calibration check, not a statistical annotation-accuracy guarantee.
- Difficulty bins are explicitly operational: easy = nearly fully visible
  sampled proximal centreline/interval; medium = sufficient but partial evidence;
  hard = a proved foreground occlusion requiring abstention. They do not measure
  amodal leaf surface visibility or physical task difficulty.
- Synthetic label provenance is recorded as automatic. Human confirmation is
  never invented. Representative stratified inspection remains part of QA.

## Release acceptance

`training_export.RELEASE_GATES` requires at least:

- 10,000 unique training RGB images; 500 validation; 500 test.
- 16/4/4 target-source families and 100/20/20 distinct target petioles.
- 20 easy, 20 medium, 20 hard examples in EACH split.
- 100 occupied 32-pixel cut-location bins in training, with query-copy baseline
  median error at least 10 pixels. More complete learned-shortcut baselines and
  model learning curves should accompany research claims.
- Exact duplicate exclusion, family/image split checks, complete file hashes,
  one-to-one image/prompt/answer consistency, and an offline portable loader pass.
- At least 5,000 localizable train examples and 250 each in validation/test;
  at least 1,000 occluded train examples and 50 each in validation/test. A large
  mostly-abstention dataset is insufficient. Repeated camera/scene conditions
  are deduplicated even if stochastic renderer noise changes their RGB hashes.
- Explicit assistant/human hold or reject records veto automatic inclusion.

These gates define a first substantive synthetic release, not universal sample
requirements or a guarantee of model quality. Further plant geometry diversity,
real evaluation and learning-curve-driven scaling remain necessary.

An exporter may create an explicitly named incomplete engineering package with
`--allow-incomplete`. Its manifest state is
`incomplete_engineering_export_do_not_claim_release`; it MUST NOT be described
as satisfying the release gates. Old pilot approval flags are never changed.

## Commands

The active scale campaign is
`data/sim_data/collection_campaigns/grounding_scale_20260909_v1/campaign.json`.
It schedules 22 remaining families in four serial lanes; the two existing
training jobs occupy reserved slots until their successful audits complete.
Supervisors were launched for all four lanes. **Do not launch these lanes again**:
their exclusive output directories intentionally reject duplicate starts.

For a *new* campaign, from `examples/greenhouse_sim`, use the same existing
Isaac Python interpreter for these commands:

```text
python -m sim_data.collection_campaign create --train-plan TRAIN_PLAN --heldout-plan HELDOUT_PLAN --after-batch EXISTING_TRAIN_BATCH_1 --after-batch EXISTING_TRAIN_BATCH_2 --output NEW_CAMPAIGN
python -m sim_data.collection_campaign run --campaign NEW_CAMPAIGN/campaign.json --lane lane_01
```

Launch each of `lane_01` through `lane_04` exactly once in separate processes.
The first two wait for the named predecessor audits; the other two can start
immediately. Do not run unrelated capture workers concurrently. Source/split
checks, a per-job disk reserve, 4-hour worker timeouts and stop-on-error remain
mandatory. There is no automatic retry, resume-overwrite, deletion or visual
approval. Lane `request.json` is initial-state provenance, not live status;
per-job capture/result files show progress, and lane `result.json` is written
on completion or controlled failure. After a failure, inspect the preserved
evidence and create a new schedule for genuinely missing views.

Complete releases additionally require explicit visual inspection records:
two examples per nonempty family/difficulty stratum (or every example if only
one exists), plus coverage of each capture profile. These are representative
checks, not a statistical accuracy estimate or human confirmation. Prepare cards
with `training_release_review prepare --audit ... --output NEW_REVIEW`, actually
inspect them, and record decisions with `training_release_review record --bundle
NEW_REVIEW/bundle.json --id ID --notes "observed evidence" --inspected`. Assistant
is the default reviewer role; it cannot impersonate a human confirmation.
Pass every relevant bundle using `training_export build --visual-review
NEW_REVIEW/bundle.json ...`. Holds, rejects, changed evidence, uninspected strata
and duplicate frames cannot pass the complete-release loader.

Run from `examples/greenhouse_sim` using Isaac's Python for collection/audit:

```powershell
& 'D:\isaac-sim-6.0.1\python.bat' -m sim_data.training_plan --targets 12 --views 64 --output NEW_PLAN
& 'D:\isaac-sim-6.0.1\python.bat' -u -m sim_data.collection_run --plan NEW_PLAN/plan.json --output NEW_BATCH --max-jobs 24 --timeout 14400
& 'D:\isaac-sim-6.0.1\python.bat' -m sim_data.training_export build --audit JOB_A/audit/audit.json --audit JOB_B/audit/audit.json --output NEW_RELEASE
& 'D:\isaac-sim-6.0.1\python.bat' -m sim_data.training_export validate --output NEW_RELEASE
& 'D:\isaac-sim-6.0.1\python.bat' -m sim_data.training_evaluate --dataset NEW_RELEASE --output NEW_BASELINE_REPORT.json
```

`--job job_003` selects an explicit scheduled job (repeatable). Every capture,
audit and release writes a new directory; no implicit retry or overwrite.

Use `training_plan --view-offset 8 --views 8 --vary-torso ...` for the next
nonoverlapping proposal window after an eight-view plan. Offset zero preserves
the original schedule. Offset windows retain exact global deterministic poses;
counts refer to maximum accepted geometry-screened views, not label quotas.

`collection_run --instance-backend fast` bypasses the legacy per-subframe JSON
conversion node. The same-callback `compare` diagnostic verified exact native
ID pixels and all observed prim paths on the calibration/occluder scene and two
full-greenhouse frames; both that job and fast-only capture passed independent
audit with zero worker exits. This does not change RGB, depth or sensor optics.
Total wall-clock improvement has not yet been demonstrated under concurrent load.

The first completed larger coverage job contains 63 audited observations; its
task contract accepts 29 easy, 2 medium and 11 hard labels. Twenty-one uncertain
examples are excluded. Other plant families are collecting; this is not yet a
complete release. Representative assistant visual-inspection notes are under
`data/sim_data/dataset_reviews/grounding_live_20260909_v1/visual_review.md`.

## Portable layout and use

The active task contract is now `greenhouse.target_conditioned_cutpoint_rgb.v3`.
Visual QA found that v1 could choose a visible distal fragment disconnected
from an otherwise visible cut by foreground foliage. V2 requires localized
queries to share the exact eight-connected native visible petiole region with
the nominal cut; it does not fill gaps. Original minimum query arc/separation
and all cut/visibility/ambiguity gates remain. If no such query exists, exclude
the image from localization training. Hard examples still abstain on hidden
cuts without claiming a visible path. Raw observations are unchanged, but v1
label counts and changed-query approvals cannot be presented as v2 results.
Review records bind the exact task-contract hash. V3 additionally enforces the
query usability policy above and uses visual-QA schema v2 with explicit query
neighborhood inspection. Historical v1/v2 engineering exports remain immutable
diagnostics, not the current release; the current loader intentionally rejects
their old task contracts. Re-screen a prior bundle without migrating approvals:

```powershell
# From examples/greenhouse_sim; choose a NEW output path.
& 'D:\isaac-sim-6.0.1\python.bat' -B -m sim_data.training_rescreen `
  --previous-bundle D:/research/tomato-pi-policy/data/sim_data/dataset_reviews/grounding_release_wave2_20260909_v2/bundle.json `
  --output D:/research/tomato-pi-policy/data/sim_data/dataset_reviews/YOUR_NEW_RESCREEN
```

This validates source hashes, writes a per-image comparison and local browser,
and creates a new pending review bundle. Excluded rows have diagnostic cards but
no eligible bundle entry; untouched sources remain available for later study.

The legacy `vlm_eval` exploratory prompt is a separate autonomous-selection
experiment with a historical 2-5 mm visual heuristic. Do not substitute it for
the export's target-conditioned prompt and 10 mm nominal / 10-20 mm arc rule.

`images/` contains only clean RGB. `splits/train.jsonl`, `validation.jsonl`,
and `test.jsonl` use image paths relative to the release root plus system/user/
assistant chat messages. `depth/` and `labels/` are separately stored sidecars.
`index.jsonl` binds provenance, split, target, query and answer; `contract.json`
fixes the task and prompt; `exclusions.json` records dropped candidates.
The portable loader does not need original absolute source paths. Historical
source hashes in `manifest.json` are provenance, not runtime dependencies.

## Renderer qualification (in progress)

The source greenhouse carries a RealTimePathTracing override; actual runtime
settings must be recorded instead of trusting the launcher mode string. The
isolated Python experience disables preference persistence, and workers now
also request that explicitly. An optional legacy-mode diagnostic restores the
RT2 switch before exit; on this installation the legacy-mode request is still
overridden, so that diagnostic fails before saving any images.

The original randomized run saved seven audited frames with a clean worker exit
in 409.28 seconds. A full initial static scan plus USD change-notification guard
and the existing long warmup saved the same seven poses with a clean exit in
313.94 seconds. The dominant per-image cost was 27-28 seconds in six warmups and
4-5 seconds in the final render, not the now-sub-millisecond static guard.

Short rendering must pass native geometry/identity checks and a long-reference
qualification. Early single-subframe attempts failed RGB convergence/reference
comparison and produced no training frames. Never describe those attempts as
successful optimization. Multi-subframe comparisons also failed their long-reference
gates. The default therefore remains the established fixed 56-subframe budget.
`--render-reference-check` is opt-in diagnostic work, not a qualified faster profile.

A separate, explicit `--render-budget warm56_then8 --instance-backend fast`
profile has passed bounded task-level qualification at 21 matched poses
across two source families: native target masks and depth agree. Recomputed
under the current v2 contract, all 17 jointly eligible task answers agree
(14 easy / 1 medium / 2 hard), and four examples are excluded by both budgets.
The earlier 18/3 result used v1 and remains historical evidence. Query pixels
differ in 15 matched pairs, so this is not an identical-query controlled VLM
comparison. Inspected saved RGB/mask/depth evidence is under
`grounding_short_profile_20260909_v1` and `grounding_short_seed101_20260909_v1` in
`data/sim_data/dataset_reviews`. Median after-first-frame capture was about
4.8-4.9 seconds versus 32-34 seconds for the long budget in these measurements.
RGB shading differs; this does not retroactively pass photometric-equivalence
experiments or certify all future frames. Per-sample gates and stratified release
QA remain mandatory. The selected profile is recorded in every observation.

The fast native adapter retains the complete pixel buffer and every observed
ID-to-prim mapping while omitting unused renderer mapping entries. It retains
the full anatomical component catalogue. An independently audited 53-frame job
verified this path; an example raw sample occupies 6.39 MB rather than about
11.8 MB with the full unused renderer table.

`training_progress --plan PLAN/plan.json --batch BATCH` reports live saved files,
provisional easy/medium/hard labels, exclusions and independently audited counts
separately. It never grants release approval. `collection_run --profile-first-render`
records a CPU profile without changing the renderer or reducing its render budget.

The conservative geometry broad phase also has a cached equivalent; 36 random
scene/margin combinations matched every original minimum/overlap/refinement
result exactly. No collision-screen criterion was relaxed.

NVIDIA references: [renderer modes](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/rendering_modes.html),
[Kit persistence](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/configuring.html).

`training_preplan` is an optional synchronous-USD diagnostic for geometry-only
view proposals, not an alternate image generator. One seed17 coverage plan
matched every field of a completed Kit-worker plan at 1e-9 absolute tolerance,
with unchanged source/scene guards (118.67 s total). Its receipt binds the plan,
sources and proposal file. It is not integrated into production workers; native
pose checks and RGB/depth/identity capture remain mandatory. A successful
proposal comparison never grants annotation or release approval.

A trainer must load RGB from `images`, preserve the original pixel-coordinate
convention through any processor resizing, mask system/user tokens from the
supervised loss, and never load labels/masks/depth as RGB observations. A model-
specific tokenizer/processor adapter and actual fine-tuning run are separate
from packaging. Inspect learning curves and per-family/per-difficulty metrics
before claiming generalization.

Cosmos real-hard augmentation is not included. Future appearance variants must
inherit the source split and pass geometric alignment QA. Bimanual grasp/cut/
release trajectories require independently validated dynamic demonstrations.
