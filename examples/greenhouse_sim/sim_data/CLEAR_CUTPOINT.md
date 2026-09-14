# Clear cut-point localization: first focused experiment

Branch: `koh-dev/sim-data`. Profile: `clear_cutpoint_v1`.
This is a NEW experiment, not a replacement or relaxation of the old balanced
or visible/occluded release. It trains **target-conditioned 2D perception**,
not autonomous target selection, metric XYZ, occlusion reasoning or robot action.

Higher-resolution native capture is a separate qualification path; this existing
848x408 release contract has not changed. See [NATIVE_RESOLUTION.md](NATIVE_RESOLUTION.md).

## Fixed task and candidate rules

- Source for the new dataset: independently audited fresh native captures,
  materialized as a validated engineering pool with `--audited-source`. The
  older approved-release input remains supported only for reproduction. Family splits,
  RGB, native Isaac optical-Z depth, masks and anatomical labels remain unchanged.
- Only easy/clear localizations; at least 95% proximal visibility, an unbroken
  visible query-to-cut association, every sampled interval point on the native
  target mask, projected interval at least12px, median mask-radius width proxy
  at least8px, local median RGB luma at least40/255, at most10% local target
  pixels below30/255. These are legibility proxies, NOT human anatomy approval.
- Target query identifies a distal petiole, not its cut location. Nominal label
  stays10mm along that petiole; evaluation interval stays10-20mm, not a sphere.
- Max12 diverse views per target. Deterministic farthest-view sampling uses
  camera position/orientation and INPUT query, never cut coordinates. Seed41.
- Original 848x408 head RGB always remains input. Optional second image is a
 384x384 query-centred window, shifted inside frame boundaries, magnified to
 768x768 with bicubic interpolation. It adds no sensor information. Pixel
  bounds are in the prompt; outputs always refer to the FULL original frame.
- Schema remains four answer fields (`status`, `cut_point_uv`, `visibility`,
  `next_action`). All clear training answers localize; abstention competence
  must NOT be claimed. The model still may abstain; that counts as failure here.
- Canonical labels are original pixels. Use the existing Qwen normalized
 0..1000 adapter, two decimal places, for both query and answer.

Coverage gates are500/100/100 images and60/15/15 targets for train/validation/test,
with all16/4/4 reserved plant families. Thresholds were chosen before the clear
scan; insufficient coverage calls for better observations, not silent relaxation.
The shared greenhouse background is not scene-disjoint or evidence of real transfer.
RGB alone does not uniquely establish metric scale in arbitrary scenes. The
10mm rule supplies synthetic 2D supervision here; pixel accuracy is not proof of
a metrically correct or safe knife pose. Execution still requires depth/geometry,
fresh visibility checks and the separate tool/controller validation.

## Local build and review (from `examples/greenhouse_sim`)

**2026-09-15 decision:** require newly captured clearer images, not a repackaging
of old easy views. The complete old-source scan retained47 candidates only
(28/15/4 train/validation/test) and fails coverage. Keep that directory as a
diagnostic draft. The commands below for the old-source derivative are historical
reproduction, NOT the requested final training dataset.

New collection uses `training_plan --clear-capture --vary-torso --views 2` for
initial qualification. It proposes real base/torso/head snapshots0.30-0.55m
in X from the plant, retains native overlap screens, checks predicted8px width /
12px interval and uses uniform diffuse dome6000/sun1500 lighting. Optics,
848x408 resolution, source geometry and native depth remain unchanged. Do not
force a renderer the greenhouse overrides: pilot_v1 failed before images on
that check; pilot_v2 preserves the scene-selected renderer.

After successful native jobs and their independent audits, use the existing
`training_export build --allow-incomplete` to materialize an **audited source
pool**, not a trainable release. Feed that new pool to `clear_cutpoint_release
build --audited-source`. This mode verifies all source artifacts and bound plans
and refuses old/non-clear captures. The NEW clear dataset's own coverage and
review gates still apply. The old balanced/mixed release validator is unchanged.

```powershell
$env:PYTHONPATH='.'
python -m sim_data.clear_cutpoint_release build --source ../../data/sim_data/training_exports/visible_occluded_20260911_v1 --output ../../data/sim_data/training_exports/clear_cutpoint_20260914_v1
python -m sim_data.clear_cutpoint_release validate --output ../../data/sim_data/training_exports/clear_cutpoint_20260914_v1 --allow-draft
python -m sim_data.clear_cutpoint_review --dataset ../../data/sim_data/training_exports/clear_cutpoint_20260914_v1 --output ../../data/sim_data/dataset_reviews/clear_cutpoint_20260914_v1
```

Choose NEW output paths on repeat. Open review `index.html` locally. Inspect RGB
without marks first; toggle query/cut/interval marks for annotation checking.
Review two training views per target and ALL selected validation/test images.
By default, training review may be attributed to an assistant and held-out
acceptance requires human review. The explicitly declared overnight assistant
experiment mode is described below. No default accept. Enter reviewer/reason
and download decisions JSON.
Held-out QA is label/legibility review, not evaluation of model predictions.

Finalization writes a new copy and refuses incomplete coverage/reviews:

```powershell
python -m sim_data.clear_cutpoint_release finalize --source ../../data/sim_data/training_exports/clear_cutpoint_20260914_v1 --reviews C:/path/to/clear_cutpoint_reviews.json --output ../../data/sim_data/training_exports/clear_cutpoint_final_v1
python -m sim_data.clear_cutpoint_transfer --dataset ../../data/sim_data/training_exports/clear_cutpoint_final_v1 --output ../../data/sim_data/training_archives/clear_cutpoint_final_v1.zip
```

An existing hold/reject cannot be overwritten by an accept during finalization.
Holds or failed coverage require a separately audited selection/recollection;
the finalizer is not an exclusion or gate-bypass tool. Drafts cannot train or
be published by `release_archive`. No final ZIP is implied until it succeeds.

## Overnight fresh collection and review attribution

The2026-09-15 pilot used the native `robot_head_close_diffuse_v1` preset.
Pilot-v3 seed103:15 independently audited captures,8 strict clear candidates.
These are NEW RGB/depth frames, not the47 survivors of the old release scan.
The24-family serial campaign is
`data/sim_data/collection_campaigns/clear_capture_20260915_overnight_v1`.
It preserves failed searches, stops on native errors, and never auto-approves.

`training_plan --opposite-aisle` is a separately selected static capture option
for a later pilot, not part of the running original-side plan. It mirrors the
base to the negative-X aisle, facing within30 degrees of0, with the same
0.30-0.55m lateral separation, camera, torso range and geometry checks. It does
not validate a trajectory across the gutter or bypass overlap checks. Native
pilot evidence and image review are required before scaling this option.

`clear_collection_intake --campaign <campaign> --output <new-directory>` may run
alongside that one native worker. It reads only completed bound audit receipts,
then builds per-job `source/`, `draft/` and `review/index.html`. After successful
campaign completion it builds `aggregate/` in the same form. These are all
pending-review engineering artifacts, never automatic training releases. It
does not change capture jobs or their source files. Failed campaigns are not
silently aggregated as completed ones. Visual decisions stay in separate JSON
files and must match selected row IDs and exact RGB hashes before finalization.

For a user-requested assistant-reviewed synthetic experiment, build a NEW draft
with `--audited-source --assistant-reviewed-experiment` from a separately
validated fresh-capture engineering pool. This records
`review_policy=assistant_reviewed_experiment_v1` and
`independent_human_validation_claimed=false`. Review all held-out images and at
least two selected training images per target explicitly, with exact RGB hash,
named reviewer, type `assistant`, decision and image-specific reason. Missing
reviews or any hold/reject block finalization. Assistant acceptance is not human
annotation, physical safety approval or proof of anatomy against real plants.
The default remains `human_holdout_v1`; do not silently edit existing manifests
to change review policy. A human can later review the same immutable release.

Native-depth and geometry audits cover every exported row independently of this
visual review sampling. Additional visual inspection is encouraged. Minimum
coverage remains500/100/100 images,60/15/15 targets and16/4/4 plant families.
These are initial experiment gates, not a guarantee of generalization. If fresh
data cannot meet them, report the shortfall instead of packaging a draft as a
training release. Keep test predictions sealed until the model is locked.

## H200 experiment ? user-launched, no local weight downloads

Use the existing known-working Qwen3-VL-8B-Instruct BF16 snapshot and environment
from `H200_RESULTS.md`. Full FT was previously measured on4H200; this new crop
configuration is NOT measured. Keep torch2.6.0/transformers4.57.6 and the working
DeepSpeed0.17.6; see `H200_RUNBOOK.md` for environment setup. The portable validator
uses NumPy/Pillow/SciPy. Record versions; do not ignore crop/measurement mismatch.
Code must first be committed and pushed (or transferred separately); data is not
in git. Do not reset a dirty server checkout or replace ongoing jobs.

`clear_cutpoint_transfer` can include the matching lightweight code, this runbook,
native data and hashes in one ZIP. A diagnostic transfer requires explicit
`--inspection-only` and a `.inspection.zip` filename; its `TRANSFER_STATUS.json`
says training_ready=false and the trainer still refuses it. The final requested
training ZIP must use the completed fresh-capture release without that flag.
Its root `DATASET_CARD.md` is generated from the actual release counts and review
status, so it cannot silently describe the older mixed-task dataset.

The ZIP includes matching lightweight code; a server git update is optional if
you use that bundled snapshot. Choose a NEW extraction directory. Replace the
example archive basename below with the delivered filename, and verify its
adjacent checksum before extraction. Do not extract over an existing release.

```bash
cd /workspace/nhkoh/tomato-vlm
sha256sum -c clear_cutpoint_final_v1.zip.sha256 && \
  test ! -e clear-v1 && mkdir clear-v1 && \
  unzip -q clear_cutpoint_final_v1.zip -d clear-v1 && \
  cd clear-v1/code/examples/greenhouse_sim
```

Stop if the checksum or new-directory check fails. Read the delivered root
`DATASET_CARD.md` for actual counts, scope and review provenance. Use the
Qwen/H200 environment described above; the portable validation/data adapter
does not require an Isaac installation. A regression test validates an extracted
fixture and prepares RGB-plus-crop inputs in an isolated Python interpreter
using only bundled code, without importing Isaac/Omniverse/USD. That is code
portability evidence, not an H200 model or CUDA training test.

Then configure the extracted release and existing local model snapshot:

```bash
export VLM_DATA=/workspace/nhkoh/tomato-vlm/clear-v1/grounding_release
export VLM_MODEL=/path/to/local/Qwen3-VL-8B-Instruct
export VLM_RUNS=/workspace/nhkoh/tomato-vlm/runs/clear-v1
mkdir -p "$VLM_RUNS"
python -m sim_data.clear_cutpoint_release validate --output "$VLM_DATA"
python -m sim_data.clear_cutpoint_evaluate --dataset "$VLM_DATA" --output "$VLM_RUNS/baselines-validation.json"
```

Ensure `$VLM_RUNS` exists. Default evaluation is validation ONLY. Never run the
old `training_evaluate.baselines` here: that legacy helper also evaluates test.

Run each input variant from the SAME untuned base; same rows, seed41, 3 epochs,
optimizer and effective batch32. For full-frame omit `--query-crop`; for the
paired crop arm include it. Use distinct output directories throughout.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m sim_data.h200_train \
 --dataset "$VLM_DATA" --model "$VLM_MODEL" --output "$VLM_RUNS/crop-smoke" \
 --mode smoke --method full --deepspeed sim_data/clear_zero2.json \
 --learning-rate 1e-5 --vision-learning-rate 1e-6 --coordinate-decimals 2 --query-crop
```

After smoke passes, repeat with NEW output `crop-overfit`, `--mode overfit`.
This uses32 deterministic TRAIN-only rows and100 optimizer steps. Reload its
saved `model/` and generate on exactly those training IDs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sim_data.h200_evaluate generate \
 --dataset "$VLM_DATA" --model "$VLM_MODEL" --tuned-model "$VLM_RUNS/crop-overfit/model" \
 --output "$VLM_RUNS/crop-overfit-check" --split train \
 --ids-file "$VLM_RUNS/crop-overfit/run_contract.json"
```

Inspect exact-point/interval overlays before proceeding; low token loss is not
localization proof. Overfit-subset scoring is explicitly PARTIAL, not held-out
performance. Then launch a NEW run with `--mode train --epochs 3
--gradient-accumulation 8`, keeping the same full-FT settings. Do not run two
4-GPU jobs at once. No expected accuracy or duration is promised.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sim_data.h200_evaluate generate \
 --dataset "$VLM_DATA" --model "$VLM_MODEL" --tuned-model "$VLM_RUNS/crop-train/model" \
 --output "$VLM_RUNS/crop-validation"
python -m sim_data.h200_evaluate score --dataset "$VLM_DATA" \
 --run "$VLM_RUNS/crop-validation" --output "$VLM_RUNS/crop-validation-report.json"
```

For the untuned baseline omit `--tuned-model`, pass `--coordinate-decimals 2`
and the matching input-mode flag. Do not interpret invalid JSON alone as
proof that the base model cannot see the object. Compare a valid output contract.

## Success and limitations

Primary report: original projected interval hit within2px, plus stricter joint
interval-and-native-target-mask hit. Report macro averages by target and plant,
5px success over ALL images, conditional median/p90 error, invalid/abstain/missing
rates, input token counts and latency. Missing/invalid/abstaining answers fail.
Off-target pixels are measurable; fruit versus main-stem attribution is not
available from the portable target-only mask. Never invent that classification.

First prove improvement over query-copy, train-median offset and untuned Qwen.
Pre-register a deployment gate separately; perception success is not cut safety.
Use validation for model selection. Lock the prompt, crop, thresholds and weights
before ONE final test evaluation, which requires explicit `--split test --allow-test`.
Compare crop versus full frame on identical cases; report extra image tokens/
compute so a crop improvement is not portrayed as a free accuracy gain.

## Alternate static viewpoints (bounded native qualification)

Local collection controllers MUST run with Isaac's own Python bootstrap, e.g.
`D:/isaac-sim-6.0.1/python.bat -m sim_data.clear_collection_campaign ...`, not
conda/system Python: isolated workers inherit `sys.executable`. Plan generation,
non-approving intake and portable H200 data validation do not start Isaac and
may use the ordinary data environment. A failed launch must retain its directory
and exit evidence; use a new output for a corrected launch, never overwrite it.

The close original-side campaign can reject an entire family when the head is
too far away for legibility or the torso envelope intersects foliage. An empty
family is not a reason to disable the geometry or visibility checks.

Two opt-in proposal modes are prepared for small native pilots before scaling:
`--opposite-aisle` mirrors the root to the other side and faces it toward the row;
`--oblique-clear` samples root positions on a .30-.55 m horizontal-distance ring
with bearings within +/-70 degrees of the selected side. Both require the clear
preset, preserve the actual camera mount/optics, joint ranges, full source scene,
floor and geometry admission screens, and retain native depth/instance checks.
The oblique mode changes proposed XY offsets only, not the original postures,
head solver or label geometry. It can combine with the opposite-side flag.

These are individually checked STATIC snapshots, not collision-free base paths,
hardware poses or a way to move the robot through the gutter. Unqualified modes
must not be described as proven to improve yield. Original-side proposal hashes
remain regression-checked; the first oblique data-code regression passed672 tests
and47 subtests (`data/sim_data/clear_regression_20260915_v6.log`).

Additional opt-in `--lean-clear` proposes 5-30 degree forward torso lean by
setting torso_3 to bend+lean, with the same exact URDF/head/floor/scene checks.
The camera stays fixed to its real bracket. It can combine with oblique views;
it does not validate dynamic balance, self-collision or transit motions.
A seed47 lean-oblique native pilot produced two strict, individually assistant-
reviewed images (exit0,687.08s). Larger-family yield remains unmeasured.
`clear_geometry_probe` uses Isaac's USD build without SimulationApp/rendering
to diagnose proposals; its original decisions must match a completed native
reference before alternate modes are evaluated. It cannot approve training data.

Explicit uncertain visual reviews can be passed to a NEW build with
`--visual-exclusions holds.json`. Only attributed, RGB-hash-bound hold/reject
records are accepted. They and the source index are retained in the derivative;
held images are excluded before the view cap. This does not overturn a decision,
change an answer, or relax coverage. Existing drafts and sources are untouched.

`--orbit-clear` additionally requires oblique capture and rotates the actual
robot root toward the target bearing with +/-30 degree heading jitter. Unlike
the position-only oblique proposals, root yaw can span80-280 degrees on the
original side or -100 to100 degrees on the opposite side. The .30-.55m root
radius, fixed camera mount, head joint limits and all floor/geometry/visibility
screens remain checked. This is explicit static pose diversity, not a safe
navigation route or physical balance claim. A seed53 native pilot produced one
strict, individually assistant-reviewed clear image (exit0,587.79s); this is
single-family evidence, not a general yield guarantee. See `vlm_train_data.md`
for active campaign status and immutable capture/review paths.

`--near-clear` is a separate opt-in .20-.40m base-to-target X separation
proposal. With oblique capture it requires `--orbit-clear` and uses the same
.20-.40m XY radius instead. It does not change optics/resolution, source
geometry, real joint limits, floor support, the 10mm geometry margin or native
visibility/label gates. This is a closer candidate search, not permission for
robot/plant intersection, a transit motion or a hardware command. Native
qualification is pending; the original .30-.55m preset remains unchanged.

Defer partial/hidden classes, RGB-D training, action guidance and larger collection
until this visible task is demonstrably learnable. Keep native depth available
for subsequent geometry and visibility checks; never reconstruct it from RGB.
