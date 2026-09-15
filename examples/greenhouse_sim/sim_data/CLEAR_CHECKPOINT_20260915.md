# Clear-image collection checkpoint: 2026-09-15

Latest update: v11 has450 reviewed images (312 train /48 validation /90 test),
73 targets and21 original donor families. The v10 history below is preserved.
The v11 section at the end gives current paths and subsequent native work.

Branch: koh-dev/sim-data. This is a reviewed engineering checkpoint, NOT the
requested 20,000-training-image release or permission to launch training.

## Actual selected data

Root: data/sim_data/collection_intake/clear_combined_20260915_checkpoint_v10.

| Split | Images | Original targets | Original donor families |
|---|---:|---:|---:|
| Train | 311 | 51 | 14 |
| Validation | 48 | 9 | 3 |
| Test | 90 | 12 | 3 |
| Total | 449 | 72 | 20 |

All449 selected original RGBs have individual, hash-matched assistant reviews.
This is not independent human, horticultural or physical-cut validation.
No selected image has an outstanding hold;14 negatives remain excluded.
Coverage still fails the existing small qualification gates, which themselves
are NOT the20k requirement. No gate, frozen family split or old review changed.

Artifacts relative to the root:

- draft/images: clean native848x408 robot-head RGB PNGs.
- draft/depth: unchanged native Isaac optical-axis Z and validity sidecars.
- draft/labels: anatomical projection and visibility evidence.
- draft/crops: query-derived review/model crops; not extra sensor captures.
- draft/splits: canonical supervised chat records, still blocked from training.
- review/index.html: local review GUI (the policy-selected review subset).
- accepts.json, holds.json, source_reviews.json: attributed decision provenance.
- missing_visual_reviews.json: empty; checkpoint.json: counts and validation.

The449-image checkpoint replaces419 as the latest inspected selection, without
modifying that earlier checkpoint. It adds30 net images after exclusions and
the unchanged12-view-per-original-target cap. Crops and repeated source views
are not new independent targets or images toward the quota.

## Newly completed visual review

Scope and per-job decisions:
data/sim_data/dataset_reviews/clear_orbit_wave3_20260915_v1.
Reviewed76 new originals individually:69 accepts,7 holds. Combined historical
review inventory:534 records, including14 held examples. Source and label hashes
are bound; no approval was copied from another camera view.

Full images were displayed without resizing at848x408 through JPEG transport;
ambiguous regions were additionally inspected as lossless PNG crops. Original
RGB files were unchanged. Review used ground-truth coordinates as annotation
guidance; it was not blind model evaluation.

New holds concern ambiguous distal-query crossings (2), attachment obscuration
(2), and local petiole/fruit blending (3). A held fruit overlap is not a claim
that the ground-truth cut label is anatomically wrong. It fails this stricter
clear-image training task. Numeric interval visibility alone does not prove
that the attachment or query-to-target association is visually unambiguous.

The completed orbit campaign captured318 native frames across18 audited jobs;
187 met the strict numerical clear screen before aggregate selection/review.
Campaign: collection_campaigns/clear_capture_20260915_orbit_v2 under data/sim_data.
These numbers overlap earlier checkpoints and must NOT be added to449.

## Task and input verification

Input: original robot-head RGB plus a distal petiole query coordinate. Optional
second image: query-centred crop. Output: status, cut_point_uv, visibility,
next_action. The answer is the nominal10mm point along the target petiole;
evaluation also uses the10-20mm arc-length segment, not a spherical tolerance.
This is target-conditioned2D localization, not autonomous target selection,
occlusion clearing, metricXYZ control, dynamic action experience or cutting.

New read-only command from the repository root (choose a NEW output):

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
python -m sim_data.clear_input_audit --dataset data/sim_data/collection_intake/clear_combined_20260915_checkpoint_v10/draft --output data/sim_data/diagnostics/NEW_clear_input_audit.json
```

Measured artifact: data/sim_data/diagnostics/clear_checkpoint_v10_input_audit_20260915.json.
Source manifest SHA256: de29ad998139e194036f8400d5db4db2156b32092730c1f7ac775b6782b107d3.
Both full-RGB and full-RGB-plus-crop modes passed on all449 records. Training
and inference RGB/prompt inputs match. Two-decimal normalized0..1000 answer
round-trip error: maximum0.00416000000007 original pixels. Native depth stays
outside these RGB model inputs. No model weights, Hugging Face processor,
GPU training or performance evaluation were run. The strict release validator
and training loader remain unchanged and reject this incomplete draft.

Data regression:926 tests plus77 subtests in65.83s.
Log: data/sim_data/clear_regression_20260915_v23.log.

## Native continuation and limits

Original supplemental shard2 stopped before any new worker/image because the
Windows16GiB commit reserve was unavailable. Its failed directory is preserved.
Only the exact assistant-owned stranded dependency waiter was stopped; no user
applications were closed by the agent. Cancellation receipt:
diagnostics/generated_native_pair_queue_20260915_v1.cancelled.json under data/sim_data.

The user freed memory. At10:15:39 KST the bounded recovery queue measured
22.38GiB commit headroom, passed two consecutive samples and started the camera
diagnostic. It waits for18GiB to leave startup margin above the native16GiB gate.
It runs one owned renderer at a time, with1800s worker deadlines,191 unchanged
source/code/plan pins, and no automatic retry or larger collection launch.

Current native validation log:
data/sim_data/native_validation_memory_recovery_20260915_v1.log.
Worker logs and receipts:
data/sim_data/diagnostics/native_validation_memory_recovery_20260915_v1.

Both diagnostics completed with exit0:150.406s for the camera test and145.235s
for the original/generated plant pair. All four images were checked numerically
and individually assistant-reviewed, including native junction crops, masks and
depth heatmaps. See NATIVE_RESOLUTION.md and GENERATED_NATIVE_CAPTURE.md.
The native modes preserve the same ideal-sim camera, not a validated physical
D405 operating mode. Generated diagnostic images have NOT entered this449-image
checkpoint or earned training or novel-target/view-cap approval.

Explicit continuation after those tests:
collection_campaigns/clear_capture_20260915_original_shard2_train_v2 under
data/sim_data. It runs the same frozen four original TRAIN-family jobs
(seed23/47/53/73), with the original1800s per-worker deadline and all gates.
The failed v1 remains unchanged. Code/plan hashes were checked again before
launch; only one native renderer runs. Started10:23 KST on job007.
Log: data/sim_data/clear_original_shard2_recovery_20260915_v2.log.
Non-approving intake: data/sim_data/collection_intake/clear_original_shard2_recovery_20260915_v2.
New captures still require individual review before joining any checkpoint.

Before a final transfer ZIP: complete native resolution/export qualification,
inspect the generated native pilot, establish genuine target diversity and
near-duplicate grouping, then collect and review20,000 accepted training images
plus held-outs. No model download or local training is needed for this work.

## v11: reviewed seed23 data and native annotations

Latest root: data/sim_data/collection_intake/clear_combined_20260915_checkpoint_v11.
Its draft/images, draft/labels, draft/depth, review/index.html, accepts.json,
holds.json and checkpoint.json have the same meanings as v10.

Three completed native seed23/SubStem_40 frames were individually inspected,
including lossless query/junction crops: one accept, two holds for query points
at confusing neighboring-stem/fruit crossings. The held images were not relabeled
as clear or moved to other splits. Review:
data/sim_data/dataset_reviews/clear_seed23_20260915_v1/assistant_reviews.json.
All450 selected images have matching assistant reviews;16 negatives remain
excluded. Source review history has537 records. Current train coverage is
312 images /52 targets /15 families; validation48/9/3; test90/12/3.
Coverage remains incomplete. The extra family is seed23, not a new generator seed.

Supplemental v2 completed seed23 (3 images,484.995s), seed47/53 (no screened
views), then stopped before seed73 because memory fell below the reserve.
After checking the recovered reserve, only seed73 was resumed in a new v3
campaign. It finished with no screened viewpoints (601.863s). All failed/empty
attempts remain intact; the stopped v2 is not called a completed aggregate.
Only its independently completed/hash-bound seed23 job entered v11.

Actual v11 model-input audit:
data/sim_data/diagnostics/clear_checkpoint_v11_input_audit_20260915.json.
Both original-RGB and RGB-plus-query-crop prompt/image contracts passed on450
records; no HF processor or model ran. Native optical-Z remains unchanged.

New high-resolution annotation implementation and two personally inspected
pilot records: NATIVE_CLEAR_ANNOTATION.md. Full data regression954 tests plus
77 subtests in73.10s (data/sim_data/clear_regression_20260915_v24.log).
The native1696 pilot is separate from these450 legacy-resolution images.
General sampling/export, native-aware H200 integration and20k coverage are
not complete. No final training ZIP or trained-model result is claimed.
