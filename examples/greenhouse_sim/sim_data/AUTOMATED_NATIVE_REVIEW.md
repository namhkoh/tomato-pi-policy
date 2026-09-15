# Automatic native annotation review

The user authorized autonomous review on 2026-09-15: resolve clear decisions,
hold/reject unsuitable examples, and ask only when a decision really cannot be
resolved. This is **simulator-ground-truth-assisted annotation and QA**, not
self-supervised model training or independent botanical ground truth.

`automated_native_review.py` reads completed native annotation pairs. It does
not launch Isaac, change capture implementation, change prior reviews/splits,
start training, run a VLM, or approve physical actions. A new receipt is written
outside the immutable captures and assets. Existing receipts cannot be replaced.

## What is checked

- Frozen source, generator, annotation and implementation hashes; exact pair
  membership, original donor lineage, TRAIN assignment and original-target cap.
- Actual saved clean native RGB, float32 Isaac optical-Z and validity; camera,
  raw RGB and raw depth fingerprints from the native writer callback.
- Exact native component IDs and the independently compared target mask.
- Recomputed 10 mm nominal / 10-20 mm petiole-arc labels, visible parent context,
  native clarity limits, distal query and full-frame coordinate conventions.
- The entire projected petiole centerline from 8 mm to the query, including
  every chain vertex. Samples are at most 0.5 mm apart and refined to at most
  0.5 native pixel projected separation. Every probe needs the target's native
  identity and consistent native depth. A connected silhouette around an
  occluder does not by itself pass this check.
- Interior width, brightness and local query usability along that trace.
- Unchanged native query crop, exact inference/training prompt parity, answer
  normalization, and RGB-only observation paths. No GT/depth enters model input.

The depth check compares existing centerline geometry with native surface Z
using the existing radius +3 mm tolerance. It does **not** reconstruct depth.
This remains a frozen-scene check. Native engine frame IDs are unavailable in
these captures; callback fingerprints/static guards do not establish moving-scene
synchronization or hardware sensor realism.

## Decisions and escalation

| Decision | Meaning |
|---|---|
| `accept_automatic_annotation_only` | Passed the versioned numeric/native gate; no visual inspection is claimed. |
| `hold_for_visual_review` | Readability or query-to-cut trace needs assistant visual inspection. |
| `reject_clear_task` | Existing native clear-task eligibility failed, e.g. occluded cut interval. |
| `hold_integrity_or_missing_evidence` | Missing/stale evidence or validation error; never converted to acceptance. |

Assistant visual review is a **separate, attributed, hash-bound receipt**. View
the full scene and clean native-pixel junction/query detail, not only overlays.
The assistant can exclude a confusing example or resolve a numeric contrast
hold after actual inspection. A visual judgment cannot override wrong organ
identity, native occlusion, integrity failures or inherited anatomy holds.
Excluding an occluded example from the clear task does not mean there is no
anatomical target; do not invent a no-target or executable-hidden-cut label.

All receipts keep `training_approved=false`. This protects the separate release
gates: deduplication, donor-family splits, target/view caps, diversity, native
resolution exporter/processor validation and held-out evaluation. The review
tool reports unique RGB counts separately from decision counts; repeated
controls/crops are not new training-target diversity. No review program can
turn the current small inspection library into a 20,000-image training release.

## Usage

From repository root, using an ordinary CPU Python with the sim-data dependencies:

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
$env:OPENBLAS_NUM_THREADS='1'
python -m sim_data.automated_native_review --annotations PATH_TO_COMPLETED_CASE/annotations --output data/sim_data/dataset_reviews/NEW_REVIEW/automatic.json
```

`--annotations` accepts multiple explicit completed directories. Do not pass
unfinished captures and call them rejected data; missing evidence is a hold.
Run one serial reviewer beside the one native renderer to avoid memory spikes.
No remote endpoint, API key, model download or GPU is needed for these checks.

## Qualification and limits

30 adversarial CPU tests cover valid traces, one-pixel holes with connected
silhouettes, other stems/fruit, native depth failures, chain bends/perspective,
darkness/contrast, prompt/answer/crop corruption, raw buffer fingerprints,
wrong masks, dynamic records, duplicate inputs and immutable output paths.
Complete sim-data suite: **1018 tests +77 subtests passed in 65.91 s**.
Log: `data/sim_data/clear_regression_20260915_v29.log`.

The numeric reviewer shares simulator anatomy with the label generator. It
cannot independently prove that every author-provided component is botanically
correct or that a VLM can read every passing image. Visual calibration and a
held-out task evaluation remain necessary. These thresholds are engineering
screens, not measured model accuracy, grasp feasibility or cutting safety.

## First real-image qualification (2026-09-15)

Campaign: `data/sim_data/diagnostics/native_generated_seven_family_20260915_v2`.
All seven native camera workers and seven original/generated workers completed;
the queue returned 0. There were 14 original/generated native frames and 13
eligible clear-annotation candidates. Other camera/sensor controls are not
additional training examples.

Final automatic receipts in
`data/sim_data/dataset_reviews/native_generated_seven_family_20260915_v2`:

- `automatic_first_six_v2.json` and `automatic_final_pair_v2.json`:
  **4 automatic accepts, 9 visual holds, 1 clear-task rejection**, no integrity holds.
- `assistant_visual_reviews.json`: every one of the 14 actual images was
  individually inspected using its full-scene overview and listed lossless
  native RGB crops. **12 annotation-pilot accepts, 2 easiest-task exclusions**.
  No unresolved human decision. This does not edit the automatic receipts.

The seed71 generated target was hidden by a leaf: rejected numerically and
confirmed visually. The seed41 generated target had a visible cut but a
distracting near-query crossing: excluded from this first easiest subset by
assistant judgment. The remaining visual holds were local-contrast concerns;
accepted examples had zero native query-to-cut identity/depth gaps. No hidden
cut was made executable or treated as a visible target.

This is a deliberately selected small calibration batch, **not** an unbiased
yield estimate or measured VLM success rate. Numeric contrast screening is
currently conservative and still sends many cases to assistant visual review.

With the earlier eight native frames there are **20 separately accepted native
annotation-pilot frames**, not 20,000 training images. The legacy reviewed draft
remains 450 (312 train /48 validation /90 test). Production release counts,
donor-family splits and caps have not changed. No final ZIP or training started.

A next bounded campaign uses twelve additional original target identities from
the frozen reviewed-reference inventory, rather than revisiting these same
targets. Root: `data/sim_data/diagnostics/native_generated_remaining_targets_20260915_v1`.
It admits one serial renderer only after 20 GiB commit headroom/no other Kit,
retains the native 18/16 GiB gates, and waits at most two hours before launch.
After the attempt it automatically reviews completed pairs, including completed
cases before a later failure, into
`data/sim_data/dataset_reviews/native_generated_remaining_targets_20260915_v1/automatic.json`.
No retries or fabricated visual reviews. At most 24 new annotation candidates;
this is not a queued 20k release. Check `queue_result.json` for actual completion.

## Append-only implementation requalification (2026-09-15)

Default review still rejects changed annotation implementation hashes. Explicit
`--requalify-existing` records old/current known annotation-module hashes in a NEW
receipt, verifies every prior native/source binding and requires recomputed labels
to equal saved labels exactly. It does not rewrite old code proofs, annotations,
images or decisions, and does not claim the old implementation was executed.
Missing/mutated evidence stays held. Seven targeted tests cover this distinction.

After fixing curve-knot sampling, 18 completed pairs were requalified into
`data/sim_data/dataset_reviews/native_curve_requalification_20260915_v1/automatic.json`:
36 raw frames,11 automatic annotation accepts,23 visual holds,2 clear rejects,
0 integrity holds,0 final training approvals. Raw unique hashes do not establish
independent targets (controls repeat and render noise changes hashes).

Actual partial visual receipt in the same folder: `assistant_visual_partial.json`.
Six frames inspected,4 additional pilot candidates,1 easiest-task exclusion,
1 repeated control reference. This brings separately accepted native pilot
candidates to24, NOT20k or a final release. Eight additional old-similarity frames
remain unadjudicated here; the newer seed19 curved pair has automatic accepts only.
Existing20 visual accepts and legacy450 reviewed records remain unchanged.

Full regression after label/requalification fixes and observational capture timing:
1075 tests +87 subtests passed in58.49 s (`data/sim_data/clear_regression_20260915_v34.log`).

### Subsequent completed visual review

All eight remaining extra-target frames actually inspected (848px display-only
overview and lossless native-pixel crops). New append-only receipt in the same
requalification folder: `assistant_visual_remaining_eight.json`,7 pilot accepts
and1 easiest exclusion. Large-leaf occlusion removed the generated seed41/SubStem41
query; visible junction alone was not accepted for the query-associated task.

Second V1 curved pair actually inspected:
`data/sim_data/dataset_reviews/curved_diversity_native_20260915_v1/assistant_visual_case002.json`.
One generated pilot accept, original repeated reference only. Native accepted
pilot count32; no final training approval. The newer V2 rigid-leaf captures are
separate pending reviews. Interrupted/truncated image tool outputs were NOT
counted as inspections; smaller lossless crops were successfully displayed later.
