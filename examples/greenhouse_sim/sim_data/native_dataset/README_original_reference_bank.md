# Original reference bank v1

CPU-only, opt-in verifier. No source edits, generator, native launch, original
review changes, training approval or cap reset. It is NOT a V3/V4 reference bank
and is NOT accepted by the legacy paired848/1696 prerequisite.

## Implemented API

`original_reference_bank.build_bank([OriginalCapturePin(...)], reviews=[ReviewPin(...)])`
returns deterministic JSON. Every planned case survives, including pre-render
holds and captured exclusions/clarity holds. Native decisions are copied unchanged.
No reviews means no reference-ready anchor, but does not erase a strict native row.
All explicitly supplied reviews survive; negative/conflicting history withholds
the affected anchor. No review-directory discovery or global history claim.

`check_bank(bank)` independently rebuilds it, including saved native audit replay.
`verify_anchor(bank_path, bank_sha256=..., entry_id=...)` first authenticates the
caller-pinned file, rebuilds it, and returns only that exact ready anchor's evidence.
`write_bank(output, captures, reviews=...)` exclusively creates a disjoint JSON.

CLI module: `sim_data.native_dataset.original_reference_bank`

```
build --capture CAPTURE RESULT_SHA LAUNCH_RECEIPT LAUNCH_SHA
      --review REVIEW_PATH REVIEW_SHA --output NEW_BANK_JSON
check --bank BANK_JSON --bank-sha256 BANK_SHA [--entry-id EXACT_ID]
```

Use the repository CPU Python with `examples/greenhouse_sim` on `PYTHONPATH`.
There is no hash-only shortcut or injectable audit callback. V1 accepts only
`original_inventory.OriginalCapturePin`, the pinned V5 original-pilot launcher
contract. Other/serial/compact producers need a named versioned adapter; do not
rename their state fields. Unknown producer types fail closed.

The bank schema is `greenhouse.original_native_reference_bank.v1`; the proof is
`greenhouse.original_native_1696_anchor_evidence.v1`. Both keep these chains apart:

- `pose_prior`: original848 manifest/sample/old plan, exact pose/calibration,
  original source row and historical lighting declaration. Never a current image.
- `scene_authority`: current clear plan/source row/frozen TRAIN, original meshes,
  appearance/population/placement, fixed policy and actual native lighting/renderer.
- `native_observation`: actual1696 sample/RGB/label/trace, request/result/plan,
  launcher/post-exit audit, known-surface smoke and callback/geometry evidence.

Large source maps are stored once in bank `source_bindings`. Per-entry map digests
refer to the pinned historical manifest (`source_usd_sha256`) and pinned native
plan (`source_bindings`); consumers must read those exact files, not discover new
assets. This avoids repeating the full greenhouse hash map for every reference.

Saved native arrays and annotations are replayed. Historical execution/OS exit
and reviewer identity are not independently attested: launcher exit is a checked
producer declaration; visual review is pinned existing evidence, not a new review.
Full renderer settings were not recorded by the original collector and stay unknown.
Mount quantization is only a grouping index; a consumer must recheck actual mounts
at1e-9 and exact optics, scene/policy/placement before sharing a stage/proof.

## Proposed generated seam -- NOT IMPLEMENTED

1. New preparation API takes caller-pinned bank, exact anchor entry ID, one explicit
   reference entry per target (initially1..12), and a clear-plan-bound variant.
   Rebuild bank once; require ready references in one compatibility group, exact
   original donor/component/cap identities, and the anchor among selected entries.
2. Reuse the unchanged geometry generator/catalogue with
   `scene_authority.clear_plan`, NOT the historical pose's collection plan.
   Generate a NEW base-plan schema containing separate `pose_prior`,
   `scene_authority`, `original_anchor_evidence`, `source_row`, `generated_row`
   and variant qualification/code/source pins. Do not fabricate a legacy base.
3. A new scene hook reuses original-native full greenhouse construction and exact
   robot-pose restoration plus the existing session-only generated substitution.
   It verifies actual plant placement,22-joint FK/mount/calibration, loaded assets,
   and current clear lighting. Return explicit current scene evidence, not a fake
   `old_manifest`. A bounded collector adaptation must read metadata/renderer checks
   from that current context. Keep native resolution,56-subframes, static guard,
   geometry checks, callback/ID/Z/label/trace gates unchanged.
4. Shared anchor evidence covers mounted1696 optics/native callbacks only. Each
   generated pose/target still needs fresh native visibility and geometry checks.
   No paired848/1696 parity claim. Preserve default V4 producer and its old proofs.
5. A matching NEW inventory lineage adapter must authenticate both chains and
   current background/light evidence; frozen inventory's legacy manifest-plan
   equality cannot be bypassed. Global image/morphology/cap admission remains later.

First integration fixture: the completed seed73/SubStem_41 original pilot. Qualify
the new consumer and downstream inventory on a bounded generated test before
relying on the409 original proposals as generator-scale inputs. None launched here.

## Bounded morphology study -- read-only proposal, no selected cutoff

Freeze24 candidates:3 TRAIN donors,2 supported targets each,4 deterministic
recipe draws per target, within the existing TRAIN-fitted bounds. Register donor/
target rules and seeds before reviewing images; do not replenish failures until
a desired passing count is reached. Record all failures, aliases and measurements.

Use actual mesh/source lineage, rigid/scale-normalized attachment displacement,
parent-relative centerline/tangent/curvature, radius profile and per-leaf geometry/
orientation/layout. Keep component/leaf counts and descriptor blind spots explicit.
The current set pseudometric ignores multiplicity and is not a full-mesh test.
Use metrics to describe/sample coverage, not to grant independent contexts.

Future blinded matched-camera controls should mix exact/rigid/scale/photometric
negative controls and genuine authored geometry changes; reviewers should not see
recipe labels, metric distances or quota targets. Separate plausibility, visible
geometric distinction and native clear-task quality. Calibrate on controls, then
test held-out controls; no cutoff is chosen here and no new budget is granted.

Single-link near-edge connectivity can chain distant endpoints. Complete-link
clusters bound within-cluster diameter but have tie/order choices and can put
near-identical pairs in DIFFERENT clusters: cluster count is not independent
capacity. Keep exact-equivalence, unresolved relations, pairwise spacing and
training-budget policy separate; never relabel unknown as distinct to reach20k.
An eventual spacing subset must satisfy every retained pair against independently
calibrated criteria, not merely select one representative per complete-link cluster.
That subset is still not an automatic grant of new training-budget groups.
