# Original native capture ? opt-in v1

New nested implementation only. No modified worker, bank, reviews, split
reservations, original assets, or top-level imports. No fake generated plant.

## Interfaces

- **prepare.prepare_plan(...)**: CPU-only, one-case pilot; returns a dictionary.
- **prepare.prepare_batch(..., cases=[...])**: CPU-only, 1?64 explicit target/pose
  cases, all from one historical capture, original donor, and pinned clear scene.
- **prepare.check_plan(plan)**: reconstructs and byte-checks the entire contract.
- **collector.collect(app, output, plan)**: internal execution hook for a checked
  plan; one unchanged greenhouse stage, one native greenhouse render product,
  one instance/depth writer, one notice-invalidated geometry cache.
- **collector CLI**: explicit Windows Isaac launch, memory/process admission,
  fresh sensor smoke once, collection, then automatic saved-buffer replay.
- **audit.audit_capture(capture, result_sha256=...)**: independent CPU replay.
  Audit CLI without an output argument is read-only.

Prefer one exact same-target historical prior per original target, then
additional distinct views. A different requested target requires explicit
allow_other_target_pose; its prior target identity stays separate.
Optional reframe_pixel_xy uses ORIGINAL 1696?816 pixel coordinates and solves
only the actual head joints. Base, torso, arms, mount, optics and clipping stay
fixed. This is not free-camera aiming or robot navigation.

Each request is constrained to an exact target row in the pinned clear plan.
The current plan has 198 TRAIN rows, not all 584 source-anatomy targets. More
source anatomy requires a separately authorized, versioned collection plan.
Neither 198 nor the number of available priors establishes clear/reachable yield.

## Preparation

Use CPU Python; set PYTHONPATH to the repository's examples and
examples/greenhouse_sim directories. Use -B to avoid bytecode writes.

Module: sim_data.native_original_capture.prepare

Required shared flags:

    --clear-plan ABSOLUTE_CLEAR_PLAN_JSON
    --clear-plan-sha256 ORIGINAL_LOWERCASE_SHA256
    --source-capture ABSOLUTE_HISTORICAL_CAPTURE_DIRECTORY
    --source-manifest-sha256 ORIGINAL_LOWERCASE_SHA256
    --output NEW_PLAN_JSON

Default pilot adds:

    --target-id seed73_full/SubStem_41
    --sample-id sample_0443
    --source-sample-sha256 ORIGINAL_LOWERCASE_SHA256

Expansion replaces those three flags with:

    --cases EXPLICIT_CASES_JSON
    --cases-sha256 CASES_FILE_SHA256

The cases file is a JSON array. Each object has target_id, sample_id,
source_sample_sha256; optional allow_other_target_pose defaults false and
reframe_pixel_xy defaults null. Example changed-target case:

    {
      "target_id": "seed73_full/SubStem_50",
      "sample_id": "sample_0443",
      "source_sample_sha256": "0e3f2b89d0b5e20ff74ef11c0af48007fb0561ea5027880f052854ff7aba0df3",
      "allow_other_target_pose": true,
      "reframe_pixel_xy": [848, 408]
    }

This illustrates an attempt, NOT a demonstrated reachable/visible SubStem_50.
Cases cannot reset donor-target cap groups or use repeated exact cameras as
additional images. Reframed duplicates are checked again against actual native
camera/RGB evidence. Near-duplicate/global image admission remains downstream.

Preparation reports coverage in the returned plan: original TRAIN row count,
donor row count, manifest-listed same-target prior counts, distinct requested
targets, same-target/changed-target cases. Unselected historical samples are
not individually verified and old numerical clarity is never native approval.
All selected samples/payloads, original USDs, clear-plan source bindings,
package appearance/daylight files and robot URDF are byte-bound.

### Existing seed73 pilot pins

Repository-relative clear plan:

    data/sim_data/collection_plans/clear_capture_20260915_overnight_v1/plan.json
    SHA256 e017f25d071afd678697f68f75ec3ba0f8f87db265ec3d9e6661958a6b005005

Historical source_capture:

    data/sim_data/collection_campaigns/grounding_resume_20260909_v1/lane_01/capture_job_019/job_019/capture
    manifest SHA256 4ea5b982f4d4052ae4acadc6205ca29e8b840499a267d6467e68b397abb4679f

Pilot: sample_0443 / seed73_full/SubStem_41

    sample.json SHA256 0e3f2b89d0b5e20ff74ef11c0af48007fb0561ea5027880f052854ff7aba0df3

Optional second exact-target case: sample_0285 / seed73_full/SubStem_44

    sample.json SHA256 2ea84cfa74be4af5d216760ceb2e3e5e91b4fce9d575d2d51076be7a06c3ebdc

A read-only development check prepared both cases, validated the result again,
and reconstructed the old complete component catalogue. Preparation took 18.3s;
that plus replay/catalogue checks took 34.9s. It bound 10,873 source files and
73 local implementation files. This is CPU timing, not collector throughput.

Seed73 has 12 clear-plan target rows. Its legacy manifest lists same-target
priors for 10; SubStem_50 and SubStem_52 have none. The manifest lists 541 samples
across those 10 targets. No native images were rendered for this implementation.

## Explicit native execution

After CPU tests and independent review, start with ONE case. With no other Kit
worker running, use the existing bootstrap:

    D:\isaac-sim-6.0.1\python.bat -m sim_data.native_original_capture.collector --plan ABSOLUTE_PLAN_JSON --plan-sha256 PREPARED_PLAN_SHA256 --output NEW_CAPTURE_DIRECTORY

Use a new capture directory disjoint from the plan directory, sources and code.
This command actually launches native capture; preparation does not.
There are no automatic retries or process-killing helpers.

The collector rebuilds the original full scene and applies the current clear
profile (day172/13:00, nominal sun1500/dome6000, RTPT). The legacy seed73 image
had dome1200: that is recorded as pose provenance, NOT claimed photometric
reproduction. Original donor placement, source USD set, source counts, actual
mounted camera and all22 revolute joints are checked. Authored static wheel and
gripper zero semantics are explicit; missing/unknown arm joints are rejected.

Every case receives fresh robot-bound/triangle screening. Non-robot scene
changes invalidate the batch. Pose/screen/clipping failures are held before
rendering and do not prevent later cases. Integrity/sensor failures fail closed.
Frames are fresh native1696?816 RGB, optical-Z, validity and raw instance IDs
from one callback. No depth reconstruction, image upscaling, source-label
transplant, isolated rendering, material change or greenhouse removal.

## Automatic audit and release boundary

The collector automatically reopens saved buffers, reconstructs original
component/parent hierarchy and ID masks, verifies callback hashes and mounted
FK, and recomputes labels, parent/interval clarity and strict query-to-cut trace.
Decisions are accept_strict_automatic_annotation_candidate,
exclude_geometry_or_visibility, hold_visual_clarity, or held_pre_render.
This is simulator-GT-assisted automatic annotation review, not human review.

Independent replay:

    python -B -m sim_data.native_original_capture.audit --capture ABSOLUTE_CAPTURE_DIRECTORY --result-sha256 RESULT_JSON_SHA256

Optional --output writes only a new, disjoint receipt. Callers must additionally
verify native worker exit0 and absence of failure.json. A result written before
native cleanup is not by itself proof of successful process exit.

All acceptance is ANNOTATION ONLY: training_approved=false, source_cap_reset=false,
zero new biological-family credit. Original target cap keys are preserved.
Global caps, held/excluded-image history, exact/near-image duplicates and final
release admission must still run; this package contains no dataset exporter or
global cap ledger. Fresh automatic audit never promotes historical RGB/labels.

## Minimal observed-inventory adapter (NOT IMPLEMENTED)

The pilot is runnable without this adapter; its output is NOT currently an
input to native_dataset.inventory.build_inventory. That reader's _receipt
requires the generated completion/audit states, and _case_lineage requires
anchor_pair_plan, variant_directory and generated qualification.json.
Original captures have none of those. Do not manufacture them or relabel this
audit as completed_automatic_annotation_replay.

Smallest next module: an opt-in original-only observed-row adapter, not another
collector, bank generator or release framework. Proposed interface:

    original_observed_rows(capture, *, result_sha256, launcher_exit_receipt)
        -> (records, capture_summary, source_and_implementation_bindings)

This interface is a proposal, not an available command. The launcher receipt
must bind the owned process, exact request/result pins and successful exit0;
neither a result file nor this module's audit output proves process exit.

Required adapter work:

1. Require no failure.json, verify the launcher receipt, and call
   audit.audit_capture(capture, result_sha256=...) for fresh saved-buffer replay.
   Pin result, request, plan, automatic_audit.json, selected source ancestry,
   sample/label/trace, all logical payloads and the adapter/replay implementation.
   The current optional audit --output contains the replay object only, not
   the capture locator/result pin expected by native_dataset. Carry those
   explicitly; do not feed that file to its existing ReceiptPin reader.

2. Emit one common observed row for each native_captured case, including strict,
   hold and exclude decisions. Keep held_pre_render cases in the capture summary
   with reasons and no invented image row. Reconcile complete scheduled/captured/
   held membership, ordering, counts and callback freshness. Only fresh strict
   label PLUS strict trace gets annotation_review.passed=true, method=automatic.
   Preserve training_approved=false, source_cap_reset=false, depth_recomputed=false.

3. Map target_id and the compatibility field variant_target to the exact
   requested ORIGINAL target. source_family=original_donor_family is the frozen
   donor; source_target=conservative_view_cap_group is that requested target,
   never the historical prior target. Add source_kind=original_native and a
   versioned original lineage object: pinned clear plan/row, frozen24 assignments,
   original manifest/component/USD hashes, source capture and separate pose-prior
   identity. No generated_* lineage keys or independent augmentation credit.

4. Match inventory.py's common record evidence fields: sample_id/candidate_id/
   paths, resolution, decision, encoded_rgb_sha256/image_bytes, decoded_rgb_sha256/
   decoded_rgb_bytes, callback_rgb_sha256, scene/camera/scene_camera hashes and
   identity bases, annotation_review and provenance.logical_files. Compute the
   decoded digest with native_dataset.admission.decoded_rgb_digest, not a PNG
   hash or an unprefixed callback digest. Use actual saved calibration, mounted
   robot pose and fresh clear-scene evidence, never prior-image observations.
   Record exactly which buffer/label/trace checks were replayed; do not copy the
   generated reader's different audit-declaration fields as if they ran here.

5. Original context identity is original:SOURCE_TARGET, with geometry from its
   authenticated original manifest/assets and parent relationships. Scene
   identity must bind the full original population/placements, background USD,
   appearance, current clear lighting, actual renderer settings and target
   placement. The seed73 prior had dome1200; fresh capture requests dome6000.
   Blind reuse of _scene_basis/_features would compare against the wrong light.
   Use a versioned neutral scene-identity normalization (plant_to_world, not a
   fictitious generated replacement), applied consistently to old and new rows
   when comparing them. Keep provenance kind outside the physical-scene hash.
   Asset-content hashes are exact evidence, NOT rigid/scale/near-geometry tests.

For the first prototype, a versioned union builder can preserve the common
greenhouse.pinned_native_observed_inventory.v1 packet/row contract consumed by
native_dataset.provisional.select_observed_train, with additive source-kind/
adapter-policy provenance. Authenticate the existing generated receipts through
their current reader; add these separately authenticated original rows; recompute
packet counts, bindings, duplicates and seal. Do not overwrite the frozen
686-row snapshot. A new union requires a newly bound complete near-image pair
coverage receipt; the old inventory digest/coverage cannot be reused. The
provisional selector then retains its current donor-target source cap unchanged.
If changing required packet semantics, use a new schema and explicit selector
support instead of claiming v1 compatibility.

This is not full geometry-policy integration. geometry_admission_v2._contexts
currently forbids associated_sample_ids on original descriptors and validates
generated qualification for joined samples. A later explicit original join
must authenticate the original component/parent extraction and join its views
to the SAME original context (qualified_geometry=false, self source_context_id).
No synthetic generated qualification or new biological donor is warranted.
Global morphology/rigid/scale and near-image checks, cross-split rejection and
held/excluded history remain required before any release.

Minimum adapter tests: changed ancestry/split/pose-prior target, absent or
nonzero launcher exit, missing/reordered case, rehashed stale label/trace,
old-vs-current lighting, original/generated identical decoded RGB and physical
scene/camera, shared original-target cap, cross-split duplicates, retained holds,
and refusal to reuse old pair coverage. This package does not implement that
adapter, mutate the current inventory or grant dataset admission.

## Throughput and extension boundary

Full source hashes, original structural reports, scene construction and product
setup are shared across cases, not repeated for every target. Runtime static
guards and fresh robot bounds still run per case. One donor currently offers
roughly a dozen original rows; 64 is a bound on cases/frames, not evidence for
64 distinct original targets in the current seed73 plan.

v1 deliberately fixes legacy instance IDs and 56 subframes for EVERY view.
One sensor-smoke sequence is paid per job. No 20k runtime/yield claim is made.
An explicit future version may reuse native_instance_adapter and native_budget
for qualified fast IDs and warm56-then8. It must bind backend/equivalence evidence,
budget/profile, first/subsequent-frame rules and auto-audit policy, and pass a
native parity pilot. There is no silent short-budget or fast-backend default.
That extension is not required to review or qualify this baseline pilot.

## CPU tests

    python -B -m pytest -q -p no:cacheprovider examples/greenhouse_sim/sim_data/native_original_capture

Tests cover hash/split/pose/target mutations, missing arm joints, rigid mounts,
same-target and explicit reframing contracts, source verification reuse,
native-sized synthetic label/trace failures, stored-buffer tampering, AST parity
with existing robot FK application, one-product batch structure and cleanup.
Synthetic fixtures are not native/visibility/throughput qualification.
