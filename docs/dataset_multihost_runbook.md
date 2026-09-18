# Shared collection contract: 25,000 samples by Saturday

Planning deadline: **2026-09-19 23:59 Asia/Seoul**. Target: **at least 25,000 globally distinct, accepted samples collectively**, including the existing verified release. The starting release has 491 images. Generated assets, proposals, raw frames, preview controls, and duplicate or quality holds do not increment that total. This is a delivery target, not a measured capacity guarantee.

Start with the machine-specific instructions: [thor1 / four L40S](dataset_thor1_runbook.md), [thor3 / two RTX PRO 6000](dataset_thor3_runbook.md). Local collection uses [local5090.json](../configs/dataset_capture/local5090.json).

## What is implemented and what must pass on the servers

Implemented: host configuration validation, exclusive foreground donor assignments, distinct generation seed namespaces, a callable qualified plant generator, shared annotation entry points and source lock, portable geometry/view identities, a single HTTP/SQLite reservation coordinator, original-sweep and generated-batch reservation adapters, and exact/perceptual duplicate screening.

**The original native capture owner is Windows-specific.** The server agents must finish the Linux process/GPU adapter and qualify it against the common numerical annotation code. The existing `native848_scene_sweep_raw_capture_v1` owner is not a Linux launch command. This change does not pretend that installing Isaac Sim or obtaining the assets completes that port. Never edit frozen numerical modules to make an unsupported capture pass ownership checks; add a typed Linux adapter.

Both servers reported no active production collection: thor3 was preparing a two-image preview; thor1 was installing Isaac Sim. Preview controls remain outside dataset totals.

## 1. Obtain one code revision and the assets

Both servers must use the same committed coordination/generator/annotation source revision. The older Hugging Face code snapshot predates this change. Keep server-specific port changes additive and record their hashes.

The verified asset bundle is private dataset `namhokaist/tomato-greenhouse-sim-assets`, revision `e9b73a1c2161ba04ebd8a103f7a056ac04a49efc`, directory `releases/20260918_v1`. Follow the asset extraction commands in the [L40S guide](dataset_four_l40s.md#2-download-the-published-scene-bundle). Isaac Sim 6.0.1 is installed separately. Build destination-specific path/runtime bindings; do not replace Windows strings inside old receipts and retain their old hashes.

The supplied `native848_fixed_scene_diverse_aisle131_prepare_20260918_v1/request.json` is a **historical diagnostic**, not a production queue. It yielded 17 raw frames and one accepted frame locally.

From the repository root, using a Python environment with the required NumPy/Pillow/OpenUSD dependencies:

```bash
export PYTHONPATH="$PWD/examples:$PWD/examples/greenhouse_sim"
export OPENBLAS_NUM_THREADS=1
python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" verify
```

`verify` checks the same exact source bytes and annotation contract on every host. Both servers must report the same `toolchain_lock_sha256`. Do not regenerate the lock independently on a server to hide code drift.

## 2. Unique ownership and generation

| Machine | Foreground original donors | New generator seeds | GPU workers |
| --- | --- | --- | --- |
| local5090 | 19, 41, 67, 103 | 1,000,000–1,999,999 | 1 |
| thor1 | 7, 11, 43, 47, 73, 101 | 2,000,000–2,999,999 | 4 |
| thor3 | 17, 23, 53, 71, 83, 89 | 3,000,000–3,999,999 | 2 |

These donor numbers name existing plant shapes. They are not random generation seeds. Every generated descendant keeps its original donor and TRAIN split; it is never a new independent original plant. All three production configs collect TRAIN. Server names are not train/validation/test partitions.

The generator command is executable and uses `procedural_petiole_controlled_v4`. It keeps the complete unpruned donor plant and deforms explicitly chosen petiole/leaf subtrees while protecting the proximal cut geometry. Different seeds choose direction controls. Generator qualification and content hashes do not establish visual novelty: compare actual images, cap each geometry at approximately 3–4 meaningful camera angles, and prioritize new target identities.

The common CLI takes a **fresh, destination-authenticated** `--source-plan` and its actual SHA256. Example commands are in each server runbook. `--recipe-only` writes the proposed recipe without generating geometry. Without that flag, the command generates the actual assets and current 9mm geometry evidence; failure remains a hold. Never lower a geometry threshold or silently search until an arbitrary seed passes.

## 3. Background generation and scene identity

Use `native848_fully_labeled_scene_v1.populate_labeled` through the qualified scene builder. Every scene retains **144 complete, unpruned, manifest-backed labeled plants** in the greenhouse. A generated foreground replaces one full plant; the other 143 remain complete. Native scene population and the CPU expected census must match exactly. Register every plant/component in the renderer-ID catalogue.

The current qualified background recipe fills the original greenhouse slots from the same-split donor pool. **The layout is shared across machines; these configs do not claim independently randomized greenhouse backgrounds.** Different foreground geometries and views produce different samples. If a server changes background geometry or placement, it must rebuild the full census/catalogue and scene identity, recheck robot collision and all competing cut targets, and qualify that new scene before bulk capture. Do not make unlabeled backdrops or prune plants to force a single answer.

Check the actual census, not merely a config flag:

```bash
python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" check-background \
  --census "$CENSUS" --census-sha256 "$CENSUS_SHA256"
```

This structural check does not replace native pixel coverage. Each accepted image must visibly contain background vines, have at least 40% background-plant coverage under the current exporter, and zero unknown pixels/targets.

## 4. Use ONE coordinator for all machines

One coordinator serializes reservations in a SQLite database on its **local disk**. All seven GPU workers contact that same service. Do not run separate copied databases on the servers. Stop production if the coordinator is unreachable; there is no local fallback or automatic claim expiry.

Initialize it once from the verified portable exclusion inventory (accepted images plus completed raw views), then serve it. The committed baseline is `configs/dataset_capture/baseline_inventory.jsonl` (SHA256 `19a834067de76afd9b96a57a7c8aba68b59d4aaea9126a1279ac54f721f20a3a`):491accepted+68raw-only,559unique images. The already running local coordinator is the authority; these initialization commands are for the operator creating that single service, not for each GPU server. Keep the token and database in ignored runtime storage, never Git:

```bash
python scripts/dataset_capture_coordinator.py init --db "$COORD_DB" \
  --inventory "$EXCLUSION_INVENTORY" --inventory-sha256 "$EXCLUSION_SHA256"
python scripts/dataset_capture_coordinator.py serve --db "$COORD_DB" \
  --bind 127.0.0.1 --port 8769
```

Set `DATASET_COORDINATOR_TOKEN` to the same strong secret on the coordinator and authorized clients. Connect over a private network/TLS or an SSH tunnel; do not expose this HTTP service directly to the public Internet. The service defaults to loopback. The service token is read from the environment and is not printed.

Each worker sets `DATASET_COORDINATOR_URL` and `DATASET_COORDINATOR_TOKEN`. Atomic reservations enforce host ownership, original donor split, content-based geometry identity, plant-relative camera identity, and a full-scene/world-camera alias. Hint labels, filenames, machine paths, renamed variants and seed strings do not create a new view. Scene aliases also catch one image retargeted to a different donor. Very close poses are held before rendering.

Original-plant sweep preparation is an implemented CLI. Run once per worker with its own fresh output:

```bash
python -m sim_data.native848_multihost_sweep_v1 \
  --config "$HOST_CONFIG" --worker-id "$WORKER_ID" \
  --request "$FRESH_REQUEST" --request-sha256 "$REQUEST_SHA256" \
  --output "$ASSIGNED_REQUEST_DIR"
```

It filters host donors and assigns each stable task ID using `int(task_id,16) % worker_count`, reserves only its subset, and produces a new request accepted by the unchanged request validator. A second attempt cannot reclaim it. An empty assignment produces no capture request.

Generated persistent batches use their own implemented adapter:

```bash
python -m sim_data.native848_multihost_generated_v1 \
  --config "$HOST_CONFIG" --worker-id "$WORKER_ID" \
  --request "$PERSISTENT_REQUEST" --request-sha256 "$PERSISTENT_REQUEST_SHA256" \
  --reference-context "$QUALIFIED_REFERENCE_CAPTURE/context.json" \
  --output "$NEW_RESERVATION_DIR" --reserve
```

This adapter verifies the completed reference scene and substitutes only the authenticated complete foreground plant. It assigns the **whole batch** to `int(batch_id,16) % worker_count`, where `batch_id` hashes the sorted candidate task IDs. Other workers do not contact the coordinator for that batch. Launch only when `reservation_gate.json` says `capture_request_eligible: true`; every novel candidate must have a grant. Replay controls are explicitly excluded. Partial grants hold the batch and do not authorize automatic splitting or retries. The reference must have genuine destination-native provenance; copied Windows receipts do not establish a Linux capture.

Preserve failed reservations. An operator may recover them only after proving the old worker stopped and reconciling its saved outputs. Never make a new task ID to retry a partly completed capture.

## 5. Native Isaac Sim Replicator capture

The server agent must provide a qualified Linux worker that consumes the assigned request. It must use **Isaac Sim Replicator**, reuse a resident populated scene/render product/writer, and save synchronized native annotator outputs. Bind each process to one verified physical GPU UUID; set explicit renderer and physics devices, `multi_gpu=False`, `max_gpu_count=1`. `CUDA_VISIBLE_DEVICES` alone is insufficient proof of renderer placement.

For every saved frame, retain:

- Native **848×408 RGB**, without resizing or a query overlay.
- Aligned **float32 optical-Z depth in metres** (`distance_to_image_plane`), plus a validity mask. RGB-D means this paired RGB and metric depth; a depth-colour preview alone is insufficient.
- Native instance IDs and the renderer-ID-to-component mapping for all visible plants, greenhouse structures and robot parts.
- Camera intrinsics, camera-to-world and camera-to-plant transforms, clipping/depth conventions, robot base/joints/mount and actual workspace evidence.
- Scene/census/catalogue and generated geometry content IDs, original donor/split, global task/claim IDs, physical GPU identity, source/toolchain hashes, frame timestamps and callback/reset/freshness evidence.

Retain the qualified reset and 8-subframe rendering profile for initial qualification. Any faster profile needs a separate freshness/clarity qualification. The Linux adapter must authenticate its real owner, child, GPU and successful closure. Do not invent Windows process receipts or accept partial output directories.

After saving immutable frames and successful owner closure, finish each reservation and keep its response with the frame. The original raw-sweep completion CLI authenticates the actual owner, exact granted schedule, source bindings and saved RGB before contacting the coordinator:

```bash
python -m sim_data.native848_multihost_complete_v1 \
  --config "$HOST_CONFIG" --capture "$COMPLETED_CAPTURE" \
  --reservation "$ASSIGNED_REQUEST_DIR/claims.json" \
  --output "$NEW_COMPLETION_DIR"
```

This version supports the original raw-sweep authority only; generated/persistent and Linux producers require their corresponding typed adapters. The client API is `Client.finish(reservation, rgb_metrics(rgb_path))`. Do not substitute that low-level call for authenticating a completed native job. Exact decoded-RGB repeats are held globally. dHash distance ≤6 creates a **review hold**, not automatic acceptance or deletion. No duplicate-held frame enters the final delivery. Completion only means capture accounting; `training_approved` remains false.

## 6. Shared annotation and acceptance

Use the same pinned entry point and unchanged numerical predicates on all hosts:

```bash
python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" annotate \
  --kind sweep-raw --capture "$COMPLETED_CAPTURE" --output "$NEW_ANNOTATION_DIR"
```

Supported kinds: `sweep-raw`, `sweep-ordinary`, `controlled-generated`, `persistent-generated`, `experimental-generated`. For persistent kinds, pass a segment's `capture` directory after the **whole batch** closes. These current consumers authenticate the original producer contracts; a Linux producer requires a corresponding typed provenance adapter while retaining the same label/visibility/workspace maths. A source-lock pass alone does not qualify that adapter.

Every accepted image requires:

1. Exactly **one eligible, clearly visible, robot-reachable petiole**, with no query cue and no unresolved competing answer anywhere in the full labeled scene.
2. The specific cut point **9mm along the petiole from its main-stem attachment**, with clear junction, proximal shaft and first-leaf context. Do not label the whole branch as the answer.
3. The existing native detail checks: at least 8px mask width and 12px projected support, plus full continuity/occlusion checks. The nominal diameter estimate is only a proposal ranking aid.
4. Real robot-camera FK, floor, whole-robot collision and workspace checks; aligned RGB/depth/IDs and fresh callbacks.
5. Full populated-background and zero-unknown checks; `seed41_full/SubStem_38` remains excluded.
6. Global duplicate checks, distinct-view assessment, and actual visual review of each target/view group and all flagged alternatives.

Store annotations separately from model inputs. Preserve original donor train/validation/test lineage. Both servers must run a shared small control set and compare target IDs, cut coordinates, depth convention, visibility/ambiguity decisions and reject reasons before scaling. Controls are diagnostic and excluded from totals.

## 7. Collective progress and deadline

Each hour report generated geometries, proposed views, raw saved frames, annotation-complete frames, quality-passing frames, duplicate/review holds, **accepted globally unique samples**, and counts per original donor/target/generated morphology. Include the common toolchain hash and active GPU worker count.

One publisher merges reviewed exports, checks global IDs/RGB hashes/near-repeat decisions again, verifies all sensor inputs, and publishes the authoritative total. Never add two server row counts together without that merge. Coordinator `complete` is a raw-capture state, not an accepted-image count.

Use measured throughput:

```text
remaining = max(0, 25000 - globally_unique_accepted)
required_accepted_per_hour = remaining / hours_until_2026-09-19_23:59_KST
measured_rate = new_globally_unique_accepted / elapsed_wall_clock_hours
```

There are 24,509 additional accepted samples to obtain from the initial 491 baseline. No machine has a separate 25k quota. Prioritize useful new geometry/target supply and overlap CPU preparation/annotation with GPU capture. Scale one→two→all assigned GPUs after the preview and ownership/annotation checks pass. Report a forecast shortfall immediately; do not fill it with tiny pose jitter, ambiguous views or duplicate images.


At Friday 22:00 KST, approximately 26 hours remain: the campaign needs about **944 accepted samples/hour collectively**, or **135 per GPU/hour if all seven GPUs run continuously**. Setup time and rejected frames increase the required capture rate. These are required rates, not observed performance.

## Validation at handoff

- Coordinator: 25 tests covering actual HTTP, concurrent reservations, restart, baseline aliases, split/host ownership and duplicates.
- Geometry identity/original-sweep adapter: 20 tests. Generated persistent adapter: 13 tests. Shared generator/annotation wrapper: 13 tests. Raw completion adapter: 9 tests.
- Both server-specific generator examples ran successfully locally, producing complete geometry and reconstructed 9mm evidence. They are not Linux capture validation.
- The shared annotation CLI replayed a saved 17-frame batch with exact scientific parity: all 188,156 target evaluations and 77,469 alternative assessments matched, including cut coordinates and ambiguity decisions. Background, coverage and metadata outputs matched byte-for-byte.
- The first coordinated local batch saved six RGB-D frames and completed all six claims with zero duplicate holds. None met all acceptance criteria, so the verified total remains 491. Raw throughput cannot stand in for accepted throughput.
