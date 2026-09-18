# Independent collection on thor1 and thor3

This is the current server execution mode. **Neither server needs a connection to the Windows PC, the other server, or a shared coordinator.** Each server keeps one local capture ledger for all its GPU workers. The source images and annotations are merged and checked globally afterward.

This supersedes the mandatory shared-coordinator steps in the older runbooks **for thor1 and thor3**. Local5090 can continue using its existing coordinator. Do not copy that live database to a server. An offline ledger starts from the committed, read-only exclusion inventory and owns only its assigned donors.

## 1. Fixed assignments

| Server | GPUs / worker IDs | Foreground donors | New morphology seeds |
| --- | --- | --- | --- |
| thor1 | 4 L40S / 0?3 | 7, 11, 43, 47, 73, 101 | 2,000,000?2,999,999 |
| thor3 | 2 RTX PRO 6000 / 0?1 | 17, 23, 53, 71, 83, 89 | 3,000,000?3,999,999 |

Donor identifiers are `seedN_full`. Backgrounds still contain complete labeled plants from the qualified same-split pool. Different foreground assignments and seed ranges prevent workers from intentionally repeating the same jobs; **they do not prove that resulting images are different**. Local exact/near-view and RGB checks remain required, and cross-server duplicate review is deferred to the final merge.

Use the existing host config. All workers on one host share **one SQLite ledger on that host's local disk**, never one ledger per GPU. Keep each worker's capture outputs separate. Local ledger completion means raw capture accounting, not training acceptance.

```bash
# On thor1:
export HOST_CONFIG=configs/dataset_capture/thor1.json
export SHARD_STATE=data/sim_data/diagnostics/offline_thor1_20260919_v1

# On thor3 instead:
# export HOST_CONFIG=configs/dataset_capture/thor3.json
# export SHARD_STATE=data/sim_data/diagnostics/offline_thor3_20260919_v1

export PYTHONPATH="$PWD/examples:$PWD/examples/greenhouse_sim"
export OPENBLAS_NUM_THREADS=1
python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" verify
```

## 2. Rebuild the destination source plan

Use the pinned HF asset release from the hardware guide. Define `ASSET_RELEASE` as its `releases/20260918_v1` directory and `ASSET_ROOT` as the verified extraction directory. These are real directories on the server, not Windows paths copied from historical receipts.

```bash
export HISTORICAL_PLAN="$ASSET_ROOT/workspace/data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json"
export LOCAL_PACKAGE="$ASSET_ROOT/workspace/data/sim_data/package_20260905/tomato_greenhouse_pack"
export ASSET_INDEX="$ASSET_RELEASE/scene_assets.files.json"
export FRESH_PREP_DIR="data/sim_data/diagnostics/${HOSTNAME}_linux_source_20260919_v1"

python scripts/dataset_linux_preview.py --config "$HOST_CONFIG" \
  --historical-plan "$HISTORICAL_PLAN" \
  --historical-plan-sha256 0fb6409073ada123e52af73620886d622a2f63f493a04d34962b999b98957f21 \
  --asset-index "$ASSET_INDEX" \
  --asset-index-sha256 24ea7e5d1caae230e19c6114cd6b476ca6544c3e10484f293a2bb9761c8460eb \
  --package "$LOCAL_PACKAGE" --output "$FRESH_PREP_DIR"
```

The helper authenticates package files, rebuilds the plan through the unchanged planner, compares targets/geometry/splits, and writes `source_plan/plan.json` plus `result.json`. The result contains the exact `generator_command` array for this host. Run that command using the server's Python environment with the required OpenUSD dependencies. This produces complete generated geometry and current 9mm evidence; it does not launch Isaac Sim or approve an image.

The source-plan helper passed an actual local CPU replay of 10,849 asset files, 10,230 plan bindings and 24 source families. This is not a Linux native-rendering test.

## 3. Initialize a host-local ledger

Run the offline shard helper once for this host. Use the same state directory when restarting or adding workers. No SSH tunnel, coordinator URL or shared token is required.

```bash
python scripts/dataset_offline_shard.py init --config "$HOST_CONFIG" \
  --inventory configs/dataset_capture/baseline_inventory.jsonl \
  --inventory-sha256 19a834067de76afd9b96a57a7c8aba68b59d4aaea9126a1279ac54f721f20a3a \
  --state "$SHARD_STATE"
```

Use `reserve-generated` or `reserve-raw` for fresh destination-authenticated requests; run `--help` for their exact arguments. Generated requests use a completed destination-native reference context and whole-batch worker assignment. Original sweeps partition stable task IDs among host workers. A reservation never qualifies a Linux capture adapter. Do not re-label a Windows reference receipt as a Linux capture or claim successful native preview from a source-lock pass.

For a fresh, authenticated generated batch, the reservation command is:

```bash
python scripts/dataset_offline_shard.py reserve-generated --state "$SHARD_STATE" \
  --request "$PERSISTENT_REQUEST" --request-sha256 "$PERSISTENT_REQUEST_SHA256" \
  --reference-context "$QUALIFIED_REFERENCE_CAPTURE/context.json" \
  --worker-id "$WORKER_ID" --seed "$BATCH_SEED" --output "$NEW_RESERVATION_DIR"
```

Read `result.json`: launch only when `capture_request_eligible` is true, through the qualified Linux owner. For an original-plant request, use `reserve-raw`, omit the reference-context and seed arguments, and use the returned `capture_request` rather than the unsplit parent request. The generated batch seed is declared provenance; geometry content and camera matrices determine its actual identity.

The ledger enforces the host's donor allocation, immutable claim identity, local duplicate checks and no automatic reclaim. Keep returned claims with the batch. If a request belongs to another worker, route it to that worker; do not alter its identity to make it fit. Distinct generator seed streams can use `first_seed + generator_worker_id + k * worker_count`; this is a proposal namespace, not the capture batch assignment rule.

## 4. Finish native Linux qualification, then scale

**The Linux capture/annotation adapter is still unfinished.** Independent ledgers remove the networking blocker; they do not make the Windows launcher executable on Linux.

For the on-server agents, use the [concrete Linux adapter implementation map](dataset_linux_adapter_handoff.md):

1. Rebuild destination anchor, camera-request, rendering-profile and runtime-source bindings from authenticated assets. Preserve historical receipts unchanged.
2. Add a Linux process/GPU owner and matching typed capture/annotation/completion adapters. Reuse the unchanged scene population, sensor, geometry, visibility and workspace calculations.
3. Save two excluded native preview controls on GPU 0. Verify the full 144-plant census, actual populated vine background, aligned RGB/depth/IDs, actual physical GPU assignment, successful native closure and shared numerical annotation parity.
4. Start one production worker, measure capture and acceptance rates, then scale to two and the full host GPU count after confirming separate physical GPUs and output directories. Keep the stage, render product and writer resident; annotate completed batches while the next batch renders.

Suggested division of port work: the thor3 agent implements and tests the common Linux adapter; the thor1 agent independently validates it on L40S and qualifies four-worker device isolation. Exchange that small code patch once; the eventual production runs are independent. A server can also complete its own adapter against the same contract, but any implementation difference must be checked before merging.

Do not reduce the scene or change the target rule to make a preview pass. Every accepted image still requires native 848?408 RGB, aligned float32 optical-Z depth in metres and validity, instance IDs/mapping, calibration and robot pose, exactly one eligible reachable petiole, the cut point 9mm from attachment, clear junction/first-leaf context, the existing detail and collision checks, at least 40% background-plant coverage and zero unknowns. Keep full unpruned plants. Reject seed41_full/SubStem_38. Review target/view groups and flagged cases from actual images.

## 5. Handoff and final merge

Use the offline helper's authenticated completion path for producers it supports. Linux producers need the matching typed adapter; a raw `finish` call cannot replace successful capture authentication. Export each ledger with `export --state "$SHARD_STATE" --output <fresh-ledger-json>` and retain raw RGB-D, annotation, review and producer evidence alongside it. The ledger export contains identities/scene aliases and RGB metrics, not private claim handles. It is an inventory for merging, not an accepted dataset.

Once the producer is supported and its actual process has closed successfully:

```bash
python scripts/dataset_offline_shard.py complete --state "$SHARD_STATE" \
  --reservation "$NEW_RESERVATION_DIR/result.json" \
  --reservation-sha256 "$OFFLINE_RESERVATION_SHA256" \
  --capture "$COMPLETED_CAPTURE" --output "$NEW_COMPLETION_DIR"
python scripts/dataset_offline_shard.py export --state "$SHARD_STATE" \
  --output "$NEW_PORTABLE_LEDGER_JSON"
```

Use the actual SHA256 of the saved reservation receipt. Existing completion adapters authenticate Windows producers; this command cannot authenticate a Linux producer until the corresponding typed adapter is added. There is no unchecked completion fallback.

The publisher must verify the exported sensor bytes and reviews, preserve donor splits, and compare stable task IDs, full-scene camera aliases, decoded RGB hashes, camera proximity and perceptual similarity across **both servers and the local release**. Exact duplicates are excluded; similarity flags require review. Shared backgrounds mean this final check is still necessary even with disjoint foreground donors.

Report per host: raw frames, annotation-complete frames, quality-passing frames, local duplicate/review holds, reviewed candidates, original donors, target identities, generated morphologies and views per geometry. Only the final globally checked union counts toward **25,000 accepted samples by Saturday 2026-09-19 23:59 KST**. At handoff, neither remote server has started production; no measured rate supports a completion guarantee yet.
