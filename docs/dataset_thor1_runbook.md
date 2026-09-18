# thor1 — four L40S collection runbook

**Current execution mode: [independent server collection](dataset_independent_servers.md). No connection to the Windows PC or shared coordinator is required.** The independent guide supersedes coordinator setup/reservation steps below; all image, annotation and Linux qualification requirements still apply.

**Shared goal: at least 25,000 accepted distinct samples collectively by Saturday 2026-09-19, 23:59 KST.** Read the [shared contract](dataset_multihost_runbook.md); its image, background, annotation, ownership and counting rules are mandatory.

## Exact assignment

- Config: [`configs/dataset_capture/thor1.json`](../configs/dataset_capture/thor1.json).
- Machine ID: `thor1`; GPU worker IDs: `0,1,2,3`, one L40S per worker.
- Foreground donors: **seed7, seed11, seed43, seed47, seed73, seed101** (`seedN_full` identifiers).
- New generator seed range: **2,000,000–2,999,999**.
- Output prefix: `captures/diverse9mm_20260918_multihost_v1/thor1/worker_<ID>/`.
- Original donor descendants remain TRAIN. Use the complete 144-plant labeled background, including neighboring vines.

The user reported installation in progress on thor1; no production request/output existed at handoff. Complete Isaac Sim 6.0.1 installation and its compatibility check first. Use the [L40S hardware guide](dataset_four_l40s.md) for GPU/runtime details. The native Linux owner/GPU adapter still needs qualification; the old Windows owner is not runnable as a Linux collection command.

## Execute in this order

1. Obtain the same new code revision as thor3 and the pinned HF assets. Create destination-specific source-plan/asset/runtime bindings. Keep the frozen numerical modules unchanged.
2. Set the config and verify the common source lock:

   ```bash
   export HOST_CONFIG=configs/dataset_capture/thor1.json
   export PYTHONPATH="$PWD/examples:$PWD/examples/greenhouse_sim"
   export OPENBLAS_NUM_THREADS=1
   python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" verify
   ```

3. Prepare new full-plant geometry. This executable example uses a previously supported donor/target pair; use the actual freshly authenticated Linux source-plan path/hash:

   ```bash
   python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" generate \
     --source-plan "$LOCALIZED_SOURCE_PLAN" --source-plan-sha256 "$SOURCE_PLAN_SHA256" \
     --family seed43_full --primary-target SubStem_42 \
     --controlled-targets SubStem_42 SubStem_43 --seed 2000000 \
     --output data/sim_data/generated_plants/thor1_seed43_2000000
   ```

   Preserve its qualification and `current9mm_evidence.json`. A failure is a held proposal, not an instruction to relax checks. Generated geometry must be installed as a complete plant and its full-scene census rebuilt before capture. Rotate work among assigned donors and eligible target identities; do not spend the whole campaign on 43/42.

4. Initialize one **host-local offline ledger** for this server using the [independent collection guide](dataset_independent_servers.md#3-initialize-a-host-local-ledger). No shared URL/token or SSH connection is needed. All GPUs on this host share this ledger; do not copy the live local5090 database.
5. Reserve fresh requests through `scripts/dataset_offline_shard.py reserve-raw` or `reserve-generated`, using this host's ledger and worker IDs. Follow the exact commands in the independent guide. Keep the returned task/claim IDs, use fresh outputs, and launch only the assigned worker when `capture_request_eligible` is true.
6. Qualify two preview images on GPU 0 with the server's Linux Replicator adapter. Save 848×408 RGB, float32 optical-Z depth, validity, instance IDs/mapping, calibration, robot pose, complete 144 census and process/GPU receipts. Compare annotation with the shared controls. Preview controls do not enter the dataset count.
7. Once the adapter passes, run two then four persistent workers with disjoint reserved requests. Confirm GPU UUIDs in Kit logs and utilization. Keep the 8-subframe profile initially. There is no requirement to match CUDA and Vulkan ordinal numbers.
8. Annotate completed batches through the common locked annotation entry point, record local duplicate checks and mark global merge pending, perform target/view-group visual review, and send reviewed exports to the single publisher. Report accepted counts per donor, target and generated morphology each hour.

**Before bulk launch, the thor1 agent must record:** the actual Linux launch command, four physical GPU UUIDs, common code/toolchain hash, local offline ledger path, successful owner closure and RGB-D/annotation parity receipts, exact output path, and measured accepted samples/hour. These fields cannot be filled from hardware specifications or a successful Isaac installation alone.
