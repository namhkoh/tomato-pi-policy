# Dataset capture on four NVIDIA L40S GPUs

Current collective25k campaign: [machine-specific execution runbook](dataset_thor1_runbook.md) and [shared capture/annotation contract](dataset_multihost_runbook.md). Use the new host config and coordinator before production.

Prepared September 18, 2026. This guide extends the [two-RTX-PRO-6000 guide](dataset_dual_rtx_pro_6000.md) to a four-L40S server. The source collection uses Isaac Sim **6.0.1**. The shared asset bundle was published and independently downloaded, hash-checked and extracted successfully.

**Hardware is suitable; the four-worker launcher is not implemented or qualified yet.** The existing capture owner is Windows-specific and serial. Downloading the bundle supplies the assets and source code; Linux path/runtime adaptation and explicit GPU ownership are still required. The implementation checklist below is for the agent preparing the server.

## 1. Hardware and runtime

Each [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/) has 48 GB GDDR6 ECC memory, third-generation RT cores and an Ada Lovelace GPU. NVIDIA describes it for RTX rendering and Omniverse simulation. Its maximum board power is 350 W and it uses passive server cooling. Confirm the provider supplies four full GPUs with graphics access, rather than assuming a CUDA-only environment exposes rendering devices.

[Isaac Sim 6.0.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/requirements.html) list RTX 4080 as the minimum GPU and RTX PRO 6000 Blackwell as the ideal. This is not a GeForce-only model whitelist: L40S provides the relevant RTX graphics capabilities. Our greenhouse has not been benchmarked on an L40S, so hardware suitability does not establish its frame rate or accepted-image yield.

Install Isaac Sim separately. Start with the source build, 6.0.1, on a supported OS and driver. The documented x86-64 operating systems include Ubuntu 22.04/24.04 and Windows 11; containers are supported on Linux. Follow the [current NVIDIA driver requirements and known issues](https://docs.omniverse.nvidia.com/utilities/latest/common/technical-requirements.html) and run Isaac Sim's Compatibility Checker on the destination.

For planning, use approximately **32 or more CPU cores, 256 GB host RAM and fast local NVMe storage**, then adjust from measurements. These are engineering starting points, not vendor minimums or measured requirements for this scene. Budget host RAM as four worker peaks plus annotation, caches and OS headroom. Keep the checkout, expanded assets and active capture files on local storage; a network filesystem can bottleneck scene loading and small-file verification.

Four independent workers each have **48 GB per GPU**. The proposed configuration does not provide a single 192 GB GPU-memory allocation. L40S has no NVLink; this queue-based design does not require it.

## 2. Download the published scene bundle

- Private dataset repository: [namhokaist/tomato-greenhouse-sim-assets](https://huggingface.co/datasets/namhokaist/tomato-greenhouse-sim-assets).
- Pinned Hugging Face revision: `e9b73a1c2161ba04ebd8a103f7a056ac04a49efc`.
- Release directory: `releases/20260918_v1`.
- Archives: 742,620,660 bytes compressed; about 2.53 GB before shared-file deduplication.
- Verification: all 26 published payload files were downloaded and matched SHA-256; extraction verified 15,007 archive members representing 14,902 unique files.

With the Hugging Face CLI installed, run from a directory on the server's local storage. Authenticate using your own account credentials; never embed a token in scripts or Git.

```bash
hf auth login
hf download namhokaist/tomato-greenhouse-sim-assets \
  --repo-type dataset \
  --revision e9b73a1c2161ba04ebd8a103f7a056ac04a49efc \
  --local-dir ./greenhouse-assets

python ./greenhouse-assets/releases/20260918_v1/verify_extract.py \
  --output ./greenhouse-assets-extracted
```

The extraction directory must be new. The helper verifies whole archives, member hashes and paths before reporting success. The following files are then available:

```text
greenhouse-assets-extracted/workspace/data/sim_data/package_20260905/tomato_greenhouse_pack/
greenhouse-assets-extracted/workspace/data/sim_data/diagnostics/native848_fixed_varied_greenhouse_20260918_v1/environment.json
greenhouse-assets-extracted/workspace/data/sim_data/diagnostics/native848_fixed_scene_diverse_aisle131_prepare_20260918_v1/request.json
greenhouse-assets-extracted/workspace/data/greenhouse_sim/robots/rby1a_v1.2/
```

The bundle contains the greenhouse, plant/texture USDs, robot assets, current request verification dependencies and exact source snapshot `94aeddfe6ae1991c1cac9c48192feae20f6976ad`. That historical snapshot includes the original Blackwell guide; it predates this L40S document. Read the release README and `inventory/README.md` before adapting paths.

The reference request records a completed 131-proposal trial with one accepted image. It is supplied for provenance and implementation reference, not as a production queue to rerun toward the target count. Prepare fresh diverse poses.

## 3. Four-worker layout

Use one persistent headless Isaac Sim process per physical GPU, each holding a complete greenhouse scene and one native 848x408 render product. This is the proposed layout to benchmark:

| Slot | GPU assignment | Queue | Writes |
| --- | --- | --- | --- |
| worker 0 | First selected L40S UUID | Distinct camera subset A | Dedicated batch directory and log |
| worker 1 | Second selected L40S UUID | Distinct camera subset B | Dedicated batch directory and log |
| worker 2 | Third selected L40S UUID | Distinct camera subset C | Dedicated batch directory and log |
| worker 3 | Fourth selected L40S UUID | Distinct camera subset D | Dedicated batch directory and log |

One coordinator reserves candidate IDs, balances donor/target coverage and merges accepted results. CPU processes annotate only completed immutable batches. GPU capture of the next batch can overlap annotation of the previous batch. A single publisher performs global duplicate checks and updates delivery totals.

NVIDIA's [multi-GPU guidance](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/sim_performance_optimization_handbook.html#multi-gpu-support) relates rendering scaling to camera count and notes that additional GPUs do not reduce USD scene load time. Keeping scenes resident avoids repeated setup. Do not assume four cards produce four times the end-to-end accepted-image rate.

## 4. Required implementation changes

Apply the ownership, isolation and provenance requirements in [section 4 of the shared guide](dataset_dual_rtx_pro_6000.md#4-required-launcher-changes), with **four slots** instead of two:

1. Add an explicit renderer-device and physics-device selection to a new launcher/request contract. Map each selection to a physical GPU UUID/PCI address. Renderer, CUDA and `nvidia-smi` indices need not match.
2. Replace the single host-wide owner with one coordinator and one lease per GPU. Authenticate owned siblings during admission and closure. Retain failure evidence and release a lease only after its child exits.
3. Isolate output directories, writable runtime state and logs. Share assets read-only. Confirm shared shader/runtime-cache behavior during qualification. Retain task-scoped telemetry handling.
4. Implement Linux process ownership and replace the hardcoded Windows runtime path if using Linux. Map `workspace/` and `external_dataset/` explicitly. Regenerate destination-specific requests/profile bindings without modifying old receipts or bypassing verification.
5. Extend typed capture/annotation/export adapters to recognize the new producer and GPU contract. Existing consumers require exact source hashes; do not edit frozen production sources in place.

Configuration fragment for the **new** launcher, applied to the otherwise unchanged qualified capture configuration:

```python
config.update(
    active_gpu=renderer_index_for_this_slot,
    physics_gpu=cuda_index_for_this_slot,
    multi_gpu=False,
    max_gpu_count=1,
    limit_cpu_threads=threads_allocated_to_this_worker,
)
```

These are documented [SimulationApp configuration keys](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.simulation_app/docs/index.html), not existing CLI options in our capture script. Preserve the renderer, resolution, reset, freshness and 8-subframe settings for the first server benchmark. Allocate CPU threads across all four workers and annotation rather than giving every process all cores.

Confirm physical device assignment using the active GPU table in each Kit log and per-GPU utilization/memory. `CUDA_VISIBLE_DEVICES` alone does not choose Vulkan rendering devices; see NVIDIA's [Linux troubleshooting guidance](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html).

The release intentionally excludes eight installation/host files referenced by historical verification. They are listed in `inventory/excluded_host_bindings.json`. Fresh destination runtime/profile qualification is necessary. The bundle is not a complete replay archive of every historical experiment.

## 5. Preserve the dataset contract

- Native 848x408 RGB, aligned float32 optical-Z depth, validity and calibration.
- Exactly one eligible petiole per image, no query cue, and its cut point 9 mm from the main-stem attachment.
- Clear junction/proximal petiole/first-leaf context and actual robot-camera, FK, floor, collision and workspace checks.
- All 144 complete unpruned labeled plant instances, with vines visibly populating the background.
- Full-scene competing-answer checks, exact/near duplicate rejection, and actual visual review of each target/view group and flagged alternatives.
- Preserve donor split lineage and exclude `seed41_full/SubStem_38`. Report original donors, targets, generated morphologies and physical instances separately.

The current exporter also enforces at least 40% background-plant pixels and zero unknown pixels. Preserve this implementation gate during migration. Multiple copies of one plant, repeated poses, or tiny jitter do not constitute independent plant diversity. Source-mesh prechecks can prioritize candidates, but do not replace native visibility, annotation or review.

## 6. Qualification and scaling

1. Verify a small control batch on one L40S, using multiple known donor/target examples. Check RGB-D/ID correspondence, cut-point clarity, device assignment and actual process closure. Repeated controls do not increase the dataset count.
2. Run two workers with disjoint prospective queues. Check leases, isolated writes, cleanup and throughput under shared CPU/I/O load.
3. Run all four workers with about 32-64 distinct proposals each. Annotate and review the trial completely before committing to a long queue. The existing cached annotation wrapper accepts at most 512 frames per completed batch.
4. Measure scene startup/warmup, steady capture time, whole-run wall time, peak host RAM/VRAM, rejected proposals, raw frames, accepted images and donor/target distribution. Record annotation/review backlog too.
5. Continue bounded batches when the four-worker result demonstrates useful scaling. Resume failed work only from uncommitted candidate IDs in fresh outputs; never count partial directories as accepted captures.

Calculate a rendering-based estimate using measured per-worker yield:

```text
accepted_images_per_hour = sum(3600 * acceptance_fraction_i / seconds_per_frame_i)
remaining_capture_hours = remaining_accepted_images / accepted_images_per_hour
```

Add startup, camera preparation, annotation/review and packaging time to the delivery estimate. Final acceptance yield must come from new representative camera views, not only known controls. No L40S speedup or delivery deadline has been measured for this greenhouse.

## Handoff completion

The implementing agent should provide the exact server commands, four physical GPU identities, source/runtime hashes, focused ownership/GPU-selection tests, successful one/two/four-worker receipts and accepted-image throughput. Publish only after the unchanged quality checks and global duplicate audit pass.

This documentation was checked against the existing source interfaces, the verified Hugging Face release and NVIDIA documentation. No server provisioning, Linux port or L40S capture run was performed for this documentation commit.
