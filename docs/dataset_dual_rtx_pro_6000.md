# Dataset capture on two RTX PRO 6000 Blackwell GPUs

Prepared September 18, 2026. Repository baseline: `a8baa2f` on `koh-dev/sim-data`. Current local runtime: Isaac Sim **6.0.1** on Windows. This guide covers capture and dataset preparation; training is separate.

**Recommended layout:** two persistent Isaac Sim processes, each assigned one physical GPU and a disjoint camera queue. Each process loads the complete labeled greenhouse once per batch. CPU processes annotate completed batches; one coordinator reviews, deduplicates and publishes accepted samples.

**Implementation status:** this is an execution guide, not a working dual-GPU launcher. The checked-in capture owner serializes native runs, selects the default renderer GPU and uses Windows process admission. The GPU assignment and ownership changes in section 4 must be implemented and qualified before concurrent capture. Do not simply start the current command twice.

## 1. Hardware choice and the RTX 4080 minimum

[NVIDIA's Isaac Sim 6.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html) list the RTX 4080 as the minimum GPU, 16 GB as minimum VRAM, and RTX PRO 6000 Blackwell as the ideal GPU. Windows 11 and Ubuntu 22.04/24.04 are listed; Isaac Sim containers are supported on Linux.

The RTX 3090 falls below that published GPU baseline even though it has 24 GB and RT cores. NVIDIA's [RTX renderer feature table](https://docs.omniverse.nvidia.com/utilities/latest/common/technical-requirements.html) includes Ampere RTX features. This supports possible operation, not a guarantee that Isaac Sim 6.0 or this greenhouse workload will work well on a 3090. Our scene has not been qualified on it. Use the Blackwells for this deadline; do not budget around an untested 3090 configuration.

Confirm the exact RTX PRO 6000 **Blackwell** edition. The older RTX 6000 Ada is a different product. The [Blackwell family specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/) distinguish:

| Edition | Memory per GPU | Power per GPU | Deployment |
| --- | ---: | ---: | --- |
| Server | 96 GB ECC | 400-600 W | Passive cooling in a compatible server |
| Workstation | 96 GB ECC | 600 W | Workstation chassis with suitable airflow |
| Max-Q Workstation | 96 GB ECC | 300 W | Dense multi-GPU workstation configurations |

Power limits and edition affect performance. Record them in benchmarks. Two workers each have 96 GB available on their assigned card; this arrangement does not pool them into one 192 GB allocation.

For host planning, provision approximately **32 CPU cores, 256 GB RAM and fast local NVMe with 1 TB free**, then size from measured scene, annotation and output peaks. These are conservative project planning recommendations, not NVIDIA minima or measured requirements. Two complete scene copies and concurrent CPU annotation consume host memory too. Retain source assets and existing release components when estimating storage.

## 2. Stage the actual project and assets

1. Record the checkout commit, Isaac Sim build, OS, driver, GPU UUIDs, PCI addresses, edition and power limits. Run the Isaac Sim Compatibility Checker on the destination.
2. Use a driver supported by the chosen Isaac Sim/Kit build. Consult the [current driver and known-issue table](https://docs.omniverse.nvidia.com/utilities/latest/common/technical-requirements.html), including Blackwell and Windows/Vulkan issues. A newer driver is not automatically validated for every Kit build.
3. Copy the required project assets and their transitive dependencies, including USD layers, textures, plant metadata, robot assets, source catalogues, pinned profiles and request inputs. A Git checkout alone is insufficient: many assets and collection receipts live in ignored `data/` folders or outside the repository.
4. Verify copied bytes against the source hashes. Keep completed source files immutable. Create new requests and new output directories for the destination; preserve original receipts.

The lowest-porting-effort route is Windows 11 with the current directory layout:

```text
D:\research\tomato-pi-policy              repository and scene inputs
D:\isaac-sim-6.0.1                       native runtime
C:\Users\USER\tomato-vlm-data-20260917   external dataset components
```

These paths appear in source bindings and metadata. A different username, drive layout or Linux host requires an explicit relocation manifest and fresh destination-side bindings. Do not globally replace paths in completed releases or bypass hash verification. Linux is supported by Isaac Sim, but this repository's current owner and process-identity checks need a Linux implementation before production use.

Inventory command (PowerShell or a Linux shell with `nvidia-smi` on PATH):

```text
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,power.limit --format=csv
```

### Inputs and pipeline entry points

All module names below are under `examples/greenhouse_sim/sim_data/`.

| Component | Existing source or input |
| --- | --- |
| Fixed greenhouse description | `data/sim_data/diagnostics/native848_fixed_varied_greenhouse_20260918_v1/environment.json` |
| Capture from numeric pose proposals | `native848_scene_sweep_raw_capture_v1.py` |
| Capture from prepared evaluated views | `native848_scene_sweep_capture_v1.py` |
| Offline annotation with bounded hash reuse | `native848_scene_sweep_cached_annotation_v1.py` |
| Reviewed scene-sweep export | `native848_scene_sweep_dataset_v1.py` |
| Verified delivery bundle | `native848_delivery_bundle_v3.py` |
| Latest published delivery pointer | `data/sim_data/training_exports/LATEST_DATASET_VLM.json` |

The environment description points into a larger scene/request dependency graph; it is not a standalone scene package. The fixed scene contains 144 complete labeled plants and 16 TRAIN donor shapes. Copies of those plants do not count as new independent donors.

## 3. Split the work by GPU

```text
Source geometry + diverse candidate cameras
                 |
     CPU visibility / ambiguity prefilter
                 |
       coordinator: disjoint queues
             /                 \
    GPU A: capture A      GPU B: capture B
    full scene copy      full scene copy
             \                 /
      completed immutable capture batches
                 |
       CPU annotation -> visual review
                 |
      global deduplication -> one publisher
```

Use one renderer per GPU initially. NVIDIA's [multi-GPU guidance](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/sim_performance_optimization_handbook.html#multi-gpu-support) ties scaling to camera count and notes that extra GPUs do not reduce USD scene load time. Independent workers are our proposed design for many 848x408 views; their throughput still needs measurement.

Balance donor and target identities across the queues. Reserve each candidate ID once and check prior captures for exact and near camera repeats. Include materially different angles and targets; changing a filename, plant instance or tiny camera jitter is insufficient diversity. Preserve donor TRAIN/validation/TEST lineage, and exclude `seed41_full/SubStem_38`.

Apply source-mesh visibility and ambiguity checks before rendering to reduce wasted work. They prioritize proposals only: unresolved candidates and unsuccessful IK remain held. Native scene/collision, full-scene annotation and actual image review remain the acceptance authority.

## 4. Required launcher changes

Implement these additively. Existing capture and loader bytes are pinned by published data.

1. **Explicit GPU binding.** Add separate renderer and physics device indices to a new request/launcher contract. Resolve them to physical GPU identity; do not assume renderer, CUDA and `nvidia-smi` indices match.
2. **Two owned slots.** Replace host-wide serial admission with a coordinator and one lease per GPU UUID. Authenticate both children, allow only the owned sibling during admission and closure, and prevent duplicate GPU leases. The current `owner_lock()`, `windows_worker_admission(None)` and `not native_processes()` closure check cannot admit two simultaneous workers.
3. **Isolated writes.** Give each slot separate requests, outputs, logs and writable runtime state. Share source assets read-only. Prewarm runtime/shader caches before the concurrent benchmark and verify any shared cache is safe for concurrent use. Keep the existing task-scoped telemetry handling; earlier parallel attempts failed when a shared helper was treated as a per-worker process.
4. **Explicit new provenance.** Pin the new producer and its GPU contract in requests, capture receipts and an additive annotation/export adapter. Existing consumers compare exact producer hashes and process ownership. They must explicitly understand the new producer; changing a source file and reusing an old request is invalid.
5. **Owned shutdown and recovery.** Wait for actual child exit and completed capture receipts. Retain partial outputs as diagnostics. Resume only uncommitted candidate IDs in a fresh batch; do not infer completion from file counts. Release each GPU lease only after its owned processes close.

The following is a **configuration fragment for that new launcher**, not an existing command-line option or standalone capture program. Preserve the rest of the qualified capture profile:

```python
# renderer_index and cuda_index must map to the intended physical GPU.
config = dict(
    headless=True,
    width=848,
    height=408,
    renderer="RaytracedLighting",
    sync_loads=False,
    disable_viewport_updates=True,
    multi_gpu=False,
    active_gpu=renderer_index,
    physics_gpu=cuda_index,
    max_gpu_count=1,
    limit_cpu_threads=threads_per_worker,
    extra_args=["--/app/settings/persistent=false"],
)
# Apply existing telemetry preparation, then construct SimulationApp(config).
```

These keys exist in the installed 6.0.1 `simulation_app.py` and the [SimulationApp API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.simulation_app/docs/index.html). Allocate CPU threads across both workers and annotation rather than giving every process the full host.

`CUDA_VISIBLE_DEVICES` alone is not renderer isolation: NVIDIA states that it does not select Vulkan rendering devices in its [Linux troubleshooting guide](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html). Verify the active renderer table in Kit logs against UUID/PCI identity and per-GPU memory/utilization. In containers, confirm graphics device exposure as well as CUDA access.

## 5. Preserve the image and answer contract

Every accepted sample must retain:

- Native **848x408 RGB**, aligned float32 optical-Z depth, validity and sensor calibration.
- Exactly **one eligible petiole**, without a query cue, and one cut point **9 mm from its main-stem attachment**.
- Clear junction, proximal petiole and first-leaf context; robot-mounted camera/FK, floor, collision and workspace checks.
- All 144 unpruned labeled plants, with vines visibly populating the background.
- Complete competing-petiole checks, duplicate rejection, and visual review of each target/view group and flagged alternatives.

The current exporter additionally requires at least 40% background-plant pixels and zero unknown pixels. That is a current implementation gate; the user requested visibly populated backgrounds rather than that numerical threshold. Keep it for this migration and record its effect on yield.

Retain the qualified 8-subframe production profile, renderer reset and RGB/depth/ID freshness checks for the initial benchmark. Earlier 1/2/4-subframe and FXAA trials did not establish acceptable replacement quality. Experimental asynchronous rendering or changed rendering settings require separate alignment and visual qualification.

## 6. Benchmark before starting a long queue

1. Capture a small **non-release control batch** on GPU A, then GPU B. Use known controls spanning several donor/target identities, with the same calibration/profile. Check image quality, depth/ID agreement, actual device assignment and process closure. Repeated controls never count as new samples.
2. Run two disjoint prospective queues concurrently, initially about 32-64 candidates each. Keep one loaded greenhouse and one render product per worker throughout each batch. The current annotation wrapper accepts completed batches of at most 512 frames.
3. Annotate, review and deduplicate the complete trial. Record proposal holds, raw frames and accepted images separately; benchmark accepted throughput using representative new cameras rather than only known passing controls.
4. Measure startup/warmup, render seconds/frame, whole-run wall time, annotation/review time, peak host RAM/VRAM, disk use and images per original donor/target. Confirm the two workers improve total useful throughput without exhausting CPU or storage.
5. Continue bounded batches sized from those measurements. Close a batch before annotation; current source verification expects an immutable completed capture. Let the next GPU batch overlap CPU processing of the previous one.

Existing serial Windows interface, shown for migration reference only:

```powershell
$env:PYTHONPATH = 'D:\research\tomato-pi-policy\examples;D:\research\tomato-pi-policy\examples\greenhouse_sim'
& 'C:\Users\USER\miniconda3\python.exe' -B -u -m sim_data.native848_scene_sweep_raw_capture_v1 --request '<prepared-request.json>' --request-sha256 '<request-sha256>' --output '<fresh-owner-output>'
```

Substitute a prepared, hash-verified request and a fresh output whose parent exists. This invokes the owner, which launches `D:\isaac-sim-6.0.1\python.bat`; `--capture` is its internal child mode. There is currently **no `--gpu` argument**. Use the new qualified owner for dual-GPU operation.

For a completed **existing v1** raw capture, the supported CPU annotation command is:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
& 'C:\Users\USER\miniconda3\python.exe' -B -u -m sim_data.native848_scene_sweep_cached_annotation_v1 --mode raw --capture '<completed-owner-output>\capture' --output '<fresh-independent-annotation-output>'
```

The output must be outside the capture owner's directory. Use `--mode ordinary` for ordinary scene sweeps. The new GPU-aware producer needs the explicit consumer extension described in section 4 before its captures can enter this path. Annotation success alone does not approve an image.

## 7. Publish once, after review

Keep one coordinator responsible for dataset totals and the latest pointer. After completed capture, annotation and actual review, use the scene-sweep exporter's `materialize(evaluation_pin, selected, review_pin, output, authorization_pin=...)` interface. Its authorization is a concrete root-generated execution receipt with exact source/review pins; do not fabricate review or process-completion evidence.

Assemble accepted components using the verified delivery bundle, load every model input, and check duplicates across **both workers and prior releases**. Publish only after these checks. Retain the absolute source component folders required by the reference bundle. Model inputs exclude target IDs, renderer masks, private annotations and ground-truth crops.

Report accepted images, original donors, source targets, generated morphologies and physical plant instances separately, including views per donor/target. Copies of a donor do not make independent train/test plants. Stop at the requested accepted total, not the raw frame count.

## 8. Calculate the ETA from measured yield

At this guide's baseline, the published pointer contains **491 accepted images**; 9,509 remain toward 10,000. Re-read the pointer before scheduling.

```text
accepted/hour = sum(3600 / seconds_per_frame_per_worker) * acceptance_fraction
capture_hours = remaining_accepted / accepted_per_hour
```

Include startup, rejected-pose preparation, annotation/review backlog and final packaging in the delivery ETA. A lower acceptance fraction can dominate the hardware gain.

Illustrative arithmetic for two equal workers, assuming 80% acceptance:

| Seconds per raw frame per GPU | Accepted/hour | Capture time for 9,509 |
| ---: | ---: | ---: |
| 12 | 480 | 19.8 hours |
| 4 | 1,440 | 6.6 hours |
| 2.5 | 2,304 | 4.1 hours |
| 2 | 2,880 | 3.3 hours |

These are **not RTX PRO 6000 benchmarks**. Recent local batches yielded 3/55 and 1/17 accepted frames. The source-mesh prefilter has retrospective evidence on 17 saved frames; prospective acceptance remains unmeasured. A deadline estimate requires a new-camera trial on the actual destination hardware.

## Completion evidence for the implementing agent

- Record hardware/runtime identity and the immutable source manifest.
- Commit the additive launcher, ownership and typed-consumer changes with focused tests for GPU assignment, sibling ownership, duplicate scheduling and partial-failure closure.
- Save actual single-GPU and dual-GPU trial receipts, whole-process exit codes and quality-review evidence.
- Report accepted images/hour and donor/target distribution; retain all failed or held trial evidence.
- Update [the collection plan](../dataset_vlm.md) with measured results and the exact commands that now work on the destination.

Documentation validation for this commit: inspected repository launch/annotation/export interfaces, installed 6.0.1 configuration keys and the linked NVIDIA documentation. No dual-Blackwell machine was accessed or benchmarked in preparing this guide.
