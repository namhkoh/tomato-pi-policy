# Replicator collection audit — 18 September 2026

## Conclusion

The current pipeline already uses Isaac Sim Replicator. The installed extension is
`omni.replicator.core-1.13.27+110.1.1.wx64.r.cp312`.

The first performance change should keep the greenhouse, robot, camera, render
product and writer alive **across plant morphology changes**. Replacing the writer
or enabling asynchronous rendering alone does not solve the measured bottleneck.

## What already works

- [capture_pilot.py](capture_pilot.py:49) attaches RGB, optical-axis depth, camera
  parameters, reference time and instance-ID annotators.
- [native848_fully_labeled_worker_v3.py](native848_fully_labeled_worker_v3.py:120)
  creates one render product and writer before a whole batch; it already reuses
  them for all views in that batch.
- [native848_bulk_worker_v1.py](native848_bulk_worker_v1.py:76) requests one
  synchronized Replicator callback per frame.
- [native848_bulk_io_v1.py](native848_bulk_io_v1.py:29) already writes lossless
  arrays/images asynchronously with bounded queues.
- Current output is native 848x408 RGB, float32 optical-Z depth, validity, instance
  ownership and calibration. The camera remains attached to the authenticated
  robot embodiment.

## Measured costs from completed captures

| Batch | Frames | Scene setup | Initial warmup | Mean production render request |
| --- | ---: | ---: | ---: | ---: |
| TRAIN103 central | 50 | 82.9 s | 147.8 s | 13.18 s |
| TRAIN53 expanded | 48 | 78.1 s | 161.2 s | 12.61 s |
| TRAIN103 grid | 46 | 80.7 s | 157.7 s | 14.11 s |

First geometry-screen construction takes about 35–38 seconds, compared with
about 0.4 seconds for later poses. Lossless writing takes about 0.08 seconds per
frame; total queue waiting is around 0.0013 seconds per batch. Another 43–46
seconds in these capture manifests is not separately attributed.

These measurements are baseline observations, not a forecast of a new pipeline.
Keeping a session alive cannot by itself remove the 12–14 second production
requests. At the current rate, 10,000 raw requests alone require roughly 33–39
hours, before setup, rejected views, annotation or review.

## What Replicator can provide

NVIDIA's [scene-based SDG example](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)
demonstrates loading an environment, randomizing assets, creating camera render
products and writing annotated captures. We can use the same lifecycle with
our labeled plants.

The [capture API examples](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html)
support explicit capture steps and event-driven randomization. Their warning
about nonblocking capture matters here: returned data may describe a preceding
frame. Keep synchronous frame association until a queued implementation proves
that image, geometry, depth and camera metadata belong together. Direct Fabric
updates also bypass USD; our geometry checks currently read USD, so enabling
that option requires matching state validation.

The [I/O guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/io_guidelines.html)
describes asynchronous writing and queue/thread controls. Our measured writer is
already keeping up. Increasing queue depth does not remove a rendering bottleneck.

The [subframe guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html)
explains the speed/quality tradeoff. Benchmark smaller rendering budgets against
the current reference; do not assume thin petiole details survive a reduction.

## Implementation order

1. **Finish diverse generated geometry and annotation.** Different donors plus
   meaningful petiole/leaf variation; retain donor train/test lineage and reject
   copied or near-identical morphology. Whole-plant bend alone failed the visual
   novelty check. No bulk generation from that weak pair.
2. **Persistent morphology transition.** Keep the 143 labeled backgrounds,
   greenhouse, robot, mounted camera, product and writer. Change only the
   foreground, rebuild its exact component ownership and 9mm cut geometry, then
   establish a new immutable frame context before requesting a capture.
3. **Partition the geometry cache.** The current
   [cache](static_geometry_parts_cache_v1.py:42) invalidates all obstacle geometry
   for any nonrobot scene change. Retain authenticated static background geometry
   and rebuild the changed foreground; compare results with the full rebuild.
4. **Benchmark render budgets.** Compare reference and lower-subframe results at
   the same view. Check native depth/IDs, junction continuity, native support
   width, first-leaf clarity and full-image artifacts. Record raw and accepted
   throughput separately.
5. **Capture first, annotate in one offline pass.** Save synchronized arrays,
   exact morphology/source identity, camera, robot pose and full scene census.
   Run custom cut-point/ambiguity checks afterward in bounded chunks.
6. **Scale only after a small batch passes.** At most two views per genuinely
   distinct generated geometry group. Report original donor count separately.

Replicator does not supply the application-specific 9mm petiole cut rule, the
first attached-leaf relationship, robot reach or exactly-one-eligible-petiole
decision. Those remain custom geometry and image checks.

## Current implementation status

- Whole-plant bends were generated and compared; local junction changes were too weak for diversity approval.
- Controlled petiole generation now preserves the complete proximal30mm mesh and handles curved source centerlines. Actual seed53 geometry and fresh9mm correspondence replay passed.
- Generated foreground substitution works in an actual full144 CPU scene, preserving all143 background plants. A known source camera passed fresh robot clearance and9mm reach checks after substitution.
- `native848_controlled_capture_v1.py` completed a one-frame native Replicator diagnostic, and generated annotation completed. The frame is held for two occluded/depth-inconsistent proximal probes at29/30mm; accepted increment0. The processes now run end to end, but this is not the proposed persistent collection implementation and does not establish morphology diversity.
- The current capped subset remains26 images across13 original supervised plants. Current native/annotation results are recorded at the top of the root `dataset_vlm.md` handoff.
