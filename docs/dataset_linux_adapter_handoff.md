# Linux preview implementation handoff

Status on September 19: source localization and host-local reservations work. **The native Linux capture/annotation adapter remains to be implemented and tested on a server.** Source-lock verification is not a rendering qualification.

## Smallest preview

Capture two original-plant control views in one complete144 labeled greenhouse, before adding generated foreground swaps. Both controls are excluded from the delivered dataset. The preview needs no central coordinator; production uses the [offline ledger](dataset_independent_servers.md).

Suggested work division: thor3 implements the common Linux adapter; thor1 checks the same adapter on L40S and qualifies four distinct GPU workers. Use additive modules so the shared scientific source lock remains valid.

## Concrete adapter boundaries

| New Linux responsibility | Existing code to reuse |
| --- | --- |
| Owner/child identity, actual exit, per-GPU lock | Replace Windows process boundaries in `native_generated_reference/owner_v1.py` and `native848_bulk_sibling_gate_v5.py` |
| Fresh anonymous scene bootstrap and scene specification | Bootstrap logic from `collection_worker.capture_job`; call `native848_fully_labeled_scene_v1.populate_labeled` |
| Fresh mounted-camera proposals and actual pose | `native848_original_direct_worker_v2.apply_cached_pose`, robot FK, floor and full-scene collision checks, `BulkWorkspace` |
| Native aligned RGB-D and IDs | `native848_bulk_worker_v1.make_bulk_writer`, `request_once`, `native_freshness`; unchanged `FrameSink` |
| Linux capture authentication for annotation | Add a typed alternative to the Windows authority boundary in `native848_scene_sweep_raw_9mm_v1` |
| Labels and acceptance calculations | Reuse existing full-scene coverage, 9mm geometry, joint visibility, competing-petiole ambiguity, workspace and background functions unchanged |
| Linux production completion | Authenticate real Linux closure and saved outputs, then match actual identities against the offline reservation before local completion |

Paths in this table are under `examples/greenhouse_sim/sim_data`.

## Implementation sequence

1. Run `scripts/dataset_linux_preview.py` to rebuild the destination source plan. Its output includes an exact generated-geometry command, but start native runtime qualification with original plants.
2. Build a fresh scene specification from actual CPU population. Reuse the anonymous wrapper and variant construction in `collection_worker.capture_job`; do **not** call the whole function because it performs other smoke renders. Use `launch_sim_data.populate` with `NoRenderProgress` to measure slots, variants and foreground placement.
3. Populate the native stage once with unchanged `populate_labeled`. Keep all144 complete same-split plants, exact catalogues/transforms and strong references to returned `template_stages`. Record fresh local bindings and census.
4. Use two historical camera parameter sets only as proposals. Recompute FK, floor alignment, actual whole-robot scene collision and9mm workspace. Keep the robot USD/URDF at their expected repo-relative locations: the workspace checker independently opens `robot_model.DEFAULT_ASSET`.
5. Use one render product and writer. Initially retain seven warmup requests of eight subframes and eight subframes per captured frame. Save native848?408 RGB, float32 optical-Z depth metres, validity, IDs/mapping, calibration, pose and request/callback/freshness evidence.
6. Record actual Linux process identity: PID plus `/proc` start time and boot ID, executable, command line, ancestry, physical GPU UUID and observed exit. Do not assume `python.sh` has the Windows owner?cmd?Kit process hierarchy.
7. Authenticate those receipts through the additive Linux consumer, then run the unchanged full144 numerical evaluation. Preserve flat `observation_path`/`observation_sha256` fields and context bindings expected by the numerical consumers.
8. Compare the control results and qualify each server's GPU mapping. After successful closure and sensor/annotation checks, qualify production request preparation and offline completion for the new producer. Scale one?two?all host GPUs using disjoint requests and outputs.

Installed SimulationApp configuration uses `active_gpu`, `physics_gpu`, `multi_gpu=False`, and `max_gpu_count=1`. Do not use `physics_gpu_idx`. Verify the physical GPU in Kit logs and actual utilization; CUDA ordinals alone do not prove Vulkan rendering placement.

## Historical evidence cannot be repathed into a fresh capture

- `native848_pilot_plan_v1.raw_anchor` authenticates actual historical pilot samples; `validate_raw_cache` reconstructs that anchor exactly.
- `native848_fully_labeled_scene_v1.prepare_native_scene` reopens those captures and checks their lighting, variants, robot pose and calibration. The new Linux bootstrap must measure those fields itself.
- `native848_bulk_plan_v1.check_profile` requires historical qualification evidence. A new Linux preview profile can use the same numerical settings while honestly remaining unqualified until its native controls finish.
- `native848_scene_sweep_raw_capture_v1.native_command`, `native_main` and `owned_run` use Windows launch/process rules.
- Existing `annotate --kind sweep-raw` and completion consumers authenticate that Windows producer. New Linux authority needs a corresponding additive adapter; do not fabricate Windows receipts or relax the numerical acceptance rules.

## Report back after the preview

Provide the exact launch command, source/adapter hashes, actual physical GPU UUID, output path, two saved native RGB-D/ID frames, full144 census, native process exit receipt, annotation/parity results, and measured elapsed time. Failed or incomplete previews remain diagnostic; they never count toward25k.
