# Wider native camera-view experiment

Opt-in static capture, not navigation, physics qualification or release approval.
The legacy 4cm base-offset sampler and all current capture defaults are unchanged.

The first broader campaign produced99 native images:20 strict automatic
annotation candidates,22 clarity holds,57 geometry/visibility exclusions.
Another generated seed19 plant yielded zero images because every proposed
4cm-neighbourhood pose overlapped scene geometry. This motivates testing a
larger actual-robot snapshot search; it is not evidence the new search works.

`pose.py` keeps the qualified source body heading, arms, torso and camera mount.
Only the base snapshot moves outward8/16/24cm and laterally-8/0/+8cm. Real head
joints aim the camera. The package floor alignment, head joint limits and the
collector's full native scene-overlap checks remain mandatory. Candidate poses
are not certified collision-free trajectories or reachable manipulation poses.

`plan.py` wraps an existing, fully source-bound multi-target plan and creates
nine candidate views/target. It validates immutable geometry/prerequisites and
records the separate policy and code hashes. All source cap, training approval
and physical execution flags remain false. Nine proposals do not mean nine
admitted training images, and do not reset the old source-target view cap.

`worker.py` is an explicit versioned collector entry point with the wider pose
and plan validator. It retains the complete original greenhouse, renderer,
materials,1696x816 native mounted-head RGB, native optical-Z, instance IDs,
static-frame guards and annotation checks. It supports unchanged reference56
and the separately labeled warm56_then8 trial. It is headless and has no
physical robot command API. It binds its own script hash in request/result.

Example, after creating/checking a plan with `plan.build(source_path)`:

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
& D:/isaac-sim-6.0.1/python.bat -m sim_data.native_capture_v2.worker --batch-plan data/sim_data/diagnostics/native_wider_seed19_20260915_v1/plan.json --output data/sim_data/diagnostics/native_wider_seed19_20260915_v1/new_capture --instance-backend fast --render-budget warm56_then8_trial
```

Use a new output, sufficient memory reserve and only one native renderer. A
successful process exit is insufficient: require result.json and absence of
failure.json. Actual eligible/strict counts and visual review determine whether
these images are useful. The planned native run is under
`data/sim_data/diagnostics/native_wider_seed19_20260915_v1`; do not claim it has
completed without its result. Previous99-frame results are under
`data/sim_data/collection_batches/native_diverse_multitarget_20260915_v2` and
replayed review receipts under `dataset_reviews/native_diverse_automatic_20260915_v1`.

Eight focused tests cover explicit bounds, opposite-side outward motion, rigid
frame/heading invariants, source-gate propagation, immutable plans and forged
approval rejection. Full suite before the worker promotion:1209 tests and136
subtests pass(v45,62.94s). Native validation is still pending at this checkpoint.
