# Curved and relocated petiole variants

Status (2026-09-15): CPU geometry/serialization qualified for one real source
variant; native greenhouse capture is queued. Not a production training release,
new biological family, dynamics qualification, or claim of 20,000 images.

## Purpose and scope

The earlier similarity generator rotates/scales an existing subtree around its
original attachment. These changes alone do not create new independent targets.
This extension creates genuinely changed centerlines and relocates attachment
positions along the same parent, while preserving source mesh topology, leaf
detail, UVs, materials, texture bytes and the surrounding plant population.

Only intact, leaf-bearing sub-stems directly attached to main stems are eligible.
No fruit/truss/flower or main-stem geometry is changed. The implementation uses
the existing TRAIN-only morphology envelope, not held-out families. The original
plant generator is still unavailable; this is a bounded source-derived geometry
generator, not a replacement botanical growth model.

## Geometry and evidence

- `procedural_petiole_geometry.py`: metric curved centerline, taper, continuous
  parallel-transport frames, intrinsic similarity-invariant shape descriptor;
  also a capped tube diagnostic mesh with winding/manifold tests. Detailed
  source meshes, not the diagnostic tube, are used for the real variant.
- `procedural_petiole_warp.py`: one shared curve-coordinate deformation field
  for the selected petiole and its leaves. Normals use the inverse-transpose
  numerical Jacobian. Folded or ill-conditioned sampled derivatives are rejected.
- `procedural_petiole_usd.py`: create-only output, actual parent-triangle surface
  attachment, copied/deformed detailed meshes, recalculated child attachments,
  removed inherited physics, regenerated 10 mm nominal /10?20 mm cut interval.
  The new cut interval must pass 24 radial rays against the actual mesh.
- `procedural_petiole_catalogue.py`: read-only deterministic recipe replay,
  frozen family/source/texture/code hashes, saved point/normal/extent checks,
  unchanged topology/UV/material channels, recalculated labels and attachment
  audit. Inspection assembly uses an anonymous stage only.
- Existing `plant_variant_catalogue` and `generated_capture` dispatch by explicit
  generator version. Curved plans bind the new implementation, retain original
  target view caps, and use the same native RGB/Z, camera, scene and robot checks.
  Legacy capture/annotation implementations are unchanged.

Engineering bounds (not measured biological distributions): attachment shift
14?35 mm along an eligible parent segment and at least10 mm actual displacement;
length0.85?1.15? donor, radius0.9?1.1? donor, within the TRAIN envelope; heading
and curvature perturbations bounded in the recipe. Every recipe is stored.

These local checks do **not** establish global self-intersection freedom,
botanical plausibility, protected-structure clearance, graspability or physical
cutting. Native visibility/clarity and robot/environment clearance remain gates.
Failure evidence is retained; no collision or memory guard is weakened.

## Reproduce the inspected variant (new destination required)

Use the ordinary project Python with `PYTHONPATH=examples;examples/greenhouse_sim`
and `OPENBLAS_NUM_THREADS=1`. No Isaac app is started by these two commands:

```powershell
python -m sim_data.procedural_petiole_usd --source-plan data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json --family seed101_full --targets SubStem_41 --seed 240901 --output data/sim_data/generated_plants/NEW_ROOT/seed101_full_cr_240901
python -m sim_data.procedural_petiole_catalogue --variant data/sim_data/generated_plants/NEW_ROOT/seed101_full_cr_240901 --source-plan data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json --output data/sim_data/diagnostics/NEW_catalogue.json
```

Observed real-asset evidence:

- `data/sim_data/generated_plants/curved_relocated_smoke_20260915_v1/seed101_full_cr_240901/qualification.json`
- `data/sim_data/diagnostics/curved_relocated_smoke_20260915_catalogue_v1.json`
- All404 donor components retained; one changed target passed catalogue checks,
  with no attachment warnings on its subtree. All source assets unchanged.
- New geometry tests:33 tests +10 subtests. Full suite after fixture correction:
  1057 tests +87 subtests passed in67.68s (`clear_regression_20260915_v31.log`).

Native first-pair plan:
`data/sim_data/diagnostics/curved_native_20260915_v1/plan.json`.
Queue log: `data/sim_data/curved_native_queue_20260915_v1.log`.
Native log: `data/sim_data/curved_native_20260915_v1.log`.
Queue launch16:42:29 KST, waits at most2h for20 GiB commit headroom/no other Kit,
then one capture attempt; the worker retains its own resource/geometry guards.
Check actual result/exit/review receipts: a queued job is not a successful image.

## Native validation checkpoint, 2026-09-15

The seed101/SubStem41 pair completed native capture (16:47:37-16:50:10 KST).
`diagnostics/curved_native_20260915_v1/annotations_v2` contains the successful
annotations; the first failed annotation attempt is retained. Curved anatomical
knots required matching the capture's piecewise <=1 mm interval sampling instead
of assuming exactly eleven points. Missing knots, stale geometry and occlusion
are rejected; straight-target labels remain exactly compatible.

Both clean native frames were actually visually inspected (overview plus native
pixel crop). Generated target accepted for annotation pilot; original control is
a repeated reference, not new diversity. Receipt:
`dataset_reviews/native_curve_requalification_20260915_v1/assistant_visual_partial.json`.
Paths here are relative to `data/sim_data`. Camera1696x816, native Isaac optical-Z,
75 gutters/3 populated gutters/142 backdrop instances and detailed assets retained.

Second donor seed19/SubStem42 also captured successfully (17:22:58-17:25:57 KST):
`diagnostics/curved_diversity_native_20260915_v1/case_002`. Both frames passed
automatic annotation checks, still pending actual visual assessment at this
checkpoint. Two other recipes (seed17/SubStem44 and seed103/SubStem43) were held
because the continuous field folded leaf meshes. Their failure evidence remains;
the Jacobian gate was not weakened. This motivates rigid leaf-blade transport.

Measured second-pair timings: setup28.26 s, pair137.90 s; per frame geometry
screen14.28-14.30 s, six native warm-up callbacks29.60-33.78 s, final native
callback3.90-4.39 s, artifact writing0.50 s. Observational instrumentation only:
56 requested render subframes and all gates unchanged. This is not a measured
speedup or a production-throughput result. Amortization/convergence work remains.

## Admission and scaling still required

The new shape descriptor excludes seed/name changes, rigid placement and uniform
scale as independent novelty. A difference from one donor is not global novelty.
Every output therefore still has `shape_novelty_pending=true`,
`independent_target_novelty_approved=false`, and the original source-target cap.
Do not reset the existing12-view budget by renaming a curved variant.

Before production scaling: qualify several native/visually reviewed examples;
implement a globally checked near-duplicate/geometry-admission inventory with an
explicit versioned cap policy; amortize full-scene setup across views; qualify
lossless native sidecar storage/export and Qwen3-VL-8B preprocessing; validate
held-out family separation and package only genuinely accepted training records.
Repeated original controls, crops and renderer noise are not extra training
target diversity. No model download, training, old review or split mutation.
