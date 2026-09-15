# Partial-plant visibility shadow v1

Standalone CPU diagnostic only. Not imported by any native worker, annotation
module, capture chooser or inventory. No launch, skip, ranking, camera override,
synthetic native depth, relabel or training approval is provided.

## Public API and evidence boundary

```python
snapshot, build_metrics = load_partial_plant(
    generated_variant_directory, saved_plant_to_world,
    expected_bindings=completed_parent_plan["source_bindings"],
)
evidence = inspect_interval(snapshot, saved_calibration, component_id)
snapshot.finish()  # full uncached source reread/hash, also performed after load
```

The predictor receives geometry and native calibration only, never RGB, depth,
validity, identity masks, label decisions or review status. The diagnostic joins
saved native evidence AFTER prediction. It verifies existing native receipts and
saved replay bindings; it does not claim to rerun annotation or view RGB.

Portable pip USD 0.26.3 was used in a separate CPU process. Existing pure
`audit_manifest` and `assemble_plant` reconstruct the generated plant using the
saved transform. Only anonymous USD composition is authored; source layers are
never saved. Referenced USD layers must be source-bound and remain clean. Arrays
are immutable byte-backed copies; there is no global or long-term cache.

Only generated-plant geometry is covered. Surrounding greenhouse, robot and other
plants are NOT reconstructed. A supported opaque foreground blocker can produce
`predicted_blocked`; absence of one leaves the frame `unknown`, never visible.

## Explicit uncertainty

- Broad phase uses every generated mesh's actual vertex AABB, not capsule culling.
- Triangles are tested directly. Quads require a strict-interior hit on the SAME
  physical face under both diagonals; arbitrary n-gons, subdivision, single-sided
  meshes, animated transforms/topology, unsupported materials and empty surfaces
  cannot be asserted opaque blockers.
- Material support requires explicit static opacity one on UsdPreviewSurface.
  Connected/nonconstant opacity, displacement, unsupported render contexts,
  geometry/material subsets and nonopaque display opacity are unknown.
- Five rays per native pixel (center and quarter offsets) must all predict a
  blocker for that pixel. Disagreement/grazing edges remain unknown. This is NOT
  a continuous-visibility, full pixel-footprint, collision or photometric proof.
- The interval uses unchanged 10--20 mm native arc samples, including anatomical
  knots. Additional proximal diagnostics use the existing 27 samples over
  4--30 mm; the parent is allowed only below 8 mm. They do not change frame gates.
- Parent capsule gaps and fruit AABB candidate counts are diagnostic evidence,
  never alternate truth labels or camera objectives.

## Actual bounded run

Ignored runner: `data/sim_data/diagnostics/run_visibility_shadow_97_20260916_v1.py`.
Completed output: `data/sim_data/diagnostics/visibility_shadow_97_20260916_v2/report.json`.
Report SHA256:
`df83ca89d3a40516d39cfb0e6d5dd56dd398cd2be0a2c82094ee7e004ba8bf93`.

Exactly completed V4 jobs 1--6: 97 frames, 30 old strict, 20 holds, 47 excluded.
Immutable progress receipt `progress_000018.json` SHA256:
`289cf7a886e5a40f32175b7a11f0b8c6009592fc171faf164496e620e2b7a4ce`.
No running/later job was used. Source and stored sensor bytes received a final
uncached hash pass (7,257 source/evidence bindings), without array decoding.

Measured interval results: 29 predicted blocked, exactly the 29 old
`cut_interval_not_fully_native_visible` exclusions; 0/30 strict predicted blocked.
All 29 have at least one blocker identity overlapping the saved native evidence,
not necessarily complete identity equality. The other 68 frame results are
unknown. Additional proximal probes found a blocker in all eight old proximal
clarity exclusions, but these are NOT adopted frame decisions. No result solves
contrast, visible-parent-clearance or task-query clarity.

One run: wall 74.2372 s; process CPU 95.0 s; six geometry builds total 22.7987 s;
97 predictor calls total 18.3964 s, median 0.1643 s, p95 0.3781 s; final hash
5.7930 s. Whole-process sampled peak RSS 281,493,504 bytes (20 ms sampling), OS
lifetime peak working set 283,418,624 bytes. Per-frame evidence totals 11,218,860
bytes; report 1,937,107 bytes. These are measured costs, NOT capture savings or
20k-frame scaling estimates. Concurrent native work was not controlled.

All six plants had supported opaque materials; eight empty mesh prims across
jobs 2/5/6 were retained as bounded unknowns. Initial output v1 contains only the
partial job-1 run before empty topology handling was added; it is preserved and
has no completion report. v2 is the completed run.

## Tests and adoption boundary

27 synthetic CPU tests passed in 0.87 s, including opacity uncertainty, both-quad
diagonals/same-face requirement, grazing edges, empty topology, malformed topology,
parent allowance, camera rays and immutable arrays. No native process was used.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='D:/research/tomato-pi-policy/examples/greenhouse_sim'
python -B -m pytest examples/greenhouse_sim/sim_data/native_capture_v5/test_visibility_shadow.py -q -p no:cacheprovider --basetemp data/sim_data/diagnostics/visibility_shadow_cpu_tests_20260916_v2
```

This retrospective bounded population motivated the prototype and is not an
independent prospective qualification. Any broader USD feature support, scene
coverage, geometry objective, skip/ranking use or native integration needs
separate review. Matched-parallax native worker remains a different, deferred
task; full56 versus trial8 qualification remains open.
