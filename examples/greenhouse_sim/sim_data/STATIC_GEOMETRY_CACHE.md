# Exact static geometry cache (2026-09-15)

Static perception capture only. This is not motion planning or contact physics.

`static_geometry_cache.py` retains obstacle bounds and the existing triangle
refiner across unchanged-scene snapshots. Robot bounds and the full screen are
recomputed every call. USD object changes invalidate obstacles; robot-only
changes keep obstacles but invalidate an in-progress result. Plant substitution,
geometry, visibility, purpose, activation and ancestor transforms trigger rebuilds.
Render bookkeeping outside /World is ignored. Source-file hashes, native sensor
checks, floor alignment and capture guards remain the caller's responsibility.

No cached pass/fail result is returned. Closed caches and scene changes during
screening fail closed. Every cached result retains the reference screen's limits:
robot local boxes versus plant triangle surfaces, no path/self-collision/dynamics
or closed-plant-volume containment certification.

## Measured native evidence

`data/sim_data/diagnostics/native_geometry_cache_profile_20260915_v1/result.json`
and `data/sim_data/native_geometry_cache_profile_20260915_v1.log`.

- Full original greenhouse, robot head1696x816, native Isaac optical-Z, unchanged
  renderer/light/assets and56 requested render subframes.
- Original: cold13.6604 s, warm0.8639 s. Generated: cold13.5520 s, warm0.8609 s.
- Both cold/warm results exactly equal each corresponding full reference screen.
- Cache rebuilt after generated-plant substitution (2521 invalidating notices,
  only one new build); no stale original triangles reused.
- Approx15.8x faster repeated GEOMETRY SCREEN, not whole capture throughput.
- Diagnostic pair161.17 s, setup26.88 s. It deliberately runs extra reference
  checks and is NOT a faster production capture nor new training diversity.

Reproduce with a NEW output destination and an already qualified native pair plan:

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
$env:OPENBLAS_NUM_THREADS='1'
D:/isaac-sim-6.0.1/python.bat -m sim_data.native_generated_pair --plan PATH_TO_QUALIFIED_PLAN --output NEW_OUTPUT --profile-geometry-cache
```

Retain the established launch reserve/no competing Kit checks. The flag is
opt-in; ordinary pair capture still uses its full reference screen. Production
integration must retain one cache across multiple views of an unchanged plant,
rather than rebuilding at every frame. Native moving-robot multi-view validation
and render warm-up/convergence qualification remain outstanding.

13 cache tests cover exact decisions, changed robot poses, obstacle edits,
triangle edits, closed use, in-screen mutation and diagnostic equality checks.
Full sim-data suite:1107 tests +87 subtests passed in57.81 s (v36 log).
