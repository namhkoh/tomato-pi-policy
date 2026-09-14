# Clear-cut-point scale-up:20,000 training images

## Confirmed requirement and current boundary

The user confirmed on2026-09-15 that the minimum is **20,000 accepted TRAINING
RGB images**, with validation and test sets additional. A working budget of
2,000 validation plus2,000 test images is a proposal, not a user-specified count.
Crops, RGB-D sidecars, overlays, repeated render noise and alternate prompts do
not count as additional unique training images.

The419-image checkpoint contains304 training,42 validation and73 test images,
all individually assistant-reviewed, and is still a draft. No20k release or
higher-resolution native qualification is claimed. Existing qualification
profiles, frozen splits, decisions and capture jobs are unchanged.
`clear_cutpoint_v1` remains the small qualification experiment; passing its
700-image minimum would NOT fulfill this new20k-training request.

## Recorded capacity, not visibility-qualified capacity

Source:`data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json`.
Reproducible report:`data/sim_data/diagnostics/clear_20k_capacity_20260915_v2.json`.
The read-only `clear_scale_capacity` module cross-checks target identities and
recorded counts. It does not rerun USD geometry or native visibility checks.

| Split | Source families | Currently scheduled targets | All recorded geometric candidates | Current schedule maximum images | All-candidate maximum images |
|---|---:|---:|---:|---:|---:|
| Training |16|198|584|2,376|7,008|
| Validation |4|50|144|600|1,728|
| Test |4|49|142|588|1,704|
| Total |24|297|870|3,564|10,440|

These are optimistic ceilings at the unchanged12-view-per-target cap. The
870 candidates are not all within the current height band, reachable, visible
or anatomically accepted. More proposal shards or higher-resolution pixels do
not create new source targets.

20,000 training images therefore require **at least1,667 distinct training
targets** at12 accepted views each. That is1,469 more than the active training
schedule and1,083 more than all recorded training-family geometry candidates.
A proposed20k/2k/2k budget requires at least2,001 targets across the splits.
Actual need is higher when some targets yield fewer accepted views.

## Can we build our own generator?

Yes: a new **component-based parametric generator** is feasible in principle.
It would not recreate the original algorithm or be validated tomato biology.
The package records `generator=tomato_stem_generator`, version1.1.0, seeds and
some physical constants, but not its growth rules or full morphology configuration.
The README explicitly states that generator source is absent.

Available foundations:

- Original USD meshes and materials for main stems, petioles, leaves, trusses,
  fruit and flowers.
- Component parent graph, attachment positions, axes, centerlines/capsule radii,
  semantic types and intact/deleafed state.
- `audit.py` for structural/provenance checks, and `cut_regions.py` for the
  prototype10mm nominal /10?20mm arc-length cut rule.
- Existing `candidate_branches.py` demonstrates subtree reuse, but supports only
  three specified donor placements on each of two plants. It is a visual-preview
  recipe, not a general, independently validated plant generator.

### Proposed design

1. **Deterministic morphology configuration.** Seed, stem curvature/taper,
   node spacing, petiole attachment direction, length/radius/curvature, leaf
   arrangement, and fruit/truss placement. Start with bounded distributions
   measured from TRAINING assets; do not fit generator settings on held-out plants.
   Do not treat existing stiffness constants as calibrated tissue properties.

2. **Generate genuinely varied local junction geometry.** Construct stem and
   petiole centerlines and meshes, then attach leaf/truss subtrees with explicit
   local frames. Reuse original leaf/material detail initially. New node placement,
   petiole angle/curvature and parent geometry should vary the actual cut cue,
   rather than merely translating identical complete plants.

3. **One geometry source for rendering and annotation.** Regenerate mesh geometry,
   centerlines, radii, normals, bounds, parent attachments and semantic identities
   consistently. Preserve a real attached proximal petiole and leaf-bearing
   detachable subtree. Compute nominal10mm and10?20mm supervision on the NEW
   curve; never reuse donor pixel labels. Validate the mesh near the junction,
   not only a capsule proxy. The current audit accepts only translation transforms:
   either bake new geometry/rotation into copied component-local assets and metadata
   or explicitly version/validate every transform consumer. A USD-only rotation
   is not sufficient.

4. **Lineage-safe variants and splits.** Preserve all24 original family reservations.
   Do not feed an enlarged family list through the existing ranking function and
   silently reassign original plants. Record generator version, seed, configuration
   hash, source mesh/material hashes, donor components and parent lineage. Keep
   source-derived variants and donor libraries within their assigned split.
   Do not move a copied training petiole into test under a new name.

5. **Separate counts for layout, anatomy and source templates.** A new seed is not
   automatically an independent biological/source family. Report generated layouts,
   canonical local junction geometries and original donor families separately.
   Collapse exact and near-identical target geometries before applying the view cap;
   renaming a copied subtree must not evade12-view limits. Thresholds for meaningful
   geometry novelty need to be specified and tested during the pilot.

6. **Native greenhouse capture.** Integrate generated manifests via an explicit
   new catalogue/collection schema, leaving original assets untouched. Use the real
   mounted head camera, full greenhouse and native Isaac RGB/optical-Z/masks.
   Higher-resolution capture/export qualification remains a prerequisite. Reject
   invisible, ambiguous, under-resolved or broken junctions; rendering does not
   imply acceptance.

7. **Keep the first task clear-only.** The model receives RGB/full-frame plus a
   query-centred crop and returns the target-conditioned2D cut point. RGB-D support,
   hidden-coordinate reasoning, action trajectories and physics calibration remain
   separate experiments. Do not pad this20k clear-task count with occlusion tasks.

## Smallest useful generator pilot

Proposed first acceptance packet:20 generated layouts and100?200 native views,
not a mass collection job. This is a proposed engineering milestone, not a yield
claim or an authorization to modify original source geometry.

- Generate new copied assets only; deterministic seed/config reproducibility.
- Every output passes graph, identity, units, parent attachment, mesh/centerline
  consistency and cut-rule checks.
- New geometry follows training-derived parameter bounds; protected organs and
  parent stem remain distinguishable. Intended attachments are allowed; reject
  unintended self-intersections or floating components near the target.
- Individually inspect each pilot full-frame RGB, proximal junction and native mask/
  depth evidence. Visually compare generated plants with originals in the greenhouse.
- Validate donor/split ancestry and target-morphology deduplication.
- No copied human/assistant approval, no physics or horticultural certificate.
- Measure accepted unique targets/layout, accepted views/target, render/search
  time, native peak memory and disk size before setting a plant-count or ETA.

Only after that pilot should we select the number of layouts. For illustration,
an average of10 accepted views per target requires2,000 training target geometries;
an average of8 requires2,500. These are arithmetic scenarios, not measured yields.

## Scale engineering and review

- Qualify native1696x816 sensing and the complete resolution-aware labels, crops,
  Qwen preprocessing, exporter and validator before scheduling a20k campaign.
- Amortize scene/renderer setup across many targets and views; use bounded chunks,
  durable receipts, resume-without-duplication and measured memory limits. Current
  one-family jobs frequently spend hundreds of seconds in preparation; do not
  extrapolate20k completion from GPU model specifications alone.
- Prefer additional source geometry over arbitrarily increasing views per target.
  Vary camera/base/head poses and plausible lighting after structural diversity is
  established; do not simplify the surrounding greenhouse to reach a quota.
- Automatic geometry/native-sensor QA is required for every record. Manual or
  assistant visual review must be honestly attributed. Scaling the review workflow
  needs an explicit new policy; no unseen frame gets an invented accept and no
  existing hold is silently overridden. A stratified review scheme, if adopted,
  must disclose that it is not a100% manual review.
- Preserve original held-out plants as a separate transfer check, alongside any
  generated held-out-layout set. Same-generator synthetic accuracy is not real-world
  transfer. Shared original background/materials remain a reported limitation.
- A new versioned20k release must enforce the confirmed training minimum,
  explicit held-out counts, lineage/duplicate checks, review policy, native-depth
  provenance and portable Qwen3-VL-8B loading. The old small-pilot exporter is not
  the final acceptance authority for this milestone.
- Finish with a portable ZIP, checksums, manifest, Qwen training/evaluation
  instructions and clearly separated source/template/layout statistics. Do not
  download or train Qwen locally; the user will train on the4H200 server.

## Status

Implemented:read-only capacity accounting and a bounded TRAIN-only component
variant prototype. See [PLANT_GENERATOR.md](PLANT_GENERATOR.md). A20-layout CPU
pilot generated160 target instances from16 original donor families, with156
exact morphology hashes and138 original target identities. These counts are
not accepted images or approved independent target diversity. All160 passed
sparse cut-surface checks; two inherited attachment warnings are held for review.
No source family/split, existing image decision or12-view cap was changed.

Still not implemented:a general growth/curvature/node generator, generated-plant
native collection qualification,20k campaign, resolution-aware final20k exporter
or revised review policy. Existing native collection and the queued
higher-resolution pilot remain unchanged.
