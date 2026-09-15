# Component-based plant generator: static VLM pilot

## Scope

Implemented on `koh-dev/sim-data` as `petiole_similarity_pilot.v1`.
This is a **new, limited source-derived variant generator**, not a reconstruction
of the unavailable original tomato growth generator.

It changes the orientation and uniform size of intact petiole/leaf subtrees
around their original main-stem attachment. Main stems, fruit, flowers, trusses
and unselected organs retain their original geometry. It does not yet create
new main-stem growth, node spacing, petiole curvature profiles or independently
generated botanical families. These outputs are for static-perception
qualification, not the grasp/cut physics environment.

## Geometry and labels

- Fit the observed parameter envelope using **only the 16 frozen TRAIN donors**.
  Measured 584 intact geometric candidates: centerline length
  0.0612795?0.388947 m, proximal radius 0.000665?0.003563 m, attachment angle to
  the nearest parent segment 0.000148?116.803731 degrees. These are source
  measurements, not botanically validated ranges or a learned growth distribution.
- The pilot uses an explicitly engineering-selected grid: uniform scale
  0.9/1.1, azimuth ?15 degrees and tilt ?12 degrees. Reject transforms whose
  resulting measurements leave the observed training envelope.
- A single similarity transform updates the entire selected subtree:
  mesh vertices, normals, extents, centerlines, radii, component origins,
  attachments and axes. Rotations are baked into copied meshes, preserving
  the existing translation-only manifest/assembly contract.
- Recompute the nominal **10 mm** cut location and **10?20 mm arc-length**
  interval on the NEW centerline. Do not scale an old 10 mm label into 11 mm,
  reuse pixel annotations or inherit review decisions.
- Preserve topology, UVs and copied material textures. Source files are read-only.
  Existing source zero normals and faceless stubs are preserved and reported,
  not filled with invented geometry or used as eligible cutting surfaces.
- Remove inherited physics APIs and mass/inertia/joint/stiffness metadata from
  the NEW static copies. New dynamics would require explicit rebuilding and
  calibration; donor physical constants are not valid by default after scaling.
  The existing simulator assets and physics implementation are untouched.

## Validation and integration boundary

`plant_variants.py`: deterministic planning, training-only envelope,
subtree/attachment protection and new centerline cut proposals.

`plant_variant_usd.py`: create-only copied USD/material writer, serialization
read-back, structural audit and a sparse actual-mesh surface diagnostic.
At 10/15/20 mm it casts eight radial rays per point against authored,
fan-triangulated petiole faces. Each must hit within 0.2?3 times the capsule
radius. This diagnostic checks that proposed labels have nearby rendered
geometry; it is **not** a watertightness, biological, collision, cutter-clearance
or native-rendering certificate.

`plant_variant_catalogue.py`: separately verifies generated files, donor
lineage, unchanged component metadata, stored labels and code bindings.
Surface failures and parent-proxy warnings cannot become inspection rows.
The anonymous-stage adapter works with the existing `component_catalogue`
and `target_world_geometry` functions. It does not register generated plants
with the running collector or claim a full-greenhouse capture has passed.
Selected subtrees with attachment/bounding-box warnings are withheld even when
the same warning exists in the original donor.

Provenance detail: the legacy collection plan binds USD and JSON, **not textures**.
The generator binds donor texture bytes when copying; the inspection adapter
verifies those source and copied bytes separately and reports
`texture_binding_origin=generator_copy_time`. This must not be presented as
a texture guarantee inherited from the older frozen plan.

## Measured pilot (2026-09-15)

Artifacts:
`data/sim_data/generated_plants/petiole_similarity_pilot_20260915_v2`.

- 20 layouts from 16 original training donor families.
- 160 transformed target instances, with all 160 passing the sparse surface
  diagnostic.
- 156 exact morphology hashes; four exact repeats are not new geometry.
- 138 original donor target identities.
- 665,859,812 bytes of pilot artifacts before separate inspection reports.
- 499 source zero-normal entries and 28 faceless mesh instances preserved
  across the copied layouts. Those counts include reused source assets.
- No native images, human image reviews or training approval from this pilot.
- No claim that these are 20 new independent plant families or 160 approved
  novel training targets.

Full CPU assembly checked 8,496 component instances across the 20 layouts;
maximum translation error was 1.15e-16 m (rounded up). All 160 stored nominal
labels matched the existing world-projection interface. The broad-phase audit
reported 75 attachment-to-parent-box warnings, all already present in original
donors; no newly flagged component was found. Two involve selected subtrees:
`seed71_full_pv_a76b4509ae22f4c4/SubStem_48` (2.295 mm) and
`seed89_full_pv_ff4a85ca0a196303/SubStem_48` (2.747 mm).
Those two are withheld by the inspection adapter. The remaining 158 are
inspection candidates, not training-approved or physically certified targets.
These distances are point-to-AABB diagnostics, not measured physical gaps.

Evidence:

- `data/sim_data/diagnostics/plant_variant_pilot_inspection_20260915_v1.json`:
  original all-layout projection/assembly checks.
- `data/sim_data/diagnostics/plant_variant_attachment_comparison_20260915_v1.json`:
  comparison against all 16 original donor families.
- `data/sim_data/diagnostics/plant_variant_pilot_inspection_20260915_v2.json`:
  re-inspection with inherited-attachment holds.
- `data/sim_data/clear_regression_20260915_v20.log`: 857 tests plus77 subtests
  passed in68.93s. Final v2 inspection:20 layouts,158 candidates/two holds.

Reproducibility is defined by the version, seed/configuration, training envelope,
source/texture hashes and serialized geometry. Bit-identical USDC container
encoding across USD versions is not claimed.

Earlier failed attempts are retained as diagnostic artifacts:
`petiole_similarity_smoke_20260915_v1` exposed source zero normals;
`petiole_similarity_pilot_20260915_v1` exposed a faceless, already-deleafed
source stub. Neither is a complete accepted pilot. The successful one-layout
smoke is `petiole_similarity_smoke_20260915_v2`.

## Commands

Use a Python environment with NumPy and USD for these CPU-only operations.
No Isaac app, robot connection, GPU renderer or model weights are launched.

From the repository in PowerShell:

```powershell
$env:PYTHONPATH = 'examples;examples/greenhouse_sim'
$env:OPENBLAS_NUM_THREADS = '1'
python -m sim_data.plant_variant_usd --source-plan data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json --output data/sim_data/generated_plants/NEW_pilot --layouts 20 --targets-per-layout 8 --seed 20260915
python -m sim_data.plant_variant_catalogue --pilot data/sim_data/generated_plants/NEW_pilot --source-plan data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json --output data/sim_data/diagnostics/NEW_pilot_inspection.json --assemble
```

The output directories/files must be new. A failed run is not overwritten.
Changed generator code or stale assets require explicit requalification, not
silent reuse of an old receipt.

## Before using this toward 20,000 training images

1. Complete the independent higher-resolution native-camera qualification.
2. Add an explicitly versioned generated-catalogue collection path, preserving
   the real robot head camera and full greenhouse. Existing source plans,
   frozen family splits and review decisions must stay unchanged.
3. Check proximal attachments and unwanted mesh intersections with parent,
   protected and neighboring organs in the full scene; sparse radial rays are
   insufficient. Inspect 100?200 native pilot views individually.
4. Establish meaningful local-morphology diversity and near-duplicate criteria.
   The current conservative 12-view cap still groups by the ORIGINAL donor
   target. A new seed/variant ID does not reset it. Exact shape hashes alone
   do not establish independent biological or useful visual diversity.
5. Expand actual geometry generation if this bounded similarity stage does not
   supply sufficient useful diversity. Main-stem/node/curvature variation is
   still planned, not implemented here.
6. Only then approve a larger collection/review policy and versioned 20k release.
   Existing image qualification gates are unchanged. The 20k requirement remains
   **20,000 accepted training images plus additional held-out sets**.

The latest reviewed clear-image checkpoint contains450 total
(312 train,48 validation,90 test), still a draft. Generated geometry is not
counted as captured or accepted training data.

## Native capture adapter and output preview

A separate two-frame native diagnostic implementation and read-only actual-mesh
preview command now exist. See [GENERATED_NATIVE_CAPTURE.md](GENERATED_NATIVE_CAPTURE.md)
for commands and measured native/visual evidence on one original/generated
pair. Native1696 query/cut/crop annotations are a separate tested pilot:
[NATIVE_CLEAR_ANNOTATION.md](NATIVE_CLEAR_ANNOTATION.md). Broader validation and
production collection/acceptance remain pending; old gates are unchanged.

Actual output comparison:
data/sim_data/diagnostics/generator_visual_preview_20260915_v1/plant_comparison.png.
This shows geometry with diagnostic colors, not native RGB/depth or training
input. Donor main-stem and protected-organ geometry remain unchanged.
