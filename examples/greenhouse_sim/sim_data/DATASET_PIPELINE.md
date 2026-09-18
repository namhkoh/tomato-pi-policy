# Dataset pipeline navigation

Status: 2026-09-18. This guide locates the active work; pinned run receipts remain authoritative.

## Current task and release

- Capture native **848 × 408 RGB-D**, with the complete **144-plant greenhouse**.
- Label the cut **9 mm along the petiole from its attachment**. Require exactly one eligible petiole in view, without a target query.
- Preserve calibrated robot reach, collision, depth, visibility, junction, support-width and independent ambiguity checks, followed by actual visual review.
- The current diversity rule is **at most two views per original plant**. Generated morphologies need a separate, explicit morphology identity and clone check before applying a two-view morphology cap; donor lineage and donor split must remain stable.
- A generated variant is not a new biological source family. Renaming, cloning or moving a plant does not create morphology diversity.

The authoritative pointers are [LATEST_DIVERSITY_CAPPED_UNIQUE9MM.json](../../../data/sim_data/training_exports/LATEST_DIVERSITY_CAPPED_UNIQUE9MM.json) and [LATEST_FULLY_POPULATED_UNIQUE9MM.json](../../../data/sim_data/training_exports/LATEST_FULLY_POPULATED_UNIQUE9MM.json).
They currently select `tomato_cutpoint_848x408_fullgreenhouse_diverse26_max2plants_20260918_v1`: **26 images / 13 original plants** (20 TRAIN, 4 VAL, 2 TEST), with `training_ready: false`.
The larger historical releases are archives, not the current diversity-compliant training set.

## Generating geometry

| Entry point | Purpose / boundary |
| --- | --- |
| [plant_variants.py](plant_variants.py), [plant_variant_usd.py](plant_variant_usd.py) | Donor-derived similarity changes to intact petiole subtrees; write new component assets and manifest without changing the donor. |
| [procedural_petiole_v2.py](procedural_petiole_v2.py), [procedural_petiole_usd.py](procedural_petiole_usd.py) | Curved petioles and relocated attachments, with leaf transport and complete manifest output. Older prototype cut qualifications do not establish current 9 mm eligibility. |
| [plant_morphology_generator_v1.py](plant_morphology_generator_v1.py) | Whole-plant extension in development: one deterministic map changes stem spacing/bend and organ geometry while preserving component graph and donor lineage. |
| [plant_morphology_preview_v1.py](plant_morphology_preview_v1.py) | CPU mesh preview and donor/variant comparison; not native RGB-D or visibility evidence. |

The original upstream growth generator is unavailable. These modules modify authenticated existing plants; they do not reconstruct that generator or claim new independent donor families.
Whole-plant generator receipts remain **not native-approved and not training-approved**. Its capsule radii are conservative warped proxies, not certified rendered cutting radii.

## Current full-greenhouse capture and admission

| Stage | Relevant entry points |
| --- | --- |
| Plan and full population | [native848_fully_labeled_plan_v3.py](native848_fully_labeled_plan_v3.py), [native848_fully_labeled_scene_v1.py](native848_fully_labeled_scene_v1.py) |
| CPU geometry and owned launch | [preflight_native848_fully_labeled_v3.py](../../../data/sim_data/diagnostics/collection_20k_848_20260916_v1/preflight_native848_fully_labeled_v3.py), [run_native848_fully_labeled_serial_v7.py](../../../data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_fully_labeled_serial_v7.py) |
| Native capture | [native848_fully_labeled_worker_v3.py](native848_fully_labeled_worker_v3.py): complete schedule and same-callback RGB/depth/renderer ownership, with CPU/native census agreement |
| Whole-census annotation | [native848_pilot_annotation_v16.py](native848_pilot_annotation_v16.py), [native848_annotation_cache_v9.py](native848_annotation_cache_v9.py): authenticate completed capture and closure; finite chunks of at most 16 frames |
| Geometry and joint ownership | [native848_fully_labeled_coverage_v3.py](native848_fully_labeled_coverage_v3.py), [native848_all_petiole_9mm_v2.py](native848_all_petiole_9mm_v2.py), [native848_joint_ownership_v1.py](native848_joint_ownership_v1.py) |
| Review and export | [native848_unique9mm_group_review_v7.py](native848_unique9mm_group_review_v7.py), [native848_unique_petiole_dataset_v8.py](native848_unique_petiole_dataset_v8.py) (DS8), [native848_unique9mm_batch_validation_v8.py](native848_unique9mm_batch_validation_v8.py) |

Orchestration helpers live in `data/sim_data/diagnostics/collection_20k_848_20260916_v1/`:

- `consume_native848_fullpop_completed_v6.py`: completed-capture CPU consumer.
- `prepare_native848_fullpop_completed_packets_v7.py`: bounded review packets.
- `record_fullpop_packet_actual_review_v5.py`: explicit assistant review records.

An automated sole candidate is not an accepted image. Review uses actual full native RGB, unscaled junction crops, the short route to the first attached leaf, and flagged competing structures. Whole-group holds and explicit export authorization remain separate from annotation.

## Artifact locations

- Repository `data/sim_data/diagnostics/`: source audits, plans, CPU receipts, launch records and pinned helper sources.
- `data/sim_data/diagnostics/native848_fully_labeled_bulk_prepare_20260917_v1/`: historical camera preparation and reservation ledgers. Old high-view plans are not authorization for new same-plant expansion.
- `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_fully_labeled_20260917_v1/`: native capture cases.
- `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_fullpop_CPU_20260917_v1/`: annotation chunks, review packets and unsigned export inputs.
- `C:/Users/USER/tomato-vlm-data-20260917/dataset_checkpoints/`: individually authorized export checkpoints.
- `C:/Users/USER/tomato-vlm-data-20260917/training_exports/`: assembled releases; follow the repository pointers above.
- Generator output directories are explicit arguments. Their manifests, qualification receipts, copied assets and previews belong in generated artifact directories, not over donor assets.

## Short generated-morphology integration roadmap

1. Pin the generator recipe, donor, new manifest and every geometry/material asset. Record a geometry identity separately from donor family; reject clones and retain donor split and component lineage.
2. Reuse an authenticated robot/camera pose only as pose provenance. Derive the new target geometry from the generated manifest, never from the old donor target coordinates.
3. Replace only the foreground slot in an anonymous full-144 scene. Keep all 143 background assignments and transforms fixed and in the donor's split; preserve full anatomy and verify CPU/native census equality.
4. Add an explicit generated geometry/report join and typed context. Existing original-geometry schemas must not be relabeled to admit modified plants. Recompute current 9 mm geometry, reach, collision, joint, depth, support and complete-census ambiguity checks.
5. Qualify two genuinely distinct variants with a small native diagnostic, then actual visual review. Keep native approval, morphology diversity, donor-family accounting and export admission as separate claims.

## Why older versioned files remain

`_vN` modules are often imported by newer adapters and named by immutable source hashes, completed run provenance, or typed predecessor-closure checks. A larger version number does not make earlier files disposable.
Keep these source bytes and historical artifacts in place. Add narrowly scoped versions when contracts change; do not rename, move or delete pinned predecessors as a cleanup shortcut.
