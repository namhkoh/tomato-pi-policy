# Original-native batch preparation v2

`batch_prepare_v2.prepare` generates one plant containing 1–12 explicit targets
from reviewed entries in `original_reference_bank_v2`. Entries must share the
same original TRAIN donor, current greenhouse, appearance, optics and mount.
Each target gets 1–6 proposals from the existing bounded mounted-robot planner.
Thus one future scene can supply at most 72 native frames.

```python
from sim_data.native_generated_reference import batch_prepare_v2 as prepare

result = prepare.prepare(bank_path, bank_sha256=bank_sha,
    entry_ids=exact_reviewed_entry_ids, seed=730201, output=new_directory, views=6)
```

Generation uses the current clear plan and unchanged TRAIN-fitted generator.
Target recipe order, derived seeds, actual geometry, all source bytes and the
bank replay are bound to the saved plan. Source assets are not modified. Existing
source caps and frozen splits remain in force; recipe names confer no novelty.

The profile fixes native 1696×816, 56 subframes per view and the legacy native-ID
backend. These preparation artifacts do not launch a renderer or approve data.
The separate `batch_execution_v2`, `batch_worker_v2`, `batch_owner_v2`,
`batch_audit_v2` and `batch_inventory_v2` implement that execution boundary.
Native execution qualification is still pending.
Frozen legacy multiview and two-frame bridge producers reject this new schema.

`batch_scene_v2` restores the actual fresh native reference pose and camera,
builds the same complete current greenhouse, and substitutes only the generated
foreground in the anonymous session layer. It requires an initialized native
application before importing native APIs. This hook has not yet run in Kit.

`batch_pose_v2.screened_reference` supplies the old bounded-pose helper with the
fresh original capture's separately recorded geometry screen. This is an explicit
runtime mapping, not a rewrite of historical pose evidence. `verify_pose` checks
base offsets, preserved height/heading, fixed mount and arm/torso joints, actual
native framing, renderer projection and independent camera forward kinematics.

## Validation on 2026-09-16

- 16 batch-planner tests passed in 30.93 seconds.
- Nine pose tests passed in 0.96 seconds.
- Real seed73/SubStem_41 generated plan: diagnostics directory
  `generated_original_batch_v2_seed73_20260916_v1`, plan SHA256
  `a69c6f8d64c9ca4284bf92b805e3939f56222445c6c05be94ed701047a48124c`.
  It has six future proposals at the requested resolution. Independent full
  bank/geometry/pose replay passed in 84.58 seconds without synthetic hooks.
  All six proposed bases passed the existing bound checks with preserved Z.
- The initial diagnostic reporting wrapper failed after saving the valid plan.
  The separate replay records that failure and the successful verification;
  existing artifacts were preserved. No native frames or training counts were added.

## Bounded execution v2

`batch_execution_v2.make_request` checks the exact prepared plan and the existing
same-callback compact storage proof, then binds the full local implementation
closure and explicit native runtime. `preflight_request` remains hash/stdlib-only
before `SimulationApp`; the owner and initialized worker perform full replay.
The neutral entrypoint is
`python -B -m sim_data.native_dataset.original_reference_batch_owner_v2
--request PATH --request-sha256 SHA256`. Its `--check-only` mode verifies the
actual raw CIM classification without native work. The shared owner mutex,
20 GiB commit reserve, 60 GiB disk reserve and unchanged process guard remain.

Each captured view uses seven requests of eight subframes, the fixed mounted
camera, full greenhouse and native RGB/ID/optical-Z. Only the newly generated
foreground is substituted. Every planned pose has an ordered saved decision;
pose and geometry rejections remain coverage records. The owner requires exit
zero and an independent saved-buffer replay before publishing its receipt.
The inventory replays that receipt again and emits the common observed schema,
including exclusions/holds and original source caps. It does not select data,
qualify independent geometry or grant training release.

Validation: 26 execution-boundary and whole-audit contract tests passed in
15.31 seconds after correcting the fresh decision-file pin check. One opt-in CPU
saved-buffer regression passed in 26.49 seconds, exercising full native array,
FK, hierarchy, label and trace replay plus corruption checks. That regression
uses an explicitly synthetic v2 wrapper in pytest temporary storage; it is
not a native v2 capture or qualification. Source captures stayed unchanged.

A replacement six-view request was prepared and checked in 154.06 seconds:
`original_reference_batch_v2_seed73_request_20260916_v2.json`, SHA256
`85b0eb684f67af0205c0aabd00fb7aa7a07020fdd320869b2141fedfacdd51e8`.
The v1 request and its waiting process were superseded before any native launch;
the old process exited on its source-pin check, and its artifacts are preserved.
The live serial continuation waits for the existing tail to finish and for
independent matched original/generated bridge evidence, then launches this
one trial and produces an independently replayed observed component.

The new producer has not run in Kit yet. Bulk rollout, actual image review,
geometry-extractor lineage support for this producer and fresh cumulative
selection remain subsequent work. Never send this schema to frozen v1/legacy
consumers or treat new recipe names as independent geometry.
