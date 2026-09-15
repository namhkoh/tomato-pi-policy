# Original-native reference: CPU preparation and scene seam v1

The frozen CPU seam consists of `__init__.py`, `prepare.py`, `scene.py`, and
`test_seam.py`. It consumes `original_reference_bank` v1, not a legacy pair proof.
Historical848 remains pose authority; the verified fresh1696 observation and
current clear plan provide scene authority. No approval, cap reset, or independent
morphology credit is granted. Source images and assets are not modified.

## Verified CPU fixture

Repository-relative artifact:
`data/sim_data/diagnostics/generated_original_seed73_prepare_20260916_v1/plan.json`

SHA256: `077bcd88bc6e5231c1d212c753184159bd4a1101f837143a2444e6f5a19a953f`

The actual seed73/SubStem_41 anchor produced seed73_full_cr_730001/SubStem_41.
Preparation took 126.6771907 s; independent saved-plan replay took 68.6928191 s.
These include repeated native-evidence/geometry validation, not native rendering.
The generated nominal projects in frame at [1355.6071751601148,130.169708376456];
this is CPU projection, not visibility, clarity, or collision qualification.

The plan fixes one exact authenticated pose, original_control then
generated_variant, at most two future frames, 1696x816, legacy native instances,
56 subframes per view, and the full greenhouse. All 22 joints, original base Z,
camera mount/calibration, placement and current daylight remain checked. Native
depth and labels must come from a fresh synchronized native callback.

## CPU interface

From the repository root in PowerShell, using Miniconda Python:

```powershell
$env:PYTHONPATH = (Resolve-Path examples/greenhouse_sim).Path
python -B -m sim_data.native_generated_reference.prepare check --plan ABS_PLAN --plan-sha256 SHA256
python -B -m pytest examples/greenhouse_sim/sim_data/native_generated_reference/test_seam.py -q -p no:cacheprovider
```

Actual interpreter: `C:/Users/USER/miniconda3/python.exe`. Final seam test run:
15 passed in 4.85 s. Unit hooks are synthetic and do not claim native execution;
the separate real fixture exercised the actual bank replay and CPU generator.
Creation API: `prepare(bank_path, bank_sha256=..., entry_id=..., seed=..., output=NEW_DIR)`.
Existing output is rejected; failed evidence is preserved, not overwritten.

## Native boundary (not a launchable CPU plan)

The immutable plan deliberately has `native_launch_supported=false`, no producer,
and no owner/post-exit audit binding. Future native execution requires a SEPARATE
versioned request binding this plan and all producer/context/audit code. Do not
rewrite the CPU plan to manufacture launch authority.

`scene.prepare_scene(app, plan)` and `scene.substitute_generated(context, plan)`
are only for an already initialized, running SimulationApp. Full `check_plan`
may import USD; never call it in a native interpreter before SimulationApp.
The producer must guard the app before calling the frozen scene entrypoints.
Pre-app checks must be hash/structure/resource checks only. Actual lighting,
counts and renderer are returned as `scene_evidence`, never `old_manifest`.

No producer, native callback, post-exit audit or inventory acceptance was tested
by this CPU fixture. A separate bounded producer version is the next work item.
Optional write-only observer integration must not change candidates or gates.

## Later bank coverage

Reuse `original_inventory_v2.original_observed_rows` only for its exact supported
`native_original_capture.serial_queue` receipt contract. Main's `serial_phases`
jobs need a separately named producer-specific authentication adapter after its
producer is frozen and completed receipts are explicitly pinned. Shared original
audit/row projection can be reused; receipts must not be relabelled as V5 pilot
or serial39 evidence. No extension is implemented by this seam.
