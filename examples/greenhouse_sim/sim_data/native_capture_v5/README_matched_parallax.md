# Matched parallax v1: CPU preparation only

This opt-in adapter selects **one target** from a completed V4 parent with any
supported original target count (1–12). It produces exactly three snapshots in
the order Y0, Y−20 mm, Y+20 mm, with **zero outward movement**. All three copy the
original selected target's `view_001` framing. Only candidate IDs, lateral
coordinates and the dependent distance fields change in the physical specs.

The full original scene anchor is retained even when another target is selected.
Source anatomy, greenhouse contents, body heading, arm/torso joints, lighting,
camera mount/optics, native 1696×816 resolution, source caps and frozen annotation
gates are unchanged. Head aiming remains the existing real-joint solver.

The new captures are specified as raw `native_generated_views.collect` with
`profile_render=False`, `instance_backend="fast"`, `render_budget="reference56"`.
Every captured view receives 56 subframes. The completed V4 parents used compact
storage and `warm56_then8_trial`; old pixels are not a matched render control.
Y0 must be freshly captured at reference56 and is a repeated reference, not new
diversity. No replacement attempts are permitted for geometry-rejected cells.

## Public CPU API

```python
plan = build(completed_path, completed_sha256, target_id=exact_generated_target_id)
original_anchor_base = check(plan)
cpu_receipt = write_plan(plan, new_diagnostic_json_path)
```

`completed_path` is an explicit `campaign_result.json` with exit code zero. The
adapter verifies the original V4 preparation through its existing verifier,
the copied original plan, completed native request/result and saved audit
bindings/counts. It checks all original deterministic six-view proposals, not
just the selected target. It binds the actual selected `view_001` metadata,
label, RGB and native observation files. Array data are hash-checked without
decoding or reconstructing depth; this is not a new annotation replay or visual
review. A final uncached disk reread/hash is mandatory.

`check` reconstructs the complete plan and compares canonical JSON, including
types, source/implementation bindings, evidence and receipt contract. There is
no permissible override list, global monkeypatch or long-term cache.

CLI: `python -B -m sim_data.native_capture_v5.matched_parallax plan` takes
`--completed`, `--completed-sha256`, `--target-id`, `--output`. `verify` takes
`--plan`, `--plan-sha256`. Both are CPU-only. **There is no capture command.**

## Deferred native-owner integration

After independent native admission and **after constructing SimulationApp**, a
future worker calls `validate_for_collect(app, plan)`. This requires a running
app, checks the derivative, then invokes frozen
`native_multitarget_plan.check(full_original_parent, replay_geometry=True)`.
The original parent can contain targets absent from the three-view derivative;
they are not omitted from this replay. Do not send the new schema to the frozen
multitarget CLI and do not run original geometry replay during CPU preflight.

The worker then owns a new, disjoint output, invokes frozen `collect` once with
the specified arguments and creates NEW request/result/failure receipts using
the versioned schemas in `RECEIPT_CONTRACT`. Receipts must bind the submitted
plan/hash, implementation and origin; retain all three decisions, metadata and
native callback hashes; check all source/implementation bytes again; and publish
completion only after app shutdown. An independent consumer must replay native
annotation before accepting that evidence. A receipt schema is not proof of a
native run: this CPU module never creates those receipts or grants approval.

Worker admission, renderer registry, capture CLI and native receipt publication
are deliberately **not implemented or exercised** in this first version.

## Bounded requested preparation

The ignored diagnostic script `prepare_matched_parallax_20260916_v1.py` pins only:

- V4 job0001, `seed101_full_cr_4000000/SubStem_42`;
- V4 job0003, `seed11_full_cr_4002000/SubStem_42`;
- V4 job0006, `seed23_full_cr_4005000/SubStem_40`.

It creates three plans (nine maximum snapshots), plus a CPU preparation receipt.
It cannot launch native work. No volume gain, continuous visibility, collision
certification, visual cut approval or training release is claimed.
