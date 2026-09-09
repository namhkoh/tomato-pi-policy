# Task-v3 image review GUI

Open <http://127.0.0.1:8880>. This is separate from the older prototype
reviewers on ports 8877-8879. No Isaac application, GPU rendering, robot control,
cloud inference or regenerated depth is used.

From the repository root, double-click or run:

```powershell
examples\greenhouse_sim\run_training_review.cmd
```

The launcher reopens a matching existing server; it does not start a duplicate.
For an explicit bundle, from `examples/greenhouse_sim`:

```powershell
& 'D:\isaac-sim-6.0.1\python.bat' -B -m sim_data.training_review_gui --bundle ..\..\data\sim_data\dataset_reviews\grounding_all_sources_20260909_v3\bundle.json --port 8880 --open
```

## Review an image

1. Leave **Pending only** selected. At initial launch this set has 79 pending
   cards out of 108; the other 29 have existing assistant decisions.
2. Inspect **1 · Original RGB**, then **2 · Annotated evidence**. The first is
   the original 848×408 robot-head input, with no annotations. The second is
   the saved review card, including native mask/depth and separate query crops.
   Scroll down to inspect both crop rows. **Open full size** opens the selected
   image in a separate tab; it does not edit or save a decision.
3. Enter your name and an observation note (at least 20 characters), then tick
   the explicit inspection checkbox. Both image views must have loaded before
   the decision controls are enabled.
4. **Accept** if the target query and task answer agree with RGB/native evidence;
   **Hold** if uncertain; **Reject** for a definite label/target error. A hidden
   nominal region can be correct when the answer abstains with a null cut point
   and asks to change viewpoint. Do not automatically reject every fruit/leaf
   foreground crop, or approve a dim cut because a green mask covers it.
5. The decision saves immediately and advances to the next pending card.
   **Refresh** picks up decisions made in another tab. Existing decisions are
   read-only, including assistant reviews. Ask for explicit reconciliation if
   a saved decision is mistaken; never delete a prior hold to force acceptance.

The cyan marker is the input target query, not the cut. White is the nominal
10 mm cut, magenta the projected 10–20 mm interval, green the visible target
identity. Yellow rectangles indicate crops. Depth colours display the saved
native Isaac optical-Z array (yellow nearer, purple farther); they are not
computed replacement depth. Hidden cuts have no white/magenta overlay.

## Persistence and safeguards

- Task records are stored in the selected bundle's `decisions/<image-id>.json`.
  Opening the page, viewing evidence and navigation do not record approvals.
- Source audit/sample/RGB/card/native-depth/mask hashes are checked; only
  allowlisted images are served. Changed sources or stale/duplicate saves fail.
- Accept records explicit human visual agreement with this task label only,
  not physical-cut, horticultural or complete-training-release approval. The
  locally entered human name is self-declared, not authenticated identity.
- Hold/Reject first appends a negative human source-review record under the
  source job's `audit/records/`. The existing training exporter therefore
  excludes that frame even if the task bundle is omitted. The task decision
  is written second: interruption cannot leave an accepted frame behind.
  A retry can reuse the exact first source block without duplicating history.
- No existing task decision, source image, depth array, asset, annotation,
  candidate snapshot or capture supervisor is overwritten. Snapshot totals
  must be recomputed after new holds; the GUI does not rewrite old receipts.
- The server binds only to loopback, with same-origin/token checks for writes,
  no arbitrary filesystem browsing, and no cross-origin or bulk approval API.

Launch evidence: `data/sim_data/review_gui/task_v3_20260909_v1/` contains server
logs, browser screenshots and a read-only browser check. The actual browser
loaded 848×408 RGB and 1152×1230 evidence with no JavaScript exceptions and zero
automatic real-data review writes. The default browser was opened for the user.

Validation: 62 focused reviewer tests passed; full sim-data suite: **516 tests
plus 47 subtests passed** (48.17 s). Save/restart/negative-source/HTTP tests use
temporary fixtures only; they do not impersonate a human on real dataset cards.
