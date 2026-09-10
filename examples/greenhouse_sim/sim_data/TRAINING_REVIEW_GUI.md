# Task-v3 image review GUI

The updated suggestion-enabled reviewer is at <http://127.0.0.1:8881>. The older
port-8880 process was left intact to avoid interrupting an ongoing review.
This is separate from prototype reviewers on ports 8877-8879. No Isaac application, GPU rendering, robot control,
cloud inference or regenerated depth is used.

From the repository root, double-click or run:

```powershell
examples\greenhouse_sim\run_training_review.cmd --port 8881
```

The launcher reopens a matching server, checking GUI version and suggestions
directory as well as the bundle. It refuses to reuse an older backend. The CLI
default remains port 8880; use 8881 while the old reviewer is still running.
The launcher prefers the installed lightweight Python environment when available.
For an explicit bundle, from `examples/greenhouse_sim`:

```powershell
& 'D:\isaac-sim-6.0.1\python.bat' -B -m sim_data.training_review_gui --bundle ..\..\data\sim_data\dataset_reviews\grounding_all_sources_20260909_v3\bundle.json --port 8881 --open
```

## Review an image

1. Leave **Awaiting human review** selected, or use **Assistant holds to inspect**.
   At the 2026-09-10 check, 14 human decisions were preserved and 94/108 cards
   awaited a human pass. This includes the 29 legacy assistant-reviewed cards:
   an assistant decision no longer makes a card unavailable for human review.
2. Inspect **1 · Original RGB**, then **2 · Annotated evidence**. The first is
   the original 848×408 robot-head input, with no annotations. The second is
   the saved review card, including native mask/depth and separate query crops.
   Scroll down to inspect both crop rows. **Open full size** opens the selected
   image in a separate tab; it does not edit or save a decision.
3. Optionally open **Show assistant reasoning** after inspecting RGB independently.
   **Copy suggestion into editable note** only fills an editable draft; it does
   not save, check the inspection box or approve anything. Enter your name and
   a note of at least 20 characters. For a suggested card, select Independent,
   Agree or Disagree. The decision must agree with that selection. Both image
   views must load and the explicit inspection checkbox must be ticked to save.
4. **Accept** if the target query and task answer agree with RGB/native evidence;
   **Hold** if uncertain; **Reject** for a definite label/target error. A hidden
   nominal region can be correct when the answer abstains with a null cut point
   and asks to change viewpoint. Do not automatically reject every fruit/leaf
   foreground crop, or approve a dim cut because a green mask covers it.
5. The decision saves immediately and advances to the next pending card.
   **Refresh** picks up decisions made in another tab. Existing human decisions
   are read-only. Existing assistant records are also immutable, but may receive
   a separate human final pass. Ask for reconciliation of mistaken saved human
   decisions; never delete prior holds to force acceptance. To correct numerical
   labels, Hold the sample and describe the correction: editing the note does
   not overwrite coordinates or the v3 contract.

## Assistant suggestions

The default bundle loads 79 independently inspected, hash-bound findings from
`data/sim_data/dataset_audits/independent_v3_20260909/`: 72 support, seven advisory
holds. Scope, source RGB/card/sample/depth/mask, task identity and finding hashes
must match. Changed advice blocks stale saves rather than silently rebinding it.

Suggestions do not create decisions, source exclusions or human confirmations.
Advisory holds are prioritized in the list; the holds filter shows those still
awaiting a human pass (six at the browser check). Agreement/disagreement and the
exact suggestion hash are recorded with the human decision. A user can reject
supportive advice or accept despite an advisory hold, but cannot clear an actual
source/task hold through this GUI. Existing human decisions are never overwritten.

Use `--suggestions <assessment-directory>` for another bound assessment or
`--no-suggestions` for explicit independent review. These flags do not delete
advice or previous decisions. No cloud inference or new automatic visual-review
claim is made while loading the saved findings.

The cyan marker is the input target query, not the cut. White is the nominal
10 mm cut, magenta the projected 10–20 mm interval, green the visible target
identity. Yellow rectangles indicate crops. Depth colours display the saved
native Isaac optical-Z array (yellow nearer, purple farther); they are not
computed replacement depth. Hidden cuts have no white/magenta overlay.

## Persistence and safeguards

- Task records are stored in the selected bundle's `decisions/<image-id>.json`.
  A human follow-up to an existing assistant decision goes into
  `human_decisions/<image-id>.json`, binding the unchanged earlier record's hash.
  Export verification reads both histories; a later human hold/rejection cannot
  be bypassed by an earlier assistant acceptance. Suggestions are not exported
  as approvals. Human counts refer only to actual human records.
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

Original launch evidence: `data/sim_data/review_gui/task_v3_20260909_v1/` contains server
logs, browser screenshots and a read-only browser check. The actual browser
loaded 848×408 RGB and 1152×1230 evidence with no JavaScript exceptions and zero
automatic real-data review writes. The default browser was opened for the user.

Suggestion-enabled browser evidence is under
`data/sim_data/review_gui/suggestions_20260910_v1/browser_evidence_v2/`.
The final GUI-only regression is in sibling `browser_final/`. Full sim-data
validation passed **559 tests plus 47 subtests** (43.75 s).
It verifies original RGB, evidence tabs, editable suggestion copying, hold filters,
no JavaScript exceptions and zero actual review writes. Unit/HTTP save tests use
temporary fixtures only; they do not impersonate a human on real dataset cards.

For the adjacent v4 static contrast/invalid-candidate pilot and the remaining
dynamic-recorder/physical-reveal work, see [ACTIVE_PERCEPTION.md](ACTIVE_PERCEPTION.md).
