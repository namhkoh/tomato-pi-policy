# Active-perception task-v4 prototype

Implemented 2026-09-10 on `koh-dev/sim-data`. This is an adjacent, versioned
perception contract and review-pilot builder, not a replacement for the existing
task-v3 release, a trained policy, a simulator controller, or an action dataset.

## What is implemented

- `active_perception_contract.py`: prompt, coordinate convention and validators
  distinguishing eligible/ineligible/unknown/no-target-in-an-assessed-region;
  localization, inspection, rejection, repositioning and a reveal-skill request.
- Non-localization answers must have null cutting coordinates. Privileged hidden
  geometry cannot justify observable eligibility or localization. Localizing a
  point does not authorize cutting. Repositioning requires explicit feasibility
  evidence; no-target requires a bounded, fully observed region and zero eligible
  targets. Unknown background labels are not absence evidence.
- The query prompt no longer presupposes that the query is on a valid petiole.
  Candidate-query and scene-region task modes are separate. Motion paths, force
  limits and gripper events are deliberately not invented by this schema.
- `active_perception_pilot.py`: a bounded manifest/review-page builder from
  independently audited, original 848x408 robot-head captures. Related examples
  preserve source-family splits and match target geometry plus plant transform.
  Source/human/legacy-task holds are excluded. Advisory holds are excluded until
  a human accept explicitly binds that exact current suggestion; an older
  acceptance is not retrospectively interpreted as resolving newer advice.
- Native component IDs identify candidate invalid organs. A visible-interior,
  valid-depth and RGB-usability screen proposes a query, not a human approval.
  Selection rotates preferred main-stem/fruit/peduncle negatives, with fallback
  when an organ lacks a suitable visible pixel. Unmapped pixels are never used.
- Original native Isaac `distance_to_image_plane` optical-Z files, validity,
  camera calibration and robot snapshots are linked as sidecars, not regenerated.
  Hidden XYZ is isolated under `private_evaluator`, outside `model_input`.

## Current review pilot

`data/sim_data/dataset_reviews/active_perception_20260910_v3/review.html`

The 28 examples contain eight matched visible/occluded target pairs, eight
invalid-object queries (five main stems, three fruits), and four leaf-covered
region examples with insufficient evidence. These are 28 task examples on reused
native frames, not 28 new independent captures. v1 and v2 are retained as provenance.

The region builder requires a complete 48x48 patch on one renderer-identified
leaf with valid native depth. It may produce fewer than requested (four of the
eight selected families here). It does NOT establish whether a hidden target
exists: the correct proposed response is unknown + inspect + null cut. A leaf
covering a region is not evidence that the region contains no eligible target.

Every v4 label is `pending_new_task_visual_review`, `human_confirmation=false`,
`training_eligible=false`. v3 approvals are NOT copied onto v4 instructions or
negative queries. The read-only HTML page can toggle candidate-query markers;
the underlying RGB is untouched. `pilot.json` contains exact evidence bindings,
proposed answers and private evaluator information; it must not be sent wholesale
to a VLM. It is not training-ready JSONL.

From `examples/greenhouse_sim`, prepare a NEW output directory:

```powershell
& 'D:\isaac-sim-6.0.1\python.bat' -B -m sim_data.active_perception_pilot --output ..\..\data\sim_data\dataset_reviews\active_perception_next --max-pairs 8 --max-negatives 8
```

`--max-negatives` and `--max-unknown-regions` (default eight) are upper bounds,
at most one example of each kind per selected pair;
the builder may return fewer if no native-supported query passes screening.
The default bundle requires its independent advisory assessment. Explicit bundle
and assessment paths may be provided with `--bundle` and `--suggestions`.

## What the static pairs do NOT demonstrate

A visible and occluded view of the same anatomy is a static comparison, not an
executed camera movement or left-arm reveal. The current frames explicitly lack
verified engine frame IDs/dynamic recording support. The builder reports these
gaps and always reports zero dynamic episodes. A passed metadata validator does
not establish real contact dynamics, collision safety, or horticultural validity.

## Reviewing and packaging task-v4 labels

The GUI at port 8881 still reviews existing v3 labels with separate assistant
advice. The separate v4 reviewer is at <http://127.0.0.1:8882>. From repo root:

```powershell
examples\greenhouse_sim\run_active_perception_review.cmd
```

1. Choose a case or filter. Start with untouched, full head-camera RGB.
2. Toggle review-only markers if needed: cyan circle is a candidate query; a
   cyan rectangle scopes an assessed region. Neither is a predicted cut box.
   A white cross is shown only for a proposed visible cut.
3. Inspect the native-identity and native-depth tabs. Green identifies the exact
   target/foreground component; it does not certify RGB readability. Depth is
   saved native optical-Z coloured on a fixed 0.04–2 m display range. Markers
   are hidden on the depth tab because its appended legend changes the layout.
4. Enter your name and an observation, explicitly confirm inspection, then
   Accept/Hold/Reject this one label. Corrections go in Hold notes; do not move
   simulator-derived coordinates by guessing. Saved decisions are read-only.

New records live beside the pilot in `v4_reviews/assistant/` and
`v4_reviews/human/`. Assistant assessment is separate from human confirmation;
opening the page or agreeing broadly does not create individual human reviews.
Every record binds the exact pilot/example. Existing source and task-v3 holds
remain effective; accepting a new task cannot bypass them. The v4 GUI never
modifies v3 decisions or source observations. The HTTP server is loopback-only,
uses same-origin/token write checks and serves only allowlisted evidence.

After explicit human acceptance, a CPU-only subset exporter is available:

```powershell
cd examples\greenhouse_sim
& 'D:\isaac-sim-6.0.1\python.bat' -B -m sim_data.active_perception_export --output ..\..\data\sim_data\dataset_exports\active_v4_reviewed_01
```

It refuses an empty/unreviewed set or an existing output directory; excludes
held/rejected/unreviewed examples; checks current source holds, every evidence
binding, and family/image split leakage. Original RGB and native depth/validity
are copied byte-for-byte. Local-image-path chat JSONL contains only the system
prompt, RGB, instruction and answer. Calibration, robot state and privileged
truth stay in separate sidecars, never model messages. A trainer-specific image
loader is still required. The manifest is written last; an interrupted output
without `manifest.json` is incomplete and must not be consumed.

This is explicitly a reviewed static-perception SUBSET, not approval of a full
training release or action dataset. It neither trains a model nor creates
missing physical episodes. No real v4 package has been released at this checkpoint.

## Required next implementation and collection

The builder saves `capture_requests.json` with six required classes; static
drafts now exist for the second, while the other five remain uncollected:

1. Fully observed bounded regions with no eligible target.
2. Dense foliage with insufficient evidence (not a false absence label).
3. Executed, collision-checked viewpoint changes revealing a target.
4. Physically validated left-arm foliage displacement and target-grasp transition.
5. Failed reveals, re-observation, bounded recovery or stopping.
6. Anatomically valid targets blocked by measured reachability/collision constraints.

Before collecting motion supervision, integrate a recorder that binds actual
native render frame/time identifiers, RGB/Z, calibration and robot state to the
same evolving physics state. Validate these timestamps against moving geometry;
do not substitute application counters or pair unrelated frozen captures. Replay
and verify contact, grasp, foliage deformation, visibility gain and protected
organ constraints. If holding an occluder prevents left-hand target support,
choose a different strategy rather than label an impossible simultaneous grasp.

Keep all frames, counterfactuals and appearance variants of a plant/episode in one
split. Compare cut-point accuracy AND unsafe false localization, missed valid
targets, correct abstention, reveal success, action count and collateral contact.
No-target examples and physical reveal trajectories are not yet collected by
this implementation. No simulator, model inference or physical robot is launched
by either module.
