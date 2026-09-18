# VLM dataset collection and generation — execution handoff

## Current goal: collective Saturday delivery

Deliver at least **25,000 globally distinct accepted samples collectively by Saturday, September 19, 2026, 23:59 KST** (planning cutoff). This supersedes the older Friday/10k target below. Published total remains **491**, leaving **24,509** additional accepted samples. Counts exclude proposals, raw frames, controls, duplicates and review holds.

- [thor1: four L40S runbook](docs/dataset_thor1_runbook.md), config `configs/dataset_capture/thor1.json`: foreground donors 7,11,43,47,73,101; generator seeds 2,000,000-2,999,999.
- [thor3: two RTX PRO6000 runbook](docs/dataset_thor3_runbook.md), config `configs/dataset_capture/thor3.json`: foreground donors 17,23,53,71,83,89; generator seeds 3,000,000-3,999,999.
- Local RTX5090 config `configs/dataset_capture/local5090.json`: donors 19,41,67,103; seeds 1,000,000-1,999,999.
- [Shared execution contract](docs/dataset_multihost_runbook.md): Isaac Sim Replicator RGB, aligned optical-Z depth, validity, instance IDs, calibration, robot evidence; all144 complete labeled background plants; one eligible petiole and9mm cut point; common679-file numerical toolchain lock; global duplicate controls and per-group visual review.
- The initial background layout is shared. Different generator seeds alone do not prove complete scene or image uniqueness. Fresh background geometry requires a new complete census/catalogue and qualification.
- Both servers reported no production collection: thor1 was installing Isaac Sim, thor3 preparing a two-image Linux preview. The Linux owner/GPU/provenance adapter remains a prerequisite. Root SSH connections failed; nothing was deployed remotely.
- A single local coordinator is running on loopback127.0.0.1:8769. It was initialized from559 existing unique images (491accepted+68raw-only). The committed portable baseline is `configs/dataset_capture/baseline_inventory.jsonl`. Runtime receipt: `data/sim_data/diagnostics/native848_multihost_coordinator_20260918_v2/service.json`. Remote connections are not configured; never start independent copied coordinators.
- The first coordinated local batch completed six RGB-D captures with no duplicate holds; none passed all annotation/background criteria. The accepted total remains 491. Output: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/fixed144_multihost_mesh6_native_20260918_v1`. The next 16 views across four generated donor19 morphologies are reserved and launching at `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/generated19_multihost16_native_20260918_v1`.
- 12 new generated morphologies / 44 views across donors 19, 67, 41 passed CPU preparation and have distinct content identities. Native capture/annotation/review remains pending. The actual generator CLI examples for thor1 seed43/2000000 and thor3 seed53/3000000 both closed0 locally; this is geometry qualification, not native Linux acceptance.


Dual-GPU migration: [capture guide for two RTX PRO 6000 Blackwells](docs/dataset_dual_rtx_pro_6000.md). It documents hardware requirements, asset transfer, required launcher changes, benchmarking and accepted-image accounting. Concurrent capture on these GPUs is not yet implemented or qualified.

Four-GPU alternative: [capture guide for four NVIDIA L40S GPUs](docs/dataset_four_l40s.md), including the verified private Hugging Face asset download, four-worker design and qualification steps. The multi-GPU launcher and Linux adaptation remain pending.

**User reconfirmation, September 17:** Keep exactly one eligible petiole in view. The proposed nearest-visible-reachable selection rule was explicitly declined. Continue full144 populated scenes and distinct actual camera poses; any reachable or unresolved eligible alternative holds the frame. This governs all future preparation and annotation.

## Previous delivery target - September 18, 16:00 KST (superseded)

The user requests 10,000 images by 16:00. Latest instructions allow distinct views, including 3-4 varied camera angles, with counts reported per plant. The old strict two-view cap is superseded. Prioritize different plant shapes and target identities. All 144 complete unpruned labeled plants remain present; background vines must also be visible in the actual images.

### Published verified total: 491 images

- Latest bundle: `C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_delivery491_20260918_v1`.
- 442 original +49 generated images;395 TRAIN/82 VAL/14 TEST.13 original donors,20 original source targets,14 donor-derived morphology identities. Generated variants are not new independent donors.
- All prior490 entries and sensor bytes are preserved; the reference bundle uses loader native848_delivery_bundle_v3. Exact decoded-RGB and within-geometry camera duplicate checks pass across491. The released set still has13original donors and20source targets; target-diversity shortage remains unresolved. Exact decoded-RGB and within-geometry camera duplicate checks passed. `data/sim_data/training_exports/LATEST_DATASET_VLM.json` is authoritative; counts per donor/target are in `DATASET_DELIVERY_STATUS.json`.
- The latest11 additions passed actual native capture, full-scene annotation, visual review and export. Their background-plant coverage is59.96-64.65%, with zero unknown pixels. The repeated control frame is excluded.

### Local continuation - September 18

The user requests continued collection on this machine. The local RTX 5090 is available; published total remains **491**. CPU preparation is now searching new target/view combinations and additional donor-derived full-plant morphologies. GPU capture has restarted; see the current Saturday delivery section above. Keep all native image, robot, full-greenhouse, single-answer, duplicate and visual-review checks.

The source-mesh prefilter was checked against all 17 saved aisle frames: it identified all 14 ambiguous frames and retained the sole unique passing frame; 1,164 predicted clear probe owners/depths agreed with saved native data. It remains a proposal ranking tool, not acceptance. The first 256 prospective proposals produced only two candidates, both the already heavily represented seed103/SubStem42, so that small repetitive bank was not launched.

The nominal petiole-diameter estimate falsely rejects some native detail passes (9 of 96 strict target passes in the saved 72-frame audit). New proposal preparation uses that estimate for ranking only; exact 12-pixel projected support and the native 8-pixel mask-width requirement remain unchanged. The complete native ambiguity check remains authoritative.

### Main workflow: one varied greenhouse, capture first, annotate afterward

Use one fixed fully populated greenhouse across many distinct robot-camera poses. Retain native848x408 RGB, aligned optical-Z depth, validity, calibration and renderer IDs for offline annotation. Do not require a previously accepted camera view. Verify actual robot/FK/floor/collision/workspace before saving; perform visibility, complete-scene9mm annotation, competing-answer checks, duplicate checks and target/view-group visual review afterward. Exactly one eligible petiole remains required. Do not start further ordinary2-4-frame scene reloads as the main collection path.

Prepared environment: `data/sim_data/diagnostics/native848_fixed_varied_greenhouse_20260918_v1/environment.json`:144 complete unpruned labeled plants,16 TRAIN donor shapes,61,372components and11,068petiole instances. Clones do not create independent donors. All greenhouse vines remain, with visible-background quality checked in each captured view.

Completed native sweep: root session46624 exited0,55frames/0capture holds, across10donor shapes,11physicalvines and22targets. One scene/product/writer and one initial warmup. Raw output: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/fixed144_sweep55_native_20260918_v1/capture`. Typical production request12.1s/frame; entire native run1005.4s. Three reviewed frames from this sweep are now released in490; other raw frames remain diagnostic.

Offline55 annotation completed successfully after two adapter repairs (saved-JSON normalization and required observation/context links). Failed sources/outputs are preserved; numerical predicates are unchanged. Actual corrected run44909 exited0:55frames,9unique candidates,3also pass visible-background quality. Native full images, junction/first-leaf context and3flagged alternatives were visually reviewed; all3pass. Targets67/42,19/42,41/43. Result: `data/sim_data/diagnostics/native848_fixed144_sweep55_annotation_20260918_v3/result.json`. Actual review: `data/sim_data/diagnostics/native848_fixed144_sweep55_actual_review_20260918_v1/actual_review.json`. Root export6100 exited0; bundle2298 exited0 and all490model inputs were loaded/verified. Published490 at18:56:58KST. Exact decodedRGB and within-geometry camera duplicate checks passed; all487prior entries are preserved apart from reference-entry schema version.

The first new-body raw24 trial (root52800) closed1 with0frames: all24failed actual body collision checks. Durable records show the unrestricted orbit placed the body through the crop row, with torso/gutter/growbag/neighboring-vine overlaps. No margin was relaxed. Failed trial retained at `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/fixed144_raw24_native_20260918_v1`.

Corrected main run: root native33510 exited0 at `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/fixed144_aisle131_native_20260918_v1`.131 numeric proposals across15 original TRAIN donors,42targets (17new to the prior schedule), using aisle-side body templates and head aiming. Actual native floor/body collision/9mm workspace checks apply once per proposal; numeric preparation does not claim collision or visibility admission. Final result17frames/114collisionholds. Cached annotation root24854 exited0 in52.207s; one unique/background-passing frame103/42_032 passed actual visual review. Export74521 and assembly15498 both exited0; it is included in491. The other16remain held:8multi-strict,6sole-strict with reachable visible alternatives,2zero-strict. No unknown alternatives;14of16 failures are ambiguity. No GPU capture remains active after33510closed. Exact request: `data/sim_data/diagnostics/native848_fixed_scene_diverse_aisle131_prepare_20260918_v1/request.json` (SHA563cd18f4a1cf30a27224b096ace1c4326b1bd422b87f3a79a85dde3915fe70d). One scene/product/writer and one initial warmup for the batch. Production sources are frozen while running.

The55-frame corrected annotation took about11minutes. The measured end-to-end rate and3/55 final yield remain inadequate for10,000images. No deadline or10k completion claim is made. Scene reuse removes repeated small-batch setup, but rendering cost, collision-free placement and unique clear-answer yield remain bottlenecks. Actual one-frame profiling found11.268s total,10.907s (96.8%) in hashing:10,831reads/8.58GB of mostly unchanged assets. The new consumer had omitted the existing batch cache. A small hash-only wrapper now reuses that cache with initial and closing full source verification, unchanged numerical predicates, bounded operation, and mutation-failure guards. Actual17-frame annotation completed in52.207s. Cache receipt and all17source checks closed successfully; no numerical predicates changed. The raw capture descriptive orbit/orientation flags are inaccurate for aisle translations; actual measured root/joints/calibration are independently checked and a separate erratum preserves native bytes.

Git cleanup commits d7e3ff4,ad392da and6a20d97 preserved stable sources and pinned bytes. The cache wrapper/tests also passed actual replay and whole17-frame execution. Current CPU work evaluates existing mesh-ray helpers as a cheap candidate-view ambiguity prefilter; no native rerun is justified until its predictions are checked against saved17 ground truth. Existing full_population_mesh_ray_ranking_v1.py/junction_mesh_ownership_ranking_verified_v1.py already provide source-mesh ray operations; reuse them rather than create another scene/renderer architecture. No training, remote server launch or push is requested.

### Required task and current work

Native 848x408 RGB, aligned float32 optical-Z depth and validity, sensor calibration, exactly one eligible petiole with no query cue, and one precise 9mm cut from the main-stem attachment. Keep clear proximal surface/junction/first-leaf context and actual robot camera/workspace checks. Preserve donor train/validation/test lineage. Never count raw captures, previews, candidates or generated assets as released images. Seed41/SubStem38 remains excluded; released generated41 targets SubStem45.

- Generated19, 67 and 43: all four views per morphology passed native capture, full annotation, actual visual QA and export. They are included in published470. Those ordinary native processes are closed; active throughput qualification is listed above.
- Recent example figures and direct original RGB/annotation links: `data/sim_data/diagnostics/dataset_examples_20260918_1553/README.md`.
- Remaining scalable-renderer work: additive `native848_persistent_capture_v1.py`, `native848_persistent_foreground_v1.py` and `native848_persistent_generated_9mm_v1.py` passed CPU checks. Actual native A-B-A qualification and experimental11-image export completed; those11are in487. The fixed-scene sweep is now the main workflow.
- Generated17 held on a numerical transverse-basis guard; frozen generator unchanged. Generated89 held on the frozen petiole length envelope. Donor29 is validation and is not eligible for the TRAIN-only generator.
- Root alone launches native GPU processes serially. Automated checks apply per image; inspect actual native RGB, junction/first-leaf context for each target/view group and every flagged alternative. Reject exact and near repeats. Generated geometry identity and original donor identity remain separate. No training/remote launch is requested.

### Measured throughput and remaining engineering problem

Replicator already supplies synchronized RGB, optical-Z and instance IDs. Writer persistence is approximately 0.08 seconds per frame. Ordinary production requests take about 11-14 seconds; a completed generated19 four-view owner run took 414.1 seconds including scene preparation, warmup and verification.

The new far-background representation preserves all 144 plant shapes/materials and original collision geometry, merging only distant components with positive exclusion from both actual arm workspaces. Four generated103 captures passed full annotation, actual visual review and typed export. Render requests took 3.7-4.4 seconds, but the whole owner run took 711.9 seconds: extra overlay preparation/verification outweighed render savings for four-image batches. This is a comparison of different generated donor scenes, not a controlled same-scene total-latency experiment. This historical four-frame comparison does not justify further ordinary small-batch restarts. Evidence: `data/sim_data/diagnostics/native848_fast103_vs_standard19_measured_timing_20260918_v1/result.json`.

**Persistent cross-morphology rendering passed native A-B-A qualification on September18; useful-batch throughput is still under test.** The old production path reloaded and verified the greenhouse for each generated foreground. The current persistent path retains it while refreshing foreground geometry, collision checks, component-ID mapping and annotations. Faster individual render requests alone do not solve this.

The 10,000-image target is not met and cannot be reached by 16:00 at measured throughput. Continue useful collection without weakening acceptance or labeling donor-derived variants as independent plants. Some downward views retain substantial floor; prioritize less-downward accepted cameras with more visible vines. Local cleanup commit `e5b5e7d` preserved 31 source/documentation files and all pinned source bytes; later additive adapters require a final cleanup commit.

### Historical record below

The remaining dated sections preserve earlier trials, failures and decisions. Their old counts, active-process statements and two-view caps are historical; the active delivery section and current published pointer above take precedence.

## Historical generator repair - September 18

The shortage of genuinely diverse usable samples is **not resolved**. The current diversity-capped release is still 26 images / 13 original plants / 16 targets. Generated assets and camera proposals do not increase that count.

**Latest completed trial:** One seed53 generated frame was actually captured at848x408 with aligned depth/IDs; owned native session36378 exited0, one frame and zero capture holds. Fresh annotation session30354 also exited0 and accounted for all144plants/11,070petioles with zero unknowns. The frame is HELD, accepted increment0: the unchanged proximal clarity predicate sees25/27 points, with failures at29mm and30mm. Both pixels belong to the same generated petiole, whose rendered surface is3.23mm/5.45mm closer than in the old accepted view. Camera, calibration, robot root/joints/mount are exactly equal to that old view. This demonstrates a generated surface occlusion/depth failure, not a camera-provenance error. The9mm nominal,9?19mm support and junction checks pass. Do not weaken the clarity threshold or add this image to training.

- Native RGB: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/controlled_seed53_curved_native_20260918_v1/capture/frames/seed53_full_SubStem_41_proximal30_controlled25mm_curved_view_0/rgb.png`.
- Completed annotation: `data/sim_data/diagnostics/native848_controlled_seed53_annotation_20260918_v1/result.json` (SHA85c5b475c05807ada5ce36bf160dadef870a05ab3f9d29ffb75df63d932daf2e).
- Exact old/new failed-probe comparison: same annotation directory, `proximal_hold_comparison.json`.
- Root inspected the complete native image using adjacent lossless, unscaled halves. The populated scene and junction are visible; no morphology-diversity approval follows. Review: `data/sim_data/diagnostics/native848_controlled_clear_donor53_curved_20260918_v1/native_review/actual_review.json`.

**Remaining blocker:** the capture/annotation integration now runs end to end, but the controlled deformation has not yet yielded an accepted or diversity-approved sample. Meaningful new plant geometry, no self-occlusion near the cut, and a verified clear view must be solved together. This one25mm control cannot substitute for a scalable plant generator. Next work should reject target self-occlusion with authored-mesh ray checks before expensive native trials, then qualify visibly substantial plant variations across multiple donors. Preserve all current resolution, populated-scene, exactly-one-answer, workspace and split requirements. Do not resume old repetitive queues. No native process is left collecting after this completed diagnostic.

Concrete findings:

- There are 24 original labeled donor plants. Two views per original vine can provide at most 48 images before eligibility checks. Larger releases need meaningful new geometry; original donor counts must remain separate from generated morphology counts.
- Two whole-plant bends were generated and visually compared. Their local junction appearance was nearly unchanged; they were not approved as a diversity solution.
- Existing `procedural_petiole_controlled_v2.py` produced two actual 25mm controls: `seed7_full/SubStem_44` and `seed11_full/SubStem_45`. Both preserved metadata attachment points but moved originally shared mesh vertices at the stem/petiole joint. Fresh checks held both. Evidence: `data/sim_data/diagnostics/native848_controlled_new_donors25mm_20260918_v1/current9mm_holds.json`.
- Repair implemented and actually verified: `procedural_petiole_controlled_v3.py` preserves the first30mm cut/junction surface, normals, topology and matching centerline, including complete triangles crossing that region. It shifts the selected distal petiole and direct leaves25mm. Both donor7/44 and donor11/45 passed generated catalogue replay and fresh9mm correspondence;22 focused checks passed. Donor11 retains78 whole proximal faces and all10 shared junction vertices; donor7 retains102 faces and all10 shared vertices. Source knot insertion prevents the old105mm capsule chord from moving the9mm point. Actual CPU previews show a visible bend, with an angular transition from coarse source tessellation; native realism/diversity approval is still pending. Results: `data/sim_data/diagnostics/native848_controlled_new_donors25mm_20260918_v3/current9mm_qualified_pair.json`.
- Root prepared three authenticated mounted-camera proposals for donor11 in2.5seconds. Those are camera geometry proposals, not eligible samples. Native collision, workspace, all-petiole annotation and visual checks remain necessary. Donor7/44 has no viable existing survey anchor; do not send it to a blind camera batch.
- New generated annotation uses separate geometry identity plus original donor lineage. The frozen numerical9mm, clarity, workspace and competing-answer rules stay unchanged. One generated foreground must preserve all143 labeled background plants.
- Isaac Sim Replicator already produces RGB, optical-Z and instance IDs. Measured production requests are12–14seconds each; writer persistence is about0.08seconds. See `examples/greenhouse_sim/sim_data/REPLICATOR_COLLECTION.md`. Correct geometry/annotation integration precedes persistent cross-morphology optimization and reduced-render-budget benchmarks.

The first native corrected-donor11 trial (`C:/Users/USER/tomato-vlm-data-20260917/diagnostics/controlled_seed11_proximal30_native_20260918_v1/`) exited nonzero before rendering. The camera and generator legitimately used different collection plans; the adapter now authenticates matching source assets and split assignments while retaining both plan identities. Nine focused adapter tests passed. Actual full144 CPU rehearsal then preserved all143 backgrounds but held both selected donor11 poses:22 and7 unresolved possible robot/background overlaps after triangle refinement. These are conservative possible-overlap holds, not certified exact contacts. Do not relaunch those poses. Receipt: `data/sim_data/diagnostics/native848_controlled_scene_two_plan_CPU_20260918_v1/subset02_cpu_handoff.json`.

The first donor53/SubStem41 generation attempt was held because v3 assumed the first source capsule segment exceeds60mm. Actual source analysis found a28.885mm first segment followed by a2.53degree bend, while its full0-30mm source centerline remains within the protected mesh region. The correction now retains every original capsule knot and adds blend samples at actual longitudinal intersections; protected point/radius equality and open-segment tangents are checked separately. Twenty-two focused checks passed. One fresh donor53 asset actually generated and replayed in28seconds, preserving78 proximal faces, all10 shared junction vertices and zero proximal radius difference. Physical evidence: `data/sim_data/diagnostics/native848_controlled_clear_donor53_curved_20260918_v1/current9mm_evidence.json`. Historical v3 receipts retain their historical source hashes; intentional source repairs invalidate replay against the current implementation until fresh assets/evidence are generated.

An actually accepted donor53 camera (`...271dd72359ec1d60cf22_045`) has been extracted and matched to the recorded robot/camera pose. `prepare_controlled_views_v1.py --camera-plan ... --sample-id ...` can reuse that exact camera and recompute the new geometry instead of aiming a new camera. Actual one-pose preparation completed successfully with exact camera preservation (`.../native848_controlled_clear_donor53_curved_20260918_v1/existing_camera/candidates.json`). It does not inherit collision, visibility, workspace or acceptance. The immediate objective is ONE complete generated native capture plus annotation at this known view. This is an integration control, not a new independent donor or evidence that the diversity shortage is solved.

A bounded pre-correction compatibility scan of the16 currently accepted target pairs made12 TRAIN calls and excluded4 held-out pairs. Five TRAIN targets support the original fixed25mm generator:101/41,103/42,103/47,41/45,47/41. That scan produced no assets/images and does not authorize bulk capture. Receipt: `data/sim_data/diagnostics/native848_current26_controlled_v3_compatibility_20260918_v1/result.json`.

Entry points under `examples/greenhouse_sim/sim_data/`: `prepare_controlled_views_v1.py` prepares a bounded camera proposal; `native848_controlled_capture_v1.py` is the owned capture integration draft; `native848_generated_9mm_v1.py` evaluates a separately typed generated full-scene census. One native generated capture and complete annotation now ran successfully as processes, but yielded a clarity hold; generated release admission remains unproven. Never restart the old repetitive queue to inflate counts.

## Whole-plant generator extension and Git cleanup - September 18, 11:24 KST

The user confirms the original generator is unavailable and explicitly authorizes modifying existing plants to create our own generator. They also ask to clean up the Git worktree.

- Existing implementations already exist: `plant_variant_usd.py` rotates/scales connected petiole subtrees; `procedural_petiole_v2.py` creates curved/relocated petioles and transports leaf blades. These are donor-derived static variants, not the omitted original `tomato_stem_generator` growth source. Reuse their USD/material/normal authoring code.
- Current additive implementation: `plant_morphology_generator_v1.py`, applying one deterministic, smooth, base-fixed map to the entire plant. Vary stem bend and vertical node spacing coherently across main stems, petioles, leaves, fruit and flowers. Transform actual mesh vertices, normals, attachment points, component origins and centerlines together; preserve the full component graph. Conservatively bound warped capsule approximations and measure interpolation error rather than treating original circular radii as exact.
- First bounded experiment is two generated morphologies from one TRAIN donor, with structural tests and an actual-mesh comparison. Recompute the 9mm point on the resulting centerline. Generated assets and diagnostic previews do not count as accepted native RGB-D samples.
- User intent now permits meaningful generated morphology diversity. Keep at most two accepted views of each distinct geometry group; copied/rigidly repositioned/renamed clones do not reset this cap. Report morphology counts and original donor counts separately. Preserve donor-level train/validation/test lineage; variants of a TRAIN source cannot become held-out TEST plants.
- Generated capture still needs an explicit geometry-source identity, fresh complete petiole census and truthful camera-seed provenance. Reuse the authentic robot camera and freeze the 143 existing same-split background assignments when replacing one foreground plant. Current native848x408, exact-one eligible petiole, full populated greenhouse, 9mm clear cut point, robot workspace and visual-review requirements remain binding.
- Current accepted max-two-original-plant subset remains 26 images; no generated morphology is accepted yet. Do not resume the repetitive original-family capture queue or inflate the count with raw/preview assets.
- Git cleanup: local commit `4746a73` preserves the 283 previously untracked source/documentation files (278 Python modules all syntax-checked), plus exact-byte `.gitattributes` rules. Worktree and staged blobs were SHA256-checked against the pre-cleanup inventory. No dataset/source deletion or remote publication. Older versioned files retain their paths because receipts and imports reference them. Inventory: `data/sim_data/diagnostics/git_cleanup_20260918_v1/inventory.json`.

## Latest diversity-capped subset - September 18, 11:07 KST

- **26 validated selected images,13 original supervised plants,16 cut targets.** Exactly2 views per original plant across all its targets and cloned instances. Splits20 TRAIN /4 VAL /2 TEST; TEST still only1 original plant. This is a small subset, not an adequate final training/evaluation dataset; training_ready=false.
- Path: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_diverse26_max2plants_20260918_v1/. Result SHA256 3c1fc89a1c8a5b511ac090a7eb8242fa2f9a32c7b5a5df0cac0c5be4cf07045e. Both LATEST_DIVERSITY_CAPPED_UNIQUE9MM.json and LATEST_FULLY_POPULATED_UNIQUE9MM.json now point here, explicitly marking the small subset. Original382 release remains archived unchanged.
- Gallery: data/sim_data/dataset_reviews/fullgreenhouse_diverse26_gallery_20260918_v1/index.html. Similarity examples: data/sim_data/dataset_reviews/diversity_audit382_20260918_v1/index.html.
- Root read entire subset assembler diffab721e19; focused8guards genuinely passed. Root actual assembly39520 CLOSED0. All382 parent rows/assets passed unchanged DS8/epoch/fullscene/split/duplicate checks before selection; only26 chosen rows/assets copied, all selected model inputs loaded. No new annotation or visual acceptance was invented.
- Selection allowlist71260d5d and auditeab5d38d exhaustively ranked7631 within-family pairs, rejected the same121 perceptual candidates as root's screen, and selected pairs with Hamming distance>6 (minimum16), favoring different targets then actual camera separation. This heuristic and physical separation support selection; exact-zero duplicates alone was never sufficient.
- Current repeated-vine native47 stopped with9 raw frames retained, native child75152 wait0 and rootqueue46284 exit1 because stopped schedules are deliberately inadmissible. Queued TEST31 and ready TRAIN17/TRAIN41target42 expansions must not launch under the new rule.
- Existing extra-family capture audit found zero previously unheld sole candidates outside the13 accepted families. One bounded new VAL13 investigation is source-only: two exact saved but never captured poses require current requalification and explicit reserved-plan supersession. No newfamily admission yet. New original plant generation route is being audited separately.
- The old10k target cannot be met by expanding these13plants: it needs at least5000 originalplants at2views each. Preserve strict single-answer9mm cut clarity, full144 labeled unpruned greenhouse, native848x408 RGB-D, embodiment and original split separation.

## New binding diversity requirement - September 18, 10:55 KST

User explicitly requires **only1-2 views of the same vine** and asks to check duplicates. This supersedes previous permission to maximize views per target/plant. Count original supervised source plant identities across all targets and cloned instances; clones do not add quota. New data collection must prioritize new original plant identities. Earlier382 remains archived, not compliant with the new two-view cap.

- Verified382 dataset:13 original supervised plants,20 original cut targets; TRAIN286 images from10 plants, VAL82 from2, TEST14 from1 (all TEST seed31_full/SubStem_42). There are ZERO exact decoded-RGB or exact camera duplicates, independently confirmed across all382. All source families, including background clones, remain disjoint across splits.
- Perceptual audit flagged121 candidate similar pairs involving135 images, all within the same original family; threshold is63-bit non-DC DCT Hamming<=6 and is a screening heuristic, not a universal duplicate verdict. Root actually viewed pair indices228/376 at native848x408 via pixel-identical lossless transport and confirmed highly similar framing of the same target. No claim to have personally reviewed all candidates.
-12of13 supervised sourceplants exceed2views. Strict cap2 permits at most26 of382 (20TRAIN/4VAL/2TEST);356 images exceed that cap. The one held-out TEST plant is insufficient for a diverse plant-generalization assessment.10k images at≤2perplant would require≥5000 original plants.
- Native expansions stopped: root46284 actually CLOSED1 after intentional STOP_AFTER_CURRENT_FRAME for current TRAIN47, and STOP_BEFORE_NEXT was set on its queue. Planned TEST31 did not launch. Partial assets retained, no partial admission or retry. Completed TRAIN103 new46 raw and CPU-ready TRAIN17 new19/TRAIN41target42 new6 are not approved for further repetitive-view release.
- Root/agents auditing original plant supply and preparing a create-only diversity-filtered subset from alreadyaccepted382. Select≤2 per ORIGINAL sourcefamily; require within-pair perceptual distance>6 and favor different targets and larger physical camera separation. Preserve all nativeRGB/depth, current quality checks, source splits, and old artifacts. The subset will be explicitly too small for the requested training goal.
- New annotation and export of repetitive deferred captures remain paused. No blind rerun of prior held/new-family searches. No training launched.

## Active capture-first queue - September 18, 10:41 KST

- Current released count382 (286 TRAIN /82 VAL /14 TEST),20 original targets/13 original families.
- Browsable fixed382 preview: data/sim_data/dataset_reviews/fullgreenhouse_joint382_gallery_20260918_v1/index.html. All382 RGB copied unchanged and SHA checked; cut/attachment overlays are SVG-only. No new samples created. Release resultf64fcb7903248ddce2c7f295d5abc62addb3961d4500598b9424b8f4640da667. Stable latest pointer resolves to joint382.
- TRAIN103 new46 capture genuinely closed0 (root5997/owner79236). All46 raw frames saved; NONE counted accepted. Offline request prepared at C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_fullpop_CPU_20260917_v1/seed103_reviewed19grid46_capturefirst_joint_actual_request_v1.json. Annotation is intentionally deferred until capture phase closes.
- Native root46284: TRAIN47 new27 (owner32880), then TEST31 new14; requestf7e636e0a68108e381d80d0591ffff3063243a00aa02e96236fe6ef295ba8a8d. No annotation/review/export barrier between capture jobs. Root read both entire ready handoffs and source changes before launch.
- TRAIN17 new19 CPU-qualified, helper0abe0e6d, CPU49242 genuinely0, ready ee3e3e350830703904e1c2dc0203b7292f8ad688d805f62e011d212b89a56e94 fully root-read. Native pending. TRAIN41 target42 ONE256 CPU attempt97717 active after explicit root review of source v1 and metadata-only v2(5e880956); seven exact accepted controls, corrected11 source only, targets38/43/45/46 excluded from this proposal task.
- Older TRAIN89 eighteen actual visual passes and validations85208/74055 are ready for export during postcapture processing. TRAIN41 recovered12 produced6 strong frames but all6 are visually held for uncertain short Leaf042 connection (validator19650 genuine0), no export. Old accepted controls are not reclassified.
- Source-only annotation audit confirmed repeated hashing: TRAIN53 chunks total183.884s; initial+closing SHA100.203s over approximately14GB/16998files per16. The queued offline runner preserves current chunk checks; it does not claim to remove this rehash cost. A safe shared cache requires additional validation and is deferred for this deadline.
- User's capture-first then offline annotation workflow governs remaining work. Continue saving full RGB/depth/calibration/scene metadata and reporting raw vs accepted separately. No quality requirements or original splits changed; no training launched.

## Latest verified release - September 18, 10:32 KST

- **382 accepted full-greenhouse images:** 286 TRAIN / 82 VAL / 14 TEST; 20 original targets across 13 source families.
- Dataset: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint382_20260918_v1/. Result SHA256 f64fcb7903248ddce2c7f295d5abc62addb3961d4500598b9424b8f4640da667. Stable D latest pointer resolves here.
- Added44 TRAIN53 views after47 actual full-RGB/junction reviews, with short first-leaf and flagged alternative checks. Three parent-edge singleton holds and one automated hold remain excluded. Agents legacy_reuse and native_recovery2 performed actual visual review; root read all47 reasons. Validators23849/34553/42920 genuinely closed0. Combined root export99043 genuinely closed0.
- Capture-first instruction below remains current. This release only completes processing that was already underway. Remaining new batches receive offline annotation after the capture phase, not between captures. Raw frames and prepared poses are not accepted samples.
- Native848x408 RGB/depth, full144 labeled unpruned plants, exactone eligible petiole,9mm cut, original splits, distinct physical camera poses and workspace checks retained. No training launched.

## Capture first, annotation afterward - September 18, 10:29 KST

User directed capture of distinct views first, followed by one offline annotation pass. The remaining native schedule must not wait for new annotation, visual review, export, or dataset-union publication. Preserve native RGB, aligned depth/buffers, calibration, geometry, source identity, and completed-owner provenance so all checks can run afterward. Keep inexpensive pose/workspace/detail/novelty prechecks before rendering. No acceptance criteria are relaxed.

- TRAIN103: root native session5997, owner79236, finite46 poses (target42x32, target45x14), request11b2f2dc0b9ee493b4672f865992210c734965d449c5653ed6633fcce31a9be7. Capturing; no new annotation is launched for this batch during capture phase.
- TRAIN47:27 distinct prepared poses across two authenticated anchors. CPU47705 genuinely0, ready5c4750e33a0ef303f8dad49d9b72ccbecbfa55c545bc8d8e3a6b8215e088bda7. Next native after actual predecessor closure.
- TEST31 then TRAIN17: each one finite256-proposal CPU attempt, at most64 selected, explicitly root-reviewed and authorized sequentially with sealed prior ledgers and fresh memory. Source helpers a68ff4c2caec51a313035a3c9bc09d3eb906cb90b69c1048525f273d83951c3a and0abe0e6d9fd67545008e8a96725ef2866a9b09b5b10325fadca8f40cea6e643c. Neither is counted as captured or accepted before actual completion.
- Already-running work may finish: TRAIN53 actual47 strong reviewed,44 accepted/3 parent-context holds, root export99043 active; TRAIN89 actual18 review and TRAIN41 actual6 review. No new QA jobs until post-capture phase.
- Offline entry point: data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_capture_first_offline_annotation_pass_v1.py. One finite manifest of completed capture requests; internally memory-bounded16-frame chunks using unchanged consumer6/cache9/annotation16. All requested capture owners must already be observed closed0. No automatic retry, native launch, export, or claimed visual acceptance.
- Latest released dataset remains338 until completed exports are assembled. Raw captured counts and accepted counts must remain separate. Native latest-start guard remains11:25 KST; finish annotation/review/package by noon as far as throughput permits.10k by noon remains infeasible at measured throughput.

## Latest verified dataset - September 18, 10:02 KST

- **338 accepted full-greenhouse images:** 242 TRAIN / 82 VAL / 14 TEST; 20 original targets across 13 source families. Latest: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint338_20260918_v1/. Result SHA256 48d221411992392b65ef8e20c2e424e482d4823b57f127529c625ac5a0f12410. Stable D latest pointer resolves here.
- Added21 VAL97 from25 captures after automated22strict/21strong and actual review of all21 RGB/junction/first-leaf plus21 flagged-context crops. All21 visual passes across15 groups. Root read all21 reasons; actual validations39593/75544 and combined export58162 genuinely closed0. Four unexported raw frames remain excluded. Largest source target is now50 images (14.8 percent).
- TRAIN53 new48 completed native38932/owner66044 zero; annotation64854 zero yields48strict/47strong. Visual QA is being split between legacy_reuse (first16/final15) and native_recovery2 (middle16) after packet preparation closes. None counted yet.
- TRAIN89 recovered19 native is root48374/owner81632 under request0ed4aa0cf40ad34ef74b374aed04f10ae94e03d5d19e7653a71e2bbb7aaba80a. New images require full epoch2 census/clear nativecut/parent/firstleaf review.
- TRAIN41 one finite32-ID recovery CPU is91442 on exact priorc421b17958a0042f1947bf69d5e1ae921b5f1b84c94d5f746b340dc54490d73d. Root read full helperc5d05088f62b6acebbc3df6a0d7ee9959b67c2c7724f7ce7b0eda7ee7529a9eb and contract3e7d2b6a1e7e117fc0735f5c4ddd366d874d103fb72e06b46f2d06b5e5b2c0dc;9 reconstruction and16 focused checks passed. Only8 surviving43/45 source controls, held46/banned38 excluded. Independent saved arrays exist for24 positives, not32 deferred. Actual next spacing/CPU result pending.
- TRAIN103 next grid source ffee7953ea6e74745da20845ba846ed71192815ff0b4cb0250a92c8209744c57 fully reviewed: exact19 newly exported controls42x12/45x7,256 proposals128each, at most32 selected per target/64 total. One attempt authorized only after41 genuine0 and its sealed exact ledger. No new acceptance inherited.
- TRAIN47 source-only grid draft will use21 newly accepted controls; a static comparison against prior cameras may reject already-too-close poses before costly scene work, while preserving final ranked new-camera spacing and all quality predicates. No47CPU yet.
- Native848x408 RGB/aligned depth, full144 labeled plants, exactlyone eligible petiole and9mm answer, original family splits, robot workspace,4mm AND0.5degree physical novelty remain required. Training not launched. Continue through noon;10k remains infeasible at measured throughput.

## Current release - September 18, 09:46 KST

- **317 accepted full-greenhouse images:** 242 TRAIN / 61 VAL / 14 TEST; 20 original cut targets across 13 source plant families. Latest: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint317_20260918_v1/. Result SHA256 29c531c93acd92b950631106996eb42a3f2654805145b68d29bbb50eacc66309. Stable D pointer updated. Largest original target remains33 images (10.4 percent).
- Added19 TRAIN103 views: target42x12 and45x7. All29 strong frames received actual full-native RGB, junction/first-leaf and14 flagged-context reviews;10 visually uncertain connections (47x8,43x2) held, with no extra group exclusions. Six other raw frames failed unique-target automation. Root read all29 review reasons; validations99549/44730 and exports67209/52917 genuinely zero.
- VAL97 next25 captured, annotation50107 genuinely zero:22 strict,21 strong. Actual visual review under legacy_reuse is pending; none counted.
- TRAIN53 finite48 remains active root38932/owner66044. Next TRAIN89 recovered19 is CPU-ready, not captured/accepted: original56 cap-deferred proposals reconstructed with exact32 saved positive-pose parity; fresh scene/rays passed for all56,37 spacing exclusions. CPU32767 genuinely zero, source contract403f7f4e/helper90578670 root fully reviewed; ready f3f01ffda41cc9a9a069e096c0565aadbd79c5b6ed8173d17c7902eaa4ba0729, latest planned ledger c421b17958a0042f1947bf69d5e1ae921b5f1b84c94d5f746b340dc54490d73d (922 cameras).
- Next source-only work: bounded recovery of32 original TRAIN41 cap-deferred candidates (43x31/45x1; banned38/held46 excluded), and a possible256-proposal/at-most64 selected two-target103 grid from the19 newly accepted controls. No native acceptance is inherited; new CPU attempts require source review and fresh finite admission.
- Exact single eligible petiole, native848x408, full144 labeled plants, 9mm cut point, clear native attachment/first-leaf context, original splits, physical camera novelty and robot-workspace checks remain unchanged. No training launched. Noon10k remains infeasible at measured throughput; continue useful accepted collection.

## Latest accepted release - September 18, 09:35 KST

- **298 accepted full-greenhouse images:** 223 TRAIN / 61 VAL / 14 TEST; 20 original cut targets across 13 source plant families. Native 848x408 RGB, aligned metric depth, no query cue, exactly one eligible petiole and one 9mm cut-point answer per image.
- Dataset: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint298_20260918_v1/. Result SHA256 f2803e072df80e6e7ceb4b6b823b8ceff28ee41eea220c038cec3e2d78cc1378. Stable D LATEST_FULLY_POPULATED_UNIQUE9MM.json points here. Training not launched.
- Added 21 VAL29 views after all32 raw frames and native junction/first-leaf contexts were actually reviewed. Eleven clipped parent-junction holds excluded, all32 first-leaf connectors visually clear; no visible alternatives flagged. Root read all32 reasons; both export sessions59396/20365 genuinely closed0 (12+9), with actual frozen validations78070/94488 zero.
- Latest browsable annotated preview: data/sim_data/dataset_reviews/fullgreenhouse_joint277_gallery_20260918_v1/index.html, fixed earlier277 subset. Largest source target in298 is33 images (11.1 percent); clones are not counted as new identities.
- TRAIN103 native35 completed root24232/owner73896 zero; annotation88352 zero gives29 strict/strong (targets42x12/43x2/45x7/47x8), six multi-eligible42+43 holds. Actual visual QA is ongoing.
- VAL97 native25 completed root33379/owner74536 zero. Annotation50107 active; none of these25 counted yet. Current native48 TRAIN53 is root38932/owner66044 under finite request e2142ffebde064063a5f81b69a4d3db521ab1c0e975e9f9dae7be8b5b932b2e6.
- TRAIN101 new16 strong views all held for visually uncertain short first-leaf connection; actual validation79951 zero, no export. TRAIN67 new31 also held. No old-control reassessment or source-identity inflation.
- Available memory improved to55.62GiB after Chrome cleanup; recent free disk C261.02GiB/D58.26GiB. Preparation and capture overlap. Native rendering and view clarity remain limiting.
- Next bounded supply work: deterministic recovery of56 existing TRAIN89 proposals previously deferred by selection cap, followed by possible32 TRAIN41 cap-deferred proposals (43x31/45x1; banned38/held46 excluded). These are opportunities, not qualified or accepted views; current-ledger camera spacing, fresh scene preflight and actual native/visual checks remain required.
- Noon deadline remains. Current measured throughput cannot reach10k by noon; continue useful collection without lowering the single-answer, clarity, native resolution, embodiment or diversity requirements.

## Current accepted release - September 18, 09:16 KST

- **277 accepted full-greenhouse images:** 223 TRAIN / 40 VAL / 14 TEST; 20 original targets across 13 source plant families. Native 848x408 RGB, aligned metric depth, one eligible petiole and one 9mm cut point; no query cue.
- Latest release: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint277_20260918_v1/. Result SHA256: 5277d639a9909d8b329f0cf76bc46231641dd5bebcc8780daf47053735bcd996. The stable D-drive LATEST_FULLY_POPULATED_UNIQUE9MM.json pointer now resolves here. No training was launched.
- Added 21 TRAIN47, 9 TRAIN19, and 7 TEST31 images to the previous 240. Root inspected all actual review reasons; the three combined export sessions 2979, 45002, 13695 genuinely closed with code 0. Annotation and held-out first-leaf/parent-edge holds were preserved. Assistant visual review is not a human review.
- All 31 strong new TRAIN67 views were held: the short first-leaf connection remained visually ambiguous in dark leaf overlap. Root separately viewed one exact native RGB and lossless leaf-context crop and concurred for that case; root did not claim to personally inspect all 31. Earlier accepted controls were not reclassified.
- The original noon97 queue genuinely closed 0 after TEST31x9, TRAIN101x21 and VAL29x32. A deliberate STOP_BEFORE_NEXT preserved its unstarted TRAIN103x35. Following user's Chrome-tab cleanup, physical free memory measured 60.606 GiB; the same already-reviewed TRAIN103 batch resumed alone under request 3bec7a1dde8ecbe3f386e4af0484d9550e637d14267b3f9b369fd226e1781dc5, root session24232.
- VAL97 attempt3 finite geometry preparation genuinely closed 0: 25 eligible distinct poses out of 256 proposals; 243 met geometry checks, with 218 excluded by unchanged physical camera spacing. This is 25 prepared poses, not captured or accepted images. TRAIN53 next finite preparation started once after the exact VAL97 ledger was sealed. TRAIN101/VAL29 actual visual QA runs alongside native capture.
- Earlier memory-failed attempts remain recorded. The noon deadline and previous missed 6am deadline remain explicit. Measured renderer throughput cannot reach 10k by noon; continue collection without counting raw, ambiguous or duplicated frames as accepted.

## Current execution - September 18, 08:45 KST (noon extension)

- **240 accepted full-greenhouse images**:193 TRAIN /40 VAL /7 TEST,20 original targets/13 source families. Latest C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint240_20260918_v1/; result 9f3f81cacc1adb8dec33f4363815fa9d1d8402f0dbbc67382d783fc2af967913. Existing212 gallery remains a fixed earlier subset; stable D pointer now240. No training launched.
- Added28 TRAIN53 images after all28 native RGB/crops and9flagged background contexts were actually reviewed. Source30 captures yielded30strict/28strong;2clarity holds remain. Root readall28 reasons and exactreviewpins; actual validation15986/4478 zero, bothrootexports20289/83806 genuinely zero. No rootpersonal-image-view claim.
- Remaining old captures:47x24 annotated23strict/22strong, QA ongoing (first16packet15accept1edgeleafhold);19x19 annotated19strict/18strong, all18packetprepared andQA ongoing;67x32 annotated31strict/31strong, all31packetprepared andQA next.
- Noon native97 queue75841 active:TEST31/42x9 owner86412 genuine0, all9strong annotation92660 genuine0, packet32500 genuine0, actualQA next;TRAIN101/41x21 owner74012 active, thenVAL29/41x32 andTRAIN103 fourtargetsx35. Rootrequest a7e1dd1fbd4edcaeac23169fed43c4fa8e1ff67a29d8083b7a726c1a2c08a890, lateststart11:25KST. Original6am deadline failure is not erased.
- Measured ordinary67batch32:791.66s wholeowner,386.52s native requests (12.08s/image),76.98s scene setup,143.36s sevenwarmups,44.09s geometry including32.06s initialbuild,0.50s totalposeapply,3.38s sourceclose. Existingcamera/scene is reusedwithinbatch; labels are automatic. About146raw/hour overall versus287/hour steady, beforeacceptance.10k bynoon would requireabout2800accepted/hour and is infeasible with currentrenderer; user informed.
- LargerVAL97 source auditpassed29 same-planned-target accepted controls from212,18focusedchecks. Finite256proposal/max64view geometry job failedbeforecreating scene at unchangedmemoryfloor; no newplan/ledger/native/retry. Root readhelperfull.diff (9655149da097bd3981a6a101377ed4209016229a0105110d4961e52ab23921a4 source). Allfrozenquality/body/spacing/200capchecks retained; onlyproposalbudget/bodygrid/source changed.
- CurrentRAM pressure:93.56GiB total, observed9.75GiB free then12.12GiB asjobsdrained. Chrome44processes used45.42GiB working set (46.81GiBprivate), simulator11.05GiBworking. Useraskedoptionallyto closeunusedChrome tabs; no otherappclosed. Root willverifyfreshreserve beforeexplicitnewboundedpreparationadmission.
- Farmerge prototype's2.279x requestgain remains diagnostic; productionrequires newcoarse-ID census/annotation/exportintegration exceeding60min. Conservativevisualregrouping saved2of107reviews, so neitheris a usefulnoondeadline detour. Continueusefulpreparedcapture/QA, thenlargerqualifiedbatches; neverweaken exactlyoneeligiblepetiole or clearnative9mmcut.

## Noon extension - September 18, 08:28 KST

User extended collection until12:00 KST with a10,000-sample target and requested a simpler camera/automatic-label pipeline. Preserve native848x408, distinct embodied views, full144 labeled unpruned plants, exactly one eligible petiole and9mm clear cut. Root resumes processing105 existing captures first; TRAIN47 annotation active95239, TRAIN53 actualQA assigned to coverage. Prepared finite97 native queue now root-reviewed, last start11:25 KST, no automatic retries. Original6am deadline/volume failure remains recorded below; no claim10k is attainable at current measured rate. Performance audit requested alongside useful collection.

## Current execution - September 18, 08:21 KST

**Deadline outcome:** The 06:00 KST deadline and requested15,000-20,000 image volume were missed. Current release is212 validated images, not20k. The live clock was checked at08:18 KST after resumption; the previous status used earlier work state.

- **Accepted release:**212 images =165 TRAIN /40 VAL /7 TEST;20 original cut targets across13 source families. Native848x408 RGB plus aligned metric optical-Z depth/validity/calibration,9mm cut from attachment, exactly one eligible petiole, full144 labeled unpruned plants, fixed-left robot workspace. Assistant visual review, no human verification claim.
- Dataset: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint212_20260918_v1/. RGB, annotations, depth and calibration are under frames/; index.jsonl and split indices provide exact paths. Result SHA ab67170fd3e3967402c9273b16b77515880d1dea9d8118f35c16229567d92f70; manifest SHA a4dc7d98583239e465857cec421658704b504a4e932c8a889640e2f5606c2cd4. Stable D LATEST_FULLY_POPULATED_UNIQUE9MM.json points to212. training_ready means file/package validation, not sample adequacy or permission to launch training.
- Gallery: D:/research/tomato-pi-policy/data/sim_data/dataset_reviews/fullgreenhouse_joint212_gallery_20260918_v1/index.html.212 unchanged RGB previews, toggleable9mm cut/junction overlays; checked212 cards, RGB hashes, counts and split labels. No additional samples created.
- Latest additions:13 TRAIN43 from19 captures and26 TRAIN89 from32 captures. All39 individually visually reviewed and exported with genuine0 exits; parent-edge/first-leaf/wholegroup holds excluded. Root read review reasons and verified pinned review provenance, with no claim root personally viewed these39.
- The assembly completed and updated the stable pointer. Its enclosing script then failed only at a stale documentation-heading assertion (03:12 versus actual03:11). This morning handoff independently verified result/manifest/index/pointer/counts and repaired the documentation; immutable212 assets were not rewritten.
- **105 additional captured frames are NOT accepted:**53x30(owner70496),47x24(owner33628),19x19(owner70076),67x32(owner59760). Root observed finite queue61952 genuinely close0 and all owned exits0.53 has all30 strict-unique automated results,28 strong candidates and two prepared review packets, but0 completed actual visual reviews.47/19/67 still need whole-capture annotation and QA. Exact trial paths and closure pins: D:/research/tomato-pi-policy/data/sim_data/diagnostics/native848_morning_status_20260918_v1/result.json.
- **97 CPU-qualified views remain unrendered:**TEST31/42x9,TRAIN101/41x21,VAL29/41x32,TRAIN103 fourtargetsx35. No native launch for these plans. Latest planned ledger2de067b13675510a384bb7d0cac446542d1793b254f056bb5bb2e4fc330b713c has830 distinct planned cameras, maximum70 per original target. CPU qualification is not visual acceptance.
- Parallel review/preparation agents legacy_reuse and native_recovery2 returned account usage-limit errors. No fabricated review, further worker launch, or hidden acceptance increment. Native queue is closed; no training launched. All captures and preparation artifacts preserved.
- Preserve prior exclusions: banned seed41/SubStem38, uncertain original/new41/SubStem46 first-leaf connections, multiple/unknown eligible alternatives, unclear junctions, original split separation, global200 per original target, >=4mm camera translation AND >=0.5degree target-ray separation. Clones are not new source identities. No nearest-target selection.
- Resume priority: process the already-captured105 before rendering more. Use frozen consumer6 -> packet7 -> genuine group/flag review -> DS8 -> assembler6. Native proposals or projected geometry must never substitute for actual visibility. See exact pending cases/pins in the morning receipt.

## Current execution - September 18, 03:11 KST

- **173 validated full-greenhouse images**:126 TRAIN /40 VAL /7 TEST,20 original targets/13 source families. Native848x408 aligned metricRGBD, exactly one eligible petiole,9mm cut, full144 labeled unpruned plants, fixed-left workspace, actual assistant visual QA. No human or training-run claim.
- Latest: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint173_20260918_v1/. Result 5b3534275bdaa08bfd99d60949b0c96de9b954a979cc8a6ffbab3ba591dc0cdb. Stable D LATEST_FULLY_POPULATED_UNIQUE9MM.json now points to the corrected173 and training_ready=true means files validated, not sample adequacy or training authorization. Sources: data/sim_data/diagnostics/native848_fullpop_joint173_assembly_20260918_v1/sources.json.
- Replaced held original41 checkpoint14 with corrected11; three original46 images removed after actual6-view leaf-connection consistency check. Original11 actual reviews preserved exactly, no new-view claim, all3 held groups single-member, validation22380 actual0; root correctionexport95628 actual0/93.89s. Affected historical checkpoint/unions104/118/121/128 remain held, raw files/history preserved. Old gallery118 has warning. Root personally inspected original46three/41two.
- Added29 TRAIN41 views across42x6/43x12/45x11, all3exports actual0;10 visualholds preserved. Added19 TRAIN17 views, actual20 reviewed,1wholegroup hold117; export49761 actual0. Total125 surviving prior +29+19=173.
- Native105 root24671 ended1:43expanded19 and89expanded32 completed0;53expanded30 owner83732 failed BEFORE native launch at unchanged process gate. Sole blockerPID69636 python had missing executable/command identity; now absent, exactidentity unresolved. Memory/GPU/disk passed. No53 captures/no47start. Failure preserved; root preparing a reviewed finite continuation after currentexports settle, no guardchange or automaticretry.
- TRAIN43 annotation19:16strong, actualQA underway (14 individual accepts/2holds pendinggroupvalidation). TRAIN89 annotation32 active97143; no additional acceptedclaim.
- Ready next:53x30,47x24,19x19,67x32; TEST31/42x9 independently CPU-ready. All originalsourcecamera/physicalspacing/cap/full144 checks preserved. TEST31 finalledger0ebf58dfb4a83246344e8eec6b67945b30924b9768d6968f8f06e234c4e906aa contains742 distinct planned cameras/globalmax70. NewTRAIN7 target42 finite48-proposal supply underway; ≤12selected and no secondfamily inferred.
- Continue to06:00KST;15k-20k infeasible at measured throughput. Do not relax exactly-one requirement, inflate identity counts or include heldframes.

## Current execution - September 18, 03:04 KST

- Latest128 is temporarily marked TRAINING_HOLD / training_ready=false while a corrected release is assembled. A targeted actual6-image old/new comparison could not resolve the first attached-leaf connection in original seed41/SubStem46 frames002/004/006. Root personally viewed all3 original46 and both original41 RGBs; original41two remain clear. The3 original46 are held conservatively, and new46 holds remain. No nearest-target selection or geometric threshold changed.
-125 of prior128 remain usable.29 additional TRAIN41 frames passed actual full-native/crop review and DS8 export: targets42x6,43x12,45x11. Exact native capture46 yielded39 strong;10 visualholds (46x9,41x1). Existing source14 checkpoint and affected unions104/118/121/128 have reversible holdsidecars; raw files and previous reviews preserved. Old gallery118 carries a historical hold warning.
- New TRAIN17 expanded24:20strong,19 actual visual accepts and one held117 for an unclear far-left leaf join. Root readall20 reasons; export19 running49761. Corrected original41 checkpoint expected11 after whole-group validation. Expected next union173 only if all exports complete (125+29+19), not yet assembled.
- Active native105 root24671: TRAIN43 expanded19 owner20044 completed0; annotation77846 completed0,18strict/16strong, actual QA assigned native_recovery2. TRAIN89 expanded32 owner90616 active, then TRAIN53 expanded30 and TRAIN47 expanded24. Fixed request681ac265e823d90be2f57a8363e1346a1a1c6b128cac3e65b6f4e33247cb31ec in native848_registry_train43_89_53_47_expand105_launch_20260918_v1.
- Prepared next51: TRAIN19 target42x19 and TRAIN67 target42x32, exact admitted controls, distinct physical cameras, bothCPU child/wrapper74640 actual0. Final reconciled ledger f357d57b0a8074918d91a74139fde0c085bbd56869af1cee176f78623cf96a29 at native848_fully_labeled_bulk_prepare_20260917_v1/prior_schedule_after_reviewed_expand_seed67_full_v1.json;733 exact planned cameras/globalmax70. TEST31 target42 expansion at most24 is separate bounded96-proposal CPU work; reconcile before rootnative. Root readsource19/67 and TEST31 diffs.
- TEST31 newtarget40 risk-only3: native83148 completed0 but all3 had multiple eligible petioles (39+40,39+40+41,39+40), zeroaccepted. Complete native geometry confirmedprimary40visible; exactly-one requirement correctly holds them.
- Bounded new-family scouting found only TRAIN7/SubStem42 as defensible unattempted anchor supply among remaining source families. Root authorized one48-TOTAL proposal trial at two originalanchors, max12selected, risk0/1/allgates unchanged. No secondfamily invented; no native launch yet.
- User deadline remains06:00KST. Continue useful validated collection at native848x408/full144/unpruned/exactly-one9mm/noquery/robotworkspace.15k-20k infeasible at measured rates; no manufactured count or diversity. Capture, CPU selection, actual visual QA and export proceed in parallel; onlyroot launches oneGPU native owner.

## Current execution - September 18, 02:17 KST

- **128 validated full-greenhouse images**, 21 original targets / 13 source families:81 TRAIN /40 VAL /7 TEST. Native848x408 aligned metric RGB-D, exactly one eligible petiole,9mm cut, fixed-left robot workspace, visible attachment/leaf association, all144 labeled unpruned plants. Actual assistant group/flag review; no human review or training-run claim.
- Latest release: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint128_20260918_v1/. Result 65317880aaf8cb3fe29480724c138678df3f2ffd61c6abb5cfd3cc6e0fd6374f. Stable D data/sim_data/training_exports/LATEST_FULLY_POPULATED_UNIQUE9MM.json updated. Assembly sources: data/sim_data/diagnostics/native848_fullpop_joint128_assembly_20260918_v1/sources.json. Existing gallery118 remains a fixed subset; new preview pending.
- Added2 TRAIN47/SubStem41 and5 TRAIN53/SubStem41. Both native queues and annotation jobs closed0; all7 actual full-native RGB/exactcrop reviews accepted, with supplementary leaf/context review where needed. Root exports87773/26551 completed0 in41.41s/55.30s. Historical sparse121 remains a different preserved dataset, not included in128.
- Active finite73 capture: root2131, TRAIN41 expanded46 (five targets) then TEST31 newtarget40 risk-only3 then TRAIN17 expanded24. Owner91552/native84836 started first41 after all process/resource guards passed. Exact request91d5bc06a14df00d7a6f75af5798977ed869170dbdd2d57234454432155a6bfe at data/sim_data/diagnostics/native848_registry_train41_TEST31_train17_73_launch_20260918_v1/request.json. No accepted inheritance or automatic retries.
- Prepared next:19 distinct TRAIN43/SubStem42 views, fullCPU3 child0 and physical novelty checked; plan0dac779fe36ff1963197a217672e4de8af29599e6392299dfe0467fdf9bf7bec,CPU4fc50590d0b6d1df9a21d6f3ef7a5abc254e660c2c20dbbccba59278b04b9dac. Native89 expansion capped32 and TRAIN53 expansion capped40 are independent bounded CPU work around actual accepted controls; not counted.
- Proposal calibration: high-risk89/47 supplemental views yielded0strong, largely real competing reachable petioles. TRAIN53 primary5 allstrong/visuallyaccepted; supplemental6 allheld. Risk1 is family-dependent: originalTRAIN17 had12 accepted of13 risk1 candidates; final native census remains authoritative. Queued17 is unchanged.
- Bounded unused-target TEST59/61 and TRAIN23 searches yielded no low-risk CPU plans; geometry overlap, detail and leaf framing failures remain real holds. TRAIN71 five actual views remain held for obscured leaf route; three alternate source cameras failed to prove the complete route, no new71 plan.
- Current proposed global ledger before89/53 expansions: data/sim_data/diagnostics/native848_fully_labeled_bulk_prepare_20260917_v1/prior_schedule_after_reviewed_expand_seed43_full_v1.json,59e16cecd576b606b7e97ddaa199a54e988f69eccb3cf317b112bbf708a6edac.596 exact cameras, originaltarget200cap, supersessions preserved. New branches must reconcile.
- Deadline September18 06:00KST remains active.15k-20k is infeasible at measured throughput/yield; continue maximum useful validated collection without relaxing exactly-one eligibility or inflating target diversity.

## Current execution - September 18, 01:50 KST

- **121 validated full-greenhouse images**, 19 original targets / 11 source families: 74 TRAIN, 40 VAL, 7 TEST. Native 848x408 aligned RGB-D; exactly one eligible petiole; clear 9mm cut, attached leaf route and fixed-left workspace. Full144 labeled plants, no query cue, actual assistant visual review. No human or training-run claim.
- Latest release: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint121_20260918_v1/. Resultebaffb71ebe914c326320275b392ce8705a6c5be2e487e70cc3bc8ba65fa5b3b. Stable D LATEST_FULLY_POPULATED_UNIQUE9MM.json updated. Fixed118-image subset gallery: data/sim_data/dataset_reviews/fullgreenhouse_joint118_gallery_20260918_v1/index.html. Assembly sources: data/sim_data/diagnostics/native848_fullpop_joint121_assembly_20260918_v1/sources.json.
- Added14 visually accepted TRAIN17 and3 TRAIN89/SubStem39 frames to prior104. TRAIN71: five strong geometric candidates all visually held because overlapping near leaves obscure an identifiable attached-leaf route; zero added. Historical51 remains under TRAINING_HOLD (48 individually requalified in current release; three multi-eligible excluded). Sparse121 is a separate historical dataset, not included in this fully populated121.
- TRAIN89 native23 actual root42182/owner57276 finished0,726.76s; resultc4342558696e91bea5b36a063b3fc6d9cfb1d209ab55499ae0b1cd96faf2ef81. Annotation16 root47272 completed0:23 evaluated,12 strict and3 strong. All3 actual assistant fullRGB/junction/leafcontext accepts, exported root69947 complete0 (46.44s) and assembled121. All17 risk-only43/44 proposals remain held for real reachable alternatives; zero unknowns. Summary7d4b9325f2be955a6db7f1dad3969e1e78710c6806d3f311fd488c348f8fb44a.
- Native root33879: corrected direct finite25 queue, TRAIN47(14) then TRAIN53(11), owner46588/native38056 started after all resource/process checks passed. Request41afbbc379ba029ccf4461d2364760fa1637b07eb8f05cf55034386c946e8113, data/sim_data/diagnostics/native848_registry_train47_53_direct25_launch_20260918_v1/request.json. Priorroot8362 failed before native start (unknown unparseable Python command89396, now exited); no captures, failure evidence preserved, no guard weakened. Root reviewed exact prepared source code and genuine CPU closures. Supplemental risk-only proposals retain identical native eligibility checks.
- Ready next:46 distinct TRAIN41 views across five existing targets, all finalCPU3 passed;33 mesh-risk0/13 risk1, min4.15mm camera origin and0.524deg target-ray separation against prior and new same-target cameras. No new original targets/families claimed. Original source seed41/SubStem38 excluded. Reconciled global ledger a6b5e396007c4093106af742d7c54e2031d031cd9e79a90a04f29a38ba840e96 preserves combined89/47/53 and supersessions. TEST59/31 supply preparation running independently.
- Automatic approval review service temporarily rejected new commands due review-model capacity; now recovered. Existing authorized capture/CPU jobs completed safely during interruption.
- Deadline remains September18 06:00KST.15k-20k is infeasible at measured throughput/yield. Continue useful validated collection without repeating cameras, inflating clone diversity or changing exactly-one eligibility.

## Current execution - September 18, 01:13 KST

- **104 validated full-greenhouse images**, 17 original targets /9 source families,57 TRAIN /40 VAL /7 TEST. All native848x408 aligned RGB-D, current epoch2 entire target census, exactly one eligible petiole,9mm cut, full144 labeled plants, fixed-left workspace and actual assistant visual review. No human or training-run claim.
- Latest dataset: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint104_20260918_v1/. Stable D LATEST_FULLY_POPULATED_UNIQUE9MM.json points here. Exact result a9d9dab6cc0f5f200511aec424cf025979e94d17fc3efb67d489b6153efecd98. Assembly sources data/sim_data/diagnostics/native848_fullpop_joint104_assembly_20260918_v1/sources.json. Gallery90 is a fixed prior subset at data/sim_data/dataset_reviews/fullgreenhouse_joint90_gallery_20260918_v1/index.html.
- Composition:48 freshly re-reviewed/re-exported survivors of historical51;10 newTRAIN43;29 newVAL97;3 recoveredTRAIN103/SubStem43;14 newTRAIN41 across targets41,42,43,45,46. Historical51 has TRAINING_HOLD.json and README notice, three multieligible103frames permanently excluded from that selection. Sparse121 remains separately preserved and is not in104.
- Recovery247 historical frames: after removing60 already reaudited, bounded18 ownership-affected103 frames re-evaluated.5strong;4individual visual accepts but a held group excludes one other, leaving3exported. Other positive parent-union/proximal paths checked, no further clean recovery candidate. All actual depth/alternative/leaf holds remain.
- Native25 queue ROOT84397 completed both15 TRAIN17 and10 TRAIN71 with actual owner waits0. Owners82348/42020 took584.37s/548.82s. TRAIN17 current annotation15strict/14strong, actualQA in progress; summary638b38f20d9606e857f42e103a617ee00781c7bb1303dc37d0e4a8a5d48092c3. TRAIN71 annotation5strict/5strong; summary78231f06639d9a57412283ff0b2507207d1230fadb45376d3ae434bc1d0af8a6, visualQA next. Neither counted yet.
- Native currently closed while first combined new-family CPU preparation finishes. Original primary89/47/53 plans remain uncaptured. Combined89 proposed23 across3targets includes6priority and17geometry/junction-clear poses held only by conservative alternative-mesh ranking. Risk-only candidates get identical actual native epoch2 uniqueness checks; no acceptance weakening. Full CPU3 fresh proof and original primary parity required. Combined47/53 follow. TRAIN41 additional?60 proposals are independently being prepared from exact14 actual strong camera states; global ledgers require reconciliation, no duplicate captured cameras.
- Merged-render paired diagnostic found2.279x lower native request time, exact retained component masks/depths and no visible near-target regression. Rare far-only depth/alpha differences remain documented. Fullowner900.13s; production integration estimated60-120min for onefamily and is deferred tonight. Ordinary renderer retained.
- Deadline remains September18 06:00KST. Continue maximum useful validated collection.15k-20k is infeasible at measured throughput/yield; do not promise that count, repeat poses, inflate clone diversity or relax exactly-one rule.

## Current execution - September 18, 00:36 KST

- Exactly one eligible petiole remains mandatory. Full 144-slot labeled unpruned greenhouse, distinct native 848x408 RGB-D cameras, fixed-left workspace, visible 9mm cut and leaf association, no query cue, no nearest-target fallback.
- Corrected anatomical junction ownership is now qualified as annotation epoch2. Only source-authenticated immediate main-stem ancestor branch stubs sharing the actual petiole mesh seam qualify; depth and clarity thresholds remain unchanged. Root and independent agent reviewed the frozen consumer changes; 42 geometry, 28 reconstruction and 39 consumer checks passed. Whole target censuses, including alternative targets, are recomputed.
- Historical51 full census re-audit completed:48 retain exactly the same sole answer/geometry/lineage;3 TRAIN103 frames (_005,_006,_011) now show two eligible petioles42+43 and are excluded. Historical export remains preserved with training hold notice. Audit: data/sim_data/diagnostics/native848_accepted51_joint_reaudit_execution_20260918_v1/result.json SHA56d5e2162eeebe15c5149b296856f9df7a8382ab5961cc2c383c91fd57933553. Fresh actual group reviews and current-epoch export of the48 are in progress.
- Current newly qualified release:10 TRAIN43 frames, one original target/family, actual full-native RGB plus unscaled junction crop review, no human review claim. C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_joint10_20260918_v1/. Result6611348951f7991fdc725d24e5ae23e6b9869f42e6b34c0466011a10c251c7c7; manifest825266c4535750974d7b88b4e14b14263943f2101063626d6e38988b1a749a4a. Stable LATEST_FULLY_POPULATED_UNIQUE9MM pointer now refers to this qualified subset while prior48 are re-exported. Not58 accepted yet. Sparse121 remains separate and historical.
- TRAIN43 native12 replay:10 strong,10 actual visual accepts; one genuine depth failure and one two-eligible frame stay held. VAL97 native30 completed:29 strong, visual review/export in progress; not counted yet. TRAIN11 prior31 remain0strong; mesh ray analysis confirms actual foreground parent surface, not a depth calibration error.
- Active native ROOT session96811 / owner22988 / native15372: TRAIN41 priority23, six original targets41..46, permanently banned38 excluded. Queue3 request2a4c3514e336905a42e4bb135755eb852b74caac99b02c3bdd564b5449a51184. Next prepared25 distinct poses:15 TRAIN17/SubStem41 and10 TRAIN71/SubStem43. Pending actual preceding closure; no launch/count yet.
- Merged far-render diagnostic pair completed ROOT40379/native0, owner10952, outer900.13s. Result e73534975ec1e0c86cf26a52a113317d505f187c0e1ca38e834feca56fc0ab4f. Two exact repeated reference poses add zero samples. CPU comparator and actual visual equivalence review are in progress; no production speed/quality qualification yet. Native ordinary capture continues independently.
- Work continues toward September18 06:00KST. 15k-20k is infeasible at the measured throughput and acceptance yield; do not manufacture count, relax uniqueness or inflate clone diversity.

## Current execution - September 17, 23:56 KST

- Accepted release remains51 full-greenhouse images (9 targets,6 families;33TRAIN/11VAL/7TEST), plus121 separately preserved sparse images. Stable pointer and accepted51 preview remain current.
- TRAIN11 priority31 completed actual root70955/native exit0, outer764.70s. Owner result45a832b154bf9974202c0896ad33db4d538e7289e6e9d8a7d184a8aa23c8f415; terminal9588b8e379863711850c59df7cad5fdc95fd813b582233873244527b976a1bd7. Annotation15 actual31 root74043 completed0:11strict single candidates,0strong candidates. No exportable frames under current checks; saved summaryac4f624218b71768142b4e72c7d69702c70a361e2a56069f3d2580a767b1b06d.
- Active native root58483, owner2856, case seed97_all144_labeled_parentmesh30_registry_v1:30 distinct VAL97 camera views. Root requestdb156703d52e4ac83ad1b25b54aa4799756e73a1ee8549f4aa2084e3a6087a64; exact actual completed31 predecessor pinned. Prepared full144 CPU proof unchanged.
- New correctness investigation: proposed seed43/SubStem42 all12 fail attachment ownership in arcs0-1.5mm. MainStem27 authors a secondary branchstub ending exactly at that attachment although declaredparent isMainStem28. Ten seam vertices match petiole source within1micrometre; actual RGB shows a continuous junction. A bounded source-authenticated junction ownership helper/evaluator successor is being prepared; frozen current15 consumers remain unchanged and no reacceptance is authorized yet. One43 probe also has depth-foreground evidence and must retain its separate check.
- Root first16 TRAIN11 diagnostics similarly found allattachment failures:40 depth-consistent probes on component35803 and29 foreground-depth probes on35804. This is not evidence that all holds are erroneous. Unchanged depth/clarity/ambiguity requirements remain mandatory, and any corrected evaluator must rerun allcandidates including alternatives, then audit accepted51 for changed eligibility.
- Rendering prototype integration remains pending. First CPU integration got through actual144/11177 render census/material verification and first originalcollision pose, then failed in a test-harness expectedproof nesting lookup. Failure preserved; narrow harness successor running. Immutable source/reference validation alone took298.19s and109.6GB reads, an additional measured startup overhead. No native merged capture/speedup claim.

## Current execution - September 17, 23:43 KST

- **51 accepted full-greenhouse images**, 9 original targets across 6 plant families: 33 TRAIN / 11 VAL / 7 TEST. Preserved sparse121 stays separate: 172 accepted across scene types, only51 meet populated-greenhouse requirement. All51 actual assistant full-native RGB and unscaled junction crop reviews; no human verification claimed.
- Latest release: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_accepted51_20260917_v1/. Result064f9798936982f2d51b84a8e858a2f863b6a1b14945acf7c0ea03e4fbae7da7; manifest42f7fef8a25f13549355cc877802cdb2879b0b2d87d54d2b132197d4d4700d7c. D stable LATEST_FULLY_POPULATED_UNIQUE9MM pointer updated. Fixed gallery: data/sim_data/dataset_reviews/fullgreenhouse_accepted51_gallery_20260917_v1/index.html.
- Since accepted31: TRAIN101 added2 (SubStem41), TRAIN19 added10 (SubStem42), limited production6 TRAIN67 added8 (SubStem42). All export/checkpoint/union checks completed. Global200/original-target, RGB/camera uniqueness, split separation preserved.
- TRAIN43_12 capture actual native/root0, root54355 closed; owner6 result535f8dbf1acb61a747e68f60715cc491e59bdeb01a78cc87d3ba8b7befd6982a. Actual annotation15: 11 zero-strict,1 strong; actual sole candidate was SubStem43 despite proposal42. Actual visual review holds it because leaf exits top before association is visible. Zero added samples.
- Active capture root70955 owner78156/nativecommand69176: seed11_all144_labeled_priority31_registry_v1, queue2. Request2aa293d251d8387ac6424125c63a39fbdd0ee662e94c89799a5095a968c52d52.31 existing distinct proposals (41x20,44x11) selected from uncaptured55;24 omitted poses remain uncaptured, not native holds. Fresh31 CPU proof passed. One historical anchor, one interior label position; no independent-target inflation.
- CPU-ready VAL97 expansion30: plan1b6b15a92f8bb9028c2e32d115d87679baa39567fadeffe262c909be48d57f22; CPU743a1c6d60975357dc3ff6ee818115b5c72909e8075b99283b9a337654f75305.30 additional distinct camera poses, one original target42; origins7.19cm/rays11.37degrees, one interior view. All mesh-risk0 and estimated parent clear; final native visibility/uniqueness/visual checks remain pending.
- New conservative parent-surface ranking caught all30 actual distance failures in120 saved foreground evaluations, with1 overflag among90 passes. It is only a proposal priority; it does not prove proximal visibility or uniqueness. The failed43 batch reinforces this limit.
- Root reviewed new fixed predecessor registry/owner6 and compatible annotation15/cache8/consumer stack. Geometry and eligibility functions unchanged.43 focused registry tests and29 consumer tests passed, actual12 owner6 annotation succeeded. Registry supports exact typed historical closures; no automatic retries or source acceptance inheritance.
- Far-component merge CPU prototype reduced render mesh primitives66,643 to11,177 while preserving all144 logical plants, source anatomy/materials/UVs and original collision representation. Foreground/near geometry unchanged; far coordinate bake error0.132micrometres under explicit1micrometre bound. Root read packer/closing audit. Native two-frame comparison is being prepared; no speedup or production qualification claimed. Diagnostic repeats add zero samples.
- Exactly one eligible petiole remains required, with 9mm cut, native848x408 RGB-D, full unpruned labeled greenhouse and fixed-left workspace. Nearest-target selection is prohibited. Collection continues toward06:00;15k-20k remains infeasible at measured rate/yield.

## Current execution - September 17, 22:58 KST

- Accepted release remains31 full-greenhouse images (7 original targets,4 families;13TRAIN/11VAL/7TEST), plus121 preserved sparse images. The accepted31 gallery and stable pointer remain current.
- Finite71 completed all13 TRAIN101 +58 TRAIN19 frames. Actual root session51283 and both native waits exited0; whole coordinator lock released. Parent result ed709aff583d2be74542882913817d46e3f277189ee57cd812695680f1725009; terminal14d9373520623810e6c647303ccdd218dfa99c30baa4984dd443168f3b4462f6. Two workers ran serially after memory admission; total outer1721.69s, no parallel speedup claimed.
- Root authenticated both actual slot closures and created annotation_parent_wrapper.json in the71 root launch folder, SHA940940fd11c05e6d950ebc797e95cbef9f69ba4ae95456021da80b3e8e03e18c. Annotation14 actual first13 completed:11 strict-single,2 strong automated, visual/export pending. Frozen cache7 firstchunk64.95s. Slot58 processing follows; no accepted increment yet.
- Active root native session94897, owner40768/nativecommand15024, case seed67_all144_reset6_production36_v1. Launcher SHA0a9425ff1ee17089375d2b6d78503a6f067640ec6d4aadb6742fe3b4db615971. Root read full template and worker/owner diffs, then instantiated only five actual predecessor pins. Prepared36 source cameras on TRAIN67/SubStem42; same full144 scene,7warmups at8 and production6. New native/QA evidence pending.
- New annotation14/cache7/base4/group5/dataset6/batch6 passed57 focused checks and actual completed oldserial closure. Helper4/packets5/QArecorder3 passed25 checks, then actual13-slot annotation end-to-end. Root read allnine diffs. Quality/geometry predicates unchanged; two-slot completion is authenticated explicitly, without pretending slots held a serial lock.
- New assembler4 accepts frozen DS3/4/5/6 checkpoints and uses DS6 model loader; global uniqueness/source-target200cap unchanged. Root loaded all31 existing model inputs with DS6. SHA4ae7990d7edfb53812d51d8e7fe437097ec0f74abb4618032ef36fda94d67dad. No new union yet.
- CPU-ready newTRAIN43: originalcontrol1 plus12 distinct additional camera poses; all12 mesh-risk0/1,6interior, one source target/one historical anchor,3.83cm origin and6.86degree target-ray spans. Minimum pairwise camera movement5.06mm/ray0.617degrees includingcontrol. All final fullscene/FK/leaf-route/detail/workspace proofs passed; actual native/leaf visual review still pending. Next finite family search is TRAIN71.

## Current execution - September 17, 22:38 KST

- **31 accepted full-greenhouse images** across **7 original source targets and 4 source families**: 13 TRAIN / 11 VAL / 7 TEST. The 121 accepted sparse images remain separate (152 accepted across both scene types; only31 meet the current full-greenhouse requirement).
- TRAIN10350 produced14 strong automated candidates. Actual assistant full-native RGB and unscaled crop review accepted12; two remain held because the actual candidate petiole leaves the image before its attached leaf can be verified. Added source targets: seed103/SubStem42 x10, SubStem45 x1, SubStem47 x1. No human review is claimed.
- New12 checkpoint result: c2c4637a58889f084e221146f528bd53fde8279425ca66bfead589e916aa9fd3. Combined release: C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_accepted31_20260917_v1/, result c7816a340a16b0a0f84cb2e15ff3641ff08a1dfe41fd45708dd040703ecef12e, manifest0742ffadab615f5f58c1499e56b76f62d69b2e07796c962817831c8cd5dc708e. Stable LATEST_FULLY_POPULATED_UNIQUE9MM.json now points here.
- Accepted31 gallery: data/sim_data/dataset_reviews/fullgreenhouse_accepted31_gallery_20260917_v1/index.html. Unchanged RGB copies with optional SVG cut/junction overlays; preview adds zero samples.
- Active finite71 root session51283: TRAIN101 slot0 saved13 frames and naturally exited0. TRAIN19 slot1 is capturing after serial fallback; whole coordinator/root closure still pending. Second-worker concurrent admission did not fit actual available physical RAM. A separately measured resident-memory model also missed the4GiB physical floor; active gates remain unchanged.
- Actual production8 requests take12.48-13.61s; callback normalization0.129-0.159s, copy0.0016-0.0056s, lossless saving0.076-0.089s, post-first geometry0.36-0.47s. Most delay lies in synchronous simulator render/app updates. A short production resource sample showed about1.13 CPU-core equivalents and1% GPU activity; this does not identify the exact internal bottleneck.
- Root has read the complete production6 worker/owner diffs and future launcher template. A limited36-frame TRAIN67 rollout is ready once71 actually closes. Its six-subframe diagnostic requests were~9.5s, with exact depth/label/calibration parity and actual visual QA over the finite diagnostic poses. Every rollout frame must pass the same full-census, exactly-one,9mm and visual checks.
- New CPU-ready TRAIN11 supply:55 actual camera states across targets41/42/44;36 mesh-ranking low-risk,16 interior views. All share one historical anchor, with4.7-5.3cm total camera span and7.7-7.9degrees ray spread; these are correlated views, not55 new targets. Native capture and acceptance pending.
- Annotation14/compatible consumer preparation is delegated for the explicit two-slot closure and truthful production6 profile. Frozen existing sources and accepted releases are unchanged. No nearest-target selection is permitted.

## Current execution ? September 17, 22:22 KST

- **Accepted count remains 19 full-greenhouse images**, across 4 original targets and 3 original families. The preserved 121 sparse-scene images remain separate. User explicitly requires exactly one eligible petiole; nearest-target ranking is not permitted.
- TRAIN103 finished all 50 native captures with actual native and outer exits 0 (root session 35654). Automated review found **14 strong candidates across 3 new source targets**: SubStem42 ? 11, SubStem45 ? 1, SubStem47 ? 2. Actual visual review/export are pending; these are not yet accepted additions. Annotation summary: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_fullpop_CPU_20260917_v1/seed103_central50_annotation13_v1/result.json`, SHA `fb3608e351efe3f81f4a9c17573250954e5117d425413f726d484e9ec3499cca`.
- Root started a finite **71-view collection**: TRAIN101 (13 views, 2 targets) and TRAIN19 (58 views, 2 targets). Root session 51283, coordinator 25644, case `train101_13_train19_58_all144_two_v2`. Maximum two native workers; second admission requires fresh memory/GPU/process checks. If overlap does not fit, it waits for the first child to close and runs serially. Neither throughput nor acceptance is assumed.
- The first 71-view launcher failed before spawning an owner or simulator because its request hash predicted LF text while Windows wrote CRLF. Its partial request is preserved. Additive launcher v2 fixes serialization and uses a new case; actual saved-byte parity was checked before launch.
- New mesh-ray ranking is a proposal priority only. In retrospective checks, all 7 accepted TEST31 views scored 0?1 potential alternatives and 35 held TEST31/61 views scored at least 2. All 58 new TRAIN19 views score 0?1, but native visibility, complete census, workspace and visual review remain mandatory. These 58 views span 7?8 cm camera movement and 8?11 degrees around two targets, with edge framing; they do not add 58 independent targets.
- Limited production6 preparation is ready for 36 new TRAIN67 views. Its actual profile truthfully records seven warmups at eight subframes and production requests at six, keeping renderer settings, resolution and zero-clock behavior unchanged. The old exact CPU geometry proof is explicitly reused, not claimed recomputed. Numeric and actual six-image/twelve-crop review support this limited rollout; a production worker still needs integration and source review. No diagnostic replay adds samples.
- Latest accepted release remains `C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_accepted19_20260917_v1/`. A fixed, accepted-only gallery is available at `data/sim_data/dataset_reviews/fullgreenhouse_accepted19_gallery_20260917_v1/index.html`.

## Current execution ? September 17, 21:52 KST

- **19 accepted full-greenhouse images remain available:** 1 train, 11 validation, 7 test; 4 original targets across 3 source families. The 121 accepted sparse-scene images remain separate. Exactly one eligible visible petiole is still required.
- The 19 new seed61 views completed capture successfully, but none qualify: 12 contain multiple strict candidates, and 7 have one strict candidate plus another visible reachable petiole. These are saved as held examples and add zero accepted images. Diagnostic: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_fullpop_CPU_20260917_v1/seed61_stratified19_hold_diagnostic_v1/result.json`.
- The six-frame reset6/reset8 A-B-A renderer diagnostic completed with actual native and outer exit codes 0. Root session 81211 is closed; result SHA `04c6742d76bc41d0df468cae4b22e6751665ad4e3d2a537a4516e1e5c8d55821`. These are repeated diagnostic views, not new training samples. Pixel comparison and unchanged cut-point quality review determine whether six subframes are usable; production still uses eight.
- **Active native capture:** 50 CPU-qualified views across 6 TRAIN seed103 targets, with 23 interior image positions. Root session 35654, owner 86248, case `seed103_all144_labeled_central50_v5`. Full 144-plant scene, actual robot camera poses and original split are preserved. Acceptance remains pending.
- Accepted19 visual gallery: `data/sim_data/dataset_reviews/fullgreenhouse_accepted19_gallery_20260917_v1/index.html`. All 19 unchanged RGB images can be viewed with optional cut/junction overlays and links to depth, calibration and private annotations. This preview adds no dataset samples.
- Owner5 adds only authentication of the actual latest reset diagnostic predecessor. Root prepared annotation13/cache6 and compatible consumers with unchanged geometry/quality functions; 38 focused checks passed. Existing frozen sources and accepted exports remain unchanged.
- A bounded two-worker capture implementation is being evaluated. The GPU has capacity, but the latest conservative RAM projection failed the required free-memory floor. Admission must use fresh resource measurements after finite CPU jobs release memory. No two-worker speedup is claimed.

## Latest accepted full-greenhouse checkpoint - September 17, about 21:22 KST

- **19 accepted full-greenhouse images:**1TRAIN,11VAL,7TEST;4 original source targets across3 source families. The121 sparse examples remain preserved separately, giving140 accepted across scene types; only19 meet the full-greenhouse requirement.
- Latest release: `C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_accepted19_20260917_v1/`, result24f64f9ca71c0de282c98091567fc876d5aa23601a7d47dce12a945afba524fa. Stable D pointer unchanged. Every accepted fullgreenhouse frame has actual individual assistant RGB/native-crop review; no human review claim.
- Actual target counts: seed67/SubStem42=1TRAIN; seed31/SubStem42=7TEST; seed29/SubStem41=8VAL; seed29/SubStem42=3VAL. Last3 answers are on a background plant instance, established by the complete candidate census; frame source-pose names do not determine labels. VAL11 export96.43s, result449b1cdda3ecd4240e41fc730db81be9387a05906a7449d1f449cad33ac9751f.
- Root session66611 is capturing19 stratified views across5 TESTseed61 targets with owner4/unchanged worker3, case `seed61_all144_labeled_stratified19_v4`, owner89372. Source plan3c94c772c5a4785bab80d966dcfab2afdd88d78c05e97dc134bd44c41d9256cc, CPUec7ac017606de074c5f69f0d13b9c9e9abe266af034618dd883ad4c7824b6390. Additional CPU-ready batches:9VAL97,6VAL13,13TRAIN101; native/QA pending.
- Instancing paired2 completed actual native/outer waits0, root33742 closed0. Resultb66406719795dbdfaee029b45e44fc0eef35e93488da282db78a37aff2123edf. Exact native component/organ maps, calibration, depth-validity and optical-depth pixels match the ordinary reference. RGB differs modestly and no new quality/export approval is implied. Native request total25.56s versus26.26s: only2.7% faster, insufficient. Ordinary renderer remains production; instanced duplicate replay adds0 samples.
- Additive owner4 merely authenticates this actual latest diagnostic predecessor; plan3/worker3/ordinary144 scene and all quality checks unchanged. Owner4 SHA2ba2a07a86dccbf9f02da55668f609f132bfc13527175c262447e0aa51e13a37. Annotation12/cache5 add its exact typed dispatch.35 focused+47 actual parity checks passed, every geometry/decision/proof unchanged except2 measured elapsed-time fields. Old versions remain frozen.
- Root added basevalidator2/group3/dataset4/batchvalidation4 with only evaluator imports/pins/schema changes;29 focused checks, actual annotation12 packet/crop parity and all19 existing model-input loads passed. New assembler2 supports accepted ds3/ds4 checkpoints with unchanged global200/source-target cap, family split and RGB/camera uniqueness. No additional acceptance from diagnostic parity.
- Lower native subframe budgets are not production-qualified: prior1/2 had camera/freshness failures and4 had trace-clarity failures;8 passed. A bounded reset6/8 A-B-A diagnostic is being prepared, not launched.15k-20k by06:00 remains infeasible at the unchanged measured13s/raw-view rate and observed rejection yield; maximize honest usable output without weakening exactly-one eligibility.

## Latest accepted full-greenhouse checkpoint - September 17, 21:01 KST

- **8 accepted fully populated images**, across2 original source targets/families:1TRAIN and7TEST. Preserved sparse scenes remain121, so129 accepted across both scene types; only8 meet the current full-greenhouse requirement.
- Current release: `C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_accepted8_20260917_v1/`; result SHA8ba7706498704e9ff930104c66dce4ab6edd24b837ec25bc2a5faa1bdb37ce90. Stable pointer `data/sim_data/training_exports/LATEST_FULLY_POPULATED_UNIQUE9MM.json`. All8 received actual individual assistant full-native RGB and unscaled-crop review. No human review claimed.
- TEST7 checkpoint result457a1c04f941e08caf99f7010b32fce6a10cfb0fcd1da3481fb1122ed522751d. Root authorized concrete export after native/annotation closure and all7 actual assistant visual decisions; export completed72.595s. Global assembly checks RGB/camera uniqueness, source-family splits, and200 views per original source-target across clones.
- VAL48 completed with actual outer/native exits0, owner result e1dca68e68a96ec53277774f48ef8e2ef159c75452fcc364b638a3583f9c4b2c. Root session74301 closed0, outer1044.36s. CPU annotation11/cache4 is processing16+16+16; no accepted increment yet.
- New stratified seed61 supply:19 provisional fullscene/workspace views across5 targets, including5 interior-quarter labels and13/15 image-position strata; final CPU sealing pending. Some target viewpoints differ by36-42degrees across23-25cm camera translations. Projection of nearby alternatives is only a risk score because it also flags occluded petioles; final native visibility and independent ambiguity gates remain mandatory.
- Instanced scene CPU geometry parity passed via artifact-only closure after preserving a failed exact aggregate-box comparison. All66,643 per-mesh arrays/world transforms/materials/world bounds and both robot collision screens match exactly. Only aggregate plant-root bounds differ by at most5.33e-15m (explicit1e-12m tolerance). Native component-ID, depth, quality and speed qualification still pending; no production renderer change yet.

## Latest accepted full-greenhouse checkpoint - September 17, 20:33 KST

- **1 accepted fully populated image**, plus **121 preserved sparse-scene images** (122 accepted total across the two scene types; only1 meets the new full-greenhouse requirement). First full-population checkpoint: `C:/Users/USER/tomato-vlm-data-20260917/training_exports/tomato_cutpoint_848x408_fullgreenhouse_20260917_v1/`; result SHAeea8f31d0307ec31f40449fdb22857a2b98e30d265cc4504edc9fb8af4659a18. Discoverable through `data/sim_data/training_exports/LATEST_FULLY_POPULATED_UNIQUE9MM.json` onD.
- Both actual2 native frames now have complete11,070-petiole censuses after mathematically proving far targets outside both arm reach spheres. Annotation11 result SHA4fd7452e327473f9a800cc27332fd883867ba1b70b0900b068880356f7f14ab0. First frame remains held for a genuine reachable alternative; second underwent actual root full-native RGB/unscaled-crop review and passed export. No hidden uncertainty or human verification claim.
- Fully labeled consumer source pins: frame_validation1 fdc0d1af840835f2ca07e413f69dcf1938f933211546f1ca6f8d43775a4fedb2; group_review2 f25c7c8f6e94192b5b4c39bbe3c7acd6fc27fd1d0bc46b159d94d1e724ecb0d4; dataset3 ccf544f1db4e1eeb40660512895a065cf7ce958f136c4c4aa10be64b3b5bf511; batch_validation3 51b8532702a0902a2f78b6bb49e9ac391502d1cf270a4142fdae206487a77256.26 focused root checks and23 independent actual/synthetic consumer checks passed. Export27.95s; global accepted union also checks source-target200cap across all clones/checkpoints.
- At20:54 KST, TEST23 has completed native capture with actual native/outer waits0 (owner result592020fd08cedc6d70a4094fce00c6b40c95615d5217d9add4d042b3119f6655). Annotation11 found7 strong candidates, all source seed31/SubStem42, now under actual visual review; the other16 remain held. No accepted credit before review/export. VAL29 balanced48 is actively capturing in root session74301, six original targets; its result and annotation are still pending.
- Initial23/48 templates cluster labels near corners. Future preparation now searches stratified continuous actual head-camera positions over the image, including interior positions where geometry/detail/single-answer requirements pass. Do not expand to20k using only tiny corner jitter. Report UV bins, view-ray spread, original source-target/family counts and accepted counts.
- Native rendering remains~13seconds/view. A same-geometry USD-instancing optimization is under CPU per-mesh geometry/material/collision/static-guard parity review; no speed or native-ID qualification is claimed yet. Native control remains root-only.

## Current measured progress - September 17, about 20:11 KST

- Active goal: **15,000-20,000 usable, distinct-view samples by September 18 at 06:00 KST**; maximize validated output. Native848x408 RGB-D,9mm cut point, no query cue, exactly one eligible visible petiole, full unpruned plants, all144 original greenhouse slots populated. Preserve the121 accepted sparse examples separately.
- Actual full144 native smoke completed2 distinct views of seed67/SubStem42 with true native and outer waits0. Root session84709 closed0. Owner result SHA3839acf625c95389c73eb181bcae7b32a8b1d170e8e1776d24b3a203f0172c8f; capture namespace `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_fully_labeled_20260917_v1/seed67_all144_labeled_smoke2_v1/`.
- Native census matches finalCPU:144 complete labeled plants,61,386 component instances,11,070 enumerated petioles. Root actually viewed the first native full RGB: dense vines are present. No full-population accepted increment yet.
- Annotation10/cache3 actually processed both frames in29.76s. Both have one strict foreground candidate, but32/28 unknown background target evaluations make the full census incomplete. Result `native848_fully_labeled_smoke2_annotation10_20260917_v1/result.json`, SHA0072b0d76d979077df0b73ba0870127adbdb4d5dfdf79cdab13fc911e86cc8ad. Every renderer pixel has authenticated ownership; unknown anatomy evaluations still hold acceptance. Investigate rigorous workspace exclusions, never relabel uncertainty as a negative.
- Actual native timing:370.28s process,77.32s scene setup,148.16s seven warmups,12.37/13.89s production render requests,32.60/0.38s per-pose geometry. Unchanged serial throughput cannot reach15k-20k bydeadline; optimize and report honestly. CPU survey proposes124 source targets across21families, not qualified/accepted counts.
- Source-target diversity keys always use original family/component across cloned background instances. New bulk plans may use200 views/source target, with global accepted-union enforcement and exact camera/RGB duplicate rejection. Additional distinct viewpoints do not create additional independent source targets.

## Active requirement ? fully populated greenhouse, deadline September 18 at 06:00 KST

The user explicitly permits retaining the sparse-scene examples but requires future collection in the fully populated greenhouse as in the simulation. The active user target is 15,000-20,000 usable samples by September 18, 2026 at 06:00 KST, collecting as much validated data as possible. This supersedes the interim 2,300-image pilot count. Required: native848x408 RGB-D, full target plants,9mm cut point, diverse original plant families, one eligible visually identifiable petiole per image, no query input. Existing images are preserved. No training is requested. Distinct camera viewpoints must be spread across original petiole targets and plant families; repeated background clones, duplicate RGB, or tiny image-only perturbations are not new target diversity. Earlier permission for more views per target remains active. New bulk plans may use up to 200 distinct views per original source-family/component target across all instances, with a planning aim of at least 100 targets and 20 source families; actual qualified supply and accepted counts must be reported honestly. No delivery guarantee is inferred from the deadline.

- **Preserved accepted sparse-scene checkpoint:** `data/sim_data/training_exports/tomato_cutpoint_848x408_unique9mm_pilot_incremental_20260917_v6/`,121 images (74 TRAIN /22 VAL /25 TEST),13 original plant families,15 physical targets.119 images individually reviewed and2 explicitly covered by view-group review. Result SHA `768b16cefbde018be9f7950f5e2481164ae2e5a674d1a34a6301520ec56b6835`; manifest SHA `e671f32900f329e4fd4c30bd911d5dce50742af404862cae5a65792d34aa102f`. No full-population acceptance is claimed for these.
- Root placed `native848_serial_next49_20260917_v1/STOP`. Actual native/outer waits for its first two16-frame cases (seed67/SubStem42 and seed31/SubStem40) are0; root queue52532 exited1 at the explicit STOP guard before the remaining four cases. The saved queue failure is the requested cooperative stop, not a failed native capture. Original source files and partial-pilot outputs are retained.
- Seed67 new16 has16 strong automated candidates and a visual packet, no review/export yet. Seed31 new16 raw capture completed, CPU annotation pending. Neither adds accepted credit.
- The original scene has142 merged background plant instances plus2 component plants. The sparse policy removed all142 and any cross-split component plant. The user subsequently APPROVED replacing background meshes with complete anatomy-labeled plants throughout the same fully populated rows. Preserve all144 occupied slots, original greenhouse density and the intact primary target at its original placement; background plant shapes may differ. No per-view/target removal or pruning. Use split-consistent background source families and authenticated per-instance geometry. The old filtered annotation/export contract must not be reused by pretending this scene is filtered.
- Additive full-population scene/plan/CPU preparation and native worker/owner integration are being implemented. Root alone launches native capture; source review and a small actual full-population capture precede scale-up. The lineage audit confirmed that the24 old merged background assets lack authenticated anatomical correspondence to the detailed plants. User decision: "Use fully labeled plants throughout (recommended)". New code is using fully labeled144-slot scenes. Repeated background instances retain their original source-family lineage and do not increase independent plant-family or target-diversity counts.


## Latest execution update ? 2026-09-17 19:07 KST

- **Latest continuation:** Root started a finite49-candidate queue across6 targets using serial6 after the actual2-frame telemetry-opt-out smoke and full closure checks. Queue helper `data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_serial_next49_v1.py`, SHA `0d3c7ccaddbcec29940ab1010b0669f19d1a34acb7ef922775ff170c07256a8f`; root tool session52532. Stop marker: `data/sim_data/diagnostics/native848_serial_next49_20260917_v1/STOP`. No retries or automatic expansion. First16 views expand the actually successful seed67/SubStem42 poses; next16 expand seed31/SubStem40, followed by four new targets. These are candidates, not accepted images.
- Smoke2 seed67/SubStem40 captured successfully but both images failed cut/support visibility; its remaining2 proposed views were withheld. Root metadata originally omitted Name/ParentProcessId; create-only explicitly derived metadata authenticates both missing fields from the real recorded startup snapshot after all four original fields match. Original receipts unchanged; full unchanged identity comparator passes. Closure SHA `9792cede1b311e1b81a82d7a103e8dfb365eeffebd8add46b4e3351709a8b183`. No new process observation or continuous transmitter absence is claimed.
- Annotation9 cached wrapper `native848_annotation_cache_v2.py`, SHA `b9fabd98df0671bb882d99a737e7a4b920fa50a59b4412022d3dbe5d211aa48d`, passed actual2-frame parity (21.82s direct ->16.38s cached). CPU consumer is assigned to process each actually successful next49 case and prepare review packets. All dataset acceptance gates remain active.
- **User examples:** `data/sim_data/dataset_reviews/recent_accepted_examples_20260917_v1/index.html` links three different accepted targets, original RGB, metric depth, validity, calibration and exact answer JSON. PNG previews show pink cut-point rings and cyan attachment crosses, with enlarged junction details. Original848x408 RGB copies are bit-identical and contain no overlay. Examples derive from accepted106 v5; no pending capture receives credit.

- **Accepted single-answer pilot: 106 images**, 62 TRAIN / 21 validation / 23 TEST, 9 original plant families and 11 physical targets. Latest verified union: `data/sim_data/training_exports/tomato_cutpoint_848x408_unique9mm_pilot_incremental_20260917_v5/`; stable pointer `LATEST_UNIQUE9MM_PILOT.json`. Result SHA `6692f6aff62bc3c4f81c14d5f375d85b8557d1fc0fe7563c82d651cc97e08214`. Still incomplete against the 2,300-image pilot. Dataset only; no training or human verification claimed.
- New accepted additions: seed83/SubStem43, 15 TRAIN; seed41/SubStem41, 11 TRAIN. Assistant legacy_coverage actually displayed and reviewed every full native RGB and unscaled junction crop; root inspected representatives and authorized the concrete exports. All26 passed unchanged automated predicates and export/source closure checks. Banned seed41/SubStem38 remains excluded.
- The finite42 candidate capture queue finished all11 plans with actual outer/native successful waits. Root session87003 exited0. CPU annotation and target/view-group visual review continue; raw counts do not receive acceptance credit.
- **Measured throughput fixes now in use:** annotation on15 frames 117.67s ->25.825s (4.56x); redundant pre-packet census replay removed, identical frame specs assembled in0.237s versus115.4s; packet generation86.47s ->18.608s (4.65x). Combined measured CPU preparation~319.5s ->44.7s, approximately7.2x for this batch, not an end-to-end capture speedup or completion ETA. Annotation sample/coverage bytes and decisions match; packet inventory and every native crop byte match, packet differs only by output directory.
- Implementations: `examples/greenhouse_sim/sim_data/native848_annotation_cache_v1.py` SHA45d597a50775c61abd51507b5bbc6ba45eaa7064a7dfaa73ffecb39d29349174; `native848_unique9mm_batch_validation_v2.py` SHAfa107a0ce0150f3e3deb64e2cb21a77b79c93452ca3a6cee40ec666bb4e7aad4. Each dedicated operation still authenticates each distinct source initially, uses immutable snapshots, rechecks every path alias and fully rehashes the source union before completion. No acceptance predicates removed. Exact root export authorization is retained.
- Measurement evidence: `data/sim_data/diagnostics/native848_cached_annotation_seed83_parity_20260917_v1/result.json` and `native848_fast_packet_seed83_parity_20260917_v1/parity_result.json` (SHA d9d8b980c4f7ef2a5500167f345f717c720b2cc01084db54e33fad46da9be158). Optimized exports took21.93s for15 seed83 frames and20.28s for11 seed41 frames.
- Remaining capture bottleneck: every small plan reloads a complete scene and warms the renderer. Observed native child duration118-157s for2-15 frames; about26s is outside the capture function. Persistent SimulationApp reuse is source-reviewed only and not natively qualified. New-target CPU supply is being prepared; full plants, one visually unambiguous answer, native848x408 and9mm cut remain required.




Last updated: **2026-09-17 continued collection (Asia/Seoul)**.



Workspace: `D:\research\tomato-pi-policy`.



This document records the collection and its correction after the user rejected severe target concentration. **A usable diverse 20,000-image replacement has not been delivered.** The current-execution section governs; earlier capture and review receipts remain preserved.



## Current prediction task - September 17 update

The user's latest instruction supersedes the earlier 10 mm query-conditioned task:

- **Inputs:** full native 848 x 408 RGB and aligned depth, depth validity and calibration as required. No supplied petiole query, target ID, target mask, or ground-truth-centered crop may identify the answer at inference.
- **Outputs:** the user subsequently requested **one true cut-point answer per image**, without a query cue. Admit only frames with exactly one clearly identifiable eligible petiole. Enumerate other scene candidates privately to verify uniqueness. Hold frames with multiple equally eligible targets; never choose one using hidden simulator identity. This supersedes the earlier all-target output choice.
- **Cut-point definition:** **9 mm along the 3D petiole from its attachment junction**, 1 mm closer than the previous 10 mm label. Reproject from 3D geometry; do not subtract one image pixel. The 9-19 mm segment is used only to check visible detail; the output remains the exact 9 mm point.
- Each frame is counted once, with one model-facing cut-point answer and a private complete candidate census establishing exactly one eligible petiole. Record excluded and unresolved candidates privately; incomplete coverage cannot be treated as proof of uniqueness. RGB/depth inputs must not contain annotation overlays, target masks, query coordinates, ground-truth crops, target counts or simulator target IDs.
- Reachability uses measured camera-to-robot calibration and robot pose in a separate geometry check. Whitelist those fields: the legacy robot snapshot also contains `desired_cut_pixel_xy` and `framing_error_degrees`, which must never enter model inputs.
- Reuse unchanged raw images only after new 9 mm visibility/detail, attachment association and robot-workspace checks. Label every eligible target required by the chosen policy; removing the old query alone does not make single-target annotations complete.
- Preserve the legacy dataset and labels. The 711 TRAIN / 48 validation / 90 test checkpoint below is accepted under the previous 10 mm query-conditioned task. **The two initial qualification exports remain held for competing-petiole ambiguity. Accepted checkpoints contain 80 usable images: 36 TRAIN / 21 validation / 23 TEST across seven original target plant families and nine physical cut targets. Consolidated v4 contains all 80.** Do not mix the two counts.
- Updated explanatory figure: `data/sim_data/dataset_reviews/cutpoint_9mm_example_20260917_v1/cutpoint_example.png`. Red is 9 mm, cyan is the junction, orange is the old 10 mm point; these explanatory overlays are not model inputs.
- Earlier figure, retained: `data/sim_data/dataset_reviews/cutpoint_10mm_example_20260917_v1/cutpoint_example.png`.

## Authorized 2,300-image pilot

The user approved proceeding with **2,300 unique images: 1,500 TRAIN, 300 validation and 500 test**, across **at least 20 independent original plant families**. This pilot is the current delivery priority. It does not mean the earlier 20,000-image objective has been delivered.

- **Full plants only (user reconfirmed September 17):** keep the target plants and their petiole anatomy intact. Find unambiguous camera views. Do not add pruned-plant scenes or delete competing target-plant branches. The established uniform removal of unauthenticated decorative background and cross-split component plants remains the capture policy.
- Each native 848 x 408 RGB/RGB-D view must have exactly one clearly identifiable eligible intact leaf petiole and one exact 9 mm cut-point answer. No input query or ground-truth crop/mask/target hint.
- **Reuse priority (user confirmed September 17):** audit the existing diverse 849-image collection, including the exact 477 TRAIN base, for relabeling and reuse before replacing those views. The current source has 711 TRAIN / 48 validation / 90 test; `registry_origin=user_selected_diverse_base` includes 477 TRAIN plus 138 held-out, while 234 TRAIN are later additions. Recompute the 9 mm point from original geometry/calibration and reassess visible competing petioles without a query. Preserve originals and splits. Do not blanket-reject all old images because one example had unresolved background anatomy; quantify coverage and eligibility per image. Reuse candidates are not counted as new-task accepts until validated.
- Preserve source-plant separation across splits, including geometry variants. Track unique images, physical targets and donor families separately. Spread target locations across the image and prefer diverse targets over repeated views.
- Use only authenticated component-assembled plants in the pilot's potentially reachable visible region. Remove unauthenticated merged decorative plants uniformly in the new scene recipe; keep greenhouse structure, robot embodiment, calibrated optics and correctly labeled plant geometry. CPU preflight and native rendering must use the same recipe. Do not modify old captures or invent missing backdrop anatomy.
- Begin with 2-4 new native frames and complete scene census, 9 mm geometry/visibility/workspace checks, actual visual review and usable RGB/depth export. Once that works, scale balanced multi-plant batches toward the exact split quotas.
- Automated checks apply to every frame. Visually review each target/view group and flagged cases, as authorized; the initial qualification frames get individual full-frame inspection. No training or SSH work.
- Keep a 500-training-image checkpoint for early feedback; collection can continue to the full pilot while the user handles training.
- Evaluation should separate valid output, correct petiole and point error. Use per-plant/per-target results, plant-aware uncertainty, image-center/mean-coordinate baselines and image-use checks. Correlated views do not count as independent plant tests.
- The prior 800-frame recommendation is superseded. **Current usable new-task count: 36 TRAIN / 21 validation / 23 test (80 / 2,300).** Two pipeline-qualification images were exported, then independently held because another intact petiole is visibly and kinematically eligible. Keep them as diagnostic artifacts only.
- **Held qualification artifact, not a training dataset:** `data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_unique9mm_qualification_20260917_v1/` (result SHA-256 `abcb6bfb62e447b09142d06ec8b2040e616652b09cf90a2f412115f7cf352d49`). Each row has clean native RGB, float32 metric depth, validity, calibration and exactly one `cut_point_uv`.
- Four native qualification frames completed with actual owned exit 0. Complete native census covered 153 petioles with no unknowns per image: two frames passed and two were held because multiple petioles were eligible. All four full images were actually viewed; junction/cut crops were also inspected. Subsequent independent review found target41 visible and reachable in both initially passing frames, with width7.21px and support11.88/11.93px just below8px/12px gates. Both are now held; a borderline detail threshold must not select the answer. Evidence: `data/sim_data/diagnostics/native848_unique9mm_competing41_audit_20260917_v1/result.json`. No human review is claimed.
- **Latest consolidated usable new-task checkpoint:** `data/sim_data/training_exports/tomato_cutpoint_848x408_unique9mm_pilot_incremental_20260917_v4/`. **80 images: 36 TRAIN / 21 validation / 23 TEST, seven original target plant families, nine physical targets.** Bit-identical union of six accepted checkpoints; every frame received actual full native RGB and unscaled junction-crop assistant review. Clean RGB/depth/validity/calibration are under `frames/`; `index.jsonl` and split indices have one `answer.cut_point_uv` per image. Model inputs exclude private annotations and provenance. Result SHA-256 `18ccaeb585258edb82e4869704225a54abc695f24e5ca789a33d6c1dbcb46168`; manifest `af6c89f6cf5dadcb018afe978f5a22e0964b63b0cbfa44967318c24c107dc9ae`. All 80 copied inputs loaded with the frozen whitelist; RGB/camera uniqueness and family split checks passed. Incomplete pilot; no human review or training claimed.
- Latest validation additions: `tomato_cutpoint_848x408_unique9mm_legacy_seed29_validation_20260917_v1` has 16 accepted images from 20 native replays of an authenticated old pose and nearby views. `tomato_cutpoint_848x408_unique9mm_screened_seed97_validation_20260917_v1` has all 5 accepted screened views. All21 underwent actual individual full-native RGB and native junction-crop review. No unchanged legacy image bytes received new-task credit.
- Failed parallel68 preserved all 14 TEST59 frames, then stopped when its exact recorded telemetry process remained alive 10.595 seconds after native/command exit. The saved failure and actual outer exit 1 remain unchanged. Slot 1 stopped before production, slots 2/3 never launched; all recorded processes subsequently exited. Finite recovery3 authenticated the complete 14-frame slot. Fresh annotation, individual visual review and export have now accepted all 14; these are included in v4.
- **Newest accepted addition:** `data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_unique9mm_failed68_test59_test14_20260917_v1/`, 14 TEST images from seed59/SubStem42. Result SHA-256 `4b6854d8bb681876a873e18b901fb7c3dde131e78e7626bc0efd3ec48bb5e897`. All 14 full native RGB images and exact junction crops were actually reviewed; group-review SHA `6048c5b60061144b33c916acab9bf0e5debd4e48270d9320b513fd1cc6513813`. The complete dataset has 80 accepted images, not the 2,300-image pilot target.
- Accepted target counts: seed17/SubStem41 = 4 TRAIN; seed103/SubStem42 = 8 TRAIN; seed103/SubStem45 = 11 TRAIN; seed61/SubStem41 = 9 TEST. Target IDs are private audit information.
- The failed first 103-candidate parallel batch remains failed. Explicit closure/source recovery admitted only its complete 51 seed103 and 22 seed17 raw frames to annotation. Of 39 strong automated candidates, 23 passed actual review and group propagation and were exported. The partial seed31 frame and startup-failed seed13 slot remain excluded.
- The subsequent 159-candidate parallel batch also remains failed, with actual outer exit 1. It completed 21 seed71 raw frames, then preserved one excluded partial seed89 frame; seed11 and seed19 were never launched. Failure was an unknown-process admission check, despite adequate measured resources. Missing failure-snapshot detail prevents a certain causal claim. The finite recovery audit authenticates 21 complete frames and recorded owner/worker/telemetry absence; those 21 are pending annotation and QA, not accepted.
- Serial continuation succeeded: all 57 seed19 candidate views captured, actual native child and outer owner exit 0. Native duration 328.55 seconds; new 9 mm census and visual review completed with 13 accepted images. Capture: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_unique9mm_pilot_20260917_v1/seed19_train_57_views_serial3_v1/`. Result SHA-256 `c8521dad4424e0fa5eddf9f0e2e929ae7a0ae8573000cf9387d2c481309b3966`. All57 immutable schedule entries, observation assets and fresh callbacks were verified in `native848_pilot_seed19_full57_closure_20260917_v1/result.json`, SHA-256 `374fe6ef8d4f0a8aaa54e55535b396669dd9f3ab6f00e30b3bb671e241905458`. Annotation found14 strong candidates;13 were accepted after actual individual review.
- Existing 849 reuse audit is complete for sensor/schema authentication and native visible scene coverage. All raw assets and five original schemas authenticate. All 849 retain visible plants without authenticated petiole anatomy; none can prove complete single-answer ground truth using retained evidence. 848 have an unknown-plant surface inside an arm outer sphere; this is not proof of IK reachability or an eligible competing petiole, but prevents conservative exclusion. In held-out data, 97 images contain visible components from a training family (79 contain petiole pixels), so split leakage also needs correction. All originals remain unchanged. Coverage result: `data/sim_data/diagnostics/native848_legacy849_coverage_20260917_v2/result.json`, SHA-256 `c5a331450177d8f253256cb1d9a71654ea4684ee56d14796f40b057c3467d6fa`.
- Reannotation of the exact 477 TRAIN base: 380 original primary targets pass strict 9 mm checks; 223 views have exactly one strict primary-plant candidate, including 171 where that candidate matches the original target. Background and near-eligible alternatives still require resolution, so accepted increment is zero. Result SHA-256 `d0c8823a2a86144f695dea19c226bac15197ef29a90b4fc9663b32d14ed393a8` in `data/sim_data/diagnostics/native848_legacy849_primary9mm_exact477TRAIN_20260917_v1/result.json`.
- Reuse the best authenticated camera/robot poses for new native capture in the existing uniform filtered scene. This preserves useful view/plant diversity while correcting missing background anatomy. Start with the balanced 31 original-primary singleton poses across 18 families, then expand from the 171 TRAIN poses and held-out supply. Re-rendered frames are new captures, not unchanged reuse of old image bytes. No per-target geometry deletion.
- The recovered seed71 batch yielded zero usable candidates after independent ambiguity checks: its three strict singletons still have a visibly connected and reachable competing petiole. Keep all 21 raw frames and diagnostics; do not count them. This motivates screening for projected competing junctions before native rendering.
- Recent serial captures both completed full 20-frame schedules with actual outer/native waits 0. Seed89/SubStem41 passed automated checks but all 20 actual full-image/crop reviews were held: the branch exits the top boundary before associated leaf tissue appears, so no-query identity is unclear. Review: `data/sim_data/dataset_reviews/native848_unique9mm_legacy_seed89_actual_review_20260917_v1/legacy_seed89/group_review.json`, SHA `f755522684c7c77d8ae36d6c8f63bd819c047bf83b74164423825b960ed90e27`. Seed101 has 20 strict singletons but zero strong unambiguous candidates after independent ambiguity checks; result `native848_legacy_seed101_serial20_annotation_20260917_v1/result.json`, SHA `bab13df020d6d1a1c7fda83b82c7be99478a182c301429690d8c69a9687d5e6e`. Neither batch adds accepted images.
- Source supply limit: the 849-image reannotation identifies 130 original source poses with one strict primary and no continuously visible same-plant alternative. These span only30 physical targets across15 families (TRAIN24targets/11families; VAL3/2; TEST3/2). At20planned views per target this supplies at most600 candidates, before further holds. The 2,300-image pilot therefore needs additional physical targets and at least five more families. Six legacy gap families are seed13,23,31,53,67,7. Geometry-only corner/side reframing and bounded body shifts show possible full-plant opportunities; exact CPU/native checks and actual visual review remain mandatory. Do not count projected opportunities as accepted examples.
- Prepared legacy replay inventory: `data/sim_data/diagnostics/native848_legacy849_primary9mm_pose_handoff_20260917_v1/prepared_replay_plans.json`, SHA-256 `eeba3397ed39bf3383386269b8cf93c8acf156754a39776cd7611aa15a35906e`. Thirteen plans contain228CPU candidates;12strong-source plans contain210 (156TRAIN/40VAL/14TEST). The seed31 plan is lower priority because its source has a visible alternative. Original images, source splits and full target anatomy remain unchanged.
- Additional plant families have CPU-prepared candidates; final plan-bound scene preflights, native rendering and ambiguity review remain required. Prepared/captured counts never substitute for accepted counts.


- **Previous parallel run failed (serial capture has since resumed):** `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_unique9mm_pilot_parallel_20260917_v1/four_legacy_ranked_64_candidates_v4/`. Actual root owner exit 1; result SHA `ff1987fbca75d7fcb72e90b0ef6731d5ccd344ade3f242f6a043bbc3107121bb`. Full slot 0 has 14 seed11 captures. Slot 1 has 3 partial seed41 captures, excluded; slots 2/3 never launched. Parent failure remains failed. All recorded child identities eventually exited and real waits are saved. Finite completed-slot recovery and serial continuation are being prepared; none of these frames is accepted yet.
- Additional full-plant gap plans now contain 10 CPU-qualified views across five new target families: seed31 (2 TEST), seed67 (2 TRAIN), seed23 (2 TRAIN), seed13 (1 VAL), seed7 (3 TRAIN). They require native visibility and visual QA. New proposals screen the entire connected petiole-to-leaf route in frame; this is necessary but is not proof of visibility. Ready seed83 replay has 15/15 projected leaf-association passes. Seed71 has only 10/20 passes and needs a new filtered plan/preflight before capture. Never modify active or frozen plans.
- Source expansion: an anatomy census found 297 scheduled targets across 24 plant families, but these are not visibility/workspace-qualified. A bounded geometry survey of 179 new targets found 16 promising targets across 13 families, adding 15 beyond the old opportunities. At the current 20-view cap, the pilot needs at least 115 qualified targets; available accepted diversity remains far below that. Prioritize new physical targets and full leaf/junction context; no completion ETA is established.


- **Parallel-runner root-cause finding:** its per-slot cleanup expects the telemetry helper to exit when a single worker exits. NVIDIA documents that the transmitter is shared and runs until all client processes exit; this explains why 10- and 60-second waits failed while siblings were active. Reference: https://docs.omniverse.nvidia.com/kit/docs/carbonite/206.14/api/function_Telemetry_8h_1a1bc5dc02f0969b2d984fde6c4d9c554e.html . Stop retrying larger retirement waits. A task-local opt-out proposal is being checked against installed SimulationApp/config sources; no global settings or rendering-profile changes have been applied. Serial remains the reliable immediate capture path after actual failed-run closure is authenticated.
- Serial101 ambiguity details: SubStem43 has a continuously visible junction and verified reachable cut region in all 20 images; one image also has reachable SubStem44. Their strict-label detail/clearance failures do not remove them as competing answers. All 20 remain held.

- **Serial capture resumed with a finite queue:** seed83/SubStem43 completed 15 native candidate views with actual owner/native wait 0; annotation8 and review remain pending. Root-reviewed serial5 uses unchanged worker1, no telemetry opt-out or rendering changes. A finite queue of 42 additional candidates across 11 preflighted full-plant target plans is running, starting with filtered seed41 (12) and seed71 (10), followed by small new-target/gap-family plans. Source: `data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_serial_leaf_queue42_v1.py`, SHA `81b93576b49c7276172d2570b95fdd72ca8a34ae41cdf4cb740b49654bc5445a`; receipts: `data/sim_data/diagnostics/native848_serial_leaf_framed_queue42_20260917_v1/`. Root tool session 87003 owns the queue. Every case waits for the prior real successful outer exit, then the unchanged serial owner validates full predecessor closure and fresh process/resource gates. Stops on failure or queue STOP; no automatic retry or expansion. These are capture candidates, not accepted images. A CPU agent validates/annotates successful cases and prepares visual packets; root actual visual review/export remains required. Current accepted dataset remains 80.
- Task-local telemetry helper proposal passes 49 CPU checks, but no opt-out has been applied to the running queue. Additive worker2/serial6 integration is in preparation for a later separately reviewed native smoke; current frozen worker1/serial5 source bindings stay unchanged.

<!-- CURRENT_EXECUTION -->
## Collection status at the task change

Legacy accepted checkpoint: `data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_diverse_continuation_20260917_v4/`.

- 711 TRAIN, 48 validation, 90 test; 58 TRAIN physical targets, 15 source plant families. Exact 477-image base preserved; banned seed41_full/SubStem_38 excluded.
- Result SHA-256: `09e79dcd4cac0b93f988f4ff5208a15eab26c71434a16c79dc454b52739a6624`. Resolve index rows through `asset_root` and `files`.
- Another 226 frames completed native capture (214 seed71 plus 12 seed89 parallax); final review/export pending. Their old 10 mm candidate labels are not accepted under the new task.
- 830 balanced views across eight targets/four source families are CPU-prepared: seed89 204, seed103 256, seed41 154 (never target38), seed43 216. These are planned captures, not accepted images.
- Completed bounded raw capture: `C:/Users/USER/tomato-vlm-data-20260917/diagnostics/native848_diverse_native_20260917_v1/multianchor_seed89_scale_seed17_fresh_v1`. Two workers completed all 204 seed89 views plus two new-target seed17 pilot views; both slot closures, whole-owner closure and actual outer exit 0 are saved. Geometry/renderer unchanged; new-task labeling remains pending. Capture started before the latest task change; old labels receive no new-task credit.
- C storage was added for new outputs because D has insufficient room for a 20k-scale release. Existing D assets stay unchanged; resource floors remain in force.
- New 9 mm utility passes 24 CPU checks and fresh position-IK verification. The illustrated seed67 frame has exactly one passing foreground candidate after all 153 catalogue petioles were assessed (152 excluded). This is not yet proof of one eligible target in the whole image.
- Full scene census found 142 merged background plants: 111 whole bounds outside conservative robot reach, 31 overlapping; 12 overlapping plants have native visible pixels. Their merged assets lack authenticated per-petiole anatomy/attachment lineage. The example therefore remains unaccepted under the unique-target task; do not label those possible targets absent or guess geometry from unrelated manifests.
- Census evidence: `data/sim_data/diagnostics/native848_all_petiole_example_census_20260917_v1/result.json`. Foreground evaluation: `data/sim_data/diagnostics/native848_9mm_example_evaluation_20260917_v1/result.json`.
- Next pilot scenes must provide authenticated component anatomy for every potentially eligible visible plant, or prove all merged plants outside eligibility. The RGB/depth input module excludes query/framing hints. No training has started.
- The 20,000 usable diverse TRAIN-image goal is still unmet. There is no verified completion ETA.

## Noon delivery: exact 477-image diverse base preserved

**The requested 20,000-image TRAIN dataset was not delivered by September 17, 12:00 KST.** The corrected standalone checkpoint has **529 TRAIN images across 58 physical targets and 15 source plant families**: the user's exact 477-image base plus 52 new individually reviewed native captures. The remaining shortfall is **19,471 TRAIN images**. Validation has 48 images and test has 90; neither counts toward 20,000.

### Resumed collection and user-approved review policy

The user resumed collection after the noon checkpoint and explicitly chose: **automated checks for every image, plus visual review of each target/view group and flagged cases**. This supersedes the earlier per-image visual-review requirement for future additions. Keep the 529-image checkpoint immutable and continue toward 20,000 TRAIN images from its exact 477-image diverse base and 52 verified additions.

- Every image must still pass native 848 x 408 resolution, authenticated source/robot camera geometry, robot-workspace, visible cut-point/route/detail, label consistency, split separation, duplicate and excluded-target checks.
- Visually review every target/view group using representative native frames and unscaled association crops, including view extremes and the weakest automated clarity margins. Review flagged cases individually; unresolved or failing images stay excluded.
- Define groups explicitly by physical target, original plant/source anchor and bounded camera-view region. Never use one target's review to approve another target or materially different camera group.
- Record the actual review scope honestly: group-reviewed automated acceptance is not individual visual inspection of every image and is not human verification.
- Prefer balanced multi-target batches within persistent plant scenes. Preserve all 477 starting rows and exclude every seed41_full/SubStem_38 image. Track images, physical targets and plant families separately.
- No training or SSH work. Root coordinates native rendering; agents independently prepare poses, review quality groups and export checked additions.

### Current deliverable

Folder: **data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_diverse_noon_20260917_v1/**

Inside this folder:

- train/images/: all 529 training RGB files, native 848 x 408.
- train/annotations/: their labels and target masks.
- train_index.jsonl: training examples, query coordinates, cut-point answers and relative file paths.
- train/crops/: auxiliary query-centered crops; these do not count as extra examples.
- validation_index.jsonl and test_index.jsonl: retained held-out splits.
- result.json, status.json and README.md: counts and provenance.

All manifest file references resolve inside this standalone folder. The intermediate extension folders referenced sibling base assets; this final package copies every selected asset locally.

Independent final verification: data/sim_data/diagnostics/native848_noon_root_release_audit_20260917_v1/result.json.

### Exact scope and outcome

- The latest user instruction controls: start from exactly 477 TRAIN images across the other 58 targets, excluding every seed41_full/SubStem_38 image. All 477 IDs, RGB bytes, labels, query coordinates and provenance are preserved.
- The corrected package has **zero seed41_full/SubStem_38 images**. The old tomato_cutpoint_848x408_tonight_bulk_v1 remains on disk for traceability and is excluded from this delivery.
- The additions are camera views of existing targets: **zero new physical targets or source plant families**. Do not represent camera framing changes as new plants, new targets, or demonstrated training robustness.
- Largest remaining TRAIN target: **seed43_full/SubStem_48**, 73/529 images (**13.80%**). Its original base images were retained as requested.
- Each new accepted image passed source/native/workspace/annotation/duplicate checks and actual assistant inspection of the complete native frame and exact unscaled query-to-cut crop. No independent human validation or model training was performed.
- Original Q3 decisions remain preserved. The separately reviewed Q4 background policy retains all protected foreground/local/trace/grid checks and permits only fresh-frame background pixels at least 4 pixels from the route and 0.20 m behind it. No historical manually rejected frame was revived.
- Final checks verify exact base preservation, RGB and label hashes, local asset paths, unique IDs/images, target and family counts, and disjoint plant families across splits.

### Remaining work and measured bottleneck

The goal remains 20,000 usable diverse TRAIN images. Preserve this checkpoint and continue from its per-target histogram and exclusion/duplicate registry. Prefer underrepresented targets and additional physical targets. Do not resume dominant-target bulk collection to fill the count.

Measured native jobs took about 127-134 seconds for 3-4 frames and 161.9 seconds for 15 frames across five targets. Per-job median reset8 render request times were about 3.67-4.62 seconds; writes were about 0.08-0.12 seconds. Scene loading, preflight, warmup, geometry checks and shutdown add substantial overhead. Small separate jobs waste that setup cost.

The original bulk worker already supports multiple compatible targets in a persistent family scene. Scale that path, preserving robot embodiment, native resolution, cut-point visibility and target identity. Capture more independent target/plant supply and measure accepted images per minute after quality checks before promising another deadline. Existing evidence does not establish a 20k delivery time.

Supply audit: 16 frozen TRAIN donor plants, 198 scheduled original cut sites, and 584 broader geometric candidate sites. Those are candidates, not accepted images or proven clear targets. Procedural variants preserve donor ancestry and must not be counted as independent plants.

Timing evidence: data/sim_data/diagnostics/native848_diverse_noon_capture_cost_summary_20260917_v1/result.json.

The noon native queue finished with owned exit0. Collection was subsequently resumed under the policy above. All accepted completed reviews were exported. No training was launched.

Parallel owners: root coordinated native rendering and selected actual QA; delivery_package prepared alternate-target geometry and audited supply/timing; review_completed checked annotations and actual images; throughput_capacity exported the exact base and final standalone checkpoint.

The prior plan is archived at data/sim_data/diagnostics/native848_diversity_correction_handoff_20260917_v1/dataset_vlm_before_diversity_correction.md. **All sections below are historical and do not override this current delivery.**
<!-- END_CURRENT_EXECUTION -->



## 1. User requirements



- Prepare **20,000 usable TRAIN images at native 848 × 408**, with annotations. Validation/test images are tracked separately.

- The user explicitly changed the earlier resolution requirement to **848 × 408**. Do not revert to 1696 × 816 or resize old images and count them as new native captures.

- Prioritize the **clearest examples**: visible cut interval, visible petiole details and cut-to-query connection, little occlusion, and a target within the recorded robot workspace.

- Clear/partial/negative class balancing is deferred by the user's explicit clear-first preference. Do not describe the current clear-only checkpoint as a balanced visibility dataset.

- The current deadline is **September 17, 2026, 12:00 KST**. Earlier September 16 deadlines were missed.

- **Dataset preparation only.** The user will handle training. Do not connect to the H200 server, download models, or launch training.

- Use parallel agents for independent CPU preparation, review, packaging, and analysis. Native rendering uses one exclusive coordinator; up to four exact-owned workers in the reviewed current queue are authorized subject to fresh resource admission.



### Earlier dataset is excluded



The user rejected the older dataset. Its 11,520 TRAIN rows / 14,259 total images are **not part of the replacement count**. Reusing them would not satisfy the request.



The supplied September 13 assessment identified too few independent plants/targets, excessive views per target, identity-correlated occlusion, tiny/backlit visual cues, inadequate class coverage and human verification, and misleading aggregate metrics. Relevant local assessment: `examples/greenhouse_sim/sim_data/H200_RESULTS.md`. The referenced `runs/RESULTS.md` was not found in this checkout.



## 2. Historical v18 deliverable: 372 TRAIN / 510 total



**Historical rejected bulk package:** `data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_tonight_bulk_v1/`,18,078 TRAIN/18,216 total. The current section above and this package's `status.json` supersede the frozen v18 counts below.



| Split | Accepted selected images | Original source families | Original target stems |

|---|---:|---:|---:|

| TRAIN | **372** | 15 | 59 |

| Validation | 48 | 3 | 9 |

| Test | 90 | 3 | 12 |

| Total | **510** | 21 | 80 |



TRAIN deficit: **19,628 images**. The v18 selection adds six reviewed original images, removes none, and retains all 31 visual holds.



### Images and annotations



Portable checkpoint:



`data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_20260916_v18/`



- Images: `train/images/`

- Labels and target masks: `train/annotations/`

- Query crops: `train/crops/`

- Depth and validity assets: `train/depth/` and paths recorded in the index.

- Validation/test use the corresponding `validation/` and `test/` directories.

- Read `train/index.jsonl`, `validation/index.jsonl`, `test/index.jsonl`, or root `index.jsonl` for exact paths and hashes.

- Each annotation directory contains both JSON labels and `_target.png` masks. For example, TRAIN has **744 annotation-directory files for 372 images**. Do not count those files as extra samples.



Absolute TRAIN image path:



`D:\research\tomato-pi-policy\data\sim_data\dataset_checkpoints\tomato_cutpoint_848x408_20260916_v18\train\images`



Absolute TRAIN annotation path:



`D:\research\tomato-pi-policy\data\sim_data\dataset_checkpoints\tomato_cutpoint_848x408_20260916_v18\train\annotations`



Historical 15:00 delivery README (predates v18):



`data/sim_data/dataset_checkpoints/delivery_20260916_1500_v3/README.md`



Authoritative selected records:



`data/sim_data/diagnostics/clear848_reviewed_workspace_checkpoint_20260916_v18/selection.json`



Selection SHA-256:



`0fc1ad64108ad3d8326ae29f3b8c4e2373d86afea83ed904e7f8db72aa8d0257`



### What is verified, and what remains provisional



- Selected images have individual assistant visual reviews, their applicable strict local clarity checks, and recorded-pose WorkspaceV2 checks.

- Full RGB images and annotations were preserved and export hashes checked.

- All 510 query crops were pixel-verified: a 384 × 384 query-centred region, shifted inside the frame, magnified to 768 × 768. These are auxiliary inputs, **not additional images or additional sensor detail**.

- 496/510 crops contain the entire accepted cut interval; TRAIN: 363/372. Crop framing is based on the query, not the ground-truth cut point.

- Snapshot bytes before result receipt: **1,403,727,920**; referenced assets: **1,399,577,733 bytes**. Export verified 4,115 asset/index files in 37.47 seconds.

- This is a **provisional checkpoint**, pending final training-release qualification. Human held-out review and final near-image/global diversity qualification remain incomplete.

- Do not claim all v13 rows use the newest full-query-trace annotation contract. Some earlier rows use the previous clear-label contract plus their recorded clarity/workspace/visual checks.

- WorkspaceV2 establishes recorded-pose reachability and its documented geometry/joint checks. It does not prove a complete physical cutting trajectory.



## 3. Environment and operating rules



```powershell

Set-Location 'D:\research\tomato-pi-policy'

$env:PYTHONPATH='D:\research\tomato-pi-policy\examples;D:\research\tomato-pi-policy\examples\greenhouse_sim'

$env:OPENBLAS_NUM_THREADS='1'

$env:OMP_NUM_THREADS='1'

```



- CPU Python: `C:\Users\USER\miniconda3\python.exe`; use `-B -u` for runners.

- Native Isaac Python: `D:\isaac-sim-6.0.1\python.bat`.

- Most native workers run with working directory `examples/greenhouse_sim` and the environment returned by the existing worker-environment helper.

- Native mutex: **`Global\greenhouse.native_serial_phases.owner.v1`**, implemented by `native_dataset/serial_phases_reboot_v2.py::owner_lock`.

- Preserve at least **20 GiB commit headroom and 60 GiB free disk** at native admission.

- Start background owners hidden (`Start-Process -WindowStyle Hidden`). Record actual process identity, including PID, creation time, executable, and command line.

- Native owners must wait on their actual child process and save its real exit receipt. An absent PID alone is not evidence of successful completion.

- All outputs are create-only. Existing owner scripts generally **cannot be rerun into their old output directories**. Create a new version/output for another run.

- Preserve frozen code, assets, source bindings, failures, and review holds. Do not change a module already pinned by a captured dataset.

- Do not kill user-owned Isaac/MCP processes. `verified_socket_bridge_gate_v1.py` contains a narrow authenticated exemption for the known non-rendering socket bridge; it is not permission to ignore other native processes.

- An intermittent Windows sandbox helper error affected some reads/edits. Authorized shell operations were used to work around it; no dataset files were deleted.



Path abbreviations used below:



- **D** = `data/sim_data/diagnostics/`

- **H** = `data/sim_data/diagnostics/collection_20k_848_20260916_v1/`

- **M** = `examples/greenhouse_sim/sim_data/`

- **R** = `data/sim_data/dataset_reviews/`



Most diagnostic/data files are ignored by Git. Use `rg --files --hidden --no-ignore` when locating them. Several new Python modules are untracked; a Git checkout alone does not contain this complete handoff.



## 4. What has been done



### 4.1 Original native848 collection and reviewed export — complete



The latest original queue completed all 16 fresh jobs: **188 raw frames → 103 strict/workspace candidates → 89 visual accepts and 14 visual holds**. This added only 45 selected TRAIN images to the earlier 312-image baseline because per-target caps and replacement selection apply.



- Frozen plan: `data/sim_data/collection_plans/clear848_orbit_window12_20260916_v1/plan.json`.

- Resumed owner: `H/run_direct848_priority_v2.py`.

- Completion: `data/sim_data/collection_campaigns/clear848_remaining_train_resumed_20260916_v2/`.

- Final job019 completed successfully after approximately 696 seconds with zero screened frames.

- There is **no unfinished original queue to restart**.

- Original review/join entry points: `H/join_reviewed_workspace_v1.py`, `H/preview_pinned848_v1.py`.

- Original-only portable exporter: `H/export_reviewed848_deadline_v1.py` (read its CLI/source before reuse).



### 4.2 Direct cached-camera rendering — implemented and actually tested



The user's idea is partly implemented: precompute a real robot-mounted camera pose, load a complete greenhouse once, apply cached robot/head transforms, and capture without a broad pose search.



Modules:



- `M/native848_direct_plan_v2.py`

- `M/native848_direct_worker_v2.py`

- `M/native848_direct_audit_v2.py`

- `M/native848_direct_admission_v2.py`



The current v2 route uses **56 subframes per image** and actual synchronized RGB/depth/instance buffers. It is specifically bound to generated-plant plans. Do not pass an original-only cache into it or rename a short capture as this profile.



Measured hot-frame costs: pose application about **12–14 ms**, geometry screen about **0.53 s**, artifact write about **0.2 s**, render/readback about **31–34 s**. Rendering accounts for roughly 98% of the hot path. Stage setup and calibration add substantial cold-start time.



### 4.3 Completed generated capture experiments



| Artifact under D | Actual outcome | Selected TRAIN increment |

|---|---|---:|

| `native848_generated_seed41_trial_20260916_v2` | One generated seed41/SubStem43 frame visually accepted; shared source pool already full | 0 |

| `native848_direct_camera_benchmark_20260916_v1` | Short8 comparison failed RGB/resolved-instance equality; preserve failure | 0 |

| `native848_direct_camera_fixed56_trial_20260916_v2` | 3 frames; all failed full interval visibility. Fruit_07 or Leaf_027 occluded interval probes | 0 |

| `native848_prefiltered_direct_seed410502_trial_20260916_v1` | 5 scheduled, 4 rendered, 1 strict/workspace/visual accept: SubStem42_view002; source pool already full | 0 |

| `native848_reference_wave_fixed56_capture_20260916_v1` | seed17: 3 frames, all clarity exclusions. seed103: 1 strict/workspace candidate, SubStem42_view001; source pool already full | 0 |



The last wave finished at approximately **16:04:56 KST**. Its per-family admissions are inside the wave directory as `seed17_full_admission/` and `seed103_full_admission/`.



The seed103 candidate was viewed by root as a full-scene preview and lossless native-scale cut/query crop. The stem connection appeared clear. **A sealed review file for this last candidate has not yet been written**, so do not treat that inspection as a published selection increment. Its source target already has 12 selected views regardless.



Second-plant accepted review:



`R/native848_prefiltered_direct_seed410502_20260916_v1/assistant_reviews.json`



SHA: `bdb263c3d8635f131562064d72b3ef83b4a91620940081f2243b78a0b77c1071`.



### 4.4 CPU visibility screening and generated geometry preparation



- `H/prepare_generated848_reference_group_v1.py`: prepares actual curved/relocated petiole geometry for compatible reviewed reference groups.

- `H/cache_direct_camera_reference_group848_v1.py`: precomputes real robot-mounted camera poses.

- `H/prepare_reference_group_wave848_v1.py`: prepared seed103 and seed17 groups.

- `H/prefilter_partial_plant848_v1.py`: screens rays against actual donor-plant meshes before rendering.

- `M/procedural_petiole_v2.py`: creates curved/relocated petioles while rigidly transporting detailed leaf meshes; preserves source lineage.



Prepared assets/caches:



- `D/native848_five_target_prepare_20260916_v1/` — seed410501, five seed41 targets.

- `D/native848_five_target_prepare_20260916_seed410502_v1/` — seed410502, same five original source targets.

- `D/native848_reference_group_wave_20260916_v1/` — four targets each on seed103 and seed17, 48 cached poses total.

- `D/native848_reference_group_wave_prefilter_20260916_v1/` — 48 poses, 18 predicted blocked, 15 unblocked proposals with ≥12 px intervals, covering four targets.

- `D/native848_reference_wave_fixed56_prepare_20260916_v1/` — the four-view plans subsequently executed above.



The partial-plant ray screen matched 45/45 observed interval-probe identities in three earlier occluded frames. It excludes obvious bad proposals; **it is not a full-scene/native visibility approval**. Other plants, robot geometry, appearance, and query trace still require capture checks.



### 4.5 Renderer-reset experiments — complete, no faster profile qualified



- v1 was queued but stopped cooperatively **before native launch**. Receipt: `D/native848_reset_probe_trial_20260916_v1/stopped_before_native.json`.

- v2 completed with real native exit 0; elapsed native process time approximately **341.89 seconds**.

- Evidence: `D/native848_reset_probe_trial_20260916_v2/`.

- Frozen module: `M/native848_reset_probe_v2.py`.

- Plan/handoff: `D/native848_reset_probe_prepare_20260916_v2_final/`.



V2 intended to test reset1/2/4/8/16 only after stable reset56/reset56 controls. **Both long-repeat controls failed the equality comparator, so no short branches ran.** Four long-reference frames were saved and independently replayed.



Important interpretation: both original clear long frames independently pass actual strict clarity and selected full trace, with the same selected query/answer. Their comparator failure included 230 background hanger identity pixels (zero in the target ROI), ROI RGB MAE 2.715 against a 2.5 limit, and changes among non-selected query candidates. This is failure to prove cross-render equivalence, not proof those clear images are unusable.



`reset_renderer_accumulation()` returned successfully. Actual settings recorded AA op3 / DLSS Performance0. This does **not** establish the camera's internal raster resolution. The native saved buffers are 848 × 408. Do not infer target internal downsampling solely from a startup 424 × 204 warning.



The installed Replicator global `RTSubframes` floor and motion-blur settings must be checked when testing 1/2/4; otherwise a requested short budget could secretly execute more subframes.



### 4.6 Geometry diversity and near-image diagnostics — measured, unqualified



Blind native controls:



- `D/blind_native848_controls_capture_20260916_v3/`: eight captured cases, five matched comparisons.

- Sealed review bundle: `R/blind_native848_controls_incremental_20260916_v1/review_bundle.json`.

- Reviews: C01/C03 meaningfully distinct; C02/C04/C05 uncertain; **zero insufficient-change negatives**.

- Protocol was frozen before distances/review outcomes were used: `H/native848_control_calibration_protocol_20260916_v1.json`.

- Result: `D/native848_sealed_control_calibration_20260916_v1/result.json`.

- No supported threshold can be fitted with no negative class. All leave-one-out fits are unsupported. No novelty qualification or view-cap reset resulted.



Actual geometry inventory:



- `D/native848_actual_geometry_inventory_20260916_v1/`: **324 contexts = 297 original + 27 generated**, source-bound ancestry and descriptors.

- `D/native848_geometry_pair_distances_20260916_v1/`: **52,326 unordered pair distances**, zero unsupported descriptors, fixed metric `v2_fixed_linf_symmetric_leaf_feature_set_hausdorff.v1`.

- Distances are threshold-free. A large distance does not by itself authorize a new independent target, and zero feature distance would not prove identical meshes.

- This inventory covers scheduled originals and current experiments; it is not the complete global candidate inventory. Held-out descriptors were used for comparisons, not fitting.



Near-image checks:



- `D/native848_near_image_readiness_v13_20260916_v1/`: all 495 selected rows are singletons under exact encoded RGB, decoded RGB, and recorded-view identity checks; no exact cross-split identity overlap found.

- `D/native848_selected_v13_pair_distances_20260916_v1/`: 118,828 supported comparisons across 488 images; 3,437 unknown pairs involve seven edge images without the required 129 × 129 cut patch.

- The old 1696-resolution near-image cutoff is **not calibrated for 848**. Do not reuse it silently or turn unknown comparisons into independent examples.



### 4.7 Conservative join/export for new pair/direct captures



- `M/native_dataset/native848_conservative_join_v1.py`

- `M/native_dataset/test_native848_conservative_join_v1.py` — eight focused CPU tests passed.

- `H/export_conservative_mixed848_v1.py` — new mixed exporter, prepared but no positive-increment mixed checkpoint has been published.

- Contract/request: `D/native848_conservative_join_preparation_20260916_v1/README.md` and `request.json`.



The joiner authenticates completed capture/admission/review files, replays native arrays and strict labels/query trace, checks WorkspaceV2, preserves 30 prior review holds, rejects exact image/camera duplicates and cross-split conflicts, and enforces the shared original-source 12-view pool.



Actual inspect runs:



- `D/native848_conservative_join_inspection_20260916_v1/`: fresh C01 original control and earlier generated seed41 row matched existing source-camera identities; zero additions.

- `D/native848_conservative_join_direct_inspection_20260916_v1/`: second-plant SubStem42_view002 is identity-eligible but already belongs to a full 12-view source pool; zero possible net increment.



This joiner currently supports its explicit pair-v2/direct-v2 routes. A new original-only or short-budget producer needs an explicit reviewed consumer; do not disguise a new schema as an old route.



## 5. Current bottlenecks and capacity limits



1. **Rendering:** current direct fixed56 frames cost roughly 31–34 seconds plus setup. Faster camera placement alone does not solve that cost.

2. **Low acceptance yield:** many proposals are too small, occluded, fail the selected cut-to-query trace, or fail robot geometry/workspace checks. Rendering every proposal wastes time.

3. **Limited independent target supply:** the scheduled original TRAIN set has **198 targets**, not 297; 297 includes held-out splits. Only 56 are currently represented. Fifteen selected targets are already at 12 views.

4. **The existing source pool cannot supply 20,000 under the cap:** scheduled TRAIN maximum is 198 × 12 = **2,376**; even all 584 recorded TRAIN geometry candidates give at most **7,008**, before visibility/reachability rejections. At least **1,667 qualified independent targets** are needed for 20,000 at 12 views each.

5. **Generated seed names do not establish new targets:** current curved variants keep the original source target's cap until real novelty/global duplicate qualification exists.

6. **QA and publication:** raw files, repeated frames, geometry descriptors, crops, and diagnostic controls do not increase the deliverable count. Count only selected, verified image/annotation pairs.



At the measured rate, the current pipeline cannot deliver 20,000 by 17:30. The continuation needs both a substantially faster independently validated renderer path and a much larger qualified target supply. Additional supervisors alone cannot provide either.



## 6. Latest unfinished work: original-only camera adapter



### CPU preparation completed



Script: `H/prepare_original848_seed19_probe_v1.py`.



Output: `D/original848_seed19_cached_probe_20260916_v1/`.



- 60 proposals, all 60 camera poses cached; six pass the provisional size/ray screen.

- Three suggested distinct target views: `SubStem_44_original_cpu_002`, `SubStem_42_original_cpu_002`, `SubStem_41_original_cpu_002`.

- Their existing view counts are 5, 3, and 3 respectively, leaving 7, 9, and 9 cap slots before duplicate selection.

- **Zero previously unseen original targets** were found in this small sweep. It is not a target-diversity breakthrough.

- Pose cache SHA: `a3b934d7aaba503dafe7a34ea393521d437b384dae281bdd2ce2185add3dbef4`.

- No native capture was launched for this cache.



### Adapter files exist but are not ready to execute



- `M/native848_original_direct_plan_v1.py`

- `M/native848_original_direct_worker_v1.py`

- `M/native848_original_direct_audit_v1.py`

- `M/native848_original_direct_admission_v1.py`

- Creation helper: `H/create_original848_adapter_v1.py` (already ran; create-only, do not rerun).



These were newly authored from the frozen direct-v2 checks with explicit original-only schemas. AST parsing passed. **Real CPU plan validation failed**, so there is no native-qualified original adapter or supervisor yet.



Exact failure:



```text

ValueError: Original collection source changed

native848_original_direct_plan_v1.py::check

```



Cause identified:



- Cache source plan: `collection_plans/clear848_near_orbit_20260916_v1/plan.json`.

- Anchor scene source plan: `collection_plans/clear_capture_20260915_orbit_v1/plan.json`.

- These are genuinely different paths, not a slash-normalization issue. The current check requires equality.



Next agent must either regenerate a new cache against the authenticated anchor source, or implement an explicit content-bound bridge proving matching original rows, source assets, frozen splits, scene variants, lighting/optics, and robot reference. **Do not simply remove the source check.** Inspect all newly copied worker/audit/admission logic and add meaningful negative/parity tests before native execution.



The worker's intended approach is to reuse `native_scene.prepare_native_scene(anchor_pair_plan)` **before substitution**, retain the complete original greenhouse, apply cached real robot/head poses, and label actual original rows. The anchor's generated catalogue is a source dependency of that existing helper, not geometry that should enter these original frames.



## 7. Execution plan for the next agents



Assign disjoint ownership. Only one coordinator may launch native jobs. CPU agents can work concurrently.



### Agent A — validate a practical short-budget renderer (highest throughput priority)



The next **v3 design exists only in this document/conversation; no v3 module or plan was completed** when the prior subagent stopped.



Implement a separate versioned diagnostic; do not modify frozen reset-v2 or relabel its outputs:



1. Preserve actual acceptance criteria: synchronized fresh RGB/depth/IDs, camera/FK/optics, static-scene checks, native cut visibility, existing strict clear labels and selected full query trace, fresh WorkspaceV2, and individual visual review.

2. Treat cross-render pixel equality as a diagnostic, not a substitute for actual capture validity. The clear-label policy itself does not require stochastic pixel equality; existing direct-v2 admission explicitly requires its fixed56 profile and must remain unchanged.

3. Use actual **A → B → A** camera transitions at 1/2/4 subframes:

   - Scene seed410502, A = reviewed `SubStem_42_view_002`.

   - Same scene, B = `SubStem_44_view_002`, which has a known full-trace hold.

   - Independently include seed410501/SubStem44_view002, the known Fruit_07 interval occlusion, at each tested budget.

4. Allow one explicitly recorded 56-subframe product/material initialization per scene; no hidden settling after each camera move.

5. Run low-budget known-plane, moved-camera, and occluder-on/off controls at each budget. Verify expected centre depths/identities, writer request counts, camera poses and changed buffers. Reuse the logic in `capture_pilot.py::calibration_smoke`, but do not pretend its default budget tests 1/2/4.

6. Save complete native arrays, prim maps, camera settings, request counts, query results, and workspace proofs for every tested frame, including failures.

7. Check global `RTSubframes` floor and motion blur. An effective budget different from the requested budget invalidates that speed result.

8. Review all candidate clear images at native scale. If a budget is usable, implement a **new explicit producer/audit/admission/join route** with its actual contract. It must not claim fixed56 qualification or cross-render equality.

9. Measure accepted, unique images per second, including cold setup and rejection rate, before planning a large run.



### Agent B — increase usable original targets and finish the original adapter



1. Read `D/original848_remaining_coverage_20260916_v1/result.json` and repair the explicit adapter blocker above.

2. Produce a small real original-only capture, independently audit it, derive labels/query trace/workspace, visually review it, and add an explicit original route to the conservative joiner before counting anything.

3. Expand CPU searches to the **142 scheduled original TRAIN targets absent from v13**, using compatible reviewed robot poses and actual original geometry.

4. Reject poor size/occlusion proposals before rendering. Record why a target is unavailable; do not repeatedly rerun the finished broad search.

5. Current `native_view_pose.bounded_reference_root` allows only 0–4 cm outward and ±4 cm lateral offsets while preserving body heading, torso and arms. If a different pose family is needed, define and validate a new contract with actual native geometry/workspace checks; do not silently enlarge these limits.

6. Prefer unused target identities and source pools with capacity over capped diagnostic replacements.



### Agent C — qualify genuinely new target supply



1. Use the actual procedural mesh generator, source lineage and frozen family splits. Generate candidates that preserve valid cut geometry and realistic leaf/stem dimensions.

2. Address novelty calibration: current five blind controls have no negative examples and three ambiguous/out-of-frame contexts. Create suitable visible comparison controls, seal reviews before computing/tuning distances, and retain uncertain decisions.

3. Use the existing 324-context inventory and threshold-free 52,326-pair results as diagnostics. Expand actual source/global coverage where required.

4. Establish defensible grouping and held-out leakage checks before granting additional independent target budgets. A new hash, random seed, camera pose, or renamed stem is insufficient.

5. Report available independent-target capacity before generating thousands of views. The source-pool arithmetic above must be addressed to make 20,000 possible.



### Agent D — review, select and export incrementally



1. Consume only completed owned capture/audit/admission results. Recompute/check source hashes and actual label/workspace provenance.

2. Inspect each eligible native image and its unscaled cut/query context. Record accept/hold with RGB, label and sample hashes. Do not infer a visual review from a passing automated score.

3. Preserve existing visual holds and the original-source 12-view caps. Keep generated annotations' actual target IDs plus explicit ancestry.

4. Run exact encoded/decoded-image, sample and recorded-camera checks, split checks, and the appropriate global grouping checks.

5. Inspect before publishing. Use a new `v14`, `v15`, etc. selection only when there is a justified addition or meaningful replacement. Do not overwrite v13.

6. Export native RGB/labels/masks/depth plus query-only auxiliary crops. Validate dimensions, path existence, hashes, row/image correspondence and crop pixel provenance. Count images, not masks or crops.

7. Update `dataset_vlm.md` with selected counts, unique targets/families, actual accepted throughput, active owner identity, and remaining blockers after each batch.



## 8. Capture and publication sequence



Use this order for every new producer or batch:



```text

actual source geometry + frozen split + source-cap availability

  → CPU pose/FK/optics check + provisional size/visibility screen

  → immutable plan and implementation/source hashes

  → one resource-checked serial native owner

  → owned process exit + complete native buffers

  → independent buffer/geometry audit

  → existing strict clarity + selected query trace + fresh workspace

  → actual visual review

  → duplicate/group/cap selection

  → immutable portable export + verified count

```



No raw capture becomes TRAIN merely because the worker returned 0. Clear images from a capped pool may replace another view but cannot increase the count.



For the existing conservative joiner, read its preparation README for the pinned request format. Its CLI separates `inspect` and `join`:



```text

python -B -m sim_data.native_dataset.native848_conservative_join_v1 inspect

  --request <new request.json>

  --request-sha256 <actual sha256>

  --output <new disjoint output>

```



The `join` subcommand uses the same argument shape. The mixed exporter uses:



```text

python -B H/export_conservative_mixed848_v1.py

  --selection <new selection.json>

  --selection-sha256 <actual sha256>

  --output <new portable checkpoint>

```



`H` in this example is a document abbreviation; expand it to the real path. These commands do not authorize supplying a new producer schema to an old consumer without implementing and reviewing support.



## 9. Process state at handoff



At 18:00 KST the capture queue is idle. Original owners 63976 and 78376 have completed. See section 11 before launching; recheck exact process identities because PIDs may be reused.



Completed/stopped historical owners:



| Owner | Historical PID | State |

|---|---:|---|

| Original resumed queue | 75396 | Complete |

| Blind controls v3 | 49080 | Complete |

| Prefiltered second plant | 13396 | Complete |

| Reset probe v1 | 62764 | Cooperatively stopped before native |

| Reset probe v2 | 80616 | Complete; four long controls audited, no short profile qualified |

| Two-family fixed56 wave | 78440 | Complete; native children exited, admissions written |



Agents resumed independent speed validation, original-reference support, and review/CPU pose preparation. Approval-service capacity failures intermittently block tool calls. Check live status rather than assuming an agent is active.



Durable older progress log: `H/WORKLOG.md`. `H/current_status_20260916_v12.json` predates the final wave and original CPU probe; **this document and the newest result receipts are newer**.



## 10. Completion criteria



- At least 20,000 actually selected TRAIN image/annotation pairs at native 848 × 408, with sufficient independently qualified targets for the cap policy.

- Clear cut details and cut-to-query association, appropriate robot workspace, actual per-image reviews, valid source/label provenance and frozen splits.

- No legacy-image padding, duplicated views, crop counting, failed-frame counting, or unqualified cap resets.

- A portable, hash-verified export with explicit class/target/family coverage and unresolved QA limitations reported honestly.

- Dataset preparation complete; training remains the user's responsibility.



Until these criteria are met, report the actual count and blockers. Do not mark the 20,000 goal complete because a deadline arrives or a capture queue finishes.





## 11. Authorized continuation after the handoff



The user explicitly requested execution of this plan. The v14 selection and portable export are complete: **358 TRAIN / 496 total**.



### Completed repairs and checks



- Preserved the failed source-mismatched original cachev1. Created a new preparation `H/prepare_original848_seed19_probe_v2.py` using the actual anchor source plan `clear_capture_20260915_orbit_v1`.

- New cache: `D/original848_seed19_cached_probe_20260916_v2/pose_cache.json`, SHA `b8b01039bc4db7c2eb4e0fc44053ab24efe0c6a7db0c32b06e5fe9880db276cf`. Same60cachedposes/6provisionalviews/3suggestedexistingtargets; no source check was removed.

- Full original adapter CPU check now **passes**; wrong-resolution, wrong-profile, generated-geometry and cap-reset test inputs are rejected. Validation elapsed24.82s.

- Native plan: `D/original848_seed19_direct_prepare_20260916_v1/plan.json`, SHA `dc59e2bc875d77e1da6f80673a8e4e303ba67d28a97749428c04e218baf1be19`.

- Preparation receipt SHA `6e9640d9b9b7f062fef06546dde0cd3952e06b97ab7fd664936404a8b9d02aec`.

- Independent adapter/owner read-through found no execution blocker. Source-cap headroom remains enforced at final selection; this capture alone does not approve inclusion.

- New consumer `M/native_dataset/native848_conservative_join_v2.py` supports the explicit original-only route.14CPUtests pass; frozenv1 remains unchanged. Readiness: `D/native848_original_direct_consumer_preparation_20260916_v1/readiness.json`, SHA `debd474324aa9464546dacea9b7187baf6036f9a9fde9291c60d4abac5d74035`. Actual empty and positive original-only integrations passed; the positive result is exported as v14. Future original-reference v2 production still requires explicit consumer support.

- Seed103 previous-wave visual review is now sealed at `R/native848_reference_wave_seed103_20260916_v1/assistant_reviews.json`, SHA `7129af40d906202f9fbf865e988b6bdc9df5f62547a8b93766ba51dc66bf7198`. It accepts static clarity; the12-view source pool is full, so TRAINincrement0.



### Completed bounded original capture v1



- Owner: `H/run_original_direct848_v1.py`, SHA `37bd8681f91ca74f2fafd9fd564870010e824956351f3ae5f8550e922327c8b4`.

- Historical launch identity: PID63976, creation `/Date(1789547359180)/`, executable `C:\Users\USER\miniconda3\python.exe`; full identity in `H/original_direct848_owner_v1.json`.

- Trial: `D/original848_seed19_direct_capture_20260916_v1/`.

- Admission: `D/original848_seed19_direct_admission_20260916_v1/`.

- Logs: `H/original_direct848_v1.stdout.log` and `.stderr.log`, plus trial `worker.log`.

- Three planned original targets:44,42,41 at `_original_cpu_002`. The run uses fixed56 and preserves848?408, original assets, source caps and complete native evidence.

- Completed in 103.25 seconds. All three poses were held by static geometry screening for possible right-arm visual-bound overlap with neighbouring stems; no RGB captured. This is a possible overlap, not a proven exact collision. Empty review/admission preserved.



### Completed original capture v2 and v14 delivery



- Owner: H/run_original_direct848_v2.py, historical PID 78376, creation 1789547662882; native child 76304 also exited. Frozen original v1 producer, three other cached poses of seed19/SubStem44.

- Plan: D/original848_seed19_direct_prepare_20260916_v2/plan.json, SHA 8fb921c5ba0ab9320ad0b565455b3fc5eb68c8dea5b0bbd7b465ea6b31c8a301.

- Three native frames captured. Pose000 passed strict clarity, full query trace, workspace and actual visual review. Pose001 failed local clarity; pose004 was held by full trace. Failed frames excluded.

- Trial: D/original848_seed19_direct_capture_20260916_v2; result SHA 4a660db6eeffa4274fb0ab693ac00fb1d0236f9e8ca6895f7177d2ad0d92e7bd.

- Admission: D/original848_seed19_direct_admission_20260916_v2/result.json, SHA b7f526365d39097129959f750f989e76c2fd03d2887f8eb080327ec2d528c7fb.

- Visual review: R/original848_seed19_direct_20260916_v2/assistant_reviews.json, SHA 57dae27be8da4a4105982d7e09cc52efa275f3968a6a708b69dc9d7fd2c570a8.

- Join request: D/original848_v14_join_prepare_20260916_v1/request.json, SHA e7544c342afc8cc78b5d3cae4f43e447e35b46272210b8dd4c3041de9e76e287.

- Selection adds one image, no removals, preserves 30 holds and source caps (seed19/44 pool 5 to 6).

- Portable v14 export manifest SHA 1f30eb45d79b111966c693513f3c51ab766201428a7bdb653da4b7d40397cce9; state immutable_native848_mixed_checkpoint_copied_and_verified; training_approved=false.



### New original target survey and active implementation



- Surveyed all 235 reviewed references with real robot head joints and preserved body/arm/torso. Of 1,846 proposals, 1,825 solved, 29 passed provisional size checks, and six poses cover five unselected targets. CPU screening is not native/workspace acceptance.

- D/original848_all_bank_body_survey_20260916_v2/result.json SHA 3bf7e8b16f686462ca388f5f785c115ce20a87f938d6d3636b1fd2e755952b23.

- Same folder shortlist.json SHA 32f462414efeb6838ed0609e74faafe0479d2b6ae46e82e65f5f2493631b0263.

- Targets: seed41/SubStem38 and39, seed43/SubStem48, seed89/SubStem40 and46. All still unselected in v14.

- Best widths 8.05 to10.16px, intervals12.32 to16.35px;11/11 provisional partial-scene ray probes. Full native geometry, clarity, trace, workspace and visual review remain required.

- Preparing a separate bank-reference original-only adapter v2 using actual reviewed body poses without a generated-catalogue anchor. Caches must bind v14. Frozen source assets, split groups and source caps remain unchanged.

- Speed agent is implementing separate short-budget-v3 diagnostics after its required reads succeeded. Actual1/2/4-subframe camera-change/occluder controls and full clarity/query/workspace evidence are required. Long-render pixel equality alone will not suppress short diagnostic frames. No short profile is production-qualified.



### Approval-service interruptions and deadline



Automatic approval review intermittently rejects narrow local reads/new-file work because its selected model is at capacity. An older agent report records an account-usage-limit write rejection, but its original invocation timing is unverified; do not describe it as a newly observed event. Do not bypass a rejection through another execution route. The required speed-agent reads later succeeded and work resumed; original-reference helper reads remain intermittently blocked.



The17:30 deadline was missed for20,000 images. Current count, throughput and independent-target capacity remain insufficient. Do not report completion or weaken the source-cap/quality rules.



## 12. Live execution update — 2026-09-16 18:20 KST



This section supersedes the older process-state notes above. Selected count remains **358 TRAIN / 48 validation / 90 test = 496 images** in v14.



### Short-budget diagnostic now launched



- New frozen diagnostic: `M/native848_short_validation_v3.py`, SHA `86f9b289bd9a918198bcbe9137f65bb9102048c07d3035900a29c9fc4e6639f1`.

- CPU preparation: `D/native848_short_validation_prepare_20260916_v3/handoff.json`, SHA `b90284532f74e319c6e9af81a678e96d8f6903fe3c6562338f9a2118586537de`; 19 CPU checks and full source/geometry validation passed.

- Plan SHA `e2bb958b2aa207e08d384d2c7d59b9858248dec5d9fe59dfd5e3cfdf031585fa`.

- Independent code/launch review: `D/native848_short_validation_independent_review_20260916_v3/result.json`, SHA `57b919df4b2e26684ebfa0117b6a0ce03385d9ee5059534a887fa46212735c88`. No blocking source findings; this does not qualify production.

- Owner `H/run_short_validation848_v3.py`, SHA `3c4f30017d9daa1469a4bbd3b7c4d2eb5ab514a316a9efefbee5ba9ab33b1395`.

- Launched hidden owner PID **67076**, creation **1789550382881**, executable `C:\Users\USER\miniconda3\python.exe`. Exact identity: `H/short_validation848_owner_v3.json`.

- Trial `D/native848_short_validation_trial_20260916_v3`; owner logs `H/short_validation848_v3.stdout.log` and `.stderr.log`.

- Tests actual 1/2/4 subframes, known camera/occluder controls, A→B→A and a fruit-negative case. Controls now use the same RealTimePathTracing renderer. Every raw callback is recorded before normalization; exact schedules, settings, freshness and actual labels/query/workspace are replayed.

- No profile is production-qualified yet; diagnostic images add zero TRAIN rows.



### Original unseen-target path



- Created separate `M/native848_original_reference_v2.py`, `native848_original_scene_v2.py` and `native848_original_direct_{plan,worker,audit,admission}_v2.py`. Frozen v1 modules are unchanged.

- Source-level independent review found no actual launch blocker. First seed41 anchor passed full CPU source/world/FK validation (877 source bindings). All five candidate calibration key sets and non-pose values were independently checked against their references.

- An optional plan-only exact-key-set hardening edit was capacity-rejected and not applied. The actual prepared candidates have complete keys; worker/audit still enforce complete native calibration. Do not claim that rejected edit was implemented.

- `H/prepare_original848_unseen_shortlist_v2.py` is running CPU-only and has prepared the first three of five plans. Output: `D/original848_unseen_shortlist_prepare_20260916_v2`.

- New serial owner `H/run_original848_unseen_shortlist_v2.py`, SHA `bd73510bf1812ea914fbf87ed226ee89744bc0c789bb32645f0b1d59311fcfad`, is syntax-checked and awaiting review/full preparation before launch. It captures one view per proposed target in separate original scenes, records actual owned exits, and runs fresh audits/admission.

- Consumer agent is implementing a new explicit v3 route for original-reference-v2 admissions; existing consumer v2 cannot accept that route.

- Native rendering remains serialized. Do not start overlapping Isaac captures.



### Independent target supply



Two existing matched original-control render pairs can support new blind repeatability controls without new rendering. A separate new protocol/public packet preparation is underway; their labels are unassigned. Preserve previous uncertain controls. Identity negatives alone do not qualify new generated target pools: small nonzero changes and independent clear positives are still needed.



Approval-service capacity errors remain intermittent; several previously blocked original operations succeeded after narrower same-tool retries. No safety rejection was bypassed. The deadline remains missed and the 20,000-image target remains unmet.





## 13. Execution update — 2026-09-16 18:34 KST



### Original-target native collection is active



All five CPU plans completed, with result `D/original848_unseen_shortlist_prepare_20260916_v2/result.json`, SHA `aa9537fbfc70420750e0ad8aa34bde1b7042f83e0f14e43feeceb0b6a30e6ecd`. Independent owner review `D/original848_unseen_shortlist_owner_independent_review_20260916_v2/result.json`, SHA `fad47b2b757c1d7329006430aa2ea7ad8fde0570b5c6d50cc3ba87bd3e9ae47a`, confirms complete calibration and all actual preparation pins.



- Owner PID **80584**, creation **1789550817666**, CPU Python; identity `H/original848_unseen_owner_v2.json`.

- Owner script `H/run_original848_unseen_shortlist_v2.py`; logs `H/original848_unseen_v2.stdout.log` and `.stderr.log`.

- Root output `D/original848_unseen_shortlist_native_20260916_v2`; each case has separate `trial` and `admission`.

- First case `61fa199240de_SubStem_38` completed: native child78416 exited, independent audit and fresh strict clarity/full query/workspace passed. Actual visual review is in progress. Its captured frame had10.16px estimated petiole width and16.35px interval.

- Second case `3feac01f8f9d_SubStem_39` started as native child81500. Later cases remain queued inside this owner.

- The serial mutex is fail-closed/nonblocking. **Do not launch another owner while80584 is active, including during its between-case CPU audit gaps.** Verify exact identity and final completion/failure before the next owner.



### Speed diagnostic v3 failed; separate v4 prepared



V3 owner67076 and native60732 exited; the native owned-exit receipt reports code1 after235.64s. Controls:4subframes passed5/5;1/2 had stale camera/depth on transitions. No greenhouse short-budget frame was successfully tested. All nine transition attempts hit guard errors caused by starting the monitor before lazy writer graph initialization. The final complete used-layer map differed from the baseline captured before intentional plant substitution; that result alone is not proof that a source file changed.



Preserved v3 failure and raw callback evidence in `D/native848_short_validation_trial_20260916_v3`. No production speed qualification or image credit resulted.



New frozen `M/native848_short_validation_v4.py` SHA `2f2ac2243411a0122501c5e3243f45ea2e936f44e2eeaf54f834ec294c52cc8b` verifies pre/post-substitution source content, records the complete loaded-layer baseline after explicit56warmup, then starts the unchanged static guard. No guard exclusion or quality gate was relaxed.



- Handoff `D/native848_short_validation_prepare_20260916_v4/handoff.json`, SHA `e52b02ec3f05196bad497107a8d55c61a58cd75db42de4fd9aaf8b66e79f8ce9`.

- Plan SHA `6c879872099e174fcae9c7f76a81dcfddc125552c7edf0072d8c86678a0cbd7f`;19CPU checks plus full source/geometry checks passed.

- New owner `H/run_short_validation848_v4.py`, SHA `22cba239207e9cfb5b69880ef3ea48ad78ec982a56f6d353d3b0140f22acc764`.

- Independent review `D/native848_short_validation_independent_review_20260916_v4/result.json`, SHA `db5ccf6b03565db3a85437c065ccfb736d07b33c0a764a75d534e3e14bad55c4`; no launch blockers. Saved initial baseline token is not separately cross-compared to first short-frame guard in replay; runtime ordering and per-frame guards remain enforced, noted as a diagnostic limitation.

- **V4 has not launched.** It must wait for the entire five-case owner80584 to finish.



### Consumer and novelty diagnostics



- New `M/native_dataset/native848_conservative_join_v3.py`, SHA `2e08365523821861d84a71402f254e9a785e631b0b0a29d202b6c871c7565e3d`, adds the explicit original-reference-v2 route.39CPU tests pass; a real existing accepted workspace certificate passed saved-joint replay. Frozenv1/v2 are unchanged. Existing mixed exporter supports the new row kind. Actual new-v2 integration remains pending visual review.

- Fresh blind reviews inspected all8public native/context PNGs: bothK17/K42 labelled insufficient_change, with the second limited by junction occlusion. Seal `R/blind_native848_geometry_review_20260916_v2/seal.json`, SHA `cbbf08e556332e72a6dc6a6ffbc7e4841b0c4f6252b9f714e66add65e8892724`.

- Post-seal actual original-control descriptors both measured0.0. Result `D/native848_sealed_repeatability_calibration_20260916_v1/result.json`, SHA `64cd35f8f680ff2649cddae7e8002e40448a16870e186351a6337ebf7a2b01e4`.

- Combined fixed controls:2positive/2negative/3uncertain; diagnostic midpoint0.16358828415684445, pairwise leave-one-out fits supported. **This is not a global qualification threshold:** the two positives share the same source target and one rendered side; negative controls establish repeatability only, with no small nonzero-change coverage. No novelty credit or cap reset is authorized.

- CPU work continues on bounded extra poses for the first newly clear original target and controlled single-target changes on another source family.



Published count remains **358 TRAIN / 496 total** until visual review and actual v3 selection/export pass.







## 14. Execution update ? 2026-09-16 19:04 KST



This section supersedes earlier live-process notes. Current exported checkpoint remains v14 (358 TRAIN / 496 total) until v15 selection and export finish.



### Five original targets completed; three reviewed accepts



Owner80584 completed and exited. Batch `D/original848_unseen_shortlist_native_20260916_v2/result.json` records five actual captures. Strict automated checks excluded seed41/SubStem39 (no usable visible query connected to cut) and seed89/SubStem40 (local clarity). Their exclusions are preserved.



Actual full native images and lossless association crops were visually accepted for seed41/SubStem38, seed43/SubStem48 and seed89/SubStem46. Review bundle `R/original848_unseen_shortlist_20260916_v2/review_bundle.json`, SHA `c60143541f18ea314855ca14e282dd5043516560c826f8e2ae39bc1d17f500a5`, pins all five admissions and three visual-review scopes.



New v15 request `D/original848_v15_join_prepare_20260916_v1/request.json`, SHA `f4aa13254615aab74b29d29a34700dbdfe92ab2963399fba0a422bca5af6cce9`, includes exactly those three accepts and prior v14. Actual consumer-v3 inspection passed all three, each from a previously unselected original-target pool, and preserves all30holds. The actual join is running toward `D/clear848_reviewed_workspace_checkpoint_20260916_v15`; portable v15 export has not yet completed. Never count selection/export success from the inspection alone.



### V4 speed diagnostic active



V4 launched after the entire original owner exited. Identity `H/short_validation848_owner_v4.json`: CPU owner80928, creation1789552524778; native owned child77796. Output `D/native848_short_validation_trial_20260916_v4`, logs `H/short_validation848_v4.stdout.log` and `.stderr.log`.



Actual transition results now exist. Budgets1/2 fail camera/callback freshness after camera changes. Budget4 captured registered buffers for A-before/B/A-return, but A-return failed strict clarity and full trace while A-before passed. The B trace hold remains a hold. The separate fruit-negative scene and final independent audit are pending. No short profile is production-qualified; no diagnostic frames count as TRAIN.



### Extra original views: use the wider replacement only



The first eight-pose proposal `D/original848_substem38_offsets_prepare_20260916_v1` was never launched. Its cutpoints spanned only24?16pixels near the image centre, risking a centre prior. Preserve it as an unused alternative.



Replacement `D/original848_substem38_wide_prepare_20260916_v2`: planSHA `a82c4c0f78a271e400c506a150b211ebe08c993c193ac7cab244e05d24887a9a`, cacheSHA `8cec060b92674f9896c41247f519ebfbc773154cf629e5b736efa7a582ec16d9`, resultSHA `62cdc1e451b6b34d4c98bd057338ac60213abc09035a58d70f59fe9a06136a9f`. Eight unique bounded robot roots cover eight non-centre strata spanning40% of image width and height. Projected widths9.58?11.89px, intervals15.54?20.77px, minimum source-query margin20.96px.995 partial original-mesh rays passed; full native checks remain required. Base1+replacement8 reserves9/12 views for the SAME original target. **Never combine both eight-view alternatives.**



New unlaunched owner `H/run_original848_substem38_wide_v2.py`, SHA `a3a9301f7c5dc3384e8ddfc3a0f05fef5bd9bb36fd1671ffaf7dccd34bf622c5`, awaits independent review and entire v4owner exit. Fixed56 capture remains unchanged. Output will be `D/original848_substem38_wide_trial_20260916_v2` with separate wide admission.



### Two explicit source-frame geometry controls prepared



New v1 controlled generator hit an authored-versus-composed USD metadata-checker bug; preserve its code and two hold receipts. New v2 fixes that comparison and adds a regression test.16focused CPU tests pass. The fixed2mm/25mm controls both pass generated geometry and catalogue replay; there was no amplitude/seed search.



Result `D/native848_explicit_control_seed41_prepare_20260916_v2/result.json`, SHA `db7eb773aed803869384a17c73bfdf4e2566210e05db6054445e9e7679bbc2e5`. Assets `data/sim_data/generated_plants/explicit_control_seed41_20260916_v2/{seed41_full_ec2_002,seed41_full_ec2_025}`. Only source seed41/SubStem40 and Leaf018?026 change;401other components retain exact visible attributes. Attachment/radii are preserved. The source-relative recipe uses the measured TRAIN envelope; it does not apply the old random-direction sampler's25?110degree proposal box. Observed source attachment is about116.8degrees, within the measured TRAIN envelope.



Separate `M/native848_controlled_pair_{plan,worker,audit}_v1.py` and `native848_controlled_scene_v1.py` bind the corrected controlled catalogue v2 to the original native scene. Both exact CPU pair plans completed in `D/controlled_native848_pair_prepare_20260916_v1`. Each is exactly original-control plus generated-variant at fixed56 native subframes; no scene isolation, novelty credit or cap reset. New owner `H/run_controlled_native848_v1.py` is unlaunched and awaiting independent source/owner review. These controls need actual render, independent saved-buffer audit and blind visual assessment before any geometry-distance calibration. They currently add zero dataset images.





## 15. Execution update ? 2026-09-16 19:10 KST



### V15 delivery completed



Actual v3 join passed and added exactly three original-reference images, with no removals and all30holds retained. Selection `D/clear848_reviewed_workspace_checkpoint_20260916_v15/selection.json`, SHA `a42fe2ae7d3d2da4bf3553150a485ff403580576b632d2e33e2b693b4748695f`. Portable `data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_20260916_v15` finished file/crop/hash verification:361TRAIN+48validation+90test=499total;59originalTRAINtargets/15families; all361TRAINclear. ManifestSHA `27e03c3755ad6ff1a88160f93446e0293e5ea2532671777994f3004a07142c3d`. Full interval contained in485/499querycrops (352/361TRAIN). Training approval remains false pending final diversity/human-heldout qualification. Deficit19,639TRAIN.



### V4 completed; next speed experiment remains separate



V4 child77796 owned exit0 after457.383s; entire owner80928 exited. ResultSHA `9f8577042e32775034804f6ba1bdc44830b4d6399047500795507cc0c176f6c8`; independent auditSHA `76695169896634e9bc74753fc3db3c0e41a6f98c14fcafd9e3ea277a284030cf`. Eight actual valid greenhouse buffers replayed, four transition attempts rejected as stale. Negative fruit cases correctly rejected at all three budgets. No production profile qualified.



Clarification: budget4 A-return failed upstream local clarity; full query scan was skipped, not independently evaluated and failed. Same projected interval14.262px/width10px/all15target probes and proximal visibility1.0, but local dark fraction0.2119 exceeded the unchanged0.10limit (before0.0); median luma fell from74.50to45.00. Root visually inspected both actual full848 images and observed smeared/ghosted edges after the camera move. A separate diagnostic-v5 with budgets8/16, same renderer/settings, unchanged gates and actual raw callback controls is being CPU-prepared. No DLSS/settings change or fixed56 production change is authorized by these results.



### Wider original capture launched



Independent review `D/original848_substem38_wide_independent_review_20260916_v2/result.json`, SHA `0b164ea3f60a04c3abf1b734487b90bd45fab666eb27774a982acdf541dc2d53`, found no blockers. Actual root launched hidden owner70200, creation1789553334914 at19:08:55KST. Identity `H/original848_substem38_wide_owner_v2.json`; logs `H/original848_substem38_wide_v2.stdout.log` and `.stderr.log`. The owner checks the entire v4 predecessor exited, serial mutex/resources, exact8pose plan and mutually exclusive narrow alternative. Wait for this entire owner to exit before any new native run.



### Controlled pairs launch-ready, still unlaunched



Exact native CPU preparation `D/controlled_native848_pair_prepare_20260916_v1/result.json`, SHA `b90b6ead87eb1518449b6bf9a8187468250a2dee9b9ab2896c928e6fd46163bb`. E01planSHA `e3421edaa5e2e479783fb0f7747c7f0d9fc2506e740c3306023b9ef992bf8b54`; E02planSHA `dc32975831d673b15fe35d3c6d9e56d290386d8695327f4bfbb29fed101e6aaf`.



Independent owner/adapter review `D/native848_controlled_adapter_independent_review_20260916_v1/result.json`, SHA `3430cdc582d1cc38be5a5040faae0a3d0845f5c8739a572707febdea9c4c2d63`, verified1820unique source/code/asset pins and found no blockers. Owner `H/run_controlled_native848_v1.py`, SHA `1c17fc130db0865da88a559c29ef45c8cec3be89f6a99dbd8cbde6d39632c74e`, accepts `--preparation-sha256 b90b6... --review <reviewpath> --review-sha256 3430...`. Root must wait for entire wide owner70200exit before launch. Exactly2pairs/4frames, no automatic retries, no data admission. Outputs `D/controlled_native848_pair_trial_20260916_v1`. Actual blind assessment precedes any distance computation or novelty claim.





## 16. Execution update ? 2026-09-16 19:27 KST



### Wide38 collection/review complete; v16 CPU delivery pending



Wide owner70200 completed and exited. Of eight planned poses, four hit full native robot geometry holds; four captured frames passed strict clarity/query/workspace and actual full-image plus lossless-crop visual review. Actual accepted strata are(.3,.3),(.3,.5),(.3,.7),(.5,.7), narrower than the planned eight-stratum coverage. Native child76448 exited0 after238.647714s. Trial auditSHA `72777d46e8ee200b94ded26b4c2a635a6478dcf2ee687558c2d789af902cf8c4`.



Admission `D/original848_substem38_wide_admission_20260916_v2/result.json`, SHA `f02b0c95d0cadb1a156c22b0d09663fdfd142c28d11d6a9a6ddc6f3c99a4f247`. Review `R/original848_substem38_wide_20260916_v2/assistant_reviews.json`, SHA `02c23fd52f4c557335d3035c2319e13e4020ec771ae9db9a661bd59a21b6e8e8`; scopeSHA `263dc9bb5f9305b0bf4af9cbf50a0e635fb300ff9cd82da8ae86e0c3569c6685`. Delivery agent is creating a new v16 join/export, preserving all holds and seed41/SubStem38 pool1+4=5/12. No global novelty claim or final training approval.



### Controlled native v1 failed; v2 startup fix reviewed



Owner11804, creation1789553767309, launched19:16:07 and exited. E01 child79464 owned exit0/15.889393s, but actual `capture/failure.json` reports mixed USD ABI failures (`GfVec3f`, `TfNotice`) during SimulationApp startup. The owner correctly failed because the failure receipt exists. No image was captured and E02 never launched. Evidence: `D/controlled_native848_pair_trial_20260916_v1`.



Cause: the new v1 plan's non-full check still calls the controlled catalogue, importing USD before SimulationApp configures native libraries. Frozen v1 is preserved. Separate v2 worker runs only schema/flags/hash checks before app initialization and full geometry/catalogue checks afterwards. Assets, fixed amplitudes, camera, render budget and all audit/quality rules are unchanged.



- New modules `M/native848_controlled_pair_{plan,worker,audit}_v2.py`, `native848_controlled_scene_v2.py`.

- CPU preparation `D/controlled_native848_pair_prepare_20260916_v2/result.json`, SHA `270e4f6782be5cae2399b626907aefa4cdbf682895f5089f7c4a96a08069bd47`.

- Owner `H/run_controlled_native848_v2.py`, SHA `9de8b7d1dbc6545ce6116abfa091e54807697f551eafaf189ca9df43313568e8`.

- Independent review `D/native848_controlled_adapter_independent_review_20260916_v2/result.json`, SHA `fd163242bba447dec8df6c408d71650b3fd64e132aef15f59aa961a5fecc5986`;1831pins verified, no blockers. Fresh-child import regression detected v1's premature geometry check and confirmed no pre-app USD/native import on v2. This is not native execution.

- Unlaunched output `D/controlled_native848_pair_trial_20260916_v2`; same two pairs/four-frame maximum. CLI uses preparation pin above, review path and review pin. Wait entire v5owner15044exit first.



### Speed diagnostic v5 launched



ModuleSHA `8deedc4fb36f57f6f8757877a3dd3a7af9d304b107a73f41932c8342023cd0a2`, planSHA `a8da0a0a446d2a06c26ea193dc92c1bbb762192a48f7ff954cad7c723a446c51`, handoff `D/native848_short_validation_prepare_20260916_v5/handoff.json` SHA `73f088914e70d5f123b668390889b27be359c026c297f16e755562886fb60853`.16CPU checks/full source checks passed. Root independently reviewed complete v4?v5diff and actual exited predecessor identities: review `D/native848_short_validation_independent_review_20260916_v5/result.json`, SHA `3d243d098273655f470348079af9dd5803f429f55b6f1c01f97481dda2f6ea3f`.



Owner `H/run_short_validation848_v5.py`, SHA `8b94e820916c360c690058446c44befd1c2b682b56e2bfdfbe20d3788e159e5b`, launched hidden PID15044, creation1789554328770 at19:25:29KST. Identity `H/short_validation848_owner_v5.json`; logs `H/short_validation848_v5.stdout.log` and `.stderr.log`. Tests8/16 only, ten controls and eight greenhouse observations, same renderer/DLSS/reset profile and acceptance checks. Output `D/native848_short_validation_trial_20260916_v5`. No profile is production-qualified; do not change the frozen original fixed56 producer. Throughput agent is doing read-only integration analysis for a possible future explicit profile if actual checks and visuals pass.



### Two more original wide-view CPU preparations complete



Root adapted the bounded original proposal builder to the two other actually reviewed targets, pinned v15 and their exact base visual/sample/query evidence, and added fixed source-query grid/margin checks. An initial new-script label/calibration field mismatch failed before output creation; corrected scripts use the hash-verified native sample calibration when the new label schema omits it. Frozen source producers are unchanged.



- Seed43/SubStem48:3/8proposals CPU-eligible; plan `D/original848_substem48_wide_prepare_20260916_v1/plan.json`, SHA `ec6f94832746c6c3a3edca5ddfdbe246bb77cc92e8076cdaaa52ad27329fea2c`; width8.84?9.80px, interval12.83?14.05px; one selectedbase+3=4/12.

- Seed89/SubStem46:5/8eligible; plan `D/original848_substem46_wide_prepare_20260916_v1/plan.json`, SHA `94f6c77ec152d68c2933f5e9f00a8f762bb5067f2ab0b2fc196119aee8965657`; width8.04?8.80px, interval13.04?14.05px; onebase+5=6/12.

- Aggregate `D/original848_two_targets_wide_prepare_20260916_v1/result.json`, SHA `fa5934ba217ffcbed2f44911460a2ad0586853aa8f2311be9b5e72619a291e5d`.

- New owner `H/run_original848_two_targets_wide_v1.py`, SHA `5b51e1c1d3ed8af63edc99be716dbaaeabcf8b55f82d8d7a174f0dccc7e00dba`; maximum3+5frames in two serial scenes, requires preparation and independent review pins. Review agent is examining it. Not launched; full native geometry/clarity/query/workspace/visual admission remains required. Output `D/original848_two_targets_wide_native_20260916_v1`.



### V15 selected-image metrics completed, without a duplicate threshold



Diagnostic `D/native848_selected_v15_metric_diagnostic_20260916_v1/result.json`, SHA `976b80f02be16bbfbd123385d11b475638653923f91cd703e8320d921747c400`. All499images authenticated;492support the frozen native metric's complete129?129cut patch. Seven edge-patch images remain unknown (5TRAIN/2TEST), leaving3465unknownpairs. All120786supportedpairs resolved:118828authenticatedv13pairs reused exactly+1958newpairs;124frozen-metric parity checks/readback passed. Exact encoded/decoded/view/camera graph has499singletons and no cross-split aliases. No binary near-image threshold or global independence qualification is inferred. Per-target counts/cap headroom, nearest numeric pairs and unknown comparisons are saved.



Image-free localization baselines are also recorded, using Euclidean native-pixel error against the nominal answer. TRAIN-fitted mean[431.06,184.50], fixedcentre[424,204], query-copy: validation mean203.00/204.51/105.06px; test200.51/203.62/106.70px. Per-image and equal-weight per-target summaries are included. No heldout tuning, model training or status-classification claim.





## 17. Execution update ? 2026-09-16 19:32 KST



**V16 selection and portable export are complete:365TRAIN+48validation+90test=503total.** No prior rows removed; all30holds retained. Seed41/SubStem38 now has5selected views, same source pool.59TRAINtargets/15families unchanged; deficit19,635TRAIN.



- Join request `D/original848_v16_join_prepare_20260916_v1/request.json`, SHA `2b9768889fa62fc62dae0437c1d862f85898a6dab76bf7a682c6c1a77bb18d07`.

- Selection `D/clear848_reviewed_workspace_checkpoint_20260916_v16/selection.json`, SHA `5ce821ca4ff043510422c04fa39f3a9eeea8cb41b9cf33ed4317bb42d4fd4a8f`.

- Portable `data/sim_data/dataset_checkpoints/tomato_cutpoint_848x408_20260916_v16`; manifestSHA `84d2fc7fc20eb35ef2c929bcb5784a49ab5f733c104cab0d5aa1d556901e68d1`.4,045copied files verified;1,383,568,786bytes before result. Query-onlycrop fullintervalcoverage489/503 (356/365TRAIN). Actual split index files are `train/index.jsonl`, `validation/index.jsonl`, `test/index.jsonl`.

- Selected-image metric diagnostic is being extended from authenticatedv15; no new threshold or global qualification.



**Two-target3+5wide batch is now independently reviewed, still unlaunched.** Receipt `D/original848_two_targets_wide_independent_review_20260916_v1/result.json`, SHA `ee56c47b8551619faaeb1d83200d99fa7ee5d8f131aeaa4c3fb4d8333770037c`.9,953bindings and740saved donor-only rays checked; no blockers. Root must verify the entire preceding native owner has exited before launch; this owner has a per-case mutex and does not itself pin the whole preceding owner. All three seed43/SubStem48 candidates are in the left image column; do not claim balanced eight-stratum coverage.



Active speed-v5owner15044 launched nativechild78240. Its ten known-plane/camera/occluder controls passed at8/16; full greenhouse observations, audit and actual visuals remain pending. A short profile still is not production-qualified. Controlled-pairsv2 remain the next reviewed native task after the entire v5owner exits.





## 18. Collection correction after the count challenge



Current export is **v17: 366 TRAIN / 504 total**, deficit **19,634 TRAIN**. It adds the actually reviewed `seed89_full/SubStem_46` additional001 frame. Manifest SHA `ece1e1d0b01545a0de737cf72bb73ebd58c2dc446d4bcda51be6c2cada9c8b0c`; selection SHA `87565d85b883a884002f8dc122932aaf3271677ec5b16d90e7925f2b22d4d0ae`. All 30 holds and prior images remain. The count is not 20k and the deadline was missed.



### Verified causes and fixes



- The prior 3+5 batch yielded one native candidate in 236.39 native seconds. Seven poses failed robot-to-scene body geometry; donor-only visibility checks could not detect those failures.

- New pure-USD full-scene CPU screening reproduced **all eight saved native screen dictionaries exactly**. Receipt `D/original848_cpu_fullscene_parity_20260916_v1/result.json`, SHA `b2d4235cb10baa184ab8aa26bc946d67ae5bc6df72ab84c9b3503d480989d550`. This is CPU evidence, not a native capture or label.

- Original reset8 adaptation captured three excluded A-B-A controls, all automated and actual visual passes. Native owned duration 191.68s; snapshot times 6.48-6.67s; steady total frame times 7.78-8.33s. First-frame geometry/warmup and repeated CPU validation remain substantial overhead. Do not report snapshot timing as end-to-end accepted-image throughput.

- Q2 actual reviews: `R/original848_short_adaptation_20260916_v1/assistant_reviews.json`, SHA `ea5c2aaa7c35c69cff77e20c3db504adf0a1f739ca0c63e41d37a12d3d32ee10`. Q2 qualification SHA `4dee7a139c4c261780bae01d6dc71a9451405d5f1d387ebaf448d879140b73b4`.



### Ready camera supply, not yet TRAIN images



| Source target | New CPU-eligible poses | Existing selected | Maximum after acceptance | Camera cache directory under D |

|---|---:|---:|---:|---|

| seed41/SubStem38 | 7 | 5 | 12 | `original848_substem38_fullscene_prepare_20260916_v4` |

| seed43/SubStem48 | 11 | 1 | 12 | `original848_substem48_fullscene_prepare_20260916_v1` |

| seed89/SubStem46 | 10 | 2 | 12 | `original848_substem46_fullscene_prepare_20260916_v1` |



All use previously native-passed roots with new head/framing poses, actual CPU full-scene body checks, unchanged detail/query-margin and donor-ray checks, and selected/held camera exclusion. New capture still requires actual native masks, full query association, workspace, individual visual review and consumer duplicate/cap checks.



- Seven-pose short production plan: `D/original848_short38_production_prepare_20260916_v1/plan.json`, SHA `94e24a79dec3027c960edf50ba8e253ed42a49f8a0c5d9cd5a41338074cc75bf`; preparation result SHA `a7f5c7df68040a19c22f479cfa8e25c1edaa5988abe1e605130a79f738548622`. Full CPU check 84.78s. Independent supply review SHA `6331b93c617ecc17e8af347fc468f02a51faf080fa8923357aa3d6ffb19f3717`.

- Eleven-pose48 cache SHA `93cece49971caa5d39a342015ff611b7820353cd117a9b7e57cc905d50f31fc7`; ten-pose46 cache SHA `86f0cfbafedc8f3bca82168ff7c981884cb6c0353efed7ae90ff0902e85e60b8`. Their short production plan preparation is in progress; fixed56 plans are explicit fallbacks, not short-profile approval.

- The initial v17 visual receipt omitted a required textual reviewer field. A corrected immutable review/request was created and accepted by the unchanged consumer; the original receipts were preserved. This was a record-format error, not an image rejection.

- Controlled geometry native v2 remains deferred. Do not resume it just because its earlier review exists. Optional image metric reruns remain paused.

