# Development Log — Greenhouse Deleafing Benchmark

Goal: an Isaac Sim benchmark for evaluating VLAs on tomato **deleafing** (removing
orphan/lower leaves from high-wire vines), targeting demo collection → π0.5
finetuning → deployment on the Rainbow Robotics **RB-Y1** (sim and real).

Environment: Isaac Sim **5.1.0-rc.19** (Kit 107.3.3, omni.physx 107.3.26, USD 0.24.5)
at `D:\isaac-sim`, Windows 11. Branch `koh-dev/simulator-base` (fork of openpi).

## Status

- [x] Recon: assets, Isaac 5.1 APIs, RB-Y1 SDK, repo conventions, prior art
- [x] Verified: Isaac Sim launches headless; greenhouse stage opens (7667 prims,
      3433 meshes); `breakForce`/`breakTorque`/`excludeFromArticulation` author OK
- [x] Verified: fused vine GLBs decompose into per-organ connected components
- [x] Asset pipeline: GLB → organ graph → structured USD (all 20 vines)
- [x] Greenhouse composition + launch (renders headless; see below)
- [x] Compliant vine physics — stable per-organ articulations (biological calibration pending)
- [x] Cut severance — verified: severed organ detaches, plant stays connected
- [x] Trellis support + stable interaction-contact physics (evidence below)
- [x] Physics-enabled greenhouse integration + automated pull/cut acceptance
- [x] Visible-mesh mouse pulling + explicit foliage-area airflow
- [x] Manual viewport acceptance: real Shift-drag grab/release and UI/keyboard cut
- [ ] Sustained pull/tear validation against the 32.5 N threshold
- [x] RB-Y1 Model A v1.0 import, greenhouse placement, and stable ready pose
- [x] Supplied D405 head/wrist brackets and right deleafing knife fitted
- [ ] Robot-to-vine gripper/blade contact and cut validation
- [ ] Benchmark task definition + metrics

## Findings

### Vine assets — `greenhouse/tomato_glb_20/` (2026-08-06)

20 vines, `tomato_XXX.glb` + `tomato_XXX.json` sidecar. Metadata gives units (m),
height, organ counts, and an **attachment graph**: `SubStem_XX` / `Truss_XX` /
`Fruit_XX` / `Flower_XX` with parent + 3D attach point.

The GLB itself is **one node, one mesh, no hierarchy** — organs are fused and split
only by material (`TomatoStem`, leaf `Material.0xx`, `FruitRipe_r_c`). No skins,
animations, or morph targets. ~528–557k verts / 755–791k tris per vine, of which
the stem material is ~83% of triangles.

Critically, **connected-component analysis recovers the organs** (tomato_000):

| Material class | Components | Interpretation |
|---|---|---|
| `TomatoStem` | 121 | 1 main stem (118,560 tris, full height) + 120 laterals |
| leaf materials | 113 | 109 leaf blades (~895 tris each) + 4 flowers (~5.4k) |
| `FruitRipe_*` | 8 | 5 fruits (body + calyx parts) |

Component counts match the metadata exactly (`leaves: 109`, `flowers: 4`,
`fruits: 5`). A radial profile of the main-stem component shows r99 ≈ r50 above
0.5 m, i.e. it is a clean tube with **no laterals fused into it** — so laterals are
genuinely separable. A nearest-neighbour proximity graph over components yields
~25 direct children of the main stem (vs 18 sub-stems + 3 trusses expected) and
chains up to depth 8 (petiole → petiolule → leaflet).

Coordinates: glTF is **Y-up**, metadata is Blender **Z-up**; the mapping
`(x, y, z)_meta → (x, z, −y)_glTF` was verified by landing `Fruit_00`'s attach
point on that fruit's bounding-box centre.

**Anatomy note.** 18 sub-stems and 109 leaves means `SubStem_XX` is the **petiole of
a compound leaf** and the 109 "leaves" are its leaflets. The 18 sub-stems are
therefore the deleafing targets, and each sub-stem's attach point on the main stem
is the agronomically correct (flush) cut site.

### Greenhouse stage — `greenhouse/green_house.usd`

Z-up, meters, `defaultPrim=/World`. Two cultivation zones (`Main_Cultivation_Zone`,
`..._01` offset +24.9 m in Y), each with 4 bed rows = **256 BedSet groups**. Each
BedSet payloads `objects/Bed.usd` + 4 pipes and owns a local `Strings/Cylinder_0X`
group of 6 trellis strings — the natural anchor points for vines.

Physics present: only 3 ground `Plane` prims with `PhysicsCollisionAPI`. **No
PhysicsScene, no rigid bodies, no colliders** on beds/pipes/walls. No cameras. No
tomato stems referenced into the stage yet.

Object USDs are Y-up/cm and are brought in via payload arcs with auto-inserted
`unitsResolve` xform ops (rotateX 90, scale 0.01).

### Isaac Sim 5.1 physics capabilities

- **No topological cutting of deformables exists**, and NVIDIA's own guidance is to
  fake it by pre-splitting geometry and disabling attachments (IsaacSim issue #258
  closed without a feature). Attachments have **no break force** and can be added
  but never deleted at runtime.
- **Newton is not in 5.1 at all** — it arrives with Isaac Sim 6.0. No `newton`
  package or extension is installed.
- FEM deformables are disqualified as the benchmark substrate: no static friction
  (grippers slip), OOM around 256 trivial bodies, no break threshold, and
  undocumented GPU determinism.
- `physics:breakForce` is **silently ignored on articulation joints** — breakable
  joints must be maximal-coordinate.

## Decisions

### D1 — Vine physics: compliant articulated capsule chains, not FEM

Model each organ as a chain of rigid capsule links joined by compliant D6 joints
with angular drive stiffness from Euler–Bernoulli beam theory (`Kp = E·I/ℓ`,
`I = πr⁴/4`, `Kd = 0.1·Kp`), following OrchardBench. This delivers the bending,
droop, and pull compliance the task needs while staying deterministic and fast.

Rationale: the user's requirement is *deformable behaviour* (bend, pull, cut), not
the FEM solver specifically. FEM in Isaac 5.1 cannot cut, cannot hold a grasp
(no static friction), and caps scale at tens of environments — all fatal for a
data-collection benchmark. Compliant chains give the same observable physics.
FEM is kept only as an optional visual/validation comparison.

### D2 — Cut primitive: in-place joint release (superseded architecture)

Sever by releasing the petiole's base joint (zeroing drive gains and freeing the
constraint) when the tool's blade plane intersects the link and the jaws close;
`breakForce` on the same joint provides tensile **tear** when the policy pulls
instead of cuts, which distinguishes clean cut from tear as a metric. The whole
plant stays out of any `PhysicsArticulationRootAPI` subtree so `breakForce` is
honoured. Never delete prims at runtime — deactivate.

Stub length stays continuous (not quantised by link resolution) by measuring the
blade plane against the parent stem surface geometrically and shrinking the
parent-side capsule to the cut plane.

### D3 — Asset pipeline: hybrid proximity graph + generator metadata

Segment each GLB into organ components, build a parent graph by closest-approach,
then label the primary laterals with the generator's `SubStem`/`Truss` attach
points. Metadata is authoritative for semantics and cut sites; the graph is
authoritative for which geometry detaches together. Validation invariant: every
leaf blade maps to exactly one sub-stem, and per-vine totals match the sidecar
counts.

### D4 — Code layout: `examples/greenhouse_sim/`, per repo convention

openpi examples are self-contained clients with their own interpreter that talk to
`scripts/serve_policy.py` over websocket (libero runs Python 3.8 against a 3.11
main env). The sim therefore runs inside **Isaac Sim's bundled Python** with
`openpi-client` pip-installed into it; Isaac deps never enter the root
`pyproject.toml`. RB-Y1 SDK is vendored as a `third_party/` submodule per
`.gitmodules` convention. Generated USD goes to gitignored `data/`.

Offline asset tooling depends only on packages already present in Isaac's bundled
Python (numpy 1.26, scipy 1.15, pytest 9, plus `pxr` for USD authoring), so the
pipeline is reproducible from a bare Isaac install with nothing pip-installed.

### D5 — Headline metric: residual petiole stub length

No existing plant-manipulation benchmark measures it, and it is agronomically
decisive: flush pruning wounds are near-absolutely resistant to *Botrytis cinerea*
while petiole stubs are highly susceptible (Beyers et al. 2014), with lesions
advancing 0.3–0.5 cm/day. Bins: ≤5 mm flush / 5–20 mm marginal / >20 mm risk.

## Validation

### Organ segmentation, all 20 vines (2026-08-06)

Sub-stem, truss, and foliage counts match the metadata sidecar **exactly on every
vine**; no foliage organ fails to trace to a primary lateral; the adjacency search
reaches every stem organ, so the graft-onto-root fallback never fires. Junction
gaps are 0.3 mm median / ~5 mm p95 / 9.9 mm max against a 10 mm adjacency radius.
Runtime ≈ 1.8 s per vine.

Sub-stem labels are **monotonic in height on all 20 vines**, matching the
metadata's own ordering — which is what the bottom-up "remove the lowest leaves"
rule depends on.

The assigned junctions sit a systematic **+28 mm above** and ~7 mm lateral of the
metadata attach points. This is expected rather than an error: the generator
records attach points on the main-stem **centreline**, while segmentation computes
the **surface contact** where the two organs actually meet. The horizontal offset
matches the main-stem radius. Surface contact is the correct reference for
residual stub length (D5), so the computed junction is the one to keep.

Risk: the worst junction gap (9.9 mm) is close to the 10 mm adjacency radius. A
future asset generation with looser junctions could silently fall back to
grafting. `convert_vines_to_usd.py` now asserts every one of these invariants
and exits non-zero, so this cannot regress unnoticed.

### Scene composition (2026-08-06)

Greenhouse layout as measured, not assumed: beds are 5 m troughs whose planting
surface is at **Z = 0.888 m**, spaced 5.7 m apart, with two 5 m trellis strings
at each bed's ends (not per-plant strings). 256 BedSets across two zones.

Vines compose in at 0.25 m in-row spacing with seeded yaw jitter, and organ
prims stay addressable through the reference
(`/World/Vines/Vine_0000/MainStem/SubStem_06`), with materials bound. A headless
render confirms textured foliage, fruit trusses, and plants seated in the gutter.

**Asset defect found and worked around**: the greenhouse `DomeLight` references
`/home/jhlee/Desktop/...`, an absolute path from the machine the asset was
authored on. It fails to load on any other machine and leaves lighting to a
renderer fallback — unacceptable variation for a vision benchmark. The scene
layer now clears unresolvable light textures, leaving a uniform sky at the
authored colour and intensity. Worth fixing in the source asset too.

Open question for later: the source vines lean substantially (canopy extends
~0.75 m to one side). Real high-wire tomato is trained near-vertical up a string
and only leaned along the wire. Whether to correct this at placement time is a
fidelity decision that should be made deliberately, not by accident.

### Cut mechanism (2026-08-06)

`cut_demo.py` rigs `tomato_000` as **396 capsule links / 396 joints / 120
severable organs**, settles it under gravity, and cuts the lowest sub-stem.
Result: the severed organ detaches and falls while the rest of the plant stays
connected. The severance primitive is proven.

Engine behaviour worth remembering:

- Authoring `physics:jointEnabled` **at build time** and only setting its value
  at runtime avoids forcing a PhysX resync of the whole plant on every cut.
- Kit swallows stdout and `--/app/fastShutdown=True` masks non-zero exits, so
  simulator scripts must write a machine-readable report file. Several apparent
  "crashes" during development were simply the final result never being written.

### Trellis support, resolved (2026-08-06)

Clips implemented as compliant world anchors every 0.30 m, plus a ground plane.
Stem sag over the clipped span fell from **435 mm to under 24 mm**. Four things
had to be right, and each was found by measurement rather than guessed:

1. **Springs alone are not enough.** A clip needs a *travel limit*, because any
   sustained load walks a pure spring out indefinitely however stiff it is. A
   clip is a collar (±5 mm of play, then the wall), not a spring.
2. **Clip to the growing point.** Advancing a running counter by `spacing`
   silently drops the topmost clip whenever the remaining run is shorter than
   one interval — which is exactly the stretch whose deflection matters, since
   cantilever droop grows with the *cube* of free length. Leaving 0.3 m
   unclipped cost 0.45 m of fold-over. Now targets explicit heights up to
   `top − 0.15 m`.
3. **The fitted tip radius is a trap.** The render mesh tapers the stem to a
   point (0.5 mm fitted radius), and stiffness goes as r⁴, so the top of the
   stem came out ~10⁴× too floppy. Floored at 3.2 mm, the measured tomato apex
   radius (Gao et al. 2024). Collision geometry still uses the fitted value.
4. **Anchoring itself was never the problem.** An isolated test confirmed all
   four joint types behave: locked-D6-to-world and FixedJoint both hold exactly,
   a ±5 mm limited joint sags exactly 5 mm, a free body falls. Worth knowing
   before blaming the engine.

Residual: the top ~0.15 m of unclipped growing head still nods ~0.18 m. It sits
outside the deleafing work zone and is the floppiest part of a real plant, so it
is acceptable for now but should be revisited.

**Open bug — spurious tearing at start-up.** Authoring `breakForce` at the
measured 32.5 N detachment force makes *every* petiole snap off within the first
frames: solver transients during settling far exceed 32.5 N. Visualising the
colliders is what exposed this; every scalar metric had looked plausible while
the plant was quietly shedding its leaves. Tearing is therefore disabled by
default until it can be armed after settling, which needs care because PhysX may
not pick up a `breakForce` change at runtime.

**Lesson worth keeping:** render the physics, do not trust aggregate numbers.
Both the fold-over and the tearing were invisible in the metrics.

### Runaway organs: diagnosis (superseded by the resolution below, 2026-08-06)

Organs fly apart on start-up. Ruled out by measurement, not argument:

| Hypothesis | Test | Verdict |
|---|---|---|
| `breakForce` firing on transients | set tear force to 0 | not the cause |
| Stiff chain unintegrable at 240 Hz | — | **wrong**; see below |
| Gravity / collapse under load | run at `--gravity 0` | **not the cause** — identical |
| Self-collision | run with `--no-collision` | **the driver**: 73 → 21 runaways |

Gravity-independence is the key result: with zero gravity nothing should move at
all, so this was never collapse, load, or the stiff-chain integration limit
blamed earlier. That earlier diagnosis was wrong and is retracted.

**Fixed along the way (correct regardless):**

- *Joint frames.* Only local *positions* were authored, never local rotations.
  The angular drives therefore targeted zero relative rotation between two
  identity frames, asking every link to lie parallel to its parent and snapping
  the plant straight on frame one. Joint frames are now anchored to the child's
  rest orientation, so "zero" means the authored pose.
- *Collider size.* A capsule's true length is `height + 2·radius`, so a short,
  thick segment collided as a ball up to **4.2×** longer than its link, engulfing
  neighbours. Clamped to stay within its own segment.
- *Offline auditability.* `PhysxSchema` is imported lazily, so `vine_physics` can
  be inspected without booting Kit.

**Still open.** `UsdPhysics.CollisionGroup` self-filtering has no measurable
effect here — neither adding several hundred collider targets individually nor
including the physics scope and letting the collection expand. Since collision
is demonstrably the driver, the next step is the mechanism Isaac supports
reliably for this: `PhysxArticulationAPI` with `enabledSelfCollisions = False`.
That needs the articulation to build, which needs the chain decimated well below
~400 links — worth doing anyway for throughput, since demo collection wants many
plants per scene.

An offline rig audit (`scratchpad/rig_audit2.py` pattern) confirmed the rig
itself is sound: exactly one organ attaches to the world (the main stem), and
every joint anchor lies within 30 mm of its parent link.

### Honest verification, and a correction (2026-08-06)

An earlier claim that the vine was "stable" was wrong, and is withdrawn. It
rested on a tight-framed screenshot after **one second**, while the run's own
metric reported 72 of 121 organs past 100 mm — which was explained away rather
than investigated. A favourable picture was trusted over an unfavourable number,
which is the same failure as trusting numbers over pictures, in the other
direction.

Proper check: 10 s run, wide framing that would show anything ejected.

| t (s) | max | p95 | median | organs > 200 mm |
|---|---|---|---|---|
| 1 | 458 mm | 359 | 96 | 82 |
| 2 | 507 | 363 | 98 | 105 |
| 5 | 501 | 363 | 96 | 98 |
| 10 | 499 | 360 | 96 | 101 |

What this actually shows:

- **Not divergence.** Displacement plateaus after ~2 s and holds for the rest of
  the run, and the plant's extent *shrinks* (0.39 → 0.24 m wide, 1.89 → 1.65 m
  tall). An explosion grows; this contracts.
- **Nothing detaches.** The wide 10 s frame shows an intact plant with no debris,
  which the 64 position iterations fixed.
- **The plant goes limp.** It slumps up to 0.5 m at the extremities on start-up.
  That first-second lurch is what looks like an explosion in the viewport.

**Root cause, still unfixed: the angular drives are not applied at all.** Sweeping
`stiffness_scale` over 1e2, 1e4, 1e6 and 1e8 gives *byte-identical* results
(max 500 mm, p95 370 mm, extent 0.24/0.67/1.65). Six orders of magnitude with no
response is not "too soft", it is ignored, so the plant hangs as a free-jointed
chain instead of holding its authored shape. The calibrated `stiffness_scale`
constant is therefore meaningless and should be removed once the real cause is
found.

Caveat on method: the isolated drive harness is **not deterministic** between
runs (the same case gave −20.7 mm and +44.6 mm on two runs, and a free-translation
control flew −5.9 m then +6.9 m). Small A/B results from it cannot be trusted;
only the plant-scale sweep above is reliable, because it is identical across
runs.

Next step: drives on **articulation** joints are the well-trodden path in Isaac
(every robot uses them), whereas driven non-articulation D6 joints are not. That
points back at articulations, which need the chain decimated from ~400 links —
0.05–0.10 m segments instead of 0.02 m would bring it near 100.

### Visual meshes now ride the physics

`vine_visuals.py` parents each organ's GLB mesh to the rigid body carrying it, so
the rendered plant is the original art and the capsules stay hidden. Foliage and
fruit have no bodies of their own, so they ride the nearest ancestor that does —
which is what makes a severed leaf travel with its petiole.

Two USD traps found here:

- **Purpose is inherited by the whole subtree.** Parenting art under a
  `guide`-purpose capsule hides the art too, and authoring `default` on the child
  does *not* override it. Bodies are therefore plain Xforms with the collider and
  the art as siblings — which is the conventional structure anyway.
- `Gf.Vec3f(*p)` rejects numpy scalars; array attributes must go through
  `Vt.*Array.FromNumpy`, which is also far faster at these mesh sizes.

### Blocking problem — stiff chains are unstable in maximal coordinates (resolved below)

With physically correct stiffness the chain explodes. Beam theory at the floored
3.2 mm radius and 250 MPa gives ~1030 N·m/rad per joint, while a link masses
~1e-4 kg, so the natural frequency is enormous and 240 Hz cannot integrate it.
The stability ceiling is roughly `K < 0.25·I/dt²` ≈ 6e-4 N·m/rad — about **six
orders of magnitude** below the physical value. A real tomato stem is stiff and
light, and that combination is exactly what maximal-coordinate joints handle
worst.

Lowering stiffness to fit is not an option: it reintroduces the collapse.

**The fix is to move the plant into a PhysX articulation** (reduced coordinates),
which handles stiff serial chains stably. That was ruled out earlier because
`breakForce` is ignored on articulation joints — but that only matters if
tearing depends on `breakForce`. It does not have to: severance already works by
disabling a joint, and tearing can be implemented by monitoring joint force in
Python and disabling past the 32.5 N threshold. That yields both stability and a
tear model, and it removes the spurious-tearing bug at the same time.

This supersedes the "no articulation root" decision in D2.

### Runaway vine physics and contact: resolved (2026-08-06)

The default simulator no longer explodes. The original tomato GLB visuals and
all **396 structural links / 396 joints** are retained; the fix changes how those
links enter PhysX, not which vine is shown or which organs can bend and sever.

The failure was three coupled implementation problems:

1. **Articulation topology/API mismatch.** A single ~400-link plant articulation
   exceeds the practical Isaac 5.1 limit, while maximal-coordinate chains cannot
   carry the calibrated stiffness. The stable topology is one reduced-coordinate
   articulation per organ, rooted on an Xform, with cross-organ base joints
   explicitly excluded from both articulations. Isaac 5.1 also has no
   PhysxArticulationAPI fixBase attribute; the invalid call was removed and the
   main-stem world joint provides the fixed base.
2. **Dense rest-pose collision was physically invalid.** The artistic vine pose
   contains intentional intersections at petiole/stem junctions and among
   foliage. Giving every structural capsule CollisionAPI created hundreds of
   endpoint-touching/interpenetrating shapes and made PhysX resolve the authored
   plant itself as penetration. CollisionGroup and pairwise FilteredPairs did not
   make that mixed articulation graph reliable and were removed.
3. **Inertia depended on whether a link happened to have a collider.** PhysX
   used fallback inertia for non-contact links and shape-derived inertia for
   contact links, so adding robot contact changed the plant dynamics and could
   invalidate every transform. Each link now authors mass and diagonal inertia
   independently of contact. The minimum diagonal inertia is 1e-5 kg*m^2: the
   stable petiole-probe scale and a conservative lump for unresolved leaf/flower
   art riding the structural links. This is a numerical/effective inertia and
   must remain a domain-randomisation/calibration parameter, not be cited as a
   measured tomato material property.

The public collision modes now make the separation explicit:

| Mode | Purpose | tomato_000 result |
|---|---|---|
| interaction (default) | Ten non-overlapping main-stem zones at 0.20 m spacing plus one midpoint zone on each of the 18 real SubStem deleafing petioles | 28 contact colliders, stable |
| none | Constraint/inertia isolation | 0 contact colliders, same structural motion |
| all | Negative-control diagnostic for the raw dense capsule set | reproduces invalid transforms; do not use for episodes |

This is task-directed collision geometry, not deletion of vine physics. Every
organ still has mass, inertia, compliant joints, gravity response, visuals, and
a severable attachment. Contact is authored only where the RB-Y1 should touch
the plant for deleafing; internal foliage self-collision is intentionally absent
because the source rest pose is not collision-clean.

**Validation evidence (Isaac Sim 5.1, 240 Hz):**

| Check | Result | Report |
|---|---|---|
| Contact off vs default contact, 120 steps | identical 7.2 mm maximum stem sag; 0/121 runaway organs in both; no invalid-transform or broadphase log entries | data/greenhouse_sim/interactive_inertia_floor_none.json and interactive_inertia_floor_contact.json |
| Default contact, 10 s / 2400 steps | 7.2 mm maximum stem sag; 0/121 runaway organs; no PhysX errors | data/greenhouse_sim/interactive_final_10s.json |
| Flush cut with ground contact | SubStem_00 dropped 233.22 mm; rest of plant moved 2.44 mm; 0 mm stub/quantisation; succeeded=true; no PhysX errors | data/greenhouse_sim/cut_contact_final.json |

The cut run proves the stable mode is dynamic rather than frozen: the severed
petiole falls and contacts the ground while the clipped parent vine remains
compliant and connected.

Remaining calibration work is deliberately not hidden by the stability result:
validate gripper forces and friction on the 28 interaction zones with the RB-Y1
finger geometry; calibrate the effective inertia distribution against tracked
real-vine deflection; and validate sustained-force tearing at 32.5 N. Full
leaf-blade collision proxies may be added only after producing a collision-clean
proxy asset, never by re-enabling the raw dense structural capsules.
### Physics-enabled greenhouse and mouse interaction (2026-08-06)

`interactive_greenhouse.py` is now the acceptance launcher between isolated-vine
physics and robot integration. It opens the generated greenhouse, switches to
the USD session layer, deactivates the selected static vine references there,
and rebuilds the corresponding source GLBs as articulated rigs at the exact
same bed transforms. The generated scene and source assets remain unchanged.
One of eight placements is dynamic by default; `--physics-vines` can increase
that only when the GPU/performance cost is intentional.

Integration exposed and fixed three non-solver errors:

1. Greenhouse translation was correctly applied to geometry points but was also
   being added to mesh normals. `vine_visuals.attach_organ_visuals` now accepts a
   separate direction transform, and both launchers pass translation-free
   `TransformDir`/GLB direction conversion for normals.
2. The severed-organ catch plane was still centred at world XY = (0, 0).
   `add_ground_plane` now accepts `centre_xy`, and each plane is centred beneath
   its greenhouse vine placement.
3. The first automated pull used `apply_force_at_pos(..., "Acceleration")`, a
   mode PhysX 5.1 rejects for off-centre force application. The probe now reads
   the selected rigid body's authored mass and applies `mass × requested
   acceleration` in supported `Force` mode. This also makes the requested pull
   reproducible across link masses.

The first manual acceptance exposed that native PhysX mouse grabbing was not a
usable interaction surface. Five UI cuts were recorded correctly, but no
`POINT_GRABBED` event appeared. Each SubStem exposed only one hidden structural
capsule: typically about 20 mm long and 5 mm across, just a few pixels at the
inspection-camera distance. Leaf blades had no colliders at all. Shift-clicking
the missed/hidden target also drove Isaac 5.1's transform selector into repeated
`KeyError: <class 'NoneType'>` callbacks. Raising the native coefficient alone
would not solve the missing pick surface (though Isaac's own rope demo uses 10.0
rather than the original 1.0).

The corrected GUI backend disables that native collider-only grab. A viewport
Shift-drag now raycasts the **actual rendered GLB triangles**, walks from the hit
Visual prim to its supporting rigid body, holds the hit at its original camera
depth, and applies a damped spring force at that material point. Defaults are
10 N/m stiffness, 0.02 N*s/m damping, and a hard 1 N force cap. This permits
pulling broad leaf blades, petioles, and the visible stem without inflating the
28 robot-contact proxies or reintroducing rest-pose contact. A four-frame camera
warm-up removes a startup race where the overlay could read the previous
viewport projection. Grab/release count, selected Visual/body, peak target
offset, and peak force are persisted in the report.

The answer to "should a stationary vine move?" is conditional: once damping has
removed transients, a vine in still air should settle, not jitter forever. To
model greenhouse motion honestly, `vine_interaction.Airflow` adds an explicit
aerodynamic load. For every compound petiole with foliage, it sums the actual
one-sided triangle area, finds the foliage centroid, and applies
`F = 0.5*rho*Cd*A*v^2` there with deterministic low-frequency speed and direction
variation. The current GUI default is 1.0 m/s at 0.18 Hz over 14 populated
petioles (0.1593 m2 total measured foliage area); `--airflow-speed 0` restores
still air. The speed is an initial interactive default and remains a calibration
and domain-randomisation parameter, not a measured condition of this greenhouse.

`[` / `]` still select deleafing petioles, `V` selects a dynamic vine, and `C` or
the interaction window releases the selected base joint.

**Revised integrated acceptance (Isaac Sim 5.1, 240 Hz):**

| Check | Result | Evidence |
|---|---|---|
| Greenhouse composition | 8 source placements found; Vine_0000 replaced in-session by 396 links, 6 clips, 120 severable joints, and 28 robot-contact colliders | `data/greenhouse_sim/interactive_greenhouse_airflow_pull_cut_10s.json` |
| Visible-mesh raycast pull | renderer selected a foliage Visual and mapped it to its supporting body; 0.7551 N peak force produced 29.95 mm motion and 1.80 mm residual under airflow; grab/release counters both 1; finite=true | `data/greenhouse_sim/interactive_greenhouse_visual_pull_probe.json` |
| Airflow, 10 s / 2400 steps | 14 foliage targets; 2.343 mm peak-to-peak axis motion; 2.515 mm maximum displacement; finite=true; 0 runaway organs | `data/greenhouse_sim/interactive_greenhouse_airflow_pull_cut_10s.json` |
| Bounded body-force pull after airflow | 0.1674 N produced 3.723 mm peak deflection; after recovery residual was 0.632 mm; finite=true | same report |
| Flush cut after airflow and recovery | exact BaseJoint changed enabled=true -> false; detached organ dropped 238.27 mm and travelled 247.82 mm; finite=true | same report |
| Error scan | no Python tracebacks, PhysX errors, invalid transforms, broadphase faults, NaNs, or explosion signatures | both reports' `.log` files |
| Rendered composition | greenhouse beds/trellis and the updated vine render together from the inspection camera | `data/greenhouse_sim/interactive_greenhouse_acceptance.png` |
| Manual viewport acceptance | user accepted visible Shift-drag pull/release, stationary airflow motion, and cutting in the relaunched greenhouse session | direct acceptance, 2026-08-06 |

The automated GUI probe reaches the same renderer raycast, Visual-to-body map,
and bounded-force path as Shift-drag; the headless probe independently checks
body compliance and the cut joint. Direct viewport acceptance was completed on
2026-08-06 after a clean relaunch: visible foliage could be Shift-dragged and
released, airflow produced acceptable stationary motion, and the cut mechanism
worked. This clears the interaction gate for RB-Y1 integration. Benchmarking
and RL remain gated on stable robot/tool/camera integration and contact testing.
Regression status: 34 passed, 1 failed, and 1 skipped in the complete hermetic
greenhouse suite. The only failure is the unchanged
`test_arc_length_sampling_is_continuous`, whose 10 mm
absolute tolerance is exceeded by 0.17 mm; no skeleton code changed in this
milestone, so that numerical test issue is tracked separately rather than hidden
inside the robot integration. The skipped test requires `PhysxSchema` from a
running `SimulationApp`; the integrated live soak exercises that path.

### RB-Y1 Model A v1.0, D405, and knife integration (2026-08-06)

The fitted robot is rebuilt reproducibly by `build_robot.py` from the exact
physical-robot URDF,
`third_party/rby1-sdk/models/rby1a/urdf/model_v1.0.urdf` (SHA-256
`33cb8cd34abc0f58f0e65f8dc7b59acabf3fd62cb820b1be1f40d513578a65ae`).
The v1.2 simulator asset is intentionally not substituted. Import keeps the
mobile base dynamic, preserves fixed joints and URDF inertia, disables
self-collision, and changes only the two wheel drives to velocity mode.

Collision reconstruction corrected an earlier source audit: v1.0 contains
**17 active custom capsule elements**, not 26. The other nine matches are inside
XML comments and must not become live colliders. Isaac's URDF importer drops
the active non-standard capsules, so the builder restores those 17 as sibling
prims under each link. The source also omits standard collisions for the base,
wheels, gripper bodies, and fingers; conservative proxies add 3 base/wheel and
6 end-effector/finger shapes. Eight more shapes cover the three fitted
camera/bracket assemblies and two knife components, for **34 collision shapes**
on the generated robot.

`extract_robot_hardware.py` makes the supplied CAD reproducible without a
runtime FreeCAD dependency. The external
`D:\research\freecad-mcp\deleaf_knife.stl` was copied byte-for-byte to
`greenhouse/robot_assets/deleaf_knife.stl` (SHA-256
`0237eb46c8980cec4cd9b09623f72d55e6f2491928e673a67172294c3a5f8dbd`)
and split into its two disconnected components:

- the 30.362 x 71.480 x 13.000 mm **flat straight plate**, the only prims
  carrying `tomato:cuttingSurface=true`;
- the 6.000 x 62.301 x 50.918 mm **U-shaped arc**, explicitly non-cutting
  support geometry.

The plate projects along the right tool's -Z axis from the gripper distal face.
The U arc is never accepted as a cutting surface. Both pieces retain collision
geometry, so support contact remains physical without corrupting cut semantics.

The supplied bent D405 bracket STEP was decomposed into the exact 27 x 59.920 x
34.619 mm bracket and 42.090 x 23 x 42 mm camera body while preserving the
authored bracket-to-camera transform. One assembly is mounted on the outer face
of each end effector. The supplied head bracket carries the same D405 body on
`link_head_2`. All three USD cameras use the D405's 84 by 58 degree depth FOV
and 40 mm near clip. Head optical forward/up are +robot-X/+robot-Z; wrist
cameras look down the tool and outward. The mirrored right physical mount gets
a sensor-only 180 degree roll so left and right policy images share an upright
convention.

A real transform bug was caught during rendered-camera validation: the CAD and
NumPy matrices use column vectors, but `Gf.Matrix4d` transforms row vectors.
Transposing at the Gf boundary corrected the initially inverted head view,
horizontal wrist views, and wrong blade direction. The authored-stage
regression now checks the actual Gf camera forward/up axes and actual blade
projection, not only the source NumPy matrices.

`interactive_greenhouse.py` now composes the robot by default at
`(6.6191, 2.4200, -0.3050817)` m with 90 degree yaw, where the Z value is the
measured cultivation-zone collision floor. The 250 mm negative-X offset
accounts for the ready right tool's lateral offset, aligning that tool with
dynamic `Vine_0000` at X=6.86912 m. The closer Y stance leaves approximately
213 mm between the chassis proxy and the trough front. `--no-robot` retains the accepted
vine-only launcher. All 22 torso, arm, and head joint targets and initial
PhysX joint states are authored to the official SDK ready pose before physics
starts; this prevents a zero-pose startup sweep through the greenhouse.

**Automated fit and stability acceptance (Isaac Sim 5.1):**

| Check | Result | Evidence |
|---|---|---|
| Hardware semantics | 3 D405 cameras; 1 visual/collision blade pair marked cutting; U arc visual/collision pair non-cutting | `robot_hardware_test.py`, `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| Authored transforms | head forward/up correct; wrists down/outward with common upright convention; flat plate projects along tool -Z; U arc faces tool +X/upward | 8 passed, 1 PhysX-only test skipped outside SimulationApp |
| Visual fit | head bracket, left wrist bracket/D405, and upward-arc right bracket/D405/knife all render seated on the intended links | `robot_fit_head_camera_final.png`, `robot_fit_left_wrist_final.png`, `robot_fit_right_tool_up.png` |
| Integrated closer-stance 480-step soak | 34/34 rigid bodies finite; base settled 3.675 mm laterally and 8.203 mm vertically; 2.078 degree tilt; succeeded=true | `data/greenhouse_sim/robot_front_acceptance.json` |
| Vine during robot soak | 121/121 tracked organs finite; 0 runaway organs | same report |
| Live camera path | right wrist D405 rendered from its fitted USD optical frame after the soak | `data/greenhouse_sim/d405_right_wrist_final.png` |

This clears the **asset import, hardware fit, optical-frame, and non-contact
stability** gate. It does not yet claim robot deleafing success: arm
reachability, gripper contact/friction, blade-to-petiole contact, robot-driven
cut triggering, and sustained-force tearing remain the next explicit gate.
Default ready-pose wrist images mainly see the floor/nearby structure because
the arms are stowed; task camera coverage must be evaluated in reachable
pre-contact poses rather than misreported as a mounting failure.

A follow-up visual review requested that the U arc face upward so the flat plate
is presented cleanly for cutting. Rolling the knife 90 degrees about its
unchanged blade axis moved CAD +Z from tool +Y to tool +X. The rendered ready
pose measures the arc-facing vector at Z=+0.77984 while preserving the blade
extension. The same review requested direct camera access: the interaction
window and keys `1`-`4` now switch the active viewport between inspection,
head D405, left-wrist D405, and right-wrist D405 video respectively. Headless
captures remain available through `--capture-camera`.

## Research findings, 2026-08-06 (pre-implementation)

### Stiffness — the current E is 5–15× too low

`TissueProperties.youngs_modulus_pa = 2.0e7` (20 MPa) is not defensible. Three
independent lines converge on **100–300 MPa** for fresh turgid vine tissue:
fresh greenhouse cucumber cane in tension, 280/199/137 MPa base→apex
(Xu et al. 2016, the nearest high-wire analogue); petioles measured at valid
span in four species, 110–192 MPa (Langer et al. 2021); and a self-weight
cantilever check on a tomato leaf back-solving to ~148 MPa.

Adopt **150 MPa** for petioles and young upper stem, **250 MPa** for the
lignified lower stem. The reviewer-proof sentence: at 5 MPa a tomato leaf would
deflect 2.9 m on a 0.12 m petiole, so any modulus below ~50 MPa is falsified by
the observable fact that tomato leaves hold themselves up.

The only direct tomato measurement (pedicel, 2.8–7.1 MPa, Weng et al. 2024) must
be cited but with its caveat: span/depth of 1.5–2.4 means it is shear-dominated
and is an *apparent* modulus; the Timoshenko correction reconciles it to
11–60 MPa. Its own data gives it away — modulus correlates with specimen
diameter at r = −0.893, which no real material does.

**No published Young's modulus exists for the fresh tomato main stem.** Say so
rather than imply otherwise. Other gaps: no tomato stem density (use 950 kg/m³
inferred from the measured 73–79% moisture), no tomato droop data.

Geometry defaults: stem 6.4 mm apex / 9.8 mm mid / 7.5 mm base (Gao et al. 2024,
2.1 m plants); petiole 4–8 mm (Sun et al. 2024). Detachment 32.5–40.3 N and
cutting 62–66 N, now corroborated across two independent groups.

### Trellis — clips every 0.30 m, compliant not fixed

Recommended: one fixed joint at the base elbow, then **D6 joints to world anchors
at 0.30 m spacing** (randomise 0.25–0.35 m per episode), 2–10 kN/m along the
string axis, ±5 mm lateral free play then ~250 N/m. Leave the stem joints
compliant — a real tomato stem self-buckles beyond 0.6–1.1 m, and the clips are
what carry it. Fixed joints over-stiffen laterally by 2–3 orders of magnitude;
a kinematic lower stem has infinite effective mass so the robot cannot perturb
it and contact forces become unbounded.

Consequence for metrics: with correct support the stem barely moves (1–3 cm
mid-span). **Essentially all real canopy disturbance comes from leaves and
trusses**, so the disturbance metric must be defined on leaf/truss motion, not
stem displacement, or it will stop discriminating between policies.

### Severance — manual snapping is standard commercial practice

The decisive finding. Growers routinely **snap** tomato leaves off by hand, and
that produces a *stub-free, faster-healing* wound than a knife (Decognet et al.
2010). The real acceptance criterion is "flush, no stub", not "cut". So the
RB-Y1's stock parallel gripper is sufficient and no custom tool is required —
the pull track is not a shortcut, it is the human baseline.

There is also **no COTS cobot leaf cutter to buy**; requiring one would mean
every replicating lab starts with a machining job and their blade geometry
becomes an uncontrolled cross-lab confound.

### The task as specified is shortcut-solvable

As authored the grower rule yields **the same answer on 20 of 20 vines**. Fixes,
in order of cost: the `N_retain = 15` term (free, agronomically real, alone
spreads the answer distribution); randomised pre-deleaf depth (uses the existing
joint-release primitive); ripeness rebinding. Together: 611 configurations.

Fruit ripeness **is** recoverable — the `FruitRipe_r_c` material's `c` index is
the ripeness stage, validated exact across all 155 fruits in the asset set. Bake
it as a `greenhouse:ripenessStage` attribute at conversion time.

Episode parameters must live **only in the instruction string**, with a
generation-time assertion that no observation→target mapping is a function.

### Occlusion is likely the binding constraint

Predicted to fail before reachability at the current 0.25 m spacing combined with
the assets' 0.75–1.11 m one-sided lean. Real high-wire in-row spacing is
0.40–0.50 m. Both spacing and lean are one-line changes in
`greenhouse_scene.py` and far cheaper to fix now than after thousands of demos.

### RB-Y1 asset path

**Confirmed 2026-08-06: the physical robot is RB-Y1 Model A v1.0.** Target
`models/rby1a/urdf/model_v1.0.urdf` (Apache-2.0, actively maintained). The official `rby1-sim-isaac` USD is **v1.2 kinematics** — 28.7 mm
end-effector offset and 6 cm head offset versus v1.0 — and ships with no licence
grant. v1.0 is identifiable on real hardware by a discrete FT-sensor puck between
wrist and gripper flange (wrist→EE 154.8 mm vs 126.1 mm).

Import settings that matter: `fix_base=False`, `merge_fixed_joints=False`,
self-collision off, inertia from URDF, position drives with wheels overridden to
velocity. The importer drops the URDF's non-standard `<capsule>` tags. A
comment-aware implementation audit found 17 active capsules to restore; the
earlier count of 26 incorrectly included nine capsules inside XML comments.

## Novelty position

Nearest neighbour is **OrchardBench** (GPU-parallel apple orchard, compliant
branches, fruit detachment). Differentiators, all of which must be delivered:
severance/cut quality rather than fruit pull-off, bimanual whole-body mobile
manipulation on RB-Y1, language-conditioned VLA evaluation, and real-robot
deployment (which OrchardBench explicitly disclaims).

## Commit log

One logical change per commit; no AI attribution trailers.

- 2026-08-06 — Stabilised vine physics with per-organ articulations, explicit
  contact-independent inertia, task-directed interaction colliders, cut/ground
  validation, and a clean 10-second stability soak.
- 2026-08-06 — Added visible-mesh mouse pulling, explicit foliage airflow,
  integrated greenhouse pull/cut probes, and manual viewport acceptance.
- 2026-08-06 — Imported exact RB-Y1 Model A v1.0, restored active collisions,
  fitted three supplied D405 assemblies and the right flat-plate knife, corrected
  optical/tool transforms, and passed the integrated robot-plus-vine soak.
- 2026-08-06 — Rolled the knife arc upward without changing blade extension,
  aligned the ready right tool with Vine_0000 from a collision-clear closer
  stance, and added live inspection/head/wrist camera switching.
