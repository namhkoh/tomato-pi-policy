# Development Log — Greenhouse Deleafing Benchmark

Goal: an Isaac Sim benchmark for evaluating VLAs on tomato **deleafing** (removing
orphan/lower leaves from high-wire vines), targeting demo collection → π0.5
finetuning → deployment on the Rainbow Robotics **RB-Y1** (sim and real).

Environment: Isaac Sim **5.1.0-rc.19** (Kit 107.3.3, omni.physx 107.3.26, USD 0.24.5)
at `D:\isaac-sim`, Windows 11. Active deleafing branch `koh-dev/deleaf`
(fork of openpi).

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
- [x] Supplied D405 head/wrist brackets fitted; right tongs removed and replaced by the deleafing knife
- [x] Physical cut gate: real leading-edge contact, force/work, direction, and centre crossing
- [x] Protected-contact ledger for main stem, non-target organs, neighbouring vines, structure, and robot
- [x] Bi-manual task state/retention support: left grasp -> right cut -> transport -> floor deposit
- [x] Stable non-contact robot/vine startup after collider deduplication and wrist-envelope correction
- [x] Robot-to-vine gripper/blade contact and cut validation
- [x] Hardware-bounded dual-arm approach + complete grasp/cut/transport/deposit acceptance
- [x] Simulator-only leader-arm command bridge + synchronized D405/action recording
- [x] Deterministic target selection + strict isolated-process repeatability runner
- [ ] Lab leader-arm hardware validation + multi-target physical repeatability acceptance
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

The generated benchmark scene loads this source without rescaling or rebuilding
it: `data/greenhouse_sim/scenes/deleafing_bench.usd` has
`../../../greenhouse/green_house.usd` as its root sublayer, and both layers are
Z-up with `metersPerUnit=1.0`. A direct composed-vs-source bounds audit found the
same 256 BedSets and identical first target gutter bounds: Z=0.671233–0.888308 m.
The authored cultivation floor is Z=-0.305082 m, so the gutter top is **1.193390 m
above the floor**. The gutters therefore do look high, but that height comes from
the supplied original greenhouse rather than a loader transform or unit error.

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

### D6 — Benchmark cuts require physical leading-edge evidence and bi-manual order

The full flat knife plate remains a physical collider, but it is not itself a
cutting semantic. Only contact points inside the distal 2 mm straight-edge
region can accumulate cut work. A valid severance requires the measured petiole
cut zone, transverse edge and motion alignment, forward edge motion, at least the
petiole's 66.3 N cut force, sustained work through its diameter, and a sweep
across the target centre. Separate taps do not combine. The U support, blade
face, direct `C` debug release, and tensile tear cannot produce a benchmark cut.

The required task order is explicitly bi-manual: both left fingers establish a
loaded grasp on the selected petiole, the right leading edge physically severs
that same target without protected contact, the left grasp retains and moves the
orphan at least 0.15 m, then releases it into the floor drop zone. A simulator
cut still follows physics if the policy failed to grasp first, but that episode
is recorded as a benchmark failure rather than silently counted as success.

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
Regression status: 36 passed, 1 failed, and 1 skipped in the complete hermetic
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
XML comments and must not become live colliders. Isaac's URDF importer handles
the non-standard capsules inconsistently: ordinary stage traversal did not show
usable shapes, but live contact traces later proved its instance proxies still
participated in PhysX alongside the restored siblings. The builder therefore
deactivates every importer-created `collisions` scope and restores the 17 source
capsules exactly once under authorable sibling scopes. The source also omits standard collisions for the base,
wheels, gripper bodies, and fingers; conservative proxies add 3 base/wheel and
3 retained left-end-effector/finger shapes. The original right end-effector body
and both right fingers are a knife-only tool slot: their visual and collision
scopes are inactive, while their invisible rigid links and joints remain so the
exact URDF articulation and controller joint indexing are not changed. Eight
more shapes cover the three fitted camera/bracket assemblies and two knife
components, for **31 collision shapes** on the generated robot.

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

The plate projects along the right tool's -Z axis directly from the retained
`ee_right` flange frame; the obsolete 73 mm gripper-tip offset is no longer used.
The U arc is never accepted as a cutting surface. Both pieces retain collision
geometry, so support contact remains physical without corrupting cut semantics.

The supplied bent D405 bracket STEP was decomposed into the exact 27 x 59.920 x
34.619 mm bracket and 42.090 x 23 x 42 mm camera body while preserving the
authored bracket-to-camera transform. The first wrist fit incorrectly treated
the re-origin of that extracted STL as its mounting face, so the assemblies
were only approximately placed on the outer wrist surfaces. The authoritative
RB-Y1 v1.1 assembly `RBY1_Example_setup.FCStd` fixes the actual interface at the
18 mm-spaced M3 pair: bolt centres are `(x, y, z) = (+/-9, -39, 38.5)` mm in
each mirrored end-effector frame. In the normalized bracket STL those same
centres are `(+/-9, 56.919538, 30.119184)` mm, which gives the exact bracket-root
translation `(0, -95.919538, 8.380816)` mm with identity assembly rotation on
both wrists. Robot-chain mirroring supplies the left/right reflection; rotating
the right bracket again would misalign the screw pair.

The supplied head bracket carries the same D405 body on `link_head_2`. All three
USD cameras use the D405's 84 by 58 degree depth FOV and 40 mm near clip. Head
optical forward/up are +robot-X/+robot-Z; wrist cameras look down the tool and
outward. A sensor-only 180 degree roll remains on the right optical prim to
normalize policy-image orientation without rotating the physical CAD or its
mounting holes. Authored metadata records `rby1_wrist_m3_pair` and the 18 mm
bolt spacing on both wrist assemblies.

A real transform bug was caught during rendered-camera validation: the CAD and
NumPy matrices use column vectors, but `Gf.Matrix4d` transforms row vectors.
Transposing at the Gf boundary corrected the initially inverted head view,
horizontal wrist views, and wrong blade direction. The authored-stage
regression now checks the actual Gf camera forward/up axes and actual blade
projection, not only the source NumPy matrices.

`interactive_greenhouse.py` now composes the robot on the **opposite side of the
target gutter** at `(6.99114, 3.78000, -0.3050817)` m with -90 degree yaw, where
the Z value is the measured cultivation-zone collision floor. This places the
robot in the inter-row aisle facing world -Y toward dynamic `Vine_0000`, with
about 229 mm initial chassis clearance to the target gutter and 35 mm to the
neighbouring gutter. `--no-robot` retains the accepted vine-only launcher. The
official 22-joint SDK ready vector is retained as `SDK_READY_POSE_DEGREES`; the
greenhouse launcher changes only the right arm to the exact v1.0-URDF IK vector
`[-101.724, -83.623, 34.196, -135.683, -57.431, 94.832, -74.920]` degrees. This
alternate branch keeps the elbow in the aisle, preserves joint-limit margin,
points the flat blade toward the real `SubStem_00` attachment, and keeps the
upward arc clear at spawn.

The first nominal IK attempt exposed an independent fixture bug rather than a
robot balance problem. Each dynamic vine owns a 1.2 m hidden `CatchPlane` cube
at bed height to catch severed foliage. It is synthetic episode bookkeeping,
not greenhouse structure, but it was colliding with the robot. A 10-step PhysX
trace measured 76.21 mm initial penetration and up to 1123.16 N*s impulse
between the tray and right-arm capsules, tipping the robot within 60 steps even
at the previously stable base stance. `add_ground_plane` now accepts filtered
actor paths, and each interactive catch tray filters the actual
`/World/RBY1/base` articulation root. Targeting the reference root alone was
insufficient; PhysX filtering requires the authored articulation-root prim.
This preserves tray contact for detached vine organs while removing the
invisible robot obstacle. `--contact-diagnostics` records collider pairs,
separation, and maximum impulse for future task-pose acceptance.

**Automated fit and stability acceptance (Isaac Sim 5.1):**

| Check | Result | Evidence |
|---|---|---|
| Hardware semantics | 3 D405 cameras; both wrist brackets aligned to the v1.1 18 mm M3 pair; original right EE/finger visuals and collisions inactive; 1 blade pair marked cutting; U arc pair non-cutting; 31 collision shapes | `robot_hardware_test.py`, `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| Authored transforms and placement | exact bracket and reference bolt centres coincide on both wrists; physical mount rotations are identity in mirrored EE frames; right tool faces world -Y with its U arc upward; opposite-aisle clearances are regression checked | 10 passed, 1 PhysX-only test skipped outside SimulationApp |
| Visual fit | brackets seat against the actual wrist screw faces; the right end remains knife-only; opposite-aisle robot, greenhouse, and vine render together | `data/greenhouse_sim/wrist_mount_reference_left.png`, `data/greenhouse_sim/wrist_mount_reference_right.png` |
| Opposite-aisle knife-only 480-step soak | 34/34 rigid bodies finite; base settled 11.524 mm toward the target row and 8.203 mm vertically; 2.078 degree tilt; succeeded=true | `data/greenhouse_sim/robot_wrist_screw_mount_acceptance.json` |
| Vine during robot soak | 121/121 tracked organs finite; 68.710 mm maximum compliant settling; 0 runaway organs | same report |
| Live camera path | all three camera paths remain present after right-gripper removal; explicit right-wrist D405 capture is non-black and sees the greenhouse/vines; head/wrist viewport selection remains available | `data/greenhouse_sim/right_wrist_d405_acceptance.png`; `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| Measured lower-petiole reach | settled flat blade 118.072 mm and upward U-support 65.175 mm from the actual `Vine_0000/SubStem_00` attachment; blade extension Y=-0.999876 and arc normal Z=+0.999845; no spawn contact | same report |
| Contact trace | only normal chassis/wheel support against the cultivation-zone floor; no catch-tray, gutter, vine, arm, camera, or knife contact | same report |
| Error scan | no Isaac `[Error]`, Python traceback, selector `NoneType`, ill-formed `SdfPath`, PhysX error, invalid transform, broadphase fault, or explosion signature | `kit_20260808_104554.log` |

This clears the **asset import, hardware fit, optical-frame, collision-clear
task reach, and pre-contact stability** gate. It does not yet claim robot
deleafing success: the measured 65-118 mm air gap is intentional and still
requires a controlled final approach. Blade-to-petiole contact, robot-driven cut
triggering, left-tool interaction if used, and sustained-force
tearing remain the next explicit gate. The existing `C` command directly
releases the selected cut joint; it is not evidence of physical blade-triggered
severance.

A follow-up visual review requested that the U arc face upward so the flat plate
is presented cleanly for cutting. Rolling the knife 90 degrees about its
unchanged blade axis moved CAD +Z from tool +Y to tool +X. The rendered ready
pose now measures the arc-facing vector at Z=+0.999845 while preserving a blade
extension of Y=-0.999876 toward the target row. The same review requested direct camera access: the interaction
window and keys `1`-`4` now switch the active viewport between inspection,
head D405, left-wrist D405, and right-wrist D405 video respectively. Headless
captures remain available through `--capture-camera`.

The reported `KeyError: <class 'NoneType'>` in
`omni.kit.manipulator.selector` was isolated from physics. Isaac Sim 5.1's
mixed USD/Fabric selector can mark a selection handled by setting it to `None`,
then pass that value to the transform manipulator, which indexes it as an Sdf
path type and emits the subsequent ill-formed empty-`SdfPath` warning. The
greenhouse does not use the native transform gizmo: visible-mesh Shift-drag is
owned by `VisualPull`. GUI startup now clears stale USD selection and destroys
only the default transform selector's event subscription, while retaining
viewport camera navigation and the custom pull path. The report records
`native_transform_selector_disabled=true` when this targeted workaround is
active.

### Physical blade and bi-manual deleafing foundation (2026-08-08)

This work is isolated on `koh-dev/deleaf`. It implements the benchmark substrate
without claiming that an autonomous or teleoperated robot trajectory has already
completed the task.

The exploding/violent-contact behaviour was traced to geometry and filtering,
not hidden with extra damping:

- Each new flush petiole cut-zone capsule begins at an artistic mesh junction
  that can overlap the main stem or a sibling petiole at rest. These zones must
  contact the robot but must not make PhysX depenetrate one part of the plant
  from another. Every cut zone is therefore filtered from all other plant
  interaction bodies while robot/tool contact remains enabled. The interaction
  set now contains 43 sparse colliders: protected main-stem zones plus one cut
  zone and one grasp zone for each of the 18 real petioles.
- Two nearest visual vines receive 56 static semantic safety proxies. They do
  not add articulated plant load, but make neighbouring-vine collision both
  physical and measurable.
- The URDF import had duplicate live instance proxies in addition to the 17
  restored capsules. All 34 importer-created collision scopes, including empty
  link scopes, are now inactive before the exact 17 source capsules are restored.
- The active vendor `link_*_arm_5` capsule was a whole-tool planning envelope:
  250 mm cylinder length with 75 mm end radii, centred 100 mm below the wrist.
  The source URDF preserves a precise right-wrist endpoint capsule from
  z=+2 mm to z=-50 mm. Both symmetric wrists now use that 52 mm cylinder centred
  at z=-24 mm for contact simulation. The left palm proxy also stops at z=-25 mm
  instead of filling the finger channel down to z=-73 mm.

A closer numerical dual-arm IK spawn was tested and explicitly rejected. Contact
traces showed the right U support touching the protected main stem and
`SubStem_01`, while the broad old arm-5 and left-palm proxies overlapped the
target area. The launcher was restored to the previously accepted non-contact
right pre-contact pose and official SDK-ready left arm. Starting closer is not a
substitute for collision-aware approach motion.

The physical cut implementation now exposes all 18 `SubStem_XX` junctions as
world-space targets with centre, axis, local biological radius, and cut force.
The local radius is the first centreline sample at the junction, not the
segment-average contact radius; for `SubStem_00` this is 4.189 mm rather than the
incorrect 14.420 mm downstream-flare average. Required work is
`66.3 N * max(2r, 3 mm)`. The default acceptance limits are 10 mm/s minimum
forward speed, at most 35 degrees axial alignment for both edge and motion,
2 mm before/25 mm after the junction, 4 mm radial tolerance, 3 mm minimum
travel, and four physics steps of contact memory. Force is capped at 3x while
integrating work so a single solver impulse cannot fake a full cut.

Live contact monitoring distinguishes the straight leading edge from the plate
face and U support, aggregates all contact points once per 240 Hz step, and logs
main-stem, non-target-organ, neighbouring-vine, greenhouse-structure, and robot
self-contact as blocking safety violations. A physical cut is benchmark-valid
only when it is the selected target, the safety ledger is clear, and the
bi-manual state machine accepts it. `C` remains an explicit debug-only joint
release and is always invalid for benchmark scoring.

The retained left tongs now have contact reporting on both fingers and one
pre-authored, initially disabled fixed grasp joint per petiole. Three consecutive
steps with both fingers and at least 1 N establish the grasp; enabling a
pre-authored joint avoids runtime prim deletion and preserves the orphan after
severance. The task state machine enforces `seek_grasp -> grasped ->
orphan_retained -> transported -> released -> deposited`, including wrong-target
cuts, protected contact, cut-before-grasp, early release, and premature orphan
loss as explicit failure reasons. Deposit requires the orphan to be within the
aisle floor zone, below the floor tolerance, and moving no faster than 0.15 m/s.

**Current acceptance (Isaac Sim 5.1):**

| Check | Result | Evidence |
|---|---|---|
| Hermetic regressions | cut force/direction/work, disconnected taps, protected-contact failure, complete bi-manual ordering, cut-zone filtering, knife semantics, and robot placement covered; 48 passed, 1 PhysX-only test skipped outside a running app | `python -m pytest examples/greenhouse_sim/greenhouse_sim -q` |
| Robot collider rebuild | 34 imported collision scopes inactive; exactly 17 URDF capsules restored; both wrist contact capsules 52 mm; 31 authored robot colliders | `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| 480-step integrated soak | succeeded=true; 121/121 vine bodies finite; 0 runaway organs; 68.693 mm maximum compliant settling; 34/34 robot bodies finite; 2.078 degree base tilt | `data/greenhouse_sim/deleaf_collision_fix_stability_480.json` |
| Rest safety | no robot-vine, knife-vine, gutter, camera, or arm contact; only chassis/wheel support contacts; safety_clear=true | same report |
| Safe right pre-contact | blade 117.945 mm and U support 65.033 mm from settled `SubStem_00`; arc still faces upward | same report |
| Error scan | no Python traceback, PhysX error, invalid joint, NaN, broadphase fault, or explosion signature | `kit_20260808_152754.log` |

This cleared the stable benchmark-physics and event-accounting milestone. The
robot-to-vine execution gate described here is completed in the 2026-08-09
validation below.

### Complete physical bi-manual execution (2026-08-09)

The staged RB-Y1 controller now completes the required task in the supplied
greenhouse using the original tomato mesh and the authored physics backend:

1. the left arm approaches the live moving petiole, closes both physical
   fingers, and requires three consecutive opposed-contact steps before a grasp
   can be established;
2. the left arm counterholds the exact selected organ while the right flat blade
   approaches from a collision-checked side and executes force-limited transverse
   slicing cycles;
3. the selected petiole severs only after the leading edge satisfies contact,
   cut-zone, direction, force, work, selected-target, and protected-contact
   gates;
4. the left arm retains the orphan, clears the vine row, carries it to a
   chassis-clear side-aisle zone, opens the gripper, and lets it settle on the
   greenhouse floor.

The earlier false starts identified real constraints rather than being hidden by
weaker thresholds:

- A planned `Link_3` grasp was not the contacted physical body. The manager now
  groups eligible cut/grasp colliders by body and anchors the fixed grasp joint
  to the actual opposed-finger contact on `Link_2`; contact gaps reset the
  consecutive-step gate.
- Both arm drives use the vendor URDF effort limits `(70, 70, 70, 40, 10, 10,
  8)` N m. Exact RB-Y1 v1.0 FK/IK and a point Jacobian reject a counterhold
  posture unless it can supply at least 1.10 times the 66.3 N cut requirement.
  A tested 72.1 N posture was therefore rejected rather than accepted by
  reducing the safety margin.
- Moving the entire unheld rigid petiole can no longer accumulate cut work.
  Reverse/unload motion also contributes none. The rigid-tissue fracture proxy
  can accumulate commanded penetration only while the real leading edge is in
  an admissible physical contact, the blade is moving in the required direction,
  force is above threshold, the exact target is counterheld by the left grasp,
  and the protected-contact ledger is clear.
- One large joint-space transport interpolation arced through foliage. The
  accepted route uses short Cartesian row-clearance waypoints, preserves the
  grasp pose whenever full-pose IK passes, falls back to bounded point/axis IK
  without changing its thresholds, translates sideways only after row
  clearance, and holds the terminal pose before release.
- Dropping directly over the original base stance let the long orphan rotate
  into a wheel. The accepted base is 150 mm farther back at
  `(6.99114, 3.93000, -0.3050817)` m. Its plan footprint extends beneath the
  neighbouring elevated supplied gutter, but the full 3-D contact trace has no
  gutter, robot, tool, or neighbouring-vine contact. Smaller trial offsets that
  reduced counterhold effort margin were explicitly rejected.

**Severance-model limitation, recorded rather than hidden:** PhysX cannot make a
live internal joint disappear inside the current reduced-coordinate petiole
articulation. Disabling `Joint_002` in USD left the subtree constrained in the
running solver. The severer therefore records the nearest geometric cut joint
and 21.003 mm geometric stub, but physically disables the organ's pre-authored
maximal-coordinate `BaseJoint`, yielding a realised physical stub of zero. This
is reported as `release_mode=maximal_coordinate_organ_base`. A future topology
backend must partition/rebuild the articulation at the selected internal joint
if non-zero residual stub physics is required. Likewise, the commanded
traction-separation penetration is an explicit approximation for a rigid
capsule that cannot deform or split; it does not replace the required physical
contact, force, direction, work, counterhold, or safety evidence.

**Definitive Isaac Sim 5.1 acceptance:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse suite | 63 passed, 1 PhysX-only test skipped outside a running app | `python -m pytest examples/greenhouse_sim -q` |
| 480-step integrated stability | succeeded=true; finite vine; 0 runaway organs; 68.700 mm maximum compliant motion; robot stable at 2.078 degrees tilt; pre-contact valid; empty stderr | `data/greenhouse_sim/stability_480_final.json` |
| Opposed physical grasp | actual `Link_002`; 26.603 N peak grasp force; exact selected target retained | `data/greenhouse_sim/bimanual_full_acceptance_final_pass.json` |
| Hardware effort margin | vendor joint limits active; counterhold posture admitted only above 72.93 N capacity | same report, `left_static_counterhold` |
| Physical cut | one intended cut; 73.875 N peak; 0.5910 J work vs 0.5554 J required; 8.697 mm forward travel; edge alignment 0.9825; transverse-motion alignment 0.9871; benchmark_valid=true | same report |
| Safe transport/deposit | 315.0 mm maximum clearance; endpoint hold; task phase `deposited`; floor contact=true; final speed 0.1490 m/s; zero unsafe contacts | same report |
| Whole acceptance | top-level succeeded=true; bimanual probe=true; robot stability=true; pre-contact=true; no Python stderr | same report and `.stderr.log` |

This is the known-target simulator-stability gate for the lab phase, not the
completion of the benchmark. The next section records the deterministic target
and simulator-only teleoperation layer added on top of it. Alternate-target
physical acceptance, lab leader-hardware validation, tear-force calibration,
task/scene randomisation, observation/action interfaces, VLA policy adapters,
metrics, and RL batching remain downstream and must not be claimed as complete.

### Deterministic targets and simulator-only teleoperation (2026-08-09)

Episode targets are no longer hard-coded in monitor, grasp, pre-contact, and
probe code. `--target-vine`, `--target-organ`, and `--episode-seed` resolve an
exact or seeded `auto` target from stable sorted physical cut-joint candidates;
the selected key and selection mode are persisted in every report. The visible
UI starts on the same target instead of silently resetting the managers to
`SubStem_00`. `run_bimanual_repeatability.py` launches one isolated Isaac
process per target/seed and counts an episode only when all strict checks pass:
top-level and probe success, exactly one intended benchmark cut, clear blade
safety, zero unsafe contacts, and terminal `deposited` state.

The simulator now consumes an atomic `greenhouse.teleop.v1` JSON mailbox. Each
arm has an independent deadman bit; commands must have a strictly increasing
sequence and a fresh host-monotonic timestamp, pass exact URDF joint limits,
and pass a configurable joint-speed limiter. Deadman release, stale watchdog,
invalid command, or an unsafe-contact latch actively changes the drive target
to the measured pose. Right-arm commands are also mapped through exact RB-Y1
FK to a commanded knife-edge velocity, so teleoperation still cannot bypass
the physical contact/force/direction/work/counterhold cut gate.

`rby1_leader_to_sim.py` reads the vendored dual leader arms and trigger tools
and publishes only this simulator mailbox. It contains no RB-Y1 address,
command stream, power, servo, or gripper connection; the physical robot is not
commanded. Leader torque is limited to vendor-style gravity compensation,
joint-limit resistance, and damping, and a communication fault disables leader
torque. This path is code-complete but remains **lab-hardware unverified** until
the leader devices are connected deliberately.

When `--teleop-record-dir` is supplied, each run creates a unique episode with
metadata, synchronized JSONL steps, measured arm positions/velocities, both EE
world transforms, active target/task/cut/safety state, the gated action, and
selected head/wrist D405 RGB frames. Replicator warmup occurs while the timeline
is paused so observation setup cannot advance unobserved physics.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse suite | 76 passed, 1 PhysX-only test skipped outside a running app; all new files pass Ruff | `D:\isaac-sim\python.bat -m pytest examples/greenhouse_sim/greenhouse_sim -q` |
| Final known-target physical acceptance | top-level/probe true; target `Vine_0000/SubStem_00`; one benchmark-valid cut at 71.444 N; task deposited; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/bimanual_full_target_teleop_final.json` |
| Final integrated stability | succeeded=true; finite vine; 0 runaway organs; 68.732 mm maximum compliant motion; robot stable at 2.078 degrees tilt; selected-target pre-contact valid | `data/greenhouse_sim/stability_target_teleop_480_final.json` |
| Hardware-free teleop/recording | one fresh disabled command accepted; neither arm enabled; no unsafe latch; one synchronized JSONL step; head RGB 320x180, range 1-244, 6,736 unique colors | `data/greenhouse_sim/teleop_camera_warmup_validation.json` and its reported episode directory |

### Multi-target reach/collision blocker resolved, 2026-08-09

The recorded fixed-base `SubStem_01` failure was real: distal Link 3/2 were
outside left-arm reach, while falling back to proximal Link 1 caused a
non-target collision. The fix does not relax either gate. Robot authoring now
waits for the settled physical target and uses exact RB-Y1 IK to test
deterministic 0/30/60/90 mm aisle advances. Planning enforces a 20 mm reach
reserve, a distal segment floor, and wrist-D405 clearance; fixed positioning
remains available explicitly through `--robot-position-mode fixed`.

The bimanual sequence now also:

- keeps the planned distal segment after settling instead of silently falling
  back proximally;
- scores multiple left-wrist IK branches using the authored D405 volume;
- selects the knife wing with the greatest live non-target clearance;
- evaluates exact segment-to-oriented-box clearance for both the flat blade and
  non-cutting U-support over sampled RB-Y1 joint interpolation;
- selects the safer right-retract direction family and then maximizes lateral/
  vertical separation before aisle stow; and
- selects a swept, payload-clear left transport route while the PhysX contact
  ledger remains the strict zero-waiver acceptance authority.

The force/direction/work/counterhold cut gate remains physical. Low-force
leading-edge contact can establish entry-side geometry but cannot contribute
cut work. A counterheld full-diameter crossing may account for rigid U-guide
displacement only after physical leading-edge contact and still must meet
force, direction, work, travel, intended-target, and protected-contact
requirements.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse suite | 85 passed, 1 PhysX-only test skipped; new base-planner modules pass Ruff | `D:\isaac-sim\python.bat -m pytest examples\greenhouse_sim\greenhouse_sim -q` |
| Baseline `SubStem_00` full episode | top-level/probe true; nominal base; distal Link 3 plan; `positive_x_extra_wide_high` retract; one valid cut at 72.559 N and 0.55564 J vs 0.55540 J required; task deposited with 257.780 mm clearance; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/bimanual_full_substem00_clearance_tiebreak.json` |
| Former blocker `SubStem_01` full episode | top-level/probe true; 30 mm base advance; distal Link 2 plan; positive-X blade wing; `negative_x_then_lift` retract; one valid cut at 74.462 N and 0.45248 J vs 0.43719 J required; task deposited with 264.425 mm clearance; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/bimanual_full_substem01_direction_family.json` |
| 480-step integrated stability | succeeded=true; 121/121 vine bodies finite; 0 runaway organs; 68.710 mm maximum compliant motion; 34/34 robot bodies finite; 2.078 degree base tilt; selected-target pre-contact valid; contacts limited to floor support | `data/greenhouse_sim/stability_multitarget_blocker_fix_480.json` |

This closes the known fixed-base reach/collision blocker and verifies two
distinct target geometries. It does not yet claim benchmark-wide
repeatability: the next simulator gate is the isolated-process target/seed
matrix, followed by deliberate lab leader-arm validation. Robot benchmarking,
policy/VLA integration, and RL remain downstream of those stability gates.

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
- 2026-08-06 — Removed the synthetic foliage catcher's invisible collision
  with the RB-Y1 articulation, added contact tracing and measured petiole-range
  acceptance, staged a trough-clear right-arm pre-contact pose, and disabled
  the faulty unused Isaac transform selector in the interactive GUI.
- 2026-08-08 — Verified the benchmark sublayers the original greenhouse without
  scale changes and documented its authored 1.193 m floor-to-gutter-top height;
  moved RB-Y1 to the opposite aisle, fully removed the original right tongs from
  rendering/contact, mounted the knife directly to the retained EE flange, and
  passed a 480-step robot/vine/contact acceptance soak.
- 2026-08-08 — Replaced the approximate wrist-camera offsets with the exact
  RB-Y1 v1.1 FreeCAD screw-pair datum, corrected the extracted-STL origin error,
  regression-checked both mirrored mounts, verified a live right D405 frame,
  and repeated the 480-step robot/vine/contact acceptance soak.
- 2026-08-08 — Added leading-edge force/direction/work cuts, protected-contact
  accounting, 18 petiole cut/grasp targets, and the required left-grasp/right-cut/
  retain/transport/floor-deposit state machine; removed plant self-depenetration,
  duplicate imported robot colliders, oversized wrist planning envelopes, and
  the left-palm jaw obstruction; passed 48 regressions and a clean 480-step
  integrated robot/vine/contact soak on `koh-dev/deleaf`.
- 2026-08-09 — Completed the hardware-effort-limited RB-Y1 left-grasp/right-cut/
  retain/transport/floor-deposit execution, added exact v1.0 kinematics and
  force-capacity checks, made rigid-tissue fracture/topology approximations
  explicit in reports, and passed 63 regressions plus top-level physical
  acceptance with one valid cut and zero unsafe contacts on `koh-dev/deleaf`.
- 2026-08-09 — Added deterministic physical-target episodes and strict
  isolated-process repeatability aggregation; added a simulator-only dual
  leader-arm mailbox with deadman/watchdog/URDF/speed/contact gates and
  synchronized D405/action recording; preserved the complete known-target
  physical acceptance and recorded the fixed-base `SubStem_01` reach/collision
  failure instead of relaxing criteria on `koh-dev/deleaf`.
- 2026-08-09 — Resolved the fixed-base `SubStem_01` blocker with
  target-conditioned base placement, distal-segment and D405-clear left IK,
  target-specific knife-wing selection, exact full-tool swept-volume retract
  planning, and payload-clear orphan transport; passed 85 regressions, both
  `SubStem_00`/`SubStem_01` full physical episodes with zero unsafe contacts,
  and the 480-step integrated stability soak on `koh-dev/deleaf`.
