"""Give a vine compliant physics that can be bent, pulled and cut.

Stem organs become chains of capsule rigid bodies joined by D6 joints with
angular drives, so a petiole bends and droops under load instead of being a
rigid stick. Drive stiffness comes from Euler-Bernoulli beam theory rather than
hand-tuning, so the compliance follows the organ's own measured radius and
tapers the way the plant does.

Two constraints from the engine shape everything here.

Isaac Sim 5.1 cannot cut: PhysX has no runtime topology change for deformables
and no API to re-cook one, so FEM stems could never be severed. Volumetric
deformables are also unusable for this task independently -- PhysX soft bodies
have no static friction, so a gripper cannot hold one, and they exhaust GPU
memory at a couple of hundred bodies when a data-collection benchmark needs
thousands of episodes. Compliant capsule chains reproduce the bending, drooping
and tearing the task actually depends on, deterministically and cheaply.

The plant is a PhysX **articulation** (reduced coordinates). That is not a
detail: a tomato stem is stiff and light, and beam theory puts a joint at
~1000 N*m/rad against a ~0.1 g link. Maximal-coordinate joints cannot integrate
that at any sane timestep -- the stable ceiling is some six orders of magnitude
lower -- so the chain detonates on the first frame. Softening it enough to be
stable brings back the collapse the trellis exists to prevent. Reduced
coordinates solve stiff serial chains natively and hold the real stiffness.

The cost is that `physics:breakForce` is silently ignored inside an
articulation, so tearing cannot rely on it. It does not need to: severance is a
joint being disabled either way, and `cutting.Severer` watches joint force and
disables past the measured detachment threshold. That also removes a bug the
breakForce route had, where solver transients during settling tore every petiole
off within the first frames.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from greenhouse_sim import organs
from greenhouse_sim import skeleton as skeleton_module
from greenhouse_sim import usd_env

usd_env.ensure_pxr()

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


@dataclasses.dataclass(frozen=True)
class TissueProperties:
    """Mechanical properties of living vine tissue.

    No Young's modulus has been published for the fresh tomato main stem, so
    these values rest on three converging lines, all landing at 100-300 MPa:
    fresh greenhouse cucumber cane in tension at 281/199/137 MPa base-to-apex
    (Xu et al. 2016, the nearest high-wire analogue), petioles measured at valid
    span in four species at 110-192 MPa (Langer et al. 2021), and a self-weight
    cantilever check on a tomato leaf back-solving to ~148 MPa.

    The one direct tomato measurement (pedicel, 2.8-7.1 MPa, Weng et al. 2024)
    is far lower but was taken at a span/depth ratio of 1.5-2.4, where mid-span
    deflection is shear-dominated; its own data gives this away, reporting
    modulus falling with specimen diameter at r = -0.893, which no real material
    does. Corrected for shear it reconciles to 11-60 MPa.

    The sanity check that settles it: at 5 MPa a tomato leaf would deflect 2.9 m
    on a 0.12 m petiole. Tomato leaves visibly hold themselves up, so any
    modulus below ~50 MPa is falsified by observation.
    """

    # Petioles and the young upper stem.
    youngs_modulus_pa: float = 1.5e8
    # The lignified stem base, which is roughly twice as stiff.
    lignified_youngs_modulus_pa: float = 2.5e8
    # Inferred from the directly measured 73-79% moisture content.
    density_kg_m3: float = 950.0
    damping_ratio: float = 0.1

    # Minimum rotational inertia per capsule axis. The analytical stem-only
    # inertia is too small for the stiff mixed-coordinate graph and omits the
    # leaf/flower art that rides these links. 1e-5 kg*m^2 matches the stable
    # 10 cm petiole probe and was verified to give identical motion with zero
    # and 28 active contact shapes.
    min_diagonal_inertia_kg_m2: float = 1.0e-5

    # Floor on the radius used for stiffness. The render mesh tapers the stem to
    # a modelling point (fitted radius reaches 0.5 mm at the tip), but a real
    # tomato apex is ~6.4 mm across, i.e. 3.2 mm in radius (Gao et al. 2024).
    # Bending stiffness goes as r^4, so an unclamped fitted radius makes the top
    # of the stem four orders of magnitude too floppy and the vine folds over
    # no matter how the trellis is tuned. Collision geometry still uses the
    # measured radius; only the structural value is clamped.
    min_structural_radius_m: float = 0.0032

    # Severance thresholds, from measured tomato leaf-pruning forces: a blade
    # shears the petiole at ~66 N, while pulling detaches it at ~33 N. Keeping
    # them separate is what lets the benchmark score a clean cut apart from a
    # tear, which is an agronomically real distinction.
    cut_force_n: float = 66.3
    tear_force_n: float = 32.5

    # Calibration between beam-theory stiffness and what the angular drive
    # actually delivers. Determined by a controlled cantilever test rather than
    # from the documented units: a link of this scale needs a drive stiffness
    # near 1e2 to hold itself up, while E*I/L for a petiole yields ~1e-2, and
    # sweeping the factor showed the plant only becomes coherent around 1e4.
    # Below that the joints behave as free hinges and the canopy collapses no
    # matter how the trellis or tissue properties are set.
    stiffness_scale: float = 1.0e4

    def modulus_at(self, height_fraction: float) -> float:
        """Young's modulus along the main stem, base (0.0) to growing tip (1.0).

        Tapering the modulus as well as the radius reproduces the measured
        base-to-apex stiffness gradient, which a single value cannot.
        """
        blend = min(max(height_fraction, 0.0), 1.0)
        return self.lignified_youngs_modulus_pa + blend * (self.youngs_modulus_pa - self.lignified_youngs_modulus_pa)


@dataclasses.dataclass(frozen=True)
class Link:
    """One capsule body in an organ's chain."""

    path: str
    organ: int
    index: int
    start: np.ndarray
    end: np.ndarray
    radius: float

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


@dataclasses.dataclass(frozen=True)
class Junction:
    """Where an organ meets its parent: the cut site, and where it tears.

    Carries the authored stiffness and lever arm so the load at the junction can
    be recovered from how far the joint is bent, without reaching into physics
    tensor state.
    """

    joint_path: str
    parent_path: str | None
    child_path: str
    stiffness_nm_per_rad: float
    lever_m: float
    tear_force_n: float


@dataclasses.dataclass(frozen=True)
class PlantRig:
    """The simulable form of one plant."""

    root_path: str
    links: list[Link]
    joints: dict[str, str]  # joint prim path -> child link path
    cut_joints: dict[str, str]  # organ label -> joint prim that severs it
    junctions: dict[str, Junction] = dataclasses.field(default_factory=dict)
    collider_paths: tuple[str, ...] = ()
    collision_mode: str = "interaction"

    def link_paths_for(self, organ: int) -> list[str]:
        return [link.path for link in self.links if link.organ == organ]


def beam_stiffness(youngs_modulus_pa: float, radius_m: float, length_m: float) -> float:
    """Bending stiffness of a beam segment, in N*m per radian.

    Euler-Bernoulli: K = E*I/L with second moment of area I = pi*r^4/4 for a
    circular section. The fourth power means a tapering stem naturally becomes
    far more compliant toward the tip, without that being modelled separately.
    """
    if radius_m <= 0 or length_m <= 0:
        return 0.0
    second_moment = math.pi * radius_m**4 / 4.0
    return youngs_modulus_pa * second_moment / length_m


def capsule_mass(radius_m: float, length_m: float, density_kg_m3: float) -> float:
    """Mass of a capsule: a cylinder plus its two hemispherical caps."""
    return density_kg_m3 * (math.pi * radius_m**2 * length_m + 4.0 / 3.0 * math.pi * radius_m**3)


def _orient_to(axis: np.ndarray) -> Gf.Quatf:
    """Rotation taking +Z onto `axis`, which is a capsule's local long axis."""
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    direction = axis / norm
    reference = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(reference, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        return Gf.Quatf(0.0, 1.0, 0.0, 0.0)  # 180 degrees about X
    cross = np.cross(reference, direction)
    half = math.sqrt(2.0 * (1.0 + dot))
    return Gf.Quatf(half / 2.0, Gf.Vec3f(*(cross / half)))


def _define_capsule(
    stage: Usd.Stage,
    path: Sdf.Path,
    link: Link,
    properties: TissueProperties,
    *,
    visible: bool = False,
    collidable: bool = True,
) -> Sdf.Path:
    """A rigid body carrying its collider as a child, returning the collider.

    Body and collider are separate prims rather than one capsule because USD
    purpose is inherited by an entire subtree: hiding a capsule with the "guide"
    purpose would also hide any render mesh parented to it, and the whole point
    is that the vine's own art rides on these bodies. Keeping the body a plain
    Xform lets the collider be hidden and the art stay visible as siblings.
    """
    body = UsdGeom.Xform.Define(stage, path)
    centre = 0.5 * (link.start + link.end)
    transformable = UsdGeom.Xformable(body.GetPrim())
    transformable.AddTranslateOp().Set(Gf.Vec3d(*centre))
    transformable.AddOrientOp().Set(_orient_to(link.end - link.start))

    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    # A long chain of stiff constraints needs far more position iterations than
    # the default; too few and the chain cannot resolve, so links drift apart
    # and organs detach from the plant.
    from pxr import PhysxSchema  # noqa: PLC0415

    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    physx_body.CreateSolverPositionIterationCountAttr(64)
    physx_body.CreateSolverVelocityIterationCountAttr(4)
    physx_body.CreateMaxDepenetrationVelocityAttr(0.5)
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass = capsule_mass(link.radius, link.length, properties.density_kg_m3)
    mass_api.CreateMassAttr(mass)
    # Author inertia independently of collision geometry. Otherwise PhysX uses
    # fallback inertia for non-contact links but derives capsule inertia for
    # links that own a contact proxy, so merely enabling robot contact changes
    # the plant's structural response. A slender solid-cylinder approximation
    # is sufficient at this link resolution and keeps all bodies consistent.
    inertia_radius = min(float(link.radius), 0.5 * link.length)
    transverse = mass * (3.0 * inertia_radius**2 + link.length**2) / 12.0
    axial = 0.5 * mass * inertia_radius**2
    inertia_floor = properties.min_diagonal_inertia_kg_m2
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            max(transverse, inertia_floor),
            max(transverse, inertia_floor),
            max(axial, inertia_floor),
        )
    )

    collider_path = path.AppendChild("Collider")
    capsule = UsdGeom.Capsule.Define(stage, collider_path)
    capsule.CreateAxisAttr(UsdGeom.Tokens.z)
    # A capsule is its cylinder plus two hemispherical caps, so its true length
    # is height + 2*radius. Using the fitted radius unclamped makes a short,
    # thick segment collide as a ball several times longer than the link it
    # stands for -- on these assets up to 4.2x -- which silently engulfs
    # neighbouring organs and leaves PhysX resolving overlaps that should never
    # exist. Clamping keeps every collider inside its own segment.
    radius = min(float(link.radius), 0.5 * link.length)
    capsule.CreateRadiusAttr(radius)
    capsule.CreateHeightAttr(max(link.length - 2.0 * radius, 1e-4))
    if collidable:
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    if visible:
        capsule.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.42, 0.14)])
    else:
        UsdGeom.Imageable(capsule).CreatePurposeAttr(UsdGeom.Tokens.guide)
    return collider_path


def _define_joint(
    stage: Usd.Stage,
    path: Sdf.Path,
    parent: str | None,
    child: str,
    anchor: np.ndarray,
    stiffness: float,
    properties: TissueProperties,
    *,
    breakable: bool,
    exclude_from_articulation: bool = False,
) -> None:
    """A compliant 3-DOF rotational joint anchored at an organ junction."""
    joint = UsdPhysics.Joint.Define(stage, path)
    if parent is not None:
        joint.CreateBody0Rel().SetTargets([parent])
    joint.CreateBody1Rel().SetTargets([child])

    # Both the anchor *and* the frame orientation have to be authored. The joint
    # frames default to identity, so with only positions set the angular drives
    # target zero relative rotation between two identity frames -- which asks
    # every link to lie parallel to its parent and snaps the whole plant
    # straight the instant simulation starts, with or without gravity. Anchoring
    # the joint frame to the child's rest orientation makes "zero" mean "the
    # pose the plant was authored in".
    child_world = UsdGeom.Xformable(stage.GetPrimAtPath(child)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    child_rotation = child_world.ExtractRotationQuat()
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(child_world.GetInverse().Transform(Gf.Vec3d(*anchor))))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    if parent is None:
        # Parent is the world, so the joint frame is the child's rest pose
        # expressed directly in world coordinates.
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(child_rotation))
    else:
        parent_world = UsdGeom.Xformable(stage.GetPrimAtPath(parent)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(parent_world.GetInverse().Transform(Gf.Vec3d(*anchor))))
        relative = parent_world.ExtractRotationQuat().GetInverse() * child_rotation
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(relative))

    # Translation is rigid; the organ bends, it does not stretch.
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)  # low > high locks the DOF

    # Drives pull the organ back toward its rest pose, which is what makes a
    # chain of free bodies behave like a beam rather than a rope.
    damping = 2.0 * properties.damping_ratio * math.sqrt(max(stiffness, 0.0))
    for axis in ("rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        # USD angular drives are documented per-degree while beam stiffness is
        # per-radian, hence the conversion; stiffness_scale exists to test that.
        scale = properties.stiffness_scale * math.pi / 180.0
        drive.CreateStiffnessAttr(float(stiffness * scale))
        drive.CreateDampingAttr(float(damping * scale))

    del breakable  # Tearing is force-monitored, not breakForce; see `cutting`.
    if exclude_from_articulation:
        # Joins two separate organ articulations, so it has to be solved in
        # maximal coordinates: an articulation is a single tree and cannot span
        # two roots.
        joint.CreateExcludeFromArticulationAttr(defaultValue=True)
    # Authored now, while the scene is still being built, so that severing at
    # runtime only changes a value. Creating the attribute mid-simulation
    # instead makes PhysX resync the whole plant.
    joint.CreateJointEnabledAttr(defaultValue=True)


def author_plant_physics(
    stage: Usd.Stage,
    plant: organs.Plant,
    root_path: str,
    skeletons: dict[int, skeleton_module.Skeleton],
    to_stage_frame,
    *,
    properties: TissueProperties | None = None,
    visible_colliders: bool = False,
    articulated: bool = True,
    collision_mode: str = "interaction",
) -> PlantRig:
    """Build capsule chains and joints for one plant under `root_path`.

    `to_stage_frame` converts organ-space points into the stage's frame, so the
    caller owns the coordinate convention rather than this module guessing it.

    Each organ becomes its **own** articulation, joined to its parent by a
    maximal-coordinate joint. That split is forced by two measured limits, both
    in `physics_probe.py`. Drives on a generic D6 joint bottom out at about 1 mm
    of residual sag no matter how stiff they are, while an articulation joint
    holds rigidly -- and 1 mm compounded through ninety-nine stem joints in
    series is the half-metre slump the plant used to show. But a single
    articulation holding the whole plant will not build either: PhysX manages
    128 links and crashes at 256, against a vine's ~400.

    Per-organ articulations satisfy both. The main stem's ~99 links fit inside
    one, petioles are a handful each, and the D6 floor now applies once per
    organ instead of accumulating along the stem. The connecting joint is also
    the severance point, which is where a maximal-coordinate joint is wanted
    anyway, since `breakForce` is ignored inside an articulation.

    Structural links and contact geometry are deliberately separate concerns.
    "interaction" (the default) enables contact only on the main stem and the
    18 labelled SubStem petioles used by the deleafing benchmark. The remaining
    capsules still carry mass, inertia, joints, and the original render mesh,
    but do not enter the contact solver. This matters because the source asset
    is an artistic rest pose with leaflets and trusses already intersecting;
    making every structural capsule collide asks PhysX to violently resolve a
    pose that is intentionally interpenetrating. "all" is retained only for
    diagnosing that raw collider set, and "none" isolates constraints.
    """
    properties = properties or TissueProperties()
    if collision_mode not in {"interaction", "none", "all"}:
        raise ValueError(f"unknown collision mode: {collision_mode}")
    scope = Sdf.Path(root_path).AppendChild("Physics")
    UsdGeom.Scope.Define(stage, scope)

    # Each organ articulation has native self-collision disabled below. Do not
    # add an all-to-all USD CollisionGroup here: in Isaac 5.1 the collection is
    # expanded during articulation parsing and invalidates this mixed
    # articulation/maximal-coordinate graph before the first step.

    links: list[Link] = []
    collider_paths: list[Sdf.Path] = []
    joints: dict[str, str] = {}
    cut_joints: dict[str, str] = {}
    junctions: dict[str, Junction] = {}
    organ_links: dict[int, list[Link]] = {}

    # Parents must exist before their children are jointed to them.
    ordered = sorted(
        (o for o in plant.organs if o.index in skeletons),
        key=lambda o: _depth(plant, o.index),
    )

    for organ in ordered:
        centreline = skeletons[organ.index]
        points = to_stage_frame(centreline.points)
        if points.shape[0] < 2:
            continue

        organ_scope = scope.AppendChild(f"Organ_{organ.index:04d}")
        UsdGeom.Xform.Define(stage, organ_scope)
        if articulated:
            # One articulation per organ, small enough for PhysX to build and
            # rigid enough that its internal joints do not sag.
            organ_prim = stage.GetPrimAtPath(organ_scope)
            UsdPhysics.ArticulationRootAPI.Apply(organ_prim)

            from pxr import PhysxSchema  # noqa: PLC0415

            physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(organ_prim)
            physx_articulation.CreateArticulationEnabledAttr(defaultValue=True)
            # Organs are authored interpenetrating their neighbours; resolving
            # that inside the articulation serves no purpose.
            physx_articulation.CreateEnabledSelfCollisionsAttr(defaultValue=False)
        chain: list[Link] = []
        contact_segments: set[int] = set()
        if collision_mode == "all":
            contact_segments.update(range(points.shape[0] - 1))
        elif collision_mode == "interaction":
            arcs = centreline.arc_lengths()
            centres = 0.5 * (arcs[:-1] + arcs[1:])
            if organ.index == plant.root:
                # Ten separated main-stem contact zones on a two-metre vine.
                # Dense endpoint-touching capsules are exactly what invalidated
                # PhysX; physical spacing keeps the result independent of the
                # requested structural segment length.
                for target in np.arange(0.10, float(arcs[-1]), 0.20):
                    contact_segments.add(int(np.argmin(np.abs(centres - target))))
            elif organ.label.startswith("SubStem_"):
                # One graspable zone at the middle of each of the 18 real
                # deleafing petioles, clear of its interpenetrating stem base.
                contact_segments.add(int(np.argmin(np.abs(centres - 0.5 * arcs[-1]))))

        for segment in range(points.shape[0] - 1):
            contactable = segment in contact_segments
            radius = float(max(0.5 * (centreline.radii[segment] + centreline.radii[segment + 1]), 1e-4))
            link = Link(
                path=str(organ_scope.AppendChild(f"Link_{segment:03d}")),
                organ=organ.index,
                index=segment,
                start=points[segment],
                end=points[segment + 1],
                radius=radius,
            )
            collider = _define_capsule(
                stage,
                Sdf.Path(link.path),
                link,
                properties,
                visible=visible_colliders,
                collidable=contactable,
            )
            if contactable:
                collider_paths.append(collider)
            chain.append(link)
            links.append(link)

        # The main stem stiffens toward its lignified base; every other organ is
        # young tissue at a single modulus.
        is_main_stem = organ.index == plant.root
        span = max(len(chain) - 1, 1)

        # Internal joints hold the organ together along its own length.
        for segment in range(1, len(chain)):
            modulus = properties.modulus_at(segment / span) if is_main_stem else properties.youngs_modulus_pa
            stiffness = beam_stiffness(
                modulus,
                max(chain[segment].radius, properties.min_structural_radius_m),
                max(chain[segment].length, 1e-4),
            )
            path = organ_scope.AppendChild(f"Joint_{segment:03d}")
            _define_joint(
                stage,
                path,
                chain[segment - 1].path,
                chain[segment].path,
                chain[segment].start,
                stiffness,
                properties,
                breakable=False,
            )
            joints[str(path)] = chain[segment].path

        organ_links[organ.index] = chain

        # The joint attaching this organ to its parent is the severance point.
        parent_link = _nearest_link(organ_links.get(organ.parent, []), chain[0].start)
        base_path = organ_scope.AppendChild("BaseJoint")
        base_modulus = properties.modulus_at(0.0) if is_main_stem else properties.youngs_modulus_pa
        base_stiffness = beam_stiffness(
            base_modulus,
            max(chain[0].radius, properties.min_structural_radius_m),
            max(chain[0].length, 1e-4),
        )
        _define_joint(
            stage,
            base_path,
            parent_link.path if parent_link is not None else None,
            chain[0].path,
            chain[0].start,
            base_stiffness,
            properties,
            breakable=parent_link is not None,
            # This joint spans two organ articulations, so it must be solved in
            # maximal coordinates. That is also what makes it severable.
            exclude_from_articulation=articulated and parent_link is not None,
        )
        # No extra anchor for the main stem: its base joint already has an empty
        # body0, which attaches it to the world and gives the articulation its
        # fixed base. A second world joint on the same link would close a loop
        # and crash PhysX.
        joints[str(base_path)] = chain[0].path
        if parent_link is not None:
            cut_joints[organ.label] = str(base_path)
            junctions[organ.label] = Junction(
                joint_path=str(base_path),
                parent_path=parent_link.path,
                child_path=chain[0].path,
                stiffness_nm_per_rad=base_stiffness,
                # Lever arm for converting the junction's restoring torque into
                # a pull force: the length of the organ hanging off it.
                lever_m=float(sum(link.length for link in chain)),
                tear_force_n=properties.tear_force_n,
            )

    return PlantRig(
        root_path=root_path,
        links=links,
        joints=joints,
        cut_joints=cut_joints,
        junctions=junctions,
        collider_paths=tuple(str(path) for path in collider_paths),
        collision_mode=collision_mode,
    )


def _depth(plant: organs.Plant, index: int) -> int:
    depth, organ = 0, plant.organs[index]
    while organ.parent is not None:
        organ = plant.organs[organ.parent]
        depth += 1
    return depth


def _nearest_link(chain: list[Link], point: np.ndarray) -> Link | None:
    """Link of `chain` whose axis passes closest to `point`."""
    if not chain:
        return None
    return min(chain, key=lambda link: float(np.linalg.norm(0.5 * (link.start + link.end) - point)))


# High-wire tomato is clipped to its support string roughly every 25-35 cm.
DEFAULT_CLIP_SPACING_M = 0.30

# Along the string the clip is nearly inextensible; across it there is a few
# millimetres of free play in the clip before it bears.
CLIP_AXIAL_STIFFNESS_N_M = 5000.0
CLIP_LATERAL_STIFFNESS_N_M = 250.0

# A clip is a collar: the stem can shift a few millimetres inside it and then
# meets the collar wall. Modelling that travel as a hard limit is what actually
# carries the vine -- a spring alone lets the stem drift arbitrarily far under a
# sustained load, however stiff it is.
CLIP_CLEARANCE_M = 0.005

# Growers clip just below the growing point, not a full spacing short of it.
# Leaving a longer head unsupported is not a small error: an unclipped 0.3 m of
# stem tip folds over by half a metre, because cantilever deflection grows with
# the cube of free length.
HEAD_CLEARANCE_M = 0.15


@dataclasses.dataclass(frozen=True)
class Clip:
    """One trellis attachment between the main stem and its support string."""

    joint_path: str
    link_path: str
    height_m: float


def add_trellis_clips(
    stage: Usd.Stage,
    rig: PlantRig,
    root_organ: int,
    *,
    spacing: float = DEFAULT_CLIP_SPACING_M,
    head_clearance: float = HEAD_CLEARANCE_M,
    properties: TissueProperties | None = None,
) -> list[Clip]:
    """Clip the main stem to its support string at regular heights.

    A tomato vine is not self-supporting: a stem self-buckles well before 2 m,
    which is exactly why commercial high-wire tomato is clipped to a string
    every 25-35 cm. Without this the plant collapses under its own weight no
    matter how the tissue stiffness is tuned, so the clips are load-bearing
    structure and not set dressing.

    They are springs rather than fixed joints deliberately. Welding the stem to
    the world would over-stiffen it laterally by orders of magnitude and reduce
    the vine to a static prop, destroying the plant-disturbance signal the
    benchmark scores. With compliant clips the stem still sways a few
    centimetres under contact, which is what a real one does.
    """
    properties = properties or TissueProperties()
    chain = sorted((link for link in rig.links if link.organ == root_organ), key=lambda link: link.index)
    if not chain:
        return []

    scope = Sdf.Path(rig.root_path).AppendChild("Trellis")
    UsdGeom.Scope.Define(stage, scope)

    base_height = float(chain[0].start[2])
    heights = np.array([float(0.5 * (link.start + link.end)[2]) - base_height for link in chain])
    # Clip up to just below the growing point, and no further.
    highest_clip = float(heights.max()) - head_clearance

    # Explicit target heights, then the nearest link to each. Advancing a
    # running counter instead silently drops the topmost clip whenever the
    # remaining run is shorter than one spacing -- which is exactly the stretch
    # whose deflection matters most.
    targets = list(np.arange(spacing, max(highest_clip, spacing), spacing))
    if highest_clip > 0 and (not targets or highest_clip - targets[-1] > 0.25 * spacing):
        targets.append(highest_clip)

    clips: list[Clip] = []
    used: set[int] = set()
    for target in targets:
        index = int(np.argmin(np.abs(heights - target)))
        if index in used:
            continue
        used.add(index)
        link = chain[index]
        anchor = 0.5 * (link.start + link.end)
        height = float(heights[index])

        path = scope.AppendChild(f"Clip_{len(clips):02d}")
        joint = UsdPhysics.Joint.Define(stage, path)
        # No body0 relationship: the clip anchors to the world, standing in for
        # the support string, which is far stiffer than the stem.
        joint.CreateBody1Rel().SetTargets([link.path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        # An articulation is a tree. Each clip ties a link that already has a
        # path to the root back to the world, closing a loop, and a loop joint
        # inside an articulation crashes PhysX outright. Excluding them keeps
        # the clips as maximal-coordinate constraints, which is what loop
        # closures have to be.
        joint.CreateExcludeFromArticulationAttr(defaultValue=True)
        joint.CreateJointEnabledAttr(defaultValue=True)

        mass = capsule_mass(link.radius, link.length, properties.density_kg_m3)
        for axis, stiffness in (
            ("transX", CLIP_LATERAL_STIFFNESS_N_M),
            ("transY", CLIP_LATERAL_STIFFNESS_N_M),
            ("transZ", CLIP_AXIAL_STIFFNESS_N_M),
        ):
            # The collar wall. Without this the spring alone cannot hold a 2 m
            # vine: any sustained load walks the stem out indefinitely.
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(-CLIP_CLEARANCE_M)
            limit.CreateHighAttr(CLIP_CLEARANCE_M)

            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateStiffnessAttr(stiffness)
            drive.CreateDampingAttr(2.0 * properties.damping_ratio * math.sqrt(stiffness * max(mass, 1e-6)))

        # Rotation is left free: a clip is a loose collar, not a weld, and the
        # stem must still be able to pivot within it.
        clips.append(Clip(joint_path=str(path), link_path=link.path, height_m=height))
    return clips


def add_ground_plane(
    stage: Usd.Stage,
    path: str = "/World/GroundPlane",
    *,
    height: float = 0.0,
    size: float = 20.0,
    centre_xy: tuple[float, float] = (0.0, 0.0),
) -> None:
    """A static floor, so severed organs land instead of falling forever."""
    plane = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    plane.CreateSizeAttr(1.0)
    transformable = UsdGeom.Xformable(plane.GetPrim())
    transformable.AddTranslateOp().Set(
        Gf.Vec3d(float(centre_xy[0]), float(centre_xy[1]), height - 0.5 * size * 0.01)
    )
    transformable.AddScaleOp().Set(Gf.Vec3f(size, size, size * 0.01))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdGeom.Imageable(plane).CreatePurposeAttr(UsdGeom.Tokens.guide)


def apply_scene_physics(stage: Usd.Stage, path: str = "/World/PhysicsScene", *, gravity: float = 9.81) -> None:
    """Ensure the stage has a physics scene configured for thin compliant bodies."""
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(path))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(gravity)

    # Imported here rather than at module scope: PhysxSchema ships in a
    # different extension than the USD bootstrap loads, so a top-level import
    # would make this module unusable outside a running Kit app -- and the rest
    # of it is pure geometry that is worth being able to audit offline.
    from pxr import PhysxSchema  # noqa: PLC0415

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(path))
    # Thin, stiff, strongly driven chains need iterations rather than a smaller
    # step to stay stable; too few and a petiole visibly buzzes at rest.
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateEnableCCDAttr(defaultValue=True)
