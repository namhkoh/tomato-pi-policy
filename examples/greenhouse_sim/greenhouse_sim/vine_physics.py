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


PETIOLE_CUT_ZONE_LENGTH_M = 0.025
FOLIAGE_CONTACT_MINIMUM_THICKNESS_M = 0.003
FOLIAGE_CONTACT_PADDING_M = 0.001

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
    cut_position_m: np.ndarray
    cut_axis: np.ndarray
    cut_radius_m: float
    cut_force_n: float


@dataclasses.dataclass(frozen=True)
class ColliderInfo:
    """Semantic identity of one physical plant collision shape."""

    path: str
    body_path: str
    organ: int
    organ_label: str
    segment: int
    role: str


@dataclasses.dataclass(frozen=True)
class SafetyColliderInfo:
    """Static semantic proxy for a neighbouring visual vine."""

    path: str
    vine_name: str
    organ_label: str
    role: str


@dataclasses.dataclass(frozen=True)
class PlantRig:
    """The simulable form of one plant."""

    root_path: str
    links: list[Link]
    joints: dict[str, str]  # joint prim path -> child link path
    cut_joints: dict[str, str]  # organ label -> joint prim that severs it
    junctions: dict[str, Junction] = dataclasses.field(default_factory=dict)
    collider_paths: tuple[str, ...] = ()
    colliders: tuple[ColliderInfo, ...] = ()
    collision_mode: str = "interaction"

    def link_paths_for(self, organ: int) -> list[str]:
        return [link.path for link in self.links if link.organ == organ]

    def cut_segment_frame(
        self, collider: ColliderInfo
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return a local straight-line cut frame with global arc coordinates.

        A petiole cut zone can span several articulated capsules.  Contact
        geometry must follow the particular capsule that the blade touched,
        while residual stub length must still be measured from the organ
        junction.  Extending the contacted segment axis backwards by its
        cumulative arc length gives both properties: projecting a point onto
        the returned line yields its cumulative stub coordinate, even after
        that segment bends relative to its neighbours.
        """
        if "petiole_cut_zone" not in collider.role:
            raise ValueError(f"collider is not a petiole cut zone: {collider.path}")
        chain = sorted(
            (link for link in self.links if link.organ == collider.organ),
            key=lambda link: link.index,
        )
        link = next(
            (candidate for candidate in chain if candidate.index == collider.segment),
            None,
        )
        if link is None:
            raise ValueError(f"cut-zone link is missing for collider: {collider.path}")
        axis = link.end - link.start
        length = float(np.linalg.norm(axis))
        if length <= 1e-12:
            raise ValueError(f"cut-zone link has zero length: {link.path}")
        axis = axis / length
        arc_start_m = float(
            sum(candidate.length for candidate in chain if candidate.index < link.index)
        )
        virtual_junction_m = link.start - arc_start_m * axis
        return virtual_junction_m, axis, arc_start_m


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


def _physical_collider_radius(
    link: Link, collider_radius_m: float | None = None
) -> float:
    """Clamp contact geometry without changing structural link properties."""
    requested = link.radius if collider_radius_m is None else collider_radius_m
    return min(float(requested), 0.5 * link.length)


def _define_capsule(
    stage: Usd.Stage,
    path: Sdf.Path,
    link: Link,
    properties: TissueProperties,
    *,
    visible: bool = False,
    collidable: bool = True,
    collider_radius_m: float | None = None,
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
    radius = _physical_collider_radius(link, collider_radius_m)
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


def _filter_cut_zones_from_plant(
    stage: Usd.Stage,
    collider_infos: list[ColliderInfo],
) -> None:
    """Keep flush blade zones physical without self-depenetrating the plant."""
    interaction_bodies = tuple(info.body_path for info in collider_infos)
    for info in collider_infos:
        if "cut_zone" not in info.role:
            continue
        cut_collider = stage.GetPrimAtPath(info.path)
        relation = UsdPhysics.FilteredPairsAPI.Apply(
            cut_collider
        ).CreateFilteredPairsRel()
        for body_path in interaction_bodies:
            if body_path != info.body_path:
                relation.AddTarget(Sdf.Path(body_path))


def _oriented_proxy_box(
    points: np.ndarray,
    *,
    minimum_thickness_m: float = FOLIAGE_CONTACT_MINIMUM_THICKNESS_M,
    padding_m: float = FOLIAGE_CONTACT_PADDING_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a padded, right-handed PCA box around one leaf blade."""
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 3:
        raise ValueError("foliage proxy points must have shape (N>=3, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("foliage proxy points must be finite")
    if minimum_thickness_m <= 0.0 or padding_m < 0.0:
        raise ValueError("foliage proxy dimensions must be positive")

    mean = vertices.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(vertices - mean, full_matrices=False)
    rotation = right_vectors.T
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    coordinates = (vertices - mean) @ rotation
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    local_centre = 0.5 * (minimum + maximum)
    half_extents = 0.5 * (maximum - minimum) + float(padding_m)
    half_extents = np.maximum(half_extents, 0.5 * float(minimum_thickness_m))
    centre = mean + rotation @ local_centre
    return centre, rotation, half_extents


def _gf_transform(rotation: np.ndarray, translation: np.ndarray) -> Gf.Matrix4d:
    values = np.asarray(rotation, dtype=np.float64)
    matrix = Gf.Matrix4d(1.0)
    # Gf matrices transform row vectors, whereas NumPy's columns above are the
    # box's local axes expressed in its parent frame.
    matrix.SetRotate(Gf.Matrix3d(*values.T.reshape(-1).tolist()))
    matrix.SetTranslateOnly(
        Gf.Vec3d(*np.asarray(translation, dtype=np.float64).tolist())
    )
    return matrix


def _labelled_stem_ancestor(
    plant: organs.Plant,
    organ_index: int,
    prefix: str,
) -> organs.Organ | None:
    index: int | None = int(organ_index)
    while index is not None:
        organ = plant.organs[index]
        if organ.tissue is organs.Tissue.STEM and organ.label.startswith(prefix):
            return organ
        index = organ.parent
    return None


def author_foliage_contact_proxies(
    stage: Usd.Stage,
    rig: PlantRig,
    plant: organs.Plant,
    to_stage_frame,
    *,
    visible: bool = False,
) -> tuple[ColliderInfo, ...]:
    """Add stable leaf-blade contact without enabling raw plant self-contact.

    Each visual foliage organ gets one thin oriented box on the same rigid body
    that carries its render mesh. Every box is filtered from every other plant
    body, so artistic rest-pose intersections cannot explode the vine, while
    external robot fingers, arms, and tools still receive normal PhysX contact.
    The proxy retains its owning ``SubStem_*`` identity so a two-finger pinch
    on a target leaflet can establish the same branch grasp as its petiole.
    """
    try:
        from pxr import PhysxSchema  # noqa: PLC0415
    except ImportError:  # Standalone USD tests do not load Kit's PhysX schema.
        PhysxSchema = None

    base_links: dict[int, Link] = {}
    for link in rig.links:
        current = base_links.get(link.organ)
        if current is None or link.index < current.index:
            base_links[link.organ] = link
    plant_body_paths = tuple(dict.fromkeys(link.path for link in rig.links))
    authored: list[ColliderInfo] = []

    for foliage in plant.organs:
        if foliage.tissue is not organs.Tissue.FOLIAGE:
            continue
        branch = _labelled_stem_ancestor(plant, foliage.index, "SubStem_")
        if branch is None:
            continue
        carrier_index: int | None = foliage.index
        while carrier_index is not None and carrier_index not in base_links:
            carrier_index = plant.organs[carrier_index].parent
        if carrier_index is None:
            continue
        carrier = base_links[carrier_index]
        body_prim = stage.GetPrimAtPath(carrier.path)
        body_world = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        body_inverse = body_world.GetInverse()
        world_vertices = np.asarray(
            to_stage_frame(foliage.component.vertices), dtype=np.float64
        )
        local_vertices = np.asarray(
            [
                body_inverse.Transform(Gf.Vec3d(*point.tolist()))
                for point in world_vertices
            ],
            dtype=np.float64,
        )
        centre, rotation, half_extents = _oriented_proxy_box(local_vertices)

        path = Sdf.Path(carrier.path).AppendChild(
            f"FoliageContact_{foliage.index:04d}"
        )
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(2.0)
        transformable = UsdGeom.Xformable(cube.GetPrim())
        transformable.AddTransformOp().Set(_gf_transform(rotation, centre))
        transformable.AddScaleOp().Set(Gf.Vec3f(*half_extents.tolist()))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if PhysxSchema is not None:
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            physx_collision.CreateContactOffsetAttr(0.003)
            physx_collision.CreateRestOffsetAttr(0.0005)
        if visible:
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.62, 0.20)])
            cube.CreateDisplayOpacityAttr([0.25])
        else:
            UsdGeom.Imageable(cube).CreatePurposeAttr(UsdGeom.Tokens.guide)

        prim = cube.GetPrim()
        prim.CreateAttribute(
            "tomato:organIndex", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(branch.index))
        prim.CreateAttribute(
            "tomato:sourceFoliageOrganIndex", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(foliage.index))
        prim.CreateAttribute(
            "tomato:organLabel", Sdf.ValueTypeNames.String, custom=True
        ).Set(branch.label)
        prim.CreateAttribute(
            "tomato:interactionRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("foliage_grasp")

        relation = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        for body_path in plant_body_paths:
            if body_path != carrier.path:
                relation.AddTarget(Sdf.Path(body_path))
        authored.append(
            ColliderInfo(
                path=str(path),
                body_path=carrier.path,
                organ=branch.index,
                organ_label=branch.label,
                segment=carrier.index,
                role="foliage_grasp",
            )
        )
    return tuple(authored)


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
    "interaction" (the default) enables main-stem and labelled SubStem petiole
    contact here; :func:`author_foliage_contact_proxies` adds robot-contactable
    leaf boxes after the render meshes are attached. The remaining structural
    capsules still carry mass, inertia, joints, and art but do not enter the
    contact solver. This matters because the source asset is an artistic rest
    pose with leaflets and trusses already intersecting; making every raw
    structural capsule collide asks PhysX to violently resolve a pose that is
    intentionally interpenetrating. "all" is retained only for diagnosing
    that raw collider set, and "none" isolates constraints.
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
    collider_infos: list[ColliderInfo] = []
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
        contact_segments: dict[int, str] = {}
        if collision_mode == "all":
            contact_segments.update(
                {segment: "generic_plant" for segment in range(points.shape[0] - 1)}
            )
        elif collision_mode == "interaction":
            arcs = centreline.arc_lengths()
            centres = 0.5 * (arcs[:-1] + arcs[1:])
            if organ.index == plant.root:
                # Ten separated main-stem contact zones on a two-metre vine.
                # Dense endpoint-touching capsules are exactly what invalidated
                # PhysX; physical spacing keeps the result independent of the
                # requested structural segment length.
                for target in np.arange(0.10, float(arcs[-1]), 0.20):
                    segment = int(np.argmin(np.abs(centres - target)))
                    contact_segments[segment] = "protected_main_stem"
            elif organ.label.startswith("SubStem_"):
                # Every segment intersecting the admissible stub interval is
                # physical blade contact. Previously only segment zero was
                # enabled (8.38 mm on SubStem_00), although the cut gate accepts
                # a realistic stub through 25 mm. A valid 12 mm path therefore
                # passed through no petiole collider and reached the main stem.
                cut_segments = set(_petiole_cut_zone_segments(arcs))
                for segment in cut_segments:
                    contact_segments[segment] = "petiole_cut_zone"
                grasp_segment = _petiole_grasp_segment(arcs)
                contact_segments[grasp_segment] = (
                    "petiole_cut_zone_grasp"
                    if grasp_segment in cut_segments
                    else "petiole_grasp"
                )

        for segment in range(points.shape[0] - 1):
            contactable = segment in contact_segments
            radius = float(max(0.5 * (centreline.radii[segment] + centreline.radii[segment + 1]), 1e-4))
            role = contact_segments.get(segment, "")
            # The first artistic petiole segment flares rapidly away from the
            # junction. Its average radius is appropriate for mass/stiffness,
            # but not for the physical cut interface: use the measured
            # proximal radius so the blade and guide encounter the tissue that
            # the force/work model represents.
            collider_radius = (
                float(max(centreline.radii[0], 1e-4))
                if "petiole_cut_zone" in role
                else None
            )
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
                collider_radius_m=collider_radius,
            )
            if contactable:
                collider_paths.append(collider)
                collider_prim = stage.GetPrimAtPath(collider)
                collider_prim.CreateAttribute(
                    "tomato:organIndex", Sdf.ValueTypeNames.Int, custom=True
                ).Set(int(organ.index))
                collider_prim.CreateAttribute(
                    "tomato:organLabel", Sdf.ValueTypeNames.String, custom=True
                ).Set(organ.label)
                collider_prim.CreateAttribute(
                    "tomato:interactionRole", Sdf.ValueTypeNames.String, custom=True
                ).Set(role)
                collider_infos.append(
                    ColliderInfo(
                        path=str(collider),
                        body_path=link.path,
                        organ=organ.index,
                        organ_label=organ.label,
                        segment=segment,
                        role=role,
                    )
                )
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
                cut_position_m=chain[0].start.copy(),
                cut_axis=(chain[0].end - chain[0].start)
                / max(float(np.linalg.norm(chain[0].end - chain[0].start)), 1e-12),
                # Cutting happens at the organ's first centreline sample, not
                # at the segment-average radius used by its contact capsule.
                # Artistic petiole meshes flare farther downstream; averaging
                # that flare overstated SubStem_00's cut diameter by ~2.6x and
                # therefore overstated the work required to sever it.
                cut_radius_m=float(
                    min(max(float(centreline.radii[0]), 1e-4), 0.5 * chain[0].length)
                ),
                cut_force_n=properties.cut_force_n,
            )

    # Flush cut-zone capsules begin exactly where a petiole enters the main
    # stem. The source is an artistic rest mesh, so those endpoint shapes can
    # also overlap an adjacent main-stem or sibling-petiole contact proxy by a
    # few millimetres. They must collide with the knife and gripper, but they
    # must never ask PhysX to depenetrate the plant from its own authored rest
    # pose. Filter each cut zone from every other *plant* interaction body;
    # external robot/tool contacts remain enabled.
    _filter_cut_zones_from_plant(stage, collider_infos)

    return PlantRig(
        root_path=root_path,
        links=links,
        joints=joints,
        cut_joints=cut_joints,
        junctions=junctions,
        collider_paths=tuple(str(path) for path in collider_paths),
        colliders=tuple(collider_infos),
        collision_mode=collision_mode,
    )


def _depth(plant: organs.Plant, index: int) -> int:
    depth, organ = 0, plant.organs[index]
    while organ.parent is not None:
        organ = plant.organs[organ.parent]
        depth += 1
    return depth


def _petiole_cut_zone_segments(
    arc_lengths: np.ndarray,
    maximum_stub_m: float = PETIOLE_CUT_ZONE_LENGTH_M,
) -> tuple[int, ...]:
    """Structural segments intersecting the benchmark's allowed stub zone."""
    arcs = np.asarray(arc_lengths, dtype=np.float64)
    if arcs.ndim != 1 or arcs.size < 2 or maximum_stub_m <= 0.0:
        return ()
    return tuple(
        int(index)
        for index in np.flatnonzero(arcs[:-1] < float(maximum_stub_m))
    )


def _petiole_grasp_segment(arc_lengths: np.ndarray) -> int:
    """Choose the first physical segment just distal to the cut zone."""
    arcs = np.asarray(arc_lengths, dtype=np.float64)
    if arcs.ndim != 1 or arcs.size < 2:
        raise ValueError("arc lengths must contain at least one segment")
    cut_segments = _petiole_cut_zone_segments(arcs)
    if not cut_segments:
        return 0
    return min(cut_segments[-1] + 1, len(arcs) - 2)


def _nearest_link(chain: list[Link], point: np.ndarray) -> Link | None:
    """Link of `chain` whose axis passes closest to `point`."""
    if not chain:
        return None
    return min(chain, key=lambda link: float(np.linalg.norm(0.5 * (link.start + link.end) - point)))


def author_safety_proxies(
    stage: Usd.Stage,
    plant: organs.Plant,
    root_path: str,
    skeletons: dict[int, skeleton_module.Skeleton],
    to_stage_frame,
    *,
    vine_name: str,
    visible: bool = False,
) -> tuple[SafetyColliderInfo, ...]:
    """Give neighbouring visual vines lightweight physical/safety colliders.

    These vines remain static references. Sparse main-stem and petiole capsules
    make collision avoidance observable without introducing hundreds of
    additional articulated bodies into each benchmark episode.
    """
    root = Sdf.Path(root_path)
    UsdGeom.Scope.Define(stage, root)
    authored: list[SafetyColliderInfo] = []
    for organ in plant.organs:
        centreline = skeletons.get(organ.index)
        if centreline is None:
            continue
        points = to_stage_frame(centreline.points)
        if points.shape[0] < 2:
            continue
        arcs = centreline.arc_lengths()
        centres = 0.5 * (arcs[:-1] + arcs[1:])
        if organ.index == plant.root:
            targets = np.arange(0.10, float(arcs[-1]), 0.20)
            segments = sorted(
                {int(np.argmin(np.abs(centres - target))) for target in targets}
            )
            role = "neighbour_main_stem"
        elif organ.label.startswith("SubStem_"):
            segments = [int(np.argmin(np.abs(centres - 0.5 * arcs[-1])))]
            role = "neighbour_petiole"
        else:
            continue

        organ_scope = root.AppendChild(f"Organ_{organ.index:04d}")
        UsdGeom.Scope.Define(stage, organ_scope)
        for segment in segments:
            start = points[segment]
            end = points[segment + 1]
            axis = end - start
            length = float(np.linalg.norm(axis))
            if length <= 0.0:
                continue
            fitted_radius = float(
                max(
                    0.5
                    * (
                        centreline.radii[segment]
                        + centreline.radii[segment + 1]
                    ),
                    1e-4,
                )
            )
            radius = min(fitted_radius, 0.5 * length)
            path = organ_scope.AppendChild(f"Proxy_{segment:03d}")
            capsule = UsdGeom.Capsule.Define(stage, path)
            capsule.CreateAxisAttr(UsdGeom.Tokens.z)
            capsule.CreateRadiusAttr(radius)
            capsule.CreateHeightAttr(max(length - 2.0 * radius, 1e-4))
            xform = UsdGeom.Xformable(capsule.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (start + end))))
            xform.AddOrientOp().Set(_orient_to(axis))
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            if visible:
                capsule.CreateDisplayColorAttr([Gf.Vec3f(0.65, 0.15, 0.10)])
            else:
                capsule.CreatePurposeAttr(UsdGeom.Tokens.guide)
            capsule.GetPrim().CreateAttribute(
                "tomato:vineName", Sdf.ValueTypeNames.String, custom=True
            ).Set(vine_name)
            capsule.GetPrim().CreateAttribute(
                "tomato:organLabel", Sdf.ValueTypeNames.String, custom=True
            ).Set(organ.label)
            capsule.GetPrim().CreateAttribute(
                "tomato:interactionRole", Sdf.ValueTypeNames.String, custom=True
            ).Set(role)
            authored.append(
                SafetyColliderInfo(
                    path=str(path),
                    vine_name=vine_name,
                    organ_label=organ.label,
                    role=role,
                )
            )
    return tuple(authored)


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
    filtered_paths: tuple[str, ...] = (),
) -> None:
    """A static floor, so severed organs land instead of falling forever.

    ``filtered_paths`` excludes synthetic catch trays from actors such as the
    robot.  The tray is bookkeeping for detached foliage, not greenhouse
    structure, and must not become an invisible obstacle to an approaching arm.
    """
    plane = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    plane.CreateSizeAttr(1.0)
    transformable = UsdGeom.Xformable(plane.GetPrim())
    transformable.AddTranslateOp().Set(
        Gf.Vec3d(float(centre_xy[0]), float(centre_xy[1]), height - 0.5 * size * 0.01)
    )
    transformable.AddScaleOp().Set(Gf.Vec3f(size, size, size * 0.01))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    if filtered_paths:
        filtered = UsdPhysics.FilteredPairsAPI.Apply(plane.GetPrim())
        relationship = filtered.CreateFilteredPairsRel()
        for filtered_path in filtered_paths:
            relationship.AddTarget(Sdf.Path(filtered_path))
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
