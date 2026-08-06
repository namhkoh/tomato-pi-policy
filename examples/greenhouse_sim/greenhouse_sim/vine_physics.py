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

`physics:breakForce` is silently ignored on articulation joints, so the plant is
deliberately built from maximal-coordinate bodies with no articulation root
above them. A vine authored inside an articulation would simply never tear, with
no error to explain why.
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
from pxr import PhysxSchema  # noqa: E402
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
class PlantRig:
    """The simulable form of one plant."""

    root_path: str
    links: list[Link]
    joints: dict[str, str]  # joint prim path -> child link path
    cut_joints: dict[str, str]  # organ label -> joint prim that severs it

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
    stage: Usd.Stage, path: Sdf.Path, link: Link, properties: TissueProperties, *, visible: bool = False
) -> None:
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateAxisAttr(UsdGeom.Tokens.z)
    capsule.CreateRadiusAttr(float(link.radius))
    capsule.CreateHeightAttr(max(link.length - 2.0 * link.radius, 1e-4))
    # Normally collision-only, with the render meshes living in the visual
    # asset. Made visible when the point is to see and grab the physics itself.
    if not visible:
        UsdGeom.Imageable(capsule).CreatePurposeAttr(UsdGeom.Tokens.guide)
    else:
        capsule.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.42, 0.14)])

    centre = 0.5 * (link.start + link.end)
    transformable = UsdGeom.Xformable(capsule.GetPrim())
    transformable.AddTranslateOp().Set(Gf.Vec3d(*centre))
    transformable.AddOrientOp().Set(_orient_to(link.end - link.start))

    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(capsule.GetPrim())
    mass_api.CreateMassAttr(capsule_mass(link.radius, link.length, properties.density_kg_m3))


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
) -> None:
    """A compliant 3-DOF rotational joint anchored at an organ junction."""
    joint = UsdPhysics.Joint.Define(stage, path)
    if parent is not None:
        joint.CreateBody0Rel().SetTargets([parent])
    joint.CreateBody1Rel().SetTargets([child])

    for body, target in ((parent, "LocalPos0"), (child, "LocalPos1")):
        if body is None:
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor))
            continue
        prim = stage.GetPrimAtPath(body)
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local = world.GetInverse().Transform(Gf.Vec3d(*anchor))
        getattr(joint, f"Create{target}Attr")().Set(Gf.Vec3f(local))

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
        # USD angular drives are per-degree; beam stiffness is per-radian.
        drive.CreateStiffnessAttr(float(stiffness * math.pi / 180.0))
        drive.CreateDampingAttr(float(damping * math.pi / 180.0))

    if breakable and properties.tear_force_n > 0.0:
        joint.CreateBreakForceAttr(properties.tear_force_n)
        joint.CreateBreakTorqueAttr(properties.tear_force_n)
    # Kept out of any articulation so breakForce is honoured at all.
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
) -> PlantRig:
    """Build capsule chains and joints for one plant under `root_path`.

    `to_stage_frame` converts organ-space points into the stage's frame, so the
    caller owns the coordinate convention rather than this module guessing it.
    """
    properties = properties or TissueProperties()
    scope = Sdf.Path(root_path).AppendChild("Physics")
    UsdGeom.Scope.Define(stage, scope)

    links: list[Link] = []
    joints: dict[str, str] = {}
    cut_joints: dict[str, str] = {}
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
        UsdGeom.Scope.Define(stage, organ_scope)
        chain: list[Link] = []

        for segment in range(points.shape[0] - 1):
            radius = float(max(0.5 * (centreline.radii[segment] + centreline.radii[segment + 1]), 1e-4))
            link = Link(
                path=str(organ_scope.AppendChild(f"Link_{segment:03d}")),
                organ=organ.index,
                index=segment,
                start=points[segment],
                end=points[segment + 1],
                radius=radius,
            )
            _define_capsule(stage, Sdf.Path(link.path), link, properties, visible=visible_colliders)
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
        )
        joints[str(base_path)] = chain[0].path
        if parent_link is not None:
            cut_joints[organ.label] = str(base_path)
        else:
            # The main stem is rooted: its base joint anchors it to the world.
            UsdPhysics.FixedJoint(stage.GetPrimAtPath(base_path))

    return PlantRig(root_path=root_path, links=links, joints=joints, cut_joints=cut_joints)


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
    stage: Usd.Stage, path: str = "/World/GroundPlane", *, height: float = 0.0, size: float = 20.0
) -> None:
    """A static floor, so severed organs land instead of falling forever."""
    plane = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    plane.CreateSizeAttr(1.0)
    transformable = UsdGeom.Xformable(plane.GetPrim())
    transformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height - 0.5 * size * 0.01))
    transformable.AddScaleOp().Set(Gf.Vec3f(size, size, size * 0.01))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdGeom.Imageable(plane).CreatePurposeAttr(UsdGeom.Tokens.guide)


def apply_scene_physics(stage: Usd.Stage, path: str = "/World/PhysicsScene", *, gravity: float = 9.81) -> None:
    """Ensure the stage has a physics scene configured for thin compliant bodies."""
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(path))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(gravity)

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(path))
    # Thin, stiff, strongly driven chains need iterations rather than a smaller
    # step to stay stable; too few and a petiole visibly buzzes at rest.
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateEnableCCDAttr(defaultValue=True)
