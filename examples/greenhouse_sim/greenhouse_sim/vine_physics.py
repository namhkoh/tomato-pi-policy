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

    Young's modulus for herbaceous tomato stem tissue is orders of magnitude
    below the ~7 GPa of the green wood used for orchard-tree simulation, which
    is why stiffness is parameterised rather than borrowed.
    """

    youngs_modulus_pa: float = 2.0e7
    density_kg_m3: float = 900.0
    damping_ratio: float = 0.1

    # Severance thresholds, from measured tomato leaf-pruning forces: a blade
    # shears the petiole at ~66 N, while pulling detaches it at ~33 N. Keeping
    # them separate is what lets the benchmark score a clean cut apart from a
    # tear, which is an agronomically real distinction.
    cut_force_n: float = 66.3
    tear_force_n: float = 32.5


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


def _define_capsule(stage: Usd.Stage, path: Sdf.Path, link: Link, properties: TissueProperties) -> None:
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateAxisAttr(UsdGeom.Tokens.z)
    capsule.CreateRadiusAttr(float(link.radius))
    capsule.CreateHeightAttr(max(link.length - 2.0 * link.radius, 1e-4))
    # Collision geometry only; the render meshes stay in the visual asset.
    UsdGeom.Imageable(capsule).CreatePurposeAttr(UsdGeom.Tokens.guide)

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

    if breakable:
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
            _define_capsule(stage, Sdf.Path(link.path), link, properties)
            chain.append(link)
            links.append(link)

        # Internal joints hold the organ together along its own length.
        for segment in range(1, len(chain)):
            stiffness = beam_stiffness(
                properties.youngs_modulus_pa, chain[segment].radius, max(chain[segment].length, 1e-4)
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
        base_stiffness = beam_stiffness(properties.youngs_modulus_pa, chain[0].radius, max(chain[0].length, 1e-4))
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
