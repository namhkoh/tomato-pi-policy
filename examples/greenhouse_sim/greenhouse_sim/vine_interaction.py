"""Viewport pulling and low-speed airflow for articulated tomato vines.

The renderer, not the sparse robot-contact geometry, is authoritative for mouse
selection. A Shift-drag raycasts the visible GLB mesh, maps that mesh to its
supporting rigid body, and applies a bounded spring force at the clicked point.
This makes leaf blades and thin petioles usable without inflating collision
geometry or reintroducing rest-pose contact explosions.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import carb
import numpy as np
import omni.kit.raycast.query
from omni.kit.viewport.utility import get_active_viewport_window
from omni.physx import get_physx_simulation_interface
from omni.ui import scene as sc
from pxr import Gf
from pxr import PhysicsSchemaTools
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics
from pxr import UsdUtils


@dataclasses.dataclass
class _PullState:
    body: str
    local_anchor: Gf.Vec3d
    ndc_depth: float
    previous_anchor: np.ndarray
    peak_force_n: float = 0.0
    peak_target_offset_m: float = 0.0


class _PullGestureManager(sc.GestureManager):
    """Keep a live Shift-drag ahead of selection, but leave camera gestures alone."""

    def can_be_prevented(self, gesture):
        return gesture.state != sc.GestureState.CHANGED

    def should_prevent(self, gesture, preventer):
        if isinstance(preventer, _VisualPullGesture) and preventer.state in (
            sc.GestureState.BEGAN,
            sc.GestureState.CHANGED,
        ):
            try:
                from omni.kit.manipulator.camera.gesturebase import CameraGestureBase

                if isinstance(gesture, CameraGestureBase):
                    return False
            except ImportError:
                pass
            return True
        return super().should_prevent(gesture, preventer)


class _VisualPullGesture(sc.DragGesture):
    def __init__(self, owner: "VisualPull") -> None:
        super().__init__(
            mouse_button=0,
            modifiers=carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT,
            manager=_PullGestureManager(),
        )
        self._owner = owner

    def on_began(self) -> None:
        self._owner.begin(self.sender.gesture_payload.mouse)

    def on_changed(self) -> None:
        self._owner.update(self.sender.gesture_payload.mouse)

    def on_ended(self) -> None:
        self._owner.end()


class VisualPull:
    """Renderer-picked, bounded spring pull applied to a vine rigid body."""

    def __init__(
        self,
        stage: Usd.Stage,
        body_paths: list[str],
        report: dict,
        persist: Callable[[], None],
        status: Callable[[str], None],
        *,
        stiffness_n_m: float = 10.0,
        damping_n_s_m: float = 0.02,
        max_force_n: float = 1.0,
        physics_dt: float = 1.0 / 240.0,
    ) -> None:
        self._stage = stage
        self._body_paths = set(body_paths)
        self._report = report
        self._persist = persist
        self._status = status
        self._stiffness = max(float(stiffness_n_m), 0.0)
        self._damping = max(float(damping_n_s_m), 0.0)
        self._max_force = max(float(max_force_n), 0.0)
        self._dt = float(physics_dt)
        self._simulation = get_physx_simulation_interface()
        self._stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self._raycast = omni.kit.raycast.query.acquire_raycast_query_interface()
        self._request_id = 0
        self._request_active = False
        self._latest_ndc = (0.0, 0.0)
        self._active: _PullState | None = None
        self._viewport_window = get_active_viewport_window()
        if self._viewport_window is None:
            raise RuntimeError("no active viewport window for visual vine pulling")
        self._viewport = self._viewport_window.viewport_api
        self._frame = self._viewport_window.get_frame("tomato_vine_visual_pull")
        self._frame.visible = True
        with self._frame:
            self._scene_view = sc.SceneView()
            self._viewport.add_scene_view(self._scene_view)
            with self._scene_view.scene:
                self._screen = sc.Screen(gesture=_VisualPullGesture(self))

        report["mouse_interaction"] = {
            "enabled": True,
            "backend": "visible-mesh-raycast-force",
            "gesture": "Shift + left-drag",
            "stiffness_n_m": self._stiffness,
            "damping_n_s_m": self._damping,
            "max_force_n": self._max_force,
            "pickable_bodies": len(self._body_paths),
        }

    @property
    def active_body(self) -> str | None:
        return None if self._active is None else self._active.body

    def project_world(self, position) -> tuple[float, float, float]:
        projected = self._viewport.world_to_ndc.Transform(Gf.Vec3d(*position))
        return float(projected[0]), float(projected[1]), float(projected[2])

    def begin(self, ndc) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._request_active = True
        self._latest_ndc = (float(ndc[0]), float(ndc[1]))
        self._active = None

        near = self._viewport.ndc_to_world.Transform(
            Gf.Vec3d(self._latest_ndc[0], self._latest_ndc[1], -1.0)
        )
        far = self._viewport.ndc_to_world.Transform(
            Gf.Vec3d(self._latest_ndc[0], self._latest_ndc[1], 1.0)
        )
        direction = (far - near).GetNormalized()
        ray = omni.kit.raycast.query.Ray(
            carb.Float3(float(near[0]), float(near[1]), float(near[2])),
            carb.Float3(float(direction[0]), float(direction[1]), float(direction[2])),
        )

        def completed(_ray, result, *args, **kwargs) -> None:
            del args, kwargs
            if request_id != self._request_id or not self._request_active:
                return
            if not result.valid:
                self._status("Pull missed: click visible geometry on the dynamic vine")
                return
            hit_path = str(result.get_target_usd_path())
            body = self._body_ancestor(hit_path)
            if body is None:
                self._status("That object is static; Shift-drag the focused physics vine")
                return
            hit = Gf.Vec3d(*result.hit_position)
            matrix = UsdGeom.Xformable(self._stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            local_anchor = matrix.GetInverse().Transform(hit)
            depth = float(self._viewport.world_to_ndc.Transform(hit)[2])
            anchor = np.asarray(hit, dtype=np.float64)
            self._active = _PullState(body, local_anchor, depth, anchor)
            self._report["visual_mouse_grabs"] = int(
                self._report.get("visual_mouse_grabs", 0)
            ) + 1
            self._report["last_grab"] = {"visual": hit_path, "body": body}
            self._persist()
            self._status(f"Pulling {Sdf.Path(body).GetParentPath().name}")

        self._raycast.submit_raycast_query(ray, completed)

    def update(self, ndc) -> None:
        if self._request_active:
            self._latest_ndc = (float(ndc[0]), float(ndc[1]))

    def end(self) -> None:
        self._request_id += 1
        self._request_active = False
        state, self._active = self._active, None
        if state is None:
            return
        self._report["visual_mouse_releases"] = int(
            self._report.get("visual_mouse_releases", 0)
        ) + 1
        self._report["last_pull"] = {
            "body": state.body,
            "peak_force_n": state.peak_force_n,
            "peak_target_offset_mm": state.peak_target_offset_m * 1000.0,
        }
        self._persist()
        self._status(
            f"Released: peak {state.peak_force_n:.3f} N, "
            f"{state.peak_target_offset_m * 1000.0:.1f} mm target offset"
        )

    def step(self) -> None:
        state = self._active
        if state is None:
            return
        prim = self._stage.GetPrimAtPath(state.body)
        if not prim.IsValid():
            self.end()
            return
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        anchor_gf = matrix.Transform(state.local_anchor)
        anchor = np.asarray(anchor_gf, dtype=np.float64)
        target_gf = self._viewport.ndc_to_world.Transform(
            Gf.Vec3d(self._latest_ndc[0], self._latest_ndc[1], state.ndc_depth)
        )
        target = np.asarray(target_gf, dtype=np.float64)
        delta = target - anchor
        velocity = (anchor - state.previous_anchor) / self._dt
        force = self._stiffness * delta - self._damping * velocity
        magnitude = float(np.linalg.norm(force))
        if not np.isfinite(force).all() or magnitude <= 0.0:
            state.previous_anchor = anchor
            return
        if magnitude > self._max_force > 0.0:
            force *= self._max_force / magnitude
            magnitude = self._max_force
        body_id = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(state.body))
        self._simulation.apply_force_at_pos(
            self._stage_id,
            body_id,
            carb.Float3(float(force[0]), float(force[1]), float(force[2])),
            carb.Float3(float(anchor[0]), float(anchor[1]), float(anchor[2])),
            "Force",
        )
        state.previous_anchor = anchor
        state.peak_force_n = max(state.peak_force_n, magnitude)
        state.peak_target_offset_m = max(
            state.peak_target_offset_m, float(np.linalg.norm(delta))
        )

    def close(self) -> None:
        self.end()
        if self._viewport is not None and self._scene_view is not None:
            self._viewport.remove_scene_view(self._scene_view)
        if self._frame is not None:
            self._frame.visible = False
        self._screen = None
        self._scene_view = None
        self._viewport = None
        self._viewport_window = None

    def _body_ancestor(self, hit_path: str) -> str | None:
        path = Sdf.Path(hit_path)
        while path != Sdf.Path.absoluteRootPath:
            candidate = str(path)
            if candidate in self._body_paths:
                return candidate
            path = path.GetParentPath()
        return None


@dataclasses.dataclass(frozen=True)
class _AirflowTarget:
    body: str
    local_centre: Gf.Vec3d
    area_m2: float
    phase: float


class Airflow:
    """Deterministic aerodynamic load on the measured foliage of each petiole."""

    AIR_DENSITY_KG_M3 = 1.204
    LEAF_DRAG_COEFFICIENT = 1.0

    def __init__(
        self,
        stage: Usd.Stage,
        runtimes: list,
        *,
        speed_m_s: float,
        frequency_hz: float = 0.18,
        direction_deg: float = 20.0,
        physics_dt: float = 1.0 / 240.0,
    ) -> None:
        self._stage = stage
        self._speed = max(float(speed_m_s), 0.0)
        self._frequency = max(float(frequency_hz), 0.0)
        self._direction = math.radians(float(direction_deg))
        self._dt = float(physics_dt)
        self._time = 0.0
        self._simulation = get_physx_simulation_interface()
        self._stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self._targets = self._build_targets(runtimes)

    @property
    def summary(self) -> dict:
        return {
            "enabled": self._speed > 0.0,
            "speed_m_s": self._speed,
            "frequency_hz": self._frequency,
            "targets": len(self._targets),
            "foliage_area_m2": float(sum(target.area_m2 for target in self._targets)),
            "model": "F=0.5*rho*Cd*A*v^2 at foliage centroid",
        }

    @property
    def body_paths(self) -> list[str]:
        return [target.body for target in self._targets]

    def step(self) -> None:
        self._time += self._dt
        if self._speed <= 0.0:
            return
        for target in self._targets:
            gust = 1.0 + 0.30 * math.sin(
                2.0 * math.pi * self._frequency * self._time + target.phase
            )
            gust += 0.10 * math.sin(
                2.0 * math.pi * self._frequency * 2.37 * self._time + 1.7 * target.phase
            )
            speed = max(self._speed * gust, 0.0)
            magnitude = (
                0.5
                * self.AIR_DENSITY_KG_M3
                * self.LEAF_DRAG_COEFFICIENT
                * target.area_m2
                * speed**2
            )
            angle = self._direction + math.radians(8.0) * math.sin(
                2.0 * math.pi * self._frequency * 0.61 * self._time + target.phase
            )
            force = (magnitude * math.cos(angle), magnitude * math.sin(angle), 0.0)
            prim = self._stage.GetPrimAtPath(target.body)
            if not prim.IsValid():
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            centre = matrix.Transform(target.local_centre)
            self._simulation.apply_force_at_pos(
                self._stage_id,
                PhysicsSchemaTools.sdfPathToInt(Sdf.Path(target.body)),
                carb.Float3(*force),
                carb.Float3(float(centre[0]), float(centre[1]), float(centre[2])),
                "Force",
            )

    def _build_targets(self, runtimes: list) -> list[_AirflowTarget]:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        targets = []
        count = 0
        for runtime in runtimes:
            visual_paths = {}
            root = self._stage.GetPrimAtPath(runtime.root_path)
            for prim in Usd.PrimRange(root):
                name = prim.GetName()
                if name.startswith("Visual_"):
                    try:
                        visual_paths[int(name.split("_", 1)[1])] = prim.GetPath()
                    except ValueError:
                        continue
            for organ in runtime.plant.organs:
                if not organ.label.startswith("SubStem_"):
                    continue
                links = [link for link in runtime.rig.links if link.organ == organ.index]
                if not links:
                    continue
                body = min(links, key=lambda link: link.index).path
                weighted = Gf.Vec3d(0.0)
                area_sum = 0.0
                for index in runtime.plant.descendants_of(organ.index):
                    descendant = runtime.plant.organs[index]
                    if descendant.tissue.value != "foliage":
                        continue
                    area = _surface_area(descendant.component)
                    path = visual_paths.get(index)
                    if area <= 0.0 or path is None:
                        continue
                    centroid = cache.ComputeWorldBound(self._stage.GetPrimAtPath(path)).ComputeCentroid()
                    weighted += area * centroid
                    area_sum += area
                if area_sum <= 0.0:
                    continue
                world_centre = weighted / area_sum
                matrix = UsdGeom.Xformable(self._stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                local_centre = matrix.GetInverse().Transform(world_centre)
                targets.append(
                    _AirflowTarget(body, local_centre, area_sum, count * 2.399963229728653)
                )
                count += 1
        return targets


def _surface_area(component) -> float:
    triangles = component.triangles
    vertices = component.vertices
    if triangles.size == 0:
        return 0.0
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(second - first, third - first), axis=1).sum())