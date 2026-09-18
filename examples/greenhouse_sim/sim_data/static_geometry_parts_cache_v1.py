"""Exact per-bound geometry reuse for unchanged robot geometry and world poses.

Every call reads each visible robot boundable's actual world transform. A changed
transform recomputes that part's bounds and collision screen, including the head.
All non-transform robot changes invalidate every part; scene changes invalidate
obstacles too. Callers retain source hashes, floor checks, and capture guards.
"""
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from pxr import Tf, Usd, UsdGeom
from .capture_viewpoints import static_obstacles, visible_bounds, scene_triangle_refiner
from .static_geometry_cache import under
from .training_screen import StaticBoundScreen


class StaticGeometryPartsCache:
    def __init__(self, stage, robot_root, *, include_generated_plants=False, maximum_poses_per_part=16):
        if not stage or not stage.GetPrimAtPath(robot_root):
            raise ValueError('Existing stage and robot root required')
        if type(include_generated_plants) is not bool:
            raise ValueError('Explicit generated-plant refinement boolean required')
        if type(maximum_poses_per_part) is not int or maximum_poses_per_part < 1:
            raise ValueError('Positive bounded per-part cache capacity required')
        self.stage = stage
        self.robot_root = str(robot_root)
        self.include_generated_plants = include_generated_plants
        self.capacity = maximum_poses_per_part
        self._screen = None
        self._paths = None
        self._memo = {}
        self._revision = 0
        self._closed = False
        self.builds = self.hits = self.invalidations = 0
        self.bounds_computed = self.part_screen_computed = self.part_hits = 0
        self._notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._changed, stage)

    def _clear_parts(self):
        self._paths = None
        self._memo.clear()

    def _changed(self, notice, sender):
        resynced = list(notice.GetResyncedPaths())
        changed = list(notice.GetChangedInfoOnlyPaths())
        events = [(p, True) for p in resynced] + [(p, False) for p in changed]
        events = [(p, resync) for p, resync in events
                  if not any(under(str(p), root) for root in ('/Render', '/Replicator', '/Orchestrator'))]
        if not events:
            return
        self._revision += 1
        for path, resync in events:
            text = str(path)
            if not under(text, self.robot_root):
                self._screen = None
                self._clear_parts()
                self.invalidations += 1
                continue
            # Only authored transform values/order can retain immutable geometry.
            attr = path.name if path.IsPropertyPath() else ''
            transform_only = not resync and (attr == 'xformOpOrder' or attr.startswith('xformOp:'))
            if not transform_only:
                self._clear_parts()
                continue
            # A boundable can unusually contain another boundable. Its aggregate
            # bound may change when a descendant moves without its own matrix changing.
            prim_path = str(path.GetPrimPath())
            if self._paths and any(prim_path.startswith(p + '/') for p in self._paths):
                self._clear_parts()

    @staticmethod
    def _matrix_key(matrix):
        return tuple(float(value) for row in matrix for value in row)

    def _one_bound(self, path, matrix, bbox):
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Robot prim disappeared without cache invalidation')
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == 'invisible' or imageable.ComputePurpose() not in ('default', 'render'):
            raise ValueError('Robot visibility changed without cache invalidation')
        bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError('Robot bounds became empty without cache invalidation')
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.isfinite([low, high]).all():
            raise ValueError('Non-finite visible robot bounds')
        local = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
        self.bounds_computed += 1
        return dict(path=path, min=low.tolist(), max=high.tolist(),
                    local_min=list(local.GetMin()), local_max=list(local.GetMax()),
                    local_to_world_row=[list(row) for row in matrix])

    def __call__(self, margin_m=.01):
        if self._closed:
            raise ValueError('Closed geometry cache')
        if not np.isfinite(margin_m) or margin_m < 0:
            raise ValueError('Valid nonnegative margin required')
        revision = self._revision
        obstacle_hit = self._screen is not None
        if not obstacle_hit:
            self._screen = StaticBoundScreen(static_obstacles(self.stage, self.robot_root),
                scene_triangle_refiner(self.stage, include_generated_plants=self.include_generated_plants))
            self._clear_parts()
        initial = None
        if self._paths is None:
            initial = {r['path']: r for r in visible_bounds(self.stage, self.robot_root)}
            self._paths = list(initial)
            self.bounds_computed += len(initial)
        transforms = UsdGeom.XformCache()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        parts = []
        for path in self._paths:
            prim = self.stage.GetPrimAtPath(path)
            matrix = transforms.GetLocalToWorldTransform(prim)
            key = (float(margin_m), self._matrix_key(matrix))
            memo = self._memo.setdefault(path, OrderedDict())
            if key in memo:
                result = memo.pop(key)
                memo[key] = result
                self.part_hits += 1
            else:
                bounds = initial[path] if initial is not None else self._one_bound(path, matrix, bbox)
                result = self._screen([bounds], margin_m)
                memo[key] = result
                while len(memo) > self.capacity:
                    memo.popitem(last=False)
                self.part_screen_computed += 1
            parts.append(result)
        # Each part uses the unchanged exact screen. Counts add, minimum takes
        # min, and ordered concatenation preserves the original first-30 limit.
        result = deepcopy(parts[0])
        result.update(passed=all(p['passed'] for p in parts),
            minimum_aabb_separation_m=min(p['minimum_aabb_separation_m'] for p in parts),
            possible_overlap_count=sum(p['possible_overlap_count'] for p in parts),
            possible_overlaps=[o for p in parts for o in p['possible_overlaps']][:30],
            robot_bound_count=len(parts),
            broadphase_pair_count=sum(p['broadphase_pair_count'] for p in parts),
            pairs_cleared_by_triangle_refinement=sum(p['pairs_cleared_by_triangle_refinement'] for p in parts))
        if revision != self._revision:
            self._screen = None
            self._clear_parts()
            raise ValueError('Scene changed while screening; no cached result can be accepted')
        if obstacle_hit:
            self.hits += 1
        else:
            self.builds += 1
        return result

    def diagnostics(self):
        return dict(obstacle_cache_builds=self.builds, obstacle_cache_hits=self.hits,
                    invalidating_notices=self.invalidations, scene_revision=self._revision,
                    every_robot_world_transform_checked_per_call=True,
                    changed_head_camera_body_bounds_recomputed=True,
                    bounds_computed=self.bounds_computed, part_screens_computed=self.part_screen_computed,
                    unchanged_part_hits=self.part_hits, maximum_cached_poses_per_part=self.capacity,
                    disk_source_hash_validation_is_callers_responsibility=True)

    def close(self):
        if not self._closed:
            self._notice.Revoke()
            self._screen = None
            self._clear_parts()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
