"""Notice-invalidated reuse of static obstacle bounds and triangle refiners.

Only the obstacle side is cached. Robot bounds and the exact screen are computed
on every call. All non-bookkeeping USD changes invalidate the snapshot; robot-
subtree changes retain obstacles but still invalidate an in-progress result.
Callers must retain source-file hashes, floor/camera checks and capture guards.
"""
import time
from pxr import Tf, Usd
from .capture_viewpoints import static_obstacles, visible_bounds, scene_triangle_refiner
from .training_screen import StaticBoundScreen


def under(path, root):
    return path == root or path.startswith(root+'/') or path.startswith(root+'.')


def profile_against_reference(cache, reference):
    timings=[]
    for _ in range(2):
        started=time.perf_counter()
        observed=cache(reference['margin_m'])
        timings.append(time.perf_counter()-started)
        if observed!=reference:
            raise ValueError('Cached geometry screen differs from full reference')
    return dict(first_call_seconds=timings[0],second_call_seconds=timings[1],
                both_results_equal_full_reference=True,cache=cache.diagnostics(),
                rendering_or_screen_criteria_changed=False)


class StaticGeometryScreenCache:
    def __init__(self, stage, robot_root, *, include_generated_plants=False):
        if not stage or not stage.GetPrimAtPath(robot_root):
            raise ValueError('Existing stage and robot root required')
        if type(include_generated_plants) is not bool:
            raise ValueError('Explicit generated-plant refinement boolean required')
        self.stage=stage
        self.robot_root=str(robot_root)
        self.include_generated_plants=include_generated_plants
        self._screen=None
        self._revision=0
        self._closed=False
        self.builds=0
        self.hits=0
        self.invalidations=0
        self._notice=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._changed,stage)

    def _changed(self, notice, sender):
        paths={str(p) for p in [*notice.GetResyncedPaths(),*notice.GetChangedInfoOnlyPaths()]}
        # These roots cannot contribute to visible_bounds(stage, '/World').
        # No /World path, including a material or ancestor transform, is ignored.
        paths={p for p in paths if not any(under(p,r) for r in ('/Render','/Replicator','/Orchestrator'))}
        if not paths:
            return
        self._revision+=1
        if any(not under(p,self.robot_root) for p in paths):
            self._screen=None
            self.invalidations+=1

    def __call__(self, margin_m=.01):
        if self._closed:
            raise ValueError('Closed geometry cache')
        revision=self._revision
        hit=self._screen is not None
        if not hit:
            obstacles=static_obstacles(self.stage,self.robot_root)
            refinement=scene_triangle_refiner(self.stage,include_generated_plants=self.include_generated_plants)
            screen=StaticBoundScreen(obstacles,refinement)
        else:
            screen=self._screen
        robot=visible_bounds(self.stage,self.robot_root)
        result=screen(robot,margin_m)
        if revision!=self._revision:
            self._screen=None
            raise ValueError('Scene changed while screening; no cached result can be accepted')
        self._screen=screen
        if hit:self.hits+=1
        else:self.builds+=1
        return result

    def diagnostics(self):
        return dict(obstacle_cache_builds=self.builds,obstacle_cache_hits=self.hits,
                    invalidating_notices=self.invalidations,scene_revision=self._revision,
                    robot_bounds_recomputed_per_call=True,cached_screen_result=False,
                    disk_source_hash_validation_is_callers_responsibility=True)

    def close(self):
        if not self._closed:
            self._notice.Revoke()
            self._screen=None
            self._closed=True

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
