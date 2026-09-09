"""Exact cached broad-phase screen for immutable dataset scenes.

Uses the union-box distance as a LOWER bound to omit provably distant pairs.
Reports the same minimum, overlaps and refinement counts as the reference
screen; this is an acceleration, not a relaxed collision criterion.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


class StaticBoundScreen:
    def __init__(self,obstacles,refinement=None):
        if not obstacles: raise ValueError('Nonempty obstacle geometry required')
        self.obstacles=deepcopy(obstacles)
        self.lower=np.array([o['min'] for o in obstacles],float)
        self.upper=np.array([o['max'] for o in obstacles],float)
        if not np.isfinite([self.lower,self.upper]).all() or np.any(self.upper<self.lower):
            raise ValueError('Invalid obstacle bounds')
        self.refinement=refinement

    def __call__(self,robot_bounds,margin_m=.01):
        if not robot_bounds or not np.isfinite(margin_m) or margin_m<0:
            raise ValueError('Nonempty robot geometry and valid margin required')
        low=np.array([p['min'] for p in robot_bounds],float)
        high=np.array([p['max'] for p in robot_bounds],float)
        if not np.isfinite([low,high]).all() or np.any(high<low): raise ValueError('Invalid robot bounds')
        union_gap=np.maximum(self.lower-high.max(axis=0),low.min(axis=0)-self.upper)
        lower_bound=np.linalg.norm(np.maximum(union_gap,0),axis=1)
        minimum=float('inf'); overlaps=[]; broad=cleared=0
        for part,pmin,pmax in zip(robot_bounds,low,high):
            indices=np.flatnonzero(lower_bound<=max(minimum,margin_m)+1e-12)
            if not len(indices): continue
            gap=np.maximum(self.lower[indices]-pmax,pmin-self.upper[indices])
            separation=np.linalg.norm(np.maximum(gap,0),axis=1)
            minimum=min(minimum,float(separation.min()))
            for local in np.flatnonzero(separation<=margin_m):
                i=int(indices[local]); broad+=1
                if self.refinement is not None and not self.refinement(part,self.obstacles[i],margin_m):
                    cleared+=1; continue
                overlaps.append(dict(robot_path=part['path'],scene_path=self.obstacles[i]['path'],
                    aabb_separation_m=float(separation[local]),possible_overlap_not_exact_collision=True))
        return dict(method='world_aabbs_then_scene_triangles_vs_robot_local_bounds' if self.refinement else
                    'all_visible_boundable_world_aabbs_with_instance_proxies',
            passed=not overlaps,minimum_aabb_separation_m=minimum,margin_m=margin_m,
            possible_overlap_count=len(overlaps),possible_overlaps=overlaps[:30],robot_bound_count=len(robot_bounds),
            scene_bound_count=len(self.obstacles),broadphase_pair_count=broad,pairs_cleared_by_triangle_refinement=cleared,
            floor_checked_separately=True,collision_free_certified=False,
            not_checked=['robot self collision','path between snapshots','dynamics and contact',
                         'invisible collision-only shapes','containment inside closed plant volumes'])
