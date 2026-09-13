"""Frozen native-scene whole-robot startup geometry, NOT motion authority.

No actor, filter, USD value, timeline state or physics step is changed. Every
source collider must pass a positive native control before AND after screening.
Robot capsules use gap-free enclosing sphere unions; boxes retain full margin.
The caller must load the owned scene separately and supply a live epoch guard.
This is not a grasp, path, native robot-pose tracking or cutting certificate.
"""
import time
import numpy as np
from .capsule_sphere_cover import cover


def _path(value):
    return (isinstance(value,str) and value.startswith('/') and value!='/'
        and all(p not in ('','.','..') for p in value[1:].split('/')))


class NativeStartupClearance:
    def __init__(self,query_box,query_sphere,scene_bounds,robot_paths,*,guard,
                 floor_root=None,max_queries=20000,wall_limit_s=60.,memoize_queries=False):
        if (not all(callable(f) for f in (query_box,query_sphere,guard))
                or type(memoize_queries) is not bool
                or type(max_queries) is not int or not 1<=max_queries<=20000
                or isinstance(wall_limit_s,(bool,np.bool_)) or not np.isfinite(wall_limit_s)
                or not 0<wall_limit_s<=180):
            raise ValueError('Explicit native queries, epoch guard and bounded budget required')
        paths=list(robot_paths)
        if (not paths or len(set(paths))!=len(paths) or not all(_path(p) for p in paths)
                or not scene_bounds or len(scene_bounds)>2000
                or not all(_path(p) for p in scene_bounds) or set(paths)&set(scene_bounds)
                or floor_root is not None and not _path(floor_root)):
            raise ValueError('Disjoint, complete scene/robot collider inventories required')
        self.robot_paths=set(paths);self.scene={}
        for path,(low,high) in scene_bounds.items():
            low,high=np.asarray(low,float),np.asarray(high,float)
            if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or np.any(high<low):
                raise ValueError('Finite ordered scene bounds required')
            self.scene[path]=(low.copy(),high.copy())
        self.box_query=query_box;self.sphere_query=query_sphere;self.guard=guard
        self.floor_root=floor_root;self.max_queries=max_queries;self.wall_limit_s=wall_limit_s
        self.started=time.perf_counter();self.calls=0;self.active=True
        self.covered=set();self.final_coverage=set();self.errors=[];self.checks=0;self.cleared=0
        self.validated=False
        self.memoize_queries=memoize_queries;self.query_cache={};self.cache_hits=0
        self._coverage(self.covered)

    def _check(self,request=False):
        if not self.active:raise RuntimeError('Native startup query invalid or closed')
        try:
            self.guard()
            if self.calls>self.max_queries or request and self.calls>=self.max_queries or time.perf_counter()-self.started>=self.wall_limit_s:
                raise RuntimeError('Native startup query budget exhausted')
        except Exception as exc:
            self.query_cache.clear()
            self.active=False;self.validated=False;self.errors.append(str(exc));raise

    def _query(self,function,*args,memoize=False):
        self._check()
        try:
            # Exact native-query arguments only, no pose rounding or spatial
            # extrapolation. Reuse is scoped to this guarded frozen epoch.
            # Positive actor controls never set memoize and always query again.
            key=None
            if memoize and self.memoize_queries:
                key=(function,tuple((np.asarray(a,dtype=float).shape,
                     np.asarray(a,dtype=float).tobytes()) for a in args))
                try:hash(key)
                except TypeError:key=None  # Unhashable callable: native fallback.
                if key is not None and key in self.query_cache:
                    self._check();self.cache_hits+=1
                    return set(self.query_cache[key])
            self._check(request=True);self.calls+=1
            hits=function(*args);self._check()
            if (not isinstance(hits,(list,tuple)) or not all(_path(p) for p in hits)
                    or set(hits)-self.robot_paths-set(self.scene)):
                raise RuntimeError('Incomplete or uninventoried native collider response')
            hits=set(hits)
            if key is not None and len(self.query_cache)<4096:
                self.query_cache[key]=frozenset(hits)
            return hits
        except Exception as exc:
            self.query_cache.clear()
            self.active=False;self.validated=False;self.errors.append(str(exc));raise

    def _box(self,centre,axes,half,*,memoize=False):
        centre,axes,half=np.asarray(centre,float),np.asarray(axes,float),np.asarray(half,float)
        if (centre.shape!=(3,) or axes.shape!=(3,3) or half.shape!=(3,)
                or not np.isfinite(np.r_[centre,axes.flat,half]).all() or np.any(half<=0)
                or not np.allclose(axes.T@axes,np.eye(3),atol=1e-8,rtol=0) or np.linalg.det(axes)<0):
            raise ValueError('Finite positive box and proper rotation required')
        # Bound native float32 centre/quaternion/extent conversion, in addition
        # to the independent full scene margin supplied by the caller.
        reserve=2e-6+16*np.finfo(np.float32).eps*(np.linalg.norm(centre)+np.linalg.norm(half)+1)
        return self._query(self.box_query,centre,axes,half+reserve,memoize=memoize)

    def _coverage(self,destination):
        for path,(low,high) in self.scene.items():
            if path not in self._box((low+high)/2,np.eye(3),(high-low)/2+.001):
                self.active=False;self.validated=False;self.errors.append('Missing native actor '+path)
                raise RuntimeError('Missing native actor '+path)
            destination.add(path)
        self._check()

    def _foreign(self,hits,link):
        floor=self.floor_root
        return sorted(p for p in hits-self.robot_paths
            if not (link in ('base','wheel_l','wheel_r') and floor
                and (p==floor or p.startswith(floor+'/'))))

    def can_check_with_final_controls(self,body_world,shapes,*,margin=.001):
        """Reserve complete final actor controls before another whole check.

        This is a search stopping condition, never clearance or a larger
        budget. Count the exact sphere covers for these unchanged proposed
        transforms. No native query, state change or partial check is made.
        """
        self._check()
        paths=[s[0] for s in shapes]
        if set(paths)!=self.robot_paths or len(paths)!=len(self.robot_paths):
            raise ValueError('Complete robot inventory required for query reservation')
        if isinstance(margin,(bool,np.bool_)) or not np.isfinite(margin) or not .001<=margin<=.05:
            raise ValueError('Preserve finite 1..50 mm scene margin')
        cost=0
        for path,body,link,kind,shape in shapes:
            if kind=='capsule':
                m=np.asarray(body_world[link],float);a,b,radius=shape
                centres,_=cover(m[:3,:3]@a+m[:3,3],m[:3,:3]@b+m[:3,3],radius,margin)
                cost+=len(centres)
            elif kind=='box':cost+=1
            else:raise ValueError('Unknown robot geometry cannot be ignored')
        # The owner guard requires calls STRICTLY below its hard ceiling even
        # after the final query returns. Include environment AND robot controls.
        return self.calls+cost+len(self.scene)+len(self.robot_paths)<self.max_queries

    def check(self,body_world,shapes,*,margin=.001):
        self._check();self.validated=False
        paths=[s[0] for s in shapes]
        if set(paths)!=self.robot_paths or len(paths)!=len(self.robot_paths):
            raise ValueError('Every exact active robot collider must be screened once')
        if isinstance(margin,(bool,np.bool_)) or not np.isfinite(margin) or not .001<=margin<=.05:
            raise ValueError('Preserve finite 1..50 mm scene margin')
        self.checks+=1
        for path,body,link,kind,shape in shapes:
            if not _path(body) or not path.startswith(body+'/') or body.rsplit('/',1)[-1]!=link:
                raise ValueError('Exact collider/body/link identity required')
            m=np.asarray(body_world[link],float)
            if (m.shape!=(4,4) or not np.isfinite(m).all()
                    or not np.allclose(m[3],[0,0,0,1],atol=1e-9,rtol=0)
                    or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-8,rtol=0)
                    or np.linalg.det(m[:3,:3])<0):raise ValueError('Rigid robot body poses required')
            if kind=='capsule':
                a,b,radius=shape
                centres,expanded=cover(m[:3,:3]@a+m[:3,3],m[:3,:3]@b+m[:3,3],radius,margin)
                hit=[]
                for centre in centres:
                    hit=self._foreign(self._query(self.sphere_query,centre,expanded,memoize=True),link)
                    if hit:break
            elif kind=='box':
                centre,axes,half=shape
                hit=self._foreign(self._box(m[:3,:3]@centre+m[:3,3],m[:3,:3]@axes,np.asarray(half)+margin,memoize=True),link)
            else:raise ValueError('Unknown robot geometry cannot be ignored')
            if hit:return dict(passed=False,robot_collider=path,scene_colliders=hit,
                motion_authorized=False,whole_path_certified=False)
        self._check();self.cleared+=1
        return dict(passed=True,motion_authorized=False,whole_path_certified=False)

    def validate(self):
        self._check();self.validated=False;self.final_coverage.clear()
        self._coverage(self.final_coverage);self._check();self.validated=True

    def close(self):
        self.query_cache.clear()
        self.active=False

    def report(self):
        return dict(method='whole_robot_frozen_native_scene_query',query_count=self.calls,
            exact_query_cache_enabled=self.memoize_queries,exact_query_cache_hits=self.cache_hits,
            query_cache_entries=len(self.query_cache),query_cache_capacity=4096,
            positive_controls_cached=False,
            scene_colliders=len(self.scene),robot_colliders=len(self.robot_paths),
            initial_positive_controls=len(self.covered),final_positive_controls=len(self.final_coverage),
            candidate_checks=self.checks,clear_candidates=self.cleared,final_validation_passed=self.validated,
            errors=list(self.errors),query_active=self.active,physics_or_usd_changes_made=False,
            native_robot_pose_tracking_verified=False,motion_authorized=False,whole_path_certified=False)
