"""Bounded synchronous native refinement of conservative static-scene boxes.

The proposed tool remains its enclosing box, expanded by the full margin.
Arm capsules optionally use a complete conservative sphere union after a box
hit; this tightens the planning bound, never the native collision geometry.
Only an exact static collider's coarse rejection can be cleared. No cached
convex, source visual mesh, new collider, dynamic-body assumption or force
filter substitutes for the live native overlap query. Close after ONE plan.
"""
import time
import math
import numpy as np


def capsule_enclosing_box(start,end,radius):
    """Contain the WHOLE capsule, including both spherical ends, in an OBB.

    This box is deliberately larger than the capsule. A native miss proves
    clearance; a hit does not prove a capsule collision and remains rejected.
    Expanding every half-extent by margin also contains its spherical margin.
    """
    a,b=np.asarray(start,float),np.asarray(end,float)
    if (a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all()
            or isinstance(radius,(bool,np.bool_)) or not np.isscalar(radius)
            or not np.isfinite(radius) or radius<=0):
        raise ValueError('Finite capsule endpoints and positive radius required')
    delta=b-a;length=math.hypot(*delta)
    if not math.isfinite(length):raise ValueError('Unrepresentable capsule length')
    axes=np.eye(3)
    if length:
        z=delta/length;helper=np.eye(3)[int(np.argmin(np.abs(z)))]
        x=np.cross(helper,z);x/=np.linalg.norm(x)
        axes=np.column_stack((x,np.cross(z,x),z))
    centre=a+delta/2;half=np.array([radius,radius,radius+length/2])
    if not np.isfinite([centre,half]).all():raise ValueError('Unrepresentable capsule box')
    return centre,axes,half


class NativeStaticClearance:
    def __init__(self, query, records, *, guard=lambda:None, max_queries=20000, wall_limit_s=8.,
                 close_guard=None, epoch_report=None, lazy_coverage=False, sphere_query=None, memoize_queries=False,
                 reuse_clear_regions=False,heartbeat=None):
        if (type(max_queries) is not int or not 0 < max_queries <= 20000
                or type(lazy_coverage) is not bool or type(memoize_queries) is not bool
                or type(reuse_clear_regions) is not bool
                or isinstance(wall_limit_s,(bool,np.bool_))
                or not np.isfinite(wall_limit_s) or not 0 < wall_limit_s <= 60.):
            raise ValueError('Bounded native query budget required')
        self.query=query;self.guard=guard;self.max_queries=max_queries
        if heartbeat is not None and not callable(heartbeat):raise ValueError('Callable render-only heartbeat required')
        self.heartbeat=heartbeat
        self.memoize_queries=memoize_queries;self.query_cache={};self.cache_hits=0
        from .empty_regions import EmptyRegions
        self.empty_regions=EmptyRegions() if reuse_clear_regions else None
        self.lazy_coverage=lazy_coverage
        self.started=time.perf_counter();self.wall_limit_s=wall_limit_s
        self.active=True;self.calls=0;self.clearances=0;self.blocked=0
        self.capsule_box_attempts=0
        if sphere_query is not None and not callable(sphere_query):raise ValueError('Callable native sphere query required')
        self.sphere_query=sphere_query;self.sphere_calls=0;self.sphere_covered=set();self.sphere_used=set()
        self.covered=set();self.coverage_failed=[];self.errors=[]
        self.used_paths=set();self.coverage_boxes={};self.final_coverage=[]
        self.validation_passed=False;self.closed=False
        self.close_guard=close_guard;self.epoch_report=epoch_report
        # Every cached coarse bound is retained. Optional lazy coverage defers
        # the positive native actor control until that EXACT actor is queried
        # for refinement, never past a claimed clearance. Each instance still
        # owns one synchronous epoch; final used-actor controls are unchanged.
        for path,kind,data,low,high in records:
            if kind!='box':continue
            if path in self.coverage_boxes:raise ValueError('Duplicate static collider: '+path)
            low,high=np.asarray(low,float),np.asarray(high,float)
            if (low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all()
                    or np.any(high<low)):
                raise ValueError('Invalid static coverage bounds')
            box=((low+high)/2,np.eye(3),(high-low)/2+.001)
            self.coverage_boxes[path]=tuple(v.copy() for v in box)
            if not self.lazy_coverage:self._ensure_coverage(path)

    def _ensure_coverage(self,path):
        if not self.active or path not in self.coverage_boxes:return False
        if path in self.covered:return True
        if path in self.coverage_failed:return False
        if self._overlap(path,*self.coverage_boxes[path]) is True:
            self.covered.add(path)
            return True
        self.coverage_failed.append(path)
        return False

    def _invalidate(self,reason):
        if reason not in self.errors:self.errors.append(reason)
        self.query_cache.clear()
        if self.empty_regions is not None:self.empty_regions.clear()
        self.active=False;self.validation_passed=False

    def _check(self,*,request=False):
        if not self.active:
            raise RuntimeError('Native refinement invalid or closed: '+str(self.errors))
        self.guard()
        if self.heartbeat is not None:
            self.heartbeat()
            # UI may process a stop, transform or timeline edit. No cached
            # clearance can survive that event, including a cache-hit path.
            self.guard()
        if self.calls>self.max_queries or (request and self.calls>=self.max_queries):
            raise RuntimeError('query_budget_exhausted: native_call_limit_exceeded')
        if time.perf_counter()-self.started>=self.wall_limit_s:
            raise RuntimeError('query_budget_exhausted: wall_time_limit_exceeded')

    def _overlap(self,path,centre,axes,half,*,memoize=False):
        if not self.active:return None
        try:
            self._check()
            centre,axes,half=np.asarray(centre,float),np.asarray(axes,float),np.asarray(half,float)
            if (centre.shape!=(3,) or axes.shape!=(3,3) or half.shape!=(3,)
                    or not np.isfinite(centre).all() or not np.isfinite(axes).all()
                    or not np.isfinite(half).all() or np.any(half<=0)
                    or not np.allclose(axes.T@axes,np.eye(3),atol=1e-8,rtol=0)):
                raise ValueError('Invalid native tool query box')
            # A box is unchanged by a local axis sign flip; a quaternion must
            # nevertheless represent a proper rotation, never a reflection.
            if np.linalg.det(axes)<0:
                axes=axes.copy();axes[:,2]*=-1
            key=('box',path,centre.tobytes(),axes.tobytes(),half.tobytes())
            if memoize and self.memoize_queries and key in self.query_cache:
                self._check();self.cache_hits+=1;return self.query_cache[key]
            if memoize and self.empty_regions is not None:
                if self.empty_regions.contains(path,centre,axes,half):
                    self._check();return False
                # Ask about an enclosing box first. A miss proves the
                # original query too; a hit proves NOTHING about the original
                # and falls through to its unchanged exact native query.
                expanded=half+.005
                enclosing=self._overlap(path,centre,axes,expanded,memoize=False)
                if enclosing is None:return None
                if enclosing is False:
                    self.empty_regions.add(path,centre,axes,expanded)
                    self._check();return False
            self._check(request=True)
            self.calls+=1
            result=self.query(path,centre,axes,half)
            # A native call cannot be interrupted here; reject its result if
            # it returns after the deadline or after any epoch invalidation.
            self._check()
            if type(result) is not bool:raise ValueError('Native overlap did not return an explicit bool')
            if memoize and self.memoize_queries and len(self.query_cache)<4096:self.query_cache[key]=result
            return result
        except Exception as exc:
            self._invalidate(type(exc).__name__+': '+str(exc))
            return None

    def clear_box(self,path,centre,axes,half,margin):
        if not np.isfinite(margin) or margin<.001:
            raise ValueError('Native refinement preserves >=1 mm scene margin')
        if not self._ensure_coverage(path):return False
        self.validation_passed=False
        hit=self._overlap(path,centre,axes,np.asarray(half,float)+margin,memoize=True)
        clear=hit is False
        self.clearances+=int(clear);self.blocked+=int(not clear)
        if clear:self.used_paths.add(path)
        return clear

    def clear_box_checked(self,path,centre,axes,half,margin):
        """Planner API: an invalid query is unavailable, NOT a collision hit.

        The conservative boolean API above remains useful to diagnostic
        callers. Execution planning must stop on invalidation instead of
        spending the remaining grid budget and mislabelling fallback bounds.
        """
        def check():
            try:self._check()
            except Exception as exc:
                self._invalidate(type(exc).__name__+': '+str(exc))
                raise RuntimeError('Native static query unavailable; collision not determined: '+str(exc)) from exc
        check()
        clear=self.clear_box(path,centre,axes,half,margin)
        check()
        return clear

    def clear_capsule_checked(self,path,start,end,radius,margin):
        centre,axes,half=capsule_enclosing_box(start,end,radius)
        self.capsule_box_attempts+=1
        clear=self.clear_box_checked(path,centre,axes,half,margin)
        if clear or self.sphere_query is None:return clear
        if not self._ensure_coverage(path):return False
        try:
            if path not in self.sphere_covered:
                self._sphere_control(path);self.sphere_covered.add(path)
            from .capsule_sphere_cover import cover
            centres,covered_radius=cover(start,end,radius,margin)
            for point in centres:
                if self._sphere_overlap(path,point,covered_radius,memoize=True):return False
            self._check();self.used_paths.add(path);self.sphere_used.add(path)
            self.clearances+=1;self.blocked-=1
            return True
        except Exception as exc:
            self._invalidate(type(exc).__name__+': '+str(exc))
            raise RuntimeError('Native sphere cover unavailable; clearance not determined: '+str(exc)) from exc

    def _sphere_overlap(self,path,centre,radius,*,memoize=False):
        self._check();centre=np.asarray(centre,float);radius=float(radius)
        if centre.shape!=(3,) or not np.isfinite(centre).all() or not math.isfinite(radius) or radius<=0:
            raise ValueError('Finite positive native sphere required')
        key=('sphere',path,centre.tobytes(),radius)
        if memoize and self.memoize_queries and key in self.query_cache:
            self._check();self.cache_hits+=1;return self.query_cache[key]
        self._check(request=True);self.calls+=1;self.sphere_calls+=1
        hit=self.sphere_query(path,centre,radius)
        self._check()
        if type(hit) is not bool:raise RuntimeError('Native sphere overlap must return explicit bool')
        if memoize and self.memoize_queries and len(self.query_cache)<4096:self.query_cache[key]=hit
        return hit

    def _sphere_control(self,path):
        centre,axes,half=self.coverage_boxes[path]
        point=np.asarray(centre,dtype=np.float32).astype(float)
        radius=float(np.nextafter(np.float32(np.linalg.norm(half)+np.linalg.norm(point-centre)+1e-6),np.float32(np.inf)))
        if not self._sphere_overlap(path,point,radius):
            raise RuntimeError('Native sphere actor positive control missing: '+path)

    def validate(self):
        """Acceptance gate: invalidate earlier clearances on ANY epoch failure."""
        self.validation_passed=False;self.final_coverage=[]
        try:
            self._check()
            for path in sorted(self.used_paths):
                if self._overlap(path,*self.coverage_boxes[path]) is not True:
                    raise RuntimeError('Final native actor coverage missing: '+path)
                self.final_coverage.append(path)
                if path in self.sphere_used:self._sphere_control(path)
            self._check()
            self.validation_passed=True
        except Exception as exc:
            self._invalidate(type(exc).__name__+': '+str(exc))
            raise RuntimeError('Native static clearance final validation failed: '+str(exc)) from exc

    def close(self):
        if self.closed:return
        validated=self.validation_passed
        self.query_cache.clear()
        if self.empty_regions is not None:self.empty_regions.clear()
        self.active=False;self.closed=True
        try:
            if self.close_guard is not None:self.close_guard()
            if validated:
                # Still reject a last event/time change while releasing our
                # subscriptions; the guard retains its latched epoch record.
                self.guard()
                if time.perf_counter()-self.started>=self.wall_limit_s:
                    raise RuntimeError('query_budget_exhausted_during_close')
        except Exception as exc:
            self._invalidate(type(exc).__name__+': '+str(exc))
            raise

    def report(self):
        return dict(method='live_native_static_overlap_expanded_conservative_robot_box',
                    render_only_heartbeat_enabled=self.heartbeat is not None,
                    exact_query_cache_enabled=self.memoize_queries,exact_query_cache_hits=self.cache_hits,
                    query_cache_entries=len(self.query_cache),query_cache_capacity=4096,
                    query_cache_scope='one_frozen_epoch_exact_float64_inputs_no_rounding',
                    positive_controls_cached=False,
                    native_empty_regions=None if self.empty_regions is None else self.empty_regions.report(),
                    query_count=self.calls,coarse_rejections_cleared=self.clearances,
                    capsule_enclosing_box_attempts=self.capsule_box_attempts,
                    capsule_query='whole_capsule_plus_margin_contained_not_tessellated_samples',
                    capsule_sphere_cover_enabled=self.sphere_query is not None,
                    sphere_query_count=self.sphere_calls,sphere_positive_controlled_paths=sorted(self.sphere_covered),
                    sphere_cover_cleared_paths=sorted(self.sphere_used),
                    retained_rejections=self.blocked,covered_static_colliders=sorted(self.covered),
                    coverage_mode='lazy_before_exact_actor_refinement' if self.lazy_coverage else 'eager',
                    unchecked_static_colliders=sorted(set(self.coverage_boxes)-self.covered-set(self.coverage_failed)),
                    coverage_failed=self.coverage_failed,errors=self.errors,
                    used_static_colliders=sorted(self.used_paths),
                    final_coverage_checked=list(self.final_coverage),
                    final_validation_passed=self.validation_passed,closed=self.closed,
                    epoch=None if self.epoch_report is None else self.epoch_report(),
                    max_queries=self.max_queries,wall_limit_s=self.wall_limit_s,
                    captured_convexes_used=False,collision_filters_changed=False,
                    current_plan_only=True,whole_path_certified=False,
                    all_simulation_shape_query_coverage_proved=False)


class _SceneQueryEpoch:
    """Own invalidation subscriptions; callbacks only latch, never unsubscribe."""
    def __init__(self,stage,context,timeline,physx):
        from pxr import Tf,Usd
        self.stage=stage;self.context=context;self.timeline=timeline;self.physx=physx
        self.simulation_time=float(timeline.get_current_time())
        self.timeline_state=(timeline.is_playing(),timeline.is_stopped())
        self.revision=0;self.reasons=[];self.closed=False;self.cleanup_errors=[]
        self.notice=None;self.object_subscription=None;self.step_subscription=None
        try:
            self.notice=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._usd_changed,stage)
            self.object_subscription=physx.subscribe_object_changed_notifications(
                object_creation_fn=lambda *args:self._changed('physx_object_created'),
                object_destruction_fn=lambda *args:self._changed('physx_object_destroyed'),
                all_objects_destruction_fn=lambda:self._changed('physx_all_objects_destroyed'),
                stop_callback_when_sim_stopped=False)
            if type(self.object_subscription) is not int or self.object_subscription<0:
                raise RuntimeError('Missing PhysX object-change subscription')
            self.step_subscription=physx.subscribe_physics_on_step_events(
                fn=lambda dt:self._changed('physics_pre_step'),pre_step=True,order=0)
            if self.step_subscription is None or not callable(getattr(self.step_subscription,'unsubscribe',None)):
                raise RuntimeError('Missing PhysX pre-step subscription')
            self.check()
        except BaseException as exc:
            try:self.close()
            except Exception as cleanup:exc.add_note('Epoch cleanup: '+str(cleanup))
            raise

    def _usd_changed(self,notice,sender):
        self._changed('usd_objects_changed')

    def _changed(self,reason):
        if self.closed:return
        self.revision+=1
        if reason not in self.reasons:self.reasons.append(reason)

    def check(self):
        reasons=list(self.reasons)
        if not np.isfinite(self.simulation_time):reasons.append('nonfinite_snapshot_time')
        if self.context.get_stage()!=self.stage:reasons.append('stage_changed')
        actual=self.timeline.get_current_time()
        if actual!=self.simulation_time:reasons.append(f'timeline_time_changed:{self.simulation_time}->{actual}')
        if (self.timeline.is_playing(),self.timeline.is_stopped())!=self.timeline_state:
            reasons.append('timeline_play_state_changed')
        if self.revision or reasons:
            raise RuntimeError('Native query epoch invalidated: '+str(reasons))

    def close(self):
        if self.closed:return
        # Attempt every owned release even when one fails. Never alter
        # subscriptions from an object-change or physics-step callback.
        releases=[]
        if self.step_subscription is not None:
            unsubscribe=getattr(self.step_subscription,'unsubscribe',None)
            if callable(unsubscribe):releases.append(('physics_pre_step',unsubscribe))
            else:self.cleanup_errors.append('physics_pre_step: invalid subscription holder')
        if type(self.object_subscription) is int and self.object_subscription>=0:
            releases.append(('physx_objects',lambda:self.physx.unsubscribe_object_change_notifications(self.object_subscription)))
        if self.notice is not None:releases.append(('usd_notice',self.notice.Revoke))
        try:
            for name,release in releases:
                try:release()
                except Exception as exc:self.cleanup_errors.append(name+': '+str(exc))
        finally:
            self.closed=True
            self.step_subscription=None;self.object_subscription=None;self.notice=None
        if self.cleanup_errors:raise RuntimeError('Epoch subscription cleanup failed: '+str(self.cleanup_errors))

    def report(self):
        return dict(simulation_time_s=self.simulation_time,revision=self.revision,
                    invalidation_reasons=list(self.reasons),
                    subscriptions_closed=self.closed and not self.cleanup_errors,
                    subscription_cleanup_attempted=self.closed,
                    cleanup_errors=list(self.cleanup_errors),
                    usd_objects_changed_subscribed=True,physx_object_changes_subscribed=True,
                    physics_pre_step_subscribed=True)


def current_scene_query(stage, records, *, wall_limit_s=8., max_queries=20000, lazy_coverage=False,capsule_sphere_cover=False,heartbeat=None):
    """Create only during native planning after fetched physics; never load/step."""
    import omni.usd
    import omni.timeline
    from omni.physx import get_physx_interface,get_physx_scene_query_interface
    from scipy.spatial.transform import Rotation
    from pxr import UsdGeom,UsdPhysics
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.:
        raise ValueError('Native static clearance requires a metre-unit stage')
    if type(capsule_sphere_cover) is not bool:raise ValueError('Explicit sphere-cover option required')
    timeline=omni.timeline.get_timeline_interface()
    epoch=_SceneQueryEpoch(stage,omni.usd.get_context(),timeline,get_physx_interface())
    frozen=None
    def checked():
        epoch.check()
        if frozen is not None:
            if not frozen.closed:frozen.check()
            elif timeline.is_auto_updating() is not frozen.old:
                raise RuntimeError('Planning timeline restoration changed')
    def close_owned():
        try:
            if frozen is not None:frozen.close()
            # A validated query checks its epoch again AFTER owned cleanup.
            # Already-invalid queries still need idempotent cleanup without
            # repeating their original execution fault as a cleanup failure.
        finally:epoch.close()
    def query(path,centre,axes,half):
        count=0;matched=False
        def collect(hit):
            nonlocal count,matched
            count+=1
            matched |= hit.collision==path
            return True
        reported=native.overlap_box(tuple(half),tuple(centre),tuple(Rotation.from_matrix(axes).as_quat()),collect,False)
        if reported!=count:raise RuntimeError('Incomplete native overlap callback coverage')
        return bool(matched)
    def sphere_query(path,centre,radius):
        # Documented native API; no query mesh, actor or pose is authored.
        # https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html
        count=0;matched=False
        def collect(hit):
            nonlocal count,matched
            count+=1;matched |= hit.collision==path
            return True
        reported=native.overlap_sphere(radius,tuple(centre),collect,False)
        if reported!=count:raise RuntimeError('Incomplete native sphere callback coverage')
        return bool(matched)
    try:
        if heartbeat is not None:
            from .planning_heartbeat import FrozenTimeline
            frozen=FrozenTimeline(timeline)
            checked()
        eligible=[]
        for record in records:
            path,kind,_,_,_=record
            if kind!='box':continue
            prim=stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            # No enabled rigid-body ancestor, including kinematic bodies.
            current=prim;body=False
            while current and not current.IsPseudoRoot():
                if current.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(current).GetRigidBodyEnabledAttr().Get():
                    body=True;break
                current=current.GetParent()
            if not body:eligible.append(record)
        epoch.check()
        native=get_physx_scene_query_interface()
        result=NativeStaticClearance(query,eligible,guard=checked,
            close_guard=close_owned,epoch_report=epoch.report,wall_limit_s=wall_limit_s,
            max_queries=max_queries,lazy_coverage=lazy_coverage,
            heartbeat=heartbeat,
            memoize_queries=True,reuse_clear_regions=True,
            sphere_query=sphere_query if capsule_sphere_cover else None)
        checked()
        return result
    except BaseException as exc:
        try:close_owned()
        except Exception as cleanup:exc.add_note('Epoch cleanup: '+str(cleanup))
        raise
