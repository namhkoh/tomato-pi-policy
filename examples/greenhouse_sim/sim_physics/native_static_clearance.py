"""Bounded synchronous native refinement of conservative static-scene boxes.

The proposed tool remains its enclosing box, expanded by the full margin.
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
                 close_guard=None, epoch_report=None, lazy_coverage=False):
        if (type(max_queries) is not int or not 0 < max_queries <= 20000
                or type(lazy_coverage) is not bool
                or isinstance(wall_limit_s,(bool,np.bool_))
                or not np.isfinite(wall_limit_s) or not 0 < wall_limit_s <= 60.):
            raise ValueError('Bounded native query budget required')
        self.query=query;self.guard=guard;self.max_queries=max_queries
        self.lazy_coverage=lazy_coverage
        self.started=time.perf_counter();self.wall_limit_s=wall_limit_s
        self.active=True;self.calls=0;self.clearances=0;self.blocked=0
        self.capsule_box_attempts=0
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
        self.active=False;self.validation_passed=False

    def _check(self,*,request=False):
        if not self.active:
            raise RuntimeError('Native refinement invalid or closed: '+str(self.errors))
        self.guard()
        if (self.calls>self.max_queries or (request and self.calls>=self.max_queries)
                or time.perf_counter()-self.started>=self.wall_limit_s):
            raise RuntimeError('query_budget_exhausted')

    def _overlap(self,path,centre,axes,half):
        if not self.active:return None
        try:
            self._check(request=True)
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
            self.calls+=1
            result=self.query(path,centre,axes,half)
            # A native call cannot be interrupted here; reject its result if
            # it returns after the deadline or after any epoch invalidation.
            self._check()
            if type(result) is not bool:raise ValueError('Native overlap did not return an explicit bool')
            return result
        except Exception as exc:
            self._invalidate(type(exc).__name__+': '+str(exc))
            return None

    def clear_box(self,path,centre,axes,half,margin):
        if not np.isfinite(margin) or margin<.001:
            raise ValueError('Native refinement preserves >=1 mm scene margin')
        if not self._ensure_coverage(path):return False
        self.validation_passed=False
        hit=self._overlap(path,centre,axes,np.asarray(half,float)+margin)
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
        return self.clear_box_checked(path,centre,axes,half,margin)

    def validate(self):
        """Acceptance gate: invalidate earlier clearances on ANY epoch failure."""
        self.validation_passed=False;self.final_coverage=[]
        try:
            self._check()
            for path in sorted(self.used_paths):
                if self._overlap(path,*self.coverage_boxes[path]) is not True:
                    raise RuntimeError('Final native actor coverage missing: '+path)
                self.final_coverage.append(path)
            self._check()
            self.validation_passed=True
        except Exception as exc:
            self._invalidate(type(exc).__name__+': '+str(exc))
            raise RuntimeError('Native static clearance final validation failed: '+str(exc)) from exc

    def close(self):
        if self.closed:return
        validated=self.validation_passed
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
                    query_count=self.calls,coarse_rejections_cleared=self.clearances,
                    capsule_enclosing_box_attempts=self.capsule_box_attempts,
                    capsule_query='whole_capsule_plus_margin_contained_not_tessellated_samples',
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
        if (self.revision or not np.isfinite(self.simulation_time)
                or self.context.get_stage()!=self.stage
                or self.timeline.get_current_time()!=self.simulation_time
                or (self.timeline.is_playing(),self.timeline.is_stopped())!=self.timeline_state):
            raise RuntimeError('Native query epoch invalidated: '+str(self.reasons))

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


def current_scene_query(stage, records, *, wall_limit_s=8., max_queries=20000, lazy_coverage=False):
    """Create only during native planning after fetched physics; never load/step."""
    import omni.usd
    import omni.timeline
    from omni.physx import get_physx_interface,get_physx_scene_query_interface
    from scipy.spatial.transform import Rotation
    from pxr import UsdGeom,UsdPhysics
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.:
        raise ValueError('Native static clearance requires a metre-unit stage')
    timeline=omni.timeline.get_timeline_interface()
    epoch=_SceneQueryEpoch(stage,omni.usd.get_context(),timeline,get_physx_interface())
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
    try:
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
        result=NativeStaticClearance(query,eligible,guard=epoch.check,
            close_guard=epoch.close,epoch_report=epoch.report,wall_limit_s=wall_limit_s,
            max_queries=max_queries,lazy_coverage=lazy_coverage)
        epoch.check()
        return result
    except BaseException as exc:
        try:epoch.close()
        except Exception as cleanup:exc.add_note('Epoch cleanup: '+str(cleanup))
        raise
