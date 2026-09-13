"""Complete native startup geometry before the FIRST physics step.

Explicit isolated diagnostic option, not a path/cut certificate. No collider,
filter, force threshold, body pose or timeline state is authored here. Native
parsing is allowed only on the current stopped zero-time stage. Every active
scene AND robot collider receives positive controls before and after screening.
"""
import time
import numpy as np


def screen(stage, robot):
    import omni.usd
    import omni.timeline
    from omni.physx import get_physx_interface, get_physx_scene_query_interface
    from .native_static_clearance import _SceneQueryEpoch
    return _screen(stage, robot, omni.usd.get_context(),
        omni.timeline.get_timeline_interface(), get_physx_interface(),
        get_physx_scene_query_interface(), _SceneQueryEpoch)


def _screen(stage, robot, context, timeline, physx, query, epoch_factory):
    """Dependency-injected lifecycle for contract tests; CLI uses real APIs."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from .native_startup_clearance import NativeStartupClearance
    result=dict(passed=False, method='complete_zero_step_native_startup_geometry',
        physics_steps=0, motion_authorized=False, whole_path_certified=False,
        native_robot_pose_tracking_verified=False, collision_filters_changed=False,
        initial_robot_actor_controls=0, final_robot_actor_controls=0)
    began=time.perf_counter(); counter=[0]; subscription=None; epoch=None; backend=None
    calls=0
    def guard():
        if (context.get_stage()!=stage or not timeline.is_stopped() or timeline.is_playing()
                or timeline.get_current_time()!=0 or counter[0]
                or time.perf_counter()-began>=60 or calls>=20000):
            raise RuntimeError('Native startup requires a frozen current zero-step stage and bounded budget')
        if epoch is not None:epoch.check()
    def box(c,a,h):
        nonlocal calls
        guard(); hits=[]; calls+=1
        count=query.overlap_box(tuple(h),tuple(c),tuple(Rotation.from_matrix(a).as_quat()),
            lambda hit:hits.append(str(hit.collision)) or True,False)
        guard()
        if count!=len(hits):raise RuntimeError('Incomplete native box callbacks')
        return hits
    def sphere(c,radius):
        nonlocal calls
        guard(); hits=[]; calls+=1
        count=query.overlap_sphere(radius,tuple(c),lambda hit:hits.append(str(hit.collision)) or True,False)
        guard()
        if count!=len(hits):raise RuntimeError('Incomplete native sphere callbacks')
        return hits
    def frames():
        cache=UsdGeom.XformCache()
        paths=set(robot.body_paths)|set(robot.rig.body_paths)
        result={}
        for path in paths:
            prim=stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():raise RuntimeError('Missing authored body '+path)
            value=np.asarray(cache.GetLocalToWorldTransform(prim)).T
            if value.shape!=(4,4) or not np.isfinite(value).all():raise RuntimeError('Invalid authored body frame')
            result[path]=value
        return result
    try:
        guard()
        if UsdGeom.GetStageMetersPerUnit(stage)!=1. or UsdGeom.GetStageUpAxis(stage)!='Z':
            raise ValueError('Native startup requires metre-unit Z-up stage')
        scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes)!=1 or str(scenes[0].GetPath())!='/World/QualificationPhysics':
            raise ValueError('One explicit owned qualification physics scene required')
        subscription=physx.subscribe_physics_on_step_events(
            fn=lambda dt:counter.__setitem__(0,counter[0]+1),pre_step=True,order=0)
        if subscription is None or not callable(getattr(subscription,'unsubscribe',None)):
            raise RuntimeError('Native startup pre-step observation unavailable')
        before=frames()
        physx.force_load_physics_from_usd()  # Parses actors; NEVER reset/play/step here.
        guard(); loaded=frames()
        change=max(float(np.max(abs(loaded[p]-before[p]))) for p in before)
        result['maximum_loading_matrix_element_change']=change
        # Loading quantization is separately recorded, bounded far below the
        # independent 1 mm query margin. Never silently accept a moved spawn.
        if change>1e-5:raise RuntimeError('Native loading materially changed authored body poses')
        epoch=epoch_factory(stage,context,timeline,physx);guard()
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
        environment={}; active_robot={}
        for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            if not prim.IsA(UsdGeom.Boundable):raise RuntimeError('Unbounded active collider')
            bound=bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            lo,hi=np.array(bound.GetMin()),np.array(bound.GetMax())
            if bound.IsEmpty() or not np.isfinite([lo,hi]).all():raise RuntimeError('Missing active collider bounds')
            path=str(prim.GetPath())
            (active_robot if path.startswith(robot.root+'/') else environment)[path]=(lo,hi)
        if set(active_robot)!={s[0] for s in robot.self_screen.shapes}:
            raise RuntimeError('Native startup requires exact complete robot shape inventory')
        backend=NativeStartupClearance(box,sphere,environment,active_robot,
            guard=guard,floor_root=robot.floor_root,wall_limit_s=60.)
        def robot_controls(key):
            for path,(lo,hi) in active_robot.items():
                hits=backend._box((lo+hi)/2,np.eye(3),(hi-lo)/2+.001)
                if path not in hits:raise RuntimeError('Missing native robot actor '+path)
                result[key]+=1
        robot_controls('initial_robot_actor_controls')
        world={link:loaded[body] for _,body,link,_,_ in robot.self_screen.shapes}
        self_check=robot.self_screen.check(world)
        result['self_screen']=self_check
        geometry=backend.check(world,robot.self_screen.shapes)
        result['geometry']=geometry
        if getattr(robot,'startup_right_pose_search',False):
            from .startup_pose_search import search
            result['right_pose_search']=search(robot,backend,guard)
        backend.validate();robot_controls('final_robot_actor_controls');guard()
        after=frames()
        result['maximum_authored_pose_change_during_queries']=max(float(np.max(abs(after[p]-loaded[p]))) for p in loaded)
        if result['maximum_authored_pose_change_during_queries']!=0:
            raise RuntimeError('Startup query changed authored poses')
        result['passed']=bool(geometry['passed'] and self_check['passed'] and backend.validated)
    except Exception as exc:
        result.update(passed=False,error=str(exc))
    finally:
        for resource in (backend,epoch,subscription):
            if resource is None:continue
            try:
                resource.unsubscribe() if resource is subscription else resource.close()
            except Exception as exc:result.update(passed=False,cleanup_error=str(exc))
        if backend is not None:result['native']=backend.report()
        if epoch is not None:result['epoch']=epoch.report()
        if 'right_pose_search' in result:
            controls_ok=bool(backend is not None and backend.validated and counter[0]==0
                and 'error' not in result and 'cleanup_error' not in result
                and result.get('maximum_authored_pose_change_during_queries')==0
                and result['initial_robot_actor_controls']==result['final_robot_actor_controls'])
            result['right_pose_search']['final_native_controls_passed']=controls_ok
            if not controls_ok:
                # A stance proposal owns both arms AND the base. Revoke all
                # provisional command fields on any final-control failure.
                for key in tuple(result['right_pose_search']):
                    if key.startswith('proposed_'):result['right_pose_search'][key]=None
        result.update(physics_steps=counter[0],native_query_calls=calls,wall_s=time.perf_counter()-began)
    return result
