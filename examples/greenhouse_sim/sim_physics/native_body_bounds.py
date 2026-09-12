"""Read-only synchronous native collider-bound probe; NOT motion authority.

Installed omni.physx110.1.13 exposes PhysxPropertyQueryColliderResponse local
position/rotation and AABB bounds. A complete same-stage response is required;
an asynchronous/missing/error response is unavailable, never empty geometry.
No cooking updates, simulation steps, pose writes or collider edits occur here.
"""
import numpy as np


def collect_synchronous(query,*,stage_id,body_id,expected_paths,decode_path,valid_result,guard=lambda:None,
                        disabled_paths=(),ignored_disabled=None):
    expected=set(expected_paths)
    disabled=set(disabled_paths)
    if not expected or not callable(query) or not callable(decode_path):
        raise ValueError('Expected collider inventory and native callbacks required')
    if disabled & expected or disabled and not isinstance(ignored_disabled,list):
        raise ValueError('Separate disabled inventory and explicit audit log required')
    rows=[];errors=[];bodies=[];finished=[];active=True
    def body(info):
        if not active:return
        if info.result!=valid_result or info.stage_id!=stage_id or info.path_id!=body_id:
            errors.append('invalid_body_response')
        bodies.append(info.path_id)
    def collider(info):
        if not active:return
        try:
            if finished:raise ValueError('Collider callback after completion')
            path=str(decode_path(info.path_id))
            if info.result!=valid_result or info.stage_id!=stage_id or path not in expected|disabled:
                raise ValueError('Invalid, wrong-stage or unexpected collider response: '+str(dict(
                    path=path,result=str(info.result),stage_id=info.stage_id,expected_stage_id=stage_id)))
            if path in disabled:
                ignored_disabled.append(dict(path=path,reason='explicit_collisionEnabled_false_in_same_USD_epoch'))
                return
            low,high=np.asarray(info.aabb_local_min,float),np.asarray(info.aabb_local_max,float)
            position,rotation=np.asarray(info.local_pos,float),np.asarray(info.local_rot,float)
            if (low.shape!=(3,) or high.shape!=(3,) or position.shape!=(3,) or rotation.shape!=(4,)
                    or not np.isfinite(np.r_[low,high,position,rotation,info.volume]).all()
                    or np.any(high<low) or abs(np.linalg.norm(rotation)-1)>1e-4 or info.volume<=0):
                raise ValueError('Invalid native bound/pose/volume')
            if len(rows)>=256:raise ValueError('Native collider-part budget exceeded')
            rows.append(dict(path=path,low=low.tolist(),high=high.tolist(),local_position=position.tolist(),
                local_quaternion_xyzw=rotation.tolist(),volume=float(info.volume)))
        except Exception as exc:errors.append(str(exc))
    def done():
        if active:finished.append(True)
    try:
        guard()
        query(stage_id=stage_id,prim_id=body_id,timeout_ms=100,
            finished_fn=done,rigid_body_fn=body,collider_fn=collider)
        guard()
        if errors or len(finished)!=1 or bodies!=[body_id] or {r['path'] for r in rows}!=expected:
            raise RuntimeError('Native bound query incomplete/unavailable: '+str(dict(
                errors=errors,finished=len(finished),bodies=bodies,
                missing=sorted(expected-{r['path'] for r in rows}))))
        return rows
    finally:active=False  # Late native callbacks cannot populate accepted data.


def current_body_bounds(stage,body_path,*,guard=lambda:None):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools
    from omni.physx import get_physx_property_query_interface
    from omni.physx.bindings._physx import PhysxPropertyQueryResult
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.:
        raise ValueError('Metre stage required')
    body=stage.GetPrimAtPath(body_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or not UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get():
        raise ValueError('Enabled native body required')
    expected=[];disabled=[]
    for prim in Usd.PrimRange(body):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        ancestor=prim
        while ancestor and not ancestor.IsPseudoRoot() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor=ancestor.GetParent()
        if ancestor!=body:raise ValueError('Nested rigid body requires a separate query')
        (disabled if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False else expected).append(str(prim.GetPath()))
    stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    if stage_id<=0:raise ValueError('Already-registered native stage required; no stage insertion')
    ignored=[]
    rows=collect_synchronous(get_physx_property_query_interface().query_prim,
        stage_id=stage_id,body_id=PhysicsSchemaTools.sdfPathToInt(body.GetPath()),
        expected_paths=expected,decode_path=PhysicsSchemaTools.intToSdfPath,
        valid_result=PhysxPropertyQueryResult.VALID,guard=guard,disabled_paths=disabled,ignored_disabled=ignored)
    return dict(parts=rows,disabled_property_responses=ignored,
        active_expected_paths=sorted(expected),disabled_expected_paths=sorted(disabled),
        active_native_actor_shape_identity_proved=False)


def countertransform_box(actual_body,proposed_body,centre,axes,half):
    """Move the query bound into the parked tool frame; NEVER move the tool.

    Rigid intersection is invariant under M=actual*inverse(proposed). The
    target must be completely enclosed by the input box. This mathematical
    identity alone does not establish native shape inventory/frame agreement.
    """
    a,b=np.asarray(actual_body,float),np.asarray(proposed_body,float)
    centre,axes,half=np.asarray(centre,float),np.asarray(axes,float),np.asarray(half,float)
    if a.shape!=(4,4) or b.shape!=(4,4) or centre.shape!=(3,) or axes.shape!=(3,3) or half.shape!=(3,):
        raise ValueError('Rigid poses and finite box required')
    if not np.isfinite(np.r_[a.flat,b.flat,centre,axes.flat,half]).all() or np.any(half<=0):
        raise ValueError('Finite positive box required')
    for f in (a,b):
        if (not np.allclose(f[3],[0,0,0,1],atol=1e-9,rtol=0)
                or not np.allclose(f[:3,:3].T@f[:3,:3],np.eye(3),atol=1e-6,rtol=0)
                or np.linalg.det(f[:3,:3])<0):raise ValueError('Rigid tool frames required')
    if not np.allclose(axes.T@axes,np.eye(3),atol=1e-6,rtol=0) or np.linalg.det(axes)<0:
        raise ValueError('Proper box axes required')
    m=a@np.linalg.inv(b)
    return m[:3,:3]@centre+m[:3,3],m[:3,:3]@axes,half.copy()
