"""Full instance-proxy static scan followed by unchanged strict USD notices."""
from .capture_contract import fingerprint
from .static_guard import relevant_path


def _check_prim(prim):
    from pxr import UsdPhysics
    path=str(prim.GetPath())
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
        raise ValueError('Static capture refuses enabled rigid body including proxy: '+path)
    if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
        raise ValueError('Static capture refuses animated attributes including proxy: '+path)
    if any(path.startswith(prefix) for prefix in ('/World/DraftLabels','/World/AnatomyReview','/World/Review','/World/Reachability')):
        raise ValueError('Diagnostic overlays present: '+path)


def assert_static_plant_sources(stage):
    """No mutable/animated physics is hidden inside background prototypes."""
    from pxr import Usd,UsdPhysics
    root=stage.GetPrimAtPath('/World/PackPlants')
    if not root:raise ValueError('Full plant population required')
    count=0
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        _check_prim(prim)
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get():
            raise ValueError('Static plant source contains enabled joint: '+str(prim.GetPath()))
        count+=1
    return count


def scene_guard(stage):
    """Same visibility/transform/material state as original, including proxies."""
    from pxr import Usd,UsdGeom
    states=[];cache=UsdGeom.XformCache();root=stage.GetPrimAtPath('/World')
    if not root:raise ValueError('Existing World required')
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        _check_prim(prim)
        if not prim.IsA(UsdGeom.Imageable):continue
        imageable=UsdGeom.Imageable(prim)
        states.append((str(prim.GetPath()),imageable.ComputeVisibility(),
            [list(r) for r in cache.GetLocalToWorldTransform(prim)],
            [str(p) for p in prim.GetRelationship('material:binding').GetTargets()]))
    return fingerprint(states)


class StaticSceneMonitor:
    def __init__(self,stage,robot_root='/World/RBY1'):
        from pxr import Tf,Usd
        self.baseline=scene_guard(stage);self.robot_root=robot_root
        self.generation=0;self.changed=set()
        self.subscription=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._notice,stage)

    def _notice(self,notice,sender):
        paths=[*notice.GetResyncedPaths(),*notice.GetChangedInfoOnlyPaths()]
        relevant={str(p) for p in paths if relevant_path(p)}
        if relevant:self.changed.update(relevant);self.generation+=1

    def begin(self):
        unexpected=[p for p in self.changed if not(p==self.robot_root or p.startswith(self.robot_root+'/') or p.startswith(self.robot_root+'.'))]
        if unexpected:raise ValueError('Unapproved static scene change: '+repr(sorted(unexpected)[:8]))
        self.changed.clear();return self.token()

    def token(self):
        return fingerprint(dict(method='initial_proxy_aware_static_scan_and_USD_change_notice.v1',baseline=self.baseline,generation=self.generation))

    def close(self):self.subscription.Revoke()
