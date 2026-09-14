"""Explicit pre-parse solver controls; never change a running physics scene."""


def uniform_iterations(stage,roots,pair):
    """Isolated convergence experiment on BOTH sides of robot/plant contact.

    Native islands can inherit the larger body's iteration requirement; a
    plant-only override is not a uniform 128/0 comparison. Readback is USD,
    not an unavailable native solver-iteration getter.
    """
    from pxr import Sdf,Usd,UsdPhysics
    roots=tuple(roots);pair=tuple(pair)
    if (len(roots)!=2 or len(set(roots))!=2 or any(not isinstance(p,str)
            or not p.startswith('/World/') or not stage.GetPrimAtPath(p) for p in roots)
            or any(a.startswith(b+'/') for a in roots for b in roots if a!=b)
            or len(pair)!=2 or any(type(x) is not int for x in pair)
            or pair not in ((32,8),(64,0),(96,0),(128,0),(128,8))):
        raise ValueError('Two distinct plant/robot roots and bounded diagnostic iteration pair required')
    records=[]
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for api,schema,prefix in ((UsdPhysics.RigidBodyAPI,'PhysxRigidBodyAPI','physxRigidBody'),
                    (UsdPhysics.ArticulationRootAPI,'PhysxArticulationAPI','physxArticulation')):
                if prim.HasAPI(api):records.append((str(prim.GetPath()),schema,prefix))
    if not all(any(p==root or p.startswith(root+'/') for p,_,_ in records) for root in roots):
        raise ValueError('Both roots require actual native body/ articulation schemas')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for path,schema,prefix in records:
            prim=stage.GetPrimAtPath(path);prim.AddAppliedSchema(schema)
            for suffix,value in zip(('solverPositionIterationCount','solverVelocityIterationCount'),pair,strict=True):
                prim.CreateAttribute(prefix+':'+suffix,Sdf.ValueTypeNames.Int,custom=False).Set(value)
    report=dict(iterations=list(pair),bindings=[dict(path=p,prefix=prefix) for p,_,prefix in records],
        scope='uniform_plant_and_robot_before_physics_parse',native_effective_iterations_verified=False)
    verify_iterations(stage,report)
    return report


def verify_iterations(stage,report):
    for binding in report['bindings']:
        prim=stage.GetPrimAtPath(binding['path'])
        observed=[prim.GetAttribute(binding['prefix']+':'+s).Get()
            for s in ('solverPositionIterationCount','solverVelocityIterationCount')]
        if observed!=report['iterations']:raise RuntimeError('Uniform USD solver iterations changed')


def author_contact_order(scene, *, enabled):
    """Author the documented custom attribute; readback is USD, not native.

    Caller must own a stopped/unparsed scene. NVIDIA's articulation stability
    guide recommends this numerical comparison for dynamic grasp contacts:
    https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html
    No material, iteration, collision filtering or force limit is altered.
    """
    from pxr import Sdf
    if type(enabled) is not bool:
        raise ValueError('Explicit boolean contact solver order required')
    attr=scene.GetAttribute('physxScene:solveArticulationContactLast')
    previous=attr.Get() if attr else None
    if enabled:
        scene.CreateAttribute('physxScene:solveArticulationContactLast',Sdf.ValueTypeNames.Bool).Set(True)
    observed=scene.GetAttribute('physxScene:solveArticulationContactLast').Get()
    if enabled and observed is not True:
        raise RuntimeError('Contact-last USD authoring failed')
    return dict(requested=enabled,previous_usd_value=previous,observed_usd_value=observed,
        authored=enabled,scope='owned stopped scene before native parse',
        native_effective_value_verified=False,source='NVIDIA articulation stability guide')
