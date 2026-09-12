"""Pre-physics matched plant diagnostic; never greenhouse qualification.

Retains the ENTIRE source plant, floor, lights and materials at their existing
transforms. Only session-layer scene branches are deactivated before physics.
"""


def isolate(stage, *, plant_root='/World/Plant', floor_root):
    from pxr import Sdf, Usd, UsdLux, UsdShade
    roots = [Sdf.Path(plant_root), Sdf.Path(floor_root)]
    if (not stage.GetPrimAtPath('/World') or
            any(not p.IsAbsolutePath() or p == Sdf.Path('/World')
                or not p.HasPrefix('/World') or not stage.GetPrimAtPath(p) for p in roots)
            or roots[0].HasPrefix(roots[1]) or roots[1].HasPrefix(roots[0])):
        raise ValueError('Distinct existing plant and floor roots under /World required')
    resources = [p.GetPath() for p in stage.Traverse()
                 if p.IsA(UsdShade.Material) or p.IsA(UsdShade.Shader)
                 or p.HasAPI(UsdLux.LightAPI)]
    keep = roots + resources
    removed = []
    def visit(prim):
        path = prim.GetPath()
        if any(path.HasPrefix(p) for p in keep):
            return
        if any(p.HasPrefix(path) for p in keep):
            for child in list(prim.GetChildren()):
                visit(child)
        else:
            removed.append(str(path))
    visit(stage.GetPrimAtPath('/World'))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in removed:
            stage.GetPrimAtPath(path).SetActive(False)
    return dict(scope='isolated_full_source_plant_with_full_robot',
                removed_scene_roots=removed, retained_plant_root=plant_root,
                retained_floor_root=floor_root, plant_geometry_modified=False,
                full_greenhouse_qualified=False, training_eligible=False)
