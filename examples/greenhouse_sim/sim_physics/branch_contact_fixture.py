"""Explicit pre-physics component selection for a branch CONTACT experiment.

Keeps the complete original main stem, selected petiole and ALL its leaves.
Other source components are inactive only in this session. This is deliberately
NOT an intact-plant/greenhouse access, safety, fidelity or training certificate.
No mesh, transform, material, mass, constraint or collision filter is edited.
"""


def select_components(stage, record, component_id):
    from pxr import Sdf, Usd
    from sim_data.audit import audit_manifest, descendants

    if any((prim := stage.GetPrimAtPath(path)) and prim.IsActive() and prim.IsDefined()
           for path in ('/World/InteractionPhysics', '/World/RBY1')):
        raise ValueError('Branch fixture selection must precede rig and robot construction')
    audit = audit_manifest(record['manifest_path'])
    components = audit['components']
    target = next((t for t in audit['targets'] if t['component_id'] == component_id), None)
    if (target is None or target['status'] == 'excluded' or target['protected_descendant_ids']
            or components[component_id]['type'] != 'sub_stem'
            or components.get(components[component_id]['parent'], {}).get('type') != 'main_stem'):
        raise ValueError('Audited leaf-only petiole attached to a main stem required')
    members = set(target['expected_detached_component_ids'])
    if (component_id not in members or members != set(descendants(components, component_id))
            or any(components[k]['type'] != 'leaf' or components[k]['parent'] != component_id
                   for k in members if k != component_id)):
        raise ValueError('Keep the entire selected petiole and every attached leaf')
    keep = members | {k for k, c in components.items() if c['type'] == 'main_stem'}
    paths = record['component_paths']
    root = Sdf.Path(record['plant_root'])
    if (not root.IsAbsolutePath() or root == Sdf.Path('/World') or not root.HasPrefix('/World')
            or not stage.GetPrimAtPath(root) or set(paths) != set(components)
            or len(set(paths.values())) != len(paths)):
        raise ValueError('Complete unique source component inventory under the plant required')
    for path in paths.values():
        p = Sdf.Path(path)
        if (not p.IsPrimPath() or not p.IsAbsolutePath() or p == root or not p.HasPrefix(root)
                or not stage.GetPrimAtPath(p) or not stage.GetPrimAtPath(p).IsActive()):
            raise ValueError('Every source component must exist and be active before selection')
    removed = sorted(set(components) - keep)
    removed_paths = [Sdf.Path(paths[k]) for k in removed]
    if any(Sdf.Path(paths[k]).HasPrefix(p) for k in keep for p in removed_paths):
        raise ValueError('Cannot deactivate an ancestor of retained geometry')
    roots = [p for p in removed_paths if not any(p != q and p.HasPrefix(q) for q in removed_paths)]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in roots:
            stage.GetPrimAtPath(path).SetActive(False)
    if not all(stage.GetPrimAtPath(paths[k]).IsActive() for k in keep):
        raise RuntimeError('Retained branch component unexpectedly deactivated')
    return dict(scope='original_main_stem_and_one_complete_leaf_petiole_contact_fixture',
                target=component_id, manifest_sha256=audit['manifest_sha256'],
                retained_component_ids=sorted(keep), excluded_component_ids=removed,
                inactive_session_roots=sorted(map(str, roots)),
                all_target_leaves_retained=True, retained_geometry_or_physics_changed=False,
                source_layers_modified=False, intact_source_plant_present=False,
                full_greenhouse_qualified=False, access_safety_qualified=False, training_eligible=False)
