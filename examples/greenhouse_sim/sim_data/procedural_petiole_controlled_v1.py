"""Two predeclared source-frame controls; static CPU assets, never novelty credit.

Only the requested petiole and its direct leaves can change visible attributes.
All other authored geometry/shading attributes are copied verbatim. Physics
removal is reported separately. There is no seed, amplitude search or metric.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import traceback
import numpy as np
from .audit import audit_manifest, descendants, safe_asset
from .cut_regions import load_rule, propose_cut_region
from .plant_variants import (PHYSICS_FIELDS, digest, file_hash, load_training_sources,
    training_envelope, parent_tangent, dimensions, normalized)
from .plant_variant_usd import cut_surface_probe, write_json_new
from .procedural_petiole_geometry import require, unit
from .procedural_petiole_usd import donor_curve, deform_new_copy
from .procedural_petiole_v2 import transform_metadata
from .procedural_petiole_warp import CurveWarp
from .procedural_leaf_transport import component_transport
from .procedural_petiole_catalogue import check_component

VERSION = 'source_frame_controlled_rigid_leaf_static.v1'
CONTROL_SCHEMA = 'greenhouse.source_frame_explicit_displacement_control.v1'
DIRECTION = 'nearest_parent_tangent_cross_source_proximal_tangent'
TAPER = 'smoothstep_normalized_source_arc_3t2_minus_2t3'
AMPLITUDES = (.002, .025)


def control_for(amplitude_m):
    require(type(amplitude_m) in (int, float) and amplitude_m in AMPLITUDES,
        'Only the two predeclared 2mm/25mm controls are supported')
    return dict(schema=CONTROL_SCHEMA, amplitude_m=float(amplitude_m), direction_policy=DIRECTION,
        taper_policy=TAPER, source_attachment_preserved=True, sampled_source_radii_preserved=True,
        seed_search=False, amplitude_search=False, visual_label=None)


def controlled_curve(old_points, radii, parent_axis, amplitude_m):
    control_for(amplitude_m)
    old = np.asarray(old_points, float); radii = np.asarray(radii, float)
    require(old.ndim == 2 and old.shape[1] == 3 and len(old) >= 2
        and radii.shape == (len(old),) and np.isfinite(old).all()
        and np.isfinite(radii).all() and (radii > 0).all(), 'Finite source centerline/radii required')
    lengths = np.linalg.norm(np.diff(old, axis=0), axis=1)
    require((lengths > 1e-9).all(), 'Nondegenerate source centerline required')
    arc = np.r_[0., np.cumsum(lengths)]; tangent = unit(old[1]-old[0])
    direction = unit(np.cross(unit(parent_axis), tangent))
    t = arc/arc[-1]; weights = t*t*(3-2*t)
    relative = old-old[0]
    points = relative+float(amplitude_m)*weights[:, None]*direction
    new_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    require(np.array_equal(points[0], np.zeros(3)) and np.all(np.diff(new_arc) > 1e-9),
        'Controlled curve moved attachment or became degenerate')
    return dict(points=points, radius=radii.copy(), arc=new_arc), direction


def plan_change_explicit(source, key, envelope, control):
    require(control == control_for(control['amplitude_m']), 'Explicit control fields changed')
    report = source['report']; component = report['components'][key]
    parent = report['components'][component['parent']]
    require(key in {t['component_id'] for t in source['job']['targets']}
        and component['type'] == 'sub_stem' and component['deleafed'] is False
        and parent['type'] == 'main_stem', 'Frozen intact direct-main-stem target required')
    members = descendants(report['components'], key)
    require(any(m != key for m in members)
        and all(m == key or (report['components'][m]['type'] == 'leaf'
            and report['components'][m]['parent'] == key) for m in members),
        'Only target petiole and its direct leaves are supported')
    old = donor_curve(component); old_absolute = old['points']+component['translation_plant_m']
    anchor = np.asarray(component['attachment_plant_m'], float)
    require(np.allclose(old_absolute[0], anchor, atol=1e-9, rtol=0), 'Source chain does not start at source attachment')
    axis = parent_tangent(component, parent)
    curve, direction = controlled_curve(old_absolute, old['radius'], axis, control['amplitude_m'])
    warp = CurveWarp(old_absolute, curve, anchor, radius_scale=1.)
    change = dict(component_id=key, members=members, curve=curve, warp=warp, control=deepcopy(control),
        direction=direction.tolist(), parent_tangent_plant=axis.tolist(), new_anchor_m=anchor.tolist(),
        source_anchor_m=anchor.tolist(), source_target_id=source['job']['plant_family']+'/'+key,
        source_curve_sha256=digest(dict(points=old_absolute.tolist(), radii=old['radius'].tolist())),
        controlled_curve_sha256=digest(dict(points=curve['points'].tolist(), radii=curve['radius'].tolist(), arc=curve['arc'].tolist())),
        source_centerline_length_m=float(old['arc'][-1]), controlled_centerline_length_m=float(curve['arc'][-1]),
        maximum_centerline_displacement_m=float(np.max(np.linalg.norm(curve['points']-(old_absolute-old_absolute[0]), axis=1))),
        source_radius_scale=1., attachment_relocation_m=0., new_independent_donor_family=False,
        novelty_admission_approved=False)
    raw = next(r for r in source['raw']['components'] if r['id'] == key)
    metadata = transform_metadata(raw, change)
    measured, _, _ = dimensions(normalized(metadata), parent, load_rule())
    require(all(envelope['bounds'][k][0] <= value <= envelope['bounds'][k][1] for k,value in measured.items()),
        'Predeclared control outside frozen TRAIN envelope; hold without adjustment')
    require(metadata['attach_point'] == raw['attach_point']
        and np.array_equal(curve['radius'], old['radius']), 'Source attachment/radii changed')
    change['training_envelope_measurements'] = measured
    return change


def _physics(name):
    return name.startswith(('physics:', 'physx'))


def _same(a, b):
    from pxr import Sdf
    if isinstance(a, Sdf.AssetPath):
        return isinstance(b, Sdf.AssetPath) and a.path == b.path
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except ValueError:
        return np.array_equal(a, b)


def copy_static_component(source_path, output_path, source_root, copied_textures):
    """Strip physics on a new anonymous copy; do not normalize unchanged normals."""
    from pxr import Sdf, Usd, UsdGeom
    source = Usd.Stage.Open(str(source_path)); require(bool(source) and source.GetDefaultPrim(), 'Valid source USD required')
    require(UsdGeom.GetStageMetersPerUnit(source) == 1 and UsdGeom.GetStageUpAxis(source) == 'Z'
        and not source.GetRootLayer().subLayerPaths, 'Native meter/Z single-layer component required')
    layer = Sdf.Layer.CreateAnonymous('controlled_static.usda'); layer.TransferContent(source.GetRootLayer())
    stage = Usd.Stage.Open(layer); removed_apis = []; removed_attributes = []; removed_relationships = []; meshes = 0
    for prim in stage.Traverse():
        require(not prim.HasAuthoredReferences() and not prim.HasAuthoredPayloads() and not prim.IsInstanceable()
            and prim.GetTypeName() in ('Xform','Mesh','Scope','Material','Shader','GeomSubset'),
            'Unsupported composed/animated/static component')
        transform = UsdGeom.Xformable(prim)
        if transform:
            require(np.allclose(np.asarray(transform.GetLocalTransformation()), np.eye(4), atol=1e-12, rtol=0),
                'Source component mesh transform must be identity')
        schemas = list(prim.GetAppliedSchemas())
        kept = [s for s in schemas if 'physics' not in s.lower() and 'physx' not in s.lower()]
        removed_apis.extend(dict(prim=str(prim.GetPath()), schema=s) for s in schemas if s not in kept)
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(kept))
        for relationship in list(prim.GetRelationships()):
            if _physics(relationship.GetName()) and relationship.IsAuthored():
                removed_relationships.append(dict(prim=str(prim.GetPath()), relationship=relationship.GetName()))
                prim.RemoveProperty(relationship.GetName())
        for attr in list(prim.GetAuthoredAttributes()):
            require(not attr.GetNumTimeSamples(), 'Animated source component unsupported')
            name = attr.GetName()
            if _physics(name):
                removed_attributes.append(dict(prim=str(prim.GetPath()), attribute=name)); prim.RemoveProperty(name); continue
            value = attr.Get()
            require(attr.GetTypeName() != Sdf.ValueTypeNames.AssetArray, 'Asset arrays unsupported')
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset and value is not None:
                dependency = safe_asset(source_root, value.path)
                destination = safe_asset(output_path.parent, value.path)
                require(dependency.is_file() and destination != output_path, 'Invalid source material dependency')
                sha = file_hash(dependency)
                if destination not in copied_textures:
                    require(not destination.exists(), 'No dependency overwrite')
                    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(dependency, destination)
                    copied_textures[destination] = sha
                require(copied_textures[destination] == sha, 'Conflicting texture dependencies')
        meshes += int(prim.IsA(UsdGeom.Mesh))
    require(meshes > 0 and not output_path.exists(), 'New copied mesh output required')
    require(layer.Export(str(output_path)), 'Static component export failed')
    return dict(component_source_sha256=file_hash(source_path), removed_physics_apis=removed_apis,
        removed_physics_attributes=removed_attributes, removed_physics_relationships=removed_relationships,
        visible_attributes_not_rewritten=True)


def verify_visible_component(source_path, generated_path, *, spatial_changes_allowed):
    """Require exact authored non-physics values outside the explicit mesh warp."""
    from pxr import Usd
    source, generated = Usd.Stage.Open(str(source_path)), Usd.Stage.Open(str(generated_path))
    require(bool(source) and bool(generated), 'Unreadable source/generated component')
    old_prims = {str(p.GetPath()): p for p in source.Traverse()}
    new_prims = {str(p.GetPath()): p for p in generated.Traverse()}
    require(old_prims.keys() == new_prims.keys(), 'Changed visible prim population')
    changed = []; attributes = 0
    for path, old in old_prims.items():
        new = new_prims[path]
        require(old.GetTypeName() == new.GetTypeName(), 'Changed prim type')
        expected_schemas = [s for s in old.GetAppliedSchemas() if 'physics' not in s.lower() and 'physx' not in s.lower()]
        require(list(new.GetAppliedSchemas()) == expected_schemas, 'Only physics schemas may be stripped')
        for key, value in old.GetAllAuthoredMetadata().items():
            if key not in ('apiSchemas',):
                require(_same(value, new.GetMetadata(key)), 'Changed visible prim metadata: '+key)
        old_rel = {r.GetName(): r for r in old.GetRelationships() if not _physics(r.GetName())}
        new_rel = {r.GetName(): r for r in new.GetRelationships() if not _physics(r.GetName())}
        require(old_rel.keys() == new_rel.keys(), 'Changed relationship population')
        for name, rel in old_rel.items():
            require(rel.GetTargets() == new_rel[name].GetTargets(), 'Changed material/subset binding')
        old_attrs = {a.GetName(): a for a in old.GetAuthoredAttributes() if not _physics(a.GetName())}
        new_attrs = {a.GetName(): a for a in new.GetAuthoredAttributes() if not _physics(a.GetName())}
        require(old_attrs.keys() == new_attrs.keys()
            and not any(_physics(a.GetName()) for a in new.GetAuthoredAttributes()), 'Changed visible attribute population')
        for name, attr in old_attrs.items():
            actual = new_attrs[name]; attributes += 1
            require(attr.GetTypeName() == actual.GetTypeName(), 'Changed attribute type')
            spatial = spatial_changes_allowed and name in ('points', 'normals', 'extent')
            for key, value in attr.GetAllAuthoredMetadata().items():
                if not (spatial and key == 'default'):
                    require(_same(value, actual.GetMetadata(key)), 'Changed source attribute metadata: '+name+'/'+key)
            if not _same(attr.Get(), actual.Get()):
                require(spatial, 'Changed protected visible channel: '+name)
                changed.append(dict(prim=path, attribute=name))
    return dict(visible_attributes_checked=attributes, changed_visible_attributes=changed,
        protected_visible_attributes_exactly_preserved=True, physics_excluded_and_separately_reported=True)


def verify_components(source, change, directory, raw, report):
    old_rows = {c['id']: c for c in source['raw']['components']}; new_rows = {c['id']: c for c in raw['components']}
    require(old_rows.keys() == new_rows.keys(), 'Changed source organ population')
    source_root = Path(source['report']['manifest_path']).parent
    rows = []; serialized = []; changed = []
    for key, old in old_rows.items():
        selected = key in change['members']
        expected = transform_metadata(old, change) if selected else {k:v for k,v in old.items() if k not in PHYSICS_FIELDS}
        require(new_rows[key] == expected, 'Changed source/replayed manifest geometry: '+key)
        source_path = safe_asset(source_root, old['file']); actual = safe_asset(directory, report['components'][key]['file'])
        visible = verify_visible_component(source_path, actual, spatial_changes_allowed=selected)
        if visible['changed_visible_attributes']: changed.append(key)
        rows.append(dict(component_id=key, **visible))
        if selected:
            qa = check_component(source_path, actual, np.asarray(old['transform']['translate']),
                np.asarray(expected['transform']['translate']), component_transport(old, change))
            serialized.extend(dict(component_id=key, **q) for q in qa)
    require(change['component_id'] in changed and set(changed) <= set(change['members']),
        'Control did not change actual target mesh or changed a protected component')
    return dict(allowed_changed_components=sorted(change['members']), changed_visible_components=sorted(changed),
        protected_component_count=len(old_rows)-len(change['members']), component_attribute_checks=rows,
        target_subtree_only_visible_change_verified=True), serialized


def code_bindings():
    names = ('procedural_petiole_controlled_v1.py','procedural_petiole_controlled_catalogue_v1.py',
        'procedural_petiole_geometry.py','procedural_petiole_warp.py','procedural_petiole_usd.py',
        'procedural_petiole_v2.py','procedural_leaf_transport.py','procedural_petiole_catalogue.py',
        'plant_variant_usd.py','plant_variants.py','audit.py','cut_regions.py','geometry.py')
    return {name:file_hash(Path(__file__).with_name(name)) for name in names}


def generate(plan_path, family, target, control, output):
    from pxr import Usd
    from .geometry import audit_geometry
    output = Path(output).resolve(); require(not output.exists(), 'New controlled output only')
    plan,sources = load_training_sources(plan_path); require(family in sources, 'Frozen TRAIN family required')
    source = sources[family]; envelope = training_envelope(plan, sources)
    source_path = Path(source['report']['manifest_path']).resolve()
    require(not output.is_relative_to(source_path.parent) and not source_path.parent.is_relative_to(output), 'Output must be disjoint from source')
    output.mkdir(parents=True)
    write_json_new(output/'intent.json', dict(version=VERSION, source_family=family, component_id=target,
        source_plan_path=str(Path(plan_path).resolve()), source_plan_sha256=file_hash(plan_path),
        control=control, training_eligible=False, native_launched=False, amplitude_search=False))
    try:
        change = plan_change_explicit(source, target, envelope, control)
        raw = deepcopy(source['raw']); raw.update(generator=VERSION, version='1.0.0', physics_supported=False,
            leaf_transport_policy='rigid_leaf_blade.v1', intended_use='explicit_unlabelled_static_geometry_control_pending_native_review')
        for key in PHYSICS_FIELDS: raw.pop(key, None)
        rows = []; textures = {}; copies = {}; derivative_checks = []; output_hashes = {}
        for old in source['raw']['components']:
            selected = old['id'] in change['members']
            row = transform_metadata(old, change) if selected else deepcopy(old)
            for key in PHYSICS_FIELDS: row.pop(key, None)
            destination = safe_asset(output, old['file']); destination.parent.mkdir(parents=True, exist_ok=True)
            copies[old['id']] = copy_static_component(safe_asset(source_path.parent, old['file']), destination, source_path.parent, textures)
            if selected:
                checks = deform_new_copy(destination, np.asarray(old['transform']['translate']),
                    np.asarray(row['transform']['translate']), component_transport(old, change))
                derivative_checks.extend(dict(component_id=old['id'], **q) for q in checks)
            rows.append(row); output_hashes[old['file']] = file_hash(destination)
        raw['components'] = rows; raw['component_count'] = len(rows); write_json_new(output/'manifest.json', raw)
        report = audit_manifest(output/'manifest.json'); require(report['status'] != 'blocked', 'Generated anatomy audit blocked')
        component = report['components'][target]; parent = report['components'][component['parent']]
        measured, _, _ = dimensions(component, parent, load_rule())
        require(measured == change['training_envelope_measurements']
            and all(envelope['bounds'][k][0] <= v <= envelope['bounds'][k][1] for k,v in measured.items()), 'Serialized output left TRAIN envelope')
        proposal = propose_cut_region(component, parent, load_rule())
        require(proposal['status'] == 'proposed_geometry_only' and not proposal['geometry_warnings'], 'Controlled cut interval has geometry warnings')
        surface = cut_surface_probe(Usd.Stage.Open(str(output/component['file'])), component, parent)
        require(surface['passed'], 'Controlled cut interval does not pass actual surface probes')
        geometry = audit_geometry(deepcopy(report))
        require(geometry['maximum_translation_error_m'] <= 1e-6, 'Controlled component assembly frame differs')
        warnings = sorted({w['code'] for w in geometry['warnings'] if w['component_id'] in change['members']})
        require(not warnings, 'Controlled subtree geometry warnings: '+str(warnings))
        proof, serialized = verify_components(source, change, output, raw, report)
        source_bindings = {str(source_path):file_hash(source_path)}
        for component in source['report']['components'].values():
            path = safe_asset(source_path.parent, component['file'])
            require(file_hash(path) == component['asset_sha256'], 'Source mesh changed during generation')
            source_bindings[str(path)] = component['asset_sha256']
        for path,sha in textures.items():
            relative = path.relative_to(output).as_posix(); output_hashes[relative] = sha
            donor = safe_asset(source_path.parent, relative)
            require(file_hash(donor) == sha, 'Source material texture changed'); source_bindings[str(donor)] = sha
        output_hashes['manifest.json'] = file_hash(output/'manifest.json')
        recipe = {k:v for k,v in change.items() if k not in ('curve','warp')}
        target_receipt = dict(component_id=target, source_target_id=change['source_target_id'],
            conservative_view_cap_group=change['source_target_id'], cut_region_proposal=proposal, cut_surface_probe=surface,
            shape_novelty_pending=True, training_eligible=False)
        receipt = dict(version=VERSION, state='cpu_explicit_control_rigid_leaf_pending_native_review', source_family=family,
            split='train', split_group=family, variant_id=output.name, source_plan_path=str(Path(plan_path).resolve()),
            source_plan_sha256=file_hash(plan_path), source_manifest_path=str(source_path),
            frozen_family_assignments=plan['family_assignments'], source_bindings=source_bindings, output_hashes=output_hashes,
            recipes=[recipe], targets=[target_receipt], explicit_control=control, leaf_transport_policy='rigid_leaf_blade.v1',
            static_copy_receipts=copies, protected_attribute_proof=proof, serialized_geometry_checks=serialized,
            mesh_derivative_diagnostics=derivative_checks, training_envelope_sha256=digest(envelope),
            training_envelope_bounds=envelope['bounds'], training_envelope_measurements=measured,
            source_assets_unchanged=True, training_eligible=False, native_capture_validated=False, physics_validated=False,
            independent_target_novelty_approved=False, new_donor_family_created=False, source_cap_reset=False,
            geometry_distances_computed=False, visual_label=None, visual_review_performed=False,
            code_sha256=code_bindings())
        write_json_new(output/'qualification.json', receipt)
        print('EXPLICIT_CONTROL_CPU_QUALIFIED', output.name, control['amplitude_m'], flush=True)
        return receipt
    except BaseException as exc:
        write_json_new(output/'FAILED.json', dict(state='predeclared_control_held_without_search',
            error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc(), control=control,
            training_eligible=False, native_launched=False, amplitude_adjusted=False, seed_search=False))
        raise
