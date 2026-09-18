"""Replace one anonymous full-144 foreground with authenticated generated anatomy.

Assembly only: no native launch, camera search, collision/visibility qualification,
annotation admission, or claim that conservative capsule radii are mesh thickness.
The original full-population contracts remain unchanged and are not emitted here.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .audit import audit_manifest, safe_asset
from .capture_contract import fingerprint
from .dataset_review import read_json, require, verify_bindings
from .depth_preview import sha256
from . import native848_fully_labeled_scene_v1 as full

SCHEMA = 'greenhouse.native848_generated_morphology_scene.v1'
CENSUS_SCHEMA = 'greenhouse.native848_generated_morphology_census.v1'
QUALIFICATION_SCHEMA = 'greenhouse.donor_whole_plant_morphology_qualification.v1'
GENERATOR_SHA256 = '0bf4a7d00423417a0abcde4cd66667c8f8d3b1a3e0b39262687a67f368a97080'
FULL_SCENE_SHA256 = 'f00841a7b7e246796bcd34e2c31ffd9d1ac4358a5294340d33c355c9239ddd17'
RADIUS_POLICY = 'conservative_warped_donor_capsule_bound_not_certified_mesh_radius'
CONTROLLED_VERSION = 'source_frame_proximal_preserved_rigid_leaf_static.v3'
CONTROLLED_CENSUS_SCHEMA = 'greenhouse.native848_controlled_morphology_census.v1'
CONTROLLED_RADIUS_POLICY = 'source_radii_preserved_pending_current_9mm_evidence'


def _remember(bindings, path, expected):
    path = Path(path).resolve()
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'Explicit SHA256 required')
    require(path.is_file() and sha256(path) == expected, 'Changed source: '+str(path))
    require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting source pins')
    bindings[str(path)] = expected
    return path


def authenticate_morphology(qualification_pin, donor_plan_pin, anchor, generated_report=None):
    """Read pinned receipt/assets and reconstruct, rather than trust, the report."""
    bindings = {}
    qp = _remember(bindings, qualification_pin['path'], qualification_pin['sha256'])
    pp = _remember(bindings, donor_plan_pin['path'], donor_plan_pin['sha256'])
    q, plan = read_json(qp), read_json(pp)
    family = anchor['source_family']
    require(q.get('schema') == QUALIFICATION_SCHEMA, 'Typed morphology receipt required')
    require(anchor['split'] == q.get('source_split') == 'train'
            and plan['family_assignments'].get(family) == 'train', 'TRAIN donor lineage required')
    require(q.get('source_family') == q.get('split_group') == q.get('original_source_family_cap_group') == family,
            'Donor family cannot be renamed')
    require(Path(anchor['source_collection_plan']).resolve() == pp, 'Anchor donor plan differs')
    for key in ('source_assets_unchanged', 'full_component_graph_preserved', 'source_deleaf_state_preserved'):
        require(q.get(key) is True, 'Required preservation evidence: '+key)
    for key in ('training_approved', 'native_capture_validated', 'rendered_geometry_qualified',
                'morphology_diversity_approved', 'new_independent_source_family',
                'review_decisions_inherited', 'capsules_are_certified_mesh_geometry'):
        require(q.get(key) is False, 'Unqualified generated state required: '+key)
    require(q.get('accepted_training_increment') == 0 and q.get('static_perception_only') is True
            and q.get('physics_supported') is False and q.get('capsule_radius_policy') == RADIUS_POLICY,
            'Static, uncertified geometry contract required')
    require(q.get('generated_geometry_hash') and q['generated_geometry_hash'] != q.get('source_geometry_hash'),
            'Identical donor geometry receipt rejected')
    for group in ('source_bindings', 'texture_source_bindings', 'implementation_hashes'):
        require(isinstance(q.get(group), dict), 'Missing receipt bindings: '+group)
        for path, expected in q[group].items():
            _remember(bindings, path, expected)
    generator = Path(__file__).with_name('plant_morphology_generator_v1.py').resolve()
    require(q['implementation_hashes'].get(str(generator)) == GENERATOR_SHA256, 'Unreviewed generator implementation')
    _remember(bindings, generator, GENERATOR_SHA256)
    _remember(bindings, full.__file__, FULL_SCENE_SHA256)
    _remember(bindings, __file__, sha256(__file__))
    require(q['source_bindings'].get(str(pp)) == donor_plan_pin['sha256'], 'Receipt donor plan differs')
    jobs = [j for j in plan['jobs'] if j['plant_family'] == family]
    require(len(jobs) == 1 and jobs[0]['split'] == 'train', 'One exact TRAIN donor job required')
    donor = Path(jobs[0]['source_manifest_path']).resolve()
    require(q['source_bindings'].get(str(donor)) == plan['source_bindings_sha256'].get(str(donor)),
            'Donor manifest does not match source plan')
    _remember(bindings, donor, q['source_bindings'].get(str(donor)))
    require(qp.name == 'qualification.json' and qp.parent.name == q.get('variant_id'), 'Variant directory mismatch')
    manifest_path = qp.parent/'manifest.json'
    outputs = q.get('output_hashes', {})
    require('manifest.json' in outputs, 'Generated manifest pin required')
    for relative, expected in outputs.items():
        _remember(bindings, safe_asset(qp.parent, relative), expected)
    manifest, original = read_json(manifest_path), read_json(donor)
    require(manifest.get('variant_id') == q['variant_id'] and manifest.get('source_family') == family
            and manifest.get('split_group') == family and manifest.get('source_split') == 'train',
            'Generated manifest lineage differs')
    require(manifest.get('capsule_radius_policy') == RADIUS_POLICY, 'Generated radius policy differs')
    graph = lambda rows: [(c['id'], c['parent'], c['type'], c.get('deleafed')) for c in rows]
    require(graph(manifest['components']) == graph(original['components']), 'Generated component graph changed')
    require(len(manifest['components']) == q['source_component_count'], 'Generated component count differs')
    for old, new in zip(original['components'], manifest['components']):
        path = safe_asset(donor.parent, old['file'])
        require(q['source_bindings'].get(str(path)) == plan['source_bindings_sha256'].get(str(path)),
                'Donor component not authenticated by original plan')
        _remember(bindings, path, q['source_bindings'].get(str(path)))
        require(new['file'] in outputs, 'Generated component not bound')
    report = audit_manifest(manifest_path)
    require(report['status'] != 'blocked' and report['plant_id'] == q['variant_id'], 'Generated structural audit failed')
    require(generated_report is None or generated_report == report, 'Supplied generated report differs from pinned assets')
    return dict(qualification=q, report=report, donor_manifest=donor, plan=plan, bindings=bindings,
                qualification_pin=dict(path=str(qp), sha256=qualification_pin['sha256']))



def authenticate_controlled(qualification_pin, donor_plan_pin, anchor, generated_report=None):
    """Replay v3 controlled assets; this is not native or physical-radius admission."""
    from .procedural_petiole_controlled_catalogue_v3 import load_for_inspection
    bindings = {}
    qp = _remember(bindings, qualification_pin['path'], qualification_pin['sha256'])
    pp = _remember(bindings, donor_plan_pin['path'], donor_plan_pin['sha256'])
    q, plan = read_json(qp), read_json(pp)
    family = anchor['source_family']
    require(q.get('version') == CONTROLLED_VERSION, 'Only proximal-preserving controlled v3 is supported')
    require(anchor['split'] == q.get('split') == 'train'
            and q.get('source_family') == q.get('split_group') == family
            and plan['family_assignments'].get(family) == 'train', 'Controlled TRAIN lineage differs')
    require(Path(q['source_plan_path']).resolve() == pp
            and q['source_plan_sha256'] == donor_plan_pin['sha256'], 'Controlled generator plan differs')
    scene_path = Path(anchor['source_collection_plan']).resolve()
    scene_sha = anchor['source_bindings'].get(str(scene_path))
    _remember(bindings, scene_path, scene_sha)
    scene_plan = read_json(scene_path)
    scene_plan_pin = dict(path=str(scene_path), sha256=scene_sha)
    require(scene_plan['family_assignments'] == plan['family_assignments'],
            'Generator and scene family assignments differ')
    scene_jobs = [j for j in scene_plan['jobs'] if j['plant_family'] == family]
    require(len(scene_jobs) == 1 and scene_jobs[0]['split'] == 'train',
            'One matching TRAIN scene donor job required')
    require(qp.name == 'qualification.json' and qp.parent.name == q.get('variant_id'),
            'Controlled variant directory mismatch')
    catalogue = load_for_inspection(qp.parent, pp)
    require(catalogue['schema_version'] == 'greenhouse.proximal_preserved_control_inspection_catalogue.v3'
            and catalogue['qualification_sha256'] == qualification_pin['sha256']
            and catalogue['source_plan_sha256'] == donor_plan_pin['sha256'], 'Controlled replay identity differs')
    require(catalogue['source_family'] == catalogue['split_group'] == family
            and catalogue['variant_id'] == q['variant_id']
            and catalogue['training_eligible'] is False and catalogue['source_cap_reset'] is False,
            'Controlled catalogue lineage or admission differs')
    require(len(catalogue['rows']) == 1 and catalogue['rejected'] == []
            and len(q['recipes']) == len(q['targets']) == 1, 'Controlled target held by actual replay')
    key = q['recipes'][0]['component_id']
    require(catalogue['rows'][0]['component_id'] == q['targets'][0]['component_id'] == key
            and q['targets'][0]['source_target_id'] == family+'/'+key,
            'Controlled target identity differs')
    proof = catalogue['proximal_mesh_identity']
    require(proof == q['proximal_mesh_identity'] and proof['protected_source_arc_m'] == [0., .030]
            and proof['complete_intersecting_faces_exact'] is True
            and proof['source_mesh_surface_equality_continuous_in_protected_halfspace'] is True
            and proof['distal_faces_remain_outside_protected_halfspace'] is True
            and proof['unchanged_numeric_radius_definition'] is True,
            'Exact proximal30mm source mesh preservation missing')
    for path, expected in q['source_bindings'].items():
        _remember(bindings, path, expected)
    for name, expected in q['code_sha256'].items():
        require(Path(name).name == name, 'Controlled code binding must be a module filename')
        _remember(bindings, Path(__file__).with_name(name), expected)
    for relative, expected in q['output_hashes'].items():
        _remember(bindings, safe_asset(qp.parent, relative), expected)
    _remember(bindings, full.__file__, FULL_SCENE_SHA256)
    _remember(bindings, __file__, sha256(__file__))
    jobs = [j for j in plan['jobs'] if j['plant_family'] == family]
    require(len(jobs) == 1 and jobs[0]['split'] == 'train', 'One original TRAIN donor required')
    donor = Path(jobs[0]['source_manifest_path']).resolve()
    require(Path(q['source_manifest_path']).resolve() == donor
            and Path(scene_jobs[0]['source_manifest_path']).resolve() == donor
            and q['source_bindings'].get(str(donor)) == plan['source_bindings_sha256'].get(str(donor))
            and q['source_bindings'].get(str(donor)) == scene_plan['source_bindings_sha256'].get(str(donor)),
            'Controlled donor manifest differs between generator and scene plans')
    _remember(bindings, donor, q['source_bindings'].get(str(donor)))
    report = catalogue['report']
    mp = qp.parent/'manifest.json'
    require(Path(report['manifest_path']).resolve() == mp and report['plant_id'] == qp.parent.name
            and report['manifest_sha256'] == q['output_hashes'].get('manifest.json'),
            'Controlled report must identify the actual generated directory')
    require(generated_report is None or generated_report == report, 'Supplied controlled report differs')
    original, generated = read_json(donor), read_json(mp)
    graph = lambda rows: [(r['id'], r['parent'], r['type'], r.get('deleafed')) for r in rows]
    require(graph(original['components']) == graph(generated['components']), 'Controlled full organ graph changed')
    for old, new in zip(original['components'], generated['components']):
        source = safe_asset(donor.parent, old['file'])
        require(q['source_bindings'].get(str(source)) == plan['source_bindings_sha256'].get(str(source))
                and q['source_bindings'].get(str(source)) == scene_plan['source_bindings_sha256'].get(str(source)),
                'Controlled donor component differs between generator and scene plans')
        _remember(bindings, source, q['source_bindings'].get(str(source)))
        require(q['output_hashes'].get(new['file']) == report['components'][new['id']]['asset_sha256'],
                'Controlled generated component is unbound')
    verify_bindings(bindings)
    return dict(qualification=q, report=report, donor_manifest=donor, plan=plan, bindings=bindings,
        qualification_pin=dict(path=str(qp), sha256=qualification_pin['sha256']),
        generated_geometry_hash=fingerprint(q['output_hashes']),
        radius_policy=CONTROLLED_RADIUS_POLICY, census_schema=CONTROLLED_CENSUS_SCHEMA,
        controlled_target_component_id=key, proximal_mesh_identity=proof,
        scene_plan=scene_plan, scene_plan_pin=scene_plan_pin,
        generator_source_collection_plan=dict(path=str(pp), sha256=donor_plan_pin['sha256']))


def replace_controlled_foreground(stage, population, *, qualification_pin, donor_plan_pin,
                                  anchor, generated_report=None):
    auth = authenticate_controlled(qualification_pin, donor_plan_pin, anchor, generated_report)
    return _replace_authenticated(stage, population, auth=auth,
                                  donor_plan_pin=donor_plan_pin, anchor=anchor)


def _layer_hash(layer, omit=None):
    from pxr import Sdf
    if omit is not None:
        copy = Sdf.Layer.CreateAnonymous('generated-invariance')
        copy.TransferContent(layer)
        if copy.GetPrimAtPath(omit):
            edits = Sdf.BatchNamespaceEdit(); edits.Add(omit, Sdf.Path.emptyPath)
            require(copy.Apply(edits), 'Cannot snapshot outside foreground')
        layer = copy
    return hashlib.sha256(layer.ExportToString().encode('utf-8')).hexdigest()


def _world(stage, path):
    from pxr import UsdGeom
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)), float).tolist()


def _population_snapshot(stage, population, anchor, plan):
    """Validate the supplied current population; freeze background identity/placement."""
    receipt = population['scene_policy_evidence']
    require(receipt['schema'] == full.CENSUS_SCHEMA and receipt['policy'] == full.policy('train'),
            'Original full144 TRAIN population required')
    require(receipt['deterministic_census_sha256'] == fingerprint({k:v for k,v in receipt.items()
            if k != 'deterministic_census_sha256'}), 'Original census fingerprint differs')
    require(receipt.get('complete_active_plant_anatomy') is True and receipt.get('source_assets_modified') is False,
            'Complete unchanged source anatomy required')
    records = {r['plant_root']:r for r in population['records']}
    variants = {r['plant_root']:r for r in population['variants']}
    census = {r['plant_root']:r for r in receipt['all_plant_roots']}
    roots = {str(p.GetPath()) for p in stage.GetPrimAtPath('/World/PackPlants').GetChildren()}
    require(len(records) == len(variants) == len(census) == len(roots) == 144
            and len(population['records']) == len(population['variants']) == len(receipt['all_plant_roots']) == 144
            and records.keys() == variants.keys() == census.keys() == roots, 'Exactly144 distinct full plant slots required')
    require(population['counts'] == receipt['active_counts']
            and population['counts']['component_plants'] == 144 and population['counts']['backdrop_instances'] == 0
            and population['counts']['components'] == sum(len(r['component_paths']) for r in records.values()),
            'Full population counts differ')
    primary = anchor['original_variant']['plant_root']
    require(variants.get(primary) == anchor['original_variant'], 'Original foreground variant differs')
    require(census[primary]['source_family'] == anchor['source_family'], 'Foreground donor differs')
    snapshots = {}
    for path in sorted(roots):
        prim = stage.GetPrimAtPath(path); row = census[path]; record = records[path]; variant = variants[path]
        require(prim.IsActive() and not prim.IsInstanceable(), 'Ordinary active plant slot required')
        family = row['source_family']
        require(row['source_split'] == 'train' and plan['family_assignments'].get(family) == 'train'
                and variant['source_plant_id'] == variant['split_group'] == family
                and row['source_plant_id'] == family, 'Mixed or forged background lineage')
        require(row['source_geometry_modified'] is False and variant['source_geometry_modified'] is False,
                'Original generated-free population required')
        require(row['component_count'] == len(record['component_paths']) and row['authenticated_component_plant'] is True,
                'Population component inventory differs')
        require(Path(record['manifest_path']).resolve() == Path(row['manifest_path']).resolve(), 'Record manifest differs')
        require(population['original_population_bindings'].get(str(Path(row['manifest_path']).resolve())) == row['manifest_sha256'],
                'Population manifest not bound')
        world = _world(stage, path)
        require(np.array_equal(world, row['plant_to_world_usd_row_vectors']), 'Population world transform differs')
        for component_path in record['component_paths'].values():
            require(component_path.startswith(path+'/') and stage.GetPrimAtPath(component_path).IsActive(),
                    'Component missing or assigned to another instance')
        if path != primary:
            snapshots[path] = dict(record=deepcopy(record), variant=deepcopy(variant), census=deepcopy(row), world=world)
    return primary, snapshots


def replace_foreground(stage, population, *, qualification_pin, donor_plan_pin, anchor, generated_report=None):
    """Mutate only one anonymous slot and return a separately typed population.

    Caller must retain the returned template_stages while using the USD stage.
    On a failed post-replacement invariant the original session layer is restored.
    This API intentionally does not produce an original full-scene policy/context.
    """
    auth = authenticate_morphology(qualification_pin, donor_plan_pin, anchor, generated_report)
    return _replace_authenticated(stage, population, auth=auth,
                                  donor_plan_pin=donor_plan_pin, anchor=anchor)


def _replace_authenticated(stage, population, *, auth, donor_plan_pin, anchor):
    """Shared anonymous-slot mutation; all 143-background checks and rollback retained."""
    from pxr import Sdf, Usd, UsdGeom, Gf
    require(stage.GetRootLayer().anonymous and stage.GetSessionLayer().anonymous, 'Anonymous stage required')
    bindings = dict(auth['bindings'])
    for path, expected in population['original_population_bindings'].items():
        _remember(bindings, path, expected)
    scene_plan_pin = auth.get('scene_plan_pin',
        dict(path=str(Path(donor_plan_pin['path']).resolve()), sha256=donor_plan_pin['sha256']))
    require(population['scene_policy_evidence']['source_collection_plan'] == scene_plan_pin,
            'Population actual scene plan differs')
    primary, backgrounds = _population_snapshot(stage, population, anchor, auth.get('scene_plan', auth['plan']))
    require(Path(next(r['manifest_path'] for r in population['records'] if r['plant_root'] == primary)).resolve()
            == auth['donor_manifest'], 'Foreground manifest differs from generator donor')
    donor_components = {c['id']:c for c in read_json(auth['donor_manifest'])['components']}
    foreground_record = next(r for r in population['records'] if r['plant_root'] == primary)
    for cid, path in foreground_record['component_paths'].items():
        asset = safe_asset(auth['donor_manifest'].parent, donor_components[cid]['file'])
        layers = {str(Path(spec.layer.realPath).resolve()) for spec in stage.GetPrimAtPath(path).GetPrimStack()
                  if not spec.layer.anonymous}
        require(str(asset) in layers, 'Current foreground is not original donor; sequential replacement is unsupported')
    session = stage.GetSessionLayer()
    require(session.GetPrimAtPath(primary), 'Foreground must be authored in anonymous session')
    original_layers = list(stage.GetUsedLayers())
    require(not any(not l.anonymous and l.dirty for l in original_layers), 'Source layer already dirty')
    before_layers = {l.identifier:_layer_hash(l, primary if l == session else None) for l in original_layers if l.anonymous}
    old_world = _world(stage, primary)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(primary))
    local, reset = xform.GetLocalTransformation(), xform.GetResetXformStack()
    template, relative = full._template(auth['report']['manifest_path'])
    for layer in template.GetUsedLayers():
        if not layer.anonymous:
            path = str(Path(layer.realPath).resolve())
            require(path in bindings, 'Unbound generated USD dependency: '+path)
    backup = Sdf.Layer.CreateAnonymous('foreground-rollback'); backup.TransferContent(session)
    old_edit = stage.GetEditTarget()
    try:
        stage.SetEditTarget(session)
        require(stage.RemovePrim(primary) and not stage.GetPrimAtPath(primary), 'Foreground has unsupported weaker opinions')
        prim = UsdGeom.Xform.Define(stage, primary).GetPrim()
        new_xform = UsdGeom.Xformable(prim); new_xform.AddTransformOp().Set(Gf.Matrix4d(local))
        new_xform.SetResetXformStack(reset)
        prim.GetReferences().AddReference(template.GetRootLayer().identifier, '/Plant'); prim.SetInstanceable(False)
        require(np.allclose(_world(stage, primary), old_world, rtol=0, atol=1e-12), 'Foreground slot moved')
        for path, snapshot in backgrounds.items():
            require(np.array_equal(_world(stage, path), snapshot['world']), 'Background world transform changed')
        for layer in original_layers:
            if layer.anonymous:
                require(_layer_hash(layer, primary if layer == session else None) == before_layers[layer.identifier],
                        'Anonymous content outside foreground changed')
        verify_bindings(bindings)
        require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()), 'Source layer modified')
        q = auth['qualification']; morphology = q['variant_id']; family = q['source_family']
        paths = {cid:primary+path[len('/Plant'):] for cid,path in relative.items()}
        require(set(paths) == set(auth['report']['components']) and all(stage.GetPrimAtPath(p).IsActive() for p in paths.values()),
                'Generated component graph missing after composition')
        result = {k:v for k,v in population.items() if k not in ('records','variants','counts','scene_policy_evidence','template_stages','original_population_bindings')}
        result.update(schema=SCHEMA, records=deepcopy(population['records']), variants=deepcopy(population['variants']),
                      counts=deepcopy(population['counts']), template_stages=dict(population['template_stages']),
                      source_bindings=bindings, original_population_bindings=dict(population['original_population_bindings']))
        identity = dict(morphology_id=morphology, geometry_source_id=morphology, donor_source_family=family,
                        source_family=family, split_group=family, source_split='train', source_geometry_modified=True,
                        original_variant_id=anchor['original_variant']['variant_id'],
                        generated_geometry_hash=auth.get('generated_geometry_hash', q.get('generated_geometry_hash')))
        record = dict(plant_root=primary, manifest_path=auth['report']['manifest_path'], component_paths=paths, **identity)
        variant = dict(plant_root=primary, variant_id=morphology, source_plant_id=family,
                       added_components={}, added_component_paths={}, **identity)
        result['records'] = [record if r['plant_root'] == primary else r for r in result['records']]
        result['variants'] = [variant if r['plant_root'] == primary else r for r in result['variants']]
        result['template_stages'][morphology] = (template, relative)
        roots = deepcopy(population['scene_policy_evidence']['all_plant_roots'])
        foreground = next(r for r in roots if r['plant_root'] == primary)
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render']).ComputeWorldBound(prim).ComputeAlignedRange()
        require(not box.IsEmpty(), 'Empty generated foreground')
        foreground.update(identity, variant_id=morphology, manifest_path=auth['report']['manifest_path'],
            manifest_sha256=auth['report']['manifest_sha256'], source_asset_paths=sorted(str(safe_asset(Path(auth['report']['manifest_path']).parent,c['file']))
                for c in auth['report']['components'].values()), decision='generated_foreground_pending_current_geometry_and_native_qualification',
            plant_to_world_usd_row_vectors=_world(stage, primary), world_bounds_m=dict(min=list(box.GetMin()),max=list(box.GetMax())))
        status = dict(source_geometry_modified=True, source_assets_modified=False, source_assets_unchanged=True,
            dataset_split='train', full_component_graph_preserved=True, complete_active_plant_anatomy=True,
            unchanged_background_count=143, all_original_slots_retained=True, removed_roots=[],
            native_capture_validated=False, native_population_census_verified=False, current_9mm_geometry_qualified=False,
            capsule_radius_policy=auth.get('radius_policy', RADIUS_POLICY), capsules_are_certified_mesh_geometry=False,
            morphology_diversity_approved=False, new_independent_source_family=False,
            training_approved=False, accepted_training_increment=0, review_decisions_inherited=False)
        census = dict(schema=auth.get('census_schema', CENSUS_SCHEMA), **status, foreground_identity=identity, foreground_root=primary,
            active_counts=result['counts'], all_plant_roots=roots, source_qualification=auth['qualification_pin'],
            original_census_sha256=population['scene_policy_evidence']['deterministic_census_sha256'],
            background_invariance_sha256=fingerprint(backgrounds), source_collection_plan=deepcopy(scene_plan_pin),
            implementation=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)))
        if 'controlled_target_component_id' in auth:
            census.update(controlled_target_component_id=auth['controlled_target_component_id'],
                generator_source_collection_plan=deepcopy(auth['generator_source_collection_plan']),
                proximal_source_mesh_identity=auth['proximal_mesh_identity'],
                physical_9mm_evidence_authenticated_by_scene=False)
        census['deterministic_census_sha256'] = fingerprint(census)
        lineage = {cid:dict(target_id=morphology+'/'+cid, component_id=cid, source_component_id=cid,
            source_target_id=family+'/'+cid, source_family=family, geometry_source_id=morphology,
            morphology_id=morphology, component_path=path) for cid,path in paths.items()}
        result.update(status, scene_policy_evidence=census, generated_report=auth['report'], foreground_identity=identity,
                      generated_component_lineage=lineage)
        return result
    except BaseException:
        session.TransferContent(backup)
        raise
    finally:
        stage.SetEditTarget(old_edit)
