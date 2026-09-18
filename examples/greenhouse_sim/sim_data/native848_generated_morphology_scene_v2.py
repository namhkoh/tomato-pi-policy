"""V4 multi-subtree authentication with unchanged full144 scene replacement."""
from pathlib import Path
from . import native848_generated_morphology_scene_v1 as predecessor
from .dataset_review import read_json,require,verify_bindings
from .depth_preview import sha256
from .audit import safe_asset
from .capture_contract import fingerprint
from . import native848_fully_labeled_scene_v1 as full
CONTROLLED_VERSION='source_frame_multisubtree_signed_control_static.v4'
CONTROLLED_CENSUS_SCHEMA=predecessor.CONTROLLED_CENSUS_SCHEMA
CONTROLLED_RADIUS_POLICY=predecessor.CONTROLLED_RADIUS_POLICY
FULL_SCENE_SHA256=predecessor.FULL_SCENE_SHA256
_remember=predecessor._remember
_replace_authenticated=predecessor._replace_authenticated

def authenticate_controlled(qualification_pin, donor_plan_pin, anchor, generated_report=None):
    """Replay v4 multi-subtree controlled assets; this is not native or physical-radius admission."""
    from .procedural_petiole_controlled_catalogue_v4 import load_for_inspection
    bindings = {}
    qp = _remember(bindings, qualification_pin['path'], qualification_pin['sha256'])
    pp = _remember(bindings, donor_plan_pin['path'], donor_plan_pin['sha256'])
    q, plan = read_json(qp), read_json(pp)
    family = anchor['source_family']
    require(q.get('version') == CONTROLLED_VERSION, 'Only proximal-preserving controlled v4 is supported')
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
    require(catalogue['schema_version'] == 'greenhouse.multisubtree_controlled_inspection_catalogue.v4'
            and catalogue['qualification_sha256'] == qualification_pin['sha256']
            and catalogue['source_plan_sha256'] == donor_plan_pin['sha256'], 'Controlled replay identity differs')
    require(catalogue['source_family'] == catalogue['split_group'] == family
            and catalogue['variant_id'] == q['variant_id']
            and catalogue['training_eligible'] is False and catalogue['source_cap_reset'] is False,
            'Controlled catalogue lineage or admission differs')
    keys=[r['component_id'] for r in q['recipes']]
    require(keys and len(keys)==len(set(keys)) and catalogue['rejected']==[]
            and set(keys)=={r['component_id'] for r in catalogue['rows']}
            and len(catalogue['rows'])==len(q['targets'])==len(keys), 'Controlled target held by actual replay')
    key=q['primary_component_id']
    require(key in keys and q['primary_source_target_id']==family+'/'+key,
            'Primary controlled target identity differs')
    require({r['component_id']:r['source_target_id'] for r in q['targets']}=={k:family+'/'+k for k in keys},
            'Controlled source target lineage differs')
    proofs=catalogue['proximal_mesh_identity_by_component']
    require(proofs==q['proximal_mesh_identity_by_component'] and set(proofs)==set(keys),
            'Every modified petiole requires proximal proof')
    for item in proofs.values():
        require(item['protected_source_arc_m']==[0.,.030]
                and item['complete_intersecting_faces_exact'] is True
                and item['source_mesh_surface_equality_continuous_in_protected_halfspace'] is True
                and item['distal_faces_remain_outside_protected_halfspace'] is True
                and item['unchanged_numeric_radius_definition'] is True,
                'Exact proximal30mm source mesh preservation missing')
    proof=proofs[key]
    for path, expected in q['source_bindings'].items():
        _remember(bindings, path, expected)
    for name, expected in q['code_sha256'].items():
        require(Path(name).name == name, 'Controlled code binding must be a module filename')
        _remember(bindings, Path(__file__).with_name(name), expected)
    for relative, expected in q['output_hashes'].items():
        _remember(bindings, safe_asset(qp.parent, relative), expected)
    _remember(bindings, full.__file__, FULL_SCENE_SHA256)
    _remember(bindings, __file__, sha256(__file__))
    _remember(bindings, predecessor.__file__, sha256(predecessor.__file__))
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

