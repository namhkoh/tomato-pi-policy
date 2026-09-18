"""Bounded multi-target generation from reviewed original-native bank v2.

One original donor/current greenhouse/appearance group, one generated plant,
1..12 distinct targets, and 1..6 existing mounted-robot pose proposals each.
Full native rendering, geometry, visibility and admission are still required.
No legacy paired848 proof, source-cap reset, or execution authority is invented.
"""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np

from ..native_dataset import original_reference_bank_v2 as bank_api
from ..native_original_capture import contracts as oc
from ..native_view_plan import propose_specs

SCHEMA = 'greenhouse.generated_original_native_batch_plan.v2'
STATE = 'cpu_prepared_batch_pending_bound_native_producer'
PROFILE = 'reviewed_original_current_scene_mounted_views_reference56.v2'
FLAGS = dict(training_approved=False, source_cap_reset=False, native_launched=False,
    independent_geometry_qualification=False, generated_native_qualification=False,
    physical_motion_commanded=False, paired_848_1696_proof=False,
    hidden_cut_coordinates_executable=False, native_launch_supported=False)


def _closure():
    root = Path(__file__).resolve().parent
    search = (root.parent.parent, root.parents[2])
    pending, found = [Path(__file__).resolve(), root / 'batch_scene_v2.py'], {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes()
        found[str(path)] = oc.digest(raw)
        base = next(p for p in search if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.' * node.level + (node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name] + [name + '.' + a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in search:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate / '__init__.py') if p.is_file())
    return dict(sorted(found.items()))


_LOADED = _closure()


def implementation_bindings():
    oc.bind_all(_LOADED)
    bank_api.implementation_bindings()
    return dict(_LOADED)


def _bank(path, pin, entry_ids):
    path = oc.pin(path, pin)
    bank = bank_api.check_bank(oc.read_json(path))
    oc.require(isinstance(entry_ids, list) and 1 <= len(entry_ids) <= 12
        and all(isinstance(s, str) and s for s in entry_ids)
        and len(set(entry_ids)) == len(entry_ids), 'One-to-twelve explicit unique reference entries required')
    by_id = {e['id']: e for e in bank['entries']}
    oc.require(all(key in by_id for key in entry_ids), 'Unknown reference entry')
    entries = [deepcopy(by_id[key]) for key in entry_ids]
    oc.require(all(e['reference_status']['usable_as_original_reference'] is True
        and e['native_decision'] == bank_api.inv.STRICT and e['native_observation'] is not None
        and e['split'] == 'train' and oc.FROZEN_SPLITS.get(e['source_family']) == 'train' for e in entries),
        'Every selected target needs a reviewed native original reference')
    oc.require(len({e['target_id'] for e in entries}) == len(entries), 'Repeated target cannot expand a batch')
    anchor = entries[0]
    for entry in entries:
        oc.require(entry['source_family'] == anchor['source_family']
            and entry['compatibility_group'] == anchor['compatibility_group']
            and entry['compatibility_basis'] == anchor['compatibility_basis'], 'References have incompatible current scenes')
        native = entry['native_observation']
        oc.require(np.allclose(native['robot_snapshot']['camera_to_head_column_vectors'],
            anchor['native_observation']['robot_snapshot']['camera_to_head_column_vectors'], atol=1e-9, rtol=0),
            'Actual mounted camera differs despite grouping index')
        oc.require(all(native['calibration'][k] == anchor['native_observation']['calibration'][k]
            for k in bank_api._OPTICS), 'Actual native optics differ')
    oc.pin(path, pin)
    return bank, entries


def _generate(clear, family, components, seed, directory):
    from ..procedural_petiole_v2 import generate
    return generate(clear, family, components, seed, directory)


def _catalogue(directory, clear):
    from ..plant_variant_catalogue import load_for_inspection
    return load_for_inspection(directory, clear)


def _case(entry, row, seed_key, views):
    from ..capture_contract import transform_points, project
    scene, native = entry['scene_authority'], entry['native_observation']
    source = scene['source_row']
    meta = oc.read_json(oc.pin(native['sample']['path'], native['sample']['sha256']))
    oc.require(meta['schema_version'] == oc.SAMPLE_SCHEMA and meta['training_sample_approved'] is False
        and meta['historical_labels_inherited'] is False and meta['supervision']['target_id'] == entry['target_id']
        and meta['calibration'] == native['calibration'] and meta['robot_snapshot'] == native['robot_snapshot']
        and meta['calibration']['resolution'] == oc.RESOLUTION and meta['geometry_screen']['passed'] is True
        and meta['lighting'] == scene['actual_lighting'] and meta['renderer'] == scene['actual_renderer']
        and scene['policy'] == oc.SCENE_POLICY, 'Fresh original scene/pose evidence differs')
    oc.require(row['source_plant_id'] == row['split_group'] == entry['source_family']
        and row['component_id'] == source['component_id'] and row['conservative_view_cap_group'] == entry['target_id']
        and row['cut_region_proposal'] != source['cut_region_proposal'], 'Generated source ancestry or anatomy differs')
    original_world = {k: deepcopy(meta['supervision'][k]) for k in
        ('plant_to_world_usd_row_vectors', 'nominal_world_m', 'interval_world_m')}
    nominal = transform_points([row['cut_region_proposal']['nominal']['point_plant_m']],
        original_world['plant_to_world_usd_row_vectors'])[0]
    specs = propose_specs(native['robot_snapshot']['robot_root_to_world_usd_row_vectors'], nominal, seed_key, views)
    for spec in specs:
        spec['candidate_id'] = source['component_id'] + '_' + spec['candidate_id']
    return dict(reference_entry_id=entry['id'], source_row=deepcopy(source), generated_row=deepcopy(row),
        target_id=row['target_id'], conservative_view_cap_group=entry['target_id'],
        expected_robot_snapshot=deepcopy(native['robot_snapshot']), expected_calibration=deepcopy(native['calibration']),
        expected_original_world=original_world, expected_nominal_world_m=nominal.tolist(),
        generated_nominal_projection_cpu_only=project([nominal], native['calibration'])[0],
        pose_prior=deepcopy(entry['pose_prior']), native_reference=deepcopy(native), views=specs)


def _assemble(request, request_pin):
    oc.require(request['schema'] == SCHEMA and request['profile'] == PROFILE
        and all(request[k] is v for k, v in FLAGS.items()), 'Changed batch preparation scope')
    seed, count = request['seed'], request['views_per_target']
    reference = request['reference_bank']
    oc.require(type(seed) is int and 0 <= seed < 2**32 and type(count) is int and 1 <= count <= 6,
        'Bounded explicit seed and existing one-to-six pose proposals required')
    bank, entries = _bank(reference['path'], reference['sha256'], reference['entry_ids'])
    oc.require(seed + (len(entries)-1)*104729 < 2**32, 'Derived target recipe seeds overflow uint32')
    anchor = entries[0]
    family = anchor['source_family']
    scene = anchor['scene_authority']
    directory = Path(request['variant_directory']).resolve()
    oc.require(directory.name == family + '_cr_' + str(seed), 'Variant naming/source seed differs')
    qualification_path = directory / 'qualification.json'
    qualification = oc.read_json(qualification_path)
    expected_recipes = [(seed+i*104729, e['target_id']) for i, e in enumerate(entries)]
    oc.require(qualification['version'] == 'curved_relocated_rigid_leaf_static.v2'
        and [(r['seed'], r['source_target_id']) for r in qualification['recipes']] == expected_recipes
        and qualification['source_plan_sha256'] == scene['clear_plan']['sha256']
        and Path(qualification['source_plan_path']).resolve() == Path(scene['clear_plan']['path']).resolve(),
        'Variant recipes must match the current clear-plan targets in order')
    catalogue = _catalogue(directory, scene['clear_plan']['path'])
    generated = {r['component_id']: r for r in catalogue['rows']}
    oc.require(len(generated) == len(catalogue['rows']) == len(entries)
        and set(generated) == {e['scene_authority']['source_row']['component_id'] for e in entries}
        and catalogue['variant_id'] == directory.name and catalogue['independent_target_novelty_approved'] is False,
        'Generated target membership differs or claims novelty')
    cases = []
    for entry in entries:
        component = entry['scene_authority']['source_row']['component_id']
        row = generated[component]
        oc.require(row['target_id'] == directory.name + '/' + component and row['variant_id'] == directory.name,
            'Generated target identity differs')
        key = oc.digest(oc.canonical(dict(reference_entry_id=entry['id'], seed=seed,
            qualification_sha256=oc.sha256(qualification_path), views=count)))
        cases.append(_case(entry, row, key, count))
    first = cases[0]
    outputs = {str(oc.safe_file(directory, rel)): pin for rel, pin in qualification['output_hashes'].items()}
    bindings = oc.merge_bindings(bank['source_bindings'], outputs, catalogue['texture_bindings'],
        {str(Path(reference['path']).resolve()): reference['sha256'], request_pin['path']: request_pin['sha256'],
         str(qualification_path): oc.sha256(qualification_path)})
    oc.bind_all(bindings)
    plan = dict(schema=SCHEMA, state=STATE, profile=PROFILE, preparation_request=deepcopy(request_pin),
        reference_bank=deepcopy(reference), source_family=family, split='train', split_group=family,
        variant_directory=str(directory), seed=seed, scene_authority=deepcopy(scene),
        source_row=deepcopy(first['source_row']), generated_row=deepcopy(first['generated_row']),
        expected_robot_snapshot=deepcopy(first['expected_robot_snapshot']),
        expected_calibration=deepcopy(first['expected_calibration']),
        expected_original_world=deepcopy(first['expected_original_world']), pose_prior=deepcopy(first['pose_prior']),
        target_cases=cases, views_per_target=count, maximum_native_frames=sum(len(c['views']) for c in cases),
        render_subframes_per_view=56, native_instance_backend='legacy', camera_path=oc.HEAD_CAMERA,
        resolution=list(oc.RESOLUTION), input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
        source_bindings=bindings, implementation_bindings=implementation_bindings(),
        geometry_guided_capture_not_blind_evaluation=True,
        sensor_qualification_scope='reviewed_original_native_optics_not_generated_target_visibility', **FLAGS)
    return plan, catalogue


def check_plan(plan):
    oc.require(isinstance(plan, dict) and plan.get('schema') == SCHEMA and plan.get('state') == STATE,
        'Unknown original-native batch plan')
    oc.bind_all(plan['source_bindings']); oc.bind_all(plan['implementation_bindings'])
    rp = plan['preparation_request']
    request = oc.read_json(oc.pin(rp['path'], rp['sha256']))
    expected, catalogue = _assemble(request, rp)
    oc.require(plan == expected, 'Batch plan differs from fresh bank/geometry/pose replay')
    return catalogue


def prepare(bank_path, *, bank_sha256, entry_ids, seed, output, views=6):
    oc.require(type(seed) is int and 0 <= seed < 2**32 and type(views) is int and 1 <= views <= 6,
        'Explicit uint32 seed and one-to-six views required')
    output = oc.new_destination(output, ())
    implementation_bindings()
    bank, entries = _bank(bank_path, bank_sha256, entry_ids)
    oc.require(seed + (len(entries)-1)*104729 < 2**32, 'Derived recipe seed overflow')
    scene = entries[0]['scene_authority']
    protected = [scene['package'], bank_path, *bank['source_bindings']]
    protected += [Path(e['native_observation']['sample']['path']).parent.parent for e in entries]
    protected += [Path(e['pose_prior']['manifest']['path']).parent for e in entries]
    oc.new_destination(output, protected)
    family = entries[0]['source_family']
    variant = output / (family + '_cr_' + str(seed))
    output.mkdir(parents=True)
    request = dict(schema=SCHEMA, profile=PROFILE, reference_bank=dict(path=str(Path(bank_path).resolve()),
        sha256=bank_sha256, entry_ids=list(entry_ids)), variant_directory=str(variant), seed=seed,
        views_per_target=views, **FLAGS)
    path = output / 'prepare_request.json'
    oc.write_new(path, request)
    try:
        _generate(scene['clear_plan']['path'], family,
            [e['scene_authority']['source_row']['component_id'] for e in entries], seed, variant)
        plan, _ = _assemble(request, dict(path=str(path), sha256=oc.sha256(path)))
        plan_path = output / 'plan.json'
        oc.write_new(plan_path, plan)
        result = dict(schema=SCHEMA, state=STATE, plan_path=str(plan_path), plan_sha256=oc.sha256(plan_path),
            target_count=len(entries), maximum_future_native_frames=plan['maximum_native_frames'], **FLAGS)
        oc.write_new(output / 'prepared.json', result)
        return result
    except BaseException as exc:
        oc.write_new(output / 'failure.json', dict(state='batch_preparation_failed', error=repr(exc), **FLAGS))
        raise
