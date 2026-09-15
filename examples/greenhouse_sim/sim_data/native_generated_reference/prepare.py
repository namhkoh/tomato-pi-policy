"""One-target generated scene preparation from a genuine original1696 reference.

No legacy bank/base/pair schema, native launcher, image output or approval.
The first experiment fixes the exact authenticated pose for original/generated
controls, two FUTURE native frames at56 subframes. A producer and post-exit audit
adapter must be explicitly implemented/bound before any native execution.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json

from ..native_dataset import original_reference_bank as bank_api
from ..native_original_capture import contracts as oc

SCHEMA = 'greenhouse.generated_from_original_reference_plan.v1'
PROFILE = 'one_target_exact_pose_original_generated_56.v1'
STATE = 'cpu_prepared_scene_only_no_native_producer_bound'
EXECUTION = dict(native_launch_supported=False, producer_module=None, producer_bindings=None,
                 owned_launcher_receipt=None, postexit_audit_adapter=None)
FLAGS = dict(training_approved=False, source_cap_reset=False, native_launched=False,
             generated_native_qualification=False, independent_target_novelty_approved=False,
             physical_motion_commanded=False, paired_848_1696_proof=False)
_ROOT = Path(__file__).resolve().parent
_LOADED = oc.merge_bindings(bank_api.implementation_bindings(),
    {str(_ROOT/name): oc.sha256(_ROOT/name) for name in ('__init__.py', 'prepare.py', 'scene.py')})


def implementation_bindings():
    oc.bind_all(_LOADED)
    return dict(_LOADED)


def _variant(directory, clear_plan):
    from ..plant_variant_catalogue import load_for_inspection
    return load_for_inspection(directory, clear_plan)


def _generate(clear_plan, family, component, seed, directory):
    from ..procedural_petiole_v2 import generate
    return generate(clear_plan, family, [component], seed, directory)


def _facts(proof, directory, catalogue):
    from ..capture_contract import transform_points, project
    oc.require(proof['schema'] == bank_api.PROOF_SCHEMA
        and proof['paired_848_1696_proof'] is False
        and proof['training_approved'] is False and proof['source_cap_reset'] is False,
        'Genuine original1696 proof required, not a legacy pair')
    scene, pose = proof['scene_authority'], proof['pose_prior']
    source = scene['source_row']
    family, key = source['source_plant_id'], source['component_id']
    rows = [r for r in catalogue['rows'] if r['component_id'] == key]
    oc.require(len(rows) == len(catalogue['rows']) == 1, 'Exactly one qualified generated target required')
    row = rows[0]
    oc.require(row['source_plant_id'] == row['split_group'] == family
        and row['target_id'] == directory.name + '/' + key
        and row['variant_id'] == catalogue['variant_id'] == directory.name
        and row['conservative_view_cap_group'] == source['target_id'] == proof['target_id']
        and row['cut_region_proposal'] != source['cut_region_proposal']
        and catalogue['independent_target_novelty_approved'] is False,
        'Generated anatomy/source/cap mismatch or unchanged target')
    native = proof['native_observation']
    sample = oc.read_json(oc.pin(native['sample']['path'], native['sample']['sha256']))
    oc.require(sample['schema_version'] == oc.SAMPLE_SCHEMA
        and sample['supervision']['target_id'] == proof['target_id']
        and sample['calibration'] == native['calibration']
        and sample['robot_snapshot'] == native['robot_snapshot'] == pose['robot_snapshot']
        and sample['lighting'] == scene['actual_lighting']
        and sample['renderer'] == scene['actual_renderer']
        and scene['policy'] == oc.SCENE_POLICY, 'Native reference/scene/pose mismatch')
    world = {k: deepcopy(sample['supervision'][k]) for k in
             ('plant_to_world_usd_row_vectors', 'nominal_world_m', 'interval_world_m')}
    generated_world = transform_points([row['cut_region_proposal']['nominal']['point_plant_m']],
                                      world['plant_to_world_usd_row_vectors'])[0]
    return dict(schema=SCHEMA, state=STATE, profile=PROFILE, source_family=family,
        split='train', split_group=family, source_row=deepcopy(source), generated_row=deepcopy(row),
        conservative_view_cap_group=source['target_id'], variant_directory=str(directory),
        pose_prior=deepcopy(pose), scene_authority=deepcopy(scene),
        expected_robot_snapshot=deepcopy(native['robot_snapshot']),
        expected_calibration=deepcopy(native['calibration']), expected_original_world=world,
        expected_generated_nominal_world_m=generated_world.tolist(),
        generated_nominal_projection_cpu_only=project([generated_world], native['calibration'])[0],
        camera_path=oc.HEAD_CAMERA, resolution=list(oc.RESOLUTION),
        modes=['original_control', 'generated_variant'], sample_count_limit=2,
        render_subframes_per_view=56, native_instance_backend='legacy',
        pose_changes_between_modes=False, input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
        execution=deepcopy(EXECUTION), **FLAGS)


def _assemble(proof_pin, request_pin):
    """Full read-only replay; no opt-out callback or hash-only launch shortcut."""
    proof = oc.read_json(oc.pin(proof_pin['path'], proof_pin['sha256']))
    request = oc.read_json(oc.pin(request_pin['path'], request_pin['sha256']))
    oc.require(type(request['seed']) is int and 0 <= request['seed'] < 2**32, 'Explicit uint32 recipe seed required')
    reference = request['reference']
    expected = bank_api.verify_anchor(reference['bank_path'], bank_sha256=reference['bank_sha256'],
                                      entry_id=reference['entry_id'])
    oc.require(proof == expected, 'Stored anchor proof differs from independent native replay')
    directory = Path(request['variant_directory']).resolve()
    clear = proof['scene_authority']['clear_plan']
    qualification_path = directory/'qualification.json'
    qualification = oc.read_json(qualification_path)
    oc.require(qualification['version'] == 'curved_relocated_rigid_leaf_static.v2'
        and len(qualification['recipes']) == 1
        and qualification['recipes'][0]['seed'] == request['seed']
        and qualification['recipes'][0]['source_target_id'] == request['source_target']
        and qualification['source_plan_sha256'] == clear['sha256']
        and Path(qualification['source_plan_path']).resolve() == Path(clear['path']).resolve(),
        'Variant must derive from CURRENT clear plan, not historical pose plan')
    catalogue = _variant(directory, clear['path'])
    plan = _facts(proof, directory, catalogue)
    oc.require(request['schema'] == SCHEMA and request['profile'] == PROFILE
        and request['source_target'] == plan['source_row']['target_id']
        and request['execution'] == EXECUTION and all(request[k] == v for k, v in FLAGS.items()),
        'Preparation request changed')
    bank = oc.read_json(oc.pin(reference['bank_path'], reference['bank_sha256']))
    outputs = {str(oc.safe_file(directory, rel)): sha for rel, sha in qualification['output_hashes'].items()}
    bindings = oc.merge_bindings(bank['source_bindings'], outputs, catalogue['texture_bindings'],
        {str(Path(reference['bank_path']).resolve()): reference['bank_sha256'],
         proof_pin['path']: proof_pin['sha256'], request_pin['path']: request_pin['sha256'],
         str(qualification_path): oc.sha256(qualification_path)})
    oc.bind_all(bindings)
    plan.update(anchor_evidence=deepcopy(proof_pin), preparation_request=deepcopy(request_pin),
                source_bindings=bindings, implementation_bindings=implementation_bindings())
    return plan, catalogue


def check_plan(plan):
    oc.require(isinstance(plan, dict) and plan.get('schema') == SCHEMA and plan.get('state') == STATE,
               'Unknown original-reference generated scene plan')
    oc.require(plan['execution'] == EXECUTION, 'CPU scene plan cannot acquire launch/audit authority')
    oc.bind_all(plan['source_bindings']); oc.bind_all(plan['implementation_bindings'])
    expected, catalogue = _assemble(plan['anchor_evidence'], plan['preparation_request'])
    oc.require(plan == expected, 'Generated scene plan differs from full proof/geometry replay')
    return catalogue


def prepare(bank_path, *, bank_sha256, entry_id, seed, output):
    oc.require(type(seed) is int and 0 <= seed < 2**32, 'Explicit uint32 recipe seed required')
    output = oc.new_destination(output, ())
    implementation_bindings()
    proof = bank_api.verify_anchor(bank_path, bank_sha256=bank_sha256, entry_id=entry_id)
    scene = proof['scene_authority']
    bank = oc.read_json(oc.pin(bank_path, bank_sha256))
    protected = [scene['package'], Path(proof['native_observation']['sample']['path']).parent.parent,
                 Path(proof['pose_prior']['manifest']['path']).parent, *bank['source_bindings'], bank_path]
    oc.new_destination(output, protected)
    source = scene['source_row']
    variant = output/(source['source_plant_id'] + '_cr_' + str(seed))
    output.mkdir(parents=True)
    proof_path = output/'original_anchor_evidence.json'
    oc.write_new(proof_path, proof)
    request_path = output/'prepare_request.json'
    request = dict(schema=SCHEMA, profile=PROFILE, reference=dict(bank_path=str(Path(bank_path).resolve()),
        bank_sha256=bank_sha256, entry_id=entry_id), seed=seed, source_target=source['target_id'],
        variant_directory=str(variant), execution=deepcopy(EXECUTION), **FLAGS)
    oc.write_new(request_path, request)
    # Only generated copies are written. Source assets and capture data stay frozen.
    _generate(scene['clear_plan']['path'], source['source_plant_id'], source['component_id'], seed, variant)
    plan, _ = _assemble(dict(path=str(proof_path), sha256=oc.sha256(proof_path)),
                        dict(path=str(request_path), sha256=oc.sha256(request_path)))
    plan_path = output/'plan.json'
    oc.write_new(plan_path, plan)
    result = dict(schema=SCHEMA, state=STATE, plan_path=str(plan_path), plan_sha256=oc.sha256(plan_path),
        reference=request['reference'], source_target=source['target_id'], generated_target=plan['generated_row']['target_id'],
        variant_directory=str(variant), maximum_future_native_frames=2, execution=deepcopy(EXECUTION), **FLAGS)
    oc.write_new(output/'prepared.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    build = commands.add_parser('prepare')
    for flag in ('bank', 'bank-sha256', 'entry-id', 'output'):
        build.add_argument('--'+flag, required=True)
    build.add_argument('--seed', type=int, required=True)
    check = commands.add_parser('check')
    check.add_argument('--plan', required=True); check.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        result = prepare(args.bank, bank_sha256=args.bank_sha256, entry_id=args.entry_id, seed=args.seed, output=args.output)
    else:
        path = oc.pin(args.plan, args.plan_sha256)
        check_plan(oc.read_json(path)); oc.pin(path, args.plan_sha256)
        result = dict(state='cpu_plan_revalidated_no_native_launch', execution=EXECUTION)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
