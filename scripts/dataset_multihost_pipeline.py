"""Common locked generator/annotation entry points for the three capture hosts.

This does not install Isaac Sim or port its Windows owner to Linux. Geometry
generation alone never grants capture, visibility or training approval.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'examples'), str(ROOT / 'examples/greenhouse_sim')]
from dataset_capture_coordinator import ALLOCATION, digest, require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def host_config(path):
    config = read(path)
    require(config['schema'] == 'greenhouse.multihost_capture_config.v1', 'Wrong host configuration')
    require(config['allocation_sha256'] == digest(ALLOCATION), 'Different donor allocation')
    require(config['allowed_foreground_families'] == ALLOCATION[config['host_id']], 'Altered donor ownership')
    for key, value in dict(resolution=[848, 408], cut_offset_m=.009,
                           exactly_one_eligible_petiole=True, query_input=False,
                           fully_labeled_plant_count=144, full_unpruned_plants=True,
                           minimum_background_plant_fraction=.40,
                           native_mask_width_minimum_px=8,
                           projected_cut_support_minimum_px=12, split='train').items():
        require(config.get(key) == value, 'Changed shared dataset contract: ' + key)
    require(config['worker_count'] == {'local5090': 1, 'thor1': 4, 'thor3': 2}[config['host_id']], 'Wrong worker count')
    require(config['new_morphology_seed_range'] == {'local5090': [1000000, 1999999], 'thor1': [2000000, 2999999], 'thor3': [3000000, 3999999]}[config['host_id']], 'Altered host seed namespace')
    return config


def verify_lock(path):
    lock = read(path)
    require(lock['schema'] == 'greenhouse.multihost_toolchain_lock.v1', 'Wrong toolchain lock')
    require(lock['annotation_contract'] == dict(resolution=[848, 408], nominal_cut_arc_m=.009,
            exactly_one_eligible_petiole=True, query_input=False, full_labeled_plant_count=144,
            minimum_background_plant_fraction=.40), 'Changed annotation contract')
    require(lock['files'], 'Empty toolchain lock')
    for relative, expected in lock['files'].items():
        path_on_disk = (ROOT / relative).resolve()
        require(path_on_disk.is_relative_to(ROOT) and path_on_disk.is_file(), 'Missing or unsafe locked source: ' + relative)
        require(sha(path_on_disk) == expected, 'Toolchain drift: ' + relative)
    return dict(toolchain_lock_sha256=sha(path), verified_files=len(lock['files']), annotation_contract=lock['annotation_contract'])


def build_controls(config, family, seed, targets):
    require(family in config['allowed_foreground_families'], 'Donor belongs to another host')
    low, high = config['new_morphology_seed_range']
    require(type(seed) is int and low <= seed <= high, 'Seed is outside this host namespace')
    require(1 <= len(targets) <= 8 and len(set(targets)) == len(targets), 'Unique bounded controlled targets required')
    require(not (family == 'seed41_full' and 'SubStem_38' in targets), 'Banned target')
    from sim_data.procedural_petiole_controlled_v4 import control_for
    rng = random.Random(seed)
    return [dict(component_id=target, control=control_for(.025, rng.uniform(-180, 180))) for target in targets]


def check_background(census):
    """Structure check; actual pixel coverage remains the frozen annotator's job."""
    require(census['complete_active_plant_anatomy'] is True, 'Incomplete anatomy')
    require(census['dataset_split'] == 'train' and census['full_active_catalogue_required'] is True, 'Complete TRAIN catalogue required')
    require(census['active_counts']['backdrop_instances'] == 0, 'Unlabeled merged backdrop remains')
    require(census['removed_merged_or_unknown_count'] == 0 and census['removed_cross_split_count'] == 0, 'Removed plant population')
    plants = census['all_plant_roots']
    require(len(plants) == 144 and len({p['plant_root'] for p in plants}) == 144, 'Exactly144 distinct plant slots required')
    require(census['active_counts']['component_plants'] == 144, 'Incorrect active plant count')
    require(not census.get('removed_roots') and not census.get('extra_plant_asset_roots_outside_packplants'), 'Removed or untracked plants')
    require(all(p['active_after_policy'] is True and p['authenticated_component_plant'] is True and p['source_split'] == 'train' and p['component_count'] > 0 for p in plants), 'Unlabeled/inactive/cross-split background')
    return dict(plant_slots=144, donor_families=sorted({p['source_family'] for p in plants}),
                actual_image_background_coverage_still_required=True, training_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=ROOT / 'configs/dataset_capture/toolchain.lock.json')
    parser.add_argument('--config', type=Path, required=True)
    subs = parser.add_subparsers(dest='action', required=True)
    subs.add_parser('verify')
    generate = subs.add_parser('generate')
    generate.add_argument('--source-plan', type=Path, required=True)
    generate.add_argument('--source-plan-sha256', required=True)
    generate.add_argument('--family', required=True)
    generate.add_argument('--primary-target', required=True)
    generate.add_argument('--controlled-targets', nargs='+', required=True)
    generate.add_argument('--seed', type=int, required=True)
    generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--recipe-only', action='store_true')
    background = subs.add_parser('check-background')
    background.add_argument('--census', type=Path, required=True)
    background.add_argument('--census-sha256', required=True)
    annotate = subs.add_parser('annotate')
    annotate.add_argument('--kind', choices=('sweep-raw', 'sweep-ordinary', 'controlled-generated', 'persistent-generated', 'experimental-generated'), required=True)
    annotate.add_argument('--capture', type=Path, required=True)
    annotate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = host_config(args.config)
    proof = verify_lock(args.lock)
    if args.action == 'verify':
        result = dict(proof, host=config['host_id'], foreground_families=config['allowed_foreground_families'])
    elif args.action == 'generate':
        require(sha(args.source_plan) == args.source_plan_sha256, 'Changed generator source plan')
        plan = read(args.source_plan)
        require(plan['family_assignments'][args.family] == 'train', 'Donor lineage must remain TRAIN')
        require(args.primary_target in args.controlled_targets, 'Primary target must have current protected geometry checks')
        require(not args.output.exists(), 'Fresh generated output required')
        controls = build_controls(config, args.family, args.seed, args.controlled_targets)
        recipe = dict(host=config['host_id'], seed=args.seed, family=args.family,
                      controls=controls, primary_target=args.primary_target,
                      generator='sim_data.procedural_petiole_controlled_v4',
                      full_unpruned_donor_retained=True, independent_original_donor=False,
                      training_approved=False, native_capture_pending=True, **proof)
        if args.recipe_only:
            write(args.output, recipe)
            result = dict(recipe=args.output.as_posix(), native_launched=False)
        else:
            from sim_data.procedural_petiole_controlled_v4 import generate
            from sim_data.native848_controlled_9mm_evidence_v4 import reconstruct
            generate(args.source_plan, args.family, args.primary_target, controls, args.output)
            qualification = args.output / 'qualification.json'
            physical, _, _ = reconstruct(dict(path=str(qualification.resolve()), sha256=sha(qualification)))
            require(physical['all_controlled_targets_qualified'], 'Generated9mm geometry held')
            write(args.output / 'multihost_recipe.json', recipe)
            write(args.output / 'current9mm_evidence.json', physical)
            require(verify_lock(args.lock) == proof, 'Toolchain lock changed during generation')
            result = dict(output=str(args.output), training_approved=False,
                          full144_scene_capture_and_visibility_pending=True, **proof)
    elif args.action == 'check-background':
        require(sha(args.census) == args.census_sha256, 'Changed scene census')
        result = dict(check_background(read(args.census)), **proof)
    else:
        if args.kind.startswith('sweep-'):
            from sim_data.native848_scene_sweep_cached_annotation_v1 import run
            evaluation = run(args.capture, args.output, mode=args.kind.split('-')[1])
        else:
            module, function = {
                'controlled-generated': ('native848_generated_9mm_v3', 'evaluate_capture'),
                'persistent-generated': ('native848_persistent_generated_9mm_v1', 'evaluate_segment'),
                'experimental-generated': ('native848_persistent_experimental_9mm_v1', 'evaluate_segment'),
            }[args.kind]
            evaluation = getattr(importlib.import_module('sim_data.' + module), function)(args.capture, args.output)
        require(verify_lock(args.lock) == proof, 'Toolchain lock changed during annotation')
        write(args.output / 'shared_toolchain_receipt.json', dict(proof, host=config['host_id'],
              capture_kind=args.kind, annotation_result_sha256=sha(args.output / 'result.json'),
              coordinator_duplicate_check_and_visual_review_still_required=True, training_approved=False))
        result = dict(output=str(args.output), frames=evaluation['frames_evaluated'], training_approved=False, **proof)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
