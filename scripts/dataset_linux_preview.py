"""Build destination-authenticated generator inputs; never launch native capture.

This is portable CPU preparation, not Linux renderer/owner qualification. The
historical plan and every source asset remain unchanged. A separate native Linux
adapter must qualify two excluded control frames before production can start.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'examples'),
                str(ROOT / 'examples/greenhouse_sim')]
from dataset_multihost_pipeline import host_config, verify_lock


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def bound(path, expected):
    require(sha(path) == expected, 'Changed pinned input: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def relative_source(path, original_package):
    """Parse foreign Windows paths even when running on Linux; no string replace."""
    cls = PureWindowsPath if PureWindowsPath(original_package).drive else PurePosixPath
    source, root = cls(path), cls(original_package)
    require(source.is_absolute() and root.is_absolute(), 'Absolute source paths required')
    require('..' not in source.parts and '..' not in root.parts, 'Parent traversal in source path')
    try:
        relative = source.relative_to(root)
    except ValueError:
        raise ValueError('Source is outside the historical package: ' + str(path)) from None
    require(relative.parts, 'File path cannot equal package directory')
    return relative.parts


def destination(path, original_package, package):
    package = Path(package).resolve()
    target = package.joinpath(*relative_source(path, original_package)).resolve()
    require(target.is_relative_to(package), 'Destination symlink escapes package')
    return target


def verify_package(plan, index, package):
    """Authenticate all indexed package bytes, including textures, before USD reads."""
    original = plan['package']
    archived = {}
    for row in index['files']:
        try:
            parts = relative_source(row['source_path'], original)
        except ValueError:
            continue  # Other archive components are not this plant package.
        target = destination(row['source_path'], original, package)
        key = str(target)
        require(key not in archived, 'Repeated archive destination')
        require(target.is_file() and target.stat().st_size == row['bytes'], 'Missing/changed package file: ' + key)
        require(sha(target) == row['sha256'], 'Archive content mismatch: ' + key)
        archived[key] = row['sha256']
    require(archived, 'Archive index contains no historical package files')
    localized = {}
    for old, expected in plan['source_bindings_sha256'].items():
        target = str(destination(old, original, package))
        require(target not in localized, 'Two historical bindings map to one destination')
        require(archived.get(target) == expected, 'Historical binding is absent from or differs from archive: ' + target)
        localized[target] = expected
    require(localized, 'Empty historical source bindings')
    return localized, archived


def compare_rebuilt(original, rebuilt, package, localized):
    require(rebuilt['schema_version'] == original['schema_version'] == 'greenhouse.grounding_collection_plan.v1', 'Unsupported plan schema')
    require(rebuilt['package'] == str(Path(package).resolve()), 'Wrong rebuilt package')
    require(rebuilt['source_bindings_sha256'] == localized, 'Rebuilt source closure differs')
    jobs = deepcopy(original['jobs'])
    for job in jobs:
        job['source_manifest_path'] = str(destination(job['source_manifest_path'], original['package'], package))
    require(rebuilt['jobs'] == jobs, 'Destination changed source targets or geometry')
    for key in ('configuration', 'cut_rule', 'cut_rule_sha256', 'family_assignments',
                'selection_audit', 'split_scope', 'camera_requirement'):
        require(rebuilt[key] == original[key], 'Destination changed ' + key)
    require(rebuilt['training_dataset_approved'] is False, 'Source plan cannot approve training')


def prepare(config_path, historical_plan, historical_sha, asset_index, index_sha,
            package, output, lock):
    started = time.perf_counter()
    package, output = Path(package).resolve(), Path(output).resolve()
    require(package.is_dir(), 'Extracted package directory required')
    require(not output.exists() and not output.is_relative_to(package)
            and not package.is_relative_to(output), 'Fresh output disjoint from source package required')
    config = host_config(config_path)
    require(config['host_id'] in ('thor1', 'thor3'), 'This preparation is for the Linux server hosts')
    proof = verify_lock(lock)
    inputs = [pin(config_path), pin(historical_plan), pin(asset_index), pin(lock), pin(__file__)]
    original = bound(historical_plan, historical_sha)
    index = bound(asset_index, index_sha)
    require(index.get('schema') == 'greenhouse.scene_asset_transport.files.v1', 'Unsupported published asset index')
    require(original['schema_version'] == 'greenhouse.grounding_collection_plan.v1'
            and original['training_dataset_approved'] is False, 'Historical source plan cannot authorize labels')
    localized, archived = verify_package(original, index, package)
    # Import only after exact package and numerical source authentication.
    from sim_data import training_plan
    output.mkdir(parents=True)
    try:
        rebuilt = training_plan.build_plan(package, output / 'source_plan', original['configuration'])
        new_path = output / 'source_plan/plan.json'
        replay, reports = training_plan.load_plan(new_path)
        require(replay == rebuilt, 'Fresh plan failed unchanged loader replay')
        compare_rebuilt(original, rebuilt, package, localized)
        family, primary, secondary = ('seed43_full', 'SubStem_42', 'SubStem_43') if config['host_id'] == 'thor1' else ('seed53_full', 'SubStem_41', 'SubStem_42')
        require(family in config['allowed_foreground_families'] and rebuilt['family_assignments'][family] == 'train', 'Wrong example donor lineage')
        seed = config['new_morphology_seed_range'][0]
        command = [sys.executable, str(ROOT/'scripts/dataset_multihost_pipeline.py'), '--config', str(Path(config_path).resolve()),
                   '--lock', str(Path(lock).resolve()), 'generate', '--source-plan', str(new_path), '--source-plan-sha256', sha(new_path),
                   '--family', family, '--primary-target', primary, '--controlled-targets', primary, secondary,
                   '--seed', str(seed), '--output', str(output/'generated_preview')]
        require(verify_lock(lock) == proof, 'Shared lock changed during preparation')
        for path, expected in archived.items():
            require(sha(path) == expected, 'Package changed during source-plan reconstruction: ' + path)
        require(all(pin(p['path']) == p for p in inputs), 'Preparation input changed')
        receipt = dict(schema='greenhouse.linux_preview_source_preparation.v1', host=config['host_id'],
            source_plan=pin(new_path), historical_source_plan=inputs[1], published_asset_index=inputs[2],
            source_bindings=localized, archive_package_bindings=archived, source_input_pins=inputs,
            family_count=len(reports), original_family_assignments_preserved=True, source_targets_and_geometry_preserved=True,
            source_assets_unchanged=True, destination_plan_loader_replayed=True, generator_command=command,
            intended_preview_frames=2, preview_controls_excluded_from_release=True,
            historical_plan_cut_rule_is_not_current_9mm_acceptance=True,
            remaining_runtime_gates=['Destination anchor, camera request, profile and runtime-source bindings',
                                     'Linux process/child/GPU identity adapter', 'Two native excluded RGB-D/ID controls in full144 scene',
                                     'Corresponding typed Linux annotation/completion authority and shared numerical parity'],
            native_linux_qualified=False, capture_request_eligible=False, coordinator_contacted=False,
            native_launched=False, training_approved=False, accepted_training_increment=0,
            elapsed_seconds=time.perf_counter()-started, **proof)
        write(output/'result.json', receipt)
        return receipt
    except BaseException as exc:
        write(output/'failure.json', dict(error=repr(exc), source_input_pins=inputs,
            native_launched=False, training_approved=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','historical-plan','asset-index','package','output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--historical-plan-sha256', required=True)
    p.add_argument('--asset-index-sha256', required=True)
    p.add_argument('--lock', type=Path, default=ROOT/'configs/dataset_capture/toolchain.lock.json')
    a = p.parse_args()
    result = prepare(a.config,a.historical_plan,a.historical_plan_sha256,a.asset_index,a.asset_index_sha256,a.package,a.output,a.lock)
    print(json.dumps(dict(result=pin(a.output/'result.json'),source_plan=result['source_plan'],
        native_linux_qualified=False,generator_command=result['generator_command'])))


if __name__ == '__main__':
    main()
