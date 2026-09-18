"""Reserve original-plant sweep views before preparing a native request.

Only preparation is supported. A reservation is neither a captured frame nor an
accepted training example. Frozen native and annotation predicates stay intact.
"""
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from . import native848_geometry_identity_v1 as geometry

IDENTITY_SCHEMA = 'greenhouse.original_plant_multihost_view.v1'
PREPARATION_SCHEMA = 'greenhouse.multihost_sweep_preparation.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def numeric(value, shape):
    """Round numerical transfer noise, normalizing signed zero at 1 nm scale."""
    array = np.asarray(value, dtype=float)
    require(array.shape == shape and np.isfinite(array).all(), 'Invalid identity matrix')
    array = np.round(array, 9)
    array[array == 0] = 0.0
    return array.tolist()


def rigid_rows(value):
    matrix = np.asarray(value, dtype=float)
    numeric(matrix, (4, 4))
    require(np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            and np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3),
                            atol=1e-7, rtol=0)
            and abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-7,
            'Identity needs a proper rigid USD row-vector transform')
    return matrix


def original_scene_identity(census, geometry_hashes_by_plant_root):
    require(census['source_assets_modified'] is False,
            'Generated or modified geometry needs a different typed adapter')
    require(all(slot['source_geometry_modified'] is False for slot in census['all_plant_roots']),
            'Generated slot is not an original plant')
    return geometry.scene_identity(census, geometry_hashes_by_plant_root)


def view_identity(record, slot, scene_identity, geometry_sha256):
    """No target/component/sample ID enters either deduplication identity.

    The local key catches a clone-local repeat. The scene-camera alias also
    catches one image retargeted to a different donor or physical instance.
    """
    require(record['source_geometry_modified'] is False
            and record['target_variant']['source_geometry_modified'] is False
            and slot['source_geometry_modified'] is False,
            'Original unchanged geometry only')
    family = slot['source_family']
    require(record['source_family'] == record['source_row']['source_plant_id'] == family
            and record['split'] == slot['source_split'], 'Source identity mismatch')
    require(record['source_row']['source_manifest_sha256'] == slot['manifest_sha256'],
            'Record and slot have different original source manifests')
    cal = record['calibration']
    require(cal['resolution'] == [848, 408] and cal.get('crop_resize') is None,
            'Native full-frame 848x408 required')
    camera = rigid_rows(cal['camera_to_world_usd_row_vectors'])
    plant = rigid_rows(slot['plant_to_world_usd_row_vectors'])
    # USD row convention: camera-local -> world -> plant-local.
    relative = camera @ np.linalg.inv(plant)
    common = dict(intrinsics=numeric(cal['intrinsics'], (3, 3)),
                  resolution=list(cal['resolution']))
    require(isinstance(geometry_sha256, str) and geometry.HEX.fullmatch(geometry_sha256),
            'Authenticated canonical geometry hash required')
    local = dict(schema=IDENTITY_SCHEMA, geometry_sha256=geometry_sha256,
                 source_family=family, split=slot['source_split'],
                 camera_to_plant=numeric(relative, (4, 4)), **common)
    image = dict(schema='greenhouse.fixed_scene_camera_identity.v1',
                 scene_identity_sha256=digest(scene_identity),
                 camera_to_world=numeric(camera, (4, 4)), **common)
    return dict(local_view_key=digest(local), scene_camera_key=geometry.scene_camera_key(scene_identity, cal),
                local_view_identity=local, scene_camera_identity=image)


def schedule_identities(records, census, geometry_hashes_by_plant_root):
    scene = original_scene_identity(census, geometry_hashes_by_plant_root)
    slots = {r['plant_root']: r for r in census['all_plant_roots']}
    require(len(slots) == 144, 'Duplicate physical plant roots')
    seen = set()
    result = []
    for record in records:
        sample = record['sample_id']
        require(sample not in seen, 'Repeated source sample ID')
        seen.add(sample)
        require(record['plant_root'] in slots, 'Unknown physical instance')
        result.append(dict(sample_id=sample,
                           **view_identity(record, slots[record['plant_root']], scene,
                                           geometry_hashes_by_plant_root[record['plant_root']])))
    return result



def coordinator_module():
    path = Path(__file__).resolve().parents[3] / 'scripts/dataset_capture_coordinator.py'
    spec = importlib.util.spec_from_file_location('dataset_capture_coordinator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coordinator_identity(item):
    local = item['local_view_identity']
    return dict(geometry_sha256=local['geometry_sha256'],
                donor_source_family=local['source_family'], split=local['split'],
                camera_to_plant_usd_row_vectors=local['camera_to_plant'],
                intrinsics=local['intrinsics'], resolution=local['resolution'],
                scene_camera_key=item['scene_camera_key'])


def granted_indices(identities, response, host, coordinator):
    reservations = response['reservations']
    require(len(reservations) == len(identities), 'Incomplete coordinator response')
    indices = []
    keys = set()
    aliases = set()
    for index, (item, reservation) in enumerate(zip(identities, reservations)):
        expected = coordinator.task_id(coordinator_identity(item))
        require(reservation['task_id'] == expected and type(reservation['granted']) is bool,
                'Coordinator response identity/order differs')
        if reservation['granted']:
            require(reservation['host'] == host and bool(reservation['claim_id']),
                    'Foreign or absent reservation authority')
            require(expected not in keys and item['scene_camera_key'] not in aliases,
                    'Coordinator granted duplicate local/image identity')
            keys.add(expected)
            aliases.add(item['scene_camera_key'])
            indices.append(index)
    return indices


def partition_identities(identities, config, worker_id, coordinator):
    require(config['schema'] == 'greenhouse.multihost_capture_config.v1', 'Unknown host config')
    host = config['host_id']
    require(host in coordinator.ALLOCATION, 'Unknown allocated host')
    require(config['allocation_sha256'] == coordinator.digest(coordinator.ALLOCATION)
            and config['allowed_foreground_families'] == coordinator.ALLOCATION[host],
            'Host allocation differs from immutable coordinator policy')
    workers = config['worker_count']
    require(type(workers) is int and 1 <= workers <= 64 and type(worker_id) is int
            and 0 <= worker_id < workers, 'Invalid fixed worker partition')
    selected, skipped = [], []
    for index, item in enumerate(identities):
        value = coordinator_identity(item)
        key = coordinator.task_id(value)
        owner_worker = int(key, 16) % workers
        if value['donor_source_family'] not in config['allowed_foreground_families']:
            reason = 'donor_assigned_to_other_host'
        elif owner_worker != worker_id:
            reason = 'task_assigned_to_other_worker'
        else:
            selected.append(index)
            continue
        skipped.append(dict(source_index=index, sample_id=item['sample_id'], task_id=key,
                            reason=reason, assigned_worker_id=owner_worker))
    return selected, skipped


def prepare(request_path, request_sha256, output, config_path, worker_id, client=None):
    """Authenticate, atomically claim, then derive a fresh granted-only schedule.

    A network error has no local fallback. Never retry or release an uncertain
    claim automatically: inspect the coordinator's durable ledger first.
    """
    from . import native848_scene_sweep_raw_capture_v1 as baseline
    from .dataset_review import verify_bindings
    from .native848_bulk_io_v1 import save_json

    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    source_pin = baseline.pin(request_path)
    require(source_pin['sha256'] == request_sha256, 'Source request hash differs')
    source = baseline.bound(source_pin)
    checked = baseline.check_request(source)
    require(not output.exists(), 'Fresh preparation output required')
    output.mkdir(parents=False)
    coordinator = coordinator_module()
    config_pin = baseline.pin(config_path)
    config = baseline.bound(config_pin)
    if client is None:
        client = coordinator.Client(os.environ.get(config['coordinator_url_env'], ''),
                                    os.environ.get(config['coordinator_token_env'], ''),
                                    config['host_id'])
    require(client.host == config['host_id'], 'Client belongs to another host')
    bindings = {}

    def bind(path, value):
        require(path not in bindings or bindings[path] == value, 'Conflicting source bindings')
        bindings[path] = value

    for path, value in checked['cpu_preflight']['source_bindings'].items():
        bind(path, value)
    for path, value in source['implementation_bindings'].items():
        bind(path, value)
    for item in [source_pin, config_pin, baseline.pin(__file__), baseline.pin(geometry.__file__), baseline.pin(coordinator.__file__),
                 *(source[key] for key in ('scene_anchor', 'expected_census', 'profile',
                                          'records', 'cpu_preflight'))]:
        bind(item['path'], item['sha256'])
    records, census = checked['records'], checked['census']
    # Authenticate all original components against the scene's actual source plan.
    # Missing historical texture pins are explicitly NEW current identity inputs.
    collection_pin = census['source_collection_plan']
    collection = baseline.bound(collection_pin)
    bind(collection_pin['path'], collection_pin['sha256'])
    authored_bindings = dict(collection['source_bindings_sha256'])
    for path, sha in bindings.items():
        require(path not in authored_bindings or authored_bindings[path] == sha,
                'Source plan and current proof disagree')
        authored_bindings[path] = sha
    geometry_results = {}
    geometry_by_root = {}
    for slot in census['all_plant_roots']:
        manifest_pin = baseline.pin(slot['manifest_path'])
        require(manifest_pin['sha256'] == slot['manifest_sha256'], 'Original manifest changed')
        key = manifest_pin['path']
        if key not in geometry_results:
            geometry_results[key] = geometry.geometry_identity(key, authored_bindings,
                                                allow_new_texture_bindings=True)
            for path, sha in geometry_results[key]['source_bindings'].items():
                bind(path, sha)
        geometry_by_root[slot['plant_root']] = geometry_results[key]['geometry_sha256']
    save_json(output / 'geometry_identities.json', dict(
        schema='greenhouse.original_scene_geometry_identity_audit.v1',
        source_plan=collection_pin, geometry_by_plant_root=geometry_by_root,
        manifests=geometry_results, shared_geometry_helper=baseline.pin(geometry.__file__),
        texture_identity_upgrade='New current referenced texture inputs; not historical capture bindings.'))
    geometry_pin = baseline.pin(output / 'geometry_identities.json')
    bind(geometry_pin['path'], geometry_pin['sha256'])
    identities = schedule_identities(records, census, geometry_by_root)
    verify_bindings(bindings)
    save_json(output / 'identities.json', dict(schema=IDENTITY_SCHEMA, source_request=source_pin,
                                              records=identities, source_bindings=bindings))
    submitted, skipped = partition_identities(identities, config, worker_id, coordinator)
    submitted_identities = [identities[index] for index in submitted]
    response = (client.claim_many([coordinator_identity(item) for item in submitted_identities])
                if submitted else dict(schema='greenhouse.capture_reservations.v1', reservations=[]))
    # Persist returned authority before dependent work. Claim handles are private;
    # the coordinator's portable audit export deliberately omits them.
    claims = dict(schema='greenhouse.multihost_sweep_claim_receipt.v1',
                  host=client.host, coordinator_url=client.url, worker_id=worker_id, worker_count=config['worker_count'],
                  config=config_pin, submitted_source_indices=submitted, partition_skips=skipped,
                  source_request=source_pin,
                  identities=baseline.pin(output / 'identities.json'), geometry_identities=geometry_pin, response=response,
                  adapter=baseline.pin(__file__), coordinator=baseline.pin(coordinator.__file__),
                  training_approved=False, accepted_training_increment=0)
    save_json(output / 'claims.json', claims)
    selected = [submitted[index] for index in
                granted_indices(submitted_identities, response, client.host, coordinator)]
    claims_pin = baseline.pin(output / 'claims.json')
    bind(claims_pin['path'], claims_pin['sha256'])
    identity_pin = baseline.pin(output / 'identities.json')
    bind(identity_pin['path'], identity_pin['sha256'])
    selected_records = [deepcopy(records[index]) for index in selected]
    final_request = None
    if selected_records:
        # Selected record dictionaries equal their authenticated parents exactly.
        save_json(output / 'records.json', dict(records=selected_records,
                  parent_records=source['records'], source_bindings=bindings,
                  coordinator_claims=claims_pin, proposal_only=True,
                  scene_collision_pending=True,
                  native_floor_and_workspace_validation_required=True, training_approved=False))
        records_pin = baseline.pin(output / 'records.json')
        proof = dict(schema='greenhouse.multihost_original_sweep_numeric_subset.v1',
                     records=records_pin, expected_census=source['expected_census'],
                     parent_cpu_preflight=source['cpu_preflight'], parent_request=source_pin,
                     source_bindings=bindings, coordinator_claims=claims_pin,
                     selected_source_indices=selected, selected_sample_ids=[r['sample_id'] for r in selected_records],
                     proposal_only=True, scene_collision_pending=True,
                     native_floor_and_workspace_validation_required=True,
                     source_priority_only=True, training_approved=False,
                     accepted_training_increment=0, native_launched=False)
        save_json(output / 'cpu_preflight.json', proof)
        derived = deepcopy(source)
        derived.update(records=records_pin, cpu_preflight=baseline.pin(output / 'cpu_preflight.json'),
                       max_frames=len(selected_records), coordinator_claims=claims_pin,
                       multihost_adapter=baseline.pin(__file__), parent_request=source_pin)
        baseline.check_request(derived)
        save_json(output / 'request.json', derived)
        final_request = baseline.pin(output / 'request.json')
    verify_bindings(bindings)
    result = dict(schema=PREPARATION_SCHEMA, source_request=source_pin,
                  coordinator_claims=claims_pin, identities=identity_pin, geometry_identities=geometry_pin,
                  request=final_request, host=client.host, worker_id=worker_id,
                  worker_count=config['worker_count'], config=config_pin, proposed=len(records),
                  submitted_to_coordinator=len(submitted), partition_skips=skipped,
                  granted=len(selected), skipped=len(records)-len(selected),
                  granted_sample_ids=[r['sample_id'] for r in selected_records],
                  actual_baseline_request_check_passed=bool(selected_records),
                  state='reserved_pending_native' if selected_records else 'no_granted_views',
                  source_priority_only=True, training_approved=False,
                  accepted_training_increment=0, native_launched=False,
                  source_bindings=bindings)
    save_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker-id', type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.request, args.request_sha256, args.output, args.config, args.worker_id)
    print(json.dumps({key: result[key] for key in ('host', 'proposed', 'granted', 'skipped', 'request')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
