"""Complete reserved original raw-sweep captures; other capture types fail closed.

Run with ``python -m sim_data.native848_multihost_complete_v1`` and PYTHONPATH
including examples/greenhouse_sim. Coordinator token/URL come only from the
configured environment variables. Completion never grants training approval.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

from .dataset_review import require, verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
from . import native848_multihost_sweep_v1 as adapter
from . import native848_scene_sweep_raw_9mm_v1 as authority

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from dataset_multihost_pipeline import host_config

class Inputs:
    def __init__(self):
        self.bindings = {}

    def local(self, path):
        path = Path(path).resolve()
        digest = sha256(path)
        require(str(path) not in self.bindings or self.bindings[str(path)] == digest, 'Input changed during completion')
        self.bindings[str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def read(self, value, within=None):
        require(isinstance(value, dict) and 'path' in value and 'sha256' in value, 'Pinned input required')
        path = Path(value['path']).resolve()
        require(within is None or path.is_relative_to(Path(within).resolve()), 'Input escaped capture authority')
        require(self.local(path)['sha256'] == value['sha256'], 'Changed pinned input')
        return json.loads(path.read_text(encoding='utf-8'))

    def bind(self, bindings):
        for path, digest in bindings.items():
            key = str(Path(path).resolve())
            require(key not in self.bindings or self.bindings[key] == digest, 'Conflicting source binding')
            self.bindings[key] = digest

    def close(self):
        verify_bindings(self.bindings)

def match_reservations(claims, identities, source_records, actual_records, config, coordinator):
    require(claims['schema'] == 'greenhouse.multihost_sweep_claim_receipt.v1', 'Only original raw-sweep reservations are supported')
    require(claims['host'] == config['host_id'] and claims['worker_count'] == config['worker_count'], 'Wrong completion host')
    require(len(identities) == len(source_records) and len({r['sample_id'] for r in source_records}) == len(source_records), 'Incomplete or repeated source schedule')
    require([r['sample_id'] for r in source_records] == [r['sample_id'] for r in identities], 'Identity sample order differs')
    submitted, skipped = adapter.partition_identities(identities, config, claims['worker_id'], coordinator)
    require(submitted == claims['submitted_source_indices'] and skipped == claims['partition_skips'], 'Reservation partition differs')
    candidates = [identities[i] for i in submitted]
    granted = adapter.granted_indices(candidates, claims['response'], claims['host'], coordinator)
    indices = [submitted[i] for i in granted]
    require(actual_records == [source_records[i] for i in indices], 'Captured request differs from exact granted source records')
    require(indices, 'No granted captures to complete')
    return {source_records[submitted[i]]['sample_id']: claims['response']['reservations'][i] for i in granted}

def prepare(config_path, capture, reservation_path):
    inputs = Inputs()
    config_pin = inputs.local(config_path)
    config = host_config(config_pin['path'])
    capture = Path(capture).resolve()
    if capture.name != 'capture':
        capture = capture / 'capture'
    reservation_pin = inputs.local(reservation_path)
    claims = inputs.read(reservation_pin)
    require(claims.get('schema') == 'greenhouse.multihost_sweep_claim_receipt.v1', 'Generated/persistent completion is not qualified by this version')
    require(claims['config'] == config_pin and claims['training_approved'] is False, 'Reservation configuration differs')
    # This authenticates the real exited owner and calls unchanged raw validators.
    complete, result, request, checked, context, observations = authority.authenticate_capture(capture, inputs.read, inputs.local)
    require(request.get('coordinator_claims') == reservation_pin and request.get('parent_request') == claims['source_request'], 'Actual native request is not bound to these reservations')
    coordinator = adapter.coordinator_module()
    require(inputs.local(adapter.__file__) == claims['adapter'] == request['multihost_adapter'], 'Reservation adapter drift')
    require(inputs.local(coordinator.__file__) == claims['coordinator'], 'Coordinator implementation drift')
    inputs.local(__file__)
    inputs.local(authority.__file__)
    original = inputs.read(claims['source_request'])
    source_records = inputs.read(original['records'])['records']
    identities = inputs.read(claims['identities'])
    geometry = inputs.read(claims['geometry_identities'])
    require(identities['source_request'] == claims['source_request'], 'Foreign identity source request')
    require(original['expected_census'] == request['expected_census'], 'Reservation and actual scene differ')
    recomputed = adapter.schedule_identities(source_records, inputs.read(request['expected_census']), geometry['geometry_by_plant_root'])
    require(recomputed == identities['records'], 'Reserved local/world camera identities differ from actual source records')
    reservations = match_reservations(claims, recomputed, source_records, checked['records'], config, coordinator)
    inputs.bind(identities['source_bindings'])
    inputs.bind(context['source_bindings'])
    inputs.bind(request['implementation_bindings'])
    for evidence in geometry['manifests'].values():
        inputs.bind(evidence['source_bindings'])
    actual = {o['sample_id']: o for o in observations}
    held = {h['sample_id']: h for h in result['holds']}
    require(len(actual) == len(observations) and len(held) == len(result['holds']) and not set(actual) & set(held)
            and set(actual) | set(held) == set(reservations), 'Frames/holds do not cover exactly the granted schedule')
    actions = []
    for sample, grant in reservations.items():
        if sample in actual:
            observation = actual[sample]
            rgb_pin = observation['files']['rgb']
            rgb_path = Path(rgb_pin['path']).resolve()
            require(rgb_path == capture/'frames'/sample/'rgb.png', 'RGB escaped its native frame')
            require(inputs.local(rgb_path) == rgb_pin, 'Changed native RGB bytes')
            metrics = coordinator.rgb_metrics(rgb_path)
            require(metrics['encoded_rgb_sha256'] == rgb_pin['sha256'], 'RGB changed while decoding')
            actions.append(dict(sample_id=sample, reservation=grant, outcome='captured', metrics=metrics, rgb=rgb_pin))
        else:
            actions.append(dict(sample_id=sample, reservation=grant, outcome='no_frame', metrics=None, native_hold_reason=held[sample]['reason']))
    inputs.close()
    return dict(inputs=inputs, config=config, claims=claims, coordinator=coordinator, actions=actions,
                capture_result=complete['result'], owner_complete=inputs.local(capture.parent/'owner_complete.json'), reservation=reservation_pin)

def run(config, capture, reservation, output, client=None):
    output = Path(output).resolve()
    require(not output.exists(), 'Fresh completion output required')
    prepared = prepare(config, capture, reservation)
    cfg, coordinator = prepared['config'], prepared['coordinator']
    if client is None:
        url = os.environ.get(cfg['coordinator_url_env'], '')
        token = os.environ.get(cfg['coordinator_token_env'], '')
        client = coordinator.Client(url, token, cfg['host_id'])
    require(client.host == cfg['host_id'] and client.url == prepared['claims']['coordinator_url'], 'Different coordinator or host')
    output.mkdir(parents=True)
    results = []
    # All local inputs are checked before the first state-changing HTTP call.
    # On transport failure, completed tasks remain sealed; repeat into a fresh
    # output directory uses coordinator idempotence, never reclaims or recaptures.
    for index, action in enumerate(prepared['actions']):
        result = client.finish(action['reservation'], action['metrics'], action['outcome'])
        require(result['task_id'] == action['reservation']['task_id'] and result['host'] == cfg['host_id']
                and result['outcome'] == action['outcome'] and result['training_approved'] is False,
                'Coordinator completion response differs')
        require(result['state'] in ('complete','duplicate_hold') if action['outcome'] == 'captured' else result['state'] == 'no_frame', 'Unexpected completion state')
        item = dict(sample_id=action['sample_id'], **result)
        save_json(output/f'completion_{index:04d}.json', item)
        results.append(item)
    prepared['inputs'].close()
    receipt = dict(schema='greenhouse.multihost_capture_completion.v1', host=cfg['host_id'], capture_kind='sweep-raw',
        owner_complete=prepared['owner_complete'], capture_result=prepared['capture_result'], reservation=prepared['reservation'],
        frames_captured=sum(a['outcome']=='captured' for a in prepared['actions']), no_frame=sum(a['outcome']=='no_frame' for a in prepared['actions']),
        duplicate_holds=sum(r['state']=='duplicate_hold' for r in results), completions=results,
        actual_owner_exit_zero_verified=True, frozen_raw_producer_validation_passed=True, source_bindings=prepared['inputs'].bindings,
        training_approved=False, accepted_training_increment=0, quality_annotation_and_visual_review_still_required=True,
        generated_persistent_supported=False, native_launched=False)
    save_json(output/'result.json', receipt)
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','reservation','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config,args.capture,args.reservation,args.output)
    print(json.dumps({key:result[key] for key in ('frames_captured','no_frame','duplicate_holds','training_approved')}))

if __name__ == '__main__':
    main()
