"""Four serial collection lanes; two can inherit slots from audited predecessors.

No renderer changes, robot commands, retries, deletion or implicit visual approval.
The four-worker bound covers these lanes plus their two named predecessors only;
do not launch unrelated capture workers alongside the campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from .collection_plan import load_plan
from .collection_run import run_jobs
from .dataset_review import read_json, require, verify_bindings, write_json
from .depth_preview import sha256

SCHEMA = 'greenhouse.bounded_collection_campaign.v1'
COMPLETE = 'complete_bounded_batch_pending_visual_review'
AUDITED = 'audited_prototype_pending_visual_review'


def allocate(train, heldout, predecessors):
    """Freeze all remaining families exactly once, keeping the reserved split."""
    for key in ('schema_version', 'package', 'family_assignments', 'cut_rule_sha256', 'source_bindings_sha256'):
        require(train[key] == heldout[key], 'Incompatible training and held-out plans')
    require(train['schema_version'] == 'greenhouse.grounding_collection_plan.v1', 'Expected grounding plans')
    tc, hc = (dict(p['configuration']) for p in (train, heldout))
    tc.pop('render_views_per_target'); hc.pop('render_views_per_target')
    require(tc == hc and tc.get('view_offset', 0) > 0, 'Preserve geometry, split and nonzero view shard')
    other = {j['job_id']: j for j in heldout['jobs']}
    require(set(other) == {j['job_id'] for j in train['jobs']}, 'Mismatched family jobs')
    for job in train['jobs']:
        require({k:v for k,v in job.items() if k != 'max_rendered_views_per_target'} ==
                {k:v for k,v in other[job['job_id']].items() if k != 'max_rendered_views_per_target'},
                'Changed family targets or split')
    reserved = [p['job_id'] for p in predecessors]
    require(len(reserved) == 2 and len(set(reserved)) == 2, 'Need two distinct existing jobs')
    require(set(reserved) <= {j['job_id'] for j in train['jobs'] if j['split'] == 'train'},
            'Existing jobs must be reserved training families')
    lanes = [dict(lane_id=f'lane_{i+1:02d}', predecessor=predecessors[i] if i < 2 else None,
                  items=[]) for i in range(4)]
    loads = [p['remaining_view_estimate'] for p in predecessors] + [0, 0]
    require(all(type(v) is int and v >= 0 for v in loads), 'Invalid scheduling estimate')
    items = []
    for original in train['jobs']:
        if original['job_id'] in reserved:
            continue
        role = 'train' if original['split'] == 'train' else 'heldout'
        job = original if role == 'train' else other[original['job_id']]
        items.append(dict(job_id=job['job_id'], family=job['plant_family'], split=job['split'],
                          plan_role=role, maximum_frames=len(job['targets'])*job['max_rendered_views_per_target']))
    for item in sorted(items, key=lambda r: (-r['maximum_frames'], r['job_id'])):
        index = min(range(4), key=lambda i: (loads[i], i))
        lanes[index]['items'].append(item); loads[index] += item['maximum_frames']
    for lane in lanes:
        # Complete held-out collection early; no model is fitted to these labels.
        lane['items'].sort(key=lambda r: (r['split'] == 'train', r['job_id']))
    require(sorted(reserved + [r['job_id'] for lane in lanes for r in lane['items']]) == sorted(other),
            'Incomplete or duplicate campaign allocation')
    return lanes


def predecessor_ready(predecessor):
    root = Path(predecessor['batch'])
    require(sha256(root/'request.json') == predecessor['request_sha256'], 'Changed predecessor request')
    result_path = root/predecessor['job_id']/'result.json'
    if result_path.is_file():
        result = read_json(result_path)
        require(result.get('state') == AUDITED and result.get('returncode') == 0 and not result.get('timed_out'),
                'Predecessor capture or audit failed; do not start this lane')
    manifest_path = root/'manifest.json'
    if not manifest_path.is_file():
        return False
    ledger = read_json(manifest_path)
    require(ledger['state'] == COMPLETE and len(ledger['jobs']) == 1, 'Predecessor batch failed or changed')
    require(result_path.is_file() and ledger['jobs'][0] == result, 'Predecessor completion is inconsistent')
    require(result['job_id'] == predecessor['job_id'], 'Wrong completed predecessor job')
    audit_path = result_path.parent/'audit/audit.json'
    require(sha256(audit_path) == result['audit_sha256'], 'Changed predecessor audit')
    return True


def wait_predecessor(predecessor, timeout_s=14400):
    started = time.monotonic()
    while not predecessor_ready(predecessor):
        require(time.monotonic()-started < timeout_s, 'Predecessor wait timed out; lane stopped')
        time.sleep(5)


def create(train_path, heldout_path, after_batches, output):
    paths = {k:Path(p).resolve() for k,p in [('train',train_path),('heldout',heldout_path)]}
    plans = {k:load_plan(p)[0] for k,p in paths.items()}
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(Path(plans['train']['package'])),
            'Choose a new campaign outside source assets')
    require(len(after_batches) == 2, 'Name the two existing scale batches')
    predecessors = []
    for name in after_batches:
        root = Path(name).resolve(); request = read_json(root/'request.json')
        require(request['plan_sha256'] == sha256(paths['train']) and len(request['selected_job_ids']) == 1,
                'Predecessor must contain one job from this scale plan')
        require(request['instance_backend'] == 'fast' and request['render_budget'] == 'warm56_then8',
                'Unexpected predecessor capture profile')
        job_id = request['selected_job_ids'][0]
        prepared = read_json(root/job_id/'capture/planned_views.json')
        planned = sum(len(v) for v in prepared['selected'].values())
        saved = len(list((root/job_id/'capture').glob('sample_*/sample.json')))
        predecessors.append(dict(batch=str(root), job_id=job_id, request_sha256=sha256(root/'request.json'),
                                 remaining_view_estimate=max(0, planned-saved)))
    lanes = allocate(plans['train'], plans['heldout'], predecessors)
    maximum_frames = max(item['maximum_frames'] for lane in lanes for item in lane['items'])
    result = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                  state='scheduled_collection_not_dataset_approval', maximum_capture_workers=4,
                  worker_limit_scope='four_lanes_plus_two_named_predecessors_no_unrelated_captures',
                  plans={k:dict(path=str(p),sha256=sha256(p)) for k,p in paths.items()}, lanes=lanes,
                  instance_backend='fast', render_budget='warm56_then8', worker_timeout_s=14400,
                  # Reserve 64 GiB plus four worst-job 12 MiB/sample envelopes.
                  minimum_free_disk_bytes=64*2**30+4*maximum_frames*12*2**20,
                  automatic_retries=False, training_release_approved=False,
                  implementation_sha256=sha256(Path(__file__)))
    output.mkdir(parents=True)
    write_json(output/'campaign.json', result)
    return result


def run_lane(campaign_path, lane_id):
    campaign_path = Path(campaign_path).resolve(); campaign_hash = sha256(campaign_path)
    campaign = read_json(campaign_path)
    require(campaign['schema_version'] == SCHEMA and campaign['maximum_capture_workers'] == 4,
            'Unexpected campaign schema or worker limit')
    require(campaign['implementation_sha256'] == sha256(Path(__file__)), 'Campaign runner changed; create a new campaign')
    plans = {k:load_plan(v['path'])[0] for k,v in campaign['plans'].items()}
    verify_bindings({v['path']:v['sha256'] for v in campaign['plans'].values()})
    predecessors = [lane['predecessor'] for lane in campaign['lanes'] if lane['predecessor']]
    require(campaign['lanes'] == allocate(plans['train'], plans['heldout'], predecessors), 'Changed lane allocation')
    lane = next((r for r in campaign['lanes'] if r['lane_id'] == lane_id), None)
    require(lane is not None, 'Unknown lane')
    output = campaign_path.parent/lane_id
    output.mkdir()  # Exclusive: a duplicate lane launch cannot spawn another worker.
    status = dict(schema_version=SCHEMA, lane_id=lane_id, campaign_sha256=campaign_hash,
                  state='waiting_for_predecessor' if lane['predecessor'] else 'ready', jobs=[],
                  created_utc=datetime.now(timezone.utc).isoformat(), training_release_approved=False)
    write_json(output/'request.json', status)
    print('CAMPAIGN_LANE_START', lane_id, status['state'], flush=True)
    try:
        if lane['predecessor']:
            wait_predecessor(lane['predecessor'])
        for item in lane['items']:
            require(sha256(campaign_path) == campaign_hash, 'Campaign changed while running')
            require(shutil.disk_usage(output).free >= campaign['minimum_free_disk_bytes'],
                    'Insufficient disk reserve; no capture started and no files deleted')
            plan = campaign['plans'][item['plan_role']]
            require(sha256(plan['path']) == plan['sha256'], 'Collection plan changed while running')
            batch = output/('capture_'+item['job_id'])
            result = run_jobs(plan['path'], batch, max_jobs=1, timeout_s=14400,
                              job_ids=[item['job_id']], instance_backend='fast', render_budget='warm56_then8')
            status['jobs'].append(dict(job_id=item['job_id'], batch=str(batch), state=result['state'],
                                       manifest_sha256=sha256(batch/'manifest.json')))
            require(result['state'] == COMPLETE, 'Collection job failed; lane stopped without retry')
        status['state'] = 'complete_collection_lane_pending_visual_review'
    except BaseException as exc:
        status.update(state='stopped_collection_lane', error=str(exc))
        raise
    finally:
        write_json(output/'result.json', status)
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('create'); c.add_argument('--train-plan', type=Path, required=True)
    c.add_argument('--heldout-plan', type=Path, required=True); c.add_argument('--after-batch', action='append', required=True)
    c.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--campaign', type=Path, required=True); r.add_argument('--lane', required=True)
    a = p.parse_args(argv)
    if a.command == 'create':
        result = create(a.train_plan, a.heldout_plan, a.after_batch, a.output)
        print('CAMPAIGN_CREATED', [(v['lane_id'],len(v['items'])) for v in result['lanes']], flush=True)
    else:
        result = run_lane(a.campaign, a.lane); print('CAMPAIGN_LANE_RESULT', a.lane, result['state'], flush=True)


if __name__ == '__main__': main()
