"""Read-only collection/label-yield status; never promotes a capture to a release."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .audit import audit_manifest
from .dataset_review import read_json, require, safe_file
from .depth_preview import sha256
from .training_contract import derive_label


def summarize(plan_path, batch):
    plan_path, batch = Path(plan_path).resolve(), Path(batch).resolve()
    plan = read_json(plan_path)
    require(plan.get('schema_version') == 'greenhouse.grounding_collection_plan.v1', 'Expected grounding plan')
    reports, jobs = {}, []
    for scheduled in plan['jobs']:
        folder = batch / scheduled['job_id']
        if not folder.is_dir():
            continue
        capture = folder / 'capture'
        result = read_json(folder / 'result.json') if (folder / 'result.json').is_file() else None
        audit_path = Path(result.get('audit_path', str(folder/'audit/audit.json'))) if result else folder/'audit/audit.json'
        require(audit_path.resolve().is_relative_to(folder), 'Audit receipt escapes the job directory')
        audited = bool(result and result.get('returncode') == 0 and not result.get('timed_out')
                       and result.get('state') == 'audited_prototype_pending_visual_review'
                       and audit_path.is_file() and sha256(audit_path) == result.get('audit_sha256'))
        family = scheduled['plant_family']
        if family not in reports:
            reports[family] = audit_manifest(scheduled['source_manifest_path'])
        rows, incomplete = [], []
        for path in sorted(capture.glob('sample_*/sample.json')):
            try:
                metadata = read_json(path)
                # A live writer may still be finishing this sample. Do not count it.
                for name, detail in metadata['files'].items():
                    require(sha256(safe_file(path.parent, name)) == detail['sha256'], 'Unfinished or changed artifact')
                label = derive_label(path.parent, metadata, reports[family])
                rows.append(dict(sample_id=path.parent.name, target_id=label['target_id'],
                                 eligible_under_label_contract=label['eligible'], reason=label['reason'],
                                 difficulty=label.get('difficulty'), answer=label.get('answer')))
            except (OSError, ValueError, KeyError) as exc:
                incomplete.append(dict(sample_id=path.parent.name, reason=str(exc)))
        jobs.append(dict(job_id=scheduled['job_id'], family=family, split=scheduled['split'],
                         worker_state=result['state'] if result else 'running_or_unfinalized',
                         independently_audited=audited, complete_sample_files=len(rows),
                         provisional_label_counts=dict(Counter(r['difficulty'] for r in rows if r['eligible_under_label_contract'])),
                         exclusion_counts=dict(Counter(r['reason'] for r in rows if not r['eligible_under_label_contract'])),
                         incomplete_or_changed_samples=incomplete, samples=rows))
    return dict(schema_version='greenhouse.grounding_progress.v1',
                state='status_only_not_a_training_release', plan_sha256=sha256(plan_path),
                total_complete_sample_files=sum(j['complete_sample_files'] for j in jobs),
                audited_sample_files=sum(j['complete_sample_files'] for j in jobs if j['independently_audited']),
                release_approved=False, jobs=jobs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--details', action='store_true')
    args = p.parse_args(argv)
    result = summarize(args.plan, args.batch)
    if not args.details:
        for job in result['jobs']: job.pop('samples')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
