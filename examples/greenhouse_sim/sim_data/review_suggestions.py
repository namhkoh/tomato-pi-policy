"""Read-only, hash-bound assistant advice. Never writes a review or export gate."""
from pathlib import Path

from .dataset_review import read_json, require, safe_file
from .depth_preview import sha256
from .training_release_review import identity


class Suggestions:
    def __init__(self, directory, bundle_hash, entries, source_files):
        self.rows, self.bindings = {}, {}
        if directory is None:
            return
        directory = Path(directory).resolve()
        scope_path, report_path = directory/'scope.json', directory/'assessment.json'
        scope, report = read_json(scope_path), read_json(report_path)
        require(scope.get('schema') == 'independent_advisory_visual_inspection.v1'
                and scope.get('bundle_sha256') == bundle_hash, 'Suggestions bind a different bundle')
        require(report.get('schema') == 'independent_advisory_assessment.v1'
                and report.get('bundle_sha256') == bundle_hash
                and report.get('scope_sha256') == sha256(scope_path), 'Stale suggestion assessment')
        self.bindings = {scope_path: sha256(scope_path), report_path: sha256(report_path)}
        scoped = {e['id']: e for e in scope['entries']}
        require(len(scoped) == len(scope['entries']), 'Duplicate suggestion scope')
        require(set(report['finding_files']) == set(scoped)
                and report['inspected_count'] == len(scoped), 'Incomplete suggestion assessment')
        for sid, expected in report['finding_files'].items():
            require(sid in entries, 'Unknown suggestion sample')
            path = safe_file(directory, 'findings/'+sid+'.json')
            require(sha256(path) == expected, 'Changed suggestion finding')
            row = read_json(path)
            require(row.get('schema') == 'independent_advisory_visual_finding.v1'
                    and row.get('id') == sid and row.get('entry') == scoped[sid]
                    and row.get('scope_sha256') == sha256(scope_path), 'Stale suggestion binding')
            require(identity(row['entry']) == identity(entries[sid])
                    and row['entry']['card_sha256'] == entries[sid]['card_sha256'], 'Suggestion label mismatch')
            require(row.get('reviewer_role') == 'assistant' and row.get('actual_visual_inspection') is True
                    and all(row.get(k) is False for k in ('human_confirmation', 'training_release_approved',
                                                        'physical_execution_approved', 'gui_decision_written', 'source_hold_written')),
                    'Suggestion cannot claim approval or saved decisions')
            require(row.get('assessment') in ('support', 'hold_recommended')
                    and isinstance(row.get('notes'), str) and len(row['notes'].strip()) >= 20,
                    'Invalid advisory assessment')
            bound = {str(p.resolve()): h for p, h in source_files[sid].items()}
            require(row['entry']['binding_sha256'] == bound, 'Suggestion source evidence mismatch')
            self.bindings[path] = expected
            self.rows[sid] = dict(assessment=row['assessment'], issue=row.get('issue'), notes=row['notes'],
                suggested_decision='accept' if row['assessment'] == 'support' else 'hold',
                reviewer_role='assistant', human_confirmation=False, sha256=expected,
                created_utc=row['created_utc'], source_path=str(path))
        self.check()

    def check(self):
        for path, expected in self.bindings.items():
            require(sha256(path) == expected, 'Suggestion evidence changed; restart after reconciliation')
