"""Local task-v3 card reviewer. No simulator, image generation or model calls.

Human decisions are append-only; legacy assistant decisions can receive a
separate human follow-up. Suggestions never count as final decisions.
Negative decisions block the source first, so interrupted saves fail closed.
Human is a local self-declared reviewer role, not authenticated identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.request import urlopen
import uuid
import webbrowser

from .dataset_package import REVIEW_SCHEMA, active_reviews, validate_record
from .dataset_review import HEAD_CAMERA, SCHEMA as AUDIT_SCHEMA, read_json, require, safe_file, write_json
from .depth_preview import sha256
from .review_gui import make_server
from .training_contract import contract_hash, validate_answer
from .training_release_review import CHECKLIST, POLICY, SCHEMA, record
from .review_suggestions import Suggestions

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT/'data/sim_data/dataset_reviews/grounding_all_sources_20260909_v3/bundle.json'
UI = Path(__file__).with_name('training_review_gui_assets')
GUI_VERSION = 'assistant_suggestions_and_human_followup.v1'
DEFAULT_SUGGESTIONS = ROOT/'data/sim_data/dataset_audits/independent_v3_20260909'
SOURCE_FILES = ('inputs/rgb.png', 'inputs/depth_m.npy', 'inputs/depth_valid.png', 'supervision/target_visible.png')


class CheckedImages:
    """Allowlisted lazy reads; don't preload hundreds of megabytes into RAM."""
    def __init__(self): self.paths = {}
    def __contains__(self, key): return key in self.paths
    def __getitem__(self, key):
        path, expected = self.paths[key]
        data = path.read_bytes()
        import hashlib
        require(hashlib.sha256(data).hexdigest() == expected, 'Displayed evidence changed; reload only after re-audit')
        return data


class TrainingReviewApp:
    def __init__(self, bundle_path, suggestions=None):
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle_hash = sha256(self.bundle_path)
        bundle = read_json(self.bundle_path)
        require(bundle.get('schema_version') == SCHEMA and bundle.get('policy') == POLICY, 'Expected current task-v3 visual bundle')
        self.records = self.bundle_path.parent/'decisions'
        self.lock = threading.Lock()
        self.token = secrets.token_hex(32)
        self.images = CheckedImages()
        self.entries = {}
        self.sources = {}
        audits = {}
        for name, expected in bundle['source_audits_sha256'].items():
            path = Path(name).resolve()
            require(sha256(path) == expected and expected not in audits, 'Changed or duplicate source audit')
            a = read_json(path)
            require(a['schema_version'] == AUDIT_SCHEMA and a['state'] == 'complete_engineering_audit_not_approval'
                    and a['training_dataset_approved'] is False, 'Source audit not complete')
            samples = {}
            for p, h in a['bindings_sha256'].items():
                if Path(p).name == 'sample.json': samples.setdefault(h, []).append(Path(p).resolve())
            audits[expected] = (path, a, samples)
        for e in bundle['entries']:
            sid = e['id']
            require(isinstance(sid, str) and re.fullmatch(r'[A-Za-z0-9_-]+', sid) and sid not in self.entries, 'Unsafe or duplicate entry ID')
            require(e['task_contract_sha256'] == contract_hash(), 'Stale task contract')
            validate_answer(e['answer'])
            path, audit, samples = audits[e['source_audit_sha256']]
            matches = samples.get(e['source_sample_sha256'], [])
            require(len(matches) == 1, 'Missing or ambiguous source sample binding')
            metadata_path = matches[0]
            directory = metadata_path.parent
            require(directory.parent == Path(audit['source_run']).resolve(), 'Source sample outside audited capture')
            checked = next((s for s in audit['samples'] if s['sample_id'] == directory.name), None)
            require(checked and checked['integrity_and_recomputed_annotations_passed'] is True
                    and checked['camera']['mounted_robot_pov_verified'] is True
                    and checked['camera']['camera_path'] == HEAD_CAMERA, 'Expected audited mounted robot-head frame')
            require(sha256(metadata_path) == e['source_sample_sha256'], 'Changed source sample')
            metadata = read_json(metadata_path)
            require(metadata['files']['inputs/rgb.png']['sha256'] == e['rgb_sha256'], 'Wrong original RGB binding')
            card = safe_file(self.bundle_path.parent, e['card'])
            files = {metadata_path: e['source_sample_sha256'], card: e['card_sha256']}
            for name in SOURCE_FILES:
                source = safe_file(directory, name)
                expected = metadata['files'][name]['sha256']
                require(audit['bindings_sha256'].get(str(source)) == expected, 'Observation not bound by independent audit')
                files[source] = expected
            self.entries[sid] = e
            self.sources[sid] = dict(audit_path=path, audit=audit, sample=checked, files=files)
            self.images.paths[sid, 'card'] = (card, e['card_sha256'])
            self.images.paths[sid, 'rgb'] = (directory/'inputs/rgb.png', e['rgb_sha256'])
            self._check_source(sid)
        self.suggestions = Suggestions(suggestions, self.bundle_hash, self.entries,
                                       {sid: s['files'] for sid, s in self.sources.items()})
        self.suggestions_directory = str(Path(suggestions).resolve()) if suggestions else None
        self._state()

    def _check_source(self, sid):
        s = self.sources[sid]
        require(sha256(self.bundle_path) == self.bundle_hash, 'Bundle changed; restart after review reconciliation')
        require(sha256(s['audit_path']) == self.entries[sid]['source_audit_sha256'], 'Source audit changed')
        for path, expected in s['files'].items(): require(sha256(path) == expected, 'Source evidence changed: '+str(path))

    def _source_history(self, sid):
        source = self.sources[sid]
        paths = sorted((source['audit_path'].parent/'records').glob('*.json'))
        rows = [read_json(p) for p in paths]
        active = active_reviews(rows, source['audit'], self.entries[sid]['source_audit_sha256'])
        return active, {r['review_id']: p for r, p in zip(rows, paths)}

    def _state(self):
        require(sha256(self.bundle_path) == self.bundle_hash, 'Bundle changed; restart reviewer')
        self.suggestions.check()
        decisions = {}
        previous = {}
        paths = sorted(self.records.glob('*.json')) + sorted((self.bundle_path.parent/'human_decisions').glob('*.json'))
        for path in paths:
            d = read_json(path); e = d['entry']; sid = e['id']
            require(d['schema_version'] == SCHEMA and d['bundle_sha256'] == self.bundle_hash
                    and e == self.entries.get(sid) and path.name == sid+'.json', 'Stale or unknown saved decision')
            require(d['decision'] in ('accept', 'hold', 'reject') and d['reviewer_role'] in ('human', 'assistant')
                    and d['inspected'] == CHECKLIST and d['human_confirmation'] is (d['reviewer_role'] == 'human')
                    and d['physical_execution_approved'] is False, 'Invalid saved review scope')
            require(sha256(self.images.paths[sid, 'card'][0]) == e['card_sha256'], 'Saved review card changed')
            if path.parent.name == 'human_decisions':
                require(d['reviewer_role'] == 'human' and sid in previous
                        and d.get('prior_assistant_review_sha256') == sha256(self.records/(sid+'.json')),
                        'Invalid human follow-up history')
            if d['reviewer_role'] == 'human':
                require(sid not in decisions, 'Duplicate human decision')
                decisions[sid] = d
            else:
                previous[sid] = d
        result = []
        for sid, e in self.entries.items():
            active, _ = self._source_history(sid)
            blocks = [r for (sample, _), r in active.items() if sample == self.sources[sid]['sample']['sample_id'] and r['decision'] in ('hold', 'reject')]
            result.append(dict(id=sid, target=e['target_id'], split=e['split'], difficulty=e['difficulty'],
                query=e['query_pixel_uv'], answer=e['answer'], decision=decisions.get(sid),
                previous_decision=previous.get(sid), suggestion=self.suggestions.rows.get(sid),
                source_blocks=[dict(decision=r['decision'], notes=r['notes'], role=r['reviewer_role']) for r in blocks],
                images={kind:f'/image/{sid}/{kind}' for kind in ('rgb', 'card')}))
        return dict(gui_version=GUI_VERSION, bundle_sha256=self.bundle_hash, bundle_path=str(self.bundle_path), samples=result,
                    suggestions_directory=self.suggestions_directory,
                    total=len(result), pending=len(result)-len(decisions), recorded=dict(Counter(d['decision'] for d in decisions.values())),
                    human_recorded=sum(d['reviewer_role']=='human' for d in decisions.values()),
                    legacy_assistant_recorded=len(previous), suggestions_count=len(self.suggestions.rows),
                    advisory_holds=sum(s['suggested_decision']=='hold' for s in self.suggestions.rows.values()),
                    records_directory=str(self.records), training_release_approved=False)

    def state(self):
        with self.lock: return self._state()

    def _block_source(self, sid, reviewer, decision, notes):
        """Negative-only append; uses existing source-review schema/export gate.

        Source/audit/card/native files for this inspected frame are checked before
        saving. This cannot recommend/confirm a numerical pilot or clear a hold.
        """
        source = self.sources[sid]; sample = source['sample']
        active, paths = self._source_history(sid)
        prior = active.get((sample['sample_id'], 'human'))
        reason = 'Task-v3 human GUI '+self.entries[sid]['id']+': '+notes
        if prior and prior['decision'] == decision and prior['notes'] == reason and prior['reviewer'] == reviewer:
            return paths[prior['review_id']]  # Retry after interrupted task record; no duplicate history.
        row = dict(schema_version=REVIEW_SCHEMA, review_id=uuid.uuid4().hex,
            created_utc=datetime.now(timezone.utc).isoformat(), audit_sha256=self.entries[sid]['source_audit_sha256'],
            sample_id=sample['sample_id'], target_review_id=sample['target_review_id'], reviewer_role='human',
            reviewer=reviewer, decision=decision, notes=reason,
            inspected=['full_scene_rgb', 'cut_overlay', 'native_identity_mask', 'native_depth'],
            human_prototype_label_confirmation=False, training_eligible=False, physical_cut_approved=False,
            horticultural_validation='pending', supersedes_review_id=prior['review_id'] if prior else None,
            identity_authentication='local_self_declared_role_not_authenticated')
        validate_record(row, source['audit'], row['audit_sha256'])
        output = source['audit_path'].parent/'records'; output.mkdir(exist_ok=True)
        path = output/(row['review_id']+'.json'); write_json(path, row)
        return path

    def save(self, payload):
        with self.lock:
            required = {'sample_id', 'reviewer', 'decision', 'notes', 'inspected', 'bundle_sha256'}
            require(isinstance(payload, dict) and required <= set(payload) <= required | {'suggestion_context'}, 'Invalid review fields')
            sid, reviewer, decision, notes = (payload[k] for k in ('sample_id', 'reviewer', 'decision', 'notes'))
            require(isinstance(sid, str) and sid in self.entries, 'Unknown sample')
            require(payload['bundle_sha256'] == self.bundle_hash, 'Stale browser bundle')
            require(isinstance(reviewer, str) and 0 < len(reviewer.strip()) <= 120, 'Enter your name')
            require(isinstance(notes, str) and 20 <= len(notes.strip()) <= 4000, 'Add an inspection note of at least 20 characters')
            require(isinstance(decision, str) and decision in ('accept', 'hold', 'reject'), 'Invalid decision')
            require(payload['inspected'] is True, 'Confirm actual RGB, query, answer, mask and native-depth inspection')
            self._check_source(sid)
            state = self._state()
            current = next(s for s in state['samples'] if s['id'] == sid)
            require(current['decision'] is None, 'Already recorded; human decisions are read-only. Refresh the page.')
            suggestion = self.suggestions.rows.get(sid)
            context = payload.get('suggestion_context')
            if suggestion is not None:
                require(isinstance(context, dict) and set(context) == {'sha256', 'response'}
                        and context['sha256'] == suggestion['sha256']
                        and context['response'] in ('independent', 'agree', 'disagree'), 'Choose how your decision relates to the current suggestion')
                require(context['response'] != 'agree' or decision == suggestion['suggested_decision'], 'Decision does not agree with suggestion')
                require(context['response'] != 'disagree' or decision != suggestion['suggested_decision'], 'Decision does not differ from suggestion')
            else:
                require(context is None, 'No current suggestion for this card')
            prior = current['previous_decision']
            require(decision != 'accept' or not prior or prior['decision'] == 'accept', 'Acceptance cannot clear an earlier task hold')
            active, _ = self._source_history(sid)
            blocked = any(r['decision'] in ('hold','reject') for (sample,_),r in active.items() if sample == self.sources[sid]['sample']['sample_id'])
            require(decision != 'accept' or not blocked, 'Source is held/rejected; acceptance cannot clear it')
            source_record = None
            if decision in ('hold', 'reject'):
                source_record = self._block_source(sid, reviewer.strip(), decision, notes.strip())
            path = record(self.bundle_path, [sid], notes.strip(), decision=decision, role='human', reviewer=reviewer.strip(), inspected=True,
                          human_followup=prior is not None, suggestion_context=context)[0]
            return dict(saved=read_json(path), source_block_path=str(source_record) if source_record else None, state=self._state())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    p.add_argument('--port', type=int, default=8880)
    p.add_argument('--open', action='store_true')
    p.add_argument('--suggestions', type=Path, help='Hash-bound independent assessment directory')
    p.add_argument('--no-suggestions', action='store_true', help='Explicitly review without advisory suggestions')
    a = p.parse_args(argv); url = f'http://127.0.0.1:{a.port}'
    require(not (a.suggestions and a.no_suggestions), 'Choose suggestions or no-suggestions')
    suggestions = a.suggestions
    if not a.no_suggestions and suggestions is None and a.bundle.resolve() == DEFAULT_BUNDLE.resolve() and DEFAULT_SUGGESTIONS.exists():
        suggestions = DEFAULT_SUGGESTIONS
    if a.open and a.port:
        existing = None
        try:
            with urlopen(url+'/api/state', timeout=2) as response: existing = json.load(response)
            if (existing.get('gui_version') == GUI_VERSION and existing.get('bundle_sha256') == sha256(a.bundle)
                    and Path(existing.get('bundle_path','')).resolve() == a.bundle.resolve()
                    and existing.get('suggestions_directory') == (str(suggestions.resolve()) if suggestions else None)):
                webbrowser.open(url); print('Opened existing task-v3 reviewer: '+url, flush=True); return
        except (OSError, ValueError, AttributeError): pass
        if existing is not None:
            p.error(f'Port {a.port} already serves a different/older reviewer. Leave ongoing reviews intact; choose --port 8881 (or another free port).')
    app = TrainingReviewApp(a.bundle, suggestions=suggestions)
    with make_server(app, a.port, ui=UI) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Task-v3 reviewer ready: {url} | {app.state()["pending"]} pending | no simulator/GPU', flush=True)
        if a.open: webbrowser.open(url)
        try: server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt: pass


if __name__ == '__main__': main()
