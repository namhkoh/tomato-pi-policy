"""Task-v4 native-evidence reviewer. New records never modify task-v3 decisions.

Human final decisions and assistant advice have separate append-only files.
The shared HTTP server enforces loopback, origin/token and allowlisted resources.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.request import urlopen
import uuid
import webbrowser

import numpy as np
from PIL import Image

from .active_perception_contract import TASK_ID, contract_hash, user_prompt, validate_evidence
from .active_perception_pilot import SCHEMA as PILOT_SCHEMA, read_source
from .capture_contract import fingerprint
from .dataset_review import HEAD_CAMERA, read_json, require, write_json
from .depth_preview import sha256, labelled_heatmap
from .review_gui import make_server
from .training_review_gui import TrainingReviewApp, ROOT

GUI_VERSION = 'active_perception_native_review.v1'
REVIEW_SCHEMA = 'greenhouse.active_perception_review.v1'
CHECKLIST = ['original_full_rgb', 'scoped_query_or_region', 'proposed_answer', 'native_identity', 'native_optical_z']
UI = Path(__file__).with_name('active_perception_review_assets')
DEFAULT_PILOT = ROOT/'data/sim_data/dataset_reviews/active_perception_20260910_v3/pilot.json'


class EvidenceImages:
    def __init__(self, app): self.app = app
    def __contains__(self, key): return key[0] in self.app.entries and key[1] in ('rgb', 'mask', 'depth')
    def __getitem__(self, key):
        sid, kind = key
        self.app.check(sid)
        rgb, depth, valid, mask = self.app.buffers(sid)
        if kind == 'rgb': return Path(self.app.entries[sid]['model_input']['rgb_path']).read_bytes()
        if kind == 'depth':
            picture = labelled_heatmap(depth, valid, .04, 2., 'Saved Isaac native optical-Z; display only')
        else:
            pixels = rgb.copy(); pixels[mask] = [0,255,80]
            picture = Image.fromarray(pixels)
        stream = io.BytesIO(); picture.save(stream, format='PNG'); return stream.getvalue()


class ActiveReview:
    def __init__(self, pilot_path):
        self.path = Path(pilot_path).resolve(); self.pilot_hash = sha256(self.path)
        self.pilot = read_json(self.path)
        require(self.pilot.get('schema') == PILOT_SCHEMA and self.pilot.get('task_id') == TASK_ID
                and self.pilot.get('task_contract_sha256') == contract_hash()
                and self.pilot.get('training_eligible') is False and self.pilot.get('dynamic_episodes_collected') == 0,
                'Expected current static, unapproved task-v4 pilot')
        self.bindings = {Path(p).resolve(): h for p,h in self.pilot['bindings_sha256'].items()}
        bundle_paths = [p for p,h in self.bindings.items() if p.name == 'bundle.json' and h == self.pilot['source_bundle_sha256']]
        require(len(bundle_paths) == 1, 'Ambiguous or missing source bundle')
        advice = [p.parent for p in self.bindings if p.name == 'assessment.json']
        require(len(advice) <= 1, 'Ambiguous assistant assessment')
        self.source_app = TrainingReviewApp(bundle_paths[0], advice[0] if advice else None)
        self.records = self.path.parent/'v4_reviews'
        self.token = secrets.token_hex(32); self.lock = threading.RLock()
        self.entries, self.metadata = {}, {}
        self._verify_bindings()
        for entry in self.pilot['examples']:
            sid = entry['id']
            require(isinstance(sid,str) and re.fullmatch(r'[A-Za-z0-9_-]+',sid) and sid not in self.entries, 'Invalid example ID')
            require(entry['task_id'] == TASK_ID and entry['task_contract_sha256'] == contract_hash()
                    and entry['training_eligible'] is False and entry['physical_execution_approved'] is False
                    and entry['human_confirmation'] is False, 'New task cannot inherit approvals')
            require(entry['source_id'] in self.source_app.entries, 'Unknown native source')
            base = self.source_app.entries[entry['source_id']]
            require(entry['split'] == base['split'] and entry['source_plant_family'] == base['source_plant_family'], 'Changed source split/family')
            directory, meta = read_source(self.source_app, entry['source_id'])
            inp, side = entry['model_input'], entry['native_sidecars']
            require(Path(inp['rgb_path']).resolve() == directory/'inputs/rgb.png'
                    and Path(side['depth_path']).resolve() == directory/'inputs/depth_m.npy'
                    and Path(side['depth_valid_path']).resolve() == directory/'inputs/depth_valid.png', 'Wrong native source paths')
            require(side['calibration'] == meta['calibration'] and side['robot_snapshot'] == meta['robot_snapshot']
                    and side['synchronization'] == meta['synchronization'], 'Changed native calibration/state')
            require(meta['calibration']['resolution'] == [848,408] and meta['calibration']['camera_path'] == HEAD_CAMERA
                    and meta['calibration']['depth_convention'] == 'optical_axis_z_metres_not_ray_range', 'Expected robot-head native optical-Z')
            for p in self.source_app.sources[entry['source_id']]['files']:
                require(self.bindings.get(p) == sha256(p), 'Source evidence missing from pilot binding')
            query, region = inp.get('query_pixel_uv'), inp.get('region_xyxy')
            require(inp['instruction'] == user_prompt(query=query, region=region) and inp['history'] == [], 'Changed input scope or invented history')
            mode = 'scene_region' if region is not None else 'candidate_query'
            validate_evidence(entry['proposed_answer'], entry['proposed_observable_evidence'], mode=mode)
            kind, answer = entry['kind'], entry['proposed_answer']
            require(kind in ('visible','occluded','invalid_candidate','uncertain_region'),
                    'Unsupported capture class: true absence still needs a complete scoped capture audit')
            if kind in ('visible','occluded'):
                require(query == base['query_pixel_uv'] and region is None, 'Changed native target query')
                require((kind == 'visible' and base['answer']['status'] == 'localized' and answer['decision'] == 'localize'
                         and answer['cut_point_uv'] == base['answer']['cut_point_uv'])
                        or (kind == 'occluded' and base['answer']['status'] == 'abstain' and answer['decision'] == 'inspect'
                            and answer['target_state'] == 'unknown' and answer['cut_visibility'] == 'occluded'), 'Changed source cut/occlusion label')
            if kind == 'invalid_candidate':
                require(query is not None and region is None and answer['decision'] == 'reject', 'Invalid candidate task mismatch')
            if kind == 'uncertain_region':
                require(region is not None and query is None and answer['decision'] == 'inspect'
                        and answer['target_state'] == 'unknown' and answer['cut_visibility'] == 'unknown', 'Obstruction is not absence')
            self.entries[sid] = entry; self.metadata[sid] = (directory, meta)
            self.buffers(sid)  # Verify native identity and validity before displaying.
        self.images = EvidenceImages(self)
        self.state()

    def _verify_bindings(self):
        require(sha256(self.path) == self.pilot_hash, 'Pilot changed; review reconciliation required')
        for path, expected in self.bindings.items(): require(sha256(path) == expected, 'Bound evidence changed: '+str(path))

    def check(self, sid):
        require(sid in self.entries and sha256(self.path) == self.pilot_hash, 'Unknown or changed pilot')
        self.source_app._check_source(self.entries[sid]['source_id'])
        directory, meta = self.metadata[sid]
        if self.entries[sid]['kind'] in ('invalid_candidate','uncertain_region'):
            for name in ('supervision/component_id.npy','supervision/identities.json'):
                p=directory/name; h=meta['files'][name]['sha256']
                require(self.bindings.get(p) == h and sha256(p) == h, 'Changed native region/component identity')

    def buffers(self, sid):
        self.check(sid)
        entry = self.entries[sid]; directory, meta = self.metadata[sid]
        rgb = np.asarray(Image.open(directory/'inputs/rgb.png'))
        depth = np.load(directory/'inputs/depth_m.npy',allow_pickle=False)
        validity = np.asarray(Image.open(directory/'inputs/depth_valid.png'))
        require(rgb.shape == (408,848,3) and rgb.dtype == np.uint8 and depth.shape == (408,848)
                and depth.dtype == np.float32 and validity.shape == depth.shape, 'Native observation format mismatch')
        near, far = meta['calibration']['clipping_range_m']
        valid = np.isfinite(depth) & (depth>0) & (depth>=near) & (depth<=far)
        require(np.array_equal(validity, valid.astype(np.uint8)*255), 'Validity is not the captured native depth validity')
        if entry['kind'] in ('visible','occluded'):
            mask = np.asarray(Image.open(directory/'supervision/target_visible.png')) == 255
        else:
            ids = np.load(directory/'supervision/component_id.npy',allow_pickle=False)
            require(ids.shape == depth.shape and np.issubdtype(ids.dtype,np.integer), 'Invalid native component array')
            organ = entry['private_evaluator']['observed_organ']
            catalogue = read_json(directory/'supervision/identities.json')['component_catalogue']
            require(organ in catalogue, 'Proposed organ is not in the native catalogue')
            mask = ids == organ['component_index']
            if entry['kind'] == 'invalid_candidate':
                x,y = np.floor(entry['model_input']['query_pixel_uv']).astype(int)
                require(mask[y,x] and valid[y,x] and organ['organ_type'] == entry['proposed_observable_evidence']['identified_ineligible_organ'],
                        'Invalid query does not hit the claimed visible organ')
            else:
                region = entry['model_input']['region_xyxy']
                require(all(type(v) is int for v in region), 'Native region uses integer edge coordinates')
                x,y,r,b = region
                require(organ['organ_type'] == 'leaf' and mask[y:b,x:r].all() and valid[y:b,x:r].all(), 'Unknown-region proposal is not a fully leaf-covered valid-depth patch')
        require(mask.shape == depth.shape, 'Invalid native mask')
        return rgb, depth, valid, mask

    def history(self):
        rows = {}
        for role in ('assistant','human'):
            for path in sorted((self.records/role).glob('*.json')):
                row=read_json(path); sid=row['sample_id']
                require(sid in self.entries and path.name == sid+'.json' and row['schema'] == REVIEW_SCHEMA
                        and row['pilot_sha256'] == self.pilot_hash and row['example_sha256'] == fingerprint(self.entries[sid]), 'Stale v4 review')
                require(row['reviewer_role'] == role and row['human_confirmation'] is (role == 'human')
                        and row['decision'] in ('accept','hold','reject') and row['inspected'] == CHECKLIST
                        and row['physical_execution_approved'] is False, 'Invalid v4 review scope')
                require(isinstance(row.get('notes'),str) and 20 <= len(row['notes'].strip()) <= 4000
                        and isinstance(row.get('reviewer'),str) and 0 < len(row['reviewer'].strip()) <= 120,
                        'Invalid v4 review attribution/note')
                rows[sid,role] = row
        return rows

    def state(self):
        with self.lock:
            require(sha256(self.path) == self.pilot_hash, 'Pilot changed')
            history = self.history()
            source_state = {s['id']:s for s in self.source_app.state()['samples']}
            samples=[]
            for sid,e in self.entries.items():
                source=source_state[e['source_id']]; blocks=list(source['source_blocks'])
                for name in ('decision','previous_decision'):
                    if source.get(name) and source[name]['decision'] != 'accept': blocks.append(dict(decision=source[name]['decision'],notes=source[name]['notes']))
                suggestion=source.get('suggestion'); human=source.get('decision') or {}
                if suggestion and suggestion['suggested_decision']=='hold' and not (
                    human.get('decision')=='accept' and (human.get('suggestion_context') or {}).get('sha256')==suggestion['sha256']):
                    blocks.append(dict(decision='hold',notes='Unresolved source advisory flag'))
                samples.append(dict(id=sid,kind=e['kind'],split=e['split'],family=e['source_plant_family'],
                    input=e['model_input'],answer=e['proposed_answer'],source_blocks=blocks,
                    assistant_review=history.get((sid,'assistant')),human_review=history.get((sid,'human')),
                    images={k:f'/image/{sid}/{k}' for k in ('rgb','mask','depth')}))
            return dict(gui_version=GUI_VERSION,pilot_sha256=self.pilot_hash,pilot_path=str(self.path),samples=samples,
                total=len(samples),human_reviewed=sum(s['human_review'] is not None for s in samples),
                assistant_reviewed=sum(s['assistant_review'] is not None for s in samples),
                kinds=dict(Counter(s['kind'] for s in samples)), records_directory=str(self.records),
                training_release_approved=False,dynamic_episodes=0)

    def record(self,sid,decision,notes,*,role,reviewer,inspected,expected_hash):
        with self.lock:
            require(role in ('assistant','human') and decision in ('accept','hold','reject'), 'Invalid reviewer/decision')
            require(isinstance(sid,str) and sid in self.entries and expected_hash == self.pilot_hash, 'Stale pilot or unknown example')
            require(inspected is True and isinstance(notes,str) and 20 <= len(notes.strip()) <= 4000,
                    'Actual inspection and a substantive note are required')
            require(isinstance(reviewer,str) and 0 < len(reviewer.strip()) <= 120, 'Reviewer name required')
            self._verify_bindings(); self.check(sid)
            state=self.state(); current=next(s for s in state['samples'] if s['id']==sid)
            require(current[role+'_review'] is None, 'Never overwrite a saved v4 review')
            require(decision != 'accept' or not current['source_blocks'], 'Source is held; v4 cannot bypass source review')
            row=dict(schema=REVIEW_SCHEMA,review_id=uuid.uuid4().hex,created_utc=datetime.now(timezone.utc).isoformat(),
                sample_id=sid,pilot_sha256=self.pilot_hash,example_sha256=fingerprint(self.entries[sid]),
                reviewer_role=role,reviewer=reviewer.strip(),decision=decision,notes=notes.strip(),inspected=CHECKLIST,
                human_confirmation=role=='human',physical_execution_approved=False,
                identity_authentication='local_self_declared_role_not_authenticated')
            folder=self.records/role; folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/(sid+'.json'),row)
            return row

    def save(self,payload):
        require(isinstance(payload,dict) and set(payload)=={'sample_id','decision','notes','reviewer','inspected','pilot_sha256'}, 'Invalid human-review fields')
        row=self.record(payload['sample_id'],payload['decision'],payload['notes'],role='human',reviewer=payload['reviewer'],
                        inspected=payload['inspected'],expected_hash=payload['pilot_sha256'])
        return dict(saved=row,state=self.state())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=DEFAULT_PILOT); p.add_argument('--port',type=int,default=8882); p.add_argument('--open',action='store_true')
    a=p.parse_args(argv); url=f'http://127.0.0.1:{a.port}'
    if a.open and a.port:
        try:
            with urlopen(url+'/api/state',timeout=2) as response: existing=json.load(response)
        except (OSError,ValueError): existing=None
        if existing is not None:
            require(existing.get('gui_version')==GUI_VERSION and existing.get('pilot_sha256')==sha256(a.pilot)
                    and existing.get('pilot_path')==str(a.pilot.resolve()), 'Port is serving another reviewer; use a free port')
            webbrowser.open(url); print('Opened existing v4 review: '+url,flush=True); return
    app=ActiveReview(a.pilot)
    with make_server(app,a.port,ui=UI) as server:
        url=f'http://127.0.0.1:{server.server_port}'; print('Task-v4 reviewer ready: '+url,flush=True)
        if a.open: webbrowser.open(url)
        try: server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt: pass


if __name__=='__main__': main()
