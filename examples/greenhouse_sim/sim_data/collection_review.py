"""One lightweight review page for a completed multi-plant collection batch.

Routes explicit human decisions to the original per-job audit/records directory.
No capture files, old review snapshots or existing browser sessions are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.request import urlopen
import webbrowser

from .dataset_review import read_json, require
from .depth_preview import sha256
from .review_gui import ReviewApp, make_server


class BatchReview:
    def __init__(self, batch, app_factory=ReviewApp):
        self.batch = Path(batch).resolve()
        self.manifest_path = self.batch/'manifest.json'
        manifest = read_json(self.manifest_path)
        require(manifest.get('state')=='complete_bounded_batch_pending_visual_review'
                and manifest.get('training_dataset_approved') is False, 'Expected a completed audited prototype batch')
        self.audit_hash = sha256(self.manifest_path)
        self.token = secrets.token_hex(32)
        self.lock = threading.RLock()
        self.apps, self.routes, self.images = {}, {}, {}
        for job in manifest['jobs']:
            jid = job['job_id']
            require(re.fullmatch(r'job_[0-9]{3}',jid) is not None and jid not in self.apps, 'Invalid/duplicate batch job')
            require(job['state']=='audited_prototype_pending_visual_review' and job['returncode']==0
                    and job['training_eligible'] is False, 'Job did not pass process and audit gates')
            audit = self.batch/jid/'audit/audit.json'
            require(audit.resolve().is_relative_to(self.batch) and sha256(audit)==job['audit_sha256'], 'Batch audit changed')
            app = app_factory(audit,audit.parent/'records')
            self.apps[jid] = app
            for (sid, kind), content in app.images.items():
                key = jid+'_'+sid
                self.routes[key] = (jid, sid)
                self.images[(key,kind)] = content
        require(bool(self.apps), 'Empty review batch')

    def _state(self):
        require(sha256(self.manifest_path)==self.audit_hash, 'Batch changed; restart reviewer')
        samples=[]
        for jid, app in self.apps.items():
            for source in app.state()['samples']:
                key=jid+'_'+source['sample_id']
                samples.append({**source,'sample_id':key,
                    'images':{kind:f'/image/{key}/{kind}' for kind in source['images']}})
        samples.sort(key=lambda s:(not s['recommended'],s['sample_id']))
        return {'samples':samples,'audit_sha256':self.audit_hash,'capture_run_name':self.batch.name,
                'total':len(samples),'reviewed':sum(s['human_review'] is not None for s in samples),
                'training_eligible':0,'records_directory':str(self.batch)+'/job_XXX/audit/records'}

    def state(self):
        with self.lock:
            return self._state()

    def save(self,payload):
        with self.lock:
            self._state()  # Reject changed batch/audits before routing a write.
            require(isinstance(payload,dict) and isinstance(payload.get('sample_id'),str), 'Invalid review request')
            route=self.routes.get(payload['sample_id'])
            require(route is not None, 'Unknown batch sample')
            jid,sid=route
            saved=self.apps[jid].save({**payload,'sample_id':sid})
            return {'saved':saved['saved'],'state':self._state()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8879)
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args(argv)
    url=f'http://127.0.0.1:{args.port}'
    if args.open and args.port:
        try:
            with urlopen(url+'/api/state',timeout=2) as response:
                existing=json.load(response)
            if existing.get('audit_sha256')==sha256(args.batch/'manifest.json'):
                webbrowser.open(url)
                print('Opened existing batch review:',url,flush=True)
                return
        except (OSError,ValueError):
            pass
    app=BatchReview(args.batch)
    with make_server(app,args.port) as server:
        url=f'http://127.0.0.1:{server.server_port}'
        print('Native multi-plant review ready:',url,flush=True)
        if args.open: webbrowser.open(url)
        try: server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt: pass


if __name__=='__main__':
    main()
