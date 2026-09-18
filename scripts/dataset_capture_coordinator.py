"""Cross-host capture reservation service; no Isaac imports or automatic lease expiry.

Run ONE coordinator. Clients fail closed if it is unavailable. SQLite stays on
the coordinator's local disk; HTTP clients never open a shared SQLite file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SCHEMA = 'greenhouse.capture_identity.v1'
HOSTS = ('local5090', 'thor1', 'thor3')
ALLOCATION = {
    'local5090': ['seed19_full', 'seed41_full', 'seed67_full', 'seed103_full'],
    'thor1': ['seed7_full', 'seed11_full', 'seed43_full', 'seed47_full', 'seed73_full', 'seed101_full'],
    'thor3': ['seed17_full', 'seed23_full', 'seed53_full', 'seed71_full', 'seed83_full', 'seed89_full'],
}
RGB_HEADER = b'RGB|uint8|408|848|3\n'
HEX = re.compile(r'^[0-9a-f]{64}$')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def matrix(value, size):
    require(isinstance(value, list) and len(value) == size, 'Wrong matrix size')
    result = []
    for row in value:
        require(isinstance(row, list) and len(row) == size, 'Wrong matrix row')
        require(all(type(x) in (int, float) and math.isfinite(x) for x in row), 'Nonfinite matrix')
        result.append([round(float(x), 9) or 0.0 for x in row])
    return result


def identity(value):
    """Target hints, file paths, host names and random seeds cannot change a view ID."""
    geometry = value['geometry_sha256']
    require(isinstance(geometry, str) and HEX.fullmatch(geometry), 'Invalid geometry digest')
    family = value['donor_source_family']
    require(isinstance(family, str) and re.fullmatch(r'seed[0-9]+_full', family), 'Invalid donor family')
    split = value['split']
    require(split in ('train', 'validation', 'test'), 'Invalid donor split')
    require(value['resolution'] == [848, 408], 'Native848x408 required')
    camera = matrix(value['camera_to_plant_usd_row_vectors'], 4)
    require(all(abs(camera[i][3] - (1.0 if i == 3 else 0.0)) < 1e-8 for i in range(4)), 'Row-vector rigid transform required')
    rotation = [r[:3] for r in camera[:3]]
    for i in range(3):
        for j in range(3):
            require(abs(sum(rotation[i][k] * rotation[j][k] for k in range(3)) - (i == j)) < 1e-6, 'Nonrigid rotation')
    a, b, c = rotation
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    require(abs(determinant - 1) < 1e-6, 'Reflected camera')
    intrinsics = matrix(value['intrinsics'], 3)
    require(intrinsics[0][0] > 0 and intrinsics[1][1] > 0 and intrinsics[2] == [0, 0, 1], 'Invalid intrinsics')
    return dict(schema=SCHEMA, geometry_sha256=geometry, donor_source_family=family,
                split=split, camera_to_plant_usd_row_vectors=camera,
                intrinsics=intrinsics, resolution=[848, 408])


def task_id(value):
    return digest(identity(value))


def near_pose(a, b):
    if a['geometry_sha256'] != b['geometry_sha256'] or a['intrinsics'] != b['intrinsics']:
        return False
    x, y = a['camera_to_plant_usd_row_vectors'], b['camera_to_plant_usd_row_vectors']
    distance = math.sqrt(sum((x[3][i] - y[3][i]) ** 2 for i in range(3)))
    cosine = (sum(x[i][j] * y[i][j] for i in range(3) for j in range(3)) - 1) / 2
    angle = math.degrees(math.acos(max(-1, min(1, cosine))))
    return distance <= .004 and angle <= .5


def rgb_metrics(path):
    """Lossless decoded RGB identity, plus a heuristic64-bit similarity screen."""
    from PIL import Image
    with Image.open(path) as source:
        require(source.size == (848, 408) and source.mode in ('RGB', 'RGBA'), 'Expected native8-bit RGB/RGBA image')
        rgb = source.convert('RGB')
        pixels = rgb.tobytes()
        small = list(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (small[row*9+col+1] > small[row*9+col])
    return dict(decoded_rgb_sha256=hashlib.sha256(RGB_HEADER + pixels).hexdigest(),
                dhash64_hex=f'{bits:016x}', encoded_rgb_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def metrics(value):
    require(isinstance(value.get('decoded_rgb_sha256'), str) and HEX.fullmatch(value['decoded_rgb_sha256']), 'Invalid RGB hash')
    require(isinstance(value.get('dhash64_hex'), str) and re.fullmatch('[0-9a-f]{16}', value['dhash64_hex']), 'Invalid perceptual hash')
    return {k: value[k] for k in ('decoded_rgb_sha256', 'dhash64_hex')}


class Store:
    def __init__(self, path, hosts=HOSTS):
        self.path = str(path)
        self.hosts = tuple(hosts)
        require(len(set(hosts)) == len(hosts) and hosts, 'Unique host roster required')
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS donors(family TEXT PRIMARY KEY, split TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS geometries(hash TEXT PRIMARY KEY, family TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY, identity_json TEXT NOT NULL, geometry TEXT NOT NULL,
                    host TEXT NOT NULL, claim TEXT NOT NULL, state TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL,
                    rgb TEXT, dhash TEXT, result_json TEXT);
                CREATE INDEX IF NOT EXISTS geometry_index ON tasks(geometry);
                CREATE INDEX IF NOT EXISTS rgb_index ON tasks(rgb);
                CREATE TABLE IF NOT EXISTS scene_aliases(alias TEXT PRIMARY KEY, task_id TEXT NOT NULL);
            ''')
            config = db.execute("SELECT value FROM config WHERE key='hosts'").fetchone()
            if config:
                require(json.loads(config[0]) == list(hosts), 'Host roster differs from existing database')
            else:
                db.execute("INSERT INTO config VALUES('hosts',?)", (canonical(list(hosts)),))
            allocation = db.execute("SELECT value FROM config WHERE key='allocation'").fetchone()
            if allocation:
                require(json.loads(allocation[0]) == ALLOCATION, 'Donor allocation changed; use a reviewed campaign migration')
            else:
                db.execute("INSERT INTO config VALUES('allocation',?)", (canonical(ALLOCATION),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def _lineage(self, db, item):
        family, split, geometry = item['donor_source_family'], item['split'], item['geometry_sha256']
        old = db.execute('SELECT split FROM donors WHERE family=?', (family,)).fetchone()
        require(old is None or old[0] == split, 'Donor split leakage rejected')
        old = db.execute('SELECT family FROM geometries WHERE hash=?', (geometry,)).fetchone()
        require(old is None or old[0] == family, 'Geometry renamed as a different donor')
        db.execute('INSERT OR IGNORE INTO donors VALUES(?,?)', (family, split))
        db.execute('INSERT OR IGNORE INTO geometries VALUES(?,?)', (geometry, family))

    def seed(self, rows):
        """Seed completed local views BEFORE serving; repeated imports are idempotent."""
        prepared = [(identity(r), metrics(r), r.get('stage', 'accepted'), r.get('scene_camera_key')) for r in rows]
        require(all(a is None or isinstance(a, str) and HEX.fullmatch(a) for _, _, _, a in prepared), 'Invalid baseline scene alias')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            require(not db.execute("SELECT 1 FROM tasks WHERE host != 'baseline' LIMIT 1").fetchone(), 'Import inventory before live claims')
            for item, rgb, stage, alias in prepared:
                self._lineage(db, item)
                key = digest(item)
                if alias is not None:
                    existing_alias = db.execute('SELECT task_id FROM scene_aliases WHERE alias=?', (alias,)).fetchone()
                    require(existing_alias is None or existing_alias[0] == key, 'Baseline contains repeated scene-camera views')
                    db.execute('INSERT OR IGNORE INTO scene_aliases VALUES(?,?)', (alias, key))
                old = db.execute('SELECT rgb FROM tasks WHERE id=?', (key,)).fetchone()
                if old:
                    require(old[0] == rgb['decoded_rgb_sha256'], 'Conflicting baseline pose/RGB')
                    continue
                now = time.time()
                db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                           (key, canonical(item), item['geometry_sha256'], 'baseline', 'baseline',
                            'baseline_' + stage, now, now, rgb['decoded_rgb_sha256'], rgb['dhash64_hex'], canonical({'inventory_stage': stage})))
            db.execute("INSERT OR REPLACE INTO config VALUES('seeded','true')")
        return self.stats()

    def claim_many(self, host, values):
        require(host in self.hosts, 'Unknown host')
        require(isinstance(values, list) and 1 <= len(values) <= 4096, 'Finite nonempty batch required')
        prepared = [identity(v) for v in values]
        aliases = [v.get('scene_camera_key') for v in values]
        require(all(a is not None and isinstance(a, str) and HEX.fullmatch(a) for a in aliases), 'Scene-camera alias required')
        answers = []
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            require(db.execute("SELECT 1 FROM config WHERE key='seeded'").fetchone(), 'Baseline inventory not initialized')
            for item, alias in zip(prepared, aliases):
                key = digest(item)
                assigned = next((h for h, families in ALLOCATION.items() if item['donor_source_family'] in families), None)
                if host != assigned:
                    answers.append(dict(task_id=key, granted=False, reason='donor_assigned_to_other_host', assigned_host=assigned))
                    continue
                require(item['split'] == 'train', 'This production campaign is TRAIN only')
                self._lineage(db, item)
                old = db.execute('SELECT id,host,state FROM tasks WHERE id=?', (key,)).fetchone()
                reason = 'exact_view_already_reserved'
                if old is None:
                    old = db.execute('SELECT t.id,t.host,t.state FROM tasks t JOIN scene_aliases a ON a.task_id=t.id WHERE a.alias=?', (alias,)).fetchone()
                    reason = 'same_scene_camera_already_reserved'
                if old is None:
                    for other in db.execute('SELECT id,identity_json,host,state FROM tasks WHERE geometry=? ORDER BY created,id', (item['geometry_sha256'],)):
                        if near_pose(item, json.loads(other['identity_json'])):
                            old = other
                            reason = 'near_pose_already_reserved'
                            break
                if old:
                    answers.append(dict(task_id=key, granted=False, reason=reason,
                                        existing_task_id=old['id'], existing_host=old['host'], existing_state=old['state']))
                    continue
                claim = secrets.token_hex(24)
                now = time.time()
                db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,NULL,NULL,NULL)',
                           (key, canonical(item), item['geometry_sha256'], host, claim, 'reserved', now, now))
                db.execute('INSERT INTO scene_aliases VALUES(?,?)', (alias, key))
                answers.append(dict(task_id=key, granted=True, host=host, claim_id=claim))
        return dict(schema='greenhouse.capture_reservations.v1', reservations=answers)

    def finish(self, host, key, claim, rgb=None, outcome='captured'):
        require(host in self.hosts and outcome in ('captured', 'no_frame', 'failed'), 'Invalid completion')
        value = metrics(rgb) if outcome == 'captured' else None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM tasks WHERE id=?', (key,)).fetchone()
            require(row is not None and row['host'] == host and hmac.compare_digest(row['claim'], claim), 'Claim ownership mismatch')
            if row['state'] != 'reserved':
                previous = json.loads(row['result_json'])
                require(previous['outcome'] == outcome and previous.get('metrics') == value, 'Completion differs from sealed result')
                return previous
            duplicate = []
            if value:
                for other in db.execute('SELECT id,rgb,dhash FROM tasks WHERE rgb IS NOT NULL ORDER BY created,id'):
                    distance = (int(value['dhash64_hex'], 16) ^ int(other['dhash'], 16)).bit_count()
                    exact = value['decoded_rgb_sha256'] == other['rgb']
                    if exact or distance <= 6:
                        duplicate.append(dict(task_id=other['id'], kind='exact_rgb' if exact else 'perceptual_review', dhash_hamming=distance))
            state = 'duplicate_hold' if duplicate else 'complete' if value else outcome
            result = dict(task_id=key, host=host, outcome=outcome, state=state,
                          metrics=value, duplicate_candidates=duplicate,
                          training_approved=False, quality_and_visual_review_still_required=True)
            db.execute('UPDATE tasks SET state=?,updated=?,rgb=?,dhash=?,result_json=? WHERE id=?',
                       (state, time.time(), value['decoded_rgb_sha256'] if value else None,
                        value['dhash64_hex'] if value else None, canonical(result), key))
            return result

    def stats(self):
        with self.connect() as db:
            return dict(states={r[0]: r[1] for r in db.execute('SELECT state,count(*) FROM tasks GROUP BY state')},
                        hosts={r[0]: r[1] for r in db.execute('SELECT host,count(*) FROM tasks GROUP BY host')},
                        seeded=bool(db.execute("SELECT 1 FROM config WHERE key='seeded'").fetchone()),
                        allocation=ALLOCATION, allocation_sha256=digest(ALLOCATION))

    def export(self):
        with self.connect() as db:
            # Claim handles are deliberately omitted from the portable audit export.
            return [dict(task_id=r['id'], identity=json.loads(r['identity_json']), host=r['host'],
                         state=r['state'], result=json.loads(r['result_json']) if r['result_json'] else None)
                    for r in db.execute('SELECT * FROM tasks ORDER BY id')]


class Client:
    def __init__(self, url, token, host, timeout=30):
        require(url.startswith(('http://', 'https://')) and len(token) >= 32, 'Coordinator URL/token required')
        self.url, self.token, self.host, self.timeout = url.rstrip('/'), token, host, timeout

    def call(self, route, value=None):
        payload = None if value is None else canonical(value).encode()
        req = Request(self.url + route, data=payload,
                      headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        # No local fallback, alternate coordinator, redirects or automatic reclaim.
        from urllib.request import HTTPRedirectHandler, build_opener
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                raise ValueError('Coordinator redirects are not allowed')
        try:
            with build_opener(NoRedirect).open(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise ValueError('Coordinator rejected request: ' + exc.read(4096).decode(errors='replace')) from None

    def claim_many(self, values):
        return self.call('/claim', dict(host=self.host, identities=values))

    def finish(self, reservation, rgb=None, outcome='captured'):
        require(reservation.get('granted') is True, 'A granted reservation is required')
        return self.call('/finish', dict(host=self.host, key=reservation['task_id'],
                         claim=reservation['claim_id'], rgb=rgb, outcome=outcome))


def make_server(address, store, token):
    require(len(token) >= 32, 'Use a strong coordinator token')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log tokens, request bodies or claim handles.

        def reply(self, code, body):
            payload = canonical(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.reply(401, {'error': 'Authentication required'})
                return
            try:
                if self.command == 'GET' and self.path == '/status':
                    answer = store.stats()
                elif self.command == 'POST':
                    length = int(self.headers.get('Content-Length', '0'))
                    require(0 < length <= 16 * 1024 * 1024, 'Bounded JSON body required')
                    body = json.loads(self.rfile.read(length))
                    if self.path == '/claim':
                        answer = store.claim_many(body['host'], body['identities'])
                    elif self.path == '/finish':
                        answer = store.finish(**body)
                    else:
                        raise ValueError('Unknown endpoint')
                else:
                    raise ValueError('Unknown endpoint')
                self.reply(200, answer)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception:
                self.reply(500, {'error': 'Coordinator failed; do not capture without a reservation'})
        do_GET = dispatch
        do_POST = dispatch
    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for command in ('init', 'serve', 'export'):
        p = sub.add_parser(command)
        p.add_argument('--db', type=Path, required=True)
        p.add_argument('--hosts', nargs='+', default=list(HOSTS))
        if command == 'init':
            p.add_argument('--inventory', type=Path, required=True)
            p.add_argument('--inventory-sha256', required=True)
        elif command == 'serve':
            p.add_argument('--bind', default='127.0.0.1')
            p.add_argument('--port', type=int, default=8769)
            p.add_argument('--token-env', default='DATASET_COORDINATOR_TOKEN')
        else:
            p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.db.parent.is_dir(), 'Database parent must exist')
    store = Store(args.db, args.hosts)
    if args.action == 'init':
        raw = args.inventory.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == args.inventory_sha256, 'Inventory hash mismatch')
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        require(rows, 'Nonempty baseline inventory required')
        print(canonical(store.seed(rows)))
    elif args.action == 'export':
        with args.output.open('x', encoding='utf-8') as stream:
            for row in store.export():
                stream.write(canonical(row) + '\n')
        print(canonical(store.stats()))
    else:
        require(store.stats()['seeded'], 'Initialize the verified baseline before serving')
        token = os.environ.get(args.token_env, '')
        server = make_server((args.bind, args.port), store, token)
        print(canonical(dict(listening=server.server_address, hosts=args.hosts, automatic_reclaim=False)), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
