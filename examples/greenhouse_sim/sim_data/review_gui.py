"""Loopback-only browser review for immutable robot-head RGB-D pilots.

No Kit, GPU, cloud services or robot connection. Writes explicit human prototype
review records only; it never enables training or physical execution.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit
from urllib.request import urlopen
import webbrowser

import numpy as np
from PIL import Image

from .dataset_package import active_reviews, checked_audit, queue_for, record
from .dataset_review import read_json, require, safe_file
from .depth_preview import labelled_heatmap, sha256

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = ROOT / "data/sim_data/dataset_reviews/review_20260908_v2/audit.json"
UI = Path(__file__).with_name("review_gui_assets")
DEFAULT_NOTES = {
    "confirm": "I inspected RGB, cut overlay, native identity mask and depth; the prototype point/interval matches the visible petiole.",
    "hold": "I inspected the evidence but remain uncertain; keep pending for further review or another robot-head view.",
    "reject": "I inspected the evidence and the proposed point/interval does not match the target anatomy in this view.",
}


class ReviewApp:
    def __init__(self, audit_path, records):
        self.audit_path, self.records = Path(audit_path).resolve(), Path(records).resolve()
        self.audit = checked_audit(self.audit_path)
        self.audit_hash = sha256(self.audit_path)
        self.run = Path(self.audit["source_run"])
        require(not self.records.is_relative_to(self.run), "Reviews must remain outside capture inputs")
        self.lock = threading.Lock()
        self.token = secrets.token_hex(32)
        self.images = {}
        self.summaries = {}
        # Cache only verified images. No URL can select arbitrary filesystem paths.
        for sample in self.audit["samples"]:
            sid = sample["sample_id"]
            metadata = read_json(self.run / sid / "sample.json")
            for kind, relative in {"rgb": "inputs/rgb.png", "overlay": "review/overlay.png",
                                   "mask": "review/visible_target.png"}.items():
                content = safe_file(self.run / sid, relative).read_bytes()
                require(hashlib.sha256(content).hexdigest() == metadata["files"][relative]["sha256"], "Image changed during loading")
                self.images[(sid, kind)] = content
            content = safe_file(self.audit_path.parent, sid + ".png").read_bytes()
            require(hashlib.sha256(content).hexdigest() == self.audit["cards_sha256"][sid + ".png"], "Review card changed")
            self.images[(sid, "card")] = content
            depth_path, valid_path = self.run / sid / "inputs/depth_m.npy", self.run / sid / "inputs/depth_valid.png"
            depth = np.load(depth_path, allow_pickle=False)
            valid = np.asarray(Image.open(valid_path)) > 0
            for path in (depth_path, valid_path):
                relative = path.relative_to(self.run / sid).as_posix()
                require(sha256(path) == metadata["files"][relative]["sha256"], "Native depth changed during loading")
            preview = labelled_heatmap(depth, valid, .04, 2., sid + " - saved native Isaac depth (review only)")
            stream = io.BytesIO()
            preview.save(stream, format="PNG")
            self.images[(sid, "depth")] = stream.getvalue()
            self.summaries[sid] = sample
        self._history()  # Fail on stale/conflicting records before opening the UI.

    def _history(self):
        require(sha256(self.audit_path) == self.audit_hash, "Audit changed; restart after re-auditing")
        paths = sorted(self.records.glob("*.json")) if self.records.exists() else []
        rows = [read_json(path) for path in paths]
        active = active_reviews(rows, self.audit, self.audit_hash)
        return active, {row["review_id"]: path for row, path in zip(rows, paths)}

    def _state(self):
        active, _ = self._history()
        result = []
        for sid, sample in self.summaries.items():
            assistant, human = (active.get((sid, role)) for role in ("assistant", "human"))
            quality = sample["quality"]
            result.append({"sample_id": sid, "target": sample["target_review_id"],
                "recommended": bool(assistant and assistant["decision"] == "recommend"),
                "queue": queue_for(sample, assistant, human), "quality": quality,
                "camera": sample["camera"], "assistant_review": assistant, "human_review": human,
                "can_confirm": quality["clear_view_gate_passed"] is True,
                "nominal_pixel_xy": sample["nominal_pixel_xy"],
                "visibility": sample["sampled_interval_visible_pixel_fraction"],
                "images": {kind: f"/image/{sid}/{kind}" for kind in ("card", "rgb", "overlay", "mask", "depth")}})
        result.sort(key=lambda r: (not r["recommended"], r["sample_id"]))
        return {"samples": result, "audit_sha256": self.audit_hash,
                "capture_run_name": self.run.name,
                "total": len(result), "reviewed": sum(r["human_review"] is not None for r in result),
                "training_eligible": 0, "records_directory": str(self.records)}

    def state(self):
        with self.lock:
            return self._state()

    def save(self, payload):
        with self.lock:
            require(isinstance(payload, dict) and set(payload) == {
                "sample_id", "reviewer", "decision", "notes", "inspected", "expected_review_id"}, "Invalid review fields")
            sid, reviewer, decision = (payload[k] for k in ("sample_id", "reviewer", "decision"))
            require(isinstance(sid, str) and sid in self.summaries, "Unknown sample")
            require(isinstance(reviewer, str) and 0 < len(reviewer.strip()) <= 120, "Enter your reviewer name")
            require(isinstance(decision, str) and decision in DEFAULT_NOTES, "Invalid human decision")
            require(isinstance(payload["notes"], str) and len(payload["notes"]) <= 4000, "Notes too long")
            require(payload["inspected"] is True, "Confirm that you inspected RGB, cut overlay, mask and depth")
            active, paths = self._history()
            previous = active.get((sid, "human"))
            previous_id = previous["review_id"] if previous else None
            require(payload["expected_review_id"] == previous_id,
                    "This review changed in another tab. Refresh before saving; no decision was overwritten")
            require(decision != "confirm" or self.summaries[sid]["quality"]["clear_view_gate_passed"],
                    "This sample failed the numerical gate. Hold or reject it; do not confirm")
            path = record(self.audit_path, sid, "human", reviewer.strip(), decision,
                          payload["notes"].strip() or DEFAULT_NOTES[decision], self.records,
                          inspected=True, supersedes=paths[previous_id] if previous_id else None)
            return {"saved": read_json(path), "state": self._state()}


def make_server(app, port=8877, *, ui=None):
    assets = Path(ui) if ui is not None else UI
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # No request bodies, names or CSRF tokens in logs.

        def send(self, code, content, mime):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(content)

        def json(self, code, value):
            self.send(code, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")

        def local_request(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return self.headers.get("Host") == expected and self.headers.get("Sec-Fetch-Site", "") != "cross-site"

        def do_GET(self):
            if not self.local_request():
                return self.json(403, {"error": "Loopback, same-site requests only"})
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    content = (assets / "index.html").read_text(encoding="utf-8").replace("__CSRF__", app.token)
                    return self.send(200, content.encode(), "text/html; charset=utf-8")
                if path in ("/ui/app.js", "/ui/style.css"):
                    mime = "text/javascript" if path.endswith(".js") else "text/css"
                    return self.send(200, (assets / path.rsplit("/", 1)[1]).read_bytes(), mime + "; charset=utf-8")
                if path == "/api/state":
                    return self.json(200, app.state())
                pieces = path.split("/")
                if len(pieces) == 4 and pieces[1] == "image" and (pieces[2], pieces[3]) in app.images:
                    return self.send(200, app.images[(pieces[2], pieces[3])], "image/png")
                return self.json(404, {"error": "Not found"})
            except (ValueError, OSError, KeyError) as exc:
                return self.json(409, {"error": str(exc)})

        def do_POST(self):
            expected_origin = f"http://127.0.0.1:{self.server.server_port}"
            if (not self.local_request() or self.headers.get("Origin") != expected_origin
                    or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), app.token)):
                return self.json(403, {"error": "Review writes require the local UI session"})
            if self.path != "/api/review":
                return self.json(404, {"error": "Not found"})
            if self.headers.get("Content-Type") != "application/json":
                return self.json(415, {"error": "Expected JSON"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(0 < length <= 16384 and not self.headers.get("Transfer-Encoding"), "Invalid request size")
                self.connection.settimeout(5)
                payload = json.loads(self.rfile.read(length))
                self.json(200, app.save(payload))
            except (ValueError, OSError, KeyError) as exc:
                self.json(409, {"error": str(exc)})

    class LocalReviewServer(ThreadingHTTPServer):
        allow_reuse_address = False  # Windows must not bind a second reviewer to the same port.
    server = LocalReviewServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--open", action="store_true", help="Open the review page in the default browser")
    args = parser.parse_args(argv)
    records = (args.records or args.audit.parent / "records").resolve()
    if args.open and args.port:
        # Double-clicking the launcher again should reopen this same review set.
        url = f"http://127.0.0.1:{args.port}"
        try:
            with urlopen(url + "/api/state", timeout=2) as response:
                existing = json.load(response)
            if (existing.get("audit_sha256") == sha256(args.audit)
                    and Path(existing.get("records_directory", "")).resolve() == records):
                webbrowser.open(url)
                print(f"Opened existing review GUI: {url}", flush=True)
                return
        except (OSError, ValueError, AttributeError):
            pass
    app = ReviewApp(args.audit, records)
    with make_server(app, args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Robot-head review ready: {url} | {len(app.summaries)} samples | no Isaac/GPU/robot connection", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
