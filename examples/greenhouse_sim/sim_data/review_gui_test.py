import http.client
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from sim_data.dataset_package import record
from sim_data.dataset_review import SCHEMA
from sim_data.depth_preview import sha256
from sim_data.review_gui import ReviewApp, make_server, main


@pytest.fixture
def gui_source(tmp_path):
    run, review = tmp_path / "capture", tmp_path / "audit"
    run.mkdir(); review.mkdir()
    bindings, cards, samples = {}, {}, []
    for number in (4, 5, 1):
        sid = f"sample_{number:04}"
        directory = run / sid
        for folder in ("inputs", "review"):
            (directory / folder).mkdir(parents=True)
        rgb = np.full((408, 848, 3), 150, dtype=np.uint8)
        rgb[150:154, 200:500] = [20, 90, 30]
        image = Image.fromarray(rgb)
        for name in ("inputs/rgb.png", "review/overlay.png", "review/visible_target.png"):
            image.save(directory / name)
        np.save(directory / "inputs/depth_m.npy", np.ones((408, 848), np.float32))
        Image.fromarray(np.full((408, 848), 255, np.uint8)).save(directory / "inputs/depth_valid.png")
        files = {}
        for path in directory.rglob("*"):
            if path.is_file():
                relative = path.relative_to(directory).as_posix()
                files[relative] = {"sha256": sha256(path), "role": "observation" if relative.startswith("inputs/") else "review_only"}
                bindings[str(path)] = sha256(path)
        metadata = directory / "sample.json"
        metadata.write_text(json.dumps({"files": files}))
        bindings[str(metadata)] = sha256(metadata)
        card = review / (sid + ".png")
        image.save(card)
        cards[card.name] = sha256(card)
        samples.append({"sample_id": sid, "target_review_id": "B05" if number in (4, 5) else "B03",
            "quality": {"clear_view_gate_passed": number != 1, "estimated_petiole_diameter_px": 3.4,
                        "projected_interval_length_px": 8., "clear_view_rejection_reasons": [] if number != 1 else ["too_thin"]},
            "camera": {"mounted_robot_pov_verified": True, "camera_path": "/World/RBY1/link_head_2/Camera"},
            "nominal_pixel_xy": [250., 152.], "sampled_interval_visible_pixel_fraction": 1.})
    audit = review / "audit.json"
    audit.write_text(json.dumps({"schema_version": SCHEMA, "state": "complete_engineering_audit_not_approval",
        "training_dataset_approved": False, "source_run": str(run), "samples": samples,
        "bindings_sha256": bindings, "cards_sha256": cards}))
    records = review / "records"
    for sid in ("sample_0004", "sample_0005"):
        record(audit, sid, "assistant", "Test engineer", "recommend", "Fixture review", records, inspected=True)
    return audit, records


@pytest.fixture
def app(gui_source):
    return ReviewApp(*gui_source)


def decision(**overrides):
    return {"sample_id": "sample_0004", "reviewer": "Test human", "decision": "confirm",
            "notes": "", "inspected": True, "expected_review_id": None, **overrides}


def test_opening_and_loading_evidence_never_confirms_or_changes_sources(app):
    before = sorted(app.records.iterdir())
    state = app.state()
    assert state["total"] == 3 and state["reviewed"] == 0
    assert state["capture_run_name"] == "capture"
    assert [s["sample_id"] for s in state["samples"][:2]] == ["sample_0004", "sample_0005"]
    assert all(s["human_review"] is None for s in state["samples"])
    assert sorted(app.records.iterdir()) == before
    for path, expected in app.audit["bindings_sha256"].items():
        assert sha256(path) == expected
    assert app.images[("sample_0004", "depth")].startswith(b"\x89PNG")


def test_explicit_save_persists_and_reload_resumes_without_training_approval(app, gui_source):
    result = app.save(decision())
    row = result["saved"]
    assert row["reviewer_role"] == "human" and row["human_prototype_label_confirmation"]
    assert not row["training_eligible"] and not row["physical_cut_approved"]
    assert row["horticultural_validation"] == "pending" and row["notes"]
    again = ReviewApp(*gui_source)
    assert again.state()["reviewed"] == 1
    assert again.state()["samples"][0]["human_review"]["review_id"] == row["review_id"]


def test_revision_preserves_history_and_stale_tabs_cannot_overwrite(app):
    first = app.save(decision())["saved"]
    with pytest.raises(ValueError, match="another tab"): app.save(decision(decision="hold"))
    second = app.save(decision(decision="hold", expected_review_id=first["review_id"]))["saved"]
    assert second["supersedes_review_id"] == first["review_id"]
    assert len(list(app.records.glob("*.json"))) == 4  # Two engineers, two human revisions.
    assert app.state()["samples"][0]["human_review"]["decision"] == "hold"


@pytest.mark.parametrize("override", [{"reviewer": " "}, {"inspected": False}, {"inspected": "true"},
    {"decision": "recommend"}, {"decision": []}, {"sample_id": "../escape"}, {"notes": "x"*4001}, {"reviewer_role": "assistant"}])
def test_invalid_or_implicit_reviews_refused(app, override):
    with pytest.raises(ValueError): app.save(decision(**override))
    assert app.state()["reviewed"] == 0


def test_failed_gate_cannot_be_confirmed_but_can_be_held(app):
    with pytest.raises(ValueError, match="failed the numerical gate"):
        app.save(decision(sample_id="sample_0001"))
    result = app.save(decision(sample_id="sample_0001", decision="hold"))
    assert result["saved"]["decision"] == "hold"
    assert not result["saved"]["human_prototype_label_confirmation"]


def test_sources_changed_after_loading_block_save(app):
    (app.run / "sample_0004/inputs/rgb.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Stale"): app.save(decision())
    assert app.state()["reviewed"] == 0


@pytest.fixture
def server(app):
    server = make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    yield server
    server.shutdown(); server.server_close(); thread.join(timeout=2)


def request(server, method, path, payload=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, body=json.dumps(payload) if payload is not None else None, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def write_headers(server, app):
    return {"Origin": f"http://127.0.0.1:{server.server_port}", "X-Review-Token": app.token,
            "Content-Type": "application/json"}


def test_http_page_images_and_state_are_local_and_read_only(server, app):
    status, headers, html = request(server, "GET", "/")
    assert status == 200 and b"Robot-head label review" in html
    assert b"__CSRF__" not in html and app.token.encode() in html
    assert headers["Cache-Control"] == "no-store" and headers["X-Frame-Options"] == "DENY"
    assert "Access-Control-Allow-Origin" not in headers
    for path in ("/ui/app.js", "/ui/style.css", "/api/state", "/image/sample_0004/card", "/image/sample_0004/rgb", "/image/sample_0004/depth"):
        assert request(server, "GET", path)[0] == 200
    assert app.state()["reviewed"] == 0


@pytest.mark.parametrize("path", ["/../dev.md", "/image/../../dev.md", "/image/sample_0004/sample.json", "/ui/../review_gui.py", "/image/sample_0004/%2e%2e%2faudit.json"])
def test_only_allowlisted_images_and_ui_assets_are_served(server, path):
    assert request(server, "GET", path)[0] == 404


@pytest.mark.parametrize("kind", ["missing_token", "wrong_token", "cross_origin", "wrong_host", "cross_site"])
def test_csrf_and_dns_rebinding_rejected(server, app, kind):
    headers = write_headers(server, app)
    if kind == "missing_token": del headers["X-Review-Token"]
    elif kind == "wrong_token": headers["X-Review-Token"] = "wrong"
    elif kind == "cross_origin": headers["Origin"] = "https://example.com"
    elif kind == "wrong_host": headers["Host"] = "attacker.example"
    else: headers["Sec-Fetch-Site"] = "cross-site"
    assert request(server, "POST", "/api/review", decision(), headers)[0] == 403
    assert app.state()["reviewed"] == 0


def test_http_save_is_explicit_and_duplicate_submission_conflicts(server, app):
    headers = write_headers(server, app)
    status, _, body = request(server, "POST", "/api/review", decision(), headers)
    assert status == 200 and json.loads(body)["saved"]["reviewer_role"] == "human"
    assert request(server, "POST", "/api/review", decision(), headers)[0] == 409
    assert request(server, "POST", "/api/review", decision(), {**headers, "Content-Type": "text/plain"})[0] == 415


def test_gui_markup_has_explicit_inspection_and_no_bulk_auto_approval():
    ui = Path(__file__).with_name("review_gui_assets")
    html, script = (ui / "index.html").read_text(encoding="utf-8"), (ui / "app.js").read_text(encoding="utf-8")
    assert 'type="checkbox" id="inspected"' in html
    assert 'id="confirm"' in html and 'id="hold"' in html and 'id="reject"' in html
    assert "expected_review_id" in script and 'setView("card")' in script
    assert "onclick = () => save" in script


def test_second_server_cannot_claim_same_port(server, app):
    with pytest.raises(OSError): make_server(app, server.server_port)


def test_launcher_reopens_existing_matching_gui_without_new_reviews(server, app, monkeypatch):
    opened = []
    monkeypatch.setattr("sim_data.review_gui.webbrowser.open", lambda url: opened.append(url))
    main(["--audit", str(app.audit_path), "--records", str(app.records), "--port", str(server.server_port), "--open"])
    assert opened == [f"http://127.0.0.1:{server.server_port}"]
    assert app.state()["reviewed"] == 0
